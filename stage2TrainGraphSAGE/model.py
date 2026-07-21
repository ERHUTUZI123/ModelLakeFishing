"""
model.py — Stage 2, Step 1: the inductive heterogeneous GraphSAGE model.

Three segments (plan.md 第一步):

  1. Node encoding   — model side = ModelNodeEncoder (frozen[name||desc] ||
                       learnable size || learnable family) -> Linear to hidden;
                       dataset side = Linear(probe_dim -> hidden). Both node
                       types reach a common hidden dim BEFORE message passing.
  2. Message passing — 2-layer GraphSAGE, heterogenized with `to_hetero` so
                       every edge relation (incl. is_base_of vs rev_is_base_of
                       vs trained_on ...) gets its OWN aggregation weights.
                       Lineage and performance edges never share a weight set.
  3. Output head     — one shared Linear to the final embedding dim, then
                       L2-normalize, so z_m and z_d live in the same cosine
                       space HNSW will index.

This file defines the network only. It produces z_dict = {'model': z_m,
'dataset': z_d}; the dot-product scorer for L_perf and the contrastive loss
are Step 3, built on top of z_dict — deliberately not here.

Inductive bottom line: there is NO nn.Embedding indexed by node id / mappedID
anywhere. The only embedding tables are inside ModelNodeEncoder and are indexed
by *semantic* discrete features (size bucket, family) that apply to unseen
models too. The __main__ smoke test asserts this.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero

# Make the repo root importable so the Stage-1 encoder resolves under the same
# dotted convention the contract checker uses
# (ModelLakeFishing.stage1BuildTransferGraph.dataset_embed.model_node_encoder).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage1BuildTransferGraph.dataset_embed.model_node_encoder import (  # noqa: E402
    ModelNodeEncoder,
    DatasetNodeEncoder,
)

# Stage 1 already ships the directed lineage pair (is_base_of base->derivative
# and rev_is_base_of derivative->base, independent weights) inside the .pt —
# see HGraph._add_directed_lineage in graph.py. No Stage-2 graph surgery here.
_HGRAPH_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "stage1BuildTransferGraph", "hgraph_zoo_xm0.pt")
)


def load_hgraph(path: str = _HGRAPH_PATH):
    """Load the Stage-1 graph. Returns (data, xm0_meta, unique_model_id)."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["data"], payload["xm0_meta"], payload["unique_model_id"]


class _SAGEBackbone(nn.Module):
    """
    Homogeneous shallow GraphSAGE — the body `to_hetero` clones per relation.

    `num_layers` is 1 or 2 (depth is the most direct over-smoothing knob; we do
    not go deeper). Kept at a fixed hidden width (non-lazy) on purpose: lazy
    SAGEConv((-1,-1)) has no parameters until the first forward, which would
    silently exclude the conv weights from the optimizer / first checkpoint if
    either is created before a warm-up pass. Both node types are already
    projected to `hidden_channels` upstream, so a fixed width is all we need.
    """

    def __init__(self, hidden_channels: int, num_layers: int = 2):
        super().__init__()
        assert num_layers in (1, 2), "keep depth shallow: num_layers must be 1 or 2"
        self.convs = nn.ModuleList(
            SAGEConv(hidden_channels, hidden_channels) for _ in range(num_layers)
        )

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:        # ReLU between layers, not after the last
                x = F.relu(x)
        return x


class HeteroGraphSAGE(nn.Module):
    """
    Inductive heterogeneous GraphSAGE producing L2-normalized node embeddings.

    Parameters
    ----------
    metadata         : data.metadata() of the PREPARED graph (must already carry
                       is_base_of + rev_is_base_of — see graph_prep.py)
    frozen_dim       : width of data['model'].x  (xm0 name_dim + desc_dim, e.g. 448)
    num_size_buckets : xm0_meta['num_size_buckets']  (e.g. 15)
    num_families     : xm0_meta['num_families']      (e.g. 136)
    dataset_in_dim   : width of data['dataset'].x    (probe embedding, e.g. 768)
    hidden_channels  : common hidden dim entering message passing
    out_dim          : final embedding dim (what HNSW indexes)
    size_dim/family_dim : learnable embedding widths inside ModelNodeEncoder
    """

    def __init__(
        self,
        metadata,
        frozen_dim: int,
        num_size_buckets: int,
        num_families: int,
        dataset_in_dim: int,
        hidden_channels: int = 128,
        out_dim: int = 128,
        size_dim: int = 16,
        family_dim: int = 16,
        num_layers: int = 2,
        # ── dataset side (xd0): optional, backward-compatible ────────────────
        # When num_task_types is given, the dataset node is encoded by a
        # DatasetNodeEncoder (frozen xd0 views || learnable task_type/n_class/
        # arity rows), symmetric to the model side. When None (old xm0-only
        # graphs), the dataset side stays a plain Linear on data['dataset'].x.
        num_task_types: int | None = None,
        n_class_buckets: int | None = None,
        num_arities: int | None = None,
        task_dim: int = 16,
        nclass_dim: int = 8,
        arity_dim: int = 4,
        # v3 Z1: learnable projection of the frozen xd0 views (None = legacy)
        dataset_frozen_proj_dim: int | None = None,
        # ── Phase 2 (edge-aware message passing): optional, backward-compatible ──
        # edge_aware=False -> the shipped to_hetero(SAGEConv) path (ignores
        # edge_attr). edge_aware=True -> EdgeAwareHetero, consuming edge_attr for
        # the relations named in `weighted_relations` (by middle relation name,
        # e.g. {"similar_to", "trained_on"}). None -> weight every relation.
        edge_aware: bool = False,
        weighted_relations=None,
        # ── Phase 5: separate model/dataset projection heads (still pure MIPS) ──
        separate_heads: bool = False,
        # ── D1 §5.3 feature-rework variants (all defaults = legacy behaviour) ──
        # name_dim is the slice boundary inside the frozen x (xm0_meta['name_dim']);
        # required only when use_desc=False or name_proj_dim is set. The variants
        # cut/reshape the ENCODER's view of x — the graph's .x is never rebuilt.
        name_dim: int | None = None,
        use_desc: bool = True,
        use_family: bool = True,
        name_proj_dim: int | None = None,
        name_proj_seed: int = 42,
        num_model_tasks: int | None = None,
        model_task_dim: int = 16,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.separate_heads = separate_heads
        self.edge_aware = edge_aware
        self.weighted_relations = (
            list(weighted_relations) if weighted_relations is not None else None)

        # ── segment 1: node encoding -> common hidden dim ────────────────────
        self.model_encoder = ModelNodeEncoder(
            frozen_dim=frozen_dim,
            num_size_buckets=num_size_buckets,
            num_families=num_families,
            size_dim=size_dim,
            family_dim=family_dim,
            name_dim=name_dim,
            use_desc=use_desc,
            use_family=use_family,
            name_proj_dim=name_proj_dim,
            name_proj_seed=name_proj_seed,
            num_model_tasks=num_model_tasks,
            task_dim=model_task_dim,
        )
        self.model_proj = nn.Linear(self.model_encoder.out_dim, hidden_channels)

        self.use_dataset_encoder = num_task_types is not None
        if self.use_dataset_encoder:
            self.dataset_encoder = DatasetNodeEncoder(
                frozen_dim=dataset_in_dim,
                num_task_types=num_task_types,
                n_class_buckets=n_class_buckets,
                num_arities=num_arities,
                task_dim=task_dim,
                nclass_dim=nclass_dim,
                arity_dim=arity_dim,
                frozen_proj_dim=dataset_frozen_proj_dim,
            )
            self.dataset_proj = nn.Linear(self.dataset_encoder.out_dim, hidden_channels)
        else:
            self.dataset_encoder = None
            self.dataset_proj = nn.Linear(dataset_in_dim, hidden_channels)

        # ── segment 2: heterogeneous 2-layer GraphSAGE ───────────────────────
        # to_hetero duplicates the backbone's convs per relation in `metadata`,
        # so each edge type carries independent weights (aggr='sum' across the
        # relations meeting at a destination node).
        # Keep `metadata` so a checkpoint can rebuild an identical hetero module
        # (same relations => same state_dict keys) — see learnable.save_checkpoint.
        self.graph_metadata = metadata
        if edge_aware:
            from ModelLakeFishing.stage2TrainGraphSAGE.edge_aware import EdgeAwareHetero
            # resolve weighted_relations (middle names) to full edge_types
            rels = None
            if self.weighted_relations is not None:
                wr = set(self.weighted_relations)
                rels = [et for et in metadata[1] if et[1] in wr]
            self.gnn = EdgeAwareHetero(hidden_channels, metadata,
                                       num_layers=num_layers, relation_weights=rels)
        else:
            self.gnn = to_hetero(_SAGEBackbone(hidden_channels, num_layers), metadata, aggr="sum")

        # ── segment 3: output head(s) + L2 norm ──────────────────────────────
        # shared head keeps z_m / z_d in one metric space; separate heads give
        # each node type its own projection. BOTH stay pure MIPS: score = z_d . z_m
        # on unit vectors, so HNSW geometry is identical either way (Phase 5).
        if separate_heads:
            self.model_head = nn.Linear(hidden_channels, out_dim)
            self.dataset_head = nn.Linear(hidden_channels, out_dim)
        else:
            self.head = nn.Linear(hidden_channels, out_dim)

    def encode_nodes(self, data) -> dict:
        """Segment 1, per node type. Frozen model.x rides through untouched.
        Dataset side uses DatasetNodeEncoder when xd0 columns are present, else a
        plain Linear on the frozen dataset features (backward-compatible)."""
        m = data["model"]
        # task_id rides on the model node store exactly like size/family ids
        # (auto-sliced by loaders); required iff the encoder has a task table.
        task_id = getattr(m, "task_id", None)
        x_model = self.model_encoder(m.x, m.size_bucket_id, m.family_id, task_id)
        d = data["dataset"]
        if self.use_dataset_encoder:
            x_dataset = self.dataset_encoder(
                d.x, d.task_type_id, d.n_class_bucket_id, d.arity_id)
        else:
            x_dataset = d.x
        return {
            "model": self.model_proj(x_model),
            "dataset": self.dataset_proj(x_dataset),
        }

    def forward(self, data) -> dict:
        """
        data : a HeteroData (full graph OR a LinkNeighborLoader mini-batch — the
               sliced size_bucket_id / family_id ride on data['model'] either way).

        Returns {'model': z_m, 'dataset': z_d}, each L2-normalized. Row order
        matches the node order in `data` (full graph => mappedID order).
        """
        x_dict = self.encode_nodes(data)
        if self.edge_aware:
            edge_attr_dict = {
                et: data[et].edge_attr for et in data.edge_types
                if getattr(data[et], "edge_attr", None) is not None
            }
            h_dict = self.gnn(x_dict, data.edge_index_dict, edge_attr_dict)
        else:
            h_dict = self.gnn(x_dict, data.edge_index_dict)
        if self.separate_heads:
            return {
                "model": F.normalize(self.model_head(h_dict["model"]), p=2, dim=-1),
                "dataset": F.normalize(self.dataset_head(h_dict["dataset"]), p=2, dim=-1),
            }
        return {nt: F.normalize(self.head(h), p=2, dim=-1) for nt, h in h_dict.items()}
