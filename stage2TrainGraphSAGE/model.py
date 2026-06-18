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
    ):
        super().__init__()
        self.num_layers = num_layers

        # ── segment 1: node encoding -> common hidden dim ────────────────────
        self.model_encoder = ModelNodeEncoder(
            frozen_dim=frozen_dim,
            num_size_buckets=num_size_buckets,
            num_families=num_families,
            size_dim=size_dim,
            family_dim=family_dim,
        )
        self.model_proj = nn.Linear(self.model_encoder.out_dim, hidden_channels)
        self.dataset_proj = nn.Linear(dataset_in_dim, hidden_channels)

        # ── segment 2: heterogeneous 2-layer GraphSAGE ───────────────────────
        # to_hetero duplicates the backbone's convs per relation in `metadata`,
        # so each edge type carries independent weights (aggr='sum' across the
        # relations meeting at a destination node).
        # Keep `metadata` so a checkpoint can rebuild an identical hetero module
        # (same relations => same state_dict keys) — see learnable.save_checkpoint.
        self.graph_metadata = metadata
        self.gnn = to_hetero(_SAGEBackbone(hidden_channels, num_layers), metadata, aggr="sum")

        # ── segment 3: shared output head + L2 norm ──────────────────────────
        # one head for both node types keeps z_m / z_d in a single metric space.
        self.head = nn.Linear(hidden_channels, out_dim)

    def encode_nodes(self, data) -> dict:
        """Segment 1, per node type. Frozen model.x rides through untouched."""
        m = data["model"]
        x_model = self.model_encoder(m.x, m.size_bucket_id, m.family_id)
        return {
            "model": self.model_proj(x_model),
            "dataset": self.dataset_proj(data["dataset"].x),
        }

    def forward(self, data) -> dict:
        """
        data : a HeteroData (full graph OR a LinkNeighborLoader mini-batch — the
               sliced size_bucket_id / family_id ride on data['model'] either way).

        Returns {'model': z_m, 'dataset': z_d}, each L2-normalized. Row order
        matches the node order in `data` (full graph => mappedID order).
        """
        x_dict = self.encode_nodes(data)
        h_dict = self.gnn(x_dict, data.edge_index_dict)
        return {nt: F.normalize(self.head(h), p=2, dim=-1) for nt, h in h_dict.items()}
