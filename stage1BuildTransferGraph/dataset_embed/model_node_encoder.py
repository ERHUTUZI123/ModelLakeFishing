"""
ModelNodeEncoder — training-time assembly of x_m^(0).

Completes the concat that xm0_builder.build_xm0 deliberately leaves half-done:

    x_m^(0) = [ frozen(e_name || e_desc)  ||  e_size  ||  e_fam ]
                from data['model'].x         looked up here (LEARNABLE)

This is the counterpart of the two concat sites in ModelLens MLP.py
(encode_model():879-911 and forward():950-964), with three deliberate
differences:

  * no e_id / ID-dropout — GraphSAGE must stay inductive, so no ID embedding
    exists; cold-start hardening is edge dropout at training time instead
  * one concat site, per node, before the GNN — ModelLens splits the concat
    across two call sites only because size/family are gated optional inputs
    of a pairwise (model, dataset) scorer; a node-feature pipeline needs one
  * no clamp()-based safe_ids fallback — clamping an out-of-range index
    silently aliases it onto a real model's row, the exact failure mode the
    mappedID row-order contract exists to prevent; out-of-range indices must
    crash here, not be papered over

Usage (inside the future GraphSAGE model):

    enc = ModelNodeEncoder(
        frozen_dim=xm0["frozen"].shape[1],
        num_size_buckets=xm0["num_size_buckets"],
        num_families=xm0["num_families"],
    )
    x_m0 = enc(
        data["model"].x,
        data["model"].size_bucket_id,
        data["model"].family_id,
    )   # [num_models_in_batch, enc.out_dim]

Works unchanged on NeighborLoader mini-batches: size_bucket_id / family_id
are [num_nodes] attributes on the model node store, so PyG slices them
together with x and row alignment is preserved automatically.
"""

import torch
import torch.nn as nn


class ModelNodeEncoder(nn.Module):
    """
    x_m^(0) = [x_frozen || size_embedding(size_bucket_id) || family_embedding(family_id)]

    size_embedding / family_embedding are the LEARNABLE halves of x_m^(0):
    nn.Embedding tables updated by gradients during GraphSAGE training.
    x_frozen passes through untouched (no grad path into the graph's .x).

    Parameters
    ----------
    frozen_dim       : width of the frozen matrix (name_dim + desc_dim from build_xm0)
    num_size_buckets : xm0["num_size_buckets"] (bucket 0 = unknown)
    num_families     : xm0["num_families"] (id 0 = Other); row identity is
                       pinned by the family vocab CSV — version it with the
                       checkpoint or the trained rows become orphans
    size_dim         : e_size width; keep modest (plan.md step 4: model-side
                       dim must not dwarf the dataset side)
    family_dim       : e_fam width
    """

    def __init__(
        self,
        frozen_dim: int,
        num_size_buckets: int,
        num_families: int,
        size_dim: int = 16,
        family_dim: int = 16,
    ):
        super().__init__()
        self.frozen_dim = frozen_dim
        self.size_embedding = nn.Embedding(num_size_buckets, size_dim)
        self.family_embedding = nn.Embedding(num_families, family_dim)
        self.out_dim = frozen_dim + size_dim + family_dim

    def forward(
        self,
        x_frozen: torch.Tensor,          # [B, frozen_dim] float
        size_bucket_id: torch.Tensor,    # [B] long
        family_id: torch.Tensor,         # [B] long
    ) -> torch.Tensor:                   # [B, out_dim]
        assert x_frozen.shape[-1] == self.frozen_dim, (
            f"frozen width {x_frozen.shape[-1]} != expected {self.frozen_dim}; "
            "graph .x and encoder were built from different xm0 outputs"
        )
        h_size = self.size_embedding(size_bucket_id)
        h_family = self.family_embedding(family_id)
        return torch.cat([x_frozen, h_size, h_family], dim=-1)
