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
    Default (legacy) form:

        x_m^(0) = [x_frozen || size_embedding(size_bucket_id) || family_embedding(family_id)]

    D1 feature-rework form (remediation plan v2 §5.3), reachable via kwargs:

        x_m^(0)' = [ P·e_name || e_size || e_task ]      (F4: 16 + 16 + 16 dims)

    where P is a FROZEN Gaussian random projection (a buffer, not a Parameter:
    saved with the checkpoint, moved by .to(device), receives no gradient) with
    entries ~ N(0, 1/name_proj_dim), so E||e_name·P||^2 = ||e_name||^2
    (Johnson-Lindenstrauss scaling). Structural down-weighting: a scalar factor
    on e_name would be absorbed by the first Linear; shrinking its DIMENSION
    cannot be.

    size_embedding / family_embedding / task_embedding are the LEARNABLE parts:
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
    name_dim         : slice boundary inside x_frozen ([:name_dim] = e_name,
                       [name_dim:] = e_desc). REQUIRED for any variant that
                       drops desc or projects the name; None = legacy passthrough.
    use_desc         : False -> e_desc (the 384-d template-noise block) is cut
                       before the concat (F1+). Frozen x in the graph is unchanged.
    use_family       : False -> no family table (F2+): family structure is
                       carried by the lineage edges, not the feature layer.
    name_proj_dim    : e.g. 16 -> e_name goes through the frozen projection P
                       [name_dim x name_proj_dim] (F4). None = keep raw width.
    name_proj_seed   : seed generating P; recorded in the checkpoint repro dict —
                       same seed + same dims => bit-identical P.
    num_model_tasks  : rows of the MODEL-side task table (F3+; id 0 = Other, row
                       identity pinned by task_vocab.csv — the family discipline).
                       None = no task table (legacy).
    task_dim         : e_task width.
    """

    def __init__(
        self,
        frozen_dim: int,
        num_size_buckets: int,
        num_families: int,
        size_dim: int = 16,
        family_dim: int = 16,
        *,
        name_dim: int | None = None,
        use_desc: bool = True,
        use_family: bool = True,
        name_proj_dim: int | None = None,
        name_proj_seed: int = 42,
        num_model_tasks: int | None = None,
        task_dim: int = 16,
    ):
        super().__init__()
        self.frozen_dim = frozen_dim
        self.name_dim = name_dim
        self.use_desc = use_desc
        self.use_family = use_family
        self.name_proj_dim = name_proj_dim
        self.name_proj_seed = name_proj_seed
        self.num_model_tasks = num_model_tasks
        if (not use_desc or name_proj_dim is not None) and name_dim is None:
            raise ValueError(
                "name_dim is required to slice x_frozen when dropping desc or "
                "projecting the name (pass xm0_meta['name_dim'])"
            )

        self.size_embedding = nn.Embedding(num_size_buckets, size_dim)
        self.family_embedding = nn.Embedding(num_families, family_dim) if use_family else None
        self.task_embedding = (nn.Embedding(num_model_tasks, task_dim)
                               if num_model_tasks is not None else None)

        if name_proj_dim is not None:
            g = torch.Generator().manual_seed(name_proj_seed)
            proj = torch.randn(name_dim, name_proj_dim, generator=g) / (name_proj_dim ** 0.5)
            self.register_buffer("name_proj", proj)   # frozen: buffer, never a Parameter
        else:
            self.name_proj = None

        if use_desc and name_proj_dim is None:
            frozen_out = frozen_dim                     # legacy passthrough, uncut
        else:
            name_out = name_proj_dim if name_proj_dim is not None else name_dim
            frozen_out = name_out + ((frozen_dim - name_dim) if use_desc else 0)
        self.frozen_out_dim = frozen_out
        self.out_dim = (frozen_out + size_dim
                        + (family_dim if use_family else 0)
                        + (task_dim if num_model_tasks is not None else 0))

    def forward(
        self,
        x_frozen: torch.Tensor,          # [B, frozen_dim] float
        size_bucket_id: torch.Tensor,    # [B] long
        family_id: torch.Tensor,         # [B] long (ignored when use_family=False)
        task_id: torch.Tensor | None = None,   # [B] long (required with a task table)
    ) -> torch.Tensor:                   # [B, out_dim]
        assert x_frozen.shape[-1] == self.frozen_dim, (
            f"frozen width {x_frozen.shape[-1]} != expected {self.frozen_dim}; "
            "graph .x and encoder were built from different xm0 outputs"
        )
        if self.use_desc and self.name_proj is None:
            frozen_part = x_frozen                      # legacy: whole block rides through
        else:
            name = x_frozen[:, :self.name_dim]
            if self.name_proj is not None:
                name = name @ self.name_proj            # frozen JL projection, no grad path
            frozen_part = (torch.cat([name, x_frozen[:, self.name_dim:]], dim=-1)
                           if self.use_desc else name)
        parts = [frozen_part, self.size_embedding(size_bucket_id)]
        if self.use_family:
            parts.append(self.family_embedding(family_id))
        if self.task_embedding is not None:
            if task_id is None:
                raise ValueError(
                    "encoder has a model-task table but no task_id was passed — "
                    "attach data['model'].task_id (see d1_model_task_vocab.py)"
                )
            parts.append(self.task_embedding(task_id))
        return torch.cat(parts, dim=-1)


class DatasetNodeEncoder(nn.Module):
    """
    Dataset-side analogue of ModelNodeEncoder (see xd0_builder.py /
    dataset_embedding_redesign.md). Cashes in the three discrete xd0 descriptors
    as LEARNABLE embedding rows:

        x_d^(0) = [ x_frozen(e_domain||e_label||e_card||e_stats)
                    || task_type_embedding(task_type_id)
                    || n_class_embedding(n_class_bucket_id)
                    || arity_embedding(arity_id) ]

    Same design rules as the model side: the frozen multi-view features pass
    through untouched (no grad into the graph's .x), the three index columns ride
    on data['dataset'] so loaders slice them with row alignment, and there is NO
    node-id embedding (inductive). Out-of-range indices crash (no clamp aliasing).

    Parameters
    ----------
    frozen_dim      : width of data['dataset'].x (xd0 frozen views)
    num_task_types  : xd0_meta['num_task_types'] (id 0 = Other; row identity pinned
                      by task_type_vocab — version it with the checkpoint)
    n_class_buckets : xd0_meta['n_class_buckets'] (bucket 0 = regression/unknown)
    num_arities     : xd0_meta['num_arities'] (single / pair / multi)
    task_dim/nclass_dim/arity_dim : learnable widths; kept modest so the dataset
                      side does not dwarf the model side.
    """

    def __init__(
        self,
        frozen_dim: int,
        num_task_types: int,
        n_class_buckets: int,
        num_arities: int,
        task_dim: int = 16,
        nclass_dim: int = 8,
        arity_dim: int = 4,
        *,
        # v3 Z1: LEARNABLE projection of the frozen block before the concat.
        # The frozen views are 4618-d vs 28 learnable dims (99.4% : 0.6%) —
        # in a concat+Linear encoder, dimension share is capacity share
        # (the F4 lesson: a scalar down-weight is absorbed by the next Linear;
        # a dimension cut is not). frozen_proj_dim=128 -> 128:28 = 82% : 18%.
        # Learnable (unlike F4's frozen P): the probe views are informative but
        # redundant, so training should choose the surviving subspace.
        frozen_proj_dim: int | None = None,
    ):
        super().__init__()
        self.frozen_dim = frozen_dim
        self.frozen_proj_dim = frozen_proj_dim
        self.frozen_proj = (nn.Linear(frozen_dim, frozen_proj_dim, bias=False)
                            if frozen_proj_dim is not None else None)
        self.task_type_embedding = nn.Embedding(num_task_types, task_dim)
        self.n_class_embedding = nn.Embedding(n_class_buckets, nclass_dim)
        self.arity_embedding = nn.Embedding(num_arities, arity_dim)
        frozen_out = frozen_proj_dim if frozen_proj_dim is not None else frozen_dim
        self.out_dim = frozen_out + task_dim + nclass_dim + arity_dim

    def forward(
        self,
        x_frozen: torch.Tensor,          # [B, frozen_dim] float
        task_type_id: torch.Tensor,      # [B] long
        n_class_bucket_id: torch.Tensor, # [B] long
        arity_id: torch.Tensor,          # [B] long
    ) -> torch.Tensor:                   # [B, out_dim]
        assert x_frozen.shape[-1] == self.frozen_dim, (
            f"frozen width {x_frozen.shape[-1]} != expected {self.frozen_dim}; "
            "graph dataset.x and encoder were built from different xd0 outputs"
        )
        if self.frozen_proj is not None:
            x_frozen = self.frozen_proj(x_frozen)
        return torch.cat([
            x_frozen,
            self.task_type_embedding(task_type_id),
            self.n_class_embedding(n_class_bucket_id),
            self.arity_embedding(arity_id),
        ], dim=-1)
