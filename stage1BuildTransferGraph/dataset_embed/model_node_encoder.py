import torch
import torch.nn as nn


class ModelNodeEncoder(nn.Module):

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
            self.register_buffer("name_proj", proj)
        else:
            self.name_proj = None

        if use_desc and name_proj_dim is None:
            frozen_out = frozen_dim
        else:
            name_out = name_proj_dim if name_proj_dim is not None else name_dim
            frozen_out = name_out + ((frozen_dim - name_dim) if use_desc else 0)
        self.frozen_out_dim = frozen_out
        self.out_dim = (frozen_out + size_dim
                        + (family_dim if use_family else 0)
                        + (task_dim if num_model_tasks is not None else 0))

    def forward(
        self,
        x_frozen: torch.Tensor,
        size_bucket_id: torch.Tensor,
        family_id: torch.Tensor,
        task_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert x_frozen.shape[-1] == self.frozen_dim, (
            f"frozen width {x_frozen.shape[-1]} != expected {self.frozen_dim}; "
            "graph .x and encoder were built from different xm0 outputs"
        )
        if self.use_desc and self.name_proj is None:
            frozen_part = x_frozen
        else:
            name = x_frozen[:, :self.name_dim]
            if self.name_proj is not None:
                name = name @ self.name_proj
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
        x_frozen: torch.Tensor,
        task_type_id: torch.Tensor,
        n_class_bucket_id: torch.Tensor,
        arity_id: torch.Tensor,
    ) -> torch.Tensor:
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
