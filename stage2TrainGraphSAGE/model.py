import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage1BuildTransferGraph.dataset_embed.model_node_encoder import (
    ModelNodeEncoder,
    DatasetNodeEncoder,
)

_HGRAPH_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "stage1BuildTransferGraph", "hgraph_zoo_xm0.pt")
)


def load_hgraph(path: str = _HGRAPH_PATH):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["data"], payload["xm0_meta"], payload["unique_model_id"]


class _SAGEBackbone(nn.Module):

    def __init__(self, hidden_channels: int, num_layers: int = 2):
        super().__init__()
        assert num_layers in (1, 2), "keep depth shallow: num_layers must be 1 or 2"
        self.convs = nn.ModuleList(
            SAGEConv(hidden_channels, hidden_channels) for _ in range(num_layers)
        )

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x


class HeteroGraphSAGE(nn.Module):

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
        num_task_types: int | None = None,
        n_class_buckets: int | None = None,
        num_arities: int | None = None,
        task_dim: int = 16,
        nclass_dim: int = 8,
        arity_dim: int = 4,
        dataset_frozen_proj_dim: int | None = None,
        edge_aware: bool = False,
        weighted_relations=None,
        separate_heads: bool = False,
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

        self.graph_metadata = metadata
        if num_layers == 0:
            self.gnn = None
        elif edge_aware:
            from ModelLakeFishing.stage2TrainGraphSAGE.edge_aware import EdgeAwareHetero
            rels = None
            if self.weighted_relations is not None:
                wr = set(self.weighted_relations)
                rels = [et for et in metadata[1] if et[1] in wr]
            self.gnn = EdgeAwareHetero(hidden_channels, metadata,
                                       num_layers=num_layers, relation_weights=rels)
        else:
            self.gnn = to_hetero(_SAGEBackbone(hidden_channels, num_layers), metadata, aggr="sum")

        if separate_heads:
            self.model_head = nn.Linear(hidden_channels, out_dim)
            self.dataset_head = nn.Linear(hidden_channels, out_dim)
        else:
            self.head = nn.Linear(hidden_channels, out_dim)

    def encode_nodes(self, data) -> dict:
        m = data["model"]
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
        x_dict = self.encode_nodes(data)
        if self.gnn is None:
            h_dict = x_dict
        elif self.edge_aware:
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
