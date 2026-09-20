import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as pyg_softmax


class WeightedSAGEConv(MessagePassing):

    def __init__(self, in_channels, out_channels, *, normalize_weights=True):
        super().__init__(aggr="add")
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.normalize_weights = normalize_weights

    def forward(self, x, edge_index, edge_weight=None):
        if isinstance(x, (tuple, list)):
            x_src, x_dst = x
        else:
            x_src = x_dst = x
        num_dst = x_dst.size(0)

        if edge_weight is not None and self.normalize_weights:
            dst = edge_index[1]
            denom = torch.zeros(num_dst, device=x_src.device, dtype=x_src.dtype)
            denom.scatter_add_(0, dst, edge_weight.to(x_src.dtype))
            norm = edge_weight.to(x_src.dtype) / (denom[dst] + 1e-12)
        elif edge_weight is not None:
            norm = edge_weight.to(x_src.dtype)
        else:
            dst = edge_index[1]
            deg = torch.zeros(num_dst, device=x_src.device, dtype=x_src.dtype)
            deg.scatter_add_(0, dst, torch.ones(edge_index.size(1), device=x_src.device, dtype=x_src.dtype))
            norm = 1.0 / (deg[dst] + 1e-12)

        agg = self.propagate(edge_index, x=(x_src, x_dst), norm=norm, size=(x_src.size(0), num_dst))
        return self.lin_self(x_dst) + self.lin_neigh(agg)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class EdgeAwareHetero(nn.Module):

    def __init__(self, hidden_channels, metadata, *, num_layers=1,
                 relation_weights=None, normalize_weights=True):
        super().__init__()
        self.node_types, self.edge_types = metadata
        self.num_layers = num_layers
        self.relation_weights = set(relation_weights) if relation_weights is not None else set(self.edge_types)
        self.layers = nn.ModuleList()
        self.gates = nn.ModuleList()
        for _ in range(num_layers):
            convs = nn.ModuleDict()
            gate = nn.ParameterDict()
            for et in self.edge_types:
                key = "__".join(et)
                convs[key] = WeightedSAGEConv(hidden_channels, hidden_channels,
                                              normalize_weights=normalize_weights)
                gate[key] = nn.Parameter(torch.ones(()))
            self.layers.append(convs)
            self.gates.append(gate)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        edge_attr_dict = edge_attr_dict or {}
        h_dict = x_dict
        for li in range(self.num_layers):
            convs, gate = self.layers[li], self.gates[li]
            out = {nt: torch.zeros(h_dict[nt].size(0), h_dict[nt].size(1),
                                   device=h_dict[nt].device, dtype=h_dict[nt].dtype)
                   for nt in self.node_types}
            counts = {nt: 0 for nt in self.node_types}
            for et in self.edge_types:
                src_t, _rel, dst_t = et
                key = "__".join(et)
                ei = edge_index_dict.get(et)
                if ei is None or ei.numel() == 0:
                    continue
                ew = edge_attr_dict.get(et) if et in self.relation_weights else None
                if ew is not None and ew.dim() > 1:
                    ew = ew.view(-1)
                msg = convs[key]((h_dict[src_t], h_dict[dst_t]), ei, ew)
                out[dst_t] = out[dst_t] + gate[key] * msg
                counts[dst_t] += 1
            new_h = {}
            for nt in self.node_types:
                new_h[nt] = out[nt] if counts[nt] > 0 else h_dict[nt]
            if li < self.num_layers - 1:
                new_h = {nt: torch.relu(v) for nt, v in new_h.items()}
            h_dict = new_h
        return h_dict

    def relation_gate_report(self):
        return [{"__".join(et): float(self.gates[li]["__".join(et)].detach())
                 for et in self.edge_types} for li in range(self.num_layers)]
