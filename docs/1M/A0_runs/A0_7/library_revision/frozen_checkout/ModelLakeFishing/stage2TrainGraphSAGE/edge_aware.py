"""
edge_aware.py -- Phase 2: edge-attribute-aware heterogeneous message passing.

The shipped model (model.HeteroGraphSAGE with to_hetero(SAGEConv)) NEVER passes
edge_attr, so similarity 0.95 == similarity 0.01, accuracy 0.91 == 0.55, strong
lineage == weak lineage (Kendall action guide bug #2). This module is the
smallest cleanly-ablatable edge-aware replacement:

  WeightedSAGEConv  -- a SAGE-style conv with an explicit self/residual path and
                       neighbour messages scaled by a per-destination-normalized
                       edge weight. edge_weight=None -> plain unweighted SAGE
                       (so a relation with no meaningful attr is unaffected).
  EdgeAwareHetero   -- per-relation WeightedSAGEConv with a learnable per-relation
                       gate (init 1) so each relation's contribution is inspectable
                       and a relation can be zeroed; destination aggregation sums
                       gated relation outputs (matching to_hetero aggr='sum').

Invariants the Phase 2 tests assert:
  * changing edge_attr while holding topology fixed changes the output;
  * zeroing a relation's edge weight removes its neighbour contribution (only the
    self path of that relation survives);
  * with all edge_weight=None and gates=1 the output matches a plain SAGE pass in
    spirit (unweighted mean aggregation + self path).
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as pyg_softmax  # noqa: F401  (available if needed)


class WeightedSAGEConv(MessagePassing):
    """GraphSAGE-mean with optional per-edge weights and an explicit self path.

    out_i = W_self . x_dst_i  +  W_neigh . (sum_j a_ij x_j)
    where a_ij = w_ij / (sum_{j'} w_ij' + eps)  (per-destination normalized) when
    edge_weight is given, else a_ij = 1/deg_i (plain mean).
    """

    def __init__(self, in_channels, out_channels, *, normalize_weights=True):
        super().__init__(aggr="add")
        # bias=False on the neighbour path so a destination with no (or fully
        # zero-weighted) in-edges reduces EXACTLY to its self path -- a cold node
        # never receives a phantom bias from an empty neighbourhood.
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.normalize_weights = normalize_weights

    def forward(self, x, edge_index, edge_weight=None):
        # x is a (x_src, x_dst) tuple under HeteroConv-style bipartite calls.
        if isinstance(x, (tuple, list)):
            x_src, x_dst = x
        else:
            x_src = x_dst = x
        num_dst = x_dst.size(0)

        if edge_weight is not None and self.normalize_weights:
            # normalize weights over the in-edges of each destination
            dst = edge_index[1]
            denom = torch.zeros(num_dst, device=x_src.device, dtype=x_src.dtype)
            denom.scatter_add_(0, dst, edge_weight.to(x_src.dtype))
            norm = edge_weight.to(x_src.dtype) / (denom[dst] + 1e-12)
        elif edge_weight is not None:
            norm = edge_weight.to(x_src.dtype)
        else:
            # plain mean: weight 1 per edge, normalized by in-degree
            dst = edge_index[1]
            deg = torch.zeros(num_dst, device=x_src.device, dtype=x_src.dtype)
            deg.scatter_add_(0, dst, torch.ones(edge_index.size(1), device=x_src.device, dtype=x_src.dtype))
            norm = 1.0 / (deg[dst] + 1e-12)

        agg = self.propagate(edge_index, x=(x_src, x_dst), norm=norm, size=(x_src.size(0), num_dst))
        return self.lin_self(x_dst) + self.lin_neigh(agg)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class EdgeAwareHetero(nn.Module):
    """One WeightedSAGEConv per relation + learnable per-relation gate; destination
    nodes sum gated relation outputs (a residual self path lives inside each conv).

    metadata : (node_types, edge_types) like data.metadata().
    relation_weights : set of edge_types whose edge_attr should be CONSUMED as a
                       weight. Relations not listed are passed edge_weight=None
                       (topology-only) -- e.g. reverse relations with no meaningful
                       attr, unless explicitly included.
    """

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
            # accumulate gated per-relation outputs at each destination node type
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
            # destinations that received no relation keep a learned self transform
            # via any relation; if a node type had zero in-relations this layer,
            # fall back to its previous representation (no update).
            new_h = {}
            for nt in self.node_types:
                new_h[nt] = out[nt] if counts[nt] > 0 else h_dict[nt]
            if li < self.num_layers - 1:
                new_h = {nt: torch.relu(v) for nt, v in new_h.items()}
            h_dict = new_h
        return h_dict

    def relation_gate_report(self):
        """Current gate magnitude per relation per layer (inspection)."""
        return [{"__".join(et): float(self.gates[li]["__".join(et)].detach())
                 for et in self.edge_types} for li in range(self.num_layers)]
