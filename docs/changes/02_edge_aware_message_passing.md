# 02 — GNN 消息传播忽略 edge_attr（bug #2）

**文件：** `stage2TrainGraphSAGE/model.py`（`HeteroGraphSAGE`）+ 新增 `stage2TrainGraphSAGE/edge_aware.py`
**Phase：** 2

## 问题

`HeteroGraphSAGE.forward` 只把 `x_dict` 和 `edge_index_dict` 喂给 GNN，从不传 `edge_attr`。
`to_hetero(SAGEConv)` 本身也不消费边权。结果消息传播中：

```
similarity 0.95 == similarity 0.01
accuracy   0.91 == accuracy   0.55
强 lineage       == 弱 lineage
```

即所有 `edge_attr`（dataset 相似度、trained_on 准确率、lineage 有序权重）被丢弃，
同一关系的所有边在聚合时等权。这与图数据契约（边携带语义权重）错位。

## 原来代码

```python
# model.py 构造
self.gnn = to_hetero(_SAGEBackbone(hidden_channels, num_layers), metadata, aggr="sum")
self.head = nn.Linear(hidden_channels, out_dim)

# model.py forward
def forward(self, data) -> dict:
    x_dict = self.encode_nodes(data)
    h_dict = self.gnn(x_dict, data.edge_index_dict)        # <-- 没有 edge_attr
    return {nt: F.normalize(self.head(h), p=2, dim=-1) for nt, h in h_dict.items()}
```

## 为什么这么改

guide 要求"最小、可干净消融的 edge-aware 异构消息传播"，且测试需证明：
拓扑不变时改 `edge_attr` 会改变输出；把某关系权重置零会移除其邻居贡献。
同时要保留**显式 self/residual 路径**，使冷启动/稀疏节点不完全依赖邻居。

## 怎么改

新建 `edge_aware.py`：
- `WeightedSAGEConv`：SAGE 风格卷积，邻居消息按"每目标节点归一化"的边权缩放，
  显式 self 路径 `lin_self(x_dst)`；`lin_neigh` 设 `bias=False`，使无邻居/零权时
  精确退化为 self 路径（不引入幽灵 bias）。`edge_weight=None` 时退化为普通 mean。
- `EdgeAwareHetero`：每个关系一个 `WeightedSAGEConv` + 可学习**每关系门控**（init 1，
  可观测/可置零），目标节点对各关系输出求和（对齐 `to_hetero(aggr='sum')`）。

`HeteroGraphSAGE`：加 `edge_aware`/`weighted_relations` 开关（默认 False = 原 `to_hetero` 路径，
向后兼容）。`forward` 在 edge_aware 时构建 `edge_attr_dict` 并传入 GNN。

## 改后的代码

`edge_aware.py`（核心）：

```python
class WeightedSAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, *, normalize_weights=True):
        super().__init__(aggr="add")
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)  # 无邻居=>无幽灵bias
        self.lin_self = nn.Linear(in_channels, out_channels)               # 显式 self 路径
        self.normalize_weights = normalize_weights

    def forward(self, x, edge_index, edge_weight=None):
        x_src, x_dst = x if isinstance(x, (tuple, list)) else (x, x)
        num_dst = x_dst.size(0)
        if edge_weight is not None and self.normalize_weights:
            dst = edge_index[1]
            denom = torch.zeros(num_dst, ...).scatter_add_(0, dst, edge_weight.to(...))
            norm = edge_weight / (denom[dst] + 1e-12)        # 每目标归一化的相似度权
        elif edge_weight is not None:
            norm = edge_weight
        else:
            dst = edge_index[1]; deg = ...scatter_add_(0, dst, ones); norm = 1.0/(deg[dst]+1e-12)
        agg = self.propagate(edge_index, x=(x_src, x_dst), norm=norm, size=(...))
        return self.lin_self(x_dst) + self.lin_neigh(agg)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
```

`model.py`（构造 + forward）：

```python
# 构造：开关 + 解析 weighted_relations(中间名) -> 完整 edge_types
self.edge_aware = edge_aware
self.weighted_relations = list(weighted_relations) if weighted_relations is not None else None
if edge_aware:
    from ...edge_aware import EdgeAwareHetero
    rels = None
    if self.weighted_relations is not None:
        wr = set(self.weighted_relations)
        rels = [et for et in metadata[1] if et[1] in wr]
    self.gnn = EdgeAwareHetero(hidden_channels, metadata, num_layers=num_layers, relation_weights=rels)
else:
    self.gnn = to_hetero(_SAGEBackbone(hidden_channels, num_layers), metadata, aggr="sum")

# forward
def forward(self, data) -> dict:
    x_dict = self.encode_nodes(data)
    if self.edge_aware:
        edge_attr_dict = {et: data[et].edge_attr for et in data.edge_types
                          if getattr(data[et], "edge_attr", None) is not None}
        h_dict = self.gnn(x_dict, data.edge_index_dict, edge_attr_dict)
    else:
        h_dict = self.gnn(x_dict, data.edge_index_dict)
    ...
```

## 实验结论

edge-aware **结构**（self/residual 路径，不用权重）= tau 0.254±0.054（B0 0.198→0.254）——干净增益。
但**消费边权**在 hinge 下有帮助、在 RankNet 下反而有害（0.398→0.354）→ 最终保留 unweighted。
对应测试：`tests/test_phase2_edge_aware.py`（改 edge_attr 改变输出、置零关系移除贡献、checkpoint roundtrip）。
