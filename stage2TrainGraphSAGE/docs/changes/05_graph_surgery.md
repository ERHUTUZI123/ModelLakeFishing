# 05 — 图手术 graph_surgery.py（新增）

**文件：** 新增 `stage2TrainGraphSAGE/graph_surgery.py`
**Phase：** 1

## 问题

Phase 1 的三组消融（删除 similar_to / top-k unweighted / weighted top-k）需要不同的
dataset 图。但从 stage-1 重建 hf1000d 图要跑很重的 embedding 流水线。而已构建的 `.pt`
里 `similar_to.edge_attr` 已经是归一化相似度（因为 bug #1 用 threshold=1 保留了全部边），
所以可以直接在加载好的图上做"手术"得到各变体，与重建结果一致。

## 原来代码

无（新文件）。

## 为什么这么改

- 避免重跑 stage-1：在内存图上派生变体，几秒完成，便于快速消融；
- 与修好的 builder（#01 的 top-k）选择**相同**（都基于同一归一化相似度做 top-k，
  已用合成矩阵测试两者选出相同邻居，见 `test_phase1_graph.py`）；
- 只动 dataset-dataset `similar_to`（消息结构），绝不碰 trained_on 监督或 lineage；
- 带度数断言，防止"稀疏"变体意外接近全连接。

## 怎么改

提供：
- `drop_similar_to(data)`：清空 similar_to（B1）；
- `topk_similar_to(data, k, weighted=True)`：每 dataset 保留 top-k 最相似邻居；
  `weighted=True` 保留相似度到 edge_attr（B3），`weighted=False` 全置 1（B2，仅拓扑）；
- `apply_similar_to_mode(data, mode, k)`：被 `ablation.py` 调度（dense/none/topk/topk_unweighted）。

## 改后的代码（核心）

```python
SIMILAR_TO = ("dataset", "similar_to", "dataset")

def topk_similar_to(data, k, *, weighted=True):
    data = data.clone(); n = data["dataset"].num_nodes
    ei = data[SIMILAR_TO].edge_index
    ea = getattr(data[SIMILAR_TO], "edge_attr", None)
    ea = torch.ones(ei.size(1)) if ea is None else ea
    W = torch.full((n, n), float("-inf"))
    W[ei[0], ei[1]] = ea.float(); W[ei[1], ei[0]] = ea.float()   # 对称
    W.fill_diagonal_(float("-inf"))                              # 排除自环
    kk = int(min(k, n - 1)); src, tgt, w = [], [], []
    for i in range(n):
        vals, idx = torch.topk(W[i], kk)
        for v, j in zip(vals.tolist(), idx.tolist()):
            if v != float("-inf"):
                src.append(i); tgt.append(j); w.append(v)
    edge_index = torch.tensor([src, tgt], dtype=torch.long)
    edge_attr = torch.tensor(w) if weighted else torch.ones(len(w))
    deg = torch.bincount(edge_index[0], minlength=n)
    assert int(deg.max()) <= kk and int(deg.max()) < n - 1       # 度数断言
    return _set_similar_to(data, edge_index, edge_attr)

def apply_similar_to_mode(data, mode, *, k=10):
    return {"dense": data, "none": drop_similar_to(data),
            "topk": topk_similar_to(data, k, weighted=True),
            "topk_unweighted": topk_similar_to(data, k, weighted=False)}[mode]
```

## 实验结论

支撑 Phase 1 三组消融（dense/none/topk）。结论：dense 轻微有害，但剪枝在 RankNet 损失下
非独立增益。最终 accepted 配置用 `topk_unweighted`（k=10）。
