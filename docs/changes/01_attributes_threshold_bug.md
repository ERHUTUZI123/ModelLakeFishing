# 01 — dataset 相似度阈值被强制覆盖（明确 bug #1）

**文件：** `stage1BuildTransferGraph/attributes.py::GraphAttributes.get_dataset_edge_index`
**Phase：** 1 — 优先级最高，是 guide 中唯一确认的覆盖参数 bug。

## 问题

`get_dataset_edge_index` 函数签名接受 `threshold` 参数（调用方传 `args.distance_thres`），
但函数体第一行把它无条件改写成 `threshold = 1`。结果：剪枝条件 `attr > (1 - threshold)`
变成 `attr > 0`，即保留**几乎所有** dataset-dataset 边。在 hf1000d/2000m 上产生 130,680 条
`similar_to` 边；362 个 dataset 的完整有向无自环图最多 `362×361 = 130,682` 条，所以图事实上
接近全连接（每个 dataset 出度 ≈ 361）。近全连接的相似度图会在消息传播中造成破坏性平均
（每个 dataset 都从所有其他 dataset 聚合，区分度被抹平）。

## 原来代码

```python
def get_dataset_edge_index(self, threshold=0.3, base_dataset='imagenet', sim_method='cosine'):
    threshold = 1          # <-- 覆盖调用方传入的任何阈值
    ...
    # data normalization
    attr = np.asarray([(float(i) - min(attr)) / (max(attr) - min(attr)) for i in attr])
    # pruning：只保留 weight > 1 - threshold 的边（threshold=1 => 保留全部）
    index = np.where(attr > (1 - threshold))
    attr = attr[index]
    data_source = np.asarray(data_source)[index]
    data_target = np.asarray(data_target)[index]
```

调用点（同文件）：

```python
self.edge_index_dataset_to_dataset, self.edge_attr_dataset_to_dataset = self.get_dataset_edge_index(
    base_dataset=self.base_dataset,
    threshold=args.distance_thres,
    sim_method=args.dataset_distance_method
)
```

## 为什么这么改

1. **让配置真正生效**：删除 `threshold = 1`，使 `args.distance_thres` 恢复作用。
2. **用有界 top-k 替代全局阈值**（guide Phase 1 首选）：全局阈值对相似度分布敏感、难调；
   每个 dataset 保留相似度最高的 `k` 个邻居更稳健、度数可控。
3. **加度数断言**：图构建后若一个"稀疏"图意外接近全连接则直接失败，防止 bug 再次潜伏。

## 怎么改

- 删除 `threshold = 1`。
- 新增 `top_k` 参数（默认 `None`，未传时从 `args.dataset_top_k` 读取）。
- 当 `top_k` 给定：用对称相似度矩阵 `W`，每个源 dataset 取 top-k 目标（排除自环），
  相似度保留到 `edge_attr`，并断言 `max_degree <= k` 且 `< n-1`。
- 当 `top_k` 为 `None`：走原阈值路径，但 `threshold` 现在真实生效。
- 调用点显式传 `top_k=getattr(args, 'dataset_top_k', None)`。

## 改后的代码

```python
def get_dataset_edge_index(self, threshold=0.3, base_dataset='imagenet', sim_method='cosine',
                           top_k=None):
    # NOTE: 原先此处的 `threshold = 1` 会静默覆盖调用方阈值，使 dataset 图近全连接
    # （hf1000d/2000m 上 ~130k 条 similar_to 边）—— Kendall guide bug #1。已删除，
    # 使 threshold（及新的 top_k）真正生效。
    if top_k is None:
        top_k = getattr(self.args, 'dataset_top_k', None)
    ...
    attr = np.asarray([(float(i) - min(attr)) / (max(attr) - min(attr)) for i in attr])
    data_source = np.asarray(data_source)
    data_target = np.asarray(data_target)

    if top_k is not None:
        # 有界 top-k 图：对称权重矩阵，每个源取相似度最高的 top_k 个邻居（排除自环）
        W = np.zeros([n, n], dtype=float)
        W[data_source, data_target] = attr
        W[data_target, data_source] = attr
        np.fill_diagonal(W, -np.inf)
        kk = int(min(top_k, n - 1))
        src_list, tgt_list, w_list = [], [], []
        for i in range(n):
            for j in np.argsort(-W[i])[:kk]:
                if np.isfinite(W[i, j]):
                    src_list.append(i); tgt_list.append(int(j)); w_list.append(float(W[i, j]))
        data_source = np.asarray(src_list); data_target = np.asarray(tgt_list)
        attr = np.asarray(w_list)
        max_deg = int(np.bincount(data_source, minlength=n).max()) if data_source.size else 0
        assert max_deg <= kk, f"top-k graph degree {max_deg} exceeds k={kk}"
        assert max_deg < n - 1, (
            f"top-k graph is nearly complete (deg {max_deg} of {n-1}); check top_k")
    else:
        # 全局阈值：保留 weight > 1 - threshold 的边（threshold 现已真实生效）
        index = np.where(attr > (1 - threshold))
        attr = attr[index]; data_source = data_source[index]; data_target = data_target[index]
```

调用点：

```python
self.edge_index_dataset_to_dataset, self.edge_attr_dataset_to_dataset = self.get_dataset_edge_index(
    base_dataset=self.base_dataset,
    threshold=args.distance_thres,
    sim_method=args.dataset_distance_method,
    top_k=getattr(args, 'dataset_top_k', None),
)
```

## 实验结论

dense（保留全部）vs 删除 similar_to vs top-k(10)：dense 轻微有害（mean_cos 0.44、tau 0.193±0.059），
删除得到 0.212±0.027、collapse 最低。**但剪枝不是独立增益**（一旦换成 RankNet 损失，图选择影响很小）。
对应测试：`tests/test_phase1_graph.py`。
