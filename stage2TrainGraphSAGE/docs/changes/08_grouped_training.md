# 08 — dataset 分组宏平均训练 train_grouped（新增，被否）

**文件：** `stage2TrainGraphSAGE/train.py`（新增 `train_grouped` + `_full_masks`）
**Phase：** 3

## 问题

原训练把监督边**全局打乱**成 batch，排序损失再合并所有 dataset 的 pair 统一平均，
于是 pair 多的 dataset 主导梯度；但 `tau_macro` 是先逐 dataset 算 τ 再等权宏平均。
训练目标与评估聚合方式错位（pair-rich dataset 过度主导）。

## 原来代码

```python
def train(model, scorer, train_data, eli, target, M, comp, *, ...):
    loader = make_link_loader(train_data, eli, target, ...)     # 全局打乱监督边
    for epoch in range(epochs):
        for batch in loader:
            ...
            lrank = perf_ranking_loss(scorer, z, eli_b, el_b)   # 合并所有 pair 后平均
```

## 为什么这么改

guide Phase 3：用 dataset 分组 batch，每 dataset 先求 mean loss，再对 dataset 求 mean，
使每个 dataset 等权（与 tau_macro 一致）。在 2000-model 图上，全图前向很便宜（~2s/epoch），
所以可用"全图一步 + 对 dataset 宏平均的排序损失"精确实现宏平衡（`raw_dot_ranknet_loss` /
`perf_ranking_loss` 本就对 dataset 宏平均），并保留 edge dropout 的冷启动硬化。

## 怎么改

新增 `train_grouped`：每 epoch 对 train 消息图的克隆做 edge dropout，全图前向一次，
用**全部**监督边算排序损失（已是对 dataset 宏平均），对比损失用全 NxN 掩码（2000 规模可行），
按 epoch 记录参与 dataset 数 / pair 数。

## 改后的代码（核心）

```python
def _full_masks(M, comp):
    pos = (M @ M.t()) > 0; pos.fill_diagonal_(False)
    hub = comp.unsqueeze(0) == comp.unsqueeze(1); hub.fill_diagonal_(False)
    # 单例不算同 hub
    return pos, hub

def train_grouped(model, scorer, train_data, eli, target, M, comp, *, epochs=40,
                  rank_loss="ranknet", ...):
    base = train_data.clone().to(device); pos_mask, hub_mask = _full_masks(M, comp)
    for epoch in range(epochs):
        g = base.clone(); apply_edge_dropout(g, p=p, p_lineage=p_lineage)
        z = model(g)                                    # 全图一步
        lrank, st = raw_dot_ranknet_loss(z, eli, target, ..., return_stats=True)  # 对 dataset 宏平均
        lc = contrastive_loss(z["model"], pos_mask, hub_mask)
        total = lambda_rank*lrank + lambda_contrast*lc + (可选 dm_contrast/uniform)
        total.backward(); opt.step()
        # 记录 st["n_datasets"], st["n_pairs"]
```

## 实验结论（被否）

| 配置 | tau_macro | mean_cos | hit@10 |
|---|---|---|---|
| B2e_ctrl（per-batch）| 0.254 | 0.391 | 0.817 |
| **B4_grouped（full-batch, hinge）** | **0.175±0.025** | 0.070 | 0.738 |

全图训练显著变差且 mean_cos 坍到 0.07（近正交，丢失结构）——**per-batch 采样 + edge dropout
的随机性很重要**。kitchen-sink BEST（grouped+ranknet+heads）也被 grouped 拖累。
结论：**拒绝 grouped**。代码保留在 flag 后。宏平衡其实已由 `raw_dot_ranknet_loss` 的逐 dataset
平均在 per-batch 路径中实现。对应测试：`tests/test_phase3_macro.py`（逐 dataset 均值与 pair 数无关）。
