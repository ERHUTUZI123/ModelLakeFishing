# 10 — dataset→model 对比损失（服务关系，Pareto 变体）

**文件：** `stage2TrainGraphSAGE/losses.py`（新增 `dataset_to_model_contrastive` +
`dataset_to_model_contrastive_from_edges`）+ `train.py`（接入）
**Phase：** 6

## 问题

原对比损失只训练 `z_m → z_m`（让共同高性能模型互相靠近），它**不直接**训练生产查询关系
`z_d → z_m`（给 dataset 检索最优 model）。线上 HNSW 实际是用 z_d 查询 z_m 索引，二者错位（bug #6）。

## 原来代码

```python
def contrastive_loss(z_model, pos_mask, hub_mask, *, temperature=0.2, hard_neg_weight=2.0):
    sim = (z_model @ z_model.t()) / temperature        # 只有 z_model，没有 z_dataset
    ...
```

## 为什么这么改

guide Phase 6：在排序流水线修正后，增加 dataset→model 对比目标，直接训练服务关系。
positive = 该 dataset 训练可见的高性能 model；可靠 negative = 已观测低性能 model；
**不把未观测 pair 当负例**（missing ≠ negative）。由于 grouped 路径被否（#08），dm-contrast
必须在获胜的 per-batch 路径里也能用，故再写一个从 batch 监督三元组直接构建的版本。

## 怎么改

- `dataset_to_model_contrastive(z_dict, ti, M, ...)`：全图版，候选 = 训练可见 trained_on 的
  model，positive = M[:,d]，InfoNCE：`-log( Σ_pos e^{s/τ} / Σ_cand e^{s/τ} )`，对 dataset 宏平均；
- `dataset_to_model_contrastive_from_edges(z_dict, eli, el, top_frac, ...)`：**per-batch 版**，
  每 dataset 的候选 = 其 batch 监督 model，positive = 按准确率 top ceil(top_frac·n)，
  其余观测候选为可靠负例；接入 per-batch `train()`（获胜路径）；
- 两版都只用观测候选（不把缺失当负）、高分负例在 InfoNCE 分母里自然成为硬负例。

## 改后的代码（per-batch 版核心）

```python
def dataset_to_model_contrastive_from_edges(z_dict, edge_label_index, edge_label, *,
                                            top_frac=0.10, temperature=0.1, min_models=2):
    z_m, z_d = z_dict["model"], z_dict["dataset"]
    src, dst, acc = edge_label_index[0], edge_label_index[1], edge_label.float()
    losses = []
    for d in torch.unique(dst).tolist():
        sel = (dst == d).nonzero().flatten(); n = int(sel.numel())
        if n < min_models: continue
        models_d, a = src[sel], acc[sel]
        kk = max(1, round(top_frac*n))
        if kk >= n: continue                            # 需同时有正负
        pos = topk(a, kk) -> pos_flag
        s = (z_m[models_d] @ z_d[d]) / temperature      # z_d 查询 -> 候选 z_m（服务几何）
        losses.append(logsumexp(s) - logsumexp(s[pos_flag]))   # 仅观测候选为负
    return torch.stack(losses).mean()                   # 对 dataset 宏平均

# train.py（per-batch 块）
if lambda_dm_contrast > 0:
    ldm = dataset_to_model_contrastive_from_edges(z, eli_b, el_b, top_frac=dm_top_frac, ...)
    total = total + lambda_dm_contrast * ldm
```

## 实验结论（Pareto：服务侧增益、tau 代价）

| 配置 | tau_macro | hit@10 | recall_top3@10 |
|---|---|---|---|
| B5_ranknet | 0.398 | 0.850 | 0.841 |
| **P6_dm10（+dm 1.0）** | 0.328 | **0.905** | **0.902** |

**B5→P6_dm10 配对：tau −0.070（CI [−0.137,−0.004] 显著下降），但 hit@10 +0.055、recall_top3 +0.060。**
即 dm-contrast 直接优化服务查询，锐化检索列表顶部，代价是全序一致性。
**定位：服务/head 优先的变体**（"给 dataset 检索最优 model"是字面目标时用）；否则 RankNet-only 取 tau。
