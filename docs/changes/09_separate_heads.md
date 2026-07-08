# 09 — 模型/数据集分离投影头（新增，被否/中性）

**文件：** `stage2TrainGraphSAGE/model.py`（`HeteroGraphSAGE`）
**Phase：** 5

## 问题

model 与 dataset 两类节点共享同一个 final projection head。但二者特征语义和上游 encoder
完全不同，共享 head 是不必要的表达力限制（guide bug #7）。两个独立 projection 仍可通过
dot product 对齐到同一 metric space，且依旧是纯 MIPS、兼容 HNSW。

## 原来代码

```python
# 构造
self.head = nn.Linear(hidden_channels, out_dim)

# forward
return {nt: F.normalize(self.head(h), p=2, dim=-1) for nt, h in h_dict.items()}
```

## 为什么这么改

guide Phase 5：用分离的 ANN 兼容投影头替换共享头，单独比较 shared vs separate（其他设置固定），
且必须更新 checkpoint save/load 与 roundtrip；不能仅凭参与率改善就接受，必须 τ 或 head retrieval 提升。

## 怎么改

加 `separate_heads` 开关（默认 False，向后兼容）。True 时建 `model_head` / `dataset_head`，
forward 分别投影并 L2 归一化；分数仍为 `z_d · z_m`（单位向量内积，HNSW 几何不变）。
`learnable.py` 记录 `separate_heads` 并据此重建（见 #12）。

## 改后的代码（核心）

```python
# 构造
self.separate_heads = separate_heads
if separate_heads:
    self.model_head = nn.Linear(hidden_channels, out_dim)
    self.dataset_head = nn.Linear(hidden_channels, out_dim)
else:
    self.head = nn.Linear(hidden_channels, out_dim)

# forward
if self.separate_heads:
    return {"model":   F.normalize(self.model_head(h_dict["model"]),   p=2, dim=-1),
            "dataset": F.normalize(self.dataset_head(h_dict["dataset"]), p=2, dim=-1)}
return {nt: F.normalize(self.head(h), p=2, dim=-1) for nt, h in h_dict.items()}
```

## 实验结论（被否/中性）

| 配置 | tau_macro |
|---|---|
| B2e_ctrl（shared, hinge）| 0.254±0.054 |
| B6_heads（separate, hinge）| 0.172±0.021（退化）|
| R_heads（separate, RankNet）| 0.383±0.013（≈ B5 0.398，方差最低）|

separate heads 在 hinge 下退化、在 RankNet 下基本中性（不是增益）。按 guide"必须有 τ/head 提升
才接受"——**不采纳**为 accepted 配置（保留共享头）。代码保留在 flag 后。
对应测试：`tests/test_phase4_5_loss_heads.py`（分离头仍单位归一化、训练分数 == HNSW 内积几何）。
