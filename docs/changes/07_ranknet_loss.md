# 07 — raw-dot RankNet 排序损失（主胜负手，bug #5）

**文件：** `stage2TrainGraphSAGE/losses.py`（新增 `raw_dot_ranknet_loss`）+ `train.py`（接入 `rank_loss` 选项）
**Phase：** 4

## 问题

原排序损失是在 `sigmoid(scale·dot + bias)` 上的 hinge，而非 HNSW 实际排序的 **raw dot**。
问题：① `scale/bias` 改变 margin=0.05 的含义、sigmoid 压缩梯度；② `scale` 未约束为正，
一旦为负排序方向与 HNSW 相反；③ pair 用 `torch.randint` 有放回采样、大量无效抽样；
④ `min_gap=1e-3` 把测量噪声当严格偏序。整体上损失与服务几何错位。

## 原来代码

```python
class PerfScorer(nn.Module):           # mode="dot"
    def forward(self, z_model, z_dataset, edge_label_index):
        s = (zm * zd).sum(-1)
        return torch.sigmoid(self.scale * s + self.bias)     # <-- sigmoid+scale/bias

def perf_ranking_loss(scorer, z_dict, edge_label_index, edge_label, *, margin=0.05,
                      max_pairs_per_dataset=256, min_gap=1e-3, ...):
    for d in datasets:
        p = torch.randint(n, (cap,)); q = torch.randint(n, (cap,))   # 有放回、含无效
        good = a[p] > a[q] + min_gap
    ...
    loss = torch.relu(margin - (s_hi - s_lo)).mean()                 # 在 sigmoid 分上 hinge
```

## 为什么这么改

guide Phase 4：排序损失必须作用在 HNSW 排序的那个**原始 raw dot**（单位向量内积）上，
去掉 sigmoid/scale/bias，使损失与检索分数之间没有任何单调代理；temperature 保持正；
测 `min_gap ∈ {0,0.005,0.01,0.02}`（把测量噪声当严格偏序会有害）；小列表枚举唯一对、
大列表挖**当前预测下的反序/近界硬对**；保留不加权版本（Kendall 对所有非平局对等权）。

## 怎么改

新增 `raw_dot_ranknet_loss(z_dict, eli, el, *, temperature, min_gap, ...)`：
- 直接用 `s = <z_m, z_d>`（单位向量），不经 scorer/sigmoid；
- 每 dataset 内枚举所有 `acc_i - acc_j > min_gap` 的有序对；超上限时按 `s_hi - s_lo`
  排序取硬对 head + 随机 tail；
- `L_d = mean softplus(-(s_hi - s_lo)/temperature)`，`L = mean_d L_d`（对 dataset 宏平均）；
- `temperature > 0` 断言；可选 `gap_weighted`，默认不加权。

`train.py`：加 `rank_loss="hinge"|"ranknet"` 开关，`lambda_rank` 块按选项调用对应损失。

## 改后的代码（核心）

```python
def raw_dot_ranknet_loss(z_dict, edge_label_index, edge_label, *, temperature=0.1,
                         min_gap=0.0, max_pairs_per_dataset=256, gap_weighted=False,
                         hard_frac=0.5, generator=None, return_stats=False):
    assert temperature > 0
    z_m, z_d = z_dict["model"], z_dict["dataset"]
    src, dst, acc = edge_label_index[0], edge_label_index[1], edge_label.float()
    s_all = (z_m[src] * z_d[dst]).sum(-1)              # raw dot，HNSW 排序的同一分数
    losses = []
    for d in torch.unique(dst).tolist():
        sel = (dst == d).nonzero().flatten(); a = acc[sel]; s = s_all[sel]
        better = (a.unsqueeze(1) - a.unsqueeze(0)) > min_gap     # 严格非平局对
        ii, jj = better.nonzero(as_tuple=True)
        if ii.numel() == 0: continue
        if ii.numel() > max_pairs_per_dataset:                  # 硬对挖掘：最反序优先
            margin = (s[ii] - s[jj]).detach(); order = torch.argsort(margin)
            ii, jj = 取 hard_frac*cap 个 head + 其余随机
        loss = F.softplus(-(s[ii] - s[jj]) / temperature)
        if gap_weighted: loss = loss * (a[ii]-a[jj]).clamp(max=1.0)
        losses.append(loss.mean())
    return torch.stack(losses).mean()                  # 对 dataset 宏平均

# train.py
if rank_loss == "ranknet":
    lrank = raw_dot_ranknet_loss(z, eli_b, el_b, temperature=rank_temperature, min_gap=rank_min_gap, ...)
else:
    lrank = perf_ranking_loss(scorer, z, eli_b, el_b, margin=rank_margin)
```

## 实验结论（主胜负手）

| 配置 | tau_macro | hit@10 | 备注 |
|---|---|---|---|
| B0 hinge | 0.193±0.059 | 0.809 | 基线 |
| RankNet（to_hetero）| 0.315 | 0.857 | 仅换损失 = +0.12 |
| **RankNet + edge-aware（min_gap 0.02）** | **0.403±0.022** | 0.857 | accepted |

**B0→RankNet 配对 bootstrap：Δtau +0.205, 95% CI [+0.122,+0.294], P>0=1.00 排除 0 → 提升**，
head retrieval 全面提升。min_gap 扫描：0.0→0.363（最噪）, 0.01→0.398, **0.02→0.403（最佳最稳）**。
对应测试：`tests/test_phase4_5_loss_heads.py`（raw dot 按准确率正确排序、min_gap 跳过近平局、temperature 正性）。
