# v5 P4 可审计记录 —— gold-gap@K 落地评测 harness(共同主指标)

**日期:** 2026-07-19
**上游方案:** `../RETRIEVAL_QUALITY_REMEDIATION_PLAN.md` §0.3 P4 / §0.6-3
**δ 依据:** P0(`P0_DIAGNOSIS.md` §2.3)实测——70% 数据集第 2 名在 gold−0.01 内、
gold/2nd 中位差 0.0017、P25 精确并列;严格 gold 低估真实服务效用 ~3 倍。
**性质:** 加一个共同主指标,**gold@K 球门柱不动**。

---

## 1. 语义定义

严格 gold@K 问:"**那一个**标签最高的模型是否活进全湖 top-K"。
gold-gap@K 把"那一个"松弛为"**一个足够好的**":

```
near(d)      = {候选模型 m : acc(m) ≥ gold_acc(d) − δ}   (含 gold 自身,gap 0)
gap_rank(d)  = min_{m ∈ near(d)}  globalRank(m)          (tie-safe,与 gold_rank 同口径)
gold-gap@K   = 1[gap_rank ≤ K]
δ = 0.01(GOLD_GAP_DELTA,代码常量,可审计)
```

**数学性质:** gold ∈ near,故 `gap_rank ≤ gold_rank` 恒成立 →
**gold-gap@K 是 gold@K 的严格松弛(下界不劣)**。它回答产品问题
("top-K 里有没有一个够用的模型"),而非学术问题("那唯一最优在不在")。

## 2. 代码改动(一处源头,全链继承)

| 文件 | 改动 |
|---|---|
| `stage2TrainGraphSAGE/top1_eval.py` | 加常量 `GOLD_GAP_DELTA=0.01`;`five_metric_eval` 每数据集算 `gold_gap_rank` + `gold_gap@K` + `n_within_gap_delta`;`aggregate` 输出 `gold_gap@K` + `median_gold_gap_rank`;`random_baselines` 加 gold-gap 的超几何解析期望 |
| `stage2TrainGraphSAGE/w1_dzero.py` | `PRIMARY_PLUS = PRIMARY + ["gold_gap@10"]`;root_macro / flat / pooled(配对 bootstrap)/ MD 表全部报告 gold-gap@10。**gate() 仍用 PRIMARY 的严格 gold@10**(球门柱不动) |

`gold_rank` 是 `five_metric_eval` 里唯一的 gold 计算源头,gold-gap 紧邻它计算,
因此所有下游(aggregate / root_macro / paired_bootstrap / 各 phase runner /
Stage-3 replay)零改动自动继承。

## 3. 验证

**机制测试(合成,断言全过):** gap_rank ≤ gold_rank;gold-gap@K ≥ gold@K;
aggregate 与 random_baselines 均携带 gold_gap;植入"第 2 名差 0.005"后
near-set ≥2。

**真实读数(s0 冻结 checkpoint `L1L3b_s0_i0.pt` 过新 harness,零重训):**

| 视图 | gold@10 | **gold-gap@10** | 中位 gold rank | 中位 gap rank |
|---|---|---|---|---|
| flat | 0.118 | **0.551** | 36 | **4** |
| root_macro | 0.186 | **0.351** | — | — |
| 随机全湖基线 | 0.001 | 0.011 | — | — |

与 P0 的 s0 spot-check(root 0.351 / flat 0.551)**逐位一致**——metric 复现可信。
**最醒目的数字:中位 gold rank 36 → 中位 gap rank 4。** 严格口径下"gold"典型
排在第 36 位(掉出 top-10),但一个"差 ≤1%"的够用模型典型排在**第 4 位**
(稳进 top-10)。这量化了 §0.2-E16 的判断:系统的真实服务效用远好于
严格 gold@10 显示的,低估集中在近平局根上。

## 4. 复现

```bash
cd ModelLakeFishing  # 仓库根 codes/
# 机制:任何走 five_metric_eval 的评测即自动产出 gold_gap@K
ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.stage2TrainGraphSAGE.w1_dzero \
    --rows L1L3b --seeds 0   # 报表新增 gold_gap@10 列
```

复核片段:

```python
from ModelLakeFishing.stage2TrainGraphSAGE.top1_eval import GOLD_GAP_DELTA
assert GOLD_GAP_DELTA == 0.01
import json
r = json.load(open(r'ModelLakeFishing\stage2TrainGraphSAGE\artifacts\ablation\d0\p0_funnel\p4_goldgap_s0.json'))
assert abs(r['root_macro']['gold_gap@10'] - 0.351) < 1e-3     # P0 复现
assert r['flat']['median_gold_gap_rank'] == 4.0
```

## 5. 对 v5 后续的作用

- **gold-gap@10 即日进所有 D0 报表**;组合行(P1×P3)的 8-seed 终裁同时看
  gold@10(球门柱)与 gold-gap@10(产品效用),两者都进配对 bootstrap;
- 现任 L1L3b 基线锚点:**root gold-gap@10 = 0.148 ± 0.093(8-seed,已实测)**
  ——注意 §3 表里的 s0 单 seed 值 0.351/0.551 是**高种子**,8-seed 真实均值
  低得多(root 严格 gold@10 0.100 → gold-gap 0.148,近平局松弛 +0.048);
  flat 口径松弛更大(大兄弟家族权重高),但 root 口径才是公平判据;
- 承诺对照(§0.6-5):关闭 gold@10 0.100→0.563 差距 25–40% 的同时,
  gold-gap@10 预期从 0.35 抬向 0.45+。