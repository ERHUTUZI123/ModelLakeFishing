# 03 — 统一评测路径 eval_harness.py（新增）

**文件：** 新增 `stage2TrainGraphSAGE/eval_harness.py`
**Phase：** 0 — 所有消融复用同一条评测路径。

## 问题

原代码的评测分散且与生产目标错位：
1. `train.eval_perf` 只给聚合 `tau_macro`，不返回**逐 dataset** 值，无法做配对检验；
2. 没有**精确 z_d→z_m head retrieval**（生产真正的查询：给 dataset 检索最优 model）的指标；
3. `split_trained_on(seed=seed)` 让同一 seed 同时决定 split 和初始化，配置间不可配对比较；
4. 没有配对 bootstrap 不确定性估计（只有 ~42 个可评分 dataset，几千分位的差异不可信）。

## 原来代码

无（新文件）。相关旧逻辑散落在 `train.eval_perf`（仅聚合 tau）和 `train.hnsw_recall`
（错误地用 z_m 查询 z_m，见 13_diagnostics_hnsw_fix.md）。

## 为什么这么改

guide "Split discipline" + "Primary metrics" 要求：固定 split 复用、split_seed 与 init_seed 分离、
逐 dataset 指标、精确 z_d→z_m head retrieval（Hit@K / Recall@K / NDCG@K / regret@K）、
配对 bootstrap。需要一个被每个消融复用的单一评测模块来保证可比性与无泄漏。

## 怎么改

新建模块，提供：
- `make_fixed_splits(data, split_seed=...)`：一次性物化 train/val/test，供所有配置复用；
- `per_dataset_tau(...)`：逐 dataset within-dataset Kendall τ + 候选数 + 可比非平局对数；
- `head_retrieval(...)`：对每个 test dataset 的**已观测 held-out 候选**用精确 `<z_d,z_m>`
  排序，算 hit@K、recall_top3@K、recall_top10pct@K、ndcg@K、regret@K，逐 dataset 后宏平均；
- `dataset_to_model_hnsw_recall(...)`：索引全部 z_m、用 z_d 查询的 ANN 保真度（见 13_…）；
- `model_to_model_hnsw_recall(...)`：旧 z_m→z_m 诊断改名保留；
- `paired_bootstrap(...)`：对相同 dataset 的逐 dataset 指标差做 95% 配对 bootstrap CI。

评测宇宙明确写在 docstring：test dataset d 的候选 = 与 d 有 held-out trained_on 边的 model
（已观测但未训练其性能），非全 lake recall、无泄漏。

## 改后的代码（关键函数签名 + 要点）

```python
def make_fixed_splits(data, *, split_seed, num_val=0.1, num_test=0.2,
                      neg_ratio=1.0, disjoint_train_ratio=0.3):
    return split_trained_on(data, num_val=..., seed=split_seed)   # split 仅由 split_seed 决定

def per_dataset_tau(scorer, z_dict, split_data, lookup, *, min_per_dataset=3):
    # 返回 (macro_tau, {dataset_idx: {tau, n_candidates, n_comparable_pairs}})

def head_retrieval(z_dict, split_data, lookup, *, ks=(10,50,100,200), top_frac=0.10):
    z_m = F.normalize(z_dict["model"]); z_d = F.normalize(z_dict["dataset"])
    for d in 各 test dataset:
        cand = 该 d 的 held-out 候选 model; a = 其归一化准确率
        score = z_m[cand] @ z_d[d]                 # 精确 MIPS（HNSW 排序的同一几何）
        order = argsort(-score); truth = argsort(-a)
        # hit@K / recall_top3@K / recall_top10pct@K / ndcg@K / regret@K
    # 逐 dataset 后宏平均

def paired_bootstrap(per_a, per_b, metric="tau", *, n_boot=10000, seed=0):
    common = 两配置都评分的 dataset
    diff = b - a; 重采样 -> {mean_delta, ci_low, ci_high, prob_positive}
```

## 实验结论

这是 Phase 0 的门控：所有后续消融（B0..P7）都通过 `ablation.py` 调用本模块，
配对 bootstrap 是 RankNet（07_ranknet_loss.md）通过严格门槛（CI [+0.122,+0.294]、P>0=1.00）的依据。
关键再框定：head retrieval 在 baseline 已很强（hit@10≈0.81），tau_macro 才是更难的信号。
