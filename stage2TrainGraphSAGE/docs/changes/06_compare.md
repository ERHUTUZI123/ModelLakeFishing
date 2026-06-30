# 06 — 配对比较 compare.py（新增）

**文件：** 新增 `stage2TrainGraphSAGE/compare.py`
**Phase：** 0（实现 guide 的 Acceptance rule）

## 问题

只有 ~42 个可评分 dataset，单次运行方差大，比较两配置时几千分位的差异不可信。
需要 guide 规定的"配对 bootstrap 在相同 dataset 上"且"head retrieval 不显著退化"的门槛判定。

## 原来代码

无（新文件）。

## 为什么这么改

guide Acceptance rule：仅当 ① Δtau_macro 的 95% 配对 bootstrap 区间排除 0
（或在 3 个固定 split 上一致复现）② head retrieval 不显著退化 ③ 分数仍可被 cosine/内积
ANN 复现 ④ 无泄漏，才提升一个改动。需要一个读取两个 artifact、按相同 (split,dataset)
tag 配对、给出 CI 与 head 退化检查的工具。

## 怎么改

读取两个 `<name>.json`，把逐 (split,init,dataset) 的 τ 按 `tag::dataset` 配对（保证完全对齐），
调用 `eval_harness.paired_bootstrap`，并逐项检查 head 聚合是否退化（regret 越低越好，其余越高越好）。

## 改后的代码（核心）

```python
def _pool_per_dataset(art, block, metric):
    out = {}
    for tag, per in art[block].items():               # tag = s{split}_i{init}
        for d, row in per.items():
            if metric in row: out[f"{tag}::{d}"] = {metric: row[metric]}
    return out

def main():
    a, b = _load(args.base), _load(args.cand)
    pa = _pool_per_dataset(a, "per_dataset_tau", "tau")
    pb = _pool_per_dataset(b, "per_dataset_tau", "tau")
    r = paired_bootstrap(pa, pb, metric="tau")
    gate = "EXCLUDES 0 (promote)" if (r["ci_low"] > 0 or r["ci_high"] < 0) else "includes 0"
    print(f"tau: delta={r['mean_delta']:+.4f} 95% CI [{r['ci_low']:+.4f},{r['ci_high']:+.4f}] "
          f"P(>0)={r['prob_positive']:.2f} -> {gate}")
    for k in ["hit@10","recall_top3@10","ndcg@10","ndcg@50","regret@10"]:
        av, bv = a["head_aggregate"][k][0], b["head_aggregate"][k][0]; d = bv-av
        worse = (d > 1e-4) if k.startswith("regret") else (d < -1e-4)
        print(f"{k}: base={av:.4f} cand={bv:.4f} delta={d:+.4f} {'REGRESS' if worse else 'ok'}")
```

用法：`python -m ...compare --base B0 --cand B5_ranknet`

## 实验结论

判定了所有关键门槛：B0→B5_ranknet **Δtau +0.205, CI [+0.122,+0.294], P>0=1.00 排除 0 → 提升**，
且 head 全面不退化。B5→P6_dm10 则量化了 Pareto（tau −0.070 显著、但 hit@10 +0.055）。
