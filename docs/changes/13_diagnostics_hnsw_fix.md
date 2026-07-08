# 13 — 诊断 HNSW 查询方向修正 + head retrieval（bug #8.3）

**文件：** `stage2TrainGraphSAGE/diverse_diagnostics.py`（+ 复用 `eval_harness` 的新函数）
**Phase：** 0 / guide 8.3

## 问题

`train.hnsw_recall` 索引 z_m、用 **z_m 查询 z_m**，验证的是 `z_m → z_m`。但生产检索是
`z_d → z_m`（用 dataset 向量查询 model 索引）。诊断没有验证真正的服务路径；
且没有报告"精确 top-K 是否包含真实性能 top model"的语义检索质量。

## 原来代码

```python
# diverse_diagnostics.py
from ...train import collapse_report, eval_perf, oversmoothing_report, hnsw_recall
...
rec = hnsw_recall(z_m, near_hub, k=50)                 # z_m -> z_m（错误关系）
out["hnsw_recall@50"] = rec if rec is not None else "skipped ..."
return out
```

`train.hnsw_recall`（z_m 索引、z_m 查询、按 near/away hub 分）。

## 为什么这么改

guide 8.3：把旧函数改名为明确的 `model_to_model_hnsw_recall`（仅作结构诊断），
新增真正的 `dataset_to_model_hnsw_recall`（索引全部 z_m、用 z_d 查询，比对 HNSW top-K
与精确 dot top-K 的 ANN 保真度）。并分别报告 ANN 保真度与语义检索（精确 head retrieval）——
前者高不能证明后者高。

## 怎么改

- 在 `eval_harness.py` 提供 `model_to_model_hnsw_recall`（旧逻辑改名）与
  `dataset_to_model_hnsw_recall`（z_d 查询 z_m，对比精确 dot top-K）；
- `diverse_diagnostics.trained_metrics` 改为同时报告两者 + 精确 z_d→z_m `head_retrieval`。

## 改后的代码（核心）

```python
# eval_harness.py
def dataset_to_model_hnsw_recall(z_dict, *, k=50, query_datasets=None):
    z_m = normalize(z_dict["model"]); z_d = normalize(z_dict["dataset"])
    index.add_items(z_m, arange(N))                    # 索引 z_m
    q = z_d[query_datasets or :]                       # 用 z_d 查询
    exact = argsort(-(q @ z_m.T))[:, :k]; ann, _ = index.knn_query(q, k=k)
    return mean(overlap(ann, exact))                   # ANN 保真度

# diverse_diagnostics.py
rec_mm = model_to_model_hnsw_recall(z_m, near_hub, k=50)        # 结构诊断（非服务）
out["model_to_model_hnsw_recall@50"] = rec_mm or "skipped ..."
rec_dm = dataset_to_model_hnsw_recall(z, k=50)                  # 真正的服务路径
out["dataset_to_model_hnsw_recall@50"] = rec_dm or "skipped ..."
head_macro, _ = head_retrieval(z, test_data, lookup)           # 精确语义检索
out["head_retrieval"] = {k: round(v, 4) for k, v in head_macro.items()}
```

## 实验结论

在 accepted checkpoint 上端到端验证：诊断正确报告 `tau_macro 0.373`、
`head_retrieval.hit@10 0.833 / ndcg@50 0.846`，并新增 `dataset_to_model_hnsw_recall@50` 字段
（本环境未装 hnswlib 故 skipped，精确 head retrieval 已覆盖语义检索）。

注（非目标）：guide 明确 "不要把装/调 HNSW 当作修语义检索的替代"。精确 dot 才是 ground truth，
HNSW 保真度只是 ANN 实现的二级检查。
