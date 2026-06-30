# Kendall action guide — 改动清单（每处改动一个文档）

按 guide 的 Phase 顺序，每个文件记录：**问题 / 原来代码 / 为什么这么改 / 怎么改 / 改后的代码**。
实验结论见 `../../artifacts/ablation/RESULTS.md`。

| # | 文件 | 涉及代码 | Phase | 性质 |
|---|---|---|---|---|
| 01 | [dataset 相似度阈值 bug](01_attributes_threshold_bug.md) | `stage1BuildTransferGraph/attributes.py` | 1 | 明确 bug 修复 |
| 02 | [GNN 忽略 edge_attr](02_edge_aware_message_passing.md) | `model.py` + 新 `edge_aware.py` | 2 | 契约错位修复 |
| 03 | [统一评测路径](03_eval_harness.md) | 新 `eval_harness.py` | 0 | 新增基础设施 |
| 04 | [消融驱动器](04_ablation_driver.md) | 新 `ablation.py` | 0 | 新增基础设施 |
| 05 | [图手术](05_graph_surgery.md) | 新 `graph_surgery.py` | 1 | 新增基础设施 |
| 06 | [配对比较](06_compare.md) | 新 `compare.py` | 0 | 新增基础设施 |
| 07 | [raw-dot RankNet 损失](07_ranknet_loss.md) | `losses.py` + `train.py` | 4 | 主胜负手 |
| 08 | [dataset 分组宏平均训练](08_grouped_training.md) | `train.py` | 3 | 新增（被否） |
| 09 | [模型/数据集分离投影头](09_separate_heads.md) | `model.py` | 5 | 新增（被否/中性） |
| 10 | [dataset→model 对比损失](10_dataset_to_model_contrastive.md) | `losses.py` + `train.py` | 6 | 服务关系（Pareto） |
| 11 | [验证集早停 + lr](11_early_stopping.md) | `train.py` + `ablation.py` | 7 | 新增 |
| 12 | [checkpoint 架构持久化](12_checkpoint_arch.md) | `learnable.py` | 2/5 | 兼容性 |
| 13 | [诊断 HNSW 方向修正](13_diagnostics_hnsw_fix.md) | `diverse_diagnostics.py` | 0/8.3 | 协议修正 |

**最终结论：** 单一主胜负手是 #07（raw-dot RankNet 损失，tau_macro 0.193→0.40，通过严格配对
bootstrap 门槛且 head retrieval 全面提升）。#02 的 self/residual 结构是干净的次级增益。
#01/#05 证明 dense 图轻微有害但剪枝非独立增益。#08/#09 被否。#10 是服务侧（head retrieval）
的 Pareto 变体。
