# ModelLens 发布版部署推荐质量重评估

**执行日期：** 2026-07-28  
**主代码：** `scale/deployment_quality_metrics.py`  
**结果：** `docs/scale/P5/artifacts/deployment_quality_metrics.json`  
**状态：** 完成；本报告取代 `list_quality_metrics.json` 作为“实际推荐可用性”的主口径。

## 0. 结论

旧评估回答的是“补齐表示并由我们重训后，ModelLens 架构能否在已标注候选中学习排序”；本评估回答的是“用户从已发布 ModelLens checkpoint 能否得到可用的全局 top-10 推荐”。两者不是同一个问题，不能共享结论。

在同一 **12,000 模型候选池**、同一 **517 个 root-aware held-out 查询**上，发布件可复现的 ModelLens 推荐质量发生系统性失效：

- **90.33%** 的查询 top-10 完全没有任何该数据集的公开结果；
- **94.20%** 的查询 top-10 没有一个已证实距最优不超过 0.01 的模型；
- top-10 中仅 **11.99%** 有精确同任务训练证据，**72.94%** 有证据但只来自其他任务；
- **89.17%** 的查询至少一半榜单为已知偏题模型，任务纯净查询为 **0%**；
- gold@10 失败率 **98.84%**，gold 中位排名 **2,418 / 12,000**；
- 对 task 与 metric 相同、但数据集不同的 3,014 对查询，top-10 **100% 完全相同**，数据集特异性为 **0**。

Model Lake 在相同口径下也并非完美，但显著更可用：verified precision **37.62% vs 3.17%**、certified near-optimal precision **15.20% vs 2.75%**、certified utility lower bound **33.25% vs 3.09%**、gold 中位排名 **21 vs 2,418**。

因此可辩护的结论是：

> **ModelLens 的公开发布件不能产生可靠、数据集特异的 top-10 推荐。它在相同 task+metric 下复用完全相同的榜单，且绝大多数查询没有可验证或近优推荐。就用户能够实际复现和使用的推荐器而言，Model Lake 显著优于 ModelLens released system。**

该结论不外推到“掌握未发布描述矩阵和私有训练流水线的论文内部满血系统”。

## 1. 为什么旧指标被撤出主表

旧的 `labeled_nDCG@10`、Spearman 和 Kendall 先把没有真实成绩的全局干扰项移除，再在已标注候选中评排序。它们可以在实际 top-10 被无关模型占满时仍然很高。`best_observed_regret@10` 只在至少命中一个已观测模型的查询上计算，同样不会惩罚“九个垃圾候选加一个不错候选”。Pointwise MAE/ECE 则评估由我们重训的另一个 head，不是 demo 返回的 ranking score。

这些值不是算错，而是回答了与部署推荐质量不同的问题。它们仅可作为“重训代理在已标注闭集内的诊断”，不得再用于称赞实际 ModelLens 推荐质量。

## 2. 评估对象与公平控制

| 项目 | 口径 |
|---|---|
| 候选宇宙 | 两边完全相同的 12,000 个模型 |
| 查询 | 517 个 root-aware held-out 数据集 |
| Model Lake | held-out `z_m_eval/z_d_eval`，纯 MIPS |
| ModelLens | 官方源码 + 已发布 `ModelLens.pt`，真实 global model IDs |
| ModelLens 缺失项 | `dataset_desc_matrix` 与 `dataset2id` 未发布 |
| 实际后果 | 发布版只能按 task、metric 和模型特征排序，看不见具体数据集 |
| 缺失成绩 | 默认记为 unknown；只在明确命名的 certified lower bound 中按 0 计 |

这里比较的是**发布/部署可复现性**，不是对论文私有系统的同条件方法复现。由我们补齐表示并重训的 ModelLens 保留为架构反事实实验，不进入本报告的产品推荐质量主表。

## 3. 新主指标：实际 top-10 部署质量

| 指标 | 定义 | Model Lake | ModelLens released |
|---|---|---:|---:|
| `verified_precision@10` | top-10 中有该查询数据集真实成绩的比例 | **0.3762** | 0.0317 |
| `all_unverified_query_rate@10` | top-10 完全无真实成绩的查询比例，越低越好 | **0.3095** | 0.9033 |
| `certified_near_optimal_precision@10` | top-10 中真实成绩距观测最优 ≤0.01 的比例 | **0.1520** | 0.0275 |
| `no_certified_near_optimal_query_rate@10` | top-10 没有任何已证近优项的查询比例，越低越好 | **0.4913** | 0.9420 |
| `certified_utility_lower_bound@10` | 已验证项相对该查询最优值的整榜平均；未验证项只在本下界中按 0 计 | **0.3325** | 0.0309 |
| `exact_task_evidence@10` | top-10 中有非 held-out 精确同任务证据的比例 | **0.6663** | 0.1199 |
| `known_off_task@10` | 有训练证据、但证据全部来自其他任务的比例，越低越好 | **0.3325** | 0.7294 |
| `no_exact_task_evidence@10` | 无精确同任务证据的比例，越低越好 | **0.3337** | 0.8801 |
| `task_pure_query_rate@10` | 至少 8/10 推荐有精确同任务证据的查询比例 | **0.4565** | 0.0000 |
| `majority_known_off_task_query_rate@10` | 至少 5/10 为已知偏题模型的查询比例，越低越好 | **0.3250** | 0.8917 |
| `gold_miss_rate@10` | 最优观测模型未进入 top-10 的查询比例，越低越好 | **0.5841** | 0.9884 |
| `top3_miss_rate@10` | 观测 top-3 全部未进入 top-10 的查询比例，越低越好 | **0.4681** | 0.9420 |
| `median_gold_rank` | 最优观测模型的中位全局名次，越低越好 | **21** | 2,418 |

这些指标全部直接查看实际返回的全局 top-10，不先移除无标签或偏题干扰项。

## 4. 查询响应性与榜单坍缩

仅看命中仍不足以解释 demo 为什么反复返回同一撮模型，因此增加榜单响应性审计。

| 指标 | Model Lake | ModelLens released |
|---|---:|---:|
| top-10 共使用的不同模型数 | **766** | 78 |
| `catalog_coverage@10` | **0.0638** | 0.0065 |
| `unique_top10_list_fraction` | **0.9246** | 0.2205 |
| `top1_max_query_share`，越低越好 | **0.1238** | 0.3907 |
| 同 task+metric 查询对 top-10 平均 Jaccard，越低越有区分度 | **0.4472** | 1.0000 |
| 同 task+metric 查询对完全相同榜单比例，越低越好 | **0.0325** | 1.0000 |
| `dataset_specificity@10 = 1 - mean Jaccard` | **0.5528** | 0.0000 |

ModelLens 的 `same_task_metric_identical_list_pair_rate@10 = 1.0` 不是统计波动：由于发布件没有数据集描述矩阵与 ID 映射，相同 task+metric 的查询输入完全相同，所以榜单按结构必然碰撞。它无法区分同一任务下的不同数据集。

## 5. 实际 demo 复核：MTEB STS17 (fr-en)

用户在公开 demo 中输入本地原始描述与 task type `STS`，得到的 top-10 为 Skywork role-play LLM、日文 instruction LoRA、Mistral Instruct、WNUT-17 NER、SantaCoder、英文 SimCSE、FSG-Net 图像分割和英文 BGE 等模型。

审计结果：

- 该数据集真实成绩覆盖：**0/10**；
- 严格法英跨语言 STS 兼容：**0/10**；
- 宽松“属于句向量模型”口径：仅 **2/10**，但两者都是英文模型，不满足 `fr-en`；
- 明确跨任务污染：通用/角色扮演 LLM、NER、代码生成和图像分割模型占据榜单。

这与全量发布版指标方向一致，而与本地重训代理返回的 Jina 多语言模型榜单完全不同。后者只能说明“补齐输入并由我们重训后，这个架构能够学习”，不能证明公开 demo 可用。

### 5.1 九类任务公共 demo 审计

公共 demo 审计进一步覆盖文本分类、检索、重排、跨语言 STS、聚类、双语文本挖掘、问答推理、多模态 VQA 和机器翻译九类任务。查询集合在收集其余 demo 输出之前冻结。完整输入、两边的完整榜单和逐案例判定见 [cross-task public-demo audit](../../../../../weeks/week9_scale/PUBLIC_DEMO_CROSS_TASK_AUDIT.md)。

| 审计量 | Model Lake | ModelLens public demo |
|---|---:|---:|
| 精确数据集可验证推荐 | **72/90** | 5/88 |
| 精确数据集近优推荐 | **33/90** | 5/88 |
| 严格满足任务、模态和语言要求 | 未做同类人工重标 | 26/88 |
| 已确认偏题推荐 | 未做同类人工重标 | 49/88 |

ModelLens 在 **8/9** 个案例中没有任何精确数据集可验证项。唯一例外是 `MTEB Tatoeba (ast-eng)`：公共榜单正文的九项中有五个 multilingual E5 变体在该数据集上具有精确、观测最优记录。该成功案例被保留在汇总中。

AskUbuntu 与 Tatoeba 的粘贴结果正文各含九行；页面上与正文不一致的孤立 rank-10 卡片没有被拼接进榜单。另有两个预注册案例 `MMLU-Pro(acc)` 和 `CrossNER_AI` 因官方 ModelLens 描述为 missing/NaN 而未提交，并从所有分子和分母中剔除。因此，本审计没有把“无法形成有效查询”算作 ModelLens 推荐失败。

## 6. 可复现命令

```powershell
Set-Location D:\research\model_lake\codes\ModelLakeFishing
.\.venv\Scripts\python.exe -m scale.deployment_quality_metrics --selftest
.\.venv\Scripts\python.exe -m scale.deployment_quality_metrics `
  --graph stage1BuildTransferGraph\hgraph_ml_v2_sub.pt
.\.venv\Scripts\python.exe -m scale.cross_task_demo_audit --selftest
.\.venv\Scripts\python.exe -m scale.cross_task_demo_audit
```

结果文件：`docs/scale/P5/artifacts/deployment_quality_metrics.json`。

## 7. 声明边界

1. “未验证”不等于“真实性能为零”；所以主表分别报告覆盖、偏题与近优证据。只有 `certified_utility_lower_bound@10` 明确把未验证项按 0 计，含义是可担保下界。
2. 任务兼容指标仍是可审计的观察证据，不替代逐模型人工审卡；STS17 demo 另做了严格的任务、模态和语言人工审计。
3. 结论针对公开发布件及用户实际可获得的推荐路径，不声称论文作者未发布的内部系统同样失效。
4. 本次没有为了结论挑单一复合分数；覆盖、近优、偏题、gold 排名和查询特异性五类相互独立指标给出一致方向。
