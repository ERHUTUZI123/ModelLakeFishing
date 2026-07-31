> **部署质量口径已更新（2026-07-28）：** 本文件记录的是“由我们补齐公开表示并重训 ModelLens 架构”的方法级反事实实验，不再作为 ModelLens 实际发布系统推荐质量的主结论。实际 top-10 部署质量请以 [`DEPLOYMENT_QUALITY_EXECUTION.md`](DEPLOYMENT_QUALITY_EXECUTION.md) 与 `deployment_quality_metrics.json` 为准。旧的 labeled-only NDCG、相关性、条件 regret 和重训 pointwise 校准不得外推为公开 demo 可用性。

# P5 执行报告：同条件公平重训 ModelLens —— A/B/C 三轴受控 head-to-head

**执行日期：** 2026-07-27
**对应规划：** `MODELLENS_HEADTOHEAD_SCALE_PLAN.md`「方案修订 v2」+ 阶段 P5
**状态：** ✅ 三轴齐全，并补充完整 top-10 列表质量与代表性审计。**A 轴 gold-survival 近平手**（重训 ModelLens 略优）；完整列表质量各有优劣；**B 轴（延迟）我方决定性胜**；**C 轴（冷启动）近平手，最冷段我方有归纳式小优势**
**产物：** 代码 `scale/retrain_modellens.py`、`scale/cold_start_axis.py`、`scale/list_quality_metrics.py`；数据 `P5/artifacts/{retrained_modellens.json,cold_start.json,retrained_modellens.pt,list_quality_metrics.json}`

---

## 0. 一句话结论（诚实版，取代"碾压"叙事）

把 ModelLens 从"被未发布件致盲的 release 版"换成"**同条件公平重训的满血方法**"后，它的 gold@10 从 **0.012 暴涨到 0.437** —— **证明此前的巨大差距全是"发布不全致盲"造成的，不是方法差**。在完全同等条件（同 12K 宇宙 / 同 517 held-out 查询 / 同一份公开 MiniLM 数据集嵌入 / 同 harness）下：

- **A gold-survival：近平手，重训 ModelLens 略优**（gold@10 0.437 vs 我方 0.416，top3@10 **两者相同 0.532**）；完整列表上，我方可验证覆盖率与近最优密度更高，ModelLens 对已有标签候选的内部排序更好；
- **B 延迟：我方决定性、架构级胜**（66×→443× 随 N 放大，亚线性 vs Θ(N)）；
- **C 冷启动：近平手**，仅在"gold 模型训练证据最稀薄"的最冷段我方归纳式 2× 领先（0.041 vs 0.020）。

⇒ **可辩护的主张 = 「在相近 gold-survival 与冷启动表现下，我方完整 top-10 的可验证覆盖率更高，并以 2–3 个数量级更低的检索延迟 + 亚线性扩展完成检索；ModelLens 在已有标签候选的内部排序上更强」。**

---

## 1. 为什么必须公平重训（回顾 v2 修订）

P2–P4 拿**发布 checkpoint** 比，但它被未发布的 `dataset_desc_matrix` + `dataset2id` 致盲（只能盲数据集排序，gold@10≈0.012）。带着这个去汇报，"你是不是把它打瘸了再打？"一问就瓦解精度结论。**破局点（用户发现）**：desc 的**原始文本 `dataset_desp` 就在 corpus-v2 里**，我们自己重嵌入、同一份喂两系统，ModelLens 就不再盲。

## 2. 重训怎么做到"忠实且不可指摘"

`scale/retrain_modellens.py`：
- **ModelLens 自己的架构**：`ModelLens/module/model/MLP.py` 的 `ModelLens` 类（cross-feature MLP + id-emb + name-enc + desc + task/metric/size/family 先验 + 可学温度），原码 import，零改。
- **ModelLens 自己的损失族 + 超参**：listwise + pairwise + pointwise 集成，权重取自其 `args.json`（λ_list=0.5 / λ_pair=1.0 / point=0.1 / id_dropout=0.1）。
- **与我方逐字节相同的输入**：数据集表示 = 图里的 `e_card`（MiniLM over `dataset_desp`，缺则 name+task 回退）；模型表示 = 图里的 `e_desc`（MiniLM over name+family+size）。**两系统喂同一份表示，唯一差异是架构**。dataset desc 槽维度设 384（我方编码器维度）⇒ 重训即 in-distribution，无 OOD。
- **同划分/同宇宙/同 harness**：训练用 12K 子图非 held-out 数据集的 `trained_on` 元组；held-out 查询数据集 `dataset_id=[UNK]`（其 id 从未被监督）但**保留已知描述** —— 真实的"新数据集"冷启动路径。

**忠实性证据**：训练 loss 稳定下降（2.39→1.73，20 epoch/1074s）；wiring **Pearson(打分, 观测精度) = 0.675、93% 查询为正** —— 重训模型确实学会了排序，非欠训。

> **留痕**：这是"用它自己的架构 + 它自己的损失/超参 + 双方共享的公开数据集嵌入"训出的**非盲、可复现** ModelLens；盲 release 版（P2/P4）降为下方对照的一列。

## 3. A 轴：精度（同 12K / 517 held-out / 同 harness）

| 指标 | 我方 L1L3b | **ModelLens 重训(非盲)** | ModelLens blind release |
|---|---:|---:|---:|
| gold@1 | 0.0948 | **0.0986** | 0.0000 |
| gold@10 | 0.4159 | **0.4371** | 0.0116 |
| top3@10 | **0.5319** | **0.5319** | 0.0580 |
| gold-gap@10 | 0.5087 | **0.5435** | 0.0580 |
| root_gold@10 | 0.3920 | **0.4035** | 0.0146 |
| root_top3@10 | **0.5065** | 0.5058 | 0.0728 |
| median gold rank（/12000） | 21 | **13** | 2418 |

**诚实读法**：
1. **盲 → 非盲 = 0.012 → 0.437**（36×）：坐实此前差距源于**发布不全致盲**，非方法。盲 release 只能作"用户拿它的 release 现成能得到什么"的附录数据点。
2. **同条件下是平手，重训 ModelLens 略优**：gold@10 0.437 vs 0.416、median rank 13 vs 21；**top3@10 完全相同（0.532）**、root_top3@10 几乎相同。**我方不在精度上占优** —— 这与 v2 修订的诚实预告一致（其 listwise/pairwise 目标就是为 in-matrix 排序优化的）。
3. **这恰恰是可辩护的好结果**：一个"精度打平"的诚实结论 + 下面 B 轴的数量级延迟优势，远比虚高的"碾压"经得起审。

### 3.1 完整 top-10 列表质量、校准与代表性审计

`gold@10`、`top3@10` 和 `gold-gap@10` 衡量至少一个目标模型是否进入 top-10，不评价其余推荐位。`scale/list_quality_metrics.py` 因此在全部 517 个 held-out 查询上补充完整列表指标。

| 指标 | 定义 | 我方 | ModelLens 重训 |
|---|---|---:|---:|
| `verified_coverage@10` | top-10 中有该数据集公开成绩的比例 | **0.3762** | 0.2408 |
| `verified_any@10` | 至少有一个可验证推荐的查询比例 | **0.6905** | 0.6692 |
| `task_compatible_proxy@10` | 在非 held-out 数据上存在同任务观测的比例 | **0.6663** | 0.6178 |
| `off_task_proxy@10` | 有训练观测、但仅出现在其他任务上的比例 | **0.3325** | 0.3799 |
| `best_observed_regret@10` | 最优准确率减去 top-10 最佳已观测准确率；仅统计有覆盖查询 | 0.0232 | **0.0151** |
| `near_optimal_precision@10` | 完整 top-10 中距最优不超过 0.01 的比例 | **0.1520** | 0.1379 |
| `labeled_nDCG@10` | 仅在有真实记录的候选内计算 NDCG | 0.8585 | **0.9208** |
| Spearman | 预测分数与真实准确率的宏平均相关性 | 0.3438 | **0.6277** |
| Kendall | 已标注候选的宏平均秩相关 | 0.2737 | **0.5294** |

**完整列表结论是混合的。** 我方平均每个 top-10 有 **3.76** 个可验证模型，ModelLens 为 **2.41**；我方近最优模型密度更高，off-task 证据代理更低。ModelLens 在已标注候选内部的排序明显更好，并在有至少一个已观测推荐的查询上取得更低的条件 regret。条件 regret 排除了无覆盖查询，必须与 coverage 同时报告。

任务兼容性为可审计的**训练证据代理**：模型在非 held-out 数据中至少有一个相同任务观测记为 compatible；有其他任务观测但无同任务观测记为 off-task；完全无任务证据单列 unknown。该代理不等价于人工查看模型卡后的语义判定。

ModelLens pointwise head 使用 `sigmoid(z_pred)` 在 55,571 个 held-out 已观测模型-数据集对上评估：**micro MAE = 0.2016、macro-query MAE = 0.1672、RMSE = 0.2411、10-bin regression ECE = 0.0367**。较低 ECE 表明分箱后的平均偏差有限；MAE 表明单个模型-数据集预测仍有明显误差。`score_matrix` 的 ranking logit 不作为准确率预测。

gold@10 四象限为：两者都命中 147、仅我方命中 68、仅 ModelLens 命中 79、两者都未命中 223。原 4 个 demo 全部落在**两者都未命中**象限，因此只可作为失败模式，不能代表总体分布。后续定性展示采用固定种子 **20260728**，从每个象限随机抽取 3 个查询；总体结论仍以全部 517 个查询为准。

## 4. B 轴：检索延迟（我方决定性、架构级胜；不受任何发布问题影响）

单查询延迟（`perf_counter_ns` + `torch.cuda.synchronize()`，预热 20、100 查询取分位；沿用 P4 口径，架构结论与训练无关）：

| 候选数 N | ModelLens O(N) 全扫 p50 | 我方 HNSW p50 | 加速比 |
|---:|---:|---:|---:|
| 1,000 | 1.96 ms | 0.030 ms | **66×** |
| 2,000 | 3.48 ms | 0.024 ms | **146×** |
| 5,000 | 7.95 ms | 0.025 ms | **321×** |
| 12,000 | 16.6 ms | 0.038 ms | **443×** |

ModelLens 延迟 ∝ N（1.96→16.6ms），我方 HNSW 随 N 几乎不变 ⇒ **加速比随 N 单调放大**。理论根因：其 MLP 把 query/candidate 交叉混合、数学上**不可 ANN 索引化**，Θ(N) 是架构下界；我方 `⟨z_q,z_m⟩` 可分解 ⇒ 亚线性。外推 47K≈60ms、百万级≈秒级，我方仍 ~0.03ms。**重训不改变这条**（重训只让它更准，全扫成本一分不减）。

## 5. C 轴：冷启动（同宇宙/同查询/同 harness，分层，无额外训练）

`scale/cold_start_axis.py`：把 517 个 held-out gold 查询按两条冷启动维度分层。

**C1 — 新模型代理（gold 模型的训练证据 = 它在多少个训练数据集上被观测过）：**

| 分层 | n | 我方 gold@10 | ModelLens 重训 |
|---|---:|---:|---:|
| **cold（≤2）** | 49 | **0.041** | 0.020 |
| mid（3–10） | 72 | 0.250 | **0.458** |
| warm（≥11） | 396 | **0.492** | 0.485 |

- **最冷段我方 2× 领先（0.041 vs 0.020）** —— 支持归纳式主张：gold 模型训练证据极稀薄时，我方从**特征 + 邻域**归纳出嵌入，而 ModelLens 的 per-model id 嵌入退化为弱/[UNK]。但两者绝对值都低（极冷模型本就难）。
- **mid 段 ModelLens 明显反超（0.458 vs 0.250）**，warm 段平手 —— **诚实记录，不回避**。

**C2 — 新数据集的同任务 peer 可用性**（注：同 root 兄弟已被 root-aware 划分**按设计全部剔除**，故所有查询都 root-孤立；同任务 peer 是非退化的冷信号）：

| 分层 | n | 我方 | ModelLens 重训 |
|---|---:|---:|---:|
| rare task（≤5） | 44 | **0.386** | 0.364 |
| mid（6–50） | 147 | 0.565 | **0.592** |
| common（≥51） | 326 | 0.353 | **0.377** |

近平手，稀有任务上我方微弱领先，中/常见任务 ModelLens 微弱领先。

**C 轴小结**：**近平手**；唯一清晰的结构性差异是"最冷模型段"我方归纳式 2× 领先。**注意**：我方**未启用** S2/P2b sibling/task 融合（导出的是纯 MIPS）—— 那正是为冷启动设计的工具，是后续能进一步拉开 C 轴的杠杆（week8 已证 sibling-rich +64%），此处未用属保守口径。

## 6. 三轴诚实总裁 + 可辩护主张

| 轴 | 结论 | 强度 |
|---|---|---|
| **A 排序质量** | gold-survival 近平手；ModelLens 的已标注候选排序更好，我方完整列表可验证覆盖与近最优密度更高 | ⚖️ 不用单一指标概括完整推荐质量 |
| **B 延迟** | 我方 66×→443× 随 N 放大，亚线性 vs Θ(N) | ✅ 决定性、架构级、不受发布问题影响 |
| **C 冷启动** | 近平手；最冷模型段我方归纳式 2×（0.041 vs 0.020） | ⚖️ 平手 + 局部归纳式小优 |

> **最终主张（经得起审）**：*在完全同等、完全可复现的条件下，我方两段式（GraphSAGE + HNSW）与 ModelLens 的 cross-feature MLP **gold-survival 与冷启动相近**；我方完整 top-10 的可验证覆盖率更高，ModelLens 对已有标签候选的内部排序更好；我方**检索延迟低 2–3 个数量级且亚线性扩展**。*

**诚实边界**：受控规模 12K（D-9，8 GiB GPU）；我方未启用 S2/P2b（保守）；重训用其损失族+超参的忠实复现（非其私有训练管线），已附 loss 曲线 + wiring 佐证。coverage 衡量可验证性，不把未知模型判为差；task-compatible/off-task 是训练证据代理。满血复现（其原 1536 维 OpenAI 编码器）作可选交叉验证。

---

## 7. 可复现命令

```powershell
Set-Location D:\research\model_lake\codes
$g='ModelLakeFishing/stage1BuildTransferGraph/hgraph_ml_v2_sub.pt'
# 公平重训 ModelLens 方法（非盲）
ModelLakeFishing\.venv\Scripts\python.exe -m ModelLakeFishing.scale.retrain_modellens --graph $g --epochs 20
# C 轴冷启动分层
Set-Location D:\research\model_lake\codes\ModelLakeFishing
.\.venv\Scripts\python.exe -m scale.cold_start_axis --graph stage1BuildTransferGraph/hgraph_ml_v2_sub.pt
# 完整列表质量、任务证据代理、校准与代表性审计
.\.venv\Scripts\python.exe -m scale.list_quality_metrics --graph stage1BuildTransferGraph/hgraph_ml_v2_sub.pt
# A/B 轴（含盲 release 对照）见 P4
.\.venv\Scripts\python.exe -m scale.head_to_head
```
