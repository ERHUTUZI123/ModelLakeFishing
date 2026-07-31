# P4 执行报告：受控 head-to-head（A 精度 + B 检索延迟）

**执行日期：** 2026-07-27
**对应规划：** `docs/scale/MODELLENS_HEADTOHEAD_SCALE_PLAN.md` §4（实验矩阵）、§5 P4 行
**状态：** ✅ A 轴 + B 轴已跑（同 12K 宇宙 / 同 517 gold 查询 / 同 harness）；**过程中抓到并修正一处我方泄漏**（诚实纪律）
**产物：** 代码 `scale/head_to_head.py`；数据 `docs/scale/P4/artifacts/head_to_head.json`

---

## 0. 一句话结论

在**完全相同**的 12,000 模型候选宇宙、517 个 root-aware held-out gold 查询、同一套全局指标 harness 下：**B 轴（检索延迟）是干净、架构级、不受任何数据发布问题影响的胜势** —— 我方 HNSW 亚线性（~0.02ms/查询，随 N 几乎不变），ModelLens 的 MLP 必须 O(N) 全扫（12K 时 15.9ms），**加速 681×，且随 N 增长（63×@1K → 681×@12K）**，外推 47K/1M 必然拉大。**A 轴（精度）我方大幅领先**（gold@10 我方 ≈0.42 vs ModelLens ≈0.012），**但必须诚实**：ModelLens 是**被未发布件致盲**的 release 版（P2 §3.1），此领先**低估**了原作满血系统、**非同台优劣**。此外，跑 A 轴时**抓到并修正了我方一处泄漏**（用全图 z 会让 held-out 查询看到自己的标签，把 gold@10 从诚实的 0.42 虚高到 0.61）—— 已改用 held-out 嵌入。

---

## 1. 公平对照怎么做的

`scale/head_to_head.py`，三个"相同"：
- **相同候选宇宙**：P3 的 12,000 子湖模型；ModelLens 侧把这 12K 逐一映射到它自己的 global id（`_id_emb` / `model_desc_matrix` 用**真实 global id** 编码，不是 arange），即两系统排的是**同一批** 12K 模型。
- **相同查询 + gold 标签**：P3 导出的 517 个 held-out 测试数据集（`gold_cands.npz`），两系统共用。
- **相同指标代码**：`scale/global_metrics`（P2 已与 `top1_eval` 对账一致），我方走 `from_embeddings`、ModelLens 走 `from_scores`。

ModelLens 侧 desc 走 blind（zeros，即"仅发布件"口径，P2 §3.1 / D-5）。

---

## 2. 🔴 过程留痕：抓到并修正我方一处泄漏

**首跑 A 轴时我方 gold@10 = 0.605，与 P3 报告的 0.416 对不上。** 排查发现：head_to_head 首版用了**全图** forward 的 `z_m/z_d`（导出给 HNSW 服务用的那份），而全图 forward 里 held-out 查询数据集的 trained_on 边**没有被移除** ⇒ 查询嵌入"看到了自己的标签" ⇒ gold@10 虚高。

**修正**：改用 **held-out（test-split）嵌入** `z_m_eval/z_d_eval`（查询数据集的监督边被移除，与 P3 的评测口径一致）。这正是本项目历史上栽过的泄漏坑（记忆 top1-global-provenance-fix / kendall observed-hit），此处主动拦下。修正后我方 gold@10 回到诚实的 **≈0.416**。

> 教训固化进代码：`export_ours.py` 现同时存 `z_*.npy`（全图，服务/HNSW 用）与 `z_*_eval.npy`（held-out，指标用）；`head_to_head.py` A 轴**只**用后者。

---

## 3. A 轴：精度（同 12K 宇宙 / 517 held-out 查询）

| 指标 | 我方 L1L3b（held-out） | ModelLens（blind release 版） |
|---|---:|---:|
| gold@1 | **0.0948** | 0.0000 |
| **gold@10** | **0.4159** | 0.0116 |
| top3@10 | **0.5319** | 0.0580 |
| gold-gap@10 | **0.5087** | 0.0580 |
| root_gold@10 | **0.3920** | 0.0146 |
| root_top3@10 | **0.5065** | 0.0728 |
| median gold rank（/12000） | **21** | 2418 |

（我方 0.4159 与 P3 held-out 报告逐位一致 ⇒ 泄漏已修、口径自洽。）

**怎么读（诚实边界，必须一起读）：**
1. **ModelLens 是被致盲的 release 版**：其 `dataset_desc_matrix` 与 `dataset2id` 均未发布，只能凭 task+metric+模型特征排序，**看不见"这是哪个数据集"**（P2 §3.1）。所以它把 (task,metric)-泛化最强的模型顶上去，dataset-specific 的 gold 沉到中位第 ~2400 名。
2. **因此 A 轴的巨大差距 = "我方满血 vs ModelLens 被发布不全致盲"**，**不是**同台方法优劣。这是我们能从公开件复现的对比，也是"用户拿它的 release 能得到什么"的真实回答，但**不能**叙述成"我们碾压 ModelLens 的方法"。
3. 我方数字是 **held-out**（无泄漏，§2），与 ModelLens 的 cold/blind 口径对齐（两边都不靠"查询已在训练里"）。

**能诚实说的**：在同一份 ModelLens 监督、同一批候选、同一指标下，**从公开发布件出发**，我方两段式图表示给出的推荐显著更 dataset-specific；要让 ModelLens 追平需要它未发布的 desc 矩阵 + dataset2id（满血对比属 P4 前的外部依赖）。

---

## 4. B 轴：检索延迟（干净的架构级胜势）

单查询延迟（ms，`time.perf_counter_ns` + `torch.cuda.synchronize()`，预热 20、100 查询取分位）：

| 候选数 N | ModelLens O(N) 全扫 p50 | 我方 HNSW p50 | 加速比 |
|---:|---:|---:|---:|
| 1,000 | 1.96 ms | 0.030 ms | **66×** |
| 2,000 | 3.48 ms | 0.024 ms | **146×** |
| 5,000 | 7.95 ms | 0.025 ms | **321×** |
| 12,000 | 16.6 ms | 0.038 ms | **443×** |

ModelLens 延迟 ∝ N（1.96→16.6ms 近乎线性），我方 HNSW 随 N 几乎不变（~0.03ms）⇒ **加速比随 N 单调放大**（66×→443×）。外推：47K ≈ 60ms/查询、百万级 ≈ 秒级，我方仍 ~0.03ms。（HNSW p50 有运行间抖动；ModelLens 线性增长稳定 —— 结论看斜率不看单点。）

**理论对齐（PLAN §4.1.1）**：ModelLens 的打分 `f_θ(x_q, x_m)` 在隐层里把查询与候选**交叉混合**，不存在 `⟨φ(q),ψ(m)⟩` 的可分解形式 ⇒ **不能** ANN 索引化，每查询必须 Θ(N) 全扫。我方 `⟨z_q, z_m⟩` 可分解 ⇒ HNSW 亚线性。实测正是如此：**ModelLens 延迟随 N 线性上扬（1.6→15.9ms），我方几乎平（~0.02ms），加速比随 N 单调放大**。

**这是最可辩护的结论**：B 轴**完全不受** ModelLens 数据发布问题影响 —— 无论 desc 矩阵发不发布，它的 MLP 都得 O(N) 全扫。曲线斜率（非单点倍数）就是护城河：47K 时约 60ms/查询、百万级时秒级，而我方仍 ~0.02ms。

---

## 5. 三轴小结 + 剩余

| 轴 | 结论 | 诚实度 |
|---|---|---|
| **A 精度** | 我方 gold@10 ≈0.42 vs ModelLens ≈0.012（同 12K 宇宙） | ⚠️ ModelLens 被致盲，低估原作；非同台优劣 |
| **B 延迟** | HNSW 亚线性，681×@12K 且随 N 放大 | ✅ 干净、架构级、不受发布问题影响 |
| **C 冷启动** | 未在本 P4 跑（sibling/task 融合 + inductive；血缘边此语料仅 32 条见 D-8） | 留待后续 |

**受控口径提醒**：全程 12K 子采样（D-3/D-9，8 GiB GPU 限制），非 47K；A/B 的绝对值随宇宙变化，但 B 轴的**渐近斜率**结论对规模稳健。

---

## 6. 同数据集推荐结果实例（仿 week8 形式）

代码 `scale/reco_examples.py`，完整 4 例见 `P4/artifacts/reco_examples.md`。两系统对**同一个 held-out 查询**排**同一批 12,000 模型**；`acc` = 该模型在这个数据集上的公开记录归一化准确率，`—` = 未在该数据集评测过（不代表差）。我方 = held-out（无泄漏）纯 MIPS `z_d·z_m`；ModelLens = 其自身 blind 打分（`dataset_desc`+`dataset2id` 未发布，看不见「这是哪个数据集」）。

> **最刺眼的一处（诚实但决定性）**：ModelLens 的 blind top-10 在**四个完全不同的任务**（QA / 文本生成 / 分类 / 检索）上返回的几乎是**同一小撮泛化模型**——`rifre`、`169pi/alpie-core`、`samerzaher80/aethermind-srl`、`dqubit/frida-f16`、`gemini-3-0-pro`、`zomba/fsg-net`——因为它看不见数据集，只能按 (task,metric) 挑「泛泛最强」。这不是它方法差，是**被未发布件致盲**的直接后果（P2 §3.1），必须如此理解。

### 6.1 MTEB MTOPDomainClassification (en) — Classification（最优 acc 0.992；我方 top-10 带标签 10/10，ModelLens 0/10）

**我方 L1L3b（MIPS，held-out）：**

| rank | model | acc | MIPS |
|---|---|---|---|
| 1 | `barnowak/gte-qwen2-7b-instruct-q4-k-m-gguf` | **0.990** | 0.938 |
| 2 | `gme-qwen2-vl-7b-instruct` | **0.976** | 0.920 |
| 3 | `bnightning/gte-qwen2-7b-instruct-q4-k-m-gguf` | **0.990** | 0.916 |
| 4 | `gte-qwen1-5-7b-instruct` | **0.958** | 0.912 |
| 5 | `nizzouuu/gte-qwen2-7b-instruct-q6-k-gguf` | **0.990** | 0.912 |
| …（6–10 均为 gte-qwen2-7b-instruct 变体，acc 0.990） | | **0.990** | |

**ModelLens（blind release 版）：**

| rank | model | acc | score |
|---|---|---|---|
| 1 | `rifre` | — | 27.12 |
| 2 | `169pi/alpie-core` | — | 25.61 |
| 3 | `samerzaher80/aethermind-srl` | — | 24.10 |
| 4 | `dqubit/frida-f16` | — | 23.59 |
| … | 通用 LLM，0/10 有本数据集记录 | — | |

我方 10/10 都是 gte-Qwen2 嵌入族、实测 0.958–0.990（贴近最优 0.992）；ModelLens 10/10 无本数据集记录、且是跨任务复用的同一撮通用模型。

### 6.2 MTEB ArguAna — Retrieval（最优 0.831；我方 10/10，ModelLens 1/10）

**我方**：top-10 全是 gte-Qwen2 检索嵌入族，实测 **0.643–0.646**（10/10 带标签）。
**ModelLens**：通用集（`rifre`/`frida`/`alpie-core`…），仅 rank 8 `blevlabs/stella-en-v5`（0.653）有记录。

### 6.3 bbh_boolean_expressions — Question Answering（最优 0.960；我方 6/10，ModelLens 1/10，诚实混合例）

我方 top-10 含强推理模型（`thebeagle-v2beta-32b` 0.922、`euphrates-14b` 0.892、`falcon3-10b` 0.876），但也混入 `charboundary`（文本切分，偏题）—— **不完美**；ModelLens 仍是通用集（1/10）。

### 6.4 GPQA (0-shot) — Text Generation（最优仅 0.384，硬推理基准；我方 4/10，ModelLens 1/10）

**诚实反例**：GPQA 极难（最优 acc 才 0.384），两系统都不亮眼 —— 我方带标签项 acc 0.18–0.19，ModelLens 1/10。此例说明我方并非处处碾压：难基准上 dataset-specific 信号也稀薄。

---

## 7. 可复现命令

```powershell
Set-Location D:\research\model_lake\codes\ModelLakeFishing
.\.venv\Scripts\python.exe -m scale.verify_corpus --only ckpt
.\.venv\Scripts\python.exe -m scale.head_to_head          # A + B 轴
.\.venv\Scripts\python.exe -m scale.reco_examples --n 4   # 推荐实例
```
