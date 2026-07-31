# P1 执行报告：从 ModelLens 语料重建 ModelLakeFishing HGraph

**执行日期：** 2026-07-23
**对应规划：** `docs/scale/MODELLENS_HEADTOHEAD_SCALE_PLAN.md` §1.2 / §1.3 / §3.1、§5 P1 行
**状态：** ✅ **P1 出闸通过**（图已建、Stage-2 契约 assert 全过、intake 可复现）；但**两项重量级发现改写下一步**（真实可训规模 30K≠47K；血缘边几近为零）
**产物：**
- 代码：`scale/modellens_intake.py`、`scale/modellens_build_graph.py`、`scale/p1_restore_drill.py`
- 中间件：`stage1BuildTransferGraph/artifacts/modellens_v2_lake/`（三件 D0 契约产物 + 报告）
- 图：`stage1BuildTransferGraph/hgraph_ml_v2.pt`

---

## 0. 一句话结论

ModelLens v2 语料已成功转成 D0 契约并重建为 `hgraph_ml_v2.pt`，**节点特征维度（模型 448 / 数据集 458）、5 类边、xm0/xd0 meta、mappedID 行序契约与 `hgraph_d0_v1.pt` 逐项一致**，Stage-2 可直接消费。P0 的六项修正全部落地并生效。但重建暴露两个必须改写规划的事实：**(1) 全语料 47,242 模型里只有 30,183 个进入可训练严格湖**（其余 1.7 万只在无界指标上被测过）；**(2) 名字精确匹配的血缘推断只得到 42 条边**，r_mm' 这条卖点在此语料上近乎失效，需 P2 专门补强。

---

## 1. 做了什么（过程与代码）

### 1.1 入口守卫（§1.4）
进入 P1 前跑 `verify_corpus`：v2 10/10、ckpt 3/3 全绿，语料自 P0 未被改动。

### 1.2 两段式：intake → build

| 脚本 | 职责 | 关键点 |
|---|---|---|
| `scale/modellens_intake.py` | CSV+7 JSON → 三件 D0 契约产物 + 逐行 provenance | 六项 P0 修正全在此落地；node=(dataset,task) |
| `scale/modellens_build_graph.py` | 三件产物 + ModelLens 特征 → `hgraph_ml_v2.pt` | **复用** D0 的 `build_name_embeddings`/`param_count_to_size_bucket`/family vocab/MiniLM |
| `scale/p1_restore_drill.py` | P1 出闸：corpus 校验 + intake 可复现性 | 离机恢复经 D-6 降级为「单机确定性复跑」 |

### 1.3 节点粒度决策（从数据定，非拍脑袋）
探针实测：dataset 名自由文本（仅 14.9% 含「/」，非 HF config 分隔符）；(dataset,task) 三元组 15,927 个；**(dataset,task,model,metric) 零重复**。据此定：

- **node = (dataset, task)** —— 一个连贯的检索目标，每节点选**一个**指标；
- **root_a = dataset 名** —— 划分口径 (a)，即 ModelLens 自己的名字粒度（已比其 (ds,task,metric) 三元组严格，见 P0 §1.2c）；
- **root_b = 规范化 dataset** —— 划分口径 (b)，剥括号 + 标点归一，保守（不合并 size 变体）。

### 1.4 六项 P0 修正的落地实证

| P0 修正 | 实现 | 实测效果 |
|---|---|---|
| #4 @k 感知 BOUNDED | `base_metric()` 剥 `@\d+` 再匹配白名单 | 1,807,133 → **972,215** 有界行（53.8%，复现 P0） |
| #3 逐 (node) 量纲归一 | 按节点中位数 >1.5 判 percent → /100，越界丢弃 | **percent 129,665 / unit 183,321**；丢弃 1,058（0.34%）脏值（如 accuracy=2317） |
| #5a size "unknown"→缺失 | 字符串 `"unknown"` 判缺失；十亿→原始计数 | size 覆盖 14,461/30,183（47.9%），unknown bucket 52% |
| #5a family "unknown"→缺失 | 缺失折叠 Other（**不**名字推断，见 §3.3） | family 覆盖 27,264/30,183（90.3%） |
| #5b popularity 取 `["models"]` | 只认 `status=="ok"` | 覆盖 13,666（弱信号，未入图特征，与 D0 一致） |
| #6 gold 深度≥10 | `gold_evaluable = depth≥10` | gold 节点 **1,686**；含描述 963；深度≥30 的 824 |

### 1.5 一个被 assert 拦下的真 bug（过程留痕）
首跑 intake 时 assert `(node,model)` 唯一失败（104,389 重复）。根因：@k 感知的 base metric 把 `ndcg@1/@3/@10` 折成同一个 `ndcg`，一个 (node,model) 于是有多行。**修正：节点的 canonical 指标用全名（`ndcg@10`），base 只用于「是否有界」的成员判定**。@1 与 @10 是不同 cutoff，绝不混folding。修后 (node,model) 天然唯一，零重复。这条纪律已写进代码注释。

---

## 2. 产出的图（Stage-2 契约对账）

`hgraph_ml_v2.pt`（`ml_graph_report.json`）：

| 项 | 值 | 对照 D0-v1 |
|---|---:|---|
| models | 30,183 | D0: 9,491 |
| dataset_nodes (dataset,task) | 9,603 | D0: 1,463 |
| roots_a (dataset 名) | 5,568 | — |
| roots_b (规范化) | 3,506 | — |
| trained_on 边 | 312,986 | D0: ~6 万 |
| similar_to 边 (KNN k=20) | 192,060 | 同构 |
| **is_base_of 血缘边** | **42** ⚠️ | D0 有实血缘 |
| gold 节点 (depth≥10) | 1,686 | D0: 613 |
| task 词表 | 196（Other 32%）| D0: 17（Other 3.1%）|
| family 词表 | **341**（= ModelLens 332 curated + KNOWN）| D0: 依采集 |
| size unknown 占比 | 52% | — |
| model x 维 / dataset x 维 | 448 / 458 | **完全一致** |
| sha256 | `e6ae2df…` | — |

**契约 assert（全过）**：模型 `x` = (30183, 448)、数据集 `x` = (9603, 458)、无 NaN、边索引不越界、`family_vocab["Other"]==0`、`task_vocab["Other"]==0`、`value_norm∈[0,1]`、恰 5 类边、`unique_model_id`/`unique_dataset_id` 按 mappedID 行序且 `unique_dataset_id` 带 `root`+`root_b` 两列。**与 `hgraph_d0_v1.pt` 的 Stage-2 输入契约完全同构，Stage-2 无需改代码即可训练。**

---

## 3. 改写下一步的发现

### 3.1 🔴 真实可训规模是 30,183 / 9,603，不是 47,242 / 10,479

**监督漏斗（透明可复现）：**

```
1,807,133 行 (全语料)
  └─#4 @k 感知有界指标 ───────────→   972,215 行 (53.8%)
      └─每 (dataset,task) 选一个全名指标 → 314,044 行
          └─#3 量纲归一 + 越界丢弃 ───→   312,986 条 trained_on 边 (丢 1,058)

模型:  47,242 (语料) → 30,183 严格湖 (has_obs ∨ has_lineage)
数据集: 10,479 → 5,568 (有界观测的) → 9,603 个 (dataset,task) 节点
```

**为什么是 30,183 而非 47,242**：1.7 万模型**只**在**无界/未白名单**指标上被测过（perplexity、cosine_\*、pass@k、quasi_exact_match 等），无法贡献 [0,1] 的 y_md 边。

**这与 D0 的设计完全一致，不是缩水**：D0 本身就是「strict intake = has_model_index ∨ has_lineage」（9,491 严格模型来自更大的采集池）。这里 47,242 是 canonical 池（类比 D0 的大采集池），30,183 是严格可训湖（类比 D0 的 9,491）。**同一套纪律，同一个漏斗。**

**但它对 head-to-head 提出一个新决策（D-7）**：候选宇宙用哪一个？
- (i) 只用 30,183 严格湖（两系统都限定在此，最干净、可比）；
- (ii) 全 47,242（1.7 万无边模型作**冷特征节点**入图，正好喂 C 轴 inductive/冷启动实验）。
- **倾向**：P4 主表用 (i)（同宇宙、无争议）；C 轴专门用 (ii) 展示我们对纯特征冷节点的处理。**待 P2/P4 定。**

### 3.2 🔴 名字血缘几近为零：42 条边

高精度、名字精确匹配的血缘推断（剥 gguf/awq/q4_k_m/lora 等 marker，要求剥后基名仍在 47K 集内）只得到 **92 个 lineage_base、42 条 pool 内边**。原因：GGUF/量化/LoRA 的上传者与基座模型**通常是不同 HF 用户**，且基座常不在这 47K 语料内，导致同名精确匹配几乎打不中。

**影响**：r_mm' 血缘边是 CLAUDE.md 列的**核心贡献之一**，也是 C 轴（仅血缘边冷启动）的关键。42 条边意味着**在此语料上血缘通道近乎空转**。这是继 P0「数据集描述矩阵未发布」之后第二个重量级发现。

**对下一步**：新增 **D-8（P2 处理）** —— 血缘补强。候选：
- (a) 跨用户段匹配（剥 marker 后按最后一段 fuzzy 匹配，牺牲一点精度换召回）；
- (b) 用 `family` + size + 名字公共前缀聚类近似血缘（弱标签）；
- (c) 承认此语料血缘稀疏，把 C 轴重心移到 **sibling/task 融合（S2/P2b）** 与 **inductive 特征节点**（这两条不依赖血缘边，且 P0 已证 sibling 通道健康）。
- **倾向 (c) 为主 + (a) 试点**：不为凑边数牺牲精度；诚实报告血缘在此语料的局限。

### 3.3 🟡 family 词表污染，两步修干净（过程留痕）
两处污染源，逐一修掉：
1. **我方注入**：首 build 用 `_infer_one_family` 兜底缺失 family，注入 673 个垃圾家族（`Test`/`output`/单字母），vocab → 808。**改为缺失即折叠 Other**（不名字推断）→ 降到 698。
2. **ModelLens 自带**：`model_profile.family` 字段本身含 ~700 个噪声值（含 `a`/`trash`/`xxx`/`this`）。**改为只认 ModelLens 自己的 curated `family2id.json`（332）**，其余折叠 Other → 最终 **341**（= 332 出现 ≥min_count 者 + KNOWN_FAMILIES）。

最终 family 覆盖 19,927/30,183（66%，只算 curated 家族），vocab 与 ModelLens 自己的 332 行 `family_embedding` 对齐 —— 干净且同源。

### 3.4 🟡 task Other 占比偏高（32%）
ModelLens 有 2,581 个 task（长尾），min_count=5 后保留约 196 类，**32% 节点落 Other**（D0 仅 3.1%）。这是语料特性（task 词表极大且碎），非 bug。P2 可考虑 task 规范化折叠（合并 `HELM-*` 来源标签、大小写归一）降低 Other 占比。

---

## 4. 出闸：restore drill

`scale/p1_restore_drill.py`：
1. `verify_corpus`（v2）—— 冻结语料未变；
2. **intake 确定性复跑** —— 重跑 intake 到临时目录，三件产物的内容摘要（排序去索引后 sha256）与已提交版**逐件 MATCH** ⇒ 从冻结 raw/ 可无损重建。

> **口径说明**：§1.4 原 restore drill 要「从离机副本恢复」；**D-6 已裁定只留本地 D 盘**，离机恢复不适用。故 drill 降级为「语料完整性 + intake 可复现性」，这是单副本策略下仍然有意义的保证。

---

## 5. 对 PLAN 的净影响（回填要点）

1. §3 规模数字：**可训湖 30,183 / 9,603 / 312,986 边**（canonical 池仍 47,242 / 10,479）。
2. 新决策 **D-7**（候选宇宙 30K vs 47K）与 **D-8**（血缘补强），均记入 §7。
3. §1.2 crosswalk 的血缘行需加注：**名字精确匹配在此语料只得 42 边**。
4. Stage-2（P3）可直接用 `hgraph_ml_v2.pt`，无需改契约。

---

## 6. 可复现命令

```powershell
Set-Location D:\research\model_lake\codes\ModelLakeFishing
.\.venv\Scripts\python.exe -m scale.verify_corpus --only v2      # 入口守卫
.\.venv\Scripts\python.exe -m scale.modellens_intake             # 三件产物 (~13s)
.\.venv\Scripts\python.exe -m scale.modellens_build_graph        # hgraph_ml_v2.pt
.\.venv\Scripts\python.exe -m scale.p1_restore_drill             # 出闸
```
