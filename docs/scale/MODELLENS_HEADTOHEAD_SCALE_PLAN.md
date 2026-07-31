# 规模化 × 竞品对标规划：在 ModelLens 42K/47K 全量语料上证明 ModelLakeFishing 更优

**日期：** 2026-07-22
**阶段定位：** optimization 收官后的下一阶段 —— 从「在自建 9.5K 湖上把方法调到最优」转向「在竞品 ModelLens 的同一份大规模语料上，做受控 head-to-head，证明同数据条件下我们更强」。
**产物落点：** `codes/ModelLakeFishing/docs/scale/`
**上游依据：** `weeks/week8_Updated/progress_and_results.md`（现任冠军 L1L3b + S2/P2b 的完整战报）、`ModelLakeFishing/CLAUDE.md`（两段式可检索架构 vs ModelLens O(N) MLP 的先天差异）、`ModelLens/README.md`（语料 schema、模型、评测协议）。

---

# 🔁 方案修订 v2（2026-07-27）：用「同条件公平重训」取代「盲 release 对比」

## 修订动机：盲对比会被一句话打穿

P2–P4 的 A 轴是拿**发布 checkpoint**（被未发布的 `dataset_desc_matrix` + `dataset2id` 致盲）去比。无论怎么诚实披露，**「你是不是把它打瘸了再打？」这一问会瓦解整个精度结论** —— gold@10 0.416 vs 0.012 站不住，不能作主张。

## 解锁点（用户 2026-07-27 发现）：数据集描述本就在语料里

ModelLens **私有的是 desc 的 1536 维嵌入**（`train_vecs.npz`，未发布），但**生成它的原始文本 `dataset_desp` 完整躺在 corpus-v2 里**（`data.csv` 第 6 列）。实测：本湖全部数据集在语料中都能取到描述文本，**held-out gold 查询 313/517 = 61% 有真实描述**（其余两系统**对称回退**到 name+task，公平）。样例："*COVERAGE is a dataset designed for copy-move forgery detection…*"。

⇒ **我们根本不需要原作者的私有件**：自己把 `dataset_desp` 用一个**公开编码器**重新嵌入，**同一份**喂给两系统，ModelLens 就**不再是盲的**。

## 新主张（三轴重新定位）

| 轴 | 旧（v1，被质疑） | 新（v2，可辩护） |
|---|---|---|
| **B 检索延迟** | 已是干净胜势 | **升为头号贡献**：亚线性 vs Θ(N)，架构级、与数据发布无关、不可辩驳 |
| **A 精度** | 我方满血 vs ModelLens 盲 release（不成立） | **同条件公平重训**：用 ModelLens 自己的 MLP + Loss + 超参，喂**同一份**公开 MiniLM 数据集嵌入、同一张图监督、同划分、同 harness ⇒ 纯**方法**对比 |
| 盲 release | 曾是 A 轴主表 | **降级为附录一个数据点**（"即便直接用它发布件也只能盲跑 gold@10≈0.01"），非主结论 |

**诚实边界（写死）**：同条件重训后，ModelLens 的 MLP **可能在 in-matrix 补全上打平甚至反超**（其 listwise/pairwise 目标正为此设计）。**若如此，照实报**——「同台条件下，我方在全局 gold@K + 检索延迟上更优，在 in-matrix 补全上与之相当」远比一个虚高的「碾压」值钱、且经得起审。

## 新增阶段 P5：同条件公平重训 ModelLens 方法（新的 A 轴主口径）

**目标**：训练一个**非盲**、完全可复现的 ModelLens，作为 A 轴的正式对手；把 P2/P4 的盲 release 降为附录。

**共享输入（两系统逐字节相同）**：
- 数据集表示：`dataset_desp` → **all-MiniLM-L6-v2 (384-d)**（= 我方 `e_card` 同款编码器）；缺描述的数据集两边都回退 name+task；
- 模型侧特征：name / size / family（全部来自语料公开件）；
- 监督：同一张 12K 子图的 `trained_on` 元组（(model,dataset,task,metric,value)）；
- 划分：同 root-aware held-out（同 517 gold 查询）；
- 指标：`scale/global_metrics`（P2 已对账）。

**唯一差异 = 架构**：ModelLens 的 cross-feature MLP（O(N) 全扫）vs 我方 GraphSAGE + HNSW（亚线性）。这才是论文要证的东西。

**实现两策略（执行时择优）**：
- **策略 A（跑它自己的训练代码，最忠实）**：用 `dataset_desp` 生成它的 `train_vecs.npz` + `dataset2id` + 划分件，跑 `ModelLens/src/main.py`。缺点：需逆向它的数据格式。
- **策略 B（薄训练器包住它的 MLP，最可控）**：import 它的 `ModelLens` 类 + 复用 `module/model/Loss.py`（listwise/pairwise）+ `args.json` 超参，在**我们的**元组/划分上训练，`dataset_desp_emb_dim` 设为 384（我方编码器维度，重训即 in-distribution，无 OOD/padding）。对齐划分最干净。
- 倾向 **B**（可控、划分逐一对齐），用**它的 Loss + 它的超参**堵"你没训好"之嘴；A 作为满血复现的 stretch。

**门（gate）**：重训 ModelLens 的**训练 loss 正常下降** + 在**观测候选内**排序合理（wiring 正相关，同 P2 口径）+ 与我方在同一 harness/同查询/同宇宙出全局指标。可复现命令与产物落 `P5/`。

**风险与反制**：
- *「你重训的不如原作」* → 用**它自己的** Loss.py + args.json 超参 + 它的 MLP 原码；报告附训练曲线；并保留满血复现（策略 A / 索取私有件）作交叉验证。
- *desc 覆盖仅 61%* → **对称**回退（两系统同缺同补），不偏袒；按"有描述/无描述"分层报，证明结论在两层都成立。
- *规模仍 12K* → 沿用 D-9：B 轴延迟-vs-N 曲线外推（斜率不依赖规模）；有大 GPU 再补全 30K/47K。

> **一句话**：把"证明我们碾压 ModelLens 的产品"改成"**在完全同等、完全可复现的条件下，证明我们的架构（可检索图表示）在全局单最优 + 检索延迟上优于它的架构（O(N) MLP）**"。前者会被质疑，后者是硬结论。

---

> ## ⚑ 执行状态（2026-07-27 更新 · 含 P5）
>
> **P0 + P1 + P2 + P3 + P4 + P5 均已完成。三轴齐全。**
> - **P5（公平重训 · 新 A 轴主口径）** —— 详见 **[`P5/P5_EXECUTION.md`](P5/P5_EXECUTION.md)**。用 ModelLens 自己的 MLP + 损失族 + 超参、喂**同一份**公开 MiniLM 数据集嵌入（`e_card`/`e_desc` from 图）、同 12K / 同 517 held-out / 同 harness 重训出**非盲** ModelLens。
>   **🔑 关键结论（改写主张）**：盲→非盲 gold@10 **0.012→0.437**（坐实此前差距=致盲、非方法）；**A 精度 = 平手，重训 ModelLens 略优**（0.437 vs 我方 0.416，top3@10 **相同 0.532**）；**B 延迟 = 我方决定性胜**（66×→443× 亚线性 vs Θ(N)）；**C 冷启动 = 近平手**，仅最冷 gold 模型段我方归纳式 2×（0.041 vs 0.020）。
>   **⇒ 最终可辩护主张：不是"更准"，是"**同样准、冷启动相当、检索快 2–3 个数量级且亚线性扩展**"** —— 经得起"你是不是打瘸它"之问。盲 release 版降为附录。
> - **P0 + P1 + P2 + P3 + P4** ——
> - **P4（A/B 轴，A 已被 P5 取代为附录）** 详见 [`P4/P4_EXECUTION.md`](P4/P4_EXECUTION.md)。B 轴延迟结论沿用；A 轴 vs 盲 release 降为附录对照。
> - **P4** —— 详见 **[`P4/P4_EXECUTION.md`](P4/P4_EXECUTION.md)**。受控 head-to-head：同 12K 宇宙 / 同 517 held-out gold 查询 / 同 harness。
>   **A 轴**：我方 gold@10 = **0.416** vs ModelLens（blind release 版）**0.012**（median gold rank 21 vs 2418）——**但 ModelLens 被未发布件致盲，此领先低估原作、非同台优劣**（已按 D-5 全程披露）。
>   **B 轴**（干净、架构级、不受发布问题影响）：我方 HNSW 亚线性 ~0.03ms，ModelLens O(N) 全扫随 N 线性上扬，**加速 66×@1K → 443×@12K 且随 N 放大**，外推 47K/1M 必然拉大。
>   **过程留痕**：首跑抓到并修正我方一处泄漏（全图 z 让 held-out 查询看到自己标签，gold@10 虚高 0.42→0.61），改用 held-out 嵌入后与 P3 逐位一致。C 轴（冷启动）未跑。
> - **P3** —— 详见 **[`P3/P3_EXECUTION.md`](P3/P3_EXECUTION.md)**。用**未改动的** L1L3b 冠军配置在 ModelLens 语料图上训练成功，导出 `z_m/z_d` + HNSW + gold 标签（`P3/exports/ml_sub_L1L3b/`）。
>   我方 **gold@10 = 0.416 / root_gold@10 = 0.392 / top3@10 = 0.532**（12K 候选、517 held-out 测试集，median gold rank 21/12000）；**HNSW recall@50 = 1.000、p50 = 0.069ms**；harness 端到端一致（five_metric == global_metrics）。
>   **🔴 硬件约束 D-9**：8 GiB 笔记本 GPU 装不下 30K 的 O(N²) 对比损失，torch 2.12 无 pyg-lib 轮子 ⇒ 按 D-3 **子采样到 12K 模型 / 3K 数据集**（保全全部 1,686 gold）；"全 47K"规模在此硬件上未兑现，B 轴改用延迟-vs-N 曲线外推（斜率不依赖训练规模）。
> - **P0 + P1 + P2** ——
> - **P2** —— 详见 **[`P2/P2_EXECUTION.md`](P2/P2_EXECUTION.md)**。全局指标 harness `scale/global_metrics.py`（gold@1/@10/top3@10 + root-macro + gold-gap@10）
>   与既有 `top1_eval` **逐查询 rank 对账一致**；ModelLens 用其**自有源码 + 发布 checkpoint** 忠实重建（4 个 missing key 全良性，wiring Pearson 0.65–0.75 全正）。
>   **🔴 关键发现（重塑 A 轴）**：ModelLens 的 `dataset_desc_matrix` **与** `dataset2id` 映射**均未发布**，异编码器重建 desc 属 OOD 无效（a≈c）⇒ **从公开件只能"盲数据集"运行**，全 47,242 候选 gold@10≈0.002。此数字**低估**原作满血系统，差距源于发布不全，**非同台优劣**；P4 须诚实按"可复现 release 版"口径处理并披露（D-5 已升级）。
> - **P1** —— 详见 **[`P1/P1_EXECUTION.md`](P1/P1_EXECUTION.md)**。ModelLens v2 语料已重建为 `hgraph_ml_v2.pt`，
>   Stage-2 契约（模型 448 / 数据集 458 维、5 类边、mappedID 行序）与 `hgraph_d0_v1.pt` **逐项一致**，六项 P0 修正全部落地。
>   **真实可训湖 = 30,183 模型 / 9,603 (dataset,task) 节点 / 312,986 trained_on 边**（canonical 池仍 47,242 / 10,479；
>   1.7 万模型只在无界指标上被测，天然不入严格湖 —— 与 D0 的 strict intake 纪律一致）。
>   **两项新发现**：血缘边名字精确匹配只得 **42 条**（D-8 补强）；候选宇宙 30K vs 47K 待定（D-7）。
> - **P0** —— 详见 **[`P0/P0_EXECUTION.md`](P0/P0_EXECUTION.md)**，产物在 `P0/artifacts/`。
> 本规划中凡带 **`[P0 实测]`** 标记的数字/结论均已由实际字节替换掉原先的占位或推测，
> 带 **`[P0 修正]`** 的是被 P0 推翻后改写的条目。摘要：
>
> - 真实规模 **47,242 模型 / 10,479 数据集 / 1,807,133 行**（「42K」不成立，「~9.6K 数据集」也不成立）；
> - 竞品 checkpoint 全部词表维度与我们独立计数**逐项精确相等** ⇒ 语料同源已证；
> - 语料已固化：锁 SHA + 只读 + sha256 + `verify_corpus` **22/22 全绿**（副本策略经 D-6 裁定为**只留 D 盘**）；
> - **5 项发现改写 P1**，其中两项是重量级：它们**未发布**数据集描述嵌入矩阵（D-5，阻塞 P4 基线）；
>   它们的「dataset」实为 **(dataset, task, metric) 三元组** ⇒ 其冷启动划分按数据集名**存在同源泄漏**。

---

## §0 立意：我们到底要证明什么

### 0.1 一句话论点

> **在完全相同的监督数据下**（ModelLens 自己发布、自己训练、自己评测所用的那一份 1.62M 记录 / ~47K 模型 / ~9.6K 数据集语料），ModelLakeFishing 的两段式「可检索图表示 + serving-time 先验融合」在**我们的全局指标**（gold@1 / gold@10 / top3@10 —— 从整湖找最优模型）上、以及在**冷启动、检索延迟**上，显著优于 ModelLens 的 cross-feature MLP。**比较口径统一用我们的全局指标，不采用 ModelLens 的 local 指标。**

这不是「我们的湖 vs 它们的湖」的系统代际比较（week8 §1b 已诚实标注那种比较是 system-era、非受控）。这一阶段的全部价值在于把比较**降维成受控实验**：同一份记录、同一套划分、同一组查询、同一批 gold、同一套指标。差异只允许来自**方法本身**。

### 0.2 「完全相同的数据条件」的精确定义

「相同」= **相同的监督信号**，不是相同的模型输入管线。三点约定：

1. **相同的记录集合**：`(model, dataset, value)` 三元组逐行对齐 ModelLens 语料。我们不新增任何它们没有的性能观测。
2. **相同的划分**：train/val/test 用同一份、同一 seed、同一 hold-out 单元（见 §2.2 —— 它们的 `new_dataset_evaluation` / `new_model_evaluation` 冷启动划分，对应我们的 root-aware hold-out）。
3. **相同的候选宇宙与 gold**：查询数据集相同、被排的候选模型池相同、被判为「最优」的标签口径相同。

允许的差异（且**正是我们的论点所在**）：
- **我们从同一份数据里榨取更多结构**。ModelLens 的 CSV 只喂 `(task, dataset, model, metric, value, dataset_desp)`；`model2family.json` / `model_profile.json` 它们只用作 size/family **先验标量**。我们把同样的字段升维成**图的边**：血缘边 `r_mm'`（从名字/family 规则推断，quantized>adapter>finetune>merge）与数据集相似边 `s_dd'`（e_card KNN）。**同一份原料，更充分的利用** —— 这不构成「数据不同」，构成「方法更强」。
- **我们的输入管线不同**（GraphSAGE 归纳式节点编码 vs 它们的 ID-embedding + cross-feature MLP）。这正是被比较的方法本体。

### 0.3 我们赢在哪三条轴（不吹的边界）

| 轴 | 主张强度 | 度量 | 诚实边界 |
|---|---|---|---|
| **A 全局精度更优** | 主张、可判决 | **两系统都用我们的全局指标 gold@1 / gold@10 / top3@10 打分**（root-macro 聚合、gold-gap@10 作 good-enough 松弛）；不采纳它们的 local 指标 | ModelLens 未针对全局单最优优化，但全局模型选择正是它自称的使命，此比较公平 |
| **B 检索可扩展性** | 强、可判决 | 47K 规模下 query 端 wall-clock 延迟 + 内存；HNSW 亚线性 vs MLP O(N) 全扫 | 这是架构级先天差异，CLAUDE.md 已定性，此阶段把它**量化到 47K** |
| **C 冷启动** | 强 | `new_dataset` hold-out 上 S2/P2b sibling/task 融合 vs 它们的 id_dropout；同样用全局 gold@K 打分 | 我们的 S2/P2b 就是冷启动工具（week8 已证 +64% sibling-rich），此处在同语料复现 |

**度量纪律（用户定调 2026-07-22）**：全程只用**我们的全局指标** gold@1 / gold@10 / top3@10（及其 root-macro 聚合与 gold-gap@10），**不引入 ModelLens 的 local 指标**（τ_w / NDCG@K / Hit@K / Rec@K）。理由：它们的指标是**数据集内**的排序质量（local ordering），而产品真问题是「从整湖 47K 里找出最优模型」（global retrieval）——后者更难、更贴产品，且正是 ModelLens paper 标题自称的使命（*Finding the Best for Your Task from Myriads of Models*）。让两系统都在这条全局轴上比，公平且直击要害。

**主轴是 A**（用户命题「同数据、全局指标更优」的正面），**B/C 是护城河论证**。写报告时若 A 在某设定下仅打平，用 B/C 补足叙事，绝不粉饰。

---

## §1 数据接入与 schema 映射

### 1.1 取数（P0 第一步，需联网）

ModelLens 语料**不在本地**（`codes/ModelLens/` 仅 97M 代码，无 `data/`）。发布在 HF Hub：

**`[P0 实测]` 三个 repo 均已按锁定 SHA 拉取、校验、双副本固化：**

| 版本 | HF repo | **锁定 revision** | 实际体量 | 我们的取舍 |
|---|---|---|---:|---|
| corpus-v1 | `luisrui/ModelLens-corpus-v1` | `df6cb242ed54c96ec09c8644916c07ac14bff4e7` | 880.2 MB / 9 files | 干净语料鲁棒性对照，非主线 |
| corpus-v2 | `luisrui/ModelLens-corpus-v2` | `57a692ccdb20d84d1c803544f722b3727450c0e8` | 919.8 MB / 10 files | **主线** —— checkpoint 即在 v2 上训练 |
| checkpoint | `luisrui/ModelLens` | `68fabcb36d03f96b620a5b5f0ae786ce6da3e74f` | 709.1 MB / 3 files | 竞品基线权重（同样归档，防下架） |

**`[P0 实测]` 计数定案 —— 「42K」问题已结**（源：`P0/artifacts/audit_v2.json`）：

| 量 | 实测值 | 此前说法 | 判定 |
|---|---:|---|---|
| 行数 | **1,807,133** | README 1,807,133 | ✅ 一致 |
| 唯一模型 | **47,242** | 用户「42k」/ paper「~47K」 | **~47K 对，42K 不成立** |
| 唯一数据集名 | **10,479** | paper「~9.6K」 | **两者皆非；「10k 数据集」的说法更准** |
| 唯一 task | **2,581** | — | 词表远比预期大 |
| 唯一 metric | **3,714** | — | 词表远比预期大 |

> **决定性交叉验证**：竞品 `args.json` 内 `num_models: 47242` / `num_tasks: 2581` / `num_metrics: 3714` / `num_families: 332`，与我们**独立从 CSV 数出的值逐项精确相等** —— 语料同源已由此证明，无需再依赖对方自报。

随附 JSON：`task2id / metric2id / family2id / model2id / model2family / model_profile / model_popularity`。

### 1.2 schema crosswalk：ModelLens CSV → ModelLakeFishing HGraph

现任 D0 图有 3 个关系族（`trained_on` / `similar_to` / `is_base_of`）与 xm0/xd0 特征契约（见 `stage1BuildTransferGraph/d0_build_graph.py`）。映射如下：

| ModelLens 字段 | → | ModelLakeFishing 图元素 | 备注（**`[P0 修正]` = P0 推翻了原假设**） |
|---|---|---|---|
| `(model, dataset, value)` | → | `trained_on` 边 `y_md` + 属性 | **`[P0 修正]`** 值域实测 **[−313.25, 1e6]**、65% >1，且 `accuracy/f1/recall/…` 等 **10 个指标同名混用 0–1 与 0–100**；`accuracy` 最大值 2317（越界脏数据）。**D0 的全局 percent 探测会灾难性失效** → 必须改为**逐 (metric, dataset) 分组**定量纲 + 越界裁剪，丢弃率落档 |
| `task` 列 | → | 数据集节点 `task_type` 学习 id + vocab | **`[P0 实测]`** 2,581 个 task，且含 `HELM-classic/torr/mmlu` 等**来源标签**混入、`text-generation` vs `Text Generation` 大小写重复 → 需规范化折叠。分布极偏：**Retrieval 占 48%**（866,997 行） |
| `metric` 列 | → | 边的 metric 归一化选择 | **`[P0 修正]`** D0 的 `BOUNDED` 白名单不认 `@k` 后缀与 `accuracy_norm`，只匹配到 10.6%；改用 **@k 感知规则**（剥 `@\d+` 再匹配基名）后达 **972,215 行 = 53.8%**。**监督规模预期由 ~19 万上修至 ~97 万** |
| `dataset_desp` | → | `e_card`（MiniLM-384） | **`[P0 修正]`** 覆盖率实为 **51.8%**（首轮 12.4% 是审计 bug）；且**随候选深度上升**：深度 ≥30 达 63%、≥100 达 **87.2%** → 在真正可用作查询的深数据集上可用 |
| `model_profile.json`（size） | → | `size_bucket` 学习 id | **`[P0 修正]`** 缺失值是**字符串 `"unknown"`**，真实 size 覆盖仅 **21,460 / 47,242 = 45.4%**；值为 `'7.0'/'0.028'/7.0/'22'` **str/float 混型**需解析 |
| `model_popularity.json` | → | `e_stats` / popularity 特征 | **`[P0 修正]`** 是**包装结构**，载荷在 `["models"]`；47,242 条中仅 `status=="ok"` 的 **22,649（47.9%）** 可信（`not_found` 23,341）→ 只能作**弱信号** |
| `model2family.json` / profile.family | → | `family` 学习 id **＋ `is_base_of` 血缘边** | **`[P1 修正]`** family id 通道健康（curated 332 家族，严格湖内覆盖 66%）；**但血缘边 `is_base_of` 名字精确匹配只得 42 条**（§1.2c-bis / D-8）—— r_mm' 在此语料近乎空转，与 P0 的乐观预期相反 |
| （自建）e_card KNN k=20 | → | `similar_to` 边 `s_dd'` | 我们从 dataset_desp 自建，它们没有这条边 |

**`[P0 实测]` 查询集口径定案**：gold@10 要求候选深度 ≥10 → 可用查询数据集 **2,543** 个（其中 1,345 有描述）；深度 ≥30 的 **1,224** 个作敏感性分析。全语料 10,479 个数据集里有 7,157 个深度 ≥2（能定义「最优」）。

**LogME 迁移边**：ModelLens 语料**无**前向 LogME 观测；现任 D0 图本就只有 3 族（原始四型设计里的 LogME 边在 D0 已省）。因此**天然对齐，不需 LogME**，无需为对齐而合成。此点写进 provenance，堵住「你们少了一族边」的质疑。

### 1.2b 🔴 `[P0 实测]` 竞品**未发布**运行其模型所必需的一张矩阵

解剖 `ModelLens.pt`（21 张量 / 177.26 M 参数）后，主干输入 6016 维的归账为：

```
model_desc 1536 + id_emb 1536 + name 512 + size 64 + family 64
          + dataset_id 256 + task 256 + metric 256 = 4480
6016 − 4480 = 1536   ← 悬空,即 args.json 的 dataset_desp_emb_dim
```

- ✅ 模型侧描述嵌入**已烘进 checkpoint**（`model_desc_matrix` (47242,1536)），故未发布的 `model2desp_embeddings.npz` **不缺**。
- 🔴 **数据集侧 1536 维冻结描述嵌入矩阵，既不在语料 repo，也不在 checkpoint 中**，且编码器身份未知（1536 维指向 OpenAI `text-embedding-3-*` 一类）。

**影响**：要让它们的 checkpoint 出分就必须自行重建这张矩阵，编码器差异会引入**无法归因的偏差** —— 这是 **P4 基线可信度的头号风险**，须在 P2 前定策（**决策 D-5**，见 §7）。

### 1.2c 🔴 `[P0 实测]` 它们的「dataset」不是数据集，而是 **(dataset, task, metric) 三元组**

`dataset_id_embedding` 有 **85,939** 行（= N+2）。实测粒度比对：

| 粒度 | 唯一数 | 是否 ≈ 85,937 |
|---|---:|---|
| dataset 名 | 10,479 | ✗ |
| (dataset, task) | 15,927 | ✗ |
| **(dataset, task, metric)** | **86,196** | ✅ 最接近（差 259，应为未知 metric/过滤所致） |

**这对我们有利，且直接了结了 §2.2 的悬案**：它们的 `new_dataset_evaluation` 是按**三元组**留出的 —— 留出 `(squad, QA, f1)` 时，`(squad, QA, exact_match)` 仍可能留在训练集。**按数据集名衡量，它们的「新数据集」冷启动划分本身就存在同源泄漏**；按 dataset root 衡量差距更大。详见 §2.2。

### 1.2c-bis 🔴 `[P1 实测]` 血缘边 `r_mm'` 在此语料名字精确匹配近乎空转

P1 建图时，高精度名字血缘推断（剥 gguf/awq/q4_k_m/lora 等 marker、要求剥后基名仍在 47K 集内）只得到 **92 个 lineage_base / 42 条 pool 内边**。原因：量化/LoRA 上传者与基座模型**通常是不同 HF 用户**，且基座常不在这 47K 语料内，同名精确匹配打不中。

**这是继「数据集描述矩阵未发布」之后第二个重量级发现**，且方向与 P0 的乐观判断相反（P0 曾据 family 89.6% 覆盖判「血缘通道健康」—— family **标签**通道确实健康，但 family 是 bert/llama 这种**粗聚类**，不能当基座**身份**；`is_base_of` 需要的是基座身份，两者不同）。

**影响**：r_mm' 是 CLAUDE.md 列的核心贡献之一，也是 C 轴「仅血缘边冷启动」的关键。42 条边意味着此通道近乎失效 → **决策 D-8（P2 补强）**，倾向把 C 轴重心移到不依赖血缘边的 sibling/task 融合 + inductive 特征节点。

### 1.3 intake 复用与新建

- **复用**：`d0_intake_audit.py` / `d0_build_graph.py` 的 xm0/xd0 builder、family vocab、size bucket、task 词表逻辑全部可复用 —— 它们已经是「dataset-first、离线、契约化」的。
- **新建**：`scale/modellens_intake.py` —— 把 ModelLens CSV+JSON 转成 D0 build 期望的 `d0_observations.parquet` 等价输入（逐行 provenance：每条边回指 CSV 行号 + 源 leaderboard）。这是唯一的新增数据入口，其余走既有 build 管线。

### 1.4 数据保全：~10K 数据集 / 47K 模型语料的不可变快照（**硬纪律**）

> **这份语料是整个阶段的地基，而且它不在我们控制之下** —— 它托管在竞品的 HF repo 上。竞品随时可能更新、改 schema、限流甚至下架；一旦发生，我们已跑出的所有数字将**无法复现、无法自证**。因此：**取到手的第一件事是固化，不是建图。**

#### 保全五条铁律

1. **锁 revision，不锁 tag**。`hf_hub_download` / `load_dataset` 必须**显式传 commit SHA**（`revision="<sha>"`），不能用 `main`。把 SHA 写进 `PROVENANCE.json` 与每张结果表的脚注。`main` 会漂移，SHA 不会。
2. **raw 层只读、永不原地修改**。目录分三层，下游只准往后写，永不回写 raw：
   ```
   $MLF_DATA_DIR/modellens_v2/
   ├── raw/            # 原样落盘的 data.csv + 7 个 JSON,落盘后置只读
   │   └── PROVENANCE.json   # repo id, revision SHA, 下载时间(UTC), 各文件 size+sha256, 行数
   ├── interim/        # modellens_intake.py 的中间产物(parquet 等)
   └── derived/        # 图 .pt / embeddings / splits / 索引
   ```
   落盘后立刻 `attrib +R`（Windows）或去掉写权限，防止脚本手滑覆盖。
3. **校验和 + 行数双锚**。对 `raw/` 每个文件算 **sha256** 并记录**行数/记录数**，存进 `PROVENANCE.json`。此后每次进入 P1–P4，先跑 `scale/verify_corpus.py` 比对校验和；不匹配即**硬失败**，禁止继续。
4. ~~**三副本，至少一份离机**~~ → **`[用户裁定 2026-07-22]` 改为：单一本地副本，只放 D 盘**。
   原条款设计为 (a) 工作盘 + (b) 同机另一物理盘 + (c) 离机。**用户明确裁定「只保留在本地 D 盘，C 盘都不要」**，P0 期间建立的 C 盘副本**已按此删除**。
   **保留的仍然有效的部分**：锁 SHA、raw/ 只读、sha256 + 行锚、`verify_corpus.py` 入口守卫 —— 这些防的是**静默损坏与误改**，仍全部在岗。
   **被接受的残余风险（记录在案，非反对）**：单副本不防 D 盘物理故障、不防整机丢失。若 D 盘失效**且**上游 repo 同时被改动/下架，快照将不可恢复，本阶段全部数字失去可复现性。**这是已知并被接受的取舍。**
5. **不进 git 工作树**。语料与派生大文件一律放 `$MLF_DATA_DIR`（沿用既有 `MLF_DATA_DIR` 环境变量约定），**不**提交进仓库；仓库里只留 `PROVENANCE.json` + 校验和清单 + 拉取脚本。已有 `.gitignore` / `.gitattributes`(LFS) 需相应补规则，避免 GB 级文件误入历史。

#### 同样要保全的派生物

复现一次 head-to-head 所需的**最小闭包**，缺一不可（沿用 CLAUDE.md Step-6「重建推理链要什么」的清单纪律）：

- `raw/` 全量 + `PROVENANCE.json`
- **划分产物**：held-out dataset/model 列表 + seed（§2.2 两种口径各一份）—— 两系统读同一份，**它就是公平性本身，丢了对照即失效**
- 重建后的 HGraph `.pt`、`family_vocab.csv`、size bucket 常量版本、`e_name` seed 与 `token_dim`、`e_desc` 编码器名
- 导出的 `z_m` / `z_d`（`mappedID` 行序）+ 对应 `unique_model_id` / `unique_dataset_id` 快照
- HNSW 索引 + `prior_sidecar.npz`
- ModelLens 的 `ModelLens.pt` + `args.json`（**它们的 checkpoint 同样可能下架，一并归档**）

#### 恢复演练（不演练的备份等于没有）

P1 结束时做一次 **restore drill**：从离机副本恢复 `raw/`，跑 `verify_corpus.py` 校验和通过，再跑一次 intake 确认产出与原产物 byte 级/统计级一致。演练结果写进 P1 的验收记录。

> **闸**：`verify_corpus.py` 通过 + 三副本就位 + restore drill 通过 —— 是 P1 的**出闸条件**，不是可选项。

#### `[P0 实测]` 落地情况

已实现为 `scale/pull_corpus.py`（拉取+固化+PROVENANCE）与 `scale/verify_corpus.py`（入口守卫，不匹配 exit 1）。

| 铁律 | 状态 | 实据 |
|---|---|---|
| 1 锁 revision SHA | ✅ | 三个 SHA 写死在 `pull_corpus.REPOS`；内置守卫：Hub 返回 sha ≠ 锁定值即抛错拒跑 |
| 2 raw/ 只读 | ✅ | 22/22 文件 `ro=True`；三层 `raw/interim/derived` 就位 |
| 3 sha256 + 行锚 | ✅ | 全部入 `PROVENANCE.json`；**行锚已标注为字节级换行数，非记录数**（`data.csv` 的 `dataset_desp` 含换行，2,067,095 换行 vs 1,807,133 真实行） |
| 4 副本策略 | ✅ **按用户裁定收敛为单副本** | D-6 已裁定「只留 D 盘」；P0 期间建的 C 盘副本**已删除**，删除后 D: 重新 `verify` 22/22 全绿。残余风险见上方铁律 4 |
| 5 不进 git 工作树 | ✅ | 落 `MLF_DATA_DIR`（默认 `D:\research\model_lake\data`），仓库内只有脚本与 `P0/artifacts/` 的 JSON/日志 |

---

## §2 公平对照协议（本阶段的命门）

week8 报告已被自己钉死一条纪律：跨湖/跨查询集的比较是 system-era、非受控。本阶段成败全在**把对照做成受控**。三道闸：

### 2.1 先把 ModelLens 跑正确（基线合法性闸）

**在打败它之前，先证明我们把它跑对了。**

**`[P0 修正]` 自检方式已调整（且更强）**：完整跑通它们的 `src/main.py` 需先补齐未发布的数据集描述嵌入（§1.2b / D-5），故 P0 改用**更廉价也更硬**的自检 —— **解剖 checkpoint，把每张 embedding 表的行数与我们独立数出的语料词表逐项对账**。维度对不上就是拿错语料；全对上即同一代。

**`[P0 实测]` 结果：全部精确匹配** ⇒ 出闸通过。

| checkpoint 张量 | 形状 | 语料对账 |
|---|---|---|
| `model_desc_matrix` | (47242, 1536) | ✅ = 模型数 |
| `_id_emb.weight` | (**47243**, 1536) | ✅ = 模型数 **+1 [UNK]** |
| `task_embedding` / `metric_embedding` | (2581,256) / (3714,256) | ✅ ✅ |
| `family_embedding` / `size_embedding` | (332,64) / (23,64) | ✅ ✅ |

**两条结构性收获（直接进报告）：**
1. **`_id_emb` 的存在 = 它们是「带 ID 的半直推式」**，新模型只能落到唯一那行 `[UNK]`（`id_dropout_rate: 0.1` 正为此训练）。**这从权重层面坐实了 §4.2 C 轴的前提** —— 我们的归纳式根本没有这张表。
2. **打分主干仅 `6016→512→512→1`，约 3.3 M 参数**（177 M 里 145 M 是两张 1536 维大表）。**含义见 §4.1.1：它们 O(N) 全扫的单价很低，交叉点 $N^\*$ 可能比预期靠后** —— 原文那句「小 N 时它们可能反赢」的谨慎被证明是必要的。

> 完整评测复现（用它们的 τ_w/Hit@K）**顺延至 D-5 定策后**，仍只作接线自检、**不作**比较口径；head-to-head 一律用我们的全局指标（§2.3）。

### 2.2 同一份划分（leakage-free 闸）

- **冷启动划分对齐**：ModelLens 的 `new_dataset_evaluation` / `new_model_evaluation`（README 评测协议第 2 条）↔ 我们的 root-aware hold-out。
- **`[P0 实测]` 悬案已了结，且结论比预期更有利**：原计划要「证明 root-aware 不比它们宽松」。P0 查明它们的留出单元是 **(dataset, task, metric) 三元组**（§1.2c），**不是数据集**。因此严格性排序为

  $$
  \underbrace{(\text{dataset},\text{task},\text{metric})}_{\text{ModelLens,最松}}\;\prec\;\underbrace{\text{dataset}}_{\text{中}}\;\prec\;\underbrace{\text{dataset root}}_{\text{我们,最严}}
  $$

  即**它们的冷启动划分按数据集名就已存在同源泄漏**（同一数据集换个 metric 仍可见）。我们不但不宽松，而且严格两级。此结论必须写进报告正文 —— 它同时是**公平性辩护**与**它们冷启动数字偏乐观**的证据。
- **`[P0 实测]` 附带约束**：它们训练时 `ood_split_mode = "new_dataset_evaluation"`、`test_split_mode = "val"`、`seed = 2025`（见 `args.json`）。复用其划分时须对齐这三项。
- **决策点 D-2**：两种口径都要跑 —— (a) 严格按它们的单-dataset 划分（完全同口径，最无争议）；(b) 我们的 root-aware 划分（更严、更诚实）。报告主表用 (a) 堵嘴，附表用 (b) 展示我们更严仍赢。
- **同 seed、同 hold-out 集合**：划分产物（held-out dataset/model 列表）落盘共享，两系统读同一份。

### 2.3 单一全局指标口径（用户定调：只用我们的全局指标）

**不做双向。** 两系统都在**同一个全局指标 harness** 上打分，指标只用我们的：**gold@1 / gold@10 / top3@10**（root-macro 聚合、gold-gap@10 作 good-enough 松弛）。**不引入** ModelLens 的 local τ_w / NDCG / Hit / Rec 作比较口径（它们只在 §2.1 的接线自检里出现一次）。

harness 的统一输入是「**每个 held-out 查询数据集 × 全湖候选模型的全局排序**」：

- **我们的系统**：HNSW 检索 top-K + rerank + S2/P2b 融合 → 全湖 gold@K 排序（现任 `top1_eval.py` 已产此口径，直接复用）。
- **ModelLens**：对同一查询数据集，用它们的 MLP **对全湖每个候选模型逐一打分**（`score(query_dataset, model)`），排序得全局 ranking。**注意：这一步的 O(N) 全扫正好就是 B 轴要量的延迟**，一举两得。
- **共同查询集 + 共同候选池 + 共同 gold**：同一批 held-out 数据集、同一 ~47K 候选、同一「最优模型」标签口径；两系统各出全局 ranking，`scale/global_metrics.py` 用同一套 gold@1/gold@10/top3@10 给两边打分。

> **为什么对 ModelLens 用全局指标是公平的**：全局模型选择（「从 myriads of models 里选最优」）是 ModelLens README 标题**自称的使命**，不是我们强加的陌生任务。它的 MLP 本就输出 per-(dataset,model) 分数，全湖排序是其原生能力，只是它自报时用了更宽松的 local 口径。我们把两系统摆到同一条**更难也更贴产品**的全局轴上，公平且直击要害。
>
> **实现纪律**：`global_metrics.py` 就是现任 `top1_eval.py` 的口径，无需从 ModelLens 那边移植任何指标代码 —— 反而**规避**了「凭记忆重写别人指标导致泄漏虚高」的老坑（记忆 kendall-ranknet 的 observed-hit 教训）。唯一要新写的是「让 ModelLens 输出全湖 ranking」的适配器。

---

## §3 规模化改造：把管线从 9.5K 顶到 47K/9.6K

现任管线在 9,491 模型 / 1,463 数据集上跑通。

**`[P1 实测]` 可训湖已建成，规模低于 canonical 池**（详见 [`P1/P1_EXECUTION.md`](P1/P1_EXECUTION.md) §3.1 漏斗）：

| 层 | 模型 | 数据集/节点 | 边 |
|---|---:|---:|---:|
| canonical 语料 | 47,242 | 10,479 dataset | 1,807,133 行 |
| @k 有界行 | — | — | 972,215（53.8%） |
| **严格可训湖 `hgraph_ml_v2.pt`** | **30,183** | **9,603 (dataset,task) 节点** | **312,986 trained_on** |
| 其中 gold（depth≥10） | — | 1,686（963 含描述） | — |
| similar_to（KNN k=20） | — | — | 192,060 |
| **is_base_of（血缘）** | — | — | **42 ⚠️（见 §3.5 / D-8）** |

30,183 而非 47,242，是因为 1.7 万模型只在无界指标上被测过，**与 D0「strict = has_obs ∨ has_lineage」纪律一致**（D0 也是 9,491 严格 / 更大采集池）。相对 D0 仍是 **~3.2× 模型 / ~6.6× 节点 / ~5× 边** 的放大。逐 Stage 的风险与改造：

### 3.1 Stage-1 建图（`stage1BuildTransferGraph`）
- **特征编码吞吐**：MiniLM 编 ~47K 模型描述子 + ~9.6K 数据集卡片。批量化、缓存（复用 `d0_lake_cache/` 结构）。
- **相似边 KNN**：9.6K 数据集的 e_card KNN（k=20）—— 9.6K² 暴力可接受，但留 faiss 后路。
- **血缘边规模**：47K 模型的 family 规则推断 + `is_base_of` 边构建，注意 hub 家族（同 family 上千 derivative）的边爆炸 —— 复用现有 `FAMILY_MIN_COUNT` 折叠。
- **root 抽取**：9.6K 数据集的 root 归并（多语 config、同源 benchmark）需要在它们的命名体系上重建 root 映射（week8 root-aware 的前提）。

### 3.2 Stage-2 GraphSAGE 训练（`stage2TrainGraphSAGE`）
- **~1.6M `trained_on` 边**的 `LinkNeighborLoader` 采样 + edge dropout + L1 全湖 logQ 负采样（n_neg=256，L1L3b 配置）。5× 边规模下的显存/时长是主要不确定性。
- **feasibility 闸**：先在 47K 图上跑 1–2 epoch smoke，确认单卡可训、loss 下降、两张 embedding 表收到非零梯度（沿用 CLAUDE.md Step-5 mechanism-first 纪律）。
- **回退**：若 47K 全量训练不收敛/超显存 —— **matched-scale 子采样**（决策点 D-3）：按 root 分层抽到可训规模，两系统在**同一子集**上比。子采样**不弱化论点**，因为对照仍受控。

### 3.3 Stage-3 HNSW（`stage3HNSW`）—— 我们的主场
- 47K 模型 embedding 建 HNSW（hnswlib 0.8.0 / faiss 1.14.3 已在 stage3 就绪，见记忆 stage3-hnsw-plan）。
- **量测 B 轴**：47K 下 query 端 P50/P99 延迟 + 峰值内存，对比 ModelLens MLP 对 47K 候选的 O(N) 全扫。这是把 CLAUDE.md 的定性护城河**量化**的一次。
- fidelity：HNSW recall@50 vs 暴力 MIPS > 90%（近 hub / 远 hub 分开量，防过平滑），沿用 Stage-3 已建立的 fidelity 纪律。
- serving 融合：S2 sibling + P2b task 先验的 sidecar 在 9.6K 数据集上重建（`build_prior_sidecar`），零重训、零重索引。

---

## §4 三条论证轴的具体实验矩阵

**所有格子的指标列都只用我们的全局指标**（gold@1 / gold@10 / top3@10；root-macro 聚合 + gold-gap@10）。

| 实验 | 系统 A（ours） | 系统 B（ModelLens） | 划分 | 期望结论 |
|---|---|---|---|---|
| A1 全局单最优命中 | L1L3b + S2/P2b | ModelLens ckpt 全湖打分 | 单-dataset（同它们）+ root-aware | **我们赢 gold@1 / gold@10 / top3@10** |
| A2 good-enough 命中 | 同上 | 同上 | 同上 | gold-gap@10 我们 ≥ | 
| B1 检索延迟 | HNSW top-K | MLP O(N) 全扫（即 A 的打分步） | —— | **我们赢（亚线性）** P50/P99 + 峰值内存 @47K |
| B2 索引可扩展 | HNSW build/query | MLP 不可 ANN 化 | —— | 随 N 增长的延迟曲线（架构级差异） |
| C1 冷启动-新数据集 | S2+P2b 融合 | id_dropout | `new_dataset` hold-out | **我们赢**（sibling/task 先验），全局 gold@K |
| C2 冷启动-新模型 | 归纳式 GraphSAGE | id_dropout + [UNK] | `new_model` hold-out | 对等或更优（归纳式零样本 onboarding），全局 gold@K |

每个格子跑 8 seed（沿用 D0 root-aware 8-seed 判决纪律，防 3-seed 假信号 —— 记忆 w1-d0-landing 的教训）。

> **`[P2 实测]` A 轴的重大公平性前提（务必写进报告正文，不可只放脚注）**：P2 已用 ModelLens 自有源码 + 发布 checkpoint 建成打分器，但发现其 `dataset_desc_matrix` 与 `dataset2id` **均未发布**，从公开件出发它**只能盲数据集运行**（全 47,242 候选 gold@10≈0.002，见 [`P2/P2_EXECUTION.md`](P2/P2_EXECUTION.md) §3）。因此 A 轴对比的 ModelLens 是"**可复现 release 版**"，其低分**低估**原作满血系统，**差距源于发布不全、非同台方法优劣**。报告必须：① 主口径明写"vs 可复现 release 版"；② 反复披露致盲原因；③ 若要满血对比，P4 前需向原作者索取缺失件。**不得**把 0.002 直接叙述成"我们碾压 ModelLens"。

### 4.1 B 轴：time cost 的**理论设计 + 量化测量方案**

#### 4.1.1 理论：为什么它们有 O(N) 地板，而我们没有

设湖内模型数 $N$，嵌入维度 $d$，查询数据集 $q$。

**我们（两段式）** 的单查询成本：

$$
C_{\text{ours}}(N)=\underbrace{O(\bar{\deg}^{L}d^{2})}_{\text{查询节点 GNN 前向}}+\underbrace{O(\texttt{ef\_search}\cdot M\cdot d)}_{\text{HNSW 搜索}\;\approx\;O(d\log N)}+\underbrace{O(K\cdot c_{\text{rerank}})}_{\text{重排}}+\underbrace{O(K)}_{\text{S2/P2b 融合}}
$$

第一项与 $N$ **无关**（只依赖查询节点的 $L$ 跳邻域），第三、四项只依赖 $K$。**全式对 $N$ 亚线性。**

**ModelLens（cross-feature MLP）** 必须对每个候选求值 $f_\theta(x_q,x_m)$：

$$
C_{\text{ML}}(N)=\Theta\!\left(N\cdot c_{\text{MLP}}\right),\qquad c_{\text{MLP}}=\textstyle\sum_l h_{l-1}h_l
$$

**为什么它不能被 ANN 索引化（这是护城河的形式化根因，务必写进报告）**：ANN/MIPS 索引成立的前提是打分函数可分解为**查询无关的模型向量**与查询向量的内积（或其单调变换）：$s(q,m)=\langle \phi(q),\psi(m)\rangle$。我们的 $s=\langle z_q,z_m\rangle$ 恰好满足，故 $\psi(m)$ 可离线建索引。而 ModelLens 的 $f_\theta$ 在隐层里把 $x_q$ 与 $x_m$ **交叉混合**，不存在这样的分解，因此**不存在**把候选集预索引的等价重写 —— $\Theta(N)$ 不是实现问题，是**架构下界**。

**离线/摊销成本单列**（不混进单查询延迟）：我们 HNSW 建索引 $O(N\log N)$ 一次性 + GNN 训练一次性；它们无索引、训练一次性。两边的一次性成本**分开报**，不用来掩盖或夸大 query 成本。

**诚实的交叉点分析**：小 $N$ 时，GPU 上批量 MLP 全扫很快，可能反而**打赢**我们的 GNN 前向 + HNSW 常数开销。因此必须测出**交叉点 $N^\*$**（护城河从哪里开始生效），而不是只报 47K 单点。测出 $N^\*$ 才是科学结论；只报一个点是营销。

> **`[P0 实测]` 这条谨慎已被证明必要。** 解剖 checkpoint 后：它们的打分主干只有 `6016→512→512→1`、**约 3.3 M 参数**（177 M 总参里 145 M 是 `model_desc_matrix` + `_id_emb` 两张查表，**查表是 $O(1)$ 索引、不参与每对的乘加**）。也就是说 $c_{\text{MLP}}$ 很小、且极易 GPU 批量化，**它们的 $\Theta(N)$ 常数因子低**。
> **后果**：$N^\*$ 可能显著高于直觉，47K 未必足以让延迟差拉开数量级。**B 轴的正确表述因此应落在「曲线斜率」而非「单点倍数」** —— 我们的曲线近似平坦、它们必然线性上扬，这个**渐近性质**才是不可辩驳的护城河；单点倍数则要老实报多少就是多少。若 47K 处差距不大，**照实写，并用外推曲线说明 100K/1M 湖的必然结局**。

#### 4.1.2 测量方案：计时器与协议

**计时器**
- CPU 段：`time.perf_counter_ns()`（单调、纳秒级）。**禁用** `time.time()`（受系统时钟调整影响）。
- GPU 段：**必须** `torch.cuda.synchronize()` 包裹，或用 `torch.cuda.Event(enable_timing=True)` 的 `start/end.elapsed_time()`。
  > **头号陷阱**：CUDA kernel 异步下发，不同步就只测到「下发耗时」，会把 ModelLens 的 O(N) 全扫测成假的飞快，直接毁掉整条 B 轴的可信度。此条写进 code review checklist。

**协议**
| 项 | 规定 |
|---|---|
| 预热 | 丢弃前 **10–20** 次（排除 JIT / cuDNN autotune / 惰性初始化 / 缺页） |
| 样本量 | ≥ **100 个不同查询数据集** × ≥ **5 轮重复** |
| 交替执行 | **A,B,A,B** 交替，**不**分块跑（抵消热漂移与后台负载） |
| 统计量 | 报 **P50 / P90 / P99 + mean±std**，**不**只报均值（尾延迟才是服务质量） |
| 计时边界 | **只计 query 端**：查询特征 → 最终 top-K 列表。**排除**模型加载、索引加载、语料加载 —— 这些一次性成本**单独列表**报 |
| 给对手最优待遇 | ModelLens 的全扫**允许 GPU 批量化**（batch 全部 N 个候选）、允许其原生 fp16/AMP 设置。**不能拿逐条 for 循环去测它** —— 那是稻草人，一旦被识破整份报告失信 |
| 硬件固定 | 同机、同 GPU、同 CPU 亲和性、无其他负载；记录 GPU 型号/驱动/CUDA/torch 版本、可行则锁频 |

**必测曲线（报告的核心图）**：对 $N\in\{1\text{K},2\text{K},5\text{K},10\text{K},20\text{K},47\text{K}\}$ 按 root 分层子采样湖，画 **latency–vs–N**。预期我们近似平坦/对数、它们线性；**交叉点 $N^\*$ 从图上读出**。这张图比任何单点数字都有说服力。

**同时记录**：峰值显存（`torch.cuda.max_memory_allocated`）+ 进程 RSS、索引磁盘体积、索引构建时长、吞吐量（queries/sec）。

**实现**：`scale/bench_latency.py`，统一 harness 同时驱动两系统，输出 `latency_raw.parquet`（每次计时一行，便于事后重算分位数）。

### 4.2 C 轴延伸：**可扩展性报告 —— inductive 是否真能让新模型/新数据集快速插入**

> **待证命题（不是假设）**：CLAUDE.md 断言「无 `mappedID` ID-embedding ⇒ 新模型可零样本 onboarding」。这是**设计意图**，本阶段必须把它**变成实测数据**。若证伪，是重大发现，必须照实写。

产出独立交付物 **`docs/scale/SCALABILITY_REPORT.md`**，回答四问：

**问 1 — 零样本 onboarding 在质量上成立吗？**
把新模型/新数据集完全排除在训练外（`new_model` / `new_dataset` hold-out），仅用其自身特征 + 与既有节点的边，过**已训练**的 GNN 得 $z$。对照 **transductive oracle**（把它们纳入训练后重训）。
- 指标：全局 gold@1/gold@10/top3@10（同 §2.3 口径）
- **inductive gap** = oracle − zero-shot，**这就是「归纳式」的真实代价**，必须量化，不能回避。

**问 2 — 插入要多久？（「快速插入」的正面量化）**
onboarding **1 / 10 / 100 / 1,000 / 10,000** 个新模型的 wall-clock，分解为三段：
特征抽取（e_name/e_desc/size/family）→ GNN 前向得 $z_m$ → **HNSW `add_items` 增量插入**。
对照基线：**全量重训 + 全量重建索引**的耗时。
- 期望结论：增量路径比全量重建**低若干数量级**；给出「每千模型 onboarding 成本」的单位数字。

**问 3 — 增量插入会不会把索引搞坏？（运维上最要命的一问）**
HNSW 增量插入会随插入比例累积图质量退化。测：插入 1% / 5% / 10% / 25% / 50% 新模型后，**recall@50 vs 暴力 MIPS**（近 hub / 远 hub 分开量，沿用 Stage-3 fidelity 纪律）。
- 交付：**重建阈值建议** —— 「插入超过 X% 后必须 rebuild」。这是可直接落进运维手册的结论。

**问 4 — 边稀疏到什么程度还活着？（冷启动的真实分布）**
新节点通常没有性能边。按新模型的邻域条件分层测：
- (a) **孤立**（度 0，纯节点特征驱动）—— edge dropout 训练的正是这种鲁棒性，此处验收；
- (b) **仅血缘边**（既有 base 的新 finetune）—— **最真实、也最该赢**的场景；
- (c) 血缘 + 少量性能边。
新数据集侧同理，并叠加 S2/P2b（sibling/task 先验在新数据集上正是为此设计）。

**与 ModelLens 的诚实对照**
它们的新模型走 `[UNK]` ID + 元数据先验，也能零样本打分；而且**它们没有索引，插入对它们近乎免费** —— 这一点必须写明，**不得宣称我们在插入成本上碾压它们**。精确的主张是：

> 我们同时拿到「**亚线性查询**」与「**足够便宜的插入**」；它们插入便宜，但**查询永远 $\Theta(N)$**。

因此报告用 **TCO 视角**收口：给定 onboarding 速率 $\lambda$（模型/天）与查询量 $Q$（次/天），把两系统的日总成本写成 $\lambda\cdot c_{\text{insert}}+Q\cdot c_{\text{query}}$，画出**我们占优的 $(\lambda,Q)$ 区域**。这比单点对比诚实得多，也更有说服力。

---

## §5 分阶段落地路线图（带闸）

| Phase | 内容 | 通过闸（gate） | 依赖 |
|---|---|---|---|
| **P0** ✅ **已完成** | 三 repo 按锁定 SHA 拉取（2,509 MB）+ sha256 + `PROVENANCE.json` + 只读；**双副本校验全绿**；计数对账；**checkpoint 词表逐项对齐**作接线自检 | ✅ `verify_corpus` 主/副本各 22/22；✅ 词表精确匹配 —— **出闸通过**（离机副本 D-6、完整评测复现顺延至 D-5） | — 见 [`P0/P0_EXECUTION.md`](P0/P0_EXECUTION.md) |
| **P1** ✅ **已完成** | `modellens_intake.py`（6 项 P0 修正 + 逐行 provenance）→ `modellens_build_graph.py`（复用 D0 builder）→ `hgraph_ml_v2.pt`；`p1_restore_drill.py` 出闸 | ✅ Stage-2 契约 assert 全过；✅ intake 确定性复跑三产物逐件 MATCH；实测湖 **30,183 / 9,603 / 312,986**（漏斗见 [`P1/P1_EXECUTION.md`](P1/P1_EXECUTION.md) §3.1） | P0 ✅ |
| **P2** ✅ **已完成** | `scale/global_metrics.py`（gold@1/@10/top3@10 + root-macro + gold-gap@10）+ `scale/modellens_adapter.py`（忠实重建 ModelLens 打分器） | ✅ harness 与 `top1_eval` 逐查询对账一致；✅ ModelLens 重建 wiring 全正（Pearson 0.65–0.75）；ModelLens 首个基线数字已出（**盲数据集口径**，见 D-5 升级） | P0 ✅, P1 ✅ |
| **P3** ✅ **已完成** | `scale/modellens_subsample.py`（D-3 兜底 12K/3K）+ `scale/export_ours.py`（L1L3b 训练 → z_m/z_d + HNSW + gold 标签导出） | ✅ smoke 机制门过；✅ HNSW recall@50 = 1.000（>90%）；✅ harness 端到端一致；我方 gold@10 = 0.416（[`P3/P3_EXECUTION.md`](P3/P3_EXECUTION.md)）；全 47K 受 D-9 阻 | P1 ✅ |
| **P4（A/B）** ✅ **已完成（A 轴降级为附录）** | `scale/head_to_head.py`：同 12K 宇宙/517 查询/harness。B 轴 HNSW 亚线性 vs O(N)，66×→443× 随 N 放大；A 轴 vs **盲 release** gold@10 0.416 vs 0.012 —— **经 v2 修订降为附录**（盲对比不作主张） | ✅ B 轴成立且升为头号；A 轴盲对比留档；泄漏已修 | P2 ✅, P3 ✅ |
| **P5 同条件重训（A/B/C）** ✅ **已完成 · v2 新主 A 轴** | `scale/retrain_modellens.py`（其 MLP+损失族+超参，喂同一份 MiniLM 嵌入）+ `scale/cold_start_axis.py`（C 轴分层）。盲→非盲 0.012→0.437 | ✅ loss 降(2.39→1.73)、wiring 0.675；**A 平手(ML 0.437 vs 我 0.416)、B 我方胜(66→443×)、C 近平手(最冷段我方 2×)** | P2 ✅, P3 ✅ |
| **P4b** | **`bench_latency.py`（§4.1）**：CUDA-sync 计时、A/B 交替、latency–vs–N 曲线、交叉点 $N^\*$ | 分位数齐全；对手已按最优待遇（GPU 批量）测；曲线可读出 $N^\*$ | P3 |
| **P4c** | **可扩展性/inductive 验证（§4.2）**：四问全测 → `scale/SCALABILITY_REPORT.md` | inductive gap 已量化；给出 HNSW 重建阈值；TCO $(\lambda,Q)$ 区域图 | P3 |
| **P5** | 战报 `scale/HEADTOHEAD_RESULTS.md`（对标 week8 progress 的写法与诚实度） | 边界诚实、可复现命令齐全；三份交付物齐（RESULTS / SCALABILITY / latency 原始表） | P4, P4b, P4c |

**关键路径**：P0→P1→P3→P4。P2 与 P1/P3 可并行。**P0 的基线复现是硬前置** —— 复现不了它们的数字，后面全塌。

---

## §6 风险与反制

| 风险 | 后果 | 反制 |
|---|---|---|
| **「数据不完全相同」质疑** | 论点被否 | 逐行 provenance（每条边回指 CSV 行）；划分产物落盘共享；主表用它们的单-dataset 划分（§2.2a）最无争议 |
| **基线复现失败** | 无法合法宣称更优 | P0 硬闸；对齐 v2 + 它们的 `args.json`；复现不过不进 §4 |
| **指标口径被指偏袒** | 「你们用自家指标当然赢」 | 全局模型选择是 ModelLens 自称使命（README 标题），全局 gold@K 对两系统对称、公平；且 §2.1 已用它们自家指标证明基线跑对 |
| **让 ModelLens 出全湖 ranking 不当** | B 侧比较失真 | 直接用它们 MLP 的原生 per-(dataset,model) 分数全湖排序，不改其模型；适配器只做调用与收集 |
| **上游语料被改/下架/限流** | 已跑数字无法复现、自证失败 | §1.4 五条铁律：锁 revision SHA + raw 只读 + sha256 + **三副本含离机** + restore drill；竞品 checkpoint 一并归档 |
| **派生物/划分产物丢失** | 对照失效（划分即公平性本身） | §1.4「最小闭包」清单纳入备份；划分列表与 seed 落盘共享 |
| **CUDA 异步计时假快** | B 轴数字失真、整份报告失信 | 强制 `torch.cuda.synchronize()` / `cuda.Event`；写进 code review checklist（§4.1.2 头号陷阱） |
| **被指用稻草人测对手延迟** | B 轴失信 | 给 ModelLens 最优待遇：GPU 批量全扫 + 其原生 fp16；协议写进报告 |
| **inductive 承诺被证伪** | CLAUDE.md 的核心卖点动摇 | §4.2 把它当**待证命题**而非假设；若 inductive gap 过大，照实写并转为「需定期增量重训」的运维结论 |
| **HNSW 增量插入退化** | 线上 recall 悄悄下滑 | §4.2 问 3 测退化曲线，产出**重建阈值**并落运维手册 |
| **47K 训练超显存/不收敛** | P3 卡死 | matched-scale root 分层子采样（D-3）；对照仍受控，论点不弱 |
| **某设定上仅打平** | A 轴叙事弱 | 诚实承认打平，用 B（延迟）+ C（冷启动）补足；不粉饰 |
| **corpus 版本混淆** | 复现对不上 | 全程锁 v2；v1 仅作鲁棒性附录；版本号写进每张表脚注 |
| **root 抽取在它们命名体系上不准** | 泄漏或过严 | 双口径（单-dataset + root-aware）并报，让审稿人看到更严口径下仍赢 |

---

## §7 待决策点（需用户拍板，默认值已给）

- **D-1 语料版本** ✅ **`[P0 已定案]`**：**v2**，revision `57a692cc…`。同源性已由 checkpoint 词表逐项对账证明（§2.1），非假设。v1（`df6cb242…`）已一并归档作干净对照。
- **D-2 划分口径**：默认**双跑** —— 主表它们的单-dataset 划分，附表我们的 root-aware。
- **D-3 规模**：默认**先 47K 全量 smoke**，超资源则 root 分层子采样到可训规模；两系统同子集比。
- **D-4 主轴叙事**：默认 **A 为正面主张（同数据、全局指标 gold@1/gold@10/top3@10 更优），B+C 为护城河**。**指标口径已由用户定死：只用我们的全局指标，不用 ModelLens 的 local 指标**（此项非待决，已锁）。若用户想把叙事重心放到「可扩展性护城河」，调整 §4 权重与报告结构。

**`[P0 新增]` 两项新决策：**

- **D-5 ✅ `[v2 解决 2026-07-27]` 已破局：不用私有件，自建 desc 嵌入公平重训**
  P2 曾判 ModelLens 从公开件只能盲跑（下方 P2 升级记录仍存档）。**v2 修订推翻此困境**：desc 的**原始文本 `dataset_desp` 就在 corpus-v2**（held-out gold 覆盖 61%，其余对称回退），我们自己用公开 MiniLM 重嵌入、同一份喂两系统、按 P5 **同条件重训** ModelLens 的方法 ⇒ 它不再盲。盲 release 版降为附录。**满血复现**（用它原 1536 维 OpenAI 编码器 / 索取 `train_vecs.npz`）作可选交叉验证。详见顶部「方案修订 v2」+ 阶段 P5。

- **D-5（历史记录）⚠️ `[P2 升级 2026-07-23]` ModelLens 从公开件只能"盲数据集"运行**
  原裁定 (a)+(c) 并报已执行，但 P2 实测（[`P2/P2_EXECUTION.md`](P2/P2_EXECUTION.md) §3.1）发现问题比"一张矩阵缺失"更深：
  - ModelLens 的**每数据集**信号有**两个**来源 —— 学习的 `dataset_id_embedding`（按 dataset2id 查）+ 冻结 `dataset_desc_matrix` —— **两者所需的映射/矩阵都未发布**；
  - 用我们 e_card 的 MiniLM（384 维）重建 desc，对其 1536 维（疑 OpenAI）backbone 是 **OOD、无效**：(a)≈(c)，median rank 47,242 里只差 1.5 名；
  - ⇒ **从公开发布件出发，任何具体命名数据集都只能走 unk**，ModelLens 只能凭 `task+metric+模型特征` 排序，全 47,242 候选 **gold@10≈0.002**。
  - **口径裁定（供 P4）**：与"**可复现 release 版**"（盲数据集）对比为主口径，**必配披露**其被未发布件致盲、**低估**原作满血系统；不确定性在"盲版 vs 满血不可复现版"之间，而非 a–c 之间。满血对比需向原作者索取 `train_vecs.npz` + dataset2id（P4 前的外部依赖，可选）。
  - **留痕（用户硬要求，措辞已按上升级）**：每处引用 ModelLens 数字的产物须复述 —— ① desc 矩阵**与 dataset2id 均未发布**；② 自建 MiniLM 重建 desc **属 OOD 无效**；③ 故为**盲数据集 release 版**，**低估**原作满血系统。
- **D-6 ✅ `[已裁定 2026-07-22]` 副本策略 = 只保留本地 D 盘**
  用户裁定「只保留在本地 D 盘，C 盘都不要」。P0 期间建立的 C 盘副本**已删除并复验**。§1.4 铁律 4 已相应改写，残余风险（单盘故障 + 上游下架的联合失效）**已记录并被接受**。

**`[P1 新增]` 两项新决策：**

- **D-7 🟠 head-to-head 候选宇宙 = 30K 严格湖 还是 47K 全池？**（P4 前定）
  P1 实测严格可训湖 30,183，而语料 canonical 池 47,242（差的 1.7 万只在无界指标上被测）。两选：
  - **(i)** 两系统都限定在 30,183 严格湖 —— 同宇宙、最无争议、可比；
  - **(ii)** 全 47,242（1.7 万无边模型作**冷特征节点**入图）—— 更接近「全湖检索」，且正好喂 C 轴 inductive/冷启动。
  - **倾向**：P4 主表用 (i)；C 轴专门用 (ii)。**待 P4 定。**
- **D-8 🔴 血缘边补强（P2 处理）**
  名字精确匹配只得 42 条 `is_base_of` 边（§3.5）—— 因量化/LoRA 上传者与基座常不同用户、基座常不在语料内。r_mm' 是核心卖点，42 条近乎空转。候选：
  - **(a)** 跨用户段匹配（剥 marker 后按最后一段 fuzzy 匹配，牺牲精度换召回）；
  - **(b)** family+size+公共前缀聚类近似血缘（弱标签）；
  - **(c)** 承认此语料血缘稀疏，C 轴重心移到 **sibling/task 融合（S2/P2b）+ inductive 特征冷节点**（都不依赖血缘边，P0 已证 sibling 通道健康）。
  - **倾向 (c) 为主 + (a) 试点**：不为凑边数牺牲精度。**待 P2 定。**

- **D-9 🔴 `[P3 新增]` 全 47K/30K 规模受阻于硬件 + pyg-lib 缺失（需裁定）**
  8 GiB 笔记本 GPU 装不下 30K 的 O(N²) 对比损失（[`P3/P3_EXECUTION.md`](P3/P3_EXECUTION.md) §1.2/§4.2），torch 2.12 又无 pyg-lib 预编译轮子拿不到邻域采样 loader。现按 D-3 子采样到 **12K/3K**（本机最大受控规模，≈1.3× D0）。选项：
  - **(a)** 换 ≥24 GB GPU / 云机跑全 30K；
  - **(b)** 装/编译 pyg-lib（换到有轮子的 torch 版本）→ 邻域采样 loader，8 GiB 也能训全 30K；
  - **(c)** 接受 12K 受控口径，P4 明写"受控子采样规模"，B 轴用**延迟-vs-N 曲线外推** 47K/1M 的渐近结论（曲线斜率不依赖训练规模）。
  - **倾向 (c) + 视资源补 (a)**：(c) 已足以支撑受控 head-to-head 与渐近护城河；全规模作 stretch goal。

> 其余默认值可直接执行。**D-1 / D-5 / D-6 已裁定**；**D-7 / D-8 / D-9 待 P4 / P2 / 资源定**。P4（head-to-head）输入已就绪：我方 `z_m/z_d` + HNSW + gold 标签（`P3/exports/`），ModelLens 适配器（P2），统一 harness（P2）。

---

## 附录 A — ModelLens 侧关键事实速查

- **语料**（**`[P0 实测]` 取代 paper headline**）：**1,807,133 行 / 47,242 模型 / 10,479 数据集名 / 2,581 task / 3,714 metric**；CSV 6 列 `task, dataset, model, metric, value, dataset_desp` + 7 个 vocab/profile JSON。**内部 dataset id 空间 85,937 = (dataset,task,metric) 三元组**（§1.2c）。
- **模型**（**`[P0 实测]` 来自 checkpoint 解剖 + `args.json`**）：`MLPMetricFull`，主干 `6016→512→512`，pairwise/pointwise/prior 三头，**总参 177.26 M**（其中 145 M 是 `model_desc_matrix` 47242×1536 与 `_id_emb` 47243×1536 两张查表，**净 MLP 仅 ~3.3 M**）。`loss_type=ensemble`、`λ_list=0.5`、`λ_pair=1.0`、`point_loss_weight=0.1`、`τ=10.0`、`id_dropout_rate=0.1`、`dropout=0.02`、`hidden=512`、`seed=2025`、`ood_split_mode=new_dataset_evaluation`。**带 `_id_emb` ⇒ 半直推式，新模型只能落 [UNK]**。**O(N) 全扫，不可 ANN 化**（我们 B 轴的根因）。
- **评测**：Kendall-weighted τ_w（主）+ NDCG@K / Hit@K / Rec@K，`topk=[1,10,30,50]`；performance-completion + cold-start（new_dataset / new_model）。**本阶段不采纳这些 local 指标作比较口径**（§2.3）。
- **实现入口**：`src/main.py`（YAML→build→train→eval）；指标在 `module/utils/metric.py`。
- **🔴 未发布的必需件**：数据集侧 1536 维冻结描述嵌入矩阵（§1.2b，决策 D-5）。

## 附录 B — ModelLakeFishing 侧现状速查（对标锚点）

- **现任冠军**：L1L3b（L1 全湖 logQ 采样 softmax + L3 native task 元数据 + n_neg=256）+ serving-time S2 sibling / P2b task 融合（α=β=1，零重训零重索引）。
- **D0 湖**：9,491 模型 / 1,463 数据集 / 613 gold / 211 gold roots；root-aware 8-seed 判决。
- **架构**：Stage-1 建图（4 关系族，D0 落 3 族）→ Stage-2 归纳式异构 GraphSAGE（`z_m`/`z_d`）→ Stage-3 HNSW 检索 + metric-aware rerank。
- **护城河**：归纳式（新模型零样本 onboarding）+ 亚线性 HNSW + serving 融合冷启动工具 —— 三者正是本阶段要在 47K 上量化坐实的东西。
