# P2 执行报告：全局指标 harness + ModelLens 全湖打分适配器（D-5 落地）

**执行日期：** 2026-07-23
**对应规划：** `docs/scale/MODELLENS_HEADTOHEAD_SCALE_PLAN.md` §2.1 / §2.3、§5 P2 行、决策 D-5
**状态：** ✅ **P2 出闸通过** —— 全局指标 harness 与 `top1_eval` 对账一致；ModelLens 用其**自有源码 + 发布 checkpoint** 忠实重建，接线自检强正相关；**D-5 (a)+(c) 双路已实现并跑出首个数字**
**产物：**
- 代码：`scale/global_metrics.py`、`scale/modellens_adapter.py`
- 数据：`docs/scale/P2/artifacts/modellens_baseline_*.json`

---

## 0. 一句话结论

本阶段造好了 head-to-head 的**度量脊柱**：一个系统无关的全局指标 harness（gold@1/gold@10/top3@10 + root-macro + gold-gap@10），并**用 ModelLens 自己的源码 + 发布权重忠实重建了它的打分器**，让它能在**我们的**指标上被评。两条硬证据：harness 与既有 `top1_eval` 在 gold@K 上**逐查询 rank 完全一致**；重建的 ModelLens 打分与观测精度的 **Pearson 相关 0.65–0.75、97–100% 查询为正** —— 证明前向忠实。

**但跑通后浮出一个比 D-5 预想更严重的发现（§3.1）**：ModelLens 的**每数据集**输入有两个来源（学习的 `dataset_id_embedding` + 冻结 `dataset_desc_matrix`），而**两者所需的 dataset2id 映射与 desc 矩阵都未发布**。用不同编码器重建 desc 又属 OOD（a≈c，无效）。结果：**从公开发布件出发，ModelLens 只能"盲数据集"运行**（仅凭 task+metric+模型特征排序），全 47,242 候选上 gold@10≈0.002。这**不是**我们碾压它，而是它被"发布不全"致盲、**低估**了原作满血系统 —— P4 必须诚实按此口径处理（§3.2）。

---

## 1. 全局指标 harness（`scale/global_metrics.py`）

### 1.1 设计：一套数学，两个入口
用户 2026-07-22 定死：两系统都只用**我们的**全局指标，不采纳 ModelLens 的 local τ_w/NDCG/Hit/Rec。harness 提供两个入口、共用同一核心，保证两系统走**完全相同**的打分：

- `from_embeddings(z_m, z_d, cands, roots)` —— MIPS 点积（我方系统，P3 产出 z 后用）；
- `from_scores(scores_by_query, cands, roots)` —— 任意打分器（ModelLens，其分数非点积）。

指标定义（$r_d(m)$ = m 在 d 的全湖排名，$g_d$ = 标注最优，$T_d^{(3)}$ = 标注前三，$\text{near}_\delta$ = 精度距最优 ≤δ 者）：

$$
\text{gold@}K=\mathbf 1[r_d(g_d)\le K],\quad
\text{top3@10}=\mathbf 1[\min_{m\in T_d^{(3)}}r_d(m)\le10],\quad
\text{gold-gap@}K=\mathbf 1[\min_{m\in\text{near}_\delta}r_d(m)\le K]
$$

排名 1-indexed、tie-safe（`rank = #严格更优 + 1`）。root-macro 先按 root 内平均再对 root 平均（root_a=dataset 名 / root_b=规范化，见 P1）。

### 1.2 验证：与 `top1_eval` 对账（防"凭记忆重写指标"老坑）
`--selftest` 在随机 z + 随机候选上，对同一 (scores, labels) **同时**跑我方 harness 与既有 `stage2TrainGraphSAGE/top1_eval.py` 的 `five_metric_eval`：

```
gold@1  ours=0.0000  top1_eval=0.0000
gold@10 ours=0.0500  top1_eval=0.0500
top3@10 / gold-gap@10 hand cases: OK
*** SELFTEST PASSED: harness matches top1_eval on gold@K ***
```

且**逐查询 gold_rank 断言完全相等**。top3@10 / gold-gap@10（`top1_eval` 未实现）用手算小样例断言。**结论：harness 与既有评测器口径一致、无 bug**，且未从 ModelLens 移植任何指标代码 —— 规避了记忆 [[kendall-ranknet-win]] 的 observed-hit 泄漏教训。

---

## 2. ModelLens 全湖打分适配器（`scale/modellens_adapter.py`）

### 2.1 忠实重建（不是仿写，是跑它自己的代码）
适配器把 `ModelLens/` 加入 path，**import 它自己的 `ModelLens` 类**，用发布的 `ModelLens.pt`（`strict=False`）加载。关键工程点：

- **候选宇宙 = 全 47,242 模型，按 global-id 顺序排列** ⇒ 它的 `build_model_cache` 内部 `arange(M)` 恰好等于 global id，`_id_emb[id]` 与 `model_desc_matrix[id]` 天然对齐，无需 id 重映射。
- **buffer 尺寸修复**：`model_desc_matrix` 初始化依赖 `model2id_path`（默认缺失→建成 `[0,1536]`，导致 `strict=False` 仍因 size mismatch 报错）。指向冻结的 `model2id.json` 后建成 `[47242,1536]`，checkpoint 正常覆盖为真值。
- **词表全用 ModelLens 自己的**：task2id / metric2id / family2id / model2id（同源语料，直接查表）。size bucket 用 args.json 的边界 searchsorted 重建（其发布件缺 `modelid2bucket`，**它自己的默认是全 0**；`--size-zeros` 可复现该默认）。

### 2.2 载入对账：4 个 missing key 全部良性
`load_state_dict(strict=False)` → **missing 4 / unexpected 0**：

| missing key | 性质 | 是否影响前向 |
|---|---|---|
| `dataset_desc_matrix` | D-5 未发布件 → 保持零（= 下界 c） | 是（D-5，已按 (a)/(c) 处理） |
| `model_embedding.weight` | 基类 MLP 的并行 id 表，ModelLens 用 `_id_emb` 取代 | **否**（死参数，前向不用） |
| `model_info_encoder.tok_emb.weight` | 基类名字编码器，ModelLens 用 `_name_encoder` 取代 | **否**（死模块） |
| `model_info_encoder.id_emb.weight` | 同上 | **否**（死模块） |

而 ModelLens **真正**用的 `_id_emb.weight (47243,1536)`、`_name_encoder.tok_emb.weight (10000,512)`、`model_desc_matrix (47242,1536)`、task/metric/size/family 表、backbone、三头、temperature **全部从 checkpoint 载入**。**结论：活跃计算图 100% 来自发布权重**；4 个 missing 里 3 个是基类死模块、1 个是 D-5。

### 2.3 接线自检：前向忠实性
在 60 个 gold 查询上算 **Pearson(ModelLens 打分, 观测精度)**：

```
[wiring] mean Pearson(score, observed acc) = 0.7522  (frac>0 = 1.00)
```

ModelLens 本就被训练来把高精度模型打高分；**0.75 的正相关、100% 查询为正**，是重建忠实的强证据（若前向拼错，相关会塌到 0 附近）。这也补上了 P0 §2.1 因 D-5 而顺延的"接线自检"（当时无法跑其完整 eval；现用其自有源码达成，且**未触碰 local 指标**）。

### 2.4 D-5 落地：(a)+(c) 双路（用户要求留痕）
- **(c) 下界**：checkpoint 无 `dataset_desc_matrix` → 保持零，即 ModelLens 自身在"只有发布件"时的默认行为。**这是诚实的地板**。
- **(a) 主基线**：用**与我们 e_card 相同的 MiniLM**（all-MiniLM-L6-v2, 384 维）编码 `dataset_desp`，填入 1536 槽的前 384 维（用它自己 `use_dim=min(...)` 的部分填充约定）。held-out 数据集无 dataset2id（也未发布）⇒ id 走 unk，仅 desc 通道注入 —— 正是"新数据集"的冷启动路径。

> **留痕（每处引用 ModelLens 基线数字的产物都须复述，D-5 硬要求，已按 §3.1 升级）**：① 数据集描述嵌入矩阵**与 dataset2id 映射均未随语料/checkpoint 发布**；② 我方用**自建 MiniLM 替代嵌入**（all-MiniLM-L6-v2 / 384 维 / `scale/modellens_adapter.py`）重建 desc 属 **OOD、无效**（a≈c）；③ 故从公开件出发 ModelLens **只能盲数据集运行**，其数字是"**可复现 release 版**"，**低估**原作满血系统；不确定性在"盲版 vs 满血不可复现版"之间，而非 a–c 之间。

---

## 3. 首个数字：ModelLens 在我们全局指标上（单边预览）

| 指标（全 47,242 候选，1,650 gold 查询） | desc=c 下界 | desc=a 主(MiniLM) | 随机基线 |
|---|---:|---:|---:|
| gold@1 | 0.0006 | 0.0006 | 0.00002 |
| gold@10 | 0.0024 | 0.0024 | 0.0002 |
| top3@10 | 0.0053 | 0.0053 | ~0.0006 |
| gold-gap@10 | 0.0771 | 0.0771 | — |
| median gold rank | 5762 | 5760.5 | 23621 |
| wiring Pearson(score,acc) | 0.647 | 0.647 | 0 |

### 3.1 🔴 关键发现：ModelLens 从"仅发布件"只能**盲数据集**运行，(a)≈(c)

**(a) 与 (c) 几乎完全相同**（median 5762 vs 5760.5，47,242 里差 1.5 名）。注入 MiniLM 描述**机械上生效**（分数确有微变，非 no-op），但**语义上被忽略**。原因有二，且比 P0/D-5 预想更严重：

1. **编码器/维度不匹配 = OOD**：backbone 在 1536 维（疑 OpenAI）描述上训练；把 384 维 MiniLM 塞进前 384 维，是分布外输入，其对应权重期待的是另一种分布，故近乎噪声被无视。
2. **更根本 —— dataset2id 也未发布**：ModelLens 的**每数据集**信号有两个来源：学习的 `dataset_id_embedding`（按 global dataset id 查）+ 冻结 `dataset_desc_matrix`。**两者所需的映射/矩阵都未发布**。因此从公开件出发，任何具体命名数据集**都只能走 unk** —— ModelLens 只能凭 `task_id + metric_id + 模型侧特征` 排序，**看不见"这是哪个数据集"**。

**证据自洽**：wiring 相关 0.65 为正（在**已标注小候选集内**排序合理），但全 47,242 的 gold@10≈0.002（gold 沉到中位 5762 名）。因为盲数据集时它给 (task,metric) 挑**泛化最强**的模型，而非**该数据集专属**最优 —— 于是数千个泛化强的无标注模型盖过了 dataset-specific 的 gold。这正是"看不见数据集"的必然表现。

### 3.2 对 head-to-head 公平性的影响（必须诚实处理）

- **这不能被解读为"我们碾压 ModelLens"**。0.002 是 ModelLens 在**被未发布件致盲**下的表现，**低估**了 paper 里那个训练/评测时握有 desc 矩阵与 dataset id 的完整 ModelLens。差距源于**发布不全**，不是同台竞技下的方法优劣。
- **诚实且可辩护的口径**（供 P4 采纳）：
  - **主口径**：与"**从公开发布件可复现的** ModelLens"对比（即盲数据集版）——这回答"用户拿它的 release 能得到什么"，公平且真实；
  - **必配披露**：明确标注该版本因**未发布 dataset_desc 矩阵 + dataset2id 映射**而被致盲，**低估**原作完整系统；
  - **可选**：若要"满血"对比，需向原作者索取 `train_vecs.npz` + dataset2id（或其 desc 编码器），属 P4 前的外部依赖。
- **D-5 结论升级**：原 (a) 主 / (c) 下界的"带宽"实为**塌缩到一点**（a≈c）——不确定性不在 a 与 c 之间，而在"**盲数据集可复现版 vs 满血不可复现版**"之间。这是比 P0 预判更强的结论，必须写进每处引用 ModelLens 数字的脚注。

> **口径与边界**：查询集 = P1 的 **1,686 个 gold 节点（depth≥10）**；候选宇宙 = 全 **47,242** 模型；held-out 数据集全走 unk id（冷启动）。**我方系统数字待 P3 训练出 z 后才有**，故此表是 ModelLens 单边基线，**不是** head-to-head 结论（那是 P4）。(a) 与 (c) 的差即"未发布 desc 矩阵"带来的不确定带宽。

---

## 4. 对下一步的影响

1. **P3（Stage-2 训练）** 产出 `z_m/z_d` 后，用 `global_metrics.from_embeddings` 出我方数字，与本表 ModelLens 基线在**同一 harness、同一 1,686 查询、同一候选宇宙**上对比 —— 这就是 P4 head-to-head 的 A 轴。
2. **D-7（候选宇宙 30K vs 47K）** 在此已可两算：适配器对全 47,242 打分，metric 端可限制到 30,183 子集统计。P4 主表用 30K、C 轴用 47K 的方案不受阻。
3. **B 轴（延迟）** 复用适配器的 `score_matrix` 全扫路径（即 §4.1 的 O(N) 计时点）—— P4b 直接量。
4. **harness 已冻结**：P4 不需再动指标代码，两系统只是不同的 ranking 输入。

---

## 5. 可复现命令

```powershell
Set-Location D:\research\model_lake\codes\ModelLakeFishing
.\.venv\Scripts\python.exe -m scale.verify_corpus --only ckpt          # 入口守卫
.\.venv\Scripts\python.exe -m scale.global_metrics --selftest          # harness 对账
.\.venv\Scripts\python.exe -m scale.modellens_adapter --wiring-check --limit 60
.\.venv\Scripts\python.exe -m scale.modellens_adapter --eval --desc c  # 下界
.\.venv\Scripts\python.exe -m scale.modellens_adapter --eval --desc a  # 主基线
```
