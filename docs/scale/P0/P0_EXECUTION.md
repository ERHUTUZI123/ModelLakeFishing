# P0 执行报告：语料固化、计数对账与竞品基线接线自检

**执行日期：** 2026-07-22
**对应规划：** `docs/scale/MODELLENS_HEADTOHEAD_SCALE_PLAN.md` §1.1 / §1.4 / §2.1、§5 P0 行
**状态：** ✅ **P0 出闸通过**（`verify_corpus` 22/22 全绿；竞品 checkpoint 词表与语料**逐项精确对齐**）；决策 **D-1 / D-5 / D-6 均已裁定**
**产物目录：** `docs/scale/P0/artifacts/`（含 `audit_v2.json`、`audit_blockers_v2.json`、三份原始日志）

---

## 0. 一句话结论

语料已按 §1.4 固化（锁 SHA、只读、sha256、双副本、校验通过），**真实规模为 47,242 模型 / 10,479 数据集 / 1,807,133 行**；竞品 checkpoint 的全部词表维度与我们独立数出的计数**逐项精确相等**，证明拿到的就是它训练用的那一份。同时挖出 **5 个会直接改写 P1 设计的硬约束**，其中 1 个是**它们未发布的必需文件**，1 个是**它们的「数据集」根本不是数据集**。

---

## 1. 做了什么（过程与代码）

新增 Python 包 `ModelLakeFishing/scale/`，四个脚本，全部可重跑：

| 脚本 | 职责 | 对应 PLAN 条款 |
|---|---|---|
| `scale/pull_corpus.py` | 按**锁定的 commit SHA** 拉取三个 repo，落 `raw/`，算 sha256 + 行锚，写 `PROVENANCE.json`，置只读 | §1.4 规则 1–3 |
| `scale/verify_corpus.py` | 重算校验和比对 `PROVENANCE.json`，**不匹配即硬失败**（exit 1） | §1.4 规则 3；P1–P4 入口守卫 |
| `scale/audit_corpus.py` | 解析 CSV 定真实计数 + 侧文件覆盖率 | §1.1 计数落档 |
| `scale/audit_intake_blockers.py` | **修正**首轮审计的错误结论，量化 P1 真正的拦路石 | §1.2 crosswalk 校准 |

### 1.1 锁定的 revision（**不可改成 `main`**）

```
v2   luisrui/ModelLens-corpus-v2  57a692ccdb20d84d1c803544f722b3727450c0e8   919.8 MB / 10 files
v1   luisrui/ModelLens-corpus-v1  df6cb242ed54c96ec09c8644916c07ac14bff4e7   880.2 MB /  9 files
ckpt luisrui/ModelLens            68fabcb36d03f96b620a5b5f0ae786ce6da3e74f   709.1 MB /  3 files
```

`pull_corpus.py` 内置守卫：若 Hub 返回的 `info.sha` 与写死的 revision 不符，**直接抛错拒绝继续** —— 上游改动必须被发现，不能被静默吞掉。

### 1.2 执行命令与结果

```powershell
Set-Location D:\research\model_lake\codes\ModelLakeFishing
.\.venv\Scripts\python.exe -m scale.pull_corpus            # 2,509 MB, 22 files, 全部 ro=True
.\.venv\Scripts\python.exe -m scale.verify_corpus          # VERIFY OK: 22 files intact
.\.venv\Scripts\python.exe -m scale.audit_corpus --corpus v2
.\.venv\Scripts\python.exe -m scale.audit_intake_blockers
```

### 1.3 保全落地（§1.4 规则 4–5）

**`[用户裁定 2026-07-22 · D-6]` 副本策略收敛为「只保留本地 D 盘」。**

| 副本 | 位置 | 状态 |
|---|---|---|
| 主（**唯一**） | `D:\research\model_lake\data\` | ✅ `verify` 22/22 |
| ~~副本 2 (C:)~~ | ~~`C:\Users\17982\ModelLakeArchive\`~~ | 🗑️ P0 期间曾建立并校验通过，**已按用户裁定删除**（54 files / 2,392.9 MB），删除后 D: 重新 `verify` 22/22 全绿 |
| ~~副本 3 离机~~ | — | 不做（G:/H: 为 Google Drive 挂载，G: 已满、H: 余 13.3 GB；未采用） |

目录布局按 §1.4 三层制：`raw/`（只读，含 `PROVENANCE.json`）/ `interim/` / `derived/`。语料**不在 git 工作树内**，走 `MLF_DATA_DIR` 约定（未设置时默认 `D:\research\model_lake\data`）。

> **仍在岗的保全机制**：锁 SHA、`raw/` 只读、sha256 + 行锚、`verify_corpus.py` 入口守卫 —— 这些防的是**静默损坏与误改**，不受副本策略影响。
> **已接受的残余风险（记录在案）**：单副本不防 D 盘物理故障、不防整机丢失。若 D 盘失效**且**上游 repo 同时被改动/下架，快照不可恢复、本阶段全部数字失去可复现性。**这是知情下的取舍，不是疏漏。**

---

## 2. 计数对账：42K？47K？—— 定案

**PLAN §1.1 当时拒绝预填数字，要求「以实际计数为准」。现在从字节数出来了：**

| 量 | 实测（v2） | 此前说法 | 判定 |
|---|---:|---|---|
| 行数 | **1,807,133** | README 称 1,807,133 | ✅ 完全一致 |
| 唯一模型 | **47,242** | 用户「42k」/ paper「~47K」 | **paper 对，42K 不对** |
| 唯一数据集名 | **10,479** | paper「~9.6K」 | **都不是 9.6K；「10k 数据集」的直觉是对的** |
| 唯一 task | **2,581** | — | 词表极大（非小 task 集） |
| 唯一 metric | **3,714** | — | 词表极大 |

**交叉验证（决定性）**：竞品 `args.json` 写着 `num_models: 47242`、`num_tasks: 2581`、`num_metrics: 3714`、`num_families: 332`，与我们**独立从 CSV 数出来的值逐项精确相等**。这比任何自报数字都硬 —— 证明我们下载的 v2 就是训 checkpoint 的那一份。

---

## 3. 基线接线自检（§2.1）

**做法调整**：完整跑通它们的 `src/main.py` 训练/评测需要补齐缺失依赖（见 §4.1），因此 P0 采用**更强也更廉价**的自检 —— 直接解剖 checkpoint，把每张 embedding 表的行数与语料词表对账。**维度对不上就是拿错了语料；全对上就是同一代**。

`ModelLens.pt` 共 21 个张量 / **177.26 M 参数**（fp32 ≈ 709 MB，与文件大小自洽）：

| 张量 | 形状 | 与语料对账 |
|---|---|---|
| `model_desc_matrix` | (47242, 1536) | ✅ = 模型数 |
| `_id_emb.weight` | (**47243**, 1536) | ✅ = 模型数 **+1 个 [UNK]** |
| `task_embedding.weight` | (2581, 256) | ✅ = task 数 |
| `metric_embedding.weight` | (3714, 256) | ✅ = metric 数 |
| `family_embedding.weight` | (332, 64) | ✅ = `family2id` |
| `size_embedding.weight` | (23, 64) | ✅ = `num_size_buckets` |
| `dataset_id_embedding.weight` | (**85939**, 256) | ⚠️ **不是 10,479** —— 见 §4.2 |
| `_name_encoder.tok_emb.weight` | (10000, 512) | 名字 tokenizer 词表 1 万 |
| `backbone.0/3` | (512, **6016**) → (512, 512) | 主干极小，见 §4.4 |

**关键结构确认（直接支撑我们的论点）：**

- **`_id_emb` 存在 ⇒ 它们是「带 ID 的半直推式」**，新模型只能落到那一行 `[UNK]`（`id_dropout_rate: 0.1` 就是为此训练的）。**这从权重层面坐实了 PLAN §4.2 的 C 轴前提** —— 我们的归纳式没有这张表。
- **打分主干 `6016→512→512→1`**，仅约 3.3 M 参数；177 M 里 145 M 是两张 1536 维大表（`model_desc_matrix` + `_id_emb`）。**含义见 §4.4：它们的 O(N) 全扫单价很低，B 轴的交叉点 $N^\*$ 可能比预期靠后 —— PLAN §4.1.1 里那句「小 N 时它们可能反赢」的谨慎是对的。**

---

## 4. 五个改写 P1 的发现

### 4.1 🔴 **必需文件未发布：数据集描述嵌入（1536 维）**

主干输入 6016 维，可归账的槽位是：

```
model_desc 1536 + id_emb 1536 + name 512 + size 64 + family 64
          + dataset_id 256 + task 256 + metric 256  = 4480
6016 − 4480 = 1536  ← 悬空
```

这 1536 维就是 `args.json` 里的 `dataset_desp_emb_dim: 1536`，即**冻结的数据集描述嵌入矩阵**。它**既不在语料 repo，也不在 checkpoint 里**。

- 好消息：`model_desp_emb_path` 指向的 `model2desp_embeddings.npz` **同样没发布，但它已被烘进 checkpoint 的 `model_desc_matrix`**，所以模型侧不缺。
- 坏消息：**数据集侧缺**，且编码器身份未知（1536 维强烈指向 OpenAI `text-embedding-3-*` 一类）。

**对下一步的影响**：要让它们的 checkpoint 出分，必须**自行重建这张矩阵**，而编码器不同会引入我们无法归因的偏差。**这是 P4 基线可信度的头号风险**，必须在 P2 之前定策（见 §5 决策 D-5）。

### 4.2 🔴 **它们的「dataset」不是 dataset，是 (dataset, task, metric) 三元组**

`dataset_id_embedding` 有 85,939 行（= N+2）。实测：

| 粒度 | 唯一数 | vs 85,937 |
|---|---:|---|
| dataset 名 | 10,479 | ✗ |
| (dataset, task) | 15,927 | ✗ |
| **(dataset, task, metric)** | **86,196** | ✅ 最接近（差 259，应为过滤/未知 metric 所致） |

**影响极大，且对我们有利**：它们的 `new_dataset_evaluation` 冷启动划分是按**这个三元组**留出的 —— 也就是说，**留出 `(squad, QA, f1)` 时，`(squad, QA, exact_match)` 仍可能留在训练集里**。按数据集名（更别说按 dataset root）衡量，**它们的「新数据集」划分存在同源泄漏**。

这正面回答了 PLAN §2.2 悬而未决的问题（「证明 root-aware 不比它们宽松」）：**不仅不宽松，我们严格得多**。§2.2 的双口径策略必须保留，并在报告中明确写出这一粒度差异。

### 4.3 🟠 **值域完全未归一，且同名指标混用 0–1 与 0–100**

`value` 实测范围 **[−313.25, 1,000,000]**，65% 的值 > 1；分位数 p25=0.65 / p50=25.7 / p75=59.9 / p99=99.6。

更棘手的是**同一个指标名同时以两种量纲出现**：

```
mixed-scale: accuracy, accuracy_norm, auc, bleu, exact_match,
             f1, macro_f1, precision, recall, roc_auc
```

且 `accuracy` 的最大值达 **2317**（越界脏数据）。

**影响**：D0 现有的「percent-scale 自动探测 → /100」是**全局规则**，在这里会**灾难性失效**。P1 必须改为
**逐 (metric, dataset) 分组**判定量纲 + 越界裁剪/丢弃，并把丢弃率落档。

### 4.4 🟠 **可用监督远比首轮估计的多：53.8%，不是 10.6%**

首轮用 D0 的 `BOUNDED` 白名单只匹配到 10.6% 的行。**该白名单不认 `@k` 后缀，也不认 `accuracy_norm`** —— 而本语料 **48% 是 Retrieval 任务**，其 `ndcg@10 / recall@100 / map@10 / mrr@10` 全都是有界指标。

改用 **@k 感知规则**（剥掉尾部 `@\d+` 再匹配基名）后：**972,215 行可用 = 53.8%**。基名 top：`recall 132,674 / precision 130,665 / accuracy_norm 123,747 / ndcg 119,759 / map 118,868 / mrr 117,172 / accuracy 88,055 / f1 49,748 / exact_match 47,790`。

**影响**：P1 的监督边规模从「~19 万」改写为 **~97 万**，这是**数量级层面的规划修正**；`BOUNDED` 必须升级为 @k 感知版本。

### 4.5 🟡 侧文件覆盖率（**首轮三个数字都是错的，已修正**）

| 项 | 首轮（错） | 修正后 | 错因 |
|---|---|---|---|
| `model_profile` 有真实 size | 47,242（100%） | **21,460（45.4%）** | 缺失值是**字符串 `"unknown"`**，朴素真值测试放行了它 |
| `model_popularity` 覆盖 | 5 条 / 覆盖 0 | **47,242 条**（其中 `ok` 仅 **22,649**） | 该 JSON 是**包装结构**，载荷在 `["models"]` 下 |
| `dataset_desp` 覆盖 | 1,298（12.4%） | **5,428（51.8%）** | 首轮只查每个数据集**首次出现**的那一行；首行为空即被永久判定为无描述 |

补充：`family` 真实覆盖 **42,312（89.6%）** —— 对血缘边 `r_mm'` 是好消息；size 值是 `'7.0' / '0.028' / 7.0 / '22'` 的 **str/float 混型**，需解析。
`model_popularity` 状态分布：`ok 22,649 / not_found 23,341 / invalid 137 / non_hf_name 1,115` —— **近半模型在 HF 上已查无此名**，popularity 特征只能作弱信号。

**描述覆盖随候选深度显著上升（决定性的好消息）：**

| 候选深度 | 数据集数 | 有描述 | 占比 |
|---|---:|---:|---:|
| 全部 | 10,479 | 5,428 | 51.8% |
| ≥2 | 7,157 | 3,864 | 54.0% |
| ≥10 | 2,543 | 1,345 | 52.9% |
| ≥30 | 1,224 | 771 | 63.0% |
| **≥100** | **564** | **492** | **87.2%** |

**影响**：能当 gold@10 查询的数据集（深度 ≥10）共 **2,543** 个，其中 1,345 个有描述；深度 ≥30 有 1,224 个。**越深的数据集描述越全**，所以 12.4% 那个吓人的数字是假警报，`e_card` 通道在真正用得上的查询集上是可用的。

---

## 5. 对下一步的影响（净结论）

**放行**：P1 可以开工 —— 语料已固化可验证，规模已定案，词表已与 checkpoint 对齐。

**必须先改的**（写回 PLAN）：

1. `BOUNDED` 升级为 **@k 感知**；监督规模预期从 19 万改为 **~97 万**。
2. 值归一化改为**逐 (metric, dataset) 分组**量纲判定 + 越界裁剪，**不得**沿用全局 /100。
3. `model_profile` 解析必须把 **`"unknown"` 当缺失**；size 需 str/float 混型解析。
4. `model_popularity` 按 **`["models"]`** 读，且只有 `status=="ok"` 的 22,649 条可信。
5. 查询集口径定为**候选深度 ≥10 的 2,543 个数据集**（gold@10 的必要条件），敏感性分析用 ≥30 的 1,224 个。
6. §2.2 双划分口径**必须保留**，并写明它们的留出单元是 (dataset, task, metric)、按数据集名存在同源泄漏。

**`[已裁定]` 两项决策（2026-07-22）：**

- **D-5 ✅ = (a) + (c) 并报，且必须留痕**（用户裁定「做 a+c 但要把这个事情记录下来」）
  - **(a) 主基线**：用与我们 `e_card` **相同的编码器**重建那张 1536 维矩阵 —— 同源、可控、零成本；
  - **(c) 敏感性下界**：该槽位**置零**再跑一次，量化「基线被削弱多少」；
  - **(b) 商用编码器复刻不采用。**

  **留痕是硬要求**，凡出现 ModelLens 基线数字的对外产物（`HEADTOHEAD_RESULTS.md`、`SCALABILITY_REPORT.md`、及所有相关表格脚注）都必须写明固定三句：
  > ① 该矩阵由原作者**未随语料或 checkpoint 发布**；② 我们的基线使用**自建替代嵌入**（编码器名 + 维度 + 生成脚本），**非原件**；③ 同时给出置零下界，基线真值落在 (a)–(c) 之间。

  这是整个 head-to-head 中**唯一一处无法完全复现对手输入**的环节 —— 主动反复披露远胜于被审稿人发现；隐瞒会让 A 轴全部结论失去可信度。

- **D-6 ✅ = 只保留本地 D 盘**（用户裁定「只保留在本地 D 盘，C 盘都不要」）。C 盘副本已删除并复验；残余风险已记录并接受（见 §1.3）。

---

## 6. 附：可复现命令

```powershell
Set-Location D:\research\model_lake\codes\ModelLakeFishing

# 重新拉取（会因已存在而跳过；--force 强制重下）
.\.venv\Scripts\python.exe -m scale.pull_corpus

# 进入任何后续 Phase 之前的入口守卫（P1-P4 必跑）
.\.venv\Scripts\python.exe -m scale.verify_corpus
.\.venv\Scripts\python.exe -m scale.verify_corpus --root 'C:\Users\17982\ModelLakeArchive\modellens_snapshot'

# 计数与拦路石审计
.\.venv\Scripts\python.exe -m scale.audit_corpus --corpus v2
.\.venv\Scripts\python.exe -m scale.audit_intake_blockers
```

**校验和速查**（完整清单见各 `PROVENANCE.json`）：

```
v2/data.csv            sha256 0db6d8c73b777338...  898,847,426 B
v1/data_clean.csv      sha256 b73f2f6daa22bf53...  864,902,042 B
ckpt/ModelLens.pt      sha256 (见 modellens_ckpt/PROVENANCE.json)  709 MB
```
