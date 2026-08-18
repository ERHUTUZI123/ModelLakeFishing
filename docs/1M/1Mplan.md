# 全量档（RF）执行计划：把候选宇宙从 100K 拉到 HuggingFace 全站

上级规划：[`100k/plan.md`](100k/plan.md)（五档梯子 R0–R4、双层湖设计、S0–S5 阶段划分）。
已验证的执行体例：[`100k/100kplan.md`](100k/100kplan.md)（R2 的完整 runbook 与出闸门）。

本计划把 R2 验证过的流水线原样放大到 HF 全站模型，作为梯子的最后一档。
原规划里的 R3（500K）与 R4（1M）不再作为独立的爬取与选样档，改为本档内部的一个训练中间档与若干索引子采样点，理由见 §3.2。

---

## §1 目标与范围

要交付的是一个候选宇宙等于 HuggingFace 全部可枚举公开模型的检索系统，以及它在这个规模上的三轴数字。

具体地：

1. 枚举 HF 上全部可访问的公开模型元数据，形成一份带时间戳、可校验、只读的快照。
2. 用 R2 已验证的流水线（canonical 化、梯子、特征、建图、训练、导出、索引、评测）在全量湖上跑通一遍。
3. 给出精度、延迟、冷启动三轴在全量规模下的实测值，以及跨 N 的 scaling 曲线。

不在本档范围内的，与 R2 保持一致：ModelLens 的对照实测仍为外推（D-16）；HALO 标签仍不进监督（D-10）；
C 轴发现的塌缩问题（G-E1）的修复属于新实验，不在本档做。
另外本档也不做增量式在线更新系统，只做一次性全量快照。

### 1.1 N 的取值不预设

N 由枚举实测决定。F1 已经跑完：2026-08-18 的枚举得到 3,003,759 条可枚举的公开模型，
**N_full = 3,020,418**（加上今天已不在 HF 上、但特征来自冻结图的 16,659 个 CORE 模型）。
容量测算仍按「每百万模型」给出（§9），便于 R3/R4 之后的快照重新折算。

`N_full = |CORE ∪ 全部可枚举的公开 HF 模型|`。CORE 的 30,183 个模型无论是否还能在 HF 上访问都计入，
因为它们的特征来自冻结图而不是爬取（约束 1）。

---

## §2 与 R2 的关系

继承而不改动的部分：

- CORE 监督核（30,183 模型 / 9,603 数据集节点 / 312,986 `trained_on` 边 / 517 个 held-out 查询 / gold 标签）逐字节冻结。
- 训练配置 `l1l3b_config()` 零改动，编码器仍是 `all-MiniLM-L6-v2`，`e_name` 仍是 seed 42 / 64 维。
- T0 的七项规模化改造已在 R2 上跑通，其中扇出上限与对比损失采样已由 [`100k/T7.5.md`](100k/T7.5.md) 用三档 × 三臂 × 三 seed 验证过误差走向：
  前者随 N 增值，后者随 N 收窄，都不构成放大到全量的障碍。
- 评测 harness、泄漏门、行序门、iso-recall 协议全部沿用。

必须改动的部分：

| 项 | R2 的做法 | 全量档的做法 | 原因 |
|---|---|---|---|
| 候选来源 | 532 个查询发现 537,025 条候选 | 单流穷举枚举全站 | 全量的定义就是不做发现式抽样 |
| HALO 选样 | 配额均衡选出 69,817（G1–G20） | 不选样，全收 | 选样是抽样偏差的补救；全量没有抽样，补救对象消失 |
| 排序键 | `sort=downloads` | `sort=createdAt&direction=-1` | `downloads` 每天变化，游标建在可变字段上会在长时间枚举中跳过记录（F-T2-9）；`createdAt` 的游标实测只用不可变的 `_id`（F0 §2.2） |
| 特征存储 | fp32，单个 `.pt` 里的张量 | 仍然 fp32，改为独立 `.npy` 并在加载时内存映射 | 精度不动（D-55）。每百万 fp32 是 1.79 GB，省内存靠映射而不是靠降精度 |
| 图文件 | 单个 `.pt`，约 250 MB | 拆件加 memmap | 单文件预估到数 GB，`torch.save`/`load` 的一次性内存峰值不可接受 |

前两项是本计划与 R2 之间最重要的差别，它改变了实验测量的对象，见 §3.3。

---

## §3 实验设计

### 3.1 严格嵌套与行序

行序设计沿用 R2 并向外扩展一层：

```
mappedID  0 .. 30,182       CORE，来自冻结图，顺序不变
mappedID  30,183 .. 99,999  R2 的 69,817 个 HALO，顺序不变
mappedID  100,000 .. N-1    其余全量模型，按 (createdAt desc, id) 确定性排序
```

这样 `L_100K` 是 `L_full` 的精确前缀，前 100,000 行就是 R2 的湖。带来三个直接好处：

- `x` 矩阵的前 100,000 行可以从 R2 的 `feats/100k/x_m.npy` 原样复制，不重新嵌入，省掉一次计算，也让前缀与 R2 逐字节相同。
  这一点成立的前提是描述符只依赖 id、family 字符串与 `size_b`，而 family 字符串由四级解析（D-36）决定、不依赖湖的规模。
- gold 标签、划分文件、517 个查询完全复用，零改动。
- 任何在前 100,000 行上重跑的评测都应当复现 R2 的数字，这本身是一条免费的正确性检查（见 §7 的 G-F3）。

需要注意 `family_id` 不在这条复制规则里。`family_id` 是 `family_vocab` 的行号，
而 vocab 由 `FAMILY_MIN_COUNT` 在当前湖上的计数决定：某个在 R2 里不够门槛、落到 `Other` 的家族，
在全量湖里可能越过门槛而获得自己的行。因此 `family_id` 对全部 N 行重算，
并记录前 100,000 行里有多少行的 `family_id` 发生了变化。`size_bucket_id` 走固定常数，不受影响。

R2 的 69,817 个 HALO 里可能有一部分在枚举当天已被删除、转私有或 gated。这些模型仍留在梯子里、特征沿用 R2 的副本，
但要在 `LADDER_REPORT.json` 里单独计数并在报告中说明，因为它们是「快照 A 里存在、快照 B 里不存在」的一批。

### 3.2 两条 scaling 曲线，成本差两个数量级

R0 到 R2 的梯子把两件事绑在了一起：候选宇宙变大，训练时看到的图也同时变大。
全量档把它们拆开，因为二者的测量成本相差很大，而 `plan.md §1.1` 对 N 的定义本来就是「HNSW 索引里可被检索到的模型总数」。

检索侧曲线，成本低。模型是归纳式的，`z_m` 只算一次；把索引建在 `z_m` 的随机子集上就能得到任意 N 处的
A 轴与 B 轴数字。子集必须包含全部 CORE 行（gold 候选都在 CORE，抽掉它们指标就没有意义），只在 HALO 里均匀抽样，
每个 N 用三个抽样 seed。计划取 N ∈ {100K, 250K, 500K, 1M, N_full}，共 5 个点。
它回答的问题是：在同一个训练好的表示下，候选池变大会让 gold 掉多少。

训练侧曲线，成本高。每个点要重新建图并训练三个 seed。计划只取两个新点：
一个 500K 的中间档和全量档本身，加上已有的 12K / 30K / 100K，训练侧共 5 个点。
它回答的问题是：在更大的图上训练出来的表示本身是否更好或更差。

500K 中间档同时承担两个角色：它是全量档之前的可行性探针（图能不能建、训练会不会 OOM、墙钟是多少），
也是训练侧曲线的中间点。它的湖定义为 `CORE ∪ R2 的 69,817 ∪ 从其余全量模型里按 seed 均匀抽样补到 500,000`，
这样既保持对 100K 前缀的嵌套，新增部分又与全量湖同分布。

如果时间或配额只够一次全量训练，先砍掉 500K 中间档的三个 seed 只留一个，再砍掉中间档本身；
检索侧曲线不要砍，它是这一档成本最低、信息量最高的一条。

### 3.3 分布口径变了，这是本计划最重要的一条限制

R2 的 HALO 是配额均衡后的 69,817 条：最大发布者 0.902%，top-10 发布者 4.045%，
有效发布者数 2,889.1，第三方量化搬运 14.999%。全量湖没有这些上限，它就是 HF 的真实构成。

本节原先按 v1 的 15 万条下载量前缀预期真实构成会是「最大发布者 24.21%、量化搬运 61.98%」。
F1 实测推翻了这个预期：**全量湖的最大发布者占 2.276%，top-10 占 5.307%，有效发布者数 1,268.4，
带量化标签的占 9.75%**。v1 那两个数是按下载量排序的产物，不是总体的性质——
自动化流水线反复拉取的搬运仓库被系统性抬到下载榜头部，把全站枚举一遍它们只占个位数百分比。

因此 R2 到全量档之间，分布确实变了，但不是数量级的变化：最大发布者 0.90% → 2.28%，
有效发布者 2,889 → 1,268。变化的仍然不只是 N，两档之间的差值仍然不能干净地归因于 N。
处理办法有三条，都要做：

1. 用 `report_distribution.py` 在全量湖上产出与 R2 同口径的分布表（HHI、熵、有效发布者数、top-k 份额），
   让读者看到两个湖差在哪里，而不是只看到一个下降的数字。
2. §3.2 的检索侧曲线在同一个湖内部变 N，分布固定，所以它给出的是干净的 N 效应。跨档比较用它，不用 R2 与全量档的直接相减。
3. 报告里对这两个数字分别命名：R2 测的是「均衡干扰项下的检索」，全量档测的是「真实 HF 湖上的检索」，
   不把后者叙述成前者的自然延伸。

### 3.4 受控变量表

| | 固定 | 变化 |
|---|---|---|
| 监督边 | CORE 的 312,986 条 `trained_on` | — |
| 查询集 | 517 个 root-aware held-out 数据集 | — |
| gold 标签 | 同一批 gold 模型 | — |
| 模型架构与超参 | L1L3b，零改动 | — |
| 编码器 | `all-MiniLM-L6-v2` | — |
| 评测 harness | `scale/global_metrics.py` | — |
| 候选宇宙 N | — | 12K → 30K → 100K → 500K → N_full |
| HALO 分布 | — | 均衡（R2）→ 真实（本档），见 §3.3 |

最后一行是 R2 的表里没有的。它必须写在这里，否则读者会默认只有 N 在变。

---

## §4 三条实验有效性约束在全量档的形态

沿用 R2 的编号，在执行记录里它们仍被称为约束 1 到 3（早期文档里叫铁律 1 到 3）。

约束 1，CORE 逐字节冻结。全量图的 `x[:30183]`、`size_bucket_id[:30183]`、`family_id[:30183]`、
`trained_on` 边、数据集侧全部张量都从 `hgraph_ml_v2.pt` 原样复制，不重算也不重新嵌入。
建图末尾的六条断言与 R2 相同。

约束 2，HALO 使用与 CORE 相同的描述符函数。仍然只从 HF 元数据里取 `family` 与 `size_b` 两个量，
喂进 `scale/modellens_build_graph.py` 的 `model_descriptor`，HF 的 tags、pipeline_tag、library_name 除用于推断 family 外不使用。
`size_b` 仍然只取 `safetensors.total`，不加名字正则兜底（D-26）。

这条约束在 R2 上的检验结果是已知的：绝对可分性 AUC 0.9729，超出原定的 [0.5, 0.75]，
按 D-39 改判为「绝对可分性 + 流水线增量 + 只用名字的基线」三个数一起报。全量档沿用这个口径，
并且要预期绝对 AUC 进一步升高，因为全量湖里量化搬运仓库的比例远高于 R2 的均衡湖，命名与元数据模式与 CORE 差得更远。
这不是流水线的问题，是两个总体本来就不同；能调的仍然只有描述符贡献的那部分增量。

约束 3，评测时候选池必须等于 N_full。`--expect-n` 断言与 `MANIFEST.json` 回写照旧。
检索侧曲线的每个子采样点也各自断言自己的 N，并把 CORE 是否全在索引里作为一条独立断言。

---

## §5 执行计划

任务编号沿用 R2 的 T 系列语义，前缀改为 F 以区分档位。依赖顺序：

```
F0 前置探针与两项改造（本地）
      │
      ├─ F1 全量枚举 ──> F2 canonical ──> F3 梯子 ──> F4 特征 ──> F5 建图 ──┐
      │                                                                      ▼
远端环境（沿用 R2 的 watGPU 配置）──────────────────────────> F6 训练 ──> F7 导出与索引 ──> F8 评测
```

F0 与 F1 都不依赖远端，可以先开。F1 是唯一一个墙钟以小时计的数据阶段。

### F0 前置探针与代码改造（已完成，2026-08-18，见 [`F0.md`](F0.md)）

三件事，都在本地做，做完再开 F1。实测结论：枚举可行性成立，`createdAt` 的游标只用不可变的 `_id`，
连续 200 页零重复零乱序；爬虫已支持 `--sort`/`--direction`；图拆件与内存映射在真实 100K 图上逐张量等价，
加载峰值从 243 MB 降到 50 MB。全站规模估计 2.89×10⁶（±15%）。

**枚举可行性探针。** 这是全量档唯一一个「不试就不知道」的前置问题：HF 的 keyset 分页能不能一直翻到几百万条。
R2 的爬取最深只翻了 150 页，deep 计划里每个查询上限 12 页，所以三千页量级的连续枚举没有被验证过。
探针的做法是按 `sort=createdAt&direction=-1` 连续翻 200 页，检查游标是否一直前进、返回条数是否稳定在 1000、
id 是否零重复、`Link: rel="next"` 是否在深处仍然出现。实测 21.7 秒跑完，产物在 [`F0.md`](F0.md) 与 [`F0_runs/`](F0_runs/)。
若在某个深度被截断，退路是按 `createdAt` 分区间枚举，即把时间轴切成若干段，每段单独翻页，段间用 `createdAt` 区间过滤拼接。

**爬虫加排序参数。** `scale1m/hf_crawl.py` 的 `crawl_single` 目前把 `sort=downloads&direction=-1` 写死
（[hf_crawl.py:398](../../scale1m/hf_crawl.py#L398)），需要加 `--sort` 与 `--direction`，默认值保持不变以免影响 R2 的复现。
终止条件已经是「`--limit` 到了或游标耗尽」，把 `--limit` 设得足够大即可靠游标自然结束。
去重用的 `seen` 集合在几百万条时占几百 MB 内存，续爬时 `replay_seen_ids` 还会重读全部分片；
先按现状跑，把实测内存记进 `F1.md`，超出再换成分片内有序去重。

**补做 `plan.md §4` 的第 10 项。** 该项在 100K 档按 D-21 推迟，现在到期。
第 9 项（fp16 特征）取消，特征保持 float32（D-55）。

第 10 项，图拆件与内存映射。全量图的单文件预估在数 GB，`torch.save` 与 `torch.load` 会有一次性的内存峰值。
改为 `x_model.npy`、`x_dataset.npy`、`nodes.npz`、`edges.npz`、`meta.json`，加载时组装并映射两个特征矩阵（D-13）。
验收方式是在 100K 图上拆件后重新加载，与原单文件版本逐张量相等，并且 `check_stage2_contract` 仍然通过。
实现是 [`scale1m/graph_store.py`](../../scale1m/graph_store.py)，两项验收均已通过。

出闸门：探针结论已写入 [`F0.md`](F0.md)；`--sort` 参数有单测覆盖；拆件在 100K 图上通过等价性验收。

### F1 全量枚举（已完成，2026-08-18，见 [`F1.md`](F1.md)）

实测 3,003,759 条、3,004 页、27.5 分钟、0 重复、0 乱序、游标耗尽正常终止，两条出闸门均通过。
快照窗口 2026-08-18T18:01:49Z 到 18:29:18Z，上界是最新记录的 `createdAt` 2026-08-18T18:01:36Z。


```bash
C=$MLF_DATA_DIR/data1m/candidates_full
python -m scale1m.hf_crawl --sort createdAt --direction -1 \
       --limit 100000000 --v2-fields --shard-size 50000 --out $C
```

字段集沿用 v2 的 `expand[]`，不能用 `full=true&cardData=true`：后者不返回 `safetensors`，
会让 `size_b` 全部缺失而不报错，同时返回整个 repo 文件列表使响应膨胀。
六个 v2 字段（`author`、`baseModels`、`config`、`lastModified`、`gguf`、`gated/disabled/private`）一个都不能少，
其中 `baseModels` 带 `relation`，是 `r_mm'` 离散有序权重的唯一结构化来源。

爬取过程中要记录并在结束后写进 `PROVENANCE.json`：起止 UTC 时间戳、总页数、总条数、
每片 sha256、限流触发次数与 `retry_reasons`、是否带 token、排序键。分片与 `PROVENANCE.json` 设为只读。

枚举窗口是数小时量级，期间会有模型被创建和删除。按 `createdAt` 降序枚举时，新建的模型出现在已经翻过的头部，
因此不会被计入，也不会导致漏掉更早的模型；这正是选这个排序键的原因。快照口径记为一个时间区间而不是一个时刻。

出闸门：`PROVENANCE.json` 完整；每片 sha256 复算一致；分片只读；规范化 id 去重后条数不变；
枚举以游标耗尽而不是以 `--limit` 结束；把实测总条数记为 `N_crawled` 并与 HF 站点公开的模型总数对照，
差异超过 5% 时先查枚举完整性再往下走。

`select_balanced_halo.py` 与 G1–G20 在本档不运行。`report_distribution.py` 仍然运行，
但产出的是描述性分布表而不是闸门。

### F2 canonical 化

```bash
python -m scale1m.hf_canonicalize --candidates $C --core .../hgraph_ml_v2.pt
```

规则与 R2 完全一致，不做任何修改：`unique_model_id` 用同一个 `normalize`；
`size_b` 只取 `safetensors.total`；`family` 走四级解析并记 `family_source`（D-36）；
`lineage_base` 以 `baseModels` 优先、`cardData.base_model` 兜底并保留 `relation`（D-37）。
`layer` 分层继续只进审计不进训练，其中 `dropped` 判据已知恒为 0（D-27），保留不动但在报告里写明。

这一步在全量上的成本是 CPU 时间与内存。R2 在 537K 候选上约 3 分钟，按线性放大到数百万约 15–30 分钟，
内存峰值取决于是否一次性读入 parquet；若超出本机内存，改为按分片处理后合并。

要在 `CANON_REPORT.json` 里记录并与 R2 对照的量：`size_b` 缺失率、`family` 取值数、
落进 CORE 用过的 207 个家族的比例、`family_source` 的四级分布、`lineage_base` 的命中率与 `relation` 分布。
R2 的对应值分别是 42.25%、9,292、48.47%，以及 `baseModels` 在 537K 候选上命中 234,445 条。

### F3 梯子

```bash
python -m scale1m.build_ladder --rung full --n <N_full> \
  --core  stage1BuildTransferGraph/hgraph_ml_v2.pt \
  --prefix .../ladder/100k_model_ids.csv \
  --canon $C/canon/hf_canon.parquet \
  --out   .../ladder
```

`--prefix` 是本档新增的参数：把 R2 的 100,000 行梯子作为固定前缀接在 CORE 之后，其余模型按
`(createdAt desc, id)` 排在 100,000 之后。断言在 R2 八条的基础上增加三条：

```python
assert ladder.iloc[:100_000]["model"].tolist() == r2_ladder["model"].tolist()   # 前缀逐项相等
assert ladder.mappedID.tolist() == list(range(n_full))                          # 行序连续
assert ladder["model"].map(normalize).nunique() == n_full                       # 全表去重
```

产物是 `ladder/full_model_ids.csv` 与 `LADDER_REPORT.json`，后者要记 sha256、
R2 前缀里已从 HF 消失的模型数、以及 `layer` 分布。

同时按 §3.2 生成 500K 中间档的梯子（`--rung 500k --sample-seed 0`），它在 100K 前缀之后按 seed 均匀抽样补齐。

### F4 特征

分三段，前两段是复制，只有第三段要算：

```python
x_core   = core["data"]["model"].x                      # [30183, 448]，冻结图复制
x_r2halo = np.load("feats/100k/x_m.npy")[30183:100000]  # [69817, 448]，R2 产物复制
x_new    = embed(descriptors_of(ladder[100000:]))       # 其余全部，新算
```

新算部分用与 CORE 相同的 `model_descriptor`，`e_name` 用 seed 42、64 维，`e_desc` 用 `all-MiniLM-L6-v2`。
R2 实测在 RTX 6000 Ada 上 batch 256、fp32 时约 1,343 条每秒，按此每百万约 12 分钟。
分片写 `.npy`（每 100 万一片）并支持断点续，合并后按 float32 存盘（D-55），由 `graph_store` 在加载时内存映射。

`family_vocab` 仍然 append-only，CORE 已有行的 id 不移位，断言与 R2 相同。
R2 时 vocab 从 341 行扩到约 2,000 行，全量下预期扩到万行量级，这会让 family 嵌入表变大，
同时让很少被采样到的稀有行更难学到东西。按 CLAUDE.md Step 2 的要求，训练后检查稀有行的范数是否仍停在初始化附近，
把结果写进报告。

出闸门：`torch.equal(x[:30183], x_core)`；`torch.equal(x[30183:100000], r2_halo)`；
形状为 `(N_full, 448)` 且无 NaN、无 Inf；抽 100 个新行按 id 重算 `e_name` 与 `x[i,:64]` 相等；
G-B4 的可分性 AUC 与子句消融按 D-39 的三个数报出。

### F5 建图

```bash
python -m scale1m.build_graph_rung --rung full \
  --core ... --ladder ... --feats ... --canon $C/canon/hf_canon.parquet --shard-out ...
```

`--canon` 不能漏。R2 第一次建图时它没有传到远端，血缘 `relation_id` 全部落成 `unknown`，
D-37 拿回来的血缘声明没有进图，后来按 D-44 重建（[`100k/T6more.md`](100k/T6more.md)）。

三类边的构建方式与 R2 相同：`trained_on` 与 `similar_to` 从 CORE 原样复制，
`is_base_of` 用规范化 id 的 hashmap join，复杂度 O(N)。

血缘边是本档相对 R2 变化最大的一处，也是最值得看的一处。R2 的 100K 湖里，
HALO 声明 `base_model` 的有 27,672 条，能在湖内解析的只有 16,385 条，因为很多 base 模型不在湖里。
全量湖里几乎所有 base 都在湖内，解析率应当大幅上升，边数预期到 10⁵ 到 10⁶ 量级。
这是 `is_base_of` 这条设计第一次有规模化证据的机会，但同时带来两个具体风险：

- 图的内存与存储随边数线性增长，这正是 F0 第 10 项拆件要解决的。
- hub 度数急剧变大。R2 上已经观察到 `cold` 层比 `frozen` 层塌得更厉害（0.935 对 0.815，G-E1），
  血缘边把派生族拉成一团。全量下同一个 base 可能挂着上万个量化变体，这个效应只会更强。
  扇出上限是现有的对冲手段，且 T7.5 证明它随 N 增值；但要在 `lineage_stats.json` 里把度数分布报出来，
  不能只报边数。

`lineage_stats.json` 必须产出，字段与 R2 相同，另加度数分布的分位数与最大 hub 的规模。

出闸门：`verify_rung_graph --rung full` 全绿；`check_stage2_contract` 通过；
划分不变性检查，即 `make_root_aware_splits` 在全量图与 CORE 图上给出相同的 test 边 sha256
（R2 实测 `66c81823cb731063…`，65,968 边；HALO 无监督边，所以这个值应当保持不变）。

### F6 训练

配置零改动沿用 `l1l3b_config()`。执行顺序是先中间档后全量，每档先冒烟再全量：

```bash
RUNG=500k  SEED=0 EPOCHS=2  sbatch scripts/watgpu/train_rung.sh    # 冒烟，记显存与墙钟
RUNG=500k  SEED=0,1,2 EPOCHS=25 sbatch scripts/watgpu/train_rung.sh
RUNG=full  SEED=0 EPOCHS=2  sbatch scripts/watgpu/train_rung.sh    # 冒烟
RUNG=full  SEED=0,1,2 EPOCHS=25 sbatch scripts/watgpu/train_rung.sh
```

监督边数在所有档上都是 312,986，不随 N 变化。这是这套设计能放大到全量的根本原因：
训练成本的增量只来自邻域采样和全 N 的推理，不来自监督量。R2 实测 25 epoch 的墙钟是
12K 约 92 秒、30K 约 431 秒、100K 约 109 秒，三者之间没有随 N 的单调关系，说明这一段的成本主要由图结构而非 N 决定。
全量档的墙钟以冒烟实测为准，`--time` 按实测的若干倍申请，不留大额占位值（D-45）。

`--seed` 的语义仍是 `split_seed`，`init_seed` 恒为 0（D-43），因此三 seed 给出的是划分噪声区间。
报任何精度数字都要带这个区间，R2 实测三 seed 的标准差在 0.020 到 0.074 之间，大于多数被比较的差值。

checkpoint 与 resume 的要求与 R2 相同，包括两张 embedding 表、`family_vocab.csv` 的路径与 SHA-256、
size bucket 常数版本、`e_name` seed 与 `token_dim`、encoder 名。

出闸门：机制门（loss 下降、无 NaN、两张 embedding 表梯度非零、frozen `x` 梯度为 `None`）；
`MANIFEST.json` 含 `peak_gpu_mem_gb` 与 `wallclock_s`；`git_head` 非空；as-run 代码对账通过。

### F7 导出与索引

```bash
python -m scale1m.export_rung --rung full --run $OUT \
  --chunk 100000 --hnsw-M 32 --ef-construction 200 --hnsw-threads 8
```

分块推理按 `batch_size=100_000` 遍历模型节点，写入必须用 `z_m[batch['model'].n_id] = out`。
R2 上分块与全图前向的差是 0.00e+00，全量下无法再跑全图前向做对照，改为在 100K 前缀上做同样的比较。

索引建完落盘，并用 `tune_ef_for_recall(target=0.99)` 二分出最小 `ef_search`。
R2 三档的 ef 都停在下界附近（52/52/50），说明到 100K 还没有逼出 ANN 的代价；全量档是第一次可能看到 ef 上升的规模，
这一列的变化本身就是结果。

检索侧曲线的子采样索引在这一步一起建：对每个 N ∈ {100K, 250K, 500K, 1M, N_full} 与三个抽样 seed，
从 `z_m` 里取 CORE 全部行加上 HALO 的均匀抽样，各建一个索引并各自 iso-recall 调 ef。

出闸门：泄漏门，`gold10(z_eval) < gold10(z_full)`；行序门，抽 100 个 `mappedID` 反查 `model_ids.csv` 与 ladder 逐项相等；
候选池断言 `z_m.shape[0] == N_full`；每个子采样索引各自断言自己的 N 且 CORE 全在。

### F8 评测

三轴口径与 R2 完全一致，只是多了子采样曲线这一维。

A 轴报 gold@1、gold@10、top3@10、gold-gap@10、root_gold@10、median gold rank、median rank / N、vs random，
全部带三 seed 区间。核心列是 `median rank / N` 而不是 `gold@10`，理由见 §6 的预注册。

B 轴严格按 iso-recall 协议：先二分 ef 使 recall@50 ≥ 0.99 再计时，单线程逐查询，warmup 100 次丢弃后测 1000 次，
报 p50/p95/p99，所有档在同一次作业内跑完，另报多线程 QPS。ModelLens 那一列仍标注为外推而非实测（D-16）。

C 轴按 warm / cool / cold / frozen 四层报层内平均余弦、有效维度、与 warm 层的质心余弦，
并把 R2 已经发现的 G-E1 现象在全量上复测：`cold` 是否仍比 `frozen` 更塌，血缘的净分离是否继续被随机对余弦的上升吃掉。
R2 实测随机对余弦随 N 从 0.61 涨到 0.74，全量是这条趋势最有力的检验点。

D 轴报端到端建库墙钟、峰值 GPU 与 CPU 内存、索引磁盘、增量上线一个新模型的耗时、多线程 QPS。
最后一项在全量规模上才真正有说服力：新模型上线是特征加一次前向加一次 `add_items`，不重训也不重建索引。

`displacement_quality` 在 R2 上只给出了成分而没有给出质量，因为 `layer=labeled` 只记了有无、没有物化 model-index 的数值。
本档沿用这个口径，只报成分，不假装做过质量对比；若要补齐，需要把 model-index 的 (dataset, metric, value) 与 CORE 的数据集节点做名字对齐，
那是一次命中率未知的独立数据工程，不放在本计划里。

---

## §6 预注册预测

在跑之前写下，用于防止事后按数字挑叙述。按 F1 实测的 N_full = 3,020,418 折算，即 100K 档的 30.2 倍。

A 轴。R2 在 100K 上的 `median rank / N` 是 9.8×10⁻⁴，`gold@10` 是 0.2071。
若 rank 分布随 N 等比放大，则 `median rank / N` 保持在 1×10⁻³ 附近，中位 gold rank 会到数千，
`gold@10` 相应落到 0.05 以下。若表示对全量湖里大量离题模型有额外判别力，`median rank / N` 会下降。
预注册区间：`median rank / N ∈ [3×10⁻⁴, 1.2×10⁻³]`，`gold@10 ∈ [0.02, 0.12]`。
落在区间内按正常结果报告；`gold@10 > 0.20`（几乎没掉）先查三处再庆祝，即候选池是否真是 N_full、
描述符是否用对、泄漏门是否通过；低于 0.02 则说明在这个规模上 gold 已经基本进不了 top-10，这是真结果，
照实报，并把论证重心移到 `median rank / N` 与 vs random 上。

这里要提前说清楚一件事：在全量规模上 `gold@10` 很可能不再是一个有分辨力的指标，因为 517 个查询、
每个查询的 gold 只有几个，而候选池是数百万。这不是失败，是度量的分辨率问题。
主指标因此明确定为 `median rank / N` 与 vs random，`gold@10` 作为兼容历史的次要列保留。

B 轴。HNSW 的 p50 对 N 的幂律指数在 R2 三档实测为 0.032，按此外推到 3,020,418 约 0.0178 ms。
全扫 p50 按线性外推约 56.9 ms，加速比约 3,200 倍。ModelLens 的 Θ(N) 外推约 4.0 秒每查询。
若实测 HNSW p50 显著高于 0.03 ms 或 ef 被迫升到三位数，说明规模开始逼出 ANN 的代价，这本身是有价值的结果，
按实测报并给出 ef 与 recall 的权衡曲线。

C 轴。预期 `frozen` 层占比进一步上升（R2 实测 50.01%），`cold` 层因血缘解析率上升而大幅扩大——
F1 实测全量湖有 67.59% 的模型既无标注也无可解析血缘，而血缘边有 852,189 条，两者共同决定分层。
预期 `cold` 仍比 `frozen` 更塌，随机对余弦继续上升。若 `cold` 反而好于 `frozen`，
那与 R2 的结论相反，需要先排查血缘边的构建是否正确再下结论。

---

## §7 出闸门总表

| ID | 阶段 | 判据 | 不过怎么办 |
|---|---|---|---|
| G-F0 | F0 | 枚举探针连续 200 页游标不断、零重复、每页 1000 条 | 改按 `createdAt` 区间分段枚举 |
| G-F1a | F1 | 已过。`PROVENANCE.json` 完整，61/61 分片只读且 sha256 复算一致 | 停止，重爬 |
| G-F1b | F1 | 已过。`exhausted_cursor=true`；3,003,759 条去重后不变、0 重复、0 乱序；实测值比 F0 的估计 2.89M 高 3.9%，落在预注册的 ±15% 带内 | 先查枚举完整性再往下 |
| G-F2 | F2 | canonical 报告的六项统计已产出并与 R2 对照 | 补齐后再建梯子 |
| G-F3 | F3 | 梯子前 100,000 行与 R2 逐项相等；`mappedID` 连续；全表 id 去重等于 N_full | 停止，行序错位不会自己暴露 |
| G-F4a | F4 | `x[:30183]` 与冻结图相等，`x[30183:100000]` 与 R2 特征相等 | 停止，违反约束 1 或破坏嵌套 |
| G-F4b | F4 | 形状为 `(N_full, 448)`，无 NaN 与 Inf；`family_vocab` 只增不移 | 停止 |
| G-F4c | F4 | G-B4 按 D-39 报三个数：绝对 AUC、流水线增量、只用名字的基线 | 不回滚描述符，按通道归因并列为混杂因素 |
| G-F5a | F5 | `verify_rung_graph --rung full` 与 `check_stage2_contract` 全绿 | 停止 |
| G-F5b | F5 | 划分不变性：test 边 sha256 与 CORE 图相同（`66c81823cb731063…`） | 停止，说明 CORE 边序被改动 |
| G-F5c | F5 | `lineage_stats.json` 已产出，含度数分布与最大 hub 规模 | 补齐，无论结论好坏 |
| G-F6 | F6 | 机制门：loss 下降、无 NaN、两张 embedding 表梯度非零、frozen `x` 无梯度 | 回到 F0 |
| G-F7a | F7 | 候选池断言 `z_m.shape[0] == N_full`；每个子采样索引各自断言 N 且 CORE 全在 | 停止，违反约束 3 |
| G-F7b | F7 | 泄漏门 `gold10(z_eval) < gold10(z_full)` | 停止 |
| G-F7c | F7 | 行序门：抽 100 个 `mappedID` 与 ladder 逐项相等 | 停止 |
| G-F7d | F7 | 分块推理与 100K 前缀上的全图前向 `max\|Δ\| < 1e-5` | 查 `n_id` scatter |
| G-F8a | F8 | iso-recall：每个索引 recall@50 ≥ 0.99，延迟在该 ef 下测 | 调 ef 或 M；调不上去查过平滑 |
| G-F8b | F8 | 前缀复现：在 100K 子采样点上的 A 轴数字与 R2 的三 seed 区间重叠 | 不重叠说明流水线在放大过程中被改动，逐项回滚定位 |
| G-F8c | F8 | 预注册区间：`median rank / N ∈ [3×10⁻⁴, 1.2×10⁻³]` | 超界按 §6 的三条先查 bug |

G-F8b 是本计划里成本最低、价值最高的一条检查。它用同一套代码在同一份 `z_m` 的 100K 子集上重跑 R2 的测量，
如果结果落在 R2 的区间之外，说明问题出在放大过程本身，而不是规模效应。

---

## §8 风险与未决问题

**枚举可能翻不到底。** HF 的分页在几千页深度上的行为没有被验证过。若在某个深度被截断，
按 `createdAt` 分区间枚举可以绕开，代价是需要拼接与去重，且区间边界处要重叠一段以免漏。
这是 F0 探针要回答的第一个问题，也是唯一一个可能改变整个计划形态的问题。

**血缘边很多，F1 已经把量算出来了。** 按 F2 将使用的规则实测：声明率 29.77%，
湖内解析率 96.08%，按 D-28 剔除以 CORE 为子节点的 6,867 条后是 **852,189 条边**，是 R2 的 52 倍。
度数分布是长尾的：每个父节点的子节点数 p50 为 1、p99 为 118，但有 119 个父节点各带 1,000 个以上，
最大的 `black-forest-labs/flux.1-dev` 带 43,346 个。
后果有两个：图的内存与磁盘增长；hub 附近的过平滑加剧，C 轴可能比 R2 更差。
检测方式是 `lineage_stats.json` 的度数分布与 C 轴的层内余弦，并把最大的若干 hub 单独列出来。
缓解手段现有的只有扇出上限（T7.5 证明它随 N 增值）与血缘边的独立丢弃率；
若实测显示过平滑严重恶化，处理方式是照实报告并把它列为下一步实验，不在本档里改模型。

**分布口径变化会与 N 效应混淆。** 见 §3.3，靠三条措施处理，其中检索侧曲线是主要手段。

**度量分辨率下降。** 517 个查询在数百万候选下，`gold@10` 可能退化成一个几乎全是 0 的列。
预注册已经把主指标改到 `median rank / N`，但如果连它的三 seed 区间都宽到无法分辨相邻档，
唯一的解法是扩大查询集，而查询集来自 CORE、扩大它会动到监督核，与约束 1 冲突。
这种情况下正确的做法是承认这一档的精度轴只能给出量级而不能给出排序，把结论落在 B 轴与 D 轴上。

**R2 的锚点口径仍未定案。** G-C2 在 R2 上没有通过：12K 在新环境下是 0.3868，P3 存档是 0.4159，
T7.5 已排除两个近似为原因，差异归到环境与采样后端。本计划假定沿用 R2 批次的基线，
即以本批次内部的 12K/30K/100K 数字为锚点，历史值 0.4159 单独标注口径。
若你另有裁定，只影响报告怎么写，不影响本计划的执行。

**存储与配额。** 全量档的磁盘占用按 §9 估算在每百万 5 GB 量级，加上三个 seed 的运行目录。
远端 quota 未确认前不要开始大数据上传，这一条与 R2 的 T1 要求相同。

---

## §9 资源测算

所有数字按每百万模型给出，乘以实测 N 即可。基准是 R2 在 RTX 6000 Ada 48 GB 上的实测值。

| 项 | 每百万模型 | 依据 |
|---|---|---|
| 爬取请求数 | 1,000 页 | `limit=1000` 为每页上限 |
| 爬取墙钟 | 约 9 分钟 | F1 实测：3,004 页 / 27.5 分钟，其中 996 秒是限流休眠 |
| 爬取下载量 | 约 4.75 GB | F0 实测 4,753 B/条（`expand[]` 路径） |
| 爬取落盘 | 约 67 MB（gz） | F1 实测 67 B/条。真实湖元数据稀疏，R2 均衡湖是 165 B/条 |
| canonical 化 | 5–10 分钟（CPU） | R2 在 537K 上约 3 分钟 |
| MiniLM 嵌入 | 约 12 分钟 | R2 实测 1,343 条/秒，batch 256，fp32。N=3.02M 全档约 36 分钟 |
| `x_m` 存储 | 1.79 GB（float32，不降精度） | 448 维，D-55 |
| `z_m` 存储 | 0.51 GB | 128 维 fp32 |
| HNSW 索引 | 约 0.73 GB | R2 实测 73.2 MB / 100K，M=32 |
| 全扫 p50 | 约 18.9 ms | R2 实测 1.8856 ms / 100K，线性 |
| HNSW p50 | 与 N 几乎无关，约 0.016–0.018 ms | R2 实测幂律指数 0.032 |

作业申请建议：GPU 一卡，显存需求由分块推理决定而不是由 N 决定，R2 实测训练峰值 1.737 GB，
全量下主要压力在导出与索引；`--mem` 从 128G 起，按冒烟实测调整；`--cpus-per-task` 8 用于 HNSW 多线程；
`--time` 按冒烟实测的若干倍申请，不留大额占位值。

---

## §10 产物

```
$WORK/model_lake/data1m/
├── candidates_full/   hf_models_000{00..60}.jsonl.gz（3,003,759 条 / 202 MB / 只读）
│                      PROVENANCE.json（含 snapshot_window_utc）  CURSOR.json  SHARDS.json
│                      canon/hf_canon.parquet  CANON_REPORT.json
│                      DISTRIBUTION_REPORT.md（描述性，非闸门）
├── ladder/            full_model_ids.csv  500k_model_ids.csv  LADDER_REPORT.json
├── feats/full/        x_m.npy（float32 分片）  size_bucket_id.npy  family_id.npy
│                      family_vocab.csv  FEATS_REPORT.json
└── graphs/            hgraph_full/{x_model.npy, x_dataset.npy, nodes.npz, edges.npz,
                                    unique_*.parquet, meta.json}  lineage_stats.json
                       hgraph_500k/…

$WORK/model_lake/runs/RF_s{0,1,2}_<date>/
├── MANIFEST.json  ckpt/  stdout/  metadata/
├── exports/       z_m.npy  z_m_eval.npy  z_d_eval.npy
│                  hnsw_full.bin  hnsw_sub_{100k,250k,500k,1m}_s{0,1,2}.bin
│                  model_ids.csv  gold_cands.npz  ef_tuning.json
└── metrics/       a_axis.json  b_axis.json  c_axis.json  d_axis.json
                   scaling_curve_retrieval.json  scaling_curve_training.json

codes/ModelLakeFishing/docs/1M/
├── 1Mplan.md      本文件
├── 100k/          R2 的全部计划与执行记录
├── F0.md … F8.md  各阶段执行记录
├── F0_runs/       探针脚本与原始 JSON
├── F1_runs/       PROVENANCE 副本、校验日志、血缘与分布预测、爬取日志
└── RF_EXECUTION.md
```

`RF_EXECUTION.md` 沿用 P0–P5 与 R2 的体例，包含失败过程留痕。

---

## §11 决策登记（续 `100k/100kplan.md` D-45）

| ID | 决策 | 定案 | 理由 |
|---|---|---|---|
| D-46 | 全量档还做不做配额均衡选样 | 不做。`select_balanced_halo` 与 G1–G20 在本档不运行，`report_distribution` 降级为描述性报告 | 均衡选样是对发现式抽样偏差的补救。全量枚举没有抽样，补救对象不存在。代价是干扰项分布从均衡变为真实，按 §3.3 的三条措施披露 |
| D-47 | 枚举用哪个排序键 | `createdAt` 降序 | 游标是建在排序字段上的 keyset，`downloads` 每天变化会在长时间枚举中跳过记录（F-T2-9）。`createdAt` 不变，且降序枚举时新建模型落在已翻过的头部，不影响已有记录的完整性 |
| D-48 | 全量湖的行序 | CORE 0..30182，R2 的 69,817 个 HALO 接在其后，其余按 `(createdAt desc, id)` 排在 100,000 之后 | 使 `L_100K` 成为 `L_full` 的精确前缀，特征前 100,000 行可复制、gold 与划分零改动复用，并得到 G-F8b 这条免费的正确性检查 |
| D-49 | 前 100,000 行的特征是否重算 | 不重算，从 R2 的 `feats/100k/x_m.npy` 原样复制 | 描述符只依赖 id、family 字符串与 `size_b`，family 字符串由四级解析决定、不依赖湖规模，因此重算结果应当相同；复制则连浮点末位差异也一并排除 |
| D-50 | `family_id` 是否也复制 | 不复制，对全部 N 行按全量湖重算，并记录前 100,000 行里变化的行数 | `family_id` 是 vocab 行号，由 `FAMILY_MIN_COUNT` 在当前湖上的计数决定；在 R2 里不够门槛的家族在全量里可能越过门槛。CORE 行仍然不移位 |
| D-51 | R3/R4 是否仍作为独立档 | 不再作为独立的爬取与选样档。500K 保留为一个训练中间档兼可行性探针，1M 及其他中间规模改为索引子采样点 | 检索侧的 N 效应可以用同一份 `z_m` 的子集索引测得，成本相差两个数量级；训练侧只在必要的少数点上重训。见 §3.2 |
| D-52 | 全量档的主指标 | `median rank / N` 与 vs random 为主，`gold@10` 保留为兼容历史的次要列 | 517 个查询、每查询数个 gold，对数百万候选池而言 `gold@10` 的分辨率不足；`median rank / N` 是唯一跨 N 可比的量。这一条在看到数字之前定下 |
| D-53 | `plan.md §4` 的第 9、10 项 | 只补做第 10 项（图拆件加 memmap）；第 9 项按 D-55 取消 | 100K 档按 D-21 推迟第 10 项的理由（图约 250 MB）在全量下不再成立。F0 已完成，实测加载峰值 243 MB → 50 MB |
| D-55 | 特征是否降精度存储 | 不降。`x` 保持 float32，省内存靠内存映射 | 用户 2026-08-18 定。`x` 是模型输入里被冻结的那一半，也是整条梯子跨档保持逐字节相同的量（约束 1），存储格式不是引入数值差异的地方。映射方案在不动精度的前提下把加载峰值降到约五分之一，fp16 带来的额外收益不值得为它承担一次全链路的数值验证 |
| D-54 | `displacement_quality` 是否补齐质量对比 | 不补，只报成分 | 需要把 model-index 的 (dataset, metric, value) 与 CORE 数据集节点做名字对齐，命中率未知，属独立数据工程。不假装做过 |

---

## §12 与 R2 runbook 的对应

| R2（[`100k/100kplan.md`](100k/100kplan.md)） | 本档 | 差异 |
|---|---|---|
| T0 代码改造七项 | F0 | 七项沿用，补做第 10 项（拆件）；第 9 项（fp16）取消，见 D-55 |
| T1 watGPU 环境 | 沿用，不重做 | 环境已在 R2 打通；仅按 §9 重新申请资源 |
| T2 候选发现与均衡选样 | F1 | 改为单流穷举枚举，不选样（D-46、D-47）。已完成：3,003,759 条 |
| T3 canonical 与梯子 | F2、F3 | 规则不变，梯子增加固定前缀（D-48） |
| T4 特征 | F4 | 前 100,000 行复制（D-49），其余新算，float32 存盘 |
| T5 建图 | F5 | 拆件存储；血缘边数量级变化是主要观察点 |
| T6 训练 | F6 | 配置零改动；新增 500K 中间档 |
| T7 导出与索引 | F7 | 增加子采样索引 |
| T8 三轴评测 | F8 | 三轴口径不变，增加两条 scaling 曲线，主指标改为 `median rank / N`（D-52） |
