# HuggingFace 全量模型湖：端到端证据来源技术报告（中文）

版本截止：**2026-09-01**。项目状态：**已进入 X 系列，当前到 X5；F0–F9 是已完成的全量基线建库与评测链，不是当前最终方法。**

配套英文版见 [`EVIDENCE_SOURCE_LIBRARY_en.md`](EVIDENCE_SOURCE_LIBRARY_en.md)；逐文件哈希、代码快照与验证边界见 [`EVIDENCE_SOURCE_MANIFEST.md`](EVIDENCE_SOURCE_MANIFEST.md)。本文依照 `tech_reports_writing_skills.md` 的 Conclusion → Evidence → Impact 与 Problem → Evidence → Decision → Verification 结构编写。

## 1. 报告边界、证据等级与当前结论

### 1.1 什么是“当前结果”

当前可辩护的主结果不是 F8 的 `gold@10 = 0.0598`，也不是把 X2/X3 的先验收益乘到 X4 上。当前主结果是：

- **表示**：X4 的合并臂 GD，即训练时同时使用 `lake_gamma=0.5` 与 `global_n_datasets=128`；
- **主协议**：root-aware held-out 前向；
- **查询口径**：X5 的 `gold_eligible=True`；
- **候选宇宙**：全部 **3,016,439** 个模型；
- **三 split seed `gold@10`**：0.1355 / 0.1417 / 0.1508，均值 **0.1427**；
- 同口径 F6 基线为 0.0738 / 0.0418 / 0.0718，均值 **0.0625**；因此 X4 GD 是当前合格查询基线的 **2.28×**。[M: [`X5_runs/X5_GD_ELIGIBILITY.json`](X5_runs/X5_GD_ELIGIBILITY.json), [`X5_runs/X5_F6_ELIGIBILITY.json`](X5_runs/X5_F6_ELIGIBILITY.json); D: `0.14266660 / 0.06249103 = 2.283`]

X2 的 task 先验 5.03×与 X3 的兄弟先验增量是在 **F6 冻结表示**上测得；X4 的九份新表示尚未重跑 P/S 轴。因此它们是独立的服务侧证据分支，不能与 X4 GD 的 0.1427 合成一个“最终系统数字”。[M: [`X2.md`](X2.md) §1, [`X3.md`](X3.md) §1, [`X4.md`](X4.md) §8]

### 1.2 证据标签

本文使用以下标签：

| 标签 | 含义 | 可用于论文的方式 |
|---|---|---|
| `[M]` | 归档产物或日志中的直接测量 | 可报告，必须带协议、样本、硬件或候选池条件 |
| `[D]` | 从直接测量算出的比值、差值或汇总 | 可报告，要给出原始量与计算式 |
| `[I]` | 代码或配置中已经实现的事实 | 可描述方法；不等同于效果已经验证 |
| `[P]` | 运行前计划或预注册 | 用于证明事前约束，不能写成实测 |
| `[U]` | 未执行、无法复核或证据冲突 | 只能写成限制或待办 |

证据优先级为：**原始 JSON/MANIFEST/resolved config/日志 > 与运行绑定的代码哈希 > 执行记录文档 > 计划和指南**。例如 `1Mplan.md` 的早期概述称“2 层 GraphSAGE”，但 F6 与全部 X4 `resolved_config.json` 均记录 `num_layers=1`；本文按 as-run 产物写 **1 层**，并在 §22 登记冲突。

### 1.3 一句话结论

本项目已建立一个由 2026-08-18 HuggingFace 全站快照与六来源历史监督共同构成的、可追溯的 3.016M 模型检索实验系统。F 系列证明全量建库、训练、导出、ANN 检索和四轴测量均可运行；X 系列进一步定位并干预训练稀疏性。当前最可靠的效果结论是：**混合提议分布是主要有效因素，扩大每步数据集覆盖只在与其组合时稳定增益；GD 在 X5 合格查询口径下把 `gold@10` 从 0.0625 提到 0.1427。** 该指标是历史记录恢复率，不是未标注模型的真实推荐正确率。

## 2. 端到端数据流与阶段状态

```text
HF /api/models ─F0探针─F1全量快照──────────────┐
                                                ├─F2规范化+六源监督─F3行序冻结─F4模型特征─┐
HF /api/datasets ─F1.5全量卡片与保守匹配──────┘                                      │
历史五源测量图───────────────────────────────────────────────────────────────────────┤
                                                                                       ▼
                                      F5异构图─F6基线训练─F7 held-out导出/HNSW─F8四轴测量
                                                                                 ├─F9效用记分卡
                                                                                 └─X1诊断
                                                                                    ├─X2 task先验
                                                                                    ├─X3 node-level兄弟协议
                                                                                    └─X4 G/D/GD重训─X5查询资格修正
```

| 阶段 | 状态 | 核心输入 | 核心输出 | 当前意义 |
|---|---|---|---|---|
| F0 | 完成 | HF API、100K 图 | 枚举可行性、容量估算、sharded graph store | 证明全量工程可行 |
| F1 | 完成 | HF `/api/models` | 3,003,759 条只读快照 | 候选快照前缀 |
| F1.5 | 完成 | HF `/api/datasets`、数据集节点 | 1,008,417 仓库索引与保守卡片映射 | 数据集文本视图 |
| F2 | 完成 | 快照 model-index + 历史五源 | 247,803 监督边、18,729 节点、冻结 gold 规则 | 标签与查询定义 |
| F3 | 完成 | 规范化节点、监督 | 3,016,439 模型与 18,729 数据集的固定行序 | 全流程主键 |
| F4 | 完成 | 固定模型行序 | `[3,016,439,448]` float32 `x_m` + 离散 id | 模型初始特征 |
| F5 | 完成 | F2/F3/F4、卡片 | 5 类边的 sharded 异构图 | 训练输入 |
| F6 | 完成 | 图、L1L3b 配置 | 3 个 H200 基线 checkpoint | 历史 F 基线 |
| F7 | 完成 | checkpoint、图 | held-out 嵌入、全湖/子采样 HNSW | 冻结评测输入 |
| F8 | 完成 | F7 产物 | A/B/C/D 四轴 | 全量基线测量 |
| F9 | 部分完成 | F7/F8、元数据 | 两层效用记分卡、`t0` 列表 | 解释“命中”和可用性；盲审/时间后验未做 |
| X1 | 完成 | F6/F7/F8/F9 | 三 seed 诊断 | 找到负样本与覆盖缩水 |
| X2 | 完成 | F6 表示、sidecar | task 先验 P 轴 | 服务侧 5×，但以任务先验为主 |
| X3 | 完成 | F6 表示、完整 sidecar | node-level S 轴 | 兄弟通道在第二协议有效 |
| X4 | 完成 | 同图、G/D 两个干预 | G、D、GD 各 3 seed，共 9 runs | 当前表示改进证据 |
| X5 | 完成 | F6/X4 表示、`gold_eligible` | 合格/排除查询并列评测 | 当前主查询口径 |

## 3. 实验对象、数据契约与指标定义

### 3.1 候选、监督和“湖”不是同一个集合

HF 快照本身有 **3,003,759** 个模型。六来源监督涉及一批快照外的历史模型，F2 最终追加 **12,680** 个，因此实际训练、导出和主评测的候选宇宙是 **3,016,439**。[M: [`F1_runs/PROVENANCE.json`](F1_runs/PROVENANCE.json), [`F2_runs/F2_MERGE_REPORT.json`](F2_runs/F2_MERGE_REPORT.json)]

快照仍是固定行序的精确前缀；历史模型只追加在尾部。这允许同时回答两个不同问题：

1. 全体候选口径：系统能否在包含历史测量模型的监督宇宙中恢复 gold；
2. `in_snapshot=true` 口径：今天仍可由该快照枚举的模型中表现如何。

两套口径会重新决定哪些查询仍有至少 3 个可比模型，因此查询集不同，不能把它们读成“只删 0.42% 候选后的同题变化”。[M: [`F7.md`](F7.md) §3]

### 3.2 监督、gold 与查询资格

每条监督边表示一个模型在一个规范化 `(dataset, task)` 节点上的定向、组内归一化表现。快照 model-index 的原始值先按 `(dataset, task, full_metric_name)` 分组；非数值/非有限值丢弃；同一四元组重复值取中位数；已知 lower-is-better 指标翻转；未知方向保留边但不进入 gold；常数组置 0.5。[I: [`canonicalize_rf.py`](../../scale1m/canonicalize_rf.py), [`metric_semantics.py`](../../scale1m/metric_semantics.py)]

历史策展源只留下已经定向的权重，因此使用第五类 `curated`；在 `(dataset node, source)` 内再 min-max。跨源同一 `(model,dataset node)` 按 `modellens_v2 > d0_v1_5 > a_ctrl_2000m > hf_effective > diverse_zoo > hf_model_index` 取一条，被覆盖项留在冲突表。每个数据集节点在冲突消解后最多保留 200 条边，并按值分层抽样。[M/I: [`F2.md`](F2.md), [`merge_supervision.py`](../../scale1m/merge_supervision.py)]

冻结规则为 `rf-gold-2.0`，SHA-256 `be3fb05e…`。F2 的 `gold_eligible` 定义是：方向已知、非 RL、非占位数据集名、候选模型数至少 3。F8 旧 A 轴只使用“至少 3 且非恒定”这一可计算条件；X5 才把 `gold_eligible` 接入主评测。[M: [`F2_runs/rf_gold_rules.json`](F2_runs/rf_gold_rules.json), [`X5.md`](X5.md)]

### 3.3 划分与泄漏控制

主协议以数据集 root 为划分单位，而不是按边或节点逐条随机切。`make_root_aware_splits()` 用 `split_seed` 打乱 root，按边量近似分配 test 20%、val 10%、train 其余；训练边中 30% 作为 disjoint supervision，从消息图移除；val 的消息图只含 train，test 的消息图只含 train+val；反向 `rev_trained_on` 同步裁剪；负边按 1:1 从全图不存在的 `(model,dataset)` 对中采样。[I: [`d0_splits.py`](../../stage2TrainGraphSAGE/d0_splits.py#L38)]

`split_seed ∈ {0,1,2}`，`init_seed=0` 固定。因此三 seed 区间表示**划分噪声**，不是初始化噪声。[M: F6/X4 MANIFEST]

### 3.4 主指标的确切语义

对查询 `d`，`g_d` 是有记录候选中归一化表现最高的模型，排序分数是 `z_d · z_m`：

- `gold@K = 1[rank(g_d) ≤ K]`；
- `top3@10`：有记录前三名至少一个进入 top-10；
- `gold-gap@10`：距记录最优值不超过 0.01 的模型至少一个进入 top-10；
- `root_gold@10`：先按 query root 做宏平均；
- `median rank / N`：gold 的中位绝对名次除以候选池大小；
- `vs random = gold@10 / (10/N)`。

名次采用 `# strictly better + 1`，并显式屏蔽探针模型与自己的浮点路径差异。[I: [`global_metrics.py`](../../scale/global_metrics.py#L37), [`global_metrics.py`](../../scale/global_metrics.py#L148)]

**关键解释限制**：`gold@10` 只测历史最优记录能否被恢复。未在该 query 上出现记录的返回模型是“未知”，不是“错误”。F9 实测 F6 seed 0 top-10 的 95.24% 槽位在目标 query 上没有记录。[M: [`F9.md`](F9.md) §4]

## 4. F0：枚举与存储前置探针

**问题。** 全量 HF API 是否能稳定分页；单文件 PyTorch 图是否会在 3M 规模造成不可接受的加载峰值。

**证据。** `createdAt` 降序探针连续取 200 页 × 1,000 条，共 200,000 条；0 重复、0 `createdAt` 乱序、200 页全部满页且最后仍有 next cursor；21.7 秒、74,491,136 bytes，页面中位耗时约 0.103 秒。[M: [`F0_runs/f0_probe.json`](F0_runs/f0_probe.json)] 游标解码只含 `_id < ObjectId`，选择不可变 `_id` 作为分页边界。[M: [`F0.md`](F0.md)]

54 个按月速率探针外推总量约 2.89M，标注 ±15%；完整字段页约 4,753 B/record，预估下载约 13.8 GB、gzip 落盘约 480 MB、30–40 分钟。[D/P: [`F0_runs/f0_size_estimate.json`](F0_runs/f0_size_estimate.json)] 后续 F1 实测 3.003759M，外推误差约 −3.9%。

图存储探针把 100K 图拆为 `.npy` mmap、NPZ 边/节点、parquet ID 表和 meta 哈希；逐张量 round-trip 相等。单文件 `torch.load` 内存增量 243.3 MB，mmap 初载 50.3 MB；全触碰后 237.8 MB，说明 mmap 降低加载峰值但不会消灭真正访问数据的成本。[M: [`F0.md`](F0.md)]

**决定。** `hf_crawl.py` 新增 `--sort/--direction`，旧默认仍保留；全量用 `createdAt -1`。特征保持 float32，节省内存依靠 `graph_store.save_sharded/load_sharded()`，不降为 fp16。[I: [`hf_crawl.py`](../../scale1m/hf_crawl.py#L757), [`graph_store.py`](../../scale1m/graph_store.py#L65)]

**影响。** F0 只证明工程可行，不是最终数据规模或性能结果；2.89M 与容量预测均被后续实测取代。

## 5. F1：模型全站快照

执行入口为：

```bash
python -m scale1m.hf_crawl --sort createdAt --direction -1 \
  --limit 4000000 --shard-size 50000 --v2-fields --out <DATA>/data1m/candidates_full
```

实际从 2026-08-18 18:01:49 UTC 到 18:29:18 UTC 完成：**3,003,759** 条、3,004 页、61 shards、1,648.6 秒；1 次 retry；6 次限流休眠共 996 秒；0 duplicate；游标耗尽正常结束。[M: [`F1_runs/PROVENANCE.json`](F1_runs/PROVENANCE.json)] 最新 `createdAt` 为 2026-08-18 18:01:36Z，最末记录字段为 2022-03-02；后者是 API 字段回填边界，不能称为 HF 最早模型发布日期。[M/U: [`F1.md`](F1.md) §2]

爬虫只保留后续所需字段：模型 id、createdAt/updatedAt/downloads/likes、标签、pipeline/library、private/gated/disabled、safetensors、config、cardData 中 model-index/base_model 等。[I: `trim()`, `trim_model_index()`, `trim_base_models()`, `trim_config()` in [`hf_crawl.py`](../../scale1m/hf_crawl.py)] 每个 shard 有 SHA-256，`PROVENANCE.json` 固定 snapshot window 与游标终止状态。

快照内原生监督盘点：[M: [`F1_native_supervision.json`](F1_runs/f1_native_supervision.json)]

| 项 | 实测 |
|---|---:|
| 带可解析 model-index 的模型 | 107,210 |
| model–dataset 对 | 208,035 |
| model–dataset–metric 三元组 | 2,158,375 |
| 不同数据集 / dataset-task 节点 | 7,700 / 8,793 |
| ≥3 / ≥10 / ≥20 模型的数据集 | 2,586 / 964 / 646 |

血缘声明 894,089 个模型；可解析到湖内父节点 859,056（96.08%）；76,669 个不同父节点；子数 p50/p90/p99 为 1/6/118；119 个父节点超过 1,000 个子模型；最大 hub `black-forest-labs/flux.1-dev` 有 43,346 个子模型。[M: [`F1.md`](F1.md)]

初始分层为 labeled 110,806（3.69%）、lineage 862,666（28.72%）、plain 2,030,287（67.59%）。`safetensors.total` 只覆盖 28.19%。这两项后来直接解释训练稀疏性和 size 特征缺失。

## 6. F1.5：数据集全量枚举与卡片映射

逐 id 请求 7,700 次会跨越多个限流窗口，因此改为一次全量枚举：

```bash
python -m scale1m.hf_crawl_datasets --out <DATA>/data1m/datasets_full
```

实测 **1,008,417** 个数据集仓库、1,009 页、11 shards、411.5 秒；0 retry、0 duplicate、2 次限流睡眠共 200 秒、游标耗尽。[M: [`F15_runs/PROVENANCE.json`](F15_runs/PROVENANCE.json)]

匹配策略经过两步收紧：规范化 id 完全相同则采用；只有 model-index 名字不含 owner 且 basename 唯一时才采用。跨 owner 的 85 个 basename 与 1,626 个歧义 basename 均拒绝，不按下载量猜。否则虽然节点卡片覆盖能从 33.29% 表面抬到 52.75%，但会制造错误归属，例如把 `abdshhayan/mbpp` 指到 `google-research-datasets/mbpp`。[M: [`F15_runs/f15_policy.json`](F15_runs/f15_policy.json)]

初始 8,793 节点有 2,927 个（33.29%）得到真实卡片；按监督记录加权覆盖 54.18%。非 RL 且 ≥3 模型的查询候选节点覆盖 35.50%，按记录加权 59.26%。未命中按模型数计 61.03% 属于 RL 环境名。[M]

六源节点合并后，18,729 个节点中 3,928（20.97%）有真实卡片；按 247,803 边加权为 27.25%。7,859 个合格查询中有卡片 1,437（18.28%），边加权 27.87%；ModelLens 来源的查询卡片覆盖只有 5.19% 节点 / 2.83% 边。[M: [`F15_runs/f15b_coverage.json`](F15_runs/f15b_coverage.json)]

未命中节点不丢弃，退回 `dataset_descriptor()` 的清洗名字 + task。卡片文本使用名字、task categories、非冒号 tags 和最多 400 字 description；MiniLM 本身仍受 512 token 截断。[I: [`match_dataset_cards.py`](../../scale1m/match_dataset_cards.py), [`d0_build_graph.py`](../../stage1BuildTransferGraph/d0_build_graph.py#L138)]

## 7. F2：canonical 化、指标方向与六来源监督冻结

### 7.1 快照 model-index 规范化

```bash
python -m scale1m.canonicalize_rf --candidates <DATA>/data1m/candidates_full --out <DATA>/data1m/rf
```

3,003,759 条模型记录与 F1 的九个锚点逐项一致。2,158,375 条原始指标记录中 61,294 条不可解析或非有限，余 2,097,081；按四元组中位数去重后 1,435,162；38,358 个常数组置 0.5。生成 7,944 dataset-task 节点、143,478 条主指标边；每节点 200 条上限后为 74,346 条、131 个节点被截断。[M: [`F2_runs/F2_REPORT.json`](F2_runs/F2_REPORT.json)]

原始方向分类为 higher 1,539,828（70.96%）、lower 15,835（0.73%）、reward 58,456（2.69%）、unknown 555,929（25.62%）。去重后的行数分布不同：higher 986,366、lower 12,115、reward 198、unknown 436,483；不要混用两套分母。[M: [`F2.md`](F2.md), `F2_REPORT.json`]

`mean_reward` 的 97% 左右是 `"11.05 +/- 5.90"` 一类字符串；项目决定不增加解析器。RL 最终监督占比低，不是因为排除策略成功，而是大部分在数值解析阶段已经丢失。[M/D: [`F2.md`](F2.md) §5]

边上限使集中度改善：有效数据集数 `1/HHI` 从 177.6 到 520.8，top-10 数据集份额从 14.95% 到 5.32%。这说明上限改变了分布；不能声称它完全消除了生态偏斜。[M]

### 7.2 六源合并

```bash
python -m scale1m.merge_supervision --rf <DATA>/data1m/rf
```

| 来源 | cap 后边数 | gold 查询数 | 证据性质 |
|---|---:|---:|---|
| `modellens_v2` | 117,898 | 4,681 | 历史策展测量 |
| `d0_v1_5` | 45,992 | 1,143 | 历史策展测量 |
| `a_ctrl_2000m` | 3,670 | 65 | 历史测量 |
| `hf_effective` | 5,223 | 75 | 历史测量 |
| `diverse_zoo` | 1,350 | 17 | 历史测量 |
| `hf_model_index` | 73,670 | 1,878 | 作者自报 model-index |

输入 531,958 行；源内重复折叠 5,102；跨源冲突 1,502；冲突消解后 525,354；339 个节点受 cap，最终 **247,803** 边、**18,729** 节点。合格查询深度：≥3 为 **7,859**、≥5 5,526、≥10 3,450、≥20 1,909。[M: [`F2_runs/F2_MERGE_REPORT.json`](F2_runs/F2_MERGE_REPORT.json)]

历史源涉及的模型与快照 join 后实际只有 12,680 个不在快照，故最终候选为 3,016,439。计划早期的 16,713 是合并前估计，已被 F2 实测取代。[M]

## 8. F3–F5：行序、特征与异构图

### 8.1 F3 行序冻结

`build_ladder_rf.build()` 把快照 3,003,759 行原序固定为 `mappedID=0..3,003,758`，历史 12,680 模型按规范化 id 排序追加。历史行无 `size_b`；family 从历史图词表恢复：ModelLens 12,658、diverse zoo 14、D0 8；`layer=no_snapshot_record`；不复制历史 `x`。[I/M: [`build_ladder_rf.py`](../../scale1m/build_ladder_rf.py), [`F3_runs/LADDER_REPORT.json`](F3_runs/LADDER_REPORT.json)]

八条断言全过：总数、连续 mappedID、规范化 id 唯一、快照精确前缀、追加行不在快照、每个监督模型有行、数据集 mappedID 连续、每个监督数据集有行。模型 ladder SHA `fee360d1…`，数据集 ladder SHA `31c027ff…`。

### 8.2 F4 模型特征

实际冻结矩阵是 `x_m = [e_name64 || e_desc384]`，形状 `[3,016,439,448]`、float32；size 和 family 不是塞进这 448 维，而是另存 `size_bucket_id` 与 `family_id`，训练时分别查 16 维可学习 embedding 后拼接。[I: [`embed_lake_rf.py`](../../scale1m/embed_lake_rf.py), [`model_node_encoder.py`](../../stage1BuildTransferGraph/dataset_embed/model_node_encoder.py#L45)]

`e_name` 为 seed 42 的 64 维哈希平均；`e_desc` 是 `all-MiniLM-L6-v2` 对 `model_descriptor()` 的编码，文本由清洗模型名、family、可用时的粗粒度参数量组成。无原始模型卡正文。[I: [`modellens_build_graph.py`](../../scale/modellens_build_graph.py#L67)]

本地 RTX 4060 用 batch 256、四个约百万行分片编码并 mmap 组装，712.7 秒。`x_m.npy` 5.41 GB，SHA `ba102087…`；41,056 个 family rows，`Other` 13.561%，vocab SHA `00d304df…`；size unknown 71.929%。全部行无 NaN/Inf/全零，头尾各抽 100 行重算，最大差 0。[M: [`F4_runs/FEATS_REPORT.json`](F4_runs/FEATS_REPORT.json)]

### 8.3 F5 数据集特征与图

数据集特征为 `[e_name64(seed 43) || e_card384(MiniLM) || e_stats10] = 458` 维。`similar_to` 只在 384 维 card 视图上做 k=20 余弦 KNN；最后两个内容统计维度保持禁用。18,729 节点中 3,928 有 HF 卡片，其余用名字兜底；描述符平均 97.07 字符。[I/M: [`build_graph_rf.py`](../../scale1m/build_graph_rf.py), [`F5_runs/GRAPH_REPORT.json`](F5_runs/GRAPH_REPORT.json)]

图在 47.6 秒内建成，sharded 目录约 5.66 GB：

| 关系 | 正向边 | 反向边 | 说明 |
|---|---:|---:|---|
| `trained_on` | 247,803 | 247,803 | 监督，反向供消息传递 |
| `similar_to` | 374,580 | — | 每个数据集 20 个近邻 |
| `is_base_of` | 859,065 | 859,065 | 基础模型到派生模型及反向 |
| 合计 |  | **2,588,316** | 五种 edge types |

血缘声明 894,089，去掉 2,008 自指，解析率 96.083%；76,672 父节点；p50/p90/p99 子数 1/6/118；119 个超千子；最大 43,346。比 F1 多 9 条解析边，来自追加历史模型。[M: [`F5_runs/lineage_stats.json`](F5_runs/lineage_stats.json)]

`task_type_id`、class bucket、arity 在 RF 图中退化为各一类，`num_task_types=1`；训练不会报错，但三张数据集离散 embedding 只相当于共享偏置。X2 的服务侧 task 分组后来改读 parquet 的真实 task 列（归一后 2,198 组），但 F6/X4 训练图本身没有补建。[M/I: [`X2.md`](X2.md) §1, D-68 in [`1Mplan.md`](1Mplan.md)]

图合同检查曾发现 `family_vocab.csv` 中字面量 `nan`/`null` 被 pandas 默认当 NA，导致词表少一行；改为 `keep_default_na=False` 后通过。该失败是已修复的实现证据，不应从报告中抹去。[M: [`F5.md`](F5.md)]

## 9. F6：全量基线训练

### 9.1 as-run 模型与目标函数

F6 的实际配置来自每个 run 的 `metadata/resolved_config.json`，不是 `1Mplan.md` 的架构概述。实际模型为：

- **1 层**异构、edge-aware GraphSAGE；hidden 128，输出 128，输出 L2 归一化；
- model encoder：448 维冻结文本 + 16 维 size embedding + 16 维 family embedding，再投影到 hidden；
- dataset encoder：458 维冻结视图 + 16/8/4 维 task/class/arity embedding，但后三张表各只有 1 行；
- 每个关系有独立消息传递参数；`weighted_relations=[]`，所以本次不消费连续 edge weight；
- shared output head，dot scorer；`ranknet`，`rank_min_gap=0.02`；
- `top_frac=0.1`，`lambda_rank=1`，`lambda_contrast=1`，`lambda_global=1`，`lambda_dm_contrast=0`；
- whole-lake logQ sampled softmax：`q(m) ∝ (deg_train(m)+1)^0.75`，`global_n_neg=256`，基线 `global_n_datasets=16`；
- `batch_size=1024`，fanout loader、sparse membership、256 sampled contrastive negatives、50,000 分块推理；AMP 关闭，诊断在训练内跳过、由 F8 重算。[M/I: [`F6_runs/RF_full_s0_e25/metadata/resolved_config.json`](F6_runs/RF_full_s0_e25/metadata/resolved_config.json), [`losses.py`](../../stage2TrainGraphSAGE/losses.py#L1015), [`train_rung.py`](../../scale1m/train_rung.py#L176)]

总体训练损失包含 RankNet 性能排序、模型侧监督对比和全湖 logQ 全局检索项。全局项对抽到的 dataset 做平均；这使 D 臂与 F6 的 loss 可比较。改变 `gamma` 会改变 logQ 分母，因此 G/GD 的 loss 不能与 F6/D 横向解释。[I/M: [`ablation.py`](../../stage2TrainGraphSAGE/ablation.py#L118), [`X4.md`](X4.md) §4/§6]

### 9.2 执行与结果

远端 as-run 命令保存在每个 `resolved_config.json`。等价的调度形式为：

```bash
for s in 0 1 2; do
  RUNG=full SEED=$s EPOCHS=25 \
  sbatch --export=ALL,RUNG,SEED,EPOCHS scripts/watgpu/train_rung_rf.sbatch
done
```

三条均在 `watgpu508` 的 NVIDIA H200 NVL 上运行，Python 3.11.4、PyTorch 2.12.0+cu130、CUDA runtime 13.0、`pyg_lib 0.8.0+pt212cu130`。[M: resolved configs, [`F6.md`](F6.md)]

| split seed | Slurm job | 墙钟 | 峰值显存 | epoch 0 → 24 loss | 机制门 |
|---:|---:|---:|---:|---:|---|
| 0 | 1515431 | 426.4 s | 38.001 GB | 22.2925 → 16.6288 | PASS |
| 1 | 1515432 | 498.8 s | 38.015 GB | 22.5152 → 16.4704 | PASS |
| 2 | 1515433 | 379.4 s | 37.976 GB | 22.2175 → 16.2624 | PASS |

机制门要求 loss 下降、无 NaN、size/family 可学习表发生更新；冻结 `x` 不应有梯度。三份 checkpoint 绑定同一图摘要 `0e80b839…`、同一 family vocab `00d304df…`，N=3,016,439。[M: F6 MANIFESTs]

训练可见带监督模型仅 37,755 / 39,930 / 35,991，即约 1.25% 的湖。日志中 `q_head≈1e-4`，当前 `q` 落在带监督模型上的质量后来由 X1 重算为约 3.52%。[M]

### 9.3 复现限制

F6 的 `git_head` 指向远端旧 commit；`uncommitted.patch` 只覆盖 tracked 文件，而当时的 `graph_store.py` 和 sbatch 是 untracked。因此 F6 可由 checkpoint binding、运行文件哈希表和产物复核，但不能只凭一个 Git commit 完整恢复。[U: [`F6.md`](F6.md) §7]

500K 训练侧 scaling 点没有构建。项目先验证全量能否运行，配额不足时按计划优先砍掉 500K；因此现有 scaling 曲线只有**固定表示下改变检索候选池**，没有“重训随 N 变化”的曲线。[U]

## 10. F7：held-out 导出与 HNSW 索引

`export_rf.py` 分四个独立进程：`embed`、`metrics`、`index`、`curve`。这是内存约束下的实现决定：`x_model.npy` 5.41 GB、`z_m` 1.54 GB、HNSW 2.2 GB 不能稳定同时驻留。[I: [`export_rf.py`](../../scale1m/export_rf.py)]

```bash
python -m scale1m.export_rf --run docs/1M/F6_runs/RF_full_s0_e25 --stage embed   --out <EXPORT>
python -m scale1m.export_rf --run docs/1M/F6_runs/RF_full_s0_e25 --stage metrics --out <EXPORT>
python -m scale1m.export_rf --run docs/1M/F6_runs/RF_full_s0_e25 --stage index   --out <EXPORT>
python -m scale1m.export_rf --run docs/1M/F6_runs/RF_full_s0_e25 --stage curve   --out <EXPORT>
```

每个 seed 导出：

- `z_m.npy/z_d.npy`：完整消息图前向，供服务；
- `z_m_eval.npy/z_d_eval.npy`：该 seed 的 test held-out 消息图前向，A 轴只能使用这对；
- `gold_cands.npz`、模型/数据集 id parquet；
- 全湖 `hnsw_full.bin` 和调参轨迹；seed 0 另有 12 个子采样索引。

嵌入阶段 55.8–72.1 秒；metrics 26–33 秒；index 110–113 秒。HNSW `M=32`、construction ef 200；服务 `ef_search` 用二分找达到 `recall@50 ≥ 0.99` 的最小值。[M/I: [`F7.md`](F7.md)]

八道门均通过：N 断言；held-out `gold@10` 小于 full-message `gold@10`；100 个行序抽样及全表对账；100K 子图上分块/整图前向最大差 7.5e-8 / 1.0e-7 / 8.9e-8；两套 metric harness 逐位一致；HNSW recall；子索引包含全部 46,146 个监督/gold 行；两个候选池均报告。[M: F7 EXPORT_MANIFESTs]

泄漏门的 full vs held-out `gold@10` 是 0.1046 vs 0.0700、0.0555 vs 0.0399、0.1674 vs 0.0696。若误用完整消息图，最高会虚高约 2.4×。[M]

全湖 HNSW 2,209.3 MB，`ef=50`，`recall@50=0.9983–0.9988`。每 10 万模型约 73.3 MB，与 100K 的 73.2 MB 基本线性。子采样候选宇宙为 100K/250K/500K/1M，各 3 个采样 seed；全部先固定保留 46,146 个监督与 gold 模型，再从无标签长尾均匀抽样。[M]

F7 在本地 RTX 4060 笔记本执行，而 F6 在 H200。嵌入内容是确定性前向，索引 recall 与磁盘不依赖机器；绝对延迟依赖机器，因此 F8 只在同一本机内比较 HNSW 与全扫。[M/U]

## 11. F8：F 系列四轴基线

### 11.1 A 轴：历史记录恢复

全体 3,016,439 候选、held-out 前向、旧查询资格：

| 指标 | seed 0 | seed 1 | seed 2 | 三 seed 均值 |
|---|---:|---:|---:|---:|
| `gold@1` | 0.0019 | 0.0078 | 0.0107 | 0.0068 |
| `gold@10` | 0.0700 | 0.0399 | 0.0696 | **0.0598** |
| `top3@10` | 0.1232 | 0.1570 | 0.1147 | 0.1317 |
| `gold-gap@10` | 0.0924 | 0.0703 | 0.0859 | 0.0829 |
| `root_gold@10` | 0.0226 | 0.0328 | 0.0407 | 0.0320 |
| median gold rank | 1,908 | 881 | 3,363 | 2,051 |
| `median rank/N` | 6.33e-4 | 2.92e-4 | 1.11e-3 | 6.80e-4 |
| vs random | 21,103 | 12,034 | 20,992 | 18,043 |
| 查询数 | 1,558 | 1,153 | 1,595 | 1,435.3 |

[M: [`F8_runs/F8_REPORT.json`](F8_runs/F8_REPORT.json)] 预注册的 `median rank/N ∈ [1e-5,5e-3]` 与 `gold@10 ∈ [0.005,0.15]` 且 vs random ≥1,000 均命中。[P/M: [`1Mplan.md`](1Mplan.md) §6]

固定 seed 0 表示，候选从 100K 扩到 3.016M：`gold@10` 0.0706 → 0.0700，中位绝对排名约 1,736 → 1,908；30× 候选只使命中下降 0.85%、绝对排名后移 9.9%。因此全量基线低的主要原因不是无标签长尾挤占，而是 46,146 个带监督模型之间排序不足。[M/D]

仅快照候选的 `gold@10` 为 0.0229 / 0.0532 / 0.0613，对应查询 1,003 / 752 / 636；与全体候选列不是同题，不计算简单差值。[M]

### 11.2 B 轴：iso-recall 延迟

同一台本地机器、同一进程，先调到 `recall@50≥0.99`，100 次 warmup 后逐查询 1,000 次：

| N | ef | recall@50 | HNSW p50 | brute-force p50 | 加速 |
|---:|---:|---:|---:|---:|---:|
| 100K | 50 | ≥0.998 | 0.0175–0.0187 ms | 1.094–1.262 ms | 59–67× |
| 250K | 50 | ≥0.998 | 0.0199–0.0208 ms | 4.527–5.216 ms | 227–251× |
| 500K | 50 | ≥0.998 | 0.0185–0.0214 ms | 9.439–10.655 ms | 493–512× |
| 1M | 50 | ≥0.998 | 0.0195–0.0210 ms | 18.388–20.016 ms | 925–991× |
| 3.016M | 50 | 0.9989 | 0.0211 ms | 57.261 ms | **2,714×** |

HNSW p50 对 N 的幂律指数 0.036，全扫约 1。批量 1,558 查询吞吐从单线程 52,655 QPS 到 24 逻辑线程 296,417 QPS。逐查询调用时 hnswlib 不会利用查询批并行，首版“多线程 QPS”曾出现 65× 不稳定值，已废弃并保留为负面执行记录。[M: [`F8.md`](F8.md) §3]

这些绝对毫秒数是本地笔记本值，不能与历史 watGPU 绝对延迟并列；2,714× 是同机比值。[U]

### 11.3 C 轴：冷启动与塌缩

| 层 | 定义 | 模型数 | 占比 | 平均层内余弦 | 有效维度 |
|---|---|---:|---:|---:|---:|
| warm | `trained_on` 度 ≥10 | 2,949 | 0.10% | 0.394–0.409 | 3.3–3.9 |
| cool | 度 1–9 | 43,197 | 1.43% | 0.293–0.426 | 3.8–4.0 |
| cold | 度 0、有血缘 | 875,314 | 29.02% | 0.882–0.931 | 3.0–3.3 |
| frozen | 度 0、无血缘 | 2,094,979 | 69.45% | 0.879–0.927 | 2.7–3.4 |

`cold+frozen=98.47%`。预期的“cold 比 frozen 更塌”没有复现，二者三 seed 均值约 0.908 vs 0.907。兄弟—随机余弦分离从历史 12K/30K 的 0.31–0.35、100K 的 0.15 降到全量的 0.028–0.049；不是兄弟更近，而是随机对也大多从塌缩的 98.5% 中抽到。[M]

hub 例：`qwen/qwen1.5-0.5b` 32,534 个子模型两两余弦 0.9863，`google/gemma-2b` 24,040 个为 0.9877，`flux.1-dev` 43,346 个为 0.9342。ANN recall 已达 0.998 且 ef 在下界，故这是表示问题而非索引参数问题。[M/D]

### 11.4 D 轴：系统成本

| 阶段/对象 | 实测 |
|---|---:|
| 模型快照爬取 | 27.5 min，202 MB gzip shards |
| 数据集索引 | 6.9 min，82 MB |
| 模型特征 | 712.7 s，`x_m` 5.41 GB |
| 建图 | 47.6 s，图 5.27 GiB/约 5.66 GB（单位口径不同） |
| F6 训练 | 379–499 s/seed，38.0 GB GPU |
| 导出 | 55.8–72.1 s/seed |
| 全湖索引 | 109.9–113.1 s/seed；2.21 GiB |
| 12 个曲线子索引 | 4.07 GiB |
| HNSW 单模型插入 | p50 0.160 ms，p95 0.255 ms |

HNSW 插入只测索引侧；完整新模型上线还需要生成 descriptor、MiniLM 编码、离散特征与归纳式 GNN 前向，这些组件存在但没有作为一个端到端单模型事务计时。[M/U]

`displacement_quality` 因其证据会与训练标签重叠而没有重写完成；这一 C/D 交叉指标没有结论。[U]

## 12. F9：推荐效用记分卡

F9 没有下载或运行模型；它执行了设计的第一层“历史记录恢复”和第二层“元数据可用性”，冻结了第四层的 `t0` 列表；第三层盲审与第四层时间后验尚未执行。[M/U: [`F9.md`](F9.md)]

五个排序源使用同一批 seed 0 的 1,558 个查询：F6 图表示、纯文本 MiniLM、全湖 popularity、`random_task_pool`、`random_lake`。纯文本基线使用 `x_m[:,64:448]` 与 `x_d[:,64:448]` 的相同 MiniLM 空间，没有图或训练。

| 指标 | F6 图表示 | 纯文本 | 解释 |
|---|---:|---:|---|
| recorded `gold@10` | 0.0700 | 0.0392 | 1.79× |
| recorded `top3@10` | 0.1232 | 0.0591 | 2.09× |
| 历史 best 中位排名 | 1,908 | 160,270 | 约 84× 前移 |
| same-task evidence@10 | 0.2963 | 0.0503 | 5.9× |
| record coverage@10 | 0.0476 | 0.0166 | 绝大多数返回项未知 |

[M: [`F9_runs/F9_SCORECARD.json`](F9_runs/F9_SCORECARD.json)] `random_task_pool` 的 recorded `gold@10=0.2349` 不是更强系统：它已知 query task，并从同任务已评测池中均匀抽 10；21.3% 查询的池不超过 20 个，按池大小分布算出的期望恰为约 0.235。[M/D]

F9 seed 0 第二层：

| 指标 | F6 图表示 | 纯文本 | 全湖随机 |
|---|---:|---:|---:|
| available@10 | 0.7099 | 0.9344 | 0.9794 |
| licensed@10 | 0.5978 | 0.3816 | 0.3625 |
| endpoint-compatible@10 | 0.5161 | 0.3289 | 0.3363 |
| library-tag@10 | 0.6006 | 0.4100 | 0.4996 |
| family diversity@10 | 0.4989 | 0.5105 | 0.9191 |
| at least one feasible@10 | 0.9724 | 0.9159 | 0.9987 |
| top-10 参数量中位数 | 1.77B | 1.29B | 3.08B |

F6 检索偏好有历史监督的模型，因此元数据质量较好，但可获取性更差；top-10 平均仅约 5 个 family，单 family 约占 4.2 个位置。X2 后来补出三 seed available@10 为 0.7099 / 0.6273 / 0.5729，证明 F9 的 0.7099 只是一条 seed 0 单点。[M]

冻结的 `t0_recommendations.parquet` 有 1,558×10=15,580 行，冻结日 2026-08-21。它允许未来只对当时未知、之后新增同 query 记录的推荐做时间后验，不允许用后来的标签回写 t0 排名。[I/M]

未标注候选的真实表现未测。资源审计估计全量密集重评需约 73.5 TB 传输与 5,300 GPU-hours；该数是资源推演，不是已发生消耗。[D/P: [`RESOURCE_E_AXIS_DENSE_REEVALUATION.md`](RESOURCE_E_AXIS_DENSE_REEVALUATION.md)]

## 13. X1：从 F 基线到 X 系列的诊断转折

X1 没有重训或改动冻结产物；它把六个一次性测量固化为 `fast_lever_audit.py` 的 `q/splits/task/ranks` 四阶段，并在三个 split seed 上复跑。总耗时约 84 分钟，其中全湖 ranks 4,981.3 秒。[M/I: [`X1.md`](X1.md), [`fast_lever_audit.py`](../../scale1m/fast_lever_audit.py)]

```bash
python -m scale1m.fast_lever_audit --stage all --seeds 0 1 2 --check
```

### 13.1 无标签长尾不是主要挤占者

从候选池删除 2,970,293 个无标签模型后，`gold@10` 仅从 0.0700/0.0399/0.0696 变为 0.0706/0.0408/0.0696，最大绝对变化 0.0009。只保留训练可见监督模型与保留全部 46,146 监督模型在 `gold@10` 上也相同。[M]

因此 F8 的低命中不是“3M 候选太多”的机械结果；有效训练信号与判别性对手不足更合理。

### 13.2 两个训练信号缩水

训练实际使用 `deg_train` 而不是全图度数。100K 与 3M 的对照为：

| 量 | 100K 三 seed 均值 | 3M 三 seed 均值 | 缩水 |
|---|---:|---:|---:|
| `q` 落在训练可见带监督模型上的质量 | 0.6138 | 0.0352 | 17.4× |
| 256 negatives 中带监督期望数 | 157.1 | 9.0 | 17.4× |
| 25 epochs 每个训练可见 dataset 被全局项触达 | 4.06 | 1.47 | 2.77× |
| 判别性触达组合量 | — | — | 约 **48×** |

[M/D: [`X1_runs/X1_FAST_LEVERS.json`](X1_runs/X1_FAST_LEVERS.json)] `global_n_datasets=16` 下，全量每 epoch 46–51 steps、训练可见 datasets 12,758–14,101；此前审计误把采样负边也算成 steps，绝对次数高估 2×，X1 已纠正。

混合提议 `q=(1−γ)q_degree+γ Uniform(labeled)` 的 seed 0 投影：γ=0/.25/.5/.75 时带监督质量 0.0355/.2766/.5177/.7589，对应 256 negatives 中 9.1/70.8/**132.5**/194.3。γ=0.5 与 100K 的 157.1 同量级，因此成为 X4 的预注册干预值。[M/D]

### 13.3 零训练通道与否定结果

| 排序 | 三 seed `gold@10` 均值 | 结论 |
|---|---:|---|
| 全湖 MIPS | 0.0598 | F 基线 |
| 同任务池均匀抽 10 | 0.1857 | 任务过滤上界参照 |
| 同任务池内用 MIPS | 0.2206 | 任务池有效 |
| 同任务其它数据集的收缩平均表现先验 | 0.3014 | 强服务先验 |
| 该先验的 pool ceiling | 0.5905 | 受划分强烈影响 |

CSLS 是明确负结果：测试侧估计 `r̄` 时三 seed +0.0051/−0.0191/−0.0596，均值从 0.0598 降到 0.0353；训练侧可部署估计也只有 0.0357，三个 seed 的 `gold@10` 均低于各自基线。因此 CSLS 从候选方案删除。[M]

`pipeline_tag` 过滤在可比查询上把 0 提到 0.0234，但远低于同任务**有记录**池的 0.2206；说明主要价值来自“在同任务上有测量记录”，不是声明式任务字符串。[M]

### 13.4 top-10 集中、兄弟覆盖与查询资格

F6 top-10 中 96.72%–99.35% 槽位是带监督模型；最热 100 个模型占 31.3%–51.4%；快照内比例仅 73.2%/63.8%/57.9%；平均不同 family 4.99/4.79/4.39。[M]

root-aware 下 66.0%/67.5%/77.7% 查询来自多节点 root，但训练可见兄弟查询数三 seed均为 **0**。这不是兄弟信号不存在，而是主协议按定义将整 root 留出。[M]

旧查询集中 `gold_eligible=False` 有 82/52/50 条，主要为方向未知 61/48/45；这成为 X5 的口径修正依据。[M]

X1 自比较屏蔽后，600 个抽样查询名次与 `global_metrics` 逐项一致；复算的训练可见模型数与 F6 日志逐 seed 相同。X1 的基线回到 F8 的 0.0700/0.0399/0.0696。[M]

## 14. X2：task 先验 P 轴

X2 适配 `build_prior_sidecar.py` 读取 sharded graph、parquet ids 与 split seed。关键修复是：RF 图的 `task_type_id` 恒 0，不能用于 task 分组；正确分组改读 `dataset_nodes_merged.parquet.task`，规范化后 2,198 组。[I: [`build_prior_sidecar.py`](../../stage3HNSW/build_prior_sidecar.py), [`X2.md`](X2.md)]

split-specific sidecar 只使用 train+val 消息边，并断言没有幸存边属于 test 数据集；三 seed 为 198,216 / 196,912 / 196,124 条。旧 D0 不带 split 的调用重建后，五个数组与历史 sidecar 逐个 `array_equal`，证明适配没有改变 D0 路径。[M]

当前 post-X5 `eval_rf.py` 还封闭了两个路径风险：`--full-sidecar` 不再在模块加载时写死为 `exports_rf/RF_full_s0_e25`，而是在解析 `args.run_fmt` 后动态生成；训练 MANIFEST 的目录另由 `--f6-runs/--f6-run-fmt` 解析，任一 seed 缺失即报错，不再把 `train_per_seed` 静默留空。这是当前代码的复现加固，不代表 X4 的 P/S 已经执行。[I: [`eval_rf.py`](../../scale1m/eval_rf.py), [`X4GPU.md`](X4GPU.md) §4.5]

```bash
python -m ModelLakeFishing.stage3HNSW.build_prior_sidecar \
  --graph-store <DATA>/graphs/hgraph_rf --export <EXPORT> \
  --split-seed 0 --task-nodes <DATA>/rf/canon/dataset_nodes_merged.parquet
python -m scale1m.eval_rf --axis p
python -m scale1m.eval_rf --axis pinf
```

融合分数在全湖上做 min-max，`fused = minmax(MIPS) + α·sibling_boost + β·task_boost`；root-aware 下 sibling 数被断言为 0，所以 P 轴只报告 task。task boost 是同 task 其它数据集上模型归一化表现的 k=5 收缩均值。排名扫描全部 3,016,439 模型，不受 serving `POOL=512` 上限干扰。[I]

| 排序 | seed 0 | seed 1 | seed 2 | 均值 |
|---|---:|---:|---:|---:|
| 纯 MIPS | 0.0700 | 0.0399 | 0.0696 | 0.0598 |
| 图中退化“单 task 组”先验 | 0.1232 | 0.1127 | 0.0940 | 0.1100 |
| task β=0.5 | 0.2914 | 0.3174 | 0.2464 | 0.2851 |
| task β=1 | 0.3132 | 0.3356 | 0.2539 | **0.3009** |
| task β=2 | 0.3087 | 0.3400 | 0.2558 | 0.3015 |
| β→∞ 下界 | 0.3338 | 0.3322 | 0.2301 | 0.2987 |

[M: [`X2_runs/X2_PRIOR_FUSION.json`](X2_runs/X2_PRIOR_FUSION.json)] β=1 与 β=2/极限基本持平，说明采纳权重已接近先验通道的饱和点。β=1 的 5.03× 主要来自 task 先验，不是与 MIPS 的协同；MIPS 主要在先验同分或 gold 不在 task pool 时发挥作用。[D]

第一层收益伴随第二层代价（三 seed均值）：available@10 0.6367→0.6017；at-least-one-feasible 0.9118→0.8973；family diversity 0.4722→0.4320；top-10 参数量中位数 **2.42B→7.57B**；record coverage 0.0498→0.1764；same-task evidence 0.2763→0.7574。[M] 因此不能只报告 5× 而省略体积和可用性。

## 15. X3：第二协议与兄弟先验

X3 的预注册文件 SHA-256 `4ee14720…` 被写入结果 JSON，证明预注册先于测量。[M: [`X3_runs/PREREGISTRATION.md`](X3_runs/PREREGISTRATION.md), [`X3_runs/X3_PROTOCOL_B.json`](X3_runs/X3_PROTOCOL_B.json)]

协议 A（主协议）按 root 切分，服务场景是“从未见过的 benchmark 家族”；协议 B 允许查询同 root 的其它 dataset/config 边可见，服务场景是“已知 benchmark 家族的新配置”。本次没有按 node-level 重新训练：查询与嵌入仍用 root-aware held-out 集，只有先验可读边集合改变。[I]

这带来不能相互抵消的偏差：现有嵌入未利用兄弟训练边，检索侧偏弱；完整 sidecar 还含真正 node-level 划分会留到 val/test 的部分标签，若用于 task(B) 则先验侧偏强。因此可辩护的第二协议行是 **`B:sibling + task(A)`**，不是 `B:sibling + task(B)`。[U/M]

| 排序 | 全部查询 | 有兄弟 | 无兄弟 | 解释 |
|---|---:|---:|---:|---|
| MIPS | 0.0598 | 0.0678 | 0.0418 | F6 表示 |
| A: task | 0.3009 | 0.3251 | 0.2551 | X2 主 task 先验 |
| B: sibling | 0.2145 | 0.2866 | 0.0418 | 兄弟单通道 |
| B: sibling（去名字相近） | 0.1209 | 0.1532 | 0.0418 | 敏感性分析 |
| **B: sibling + task(A)** | **0.3536** | **0.3975** | 0.2551 | 可辩护第二协议 |
| B: task(B) | 0.3744 | 0.4163 | 0.2622 | 先验侧上界 |
| B: sibling + task(B) | 0.4314 | 0.4961 | 0.2622 | 上界，不作主结果 |

[M] 在有兄弟层，兄弟单通道把 0.0678 提到 0.2866（4.2×）；加在 task(A) 上从 0.3251 到 0.3975（+22.3%），全部查询从 0.3009 到 0.3536（+17.5%）。

字符 3-gram Jaccard >0.5 的名字相近兄弟移除后，有兄弟层从 0.2866 降到 0.1532；这不是泄漏修复，而是说明约一半信号强度依赖高度相似配置。[M]

代价进一步放大：`B:sibling+task(A)` 的 available@10=0.5549、at-least-one-feasible=0.7615、family diversity=0.4091、top-10 参数量中位数 **12.11B**。[M]

## 16. X4：G/D/GD 三臂重训

### 16.1 预注册干预与正式执行

X4 在提交作业前冻结：

- G：仅 `lake_gamma=0.5`；
- D：仅 `global_n_datasets=128`；
- GD：两者同时；
- 每臂 split seeds 0/1/2，25 epochs；其它配置、图、行序、init seed 不变；
- 主判据：GD 三 seed均值 `gold@10≥0.09`，且每个 seed 都高于其 F6 对应值。[P: [`X4GPU.md`](X4GPU.md) §2]

X4 修改的正式远端文件与当前本地哈希一致：`losses.py c9b351…`、`ablation.py e0fa47…`、`train_rung.py 08904b…`、`train_rung_x4.sbatch f1237a…`。sbatch 初版曾为 `a0b2ab…`，增加 `X4_DIRECT=1` 绕过 watgpu608 故障的 `srun` job-step 后，正式版本变为 `f1237a…`；执行证据的 `code_sha256.txt` 与本地一致。[M: [`X4GPU.md`](X4GPU.md), [`X4_runs/x4_execution/code_sha256.txt`](X4_runs/x4_execution/code_sha256.txt)]

九条正式 run 均有 25 条 epoch history、无 NaN、机制门 PASS、N=3,016,439、图摘要 `0e80b839…`、vocab `00d304df…`。多次 TIMEOUT/抢占通过相同 RUN_ID 的 `last.pt` 续跑；no-op resume 只用于在 epoch 25 后补写门禁，不应把其短墙钟/低峰值当作完整训练成本。[M]

### 16.2 A 轴：干预效果

旧查询口径、全体候选：

| 指标 | F6 | G | D | GD |
|---|---:|---:|---:|---:|
| `gold@1` | 0.0068 | 0.0177 | 0.0071 | **0.0228** |
| `gold@10` | 0.0598 | 0.1160 | 0.0745 | **0.1429** |
| `top3@10` | 0.1317 | 0.2082 | 0.1521 | **0.2508** |
| `gold-gap@10` | 0.0829 | 0.1448 | 0.1110 | **0.1866** |
| `root_gold@10` | 0.0320 | 0.0511 | 0.0593 | **0.0947** |
| median gold rank | 2,051 | 1,326 | **1,178** | 1,590 |

[M: [`X4_runs/reports/`](X4_runs/reports/)] GD seeds 为 0.1341/0.1440/0.1505，相对各自 F6 增量 +0.0642/+0.1041/+0.0809，满足预注册。GD 均值是 F6 的 2.39×，相对 100K 参照 0.2071 关闭两档差距 56.4%；后者只是口径近似参照，因为监督/查询并非完全相同。[D/U]

G 为 0.1194/0.1075/0.1210，三个 seed 一致上升，均值 0.1160，获得总增量的 67.6%。D 为 0.0546/0.1023/0.0665，增量 −0.0154/+0.0624/−0.0031，方向不一致；D 单独不构成稳定因果证据。GD 比 G 高 23.2%，逐 seed +0.0148/+0.0365/+0.0295，说明扩大 dataset 覆盖在判别性负样本已恢复时提供叠加收益。[M/D]

注意 `median rank` 与 top-10 命中不同步：GD seed 2 的 `gold@10` 0.0696→0.1505，但中位名次 3,363→3,858；改善集中在分布头部。[M]

### 16.3 C 轴：预注册外的几何改善

| 层 | F6 平均余弦 | GD 平均余弦 | F6 有效维度 | GD 有效维度 |
|---|---:|---:|---:|---:|
| warm | 0.402 | 0.218 | 3.6 | 6.6 |
| cool | 0.350 | 0.198 | 3.9 | 7.4 |
| cold | 0.908 | 0.783 | 3.2 | 7.9 |
| frozen | 0.908 | 0.778 | 3.0 | 8.0 |

兄弟—随机分离从 0.0359 到 0.0796；G 为 0.0747，D 为 0.0337。该方向说明训练采样也影响塌缩，但机制解释是后验假设，X4 没有为它设计独立对照；并且有效维度约 8/128，塌缩只是减轻。[M/U]

### 16.4 硬件、续跑与证据纪律

F6 基线在 H200。X4 最终 MANIFEST：GD 三条为 L40S；G 三条 RTX 6000 Ada；D seed 0/2 RTX 6000 Ada、D seed 1 L40S。早期 GD seed 1/2 epoch 0–9 位于 watgpu308 的 `schoolgpu` allocation，但 TIMEOUT 段没有 MANIFEST或 GPU 型号日志；探针某次抽到 RTX A6000，而同节点同 GRES 的 no-op `1522415` 实际记录 L40S，故早期两段型号必须写 **无留档**，不能推断为 A6000。[M/U: [`X4GPU.md`](X4GPU.md) 硬件披露]

`X4D_full_s1_e25` Python 3.11.9，其余八条 3.11.4；全部 PyTorch 2.12.0+cu130。同图、代码、配置和 split 支持 A 轴比较，但不是严格同硬件/同 Python 复现。[M]

`1522415` 在启动时 checkpoint 已到 epoch 25，只执行 no-op；真正把 GD seed 1 从 epoch 10 跑到 24 的是 `1522429`。D seed 2 的 43,030 MiB 是未留档现场观察，不能由交付物复核。GD seed 1 的 no-op MANIFEST 35.6 s/5.252 GB 与 D seed 2 的 15.5 s/5.251 GB均不是完整训练成本；GD seed 1 覆盖前完整记录为 2,329.9 s/37.964 GB。[M/U]

`global_n_datasets=128` 不是精确恢复 100K 触达水平：X1 推算对齐约 44；128 给约 11.7 次触达，是 100K 4.06 的 2.9×。γ=0.5 的 132.5 个带监督 negatives 才是同量级对齐。[D]

## 17. X5：当前主查询口径

X5 的 `--axis e` 对每个表示和 seed重算三行：全部旧查询、`gold_eligible` 查询、仅被排除查询。训练、嵌入、索引不变。[I: [`eval_rf.py`](../../scale1m/eval_rf.py#L1218)]

| seed | 旧查询 | 合格查询 | 排除 | unknown direction | placeholder | RL |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,558 | 1,476 | 82 | 61 | 20 | 1 |
| 1 | 1,153 | 1,101 | 52 | 48 | 4 | 1 |
| 2 | 1,595 | 1,545 | 50 | 45 | 4 | 2 |

seed 1/2 各有一题同时符合两种排除原因，所以原因和比总排除数多 1。[M]

| 表示 | 旧 `gold@10` | X5 合格 `gold@10` | 相对变化 | 排除题 `gold@10` |
|---|---:|---:|---:|---:|
| F6 | 0.0598 | **0.0625** | +4.5% | 0.0000 |
| X4 G | 0.1160 | **0.1197** | +3.2% | 0.0291 |
| X4 D | 0.0745 | **0.0724** | −2.8% | 0.1117 |
| X4 GD | 0.1429 | **0.1427** | −0.1% | 0.1474 |

[M: [`X5_runs/`](X5_runs/)] 口径修正影响小且方向不一，X4 的核心结论不变。F6 的 184 个排除题跨三个 seed一次 top-10 都没命中，旧口径只向分母贡献；X4 GD 对这些不可信标签反而命中较高，但未知方向的 argmax 并不因此变成可信 gold。[M/D]

从现在起，论文主表应以 X5 合格口径为主，同时保留旧口径用于与 F8–X4 历史记录对账。当前 `EXPORT_MANIFEST.json` 与 `F8_REPORT.json` 仍保存旧口径；X5 结果独立存放，未回写导出阶段。[I/U]

X5 只重算全体候选，未重算 `in_snapshot_only`。训练时也未过滤不合格 dataset；若要从训练全局项排除 unknown/RL/placeholder，必须另行重训，现有证据不能回答。[U]

## 18. 当前结果总表：能并列什么，不能合并什么

### 18.1 主协议、全体候选、X5 合格查询

| 表示 | seed 0 | seed 1 | seed 2 | 均值 | 相对 F6 | 证据状态 |
|---|---:|---:|---:|---:|---:|---|
| F6 | 0.0738 | 0.0418 | 0.0718 | 0.0625 | 1.00× | 历史基线 |
| X4 G | 0.1233 | 0.1108 | 0.1249 | 0.1197 | 1.92× | 三 seed 同向 |
| X4 D | 0.0508 | 0.0990 | 0.0673 | 0.0724 | 1.16× | 方向不稳 |
| **X4 GD** | **0.1355** | **0.1417** | **0.1508** | **0.1427** | **2.28×** | 当前主表示 |

这张表是当前论文最安全的 headline 表。[M/D]

### 18.2 独立服务侧证据分支（均基于 F6 表示）

| 分支 | 协议 | `gold@10` | 可说的结论 | 不可说的结论 |
|---|---|---:|---|---|
| X2 task β=1 | root-aware，F6 表示 | 0.3009 | 同任务历史表现是强先验 | X4+task 的最终值 |
| X3 sibling+task(A) | 第二服务协议，F6 表示 | 0.3536 | 已知 benchmark 新配置可用兄弟信息 | 主 root-aware 结果或完整 node-level 重训值 |
| X3 sibling+task(B) | 先验侧上界 | 0.4314 | 上界与信号存在性 | 可部署、无泄漏主结果 |

X2/X3 与 X4 的数值不能相乘、相加或取最大作为“当前系统”。要得到统一最终服务数字，需为九个 X4 导出目录构建 split-specific/full sidecar，并重跑 P/S/E 与第二层可用性。

### 18.3 F 系统性能仍然有效、但不等于 X4 已复测

HNSW `recall@50≈0.999`、全湖 p50 0.0211 ms、2,714× 对全扫、索引 2.21 GiB是 F6 嵌入上的系统测量。X4 没有执行 `index/curve`、B/D 轴；索引算法和 N 未变，但向量几何已变，故不能把 F8 的 recall/延迟直接标为“X4 实测”。[U]

## 19. 产物、路径与复现入口

### 19.1 本地大数据根目录

文档中的 `<DATA>` 对应执行时的 `$MLF_DATA_DIR/data1m`。主要目录契约：

```text
data1m/
├── candidates_full/                    F1: 61 model shards + PROVENANCE/CURSOR/SHARDS
├── datasets_full/                      F1.5: 11 dataset shards + cards
├── rf/canon/                           F2: canon models, supervision, conflicts, nodes
├── ladder_rf/                          F3: model/dataset parquet ladders
├── feats_rf/                           F4: x_m, ids, family vocab, reports
├── graphs/hgraph_rf/                   F5: sharded graph store
├── exports_rf/RF_full_s{0,1,2}_e25/   F7: F6 embeddings and HNSW
├── metrics_rf/                         F8/X1/X2/X3 baseline reports
├── utility_rf/                         F9 model meta, scorecard, t0 list
├── exports_x4/X4{GD,G,D}_full_s*_e25/ X4 embeddings/metrics sidecars
├── metrics_x4/{GD,G,D}/               X4 A/C reports
└── metrics_x5/{F6,GD,G,D}/            X5 eligibility reports
```

仓库内可携带证据在 `docs/1M/*_runs/`；大矩阵与索引不入仓库。关键对象的嵌入哈希绑定见 [`EVIDENCE_SOURCE_MANIFEST.md`](EVIDENCE_SOURCE_MANIFEST.md)。

### 19.2 训练产物

F6 本地回收：[`F6_runs/RF_full_s{0,1,2}_e25/`](F6_runs/)；X4 九条：[`X4_runs/X4{GD,G,D}_full_s{0,1,2}_e25/`](X4_runs/)。每个标准 run 至少包含：

```text
MANIFEST.json
metadata/resolved_config.json
metadata/uncommitted.patch
metrics/train_history.json
stdout/train.log
ckpt/last.pt, best.pt, retained checkpoints, family_vocab.csv
```

X4 远端原始位置为 `/u801/x98liu/model_lake/runs/<RUN_ID>/`，交付包 `/u801/x98liu/x4_<RUN_ID>.tgz`；执行证据下载至 [`X4_runs/x4_execution/`](X4_runs/x4_execution/)。[M: [`X4GPU.md`](X4GPU.md)]

### 19.3 推荐复现顺序

以下是依据当前 CLI 重建的入口顺序；运行前必须固定数据根、HF token（若使用）、Python/PyTorch/PyG 与硬件：

```bash
# F1 / F1.5
python -m scale1m.hf_crawl --sort createdAt --direction -1 --v2-fields --out <DATA>/candidates_full
python -m scale1m.hf_crawl_datasets --out <DATA>/datasets_full

# F2–F5
python -m scale1m.canonicalize_rf --candidates <DATA>/candidates_full --out <DATA>/rf
python -m scale1m.merge_supervision --rf <DATA>/rf
python -m scale1m.build_ladder_rf --rf <DATA>/rf --out <DATA>/ladder_rf
python -m scale1m.embed_lake_rf --ladder <DATA>/ladder_rf/full_model_ids.parquet --out <DATA>/feats_rf
python -m scale1m.build_graph_rf --ladder <DATA>/ladder_rf --feats <DATA>/feats_rf --rf <DATA>/rf --out <DATA>/graphs/hgraph_rf

# F6 baseline / X4 interventions
python -m scale1m.train_rung --rung full --graph <DATA>/graphs/hgraph_rf --out <RUN> \
  --seed 0 --epochs 25 --family-vocab <DATA>/feats_rf/family_vocab.csv \
  --fanout --sparse-M --contrast-n-neg 256 --chunked-infer 50000 --skip-diagnostics
# X4 GD adds: --lake-gamma 0.5 --global-n-datasets 128

# F7 / F8 / F9
python -m scale1m.export_rf --run <RUN> --stage embed --out <EXPORT>
python -m scale1m.export_rf --run <RUN> --stage metrics --out <EXPORT>
python -m scale1m.export_rf --run <RUN> --stage index --out <EXPORT>
python -m scale1m.export_rf --run <RUN> --stage curve --out <EXPORT>
python -m scale1m.eval_rf --axis a --exports <EXPORT_ROOT> --run-fmt '<RUN_FMT>' --out <METRICS>
python -m scale1m.eval_rf --axis b --exports <EXPORT_ROOT> --run-fmt '<RUN_FMT>' --out <METRICS>
python -m scale1m.eval_rf --axis c --exports <EXPORT_ROOT> --run-fmt '<RUN_FMT>' --out <METRICS>
python -m scale1m.eval_rf --axis d --exports <EXPORT_ROOT> --run-fmt '<RUN_FMT>' --out <METRICS>
python -m scale1m.utility_scorecard --stage meta
python -m scale1m.utility_scorecard --stage score

# X1–X5
python -m scale1m.fast_lever_audit --stage all --seeds 0 1 2 --check
python -m scale1m.eval_rf --axis p
python -m scale1m.eval_rf --axis pinf
python -m scale1m.eval_rf --axis s
python -m scale1m.eval_rf --axis e --exports <EXPORT_ROOT> --run-fmt '<RUN_FMT>' --out <METRICS>
```

上述 F3–F5 命令是由当前 argparse 契约重建的可复现入口；不能冒充未归档的原始 shell history。F6/X4 的确切 as-run 命令应以各 `resolved_config.json.metadata.command` 为准。

## 20. 失败、修复与负结果登记

| 阶段 | 失败/风险 | 处理 | 证据意义 |
|---|---|---|---|
| F0/F1 | API 变动与分页跳项 | `createdAt` 降序、不可变 `_id` cursor、0重复/乱序门 | 支持快照枚举完整性，不证明历史时间字段是真实发布日期 |
| F1.5 | basename 猜测可抬覆盖 | 拒绝跨 owner 与歧义，不按下载量解歧 | 用较低覆盖换保守精度 |
| F2 | lower-is-better/unknown/reward 混用会反转 gold | 显式语义表与 `gold_eligible` | X5 才完全统一查询口径 |
| F2 | RL 字符串值不可解析 | 不写专用 parser | RL 不主导不是 cap 的成功证据 |
| F5 | `nan/null` 家族名被 pandas 吃掉 | `keep_default_na=False` | 契约门抓到真实行序风险 |
| F6 | PYTHONPATH、采样后端、matplotlib/环境问题 | 远端探针与脚本修复后再正式跑 | 失败日志是环境证据，不是模型结果 |
| F6 | Git binding 不完整 | 文件哈希 + checkpoint binding | 单一 commit 不足以复现 |
| F7 | 完整图前向会泄漏 | held-out `z_*_eval` + 不等式门 | full z 最多虚高 2.4× |
| F8 | 逐查询“多线程 QPS”不稳定 | 改为批量查询吞吐 | 首版 QPS 废弃 |
| F8/X1 | CSLS seed 0 看似正收益 | 三 seed重测后均值 −41%，删除 | 必须保留的否定结果 |
| X2 | 图中 task id 恒 0 | sidecar 改读真实 task parquet | 服务先验修复，不是训练图修复 |
| X3 | node-level 完整重训会泄漏现有嵌入 | 只改服务侧可读边，并标上下界偏差 | 第二协议不能冒充主协议 |
| X4 | H200 长排队、节点 staging/CUDA/srun 故障、抢占/TIMEOUT | 扩展 GPU 池、探针、`X4_DIRECT=1`、相同 RUN_ID resume | 9 runs 保留；硬件异质必须披露 |
| X4 | no-op resume 覆盖 manifest cost | 单列完整段/现场证据，星号值不作成本 | 防止 5.25GB 假峰值进入论文 |
| X5 | 不可信 query 混入 | 独立 E 轴并列旧/新口径 | 口径修正，不是模型改进 |

## 21. 尚未闭合的证据缺口

以下事项不能在论文中写成“已完成”或“已证明”：

1. **X4 的 P/S/B/D 轴未重跑。** 不能给出 X4+task/sibling 的统一最终指标，也不能把 F8 的 HNSW延迟标为 X4 实测。
2. **X5 未接入导出阶段。** `EXPORT_MANIFEST` 和 `F8_REPORT` 的 A 行仍是旧查询口径；X5 只在独立 JSON 中。
3. **X5 未重算 snapshot-only。** 当前新口径只有全体候选。
4. **训练侧 500K scaling 点未执行。** 只有固定表示的检索候选规模曲线。
5. **F8 计划要求的查询深度 ≥3/5/10/20 分层结果未落在 `F8_REPORT.json`；按监督来源拆分 A 轴也未落盘。** F2 有深度和来源计数，但不是分层性能。
6. **`displacement_quality` 未重写。** 原定义与训练标签重叠。
7. **F9 第三层盲审未做；时间后验只冻结 t0，尚无未来标签结果。**
8. **未标注候选未做真实模型运行。** recorded gold 指标不能转写成 recommendation accuracy。
9. **数据集离散任务特征在训练图中退化。** X2 只修服务 sidecar，F6/X4 编码器仍用单行共享偏置。
10. **数据质量缺口。** 71.929% 模型无 size，79.03% 数据集节点无真实卡片；历史 `curated` 权重缺原始 metric name；12,680 历史模型不在快照。
11. **统计功效有限。** 只有 3 个 split seed；GD 同向但简单符号检验最低只能到 1/8。因果支持依赖事前方向+幅度判据，不是显著性检验。
12. **仓库不是 clean commit。** 当前 HEAD 停在 F1；完整流水线依赖 dirty/untracked 文件。应在论文封存前制作只读 release/tag 与全文件 manifest。
13. **测试入口有命名空间冲突，但套件级测试已独立重跑。** 仓库 `.venv`（Python 3.13.1、pytest 9.1.1）中，`scale1m/tests` 262 passed、`stage2TrainGraphSAGE/tests` 67 passed，合计准确复现 X5 归档的 329 passed；`stage1BuildTransferGraph/tests` 另有 8 passed，`stage3HNSW/tests` 另有 7 passed，总计 344 passed。直接从仓库根一次收集会因多个顶层目录都暴露名为 `tests` 的包而产生 18 个 collection error；stage1 还需把仓库父目录加入 `PYTHONPATH`。这些是收集/导入条件，不是断言失败。本次另验证 92 JSON 可解析、28 个正文引用/核心模块 AST 合法。[M/U]

## 22. 文档冲突与取证裁决

| 冲突 | 较弱证据 | 裁决证据 | 本报告采用 |
|---|---|---|---|
| GraphSAGE 层数 | `1Mplan.md` §2.1、`model.py` 顶部注释称 2 层 | F6 与 9 个 X4 `resolved_config.num_layers=1` | **1 层 as-run** |
| 候选池是否只含快照 | 计划早段称 3,003,759 | F2/F3/F6/F7/X4 产物全部 N=3,016,439 | **快照前缀 + 12,680 历史模型** |
| 历史湖外模型数 | 合并前估计 16,713 | F2 merge `historical_only=12,680` | **12,680** |
| X4 GD s1 真正续跑 job | 旧过程句可能指 1522415 | 日志/更正：1522429 从 epoch10→24；1522415 no-op | **1522429 训练，1522415 门禁** |
| watgpu308 早期 GPU | 探针 1522377 是 RTX A6000 | 两 TIMEOUT 段无型号；同 GRES 的1522415是L40S | **型号无留档** |
| X4 sbatch SHA | 初版 `a0b2ab…` | 正式文件与 code_sha256 `f1237a…` | **f1237a…** |
| D seed2 43,030 MiB | 现场观察 | 交付物无 nvidia-smi 输出 | **未留档观察，不作可复核测量** |
| F9 available 0.7099 | seed 0 单点 | X2 三 seed 0.7099/0.6273/0.5729 | 引用时带区间/注明单点 |
| 当前 `gold@10` | F8 0.0598 或 X4 0.1429 | X5 合格口径 | **F6 0.0625；X4 GD 0.1427** |

## 23. 可直接用于论文的 claim–evidence 库

### 23.1 当前可以安全陈述

1. **规模与可追溯性。** “系统在 3,016,439 个模型、18,729 个数据集节点与 247,803 条六来源监督边上训练；3,003,759 个模型来自 2026-08-18 HF 快照，12,680 个为历史监督追加。”证据：F1/F2/F3 JSON与 ladder/graph bindings。
2. **当前效果。** “在 root-aware held-out、X5 `gold_eligible`、全候选协议下，X4 GD 的三 seed `gold@10` 为 0.1355/0.1417/0.1508，均值 0.1427；F6 为 0.0625，提升 2.28×。”证据：X5 F6/GD JSON。
3. **机制分解。** “G 单独三 seed同向到 0.1197；D 单独 0.0724 且方向不稳；GD 0.1427。因此提议分布是主项，覆盖对其有组合增量。”证据：X4旧口径的逐 seed差值 + X5当前汇总；机制解释应保留‘原因之一’措辞。
4. **候选规模。** “在固定 F6 表示下，候选从 100K 到 3.016M，`gold@10` 只从 0.0706到0.0700。”证据：F8 retrieval curve。必须写“固定表示、监督行固定保留”。
5. **ANN系统。** “F6 表示的全湖 HNSW 在本地 `recall@50=0.9989`、`ef=50` 时 p50 0.0211ms，相对同机全扫2,714×。”证据：F8 B；必须注明本地硬件与 F6 表示。
6. **塌缩及缓解。** “F6 的 cold/frozen 约98.5%且有效维度约3；X4 GD 将 frozen 有效维度提高到8、余弦从0.908降到0.778，但远未达到128维。”证据：F8/X4 C。
7. **先验分支。** “在 F6 表示上，root-aware task先验把 `gold@10` 从0.0598到0.3009，同时参数量中位数2.42B到7.57B；第二协议兄弟+task(A)到0.3536且参数量到12.11B。”证据：X2/X3；必须与 X4 主结果分开。
8. **指标边界。** “`gold@10` 是历史记录恢复而非推荐准确率；F6 seed0 top-10中95.2%在目标 query上无记录。”证据：F9。

### 23.2 当前不能安全陈述

- “最终系统 `gold@10` 是 0.35/0.43”：这些是 F6 表示上的第二协议或上界，不是 X4+X5统一结果。
- “X4 在 HNSW 上仍有 2,714×”：尚未为 X4 建索引/跑 B 轴。
- “未标注推荐中 14.27% 真正最好”：`gold@10` 不提供该语义。
- “系统是两层 GraphSAGE”：as-run 是一层。
- “所有 X4 都在 L40S/A6000/H200 同型卡”：硬件异质，早期两段型号无留档。
- “D 独立有效”：三 seed方向不一致。
- “塌缩已解决”：有效维度仍约8/128。
- “F8 已按来源和查询深度完整分层”：对应性能表未落盘。

## 24. 收尾建议（不改变既有九个 X4 run）

若论文 evidence freeze 只允许做读写产物、不重训，优先级应为：

1. 对 X4 GD/G/D 建 split-specific 与 full sidecar，重跑 P/S/E 和同表第二层；保留 F6 分支作对照。
2. 将 X5 eligibility 接入 export metrics，生成新版本而非覆盖旧 `EXPORT_MANIFEST`；并补 snapshot-only 或明确永久取消该列。
3. 从现有 cands/edge source 补做 F8 的 ≥3/5/10/20 与 source-stratified A 轴；这是评测重算，不需训练。
4. 若论文需要系统结论，为 X4 GD 建全湖 HNSW并按 F8同机协议跑 B/D；不要复用 F8标签。
5. 生成 clean release commit/tag、锁定环境文件与全文件 SHA manifest；保留当前 dirty tree的 tarball作为过渡证据。
6. 若允许新 GPU 实验，再考虑补 seed、D≈44 对齐臂与真正 node-level 重训；它们是新实验，不应改写现有结果。

本文到此形成从原始 API 枚举、规范化监督、特征和图、训练、held-out导出、ANN评测、效用解释，到 X1–X5诊断/干预/口径修正的完整证据链。所有 headline 结论均以 X 系列当前状态为主，F 系列仅作为可追踪基线。
