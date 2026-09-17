# ModelLakeFishing：最终三百万规模检索系统的 A0 证据库

**证据版本：** A0.1–A0.7，更新于 2026-09-14（America/Toronto）。

**代码仓库：** `D:\research\model_lake\codes\ModelLakeFishing`  
**最终系统：** **X4G+D → HNSW top1000 → task prior → top10**。  
**证据链：** [A0 来源清单](A0_runs/A0_SOURCE_MANIFEST.json)、[A0.7 清单](A0_runs/A0_7/MANIFEST.json)、[A0.7 报告](A0_runs/A0_7/results/A0_REPORT.json)、[指标清单](A0_runs/A0_7/results/A0_METRIC_INVENTORY.json)。

本版描述最终方法及 A0.1–A0.7 产生的结果。A0 保留冻结的候选全集、节点标识、监督数据、root 划分、三个 seed 的训练配方和固定检索配置，将数据集输入中七个性能派生列归零，重新训练三个 25 epoch checkpoint，导出新表示，构建新 HNSW 索引，并独立复算产生的测量结果。两条精确检索路径是同一最终系统的离线诊断参照。

检索与成本复算通过，包括与评测器额外进行的 470 项对照。完整清单包含 982 项：838 项已复算、86 项已核验、3 项已禁用、11 项不适用、42 项无定义、2 项缺失。缺失项为原始模型采集的 API 分页次数和重复记录丢弃次数。这些历史事件缺少完整原始日志，因此总完整性标记仍为 `false`；检索结果已产生并通过验收。[A0.7 验证](A0_runs/A0_7/results/A07_VALIDATION.json)、[A0.7 报告](A0_runs/A0_7/results/A0_REPORT.json)。

在 3,016,439 个候选和 1,476 / 1,101 / 1,545 次合格留出查询观测上，最终 `gold@10` 为 **0.3320 / 0.3370 / 0.2214**（三个 seed 等权均值为 **0.2968**）。全库精确融合达到 0.3174。最终系统相对于全库融合的 `gold@10` 保留率为 93.47%，按三个 seed 各自的比值取平均。正式 Linux 评测中，各 seed 检索 p50 / p95 的均值为 **0.747 / 1.102 ms**；计时从预计算的查询嵌入开始。上述数值均来自 [A0.7 复算报告](A0_runs/A0_7/results/A0_REPORT.json)。

## 1. 系统形式化与计算边界

令 $\mathcal M=\{m_1,\ldots,m_N\}$ 表示候选模型库，$\mathcal D=\{d_1,\ldots,d_Q\}$ 表示冻结的数据集–任务节点表，其中

$$
N=3{,}016{,}439,\qquad Q=18{,}729.
$$

离线阶段，系统规范化模型、数据集、任务、谱系和评估元数据，构建有类型的证据图，应用 A0 特征规则，训练图编码器，导出模型与数据集–任务表示，在模型向量上构建 HNSW，并生成各划分对应的任务先验 sidecar。两类节点均表示为 $\mathbb R^{128}$ 中的单位归一化向量，稠密分数为

$$
s_\theta(d,m)=z_d^\top z_m=\cos(z_d,z_m).
$$

对于已物化的查询节点 $d=(\mathrm{dataset},t)$，HNSW 召回

$$
\mathcal P_{1000}(d)=\operatorname*{ANN\text{-}Top1000}_{m\in\mathcal M}s_\theta(d,m).
$$

令 $n_{tm}$ 和 $A_{tm}$ 分别表示模型 $m$ 在规范任务 $t$ 上方向统一后的性能值数量与总和，统计范围为该划分的训练和验证边。接受计分的测试 root 从这些证据中排除。固定先验与融合公式为

$$
p_t(m)=
\begin{cases}
\dfrac{A_{tm}+0.5\times5}{n_{tm}+5},&n_{tm}>0,\\[5pt]
0,&n_{tm}=0,
\end{cases}
\qquad
r(d,m)=\frac{s_\theta(d,m)+1}{2}+p_t(m).
$$

系统读取 1,000 个候选的先验，返回融合分数最高的十个模型；分数完全相同时，使用固定、与标签无关的模型 ID 排列确定顺序。融合采用 `beta=1`、收缩参数 `k=5` 和上面的原始余弦变换。先验是由历史证据确定性计算得到的。HNSW 避免全库稠密打分，固定候选池将第二阶段的查询与融合限制在 1,000 个模型内，这部分工作量与 $N$ 无关。HNSW 搜索本身的成本仍可能随索引规模和搜索参数变化。[A0 协议](A0_runs/A0_PROTOCOL.json)、[A0 评测器](../../scale1m/a0_evaluation.py)。

## 2. 冻结元数据与规范化

### 2.1 模型中心快照与重新计数范围

A0 复用冻结的模型中心快照字节，并在本地进行审计。采集器使用服务器提供的下一页游标、可恢复的 JSONL-GZIP 分片和 SHA-256 绑定，枚举 Hugging Face 的 `/api/models` 与 `/api/datasets` 接口。模型采集器保留仓库标识、时间戳、任务/库标签、作者、下载量、点赞数、safetensors 元数据、基础模型关系和相关模型卡字段。数据集采集器保留标识、任务类别、标签、描述、语言、规模类别、许可证和来源元数据。采集器实现：[模型](../../scale1m/hf_crawl.py)、[数据集](../../scale1m/hf_crawl_datasets.py)。

| 冻结快照 | 解析记录数 | 分片数 | 原始唯一 ID 数 | strip/lower 后的唯一 ID 数 |
|---|---:|---:|---:|---:|
| 模型 | 3,003,759 | 61 | 3,003,759 | 3,003,759 |
| 数据集 | 1,008,417 | 11 | 1,008,417 | 1,008,416 |

审计确认该大小写冲突在冻结节点/数据集卡表中影响零行，并保持匹配输入原样。快照声明日期为 2026-08-18；这里核验的是冻结元数据中的日期声明，而非独立重建当时的 API 状态。计数、各分片哈希、顺序哈希和冲突追踪见 [A0.1 快照审计](A0_runs/audit/A0_SNAPSHOT_AUDIT.json)。

保留的模型分片中重复标识数为零。采集过程中丢弃了多少重复记录，则属于另一项事件统计。原始 API 分页次数和重复丢弃次数在 [A0.7 来源复算](A0_runs/A0_7/results/A0_SOURCE_COUNTS.json) 中仍标记为 `missing`。

### 2.2 规范模型标识与结构属性

所有连接操作统一采用以下模型键规则：

$$
\operatorname{id}_{\mathrm{norm}}(m)
=\operatorname{lower}(\operatorname{strip}(\operatorname{id}(m))).
$$

规范化标识一旦重复便直接拒绝，而不是静默合并。参数量只接受 `safetensors.total`：

$$
\operatorname{sizeB}(m)=
\begin{cases}
\texttt{safetensors.total}/10^9, & \text{if present},\\
\mathrm{NA}, & \text{otherwise}.
\end{cases}
$$

若 `config.model_type` 存在，则以它作为规范模型族；否则应用小写名称规则，最后回退为 `other`。对声明的父模型，优先读取 Hugging Face 的结构化字段 `baseModels.ids[0]`；只有结构化关系缺失时才使用自由文本 `cardData.base_model`。这些选择由 `size_b_of`、`family_of` 和 `lineage_base_of` 实现，代码位于 [`scale1m/hf_canonicalize.py`](../../scale1m/hf_canonicalize.py)，并由 [`scale1m/canonicalize_rf.py`](../../scale1m/canonicalize_rf.py) 调用。

### 2.3 数据集--任务节点标识与数据集卡匹配

查询节点不是单独的数据集名称，其主键为

$$
u_d=(\operatorname{normalize}(\text{dataset}),\operatorname{task}),
$$

内部序列化为 `dataset + "\t" + task`。这样，同一仓库在不同任务上的评估不会被合并为一个排序问题。

数据集卡匹配采取保守策略。规范节点的标准化标识与仓库标识完全相同时才直接获得数据集卡。不带所有者的 basename 只有在数据集快照中唯一时才会被接受；包含 `/` 的名称不会跨所有者按 basename 匹配。对于 `ag_news/default` 这类 dataset/config 名称，匹配器可以使用无歧义的父级数据集卡，并单独记录为 `hf_card_via_parent`。所有无法解析或存在歧义的节点都回退到清洗后的节点名称。算法与审计标签位于 [`scale1m/match_dataset_cards.py`](../../scale1m/match_dataset_cards.py)。

最终节点集合中，有 3,928 个节点匹配到精确或父级 Hugging Face 数据集卡，总节点数为 18,729。卡片缺失作为明确状态保留，不通过流行度猜测补全。该计数已在 [A0.1 输入审计](A0_runs/audit/A0_INPUT_AUDIT.json) 中重新核验。

### 2.4 评估记录解析与指标语义

解析器从模型卡的每条 `model-index` 记录中抽取

$$
(m,\;\text{dataset},\;\text{task},\;\text{metric},\;v).
$$

布尔值、格式错误值和非有限值会被丢弃。指标名先转为小写并统一分隔符，`@k` 改写为 `_at_k`；相似度前缀和截断后缀只在判断指标方向时移除。完整的规范指标名仍作为分组键，因此 `ndcg_at_1` 与 `ndcg_at_10` 不会被合并。

每个指标属于以下四种方向类别之一：

$$
c(r)\in\{\text{higher},\text{lower},\text{reward},\text{unknown}\}.
$$

已知的 accuracy、F1、NDCG、相关性、重叠度和生成质量指标族属于 `higher`；WER、loss、perplexity 和 error 指标族属于 `lower`。reward 和方向未知的指标仍可作为图证据，但不能定义 gold 模型。显式分类器位于 [`scale1m/metric_semantics.py`](../../scale1m/metric_semantics.py)；未匹配的名称绝不会被猜测性地赋予方向。

## 3. 离线阶段 2：构建模型--数据集证据图

### 3.1 规范监督值

在 Hugging Face 原生来源内部，重复测量按

$$
(m,d,t,r).
$$

分组并取中位数。对每个组 $g=(d,t,r)$，令 $v_g^{\min}$ 和 $v_g^{\max}$ 为组内极值，归一化值为

$$
\bar v_i=
\begin{cases}
\dfrac{v_i-v_g^{\min}}{v_g^{\max}-v_g^{\min}}, & v_g^{\max}>v_g^{\min},\\[6pt]
0.5, & v_g^{\max}=v_g^{\min},
\end{cases}
$$

方向统一后的值为

$$
y_i=
\begin{cases}
1-\bar v_i, & c(r)=\text{lower},\\
\bar v_i, & \text{otherwise}.
\end{cases}
$$

对每个数据集--任务节点，优先选择方向已知的指标，再选择记录数最多的指标；记录数相同时使用确定性的词典序裁决。最终模型--节点对的方向统一值取中位数，作为 `trained_on` 边权。在解析的 2,158,375 条原始指标记录中，2,097,081 条是有限且可解析的；中位数去重后得到 1,435,162 条。原生主指标选择在单节点限额之前生成 143,478 条边，限额之后保留 74,346 条。精确构造由 `build_supervision` 实现，代码位于 [`scale1m/canonicalize_rf.py`](../../scale1m/canonicalize_rf.py)，计数由 [A0.7 来源复算](A0_runs/A0_7/results/A0_SOURCE_COUNTS.json) 独立重新生成。

### 3.2 六来源证据合并

原生观测与五个历史评估图合并。固定的冲突优先级为

$$
\text{modellens\_v2}
\succ \text{d0\_v1\_5}
\succ \text{a\_ctrl\_2000m}
\succ \text{hf\_effective}
\succ \text{diverse\_zoo}
\succ \text{hf\_model\_index}.
$$

历史来源中存储的权重已经完成方向统一。合并首先在 `(source, node, model)` 内取中位数，再在 `(node, source)` 内执行 min--max 归一化；常数组统一置为 0.5。如果多个来源提供同一个 `(node, model)` 对，则保留最高优先级来源的边，并把被覆盖的记录写入冲突表。最后将每个节点限制在 200 条边以内，采用 NumPy seed 0 的确定性权重分层采样。该过程由 [`scale1m/merge_supervision.py`](../../scale1m/merge_supervision.py) 实现。

合并从 531,958 条记录开始，删除 5,102 条来源内重复记录，登记 1,502 个跨来源冲突，并在限额前保留 525,354 个不同的模型--节点对。最终模型--数据集证据图含有 **247,803 条有向监督边**，各来源保留数量如下：

| 来源 | 保留边数 |
|---|---:|
| `modellens_v2` | 117,898 |
| `hf_model_index` | 73,670 |
| `d0_v1_5` | 45,992 |
| `hf_effective` | 5,223 |
| `a_ctrl_2000m` | 3,670 |
| `diverse_zoo` | 1,350 |
| **合计** | **247,803** |

规则文件 `rf-gold-2.0` 的 SHA-256 为 `be3fb05ecf92ea0112ba4de10b8191d71dfe096b8667a86911c2ebaeadb9cdb1`，由 [A0.1 输入审计](A0_runs/audit/A0_INPUT_AUDIT.json) 绑定。A0.7 重建合并后的监督表、节点表和冲突表，经稳定的语义排序后逐行逐列完全一致，数值比较未设置容差。合并复算和精确输出标识检查见 [A0.7 来源复算](A0_runs/A0_7/results/A0_SOURCE_COUNTS.json)。这些名称表示源数据，并非其他接受评测的检索系统。

### 3.3 候选闭包与行序契约

监督证据涉及 12,680 个未出现在 2026-08-18 模型中心快照中的模型。系统把这些模型追加到候选空间，而不是丢弃相应观测：

$$
3{,}003{,}759+12{,}680=3{,}016{,}439.
$$

行映射冻结为

$$
\operatorname{mappedID}(m_i)=i,
$$

其中模型中心快照构成精确前缀，规范化后的仅历史来源标识按排序后的顺序追加。在冻结构造中，仅历史来源行的模型规模为未知，模型族取自第一个可用历史来源，并与其他行使用相同的描述生成器和特征编码器。A0 复用该已核验模型特征矩阵。数据集--任务节点按确定性规则排序，并遵循相同的连续 `mappedID` 契约。

模型与数据集 ladder 的 SHA-256 分别为 `fee360d1fef9afa4c446b70af35bcbf4ead2c75387f47abc3b0d21c944ae3efa` 和 `31c027ff2eeb693aed6abaa5d235f4a6264bb4b3245bfd71cf1c8b85d32173cb`。行映射、快照前缀闭包和边端点由 [A0.1 输入审计](A0_runs/audit/A0_INPUT_AUDIT.json) 核验。实现位于 [`scale1m/build_ladder_rf.py`](../../scale1m/build_ladder_rf.py)。

### 3.4 模型节点特征

对于仓库名称 $n$，令 $T(n)$ 包含其小写完整名称、仓库 basename，以及按 `/`、`_`、`-` 和空白符切分得到的 token，并按原顺序去重。令 $R\in\mathbb{R}^{10000\times64}$ 为使用 seed 42 生成且逐行归一化的高斯表，并定义

$$
h(t)=\operatorname{MD5}(t)\bmod 10000.
$$

冻结的名称表示为

$$
e_{\mathrm{name}}(n)=\frac{1}{|T(n)|}\sum_{t\in T(n)}R_{h(t)}\in\mathbb{R}^{64}.
$$

模型文本描述符包含清洗后的仓库标识、规范模型族，以及 safetensors 参数量已知时的规模短语。描述符由 `all-MiniLM-L6-v2` 编码，输出不做归一化：

$$
e_{\mathrm{desc}}(m)=\operatorname{MiniLM}(\operatorname{descriptor}(m))
\in\mathbb{R}^{384}.
$$

因此，冻结模型矩阵为

$$
x_m^{\mathrm{frozen}}=
[e_{\mathrm{name}}(m)\;\Vert\;e_{\mathrm{desc}}(m)]
\in\mathbb{R}^{448}.
$$

模型族和已知参数规模已出现在描述文本中；它们的离散模型族 ID 与规模桶 ID 还会在图训练期间选择可学习的 16 维查找表。规模桶 0 表示未知。对于已知参数量 $P$，桶 1--14 以宽度为 0.5 的区间划分 $\log_{10}(P)$，范围为 5 到 12，超出范围的值归入两端的桶。模型族 ID 0 为 `Other`，动态观测到的模型族至少出现三次后才获得独立行。

最终 `x_m` 矩阵形状为 `[3,016,439, 448]`，数据类型为 `float32`，文件大小为 5.41 GB，SHA-256 为 `ba1020872ddb6e90726c755f9f08f236f4c73e506013c91d8a0075fc0c0361e6`。仅追加的 `feats_rf/family_vocab.csv` 词表有 41,056 行，SHA-256 为 `00d304df9bd72acb2c2c1ca5b84ed0dc9d4adb0cbc9f098f32a9e62109d24005`。未知规模模型占 71.929%，`Other` 占 13.561%。41,056 行表示嵌入表索引空间大小，图中实际出现 40,927 个不同模型族 ID。特征构造位于 [`scale1m/embed_lake_rf.py`](../../scale1m/embed_lake_rf.py)；当前形状、基数和覆盖率测量见 [A0.7 来源复算](A0_runs/A0_7/results/A0_SOURCE_COUNTS.json)，冻结文件哈希见 [A0.1 输入审计](A0_runs/audit/A0_INPUT_AUDIT.json)。

### 3.5 数据集节点特征

数据集描述符拼接清洗后的数据集--任务名称、数据集卡任务类别、最多十个不含冒号的标签，以及最多 400 个描述字符。如果不存在可信的数据集卡匹配，则只使用清洗后的名称和任务。使用 seed 43 生成名称哈希表后，冻结的数据集向量为

$$
x_d^{\mathrm{frozen}}=
[e_{\mathrm{name}}^{64}(d)\;\Vert\;
 e_{\mathrm{card}}^{384}(d)\;\Vert\;
 e_{\mathrm{stats}}^{10}(d)]
\in\mathbb{R}^{458}.
$$

在 A0 中，这个十槽位块仅包含一项节点表结构计数。设 $r_d$ 为冻结节点表中与节点 $d$ 具有同一 root 的数据集–任务节点数量，则

$$
e_{\mathrm{stats}}^{A0}(d)=
[0,0,0,0,0,0,\log(1+r_d),0,0,0].
$$

全部 18,729 个节点的列 448–453 和 455 均被精确设为零；本文所有列号均为从零开始的索引。这七个位置原来分别存放两项观测计数、性能均值/标准差/最小值/最大值和 `gold_eligible`。A0 保留 458 维结构：列 454 保留由结构性 root 节点计数得到的 $\log(1+r_d)$，列 456–457 继续保留为零，其余 451 列保持冻结值。因此，该修复移除了这些性能派生输入，同时保留文本块、行映射、图拓扑、划分标识和编码器维度。资格标记继续作为学习输入之外的评估查询筛选规则。修复后的图摘要为

```text
acddeb93d926efcb636c83d452fe360736a0e6e4ac68cab3f5fef210e7cef5db
```

数据集任务类型、类别数桶和 arity ID 还会分别选择维度为 16、8、4 的可学习查找表。本产物中每张表均只有一行，因此离散 ID 对所有节点均为常量。任务字符串仍影响节点标识和 MiniLM 描述。任务先验另行根据规范任务字符串确定任务分组。[A0.2 修复与验证](A0.2.md)、[特征准备](../../scale1m/prepare_a0_graph.py)、[特征构建器](../../scale1m/build_graph_rf.py)、[A0.7 来源复算](A0_runs/A0_7/results/A0_SOURCE_COUNTS.json)。

### 3.6 有类型的图关系

模型--数据集证据图定义为

$$
G=(V_M\cup V_D,E_{MD}\cup E_{DM}\cup E_{DD}\cup E_{MM}\cup E_{MM}^{-1}),
$$

其中包含以下关系：

1. `model --trained_on--> dataset` 携带归一化性能值 $y_{md}$。
2. `dataset --rev_trained_on--> model` 镜像每一条监督边，用于双向消息传递。
3. `dataset --similar_to--> dataset` 将每个数据集连接到其 20 个最近邻，邻居按 384 维数据集卡特征块的余弦相似度确定。
4. `model --is_base_of--> model` 从解析出的父模型指向其派生模型。
5. `model --rev_is_base_of--> model` 镜像模型谱系关系。

数据集相似度为

$$
w_{ij}^{DD}=\frac{e_{\mathrm{card}}(d_i)^\top e_{\mathrm{card}}(d_j)}
{\|e_{\mathrm{card}}(d_i)\|_2\|e_{\mathrm{card}}(d_j)\|_2},
$$

选择最大的 20 个值之前会排除自环。存储图中的边数如下：

| 关系 | 存储图中的有向边数 |
|---|---:|
| `trained_on` | 247,803 |
| `rev_trained_on` | 247,803 |
| `similar_to` | 374,580 |
| `is_base_of` | 859,065 |
| `rev_is_base_of` | 859,065 |
| **合计** | **2,588,316** |

A0 输入审计核验了存储的谱系端点及反向关系。训练前，数据集相似关系按确定性规则缩减为每个源节点的前 10 个邻居，并将保留的边属性设为一，得到 187,290 条仅表达拓扑的相似边。此训练期转换为 `apply_similar_to_mode(..., mode="topk_unweighted", k=10)`，位于 [`stage2TrainGraphSAGE/graph_surgery.py`](../../stage2TrainGraphSAGE/graph_surgery.py)。

图采用 memory-mapped `.npy` 特征数组、NPZ 节点/边结构、Parquet 行映射和带哈希的 JSON 元数据存储。加载器可映射 5.41 GB 的模型矩阵；随后训练会复制图数据并将其传至 GPU。分片格式由 [`scale1m/graph_store.py`](../../scale1m/graph_store.py) 实现。上表计数已在 [A0.1 输入审计](A0_runs/audit/A0_INPUT_AUDIT.json) 中核验；A0.2 保留这些数据的字节，仅替换七个输入列，并绑定 §3.5 给出的修复图摘要。全规模资源需求实测见 [A0.3](A0.3.md) 和 [A0.4](A0.4.md)。

## 4. 离线阶段 3：结构感知的图训练

### 4.1 Root-aware 监督划分

划分单位是数据集 root，而不是单条边或数据集配置。共享同一 root 的所有节点都会被分到同一侧。对于划分 seed $s\in\{0,1,2\}$，root 只洗牌一次，再按监督边配额贪心分配，直至测试集约占 20%、验证集约占 10%，其余进入训练集。

令 $E_{\mathrm{tr}},E_{\mathrm{val}},E_{\mathrm{te}}$ 表示这些 root 分区诱导出的正边。$E_{\mathrm{tr}}$ 中的百分之三十作为不相交的监督集 $E_{\mathrm{sup}}$，并从训练消息图中移除。因此，

$$
\begin{aligned}
G_{\mathrm{train}} &: E_{\mathrm{tr}}\setminus E_{\mathrm{sup}},\\
G_{\mathrm{val}} &: E_{\mathrm{tr}},\\
G_{\mathrm{test}} &: E_{\mathrm{tr}}\cup E_{\mathrm{val}}.
\end{aligned}
$$

删除每条正向边时，也会删除对应的 `rev_trained_on` 边。二元负例按 1:1 比例从完整正边集合中不存在的模型--数据集对采样；性能排序目标本身只使用正边，并从完整图读取其方向统一值。划分实现与泄漏断言位于 [`stage2TrainGraphSAGE/d0_splits.py`](../../stage2TrainGraphSAGE/d0_splits.py)。

### 4.2 节点编码器

模型侧消息传递输入为

$$
h_m^{(0)}=W_M
[x_m^{\mathrm{frozen}}\;\Vert\;
 E_{\mathrm{size}}[b_m]\;\Vert\;
 E_{\mathrm{family}}[f_m]]+b_M,
$$

其中 $x_m^{\mathrm{frozen}}\in\mathbb{R}^{448}$，两个可学习查找向量的维度均为 16，$W_M:\mathbb{R}^{480}\rightarrow\mathbb{R}^{128}$。系统不使用模型 ID embedding。

数据集侧输入为

$$
h_d^{(0)}=W_D
[x_d^{\mathrm{frozen}}\;\Vert\;
E_{\mathrm{task}}[t_d]\;\Vert\;
E_{\mathrm{class}}[c_d]\;\Vert\;
E_{\mathrm{arity}}[a_d]]+b_D,
$$

其中 $W_D:\mathbb{R}^{486}\rightarrow\mathbb{R}^{128}$。冻结特征矩阵不接收梯度；语义查找表、投影层、图层、关系门控和输出头参与学习。编码器装配实现在 [`stage1BuildTransferGraph/dataset_embed/model_node_encoder.py`](../../stage1BuildTransferGraph/dataset_embed/model_node_encoder.py)，并由 `HeteroGraphSAGE` 实例化，代码位于 [`stage2TrainGraphSAGE/model.py`](../../stage2TrainGraphSAGE/model.py)。

### 4.3 关系特定的 GraphSAGE

最终网络使用一层异构消息传递。对关系 $r$ 和目标节点 $v$，令 $\mathcal{N}_r(v)$ 为其入邻居。关系块计算

$$
\tilde h_{v,r}=
W_{\mathrm{self},r}h_v^{(0)}+b_{\mathrm{self},r}+
W_{\mathrm{nbr},r}
\left(
\frac{1}{|\mathcal{N}_r(v)|}
\sum_{u\in\mathcal{N}_r(v)}h_u^{(0)}
\right).
$$

各个入关系块通过初始化为一的可学习标量门控合并：

$$
h_v^{(1)}=
\sum_{r:\operatorname{dst}(r)=\operatorname{type}(v)}g_r\tilde h_{v,r}.
$$

每种关系都有独立的自身投影与邻居投影；自身投影含偏置。单个节点的邻域为空时，邻居均值取零；某一关系在整图上为空时，跳过该关系张量。冻结配置为 `weighted_relations=[]`，因此消息聚合中**不使用**连续 `edge_attr` 值。性能值仍用于监督排序及正例集合相关目标，数据集卡余弦仍决定保留哪些邻居拓扑；消息算子本身对每种关系使用无权均值。该精确算子由 `WeightedSAGEConv` 与 `EdgeAwareHetero` 实现，代码位于 [`stage2TrainGraphSAGE/edge_aware.py`](../../stage2TrainGraphSAGE/edge_aware.py)。

共享线性输出头把两类节点映射到同一检索空间：

$$
z_v=\frac{W_Oh_v^{(1)}+b_O}
{\|W_Oh_v^{(1)}+b_O\|_2},\qquad z_v\in\mathbb{R}^{128}.
$$

共享 $W_O$ 并逐行归一化，使模型节点和数据集节点处于同一个内积空间。冻结配置不使用分离的输出头。

### 4.4 与排序一致的局部目标

对于查询节点 $d$，令 $y_{dm}$ 为方向统一后的监督值，并定义偏好对

$$
\mathcal{P}_d=
\{(m_i,m_j):y_{di}-y_{dj}>0.02\}.
$$

局部 RankNet 项为

$$
\mathcal{L}_{\mathrm{rank}}=
\frac{1}{|\mathcal{D}_B|}
\sum_{d\in\mathcal{D}_B}
\frac{1}{|\mathcal{P}_d|}
\sum_{(i,j)\in\mathcal{P}_d}
\operatorname{softplus}
\left(-\frac{s_\theta(d,m_i)-s_\theta(d,m_j)}{0.1}\right).
$$

均值先在数据集内部计算，再在数据集之间计算，因此观测密集的查询节点不会仅因产生更多配对而占据更大权重。公式中，$\mathcal D_B$ 包含至少保留一对偏好配对的数据集；需要采样时，$\mathcal P_d$ 表示保留的配对子集。每个数据集最多保留 256 对；超过上限时，一半选取当前排序颠倒最严重的配对，另一半从剩余配对中采样。该目标直接作用于检索使用的原始单位向量点积，实现为 `raw_dot_ranknet_loss`，位于 [`stage2TrainGraphSAGE/losses.py`](../../stage2TrainGraphSAGE/losses.py)。

### 4.5 模型--模型对比目标

训练可见观测定义稀疏的高性能模型成员矩阵：

$$
A_{md}=\mathbb{1}
\left[m\in\operatorname{Top}_{\max(1,\operatorname{round}(0.1n_d))}
\{y_{dm'}\}_{m'}\right].
$$

如果模型 $i$ 和 $j$ 至少在一个共同数据集上同时进入高性能集合，则它们构成正样本对：

$$
(i,j)\in\mathcal{P}^{MM}
\iff i\ne j\;\land\;\sum_d A_{id}A_{jd}>0.
$$

该矩阵存储为 CSR/CSC 成员列表，而非稠密的 $N\times Q$ 块。在采样图批次中，以稀疏方式生成正例对并去重。对每个锚点，从批次模型 ID 中均匀有放回采样 256 个 ID，将自身 ID 和采样正例从负例项中移除。若某负例与锚点处于同一谱系连通分量、但并非正例，其分母权重为二；其他情况权重为一。令 $\phi(i,j)=z_i^\top z_j/0.2$，实现的采样目标为

$$
\mathcal{L}_{\mathrm{contrast}}=
\frac{1}{|\mathcal A|}\sum_{i\in\mathcal A}
\left[
\log\left(
\sum_{p\in\mathcal P_i}e^{\phi(i,p)}+
\sum_{n\in\mathcal N_i}w_{in}e^{\phi(i,n)}
\right)
-\frac{1}{|\mathcal P_i|}\sum_{p\in\mathcal P_i}\phi(i,p)
\right].
$$

成员集合由 `topk_membership` 构造，稀疏正例对由 `batch_positive_pairs` 生成，损失由 `contrastive_loss_sampled` 实现；代码位于 [`losses.py`](../../stage2TrainGraphSAGE/losses.py) 和 [`sampling.py`](../../stage2TrainGraphSAGE/sampling.py)。

### 4.6 全湖 sampled-softmax 目标

全局项让每个查询方向接触从完整候选模型湖抽取的负例。令

$$
\deg(m)=|\{d:(m,d)\in E_{\mathrm{tr}}\}|
$$

表示训练可见的监督度数。平滑度数提议与已标注模型均匀提议分别为

$$
q_{\deg}(m)=
\frac{(\deg(m)+1)^{0.75}}
{\sum_{m'}(\deg(m')+1)^{0.75}},
\qquad
q_{\mathrm{lab}}(m)=
\frac{\mathbb{1}[\deg(m)>0]}
{|\{m':\deg(m')>0\}|}.
$$

最终提议分布使用固定混合：

$$
q(m)=0.5q_{\deg}(m)+0.5q_{\mathrm{lab}}(m).
$$

每个训练 step 采样 128 个含有正例的数据集节点。对每个选中的节点 $d$，正样本集合 $P_d$ 是上述 top-10% 成员集合；从 $q$ 中有放回地采样 256 个负例，并移除抽中的正例。令 $T_g=0.1$，实际使用的 logQ 校正项为

$$
\mathcal{L}_{\mathrm{global}}(d)=
\frac{1}{|P_d|}\sum_{p\in P_d}
\left[
\log\left(
\sum_{p'\in P_d}e^{s(d,p')/T_g}
+\sum_{n\in S_d}e^{s(d,n)/T_g-\log q(n)}
\right)
-\frac{s(d,p)}{T_g}
\right],
$$

随后在采样的数据集节点之间取平均。由于批次子图不包含完整的负样本空间，该项会在训练消息图上执行一次完整前向传播，并重新采样消息边 dropout。`build_lake_logq` 和 `global_lake_loss` 实现了该提议分布与目标函数，代码位于 [`stage2TrainGraphSAGE/losses.py`](../../stage2TrainGraphSAGE/losses.py)。

### 4.7 最终优化问题与冻结配置

最终目标为

$$
\boxed{
\mathcal{L}=\mathcal{L}_{\mathrm{rank}}
+\mathcal{L}_{\mathrm{contrast}}
+\mathcal{L}_{\mathrm{global}}
}
$$

其中各项系数均为一。MSE、嵌入均匀性正则、数据集到模型的对比损失、全局目标中的可选挖掘负例分支、正例逆倾向加权、独立投影头和早停均被禁用。前文描述的局部 RankNet 困难配对选择和基于谱系的对比负例加权仍保持启用。

| 组件 | 冻结值 |
|---|---:|
| GraphSAGE 深度 / 隐藏宽度 / 输出宽度 | 1 / 128 / 128 |
| 优化器 / 学习率 | Adam / 0.01 |
| Epoch 数 | 25 |
| 监督批大小 | 1,024 |
| 邻居展开 | fallback loader 中使用两个 fan-out hop；普通关系每跳 10 个邻居，谱系关系每跳 20 个邻居 |
| 普通 / 谱系消息边 dropout | 0.30 / 0.05 |
| 排序温度 / 最小目标差 | 0.10 / 0.02 |
| 对比温度 / 采样负例数 | 0.20 / 256 |
| 全局温度 / 采样负例数 | 0.10 / 256 |
| 提议指数 / 平滑项 / 混合系数 | 0.75 / 1.0 / 0.50 |
| 每次全局 step 的数据集节点数 | 128 |
| 相似数据集拓扑 | top 10、无权 |
| 划分 seed / 初始化 seed | 0、1、2 / 0 |
| 导出推理分块 | 50,000 个节点 |

Mini-batch 以互不相交的训练监督边为根。fallback 采样器以无放回方式展开当前 frontier，并对每个节点施加严格的 fan-out 上限。边 dropout 只改变消息边；监督标签单独存储，绝不会被 dropout。完整训练循环位于 [`stage2TrainGraphSAGE/train.py`](../../stage2TrainGraphSAGE/train.py)，面向规模的采样器位于 [`stage2TrainGraphSAGE/sampling.py`](../../stage2TrainGraphSAGE/sampling.py)。

A0 实际配置分别为 [`seed 0`](A0_runs/A0_4/delivery_s0/metadata/resolved_config.json)、[`seed 1`](A0_runs/A0_4/delivery_s1/metadata/resolved_config.json) 和 [`seed 2`](A0_runs/A0_4/delivery_s2/metadata/resolved_config.json)。每份配置记录命令、运行时、修复图摘要、40 个有效配置键、划分 seed 和初始化 seed。各运行均完成 25 个 epoch，并通过机制门禁。A0 的三个导出均使用零基 epoch 24 对应的 `ckpt/last.pt`。这些均为重新训练的运行，每次初始化 seed 均为 0。[A0.4 完成记录与哈希](A0.4.md)。

### 4.8 留出导出与受测信息边界

导出器从每个新的末轮 checkpoint 写出两对 128 维 float32 数组：

```text
z_m.npy,      z_d.npy       # full-message graph
z_m_eval.npy, z_d_eval.npy  # training+validation performance edges only
```

报告中的索引与查询嵌入只使用 `z_m_eval.npy` 和 `z_d_eval.npy`。对应图移除了所有测试性能边及其反向边。完整消息图数组作为独立导出产物保留，其存在本身并不能证明已有另行测量的完整消息图生产索引。

A0.5 检查所有导出行均为有限值、L2 范数误差低于 `1e-5`，检查分块推理与整图推理的差异在 `1e-5` 以内，并核验模型/数据集行序。新导出的查询 ID、root、候选 ID 和方向统一值与全部 4,122 次冻结划分观测一致。[A0.5 验证](A0.5.md)、[导出器](../../scale1m/export_rf.py)。

A0.3 还在固定测试样例上扰动了测试 root 的性能值和资格输入。七列修复后，在固定 RNG 下，特征、训练使用的监督、全库采样分布、训练后参数、训练历史和留出前向输出均保持一致。这是对修复信息边界的针对性测试；冻结的节点总体和元数据拓扑仍共同定义当前评估设置。[A0.3 信息边界测试与全规模 smoke](A0.3.md)、[测试源文件](../../scale1m/tests/test_a03_information_boundary.py)。

## 5. 评估与最终检索结果

### 5.1 查询标识、标签与指标定义

接受评估的查询为冻结图中已存在的数据集–任务节点。共享同一 root 的节点被分到同一监督划分。合格查询须满足以下条件：指标方向已知或来自已整理且方向统一的来源，任务属于非强化学习任务，数据集名称不是占位符，至少有三个已观测的留出候选，且方向统一值不全相同。三个 seed 分别包含 1,476 / 1,101 / 1,545 次查询观测，以及 730 / 495 / 473 个不同查询 root。总查询数 4,122 表示跨划分的观测次数；同一节点可能在不同 seed 中重复出现。[冻结查询标识](A0_runs/audit/A0_QUERY_IDENTITY.jsonl)、[A0.1 输入审计](A0_runs/audit/A0_INPUT_AUDIT.json)。

对查询 $d$，gold 模型 $g_d$ 为方向统一值最大的已观测留出候选；该标签存在并列值时，按冻结的候选顺序和 `argmax` 选定。令 $R_{10}(d)$ 表示实际返回的十个模型 ID，则

$$
\mathrm{gold@10}=\frac{1}{|\mathcal Q_s|}\sum_{d\in\mathcal Q_s}
\mathbb 1[g_d\in R_{10}(d)].
$$

`gold@1` 检查第一个返回结果；`top3@10` 接受冻结计分器选出的历史前三个候选中的任意一个；`gold-gap@10` 接受方向统一值与最优值相差 0.01 以内的任意已观测候选。Root-macro 指标先在每个 root 内平均查询指示值，再对 root 取平均。主结果的三 seed 均值对各 seed 等权。所有比值均无量纲；以百分数呈现时显式标注。

对分数函数 $a_d(m)$ 和固定且唯一的同分排序键 $\tau(m)$，全库排名为

$$
\operatorname{rank}_d(m)=1+\sum_{j=1}^{N}
\mathbb 1\!\left[a_d(m_j)>a_d(m)\ \lor\
\big(a_d(m_j)=a_d(m)\land\tau(m_j)<\tau(m)\big)\right].
$$

精确诊断排名使用上述顺序。A0 将保存的全库精确融合 top10 对齐到同一同分顺序，并从与全库扫描形状相同的 float32 分块 GEMM 中读取探针分数。解释与早期结果的变化时，需要考虑这些 A0.2 一致性修正。对于限定为 1,000 个候选的输出，可以在 gold 进入候选池的条件下计算其重排位置；该有限输出的全库 gold 排名及其归一化/随机参照变体均无定义。[A0.2 精确计算一致性修正](A0.2.md)、[独立计分器](../../scale1m/recompute_a0.py)。

### 5.2 HNSW 构建与无标签校准

对单位向量，内积降序等价于 $1-z_d^\top z_m$ 升序，也等价于平方欧氏距离升序。A0 使用 `hnswlib.Index(space="ip", dim=128)`，设置 `M=32`、`ef_construction=200`，并采用八个构建线程；模型行 $i$ 以标签 `mappedID=i` 插入。三个文件为 `data1m/a0_20260912/metrics/hnsw_a0_s{0,1,2}.bin`。实际运行的构建器为 [A0 评测器](../../scale1m/a0_evaluation.py) 中的 `run_hnsw`。

对每个划分，校准将 HNSW 返回的 1,000 个 ID 与同一查询的精确稠密 top1,000 比较：

$$
\operatorname{Recall@1000}(ef)=\frac{1}{|\mathcal Q_s|}
\sum_{d\in\mathcal Q_s}\frac{|\operatorname{ANN}_{1000}(d;ef)\cap
\operatorname{Exact}_{1000}(d)|}{1000}.
$$

冻结的搜索网格为 1,000、1,500、2,000、3,000、5,000。选择平均召回率至少达到 0.99 的第一个值；按停止规则，后续值标记为不适用。校准使用已物化评估查询的邻居 ID 一致性，过程不读取它们的 gold 标签。这衡量的是 ANN 在该查询集合上的近似保真度。

| 划分 seed | 选定 ef_search | Recall@1000 |
|---|---|---|
| 0 | 1,000 | 0.99438144 |
| 1 | 1,500 | 0.99626067 |
| 2 | 1,000 | 0.99282006 |

### 5.3 任务先验证据与在线工作量

三个 sidecar 分别包含 198,216 / 196,912 / 196,124 条可见训练+验证边，对应 seed 0 / 1 / 2。它们从规范 `task` 列生成 2,198 个任务组，处理方式为转小写，并将空白/下划线替换为连字符。该任务映射与数据集编码器中为常量的 `task_type_id` 字段有区别。各可见记录按任务和模型分组，再按 §1 的公式收缩估计；若不存在可见模型–任务记录，返回零。

评测器核验模型/数据集行映射完全一致、root 对应关系、划分标识，以及可见先验边中不存在任何接受计分的测试 root。因此，查询节点及其同 root 的其他节点均被排除。A0.5 独立计算记录总和与数量，并将实际先验读取器的输出与公式核对。[A0.5 验证](A0.5.md)、[sidecar 生成器](../../stage3HNSW/build_prior_sidecar.py)、[先验读取器](../../scale1m/eval_y2.py)。

```text
z_q = stored_dataset_task_embedding[query_id]
ids, cosine = hnsw_top1000(z_q)
prior = task_prior[query_task, ids]
fused = (cosine + 1) / 2 + prior
return ids[order_by(descending=fused, ascending=fixed_tie_key)[:10]]
```

### 5.4 最终质量与精确诊断参照

下表各行使用相同的新 GD 表示、候选全集、留出查询和固定先验/融合。HNSW 行为最终系统，精确参照行衡量候选截断和 ANN 近似带来的成本与质量变化。

| 路径 | Seed 0 gold@10 | Seed 1 gold@10 | Seed 2 gold@10 | 等权均值 |
|---|---|---|---|---|
| 全库精确融合（离线参照） | 0.3604 | 0.3542 | 0.2375 | 0.3174 |
| 精确稠密 top1000 + prior（离线参照） | 0.3327 | 0.3370 | 0.2214 | 0.2970 |
| **HNSW top1000 + prior（最终）** | 0.3320 | 0.3370 | 0.2214 | 0.2968 |

| 划分 seed | gold@1 | gold@10 | top3@10 | gold-gap@10 | Root-macro gold@10 | gold 被召回时的位置中位数 |
|---|---|---|---|---|---|---|
| 0 | 0.1585 | 0.3320 | 0.3740 | 0.3550 | 0.2515 | 4 |
| 1 | 0.1217 | 0.3370 | 0.4096 | 0.3660 | 0.2566 | 5 |
| 2 | 0.1023 | 0.2214 | 0.2628 | 0.2311 | 0.2471 | 4 |
| 等权均值 | 0.1275 | 0.2968 | 0.3488 | 0.3174 | 0.2517 | — |

最终未舍入的 `gold@10` 分别为 0.3319783197831978 / 0.3369663941871026 / 0.22135922330097088。这些指标衡量对历史已观测排名的恢复；本次检索评估并未实际执行各候选模型。

| 诊断指标 | Seed 0 | Seed 1 | Seed 2 | 逐 seed 比值的均值 |
|---|---|---|---|---|
| 精确候选池保留率 | 92.2932% | 95.1282% | 93.1880% | 93.5365% |
| ANN 保留率 | 99.7963% | 100.0000% | 100.0000% | 99.9321% |
| 总保留率 | 92.1053% | 95.1282% | 93.1880% | 93.4738% |
| 精确 top1000 的 gold 覆盖率 | 53.8618% | 57.4024% | 37.0874% | 49.4505% |
| 全库融合 top10 在精确 top1000 内的覆盖率 | 88.1504% | 90.7084% | 92.1165% | 90.3251% |
| 实际 HNSW top1000 的 gold 覆盖率 | 53.9295% | 57.4024% | 37.0874% | 49.4731% |

精确候选池保留率为 `exact1000 / exact_full_lake`，ANN 保留率为 `hnsw1000 / exact1000`，总保留率为 `hnsw1000 / exact_full_lake`；均先使用各 seed 内的 `gold@10` 相除，再对比值取平均。精确候选池的 gold 覆盖率表示 gold ID 进入稠密候选池的查询占比。全库融合 top10 覆盖率先计算每个查询的全库精确融合 top10 ID 中已有多少进入该候选池，再对比例取平均。这些诊断量有助于区分候选池截断损失和 ANN 误差。§5 全部表格来源：[A0.7 复算报告](A0_runs/A0_7/results/A0_REPORT.json)、[完整清单](A0_runs/A0_7/results/A0_METRIC_INVENTORY.json)、[独立验证](A0_runs/A0_7/results/A07_VALIDATION.json)。

### 5.5 检索延迟、索引存储与离线成本实测

A0.6 正式评测在 `watgpu308` 上运行，环境为 Linux、Python 3.11.4、NumPy 2.4.6、PyTorch 2.12.0+cu130 和 hnswlib 0.8.0。精确向量计算使用该节点的 NVIDIA L40S；HNSW 构建/搜索及先验融合在主机 CPU 上执行。每个 seed 预热至多 50 个查询后，使用一个 HNSW 查询线程完成一次查询计时遍历。计时从预计算查询嵌入开始，覆盖 HNSW、候选先验读取、融合与排序。查询编码、索引/先验表加载、索引构建和客户端/网络开销均在测量区间之外。本机 RTX 4060/i7 复核及后续本地独立复算分别保存执行记录。[A0.6 报告与运行时绑定](A0_runs/A0_6/delivery/metrics/A0_EVALUATION_MANIFEST.json)、[A0.6 执行记录](A0.6.md)。

| Seed | HNSW p50 ms | HNSW p95 ms | 先验/融合 p50 ms | 先验/融合 p95 ms | 合计 p50 ms | 合计 p95 ms |
|---|---|---|---|---|---|---|
| 0 | 0.512913 | 0.804156 | 0.178343 | 0.229241 | 0.693406 | 1.014304 |
| 1 | 0.716782 | 1.148937 | 0.169012 | 0.209267 | 0.882341 | 1.337841 |
| 2 | 0.536364 | 0.754903 | 0.124475 | 0.208460 | 0.666437 | 0.953115 |
| 均值 | 0.588686 | 0.902665 | 0.157277 | 0.215656 | 0.747395 | 1.101754 |

每个合计样本是同一查询的 HNSW 与重排耗时之和。分位数从原始逐查询纳秒数组重新计算；均值行对三个 seed 各自的分位数取平均，不等于合并查询后的分位数，也不等于各分量分位数之和。

| Seed | 索引字节数 | 索引 GiB | 构建秒数 |
|---|---|---|---|
| 0 | 2,377,803,152 | 2.214502 | 196.151983 |
| 1 | 2,377,803,548 | 2.214502 | 208.427164 |
| 2 | 2,377,803,284 | 2.214502 | 217.314888 |

索引构建时间包含初始化、插入、序列化和最终文件替换。索引总存储量为 7,133,409,984 字节，即 6.643506 GiB。这里 GB 表示 $10^9$ 字节，GiB 表示 $2^{30}$ 字节。

| 记录区间（秒） | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| 训练区间 | 3875.623108 | 3911.995431 | 3569.225913 |
| 导出区间 | 36.909913 | 39.057960 | 40.113739 |
| 先验构建 | 7.548296 | 7.497223 | 7.522241 |
| 共享精确参照扫描 | 8.752328 | 6.686751 | 7.996606 |
| 记录的 exact + HNSW 评测区间 | 210.592887 | 220.728926 | 231.004163 |

训练耗时取记录中的 `train_eval_one` 区间，包含设置、训练/checkpoint 工作和既有的训练后评估。导出耗时取包括检查在内的导出区间，区别于内部仅嵌入生成的计时。先验耗时包括其输入核验。两条精确参照共享同一次扫描，因此使用同一计时区间。各 seed 评测耗时将 exact 和 HNSW 阶段区间各计一次，其中已包括索引构建；全局绑定检查与最终汇总在这些区间之外。因此，第二个精确参照标签和索引构建时间均不应再次相加。这些实测区间也不包含排队及失败的环境设置尝试。A0.4 训练使用 `watgpu308`/L40S，A0.5 导出使用 `watgpu608`/RTX 6000 Ada，A0.6 评测使用 `watgpu308`/L40S 加 CPU。[A0.4](A0.4.md)、[A0.5](A0.5.md)、[A0.6](A0.6.md)、[A0.7 来源描述](A0_runs/A0_7/results/A0_RECORD_SOURCES.json)。

| 记录的资源用量（GiB） | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| 训练进程 RSS 历史峰值 | 32.686615 | 32.641811 | 32.884975 |
| 训练 GPU 已分配内存峰值 | 37.947710 | 37.964684 | 37.931339 |
| 评测进程 RSS 历史峰值 | 5.952076 | 5.991192 | 6.071606 |
| 评测 GPU 已分配内存峰值 | 2.026332 | 2.026332 | 2.026332 |

GPU 数值为 PyTorch 已分配内存峰值。RSS 数值为操作系统记录的进程历史峰值，可能包含同一进程先前工作的分配量，不能解释为各 seed 独立新增的工作集需求。完整逐 epoch 损失、来源描述、资源数值和生成记录保存在 [A0.7 报告](A0_runs/A0_7/results/A0_REPORT.json) 与[清单](A0_runs/A0_7/results/A0_METRIC_INVENTORY.json) 中。

## 6. 复现与不可变证据

### 6.1 A0 阶段与输入

| 阶段 | 目的 | A0 证据 |
|---|---|---|
| A0.1 | 冻结协议、原始输入和查询标识 | [协议](A0_runs/A0_PROTOCOL.json)、[输入审计](A0_runs/audit/A0_INPUT_AUDIT.json)、[快照审计](A0_runs/audit/A0_SNAPSHOT_AUDIT.json) |
| A0.2 | 完成七列修复并核验科学设置保持一致 | [执行记录](A0.2.md)、[生成器与评测器改动](A0_runs/A0_2/A0_2_MANIFEST.json) |
| A0.3 | 执行回归、信息边界测试和全规模 smoke | [执行记录](A0.3.md)、[回归 JUnit](A0_runs/A0_3/regression.xml) |
| A0.4 | 重新训练三个 25 epoch GD checkpoint | [执行记录](A0.4.md)、[源码预检](A0_runs/A0_4/A04_PREFLIGHT.json)、[seed 0](A0_runs/A0_4/delivery_s0/MANIFEST.json)、[seed 1](A0_runs/A0_4/delivery_s1/MANIFEST.json)、[seed 2](A0_runs/A0_4/delivery_s2/MANIFEST.json) |
| A0.5 | 导出完整/评估向量、gold 候选、行映射和先验 | [执行记录](A0.5.md)、[验证与文件哈希](A0_runs/A0_5/STATUS.json) |
| A0.6 | 计算精确参照，构建/校准 HNSW，测量最终检索 | [执行记录](A0.6.md)、[原始产物清单](A0_runs/A0_6/delivery/metrics/A0_EVALUATION_MANIFEST.json)、[评测器报告](A0_runs/A0_6/delivery/metrics/A0_EVALUATION_REPORT.json) |
| A0.7 | 独立复算原始指标、成本和来源计数 | [执行记录](A0.7.md)、[报告](A0_runs/A0_7/results/A0_REPORT.json)、[清单](A0_runs/A0_7/results/A0_METRIC_INVENTORY.json)、[验证](A0_runs/A0_7/results/A07_VALIDATION.json) |

A0.1 协议保留原始英文权威文档哈希 `6c7d883734fb9a05ca99c4a8409e366955c86ba2e644bd5328c9ba696ff344cd`。该冻结文档是本轮运行的不可变输入；本修订证据库为其后续版本，归档原件字节继续保留。原始权威文档位于 [A0 冻结来源](A0_runs/frozen/EVIDENCE_SOURCE_LIBRARY_en.md)。对于隔离的远端 checkout，实际 A0 训练元数据记录 `git_head=not-a-git-repo` 和 `git_status=not-a-git-repo`。因此，源码文件 SHA 绑定、图/词表/行映射哈希、checkpoint 哈希和导出/评测清单共同构成复现链。

| Seed | 运行名称 | 末轮 checkpoint SHA-256 |
|---|---|---|
| 0 | `A0GD_full_s0_e25` | `9ee3415caf361e0ed50126c0d7992eb0989284282ebee8ed097addc6375650fb` |
| 1 | `A0GD_full_s1_e25` | `576db197c5d1429341a0b7d0fb35b98bbdcddda80b127a7414ceaba927123994` |
| 2 | `A0GD_full_s2_e25` | `9a090d58fd889b730299af226135fbb6727de5998a9fc133b9eaeb58138eb6f8` |

### 6.2 实际计算顺序与命令记录

完整实际命令、环境设置和重试记录见 [A0.4](A0.4.md)、[A0.5 运行脚本](A0_runs/A0_5/run.sbatch)、[A0.6 运行脚本](A0_runs/A0_6/run.sbatch) 和 [A0.7 执行状态](A0_runs/A0_7/STATUS.json)。下方科学计算命令主体使用实际远端目录布局；重放需要冻结源码、已验证输入和空输出目录。已完成输出继续作为绑定证据保留。新的重放应使用单独记录的输出位置，并显式记录路径迁移。

```bash
cd /u801/x98liu/model_lake/a0_formal_20260913/ModelLakeFishing
DATA=/u801/x98liu/model_lake/data1m
RUNS=/u801/x98liu/model_lake/runs/A0_20260912
GRAPH="$DATA/a0_20260912/graph"
EXPORTS="$DATA/a0_20260912/exports"
METRICS="$DATA/a0_20260912/metrics"
OPS=/u801/x98liu/model_lake/a0_eval_20260914
```

修复图由 `scale1m.prepare_a0_graph` 从已验证的冻结图准备一次，并按 §3.5 的摘要验收；该过程复用冻结输入，不重新抓取当前模型中心数据或重新计算 MiniLM 文本特征。精确准备/验证调用见 [A0.2](A0.2.md)。

对每个 `SEED=0`、`1`、`2`，A0.4 使用相同配置训练：

```bash
python -m scale1m.train_rung \
  --rung full --graph "$GRAPH" \
  --out "$RUNS/A0GD_full_s${SEED}_e25" --seed "$SEED" --epochs 25 \
  --family-vocab "$DATA/feats_rf/family_vocab.csv" \
  --fanout --sparse-M --contrast-n-neg 256 \
  --chunked-infer 50000 --skip-diagnostics \
  --lake-gamma 0.5 --global-n-datasets 128
```

三个运行全部通过后，A0.5 导出各运行并构建其先验：

```bash
python -m scale1m.export_rf \
  --run "$RUNS/A0GD_full_s${SEED}_e25" --stage embed --ckpt last \
  --graph "$GRAPH" --ladder "$DATA/ladder_rf/full_model_ids.parquet" --chunk 50000 \
  --out "$EXPORTS/A0GD_full_s${SEED}_e25"
python -m stage3HNSW.build_prior_sidecar \
  --graph-store "$GRAPH" --export "$EXPORTS/A0GD_full_s${SEED}_e25" \
  --split-seed "$SEED" \
  --task-nodes "$DATA/rf/canon/dataset_nodes_merged.parquet"
```

A0.6 依次执行 `exact`、`hnsw`、`finalize`，每个阶段内部处理全部三个 seed：

```bash
for STAGE in exact hnsw finalize; do
  python -m scale1m.eval_y2 --protocol a0 --stage "$STAGE" \
    --exports "$EXPORTS" --run-fmt 'A0GD_full_s%d_e25' \
    --sidecar-exports "$EXPORTS" --sidecar-run-fmt 'A0GD_full_s%d_e25' \
    --dataset-nodes "$DATA/rf/canon/dataset_nodes_merged.parquet" \
    --out "$METRICS" --device cuda --query-chunk 16 --model-chunk 50000 \
    --hnsw-M 32 --ef-construction 200 --hnsw-threads 8 \
    --ef-search 1000 1500 2000 3000 5000 \
    --a0-protocol-file "$OPS/inputs/A0_PROTOCOL.linux.json" \
    --a0-query-identity "$OPS/inputs/audit/A0_QUERY_IDENTITY.jsonl"
done
```

Linux 协议对冻结路径字段做了已审计的迁移，科学数值和查询标识保持固定；[迁移记录](A0_runs/A0_6/PATH_RELOCATION.json) 绑定这些变化。A0.7 通过 [run.py](A0_runs/A0_7/run.py) 在本地运行，原始 [Windows 路径别名](A0_runs/A0_7/PATH_ALIASES.json) 记录初次执行的映射。证据库修订另将正式源码和权威文档字节保存在 `A0_runs/A0_7/library_revision/frozen_checkout/ModelLakeFishing` 下，当前不可变映射见[重放路径映射](A0_runs/A0_7/library_revision/REPLAY_PATH_MAP.json)。这些映射解析生成器路径，同时保留原始产物字节。来源复算和独立读取器分别为 `scale1m.a0_source_counts` 与 `scale1m.recompute_a0`；[STATUS.json](A0_runs/A0_7/STATUS.json) 记录执行参数。最终来源描述见 [A0_RECORD_SOURCES.json](A0_runs/A0_7/results/A0_RECORD_SOURCES.json)。独立报告读取器在修正报告时单独版本化，冻结的正式训练/评测代码继续保留。

### 6.3 独立复算边界

A0.7 从已绑定的候选/数值数组推导 gold 标签，独立重建候选先验融合和完整同分顺序，核验最终 top10 成员、命中指示值、root 平均、条件位置、保留率和 ANN 重合率。它从成对的查询纳秒数组重新计算分位数，读取实际索引字节及生成器的计时/资源记录。全库排名汇总使用保存的原始整数全模型比较计数，并与独立计分的全库融合 top10 核对；A0.7 不重复执行完整稠密 GPU 扫描。来源处理另行从冻结记录重新计数，重建的规范表必须逐行逐列一致。[独立实现](../../scale1m/recompute_a0.py)、[来源复算实现](../../scale1m/a0_source_counts.py)、[A0.7 验证](A0_runs/A0_7/results/A07_VALIDATION.json)。

证据库审计修正了初版 A0.7 清单中的报告单位不一致：六项检索比值的单位声明为百分数，存储值却是小数比例。科学计算的小数比例保持不变；以百分数计量的清单条目现将这些比例乘以 100，新旧差值也使用一致单位。原始数组、已训练模型和正式评测仍绑定原始字节。读取器改动及验证记录见 [UNIT_CORRECTION.md](A0_runs/A0_7/library_revision/UNIT_CORRECTION.md)。

## 7. 评估范围与剩余局限

本结果支持对已物化数据集–任务节点进行历史排名恢复，其测试性能边在两个方向均被留出。A0 将所有节点的七个已识别性能派生输入列归零，并验证修复后的边界。节点成员、文本元数据和数据集相似/谱系拓扑仍冻结且可见，因此，这是性能边留出条件下的传导式图评估。

- **已观测排名目标。** `gold@10` 衡量冻结方向统一标签下的最优已观测留出模型是否被恢复。实际下游模型执行，以及未观测模型–查询组合的性能，需要单独证据。
- **新数据集接入。** 冻结节点表之外的数据集需要经过评估的编码与插入流程。当前测量从已物化的数据集–任务向量开始。
- **元数据覆盖。** 匹配的数据集卡覆盖 3,928 个节点，总节点数为 18,729，其余节点使用名称/任务文本。未知模型规模占候选的 71.929%，`Other` 模型族占 13.561%。数据集任务类型/类别数/arity 嵌入 ID 为常量。这些均是已测量的输入属性。
- **关系证据范围。** 图编码器在 `weighted_relations=[]` 下使用无权关系均值；方向统一性能值用于监督损失与任务先验，相似值用于选择拓扑。
- **变化来源与校准。** 三个 root 划分 seed 共享初始化 seed 0。结果离散程度反映这些划分运行的变化，而非独立初始化实验。ANN 校准使用评估查询向量和稠密 ID 一致性。每个 seed 的一次计时遍历刻画本硬件与运行设置。
- **产物区别。** 实测 HNSW 索引绑定留出评估嵌入。完整消息图向量导出独立保留，本版未给出另行测量的生产索引结果。
- **来源完整性。** 原始模型采集的 API 分页次数与重复丢弃事件数仍缺失。A0.7 清单无效项为零，全部 seed 齐全，但由于这两项计数缺失，全项完整性标记仍为 false。
- **历史比较边界。** A0 特征修复以及 A0 精确同分/GEMM 一致性修正均使本轮区别于早期测量；延迟的硬件条件也发生了变化。本版最终结果采用新验证的 A0 数值。A0 归档的新旧对照表保留早期文档的公开精度，供历史比较参考：[A0_COMPARISONS.csv](A0_runs/A0_7/results/A0_COMPARISONS.csv)。

## 8. 可直接用于论文的方法概述

ModelLakeFishing 将冻结的模型中心元数据和评估记录规范化为有类型的图，包含 3,016,439 个模型节点、18,729 个数据集–任务节点、247,803 条性能边及其反向关系，以及数据集相似边和双向谱系边。模型特征拼接 64 维名称哈希与 384 维 MiniLM 描述，可学习的规模/模型族查找表在投影前进一步扩展输入。数据集特征使用相同宽度的文本块和十个固定结构槽位；A0 将七个性能特征列归零后，仅经对数变换的结构性 root 节点计数保持有效。可学习的结构类别表和按节点类型区分的线性投影，将输入送入一层关系特定的 GraphSAGE 编码器和共享输出头，生成单位归一化的 128 维检索向量。

训练结合数据集内部 RankNet、采样模型–模型对比学习和全库 logQ 校正 sampled softmax。Root-aware 划分将每个数据集 root 整体分到同一侧，评估消息图在两个方向移除测试性能边。对于已物化的数据集–任务查询，HNSW 选出 1,000 个稠密候选，来自训练/验证 root 的固定任务先验按 $r=(\cos+1)/2+p_t(m)$ 对该候选池重排，系统返回十个 ID。A0 重跑得到的 `gold@10` 为 0.3320 / 0.3370 / 0.2214（均值 0.2968），各 seed 的 ANN ID 召回率均高于 0.99。正式 Linux 主机上，从预计算查询嵌入开始的各 seed 检索 p50/p95 均值为 0.747/1.102 ms。这些测量支持在冻结节点表内恢复留出的历史排名。[A0.7 复算报告](A0_runs/A0_7/results/A0_REPORT.json)。
