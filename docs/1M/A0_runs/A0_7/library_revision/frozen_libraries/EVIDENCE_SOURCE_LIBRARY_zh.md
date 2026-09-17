# ModelLakeFishing：三百万规模两阶段检索证据报告

**证据截止时间：** 截止 Y4 最终复跑与系统取舍，2026-09-06（America/Toronto）

**代码仓库：** `D:\research\model_lake\codes\ModelLakeFishing`
**证据清单：** [`EVIDENCE_SOURCE_MANIFEST.md`](EVIDENCE_SOURCE_MANIFEST.md)

本报告按照方法而非实验时间线描述最终系统。核心路径为：**收集并规范化元数据、构建模型--数据集证据图、训练稠密检索器、用 HNSW 召回 1,000 个候选，再以任务级历史证据确定性地重排候选池**。内部运行名称只在定位不可变证据时保留，不作为方法名称。

评测系统先在最终图训练嵌入上用内积 HNSW 召回 1,000 个候选，再用从其他 train/validation 数据集的同任务历史成绩估计出的确定性先验重排小池。冻结的融合口径为 raw `beta=1`、shrink `k=5`，且不做逐查询 min--max 归一化。在 3,016,439 个候选和 1,476 / 1,101 / 1,545 个合格 held-out 查询上，`gold@10` 为 **0.3279 / 0.3388 / 0.2427**（均值 **0.3031**），而全池精确融合为 0.3216。实际系统保留全池融合 94.25% 的分数；HNSW 重排保留 exact top-1000 重排 `gold@10` 的 99.93%。截止 Y4 的重复测量中，检索阶段 HNSW 加重排路径的 p50 / p95 为 0.694 / 1.223 ms。

## 1. 系统形式化

令

$$
\mathcal{M}=\{m_1,\ldots,m_N\},\qquad
\mathcal{D}=\{d_1,\ldots,d_Q\}
$$

分别表示模型湖和数据集--任务查询节点集合。在冻结的全湖产物中，

$$
N=3{,}016{,}439,\qquad Q=18{,}729.
$$

离线管线可以写成如下复合映射：

$$
\text{hub records}
\xrightarrow{\text{canonicalize}}
(X_M,X_D,E)
\xrightarrow{\text{证据图}}
G
\xrightarrow{f_\theta}
(Z_M,Z_D)
\xrightarrow{\text{HNSW}}
\mathcal{I}(Z_M).
$$

其中，$X_M$ 和 $X_D$ 是模型与数据集的元数据特征，$E$ 是带类型的关系证据，$f_\theta$ 是异构 GraphSAGE 编码器，$Z_M,Z_D\in\mathbb{R}^{128}$ 的每一行均经过 L2 归一化。查询是已经物化的数据集--任务节点 $d=(\text{dataset},t)$，第一阶段稠密分数为

$$
s_\theta(d,m)=z_d^\top z_m=\cos(z_d,z_m),
$$

HNSW 产生候选池

$$
\mathcal P_{1000}(d)=
\operatorname*{ANN\text{-}Top1000}_{m\in\mathcal{M}}\;s_\theta(d,m).
$$

对于模型 $m$ 和规范化任务 $t$，令 $n_{tm}$ 与 $A_{tm}$ 分别为 split-specific train+validation 可见边中、来自测试查询 root 之外的同任务有向成绩数量与总和。固定任务先验和第二阶段分数为

$$
p_t(m)=\frac{A_{tm}+0.5\times5}{n_{tm}+5},\qquad
r(d,m)=\frac{s_\theta(d,m)+1}{2}+p_t(m),
$$

没有可见记录时令 $p_t(m)=0$。系统返回 $\mathcal P_{1000}(d)$ 中分数最高的十个模型，精确平局由基于模型行 ID 的固定、无标签排列打破。该重排器是确定性的且不参与学习；它确实是服务阶段，但不是额外的表示学习贡献。

| 流水线阶段 | 具体输出 | 主要实现 |
|---|---|---|
| 收集元数据 | 不可变的模型/数据集快照和规范化记录 | [`hf_crawl.py`](../../scale1m/hf_crawl.py)、[`hf_crawl_datasets.py`](../../scale1m/hf_crawl_datasets.py)、[`canonicalize_rf.py`](../../scale1m/canonicalize_rf.py) |
| 构建证据图 | 冻结的行映射、节点特征和五种有类型边关系 | [`build_ladder_rf.py`](../../scale1m/build_ladder_rf.py)、[`embed_lake_rf.py`](../../scale1m/embed_lake_rf.py)、[`build_graph_rf.py`](../../scale1m/build_graph_rf.py) |
| 训练图表示 | 单位归一化的 128 维模型和数据集嵌入 | [`model.py`](../../stage2TrainGraphSAGE/model.py)、[`losses.py`](../../stage2TrainGraphSAGE/losses.py)、[`train.py`](../../stage2TrainGraphSAGE/train.py)、[`train_rung.py`](../../scale1m/train_rung.py) |
| HNSW 召回 | 与模型嵌入行映射绑定的内积 ANN 索引 | [`eval_y2.py`](../../scale1m/eval_y2.py)、[`export_ours.py`](../../scale/export_ours.py) |
| 重排 1,000 个候选 | split-safe 任务先验、固定融合及无标签平局规则 | [`eval_rf.py`](../../scale1m/eval_rf.py)、[`eval_y2.py`](../../scale1m/eval_y2.py) |

## 2. 离线阶段一：元数据收集与规范化

### 2.1 不可变的模型中心快照

模型收集器按照 `createdAt` 降序枚举 Hugging Face 的 `/api/models` 端点。分页沿用服务器返回的 `Link: ... rel="next"` 游标，而不是自行构造页偏移。每个完成的分片都压缩为 JSONL-GZIP、计算 SHA-256，并登记到来源记录中；游标文件保存下一页 URL 和已经提交的记录数，因此中断后的爬取可以继续执行，而不必重写已完成分片。收集器保留仓库标识、时间戳、任务/库标签、作者、下载数、点赞数、safetensors 元数据、结构化基模型关系，以及模型卡中谱系和评估抽取所需的字段。实现位于 [`scale1m/hf_crawl.py`](../../scale1m/hf_crawl.py)。

冻结的模型抓取使用：

```powershell
python -m scale1m.hf_crawl `
  --sort createdAt --direction -1 --limit 100000000 --v2-fields `
  --out <DATA>/data1m/candidates_full
```

抓取由游标耗尽而非数值上限终止。最终包含 **3,003,759 条唯一模型记录**，由 3,004 个 API 页面写入 61 个分片。归档记录显示跳过的重复项为零，并在 [`F1_runs/PROVENANCE.json`](F1_runs/PROVENANCE.json) 中以哈希绑定每个分片。

数据集元数据通过 `/api/datasets` 独立收集，并采用相同的游标、续跑、分片和哈希规范。保留字段包括仓库标识、任务类别、标签、描述、语言、规模类别、许可证和源数据集元数据。冻结的数据集抓取包含 **1,008,417 个仓库**，写入 11 个分片，并记录在 [`F15_runs/PROVENANCE.json`](F15_runs/PROVENANCE.json) 中。其实现为 [`scale1m/hf_crawl_datasets.py`](../../scale1m/hf_crawl_datasets.py)。

两份快照均以 2026-08-18 为快照日期。枚举过程中观测到的 API `createdAt` 端点是返回记录中的字段，并不构成对 Hugging Face 历史上线日期的判断。

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
\texttt{safetensors.total}/10^9, & \text{若存在},\\
\mathrm{NA}, & \text{否则}.
\end{cases}
$$

若 `config.model_type` 存在，则以它作为规范模型族；否则应用小写名称规则，最后回退为 `other`。对声明的父模型，优先读取 Hugging Face 的结构化字段 `baseModels.ids[0]`；只有结构化关系缺失时才使用自由文本 `cardData.base_model`。这些选择由 [`scale1m/hf_canonicalize.py`](../../scale1m/hf_canonicalize.py) 中的 `size_b_of`、`family_of` 和 `lineage_base_of` 实现，并由 [`scale1m/canonicalize_rf.py`](../../scale1m/canonicalize_rf.py) 调用。

### 2.3 数据集--任务节点标识与数据集卡匹配

查询节点不是单独的数据集名称，其主键为

$$
u_d=(\operatorname{normalize}(\text{dataset}),\operatorname{task}),
$$

内部序列化为 `dataset + "\t" + task`。这样，同一仓库在不同任务上的评估不会被合并为一个排序问题。

数据集卡匹配采取保守策略。规范节点的标准化标识与仓库标识完全相同时才直接获得数据集卡。不带所有者的 basename 只有在数据集快照中唯一时才会被接受；包含 `/` 的名称不会跨所有者按 basename 匹配。对于 `ag_news/default` 这类 dataset/config 名称，匹配器可以使用无歧义的父级数据集卡，并单独记录为 `hf_card_via_parent`。所有无法解析或存在歧义的节点都回退到清洗后的节点名称。算法与审计标签位于 [`scale1m/match_dataset_cards.py`](../../scale1m/match_dataset_cards.py)。

最终节点集合中，18,729 个节点有 3,928 个匹配到精确或父级 Hugging Face 数据集卡。因此，卡片缺失被作为一等状态保留，而不是用基于流行度的猜测填充。覆盖率测量归档在 [`F15_runs/f15b_coverage.json`](F15_runs/f15b_coverage.json)。

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

## 3. 离线阶段二：构建模型--数据集证据图

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

对每个数据集--任务节点，优先选择方向已知的指标，再选择记录数最多的指标；记录数相同时使用确定性的词典序裁决。最终模型--节点对的方向统一值取中位数，作为 `trained_on` 边权。在解析的 2,158,375 条原始指标记录中，2,097,081 条是有限且可解析的；中位数去重后得到 1,435,162 条。原生主指标选择在单节点限额之前生成 143,478 条边，限额之后保留 74,346 条。精确构造由 [`scale1m/canonicalize_rf.py`](../../scale1m/canonicalize_rf.py) 中的 `build_supervision` 实现，计数见 [`F2_runs/F2_REPORT.json`](F2_runs/F2_REPORT.json)。

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

历史来源中存储的权重已经完成方向统一。合并首先在 `(source, node, model)` 内取中位数，再在 `(node, source)` 内执行 min--max 归一化；常数组统一置为 0.5。如果多个来源提供同一个 `(node, model)` 对，则保留最高优先级来源的边，并把被覆盖的记录写入冲突表。最后使用 NumPy seed 0 进行确定性的权重分层采样，把每个节点限制在 200 条边以内。该过程由 [`scale1m/merge_supervision.py`](../../scale1m/merge_supervision.py) 实现。

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

冻结规则文件版本为 `rf-gold-2.0`，SHA-256 为 `be3fb05ecf92ea0112ba4de10b8191d71dfe096b8667a86911c2ebaeadb9cdb1`；合并计数与来源表位于 [`F2_runs/F2_MERGE_REPORT.json`](F2_runs/F2_MERGE_REPORT.json)，规则产物位于 [`F2_runs/rf_gold_rules.json`](F2_runs/rf_gold_rules.json)。

### 3.3 候选闭包与行序契约

监督证据涉及 12,680 个未出现在 2026-08-18 模型中心快照中的模型。系统把这些模型追加到候选空间，而不是丢弃相应观测：

$$
3{,}003{,}759+12{,}680=3{,}016{,}439.
$$

行映射冻结为

$$
\operatorname{mappedID}(m_i)=i,
$$

其中模型中心快照是精确前缀，规范化后的历史独有标识按排序结果追加。历史独有行的模型规模记为未知，模型族取自首个可用的历史来源，并与其他模型一样由同一个编码器重新计算特征，不复制旧特征向量。数据集--任务节点也按确定性顺序排列，并使用同样连续的 `mappedID` 契约。

模型与数据集 ladder 的 SHA-256 分别为 `fee360d1fef9afa4c446b70af35bcbf4ead2c75387f47abc3b0d21c944ae3efa` 和 `31c027ff2eeb693aed6abaa5d235f4a6264bb4b3245bfd71cf1c8b85d32173cb`。[`F3_runs/LADDER_REPORT.json`](F3_runs/LADDER_REPORT.json) 中的八项标识与端点检查全部通过。实现位于 [`scale1m/build_ladder_rf.py`](../../scale1m/build_ladder_rf.py)。

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

参数规模和模型族不嵌入这 448 个维度，而是存为整数列，并在图训练时索引两个可学习的 16 维表。规模桶 0 表示未知；桶 1--14 以半个十进制数量级为间隔，把 $\log_{10}$ 参数量从 $10^5$ 划分至 $10^{12}$。模型族 ID 0 为 `Other`；动态观测到的模型族只有出现至少三次后才获得独立行。

最终 `x_m` 矩阵形状为 `[3,016,439, 448]`，数据类型为 `float32`，文件大小为 5.41 GB，SHA-256 为 `ba1020872ddb6e90726c755f9f08f236f4c73e506013c91d8a0075fc0c0361e6`。仅追加的模型族词表有 41,056 行，SHA-256 为 `00d304df9bd72acb2c2c1ca5b84ed0dc9d4adb0cbc9f098f32a9e62109d24005`。未知规模模型占 71.929%，`Other` 占 13.561%。特征构造与行序检查位于 [`scale1m/embed_lake_rf.py`](../../scale1m/embed_lake_rf.py) 和 [`F4_runs/FEATS_REPORT.json`](F4_runs/FEATS_REPORT.json)。

### 3.5 数据集节点特征

数据集描述符拼接清洗后的数据集--任务名称、数据集卡任务类别、最多十个不含冒号的标签，以及最多 400 个描述字符。如果不存在可信的数据集卡匹配，则只使用清洗后的名称和任务。使用 seed 43 生成名称哈希表后，冻结的数据集向量为

$$
x_d^{\mathrm{frozen}}=
[e_{\mathrm{name}}^{64}(d)\;\Vert\;
 e_{\mathrm{card}}^{384}(d)\;\Vert\;
 e_{\mathrm{stats}}^{10}(d)]
\in\mathbb{R}^{458}.
$$

令 $n_d$ 为节点 $d$ 上保留观测的数量，$\mu_d,\sigma_d,a_d^{\min},a_d^{\max}$ 汇总方向统一后的边值，$r_d$ 为与该节点共享数据集 root 的节点数。实际实现的统计特征为

$$
e_{\mathrm{stats}}(d)=
[\log(1+n_d),\log(1+n_d),\mu_d,\sigma_d,
a_d^{\min},a_d^{\max},\log(1+r_d),
\mathbb{1}_{\mathrm{eligible}}(d),0,0].
$$

冻结实现中的前两个坐标相同，因为重复的模型--节点对已经删除。最后两个位置为禁用的内容视图预留。训练期间，数据集任务类型、类别数桶和元数 ID 还会分别索引 16、8 和 4 维可学习表。在该全湖产物中，这三个表都只有一行，因此它们只保留编码器结构，并不区分节点；任务字符串仍会影响节点标识和文本描述符。特征构建器是 [`scale1m/build_graph_rf.py`](../../scale1m/build_graph_rf.py) 中的 `build_xd`。

### 3.6 有类型的图关系

模型--数据集证据图定义为

$$
G=(V_M\cup V_D,E_{MD}\cup E_{DM}\cup E_{DD}\cup E_{MM}\cup E_{MM}^{-1}),
$$

其中包含以下关系：

1. `model --trained_on--> dataset` 携带归一化性能值 $y_{md}$。
2. `dataset --rev_trained_on--> model` 镜像每一条监督边，用于双向消息传递。
3. `dataset --similar_to--> dataset` 按 384 维数据集卡特征块的余弦相似度，把每个数据集连接到其 20 个最近邻。
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

声明的基模型关系中有 96.083% 成功解析到谱系边。训练前，数据集相似关系被确定性地裁剪为每个源节点最多 10 个邻居，保留边的属性统一置为 1，得到 187,290 条只表示拓扑的相似边。该训练期变换是 [`stage2TrainGraphSAGE/graph_surgery.py`](../../stage2TrainGraphSAGE/graph_surgery.py) 中的 `apply_similar_to_mode(..., mode="topk_unweighted", k=10)`。

图以 memory-mapped `.npy` 特征数组、NPZ 节点/边结构、Parquet 行映射和带哈希的 JSON 元数据存储，从而避免加载时物化 5.41 GB 模型矩阵。分片格式由 [`scale1m/graph_store.py`](../../scale1m/graph_store.py) 实现；构造计数和门禁位于 [`F5_runs/GRAPH_REPORT.json`](F5_runs/GRAPH_REPORT.json)。所有最终训练运行使用的冻结图摘要为 `0e80b8393846dcc4b9e354218fd5139a906d569125d2e8545e17ccd01612b76c`。

## 4. 离线阶段三：结构感知的图训练

### 4.1 Root-aware 监督划分

划分单位是数据集 root，而不是单条边或数据集配置。共享同一 root 的所有节点都会被分到同一侧。对于划分 seed $s\in\{0,1,2\}$，root 只洗牌一次，再按监督边配额贪心分配，直至测试集约占 20%、验证集约占 10%，其余进入训练集。

令 $E_{\mathrm{tr}},E_{\mathrm{val}},E_{\mathrm{te}}$ 表示这些 root 分区诱导出的正边。$E_{\mathrm{tr}}$ 中有 30% 作为不相交的监督集 $E_{\mathrm{sup}}$，并从训练消息图中移除。因此，

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

其中 $W_D:\mathbb{R}^{486}\rightarrow\mathbb{R}^{128}$。冻结特征矩阵不接收梯度；语义查找表、投影层、图层、关系门控和输出头参与学习。编码器装配实现在 [`stage1BuildTransferGraph/dataset_embed/model_node_encoder.py`](../../stage1BuildTransferGraph/dataset_embed/model_node_encoder.py)，并由 [`stage2TrainGraphSAGE/model.py`](../../stage2TrainGraphSAGE/model.py) 中的 `HeteroGraphSAGE` 实例化。

### 4.3 关系特定的 GraphSAGE

最终网络使用一层异构消息传递。对关系 $r$ 和目标节点 $v$，令 $\mathcal{N}_r(v)$ 为其入邻居。关系块计算

$$
\tilde h_{v,r}=
W_{\mathrm{self},r}h_v^{(0)}+
W_{\mathrm{nbr},r}
\left(
\frac{1}{|\mathcal{N}_r(v)|}
\sum_{u\in\mathcal{N}_r(v)}h_u^{(0)}
\right).
$$

各个入关系块通过初始化为 1 的可学习标量门控合并：

$$
h_v^{(1)}=
\sum_{r:\operatorname{dst}(r)=\operatorname{type}(v)}g_r\tilde h_{v,r}.
$$

每种关系都有独立的 self 投影和 neighbor 投影。由于冻结配置设置 `weighted_relations=[]`，连续的 `edge_attr` 值**不参与**消息聚合。性能值仍然监督排序目标和正样本集合目标，数据集卡余弦相似度仍然决定保留哪些数据集邻居拓扑，但消息算子本身是逐关系的无权均值。精确算子由 [`stage2TrainGraphSAGE/edge_aware.py`](../../stage2TrainGraphSAGE/edge_aware.py) 中的 `WeightedSAGEConv` 与 `EdgeAwareHetero` 实现。

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

损失先在单个数据集内部取平均，再跨数据集取平均，因此候选稠密的查询节点不会仅仅因为产生更多偏好对而主导目标。每个数据集最多保留 256 个偏好对；超过上限时，一半取当前排序颠倒最严重的对，另一半从其余对中采样。目标直接作用于检索所用的单位向量原始点积。实现是 [`stage2TrainGraphSAGE/losses.py`](../../stage2TrainGraphSAGE/losses.py) 中的 `raw_dot_ranknet_loss`。

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

该矩阵以 CSR/CSC 成员列表存储，而不是物化稠密的 $N\times Q$ 矩阵。在采样图批次中，正样本对以稀疏方式生成并去重。每个锚点从批内模型中采样 256 个负例。如果某个负例与锚点属于同一谱系分量、但不是正例，则其分母权重为 2；其余负例权重为 1。令 $\phi(i,j)=z_i^\top z_j/0.2$，实际使用的采样目标为

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

成员集合由 `topk_membership` 构造，稀疏偏好对由 `batch_positive_pairs` 生成，损失由 [`losses.py`](../../stage2TrainGraphSAGE/losses.py) 和 [`sampling.py`](../../stage2TrainGraphSAGE/sampling.py) 中的 `contrastive_loss_sampled` 实现。

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

随后在采样的数据集节点之间取平均。由于批次子图不包含完整的负样本空间，该项会在训练消息图上执行一次完整前向传播，并重新采样消息边 dropout。[`stage2TrainGraphSAGE/losses.py`](../../stage2TrainGraphSAGE/losses.py) 中的 `build_lake_logq` 和 `global_lake_loss` 实现了该提议分布与目标函数。

### 4.7 最终优化问题与冻结配置

最终目标为

$$
\boxed{
\mathcal{L}=\mathcal{L}_{\mathrm{rank}}
+\mathcal{L}_{\mathrm{contrast}}
+\mathcal{L}_{\mathrm{global}}
}
$$

三个损失项的系数均为 1。MSE、嵌入均匀性正则、数据集到模型对比损失、困难负例挖掘、正例逆倾向加权、分离投影头和早停均处于禁用状态。

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

三个归档的最终配置分别为 [`seed 0`](X4_runs/X4GD_full_s0_e25/metadata/resolved_config.json)、[`seed 1`](X4_runs/X4GD_full_s1_e25/metadata/resolved_config.json) 和 [`seed 2`](X4_runs/X4GD_full_s2_e25/metadata/resolved_config.json)。每份配置都记录了精确命令行、运行时版本、图摘要、配置、划分 seed 和初始化 seed。三次运行均包含 25 条 epoch 记录，损失为有限值，总损失下降，且机制门禁通过。

### 4.8 Held-out 推理与评估

评估嵌入由 $G_{\mathrm{test}}$ 生成。该图包含训练和验证消息边，但排除了全部测试 `trained_on` 边及其反向副本。导出器分别写出用于服务的全消息嵌入和用于评估的 held-out 消息嵌入：

```text
z_m.npy,      z_d.npy       # 服务图
z_m_eval.npy, z_d_eval.npy  # 泄漏受控的评估图
```

报告结果只使用 `z_m_eval` 和 `z_d_eval`。模型嵌入按 `mappedID` 顺序导出，checkpoint 采用严格加载。分块推理与整图推理在归档数值容差内一致。该路径是 [`scale1m/export_rf.py`](../../scale1m/export_rf.py) 中的 `stage_embed`。

对于合格的 held-out 查询 $d$，令 $C_d$ 为其 held-out 已观测模型集合，并令

$$
g_d=\operatorname*{arg\,max}_{m\in C_d}y_{dm}.
$$

精确全湖排名使用全部 $N$ 个模型嵌入，并以严格大于关系处理并列：

$$
\operatorname{rank}_d(m)
=1+\sum_{j=1}^{N}\mathbb{1}[s(d,m_j)>s(d,m)].
$$

因此，

$$
\operatorname{gold@10}
=\frac{1}{|\mathcal Q_{\mathrm{test}}|}
\sum_{d\in\mathcal Q_{\mathrm{test}}}
\mathbb{1}[\operatorname{rank}_d(g_d)\le 10].
$$

合格查询规则要求：（i）指标方向已知或来源已经人工整理并完成方向统一；（ii）任务不是强化学习；（iii）数据集名称不是占位符；（iv）至少有三个已观测候选。要计算排名，查询还必须具有非常数的 held-out 值。由于划分单位是 root 而不是单个节点，不同 seed 的可评估查询数不同。

评估程序还报告 `gold@1`；`top3@10`，即三个最佳已观测模型中的任意一个命中即可；`gold-gap@10`，即与最佳值之差不超过 0.01 的任意已观测模型命中即可；以及 root-macro `gold@10`。定义和流式精确排名计算位于 [`scale/global_metrics.py`](../../scale/global_metrics.py)，最终合格查询过滤器位于 [`scale1m/eval_rf.py`](../../scale1m/eval_rf.py)。

## 5. 在线检索：HNSW top-1,000 与任务证据重排

### 5.1 索引几何

由于每个嵌入都经过单位归一化，

$$
\operatorname*{arg\,max}_{m}\;z_d^\top z_m
=\operatorname*{arg\,min}_{m}\;(1-z_d^\top z_m)
=\operatorname*{arg\,min}_{m}\;\frac{1}{2}\|z_d-z_m\|_2^2.
$$

因此，内积 HNSW 搜索的正是训练目标优化的几何空间。全湖构建器创建 `hnswlib.Index(space="ip", dim=128)`，把第 $i$ 行以标签 `mappedID=i` 插入，并持久化二进制索引。固定构建参数为

$$
M_{\mathrm{HNSW}}=32,\qquad ef_{\mathrm{construction}}=200.
$$

实现是 [`scale/export_ours.py`](../../scale/export_ours.py) 中的 `build_hnsw`，由 [`scale1m/export_rf.py`](../../scale1m/export_rf.py) 中的 `stage_index` 调用。

### 5.2 无标签召回率校准

在最终深度 $K=1000$ 下，索引保真度以 exact dense top-1,000 的 ID 为参照：

$$
\operatorname{Recall@1000}(ef)=
\frac{1}{|\mathcal Q_H|}
\sum_{d\in\mathcal Q_H}
\frac{|\operatorname{ANN}_{1000}(d;ef)
\cap\operatorname{Exact}_{1000}(d)|}{1000}.
$$

每个划分从 1,000、1,500、2,000、3,000 和 5,000 中选择首个达到 `recall@1000 >= 0.99` 的 `ef_search`。选择只使用稠密近邻 ID 的一致率，从不读取测试 gold 标签。最终参数与召回率为：

| 划分 seed | `ef_search` | `recall@1000` |
|---:|---:|---:|
| 0 | 1,000 | 0.9923 |
| 1 | 1,500 | 0.9941 |
| 2 | 1,500 | 0.9932 |

所有延迟都从预计算好的查询嵌入开始，并在预热后用单个 HNSW 查询线程测量；不计查询编码、索引加载与索引构建时间。[`Y2_REPORT.json`](Y2_runs/Y2_REPORT.json)

### 5.3 Split-safe 重排与产物契约

对于已经物化的数据集--任务节点，实测在线路径为：

```text
dense_pool = hnsw.knn_query(z_q, k=1000)
prior = task_prior[query_task, dense_pool.model_ids]
score = (dense_pool.cosine + 1) / 2 + prior
top10 = stable_topk(score, fixed_label_free_tie_break)
```

三个 split-specific sidecar 只包含 train+validation 边：seed 0 / 1 / 2 分别有 198,216 / 196,912 / 196,124 条可见边。评测器断言 X4 与 sidecar 的模型和数据集行映射逐行完全一致，核验 root 映射，并断言任何可见先验边都不与测试查询共享 root。因此，查询自身及同 root sibling 都不会进入先验。归档报告中的全部映射和泄漏门均通过。[`eval_y2.py`](../../scale1m/eval_y2.py) [`Y2_REPORT.json`](Y2_runs/Y2_REPORT.json)

只有在使用构造索引时的同一个模型行映射下，HNSW 标签才有意义。三个实测评估索引绑定 3,016,439 行 X4-GD held-out 嵌入，位于 `data1m/exports_x4/X4GD_full_s{0,1,2}_e25/hnsw_y2_eval.bin`。对于不在冻结节点表中的真正新数据集，仍需验证仅查询节点的归纳编码路径；当前测量针对 performance 边被 held out 的已物化数据集--任务节点。

### 5.4 最终两阶段结果与代价机制

下表各行使用相同的合格查询、候选全集、表示与冻结重排器。它比较实际的有界候选路径和两条 exact 参照路径，而不是罗列系统的中间阶段。

| 检索路径 | 每次查询需要完成的工作 | seed 0 | seed 1 | seed 2 | `gold@10` 均值 |
|---|---|---:|---:|---:|---:|
| 稠密 + 任务先验，exact 全池 | 对全部 3,016,439 个模型计算稠密分数、读取任务先验并融合两项分数 | 0.3408 | 0.3651 | 0.2589 | 0.3216 |
| 稠密 exact top-1,000 + 任务先验 | 对全部 3,016,439 个模型计算稠密分数并选出 1,000 个，再读取和融合 1,000 项先验 | 0.3286 | 0.3388 | 0.2427 | 0.3034 |
| **稠密 HNSW top-1,000 + 任务先验** | **用 ANN 索引找到 1,000 个候选，再只读取和融合 1,000 项先验** | **0.3279** | **0.3388** | **0.2427** | **0.3031** |

Exact top-1,000 的代价高，是因为它在截断之前仍要逐一比较查询与所有模型嵌入，因此保留了随湖规模线性增长的全池扫描，并失去 HNSW 的亚线性搜索路径。Exact 全池融合不仅要做同样的全模型稠密打分，还要为每个模型读取、融合任务证据并参与排序，而不是只处理 1,000 个候选；其第二阶段处理的候选数约为实际系统的 3,016 倍，并且两条 exact 路径都会随 model lake 扩大而直接增加工作量。实际系统则把任务先验查询与融合限制在有界的 HNSW 候选池内。

最终未舍入值为 0.3279132791、0.3387829246 和 0.2427184466，算术均值为 **0.3031382168**。最终路径在每个 seed 上都超过三百万规模 BM25 基线，三 seed 均值为其 2.98 倍。Exact top-1,000 重排平均保留全池融合的 94.32%，HNSW 平均保留 exact top-1,000 重排的 99.93%，相对全池融合的完整两阶段保留率为 94.25%。稠密 top-1,000 平均包含 51.88% 查询的 held-out gold；全池精确融合 top-10 中有 89.91% 的成员已经位于该候选池。剩余质量损失因此主要来自候选池截断，而非 ANN 近似。[`Y2_REPORT.json`](Y2_runs/Y2_REPORT.json)

| 划分 seed | 查询数 | `gold@1` | `gold@10` | `top3@10` | `gold-gap@10` | root-macro `gold@10` | 已召回时 gold 重排位置中位数 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,476 | 0.1599 | 0.3279 | 0.3984 | 0.3814 | 0.2515 | 6 |
| 1 | 1,101 | 0.1117 | 0.3388 | 0.4269 | 0.3860 | 0.2591 | 7 |
| 2 | 1,545 | 0.1107 | 0.2427 | 0.2958 | 0.2777 | 0.1903 | 5 |
| 三 seed 均值 | — | 0.1274 | **0.3031** | 0.3737 | 0.3484 | 0.2336 | — |

截止 Y4 的最新延迟复测环境为 Windows 11、Intel Family 6 Model 183 CPU、24 个逻辑处理器、单查询/单线程 HNSW，并包含预热；检索阶段延迟如下：

| 划分 seed | HNSW + 重排 p50 | HNSW + 重排 p95 |
|---:|---:|---:|
| 0 | 0.536 ms | 0.905 ms |
| 1 | 0.847 ms | 1.435 ms |
| 2 | 0.699 ms | 1.329 ms |
| 均值 | **0.694 ms** | **1.223 ms** |

HNSW 与重排的 p50 分量均值分别为 0.566 和 0.123 ms。该路径从预计算的 $z_q$ 开始，不计查询编码、索引加载与索引构建。每个索引约占 2.214 GiB，三个索引合计 6.643 GiB；构建时间为 184.6 / 191.3 / 181.6 秒。[`Y2_REPORT.json`](Y2_runs/Y2_REPORT.json) [`Y4_REPORT.json`](Y4_runs/Y4_REPORT.json)

最终绝对 `gold@10` 仍为 0.3031，因此全湖模型检索距离完整恢复历史 gold 仍很远。这是一个可复现运行点，并不表示未观测推荐模型拥有 30.31% 的实际下游准确率。

## 6. 实现入口与归档结果复验

以下命令给出具体的实现顺序，并让 X4-GD 嵌入与 split-specific task sidecar 沿用归档评测中的同一套行映射。尖括号中的路径取决于部署环境；每个 seed 的实际训练命令保存在对应运行的 `metadata/resolved_config.json` 中。

```powershell
# 1. 收集不可变的模型和数据集元数据。
python -m scale1m.hf_crawl `
  --sort createdAt --direction -1 --limit 100000000 --v2-fields `
  --out <DATA>/data1m/candidates_full
python -m scale1m.hf_crawl_datasets `
  --out <DATA>/data1m/datasets_full

# 2. 规范化证据并构建模型--数据集证据图。
python -m scale1m.canonicalize_rf `
  --candidates <DATA>/data1m/candidates_full `
  --out <DATA>/data1m/rf
python -m scale1m.merge_supervision `
  --rf <DATA>/data1m/rf
python -m scale1m.build_ladder_rf `
  --rf <DATA>/data1m/rf --out <DATA>/data1m/ladder_rf
python -m scale1m.match_dataset_cards `
  --nodes <DATA>/data1m/rf/canon/dataset_nodes_merged.parquet `
  --datasets <DATA>/data1m/datasets_full `
  --out <DATA>/data1m/datasets_full/dataset_cards_merged.parquet
python -m scale1m.embed_lake_rf `
  --ladder <DATA>/data1m/ladder_rf/full_model_ids.parquet `
  --out <DATA>/data1m/feats_rf
python -m scale1m.build_graph_rf `
  --ladder <DATA>/data1m/ladder_rf `
  --feats <DATA>/data1m/feats_rf `
  --rf <DATA>/data1m/rf `
  --cards <DATA>/data1m/datasets_full/dataset_cards_merged.parquet `
  --out <DATA>/data1m/graphs/hgraph_rf

# 3. 训练并导出 X4-GD；分别使用 seed 0、1 和 2 重复。
python -m scale1m.train_rung `
  --rung full --graph <DATA>/data1m/graphs/hgraph_rf `
  --out <RUNS>/X4GD_full_s<SEED>_e25 --seed <SEED> --epochs 25 `
  --family-vocab <DATA>/data1m/feats_rf/family_vocab.csv `
  --fanout --sparse-M --contrast-n-neg 256 `
  --chunked-infer 50000 --skip-diagnostics `
  --lake-gamma 0.5 --global-n-datasets 128

python -m scale1m.export_rf `
  --run <RUNS>/X4GD_full_s<SEED>_e25 --stage embed `
  --graph <DATA>/data1m/graphs/hgraph_rf `
  --ladder <DATA>/data1m/ladder_rf/full_model_ids.parquet `
  --out <DATA>/data1m/exports_x4/X4GD_full_s<SEED>_e25
python -m scale1m.export_rf `
  --run <RUNS>/X4GD_full_s<SEED>_e25 --stage metrics `
  --out <DATA>/data1m/exports_x4/X4GD_full_s<SEED>_e25

# 4. 直接在每个 X4-GD 导出旁构建等价的 split-safe task sidecar；每个 seed 重复一次。
python -m stage3HNSW.build_prior_sidecar `
  --graph-store <DATA>/data1m/graphs/hgraph_rf `
  --export <DATA>/data1m/exports_x4/X4GD_full_s<SEED>_e25 `
  --split-seed <SEED> `
  --task-nodes <DATA>/data1m/rf/canon/dataset_nodes_merged.parquet

# 5. 复现 exact 候选池、HNSW 索引与最终 K=1000 报告。
python -m scale1m.eval_y2 --stage exact `
  --exports <DATA>/data1m/exports_x4 --run-fmt "X4GD_full_s%d_e25" `
  --sidecar-exports <DATA>/data1m/exports_x4 --sidecar-run-fmt "X4GD_full_s%d_e25" `
  --dataset-nodes <DATA>/data1m/rf/canon/dataset_nodes_merged.parquet `
  --out <DATA>/data1m/metrics_y2 `
  --device cuda --query-chunk 16 --model-chunk 50000
python -m scale1m.eval_y2 --stage hnsw `
  --exports <DATA>/data1m/exports_x4 --run-fmt "X4GD_full_s%d_e25" `
  --sidecar-exports <DATA>/data1m/exports_x4 --sidecar-run-fmt "X4GD_full_s%d_e25" `
  --dataset-nodes <DATA>/data1m/rf/canon/dataset_nodes_merged.parquet `
  --out <DATA>/data1m/metrics_y2 `
  --device cuda --hnsw-threads 8 `
  --ef-search 1000 1500 2000 3000 5000
python -m scale1m.eval_y2 --stage finalize `
  --exports <DATA>/data1m/exports_x4 --run-fmt "X4GD_full_s%d_e25" `
  --sidecar-exports <DATA>/data1m/exports_x4 --sidecar-run-fmt "X4GD_full_s%d_e25" `
  --dataset-nodes <DATA>/data1m/rf/canon/dataset_nodes_merged.parquet `
  --out <DATA>/data1m/metrics_y2 `
  --hnsw-threads 8 --ef-search 1000 1500 2000 3000 5000

# 可选：复现截止 Y4、冻结检索器的 K 敏感度曲线。
python -m scale1m.eval_y4 --stage exact --device cuda `
  --exports <DATA>/data1m/exports_x4 --run-fmt "X4GD_full_s%d_e25" `
  --sidecar-exports <DATA>/data1m/exports_x4 --sidecar-run-fmt "X4GD_full_s%d_e25" `
  --dataset-nodes <DATA>/data1m/rf/canon/dataset_nodes_merged.parquet `
  --y2-out <DATA>/data1m/metrics_y2 `
  --y2-report <DATA>/data1m/metrics_y2/Y2_REPORT.json `
  --out <DATA>/data1m/metrics_y4
python -m scale1m.eval_y4 --stage hnsw --hnsw-threads 8 `
  --exports <DATA>/data1m/exports_x4 --run-fmt "X4GD_full_s%d_e25" `
  --sidecar-exports <DATA>/data1m/exports_x4 --sidecar-run-fmt "X4GD_full_s%d_e25" `
  --dataset-nodes <DATA>/data1m/rf/canon/dataset_nodes_merged.parquet `
  --y2-out <DATA>/data1m/metrics_y2 `
  --y2-report <DATA>/data1m/metrics_y2/Y2_REPORT.json `
  --out <DATA>/data1m/metrics_y4
python -m scale1m.eval_y4 --stage finalize `
  --y2-out <DATA>/data1m/metrics_y2 `
  --y2-report <DATA>/data1m/metrics_y2/Y2_REPORT.json `
  --out <DATA>/data1m/metrics_y4
```

采集与图构建命令反映当前可执行的 CLI 契约。归档的最终训练命令对实际运行配置提供更强证据，因为其中记录了绝对集群路径、作业元数据和全部解析后的默认值。

## 7. 可复现性映射

| 方法要素 | 代码路径 | 主要证据 |
|---|---|---|
| 游标安全的模型快照 | [`scale1m/hf_crawl.py`](../../scale1m/hf_crawl.py) | [`模型来源记录`](F1_runs/PROVENANCE.json) |
| 游标安全的数据集快照 | [`scale1m/hf_crawl_datasets.py`](../../scale1m/hf_crawl_datasets.py) | [`数据集来源记录`](F15_runs/PROVENANCE.json) |
| 数据集描述与数据集卡匹配 | [`scale1m/dataset_descriptor.py`](../../scale1m/dataset_descriptor.py)、[`scale1m/match_dataset_cards.py`](../../scale1m/match_dataset_cards.py) | [`覆盖率报告`](F15_runs/f15b_coverage.json) |
| 指标方向与方向统一 | [`scale1m/metric_semantics.py`](../../scale1m/metric_semantics.py)、[`scale1m/canonicalize_rf.py`](../../scale1m/canonicalize_rf.py) | [`原生规范化报告`](F2_runs/F2_REPORT.json) |
| 六来源合并与查询规则 | [`scale1m/merge_supervision.py`](../../scale1m/merge_supervision.py) | [`合并报告`](F2_runs/F2_MERGE_REPORT.json)、[`规则文件`](F2_runs/rf_gold_rules.json) |
| 候选与数据集行映射 | [`scale1m/build_ladder_rf.py`](../../scale1m/build_ladder_rf.py) | [`ladder 报告`](F3_runs/LADDER_REPORT.json) |
| 模型元数据特征 | [`scale1m/embed_lake_rf.py`](../../scale1m/embed_lake_rf.py) | [`特征报告`](F4_runs/FEATS_REPORT.json) |
| 证据图构建 | [`scale1m/build_graph_rf.py`](../../scale1m/build_graph_rf.py) | [`图报告`](F5_runs/GRAPH_REPORT.json)、[`谱系报告`](F5_runs/lineage_stats.json) |
| 泄漏受控划分 | [`stage2TrainGraphSAGE/d0_splits.py`](../../stage2TrainGraphSAGE/d0_splits.py) | [`划分审计`](F5_runs/f5_splits.json) |
| 节点编码器与图网络 | [`model_node_encoder.py`](../../stage1BuildTransferGraph/dataset_embed/model_node_encoder.py)、[`model.py`](../../stage2TrainGraphSAGE/model.py)、[`edge_aware.py`](../../stage2TrainGraphSAGE/edge_aware.py) | 上文链接的最终解析配置 |
| 排序、对比与全局目标 | [`stage2TrainGraphSAGE/losses.py`](../../stage2TrainGraphSAGE/losses.py)、[`train.py`](../../stage2TrainGraphSAGE/train.py) | 最终运行 manifest 与训练历史 |
| Held-out 导出与精确稠密排名 | [`scale1m/export_rf.py`](../../scale1m/export_rf.py)、[`scale/global_metrics.py`](../../scale/global_metrics.py) | [`稠密合格查询指标`](X5_runs/X5_GD_ELIGIBILITY.json) |
| 三百万候选基线 | [`scale1m/baselines.py`](../../scale1m/baselines.py)、[`scale1m/eval_x6.py`](../../scale1m/eval_x6.py) | [`免训练基线报告`](X6_runs/X6_BASELINES.training_free.json) |
| Split-safe 任务 sidecar | [`stage3HNSW/build_prior_sidecar.py`](../../stage3HNSW/build_prior_sidecar.py)、[`scale1m/eval_rf.py`](../../scale1m/eval_rf.py) | [`Y2 报告`](Y2_runs/Y2_REPORT.json) |
| HNSW top-1,000 与确定性重排 | [`scale1m/eval_y2.py`](../../scale1m/eval_y2.py)、[`scale/export_ours.py`](../../scale/export_ours.py) | [`Y2 报告`](Y2_runs/Y2_REPORT.json) |
| 最终 K=1,000 检索与计时 | [`scale1m/eval_y4.py`](../../scale1m/eval_y4.py) | [`Y4 报告`](Y4_runs/Y4_REPORT.json) |

## 8. 证据限定的局限性

以下边界是当前最终产物的属性，在由本报告形成的论文中应继续明确说明。

1. **结果依赖元数据与历史观测。** `gold@10` 测试的是已观测 held-out 最佳模型的排名；它不会在新数据集上实际执行检索出的模型，也无法给没有观测的查询--模型对赋予正确性。
2. **数据集卡覆盖较稀疏。** 18,729 个节点中只有 3,928 个匹配到 Hugging Face 数据集卡，其余节点使用名称/任务文本。这同时影响数据集特征与 `similar_to` 拓扑。
3. **模型规模信息稀疏。** 71.929% 的候选使用未知规模桶。模型族覆盖更广，但仍有 13.561% 映射到 `Other`。
4. **该图中的数据集类别表退化。** 任务类型、类别数和元数 ID 各自只有一行词表。数据集--任务字符串仍然进入节点标识和文本，但三个查找表不提供节点之间的区分信息。
5. **最终图编码器感知拓扑，但不感知连续边权。** 代码支持加权关系，但冻结最终配置把加权关系列表设为空。
6. **只测量了三个划分 seed。** 初始化固定为 seed 0，因此该区间描述 root 划分变化，而不是独立初始化的不确定性。
7. **归档 HNSW 测量使用 held-out 评估嵌入。** 三个最终表示评估索引已有测量，但本次证据冻结不包含另行归档的全消息 production index。算法服务路径相同，产物区别仍须明确。
8. **从未出现过的数据集路径尚未针对最终图结构验证。** 最终结果适用于 root-aware held-out 边协议下已经物化的数据集--任务节点，不能证明全新节点的归纳编码路径已经成立。
9. **可复现性同时依赖产物哈希和 Git。** 最终运行记录了仓库 HEAD 及本地改动。主要训练文件的实际运行哈希与当前检查版本一致；图、词表、行映射、checkpoint 和导出结果则分别通过 manifest 绑定。

## 9. 可直接用于论文的方法概述

ModelLakeFishing 首先使用支持游标续跑并以哈希绑定的收集器，从公共模型中心冻结模型和数据集元数据快照。系统规范化仓库标识、模型族、参数规模、模型谱系、数据集--任务标识和异构评估记录。指标值先去重并在可比较组内归一化，再通过显式的越高越好/越低越好词典统一方向；方向未知的观测仍可作为图证据，但不进入 gold 评估。六个评估来源按照确定性的来源优先级合并，并通过权重分层采样限制单节点边数，最终在 3,016,439 个模型节点和 18,729 个数据集--任务节点之间形成 247,803 条模型--数据集监督边。

构建出的模型--数据集证据图包含双向性能边、数据集相似边和双向模型谱系边。模型节点组合 64 维哈希名称向量、384 维 MiniLM 描述向量，以及可学习的规模和模型族嵌入。数据集节点组合 64 维哈希名称向量、384 维数据集卡描述向量、十项观测统计量和结构级类别嵌入。一层关系特定的 GraphSAGE 编码器以独立参数和可学习关系门控聚合各类关系，再通过共享输出头把模型与数据集节点投影到单位归一化的 128 维检索空间。

训练最小化三个与检索分数一致的目标之和：基于方向统一性能差的单数据集 RankNet 损失、以共同入选高性能集合定义正例的模型--模型对比损失，以及经过 logQ 校正的全湖 sampled-softmax 损失。全湖提议分布以相同比例混合平滑度数分布和已标注模型上的均匀分布；每次全局更新采样 128 个查询数据集，并为每个查询采样 256 个模型负例。Root-aware 的训练/验证/测试划分保证每个数据集族只出现在一侧，并从双向消息传递中移除 held-out 边。

检索时，内积 HNSW 先召回 1,000 个稠密候选，再由确定性、split-safe 的任务先验仅在该候选池内按 $r=(\cos+1)/2+p_t(m)$ 重排。三个 root 划分 seed 的最终 `gold@10` 为 0.3279、0.3388 和 0.2427，均值为 0.3031；全池精确融合达到 0.3216，但需要逐项完成全模型稠密打分和全模型先验融合。三个索引的 `recall@1000` 均高于 0.99，截止 Y4 的最新重复测量中，从预计算查询嵌入开始的 HNSW 加重排 p50 / p95 为 0.694 / 1.223 ms。
