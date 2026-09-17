# ModelLakeFishing Methodology 写作地图

> 状态：写作前梳理稿，不是论文正文。
>
> 证据截止：2026-09-06（截止最终 Y4 复跑与系统取舍）。
>
> 主要内部事实基准：[`EVIDENCE_SOURCE_LIBRARY_zh.md`](1M/EVIDENCE_SOURCE_LIBRARY_zh.md) 与 [`EVIDENCE_SOURCE_LIBRARY_en.md`](1M/EVIDENCE_SOURCE_LIBRARY_en.md)。两份文件均为 799 行，章节、公式、表格、链接目标和事实口径互为镜像。本稿不把早期实验计划、旧架构描述或运行代号当作最终方法定义。只在两项 validity blocker 上额外核验了证据库直接链接的最终实现，核验位置在下方明确列出。

> **成稿前的最高优先级阻断项：** 定向实现核验已经确认，数据集特征中的 $e_{\mathrm{stats}}$ 由完整 `supervision_merged.parquet` 一次性计算，root-aware split 只克隆图并替换 performance 边集合，held-out export 沿用原数据集特征。因此冻结评测中的测试节点会携带由自身 held-out performance 汇总出的数量、均值、方差、极值和 eligibility。当前可以声称的是 **root-level performance-edge isolation 与 split-safe prior sidecar**，不能声称整个表示链 leakage-free。若要维持 held-out retrieval 的主要有效性声明，需按 split 重算、移除/屏蔽这些坐标并重新评测；把协议仅改名为 transductive 不能消除 label-derived feature leakage。

> **定向核验位置：** [`build_graph_rf.py`](../scale1m/build_graph_rf.py) 第 91–122、188–193 行；[`d0_splits.py`](../stage2TrainGraphSAGE/d0_splits.py) 第 38–97 行；[`export_rf.py`](../scale1m/export_rf.py) 第 200–223 行。第二项核验表明 `reward`/`unknown` 原生节点仍生成数值边，最终 split/loss/sidecar 不再携带 direction 字段做过滤；参见 [`canonicalize_rf.py`](../scale1m/canonicalize_rf.py) 第 213–247 行、[`losses.py`](../stage2TrainGraphSAGE/losses.py) 第 569–578 行和 [`build_prior_sidecar.py`](../stage3HNSW/build_prior_sidecar.py) 第 146–163 行。

## 0. 先给结论

这篇论文的 Methodology 不宜按照实验发生的时间线展开，也不宜被写成“一个 GraphSAGE 模型加一个 HNSW 索引”。它真正需要讲清楚的是一条端到端、带明确证据边界的系统方法：

```text
模型与数据集快照
    -> 身份、任务和评估记录规范化
    -> 多来源模型--数据集证据图
    -> root-aware 边隔离下的异构图表示学习
    -> 单位球上的模型/查询稠密表示
    -> HNSW 召回 1,000 个候选
    -> split-safe 任务历史先验重排
    -> top-10 模型结果
```

建议把论文 Methodology 压缩成五个主模块：

1. **问题定义与系统总览**：候选是什么、查询是什么、要优化什么、离线与在线如何衔接。
2. **证据规范化与异构图构建**：原始元数据和异构成绩如何变成可比较、可训练且行序稳定的图。
3. **带 root-aware 可见性控制的图检索器**：root-aware 划分、节点编码、关系特定 GraphSAGE 与共享检索空间；feature-level 风险另行审计。
4. **与检索一致的多目标学习**：局部排序、模型--模型对比和全湖 sampled-softmax。
5. **有界两阶段检索**：HNSW 小池召回、split-safe 任务先验及确定性融合。

评测定义与泄漏控制必须紧邻 Methodology 说明；具体结果、延迟数值、基线比较和产物哈希则应分别放入 Experimental Setup、Results 或 Reproducibility Appendix。

## 1. 首先钉死研究对象与贡献边界

### 1.1 研究对象

候选模型湖为

$$
\mathcal M=\{m_1,\ldots,m_N\},
$$

查询集合为

$$
\mathcal D=\{d_1,\ldots,d_Q\}.
$$

查询不是一个裸数据集名，而是已经物化的 **dataset--task 节点**：

$$
d=(\operatorname{normalize}(\text{dataset}),t).
$$

同一数据集在不同任务下必须构成不同的排序问题。冻结产物中，候选数为 $N=3{,}016{,}439$，查询节点数为 $Q=18{,}729$。这两个规模数字适合在 Methodology 总览中出现一次，详细来源放到 Data/Experimental Setup。

本稿后续统一把它称为“数据集–任务（dataset–task）节点”，把承载历史评估值的边称为“性能证据边（内部关系名 `trained_on`）”，把 split 单元称为“数据集根（root）”。代码标识保留反引号，其余部分优先使用中文术语。

### 1.2 当前证据真正支持的任务

当前结果支持的是：

> 对图中已经物化、但其 performance 边被 held out 的 dataset--task 节点，在约三百万模型中检索历史观测意义下的高排名模型。

当前结果**不支持**以下更强表述：

- 面向一个从未进入冻结节点表的新数据集，已经具备验证完毕的 query-only 归纳编码路径；
- 被推荐但从未观测过的模型在新数据集上一定表现良好；
- `gold@10` 等同于下游任务准确率或实际部署成功率。

因此，“inductive”如果出现在主文中，必须精确定义它指模型编码器不依赖 model-ID embedding，还是指可处理完全未见查询。后一项目前没有最终证据支持。

### 1.3 贡献层次建议

最终论文可以把贡献叙事分成三层，但“新颖性”仍需结合 Related Work 再定：

1. **表示学习层**：由多来源模型--数据集成绩、数据集相似性和模型谱系共同构成的异构证据图，以及与检索分数对齐的三目标学习。
2. **系统层**：在约三百万候选上，以 HNSW 构造有界候选池，再只对 1,000 个候选读取和融合任务历史证据。
3. **可信性层**：root-aware 划分、双向 performance 边同步移除、split-specific sidecar、冻结行映射和无标签 ANN 校准共同构成的泄漏与复现契约。

必须把边界写清：第二阶段重排器是**确定性、非学习式**的真实服务阶段，但不应被包装成另一个表示学习贡献。

## 2. 推荐的 Methodology 章节骨架

以下编号假设 Methodology 是论文第 3 节；最终可随全文调整。

### 拟议论文 §3.1：Problem Formulation and System Overview

这一节需要回答四个问题：

1. 模型候选、查询节点和返回结果分别是什么？
2. 为什么查询主键必须包含 dataset 与 task？
3. 第一阶段学习什么分数，第二阶段又加入什么证据？
4. 哪些工作在线下完成，哪些工作发生在在线检索路径？

应给出共享检索空间中的第一阶段分数：

$$
s_\theta(d,m)=z_d^\top z_m=\cos(z_d,z_m),
\qquad z_d,z_m\in\mathbb{R}^{128},
$$

以及候选池：

$$
\mathcal P_{1000}(d)
=\operatorname*{ANN\text{-}Top1000}_{m\in\mathcal M}s_\theta(d,m).
$$

总览段落不宜提前塞入所有数据清洗细节，但应在一开始告诉读者：系统先从异构历史证据中学习一个可索引的统一空间，再用同任务、跨 root 的可见历史成绩重排小池。

**主要证据定位：** 中英文证据库第 8–69 行。

### 拟议论文 §3.2：Evidence Corpus Construction

这一节最好再拆成三个逻辑单元，而不是把数据抓取、特征与边一次性混写。

#### 拟议论文 §3.2.1：Hash-bound frozen crawls and canonical identities

需要写入：

- 模型与数据集分别从 Hugging Face API 枚举；服务端 `Link` 游标、可续跑分片、SHA-256 与 provenance 使本地冻结抓取产物可追溯。
- 两个快照日期均为 2026-08-18。
- 模型 join key 统一为 `lower(strip(id))`，规范化后重复标识直接拒绝。
- 参数量仅接受 `safetensors.total`；模型族优先使用 `config.model_type`；父模型优先使用结构化 `baseModels.ids[0]`。
- 查询主键是 `(normalized dataset, task)`。
- 数据集卡只进行精确、唯一 basename 或无歧义 parent 匹配；模糊情况不做流行度猜测，而回退到清洗后的名称和任务。

写作重点不是逐字段罗列 API，而是解释这些规则怎样防止错误合并、跨 owner 误配和静默身份漂移。抓取命令、分片数和哈希可放附录。措辞上宜用 **hash-bound frozen crawl artifact**，而不要暗示多页 API 枚举是服务端原子事务意义上的快照。

#### 拟议论文 §3.2.2：Metric parsing and canonical supervision

原生 `model-index` 记录被解析为

$$
(m,\text{dataset},\text{task},\text{metric},v).
$$

必须说明：

- 布尔、格式错误和非有限值被丢弃；
- 指标名统一大小写、分隔符和 `@k` 表示，但完整规范名仍是分组键，因此不同 cutoff 不会被合并；
- 指标方向分为 `higher`、`lower`、`reward` 和 `unknown`；方向未知项可作为图证据，但不能定义 gold；
- 原生来源中的重复测量按 $(m,d,t,r)$ 取中位数；
- 在每个可比较组 $g=(d,t,r)$ 内进行 min--max 归一化，常数组置为 0.5，再把 lower-is-better 指标翻转。

核心监督变换建议保留公式：

$$
\bar v_i=
\begin{cases}
\dfrac{v_i-v_g^{\min}}{v_g^{\max}-v_g^{\min}},&v_g^{\max}>v_g^{\min},\\[4pt]
0.5,&v_g^{\max}=v_g^{\min},
\end{cases}
$$

$$
y_i=
\begin{cases}
1-\bar v_i,&c(r)=\text{lower},\\
\bar v_i,&\text{otherwise}.
\end{cases}
$$

每个 dataset--task 节点优先选择方向已知的指标，然后选择记录最多的指标，并以词典序确定性地处理平局。这里必须写清“归一化范围”和“主指标选择顺序”，否则不同任务或不同量纲的成绩为何可用于共同训练会显得没有依据。

这里存在已经由定向实现核验确认的语义问题：上述公式会把 `reward` 和 `unknown` 一并落入 `otherwise` 分支、保留 $\bar v_i$；只含这两类指标的节点仍生成数值边，而最终图、split、损失和 prior sidecar 没有 direction 字段继续过滤。因此它们是 **gold-ineligible，但仍参与拓扑、数值排序/正样本学习，并可能进入 task prior**。正式方法必须决定是显式过滤，还是分别为 `reward` 与 `unknown` 提供可辩护的有序语义；不能把所有 $y_i$ 笼统称为“已完成方向统一的监督”。

#### 拟议论文 §3.2.3：Multi-source merge and candidate closure

六个证据来源按固定优先级合并：

$$
\text{modellens\_v2}
\succ\text{d0\_v1\_5}
\succ\text{a\_ctrl\_2000m}
\succ\text{hf\_effective}
\succ\text{diverse\_zoo}
\succ\text{hf\_model\_index}.
$$

方法段需要交代四步：

1. 历史来源沿用其已经方向处理过的存储权重，并在 `(source, node, model)` 内取中位数；
2. 历史来源在 `(node, source)` 内做 min--max 归一化，常数组置为 0.5；
3. 同一 `(node, model)` 冲突时保留优先级最高的来源，并记录被覆盖项；
4. 每个节点最多保留 200 条边，采用 seed 0 的确定性、按权重分层采样。

监督证据中存在 12,680 个不在模型快照中的历史模型；它们被按规范 ID 排序后追加，而不是丢弃其观测。因此论文必须区分：

- Hugging Face 冻结快照：3,003,759 个模型；
- 闭包后的最终候选湖：3,016,439 个模型。

这些历史独有行的规模记为未知，family 取首个可用历史来源；它们和其余模型一样由同一编码器重新计算特征，不复制旧特征向量。

模型与数据集均使用冻结的连续 `mappedID` 行映射。该契约不是普通工程细节：特征矩阵、嵌入、HNSW 标签、sidecar 和评估都依赖同一行序，错位会静默破坏结果。

**主要证据定位：** 中英文证据库第 71–221 行。

> **章节顺序取决于 P0 修复。** 若最终移除 performance-derived $e_{\mathrm{stats}}$，可以维持“先构图、后划分”的叙述。若选择按 split 重算这些特征，则必须先定义 root-aware split，再定义仅由该阶段可见边构造的 $x_d^{(s)}$；不能继续把全图标签派生特征写在 split 之前。

### 拟议论文 §3.3：Heterogeneous Evidence Graph

#### 拟议论文 §3.3.1：Nodes and features

模型节点的冻结特征为

$$
x_m^{\mathrm{frozen}}
=[e_{\mathrm{name}}^{64}(m)\Vert e_{\mathrm{desc}}^{384}(m)]
\in\mathbb{R}^{448}.
$$

其中：

- 名称特征来自固定种子的高斯哈希表和 `MD5(token) mod 10000`；
- 描述特征由 `all-MiniLM-L6-v2` 编码清洗后的仓库名、模型族和可用的参数规模短语；因此 family/size 的**文本形式已经影响 384 维描述向量**；
- 448 维冻结矩阵没有独立的 structured family/size 通道；训练时还会把二者分别映射到 16 维可学习表，作为显式结构特征；
- 不使用 model-ID embedding。

规模 bucket 0 表示未知，1--14 在 $10^5$ 到 $10^{12}$ 参数之间按半个 decade 划分；family ID 0 为 `Other`，动态观测到的 family 至少出现三次后才获得独立行。这些规则决定长尾和缺失元数据如何退化，属于方法而非纯实现细节。

数据集节点的冻结特征为

$$
x_d^{\mathrm{frozen}}
=[e_{\mathrm{name}}^{64}(d)\Vert e_{\mathrm{card}}^{384}(d)
\Vert e_{\mathrm{stats}}^{10}(d)]
\in\mathbb{R}^{458}.
$$

数据集描述符由名称、任务、可信卡片的任务类别、最多十个标签和截断描述组成；无可信卡片时只使用清洗后的名称与任务。十维统计块包含观测数量、均值、标准差、极值、同 root 节点数和 eligibility 等信息。

冻结实现中的精确统计向量为

$$
e_{\mathrm{stats}}(d)=
[\log(1+n_d),\log(1+n_d),\mu_d,\sigma_d,
a_d^{\min},a_d^{\max},\log(1+r_d),
\mathbb{1}_{\mathrm{eligible}}(d),0,0].
$$

这里必须诚实处理三个实现事实：

- 十维统计块的前两维在冻结实现中相同，最后两维是被禁用的保留位；不能写成十个彼此独立的统计信号。
- task type、class-count 和 arity 三张数据集查找表在最终全湖图中各只有一行，因此不提供节点间区分；真正区分任务的仍是节点身份和文本描述。
- $e_{\mathrm{stats}}$ 中多项坐标直接由完整 supervision 的目标值聚合得到，且冻结 export 没有按 split 重算；这是当前评测已确认的 feature-level label leakage。即使逐边 `edge_attr` 不参与消息聚合，也不能据此声称编码器完全不接触 performance 数值。

#### 拟议论文 §3.3.2：Typed relations

图可以写为

$$
G=(V_M\cup V_D,
E_{MD}\cup E_{DM}\cup E_{DD}\cup E_{MM}\cup E_{MM}^{-1}).
$$

概念上是三类证据，存储上是五种有向关系：

1. 模型到数据集的 `trained_on` performance 边；这里的 `trained_on` 只是内部 schema 名，证据实际表示 observed evaluation/performance，不能推断模型曾在该数据集上训练；
2. 对应的 `rev_trained_on` 反向边；
3. 数据集到数据集的 `similar_to` 边；
4. 父模型到衍生模型的 `is_base_of` 谱系边；
5. 对应的 `rev_is_base_of` 反向边。

数据集相似边先由 384 维数据集描述块的余弦相似度取每个节点 top-20；训练前再确定性裁剪为 top-10，并把保留边属性置为 1。

必须使用准确表述：最终消息算子是 **topology-aware**，而不是 continuous-edge-weight-aware。performance 数值参与监督目标，数据集相似值决定保留哪些邻居，但冻结配置 `weighted_relations=[]`，连续逐边权重不进入消息聚合。这个限定只针对 message operator；数据集统计输入是否间接携带 performance 数值仍需另行审计。

#### 拟议论文 §3.3.3：Graph scale and storage

最终图包含 247,803 条正向 supervision 边和五种有向关系共 2,588,316 条存储边。规模和各关系计数建议放 Experimental Setup 的数据表中；memory-mapped `.npy`、NPZ、Parquet 和图摘要哈希则放 Reproducibility Appendix。主文只需解释为什么需要分片/内存映射，以及行序契约如何贯穿训练与索引。

**主要证据定位：** 中英文证据库第 223–315 行。

### 拟议论文 §3.4：Root-Aware Performance-Edge Protocol and Graph Retriever

归档报告记录了 performance 正反边和重排 sidecar 的 root-level 隔离断言均通过；定向实现核验同时确认，数据集统计特征没有随 split 重算。因此这一节只能描述 edge visibility protocol，不能把当前冻结评测概括为端到端 leakage-controlled retrieval。

#### 拟议论文 §3.4.1：Root-aware split and message graphs

划分单位是 dataset root，而不是单条边或单个 dataset configuration。同一 root 下的全部节点必须进入同一侧。对 split seed $s\in\{0,1,2\}$，root 被打乱后按 supervision 边规模贪心分配，使测试、验证和训练约为 20%、10% 和 70%。

令 $E_{\mathrm{tr}}$、$E_{\mathrm{val}}$、$E_{\mathrm{te}}$ 为 root 划分诱导的正边。训练边中 30% 作为 disjoint supervision $E_{\mathrm{sup}}$，并从训练消息图移除：

$$
\begin{aligned}
G_{\mathrm{train}} &: E_{\mathrm{tr}}\setminus E_{\mathrm{sup}},\\
G_{\mathrm{val}} &: E_{\mathrm{tr}},\\
G_{\mathrm{test}} &: E_{\mathrm{tr}}\cup E_{\mathrm{val}}.
\end{aligned}
$$

任何被移除的 `trained_on` 正向边都必须同步移除其 `rev_trained_on`，否则测试标签可以沿反向边进入消息传递。论文应把这一点作为方法设计明确写出，而不是只在实现细节中说“我们避免泄漏”。

上式只写出了性能证据边在三种消息图中的可见性变化；正文还应确认并说明数据集相似关系和模型谱系关系在三种图中是否保持不变。

#### 拟议论文 §3.4.2：Node encoders

模型输入为

$$
h_m^{(0)}=W_M
[x_m^{\mathrm{frozen}}\Vert E_{\mathrm{size}}[b_m]
\Vert E_{\mathrm{family}}[f_m]]+b_M,
$$

其中 $W_M:\mathbb{R}^{480}\rightarrow\mathbb{R}^{128}$。数据集输入为

$$
h_d^{(0)}=W_D
[x_d^{\mathrm{frozen}}\Vert E_{\mathrm{task}}[t_d]
\Vert E_{\mathrm{class}}[c_d]\Vert E_{\mathrm{arity}}[a_d]]+b_D,
$$

其中 $W_D:\mathbb{R}^{486}\rightarrow\mathbb{R}^{128}$。冻结特征矩阵不接收梯度；查找表、输入投影、图层、关系门控和输出头参与学习。

#### 拟议论文 §3.4.3：One-layer relation-specific GraphSAGE

最终网络只有**一层**异构消息传递。对关系 $r$ 和目的节点 $v$：

$$
\tilde h_{v,r}
=W_{\mathrm{self},r}h_v^{(0)}
+W_{\mathrm{nbr},r}
\left(\frac{1}{|\mathcal N_r(v)|}
\sum_{u\in\mathcal N_r(v)}h_u^{(0)}\right).
$$

不同关系拥有独立的 self/nbr 投影，并以初始化为 1 的可学习标量门控相加：

$$
h_v^{(1)}=
\sum_{r:\operatorname{dst}(r)=\operatorname{type}(v)}
g_r\tilde h_{v,r}.
$$

共享输出头把两类节点投到同一检索空间并逐行 L2 归一化：

$$
z_v=\frac{W_Oh_v^{(1)}+b_O}
{\|W_Oh_v^{(1)}+b_O\|_2}\in\mathbb{R}^{128}.
$$

这里需要强调三点：

- 模型与数据集输入投影不同，但最终输出头共享；
- 每种关系参数独立，关系门控可学习；
- 消息聚合是逐关系无权均值，不能称为 weighted GraphSAGE 的最终配置。

证据库没有给出 $\mathcal N_r(v)=\varnothing$ 时的聚合约定，也未汇总 activation、normalization 和 dropout 的精确位置；这些必须依据最终实现或 resolved config 在正文/附录中补齐。

**主要证据定位：** 中英文证据库第 317–390 行。

### 拟议论文 §3.5：Score-Aligned Multi-Objective Learning

三项损失都直接作用于最终检索所使用的单位向量点积，不应只给出总式而省略各自解决的问题。

#### 拟议论文 §3.5.1：Local rank-consistent objective

对查询节点 $d$，只有目标差超过 0.02 的模型对构成偏好：

$$
\mathcal P_d=\{(m_i,m_j):y_{di}-y_{dj}>0.02\}.
$$

局部 RankNet 损失为

$$
\mathcal L_{\mathrm{rank}}
=\frac{1}{|\mathcal D_B|}\sum_{d\in\mathcal D_B}
\frac{1}{|\mathcal P_d|}
\sum_{(i,j)\in\mathcal P_d}
\operatorname{softplus}
\left(-\frac{s_\theta(d,m_i)-s_\theta(d,m_j)}{0.1}\right).
$$

先在节点内平均、再跨节点平均，避免观测稠密的查询因产生更多 pair 而支配训练。每个节点最多保留 256 对；一半取当前最严重的逆序 pair，一半从其余 pair 采样。

#### 拟议论文 §3.5.2：Model--model contrastive objective

对含 $n_d$ 条训练可见记录的查询，高表现集合大小为

$$
k_d=\max\!\left(1,\operatorname{round}(0.1n_d)\right).
$$

排名前 $k_d$ 的模型形成 high-performer membership；两个模型只要在至少一个查询上共同入选就构成正对。模型对不需要构造稠密 $N\times Q$ 矩阵，而用稀疏 membership 列表在 batch 内生成。

每个 anchor 采样 256 个 batch 模型作为负样本。同谱系 component、但不属于正对的负样本在分母中权重为 2，其余权重为 1；温度为 0.2。应将其称为**谱系感知的负样本加权**，不要误写成启用了单独的 model-level hard-negative mining 模块。与此同时，局部 RankNet 会优先保留一半当前最严重的逆序 pair，因此可以称为 hard-pair prioritization；两者需要在术语上区分。

#### 拟议论文 §3.5.3：Whole-lake sampled-softmax objective

局部 batch 无法暴露三百万规模的完整负样本空间，因此加入全湖目标。采样分布混合带平滑的 degree proposal 与 labeled-model uniform proposal：

$$
q_{\deg}(m)\propto(\deg(m)+1)^{0.75},
$$

$$
q_{\mathrm{lab}}(m)
=\frac{\mathbb{1}[\deg(m)>0]}
{|\{m':\deg(m')>0\}|},
$$

$$
q(m)=0.5q_{\deg}(m)+0.5q_{\mathrm{lab}}(m).
$$

每个 global step 采样 128 个带正样本的数据集节点，每个节点从 $q$ 中有放回地采样 256 个负样本，移除意外抽到的正样本，并对负项应用 $-\log q(n)$ 校正。正样本同样由该查询的 top-10% high-performer set 定义，温度为 0.1。

证据库还说明，该项因为 batch 子图不包含完整负样本宇宙，会在带新一轮 message-edge dropout 的训练消息图上执行一次 full-graph forward。它在三百万节点规模下是重要的计算机制，正文或实现设置中不能被“普通 in-batch sampled softmax”一句带过。

#### 拟议论文 §3.5.4：Joint objective and training mechanics

最终目标为

$$
\boxed{
\mathcal L=\mathcal L_{\mathrm{rank}}
+\mathcal L_{\mathrm{contrast}}
+\mathcal L_{\mathrm{global}}
}
$$

三项系数均为 1。必须避免把最终目标写成 MSE、performance regression 或二元 link-prediction loss；MSE、uniformity regularization、dataset-to-model contrastive loss、hard-negative mining、positive inverse-propensity weighting、独立输出头和 early stopping 在冻结配置中均未启用。

训练细节可在本节结尾用一段交代：mini-batch 以从训练消息图移除的 disjoint supervision $E_{\mathrm{sup}}$ 为监督根；监督 batch size 为 1,024；普通/谱系消息边 dropout 为 0.30/0.05；Adam 学习率 0.01；训练 25 个 epoch；三次运行改变 root split seed 0/1/2，但初始化 seed 固定为 0。完整超参数表更适合 Experimental Setup。

**主要证据定位：** 中英文证据库第 392–523 行。

### 拟议论文 §3.6：Bounded Two-Stage Retrieval

#### 拟议论文 §3.6.1：HNSW candidate generation

由于所有输出嵌入均单位归一化，最大内积、最大余弦和最小平方欧氏距离具有相同排序：

$$
\operatorname*{arg\,max}_m z_d^\top z_m
=\operatorname*{arg\,min}_m(1-z_d^\top z_m)
=\operatorname*{arg\,min}_m\frac12\|z_d-z_m\|_2^2.
$$

因此使用 `space="ip"` 的 HNSW 与训练几何一致。固定索引构造参数为 $M_{\mathrm{HNSW}}=32$、`ef_construction=200`，模型行 $i$ 的 HNSW label 就是 `mappedID=i`。

`ef_search` 从 $\{1000,1500,2000,3000,5000\}$ 中选择第一个使 exact dense top-1000 ID agreement 达到 0.99 的值。该校准不使用 test gold 标签，因此可称为 **label-free index-fidelity calibration**。不过，证据库没有精确定义校准查询集 $\mathcal Q_H$；若它就是各 split 的实际 held-out test queries，则仍属于适配测试查询分布的系统设置。需要先核验 $\mathcal Q_H$，再决定是否仅披露该协议，或改用 validation/calibration queries 或统一的全局 `ef_search`。具体每个 split 的值放 Experimental Setup。

#### 拟议论文 §3.6.2：Split-safe task prior

对报告中的测试查询，令 $n_{tm}$ 和 $A_{tm}$ 分别为该 split 的 train+validation 可见边中，与规范任务 $t$ 相同、且来自测试查询 root 以外的**实现存储归一化边值**的数量与总和。对 higher/lower 与 curated 来源，这些值有明确方向；`reward`/`unknown` 的有序语义存在上述未解决问题。按照文字给出的真实实现，固定收缩先验应写成分段函数：

$$
p_t(m)=
\begin{cases}
\dfrac{A_{tm}+0.5\times5}{n_{tm}+5},&n_{tm}>0,\\[5pt]
0,&n_{tm}=0.
\end{cases}
$$

最终重排分数为

$$
r(d,m)=\frac{s_\theta(d,m)+1}{2}+p_t(m),
$$

即 raw `beta=1`、shrink $k=5$，且不做 query-wise min--max normalization。系统只为 $\mathcal P_{1000}(d)$ 中的候选读取先验，再以**由模型行 ID 生成的固定、无标签 permutation** 处理精确平局并返回 top-10；它不等同于简单按自然行 ID 升序裁决。

sidecar 只含 train+validation 可见边，并显式断言：模型/数据集行映射一致、root 映射一致、任何可见 prior 边都不与被评分测试查询共享 root。查询本身及其同 root sibling 因而不会向重排器泄漏成绩。

#### 拟议论文 §3.6.3：Why the bounded pipeline matters

方法段应解释复杂度机制，而不提前报告效果数字：

- exact dense top-1,000 仍需对全部 $N$ 个模型打分，保留 $O(N)$ 全湖扫描；
- exact full-pool fusion 还需为全部模型读取、融合和排序任务先验；
- 实际系统以 ANN 找 1,000 个候选，只对固定大小的小池执行 prior lookup 和融合。

这说明 HNSW 不是仅用于“近似加速同一个全池打分器”，而是把昂贵的第二阶段证据访问也限制在 $K=1000$。

**主要证据定位：** 中英文证据库第 562–629 行。

## 3. Experimental Setup 接口：评测协议

评测通常可单列为第 4 节 Experimental Setup，但以下定义必须与方法叙事无缝衔接。

### 3.1 Performance-edge-held-out inference graph

评测嵌入由 $G_{\mathrm{test}}$ 生成：它包含训练和验证消息边，但排除所有测试 `trained_on` 边及其反向边。服务用 full-message embeddings 与 performance-edge-held-out evaluation embeddings 分开导出；所有最终报告只使用 `z_m_eval` 和 `z_d_eval`。这一命名只承诺边可见性，不预先替数据集统计特征的 split-safety 背书。

论文需要清楚区分：

- **evaluation representation/index**：基于 held-out-message 图，用于报告结果；
- **full-message serving representation/index**：算法路径相同，但当前证据冻结中没有单独归档并实测的 production index。

### 3.2 Gold definition and eligibility

对合格测试查询 $d$，令 $C_d$ 为其 held-out 已观测模型集合，gold 模型为

$$
g_d=\operatorname*{arg\,max}_{m\in C_d}y_{dm}.
$$

先对任意检索路径 $a$ 定义它实际返回的十模型集合 $\widehat{\mathcal R}_{10}^{\,a}(d)$。通用命中指标应写为

$$
\operatorname{gold@10}(a)
=\frac{1}{|\mathcal Q_{\mathrm{test}}|}
\sum_{d\in\mathcal Q_{\mathrm{test}}}
\mathbb{1}\!\left[g_d\in\widehat{\mathcal R}_{10}^{\,a}(d)\right].
$$

对 exact dense 诊断路径，证据库还给出了全湖 competition rank：严格“大于”计数会让预测分数相同的模型共享名次，且不等同于服务阶段的固定 permutation：

$$
\operatorname{rank}_d(m)
=1+\sum_{j=1}^{N}\mathbb{1}[s(d,m_j)>s(d,m)].
$$

对应的 exact-dense 特例可写为

$$
\operatorname{gold@10}
=\frac{1}{|\mathcal Q_{\mathrm{test}}|}
\sum_{d\in\mathcal Q_{\mathrm{test}}}
\mathbb{1}[\operatorname{rank}_d(g_d)\le10].
$$

合格查询要求：指标方向已知或来自人工整理的方向统一来源、任务不是 reinforcement learning、数据集名不是 placeholder、至少有三个已观测候选，且 held-out 值不是常数。次要指标包括 `gold@1`、`top3@10`、`gold-gap@10` 和 root-macro `gold@10`。

对 HNSW 与重排路径，应直接使用相应的 $\widehat{\mathcal R}_{10}^{\,a}(d)$；上面的 $s(d,m)$ rank 公式只是 exact dense 路径的特例，不能拿它替代重排分数 $r(d,m)$ 的实际输出定义。

### 3.3 What the metric does and does not measure

`gold@10` 测量已观测 held-out 最佳模型能否进入检索 top-10。它不执行模型、不评估未观测 query--model pair，也不等于“新数据集上的正确率”。这一定义既是 Experimental Setup，也是全文 claim boundary。

**主要证据定位：** 中英文证据库第 525–560、617、777–789 行。

## 4. 内容应该放在哪里

| 内容 | Methodology 主文 | Experimental Setup | Results | Reproducibility Appendix | Limitations |
|---|---:|---:|---:|---:|---:|
| 问题定义、数据集–任务查询和两阶段总览 | ✓ |  |  |  |  |
| 抓取方法、身份规范化和指标方向统一 | ✓ |  |  | 细规则 |  |
| 快照日期、记录数和来源边数 | 简述 | ✓ |  | 完整表 |  |
| 六来源合并、冲突优先级和 per-node cap | ✓ |  |  | 规则文件 |  |
| 节点特征、关系与行序契约 | ✓ |  |  | 张量/哈希 |  |
| 退化数据集查找表及其实际信息量 | ✓ | ✓ |  |  | ✓ |
| root-aware split 与三种 message graph | ✓ | ✓ |  | 审计结果 |  |
| 编码器、消息传递和三项损失 | ✓ |  |  | 完整推导 |  |
| optimizer、batch、fan-out、dropout、epochs | 核心项简述 | ✓ |  | 全配置 |  |
| 硬件和软件环境 |  | 概况 |  | 精确版本 |  |
| HNSW 几何、$K=1000$、任务先验与融合 | ✓ | ✓ |  |  |  |
| `ef_search` 逐 split 值与校准集合 | 简述规则 | ✓ |  |  |  |
| `gold@10`、eligibility 和辅助指标定义 |  | ✓ |  |  |  |
| 0.3031、BM25 倍数、retention 与 latency |  |  | ✓ |  |  |
| CLI、SHA-256、产物路径和环境版本 |  |  |  | ✓ |  |
| 卡片/规模稀疏及未验证的新查询路径 |  | ✓ |  |  | ✓ |

## 5. 最容易写错的最终系统口径

下面这些应当作为写作时的“禁止漂移清单”。

| 容易误写 | 最终证据支持的准确口径 |
|---|---|
| 学习式或 metric-aware MLP reranker | 确定性、非学习式的 task prior；raw `beta=1`、shrink $k=5$ |
| 两层 GraphSAGE | 一层 relation-specific heterogeneous GraphSAGE |
| 四种边 | 三类语义证据、五种存储的有向关系 |
| weighted message passing | 冻结配置 `weighted_relations=[]`；无权逐关系均值聚合 |
| performance regression / MSE 是主损失 | RankNet + model--model contrastive + whole-lake sampled-softmax |
| 单独的 model-level hard-negative mining 已启用 | 该模块未启用；实际存在同谱系非正样本分母加权，以及 RankNet 的 hard-pair prioritization |
| 移除 held-out 边即可证明所有输入都无标签泄漏 | 只完成 edge/sidecar 隔离；$e_{\mathrm{stats}}$ 已确认由全监督图构造并沿用到 held-out export |
| `reward`/`unknown` 都已经可靠地完成方向统一 | 它们不能定义 gold，却已确认仍进入数值边、排序类信号和可能的 task prior；必须过滤或单独论证 |
| 448 维模型特征完全不含 family/size 信息 | 没有独立 structured 通道，但 family/size 文本已进入 MiniLM descriptor，另有显式可学习 lookup |
| `trained_on` 表示模型确实用该数据集训练 | 它是内部关系名，承载的是 observed evaluation/performance evidence |
| API crawl 是服务端原子快照 | 证据支持的是本地 hash-bound、可续跑并冻结的抓取产物 |
| 三次完全独立随机训练 | split seed 为 0/1/2，初始化 seed 固定为 0 |
| 所有查询都有丰富数据集卡语义 | 仅 3,928/18,729 节点匹配到精确或 parent 卡片 |
| 任务类别、class count、arity lookup 提供有效区分 | 最终图中三张表各仅一行，不提供节点间差异 |
| 对全新未见数据集已经验证在线编码 | 当前只验证已物化节点的 held-out performance 边场景 |
| 0.3031 是下游准确率 | 它是历史观测 gold 模型的 `gold@10` |
| 0.694/1.223 ms 是完整端到端延迟 | 只测预计算查询嵌入之后、warm-up 后的 single-query/single-HNSW-thread HNSW + rerank；不含查询编码、载入和建索引 |
| 模型湖就是 3,003,759 个 hub 记录 | 候选闭包后为 3,016,439，其中追加 12,680 个历史独有模型 |
| 实测索引就是完整 production index | 实测的是三份 held-out evaluation embedding index |

## 6. 正式动笔前仍需回答的问题

它们分成三类：不解决就不能维持主要有效性声明的审计、必须在主文/设置中定义的协议，以及可放附录但仍须有答案的复现细节。

### 6.1 成稿前必须完成的 validity audits

1. **修复已经确认的 dataset-statistics 标签泄漏并重做受影响评测。** $e_{\mathrm{stats}}$ 由完整 supervision 预计算，split/export 沿用原特征，因此测试节点可见自身 held-out performance 的聚合量。需要选择按 split 重算、移除/屏蔽统计块或建立不含这些坐标的版本，然后重新训练/导出/评测；仅改称 transductive 不能修复 held-out label leakage。
2. **处理已经确认进入数值目标的 `reward` 与 `unknown`。** 当前实现会保留其数值边，并将其用于训练可见 performance topology、top-performer membership/排序类目标及可能的 task prior。两者应分别决定：过滤出 value-based losses/prior、仅保留 topology，或给出可验证的方向语义；尤其不能默认 `unknown` 越大越好。
3. **最终配置是否接触过 test gold。** 至少逐项登记 $K=1000$、`beta=1`、shrink $k=5$、eligibility 规则、cap 200、top-10% positives、0.02 preference gap、三项 unit weight、epoch/checkpoint 选择，并标为预先指定、训练集导出、验证集选择或看过测试结果后的探索。若关键最终配置由同一 test gold 选择，需要 untouched test，或明确降低 confirmatory claim。
4. **Root/task 规范化能否真正阻断别名泄漏。** 除了给出 dataset root 与 normalized task 的定义，还要审计 dataset/config、owner、别名、重命名、parent-card 名称及六来源字符串差异，确认同一真实数据集不会以不同名称跨 split。
5. **Gold 与 prediction ties。** 先量化多个真实并列最优模型的查询数，再决定 single-gold、multi-gold 或确定性 gold。exact dense 的 competition rank 与服务端固定 permutation 是两套 tie 语义；若 ties 不稀少，需要敏感性重算。

### 6.2 主文或 Experimental Setup 必须明确

1. **一句话任务定义与论文 claim。** 是“model retrieval for materialized dataset–task nodes”，还是更宽泛的“model recommendation”？若用后者，必须立刻给出 root-held-out、materialized-node 的当前评测边界。
2. **六个来源的 provenance 与可比性。** 需要解释各来源的出处、时间、任务/指标覆盖、原始权重语义、许可证、覆盖偏差和优先级理由；跨来源各自 min–max 后共同训练的可比性也需要说明或敏感性分析。
3. **无历史记录时 $p_t(m)=0$ 的设计理由。** 本稿已把真实规则改写为 piecewise；论文还需解释为什么“无证据”低于 shrinkage target 0.5。
4. **Binary negatives 的最终角色。** split 模块生成 1:1 二元负例，但最终目标不含 BCE/link-prediction 项。若未使用，就不要暗示其参与最终训练；若用于其他路径，则画清数据流。
5. **一层网络与 two-hop fan-out 的关系。** 确认第二段 fan-out 是 loader 兼容设置、多采但未用，还是证据文字遗漏了实际计算。
6. **HNSW 校准查询集。** 精确定义 $\mathcal Q_H$。若使用 held-out test queries，则披露其只看 dense-neighbor ID agreement 的协议，并说明为什么允许按 split 选择不同 `ef_search`；更稳妥的替代是 validation/calibration queries 或统一全局值。
7. **Validation 的生命周期。** 说明 validation 是否先用于超参数/模型选择，以及配置冻结后为何在测试期允许 validation performance 边进入 $G_{\mathrm{test}}$ 和 task prior；若这是 train+validation refit/evidence-expansion protocol，应明确命名并保证 test 未参与选择。
8. **Eligibility 与空集合处理。** 明确 RL、placeholder、curated-oriented-source、至少三个候选和非恒定值的算法；同时说明 RankNet 无有效 pair、contrastive anchor 无正样本、top-10% cutoff tie/rounding 及重复负样本怎样处理。
9. **空邻域与图层细节。** 给出 $\mathcal N_r(v)=\varnothing$ 时的聚合约定，以及 activation、normalization、bias、dropout 的精确位置。
10. **评测 index 与 production index。** 当前测得的是 held-out embedding index；不要把算法可服务性写成已归档 production artifact 的实测结果。
11. **候选可访问性。** 追加的 12,680 个历史独有模型不在冻结 hub crawl 中；需要说明其是否仍可下载、是否合法进入“可推荐候选”，还是只作为历史图节点/评测身份存在。
12. **最终 checkpoint 与优化细节。** 汇总 checkpoint 选择、Adam betas/epsilon、weight decay、scheduler、梯度裁剪、混合精度和确定性设置；硬件/软件概况进设置，精确版本进附录。
13. **外部文献引用。** GraphSAGE、HNSW、RankNet、MiniLM、contrastive learning、sampled softmax/logQ correction 与 BM25 需要正式论文来源；内部代码与 JSON 只能证明本项目实现。

### 6.3 可放入补充材料，但必须有答案

- 权重分层的 200-edge cap 怎样划分 strata；
- family name fallback、规模 bucket 和动态 family vocabulary 的完整规则；
- metric direction dictionary 的完整条目；
- 原生来源 cap 后的 74,346 条边为何在最终来源表中变为 73,670 条；
- lineage 多 parent、环、自环、缺失父节点和 96.083% 解析率分母的处理；
- MiniLM 的精确 revision、tokenizer、pooling、最大长度与截断设置；
- fixed label-free tie-break permutation 的构造方式；
- HNSW warm-up/重复次数、库版本、线程设置和内存口径；
- chunked inference 与 whole-graph inference 的数值容差；
- 所有 artifact hash、resolved config 与复现实验命令。

## 7. 建议的图与表

### 7.1 主方法图：三面板即可

建议一张图同时表现三个层面，而不是画成纯线性流水线：

```text
Panel A: Evidence construction
Hub metadata + six evaluation sources
        -> canonicalization/orientation
        -> model--dataset evidence graph

Panel B: Representation learning
model features ----------------\
repaired dataset features ------- heterogeneous GraphSAGE -> shared unit sphere
typed topology -----------------/                     \-> three aligned losses

Panel C: Bounded retrieval
materialized query embedding -> HNSW top-1000
                                   + split-safe task prior
                                   -> deterministic top-10
```

图上应明确标出：performance edge values 用于监督、不是消息权重；test root 的 performance 边从评测消息图和 prior sidecar 中排除；performance-derived 数据集统计必须在 P0 修复后按合法可见信息生成或从输入中移除。

### 7.2 推荐表格

主文最多保留三张方法/设置表：

1. **Node and relation schema**：节点特征维度、三类语义关系/五种有向关系；
2. **Frozen training configuration**：深度、维度、三项损失、采样数、dropout、epoch 和 seed；
3. **Evaluation paths**：exact full-pool fusion、exact top-1000 + prior、HNSW top-1000 + prior 分别需要什么计算。

来源计数、哈希和 CLI 不应挤占这三张表。

## 8. 段落级写作顺序

正式成稿时，可以依照下面的“读者认知顺序”，而不是工程执行日志顺序：

1. 一段定义任务、查询和输出；
2. 一段用图或总公式概览 offline learning 与 online retrieval；
3. 两至三段说明证据如何被规范化、方向统一和合并；
4. 一段先定义 root-aware edge visibility control，确保任何标签派生特征的可见集合先被确定；
5. 一至两段定义图的节点、关系，以及 P0 修复后的模型/数据集特征；
6. 一段定义两类输入编码器与异构 GraphSAGE；
7. 三段分别解释三项损失的动机、正负样本与公式；
8. 一段给出总目标和必要训练配置；
9. 一段解释单位球几何与 HNSW 候选池；
10. 一段定义 split-safe task prior、融合和 tie-break；
11. 最后一段解释为何只重排 $K=1000$ 能把第二阶段成本与全湖规模解耦；
12. 转入 Experimental Setup，再定义 gold、eligibility、三个 split 的规模和 latency protocol。

这个顺序让读者先理解“问题与方法”，再看到“如何可信地测量”，避免在 Methodology 中被运行代号、结果数字和产物路径打断。

## 9. 证据索引

两份证据库行号完全对齐，可按语言任选其一核对：

注意：最终报告仍位于历史目录名 `docs/1M` 下，但其标题和冻结候选闭包已经是 3M-scale；论文版本命名应以最终证据内容为准，避免把目录名误当规模。

| 主题 | 行号 |
|---|---:|
| 最终系统摘要与形式化 | 8–69 |
| 快照、身份、卡片与 metric semantics | 71–142 |
| 监督规范化、六来源合并与候选闭包 | 144–221 |
| 节点特征与五种有向关系 | 223–315 |
| root-aware 划分、编码器与 GraphSAGE | 317–390 |
| 三项目标与冻结训练配置 | 392–523 |
| held-out inference 与评测定义 | 525–560 |
| HNSW、task prior 与有界检索 | 562–651 |
| 实现入口与复现证据 | 653–775 |
| 证据限定的局限性 | 777–789 |
| paper-ready method summary | 791–799 |

本稿新增的两个“已确认”判断不是来自证据库的自然语言摘要，而是来自其直接链接实现的定向只读审计：一是完整 supervision 生成的 $e_{\mathrm{stats}}$ 被 split/export 原样沿用；二是 `reward`/`unknown` 边在构图后没有 direction 字段供损失与 sidecar 过滤。对应文件和行号已列于文首，后续若修复实现，应同步更新本稿与证据库。

## 10. 方法主链的一句话摘要（P0 修复后方可用于论文）

> ModelLakeFishing 将规范化的模型元数据、数据集描述、跨来源性能记录和模型谱系组织成异构证据图，通过 root-aware 边可见性控制下的关系特定 GraphSAGE 与局部排序、模型对比及全湖 sampled-softmax 联合学习可索引的模型--查询嵌入；在线阶段用内积 HNSW 从约三百万候选中召回 1,000 个模型，再以排除测试 root 的同任务历史先验进行确定性重排并返回 top-10。

这句话覆盖了预期方法主链，但当前冻结结果还使用了由测试节点自身 held-out performance 汇总得到的数据集统计特征，因此它现在只能作为整改后的写作目标，不能直接为现有主结果背书。修复并重评后，论文仍须保留另一项范围限定：实证对象是图中已物化、performance 边被 held out 的数据集–任务节点，而不是从未进入图的新数据集。
