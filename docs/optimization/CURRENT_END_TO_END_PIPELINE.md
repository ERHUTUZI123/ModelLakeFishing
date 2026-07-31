# ModelLakeFishing 最新端到端 Pipeline

> **当前版本：D0 lake + L1L3b + HNSW + S2/P2b**  
> **状态日期：2026-07-21**  
> **本文定位：** 当前系统的单一入口说明。结论以现有代码、冻结 manifest 和可审计产物为准；历史优化文档用于解释为什么采用或否决某一分支。

## 0. 一页结论

当前主线不是“把数据集和 Model Card 做一次文本向量匹配”，而是：

1. 从 Hugging Face 缓存中的 `model-index` 与 lineage 证据建立去重、归一化的模型—数据集性能湖；
2. 把模型、数据集、性能、数据集相似性与模型血统组织成异构图；
3. 用 **L1L3b** 训练 128 维、L2 归一化的模型/数据集共同空间；
4. 冻结模型向量并建立 HNSW cosine 索引；
5. 在基础 MIPS 分数上，使用合法的湖内 sibling 与 task 证据做 **S2/P2b** 加法融合；
6. 当某类先验不存在时，对应项严格为 0；两类都不存在时退化为纯 MIPS。

```text
HF metadata/model-index/lineage cache
                  │
                  ▼
       D0 intake + observation vault
                  │  one row per (dataset node, model)
                  ▼
      D0 heterogeneous transfer graph
                  │
                  ▼
      root-aware L1L3b training/eval
                  │
                  ▼
   frozen z_m / z_d ──► HNSW cosine index
          │                    │
          └──── prior sidecar ─┘
                  │
                  ▼
 minmax(MIPS) + sibling prior + task prior
                  │
                  ▼
             final Top-K models
```

当前冻结规模与身份如下。

| 对象 | 当前值 |
|---|---:|
| 模型 | 9,491 |
| 数据集节点 | 1,463 |
| 数据集 root | 421 |
| 去重 `trained_on` 边 | 54,795 |
| lineage 边 | 990 |
| 原始 `similar_to` 边 | 29,260（每个数据集 k=20） |
| gold-evaluable 节点 | 613 |
| family 词表 | 371 |
| task 词表 | 17，`Other` 仅 3.1% |
| 冻结向量 | `z_m: 9491×128`，`z_d: 1463×128`，float32，单位向量 |
| 当前 graph SHA-256 | `45c1dd3cffa0a94bc6a5e34263b025c6b113033ced299e2367f2555d60e2c781` |
| 当前 checkpoint SHA-256 | `a07628c06b0c1c3ede5d87ce31fcbae7cc6030de95d12a2ad429b6f8ceb9bef5` |
| 当前 index SHA-256 | `9f4bf7b5751f2ea9d4f4c0a27449f05f6376efc2e076cd8344d732c46c1a2395` |

## 1. “最新版本”具体指什么

系统包含三个彼此独立、但按顺序组合的层次。

| 层次 | 当前采用项 | 不属于当前主线的项 |
|---|---|---|
| 数据与图 | D0 strict lake、原生 task metadata、root-aware split | 2K 旧湖、v5 `e_content` 图 |
| 表示与训练 | L1L3b | G2、L2/L2b、L4、P2 degree-cap、S1 root pooling |
| 检索与排序 | 归一化 MIPS/HNSW + S2 sibling + P2b task | 单独纯 MIPS 仅作为退化路径/对照 |

因此，**L1L3b 是当前冻结 embedding 的训练配置；S2/P2b 是在冻结 embedding 之上的最终排序策略**。S2/P2b 不重训 GNN，也不改变 `z_m`、`z_d` 或 `index.bin`。

需要同时区分“算法设计”与“现有可执行接线”：

| 能力 | 当前实现状态 |
|---|---|
| D0 图、L1L3b checkpoint、export、HNSW index | 已冻结并有 manifest/hash |
| warm HNSW 查询 | `query.py` 已实现并使用持久化 HNSW |
| warm S2/P2b 融合 | `serving_rerank.py` 已实现，week8 结果由它真实产生 |
| HNSW 候选池后接 S2/P2b | 文档设计如此；当前 `serving_rerank.py` 实际对 9,491 个模型做全量精确点积，尚未调用 HNSW |
| D0 cold + fused 查询 | 尚未形成可运行闭环；详见第 10 节 |

## 2. Stage 0：D0 数据湖与监督事实表

### 2.1 模型 intake

`d0_intake_audit.py` 先审计本地缓存，原始宽松候选规则为：

$$
I_{\mathrm{union}}(m)=I_{\mathrm{model\text{-}index}}(m)
\lor I_{\mathrm{lineage}}(m)
\lor I_{\mathrm{family}}(m).
$$

当前 D0 建图使用更严格的模型集合：

$$
\mathcal M=\left\{m:
\bigl(I_{\mathrm{model\text{-}index}}(m)\lor I_{\mathrm{lineage}}(m)\bigr)
\land \neg I_{\mathrm{disabled}}(m)
\land \neg I_{\mathrm{private}}(m)
\right\}.
$$

这一步排除了仅因 family 名称可解析、但没有性能或 lineage 证据的长尾模型。这里还存在一个必须区分的时间快照：冻结图在 2026-07-19 建成并包含 9,491 个模型；本地 `d0_lake` intake/funnel 又在 2026-07-20 随缓存 top-up 更新，目前报告 10,234 个 strict 候选。后者不是从 9,491 再“过滤”出来的同一批数据，而是更晚的上游快照。当前 serving 身份由 manifest 绑定的 9,491-model graph 决定。

### 2.2 性能观测与指标归一化

监督唯一入口是：

```text
stage1BuildTransferGraph/artifacts/d0_lake/d0_observations.parquet
```

它在建图前已经完成：

- 数据集名 canonicalization；
- 主指标选择；
- 同一 `(dataset_node, model_id, metric)` 多次 run 的 median folding；
- 最终一行一个 `(dataset_node, model_id)` 的去重约束；
- 原始 provenance 保留。

当前允许的 bounded higher-is-better 指标为：

```text
accuracy, f1, matthews_correlation, pearson, spearman,
exact_match, map, mrr, rougeL
```

对数据集节点 $d$，只保留它的 `chosen_metric(d)`。数值统一到 $[0,1]$：

$$
a(d,m)=\operatorname{clip}_{[0,1]}
\begin{cases}
v(d,m)/100,&v(d,m)>1.5,\\
v(d,m),&v(d,m)\le 1.5.
\end{cases}
$$

注意：这里的 $a(d,m)$ 是公开 `model-index` 记录的归一化值，不是本 pipeline 临时下载模型并重跑 benchmark 得出的新成绩。不同数据集可以选择不同主指标；**只在同一数据集内部比较模型**，不跨数据集直接比较这些数值。

### 2.3 数据集节点与 root

数据集节点使用 `dataset/config` 标识，例如：

```text
amazon_massive_intent/en
conll2003/conll2003
```

root 是 `/` 前的基准数据集名：

$$
r(d)=\operatorname{prefix}_{/}(d).
$$

不同语言或 config 可以是不同节点，但属于同一 root。当前图有 1,463 个节点、421 个 root。

## 3. Stage 1：D0 异构图

### 3.1 图结构

令异构图为 $\mathcal G=(\mathcal V_M\cup\mathcal V_D,\mathcal E)$。当前 `.pt` 中实际有五种有向 edge type，对应四类关系：

| edge type | 方向 | 数值属性 | 含义 |
|---|---|---|---|
| `trained_on` | model → dataset | $a(d,m)$ | 模型在数据集上的公开性能 |
| `rev_trained_on` | dataset → model | 同上 | 消息传递的反向镜像 |
| `similar_to` | dataset → dataset | metadata cosine | 数据集卡片语义近邻 |
| `is_base_of` | base model → derivative | 1 | 模型 lineage |
| `rev_is_base_of` | derivative → base model | 1 | lineage 反向镜像 |

建图时 `similar_to` 在 dataset card embedding 上取 k=20；L1L3b 训练和导出前通过 graph surgery 改为 `topk_unweighted, k=10`。因此：

- `d0_graph_report.json` 的 29,260 条是原始 k=20 图；
- checkpoint/manifest 记录的 serving 图语义是 k=10、拓扑保留但不使用 similarity 数值权重。

### 3.2 模型节点输入

冻结输入为：

$$
x_m^{\mathrm{frozen}}=
e_{\mathrm{name}}^{64}(m)\Vert e_{\mathrm{desc}}^{384}(m)
\in\mathbb R^{448}.
$$

- `e_name`：MD5 hash-token mean，seed 42；
- `e_desc`：`all-MiniLM-L6-v2` 编码模型 ID、pipeline tag、library、tags、datasets 等 metadata descriptor；
- 参数量映射到 15 个 size bucket，缺失值为 0；
- family 按规则推断并做最小频次折叠，未知/长尾为 `Other=0`。

进入 GNN 前：

$$
\widetilde x_m=x_m^{\mathrm{frozen}}
\Vert E_{\mathrm{size}}[b(m)]
\Vert E_{\mathrm{family}}[f(m)],
$$

其中两个可学习 embedding 均为 16 维。不存在以 `mappedID` 为索引的节点 embedding，因此模型编码器保持 inductive 形式。

### 3.3 数据集节点输入

当前 D0 主图不含 v5 的 sample-content view。冻结输入为：

$$
x_d^{\mathrm{frozen}}=
e_{\mathrm{name}}^{64}(d)\Vert e_{\mathrm{card}}^{384}(d)
\Vert e_{\mathrm{stats}}^{10}(d)
\in\mathbb R^{458}.
$$

`e_stats` 包含边数、run 数、性能均值/方差/极值、root 大小、gold flag 等统计。learnable 部分为：

$$
\widetilde x_d=x_d^{\mathrm{frozen}}
\Vert E_{\mathrm{task}}[t(d)]
\Vert E_{\mathrm{nclass}}[c(d)]
\Vert E_{\mathrm{arity}}[q(d)].
$$

当前 task 由观测中的 dominant task 原生构建，低于 5 次的类别折叠到 `Other=0`，最终 17 类；`n_class` 和 `arity` 在 D0-v1 中均为 unknown 0，但接口和 embedding table 保留。

## 4. Stage 2：L1L3b 图模型

### 4.1 编码与消息传递

model 和 dataset 输入先分别投影到 128 维。当前配置使用一层 `EdgeAwareHetero`，每个关系有独立的 GraphSAGE 参数与可学习 gate，但 `weighted_relations=[]`，所以**所有 edge_attr 均不参与消息权重**，只使用拓扑：

$$
h_v^{(1)}=
\sum_{\rho:\,*\rightarrow \tau(v)}
\gamma_\rho\left(
W_{\rho,\mathrm{self}}h_v^{(0)}+
W_{\rho,\mathrm{nbr}}
\frac{1}{|\mathcal N_\rho(v)|}
\sum_{u\in\mathcal N_\rho(v)}h_u^{(0)}
\right).
$$

model 与 dataset 共用最终线性 head，并做 L2 归一化：

$$
z_v=\frac{W_o h_v^{(1)}}{\|W_o h_v^{(1)}\|_2},
\qquad z_m,z_d\in\mathbb R^{128}.
$$

基础检索分数因此严格等于 cosine/MIPS：

$$
s(d,m)=\langle z_d,z_m\rangle=\cos(z_d,z_m).
$$

### 4.2 当前总损失

L1L3b 的实际训练目标为：

$$
\mathcal L=
\mathcal L_{\mathrm{rank}}
+\mathcal L_{\mathrm{contrast}}
+\mathcal L_{\mathrm{lake}},
$$

三个系数均为 1。MSE、uniformity、dataset-to-model contrastive 均关闭。

#### A. RankNet：直接训练被 HNSW 使用的几何分数

对同一数据集 $d$ 内的两个已观测模型，只有性能差超过 $\epsilon=0.02$ 才形成偏序对：

$$
\mathcal P_d=\{(i,j):a(d,i)-a(d,j)>0.02\}.
$$

$$
\mathcal L_{\mathrm{rank}}=
\frac{1}{|\mathcal D_B|}
\sum_{d\in\mathcal D_B}
\frac{1}{|\mathcal P_d|}
\sum_{(i,j)\in\mathcal P_d}
\operatorname{softplus}
\left(-\frac{s(d,i)-s(d,j)}{T_r}\right),
\qquad T_r=0.1.
$$

每个数据集最多使用 256 对；过多时保留一半当前最难/倒序 pair，再随机采样其余 pair。最后按数据集 macro 求均值，避免大数据集天然权重更高。

#### B. 模型—模型 supervised contrastive

先在每个数据集的 train-visible 模型中选 top 10% 为 high performers。若两个模型在至少一个数据集上共同进入 top 10%，它们互为正样本。对 anchor $i$：

$$
\mathcal L_{\mathrm{contrast}}(i)=
-\frac{1}{|P_i|}\sum_{p\in P_i}
\log
\frac{\exp(\langle z_i,z_p\rangle/T_c)}
{\sum_{a\ne i}w_{ia}\exp(\langle z_i,z_a\rangle/T_c)},
\qquad T_c=0.2.
$$

同 lineage component、但不是共同高性能模型的 pair 使用 $w_{ia}=2$ 作为 hard negative；其他 pair 为 1。大湖训练中只为当前 mini-batch 构造 mask，不创建 $9491^2$ 的稠密矩阵。

#### C. L1b：全湖 logQ sampled-softmax

令 $P_d$ 为数据集 $d$ 的 train-visible top-10% 模型，模型的 train-visible label degree 为 $\deg(m)$。proposal 为：

$$
q(m)=\frac{(\deg(m)+n_0)^{\eta}}
{\sum_j(\deg(j)+n_0)^{\eta}},
\qquad \eta=0.75,\quad n_0=1.
$$

对每步采样的数据集 $d$，从全湖按 $q$ 有放回采样 $N=256$ 个负例，去掉误采到的正例。代码实际计算：

$$
\mathcal L_{\mathrm{lake}}(d)=
-\frac{1}{|P_d|}\sum_{p\in P_d}
\log\frac{\exp(s(d,p)/T_g)}
{\sum_{p'\in P_d}\exp(s(d,p')/T_g)
+\sum_{n\in S_d\setminus P_d}
\exp(s(d,n)/T_g-\log q(n))},
\qquad T_g=0.1.
$$

每个优化 step 最多抽 16 个有正样本的数据集计算该全图项。相较旧 G2 的“只从人工定义的可靠负例池取样”，它让无标签模型也能作为检索负方向出现；`-log q(n)` 则修正 popularity-biased proposal。L1b 把负采样数从 64 提到 256，使 Monte Carlo 均值估计的方差按

$$
\operatorname{Var}(\widehat Z_N)=\frac{\operatorname{Var}(X)}{N}
$$

缩小到原来的四分之一。这一配置最终命名为 **L1L3b**。

### 4.3 固定训练超参数

| 项 | 当前值 |
|---|---:|
| layers / hidden / output | 1 / 128 / 128 |
| epochs | 25 |
| optimizer / lr | Adam / 0.01 |
| link batch | 1,024 |
| neighbor sampling | `(10, 10)`（一层模型只消费第一层邻居） |
| ordinary edge dropout | 0.30 |
| lineage edge dropout | 0.05 |
| RankNet temperature / min gap | 0.1 / 0.02 |
| high-performer fraction | 0.10 |
| lake negatives / datasets per global step | 256 / 16 |
| global temperature | 0.1 |
| scorer | normalized raw dot product |
| similar-to surgery | top-k unweighted, k=10 |

## 5. 防泄漏切分与离线评测

### 5.1 L1L3b 晋升：root-aware protocol

同一个 root 的所有 config 必须在同一侧：

$$
r(d_i)=r(d_j)\Longrightarrow
\operatorname{split}(d_i)=\operatorname{split}(d_j).
$$

root 随 seed 打乱后，按边量贪心填充约 20% test、10% validation、70% train。train 边中再抽 30% 作为 disjoint supervision，不能出现在 train message graph 中：

| 视图 | message edges | supervision edges |
|---|---|---|
| train | 70% train-message | 30% disjoint train |
| validation | 全部 train | validation |
| test | train + validation | test |

`rev_trained_on` 始终只镜像当前 message edges；held-out edge 不会通过反向边泄漏。

### 5.2 评测集合与指标

对每个 test 数据集，只在“至少 3 个已标注候选且性能非恒定”时计入评测。记其已标注候选为 $C_d$，gold 为：

$$
g_d=\arg\max_{m\in C_d}a(d,m).
$$

主要指标：

- `observed_hit1`：在 $C_d$ 内，Top-1 是否就是 $g_d$；
- `top3_hit1`：推荐模型真实性能是否落在 $C_d$ 的前三；
- `regret1`：$a(d,g_d)-a(d,\widehat m_1)$；
- `gold@K`：$g_d$ 在全湖 9,491 个模型的排序是否进入前 K；
- `MRR`：严格 gold 全湖 rank 的倒数均值。

正式汇总使用 root macro：

$$
\operatorname{root\text{-}metric}=
\frac{1}{|R_{test}|}\sum_{r\in R_{test}}
\frac{1}{|D_r|}\sum_{d\in D_r}\operatorname{metric}(d).
$$

这样不会让 `tatoeba` 或 MASSIVE 这类多语言大 root 因节点多而主导平均值。

### 5.3 gold-gap：允许“距离最好不超过 1 个百分点”

公开记录中存在大量并列或近似并列结果。定义 $\delta=0.01$：

$$
\operatorname{near}_{\delta}(d)=
\{m:a(d,m)\ge a(d,g_d)-\delta\},
$$

$$
r_d^{\mathrm{gap}}=
\min_{m\in\operatorname{near}_{\delta}(d)}r_d(m),
\qquad
\operatorname{gold\text{-}gap@K}=
\frac{1}{|D|}\sum_d\mathbf 1[r_d^{\mathrm{gap}}\le K].
$$

因为严格 gold 一定属于 near set，所以 $r_d^{\mathrm{gap}}\le r_d(g_d)$。它是产品层的补充指标，不替代严格 gold 晋升门。

### 5.4 L1L3b 的 8-seed 晋升证据

在 root-macro 正式口径下：

| 指标 | 旧 G2 | L1L3b |
|---|---:|---:|
| hit@1 | 0.299 ± 0.059 | **0.312 ± 0.037** |
| top3 | 0.472 ± 0.036 | **0.527 ± 0.023** |
| regret@1 | 0.062 | **0.055** |
| strict gold@1 | 0.009 | **0.027** |
| strict gold@10 | 0.053 ± 0.037 | **0.100 ± 0.058** |

L1L3b 在 root gold@10 上提升 89%，且 root hit@1 未付出代价，因此晋升为唯一默认训练配置。flat hit@1 上 G2 的优势主要来自少数节点极多的 root，属于构成加权差异，已保留在原审计记录中而没有隐藏。

## 6. Stage 3：冻结、导出与 HNSW

### 6.1 Freeze

`d0_freeze_export.py` 固定使用：

```text
graph       = hgraph_d0_v1.pt
config      = L1L3b
split_seed  = 0
init_seed   = 0
epochs      = 25
```

它先验证 graph hash 与记录配置，再重训。由于 54.8K 边上的 CUDA scatter 不能保证 bit-exact 重训，当前冻结门是功能性门：五个指标相对记录值的最大绝对偏差必须不超过 0.02。保存时 checkpoint 同时绑定 family/task vocab 与复现元数据。

### 6.2 Export

导出执行确定性全图 forward，并保证：

- `row i == mappedID i`；
- $z_m,z_d$ 均为 float32 单位向量；
- 连续两次 forward 位级一致；
- 20 个 spot-check 全通过；
- graph、checkpoint、vocab、所有主 export sidecar 都由 manifest 记录 SHA-256。

另有 `occupancy.npy`：对 1,463 个 `z_d` 做 exact Top-10 时，每个模型进入 Top-10 的次数，用于后续 hub 诊断，不改变基础排序。

### 6.3 HNSW

HNSW 使用 cosine space：

$$
d_{\mathrm{HNSW}}(d,m)=1-\cos(z_d,z_m)=1-\langle z_d,z_m\rangle.
$$

当前参数：

```text
M=16, ef_construction=200, random_seed=100, build_threads=1
```

索引构建后有两个硬门：

1. 落盘再加载后，所有 `z_d` 的 k=50 labels 与 distances 必须逐位一致；
2. `ef_search=200` 时 exact-vs-HNSW recall@10 必须不低于 0.99。

当前产物为 6,271,072 bytes，构建耗时记录为 0.5228 秒；全 1,463 个查询的 fidelity 为：

| recall | 值 |
|---|---:|
| @1 | 1.000000 |
| @10 | 1.000000 |
| @50 | 0.999986 |
| @100 | 0.999966 |

这些 recall 是 ANN 对 exact MIPS 的**工程忠实度**，不是推荐语义准确率。

## 7. Stage 4：S2/P2b 最终融合排序

### 7.1 先验 sidecar

`build_prior_sidecar.py` 生成：

```text
prior_sidecar.npz
├── edge_model    [54795]
├── edge_dataset  [54795]
├── edge_acc      [54795]
├── root_id       [1463]
└── task_id       [1463]
```

构建时断言 model/dataset row order 与 export CSV 完全一致。sidecar 不改变索引，只给 serving 排序提供湖内可见性能、root 与 task 映射。

### 7.2 MIPS 归一化

当前实现先对本次排序范围内所有模型做 min-max：

$$
\widetilde s(d,m)=
\frac{s(d,m)-\min_j s(d,j)}
{\max_j s(d,j)-\min_j s(d,j)+10^{-9}}.
$$

week8 的结果是在全 9,491 模型范围归一化和排序。若未来真正只对 HNSW candidate pool 重排，min/max 的集合也必须固定写入 serving contract，否则分数和次序可能漂移。

### 7.3 S2 sibling prior

对查询 $d$，仅使用同 root、不同节点的湖内可见标签：

$$
S(d)=\{d':r(d')=r(d),\ d'\ne d,\ d'\text{ lake-visible}\}.
$$

$$
b_{\mathrm{sib}}(d,m)=
\begin{cases}
\frac{1}{|S_m(d)|}\sum_{d'\in S_m(d)}a(d',m),&|S_m(d)|>0,\\
0,&|S_m(d)|=0,
\end{cases}
$$

其中 $S_m(d)=\{d'\in S(d):(d',m)\text{ 有观测}\}$。查询数据集自身 $d$ 被显式排除。

S2 的离线场景是 `cold-D, warm-siblings`：查询节点自身所有边 held out，但同 root 的已知 sibling 可以留在训练侧。这与 root-aware 的“整根全新”场景不同。8-seed sibling-rich root-macro 结果：

| 指标 | 纯 MIPS | S2，$\alpha=1$ |
|---|---:|---:|
| strict gold@10 | 0.162 | **0.266** |
| strict gold@1 | 0.038 | **0.122** |

### 7.4 P2b task prior

对 task $t(d)$，使用同 task、不同数据集节点的观测。当前 serving 代码使用固定收缩中心 $\mu_0=0.5$、强度 $\kappa=5$：

$$
b_{\mathrm{task}}(d,m)=
\begin{cases}
\dfrac{\sum_{d'\in T_m(d)}a(d',m)+\kappa\mu_0}
{|T_m(d)|+\kappa},&|T_m(d)|>0,\\
0,&|T_m(d)|=0,
\end{cases}
$$

$$
T_m(d)=\{d':t(d')=t(d),\ d'\ne d,\ (d',m)\text{ 有湖内观测}\}.
$$

这里同样不会把“完全没在同任务出现过的模型”凭空赋成 0.5；只有 $|T_m(d)|>0$ 才产生 boost。

离线 `p2b_task.py` 的 8-seed 实验用训练集所有可见 accuracy 的全局均值作为 $\mu_0$，而 standalone serving 使用固定 0.5。两者公式结构一致，但中心值并非严格同一实现；比较实验值时必须保留这一口径差异。

### 7.5 统一最终分数

$$
s_{\mathrm{fused}}(d,m)=
\widetilde s(d,m)
+\alpha b_{\mathrm{sib}}(d,m)
+\beta b_{\mathrm{task}}(d,m),
\qquad \alpha=\beta=1.
$$

8-seed node-level root-macro 的主要结果：

| 分层 | 纯 MIPS gold@10 | S2+P2b gold@10 |
|---|---:|---:|
| all | 0.137 | **0.219** |
| has sibling | 0.158 | **0.261** |
| no sibling | 0.121 | **0.180** |

退化规则无需分支特判：

- 无 sibling 证据：$b_{\mathrm{sib}}=0$，由 task + MIPS 排序；
- 无 same-task 证据：$b_{\mathrm{task}}=0$，由 sibling + MIPS 排序；
- 两者都无：$s_{\mathrm{fused}}=\widetilde s$，即纯 MIPS 的同序变换。

## 8. 查询路径与输出契约

### 8.1 warm HNSW 查询

适用于 dataset 已在 export 中的情况。`query.py` 读取已有 `z_d[mappedID]`，通过 manifest 验证的 HNSW 查询，输出：

- rank、model mappedID、model ID、cosine；
- 可选 `hub_occupancy`；
- query mode、`ef_search`、requested/effective K；
- index/checkpoint hash provenance。

默认 Stage-4 handoff K 为 200，硬上限 500；CLI `query.py` 的默认 K 为 50。

### 8.2 warm fused 查询

`serving_rerank.py` 读取同一 export 和 prior sidecar，输出：

```text
rank, mappedID, unique_model_id,
fused, minmax_mips, sibling_boost, task_boost
```

它不改变 embedding 或 index。当前实现的关键事实是：

```python
mips = z_m @ z_d[d]
order = np.argsort(-fused)[:k]
```

也就是说它目前对全湖 exact score 排序；文件中的 `POOL = 512` 尚未被实际使用，也没有调用 `query.py`/`ModelRetriever`。因此它是**正确、可复现的全量 warm re-ranker**，但还不是低延迟的“HNSW 召回 512 → 融合精排”在线接线。

### 8.3 week8 推荐表的真实性与口径

`weeks/week8_Updated/updated_recommendations.md` 中四组新/旧式表确实来自当前 `serving_rerank.fused_rerank` 的实际输出，而不是手写示例。其口径是：

| 表格列 | 真正含义 |
|---|---|
| 新推荐 | 当前 D0/L1L3b export，`alpha=1, beta=1` |
| 旧式/plain-MIPS | **同一个** D0/L1L3b export，`alpha=0, beta=0` |
| `acc` | 排序完成后，从 query 自身的公开 observation 关联出来用于展示；不进入 prior |
| `—` | 该模型没有 query 数据集的 exact public record，不代表实测为 0 |

所以这些表可以支持“融合相对同一现代 embedding 的纯 MIPS 改变了什么”，但 `alpha=beta=0` **不是历史 2K/G2 checkpoint 的逐项复跑**。历史 G2 的 amazon 错类输出来自旧 G2 index 的独立命令和产物。

四个 warm 样例的冻结证据量如下：

| dataset | exact public labels | recorded best | 新排序主要现象 |
|---|---:|---:|---|
| `amazon_massive_intent/en` | 724 | 0.861 | Top-10 均为 0.854 的 gte-Qwen2 变体，距 best 0.007 |
| `banking77/default` | 761 | 1.000 | `BERT-Banking77` 以 0.928 到 Rank 1 |
| `conll2003/conll2003` | 200 | 1.000 | CoNLL/SpanMarker 专用模型前移；Top-10 中 2 个 exact label 为 0.991 |
| `squad_v2/squad_v2` | 42 | 0.888 | 新旧都大体对题；融合主要重排 SQuAD-v2 专用模型，不能宣称新 #1 已被实测证明优于 0.888 模型 |

## 9. 端到端复现

以下命令从 `D:\research\model_lake\codes` 执行。大规模训练和重建会覆盖/新增 artifact，先确认是否确实需要重建；只复现推荐无需执行 9.1–9.3。

### 9.1 用当前缓存构建下一版 D0 intake 与图

该步骤只读本地 HF cache，不主动联网。**它不能用于 bit-for-bit 复原当前 9,491-model serving graph**：本地 intake/cache 晚于冻结 graph，直接执行会构建一张新快照并改变 graph hash，随后必须重新训练、导出、建索引和建 sidecar。若目标只是复现现有推荐，请跳过本节并直接使用 manifest 绑定产物。

```powershell
Set-Location D:\research\model_lake\codes\ModelLakeFishing\stage1BuildTransferGraph
$py = '..\.venv\Scripts\python.exe'

& $py .\d0_intake_audit.py
& $py .\d0_build_graph.py
```

核对：

```powershell
Get-Content .\artifacts\d0_lake\d0_graph_report.json
```

当前主图命令**不要**加 `--content`；带 content 的 v5 图已经被实验否决，不是现行 graph。

### 9.2 复跑 L1L3b 晋升评测

```powershell
Set-Location D:\research\model_lake\codes
$py = '.\ModelLakeFishing\.venv\Scripts\python.exe'

& $py -m ModelLakeFishing.stage2TrainGraphSAGE.w1_dzero `
  --rows G2 L1L3b `
  --seeds 0 1 2 3 4 5 6 7 `
  --tag current_rebuild
```

这是 16 次训练，成本高。当前正式 8-seed 证据已在：

```text
stage2TrainGraphSAGE/artifacts/ablation/d0/
├── w1_baselines/
└── w1_ext/
```

### 9.3 冻结、导出、建索引与 sidecar

```powershell
Set-Location D:\research\model_lake\codes
$py = '.\ModelLakeFishing\.venv\Scripts\python.exe'

# freeze -> export -> HNSW；内部带 functional freeze 与 fidelity gates
& $py -m ModelLakeFishing.stage3HNSW.d0_freeze_export

# S2/P2b 需要的湖内先验 sidecar
& $py -m ModelLakeFishing.stage3HNSW.build_prior_sidecar `
  --graph ModelLakeFishing/stage1BuildTransferGraph/hgraph_d0_v1.pt `
  --export ModelLakeFishing/stage3HNSW/artifacts/exports/d0_L1L3b
```

### 9.4 复现 HNSW warm 查询

```powershell
& $py -m ModelLakeFishing.stage3HNSW.query `
  --index ModelLakeFishing/stage3HNSW/artifacts/indexes/d0_L1L3b `
  --dataset amazon_massive_intent/en `
  --mode warm --k 10 --ef-search 64
```

### 9.5 复现最终融合与 plain-MIPS 对照

```powershell
$export = 'ModelLakeFishing/stage3HNSW/artifacts/exports/d0_L1L3b'

# 最新最终排序
& $py -m ModelLakeFishing.stage3HNSW.serving_rerank `
  --export $export --dataset amazon_massive_intent/en `
  --alpha 1 --beta 1 --k 10

# 同一新 embedding 上关闭两个先验，仅作 plain-MIPS 对照
& $py -m ModelLakeFishing.stage3HNSW.serving_rerank `
  --export $export --dataset amazon_massive_intent/en `
  --alpha 0 --beta 0 --k 10
```

复现 week8 全部四组：

```powershell
$datasets = @(
  'amazon_massive_intent/en',
  'banking77/default',
  'conll2003/conll2003',
  'squad_v2/squad_v2'
)

foreach ($dataset in $datasets) {
  & $py -m ModelLakeFishing.stage3HNSW.serving_rerank `
    --export $export --dataset $dataset --alpha 1 --beta 1 --k 10
  & $py -m ModelLakeFishing.stage3HNSW.serving_rerank `
    --export $export --dataset $dataset --alpha 0 --beta 0 --k 10
}
```

### 9.6 复跑 S2/P2b 8-seed 研究证据

这两条命令会分别重训 8 次，不是查询命令：

```powershell
& $py -m ModelLakeFishing.stage2TrainGraphSAGE.s2_sibling
& $py -m ModelLakeFishing.stage2TrainGraphSAGE.p2b_task
```

## 10. 当前已知边界与尚未闭合的接口

这些边界是当前代码事实，不应在产品说明中省略。

### 10.1 HNSW 与 fusion 尚未在一个函数内接线

目标在线形态应为：

$$
C_d=\operatorname{HNSW}(z_d,K_{pool}),\qquad
\operatorname{TopK}_{m\in C_d}s_{\mathrm{fused}}(d,m).
$$

`serving_rerank.py` 的注释给出 `POOL=512`，但当前没有使用它，实际是 $C_d=\mathcal M$。这保证 week8 排名是 exact/full-lake，但不代表 9,491 规模之外的低延迟服务已完成。

### 10.2 D0 cold query 当前不可直接运行

`query.py` 的 cold path 从 manifest 所绑定 graph 的 `xd0_meta["view_dims"]["e_domain"]` 读取 cold-neighbor feature slice；当前 D0 graph 只有：

```text
e_name=64, e_card=384, e_stats=10
```

没有 `e_domain` 键。因此当前 D0 export 的 `--mode cold` 会在加载 cold context 时失败。并且 `serving_rerank.py` 只接受 export 中已有的 dataset ID，没有外部新数据集的特征构建入口。故当前可承诺的是：

- warm HNSW 可运行；
- export 内数据集的 warm exact fusion 可运行；
- “任意全新外部数据集 → cold embedding → HNSW → S2/P2b”尚未成为一条可执行主链。

闭合该接口时应明确选择 D0 的 deployable cold 相似度 view（最自然候选是 `e_card`），同时为新 query 提供 root/task 映射，并补 D0 cold 集成测试；在代码修复前不应把旧图上的 cold-path 测试等同于 D0 production support。

### 10.3 prior sidecar 的 hash binding 弱于主 export

`build_prior_sidecar.py` 在生成时验证 mappedID 顺序，但当前 `manifest.json` 的 `files` 字段没有记录 `prior_sidecar.npz` 和其 meta 的 hash，`load_serving()` 也不复验 hash。主 index/export 的 hash 链完整，先验 sidecar 目前只靠同目录与构建时断言。正式上线前应把 sidecar 加入 manifest binding。

### 10.4 没有显式语言/任务硬过滤器

当前最终排序是连续分数融合，不是 rule-based language firewall。task prior 能显著减少跨任务漂移，sibling prior 能强化同基准专用模型，但系统没有单独的语言 hard constraint、license/size/format constraint 或 popularity filter。week8 的强结果不能外推为“所有查询 100% 不会跑偏”。

### 10.5 public score 的缺失不是负标签

未在 query 数据集上出现的模型仍可被推荐；它们的 `acc` 为未知，不是 0。gold@K 只追踪已知 gold 在全湖排序中的生存，不等价于 full-lake precision@K。对于 top-ranked 但无 exact score 的模型，结论只能是“metadata/lineage/先验上相关”，不能写成“实测最好”。

## 11. 被否决的实验分支

当前主线经过逐项 ablation，而非把所有实验功能叠加：

| 分支 | 结论 |
|---|---|
| F1–F4 模型侧 feature rework | 未成为主要增益来源 |
| Z1/Z2 query compression / geometry constraint | 否决；表示问题主要是目标函数的影子 |
| L2/L2b hard-negative mining | 否决；存在自我强化/伤害 hidden gold 的风险 |
| L4 positive IPW | 否决 |
| P2 degree cap | 8-seed root-macro 中性略负，不采用 |
| v5 sample content | 覆盖约 21%，未改善目标分层，不采用 |
| S1 training-side root pooling | 与 S2 叠加有副作用，不采用 |
| S2 serving sibling fusion | 采用 |
| P2b serving task fusion | 采用并与 S2 叠加 |

整体经验是：真正有效的杠杆是**更完整且覆盖全湖的监督目标，以及 serving 时外科式加入的合法先验**；不是无差别加深 GNN 或堆更多 feature。

## 12. 代码与证据索引

### 主代码

| 阶段 | 文件 |
|---|---|
| D0 intake | `stage1BuildTransferGraph/d0_intake_audit.py` |
| D0 graph | `stage1BuildTransferGraph/d0_build_graph.py` |
| root-aware split | `stage2TrainGraphSAGE/d0_splits.py` |
| GNN | `stage2TrainGraphSAGE/model.py`、`edge_aware.py` |
| losses | `stage2TrainGraphSAGE/losses.py`、`train.py` |
| L1L3b config/eval | `stage2TrainGraphSAGE/w1_dzero.py` |
| gold/gold-gap | `stage2TrainGraphSAGE/top1_eval.py` |
| S2/P2b studies | `stage2TrainGraphSAGE/s2_sibling.py`、`p2b_task.py` |
| freeze/export/index | `stage3HNSW/d0_freeze_export.py`、`export_embeddings.py`、`build_index.py` |
| HNSW query | `stage3HNSW/query.py`、`retrieval_service.py` |
| prior/fusion | `stage3HNSW/build_prior_sidecar.py`、`serving_rerank.py` |

### 冻结产物

```text
stage1BuildTransferGraph/hgraph_d0_v1.pt
stage1BuildTransferGraph/artifacts/d0_lake/d0_graph_report.json
stage2TrainGraphSAGE/artifacts/ablation/d0/ckpt/L1L3b_s0_i0.pt
stage3HNSW/artifacts/exports/d0_L1L3b/manifest.json
stage3HNSW/artifacts/exports/d0_L1L3b/prior_sidecar.npz
stage3HNSW/artifacts/indexes/d0_L1L3b/index_manifest.json
```

### 关键审计记录

```text
docs/optimization/v3/L_TRACK_AUDIT.md
docs/optimization/v4/W1_D0_AUDIT.md
docs/optimization/v4/W2_AUDIT.md
docs/optimization/v4/W3_STAGE3_AUDIT.md
docs/optimization/v5/P4_GOLDGAP_METRIC.md
docs/optimization/v6/S2_SIBLING_FUSION.md
docs/optimization/v6/P2b_TASK_FUSION.md
docs/optimization/v6/SERVING_INTEGRATION.md
```

### week8 展示层

```text
D:\research\model_lake\weeks\week8_Updated\progress_and_results.md
D:\research\model_lake\weeks\week8_Updated\updated_recommendations.md
```

`progress_and_results.md` 是面向汇报的结果叙事；`updated_recommendations.md` 是四个 warm 查询的实际推荐快照。二者不替代 manifest、代码与审计 JSON；当描述冲突时，应按“代码 → manifest/artifact → audit → 周报”的顺序裁决。
