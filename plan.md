# GraphSAGE Training 构建指南

目标：在已建好的 HGraph 上训练一个 inductive 的异构 GraphSAGE，产出模型节点 embedding $\mathbf{z}_m$（供 HNSW 索引）和数据集节点 embedding $\mathbf{z}_d$（查询时编码目标数据集），训练目标为 $\mathcal{L} = \lambda_{\text{perf}}\mathcal{L}_{\text{perf}} + \lambda_{\text{contrast}}\mathcal{L}_{\text{contrast}}$。

---

**第零步：先确认接口契约（和 xm0 阶段同一条铁律）**

训练阶段从上一阶段接收三样东西，缺一不可：

1. `HGraph.data` — `data['model'].x` 是 frozen 的 `[e_name || e_desc]` 矩阵；`data['model'].size_bucket_id` 和 `data['model'].family_id` 是两列离散索引，挂在节点上随采样自动切片；四类边（含 ToUndirected 自动生成的反向边）。
2. `xm0` 字典里的元信息 — `num_size_buckets`、`num_families`、`family_vocab`。前两个是 embedding 表的行数，最后一个是行身份的唯一凭证。
3. `ModelNodeEncoder` — 这是 xm0 阶段留给你的口子，训练时完成 $\mathbf{x}_m^{(0)} = [\text{frozen} \| \mathbf{e}_m^{\text{size}} \| \mathbf{e}_m^{\text{fam}}]$ 的最终拼接。

输出侧的契约和输入侧对称：最终导出的 $\mathbf{z}_m$ 矩阵第 i 行必须是 mappedID 为 i 的模型——HNSW 索引、reranker、评测全都依赖这个行序，错位不报错但全盘皆错。

---

**第一步：模型结构**

三段式：节点编码 → 异构消息传递 → 输出头。

**节点编码层。** model 侧就是 `ModelNodeEncoder`（输出 128 维量级）；dataset 侧把 probe embedding 过一个线性投影。两类节点的初始维度不必相等（异构卷积允许每类节点不同输入维度），但都要投到同一个隐层维度再进消息传递，否则后面每层都要维护两套尺寸。

**消息传递层。** 2 层 GraphSAGE（先别更深——hub 周围的衍生模型本来就容易 over-smoothing，层数是最直接的旋钮）。异构化方式用 PyG 的 to_hetero 或 HeteroConv，让四类边各自有独立的聚合权重——lineage 边和 performance 边语义完全不同，绝不能共享一套 SAGE 权重。

**输出头。** 一个投影到最终 embedding 维度 + L2 归一化（HNSW 用余弦/内积，归一化让训练目标和检索度量一致）。

整个模型里**不允许出现任何按 mappedID 查表的 ID embedding**——这是 inductive 的底线，破了它新模型就无法零样本接入，我们对 ModelLens 的扩展性优势就没了。

---

**第二步：learnable embeddings 怎么用（本阶段最关键的衔接点）**

xm0 阶段把 size 和 family 存成离散索引，就是为了在这里兑现"可学习"。具体用法：

**归属。** `size_embedding` 和 `family_embedding` 两张 `nn.Embedding` 表住在 `ModelNodeEncoder` 里，而 encoder 是 GraphSAGE 模型的子模块——所以它们自动进同一个 optimizer 的参数组，和 SAGE 卷积权重一起被梯度更新。不需要也不应该为它们单独建 optimizer 或设特殊学习率（先用默认，有证据再调）。

**前向。** 每个 batch 里，采样子图的 model store 自带切片好的 `size_bucket_id` / `family_id`（这就是当初把索引挂在节点上的原因），encoder 查表、拼接、送进第一层卷积。frozen 的 x 原样通过，不在任何参数组里，永远收不到梯度——最小测试里的梯度检查验证的就是这条边界。

**反向。** 损失的梯度沿 GNN 一路传回两张表。注意梯度是稀疏的：一个 batch 只更新被采到的 bucket/family 行。由此有两个要盯的点：(i) unknown bucket（id 0）和 Other family（id 0）是高频行，会被更新得最多，它们学到的是"缺失本身的先验"，这是特性不是 bug；(ii) 稀有 bucket/family 的行更新极少，训练末期检查这些行的范数，如果和初始化几乎没动，说明它们实际上没学到东西——这正是当初设 FAMILY_MIN_COUNT 门槛的原因，门槛之下进 Other 比留一行噪声好。

**保存。** checkpoint 必须把 encoder 权重和 `family_vocab.csv` 绑在一起存——vocab 文件是 embedding 行身份的唯一凭证，丢了它，训练好的 family 行就成了无主孤儿（xm0 阶段反复强调过，这里是真正兑付的地方）。size 表不需要 vocab（bucket 边界是固定常数），但 bucket 常数变了同样要求重训。

**推理 / 零样本新模型。** 新模型接入时：frozen 部分由 builder 离线算出；size/family 走同一套规则得到索引——参数量缺失进 unknown bucket，家族不在 vocab 里进 Other。然后用**训好的**表查出向量拼接，过**训好的** GNN 得到 $\mathbf{z}_m$。新家族只有积累到门槛、vocab 扩行、表扩行并增量训练之后才有自己的行——在那之前它就是 Other，这是设计内的降级而非故障。

---

**第三步：两个损失分别监督什么**

**$\mathcal{L}_{\text{perf}}$（性能回归）。** 在 `trained_on` 边上做监督：用 $\mathbf{z}_m$ 和 $\mathbf{z}_d$ 的打分（点积或小 MLP）回归归一化后的 accuracy 边权。这是主任务，让 embedding 空间编码"谁在哪类数据上表现好"。边的 train/val/test 切分用 HGraph 已有的 split（RandomLinkSplit），注意把对应的反向边类型一并交给 split 处理，否则反向边会把测试标签泄漏回训练图。

**$\mathcal{L}_{\text{contrast}}$（任务结构对比）。** 正样本对：在同一数据集/任务上都表现好的模型；负样本：不同任务的模型，并且**刻意加重同 hub 衍生模型之间的负样本**——这是对抗 over-smoothing 的主动手段，否则 lineage 边会把一个家族的几十个衍生模型聚成一个点，HNSW 在 hub 附近就废了。这一项注入任务结构，是单索引下 HNSW 具备 task-sensitivity 的全部来源。

$\lambda$ 配比从 1:1 起步，看两个损失的下降曲线再调，不要预先精调。

---

**第四步：采样与 edge dropout**

用 LinkNeighborLoader 围绕监督边采子图。两列索引和 frozen x 都会随子图自动对齐切片，不需要任何手工 gather——如果你发现自己在写按全局 id 查索引的代码，说明走错路了。

edge dropout 在训练时随机丢弃输入图的边（监督边除外），模拟长尾模型的稀疏邻域，强迫模型在邻居缺失时仍能从节点自身特征（恰恰是四分量 $\mathbf{x}_m^{(0)}$）恢复信号。这是我们对 ModelLens ID-dropout 的结构等价物：他们丢 ID 强迫依赖语义特征，我们丢边强迫依赖节点特征，目的相同、机制对偶。lineage 边建议单独设较低的 drop 率——它是冷启动模型唯一的边，丢光了等于自废卖点。

---

**第五步：训练循环与验证顺序**

先在小图（3 模型的 mini 图）上跑通**机制**：损失能降、两张表的梯度非零、frozen x 无梯度、checkpoint 能存能载。再上 TransferGraph zoo（348 模型）跑通**效果**，按 CLAUDE.md 的清单做 sanity check：

UMAP 按任务着色看聚类；同 hub 衍生模型两两余弦距离——若全部趋零即 over-smoothing，回头加重对比损失或减层；HNSW recall@50 对暴力检索 > 90%，hub 附近和远离 hub 分开测；held-out 边上的 $\mathcal{L}_{\text{perf}}$ 与 Kendall's τ。

全部通过之前不碰 47K 模型的 ModelLens benchmark。

---

**第六步：产出物清单**

训练结束要落盘的东西，按"重建推理链路需要什么"列全：GNN 权重（含 encoder 两张表）、`family_vocab.csv`、size bucket 常数（在代码里，记录版本即可）、name 编码的 seed 和 token_dim、desc 编码器名称、最终 $\mathbf{z}_m$ 矩阵（mappedID 行序）+ 对应的 `unique_model_id` 快照。少任何一样，三个月后的你都无法复现今天的索引。

---

**整条链路对照 xm0 阶段**

接口契约：xm0 是 mappedID 行序进 → 这里是 mappedID 行序出（$\mathbf{z}_m$）。
关键分离：xm0 是 frozen/learnable 分开存储 → 这里是 frozen 无梯度通过 / learnable 入参数组更新。
身份凭证：xm0 写下 family_vocab → 这里和 checkpoint 绑定保存。
冷启动伏笔：xm0 留 unknown/Other 兜底行 → 这里靠 edge dropout 把兜底行训练成有用的先验。
最小验证：xm0 验证产出本身 → 这里验证梯度边界 + 嵌入空间结构。

最该警惕的三个点：输出行序错位（静默全错）、vocab 文件与 checkpoint 脱钩（learnable 行成孤儿）、over-smoothing（lineage 边的副作用，靠对比损失里的同 hub 负样本压制）。
