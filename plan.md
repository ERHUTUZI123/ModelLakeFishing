# xm0_builder 构建指南

目标：把模型节点特征从随机占位换成真实的四分量 $\mathbf{x}_m^{(0)} = [\mathbf{e}_m^{\text{name}} \| \mathbf{e}_m^{\text{desc}} \| \mathbf{e}_m^{\text{size}} \| \mathbf{e}_m^{\text{fam}}]$，让 GraphSAGE 有真实的节点特征可聚合。

---


**第零步：先确认接口契约（最重要的对齐点）**

在动手之前，先看清楚 `HGraph` 现在怎么接收模型特征。在你 attach 的测试里，`model_features=[]` 且 `contain_model_feature=False`，于是 `HGraph` 内部用 `torch.rand` 生成随机特征。

`xm0_builder` 的唯一职责就是产出一个能替换那个随机张量的东西：一个形状为 `[num_models, feature_dim]` 的矩阵，**行顺序严格对应 `unique_model_id` 的 mappedID 顺序**。这是整条链路的契约——第 i 行必须是 mappedID 为 i 的那个模型的特征。任何顺序错位都会让每个模型的特征张冠李戴，而且不报错，是最危险的 bug。所以 builder 的输入必须是 `unique_model_id` 这张表，输出按它的行序排列。

---

**第一步：数据从哪里来**

四个分量有两个不同的数据源。

name 和 fam 可以直接从模型名字字符串里得到——模型名（如 `google/gemma-4-31B-it`）你已经有了，就在 `unique_model_id['model']` 这一列里，不需要额外抓取。

desc 和 size 需要从 HuggingFace 抓取——desc 来自 model card 的 README 文本，size 来自模型的参数量元数据。这一步要在 `attributes.py` 之外，写一个独立的抓取脚本（和你抓 lineage 的 `base_model` 那个脚本平级），把每个模型的 README 和参数量抓下来存成一张 CSV，结构类似：`model`、`description_text`、`param_count`。这张表和 lineage 表一样，是 builder 的"原始输入"。

---

**第二步：四个分量分别怎么算**

**name 分量。** 把模型名喂进一个轻量句子编码器（sentence-transformers 那类），得到一个固定维度向量。这是 frozen 的，算一次就不变。要决定的设计点：整名编码还是按分隔符切分后编码——先看实际输出再定。

**desc 分量。** 把 README 文本喂进同一个编码器，得到 frozen 向量。这是四个分量里最脏的一个，边界情况最多：README 为空、过长（要截断）、非英文。每种情况都要有一个一致的兜底策略。这一步也是整个 builder 里最耗时的离线步骤，必须做断点续跑和缓存，否则中断一次重头来代价极高。

**size 分量。** 参数量先做 log10 变换（因为跨四个数量级），再离散化成若干 bucket，每个 bucket 对应一个**可学习**的 embedding。注意这里和 name/desc 不同——size 是 learnable 的，不是 frozen 的。参数量缺失的模型统一进一个 "unknown" bucket。

**fam 分量。** 从模型名字和 tags 用规则表推断家族（含 `llama`→LLaMA 系，含 `qwen`→Qwen 系等），每个家族对应一个**可学习**的 embedding。识别不出的进 "Other"。同样是 learnable。

---

**第三步：frozen 和 learnable 必须分开存储（关键架构决定）**

这是和 lineage 那套流程最不一样的地方，要特别注意。

name 和 desc 是 frozen 的——预训练编码器算出来就固定，不参与梯度。size 和 fam 是 learnable 的——它们是 embedding 查找表，训练时会更新。

这意味着 builder 的产出不是一个拼好的完整矩阵，而应该分两部分：frozen 部分（name+desc）离线算好存成文件，直接当作固定特征；learnable 部分（size bucket id 和 family id）只需要存**离散的索引**（第几个 bucket、第几个家族），真正的 embedding 向量在 GraphSAGE 训练时才动态查表生成。

如果你把 size 和 fam 也提前算成固定向量塞进特征矩阵，它们就变成 frozen 的了，失去了"可学习"的意义——这违背了我们设计 $\mathbf{e}_m^{\text{size}}$ 和 $\mathbf{e}_m^{\text{fam}}$ 作为可训练 embedding 的初衷。所以这一步的产出实际上是：一个 frozen 特征矩阵 + 两列离散索引（size_bucket_id、family_id）。

---

**第四步：维度对齐**

name 和 desc 如果用同一个编码器，维度一致；如果用不同编码器，拼接前要先投影到统一维度。整个 $\mathbf{x}_m^{(0)}$ 的最终维度会一路传到 GraphSAGE，影响计算量，要心里有数控制在合理范围（参考你数据集特征的维度量级，别让模型侧维度爆炸性地大于数据集侧）。

---

**第五步：接进 HGraph**

现有 `HGraph` 在 `contain_model_feature=True` 时会读传进来的 `model_features`。所以你要做两件事：把 `contain_model_feature` 打开；把 builder 产出的 frozen 特征矩阵传进 `model_features` 参数。

但 learnable 的 size/fam 索引不能走 `model_features` 这个口子——它走的是固定特征的路径。size/fam 的 embedding 层需要在 GraphSAGE 模型内部定义，训练时用索引查表，再和 frozen 特征拼接。所以这一步实际上跨越了"建图"和"建模型"两个阶段：frozen 部分进图作为节点特征，learnable 部分的索引要作为额外信息带到模型定义里。这是你接下来写 GraphSAGE 时要接的口子，builder 现在只需要把这两列索引准备好。

---

**第六步：最小验证**

仿照你 lineage 的 sanity check 思路。在小图（那 3 个模型）上：

确认 frozen 特征矩阵的行数等于模型数，行序和 `unique_model_id` 的 mappedID 对得上（抽查第 0 行是不是 mappedID=0 那个模型的特征）。确认 size_bucket_id 和 family_id 两列没有意外的 NaN，缺失值都正确落进了 unknown/Other。确认维度符合预期。

这一步只验证 builder 的产出本身正确，不涉及训练——和你验证 lineage 边"确实进了图"是同一层级的检查。

---

**整条链路对照一下你已经做过的 lineage 流程**

数据源：lineage 的 `base_model` 字段 → 这里的 README 文本 + 参数量（外部抓取脚本，平级）。
载入时机：lineage 在 `get_node_id` 之后载入 → 这里 builder 也依赖 `unique_model_id` 已生成。
核心操作：lineage 是双重 merge 建边 → 这里是四分量编码 + 拼接建节点特征。
关键区别点：lineage 的关键是"权重不归一化" → 这里的关键是"frozen 与 learnable 分开存储"。
接进图：lineage 注册 `is_base_of` 边 → 这里 frozen 特征走 `model_features` 进图，learnable 索引留给模型定义。

每一步都有你 lineage 流程的对应物。最该警惕的两个点：行序对齐（第零步）和 frozen/learnable 分离（第三步）——前者错了会静默地张冠李戴，后者错了会让两个本该训练的 embedding 退化成固定值。

# Model Feature Embeddings — $\mathbf{x}_m^{(0)}$

$$\mathbf{x}_m^{(0)} = [\mathbf{e}_m^{\text{name}} \| \mathbf{e}_m^{\text{desc}} \| \mathbf{e}_m^{\text{size}} \| \mathbf{e}_m^{\text{fam}}]$$

---

## $\mathbf{e}_m^{\text{name}}$

**来源**：`module/model/modelNameEncoder.py:6–122` — `ModelNameAvgEncoder.forward()`

| 步骤 | 操作 |
|------|------|
| 1 | 模型名按 `-` / `_` / `/` / 空格切词 |
| 2 | 每个 token 用 MD5 hash → 映射到 10000-bucket |
| 3 | 查 learned `tok_emb: nn.Embedding(10000, token_dim)` |
| 4 | 所有 token embedding 取平均 → `[B, token_dim]` |

---

## $\mathbf{e}_m^{\text{id}}$

**来源**：`module/model/MLP.py:688–693` — `self._id_emb`

```python
self._id_emb = nn.Embedding(num_models + 1, model_dim)
```

- 直接 learned embedding，按 `model_id` 查表
- 训练时以 `model_id_dropout_rate` 概率替换为 `[UNK]`，强迫模型依赖 name/desc/size 信号以支持 zero-shot 泛化
- 与 $\mathbf{e}_m^{\text{name}}$ 一同在 `encode_model()` 内拼接后输出

---

## $\mathbf{e}_m^{\text{desc}}$

**来源**：`module/model/MLP.py:67–101` (`_build_desc_matrix`) + `MLP.py:706–713`

```python
self.register_buffer("model_desc_matrix", model_desc_matrix)  # frozen
```

- 从外部 `.npz` 文件预加载（字段：`model_names` + `embeddings`）
- 预计算文本 embedding，维度 1536（likely OpenAI / sentence-transformer）
- 注册为 `register_buffer`，**不参与反向传播**
- 推理时按 `model_id` 直接行索引 → `[B, 1536]`

---

## $\mathbf{e}_m^{\text{size}}$

**来源**：`module/model/MLP.py:116`

```python
self.size_embedding = nn.Embedding(num_size_buckets, size_dim)
```

- 模型参数量预先离散化为若干 bucket，编为整数 `size_id`
- 按 `size_id` 查 learned embedding → `[B, size_dim]`

---

## $\mathbf{e}_m^{\text{fam}}$

**来源**：`module/model/MLP.py:123–126`

```python
self.family_embedding = nn.Embedding(num_families, family_dim)
```

- 模型所属 family（如 LLaMA、Mistral 等）预先编为整数 `family_id`
- 按 `family_id` 查 learned embedding → `[B, family_dim]`

---

## 拼接位置

| 阶段 | 位置 | 内容 |
|------|------|------|
| `encode_model()` | `MLP.py:879–911` | `name \|\| id \|\| desc` → `h_model` |
| `forward()` | `MLP.py:950–964` | `h_model \|\| h_size \|\| h_family \|\| ...` → `residual_inp` |
