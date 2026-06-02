好，我按 TransferGraph 已有的数据流，从头到尾把 lineage 边的全流程讲一遍，每一步都对照现有代码里对应的位置，让你能看出它是怎么嵌进去的。

---

**第一步：数据从哪里来**

TransferGraph 现有的数据入口在 `get_finetuned_records` 里，从 `records.csv` 读 fine-tune 历史。lineage 信息不在这个文件里，所以你需要一个**新的数据源**——每个模型的 `base_model` 字段。

这个字段的来源是 HuggingFace 模型卡。你需要在数据准备阶段（`attributes.py` 之外）先把它抓下来，整理成一张 CSV，最简结构是三列：`model`（子模型名）、`base_model`（父模型名）、`relation_type`（关系类型：quantized / adapter / finetune / merge）。这张表在概念上和 `records.csv` 平级——它是 lineage 边的"原始记录表"，对应工作流第一步"Start from a records DataFrame"。

注意一个现实情况：一个 merge 模型有多个 base_model，所以这张表里同一个子模型可能出现多行，每行连向一个不同的父模型。这是正常的，不要去重成一行。

---

**第二步：数据在哪里处理、何时载入**

现有代码在 `GraphAttributes.__init__` 里调用 `get_finetuned_records` 和 `get_node_id`，先把 records 读进来，再生成 `unique_model_id`（模型名到整数 mappedID 的映射表）。

你的 lineage 表要在 `unique_model_id` **生成之后**载入，因为你需要这张映射表来把模型名翻译成节点 ID。所以合理的位置是：在 `__init__` 里 `get_node_id` 之后，加一句载入 lineage CSV，存成 `self.lineage_records`。

这里有一个关键的过滤动作要做。TransferGraph 的图里模型集合是固定的（来自 `unique_model_id`），所以你的 lineage 边只有当**子模型和父模型都在 `unique_model_id` 里**时才有意义。父模型不在图里，这条边就没有可连的目标节点。这个过滤不需要你手动写循环——它会在第四步的 merge 里用 `how='inner'` 自动完成，和现有代码丢弃孤儿边的机制完全一样。

---

**第三步：构建权重**

现有代码里，performance 边的权重来自 accuracy（连续值，做了 groupby 归一化），transferability 边的权重来自 score（连续值，做了 mean 归一化）。这些都是测量出来的连续量，所以要归一化。

你的 lineage 权重不一样——它是**按关系类型赋的有序离散值**，不是测量出来的。所以处理方式要区别对待：

建一个固定的映射表，把 `relation_type` 映射成 $r_{mm'}$。按继承强度排序：quantized 最高、adapter 次之、finetune 再次、merge 最低（比如 0.9 / 0.7 / 0.5 / 0.3）。在 lineage 表里加一列 `relation_weight`，按这个映射填进去。

关键决定：**不要**把这列权重塞进现有的 min-max 或 mean 归一化流程。那套归一化是为了处理没有绝对意义的连续测量值，而你的离散权重本身就是你设计好的、带语义的相对关系，归一化会把 0.9/0.7/0.5/0.3 的精心设计扭曲掉。这也是为什么 lineage 边要单独写一个获取方法，而不是复用 `get_model_dataset_edge_index`——后者内置了 groupby 归一化和阈值过滤，对你的离散权重不适用。

---

**第四步：名字翻译成节点 ID，构建 edge_index 和 edge_attr**

这一步完全复刻现有 `get_edges` 的双重 merge 模式，只是两端都换成 model。

现有代码里，model-dataset 边是：用 `model` 列 merge `unique_model_id` 拿模型 ID，用 `dataset` 列 merge `unique_dataset_id` 拿数据集 ID，两端来自不同的映射表。

你的 lineage 边是：用 `model` 列 merge `unique_model_id` 拿子模型 ID，再用 `base_model` 列 merge **同一张** `unique_model_id`（只是 merge 的 key 换成 base_model）拿父模型 ID。两端都来自 `unique_model_id`。两次 merge 都用 `how='inner'`，孤儿边在这一步自动被丢掉。

merge 完之后，把子模型的 mappedID 和父模型的 mappedID 用 `torch.stack` 叠成 `[2, num_edges]` 的 edge_index tensor——这和现有代码 stack 模型 ID 和数据集 ID 的写法一模一样，对应工作流第三步。权重那一列 `relation_weight` 转成 tensor 作为 edge_attr，对应第四步。

关于索引空间有一个细节要注意。在 homo 模式下，现有代码有一行 `self.unique_model_id['mappedID'] += self.max_dataset_idx + 1`，把模型索引整体平移以避免和数据集索引冲突。因为你的 lineage 边两端都是 model，两次 merge 自动拿到的就是平移后的索引，两端都对，不需要任何额外处理——这反而比 model-dataset 边简单，后者要分别处理两个不同的索引空间。hetero 模式下不平移，两端也都用原始 model 索引，同样不需要特殊处理。

---

**第五步：插入图**

现有代码在 `GraphAttributesWithDomainSimilarity.__init__` 的末尾，把三类边的 edge_index 和 edge_attr 都算好存成 `self.edge_index_xxx` 属性，然后这些属性被传进 `HGraph`，在 `HGraph.__init__` 里注册成 typed relation（比如 `self.data["model", "trained_on", "dataset"]`）。

你的 lineage 边走同样的路径：在 `__init__` 末尾调用你新写的 lineage 边获取方法，存成 `self.edge_index_lineage` 和 `self.edge_attr_lineage`；然后在 `HGraph.__init__` 里加一个新参数接收它们，注册成 `self.data["model", "derived_from", "model"].edge_index` 和对应的 edge_attr。

因为 `HGraph` 末尾会调用 `T.ToUndirected()`，你只需要存单向边（子→父），反向边会自动补上。所以建图时不用操心方向问题。

---

**第六步：最小验证**

改完先别训练。`HGraph._print()` 会打印 `self.data` 和 `metadata()`。确认输出里多了一类 `("model", "derived_from", "model")` 边，且边数和你过滤后预期的 lineage 边数量吻合。这是最快的 sanity check，确认边确实进了图，再往下走。

---

**整条链路对照一下现有代码**

数据源：`records.csv` → 你的 `lineage.csv`。
载入：`get_finetuned_records` → 你在 `__init__` 里载入 lineage 表。
ID 映射：`get_node_id` 生成的 `unique_model_id`（复用，不改）。
边构建：`get_edges` 的双重 merge → 你的 lineage 边方法，两端都 merge `unique_model_id`。
权重：accuracy/score 的归一化 → 你的离散权重映射（不归一化）。
插图：`HGraph` 注册 `trained_on` 边 → 你注册 `derived_from` 边。

每一步都有现成的对应物，你做的是平移复制加一处关键改动（权重不归一化、两端同源），没有任何一步是凭空新增的范式。