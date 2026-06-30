# Stage 2 工作流导览

这份文档面向第一次阅读 Stage 2 代码的人。它不解释公式和实现细节，只回答三个问题：

1. Stage 2 接收什么？
2. 数据按什么顺序经过哪些步骤？
3. 每一步应该看哪个 Python 文件？

---

## 1. Stage 2 到底在做什么？

Stage 1 已经把模型、数据集以及它们之间的关系做成一张异构图。Stage 2 的任务是：

> 用 GraphSAGE 把每个模型和每个数据集编码成同一向量空间中的向量，使适合某个
> 数据集的模型能通过点积/余弦相似度排在前面。

Stage 2 最重要的输出是：

- `z_m`：所有 model nodes 的向量；
- `z_d`：所有 dataset nodes 的向量；
- checkpoint：能够重新生成这些向量的模型参数与词表绑定。

未来 HNSW 会索引 `z_m`，收到数据集查询时用 `z_d` 搜索模型：

```text
dataset query z_d  --->  HNSW model index  --->  top-K model candidates
```

Stage 2 负责学好向量；HNSW 只是后续快速搜索这些向量。

---

## 2. 一张图看完整流程

```text
Stage 1 graph .pt
        |
        v
加载 HeteroData + xm0/xd0 metadata
        |
        v
可选：生成 dataset-similarity 图变体（实验路径）
        |
        v
切分 trained_on edges：train / validation / test
        |
        v
构造 ranking 与 contrastive supervision
        |
        v
创建 HeteroGraphSAGE
        |
        v
采样训练子图 + edge dropout
        |
        v
GraphSAGE forward：得到 z_m、z_d
        |
        v
计算 ranking / contrastive 等 losses
        |
        v
backprop + optimizer 更新参数
        |
        v
held-out Kendall + z_d -> z_m 检索评估
        |
        v
保存 checkpoint、实验 JSON、诊断和可视化
```

---

## 3. Stage 2 的输入

Stage 2 的输入是 Stage 1 生成的 `.pt` 图文件，例如：

```text
stage1BuildTransferGraph/hgraph_hf1000d_2000m_xm0_xd0.pt
```

加载入口在：

```text
stage2TrainGraphSAGE/model.py -> load_hgraph()
```

图文件中最重要的内容包括：

- `data`：PyG `HeteroData` 异构图；
- `xm0_meta`：model feature 相关元数据和 family vocab；
- `xd0_meta`：dataset feature 相关元数据；
- `unique_model_id`：模型行号与真实模型 ID 的映射。

`data` 主要包含两种节点：

```text
model nodes
dataset nodes
```

以及几类关系：

```text
model  --trained_on-->      dataset
dataset --rev_trained_on--> model
dataset --similar_to-->     dataset
model  --is_base_of-->      model
model  --rev_is_base_of-->  model
```

Stage 1 的构图逻辑不属于 Stage 2；如果想追溯图是怎么来的，再看：

```text
stage1BuildTransferGraph/build_graph.py
stage1BuildTransferGraph/attributes.py
stage1BuildTransferGraph/dataset_embed/utils/graph.py
```

---

## 4. 按运行顺序理解每一步

### Step 1：选择运行入口

Stage 2 目前有两条主要入口。

#### 稳定生产复现入口

```text
stage2TrainGraphSAGE/train_diverse.py
```

它负责：

- 读取命令行参数；
- 加载图；
- 创建模型和训练配置；
- 调用训练函数；
- 评估多个 seed；
- 保存候选 checkpoint。

这是当前 `README.md` 和 `artifacts/PRODUCTION.md` 所指向的生产复现入口。

#### Kendall 改进实验入口

```text
stage2TrainGraphSAGE/ablation.py
```

它也调用同一套模型和训练代码，但额外负责：

- 固定 train/validation/test splits；
- 分开控制 split seed 与模型初始化 seed；
- 开关不同图结构、GNN、loss 和 head；
- 统一计算 Kendall 和 `z_d -> z_m` 检索指标；
- 输出可配对比较的实验 JSON。

如果你要理解“目前正在做的 Kendall 实验”，优先从 `ablation.py` 看；如果你只想
复现 README 中原来的 Stage 2 candidate，则看 `train_diverse.py`。

### Step 2：可选地调整图结构

文件：

```text
stage2TrainGraphSAGE/graph_surgery.py
```

这一步只存在于实验路径。它可以在不重新运行 Stage 1 的情况下，对已经构建好的
`similar_to` relation 做轻量变体，例如：

- 保留原来的 dense graph；
- 删除 `similar_to`；
- 每个 dataset 只保留 top-k 相似邻居。

它不会修改 `trained_on` supervision 或 lineage。

### Step 3：切分 performance edges

文件：

```text
stage2TrainGraphSAGE/losses.py
```

主要函数：

```text
split_trained_on()
```

它把 `trained_on` edges 分成：

- train；
- validation；
- test。

同时处理反向的 `rev_trained_on`，避免 held-out edge 从反方向泄漏回消息图。

这里要记住两个不同概念：

```text
edge_index       = GraphSAGE 用于传消息的边
edge_label_index = loss 要求模型预测/排序的监督边
```

### Step 4：把图中的 performance 变成训练监督

文件：

```text
stage2TrainGraphSAGE/losses.py
```

相关函数包括：

```text
accuracy_lookup()
perf_supervision()
topk_membership()
lineage_components()
```

这一步大致产生两类训练信号：

1. 对某个 dataset，哪些 model 应该排在前面；
2. 哪些 models 可以作为 contrastive positives 或 same-hub negatives。

输出会交给训练循环使用。

### Step 5：创建 GraphSAGE 模型

核心文件：

```text
stage2TrainGraphSAGE/model.py
```

核心类：

```text
HeteroGraphSAGE
```

模型内部可以粗略分成三段：

```text
节点编码
  -> 异构图消息传播
  -> 输出 projection + L2 normalization
```

节点编码部分：

- model side 使用名称、描述、size bucket、family 等特征；
- dataset side 使用 xd0 多视图特征和 task/class 等离散特征；
- 两侧先投影到相同 hidden dimension。

消息传播部分根据 relation 分开学习。当前代码还提供可选的 edge-aware 路径：

```text
stage2TrainGraphSAGE/edge_aware.py
```

输出始终是：

```python
{
    "model": z_m,
    "dataset": z_d,
}
```

二者都会 L2 normalize，因此后续可以直接用 dot product 作为 cosine similarity。

### Step 6：生成训练 batch 并做 edge dropout

文件：

```text
stage2TrainGraphSAGE/sampling.py
```

主要职责：

- 以 supervision edges 为中心采样局部子图；
- 自动把全局 node ID 映射成 batch 内局部 ID；
- 保持节点特征和监督边对齐；
- 随机丢弃部分 message edges；
- 不丢弃 supervision edges。

有采样后端时使用 PyG `LinkNeighborLoader`；没有时使用项目自己的
`LightLinkLoader` fallback。

edge dropout 的目的，是让模型不能永远依赖完整邻居，从而更适应稀疏/冷启动节点。

### Step 7：执行 forward，得到 `z_m` 和 `z_d`

文件：

```text
stage2TrainGraphSAGE/model.py
```

训练 batch 进入 `HeteroGraphSAGE.forward()` 后，模型会：

1. 编码 model/dataset 的原始特征；
2. 沿不同 relation 聚合邻居；
3. 投影到最终 embedding dimension；
4. L2 normalize；
5. 返回当前 batch 的 `z_m` 和 `z_d`。

这些向量既用于 loss，也就是未来 HNSW 使用的向量空间。

### Step 8：计算 losses

loss 定义集中在：

```text
stage2TrainGraphSAGE/losses.py
```

当前代码支持的主要目标包括：

- performance/ranking loss：让高性能 model 在对应 dataset 上得分更高；
- model-to-model contrastive loss：形成任务结构并处理 lineage hub；
- dataset-to-model contrastive loss：直接训练 serving relation；
- optional MSE/uniformity 等辅助项。

训练时的总 loss 由配置中的不同 `lambda` 加权相加。

不用一开始记住所有 loss。只需先理解核心关系：

```text
score(model, dataset) = z_m dot z_d
```

ranking loss 的目标，就是让真实表现更好的 model 获得更高 score。

### Step 9：反向传播与参数更新

训练循环在：

```text
stage2TrainGraphSAGE/train.py
```

主要训练函数：

```text
train()          = mini-batch 训练路径
train_grouped()  = dataset-macro/full-batch 实验路径
```

每个训练 step 大致执行：

```text
取 batch
-> edge dropout
-> model forward
-> 计算 losses
-> total.backward()
-> optimizer.step()
```

`train()` 还支持 validation Kendall、best-state 恢复和 early stopping；是否启用由
调用入口的配置决定。

### Step 10：在 held-out edges 上评估

基础评估函数仍在：

```text
stage2TrainGraphSAGE/train.py
```

统一实验评估在：

```text
stage2TrainGraphSAGE/eval_harness.py
```

它主要计算：

- 每个 dataset 的 Kendall tau；
- `tau_macro`；
- exact `z_d -> z_m` 的 Hit/Recall/NDCG/regret；
- HNSW 与 exact-dot top-K 的一致性；
- 两个配置之间的 paired bootstrap。

要注意：Kendall 衡量排序语义质量；HNSW recall 衡量近似索引是否复现 exact
embedding neighbors。二者不是同一个指标。

### Step 11：保存 checkpoint

文件：

```text
stage2TrainGraphSAGE/learnable.py
```

主要职责：

- 保存模型结构和参数；
- 把 checkpoint 与 family/task vocab 绑定；
- 加载时检查词表和 embedding rows 是否一致；
- 支持新模型按同样规则构造输入；
- 检查 learnable embedding rows 是否真的得到训练。

这一层保证 checkpoint 不只是“一堆权重”，还保留重建推理链所需的身份信息。

### Step 12：诊断、比较和可视化

训练完成后常用文件：

```text
diverse_diagnostics.py  = 图统计、embedding 健康、Kendall、检索诊断
visualize_trained.py    = 将训练后的 z_m/z_d 投影到二维图
compare.py              = 对两个 ablation JSON 做配对比较
```

当前 Kendall ablation 的汇总结果在：

```text
stage2TrainGraphSAGE/artifacts/ablation/RESULTS.md
```

---

## 5. 核心文件与辅助文件

### 必须理解的五个核心文件

| 文件 | 一句话职责 |
|---|---|
| `model.py` | 定义如何把异构图编码成 `z_m` 和 `z_d` |
| `losses.py` | 定义数据切分、监督构造、scorer 和各种 loss |
| `sampling.py` | 生成训练子图并执行 edge dropout |
| `train.py` | 串起 forward、loss、backprop、validation |
| `learnable.py` | 保存/加载 checkpoint，并绑定 vocab 与冷启动规则 |

### 运行入口

| 文件 | 用途 |
|---|---|
| `train_diverse.py` | README 中的稳定生产复现入口 |
| `ablation.py` | 当前 Kendall 改进实验的统一入口 |

### 实验组件

| 文件 | 用途 |
|---|---|
| `graph_surgery.py` | 生成 dense/drop/top-k similarity 图变体 |
| `edge_aware.py` | 可选的 edge-weight-aware 消息传播 |
| `eval_harness.py` | 固定 split 和统一 Kendall/检索评估 |
| `compare.py` | 比较 base 与 candidate 的配对结果 |

### 诊断与历史入口

| 文件 | 用途 |
|---|---|
| `diverse_diagnostics.py` | 完整诊断 JSON |
| `visualize_trained.py` | embedding 二维可视化 |
| `experiment.py` | 旧 zoo 的 robustness matrix/legacy experiment |
| `kendall_lab.py` | 较早的独立 Phase-0 Kendall runner；当前 `ablation.py` 直接使用 `eval_harness.py` |

---

## 6. 两条实际调用链

### 生产复现调用链

```text
train_diverse.py
  -> model.load_hgraph()
  -> losses.split_trained_on()
  -> losses 中的 supervision builders
  -> model.HeteroGraphSAGE
  -> train.train()
       -> sampling.make_link_loader()
       -> model.forward()
       -> losses.*
  -> train.eval_perf()
  -> learnable.save_checkpoint()
```

### Kendall ablation 调用链

```text
ablation.py
  -> model.load_hgraph()
  -> graph_surgery.apply_similar_to_mode()
  -> eval_harness.make_fixed_splits()
  -> model.HeteroGraphSAGE
  -> train.train() 或 train.train_grouped()
  -> eval_harness.per_dataset_tau()
  -> eval_harness.head_retrieval()
  -> eval_harness.dataset_to_model_hnsw_recall()
  -> 写入 artifacts/ablation/<name>.json

compare.py
  -> 读取两个 JSON
  -> paired bootstrap
  -> 判断 candidate 是否优于 baseline
```

---

## 7. 推荐阅读顺序

如果你完全不了解 Stage 2，不建议从每个 loss 的实现细节开始。按下面顺序阅读：

1. `STAGE2_WORKFLOW_OVERVIEW.md`：先建立全局地图；
2. `ablation.py::train_eval_one()`：看一次实验如何从图走到结果；
3. `model.py::HeteroGraphSAGE`：理解 `z_m/z_d` 从哪里来；
4. `train.py::train()`：理解一个训练 step；
5. `losses.py`：再看 supervision 和 loss；
6. `sampling.py`：理解 batch 内的图从哪里来；
7. `eval_harness.py`：理解 Kendall 和检索结果怎么算；
8. `learnable.py`：最后看 checkpoint 和冷启动契约。

读到任何时候，只要能回答下面四个问题，就没有迷路：

```text
当前拿到的是完整图还是 batch 子图？
当前边是 message edge 还是 supervision edge？
当前操作产生的是 hidden feature、z_m/z_d，还是最终 metric？
当前代码属于生产复现路径还是 ablation 实验路径？
```

---

## 8. 最容易混淆的边界

### Stage 1 与 Stage 2

```text
Stage 1：决定图里有什么节点、特征和边
Stage 2：学习如何把这张图映射成可检索向量
```

### GraphSAGE 与 scorer

```text
GraphSAGE：产生 z_m 和 z_d
scorer/loss：用 z_m 和 z_d 计算训练目标
```

### Message edge 与 supervision edge

```text
message edge：作为 GNN 输入，帮助节点聚合邻居
supervision edge：作为答案，要求模型预测或比较
```

### Exact retrieval 与 HNSW

```text
exact retrieval：直接计算所有 z_d dot z_m
HNSW：近似找到相同的 top-K，目的是加速
```

### Production 与 ablation

```text
production：可复现当前正式 checkpoint 的稳定路径
ablation：验证某个改动是否真的提升指标的实验路径
```

Stage 2 的主线其实只有一句话：

> 从 Stage 1 图中学习 `z_m` 和 `z_d`，用 held-out 排名验证它们，再把模型与词表
> 一起保存，为后续 HNSW candidate retrieval 提供稳定的向量空间。
