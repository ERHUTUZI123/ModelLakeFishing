# Kendall Tau Improvement Action Guide for Claude

## Mission

Improve held-out within-dataset `tau_macro` and, more importantly, the quality of
`z_d -> z_m` candidate retrieval while preserving the production constraint:

```text
score(dataset, model) = normalized_z_d dot normalized_z_m
```

The retrieval score must remain an inner product/cosine score so model vectors can
be precomputed and indexed by HNSW. Do not replace candidate generation with a
pairwise MLP. A later reranker may use an MLP, but that is outside this task.

Work one hypothesis at a time. Every change must be compared with the current
baseline on identical data splits. Do not simultaneously change graph construction,
sampling, loss, and architecture and then attribute the result to one of them.

## Confirmed facts from the current repository

These are code- or artifact-verified facts, not hypotheses:

1. The current hf1000d/2000m result is `tau_macro = 0.211 +/- 0.032` over three
   runs. The seed-0 diagnostic is 0.1653 over 42 scorable datasets.
2. The graph has 2,000 model nodes, 362 dataset nodes, 12,205 `trained_on`
   edges, 130,680 `similar_to` edges, and 96 lineage edges. The dataset graph is
   therefore almost a complete directed graph.
3. `stage1BuildTransferGraph/attributes.py::get_dataset_edge_index()` overwrites
   its caller-provided threshold with `threshold = 1`. The threshold CLI/config
   is consequently ineffective.
4. `stage2TrainGraphSAGE/model.py` calls the heterogeneous GNN with only
   `x_dict` and `edge_index_dict`. It never passes `edge_attr`. Accuracy,
   similarity strength, and lineage weight are therefore discarded during
   message passing; all surviving edges of a relation are effectively unweighted.
5. The learned space remains low-rank: seed-0 `z_m` participation ratio is 2.741
   and `z_d` participation ratio is 2.413. The reported 10.27 value is the
   effective rank of the input/performance-bearing dataset-similarity matrix, not
   the learned `z_d` space.
6. The current ranking loss is a hinge loss on
   `sigmoid(scale * dot + bias)`, not directly on the raw cosine score used by
   HNSW.
7. `split_trained_on(disjoint_train_ratio=0.3)` gives seed 0 only 2,563 positive
   ranking-supervision edges from the 12,205 total edges. They cover 65 datasets;
   the median is four supervision edges per dataset, 12 datasets have fewer than
   two, 20 have fewer than three, and the maximum is 266. Training supervision is
   highly imbalanced across datasets.
8. Supervision edges are globally shuffled into edge batches. The loss averages
   over sampled pairs, whereas `tau_macro` weights each scorable dataset equally.
9. The active contrastive loss is model-to-model (`z_m -> z_m`). It does not
   directly train dataset-to-model retrieval (`z_d -> z_m`).
10. Model and dataset nodes share one final projection head.
11. `train_diverse.py` constructs a validation split but discards it, trains for a
    fixed number of epochs, and saves the final rather than the best-validation
    checkpoint.
12. The current HNSW diagnostic queries the model index with model vectors
    (`z_m -> z_m`). Production retrieval queries it with dataset vectors
    (`z_d -> z_m`). The diagnostic does not validate the serving path.

## Success metrics and evaluation protocol

Before modifying training, build one evaluation path that every ablation reuses.

### Split discipline

- Materialize three fixed edge splits once and reuse them for every configuration.
- Separate `split_seed` from `init_seed`. Do not let a model seed silently change
  the test set.
- Report per-dataset values so configurations can be compared with a paired test.
- Never use validation/test edges to build contrastive membership, hard negatives,
  graph messages, or pair weights.
- In addition to the current transductive edge split, report a cold-dataset split
  in which test datasets have no `trained_on` message edges. The production query
  resembles this setting more closely.

### Primary metrics

1. Held-out `tau_macro`, with the same eligible datasets for paired comparison.
2. Per-dataset Kendall τ and the number of comparable non-tied pairs.
3. Exact-dot `z_d -> z_m` retrieval:
   - Hit@K for the true best model;
   - Recall@K for true top-3 and top-10% models;
   - NDCG@K;
   - regret@K = best observed performance minus the best observed performance in
     the retrieved candidates.
4. Use K = 10, 50, 100, and 200 where the labeled candidate count permits it.

Because the performance matrix is sparse, clearly state the evaluation universe:
metrics over observed held-out candidates are not equivalent to full-lake recall.

### Secondary diagnostics

- `z_m` and `z_d` participation ratio;
- mean pairwise cosine and same-hub distance;
- per-relation degree and message magnitude;
- number of datasets and ranking pairs contributing to every training batch;
- gradient cosine between ranking and contrastive objectives;
- HNSW ANN recall against exact `z_d -> z_m` dot-product top-K.

Do not select a model by collapse metrics alone. They are diagnostics, not the
objective.

### Acceptance rule

Use paired bootstrap resampling over datasets. Promote a change only when:

- the 95% paired bootstrap interval for `delta tau_macro` excludes zero, or the
  improvement reproduces consistently across all three fixed splits;
- exact-dot head retrieval does not materially regress;
- the score remains exactly reproducible by cosine/inner-product ANN search;
- no held-out edge leakage is introduced.

With only about 42 currently scorable datasets, do not celebrate a change of a few
thousandths without a paired uncertainty estimate.

## Work plan

### Phase 0 — Freeze and reproduce the baseline

1. Add a reproducible experiment manifest containing graph path, checkpoint
   architecture, split seeds, init seeds, and all training hyperparameters.
2. Reproduce the current baseline using fixed splits.
3. Extend diagnostics to emit per-dataset τ, candidate counts, pair counts, and
   exact `z_d -> z_m` head metrics.
4. Correct `hnsw_recall` or add a new function that indexes all `z_m` and queries
   with `z_d`; keep the old model-to-model diagnostic under a distinct name if it
   remains useful.

Gate: do not begin architecture experiments until the baseline and serving-path
metrics are reproducible.

### Phase 1 — Fix dataset graph construction first

This is the highest-priority hypothesis.

1. Remove the hard-coded `threshold = 1`; make the configured threshold effective.
2. Prefer a bounded top-k graph to a global threshold. Build directed top-k
   neighbors per dataset, then test mutual-kNN or symmetric union variants.
3. Start with `k in {5, 10, 20}` and always exclude self edges.
4. Preserve normalized similarity as `edge_attr`.
5. Assert a maximum/expected degree and fail graph construction if a supposedly
   sparse graph becomes nearly complete.
6. Run these isolated ablations before changing the GNN:
   - current almost-complete unweighted graph;
   - no `similar_to` relation;
   - top-k unweighted graph.

Interpretation:

- If removing or pruning `similar_to` raises τ, the dense relation is causing
  destructive averaging.
- If pruning alone does not help, do not conclude that similarity values are
  useless; the GNN still ignores their weights until Phase 2.

### Phase 2 — Consume edge weights in message passing

Implement the smallest edge-aware heterogeneous message-passing change that can
be cleanly ablated.

Requirements:

- `similar_to`: aggregate neighbor messages using normalized similarity weight;
- `trained_on`: only training-visible message edges may use accuracy weight;
- lineage: preserve direction and consume its ordered relation weight, or encode
  the discrete lineage type explicitly;
- add learnable relation gates or normalization so relation contributions can be
  inspected rather than blindly summed;
- retain an explicit self/residual path so sparse or cold nodes do not depend only
  on neighbors.

Tests must prove that changing `edge_attr` while holding topology fixed changes the
output, and that zeroing a relation weight removes its neighbor contribution.

Run, in order:

1. top-k graph + unweighted GNN;
2. top-k graph + weighted `similar_to` only;
3. add weighted training-visible `trained_on`;
4. add lineage weight.

Do not feed validation/test accuracy into message passing.

### Phase 3 — Align training batches with `tau_macro`

Replace globally shuffled edge batches for ranking supervision with
dataset-grouped batches.

For each optimization step:

1. sample a fixed number of datasets;
2. sample several supervised models per selected dataset;
3. form unique non-tied pairs within each dataset;
4. compute one mean loss per dataset;
5. average dataset losses, giving each dataset equal weight.

Avoid sampling `(p, q)` with replacement and then discarding half the samples.
Enumerate all unique pairs for small lists; for large lists, sample hard pairs
without replacement.

Log per epoch:

- datasets with at least one valid pair;
- unique pairs used per dataset;
- skipped datasets and why;
- contribution of each dataset to the total loss.

Also test ways to use more than the current 2,563 ranking-supervision edges without
leakage. Preferred option: cross-fitted/leave-edge-out supervision folds, where a
target edge is absent from that forward pass's message graph. Do not simply place a
target `trained_on` edge back into the same graph used to score that pair.

### Phase 4 — Replace the current ranking surrogate

First implement a raw-dot RankNet-style loss:

```text
s(d, m) = dot(normalize(z_d), normalize(z_m))
L_d = mean softplus(-(s(d, better) - s(d, worse)) / temperature)
L = mean over datasets L_d
```

Requirements:

- ranking loss must operate on the raw score that HNSW ranks;
- remove sigmoid scale/bias from the ranking path;
- constrain any temperature to remain positive;
- test `min_gap in {0, 0.005, 0.01, 0.02}` because `0.001` likely treats
  measurement noise as a strict preference;
- optionally weight pairs by a capped performance gap, but retain an unweighted
  version because Kendall gives all non-tied pairs equal importance;
- mine hard inversions and near-ties under the current prediction instead of
  repeatedly sampling easy, already-correct pairs.

Compare hinge and RankNet on the exact same grouped batches. RankNet is the first
replacement to test because pairwise concordance is closely aligned with Kendall.

Only after that baseline is stable, optionally test a listwise loss. Use LambdaLoss
when prioritizing NDCG/head quality; use a differentiable sorting loss only if its
added complexity produces a reproducible gain. Do not assume a top-heavy listwise
loss must improve full-list Kendall.

### Phase 5 — Remove unnecessary two-tower restrictions

Replace the shared final projection with separate ANN-compatible projections:

```text
z_m = normalize(model_head(h_m))
z_d = normalize(dataset_head(h_d))
score = z_d dot z_m
```

This is still a pure maximum-inner-product/cosine retrieval model. Test shared-head
versus separate-head with every other setting fixed.

If low learned rank persists, test residual projections from the pre-GNN encoded
features into each output head. Do not add dimensions merely to improve a rank
diagnostic; require a ranking/retrieval gain.

### Phase 6 — Train the serving relation directly

After the ranking pipeline is corrected, add a dataset-to-model contrastive
objective rather than relying only on model-to-model clustering.

- positives: training-visible top-performing models for dataset `d`;
- reliable negatives: training-visible low-performing models;
- hard negatives: models currently ranked highly by `z_d dot z_m` but known to
  perform poorly;
- do not label every unobserved pair negative; most are missing, not negative.

Keep the existing model-to-model contrastive objective as an ablation, not an
article of faith. Test:

- ranking only;
- ranking + small model-to-model contrastive weight;
- ranking + dataset-to-model contrastive;
- all three.

Measure gradient cosine between objectives. If the contrastive gradient repeatedly
opposes ranking, use a smaller weight or a schedule. Begin with ranking warm-up and
anneal contrastive weight from zero. Do not retain `lambda_rank = lambda_contrast =
1` merely because it was the historical default.

### Phase 7 — Training selection and routine optimization

Only after Phases 1–4 establish a sound objective:

- select checkpoints by validation `tau_macro` and head retrieval metrics;
- use early stopping and save the best epoch, not the final epoch;
- test learning rates `{3e-4, 1e-3, 3e-3}` before retaining `1e-2`;
- compare AdamW with modest weight decay;
- use a scheduler and gradient clipping;
- tune edge dropout by relation. A dense similarity relation and a sparse lineage
  relation should not inherit the same assumptions;
- report initialization variance separately from split variance.

## Minimal ablation ladder

Run sequentially and retain only accepted changes:

| ID | Single change from previous accepted row |
|---|---|
| B0 | Reproduced current baseline on fixed splits |
| B1 | Remove `similar_to` |
| B2 | Restore it as top-10 unweighted |
| B3 | Weighted top-10 `similar_to` |
| B4 | Dataset-grouped, macro-balanced batches |
| B5 | Raw-dot RankNet with tie threshold |
| B6 | Separate model/dataset projection heads |
| B7 | Weighted training-visible performance messages |
| B8 | Dataset-to-model hard-negative contrastive loss |
| B9 | Validation early stopping and optimizer tuning |

If B1 beats B0, retain B1 while developing B2/B3. If B2 or B3 cannot beat B1,
leave dataset message passing out rather than forcing it into the architecture.

## Required tests

Add focused tests before long training runs:

1. Similarity threshold/top-k arguments alter graph degree as requested.
2. No self edges; no accidental nearly complete graph under top-k mode.
3. `edge_attr` affects message passing numerically.
4. Held-out and target supervision edges are absent from the corresponding message
   graph, including reverse edges.
5. Grouped batches contain the expected number of datasets and valid unique pairs.
6. Macro-balanced loss gives equal dataset weight regardless of edge count.
7. Rank loss orders raw dot products in the intended direction.
8. Separate heads still produce normalized vectors and exact score equality between
   training and HNSW query geometry.
9. HNSW diagnostic uses `z_d` queries and `z_m` index items.
10. Checkpoint roundtrip preserves both projection heads and reproduces embeddings.

## Files likely to change

- `stage1BuildTransferGraph/attributes.py`
- `stage2TrainGraphSAGE/model.py`
- `stage2TrainGraphSAGE/sampling.py`
- `stage2TrainGraphSAGE/losses.py`
- `stage2TrainGraphSAGE/train.py`
- `stage2TrainGraphSAGE/train_diverse.py`
- `stage2TrainGraphSAGE/diverse_diagnostics.py`

Create a dedicated experiment driver or manifest rather than embedding another
one-off configuration into production defaults.

## Explicit non-goals and prohibitions

- Do not install or tune HNSW as a substitute for fixing semantic retrieval.
- Do not claim model-to-model ANN recall validates dataset-to-model retrieval.
- Do not use an MLP score for candidate retrieval.
- Do not use test edges in contrastive membership or hard-negative construction.
- Do not optimize `mean_cos` or participation ratio at the expense of τ/head recall.
- Do not treat missing performance pairs as automatic negatives.
- Do not change multiple major components in one ablation.
- Do not overwrite the current production checkpoint until an accepted candidate
  passes fixed-split evaluation and checkpoint roundtrip tests.

## Deliverable expected from Claude after each phase

Return:

1. files changed and the exact hypothesis tested;
2. tests run and their outputs;
3. fixed-split per-dataset metric artifact;
4. aggregate `tau_macro`, head retrieval metrics, and uncertainty versus the paired
   baseline;
5. leakage audit;
6. decision: retain, reject, or investigate further;
7. the next single ablation.

Do not describe a loss decrease as a ranking improvement. The gate is held-out
within-dataset ranking and dataset-to-model candidate quality.

## Primary references for optional loss/architecture work

- RankNet: [Learning to Rank using Gradient Descent](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/)
- LambdaLoss: [The LambdaLoss Framework for Ranking Metric Optimization](https://research.google/pubs/the-lambdaloss-framework-for-ranking-metric-optimization/)
- NeuralSort: [Stochastic Optimization of Sorting Networks via Continuous Relaxations](https://arxiv.org/abs/1903.08850)
- Edge-aware message passing motivation: [Exploiting Edge Features in Graph Neural Networks](https://arxiv.org/abs/1809.02709)

---

## Appendix A — 八个核心问题：当前代码与对应修复

> 说明：下面只有第 1 条是非常明确的覆盖参数 bug。其余多数属于“代码能运行，
> 但与 `tau_macro` 或生产 `z_d -> z_m` 检索目标错位”。修复片段中标注为
> “伪代码”的内容不能不加测试地直接粘贴进生产代码。

### 1. Dataset similarity 阈值被强制覆盖

**类型：明确 bug。**

当前代码：`stage1BuildTransferGraph/attributes.py::get_dataset_edge_index`

```python
def get_dataset_edge_index(self, threshold=0.3,
                           base_dataset='imagenet',
                           sim_method='cosine'):
    threshold = 1
```

调用者传入的任何阈值都会立即变成 `1`。在 hf1000d/2000m 图中，这产生了
130,680 条 `similar_to` 边；362 个 dataset 的完整有向无自环图最多只有
`362 * 361 = 130,682` 条边，因此当前图事实上接近全连接。

**修复建议：**

1. 删除 `threshold = 1`，让配置参数真正生效；
2. 不把全局 threshold 当最终生产策略，优先实现每个 dataset 的 top-k；
3. 从 `k in {5, 10, 20}` 开始；排除 self edge；
4. 比较 directed top-k、mutual-kNN、symmetric union；
5. 图构建后断言最大/平均 degree，意外接近全连接时直接失败。

方向性伪代码：

```python
def get_dataset_edge_index(..., top_k=10):
    # 不再覆盖 threshold。
    # 对每个 source dataset，仅保留 similarity 最大的 top_k 个 target。
    # 保存 similarity 到 edge_attr，并排除 source == target。
    ...
```

必须先做三组隔离消融：当前 dense graph、完全删除 `similar_to`、top-k
unweighted。若删除 relation 反而提升 τ，说明 dense aggregation 正在伤害模型。

### 2. GNN 消息传播完全忽略 `edge_attr`

**类型：实现与图数据契约错位。**

当前代码：`stage2TrainGraphSAGE/model.py::forward`

```python
x_dict = self.encode_nodes(data)
h_dict = self.gnn(x_dict, data.edge_index_dict)
```

这里只传入 node features 和 edge topology，没有传入 `data.edge_attr_dict`。
因此当前消息传播中：

```text
similarity 0.95 == similarity 0.01
accuracy   0.91 == accuracy   0.55
强 lineage       == 弱 lineage
```

`sampling.py` 虽然会随 edge dropout 同步切片 `edge_attr`，但 `model.py` 最终
没有消费它。

**修复建议：**

实现最小、可消融的 edge-aware heterogeneous convolution：

```python
# 方向性伪代码，不是可直接替换的 PyG API：
message_ij = normalized_edge_weight_ij * transform(x_j)
out_i = self_path(x_i) + aggregate(message_ij)
```

要求：

- `similar_to` 使用归一化 similarity；
- `trained_on` 只使用 training-visible message edges 的 accuracy；
- lineage 保留方向并使用 ordered weight，或显式编码离散 lineage type；
- 为每个 relation 增加可观测的 gate/normalization，不盲目相加；
- 保留 self/residual path，避免冷启动节点完全依赖邻居。

必须按以下顺序消融：top-k unweighted → weighted similarity → weighted
training-visible performance → lineage weight。测试必须证明：拓扑不变时改变
`edge_attr` 会改变输出；把某 relation 的权重置零会移除其邻居贡献。

### 3. 只有少量 performance edges 直接参与 ranking supervision

**类型：训练资源分配不适合当前稀疏排序任务，不是库函数 bug。**

当前配置：`stage2TrainGraphSAGE/losses.py::split_trained_on`

```python
def split_trained_on(data, *,
                     num_val=0.1,
                     num_test=0.2,
                     neg_ratio=1.0,
                     disjoint_train_ratio=0.3,
                     seed=None):
    ...
```

自定义 `RandomLinkSplit` 先划分 train/val/test：

```python
num_train = perm.numel() - num_val - num_test
train_edges = perm[:num_train]
```

然后把 train pool 的前 30% 作为独立监督，其余作为消息边：

```python
num_disjoint = int(
    self.disjoint_train_ratio * train_edges.numel()
)

# 消息边：不包含当前监督目标
self._split(
    train_store,
    train_edges[num_disjoint:],
    is_undirected,
    rev_edge_type,
)

# ranking supervision
train_edges = train_edges[:num_disjoint]
self._create_label(store, train_edges, ..., out=train_store)
```

hf1000d/2000m seed 0 的实际流水：

```text
12,205 total performance edges
├── 1,220 validation edges
├── 2,441 test edges
└── 8,544 train-pool edges
    ├── 2,563 ranking-supervision edges (30%)
    └── 5,981 GraphSAGE message edges (70%)
```

5,981 条消息边并非完全没用，但当前 GNN 又忽略 accuracy `edge_attr`，所以它们
只告诉模型“有 trained_on 关系”，没有直接教授“谁比谁好”。2,563 条监督边仅覆盖
65 个 dataset；中位数每个 dataset 4 条，12 个 dataset 少于 2 条，无法形成有效
pair，20 个少于 3 条。

**修复建议：**

先做低成本消融：

```text
disjoint_train_ratio in {0.3, 0.5, 0.7, 0.8}
```

不要直接设为 0：监督边与消息边重叠会产生 shortcut/leakage；也不要未经实验直接
拉到接近 1，因为 performance message structure 会骤减。

最终方案是 leave-edge-out/cross-fitted folds：把 train pool 分成 K folds。每次
forward 以 fold `f` 为监督，并从当前消息图中移除 `f`；轮换所有 folds，使更多
训练边直接进入 ranking loss，同时保证一条边作为当前预测目标时不出现在当前消息
图里。Validation/test 永远不能参加轮换。

### 4. Batch 按 edge 随机采样，但 `tau_macro` 按 dataset 等权

**类型：训练目标与评估聚合方式错位。**

当前 fallback loader：`stage2TrainGraphSAGE/sampling.py::LightLinkLoader`

```python
P = self.eli.size(1)
order = torch.randperm(P) if self.shuffle else torch.arange(P)

for start in range(0, P, self.batch_size):
    idx = order[start:start + self.batch_size]
    seed = self.eli[:, idx]
```

真实 `LinkNeighborLoader` 同样使用 globally shuffled supervision edges。随后
ranking loss 把所有 dataset 产生的 pairs 合并，再统一平均：

```python
hi, lo = torch.cat(hi), torch.cat(lo)
s_hi = scorer(...)
s_lo = scorer(...)
loss = torch.relu(margin - (s_hi - s_lo)).mean()
```

因此 pair 多的数据集贡献更多梯度；但 `eval_perf()` 的 `tau_macro` 是先计算每个
dataset 的 τ，再给每个 dataset 相同权重。训练和评估不是同一个宏平均目标。

**修复建议：**

实现 dataset-grouped sampler：

```python
# 方向性伪代码
selected_datasets = sample_datasets(n_dataset_per_batch)
dataset_losses = []

for d in selected_datasets:
    edges_d = sample_models_for_dataset(d)
    unique_pairs_d = build_non_tied_pairs_without_replacement(edges_d)
    if unique_pairs_d:
        dataset_losses.append(rank_loss(unique_pairs_d))

loss = torch.stack(dataset_losses).mean()
```

每个 step 固定采若干 dataset，每个 dataset 固定采若干模型；小 list 枚举 unique
pairs，大 list 再无放回采样 hard pairs。先求每个 dataset 的 mean loss，再对
dataset 求 mean。每个 epoch 记录参与训练的数据集数、每个数据集 unique pair 数和
跳过原因。

### 5. Ranking loss 没有直接优化 HNSW 使用的 raw dot score

**类型：几何目标近似对齐，但 margin 与 serving score 不严格一致。**

当前 scorer：`stage2TrainGraphSAGE/losses.py::PerfScorer`

```python
s = (zm * zd).sum(dim=-1)
return torch.sigmoid(self.scale * s + self.bias)
```

当前 ranking loss：

```python
s_hi = scorer(...)
s_lo = scorer(...)
loss = torch.relu(margin - (s_hi - s_lo)).mean()
```

如果 `scale > 0`，sigmoid 是单调的，排序方向仍一致；但固定 `margin=0.05` 的意义
会随 scale/bias 变化，sigmoid 还可能压缩梯度。`scale` 当前也没有被约束为正数；
一旦变负，训练排序方向会与 HNSW raw dot 排序相反。

当前 pair sampling 还会有重复和无效抽样：

```python
p = torch.randint(n, (cap,), ...)
q = torch.randint(n, (cap,), ...)
good = a[p] > a[q] + min_gap
```

**修复建议：**

第一选择是直接在 normalized raw dot 上使用 RankNet-style logistic loss：

```python
# 方向性伪代码
raw_hi = (z_m[hi] * z_d[d]).sum(dim=-1)
raw_lo = (z_m[lo] * z_d[d]).sum(dim=-1)
loss_d = F.softplus(-(raw_hi - raw_lo) / temperature).mean()
```

- ranking path 删除 sigmoid/scale/bias；
- temperature 必须保持正数；
- 测 `min_gap in {0, 0.005, 0.01, 0.02}`，不要把测量噪声强行当严格偏序；
- 小 list 枚举 unique pairs；大 list 优先采当前预测反序或接近边界的 hard pairs；
- 可测试按 performance gap 截断加权，但保留不加权版本，因为 Kendall 对所有
  non-tied pairs 等权。

先在完全相同的 grouped batches 上比较现有 hinge 与 raw-dot RankNet。不要先上
更复杂的 listwise loss。

### 6. Contrastive loss 只训练 `z_m -> z_m`，没有直接训练 serving relation

**类型：辅助目标与生产查询关系错位。**

当前代码：`stage2TrainGraphSAGE/losses.py::contrastive_loss`

```python
def contrastive_loss(z_model, pos_mask, hub_mask, *, ...):
    ...
    sim = (z_model @ z_model.t()) / temperature
```

整个函数没有 `z_dataset`。它能让共同高性能模型互相靠近，但不能保证对应的 dataset
query 靠近这些模型。线上 HNSW 实际需要：

```text
query z_d -> indexed z_m
```

而不是：

```text
query z_m -> indexed z_m
```

**修复建议：**

在 ranking pipeline 修正后，再增加 dataset-to-model contrastive objective：

```python
# 方向性伪代码
logits_dm = (z_dataset @ z_model.T) / temperature
```

- positive：该 dataset 上 training-visible 的 top-performing models；
- reliable negative：该 dataset 上已知低性能模型；
- hard negative：当前 `z_d dot z_m` 排得高、但已知性能差的模型；
- 不把所有未观测 pair 当负例，因为 missing 不等于 negative。

必须消融：ranking only、ranking + 小权重 model-model contrast、ranking +
dataset-model contrast、三者同时。记录 ranking 与 contrastive 对 `z_m/z_d` 的
gradient cosine；若长期为负，减小权重或用 ranking warm-up 后逐渐加入 contrast。

### 7. Model 和 dataset 被迫共享同一个 final projection head

**类型：不必要的表达能力限制，不是明确 bug。**

当前代码：`stage2TrainGraphSAGE/model.py`

```python
self.head = nn.Linear(hidden_channels, out_dim)
```

两类节点统一通过同一个 head：

```python
return {
    nt: F.normalize(self.head(h), p=2, dim=-1)
    for nt, h in h_dict.items()
}
```

但 model 与 dataset 的特征语义和上游 encoder 完全不同。共享 head 不是进入同一
metric space 的必要条件；两个独立 projection 最终仍可通过 dot product 对齐。

**修复建议：**

```python
self.model_head = nn.Linear(hidden_channels, out_dim)
self.dataset_head = nn.Linear(hidden_channels, out_dim)

z_m = F.normalize(self.model_head(h_dict["model"]), p=2, dim=-1)
z_d = F.normalize(self.dataset_head(h_dict["dataset"]), p=2, dim=-1)
return {"model": z_m, "dataset": z_d}
```

保持：

```python
score = z_d @ z_m.T
```

因此仍然完全兼容 HNSW。必须单独比较 shared head 与 separate heads，并更新
checkpoint save/load 与 roundtrip 测试；不能仅凭 participation ratio 改善就接受，
必须要求 τ 或 head retrieval 提升。

### 8. Validation、随机种子和 HNSW 诊断没有对齐真实选择/服务流程

**类型：实验协议与 serving path 错位。**

#### 8.1 Validation 被创建但丢弃

当前代码：`stage2TrainGraphSAGE/train_diverse.py::run`

```python
train_data, _val, test_data = split_trained_on(data, seed=seed)
...
hist, _ = train(model, ..., epochs=epochs)
...
tm = eval_perf(model, scorer, test_data, lookup)
```

`_val` 没有参与 early stopping 或 checkpoint selection。训练固定 25 epochs，保存
final epoch；最佳模型可能早已出现在中间 epoch。

**修复：**每个 epoch 在固定 validation split 上计算 `tau_macro` 与 head metrics，
保存最佳 checkpoint，使用 patience early stopping；test 只在最终选择完成后运行一次。

#### 8.2 同一个 seed 同时改变 split 和初始化

```python
for si, seed in enumerate(seeds):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_data, _val, test_data = split_trained_on(data, seed=seed)
    ...
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = HeteroGraphSAGE(...)
```

因此 seed 0–2 同时更换 test pairs 和模型初始化。配置之间若没有复用完全相同的
splits，就不能做可靠的 paired comparison。

**修复：**显式分离 `split_seed` 和 `init_seed`；预先物化三个固定 splits，让所有
配置复用。分别报告 split variance 与 initialization variance，并对相同 dataset/test
pairs 做 paired bootstrap。

#### 8.3 HNSW diagnostic 查询了错误的向量类型

当前代码：`stage2TrainGraphSAGE/train.py::hnsw_recall`

```python
z = normalize(z_model)
index.add_items(z, np.arange(N))

sim = z @ z.T
brute = np.argsort(-sim, axis=1)[:, :k]
labels, _ = index.knn_query(z, k=k + 1)
```

这验证的是 `z_m -> z_m`。生产检索是 `z_d -> z_m`。

**修复方向：**

```python
# 方向性伪代码
z_m = normalize(z_model)
z_d = normalize(z_dataset)

index.add_items(z_m, model_ids)
ann_labels, _ = index.knn_query(z_d, k=k)

exact_sim = z_d @ z_m.T
exact_labels = topk(exact_sim, k)
ann_recall = overlap(ann_labels, exact_labels)
```

把旧函数重命名为明确的 `model_to_model_hnsw_recall`（如仍需保留），新增真正的
`dataset_to_model_hnsw_recall`。同时分别报告：

1. ANN fidelity：HNSW 是否复现 exact-dot top-K；
2. semantic retrieval：exact/HNSW top-K 是否包含真实性能 top models。

前者高不能证明后者高。
