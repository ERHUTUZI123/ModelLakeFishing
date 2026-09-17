# ModelLakeFishing: A0 Evidence Library for the Final 3M-Scale Retrieval System

**Evidence edition:** A0.1–A0.7, updated 2026-09-14 (America/Toronto).

**Repository:** `D:\research\model_lake\codes\ModelLakeFishing`  
**Final system:** **X4G+D → HNSW top1000 → task prior → top10**.  
**Evidence chain:** [A0 source manifest](A0_runs/A0_SOURCE_MANIFEST.json), [A0.7 manifest](A0_runs/A0_7/MANIFEST.json), [A0.7 report](A0_runs/A0_7/results/A0_REPORT.json), [metric inventory](A0_runs/A0_7/results/A0_METRIC_INVENTORY.json).

This edition describes the final method and the results produced by A0.1–A0.7. A0 retains the frozen candidate universe, node identities, supervision, root splits, three-seed training recipe, and fixed retrieval configuration. It zeros seven performance-derived dataset input columns, trains three new 25-epoch checkpoints, exports new representations, builds new HNSW indexes, and independently recomputes the resulting measurements. The two exact retrieval paths are offline diagnostic references for this same final system.

Retrieval and cost recomputation passed, including 470 additional comparisons with the evaluator. The complete inventory contains 982 entries: 838 recomputed, 86 verified, 3 disabled, 11 not applicable, 42 undefined, and 2 missing. The missing entries are the original model-crawl API-page and discarded-duplicate event counts. These historical events have no complete raw log, so the overall completeness flag remains `false`; the retrieval result is available and validated. [A0.7 validation](A0_runs/A0_7/results/A07_VALIDATION.json), [A0.7 report](A0_runs/A0_7/results/A0_REPORT.json).

Across 3,016,439 candidates and 1,476 / 1,101 / 1,545 eligible held-out query observations, final `gold@10` is **0.3320 / 0.3370 / 0.2214** (equal-weight three-seed mean **0.2968**). Exact full-lake fusion reaches 0.3174. The final system retains 93.47% of the full-lake fused `gold@10`, averaged over the three per-seed ratios. Mean per-seed retrieval p50 / p95 is **0.747 / 1.102 ms** in the formal Linux evaluation; timing starts from a precomputed query embedding. All these figures come from the [A0.7 recomputation report](A0_runs/A0_7/results/A0_REPORT.json).

## 1. System formulation and computation boundary

Let $\mathcal M=\{m_1,\ldots,m_N\}$ be the candidate lake and $\mathcal D=\{d_1,\ldots,d_Q\}$ the frozen dataset–task node table, with

$$
N=3{,}016{,}439,\qquad Q=18{,}729.
$$

Offline, the system canonicalizes model, dataset, task, lineage, and evaluation metadata; constructs the typed evidence graph; applies the A0 feature policy; trains the graph encoder; exports model and dataset–task representations; builds HNSW over model vectors; and materializes a split-specific task-prior sidecar. Both node types are represented by unit-normalized vectors in $\mathbb R^{128}$, with dense score

$$
s_\theta(d,m)=z_d^\top z_m=\cos(z_d,z_m).
$$

For a materialized query node $d=(\mathrm{dataset},t)$, HNSW retrieves

$$
\mathcal P_{1000}(d)=\operatorname*{ANN\text{-}Top1000}_{m\in\mathcal M}s_\theta(d,m).
$$

Let $n_{tm}$ and $A_{tm}$ be the count and sum of oriented performance values for model $m$ on normalized task $t$, over the split's training and validation edges. Scored test roots are excluded from that evidence. The fixed prior and fusion are

$$
p_t(m)=
\begin{cases}
\dfrac{A_{tm}+0.5\times5}{n_{tm}+5},&n_{tm}>0,\\[5pt]
0,&n_{tm}=0,
\end{cases}
\qquad
r(d,m)=\frac{s_\theta(d,m)+1}{2}+p_t(m).
$$

The system reads the 1,000 candidate priors and returns the ten highest fused scores, with a fixed label-free model-ID permutation resolving exact ties. Fusion uses `beta=1`, shrinkage `k=5`, and the raw cosine transformation above. The prior is deterministic historical evidence. HNSW avoids exhaustive dense scoring, and the fixed pool bounds the second-stage lookup and fusion to 1,000 models independently of $N$. HNSW search cost itself can still depend on index size and search parameters. [A0 protocol](A0_runs/A0_PROTOCOL.json), [A0 evaluator](../../scale1m/a0_evaluation.py).

## 2. Frozen metadata and canonicalization

### 2.1 Hub snapshots and what was recounted

A0 reuses the frozen hub snapshot bytes and audits them locally. The collectors enumerate the Hugging Face `/api/models` and `/api/datasets` endpoints using server-provided next-page cursors, resumable JSONL-GZIP shards, and SHA-256 bindings. The model collector retains repository identity, timestamps, task/library tags, author, downloads, likes, safetensors metadata, base-model relations, and relevant model-card fields. The dataset collector retains identity, task categories, tags, description, language, size category, license, and source metadata. Collector implementations: [models](../../scale1m/hf_crawl.py), [datasets](../../scale1m/hf_crawl_datasets.py).

| Frozen snapshot | Parsed records | Shards | Unique raw IDs | Unique IDs after strip/lower |
|---|---:|---:|---:|---:|
| Models | 3,003,759 | 61 | 3,003,759 | 3,003,759 |
| Datasets | 1,008,417 | 11 | 1,008,417 | 1,008,416 |

The one dataset case collision has zero affected rows in the frozen node/card tables and stays outside their matching inputs. The declared snapshot date is 2026-08-18; this verifies the frozen metadata declaration rather than independently reconstructing the historical API state. Counts, per-shard hashes, order hashes, and the collision trace are in the [A0.1 snapshot audit](A0_runs/audit/A0_SNAPSHOT_AUDIT.json).

The retained model shards have zero duplicate identifiers. This is distinct from the number of duplicate records discarded during collection. Original API-page and duplicate-discard event counts remain `missing` in the [A0.7 source recount](A0_runs/A0_7/results/A0_SOURCE_COUNTS.json).

### 2.2 Canonical model identity and structural attributes

Every join uses one model-key rule:

$$
\operatorname{id}_{\mathrm{norm}}(m)
=\operatorname{lower}(\operatorname{strip}(\operatorname{id}(m))).
$$

Duplicate normalized identifiers are rejected rather than silently merged. Parameter count is accepted only from `safetensors.total`:

$$
\operatorname{sizeB}(m)=
\begin{cases}
\texttt{safetensors.total}/10^9, & \text{if present},\\
\mathrm{NA}, & \text{otherwise}.
\end{cases}
$$

The canonical family is `config.model_type` when present; otherwise a lowercased name rule supplies the family, with `other` as the final fallback. A declared parent first uses Hugging Face's structured `baseModels.ids[0]`; free-text `cardData.base_model` is used only when the structured relation is absent. These choices are implemented by `size_b_of`, `family_of`, and `lineage_base_of` in [`scale1m/hf_canonicalize.py`](../../scale1m/hf_canonicalize.py) and consumed by [`scale1m/canonicalize_rf.py`](../../scale1m/canonicalize_rf.py).

### 2.3 Dataset--task node identity and card matching

A query node is not a bare dataset name. Its primary key is

$$
u_d=(\operatorname{normalize}(\text{dataset}),\operatorname{task}),
$$

serialized internally as `dataset + "\t" + task`. This prevents evaluations of the same repository under different tasks from being pooled into one ranking problem.

Dataset-card matching is deliberately conservative. A canonical node receives a card when its normalized identifier is an exact repository identifier. An owner-free basename is accepted only when that basename is unique in the dataset snapshot. A name containing `/` is never cross-owner matched by basename. For dataset/config names such as `ag_news/default`, the matcher may use the unambiguous parent card and records this separately as `hf_card_via_parent`. All unresolved or ambiguous nodes fall back to their cleaned node name. The algorithm and its audit labels are in [`scale1m/match_dataset_cards.py`](../../scale1m/match_dataset_cards.py).

In the final node set, 3,928 of 18,729 nodes have an exact or parent Hugging Face card. Missing cards therefore remain a first-class condition rather than being filled by a popularity-based guess. The count is newly verified in [A0.1 input audit](A0_runs/audit/A0_INPUT_AUDIT.json).

### 2.4 Evaluation-record parsing and metric semantics

For each model-card `model-index` entry, the parser extracts

$$
(m,\;\text{dataset},\;\text{task},\;\text{metric},\;v).
$$

Boolean, malformed, and non-finite values are discarded. Metric names are lowercased, separators are normalized, `@k` is rewritten as `_at_k`, and similarity prefixes and cut-off suffixes are removed only for direction classification. The full normalized metric name remains the grouping key, so, for example, `ndcg_at_1` and `ndcg_at_10` are not merged.

Each metric belongs to one of four direction classes:

$$
c(r)\in\{\text{higher},\text{lower},\text{reward},\text{unknown}\}.
$$

Known accuracy, F1, NDCG, correlation, overlap, and generation-quality families are `higher`; WER, loss, perplexity, and error families are `lower`. Reward and unknown metrics remain usable as graph evidence but cannot define a gold model. The explicit classifier is [`scale1m/metric_semantics.py`](../../scale1m/metric_semantics.py); unmatched names are never assigned a guessed direction.

## 3. Offline stage 2: constructing the model--dataset evidence graph

### 3.1 Canonical supervision values

Within the native Hugging Face source, repeated measurements are collapsed by the median over

$$
(m,d,t,r).
$$

For each group $g=(d,t,r)$, let $v_g^{\min}$ and $v_g^{\max}$ be the group extrema. The normalized value is

$$
\bar v_i=
\begin{cases}
\dfrac{v_i-v_g^{\min}}{v_g^{\max}-v_g^{\min}}, & v_g^{\max}>v_g^{\min},\\[6pt]
0.5, & v_g^{\max}=v_g^{\min},
\end{cases}
$$

and its oriented value is

$$
y_i=
\begin{cases}
1-\bar v_i, & c(r)=\text{lower},\\
\bar v_i, & \text{otherwise}.
\end{cases}
$$

For each dataset--task node, a direction-known metric is preferred and the metric with the greatest number of records is selected, with deterministic lexical tie-breaking. The `trained_on` edge weight is the median oriented value for the resulting model--node pair. Of 2,158,375 parsed raw metric rows, 2,097,081 are finite and parseable; median deduplication produces 1,435,162 rows. Native primary-metric selection yields 143,478 edges before the per-node cap and 74,346 afterward. The exact construction is `build_supervision` in [`scale1m/canonicalize_rf.py`](../../scale1m/canonicalize_rf.py), with counts independently regenerated in [A0.7 source recount](A0_runs/A0_7/results/A0_SOURCE_COUNTS.json).

### 3.2 Six-source evidence merge

The native observations are merged with five historical evaluation graphs. The fixed conflict priority is

$$
\text{modellens\_v2}
\succ \text{d0\_v1\_5}
\succ \text{a\_ctrl\_2000m}
\succ \text{hf\_effective}
\succ \text{diverse\_zoo}
\succ \text{hf\_model\_index}.
$$

For historical sources, the stored weights were already direction-oriented. The merge first takes the median within `(source, node, model)`, then applies min--max normalization within `(node, source)`, using 0.5 for constant groups. If multiple sources provide the same `(node, model)` pair, the highest-priority source supplies the retained edge and the losing rows are written to a conflict table. Finally, each node is capped at 200 edges by deterministic, weight-stratified sampling with NumPy seed 0. This procedure is implemented in [`scale1m/merge_supervision.py`](../../scale1m/merge_supervision.py).

The merge starts from 531,958 rows, removes 5,102 within-source duplicates, records 1,502 cross-source conflicts, and retains 525,354 distinct pairs before capping. The final model--dataset evidence graph contains **247,803 directed supervision edges**. Their retained source counts are:

| Source | Retained edges |
|---|---:|
| `modellens_v2` | 117,898 |
| `hf_model_index` | 73,670 |
| `d0_v1_5` | 45,992 |
| `hf_effective` | 5,223 |
| `a_ctrl_2000m` | 3,670 |
| `diverse_zoo` | 1,350 |
| **Total** | **247,803** |

The rule file `rf-gold-2.0` has SHA-256 `be3fb05ecf92ea0112ba4de10b8191d71dfe096b8667a86911c2ebaeadb9cdb1`, bound by [A0.1 input audit](A0_runs/audit/A0_INPUT_AUDIT.json). A0.7 reconstructed the merged supervision, node, and conflict tables and matched every row and column after stable semantic sorting, with no numeric tolerance. The merge recount and exact output-identity check are in [A0.7 source recount](A0_runs/A0_7/results/A0_SOURCE_COUNTS.json). These are source-data names, not alternative evaluated retrieval systems.

### 3.3 Candidate closure and row-order contract

Supervision refers to 12,680 models that are absent from the 2026-08-18 hub snapshot. They are appended to the candidate universe rather than dropping their observations:

$$
3{,}003{,}759+12{,}680=3{,}016{,}439.
$$

The row map is frozen as

$$
\operatorname{mappedID}(m_i)=i,
$$

with the hub snapshot as an exact prefix and the normalized historical-only identifiers appended in sorted order. In the frozen construction, historical-only rows have unknown size, take family identity from the first available historical source, and use the same descriptor and feature encoder as the other rows. A0 reuses that verified model-feature matrix. Dataset--task nodes are sorted deterministically and assigned the same contiguous `mappedID` contract.

The model and dataset ladders have SHA-256 values `fee360d1fef9afa4c446b70af35bcbf4ead2c75387f47abc3b0d21c944ae3efa` and `31c027ff2eeb693aed6abaa5d235f4a6264bb4b3245bfd71cf1c8b85d32173cb`. The row maps, snapshot-prefix closure, and edge endpoints are verified by [A0.1 input audit](A0_runs/audit/A0_INPUT_AUDIT.json). The implementation is [`scale1m/build_ladder_rf.py`](../../scale1m/build_ladder_rf.py).

### 3.4 Model-node features

For a repository name $n$, let $T(n)$ contain the lowercased full name, its repository basename, and tokens obtained by splitting on `/`, `_`, `-`, and whitespace, with order-preserving deduplication. Let $R\in\mathbb{R}^{10000\times64}$ be a row-normalized Gaussian table generated with seed 42 and let

$$
h(t)=\operatorname{MD5}(t)\bmod 10000.
$$

The frozen name representation is

$$
e_{\mathrm{name}}(n)=\frac{1}{|T(n)|}\sum_{t\in T(n)}R_{h(t)}\in\mathbb{R}^{64}.
$$

The textual model descriptor contains the cleaned repository identifier, the canonical family, and a parameter-size phrase when the safetensors count is known. It is encoded with `all-MiniLM-L6-v2` without output normalization:

$$
e_{\mathrm{desc}}(m)=\operatorname{MiniLM}(\operatorname{descriptor}(m))
\in\mathbb{R}^{384}.
$$

The frozen model matrix is therefore

$$
x_m^{\mathrm{frozen}}=
[e_{\mathrm{name}}(m)\;\Vert\;e_{\mathrm{desc}}(m)]
\in\mathbb{R}^{448}.
$$

Family and known parameter size already appear in the descriptor text. Their discrete family and size-bucket IDs additionally select learned 16-dimensional tables during graph training. Size bucket 0 means unknown; buckets 1--14 partition $\log_{10}$ parameter count from $10^5$ to $10^{12}$ in half-decade intervals. Family ID 0 is `Other`, and a dynamically observed family receives its own row only after at least three occurrences.

The final `x_m` matrix has shape `[3,016,439, 448]`, dtype `float32`, file size 5.41 GB, and SHA-256 `ba1020872ddb6e90726c755f9f08f236f4c73e506013c91d8a0075fc0c0361e6`. The append-only family vocabulary has 41,056 rows and SHA-256 `00d304df9bd72acb2c2c1ca5b84ed0dc9d4adb0cbc9f098f32a9e62109d24005`. Unknown size accounts for 71.929% of models and `Other` for 13.561%. The 41,056 rows are the embedding index cardinality; 40,927 distinct family IDs actually occur in the graph. Feature construction is in [`scale1m/embed_lake_rf.py`](../../scale1m/embed_lake_rf.py); current shape, cardinality, and coverage measurements are in [A0.7 source recount](A0_runs/A0_7/results/A0_SOURCE_COUNTS.json), with frozen-file hashes in [A0.1 input audit](A0_runs/audit/A0_INPUT_AUDIT.json).

### 3.5 Dataset-node features

The dataset descriptor concatenates a cleaned dataset--task name, card task categories, up to ten non-colon tags, and at most 400 description characters. If no trustworthy card match exists, only the cleaned name and task are used. With a name-hash table generated using seed 43, the frozen dataset vector is

$$
x_d^{\mathrm{frozen}}=
[e_{\mathrm{name}}^{64}(d)\;\Vert\;
 e_{\mathrm{card}}^{384}(d)\;\Vert\;
 e_{\mathrm{stats}}^{10}(d)]
\in\mathbb{R}^{458}.
$$

In A0, this ten-slot block contains only a node-table structural count. If $r_d$ is the number of frozen dataset–task nodes with the same root,

$$
e_{\mathrm{stats}}^{A0}(d)=
[0,0,0,0,0,0,\log(1+r_d),0,0,0].
$$

All 18,729 nodes have columns 448–453 and 455 set exactly to zero. These seven positions formerly carried two observation counts, the performance mean/standard deviation/minimum/maximum, and `gold_eligible`. A0 keeps the 458-dimensional schema: column 454 retains the root-node count, columns 456–457 remain reserved zeros, and the other 451 columns retain their frozen values. Thus the feature repair removes those performance-derived inputs while preserving the text blocks, row maps, graph topology, split identities, and encoder dimensions. The eligibility flag remains an evaluation-cohort rule outside the learned input. The repaired graph digest is

```text
acddeb93d926efcb636c83d452fe360736a0e6e4ac68cab3f5fef210e7cef5db
```

Dataset task-type, class-count-bucket, and arity IDs additionally select learned tables of dimensions 16, 8, and 4. Each table has one row in this artifact, so the discrete IDs are constant across nodes. Task strings still affect node identity and the MiniLM descriptor. The task prior separately derives its task groups from the canonical task strings. [A0.2 repair and verification](A0.2.md), [feature preparation](../../scale1m/prepare_a0_graph.py), [feature builder](../../scale1m/build_graph_rf.py), [A0.7 source recount](A0_runs/A0_7/results/A0_SOURCE_COUNTS.json).

### 3.6 Typed graph relations

The model--dataset evidence graph is

$$
G=(V_M\cup V_D,E_{MD}\cup E_{DM}\cup E_{DD}\cup E_{MM}\cup E_{MM}^{-1}),
$$

with the following relations:

1. `model --trained_on--> dataset` carries the normalized performance value $y_{md}$.
2. `dataset --rev_trained_on--> model` mirrors each supervision edge for bidirectional message flow.
3. `dataset --similar_to--> dataset` connects each dataset to its 20 nearest neighbors by cosine similarity of the 384-dimensional card block.
4. `model --is_base_of--> model` points from a resolved parent to its derivative.
5. `model --rev_is_base_of--> model` mirrors the lineage relation.

For dataset similarity,

$$
w_{ij}^{DD}=\frac{e_{\mathrm{card}}(d_i)^\top e_{\mathrm{card}}(d_j)}
{\|e_{\mathrm{card}}(d_i)\|_2\|e_{\mathrm{card}}(d_j)\|_2},
$$

and the self edge is excluded before selecting the 20 largest values. The stored graph contains:

| Relation | Directed edges in the stored graph |
|---|---:|
| `trained_on` | 247,803 |
| `rev_trained_on` | 247,803 |
| `similar_to` | 374,580 |
| `is_base_of` | 859,065 |
| `rev_is_base_of` | 859,065 |
| **Total** | **2,588,316** |

The A0 input audit verifies the stored lineage endpoints and reverse relation. Before training, dataset similarity is deterministically reduced to the top 10 neighbors per source and its retained edge attributes are set to one, yielding 187,290 topology-only similarity edges. This training-time transformation is `apply_similar_to_mode(..., mode="topk_unweighted", k=10)` in [`stage2TrainGraphSAGE/graph_surgery.py`](../../stage2TrainGraphSAGE/graph_surgery.py).

The graph is stored as memory-mapped `.npy` feature arrays, NPZ node/edge structures, Parquet row maps, and hashed JSON metadata. The storage loader can map the 5.41 GB model matrix; training subsequently clones graph data and transfers it to the GPU. The sharded format is implemented in [`scale1m/graph_store.py`](../../scale1m/graph_store.py). The counts above are verified in [A0.1 input audit](A0_runs/audit/A0_INPUT_AUDIT.json); A0.2 preserves their bytes while replacing the seven input columns and binding the repaired graph digest stated in §3.5. The full-scale resource requirement is measured in [A0.3](A0.3.md) and [A0.4](A0.4.md).

## 4. Offline stage 3: structure-aware graph training

### 4.1 Root-aware supervision split

The split unit is a dataset root, not an edge or a dataset configuration. All nodes sharing a root are assigned to the same side. With split seed $s\in\{0,1,2\}$, roots are shuffled once and greedily assigned until approximately 20% of supervision edges are in test, 10% in validation, and the remainder in training.

Let $E_{\mathrm{tr}},E_{\mathrm{val}},E_{\mathrm{te}}$ denote the positive edges induced by these root partitions. Thirty percent of $E_{\mathrm{tr}}$ becomes disjoint supervision $E_{\mathrm{sup}}$ and is removed from the training message graph. Consequently,

$$
\begin{aligned}
G_{\mathrm{train}} &: E_{\mathrm{tr}}\setminus E_{\mathrm{sup}},\\
G_{\mathrm{val}} &: E_{\mathrm{tr}},\\
G_{\mathrm{test}} &: E_{\mathrm{tr}}\cup E_{\mathrm{val}}.
\end{aligned}
$$

Every removed forward edge is removed from `rev_trained_on` as well. Binary negative examples are sampled at a 1:1 ratio from model--dataset pairs absent from the complete positive edge set, although the performance-ranking targets themselves use only positive edges and read their oriented values from the full graph. The split implementation and leakage assertions are in [`stage2TrainGraphSAGE/d0_splits.py`](../../stage2TrainGraphSAGE/d0_splits.py).

### 4.2 Node encoders

The model-side input to message passing is

$$
h_m^{(0)}=W_M
[x_m^{\mathrm{frozen}}\;\Vert\;
 E_{\mathrm{size}}[b_m]\;\Vert\;
 E_{\mathrm{family}}[f_m]]+b_M,
$$

where $x_m^{\mathrm{frozen}}\in\mathbb{R}^{448}$, both learned lookup vectors have dimension 16, and $W_M:\mathbb{R}^{480}\rightarrow\mathbb{R}^{128}$. There is no model-ID embedding.

The dataset-side input is

$$
h_d^{(0)}=W_D
[x_d^{\mathrm{frozen}}\;\Vert\;
E_{\mathrm{task}}[t_d]\;\Vert\;
E_{\mathrm{class}}[c_d]\;\Vert\;
E_{\mathrm{arity}}[a_d]]+b_D,
$$

where $W_D:\mathbb{R}^{486}\rightarrow\mathbb{R}^{128}$. The frozen feature matrices do not receive gradients; the semantic lookup tables, projections, graph layers, relation gates, and output head are learned. Encoder assembly is implemented in [`stage1BuildTransferGraph/dataset_embed/model_node_encoder.py`](../../stage1BuildTransferGraph/dataset_embed/model_node_encoder.py) and instantiated by `HeteroGraphSAGE` in [`stage2TrainGraphSAGE/model.py`](../../stage2TrainGraphSAGE/model.py).

### 4.3 Relation-specific GraphSAGE

The final network uses one heterogeneous message-passing layer. For relation $r$ and destination node $v$, let $\mathcal{N}_r(v)$ be the incoming neighbors. The relation block computes

$$
\tilde h_{v,r}=
W_{\mathrm{self},r}h_v^{(0)}+b_{\mathrm{self},r}+
W_{\mathrm{nbr},r}
\left(
\frac{1}{|\mathcal{N}_r(v)|}
\sum_{u\in\mathcal{N}_r(v)}h_u^{(0)}
\right).
$$

Incoming relation blocks are combined with learned scalar gates initialized to one:

$$
h_v^{(1)}=
\sum_{r:\operatorname{dst}(r)=\operatorname{type}(v)}g_r\tilde h_{v,r}.
$$

Every relation has its own self and neighbor projections; the self projection includes a bias. An empty per-node neighborhood contributes a zero neighbor mean. Globally empty relation tensors are skipped. Because the frozen configuration sets `weighted_relations=[]`, continuous `edge_attr` values are **not** used inside message aggregation. Performance values still supervise the ranking and positive-set objectives, and card cosine still determines which dataset-neighbor topology is retained, but the message operator itself is an unweighted per-relation mean. This exact operator is `WeightedSAGEConv` plus `EdgeAwareHetero` in [`stage2TrainGraphSAGE/edge_aware.py`](../../stage2TrainGraphSAGE/edge_aware.py).

A shared linear head maps both node types to the retrieval space:

$$
z_v=\frac{W_Oh_v^{(1)}+b_O}
{\|W_Oh_v^{(1)}+b_O\|_2},\qquad z_v\in\mathbb{R}^{128}.
$$

Sharing $W_O$ and normalizing every row places model and dataset nodes in the same inner-product space. The frozen configuration does not use separate heads.

### 4.4 Rank-consistent local objective

For query node $d$, let $y_{dm}$ be the oriented supervision value and define the preference pairs

$$
\mathcal{P}_d=
\{(m_i,m_j):y_{di}-y_{dj}>0.02\}.
$$

The local RankNet term is

$$
\mathcal{L}_{\mathrm{rank}}=
\frac{1}{|\mathcal{D}_B|}
\sum_{d\in\mathcal{D}_B}
\frac{1}{|\mathcal{P}_d|}
\sum_{(i,j)\in\mathcal{P}_d}
\operatorname{softplus}
\left(-\frac{s_\theta(d,m_i)-s_\theta(d,m_j)}{0.1}\right).
$$

The averaging is first within a dataset and then across datasets, so dense query nodes do not dominate merely by producing more pairs. In the formula, $\mathcal D_B$ includes datasets with at least one retained preference pair, and $\mathcal P_d$ denotes the retained subset when sampling is required. At most 256 pairs are retained per dataset; when a node has more, half are the most inverted current pairs and the other half are sampled from the remainder. The objective operates directly on the raw unit-vector dot product used by retrieval. Its implementation is `raw_dot_ranknet_loss` in [`stage2TrainGraphSAGE/losses.py`](../../stage2TrainGraphSAGE/losses.py).

### 4.5 Model--model contrastive objective

Training-visible observations define a sparse high-performer membership matrix

$$
A_{md}=\mathbb{1}
\left[m\in\operatorname{Top}_{\max(1,\operatorname{round}(0.1n_d))}
\{y_{dm'}\}_{m'}\right].
$$

Models $i$ and $j$ form a positive pair when they are both selected for at least one common dataset:

$$
(i,j)\in\mathcal{P}^{MM}
\iff i\ne j\;\land\;\sum_d A_{id}A_{jd}>0.
$$

The matrix is stored as CSR/CSC membership lists rather than a dense $N\times Q$ block. In a sampled graph batch, positive pairs are materialized sparsely and deduplicated. For each anchor, 256 batch model IDs are drawn uniformly with replacement; self IDs and sampled positives are discarded from the negative term. If a negative shares the anchor's lineage component but is not a positive, its denominator weight is two; otherwise the weight is one. With $\phi(i,j)=z_i^\top z_j/0.2$, the implemented sampled objective is

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

The membership construction is `topk_membership`, sparse pair generation is `batch_positive_pairs`, and the loss is `contrastive_loss_sampled` in [`losses.py`](../../stage2TrainGraphSAGE/losses.py) and [`sampling.py`](../../stage2TrainGraphSAGE/sampling.py).

### 4.6 Whole-lake sampled-softmax objective

The global term exposes each query direction to negatives drawn from the complete candidate lake. Let

$$
\deg(m)=|\{d:(m,d)\in E_{\mathrm{tr}}\}|
$$

be the train-visible supervision degree. The tempered degree proposal and the uniform-over-labeled proposal are

$$
q_{\deg}(m)=
\frac{(\deg(m)+1)^{0.75}}
{\sum_{m'}(\deg(m')+1)^{0.75}},
\qquad
q_{\mathrm{lab}}(m)=
\frac{\mathbb{1}[\deg(m)>0]}
{|\{m':\deg(m')>0\}|}.
$$

The final proposal is the fixed mixture

$$
q(m)=0.5q_{\deg}(m)+0.5q_{\mathrm{lab}}(m).
$$

For each training step, 128 positive-bearing dataset nodes are sampled. For each selected node $d$, the positive set $P_d$ is the top-10% membership set above, and 256 negatives are sampled with replacement from $q$; sampled positives are removed. With $T_g=0.1$, the implemented logQ-corrected term is

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

followed by a mean across sampled dataset nodes. This term performs a full training-message-graph forward, with fresh edge dropout, because batch subgraphs do not contain the complete negative universe. `build_lake_logq` and `global_lake_loss` in [`stage2TrainGraphSAGE/losses.py`](../../stage2TrainGraphSAGE/losses.py) implement the proposal and objective.

### 4.7 Final optimization problem and frozen configuration

The final objective is

$$
\boxed{
\mathcal{L}=\mathcal{L}_{\mathrm{rank}}
+\mathcal{L}_{\mathrm{contrast}}
+\mathcal{L}_{\mathrm{global}}
}
$$

with unit coefficient on every term. MSE, embedding-uniformity regularization, dataset-to-model contrastive loss, hard-negative mining, positive inverse-propensity weighting, separate projection heads, and early stopping are disabled.

| Component | Frozen value |
|---|---:|
| GraphSAGE depth / hidden width / output width | 1 / 128 / 128 |
| Optimizer / learning rate | Adam / 0.01 |
| Epochs | 25 |
| Supervision batch size | 1,024 |
| Neighbor expansion | two fan-out hops, 10 per ordinary relation and 20 per lineage relation in the fallback loader |
| Ordinary / lineage message-edge dropout | 0.30 / 0.05 |
| Rank temperature / minimum target gap | 0.10 / 0.02 |
| Contrastive temperature / sampled negatives | 0.20 / 256 |
| Global temperature / sampled negatives | 0.10 / 256 |
| Proposal exponent / smoothing / mixture | 0.75 / 1.0 / 0.50 |
| Dataset nodes per global step | 128 |
| Similar-dataset topology | top 10, unweighted |
| Split seeds / initialization seed | 0, 1, 2 / 0 |
| Export inference chunk | 50,000 nodes |

Mini-batches are rooted at disjoint training supervision edges. The fallback sampler expands the current frontier without replacement and imposes a hard per-node fan-out bound. Edge dropout changes only message edges; supervision labels are stored separately and are never dropped. The full training loop is [`stage2TrainGraphSAGE/train.py`](../../stage2TrainGraphSAGE/train.py), while the scale-safe sampler is [`stage2TrainGraphSAGE/sampling.py`](../../stage2TrainGraphSAGE/sampling.py).

The actual A0 configurations are [`seed 0`](A0_runs/A0_4/delivery_s0/metadata/resolved_config.json), [`seed 1`](A0_runs/A0_4/delivery_s1/metadata/resolved_config.json), and [`seed 2`](A0_runs/A0_4/delivery_s2/metadata/resolved_config.json). Each records the command, runtime, repaired graph digest, 40 effective configuration keys, split seed, and initialization seed. Each completed 25 epochs and passed the mechanism gate. A0 uses `ckpt/last.pt` at zero-based epoch 24 for all three exports. These are newly trained runs, each with initialization seed 0. [A0.4 completion and hashes](A0.4.md).

### 4.8 Held-out exports and the tested information boundary

The exporter writes two pairs of 128-dimensional float32 arrays from each new last checkpoint:

```text
z_m.npy,      z_d.npy       # full-message graph
z_m_eval.npy, z_d_eval.npy  # training+validation performance edges only
```

The reported indexes and query embeddings use only `z_m_eval.npy` and `z_d_eval.npy`. Their graph excludes every test performance edge and its reverse. Full-message arrays are separate export artifacts. Their existence does not establish a separately measured full-message production index.

A0.5 checks all exported rows for finite values and L2 norm error below `1e-5`, checks chunked versus full inference within `1e-5`, and verifies model/dataset row order. The newly exported query IDs, roots, candidate IDs, and oriented values match all 4,122 frozen split observations. [A0.5 validation](A0.5.md), [exporter](../../scale1m/export_rf.py).

A0.3 also perturbs test-root performance values and the eligibility input on a fixed fixture. After the seven-column repair, the features, training-used supervision, global proposal, trained parameters, history, and held-out forward output remain identical with fixed RNG. This is a targeted test of the repaired information boundary; the frozen node population and metadata topology still define the evaluation setting. [A0.3 information-boundary test and full-scale smoke](A0.3.md), [test source](../../scale1m/tests/test_a03_information_boundary.py).

## 5. Evaluation and final retrieval results

### 5.1 Query identity, labels, and metric definitions

The evaluated queries are dataset–task nodes already present in the frozen graph. All nodes sharing a root are assigned to the same supervision split. Query eligibility requires a known metric direction or a curated oriented source, a non-reinforcement-learning task, a non-placeholder dataset name, at least three observed held-out candidates, and nonconstant oriented values. The three seeds contain 1,476 / 1,101 / 1,545 query observations and 730 / 495 / 473 distinct query roots. Their summed query count, 4,122, is a count of observations across splits; nodes may recur across seeds. [Frozen query identities](A0_runs/audit/A0_QUERY_IDENTITY.jsonl), [A0.1 input audit](A0_runs/audit/A0_INPUT_AUDIT.json).

For query $d$, its gold model $g_d$ is the observed held-out candidate with maximum oriented value; the frozen candidate order and `argmax` resolve a tie in that label. Let $R_{10}(d)$ be the actual ten returned model IDs. Then

$$
\mathrm{gold@10}=\frac{1}{|\mathcal Q_s|}\sum_{d\in\mathcal Q_s}
\mathbb 1[g_d\in R_{10}(d)].
$$

`gold@1` uses the first return; `top3@10` accepts any of the three best observed candidates selected by the frozen scorer; `gold-gap@10` accepts any observed candidate within 0.01 of the best oriented value. Root-macro metrics first average query indicators within each root, then average roots. The main three-seed means weight seeds equally. All ratios are dimensionless; percentages are stated explicitly.

For a score function $a_d(m)$ and the fixed unique tie key $\tau(m)$, full-lake rank is

$$
\operatorname{rank}_d(m)=1+\sum_{j=1}^{N}
\mathbb 1\!\left[a_d(m_j)>a_d(m)\ \lor\
\big(a_d(m_j)=a_d(m)\land\tau(m_j)<\tau(m)\big)\right].
$$

Exact diagnostic ranks use that order. A0 aligns the saved exact full-fusion top10 with the same tie order and reads probe scores from the same-shaped float32 block GEMM as the full scan. These A0.2 consistency fixes matter when interpreting changes from earlier results. The bounded 1,000-candidate outputs support a conditional reranked gold position when the gold enters the pool; a full-lake gold rank and its normalized/random-reference variants are undefined for that bounded output. [A0.2 exact consistency changes](A0.2.md), [independent scorer](../../scale1m/recompute_a0.py).

### 5.2 HNSW construction and label-free calibration

For unit vectors, inner-product order is equivalent to ascending $1-z_d^\top z_m$ and to ascending squared Euclidean distance. A0 builds `hnswlib.Index(space="ip", dim=128)` with `M=32`, `ef_construction=200`, and eight construction threads; model row $i$ is inserted under label `mappedID=i`. The three files are `data1m/a0_20260912/metrics/hnsw_a0_s{0,1,2}.bin`. The as-run builder is `run_hnsw` in the [A0 evaluator](../../scale1m/a0_evaluation.py).

For each split, calibration compares HNSW's 1,000 IDs with the same query's exact dense top1,000:

$$
\operatorname{Recall@1000}(ef)=\frac{1}{|\mathcal Q_s|}
\sum_{d\in\mathcal Q_s}\frac{|\operatorname{ANN}_{1000}(d;ef)\cap
\operatorname{Exact}_{1000}(d)|}{1000}.
$$

The frozen search grid is 1,000, 1,500, 2,000, 3,000, 5,000. The first value with mean recall at least 0.99 is selected; later values are marked not applicable under the stopping rule. Calibration uses neighbor-ID agreement on the materialized evaluation queries and does not read their gold labels. It is a calibration of ANN fidelity on that query set.

| Split seed | Selected ef_search | Recall@1000 |
|---|---|---|
| 0 | 1,000 | 0.99438144 |
| 1 | 1,500 | 0.99626067 |
| 2 | 1,000 | 0.99282006 |

### 5.3 Task-prior evidence and online work

The sidecars contain 198,216 / 196,912 / 196,124 visible training+validation edges for seeds 0 / 1 / 2. They derive 2,198 task groups from the canonical `task` column, lowercasing and replacing whitespace/underscores with hyphens. This task mapping is distinct from the constant dataset `task_type_id` encoder field. Each visible record is grouped by task and model, then shrunk by the formula in §1. The absence of a visible model–task record returns zero.

The evaluator verifies exact model/dataset row-map agreement, root equivalence, the split identity, and the absence of every scored test root from the visible prior edges. This excludes both the query node and its same-root siblings. A0.5 independently sums and counts the records and checks the actual prior consumer against the formula. [A0.5 validation](A0.5.md), [sidecar producer](../../stage3HNSW/build_prior_sidecar.py), [prior consumer](../../scale1m/eval_y2.py).

```text
z_q = stored_dataset_task_embedding[query_id]
ids, cosine = hnsw_top1000(z_q)
prior = task_prior[query_task, ids]
fused = (cosine + 1) / 2 + prior
return ids[order_by(descending=fused, ascending=fixed_tie_key)[:10]]
```

### 5.4 Final quality and exact diagnostic references

All rows below use the same new GD representations, candidate universe, held-out queries, and fixed prior/fusion. The HNSW row is the final system. The exact rows measure the cost and quality implications of candidate truncation and ANN approximation.

| Path | Seed 0 gold@10 | Seed 1 gold@10 | Seed 2 gold@10 | Equal-weight mean |
|---|---|---|---|---|
| Exact full-lake fusion (offline reference) | 0.3604 | 0.3542 | 0.2375 | 0.3174 |
| Exact dense top1000 + prior (offline reference) | 0.3327 | 0.3370 | 0.2214 | 0.2970 |
| **HNSW top1000 + prior (final)** | 0.3320 | 0.3370 | 0.2214 | 0.2968 |

| Split seed | gold@1 | gold@10 | top3@10 | gold-gap@10 | Root-macro gold@10 | Median gold position, conditional on retrieval |
|---|---|---|---|---|---|---|
| 0 | 0.1585 | 0.3320 | 0.3740 | 0.3550 | 0.2515 | 4 |
| 1 | 0.1217 | 0.3370 | 0.4096 | 0.3660 | 0.2566 | 5 |
| 2 | 0.1023 | 0.2214 | 0.2628 | 0.2311 | 0.2471 | 4 |
| Equal-weight mean | 0.1275 | 0.2968 | 0.3488 | 0.3174 | 0.2517 | — |

The final unrounded `gold@10` values are 0.3319783197831978 / 0.3369663941871026 / 0.22135922330097088. The metrics are historical observed-ranking recovery; each candidate model is not executed during this retrieval evaluation.

| Diagnostic | Seed 0 | Seed 1 | Seed 2 | Mean of seed ratios |
|---|---|---|---|---|
| Exact-pool retention | 92.2932% | 95.1282% | 93.1880% | 93.5365% |
| ANN retention | 99.7963% | 100.0000% | 100.0000% | 99.9321% |
| Overall retention | 92.1053% | 95.1282% | 93.1880% | 93.4738% |
| Gold coverage in exact top1000 | 53.8618% | 57.4024% | 37.0874% | 49.4505% |
| Full-fused top10 coverage in exact top1000 | 88.1504% | 90.7084% | 92.1165% | 90.3251% |
| Gold coverage in actual HNSW top1000 | 53.9295% | 57.4024% | 37.0874% | 49.4731% |

Exact-pool retention is `exact1000 / exact_full_lake`, ANN retention is `hnsw1000 / exact1000`, and overall retention is `hnsw1000 / exact_full_lake`, using `gold@10` within each seed before averaging ratios. Exact-pool gold coverage is the fraction of queries whose gold ID enters the dense pool. Full-fused-top10 coverage averages the fraction of exact full-fusion top10 IDs already in that pool. These diagnostic quantities help separate pool truncation from ANN error. Source for all tables in §5: [A0.7 recomputation report](A0_runs/A0_7/results/A0_REPORT.json), [complete inventory](A0_runs/A0_7/results/A0_METRIC_INVENTORY.json), [independent validation](A0_runs/A0_7/results/A07_VALIDATION.json).

### 5.5 Retrieval latency, index storage, and measured offline costs

The formal A0.6 evaluation ran on `watgpu308` under Linux, Python 3.11.4, NumPy 2.4.6, PyTorch 2.12.0+cu130, and hnswlib 0.8.0. Exact vector computation used its NVIDIA L40S; HNSW construction/search and prior fusion ran on the host CPU. Each seed has one timed query pass, using one HNSW query thread after up to 50 warm-up queries. Timings start from a precomputed query embedding and cover HNSW plus candidate-prior lookup, fusion, and ordering. Query encoding, loading the index/prior tables, constructing the index, and client/network overhead lie outside that measured interval. Local RTX 4060/i7 verification and the later local independent recomputation have separate execution records. [A0.6 report and runtime binding](A0_runs/A0_6/delivery/metrics/A0_EVALUATION_MANIFEST.json), [A0.6 execution](A0.6.md).

| Seed | HNSW p50 ms | HNSW p95 ms | Prior/fusion p50 ms | Prior/fusion p95 ms | Total p50 ms | Total p95 ms |
|---|---|---|---|---|---|---|
| 0 | 0.512913 | 0.804156 | 0.178343 | 0.229241 | 0.693406 | 1.014304 |
| 1 | 0.716782 | 1.148937 | 0.169012 | 0.209267 | 0.882341 | 1.337841 |
| 2 | 0.536364 | 0.754903 | 0.124475 | 0.208460 | 0.666437 | 0.953115 |
| mean | 0.588686 | 0.902665 | 0.157277 | 0.215656 | 0.747395 | 1.101754 |

Each total sample is the paired sum of its HNSW and reranking samples. Percentiles are recomputed from the original per-query nanosecond arrays; the mean row averages the three seed percentiles. It is neither a percentile over pooled queries nor a sum of component percentiles.

| Seed | Index bytes | Index GiB | Build seconds |
|---|---|---|---|
| 0 | 2,377,803,152 | 2.214502 | 196.151983 |
| 1 | 2,377,803,548 | 2.214502 | 208.427164 |
| 2 | 2,377,803,284 | 2.214502 | 217.314888 |

Index build time includes initialization, insertion, serialization, and final file replacement. Total index storage is 7,133,409,984.0 bytes, or 6.643506 GiB. Here GB denotes $10^9$ bytes and GiB denotes $2^{30}$ bytes.

| Recorded interval (seconds) | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Training segments | 3875.623108 | 3911.995431 | 3569.225913 |
| Export segments | 36.909913 | 39.057960 | 40.113739 |
| Prior construction | 7.548296 | 7.497223 | 7.522241 |
| Shared exact-reference scan | 8.752328 | 6.686751 | 7.996606 |
| Recorded exact + HNSW evaluation segments | 210.592887 | 220.728926 | 231.004163 |

Training time uses the recorded `train_eval_one` segments, including setup, training/checkpoint work, and the existing post-training evaluations. Export time uses the recorded export segments, including checks; it differs from the inner embedding-only timer. Prior time includes its input verification. The two exact references share one scan and therefore the same time interval. Per-seed evaluation time sums the exact and HNSW stage segments once, including index construction; global binding checks and finalization are outside those segments. Consequently, neither the second exact label nor index-build time should be added again. These measured intervals also exclude queueing and failed environment-setup attempts. A0.4 training used `watgpu308`/L40S, A0.5 export used `watgpu608`/RTX 6000 Ada, and A0.6 evaluation used `watgpu308`/L40S plus CPU. [A0.4](A0.4.md), [A0.5](A0.5.md), [A0.6](A0.6.md), [A0.7 source descriptors](A0_runs/A0_7/results/A0_RECORD_SOURCES.json).

| Recorded resource (GiB) | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Training process RSS high-water | 32.686615 | 32.641811 | 32.884975 |
| Training GPU allocated peak | 37.947710 | 37.964684 | 37.931339 |
| Evaluation process RSS high-water | 5.952076 | 5.991192 | 6.071606 |
| Evaluation GPU allocated peak | 2.026332 | 2.026332 | 2.026332 |

GPU values are PyTorch allocated-memory peaks. RSS values are OS process high-water marks, which can include allocations from earlier work in the same process; they are not isolated incremental per-seed working-set requirements. The complete per-epoch losses, source descriptors, resource values, and producer records are retained in the [A0.7 report](A0_runs/A0_7/results/A0_REPORT.json) and [inventory](A0_runs/A0_7/results/A0_METRIC_INVENTORY.json).

## 6. Reproduction and immutable evidence

### 6.1 A0 stages and inputs

| Stage | Purpose | A0 evidence |
|---|---|---|
| A0.1 | Freeze the protocol, original inputs, and query identities | [Protocol](A0_runs/A0_PROTOCOL.json), [input audit](A0_runs/audit/A0_INPUT_AUDIT.json), [snapshot audit](A0_runs/audit/A0_SNAPSHOT_AUDIT.json) |
| A0.2 | Prepare the seven-column repair and verify unchanged scientific settings | [Execution record](A0.2.md), [producer and evaluator changes](A0_runs/A0_2/A0_2_MANIFEST.json) |
| A0.3 | Run regression, information-boundary testing, and full-scale smoke | [Execution record](A0.3.md), [regression JUnit](A0_runs/A0_3/regression.xml) |
| A0.4 | Train three new 25-epoch GD checkpoints | [Execution record](A0.4.md), [source preflight](A0_runs/A0_4/A04_PREFLIGHT.json), [seed 0](A0_runs/A0_4/delivery_s0/MANIFEST.json), [seed 1](A0_runs/A0_4/delivery_s1/MANIFEST.json), [seed 2](A0_runs/A0_4/delivery_s2/MANIFEST.json) |
| A0.5 | Export full/evaluation vectors, gold candidates, row maps, and priors | [Execution record](A0.5.md), [validation and file hashes](A0_runs/A0_5/STATUS.json) |
| A0.6 | Compute exact references, build/calibrate HNSW, measure final retrieval | [Execution record](A0.6.md), [raw-artifact manifest](A0_runs/A0_6/delivery/metrics/A0_EVALUATION_MANIFEST.json), [evaluator report](A0_runs/A0_6/delivery/metrics/A0_EVALUATION_REPORT.json) |
| A0.7 | Independently recompute raw metrics, costs, and source counts | [Execution record](A0.7.md), [report](A0_runs/A0_7/results/A0_REPORT.json), [inventory](A0_runs/A0_7/results/A0_METRIC_INVENTORY.json), [validation](A0_runs/A0_7/results/A07_VALIDATION.json) |

The A0.1 protocol retains the original English authority hash `6c7d883734fb9a05ca99c4a8409e366955c86ba2e644bd5328c9ba696ff344cd`. That frozen authority is an immutable input to the run; this revised library is its successor, not a replacement for its archived bytes. The original authority remains in [A0's frozen source](A0_runs/frozen/EVIDENCE_SOURCE_LIBRARY_en.md). Actual A0 training metadata records `git_head=not-a-git-repo` and `git_status=not-a-git-repo` for the isolated remote checkout. Source-file SHA bindings, graph/vocabulary/row-map hashes, checkpoint hashes, and export/evaluation manifests therefore provide the reproducibility chain.

| Seed | Run name | Last-checkpoint SHA-256 |
|---|---|---|
| 0 | `A0GD_full_s0_e25` | `9ee3415caf361e0ed50126c0d7992eb0989284282ebee8ed097addc6375650fb` |
| 1 | `A0GD_full_s1_e25` | `576db197c5d1429341a0b7d0fb35b98bbdcddda80b127a7414ceaba927123994` |
| 2 | `A0GD_full_s2_e25` | `9a090d58fd889b730299af226135fbb6727de5998a9fc133b9eaeb58138eb6f8` |

### 6.2 Actual computation order and command records

The complete as-run commands, environment setup, and retries are recorded in [A0.4](A0.4.md), [A0.5 run script](A0_runs/A0_5/run.sbatch), [A0.6 run script](A0_runs/A0_6/run.sbatch), and [A0.7 execution status](A0_runs/A0_7/STATUS.json). The scientific command bodies below use the actual remote directory layout; replay requires the frozen source, verified inputs, and empty output directories. Completed outputs remain bound evidence. A new replay should use separately recorded output locations with explicit path relocation.

```bash
cd /u801/x98liu/model_lake/a0_formal_20260913/ModelLakeFishing
DATA=/u801/x98liu/model_lake/data1m
RUNS=/u801/x98liu/model_lake/runs/A0_20260912
GRAPH="$DATA/a0_20260912/graph"
EXPORTS="$DATA/a0_20260912/exports"
METRICS="$DATA/a0_20260912/metrics"
OPS=/u801/x98liu/model_lake/a0_eval_20260914
```

The repaired graph is prepared once from the verified frozen graph by `scale1m.prepare_a0_graph` and accepted under the digest in §3.5; it is not rebuilt by rerunning current hub crawls or recomputing MiniLM text features. Its exact preparation/verification invocation is in [A0.2](A0.2.md).

For each `SEED=0`, `1`, `2`, A0.4 trains the same configuration:

```bash
python -m scale1m.train_rung \
  --rung full --graph "$GRAPH" \
  --out "$RUNS/A0GD_full_s${SEED}_e25" --seed "$SEED" --epochs 25 \
  --family-vocab "$DATA/feats_rf/family_vocab.csv" \
  --fanout --sparse-M --contrast-n-neg 256 \
  --chunked-infer 50000 --skip-diagnostics \
  --lake-gamma 0.5 --global-n-datasets 128
```

After all three runs pass, A0.5 exports each run and builds its prior:

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

A0.6 runs `exact`, then `hnsw`, then `finalize`, with all three seeds handled inside each stage:

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

The Linux protocol is an audited relocation of frozen path fields; scientific values and query identity stay fixed. Its [relocation record](A0_runs/A0_6/PATH_RELOCATION.json) binds those changes. A0.7 runs locally through [run.py](A0_runs/A0_7/run.py); its original [Windows path aliases](A0_runs/A0_7/PATH_ALIASES.json) document the initial execution mapping. The library revision separately preserves the formal source and authority bytes under `A0_runs/A0_7/library_revision/frozen_checkout/ModelLakeFishing`, with the current immutable [replay path map](A0_runs/A0_7/library_revision/REPLAY_PATH_MAP.json). These mappings resolve producer paths while preserving original artifact bytes. The source recount and independent reader are `scale1m.a0_source_counts` and `scale1m.recompute_a0`; [STATUS.json](A0_runs/A0_7/STATUS.json) records execution arguments. Finalized source descriptors are [A0_RECORD_SOURCES.json](A0_runs/A0_7/results/A0_RECORD_SOURCES.json). The independent reporting reader is separately versioned when reporting corrections are made; it does not replace the frozen formal training/evaluation code.

### 6.3 Independent recomputation boundary

A0.7 derives gold labels from the bound candidate/value arrays, independently reconstructs candidate prior fusion and the total tie order, and verifies final top10 membership, hit indicators, root averages, conditional positions, retention ratios, and ANN overlap. It recalculates percentiles from paired query nanosecond arrays and reads actual index bytes and producer timing/resource records. Full-lake rank summaries use saved raw integer all-model comparison counts, checked against the independently scored full-fusion top10; A0.7 does not repeat the full dense GPU scan. Source processing is separately recounted from frozen records, and reconstructed canonical tables must match all rows and columns. [Independent implementation](../../scale1m/recompute_a0.py), [source recount implementation](../../scale1m/a0_source_counts.py), [A0.7 validation](A0_runs/A0_7/results/A07_VALIDATION.json).

## 7. Evaluation scope and remaining limitations

The result establishes historical ranking recovery for materialized dataset–task nodes whose test performance edges are withheld in both directions. A0 removes the seven identified performance-derived input columns across every node and validates the repaired boundary. Node membership, text metadata, and dataset-similarity/lineage topology remain frozen and visible, so this is a transductive graph evaluation with held-out performance edges.

- **Observed ranking target.** `gold@10` concerns the best observed held-out model under the frozen oriented labels. Downstream model execution and performance on unobserved model–query pairs require separate evidence.
- **New dataset ingestion.** A dataset absent from the frozen node table would require an evaluated encoding and insertion procedure. The current measurements start from materialized dataset–task vectors.
- **Metadata coverage.** Matched cards cover 3,928 of 18,729 nodes; the other nodes use name/task text. Unknown model size accounts for 71.929% of candidates and `Other` family for 13.561%. Dataset task-type/class/arity embedding IDs are constant. These are measured properties of the inputs.
- **Scope of relational evidence.** The graph encoder uses unweighted relation means under `weighted_relations=[]`; oriented performance values supervise losses and task priors, while similarity values select topology.
- **Variation and calibration.** Three root split seeds share initialization seed 0. Their spread reflects those split runs rather than independent initialization experiments. ANN calibration uses the evaluation query vectors and dense ID agreement. One measured timing pass per seed characterizes this hardware/run setting.
- **Artifact distinction.** The measured HNSW indexes bind held-out evaluation embeddings. Full-message vector exports remain separate, with no separately measured production-index result in this edition.
- **Source completeness.** The original model-crawl API-page and discarded-duplicate event counters remain missing. A0.7 has zero invalid inventory entries and no missing seeds, but its all-item completeness flag remains false for those two counters.
- **Historical comparison boundary.** The A0 repair and the A0 exact tie/GEMM consistency changes both separate this run from earlier measurements; hardware also differs for latency. This edition's final results are the newly validated A0 values. A0's archived old/new comparison table preserves the earlier document's displayed precision as comparison context: [A0_COMPARISONS.csv](A0_runs/A0_7/results/A0_COMPARISONS.csv).

## 8. Paper-ready method summary

ModelLakeFishing canonicalizes frozen hub metadata and evaluation records into a typed graph with 3,016,439 model nodes, 18,729 dataset–task nodes, and 247,803 performance edges plus their reverse relation, dataset-similarity edges, and bidirectional lineage edges. Model features concatenate 64-dimensional name hashes and 384-dimensional MiniLM descriptions; learned size/family tables extend their input before projection. Dataset features combine the same text-block widths with ten fixed-schema slots, of which only the structural root-node count is active after A0's seven-column performance-feature removal. Learned schema tables and type-specific linear projections feed a one-layer relation-specific GraphSAGE encoder and a shared head, producing unit-normalized 128-dimensional retrieval vectors.

Training combines within-dataset RankNet, sampled model–model contrastive learning, and whole-lake logQ-corrected sampled softmax. Root-aware splits keep each dataset root on one side, and evaluation uses message graphs with test performance edges removed in both directions. For a materialized dataset–task query, HNSW selects 1,000 dense candidates and a fixed task prior from training/validation roots reranks that pool using $r=(\cos+1)/2+p_t(m)$; the system returns ten IDs. The A0 rerun obtains `gold@10` of 0.3320 / 0.3370 / 0.2214 (mean 0.2968), with ANN ID recall above 0.99 for every seed. Mean per-seed retrieval p50/p95 is 0.747/1.102 ms from precomputed query embeddings on the formal Linux host. These measurements support held-out historical ranking recovery within the frozen node table. [A0.7 recomputation report](A0_runs/A0_7/results/A0_REPORT.json).
