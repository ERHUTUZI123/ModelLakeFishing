# ModelLakeFishing: 3M-Scale Two-Stage Retrieval Evidence Report

**Evidence cutoff:** through the final Y4 rerun and system decision, 2026-09-06 (America/Toronto)

**Repository:** `D:\research\model_lake\codes\ModelLakeFishing`
**Evidence manifest:** [`EVIDENCE_SOURCE_MANIFEST.md`](EVIDENCE_SOURCE_MANIFEST.md)

This report describes the final system as a method, not as a chronology of experiments. The core path is: **collect and canonicalize metadata, construct a model--dataset evidence graph, train a dense retriever, retrieve 1,000 candidates with HNSW, and deterministically rerank that pool with task-level historical evidence**. Internal run names are retained only where needed to locate immutable evidence; they are not method names.

The evaluated system first retrieves 1,000 candidates with an inner-product HNSW index over the final graph-trained embeddings, then reranks that pool with a deterministic prior estimated from performance records on other training/validation datasets sharing the query task. The frozen fusion uses raw `beta=1` and shrink `k=5`, with no query-wise min--max normalization. Across 3,016,439 candidates and 1,476 / 1,101 / 1,545 eligible held-out queries, it obtains `gold@10` of **0.3279 / 0.3388 / 0.2427** (mean **0.3031**), while exact full-lake fusion reaches 0.3216. The actual system retains 94.25% of the full-lake fused score; HNSW reranking retains 99.93% of exact top-1,000 reranked `gold@10`. In the latest repeated timing, the retrieval-stage HNSW-plus-reranking path has p50 / p95 of 0.694 / 1.223 ms.

## 1. System formulation

Let

$$
\mathcal{M}=\{m_1,\ldots,m_N\},\qquad
\mathcal{D}=\{d_1,\ldots,d_Q\}
$$

denote the model lake and the set of dataset--task query nodes. In the frozen full-lake artifact,

$$
N=3{,}016{,}439,\qquad Q=18{,}729.
$$

The offline pipeline is the composition

$$
\text{hub records}
\xrightarrow{\text{canonicalize}}
(X_M,X_D,E)
\xrightarrow{\text{evidence graph}}
G
\xrightarrow{f_\theta}
(Z_M,Z_D)
\xrightarrow{\text{HNSW}}
\mathcal{I}(Z_M).
$$

Here, $X_M$ and $X_D$ are model and dataset metadata features, $E$ is typed relational evidence, $f_\theta$ is a heterogeneous GraphSAGE encoder, and every row of $Z_M,Z_D\in\mathbb{R}^{128}$ is L2-normalized. A query is a materialized dataset--task node $d=(\text{dataset},t)$. Its first-stage dense score is

$$
s_\theta(d,m)=z_d^\top z_m=\cos(z_d,z_m),
$$

and HNSW produces the candidate pool

$$
\mathcal P_{1000}(d)=
\operatorname*{ANN\text{-}Top1000}_{m\in\mathcal{M}}\;s_\theta(d,m).
$$

For model $m$ and normalized task $t$, let $n_{tm}$ and $A_{tm}$ be the count and sum of oriented performance values among the split-specific train+validation edges from roots other than the test-query root. The fixed task prior and second-stage score are

$$
p_t(m)=\frac{A_{tm}+0.5\times5}{n_{tm}+5},\qquad
r(d,m)=\frac{s_\theta(d,m)+1}{2}+p_t(m),
$$

with $p_t(m)=0$ when no visible record exists. The system returns the ten highest-scoring members of $\mathcal P_{1000}(d)$, using a fixed label-free permutation of model row IDs to break exact ties. The reranker is deterministic and non-learned; it is a real serving stage, but not an additional claimed representation-learning contribution.

| Pipeline stage | Concrete output | Main implementation |
|---|---|---|
| Collect metadata | Immutable model/dataset snapshots and canonical records | [`hf_crawl.py`](../../scale1m/hf_crawl.py), [`hf_crawl_datasets.py`](../../scale1m/hf_crawl_datasets.py), [`canonicalize_rf.py`](../../scale1m/canonicalize_rf.py) |
| Construct the evidence graph | Frozen row maps, node features, and five typed edge relations | [`build_ladder_rf.py`](../../scale1m/build_ladder_rf.py), [`embed_lake_rf.py`](../../scale1m/embed_lake_rf.py), [`build_graph_rf.py`](../../scale1m/build_graph_rf.py) |
| Train graph representations | Unit-normalized 128-dimensional model and dataset embeddings | [`model.py`](../../stage2TrainGraphSAGE/model.py), [`losses.py`](../../stage2TrainGraphSAGE/losses.py), [`train.py`](../../stage2TrainGraphSAGE/train.py), [`train_rung.py`](../../scale1m/train_rung.py) |
| Retrieve with HNSW | Inner-product ANN index bound to the model-embedding row map | [`eval_y2.py`](../../scale1m/eval_y2.py), [`export_ours.py`](../../scale/export_ours.py) |
| Rerank 1,000 candidates | Split-safe task prior, fixed fusion, and label-free tie-break | [`eval_rf.py`](../../scale1m/eval_rf.py), [`eval_y2.py`](../../scale1m/eval_y2.py) |

## 2. Offline stage 1: metadata collection and canonicalization

### 2.1 Immutable hub snapshots

The model collector enumerates the Hugging Face `/api/models` endpoint in descending `createdAt` order. Pagination follows the server-provided `Link: ... rel="next"` cursor rather than synthesizing page offsets. Each completed shard is compressed as JSONL-GZIP, hashed with SHA-256, and entered into a provenance record; the cursor file stores the next URL and committed record count so an interrupted crawl can resume without rewriting completed shards. The collector retains repository identity, timestamps, task/library tags, author, downloads, likes, safetensors metadata, structured base-model relations, and the model-card fields needed for lineage and evaluation extraction. The implementation is [`scale1m/hf_crawl.py`](../../scale1m/hf_crawl.py).

The frozen model crawl used

```powershell
python -m scale1m.hf_crawl `
  --sort createdAt --direction -1 --limit 100000000 --v2-fields `
  --out <DATA>/data1m/candidates_full
```

and terminated by cursor exhaustion, not by the numerical limit. It contains **3,003,759 unique model records**, written in 61 shards after 3,004 API pages. The archived run reports zero skipped duplicates and binds every shard by hash in [`F1_runs/PROVENANCE.json`](F1_runs/PROVENANCE.json).

Dataset metadata is collected independently from `/api/datasets` with the same cursor, resume, sharding, and hash discipline. The retained fields include repository identity, task categories, tags, description, language, size category, license, and source-dataset metadata. The frozen dataset crawl contains **1,008,417 repositories** in 11 shards and is recorded in [`F15_runs/PROVENANCE.json`](F15_runs/PROVENANCE.json). Its implementation is [`scale1m/hf_crawl_datasets.py`](../../scale1m/hf_crawl_datasets.py).

Both snapshots use 2026-08-18 as the snapshot date. The API `createdAt` endpoints observed during enumeration are fields of the returned records, not claims about the historical launch date of Hugging Face.

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

In the final node set, 3,928 of 18,729 nodes have an exact or parent Hugging Face card. Missing cards therefore remain a first-class condition rather than being filled by a popularity-based guess. Coverage measurements are archived in [`F15_runs/f15b_coverage.json`](F15_runs/f15b_coverage.json).

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

For each dataset--task node, a direction-known metric is preferred and the metric with the greatest number of records is selected, with deterministic lexical tie-breaking. The `trained_on` edge weight is the median oriented value for the resulting model--node pair. Of 2,158,375 parsed raw metric rows, 2,097,081 are finite and parseable; median deduplication produces 1,435,162 rows. Native primary-metric selection yields 143,478 edges before the per-node cap and 74,346 afterward. The exact construction is `build_supervision` in [`scale1m/canonicalize_rf.py`](../../scale1m/canonicalize_rf.py), with counts in [`F2_runs/F2_REPORT.json`](F2_runs/F2_REPORT.json).

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

The frozen rule file is `rf-gold-2.0`, SHA-256 `be3fb05ecf92ea0112ba4de10b8191d71dfe096b8667a86911c2ebaeadb9cdb1`; the merge counts and source table are in [`F2_runs/F2_MERGE_REPORT.json`](F2_runs/F2_MERGE_REPORT.json), and the rule artifact is [`F2_runs/rf_gold_rules.json`](F2_runs/rf_gold_rules.json).

### 3.3 Candidate closure and row-order contract

Supervision refers to 12,680 models that are absent from the 2026-08-18 hub snapshot. They are appended to the candidate universe rather than dropping their observations:

$$
3{,}003{,}759+12{,}680=3{,}016{,}439.
$$

The row map is frozen as

$$
\operatorname{mappedID}(m_i)=i,
$$

with the hub snapshot as an exact prefix and the normalized historical-only identifiers appended in sorted order. Historical-only rows have unknown size, take family identity from the first available historical source, and receive newly computed features under the same encoder as every other model. No old feature vector is copied. Dataset--task nodes are sorted deterministically and assigned the same contiguous `mappedID` contract.

The model and dataset ladders have SHA-256 values `fee360d1fef9afa4c446b70af35bcbf4ead2c75387f47abc3b0d21c944ae3efa` and `31c027ff2eeb693aed6abaa5d235f4a6264bb4b3245bfd71cf1c8b85d32173cb`. All eight identity and endpoint checks pass in [`F3_runs/LADDER_REPORT.json`](F3_runs/LADDER_REPORT.json). The implementation is [`scale1m/build_ladder_rf.py`](../../scale1m/build_ladder_rf.py).

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

Parameter size and family are not embedded inside these 448 dimensions. They are stored as integer columns and select learnable 16-dimensional tables during graph training. Size bucket 0 means unknown; buckets 1--14 partition $\log_{10}$ parameter count from $10^5$ to $10^{12}$ in half-decade intervals. Family ID 0 is `Other`, and a dynamically observed family receives its own row only after at least three occurrences.

The final `x_m` matrix has shape `[3,016,439, 448]`, dtype `float32`, file size 5.41 GB, and SHA-256 `ba1020872ddb6e90726c755f9f08f236f4c73e506013c91d8a0075fc0c0361e6`. The append-only family vocabulary has 41,056 rows and SHA-256 `00d304df9bd72acb2c2c1ca5b84ed0dc9d4adb0cbc9f098f32a9e62109d24005`. Unknown size accounts for 71.929% of models and `Other` for 13.561%. Feature construction and row-order checks are in [`scale1m/embed_lake_rf.py`](../../scale1m/embed_lake_rf.py) and [`F4_runs/FEATS_REPORT.json`](F4_runs/FEATS_REPORT.json).

### 3.5 Dataset-node features

The dataset descriptor concatenates a cleaned dataset--task name, card task categories, up to ten non-colon tags, and at most 400 description characters. If no trustworthy card match exists, only the cleaned name and task are used. With a name-hash table generated using seed 43, the frozen dataset vector is

$$
x_d^{\mathrm{frozen}}=
[e_{\mathrm{name}}^{64}(d)\;\Vert\;
 e_{\mathrm{card}}^{384}(d)\;\Vert\;
 e_{\mathrm{stats}}^{10}(d)]
\in\mathbb{R}^{458}.
$$

Let $n_d$ be the number of retained observations at node $d$, let $\mu_d,\sigma_d,a_d^{\min},a_d^{\max}$ summarize its oriented edge values, and let $r_d$ be the number of dataset nodes sharing its root. The implemented statistics are

$$
e_{\mathrm{stats}}(d)=
[\log(1+n_d),\log(1+n_d),\mu_d,\sigma_d,
a_d^{\min},a_d^{\max},\log(1+r_d),
\mathbb{1}_{\mathrm{eligible}}(d),0,0].
$$

The first two coordinates are identical in the frozen implementation because duplicate model--node pairs have already been removed. The last two positions reserve disabled content views. During training, dataset task type, class-count bucket, and arity IDs additionally select learnable tables of dimensions 16, 8, and 4. In this full-lake artifact each of these tables has only one row, so they preserve the encoder schema but do not distinguish nodes; the task string still affects node identity and the text descriptor. The feature builder is `build_xd` in [`scale1m/build_graph_rf.py`](../../scale1m/build_graph_rf.py).

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

Lineage resolution succeeds for 96.083% of declared base-model relations. Before training, dataset similarity is deterministically reduced to the top 10 neighbors per source and its retained edge attributes are set to one, yielding 187,290 topology-only similarity edges. This training-time transformation is `apply_similar_to_mode(..., mode="topk_unweighted", k=10)` in [`stage2TrainGraphSAGE/graph_surgery.py`](../../stage2TrainGraphSAGE/graph_surgery.py).

The graph is stored as memory-mapped `.npy` feature arrays, NPZ node/edge structures, Parquet row maps, and hashed JSON metadata. This avoids materializing the 5.41 GB model matrix during loading. The sharded format is implemented in [`scale1m/graph_store.py`](../../scale1m/graph_store.py); construction counts and gates are in [`F5_runs/GRAPH_REPORT.json`](F5_runs/GRAPH_REPORT.json). The frozen graph digest used by all final training runs is `0e80b8393846dcc4b9e354218fd5139a906d569125d2e8545e17ccd01612b76c`.

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
W_{\mathrm{self},r}h_v^{(0)}+
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

Every relation has its own self and neighbor projections. Because the frozen configuration sets `weighted_relations=[]`, continuous `edge_attr` values are **not** used inside message aggregation. Performance values still supervise the ranking and positive-set objectives, and card cosine still determines which dataset-neighbor topology is retained, but the message operator itself is an unweighted per-relation mean. This exact operator is `WeightedSAGEConv` plus `EdgeAwareHetero` in [`stage2TrainGraphSAGE/edge_aware.py`](../../stage2TrainGraphSAGE/edge_aware.py).

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

The averaging is first within a dataset and then across datasets, so dense query nodes do not dominate merely by producing more pairs. At most 256 pairs are retained per dataset; when a node has more, half are the most inverted current pairs and the other half are sampled from the remainder. The objective operates directly on the raw unit-vector dot product used by retrieval. Its implementation is `raw_dot_ranknet_loss` in [`stage2TrainGraphSAGE/losses.py`](../../stage2TrainGraphSAGE/losses.py).

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

The matrix is stored as CSR/CSC membership lists rather than a dense $N\times Q$ block. In a sampled graph batch, positive pairs are materialized sparsely and deduplicated. For each anchor, 256 batch models are sampled as negatives. If a negative shares the anchor's lineage component but is not a positive, its denominator weight is two; otherwise the weight is one. With $\phi(i,j)=z_i^\top z_j/0.2$, the implemented sampled objective is

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

The three archived final configurations are [`seed 0`](X4_runs/X4GD_full_s0_e25/metadata/resolved_config.json), [`seed 1`](X4_runs/X4GD_full_s1_e25/metadata/resolved_config.json), and [`seed 2`](X4_runs/X4GD_full_s2_e25/metadata/resolved_config.json). Each records the exact command line, runtime versions, graph digest, configuration, split seed, and initialization seed. All three contain 25 epoch records, finite losses, descending total loss, and a passing mechanism gate.

### 4.8 Held-out inference and evaluation

Evaluation embeddings are generated from $G_{\mathrm{test}}$, which contains training and validation message edges but excludes all test `trained_on` edges and their reverse copies. The exporter separately writes full-message embeddings for serving and held-out-message embeddings for evaluation:

```text
z_m.npy,      z_d.npy       # serving graph
z_m_eval.npy, z_d_eval.npy  # leakage-controlled evaluation graph
```

Only `z_m_eval` and `z_d_eval` are used for the reported result. Model embeddings are exported in `mappedID` order, and checkpoint loading is strict. Chunked inference reproduces whole-graph inference within the archived numerical tolerance. This path is `stage_embed` in [`scale1m/export_rf.py`](../../scale1m/export_rf.py).

For an eligible held-out query $d$, let $C_d$ be its held-out observed model set and let

$$
g_d=\operatorname*{arg\,max}_{m\in C_d}y_{dm}.
$$

The exact full-lake rank uses all $N$ model embeddings and is tie-safe:

$$
\operatorname{rank}_d(m)
=1+\sum_{j=1}^{N}\mathbb{1}[s(d,m_j)>s(d,m)].
$$

Therefore,

$$
\operatorname{gold@10}
=\frac{1}{|\mathcal Q_{\mathrm{test}}|}
\sum_{d\in\mathcal Q_{\mathrm{test}}}
\mathbb{1}[\operatorname{rank}_d(g_d)\le 10].
$$

The eligible query policy requires: (i) a known metric direction or curated oriented source, (ii) a non-reinforcement-learning task, (iii) a non-placeholder dataset name, and (iv) at least three observed candidates. A query must also have nonconstant held-out values for a rank to be evaluated. Because roots rather than individual nodes are split, the number of evaluable queries differs across seeds.

The metric harness also reports `gold@1`; `top3@10`, which accepts any of the three best observed models; `gold-gap@10`, which accepts any observed model within 0.01 of the best value; and a root-macro `gold@10`. Definitions and streaming exact-rank computation are in [`scale/global_metrics.py`](../../scale/global_metrics.py); the final eligibility filter is in [`scale1m/eval_rf.py`](../../scale1m/eval_rf.py).

## 5. Online retrieval: HNSW top-1,000 and task-evidence reranking

### 5.1 Index geometry

Because every embedding is unit normalized,

$$
\operatorname*{arg\,max}_{m}\;z_d^\top z_m
=\operatorname*{arg\,min}_{m}\;(1-z_d^\top z_m)
=\operatorname*{arg\,min}_{m}\;\frac{1}{2}\|z_d-z_m\|_2^2.
$$

Thus inner-product HNSW searches exactly the geometry optimized by training. The full-lake builder creates `hnswlib.Index(space="ip", dim=128)`, inserts row $i$ with label `mappedID=i`, and persists the binary index. Its fixed construction parameters are

$$
M_{\mathrm{HNSW}}=32,\qquad ef_{\mathrm{construction}}=200.
$$

The implementation is `build_hnsw` in [`scale/export_ours.py`](../../scale/export_ours.py), called by `stage_index` in [`scale1m/export_rf.py`](../../scale1m/export_rf.py).

### 5.2 Label-free recall calibration

For the final depth $K=1000$, index fidelity is measured against the exact dense top-1,000 IDs:

$$
\operatorname{Recall@1000}(ef)=
\frac{1}{|\mathcal Q_H|}
\sum_{d\in\mathcal Q_H}
\frac{|\operatorname{ANN}_{1000}(d;ef)
\cap\operatorname{Exact}_{1000}(d)|}{1000}.
$$

For each split, `ef_search` is selected from 1,000, 1,500, 2,000, 3,000, and 5,000 as the first value reaching `recall@1000 >= 0.99`. Selection uses only dense-neighbor ID agreement, never test gold labels. The chosen values and measured recalls are:

| Split seed | `ef_search` | `recall@1000` |
|---:|---:|---:|
| 0 | 1,000 | 0.9923 |
| 1 | 1,500 | 0.9941 |
| 2 | 1,500 | 0.9932 |

All latency measurements begin from a precomputed query embedding and use one HNSW query thread after warm-up. Query encoding, index loading, and index construction are excluded.[`Y2_REPORT.json`](Y2_runs/Y2_REPORT.json)

### 5.3 Split-safe reranking and artifact contract

For a materialized dataset--task node, the measured online path is:

```text
dense_pool = hnsw.knn_query(z_q, k=1000)
prior = task_prior[query_task, dense_pool.model_ids]
score = (dense_pool.cosine + 1) / 2 + prior
top10 = stable_topk(score, fixed_label_free_tie_break)
```

The split-specific sidecars contain only train+validation edges: 198,216 / 196,912 / 196,124 visible edges for seeds 0 / 1 / 2. The evaluator asserts exact equality of the X4 and sidecar model and dataset row mappings, verifies the root mapping, and asserts that no visible prior edge shares a root with any scored test query. The query itself and every same-root sibling are therefore absent from the prior. All mapping and leakage gates pass in the archived report.[`eval_y2.py`](../../scale1m/eval_y2.py) [`Y2_REPORT.json`](Y2_runs/Y2_REPORT.json)

An HNSW label is meaningful only under the model row map used to construct the index. The three measured evaluation indexes bind the 3,016,439-row X4-GD held-out embeddings and live at `data1m/exports_x4/X4GD_full_s{0,1,2}_e25/hnsw_y2_eval.bin`. A truly new dataset that is absent from the frozen node table still requires a validated query-only inductive encoding path; the present measurements concern materialized dataset--task nodes with held-out performance edges.

### 5.4 Final two-stage result and cost mechanism

All rows below use the same eligible queries, candidate universe, representation, and frozen reranker. They compare the actual bounded-candidate path with two exact reference paths; they are not intermediate system stages.

| Retrieval path | Work required per query | seed 0 | seed 1 | seed 2 | Mean `gold@10` |
|---|---|---:|---:|---:|---:|
| Dense + task prior, exact full lake | Compute a dense score, read the task prior, and fuse the two scores for all 3,016,439 models | 0.3408 | 0.3651 | 0.2589 | 0.3216 |
| Dense exact top-1,000 + task prior | Compute a dense score for all 3,016,439 models, select 1,000, then read and fuse 1,000 priors | 0.3286 | 0.3388 | 0.2427 | 0.3034 |
| **Dense HNSW top-1,000 + task prior** | **Use the ANN index to find 1,000 candidates, then read and fuse only 1,000 priors** | **0.3279** | **0.3388** | **0.2427** | **0.3031** |

The exact top-1,000 reference is costly because it must still compare the query with every model embedding before truncation; it therefore preserves a linear full-lake scan and loses HNSW's sublinear search path. Exact full-pool fusion performs that same all-model dense scoring and additionally reads, combines, and ranks task evidence for every model rather than for 1,000 candidates. Its second-stage candidate count is therefore about 3,016 times larger than that of the actual system, and both exact paths grow directly with lake size. By contrast, the actual path confines task-prior lookup and fusion to the bounded HNSW pool.

The unrounded final values are 0.3279132791, 0.3387829246, and 0.2427184466; their arithmetic mean is **0.3031382168**. The final path beats the 3M-scale BM25 baseline on every seed and by 2.98 times on the three-seed mean. Exact top-1,000 reranking retains 94.32% of the full-pool fused result on average, and HNSW retains 99.93% of exact top-1,000 reranking, for 94.25% overall two-stage retention relative to full-pool fusion. Dense top-1,000 contains the held-out gold for 51.88% of queries on average, and 89.91% of the exact full-pool fused top-ten members already occur in that pool. The remaining quality loss is therefore dominated by pool truncation rather than ANN approximation.[`Y2_REPORT.json`](Y2_runs/Y2_REPORT.json)

| Split seed | Queries | `gold@1` | `gold@10` | `top3@10` | `gold-gap@10` | root-macro `gold@10` | Median reranked gold position when retrieved |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,476 | 0.1599 | 0.3279 | 0.3984 | 0.3814 | 0.2515 | 6 |
| 1 | 1,101 | 0.1117 | 0.3388 | 0.4269 | 0.3860 | 0.2591 | 7 |
| 2 | 1,545 | 0.1107 | 0.2427 | 0.2958 | 0.2777 | 0.1903 | 5 |
| Three-seed mean | — | 0.1274 | **0.3031** | 0.3737 | 0.3484 | 0.2336 | — |

The latest Y4 timing repeat on Windows 11, an Intel Family 6 Model 183 CPU with 24 logical processors, single-query/single-thread HNSW, and warm-up gives the following retrieval-stage latency:

| Split seed | HNSW + rerank p50 | HNSW + rerank p95 |
|---:|---:|---:|
| 0 | 0.536 ms | 0.905 ms |
| 1 | 0.847 ms | 1.435 ms |
| 2 | 0.699 ms | 1.329 ms |
| Mean | **0.694 ms** | **1.223 ms** |

Mean HNSW and reranking p50 components are 0.566 and 0.123 ms. This path starts from a precomputed $z_q$ and excludes query encoding, index loading, and index construction. Each index occupies about 2.214 GiB; the three total 6.643 GiB and were built in 184.6 / 191.3 / 181.6 seconds.[`Y2_REPORT.json`](Y2_runs/Y2_REPORT.json) [`Y4_REPORT.json`](Y4_runs/Y4_REPORT.json)

The absolute final `gold@10` remains 0.3031, so full-lake model retrieval is far from complete historical-gold recovery. It is a reproducible operating point, not evidence that unobserved recommendations achieve 30.31% downstream accuracy.

## 6. Implementation entry points and archived-result verification

The following commands expose the concrete implementation order and keep the X4-GD embeddings and split-specific task sidecars on the same row mappings used by the archived evaluation. Angle-bracket paths are deployment-specific; the exact as-run training command for each seed is stored in that run's `metadata/resolved_config.json`.

```powershell
# 1. Collect immutable model and dataset metadata.
python -m scale1m.hf_crawl `
  --sort createdAt --direction -1 --limit 100000000 --v2-fields `
  --out <DATA>/data1m/candidates_full
python -m scale1m.hf_crawl_datasets `
  --out <DATA>/data1m/datasets_full

# 2. Canonicalize evidence and construct the model--dataset evidence graph.
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

# 3. Train and export X4-GD; repeat with seed 0, 1, and 2.
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

# 4. Build an equivalent split-safe task sidecar directly beside each X4-GD export.
python -m stage3HNSW.build_prior_sidecar `
  --graph-store <DATA>/data1m/graphs/hgraph_rf `
  --export <DATA>/data1m/exports_x4/X4GD_full_s<SEED>_e25 `
  --split-seed <SEED> `
  --task-nodes <DATA>/data1m/rf/canon/dataset_nodes_merged.parquet

# 5. Reproduce the exact pool, HNSW indexes, and final K=1000 report.
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

# Optional: reproduce the frozen-retriever K sensitivity curve through Y4.
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

The acquisition and graph-construction commands are the current executable CLI contracts. The archived final training commands are stronger evidence for as-run configuration because they record absolute cluster paths, job metadata, and all resolved defaults.

## 7. Reproducibility map

| Method element | Code path | Primary evidence |
|---|---|---|
| Cursor-safe model snapshot | [`scale1m/hf_crawl.py`](../../scale1m/hf_crawl.py) | [`model provenance`](F1_runs/PROVENANCE.json) |
| Cursor-safe dataset snapshot | [`scale1m/hf_crawl_datasets.py`](../../scale1m/hf_crawl_datasets.py) | [`dataset provenance`](F15_runs/PROVENANCE.json) |
| Dataset descriptors and card matching | [`scale1m/dataset_descriptor.py`](../../scale1m/dataset_descriptor.py), [`scale1m/match_dataset_cards.py`](../../scale1m/match_dataset_cards.py) | [`coverage report`](F15_runs/f15b_coverage.json) |
| Metric direction and orientation | [`scale1m/metric_semantics.py`](../../scale1m/metric_semantics.py), [`scale1m/canonicalize_rf.py`](../../scale1m/canonicalize_rf.py) | [`native canonicalization`](F2_runs/F2_REPORT.json) |
| Six-source merge and query rules | [`scale1m/merge_supervision.py`](../../scale1m/merge_supervision.py) | [`merge report`](F2_runs/F2_MERGE_REPORT.json), [`rule file`](F2_runs/rf_gold_rules.json) |
| Candidate and dataset row maps | [`scale1m/build_ladder_rf.py`](../../scale1m/build_ladder_rf.py) | [`ladder report`](F3_runs/LADDER_REPORT.json) |
| Model metadata features | [`scale1m/embed_lake_rf.py`](../../scale1m/embed_lake_rf.py) | [`feature report`](F4_runs/FEATS_REPORT.json) |
| Evidence-graph construction | [`scale1m/build_graph_rf.py`](../../scale1m/build_graph_rf.py) | [`graph report`](F5_runs/GRAPH_REPORT.json), [`lineage report`](F5_runs/lineage_stats.json) |
| Leakage-controlled split | [`stage2TrainGraphSAGE/d0_splits.py`](../../stage2TrainGraphSAGE/d0_splits.py) | [`split audit`](F5_runs/f5_splits.json) |
| Node encoders and graph network | [`model_node_encoder.py`](../../stage1BuildTransferGraph/dataset_embed/model_node_encoder.py), [`model.py`](../../stage2TrainGraphSAGE/model.py), [`edge_aware.py`](../../stage2TrainGraphSAGE/edge_aware.py) | final resolved configurations linked above |
| Ranking, contrastive, and global objectives | [`stage2TrainGraphSAGE/losses.py`](../../stage2TrainGraphSAGE/losses.py), [`train.py`](../../stage2TrainGraphSAGE/train.py) | final run manifests and histories |
| Held-out export and exact dense ranking | [`scale1m/export_rf.py`](../../scale1m/export_rf.py), [`scale/global_metrics.py`](../../scale/global_metrics.py) | [`dense eligible-query metrics`](X5_runs/X5_GD_ELIGIBILITY.json) |
| Three-million-candidate baselines | [`scale1m/baselines.py`](../../scale1m/baselines.py), [`scale1m/eval_x6.py`](../../scale1m/eval_x6.py) | [`training-free baseline report`](X6_runs/X6_BASELINES.training_free.json) |
| Split-safe task sidecar | [`stage3HNSW/build_prior_sidecar.py`](../../stage3HNSW/build_prior_sidecar.py), [`scale1m/eval_rf.py`](../../scale1m/eval_rf.py) | [`Y2 report`](Y2_runs/Y2_REPORT.json) |
| HNSW top-1,000 and deterministic reranking | [`scale1m/eval_y2.py`](../../scale1m/eval_y2.py), [`scale/export_ours.py`](../../scale/export_ours.py) | [`Y2 report`](Y2_runs/Y2_REPORT.json) |
| Final K=1,000 retrieval and timing | [`scale1m/eval_y4.py`](../../scale1m/eval_y4.py) | [`Y4 report`](Y4_runs/Y4_REPORT.json) |

## 8. Evidence-qualified limitations

The following boundaries are properties of the present final artifact and should remain explicit in a paper derived from this report.

1. **The result is metadata-and-history based.** `gold@10` tests the rank of an observed held-out best model; it does not execute the retrieved models on a new dataset and cannot assign correctness to unobserved query--model pairs.
2. **Dataset-card coverage is sparse.** Only 3,928 of 18,729 nodes have a matched Hugging Face card; the remainder use name/task text. This affects both dataset features and `similar_to` topology.
3. **Model size is sparse.** 71.929% of candidates use the unknown size bucket. Family coverage is broader, but 13.561% map to `Other`.
4. **The dataset categorical tables are degenerate in this graph.** Task-type, class-count, and arity IDs each contain one vocabulary row. The dataset--task string still appears in node identity and text, but the three lookup tables add no between-node information.
5. **The final graph encoder is topology-aware, not continuous-edge-weight-aware.** The code supports weighted relations, but the frozen final configuration sets the weighted relation list to empty.
6. **Only three split seeds are measured.** Initialization is held fixed at seed 0, so the interval describes root-split variation rather than independent initialization uncertainty.
7. **The archived HNSW measurements use held-out evaluation embeddings.** Three final-representation evaluation indexes are measured, but a separately archived full-message production index is not part of this evidence freeze. The algorithmic serving path is the same; the artifact distinction must remain explicit.
8. **A never-before-seen dataset path is not validated for the final graph schema.** The final result is for materialized dataset--task nodes under root-aware held-out edges; it does not establish an inductive encoding path for a brand-new node.
9. **Reproducibility depends on artifact hashes as well as Git.** The final runs record a repository head plus local changes. The as-run hashes for the principal training files match the current inspected versions, while graph, vocabulary, row maps, checkpoints, and exports are separately bound through manifests.

## 9. Paper-ready method summary

ModelLakeFishing first snapshots model and dataset metadata from a public model hub using cursor-resumable, hash-bound collectors. It canonicalizes repository identifiers, model family, parameter size, lineage, dataset--task identity, and heterogeneous evaluation records. Metric values are deduplicated, normalized within comparable groups, and oriented through an explicit higher/lower-is-better dictionary; observations whose direction is unknown remain graph evidence but are excluded from gold evaluation. Six evaluation sources are merged by deterministic source priority and capped by weight-stratified sampling, yielding 247,803 model--dataset supervision edges over 3,016,439 model nodes and 18,729 dataset--task nodes.

The resulting model--dataset evidence graph contains bidirectional performance edges, dataset-similarity edges, and bidirectional model-lineage edges. Model nodes combine a 64-dimensional hashed name vector, a 384-dimensional MiniLM descriptor, and learned size and family embeddings. Dataset nodes combine a 64-dimensional hashed name vector, a 384-dimensional card descriptor, ten observation statistics, and schema-level categorical embeddings. A one-layer relation-specific GraphSAGE encoder aggregates each relation with independent parameters and a learned relation gate, then projects model and dataset nodes through a shared head into a unit-normalized 128-dimensional retrieval space.

Training minimizes the sum of three score-aligned objectives: a within-dataset RankNet loss on oriented performance differences, a model--model contrastive loss whose positives are co-selected top performers, and a whole-lake logQ-corrected sampled-softmax loss. The full-lake proposal mixes a smoothed degree distribution and a uniform distribution over labeled models with equal weight; each global step samples 128 query datasets and 256 model negatives per query. Root-aware train/validation/test splits keep every dataset family on one side and remove held-out edges in both directions from message passing.

At retrieval time, inner-product HNSW retrieves 1,000 dense candidates and a deterministic, split-safe task prior reranks only that pool using $r=(\cos+1)/2+p_t(m)$. Across the three root split seeds, the final `gold@10` values are 0.3279, 0.3388, and 0.2427, with a mean of 0.3031; exact full-pool fusion reaches 0.3216 but requires all-model dense scoring and all-model prior fusion. The measured HNSW recalls at 1,000 are all above 0.99, and the latest repeated HNSW-plus-reranking p50 / p95 are 0.694 / 1.223 ms from precomputed query embeddings.
