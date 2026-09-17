"""Construct a reviewable A0-only English library; never writes active libraries."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
DOCS = ROOT / "docs/1M"
OUT = Path(__file__).parent / "EVIDENCE_SOURCE_LIBRARY_en.draft.md"
old = (DOCS / "A0_runs/frozen/EVIDENCE_SOURCE_LIBRARY_en.md").read_text(encoding="utf-8")
rep = json.loads((DOCS / "A0_runs/A0_7/results/A0_REPORT.json").read_text(encoding="utf-8"))
vals = rep["new_measurements"]


def v(key):
    item = vals[key]
    assert item["status"] in ("recomputed", "verified"), (key, item)
    return item["value"]


def f(key, n=4):
    return f"{v(key):.{n}f}"


def section(start, end):
    return old[old.index(start):old.index(end)]


def change(text, before, after):
    assert text.count(before) == 1, before[:150]
    return text.replace(before, after)


REPORT = "[A0.7 recomputation report](A0_runs/A0_7/results/A0_REPORT.json)"
COUNTS = "[A0.7 source recount](A0_runs/A0_7/results/A0_SOURCE_COUNTS.json)"
AUDIT = "[A0.1 input audit](A0_runs/audit/A0_INPUT_AUDIT.json)"

intro = r"""# ModelLakeFishing: A0 Evidence Library for the Final 3M-Scale Retrieval System

**Evidence edition:** A0.1–A0.7, updated 2026-09-14 (America/Toronto).

**Repository:** `D:\research\model_lake\codes\ModelLakeFishing`  
**Final system:** **X4G+D → HNSW top1000 → task prior → top10**.  
**Evidence chain:** [A0 source manifest](A0_runs/A0_SOURCE_MANIFEST.json), [A0.7 manifest](A0_runs/A0_7/MANIFEST.json), [A0.7 report](A0_runs/A0_7/results/A0_REPORT.json), [metric inventory](A0_runs/A0_7/results/A0_METRIC_INVENTORY.json).

This edition describes the final method and the results produced by A0.1–A0.7. A0 retains the frozen candidate universe, node identities, supervision, root splits, three-seed training recipe, and fixed retrieval configuration. It zeros seven performance-derived dataset input columns, trains three new 25-epoch checkpoints, exports new representations, builds new HNSW indexes, and independently recomputes the resulting measurements. The two exact retrieval paths are offline diagnostic references for this same final system.

Retrieval and cost recomputation passed, including 470 additional comparisons with the evaluator. The complete inventory contains 982 entries: 838 recomputed, 86 verified, 3 disabled, 11 not applicable, 42 undefined, and 2 missing. The missing entries are the original model-crawl API-page and discarded-duplicate event counts. These historical events have no complete raw log, so the overall completeness flag remains `false`; the retrieval result is available and validated. [A0.7 validation](A0_runs/A0_7/results/A07_VALIDATION.json), [A0.7 report](A0_runs/A0_7/results/A0_REPORT.json).

Across 3,016,439 candidates and 1,476 / 1,101 / 1,545 eligible held-out query observations, final `gold@10` is **@@GOLD@@** (equal-weight three-seed mean **@@MEAN@@**). Exact full-lake fusion reaches @@FULL@@. The final system retains @@RET@@% of the full-lake fused `gold@10`, averaged over the three per-seed ratios. Mean per-seed retrieval p50 / p95 is **@@P50@@ / @@P95@@ ms** in the formal Linux evaluation; timing starts from a precomputed query embedding. All these figures come from the [A0.7 recomputation report](A0_runs/A0_7/results/A0_REPORT.json).

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

"""
intro = (intro.replace("@@GOLD@@", " / ".join(f(f"hnsw1000_task_prior.gold_at_10.{s}") for s in range(3)))
    .replace("@@MEAN@@", f("hnsw1000_task_prior.gold_at_10.mean"))
    .replace("@@FULL@@", f("exact_full_lake_task_prior.gold_at_10.mean"))
    .replace("@@RET@@", f"{100*v('retrieval_diagnostics.overall_retention.mean'):.2f}")
    .replace("@@P50@@", f("retrieval_cost.total_latency_p50_ms.mean", 3))
    .replace("@@P95@@", f("retrieval_cost.total_latency_p95_ms.mean", 3)))

# Reuse the mathematical definitions that were frozen into the A0 protocol;
# replace each superseded feature, result, or external-series evidence reference.
method = section("### 2.2 Canonical model identity", "### 4.8 Held-out inference and evaluation")
method = change(method,
    "Coverage measurements are archived in [`F15_runs/f15b_coverage.json`](F15_runs/f15b_coverage.json).",
    "The count is newly verified in " + AUDIT + ".")
method = change(method,
    "with counts in [`F2_runs/F2_REPORT.json`](F2_runs/F2_REPORT.json).",
    "with counts independently regenerated in " + COUNTS + ".")
method = change(method,
    "The frozen rule file is `rf-gold-2.0`, SHA-256 `be3fb05ecf92ea0112ba4de10b8191d71dfe096b8667a86911c2ebaeadb9cdb1`; the merge counts and source table are in [`F2_runs/F2_MERGE_REPORT.json`](F2_runs/F2_MERGE_REPORT.json), and the rule artifact is [`F2_runs/rf_gold_rules.json`](F2_runs/rf_gold_rules.json).",
    "The rule file `rf-gold-2.0` has SHA-256 `be3fb05ecf92ea0112ba4de10b8191d71dfe096b8667a86911c2ebaeadb9cdb1`, bound by " + AUDIT + ". A0.7 reconstructed the merged supervision, node, and conflict tables and matched every row and column after stable semantic sorting, with no numeric tolerance. The merge recount and exact output-identity check are in " + COUNTS + ". These are source-data names, not alternative evaluated retrieval systems.")
method = change(method,
    "Historical-only rows have unknown size, take family identity from the first available historical source, and receive newly computed features under the same encoder as every other model. No old feature vector is copied.",
    "In the frozen construction, historical-only rows have unknown size, take family identity from the first available historical source, and use the same descriptor and feature encoder as the other rows. A0 reuses that verified model-feature matrix.")
method = change(method,
    "All eight identity and endpoint checks pass in [`F3_runs/LADDER_REPORT.json`](F3_runs/LADDER_REPORT.json).",
    "The row maps, snapshot-prefix closure, and edge endpoints are verified by " + AUDIT + ".")
method = change(method,
    "Parameter size and family are not embedded inside these 448 dimensions. They are stored as integer columns and select learnable 16-dimensional tables during graph training.",
    "Family and known parameter size already appear in the descriptor text. Their discrete family and size-bucket IDs additionally select learned 16-dimensional tables during graph training.")
method = change(method,
    "The append-only family vocabulary has 41,056 rows",
    "The append-only `feats_rf/family_vocab.csv` vocabulary has 41,056 rows")
method = change(method,
    "Feature construction and row-order checks are in [`scale1m/embed_lake_rf.py`](../../scale1m/embed_lake_rf.py) and [`F4_runs/FEATS_REPORT.json`](F4_runs/FEATS_REPORT.json).",
    "The 41,056 rows are the embedding index cardinality; 40,927 distinct family IDs actually occur in the graph. Feature construction is in [`scale1m/embed_lake_rf.py`](../../scale1m/embed_lake_rf.py); current shape, cardinality, and coverage measurements are in " + COUNTS + ", with frozen-file hashes in " + AUDIT + ".")
begin = method.index("Let $n_d$ be the number of retained observations")
end = method.index("### 3.6 Typed graph relations")
method = method[:begin] + r"""In A0, this ten-slot block contains only a node-table structural count. If $r_d$ is the number of frozen dataset–task nodes with the same root,

$$
e_{\mathrm{stats}}^{A0}(d)=
[0,0,0,0,0,0,\log(1+r_d),0,0,0].
$$

All 18,729 nodes have columns 448–453 and 455 set exactly to zero. These seven positions formerly carried two observation counts, the performance mean/standard deviation/minimum/maximum, and `gold_eligible`. A0 keeps the 458-dimensional schema: column 454 retains the root-node count, columns 456–457 remain reserved zeros, and the other 451 columns retain their frozen values. Thus the feature repair removes those performance-derived inputs while preserving the text blocks, row maps, graph topology, split identities, and encoder dimensions. The eligibility flag remains an evaluation-cohort rule outside the learned input. The repaired graph digest is

```text
acddeb93d926efcb636c83d452fe360736a0e6e4ac68cab3f5fef210e7cef5db
```

Dataset task-type, class-count-bucket, and arity IDs additionally select learned tables of dimensions 16, 8, and 4. Each table has one row in this artifact, so the discrete IDs are constant across nodes. Task strings still affect node identity and the MiniLM descriptor. The task prior separately derives its task groups from the canonical task strings. [A0.2 repair and verification](A0.2.md), [feature preparation](../../scale1m/prepare_a0_graph.py), [feature builder](../../scale1m/build_graph_rf.py), [A0.7 source recount](A0_runs/A0_7/results/A0_SOURCE_COUNTS.json).

""" + method[end:]
method = change(method,
    "Lineage resolution succeeds for 96.083% of declared base-model relations. Before training,",
    "The A0 input audit verifies the stored lineage endpoints and reverse relation. Before training,")
method = change(method,
    "The graph is stored as memory-mapped `.npy` feature arrays, NPZ node/edge structures, Parquet row maps, and hashed JSON metadata. This avoids materializing the 5.41 GB model matrix during loading. The sharded format is implemented in [`scale1m/graph_store.py`](../../scale1m/graph_store.py); construction counts and gates are in [`F5_runs/GRAPH_REPORT.json`](F5_runs/GRAPH_REPORT.json). The frozen graph digest used by all final training runs is `0e80b8393846dcc4b9e354218fd5139a906d569125d2e8545e17ccd01612b76c`.",
    "The graph is stored as memory-mapped `.npy` feature arrays, NPZ node/edge structures, Parquet row maps, and hashed JSON metadata. The storage loader can map the 5.41 GB model matrix; training subsequently clones graph data and transfers it to the GPU. The sharded format is implemented in [`scale1m/graph_store.py`](../../scale1m/graph_store.py). The counts above are verified in " + AUDIT + "; A0.2 preserves their bytes while replacing the seven input columns and binding the repaired graph digest stated in §3.5. The full-scale resource requirement is measured in [A0.3](A0.3.md) and [A0.4](A0.4.md).")
method = change(method,
    r"W_{\mathrm{self},r}h_v^{(0)}+",
    r"W_{\mathrm{self},r}h_v^{(0)}+b_{\mathrm{self},r}+")
method = change(method,
    "Every relation has its own self and neighbor projections.",
    "Every relation has its own self and neighbor projections; the self projection includes a bias. An empty per-node neighborhood contributes a zero neighbor mean. Globally empty relation tensors are skipped.")
method = change(method,
    "At most 256 pairs are retained per dataset;",
    "In the formula, $\\mathcal D_B$ includes datasets with at least one retained preference pair, and $\\mathcal P_d$ denotes the retained subset when sampling is required. At most 256 pairs are retained per dataset;")
method = change(method,
    "For each anchor, 256 batch models are sampled as negatives.",
    "For each anchor, 256 batch model IDs are drawn uniformly with replacement; self IDs and sampled positives are discarded from the negative term.")
method = change(method,
    "MSE, embedding-uniformity regularization, dataset-to-model contrastive loss, hard-negative mining, positive inverse-propensity weighting, separate projection heads, and early stopping are disabled.",
    "MSE, embedding-uniformity regularization, dataset-to-model contrastive loss, optional mined-negative branches of the global objective, positive inverse-propensity weighting, separate projection heads, and early stopping are disabled. The local RankNet hard-pair selection and lineage-based contrastive negative weighting described above remain active.")
method = change(method,
    "The three archived final configurations are [`seed 0`](X4_runs/X4GD_full_s0_e25/metadata/resolved_config.json), [`seed 1`](X4_runs/X4GD_full_s1_e25/metadata/resolved_config.json), and [`seed 2`](X4_runs/X4GD_full_s2_e25/metadata/resolved_config.json). Each records the exact command line, runtime versions, graph digest, configuration, split seed, and initialization seed. All three contain 25 epoch records, finite losses, descending total loss, and a passing mechanism gate.",
    "The actual A0 configurations are [`seed 0`](A0_runs/A0_4/delivery_s0/metadata/resolved_config.json), [`seed 1`](A0_runs/A0_4/delivery_s1/metadata/resolved_config.json), and [`seed 2`](A0_runs/A0_4/delivery_s2/metadata/resolved_config.json). Each records the command, runtime, repaired graph digest, 40 effective configuration keys, split seed, and initialization seed. Each completed 25 epochs and passed the mechanism gate. A0 uses `ckpt/last.pt` at zero-based epoch 24 for all three exports. These are newly trained runs, each with initialization seed 0. [A0.4 completion and hashes](A0.4.md).")
# Current root split source is retained; performance-derived input exposure is
# separately tested by the actual A0 information-boundary fixture.
method += r"""### 4.8 Held-out exports and the tested information boundary

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

@@CALIBRATION@@

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

@@QUALITY_REFERENCES@@

@@PRIMARY_QUALITY@@

The final unrounded `gold@10` values are @@UNROUNDED@@. The metrics are historical observed-ranking recovery; each candidate model is not executed during this retrieval evaluation.

@@DIAGNOSTICS@@

Exact-pool retention is `exact1000 / exact_full_lake`, ANN retention is `hnsw1000 / exact1000`, and overall retention is `hnsw1000 / exact_full_lake`, using `gold@10` within each seed before averaging ratios. Exact-pool gold coverage is the fraction of queries whose gold ID enters the dense pool. Full-fused-top10 coverage averages the fraction of exact full-fusion top10 IDs already in that pool. These diagnostic quantities help separate pool truncation from ANN error. Source for all tables in §5: [A0.7 recomputation report](A0_runs/A0_7/results/A0_REPORT.json), [complete inventory](A0_runs/A0_7/results/A0_METRIC_INVENTORY.json), [independent validation](A0_runs/A0_7/results/A07_VALIDATION.json).

### 5.5 Retrieval latency, index storage, and measured offline costs

The formal A0.6 evaluation ran on `watgpu308` under Linux, Python 3.11.4, NumPy 2.4.6, PyTorch 2.12.0+cu130, and hnswlib 0.8.0. Exact vector computation used its NVIDIA L40S; HNSW construction/search and prior fusion ran on the host CPU. Each seed has one timed query pass, using one HNSW query thread after up to 50 warm-up queries. Timings start from a precomputed query embedding and cover HNSW plus candidate-prior lookup, fusion, and ordering. Query encoding, loading the index/prior tables, constructing the index, and client/network overhead lie outside that measured interval. Local RTX 4060/i7 verification and the later local independent recomputation have separate execution records. [A0.6 report and runtime binding](A0_runs/A0_6/delivery/metrics/A0_EVALUATION_MANIFEST.json), [A0.6 execution](A0.6.md).

@@TIMING@@

Each total sample is the paired sum of its HNSW and reranking samples. Percentiles are recomputed from the original per-query nanosecond arrays; the mean row averages the three seed percentiles. It is neither a percentile over pooled queries nor a sum of component percentiles.

@@INDEX@@

Index build time includes initialization, insertion, serialization, and final file replacement. Total index storage is @@INDEX_BYTES@@ bytes, or @@INDEX_GIB@@ GiB. Here GB denotes $10^9$ bytes and GiB denotes $2^{30}$ bytes.

@@COSTS@@

Training time uses the recorded `train_eval_one` segments, including setup, training/checkpoint work, and the existing post-training evaluations. Export time uses the recorded export segments, including checks; it differs from the inner embedding-only timer. Prior time includes its input verification. The two exact references share one scan and therefore the same time interval. Per-seed evaluation time sums the exact and HNSW stage segments once, including index construction; global binding checks and finalization are outside those segments. Consequently, neither the second exact label nor index-build time should be added again. These measured intervals also exclude queueing and failed environment-setup attempts. A0.4 training used `watgpu308`/L40S, A0.5 export used `watgpu608`/RTX 6000 Ada, and A0.6 evaluation used `watgpu308`/L40S plus CPU. [A0.4](A0.4.md), [A0.5](A0.5.md), [A0.6](A0.6.md), [A0.7 source descriptors](A0_runs/A0_7/results/A0_RECORD_SOURCES.json).

@@RESOURCES@@

GPU values are PyTorch allocated-memory peaks. RSS values are OS process high-water marks, which can include allocations from earlier work in the same process; they are not isolated incremental per-seed working-set requirements. The complete per-epoch losses, source descriptors, resource values, and producer records are retained in the [A0.7 report](A0_runs/A0_7/results/A0_REPORT.json) and [inventory](A0_runs/A0_7/results/A0_METRIC_INVENTORY.json).

"""

def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join(["---"]*len(headers)) + "|"] + ["| " + " | ".join(map(str, row)) + " |" for row in rows])

method = method.replace("@@CALIBRATION@@", table(["Split seed", "Selected ef_search", "Recall@1000"],
    [[s, f"{v(f'ann_calibration.selected_ef_search.{s}'):,}", f(f"ann_calibration.selected_recall_at_1000.{s}", 8)] for s in range(3)]))
method = method.replace("@@QUALITY_REFERENCES@@", table(["Path", "Seed 0 gold@10", "Seed 1 gold@10", "Seed 2 gold@10", "Equal-weight mean"],
    [[label] + [f(f"{key}.gold_at_10.{s}") for s in (0,1,2,"mean")] for label,key in [
        ("Exact full-lake fusion (offline reference)","exact_full_lake_task_prior"),
        ("Exact dense top1000 + prior (offline reference)","exact1000_task_prior"),
        ("**HNSW top1000 + prior (final)**","hnsw1000_task_prior")]]))
quality_fields = ["gold_at_1", "gold_at_10", "top3_at_10", "gold_gap_at_10", "root_macro_gold_at_10"]
primary_rows = [[str(s)] + [f(f"hnsw1000_task_prior.{k}.{s}") for k in quality_fields] + [str(int(v(f"hnsw1000_task_prior.median_gold_position_when_retrieved.{s}")))] for s in range(3)]
primary_rows.append(["Equal-weight mean"] + [f(f"hnsw1000_task_prior.{k}.mean") for k in quality_fields] + ["—"])
method = method.replace("@@PRIMARY_QUALITY@@", table(["Split seed", "gold@1", "gold@10", "top3@10", "gold-gap@10", "Root-macro gold@10", "Median gold position, conditional on retrieval"], primary_rows))
method = method.replace("@@UNROUNDED@@", " / ".join(str(v(f"hnsw1000_task_prior.gold_at_10.{s}")) for s in range(3)))
diagnostics = [("Exact-pool retention", "retrieval_diagnostics.exact_pool_retention"), ("ANN retention", "retrieval_diagnostics.ann_retention"), ("Overall retention", "retrieval_diagnostics.overall_retention"), ("Gold coverage in exact top1000", "retrieval_diagnostics.exact_pool_gold_coverage_at_1000"), ("Full-fused top10 coverage in exact top1000", "retrieval_diagnostics.full_fused_top10_in_exact_pool"), ("Gold coverage in actual HNSW top1000", "hnsw1000_task_prior.actual_hnsw_gold_coverage_at_1000")]
method = method.replace("@@DIAGNOSTICS@@", table(["Diagnostic", "Seed 0", "Seed 1", "Seed 2", "Mean of seed ratios"], [[label] + [f"{100*v(f'{key}.{s}'):.4f}%" for s in (0,1,2,"mean")] for label,key in diagnostics]))
timings = ["hnsw_latency_p50_ms", "hnsw_latency_p95_ms", "prior_rerank_latency_p50_ms", "prior_rerank_latency_p95_ms", "total_latency_p50_ms", "total_latency_p95_ms"]
method = method.replace("@@TIMING@@", table(["Seed", "HNSW p50 ms", "HNSW p95 ms", "Prior/fusion p50 ms", "Prior/fusion p95 ms", "Total p50 ms", "Total p95 ms"], [[str(s)] + [f(f"retrieval_cost.{k}.{s}",6) for k in timings] for s in (0,1,2,"mean")]))
method = method.replace("@@INDEX@@", table(["Seed", "Index bytes", "Index GiB", "Build seconds"], [[s, f"{v(f'index_cost.index_bytes.{s}'):,}", f(f"index_cost.index_gib.{s}",6), f(f"index_cost.build_seconds.{s}",6)] for s in range(3)]))
method = method.replace("@@INDEX_BYTES@@", f"{int(v('index_cost.index_bytes.sum')):,}").replace("@@INDEX_GIB@@", f("index_cost.index_gib.sum",6))
costs = [("Training segments", "training_seconds"), ("Export segments", "export_seconds"), ("Prior construction", "prior_build_seconds"), ("Shared exact-reference scan", "exact_full_reference_seconds"), ("Recorded exact + HNSW evaluation segments", "complete_evaluation_seconds")]
method = method.replace("@@COSTS@@", table(["Recorded interval (seconds)", "Seed 0", "Seed 1", "Seed 2"], [[label]+[f(f"pipeline_cost.{key}.{s}",6) for s in range(3)] for label,key in costs]))
resource_fields = [("Training process RSS high-water", "training_peak_rss_bytes"), ("Training GPU allocated peak", "training_peak_vram_bytes"), ("Evaluation process RSS high-water", "evaluation_peak_rss_bytes"), ("Evaluation GPU allocated peak", "evaluation_peak_vram_bytes")]
method = method.replace("@@RESOURCES@@", table(["Recorded resource (GiB)", "Seed 0", "Seed 1", "Seed 2"], [[label]+[f"{v(f'pipeline_cost.{key}.{s}')/2**30:.6f}" for s in range(3)] for label,key in resource_fields]))

tail = r"""## 6. Reproduction and immutable evidence

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

The library audit corrected a reporting-unit mismatch in the initial A0.7 inventory: six retrieval ratios had been stored as fractions under a percent declaration. Scientific fractions remain unchanged; percent-valued inventory entries now multiply those fractions by 100, and old/new differences use consistent units. The raw arrays, learned model, and formal evaluation stay bound to their original bytes. The reader change and its validation are recorded in [UNIT_CORRECTION.md](A0_runs/A0_7/library_revision/UNIT_CORRECTION.md).

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

Training combines within-dataset RankNet, sampled model–model contrastive learning, and whole-lake logQ-corrected sampled softmax. Root-aware splits keep each dataset root on one side, and evaluation uses message graphs with test performance edges removed in both directions. For a materialized dataset–task query, HNSW selects 1,000 dense candidates and a fixed task prior from training/validation roots reranks that pool using $r=(\cos+1)/2+p_t(m)$; the system returns ten IDs. The A0 rerun obtains `gold@10` of @@GOLD@@ (mean @@MEAN@@), with ANN ID recall above 0.99 for every seed. Mean per-seed retrieval p50/p95 is @@P50@@/@@P95@@ ms from precomputed query embeddings on the formal Linux host. These measurements support held-out historical ranking recovery within the frozen node table. [A0.7 recomputation report](A0_runs/A0_7/results/A0_REPORT.json).
"""
tail = (tail.replace("@@GOLD@@", " / ".join(f(f"hnsw1000_task_prior.gold_at_10.{s}") for s in range(3)))
    .replace("@@MEAN@@", f("hnsw1000_task_prior.gold_at_10.mean"))
    .replace("@@P50@@", f("retrieval_cost.total_latency_p50_ms.mean",3))
    .replace("@@P95@@", f("retrieval_cost.total_latency_p95_ms.mean",3)))

text = intro + method + tail
assert "@@" not in text
assert not re.search(r"\]\((?:F|X|Y|Z)\d", text)
assert "0.3031" not in text
assert "BM25" not in text
OUT.write_text(text, encoding="utf-8", newline="\n")
print(json.dumps({"draft": str(OUT), "lines": len(text.splitlines()), "characters": len(text)}, indent=2))
