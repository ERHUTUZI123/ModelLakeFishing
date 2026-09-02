# HuggingFace Full-Lake Retrieval: End-to-End Evidence Source Technical Report

Evidence cutoff: **2026-09-01**. Project status: **the project is now at X5. F0–F9 are the completed full-lake construction and baseline-evaluation chain; they are not the current final method.**

The Chinese companion is [`EVIDENCE_SOURCE_LIBRARY_zh.md`](EVIDENCE_SOURCE_LIBRARY_zh.md). Exact input hashes, the inspected code snapshot, and the verification boundary are in [`EVIDENCE_SOURCE_MANIFEST.md`](EVIDENCE_SOURCE_MANIFEST.md). This report follows the requested evidence-writing discipline: Conclusion → Evidence → Impact and Problem → Evidence → Decision → Verification.

## 1. Scope, evidence levels, and the current result

### 1.1 What “current” means

The defensible current headline is neither the F8 baseline `gold@10=0.0598` nor a product of the X2/X3 prior gains and the X4 gain. It is:

- representation: X4 combined arm GD, trained with `lake_gamma=0.5` and `global_n_datasets=128`;
- primary protocol: root-aware held-out message passing;
- query policy: X5 `gold_eligible=True`;
- candidate universe: all **3,016,439** models;
- three split-seed `gold@10`: 0.1355 / 0.1417 / 0.1508, mean **0.1427**;
- matched F6 baseline: 0.0738 / 0.0418 / 0.0718, mean **0.0625**; X4 GD is therefore **2.28×** the eligible-query baseline.[M: [`X5_runs/X5_GD_ELIGIBILITY.json`](X5_runs/X5_GD_ELIGIBILITY.json), [`X5_runs/X5_F6_ELIGIBILITY.json`](X5_runs/X5_F6_ELIGIBILITY.json); D: `0.14266660/0.06249103=2.283`]

The X2 5.03× task-prior result and the X3 sibling-prior increment were measured on the **frozen F6 representation**. P and S have not been rerun on the nine X4 exports. They are separate serving-side evidence branches and must not be combined with X4 GD into an unmeasured “final-system” score.[M: [`X2.md`](X2.md), [`X3.md`](X3.md), [`X4.md`](X4.md) §8]

### 1.2 Evidence notation

| Tag | Meaning | Permitted use |
|---|---|---|
| `[M]` | Direct archived measurement or execution record | Report with its protocol, population, pool, and hardware conditions |
| `[D]` | Derived ratio, difference, or aggregate | Report with the source measurements and computation |
| `[I]` | Implemented in inspected code/configuration | Method description only; not proof of effect |
| `[P]` | Plan or preregistration written before execution | Establishes prior commitment, not a measurement |
| `[U]` | Unexecuted, unarchived, conflicting, or not independently verified | Limitation or open work only |

Evidence precedence is: **raw JSON/MANIFEST/resolved config/log > as-run code hash > execution report > plan/guide**. For example, an early `1Mplan.md` sentence says two-layer GraphSAGE, while every F6 and X4 resolved configuration records `num_layers=1`. This report uses the **one-layer as-run model** and records the conflict explicitly.

### 1.3 Executive conclusion

The project constructed a traceable 3.016M-model retrieval experiment from a 2026-08-18 HuggingFace snapshot plus six-source supervision. F0–F9 establish full-lake construction, training, held-out export, ANN retrieval, four-axis measurement, and a utility scorecard. X1–X5 then identify and intervene on supervision sparsity. The strongest supported mechanism statement is: **the mixed proposal distribution is the primary effective change; increasing per-step dataset coverage is unstable alone but adds value when combined with the proposal fix. Under the current X5 policy, GD moves `gold@10` from 0.0625 to 0.1427.** `gold@10` is historical-record recovery, not true accuracy on previously unmeasured models.

## 2. End-to-end flow and stage status

```text
HF /api/models ─F0 probe─F1 full snapshot────────────┐
                                                     ├─F2 canonicalize/merge─F3 ladders─F4 model features─┐
HF /api/datasets ─F1.5 full cards + conservative join┘                                                   │
five historical measured sources──────────────────────────────────────────────────────────────────────────┤
                                                                                                            ▼
                                            F5 heterograph─F6 baseline─F7 held-out export/HNSW─F8 A/B/C/D
                                                                                              ├─F9 utility
                                                                                              └─X1 diagnosis
                                                                                                 ├─X2 task prior
                                                                                                 ├─X3 node-level sibling protocol
                                                                                                 └─X4 G/D/GD retraining─X5 query correction
```

| Stage | Status | Principal output | Role now |
|---|---|---|---|
| F0 | Complete | Enumeration probe, capacity estimate, sharded graph store | Feasibility evidence |
| F1 | Complete | 3,003,759-model immutable snapshot | Exact snapshot prefix |
| F1.5 | Complete | 1,008,417 dataset repositories and conservative card mapping | Dataset text view |
| F2 | Complete | 247,803 supervision edges, 18,729 nodes, frozen gold rules | Label/query contract |
| F3 | Complete | Fixed rows for 3,016,439 models and 18,729 datasets | Cross-stage primary key |
| F4 | Complete | `[3,016,439,448]` float32 model matrix plus discrete ids | Model inputs |
| F5 | Complete | Five-edge-type sharded heterograph | Training input |
| F6 | Complete | Three H200 baseline checkpoints | Historical F baseline |
| F7 | Complete | Held-out embeddings and full/subsample HNSW indexes | Frozen evaluation inputs |
| F8 | Complete | A/B/C/D baseline measurements | Full-lake baseline |
| F9 | Partly complete | Utility layers 1–2 and frozen `t0` | Metric interpretation; blind/temporal layers pending |
| X1 | Complete | Three-seed diagnosis | Identifies proposal and coverage shrinkage |
| X2 | Complete | Task-prior P axis | Strong F6 serving prior |
| X3 | Complete | Node-level sibling S axis | Second-protocol signal |
| X4 | Complete | G, D, and GD × three seeds | Current representation evidence |
| X5 | Complete | Eligible/excluded query evaluation | Current primary query policy |

## 3. Experimental objects and metric contract

### 3.1 Candidate snapshot, supervision, and lake

The HF snapshot contains **3,003,759** models. Six-source supervision introduces **12,680** historical models absent from that snapshot. Actual training, export, and primary evaluation therefore use **3,016,439** candidates.[M: [`F1_runs/PROVENANCE.json`](F1_runs/PROVENANCE.json), [`F2_runs/F2_MERGE_REPORT.json`](F2_runs/F2_MERGE_REPORT.json)]

The snapshot remains an exact row prefix; historical-only models are appended. The required “all candidates” and `in_snapshot=true` evaluations answer different questions, and restricting the latter changes query eligibility. They are not the same queries with 0.42% of candidates removed.[M: [`F7.md`](F7.md) §3]

### 3.2 Supervision and gold construction

Native model-index values are grouped by `(dataset, task, full_metric_name)`. Nonfinite/unparseable values are removed; duplicate quadruples take their median; lower-is-better metrics are reversed after within-group normalization; unknown-direction rows remain edges but cannot define gold; constant groups receive 0.5.[I: [`canonicalize_rf.py`](../../scale1m/canonicalize_rf.py), [`metric_semantics.py`](../../scale1m/metric_semantics.py)]

Historical curated weights are already oriented but lack original metric semantics. They receive a fifth `curated` direction class and are min-max normalized within `(dataset node, source)`. Cross-source conflicts retain one edge by the fixed priority `modellens_v2 > d0_v1_5 > a_ctrl_2000m > hf_effective > diverse_zoo > hf_model_index`; displaced records go to the conflict table. A stratified 200-edge cap is applied per dataset node after conflict resolution.[I/M: [`merge_supervision.py`](../../scale1m/merge_supervision.py), [`F2.md`](F2.md)]

The frozen rule set is `rf-gold-2.0`, SHA-256 `be3fb05e…`. Its `gold_eligible` condition requires known direction, non-RL, non-placeholder name, and at least three candidates. F8 used only the weaker “computable with ≥3 nonconstant candidates” condition; X5 finally applies `gold_eligible` to primary evaluation.[M]

### 3.3 Root-aware split and leakage control

The primary split unit is a dataset root, not an individual edge or configuration. `make_root_aware_splits()` shuffles roots by `split_seed`, greedily assigns about 20% of edges to test and 10% to validation, then uses the rest for training. Thirty percent of training edges form disjoint supervision and are removed from the train message graph. Validation sees train messages; test sees train+validation messages; reverse edges are cropped consistently; negative labels are sampled 1:1 from absent model–dataset pairs.[I: [`d0_splits.py`](../../stage2TrainGraphSAGE/d0_splits.py#L38)]

`split_seed∈{0,1,2}` and `init_seed=0`. The reported intervals therefore measure **split variation**, not initialization variation.[M: F6/X4 MANIFESTs]

### 3.4 Exact metric semantics

For query `d`, `g_d` is the highest normalized recorded candidate and scores are `z_d·z_m`:

- `gold@K = 1[rank(g_d)≤K]`;
- `top3@10`: at least one of the recorded top three appears in the returned ten;
- `gold-gap@10`: at least one model within 0.01 of recorded best appears in ten;
- `root_gold@10`: root-macro gold rate;
- `median rank/N`: median absolute gold rank divided by pool size;
- `vs random = gold@10/(10/N)`.

Ranks are `# strictly better + 1`; the streaming scorer explicitly masks a probe's comparison with itself.[I: [`global_metrics.py`](../../scale/global_metrics.py)]

`gold@10` is a gold-survival/record-recovery metric. It is not an error rate on unknown candidates. F9 measured that 95.24% of F6 seed-0 top-ten slots had no record on the target query.[M: [`F9.md`](F9.md)]

## 4. F0–F1.5: acquisition and storage

### 4.1 F0 probe

The `createdAt` descending probe fetched 200 full pages × 1,000 records: 200,000 unique models, zero duplicates, zero ordering violations, a live next cursor after page 200, 21.7 s, and 74,491,136 bytes. Cursor decoding showed only an immutable `_id < ObjectId` boundary.[M: [`F0_runs/f0_probe.json`](F0_runs/f0_probe.json)]

F0's 54 monthly probes estimated 2.89M models with a declared ±15% range, roughly 13.8 GB downloaded and 480 MB gzip in 30–40 minutes. F1 later measured 3.003759M, about 3.9% above the estimate.[D/P]

A 100K graph round-trip through `.npy` mmap, NPZ structures, parquet ids, and hashed metadata was tensor-identical. Initial load memory fell from 243.3 MB for `torch.load` to 50.3 MB with mmap; touching all data rose to 237.8 MB. The decision was float32 storage with sharding/mmap, not fp16.[M/I: [`graph_store.py`](../../scale1m/graph_store.py)]

### 4.2 F1 model snapshot

```bash
python -m scale1m.hf_crawl --sort createdAt --direction -1 \
  --limit 4000000 --shard-size 50000 --v2-fields --out <DATA>/data1m/candidates_full
```

The crawl ran from 18:01:49 to 18:29:18 UTC on 2026-08-18: **3,003,759** records, 3,004 pages, 61 shards, 1,648.6 s, one retry, six rate-limit sleeps totaling 996 s, zero duplicates, and normal cursor exhaustion.[M: [`F1_runs/PROVENANCE.json`](F1_runs/PROVENANCE.json)] The last `createdAt` field is 2022-03-02, but it is an API field/backfill boundary, not evidence of the first-ever HF release.[U]

Native snapshot inventory: 107,210 models with parseable model-index data; 208,035 model–dataset pairs; 2,158,375 model–dataset–metric triples; 7,700 datasets and 8,793 dataset-task nodes. Declared lineage is 894,089, of which 859,056 resolve inside the lake (96.08%); the largest parent has 43,346 children. Initial layers are 110,806 labeled (3.69%), 862,666 lineage (28.72%), and 2,030,287 plain (67.59%). Safetensors size coverage is 28.19%.[M: [`F1.md`](F1.md)]

### 4.3 F1.5 dataset cards

```bash
python -m scale1m.hf_crawl_datasets --out <DATA>/data1m/datasets_full
```

The dataset crawl enumerated **1,008,417** repositories, 1,009 pages, 11 shards, 411.5 s, no duplicates/retries, two rate-limit sleeps totaling 200 s, and cursor exhaustion.[M: [`F15_runs/PROVENANCE.json`](F15_runs/PROVENANCE.json)]

Matching accepts exact normalized ids and a basename only when the model-index name has no owner and the basename is unique. It rejects 85 cross-owner and 1,626 ambiguous basenames. Picking the most-downloaded ambiguous result would raise apparent coverage from 33.29% to 52.75% but would create false ownership mappings.[M: [`F15_runs/f15_policy.json`](F15_runs/f15_policy.json)]

Before six-source expansion, 2,927/8,793 nodes (33.29%) had an HF card and 54.18% of records were covered. In the non-RL, ≥3-model view, node/record coverage was 35.50%/59.26%. After merge, 3,928/18,729 nodes (20.97%) had a real card; eligible-query coverage was 1,437/7,859 (18.28%) and 27.87% by edges. ModelLens query coverage was only 5.19% by nodes and 2.83% by edges.[M: [`F15_runs/f15b_coverage.json`](F15_runs/f15b_coverage.json)]

Unmatched nodes fall back to a cleaned name plus task. The card descriptor includes name, task categories, up to ten non-colon tags, and 400 description characters.[I: [`match_dataset_cards.py`](../../scale1m/match_dataset_cards.py), [`d0_build_graph.py`](../../stage1BuildTransferGraph/d0_build_graph.py#L138)]

## 5. F2–F5: canonical supervision, rows, features, and graph

### 5.1 F2 native canonicalization and six-source merge

Of 2,158,375 raw metric rows, 61,294 are unparseable/nonfinite; 2,097,081 remain. Median deduplication yields 1,435,162; 38,358 constant groups receive 0.5. Native output has 7,944 dataset-task nodes and 143,478 primary edges, capped to 74,346 over 131 affected nodes.[M: [`F2_runs/F2_REPORT.json`](F2_runs/F2_REPORT.json)]

Raw direction counts are higher 1,539,828 (70.96%), lower 15,835 (0.73%), reward 58,456 (2.69%), and unknown 555,929 (25.62%). The deduplicated denominator has different counts and must not be mixed with these percentages. Most `mean_reward` values are strings such as `11.05 +/- 5.90`; no special parser was added, so low final RL share is primarily a parse outcome, not a successful balancing intervention.[M]

The edge cap improves effective-dataset count `1/HHI` from 177.6 to 520.8 and lowers top-ten dataset share from 14.95% to 5.32%.[M]

Six-source merge starts with 531,958 rows, collapses 5,102 intra-source duplicates, records 1,502 cross-source conflicts, retains 525,354 before cap, and ends with **247,803** edges over **18,729** nodes; 339 nodes are capped. Eligible query depths are 7,859 at ≥3, 5,526 at ≥5, 3,450 at ≥10, and 1,909 at ≥20.[M: [`F2_runs/F2_MERGE_REPORT.json`](F2_runs/F2_MERGE_REPORT.json)]

| Source | Edges after cap | Gold queries |
|---|---:|---:|
| ModelLens v2 | 117,898 | 4,681 |
| D0 v1.5 | 45,992 | 1,143 |
| A-control 2000m | 3,670 | 65 |
| HF-effective | 5,223 | 75 |
| Diverse zoo | 1,350 | 17 |
| HF model-index | 73,670 | 1,878 |

The final historical-only model count is 12,680. The earlier 16,713 figure was a pre-merge estimate and is superseded.[M]

### 5.2 F3 frozen ladders

F3 keeps the 3,003,759 snapshot rows in crawl order and appends 12,680 normalized historical ids. Historical size is unknown; family is recovered for all appended models; no historical feature vector is copied. Eight integrity assertions pass. Model and dataset ladder hashes are `fee360d1…` and `31c027ff…`.[M/I: [`F3_runs/LADDER_REPORT.json`](F3_runs/LADDER_REPORT.json), [`build_ladder_rf.py`](../../scale1m/build_ladder_rf.py)]

### 5.3 F4 model features

The frozen matrix is `x_m=[e_name64||e_desc384]`, not size/family inside 448 dimensions. Size and family ids are separate and select learnable 16-dimensional embeddings during training. Name uses seed 42; MiniLM encodes a descriptor consisting of cleaned model id, family, and parameter size when known.[I]

On local RTX 4060, four mmap-written parts complete in 712.7 s. `x_m` is `[3,016,439,448]` float32, 5.41 GB, SHA `ba102087…`; family vocabulary has 41,056 rows, 13.561% `Other`, SHA `00d304df…`; 71.929% have unknown size. Shape, finite/nonzero, id-range, and 200-row head/tail recomputation gates all pass.[M: [`F4_runs/FEATS_REPORT.json`](F4_runs/FEATS_REPORT.json)]

### 5.4 F5 heterograph

Dataset features are `[name hash 64 || card MiniLM 384 || statistics 10]=458`. `similar_to` is k=20 cosine KNN on the card block. Graph construction takes 47.6 s and yields:

| Relation | Forward | Reverse |
|---|---:|---:|
| `trained_on` | 247,803 | 247,803 |
| `similar_to` | 374,580 | — |
| `is_base_of` | 859,065 | 859,065 |
| Total across five edge types |  | **2,588,316** |

Lineage resolution is 96.083%, with 76,672 distinct parents, child-count p50/p90/p99 1/6/118, 119 parents above 1,000 children, and maximum 43,346.[M: [`F5_runs/GRAPH_REPORT.json`](F5_runs/GRAPH_REPORT.json)]

RF graph task/class/arity ids each collapse to a single category. Training succeeds, but the three dataset discrete tables act as shared biases. X2 later repairs task grouping only for the serving sidecar by reading the real parquet task column; F6/X4 training features remain degenerate.[M/U]

The graph contract caught a real parsing bug: literal family names `nan` and `null` were interpreted as pandas NA and collapsed. `keep_default_na=False` fixed the row mismatch.[M]

## 6. F6: full-lake baseline training

### 6.1 As-run architecture and objective

The authoritative configuration is the archived `resolved_config.json`:

- **one-layer** relation-specific, edge-aware heterogeneous GraphSAGE; hidden/output 128; L2-normalized outputs;
- model input: frozen 448 + learned size 16 + learned family 16;
- dataset input: frozen 458 + learned task/class/arity 16/8/4, although each table has one row in RF;
- shared head, dot scorer, RankNet with `rank_min_gap=0.02`;
- `top_frac=0.1`, `lambda_rank=lambda_contrast=lambda_global=1`, `lambda_dm_contrast=0`;
- full-lake logQ sampled softmax `q(m)∝(deg_train(m)+1)^0.75`, 256 global negatives, 16 datasets per step;
- batch 1,024, fanout loader, sparse membership, 256 sampled contrastive negatives, inference chunk 50,000; AMP off; expensive diagnostics skipped in training and recomputed in F8.[M/I: [`F6_runs/RF_full_s0_e25/metadata/resolved_config.json`](F6_runs/RF_full_s0_e25/metadata/resolved_config.json), [`losses.py`](../../stage2TrainGraphSAGE/losses.py), [`train_rung.py`](../../scale1m/train_rung.py)]

The objective combines RankNet performance ranking, model-side supervised contrast, and a global lake logQ term. Absolute loss is comparable between F6 and X4 D because the global term averages across datasets; changing gamma changes the correction distribution, so G/GD loss must not be compared numerically with F6/D.[I/M]

### 6.2 Execution

All three runs used H200 NVL on `watgpu508`, Python 3.11.4, PyTorch 2.12.0+cu130, CUDA 13.0, and pyg_lib 0.8.0+pt212cu130.

| Split seed | Slurm | Wall time | Peak GPU | Loss epoch 0→24 | Gate |
|---:|---:|---:|---:|---:|---|
| 0 | 1515431 | 426.4 s | 38.001 GB | 22.2925→16.6288 | PASS |
| 1 | 1515432 | 498.8 s | 38.015 GB | 22.5152→16.4704 | PASS |
| 2 | 1515433 | 379.4 s | 37.976 GB | 22.2175→16.2624 | PASS |

The gates require loss descent, no NaN, and movement of the learned size/family tables while frozen `x` remains gradient-free. All runs bind graph digest `0e80b839…`, family vocabulary `00d304df…`, and N=3,016,439.[M: F6 MANIFESTs]

Only 37,755 / 39,930 / 35,991 models are train-visible labeled—about 1.25% of the lake. This is the sparsity later quantified by X1.[M]

F6 is not reproducible from one clean Git commit: run metadata points to an older head, the patch captures tracked changes only, and key files were untracked. Checkpoint bindings and file hashes provide partial recovery, but the limitation must remain explicit.[U: [`F6.md`](F6.md)] The planned 500K retrained scaling point was never built; the existing scaling curve changes retrieval N under a fixed representation only.[U]

## 7. F7: held-out export and HNSW

`export_rf.py` separates `embed`, `metrics`, `index`, and `curve` into processes because the 5.41 GB graph features, 1.54 GB embedding, and 2.2 GB index cannot safely co-reside.[I]

Each seed exports full-message `z_m/z_d` for serving and held-out-message `z_m_eval/z_d_eval` for evaluation, plus candidate records, id tables, a full HNSW, and—under seed 0—12 subsample indexes. Only `z_*_eval` is valid for A-axis claims.

Embedding takes 55.8–72.1 s; metrics 26–33 s; index 110–113 s. HNSW uses M=32, construction ef=200, and chooses the minimum search ef attaining `recall@50≥0.99`.[M/I]

Eight gates pass: exact N; held-out score below full-message score; row-order checks; chunked/full forward agreement on a 100K subgraph at 7.5e-8/1.0e-7/8.9e-8; independent metric-harness agreement; ANN recall; all 46,146 supervised/gold rows in every subset; and both candidate-pool reports.[M]

Full-message versus held-out `gold@10` is 0.1046 vs 0.0700, 0.0555 vs 0.0399, and 0.1674 vs 0.0696. Using the wrong embedding can inflate the score by as much as 2.4×.[M]

The full index is 2,209.3 MB, `ef=50`, recall 0.9983–0.9988. Subsets at 100K/250K/500K/1M each have three random seeds, always retaining the 46,146 supervised/gold rows before sampling the unlabeled tail.[M]

F7 ran on the local RTX 4060 laptop. Deterministic embeddings, recall, and disk size remain valid; absolute latency is machine-specific, so F8 compares HNSW and brute force within that machine only.[U]

## 8. F8: four-axis F baseline

### 8.1 A: record recovery

| Metric | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---:|---:|---:|---:|
| `gold@1` | 0.0019 | 0.0078 | 0.0107 | 0.0068 |
| `gold@10` | 0.0700 | 0.0399 | 0.0696 | **0.0598** |
| `top3@10` | 0.1232 | 0.1570 | 0.1147 | 0.1317 |
| `gold-gap@10` | 0.0924 | 0.0703 | 0.0859 | 0.0829 |
| `root_gold@10` | 0.0226 | 0.0328 | 0.0407 | 0.0320 |
| Median gold rank | 1,908 | 881 | 3,363 | 2,051 |
| Median rank/N | 6.33e-4 | 2.92e-4 | 1.11e-3 | 6.80e-4 |
| Queries | 1,558 | 1,153 | 1,595 | 1,435.3 |

[M: [`F8_runs/F8_REPORT.json`](F8_runs/F8_REPORT.json)] The preregistered rank/N and gold ranges were met.[P/M]

Under fixed seed-0 embeddings, raising N from 100K to 3.016M changes `gold@10` only 0.0706→0.0700 and median absolute rank about 1,736→1,908. The unlabeled tail is not the dominant cause; ordering among the fixed supervised population is.[M/D]

The snapshot-only row is 0.0229/0.0532/0.0613 over 1,003/752/636 queries. It is a different eligible-query population and cannot be differenced directly from the all-candidate row.[M]

### 8.2 B: iso-recall latency

| N | ef | HNSW p50 | Brute p50 | Speedup |
|---:|---:|---:|---:|---:|
| 100K | 50 | 0.0175–0.0187 ms | 1.094–1.262 ms | 59–67× |
| 250K | 50 | 0.0199–0.0208 ms | 4.527–5.216 ms | 227–251× |
| 500K | 50 | 0.0185–0.0214 ms | 9.439–10.655 ms | 493–512× |
| 1M | 50 | 0.0195–0.0210 ms | 18.388–20.016 ms | 925–991× |
| 3.016M | 50 | 0.0211 ms | 57.261 ms | **2,714×** |

All recalls exceed 0.998. HNSW p50 has fitted exponent 0.036 versus about 1 for full scan. Batched throughput is 52,655 QPS at one thread and 296,417 at 24 logical threads. An earlier per-query “multithread QPS” varied by 65× because hnswlib parallelizes across query batches; it is discarded, not hidden.[M]

### 8.3 C: cold-start geometry

| Layer | Count | Share | Within-layer cosine | Effective dimension |
|---|---:|---:|---:|---:|
| warm | 2,949 | 0.10% | 0.394–0.409 | 3.3–3.9 |
| cool | 43,197 | 1.43% | 0.293–0.426 | 3.8–4.0 |
| cold | 875,314 | 29.02% | 0.882–0.931 | 3.0–3.3 |
| frozen | 2,094,979 | 69.45% | 0.879–0.927 | 2.7–3.4 |

Cold+frozen account for 98.47%. The preregistered expectation that cold would be more collapsed than frozen did not reproduce; their means are about 0.908 and 0.907. Sibling–random separation falls to 0.028–0.049. Examples include 32,534 Qwen1.5-0.5B children at cosine 0.9863 and 24,040 Gemma-2B children at 0.9877. ANN recall is already high, so this is representation collapse, not index tuning.[M/D]

### 8.4 D: system cost

The measured chain is: model crawl 27.5 min; dataset crawl 6.9 min; model features 712.7 s; graph 47.6 s; training 379–499 s/seed at 38.0 GB; export 55.8–72.1 s/seed; full index 109.9–113.1 s/seed. Disk is approximately 5.27 GiB graph, 1.44 GiB `z_m`, 2.21 GiB full index, and 4.07 GiB for 12 subset indexes. HNSW insertion is p50 0.160 ms and p95 0.255 ms.[M]

The insertion number is index-only. Descriptor generation, MiniLM encoding, discrete ids, and inductive GNN forwarding were not timed as one end-to-end onboarding transaction.[U] `displacement_quality` remains unimplemented under a non-label-overlapping definition.[U]

## 9. F9: recommendation-utility scorecard

F9 runs no candidate model. It completes historical-record recovery and metadata feasibility, freezes the temporal `t0` list, and leaves human blind review and future temporal validation undone.[M/U]

On the same 1,558 seed-0 queries, graph training versus pure MiniLM text gives:

| Metric | F6 graph | Text-only |
|---|---:|---:|
| recorded `gold@10` | 0.0700 | 0.0392 |
| recorded `top3@10` | 0.1232 | 0.0591 |
| median historical-best rank | 1,908 | 160,270 |
| same-task evidence@10 | 0.2963 | 0.0503 |
| record coverage@10 | 0.0476 | 0.0166 |

[M: [`F9_runs/F9_SCORECARD.json`](F9_runs/F9_SCORECARD.json)] `random_task_pool` obtains 0.2349 because it is told the task and samples ten from a frequently tiny already-evaluated pool; its expected value from the measured pool-size distribution is about 0.235. It is an oracle-like reference, not a superior full-lake retriever.[D]

F6 seed-0 metadata metrics are available 0.7099, licensed 0.5978, endpoint-compatible 0.5161, library-tag 0.6006, family diversity 0.4989, at least one feasible 0.9724, and median top-ten size 1.77B. X2 later measures availability over all splits as 0.7099/0.6273/0.5729, so 0.7099 must be labeled a seed-0 point.[M]

The frozen `t0_recommendations.parquet` has 15,580 rows (1,558×10), dated 2026-08-21. No temporal outcome exists yet.[M/U] Full dense re-evaluation of unknown candidates was estimated at 73.5 TB and roughly 5,300 GPU-hours; this is a resource estimate, not consumed compute.[D/P]

## 10. X1: diagnosis that begins the current X series

X1 freezes six diagnostics into `fast_lever_audit.py` and reruns all three split seeds without retraining or modifying exports. Total time is about 84 minutes; the full-lake rank scan takes 4,981.3 s.[M/I]

Removing all 2,970,293 unlabeled models changes `gold@10` from 0.0700/0.0399/0.0696 to 0.0706/0.0408/0.0696. Dark matter is not the main displacement source.[M]

Training-side shrinkage is much larger:

| Quantity | 100K mean | 3M mean | Reduction |
|---|---:|---:|---:|
| Proposal mass on train-visible labeled models | 0.6138 | 0.0352 | 17.4× |
| Expected labeled models among 256 negatives | 157.1 | 9.0 | 17.4× |
| Global-term touches per visible dataset over 25 epochs | 4.06 | 1.47 | 2.77× |
| Combined discriminative exposure | — | — | about **48×** |

[M/D: [`X1_runs/X1_FAST_LEVERS.json`](X1_runs/X1_FAST_LEVERS.json)] The earlier audit counted negative labels as loader steps and used full-graph degree; X1 corrects both to the as-trained positive loader and train-visible degree.

For `q=(1−γ)q_degree+γUniform(labeled)`, seed-0 gamma 0/.25/.5/.75 gives labeled mass .0355/.2766/.5177/.7589 and 9.1/70.8/**132.5**/194.3 labeled negatives. Gamma 0.5 becomes the X4 intervention.[M]

Zero-training measurements give all-task random 0.1857, task-filtered MIPS 0.2206, shrunk task prior 0.3014, and task-pool ceiling 0.5905. CSLS is a negative result: the three-seed mean falls 41%, from 0.0598 to 0.0353 with test-side density and 0.0357 with deployable train-side density. It is removed from the candidate set.[M]

The F6 top ten is 96.72%–99.35% supervised models; the hottest 100 models occupy 31.3%–51.4% of slots; snapshot-resident share is only 73.2%/63.8%/57.9%; average distinct families are 4.99/4.79/4.39.[M]

Although 66.0%/67.5%/77.7% of queries lie in multi-node roots, root-aware splitting leaves exactly zero queries with a train-visible sibling. X1 also counts 82/52/50 `gold_eligible=False` queries, mostly unknown direction, motivating X5.[M]

## 11. X2: task-prior P axis

X2 adapts the prior sidecar to the sharded graph, parquet ids, and split-specific visibility. RF's graph task id is constant zero, so correct groups come from the real parquet task column, normalized into 2,198 groups.[I]

Split sidecars contain only train+validation visible edges—198,216 / 196,912 / 196,124—and assert no test-dataset edge survives. The old D0 invocation reproduces all five historical arrays exactly.[M]

The current post-X5 `eval_rf.py` closes two path hazards. `--full-sidecar` is no longer fixed at module load to `exports_rf/RF_full_s0_e25`; it is derived after parsing `args.run_fmt`. Training MANIFEST paths are resolved separately through `--f6-runs/--f6-run-fmt`, and a missing seed raises an error instead of silently leaving `train_per_seed` empty. This is reproduction hardening in the present code, not evidence that X4 P/S has run.[I: [`eval_rf.py`](../../scale1m/eval_rf.py), [`X4GPU.md`](X4GPU.md) §4.5]

The full-lake fusion is `minmax(MIPS)+alpha*sibling+beta*task`; sibling is asserted zero under root-aware. Task boost is a k=5 shrunk mean of a model's performance on other datasets in the same task. Evaluation scans all 3,016,439 models instead of a 512 retrieval pool.[I]

| Ranking | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---:|---:|---:|---:|
| MIPS | 0.0700 | 0.0399 | 0.0696 | 0.0598 |
| Degenerate one-group prior | 0.1232 | 0.1127 | 0.0940 | 0.1100 |
| Task β=.5 | 0.2914 | 0.3174 | 0.2464 | 0.2851 |
| Task β=1 | 0.3132 | 0.3356 | 0.2539 | **0.3009** |
| Task β=2 | 0.3087 | 0.3400 | 0.2558 | 0.3015 |
| β→∞ lower bound | 0.3338 | 0.3322 | 0.2301 | 0.2987 |

[M: [`X2_runs/X2_PRIOR_FUSION.json`](X2_runs/X2_PRIOR_FUSION.json)] Saturation near beta 1 and the almost identical prior-only result show that the 5.03× gain is primarily the task prior, not synergy with embeddings.[D]

The same top-ten lists worsen availability 0.6367→0.6017, feasibility 0.9118→0.8973, family diversity 0.4722→0.4320, and median parameter count **2.42B→7.57B**, while record coverage rises 0.0498→0.1764 and same-task evidence 0.2763→0.7574.[M]

## 12. X3: second protocol and sibling prior

The X3 preregistration SHA `4ee14720…` is embedded in the result JSON, establishing ordering.[M]

Protocol A holds out whole roots and represents an unseen benchmark family. Protocol B permits other configurations in the query's known root. X3 does not retrain node-level embeddings: queries and embeddings remain root-aware held out; only which prior edges are readable changes.[I]

This makes the retrieval side a lower bound—embeddings never trained on siblings—and full-sidecar task(B) an upper bound because it includes labels that a true node-level split would reserve. The defensible second-protocol line is `sibling + task(A)`, not `sibling + task(B)`.[U]

| Ranking | All queries | Sibling stratum | No-sibling stratum |
|---|---:|---:|---:|
| MIPS | 0.0598 | 0.0678 | 0.0418 |
| A: task | 0.3009 | 0.3251 | 0.2551 |
| B: sibling | 0.2145 | 0.2866 | 0.0418 |
| B: sibling, dissimilar names only | 0.1209 | 0.1532 | 0.0418 |
| **B: sibling + task(A)** | **0.3536** | **0.3975** | 0.2551 |
| B: task(B) | 0.3744 | 0.4163 | 0.2622 |
| B: sibling + task(B) | 0.4314 | 0.4961 | 0.2622 |

[M: [`X3_runs/X3_PROTOCOL_B.json`](X3_runs/X3_PROTOCOL_B.json)] Sibling alone is 4.2× MIPS in its active stratum; adding sibling to task(A) is +22.3% there and +17.5% overall. Removing name-near siblings lowers 0.2866 to 0.1532, showing high sensitivity to closely related configurations.[M]

`sibling+task(A)` lowers availability to 0.5549, feasibility to 0.7615, family diversity to 0.4091, and raises median parameter count to **12.11B**.[M]

## 13. X4: G/D/GD retraining

### 13.1 Preregistered interventions and execution evidence

Before submission, X4 fixed G as `lake_gamma=.5`, D as `global_n_datasets=128`, GD as both, three split seeds, 25 epochs, all else fixed. The primary pass condition was GD mean `gold@10≥.09` and every seed above its own F6 baseline.[P]

Formal remote hashes match the present local files: `losses.py c9b351…`, `ablation.py e0fa47…`, `train_rung.py 08904b…`, and `train_rung_x4.sbatch f1237a…`. The sbatch's earlier `a0b2ab…` version was superseded when `X4_DIRECT=1` bypassed a broken `srun` job-step layer; `f1237a…` is the as-run version.[M]

All nine runs contain 25 epoch records, no NaN, PASS mechanism gates, N=3,016,439, graph digest `0e80b839…`, and vocabulary `00d304df…`. Preemption/timeouts used same-RUN_ID `last.pt` continuation. No-op resumes at epoch 25 write final gates but do not represent full training cost.[M]

### 13.2 Results

Old query policy, all candidates:

| Metric | F6 | G | D | GD |
|---|---:|---:|---:|---:|
| `gold@1` | .0068 | .0177 | .0071 | **.0228** |
| `gold@10` | .0598 | .1160 | .0745 | **.1429** |
| `top3@10` | .1317 | .2082 | .1521 | **.2508** |
| `gold-gap@10` | .0829 | .1448 | .1110 | **.1866** |
| `root_gold@10` | .0320 | .0511 | .0593 | **.0947** |
| Median rank | 2,051 | 1,326 | **1,178** | 1,590 |

[M: [`X4_runs/reports/`](X4_runs/reports/)] GD is .1341/.1440/.1505, with +.0642/+.1041/+.0809 over matched F6 seeds, satisfying preregistration. G is .1194/.1075/.1210 and consistently positive. D is .0546/.1023/.0665, changes −.0154/+.0624/−.0031, and is not independently stable. G captures 67.6% of the combined mean gain; GD is 23.2% above G and positive by seed, supporting a conditional coverage contribution.[D]

Median rank and top-ten hit can move differently: GD seed 2 raises `gold@10` .0696→.1505 while median rank worsens 3,363→3,858. The gain concentrates in the head.[M]

### 13.3 Geometry

| Layer | F6 cosine | GD cosine | F6 effective dim | GD effective dim |
|---|---:|---:|---:|---:|
| warm | .402 | .218 | 3.6 | 6.6 |
| cool | .350 | .198 | 3.9 | 7.4 |
| cold | .908 | .783 | 3.2 | 7.9 |
| frozen | .908 | .778 | 3.0 | 8.0 |

Sibling–random separation rises .0359→.0796. This unpreregistered result suggests training sampling contributes to collapse, but the mechanism is post hoc and effective dimension remains only about 8/128.[M/U]

### 13.4 Hardware and continuation disclosure

F6 used H200. Final X4 MANIFESTs report GD on L40S; G on RTX 6000 Ada; D seeds 0/2 on RTX 6000 Ada and D seed 1 on L40S. Early GD seeds 1/2 epochs 0–9 ran under `watgpu308 schoolgpu` allocations that timed out without MANIFESTs or GPU-name logs. A probe once received RTX A6000, but another allocation under the same GRES (`1522415`) records L40S. The early segment type is therefore **unarchived**, not inferred as A6000.[M/U]

X4D seed 1 records Python 3.11.9; the other eight record 3.11.4; all use PyTorch 2.12.0+cu130. The comparisons share graph, code, config, and split, but are not same-hardware/same-Python replications.[M]

Job `1522429` performed GD seed-1 epochs 10–24. `1522415` started after epoch 25 and performed a no-op gate only. The observed 43,030 MiB for D seed 2 was never archived in nvidia-smi output and is not independently reproducible. No-op MANIFEST values 35.6 s/5.252 GB and 15.5 s/5.251 GB are not full-run costs; the pre-overwrite GD seed-1 record was 2,329.9 s/37.964 GB.[M/U]

`global_n_datasets=128` overshoots X1's approximate 100K alignment point of 44 and gives about 11.7 touches versus 4.06. Gamma .5, at 132.5 labeled negatives versus 157.1, is the closer same-order alignment.[D]

## 14. X5: current query policy

X5's E axis recomputes old-all, eligible-only, and excluded-only rows without changing training, embeddings, or indexes.[I]

| Seed | Before | Eligible | Excluded | Unknown direction | Placeholder | RL |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,558 | 1,476 | 82 | 61 | 20 | 1 |
| 1 | 1,153 | 1,101 | 52 | 48 | 4 | 1 |
| 2 | 1,595 | 1,545 | 50 | 45 | 4 | 2 |

One query in each of seeds 1 and 2 has two exclusion reasons.[M]

| Representation | Old `gold@10` | Eligible `gold@10` | Relative change | Excluded-only |
|---|---:|---:|---:|---:|
| F6 | .0598 | **.0625** | +4.5% | .0000 |
| X4 G | .1160 | **.1197** | +3.2% | .0291 |
| X4 D | .0745 | **.0724** | −2.8% | .1117 |
| X4 GD | .1429 | **.1427** | −0.1% | .1474 |

[M: [`X5_runs/`](X5_runs/)] The correction is small and inconsistent in direction, leaving the X4 conclusion intact. Unknown-direction argmax labels remain untrustworthy even when X4 ranks them well.[D]

The eligible row should now be primary while the old row remains for historical alignment. Existing EXPORT_MANIFEST and F8_REPORT files still contain the old policy; X5 is separate. Snapshot-only was not recomputed, and training still samples ineligible datasets.[U]

## 15. Current result table: what may and may not be combined

### 15.1 Primary protocol, all candidates, X5-eligible queries

| Representation | Seed 0 | Seed 1 | Seed 2 | Mean | Relative to F6 | Evidence status |
|---|---:|---:|---:|---:|---:|---|
| F6 | 0.0738 | 0.0418 | 0.0718 | 0.0625 | 1.00× | Historical representation baseline |
| X4 G | 0.1233 | 0.1108 | 0.1249 | 0.1197 | 1.92× | Same direction in all three seeds |
| X4 D | 0.0508 | 0.0990 | 0.0673 | 0.0724 | 1.16× | Directionally unstable |
| **X4 GD** | **0.1355** | **0.1417** | **0.1508** | **0.1427** | **2.28×** | **Current primary representation** |

[M: [`X5_F6_ELIGIBILITY.json`](X5_runs/X5_F6_ELIGIBILITY.json), [`X5_GD_ELIGIBILITY.json`](X5_runs/X5_GD_ELIGIBILITY.json), [`X5_G_ELIGIBILITY.json`](X5_runs/X5_G_ELIGIBILITY.json), [`X5_D_ELIGIBILITY.json`](X5_runs/X5_D_ELIGIBILITY.json)] This is the safest headline table for the paper. The exact unrounded means are 0.06249103 for F6 and 0.14266660 for GD; the ratio is 2.283.[D]

### 15.2 Separate service-side evidence branches, all on F6 representations

| Branch | Protocol | `gold@10` | Supported conclusion | Unsupported conclusion |
|---|---|---:|---|---|
| X2 task, beta=1 | Root-aware, F6 representations | 0.3009 | Same-task historical performance is a strong prior | The final value of X4+task |
| X3 sibling+task(A) | Second service protocol, F6 representations | 0.3536 | A known benchmark's new configuration can exploit sibling evidence | Main root-aware performance or a fully retrained node-level result |
| X3 sibling+task(B) | Prior-side upper bound | 0.4314 | An upper bound and signal-existence result | A deployable, leakage-free primary result |

The X2/X3 and X4 numbers must not be added, multiplied, or maximized into a purported “current system” score. A unified service result requires split-specific and full sidecars for the nine X4 export directories, rerunning P/S/E, and rerunning the second-layer utility table.[U]

### 15.3 F-series system measurements remain valid but are not X4 measurements

HNSW `recall@50≈0.999`, full-lake p50 0.0211 ms, the 2,714× speedup over exhaustive search, and the 2.21 GiB index are measurements of the F6 embeddings. X4 did not rerun the `index/curve` stages or the B/D axes. The algorithm and candidate count are unchanged, but the vector geometry changed; these values therefore cannot be labeled “measured on X4.”[U]

## 16. Artifacts, paths, and reproduction entry points

### 16.1 Large-data root

`<DATA>` in the documentation denotes `$MLF_DATA_DIR/data1m` in the executed environment. The principal directory contract is:

```text
data1m/
├── candidates_full/                    F1: 61 model shards + PROVENANCE/CURSOR/SHARDS
├── datasets_full/                      F1.5: 11 dataset shards + cards
├── rf/canon/                           F2: canonical models, supervision, conflicts, nodes
├── ladder_rf/                          F3: model and dataset parquet ladders
├── feats_rf/                           F4: x_m, ids, family vocabulary, reports
├── graphs/hgraph_rf/                   F5: sharded heterogeneous graph store
├── exports_rf/RF_full_s{0,1,2}_e25/   F7: F6 embeddings and HNSW indexes
├── metrics_rf/                         F8/X1/X2/X3 baseline reports
├── utility_rf/                         F9 model metadata, scorecard, and t0 list
├── exports_x4/X4{GD,G,D}_full_s*_e25/ X4 embeddings and metric sidecars
├── metrics_x4/{GD,G,D}/               X4 A/C reports
└── metrics_x5/{F6,GD,G,D}/            X5 eligibility reports
```

Repository-portable evidence is stored under `docs/1M/*_runs/`; large matrices and indexes are deliberately excluded from Git. Hash bindings for the large embedding objects are recorded in [`EVIDENCE_SOURCE_MANIFEST.md`](EVIDENCE_SOURCE_MANIFEST.md).

### 16.2 Training artifacts

The recovered F6 runs are under [`F6_runs/RF_full_s{0,1,2}_e25/`](F6_runs/); the nine X4 runs are under [`X4_runs/X4{GD,G,D}_full_s{0,1,2}_e25/`](X4_runs/). A standard run contains at least:

```text
MANIFEST.json
metadata/resolved_config.json
metadata/uncommitted.patch
metrics/train_history.json
stdout/train.log
ckpt/last.pt, best.pt, retained checkpoints, family_vocab.csv
```

The original X4 remote paths were `/u801/x98liu/model_lake/runs/<RUN_ID>/`; delivery archives were `/u801/x98liu/x4_<RUN_ID>.tgz`; downloaded scheduler and execution evidence is in [`X4_runs/x4_execution/`](X4_runs/x4_execution/).[M: [`X4GPU.md`](X4GPU.md)]

### 16.3 Reproduction order

The following order is reconstructed from the current command-line interfaces. Before execution, freeze the data root, HuggingFace credential if required, Python/PyTorch/PyG versions, and hardware:

```bash
# F1 / F1.5
python -m scale1m.hf_crawl --sort createdAt --direction -1 --v2-fields --out <DATA>/candidates_full
python -m scale1m.hf_crawl_datasets --out <DATA>/datasets_full

# F2–F5
python -m scale1m.canonicalize_rf --candidates <DATA>/candidates_full --out <DATA>/rf
python -m scale1m.merge_supervision --rf <DATA>/rf
python -m scale1m.build_ladder_rf --rf <DATA>/rf --out <DATA>/ladder_rf
python -m scale1m.embed_lake_rf --ladder <DATA>/ladder_rf/full_model_ids.parquet --out <DATA>/feats_rf
python -m scale1m.build_graph_rf --ladder <DATA>/ladder_rf --feats <DATA>/feats_rf --rf <DATA>/rf --out <DATA>/graphs/hgraph_rf

# F6 baseline / X4 interventions
python -m scale1m.train_rung --rung full --graph <DATA>/graphs/hgraph_rf --out <RUN> \
  --seed 0 --epochs 25 --family-vocab <DATA>/feats_rf/family_vocab.csv \
  --fanout --sparse-M --contrast-n-neg 256 --chunked-infer 50000 --skip-diagnostics
# X4 GD additionally uses: --lake-gamma 0.5 --global-n-datasets 128

# F7 / F8 / F9
python -m scale1m.export_rf --run <RUN> --stage embed --out <EXPORT>
python -m scale1m.export_rf --run <RUN> --stage metrics --out <EXPORT>
python -m scale1m.export_rf --run <RUN> --stage index --out <EXPORT>
python -m scale1m.export_rf --run <RUN> --stage curve --out <EXPORT>
python -m scale1m.eval_rf --axis a --exports <EXPORT_ROOT> --run-fmt '<RUN_FMT>' --out <METRICS>
python -m scale1m.eval_rf --axis b --exports <EXPORT_ROOT> --run-fmt '<RUN_FMT>' --out <METRICS>
python -m scale1m.eval_rf --axis c --exports <EXPORT_ROOT> --run-fmt '<RUN_FMT>' --out <METRICS>
python -m scale1m.eval_rf --axis d --exports <EXPORT_ROOT> --run-fmt '<RUN_FMT>' --out <METRICS>
python -m scale1m.utility_scorecard --stage meta
python -m scale1m.utility_scorecard --stage score

# X1–X5
python -m scale1m.fast_lever_audit --stage all --seeds 0 1 2 --check
python -m scale1m.eval_rf --axis p
python -m scale1m.eval_rf --axis pinf
python -m scale1m.eval_rf --axis s
python -m scale1m.eval_rf --axis e --exports <EXPORT_ROOT> --run-fmt '<RUN_FMT>' --out <METRICS>
```

The F3–F5 commands above are reproducible entry points reconstructed from the current `argparse` contracts; they are not represented as archived shell history. For F6 and X4, the exact as-run command is the `metadata.command` value in each run's `resolved_config.json`.[U/M]

## 17. Failure, repair, and negative-result ledger

| Stage | Failure or risk | Treatment | Evidentiary consequence |
|---|---|---|---|
| F0/F1 | API change and pagination omission | Descending `createdAt`, immutable `_id` cursor, zero-duplicate/order gates | Supports snapshot enumeration completeness; does not prove that every historical timestamp is a true publication date |
| F1.5 | Basename guessing could inflate mapping coverage | Reject cross-owner and ambiguous matches; do not break ties by downloads | Trades coverage for conservative precision |
| F2 | Mixing lower-is-better, unknown-direction, and reward metrics can reverse gold labels | Explicit semantics table and `gold_eligible` | Query policy is fully aligned only at X5 |
| F2 | RL string-valued metrics cannot be parsed | No special parser was invented | RL not dominating the cap is evidence of limitation, not successful normalization |
| F5 | Pandas interpreted literal `nan/null` family names as missing values | `keep_default_na=False` | Contract checks exposed a real row-order risk |
| F6 | PYTHONPATH, sampling backend, plotting, and environment failures | Remote probes and script repair before formal runs | Failure logs are environment evidence, not model results |
| F6 | Git binding is incomplete | File hashes plus checkpoint/run bindings | A commit hash alone cannot reproduce the run |
| F7 | Full-graph forward propagation leaks held-out information | Held-out `z_*_eval` plus inequality gates | Full embeddings can inflate recovery by as much as 2.4× |
| F8 | Per-query “multithreaded QPS” was unstable | Replaced with batch-query throughput | The first QPS result is discarded |
| F8/X1 | CSLS appeared positive on seed 0 | Three-seed rerun produced a −41% mean and CSLS was removed | A material negative result that must remain visible |
| X2 | Graph task id is constant zero | Sidecar reads the real task parquet | Repairs the service prior, not the training graph |
| X3 | A nominal node-level protocol would leak through existing embeddings | Change only service-readable edges and label lower/upper-bound bias | The second protocol is not the primary protocol |
| X4 | H200 queueing, staging/CUDA/`srun` failures, preemption, and TIMEOUT | Expanded GPU pool, probes, `X4_DIRECT=1`, same-RUN-ID resume | All nine runs remain valid; hardware heterogeneity must be disclosed |
| X4 | No-op resume overwrote manifest cost fields | Separate full training segments from no-op gates; exclude starred costs | Prevents false 5.25 GB peak-memory claims |
| X5 | Untrustworthy queries were mixed into the headline metric | Independent E axis reports old, eligible, and excluded sets | A metric-policy correction, not a model gain |

## 18. Evidence gaps that remain open

The following items must not be described as completed or proved:

1. **X4 P/S/B/D axes have not been rerun.** There is no unified X4+task/sibling result, and F8 HNSW latency is not an X4 measurement.
2. **X5 has not been integrated into export.** The A rows in `EXPORT_MANIFEST` and `F8_REPORT` retain the old query policy; X5 exists in separate JSON files.
3. **X5 snapshot-only has not been recomputed.** The corrected policy is presently available only for the all-candidate condition.
4. **The 500K training-scale point has not run.** Only a retrieval candidate-count curve with fixed representations exists.
5. **F8 performance strata for query depth ≥3/5/10/20 and supervision source are absent from `F8_REPORT.json`.** F2 counts depth and source, but those counts are not stratified performance.
6. **`displacement_quality` has not been redesigned.** Its current definition overlaps the training label.
7. **F9's blinded third layer has not run.** The temporal protocol only froze the t0 list; no future-label outcome exists.
8. **No real model execution has evaluated unlabeled recommendations.** Recorded-gold recovery cannot be restated as recommendation accuracy.
9. **Discrete dataset task features are degenerate inside the training graph.** X2 repairs only the service sidecar; the F6/X4 encoder still sees a shared single-row bias.
10. **Data quality remains incomplete.** Model size is missing for 71.929%; 79.03% of dataset nodes lack a real card; historical `curated` weights lack the original metric name; 12,680 historical models are absent from the snapshot.
11. **Statistical power is limited.** There are only three split seeds. Even three same-direction GD results give a minimum simple sign-test probability of 1/8; causal support comes from the preregistered direction and magnitude gates, not a significance test.
12. **The repository is not a clean experimental release.** HEAD predates most of the pipeline; later code and evidence are dirty or untracked. A read-only release/tag and full-file manifest are required before paper freeze.
13. **The root test entry point has a namespace collision, but every suite was rerun independently.** In the repository `.venv` (Python 3.13.1, pytest 9.1.1), `scale1m/tests` produced 262 passed and `stage2TrainGraphSAGE/tests` 67 passed, exactly reproducing X5's archived 329; `stage1BuildTransferGraph/tests` added 8 passed and `stage3HNSW/tests` 7 passed, for 344 passed overall. Collecting once from the repository root raises 18 collection errors because several top-level directories expose a package named `tests`; stage1 also requires the repository parent on `PYTHONPATH`. These are collection/import conditions, not assertion failures. This audit also parsed 92 JSON files and AST-parsed 28 report-referenced/principal Python modules.[M/U]

## 19. Documentation conflicts and evidentiary rulings

| Conflict | Weaker evidence | Deciding evidence | Ruling used here |
|---|---|---|---|
| GraphSAGE layer count | `1Mplan.md` §2.1 and the stale `model.py` header say two layers | F6 and all nine X4 `resolved_config.num_layers=1` | **One layer as run** |
| Snapshot-only candidate universe | Early planning language says 3,003,759 | F2/F3/F6/F7/X4 artifacts all bind N=3,016,439 | **Snapshot prefix plus 12,680 historical models** |
| Number of historical-only models | Pre-merge estimate 16,713 | F2 merge reports `historical_only=12,680` | **12,680** |
| True GD seed-1 continuation job | A stale process sentence pointed to 1522415 | Logs and correction: 1522429 executed epochs 10–24; 1522415 was a no-op | **1522429 trained; 1522415 gated only** |
| Early GPU type on watgpu308 | Probe 1522377 received RTX A6000 | Both TIMEOUT segments lack a GPU name; another allocation under the same GRES, 1522415, records L40S | **GPU type unarchived** |
| X4 sbatch SHA | Initial `a0b2ab…` | As-run file and `code_sha256.txt` contain `f1237a…` | **f1237a…** |
| D seed-2 43,030 MiB | Operator observation | Delivery artifacts contain no nvidia-smi output | **Unarchived observation, not a reproducible measurement** |
| F9 availability 0.7099 | Seed-0 point estimate | X2 three seeds are 0.7099/0.6273/0.5729 | Cite as a seed-0 point or include the range |
| Current `gold@10` | F8 0.0598 or old-policy X4 0.1429 | X5 eligible-query result | **F6 0.0625; X4 GD 0.1427** |

## 20. Claim–evidence library for the paper

### 20.1 Statements currently supported

1. **Scale and traceability.** “The system trains over 3,016,439 model nodes, 18,729 dataset nodes, and 247,803 supervision edges merged from six sources. Of the model nodes, 3,003,759 come from the 2026-08-18 HuggingFace snapshot and 12,680 are historical-only appendages.” Cite the F1/F2/F3 reports and ladder/graph bindings.
2. **Current effectiveness.** “Under root-aware holdout, X5 `gold_eligible`, and the all-candidate protocol, X4 GD obtains three-seed `gold@10` values 0.1355/0.1417/0.1508, mean 0.1427, versus 0.0625 for F6—a 2.28× ratio.” Cite the X5 F6/GD JSON files.
3. **Mechanism decomposition.** “G alone improves all three seeds to 0.1197; D alone reaches 0.0724 but changes direction by seed; GD reaches 0.1427. Proposal distribution is the primary contributor, while coverage supplies a conditional combined increment.” Cite seedwise X4 old-policy changes together with the X5 current summary; describe the mechanism as a contributor, not a uniquely proved cause.
4. **Candidate-scale robustness.** “With F6 representations and supervision rows forcibly retained, expanding retrieval candidates from 100K to 3.016M changes `gold@10` only from 0.0706 to 0.0700.” Cite F8 retrieval curve and retain both conditions.
5. **ANN system result.** “For F6 embeddings, full-lake HNSW obtains `recall@50=0.9989`; at `ef=50`, local p50 is 0.0211 ms, 2,714× faster than same-machine exhaustive search.” Cite F8 B and disclose local hardware and the F6 representation.
6. **Collapse and mitigation.** “Under F6, cold/frozen embeddings are approximately 98.5% collapsed with effective dimension near 3. X4 GD raises frozen effective dimension to 8 and lowers cosine from 0.908 to 0.778, but remains far below 128 dimensions.” Cite F8 and X4 C.
7. **Prior branch.** “On F6 representations, the root-aware task prior moves `gold@10` from 0.0598 to 0.3009 while median parameter count rises from 2.42B to 7.57B. In the second protocol, sibling+task(A) reaches 0.3536 and median parameters 12.11B.” Cite X2/X3 and keep this branch separate from X4.
8. **Metric boundary.** “`gold@10` measures recovery of historical records, not recommendation accuracy; for F6 seed 0, 95.2% of top-ten query–model pairs have no record for the target query.” Cite F9.

### 20.2 Statements not currently supported

- “The final system's `gold@10` is 0.35/0.43.” Those values use F6 representations under a second protocol or an upper bound, not a unified X4+X5 system.
- “X4 retains a 2,714× HNSW speedup.” X4 indexes and the B axis have not run.
- “14.27% of unlabeled recommendations are truly best.” `gold@10` does not have this meaning.
- “The system is a two-layer GraphSAGE.” The as-run system has one layer.
- “All X4 runs used the same L40S/A6000/H200 GPU.” Hardware was heterogeneous, and two early segments have no archived type.
- “D is independently effective.” Its three-seed direction is inconsistent.
- “Representation collapse is solved.” Effective dimension is still only about 8/128.
- “F8 completed performance stratification by source and query depth.” Those tables were not archived.

## 21. Closeout priorities that preserve the nine X4 runs

If the paper evidence freeze permits artifact computation but no retraining, the priority order is:

1. Build split-specific and full sidecars for X4 GD/G/D, rerun P/S/E and the matched second-layer utility table, and retain the F6 branch as the control.
2. Integrate X5 eligibility into export metrics as a new version rather than overwriting the old `EXPORT_MANIFEST`; add snapshot-only or explicitly retire that column.
3. Use the existing candidate/source fields to compute the F8 depth ≥3/5/10/20 and source-stratified A tables. This is evaluation recomputation, not training.
4. If the paper needs a system-level claim for X4, build a full-lake GD HNSW index and rerun B/D under the same-machine F8 protocol; do not relabel F8 measurements.
5. Create a clean release commit/tag, lock the environment, and emit a complete SHA manifest; preserve a tarball of the current dirty tree as transitional evidence.
6. If new GPU experiments become permissible, consider more seeds, a D≈44 alignment arm, and a genuine node-level retrain. These would be new experiments and must not rewrite the existing record.

This report closes the evidence chain from raw API enumeration through supervision normalization, features and graph construction, training, held-out export, ANN evaluation, utility interpretation, and X1–X5 diagnosis/intervention/query-policy correction. All headline conclusions use the current X-series state; the F series is retained only as a traceable baseline.
