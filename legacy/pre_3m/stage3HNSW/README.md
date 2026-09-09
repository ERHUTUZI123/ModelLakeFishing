# Stage 3 — HNSW Retrieval Layer

Plan: [`docs/STAGE3_HNSW_IMPLEMENTATION_PLAN.md`](../../../docs/STAGE3_HNSW_IMPLEMENTATION_PLAN.md).
This README records the Phase-0 environment/input freeze and the design decisions D1–D4.

## Design decisions (fixed 2026-07-07, see plan §1)

- **D1 — Backend:** hnswlib is the production backend (incremental `add_items` +
  `mark_deleted`); faiss `IndexHNSWFlat` is retained as an independent fidelity
  cross-check only.
- **D2 — Served embeddings (amended 2026-07-08, user decision): G2 ONLY.**
  Primary = `hf1000d_2000m` graph + **G2** checkpoint (deliberate override of
  the Stage-2 promotion rule, which promoted G1 — recorded in the ckpt repro
  metadata). G1/P6_dm10 are NOT indexed or reported in Stage 3; their frozen
  Phase-0 checkpoints stay on disk for future audits only. The 306-model
  diverse candidate is the mechanism-smoke checkpoint. Stage-3 code remains
  checkpoint-agnostic.
- **D3 — Metric:** cosine via L2-normalized inner product; normalization happens
  once at export time, index built in `space="cosine"` anyway.
- **D4 — Identity contract:** index label = `mappedID` = row of exported `z_m`
  = row of the `unique_model_id` snapshot. Every index binds to its manifest
  (graph sha256 ⊕ ckpt sha256 ⊕ vocab hashes ⊕ dim ⊕ code version).
  **An index file without its manifest is invalid by definition.**

## Phase 0 — Environment freeze (done 2026-07-08)

### Environment

| component | version | note |
|---|---|---|
| Python | 3.13.1 | project venv `ModelLakeFishing/.venv` |
| torch | 2.12.0+cu126 | CUDA available |
| torch-geometric | 2.7.0 | |
| numpy | 2.4.6 | |
| **hnswlib** | **0.8.0** | sdist-only on PyPI (no cp313 wheel); compiled locally with MSVC 14.44 (VS Build Tools 2022, installed 2026-07-08 for this purpose) |
| **faiss-cpu** | **1.14.3** | was already in the venv (used by `top1_eval`), now pinned in `requirements.txt` |

Both packages are appended to `requirements.txt` (kept UTF-16 LE, CRLF).
Backend smoke test: 500×64 unit vectors, hnswlib and faiss both recall@10 = 1.000
vs exact dot.

### Frozen inputs (sha256)

Graphs (Stage 1):

| file | sha256 |
|---|---|
| `stage1BuildTransferGraph/hgraph_hf1000d_2000m_xm0_xd0.pt` (primary, 2,000 models) | `25ae333bca0c00b6ae2fe4047115a02a198744284bdcd632419d02b2b16e05c0` |
| `stage1BuildTransferGraph/hgraph_diverse_xd0.pt` (smoke, 306 models) | `e53dae9928cfeb69446ebe5b1aebf3ec49d365cb05d69a4c39f2404813abbfba` |

Checkpoints (Stage 2). The G-phase/T0 runs recorded per-split `state_dict_sha256`
but never persisted weights, so G2/G1/P6_dm10 were re-trained once via
[`phase0_freeze.py`](phase0_freeze.py) (same code path as
`top1_baselines.run_configs`: deduped graph, `similar_to` surgery, fixed
split_seed=0, init_seed=0, 25 epochs, CUDA) — exactly the bounded re-train the
plan budgeted for:

| file | sha256 |
|---|---|
| `stage2TrainGraphSAGE/artifacts/ablation/top1/ckpt/G2_s0_i0.pt` (**serving primary**) | `3eade9611c3241f8bc46311f7341f4da7b69e59b5c69ff7bf4add3addf810c5f` |
| `stage2TrainGraphSAGE/artifacts/ablation/top1/ckpt/G1_s0_i0.pt` (comparison) | `aac16c0983594b52614aafa0b69dbdbaf4eb16bef5e68aeaf0353da57591f5a0` |
| `stage2TrainGraphSAGE/artifacts/ablation/top1/ckpt/P6_dm10_s0_i0.pt` (comparison) | `d4f00985cb218a686ac8ddf3f546fdb9f396bbba7ca2f704d8eda024c83e47d1` |
| `stage2TrainGraphSAGE/artifacts/stage2_diverse_xd0_candidate.pt` (smoke) | `3ceeb500ae6c9a3fbd01ad0046d2b1b822112f7d5f0135ba5a6386c22fb80da5` |

Vocab CSVs (the identity credential bound to each checkpoint):

| file | sha256 |
|---|---|
| `top1/ckpt/{G2,G1,P6_dm10}_s0_i0.family_vocab.csv` (identical) | `4b76d3c82c763c4d847143db757b273aa68135735b55d9efbf8df0f6a61a3b0f` |
| `top1/ckpt/{G2,G1,P6_dm10}_s0_i0.task_type_vocab.csv` (identical) | `9012cb89440a5ff57562de7ddf18c97863a02a9f8e55c76ba4d3ce271d817e84` |
| `artifacts/stage2_diverse_xd0_candidate.family_vocab.csv` | `f4e7d560d6f89cbf744c903d883ca34868819cfdf42b8cddaaf94704eb86045f` |
| `artifacts/stage2_diverse_xd0_candidate.task_type_vocab.csv` | `14a78f25189623b73a28fbd1c1664bb8c856bd9c3954b47ce96bd000978147bf` |

### Re-train verification (honest provenance)

Recorded in `artifacts/phase0_freeze_report.json` and inside each checkpoint's
`repro` metadata:

- **state_dict sha256 does NOT bit-match the recorded G/T0 values** for any of
  the three (CUDA training is not bit-reproducible across runs). Expected and
  reported, not hidden.
- **The decisive check passes:** replaying the five-metric evaluation on the
  recorded split 0 reproduces the recorded aggregates **exactly (delta = 0 on
  every key)** for G1 and P6_dm10; for G2 every key matches except
  `median_gold_rank` (142 vs recorded 141 — one gold rank moved by 1 under
  training nondeterminism). Reference split-0 numbers: G2 hit@1 0.4872 /
  gold@10 0.0513, G1 hit@1 0.3333 / gold@10 0.0769, P6_dm10 hit@1 0.3333 /
  gold@10 0.0256.
- Checkpoint save/load roundtrip embeddings `allclose` (atol 1e-6) and
  family-vocab binding verified for all three.

### Gate

| check | status |
|---|---|
| hnswlib imports + functional | PASS (0.8.0, recall 1.000 smoke) |
| faiss imports + functional | PASS (1.14.3, recall 1.000 smoke) |
| both pinned in requirements.txt (UTF-16 preserved) | PASS |
| input hashes recorded | PASS (this file) |
| G2 weights on disk | PASS (`G2_s0_i0.pt`, roundtrip OK) |
| G1 / P6_dm10 weights on disk | PASS (roundtrip OK) |

**Phase 0 gate: PASS.**

## Phase 1 — Embedding export (done 2026-07-08)

[`export_embeddings.py`](export_embeddings.py) turns `(graph.pt, ckpt.pt)` into
the serving triple under `artifacts/exports/<name>/`: `z_m.npy` / `z_d.npy`
(float32, L2-normalized, row i == mappedID i), `model_ids.csv`,
`dataset_ids.csv`, `model_meta.parquet` (size_bucket_id/family_id sidecar for
the Stage-4 reranker), `manifest.json` (D4 identity contract, written last).

Contract enforcement at the export boundary:

- CPU eval forward computed **twice, bitwise-equal required** (determinism);
- the message graph is rebuilt exactly as the ckpt was trained —
  `dedup_trained_on` + `similar_to` surgery are read from the ckpt's repro
  metadata, and the ckpt's recorded graph sha256 must match the `--graph` file;
- id snapshots validated as the permutation 0..N-1; 20-row spot check re-reads
  the id/meta files from disk against the graph's node attributes.

Serving semantics note: exported z is computed on the **full** (deduped,
surgered) graph — all trained_on edges act as messages — which is the
deployment-time geometry, intentionally different from the eval-time
`test_data` message graphs used in the Stage-2 metric records.

Exports produced (per the 2026-07-08 amendment, G2 only + smoke):

| export | graph | ckpt | contents |
|---|---|---|---|
| `artifacts/exports/hf1000d_G2/` | hf1000d_2000m | `G2_s0_i0.pt` | 2,000 models × dim 128, 362 datasets; surgery dedup + topk_unweighted(k=10) |
| `artifacts/exports/diverse_candidate/` | diverse_xd0 | `stage2_diverse_xd0_candidate.pt` | 306 models × dim 128, 24 datasets; no surgery (dense) |

**Gate G-A: PASS** — `tests/test_export_contract.py` green on both graphs
(determinism / ids / unit norms / manifest schema+hash truthfulness).

## Phase 2 — Index build & persistence (done 2026-07-08)

[`build_index.py`](build_index.py) builds the hnswlib cosine index
(labels == mappedID, defaults `M=16, ef_construction=200`, **single-thread
build with fixed seed → `index.bin` is bit-reproducible**), persists
`index.bin` + `index_manifest.json` (export manifest sha256 ⊕ z_m sha256 ⊕
graph/ckpt hashes ⊕ build params ⊕ index sha256 ⊕ gate results), and refuses
to hand out an index that fails its gates at build time. `load_index()` is the
only sanctioned loading path: it raises `ManifestMismatch` on a tampered
`index.bin`, a drifted export (`manifest.json` or `z_m.npy`), or a missing
manifest — an index file without its matching manifest is invalid.

Production indexes (`artifacts/indexes/<name>/`):

| index | elements | size | build | fidelity vs exact dot (efS=200, all z_d) |
|---|---|---|---|---|
| `hf1000d_G2` | 2,000 × dim 128 | 1.3 MB | 0.08 s | recall@{1,10,50,100} = **1.000** |
| `diverse_candidate` | 306 × dim 128 | 0.2 MB | 0.01 s | recall@{1,10,50,100} = **1.000** |

**Gates G-B + G-C: PASS** — `tests/test_index_roundtrip.py` green on both
scales: save/load answers bitwise-identical (labels + distances), fidelity
1.000 (hard gate ≥ 0.99@10), label set exactly 0..N-1, all three
manifest-mismatch refusal cases raise, rebuild determinism bitwise.

## Phase 3 — Query paths (done 2026-07-10)

[`query.py`](query.py) — the two ways a dataset arrives:

- **warm**: z_d row exists in the bound export → look up, query (replay path);
- **cold**: deployment path. All edges incident to the dataset are dropped
  (`cold_graph.drop_cold_dataset_edges` + any other dataset-incident edge type,
  e.g. the diverse graph's `transfer_to`), the ckpt's `similar_to` surgery is
  re-applied among remaining datasets, **deployable** incoming `similar_to`
  edges are attached from the e_domain slice only
  (`cold_graph.cold_incoming_similar_to`; dim read from `xd0_meta` — 1536
  diverse / 3840 hf1000d), one inductive forward yields z_d. Query-only
  insertion is asserted (indexed z_m bitwise-unaffected). Graph/ckpt/surgery
  are read from the export manifest, hashes re-verified before use.

Every answer carries provenance: mode, ef_search, index manifest hash, and the
cold-edge composition.

**Gate G-D: PASS** — `tests/test_query_paths.py` green on both indexes: warm
ANN top-K == exact-dot top-K (24/24 and 50-sample datasets, scores agree at
float32); cold embed bitwise-deterministic; cold ANN/exact top-50 overlap
**1.000** (gate ≥ 0.95); provenance hash matches disk. Demo: treating
`amazon_massive_intent` as cold reproduces 9/10 of its warm top-10 on the G2
index. Full outputs archived in `weeks/week7_HNSW/HNSW.md`.
Next: Phase 4 (`bench_params.py`) and Phase 5 (`update_index.py`), independent
after Phase 2.
