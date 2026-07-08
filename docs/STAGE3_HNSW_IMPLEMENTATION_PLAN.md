# Stage 3 — HNSW Retrieval Layer: Implementation Plan

**Date:** 2026-07-07
**Target folder:** `stage3HNSW/`
**Status:** plan only — no code written yet.

---

## 0. Where we are (proposal recap and current position)

### 0.1 The proposal in one paragraph

(`weeks/week1_updated_proposal/model_Lens_reading_notes.tex`) We beat ModelLens at
million-scale model lakes with a **two-stage serving architecture**:

```
query (d*, t, μ)
  → [Stage 3] HNSW retrieval over GraphSAGE embeddings   O(log N)
  → [Stage 4] metric-aware reranker over K candidates    O(K)
```

versus ModelLens's cross-feature MLP that must score all N models — O(N), not
ANN-indexable. Our supporting structure is the TransferGraph (4 edge types:
dataset–dataset similarity, model–dataset performance, model–dataset
transferability, model–model lineage) encoded by an **inductive** heterogeneous
GraphSAGE trained with `L_perf + L_contrast` + edge dropout. Inductiveness is the
contract that makes Stage 3 valuable: a new model or a new dataset gets its
embedding from one forward pass — no retraining, no ID table.

### 0.2 What is already done

| Stage | Status | Key artifacts |
|---|---|---|
| **Stage 1 — build TransferGraph** | DONE (several graph variants) | `hgraph_diverse_xd0.pt` (306 models / 24 datasets, production per `artifacts/PRODUCTION.md`); `hgraph_hf1000d_2000m_xm0_xd0.pt` (2,000 models / ~365 datasets, the Top-1/global-retrieval benchmark graph); `hgraph_hf_effective_2000m_v2_xm0_xd0.pt` (v2 provenance-reconciled) |
| **Stage 2 — train GraphSAGE** | DONE + evaluated | `stage2_diverse_xd0_candidate.pt` (production ckpt); ablation checkpoints under `stage2TrainGraphSAGE/artifacts/ablation/`; five-metric evaluator `top1_eval.py`; results in `artifacts/ablation/top1/TOP1_BASELINES.md`, `TOP1_PHASE_REPORT.md` |
| **Stage 3 — HNSW retrieval** | **empty folder** — this plan | `stage3HNSW/` |
| Stage 4 — metric-aware reranker | not started | — |

Stage-2 evaluation facts that Stage 3 must be built on:

1. **The interface contract holds.** Checkpoints roundtrip exactly; `z_m` row `i`
   ↔ `mappedID i`; family/task_type vocabs bind to checkpoints (orphan-row guard).
2. **Best clean configs (post `dedup_trained_on`, leakage = 0):**
   - **G1** (incompatible global negatives, λ=1.0) — PROMOTED for global
     retrieval: full2k_gold@10 0.010 → **0.054** (10.8× random), first nonzero
     gold@1, median gold rank 246 → 195.
   - **G2** (G1 + known-low + hard negatives) — hit@1 0.400, gold@10 0.046,
     best median gold rank 155; rejected under the promotion rule on a single
     zero split (metric granularity 1/35), flagged "investigate". **Chosen as
     the Stage-3 primary serving checkpoint by decision 2026-07-07** (see D2).
   - **P6_dm10** — best local ordering: observed hit@1 0.406, top3 0.737,
     regret@1 0.044.
3. **HNSW fidelity is already a solved *quality* question at 2K scale:**
   exact-vs-HNSW recall = **1.000** at every K in `top1_eval` (faiss
   `IndexHNSWFlat`) and in `eval_harness.dataset_to_model_hnsw_recall` (hnswlib).
   The serving score is frozen as `s(d,m) = ⟨normalize(z_d), normalize(z_m)⟩`.
4. **The open weakness is embedding geometry, NOT the index:** whole-lake gold
   survival is weak (best gold@10 = 0.054), and the hubness audit found **57
   models filling all 390 top-10 slots**, several with empty train-visible task
   profiles. No index parameter can fix this — it is Stage-2 territory
   (whole-model masking / hub penalty are the identified next levers).

### 0.3 What Stage 3 therefore is — and is not

Stage 3 is the **systems / serving layer**, not another quality experiment:

- **IS:** a persistent, versioned, incrementally updatable HNSW index over `z_m`;
  a query path that embeds a cold dataset inductively and retrieves top-K; a
  parameter study (recall–latency trade-off, hub-stratified); a scale benchmark
  substantiating the O(log N) claim; a frozen retrieval API for Stage 4.
- **IS NOT:** an attempt to raise gold@10. Retrieval quality equals embedding
  quality here (ANN fidelity is already 1.000); semantic improvements stay in the
  Stage-2 track and Stage 3 simply re-indexes better checkpoints when they land.

---

## 1. Design decisions (fixed up front)

### D1. ANN backend: **hnswlib** primary, faiss cross-check

| | hnswlib | faiss `IndexHNSWFlat` |
|---|---|---|
| already used in repo | yes (`eval_harness.py`) | yes (`top1_eval.py`) |
| incremental `add_items` after build | **yes** | yes (no deletion) |
| deletion | `mark_deleted` (tombstone) | no |
| cosine space | native | via L2-normalize + IP |
| Windows pip install | clean wheel | `faiss-cpu` wheel (CPU ok) |

Incremental insert + deletion is the proposal's "inductive incremental update"
selling point, so **hnswlib is the production backend**; faiss is retained as an
independent fidelity cross-check in benchmarks. Both consume the same exported
`.npy` matrices, so the choice is reversible.

### D2. Which embeddings to serve

- **Primary benchmark lake:** `hf1000d_2000m` graph + **G2** checkpoint
  (incompatible + known-low + hard global negatives) — **chosen by decision
  2026-07-07**. Note the provenance honestly: under the Stage-2 promotion rule
  G2 was *rejected* (one zero split at gold@10 granularity 1/35; gate detail in
  `TOP1_PHASE_REPORT.md`), while G1 was the promoted config. Serving G2 is a
  deliberate override; Phase 4/8 must therefore report G2's fidelity and
  hub-stratified numbers side by side with G1 so the choice stays auditable.
  **G1 and P6_dm10 are indexed alongside for comparison.**
- **Mechanism smoke / small tests:** production `stage2_diverse_xd0_candidate.pt`
  (306 models) — fast, contract-verified.
- Stage 3 code must be **checkpoint-agnostic**: everything flows from an exported
  `(z_m, ids, manifest)` triple, never from a hardcoded graph.

### D3. Metric and normalization

Cosine via L2-normalized inner product, identical to the evaluator's serving
score. **Normalization happens once at export time** and the manifest records
`normalized: true`; the index is built in `space="cosine"` anyway (belt and
braces — cosine of normalized vectors = IP, so both backends agree).

### D4. Identity contract (the iron rule, inherited)

Index label = `mappedID` = row index of exported `z_m` = row of
`unique_model_id` snapshot. The manifest binds: graph sha256, checkpoint sha256,
`family_vocab.csv` hash, embed dim, encoder names, export code version.
**An index file without its manifest is invalid by definition.**

---

## 2. Deliverables & folder layout

```
stage3HNSW/
├── README.md                  # decisions D1–D4 + quickstart
├── export_embeddings.py       # Phase 1: ckpt+graph → z_m/z_d + ids + manifest
├── build_index.py             # Phase 2: npy → .bin index + sidecar manifest
├── query.py                   # Phase 3: warm/cold query → top-K (CLI + API)
├── update_index.py            # Phase 5: incremental insert / delete / rebuild policy
├── bench_params.py            # Phase 4: M × efC × efS recall/latency sweep
├── bench_scale.py             # Phase 6: synthetic 20K/100K/1M scale benchmark
├── retrieval_service.py       # Phase 7: ModelRetriever class (Stage-4 handoff)
├── tests/
│   ├── test_export_contract.py
│   ├── test_index_roundtrip.py
│   ├── test_query_paths.py
│   ├── test_incremental.py
│   └── test_service_api.py
└── artifacts/
    ├── exports/<name>/        # z_m.npy, z_d.npy, model_ids.csv, dataset_ids.csv, manifest.json
    ├── indexes/<name>/        # index.bin + index_manifest.json
    ├── bench/                 # param_sweep.{json,md}, scale_bench.{json,md}, plots
    └── STAGE3_REPORT.md       # Phase 8
```

---

## 3. Phased plan

### Phase 0 — Environment & input freeze (½ day)

1. `pip install hnswlib faiss-cpu` into the project venv; record exact versions.
   (Neither is in `requirements.txt` yet — add both, and note `requirements.txt`
   is currently UTF-16; keep encoding consistent when appending.)
2. Freeze the two input checkpoints (D2): record sha256 of graph `.pt`, ckpt
   `.pt`, vocab CSVs into `stage3HNSW/README.md`.
3. Confirm the G2 checkpoint file exists on disk (the G-phase runs saved
   per-config JSONs; if the G2 *weights* were not persisted, re-run its training
   once via `ablation.train_eval_one` with the recorded seeds and save the ckpt —
   budget for this). Same check for G1 and P6_dm10 (comparison indexes).

**Gate:** both backends import; input hashes recorded; G2 weights on disk.

### Phase 1 — Embedding export with contract verification (1 day)

`export_embeddings.py --graph <pt> --ckpt <pt> --out artifacts/exports/<name>/`

1. Load graph + checkpoint exactly as `visualize_trained.py` / `top1_*` do
   (reuse `model.py` loaders — import from `stage2TrainGraphSAGE`, do not copy).
2. Full-graph forward in eval mode → `z_m`, `z_d`; L2-normalize; cast float32.
3. Write `z_m.npy` (row order = mappedID), `model_ids.csv`
   (mappedID, unique_model_id), same for datasets, plus `manifest.json` (D4
   fields + numpy/torch versions + timestamp).
4. **Row-order verification (the Stage-2 killer risk, now at the export
   boundary):** recompute `z` twice in one process (determinism check), and
   verify `model_ids.csv` against the graph's stored id mapping; sample 20
   random rows and confirm the vocab-bound features (size bucket, family id)
   match the graph's per-node attributes.

**Tests:** export determinism; ids alignment; normalization (‖row‖₂ = 1);
manifest completeness (schema-validated).
**Gate:** `test_export_contract.py` green on both diverse (306) and hf1000d (2,000).

### Phase 2 — Index build & persistence (1 day)

`build_index.py --export artifacts/exports/<name>/ --M 16 --efc 200`

1. Build hnswlib cosine index, labels = mappedID; defaults `M=16, ef_construction=200`
   (the values already validated in `eval_harness`), overridable.
2. Persist `index.bin` + `index_manifest.json` = export manifest ⊕ build params ⊕
   index sha256 ⊕ element count.
3. Load-roundtrip: reload from disk, re-run a fixed query batch → results must be
   **identical** to pre-save.
4. Fidelity gate vs exact dot (reuse `top1_eval`'s exact ranking as oracle):
   recall@{1,10,50,100} with `ef_search=200` — expected 1.000 at 2K.

**Tests:** roundtrip identity; fidelity ≥ 0.99 @10 (hard gate; expect 1.000);
manifest-mismatch refusal (loading an index whose manifest hash ≠ export hash
must raise).
**Gate:** all green at 306 and 2,000 scale.

### Phase 3 — Query paths (1–2 days)

`query.py --index <dir> --dataset <id|features> --k 50`

Two modes, matching how a dataset can arrive:

- **(a) Warm dataset** — its `z_d` row already exists in the export: look up and
  query. This is the evaluation replay path.
- **(b) Cold dataset** — the deployment path from the proposal
  ("target dataset embedded by one inductive forward pass"): take deployable
  features (`xd0` multi-view embedding + task_type/n_class/arity ids), attach
  deployable `similar_to` edges only, run the inductive forward using the
  machinery already built in `stage2TrainGraphSAGE/cold_graph.py` /
  `cold_dataset_split.py`, normalize, query. **Reuse, don't reimplement** — the
  cold-dataset guide (`docs/COLD_DATASET_FULL_VERSION_GUIDE.md`) already defines
  what "deployable" means.

Output: ranked `(rank, model_unique_id, cosine_score)` + query provenance
(which mode, ef_search used, index manifest hash) as JSON/CLI table.

**Tests:** warm query == exact-dot top-K on served embeddings; cold path
determinism; cold query of a dataset that was *held out at Stage 2* returns
plausible neighbors (smoke: its top-50 overlaps the exact-dot top-50 of its
cold embedding ≥ 95%).
**Gate:** both modes work on the diverse graph end-to-end.

### Phase 4 — Parameter study: recall–latency frontier (1–2 days)

`bench_params.py` sweeps on the 2,000-model export:

- Grid: `M ∈ {8, 16, 32, 48}` × `ef_construction ∈ {100, 200, 400}` ×
  `ef_search ∈ {16, 32, 64, 128, 256, 512}`.
- Measured per cell: recall@{1,10,50,100} vs exact dot; build time; index bytes;
  query latency p50/p95 (single query, and batch-100).
- **Hub-stratified recall** — the proposal's own "Potential Issue §
  hub-dominated graphs": split queries into near-hub / away-hub using the
  existing `model_to_model_hnsw_recall` near-hub logic + `top1_hubness.py`'s hub
  list; report recall separately. If near-hub recall lags at 2K already, that is
  the earliest possible warning for 1M.
- Cross-check 3 representative cells with faiss `IndexHNSWFlat` (fidelity
  agreement between backends).

**Deliverable:** `artifacts/bench/param_sweep.md` with the frontier plot and a
**chosen production triple** (expected: M=16, efC=200, efS≈64 saturate at 2K —
the point is the methodology, which reruns unchanged at 47K/1M).

### Phase 5 — Incremental update path (1–2 days)

The proposal's "inductive incremental update" claim, made executable:

1. **Insert:** hold out 10% of models from the initial build; embed each via the
   inductive forward (zero-shot path per `CLAUDE.md` §Inference: missing param →
   unknown bucket, unseen family → Other), `add_items`, then verify (i) each
   inserted model is retrievable as its own nearest neighbor, (ii) recall@10 on a
   fixed query set does not drop by >1pt vs a fresh full rebuild.
2. **Delete:** `mark_deleted` tombstones; verify deleted models never appear in
   results; measure recall drift as tombstone fraction grows.
3. **Rebuild policy (documented, tested):** rebuild when tombstones >20% of
   elements or recall gate fails; rebuild = rerun `build_index.py` on the
   current export (cheap by design).
4. Record per-insert latency — this is the number that backs "cost independent
   of lake size" in the proposal.

**Tests:** insert-then-retrieve; post-insert recall gate; deletion exclusion;
capacity growth (`resize_index`) handled.

### Phase 6 — Scale & latency benchmark: the O(log N) evidence (1–2 days)

We only have 2K real models, so scale is **synthetic and must say so**:

1. Generate lakes of N ∈ {2K (real), 20K, 100K, 1M} by cloning real `z_m` rows
   with small Gaussian perturbation (σ ≈ 0.05 before renormalizing), preserving
   the real cluster/hub geometry rather than sampling uniform noise. Honest
   caveat recorded: synthetic derivatives *underestimate* hub crowding effects.
2. For each N: brute-force exact dot (numpy matmul) vs HNSW query time
   (p50/p95, single-thread CPU), recall@10 vs exact, index memory, build time.
3. Deliver the log-log latency plot (brute force ~O(N) line vs HNSW ~flat/log
   line) — the concrete artifact behind the proposal's "10k users, 1 GPU, 15
   seconds" argument, with measured numbers replacing the extrapolation.
4. Report QPS at K=50 for batch queries at each N.

**Gate (targets, CPU):** recall@10 ≥ 0.95 at 1M with tuned efS; single query
p95 ≤ 10 ms at 1M; the crossover-vs-brute-force plot exists.

### Phase 7 — Retrieval service API: the Stage-4 handoff (1 day)

`retrieval_service.py` — a single class freezing what Stage 4 (reranker)
consumes:

```python
r = ModelRetriever.load("artifacts/indexes/hf1000d_G2")   # verifies manifests
out = r.retrieve(z_query, k=50, ef_search=64)
# out.candidates: [(mapped_id, unique_model_id, score), ...]
# out.z_m: (k, dim) float32   — reranker input [z_m ‖ z_d* ‖ e_size ‖ e_fam ‖ e_t ‖ e_μ]
# out.model_meta: size_bucket_id / family_id per candidate (from the export sidecar)
# out.provenance: index manifest hash, efS, timestamp
```

Note the reranker needs `e_size`/`e_fam` *ids* per candidate — so Phase 1's
export must also persist the per-model discrete ids sidecar
(`model_meta.parquet`). Add it to the Phase-1 checklist.

**Tests:** API schema stability; manifest verification on load; retrieve() ==
query.py results.
**Deliverable:** `STAGE4_INPUT_CONTRACT.md` section inside the Stage-3 report.

### Phase 8 — Report & acceptance (½ day)

`artifacts/STAGE3_REPORT.md` consolidating:

- gates table (every phase's gate, pass/fail, numbers),
- production index choice (checkpoint, params, hashes),
- hub-stratified findings and their implication for the Stage-2 backlog,
- scale plot + honest synthetic caveat,
- Stage-4 input contract.

---

## 4. Acceptance gates (summary)

| # | Gate | Threshold |
|---|---|---|
| G-A | Export row-order & determinism | exact match, both graphs |
| G-B | Index roundtrip | identical results post save/load |
| G-C | Fidelity vs exact dot @2K | recall@10 ≥ 0.99 (expect 1.000) |
| G-D | Cold-dataset query path | end-to-end, deterministic, ≥95% overlap with exact on cold z_d |
| G-E | Incremental insert | zero-shot insert retrievable; recall@10 drop ≤ 1pt vs rebuild |
| G-F | Near-hub vs away-hub recall gap @2K | report; investigate if gap > 5pts |
| G-G | 1M synthetic | recall@10 ≥ 0.95, p95 ≤ 10 ms CPU |
| G-H | Service API | manifest-verified load; schema tests green |

## 5. Risks & pre-answered questions

1. **Hub crowding is not fixable here.** 57 models filling all top-10 slots is
   Stage-2 geometry; Stage 3 measures it (G-F) and hands the evidence back.
   Do not burn time tuning efS against it.
2. **Synthetic 1M ≠ real 1M.** State the caveat everywhere the plot appears;
   the real-scale run is future work gated on harvesting a larger lake.
3. **G2 weights may not be on disk** (Phase 0 item 3) — one bounded re-train if so.
   Same applies to the G1/P6_dm10 comparison checkpoints.
4. **hnswlib tombstones don't reclaim memory** — covered by the rebuild policy.
5. **Two backends drifting** — faiss is check-only; production answers always
   come from hnswlib; benchmarks assert agreement.
6. **Import path** — Stage 3 imports Stage-2 modules (`model.py`, `cold_graph.py`,
   `top1_eval.py`); keep `stage2TrainGraphSAGE` importable (sys.path shim or
   package-ify), never copy-paste model code.

## 6. Suggested execution order & effort

Phases 0→1→2→3 are strictly sequential (~3–4 days). Phase 4 and Phase 5 are
independent after Phase 2 (~2–4 days). Phase 6 needs only Phase 2 (~1–2 days).
Phase 7 needs Phases 3+5. Phase 8 last. **Total ≈ 8–12 working days.**

First session target: Phase 0 + Phase 1 complete, gates G-A green on the
diverse graph.
