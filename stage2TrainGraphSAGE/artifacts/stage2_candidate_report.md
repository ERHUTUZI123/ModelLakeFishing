# Stage-2 GraphSAGE — 47K Pilot Candidate Report

Status: **converged**. This documents the recommended Stage-2 training config to
carry to the 47K ModelLens benchmark, the evidence behind it, and the open risks.
Generated on the real `hgraph_zoo_xm0.pt` (177 models / 23 datasets, frozen dim
448, 136 families). Not tuned for tiny-zoo Kendall tau.

## Selected candidate config (single source of truth: `experiment.CANDIDATE`)

| knob | value | why |
|---|---|---|
| `num_layers` | **1** | depth is the most direct over-smoothing knob; 2-layer reference had *negative* held-out ranking on the zoo |
| `top_frac` | **0.10** | per-dataset top-fraction positives; healthy density (~3%), avoids the unstable 0.05 and the re-collapsing 0.20 |
| `lambda_contrast` | **1.0** | raising it (2, 4) *increased* collapse, did not help ranking |
| `lambda_rank` | **1.0** | within-dataset margin ranking on the `z_m·z_d` dot score (retrieval geometry) |
| `lambda_mse` | **0.0** | MSE on the narrow [0.6,1.0] range rewards predicting the mean (collapse); kept available but off so it cannot dominate |
| `lambda_uniform` | **0.0** | anti-collapse regularizer available but OFF (diagnostic only; not tuned) |
| scorer | **dot** | aligned with HNSW cosine/inner-product retrieval; not an MLP |
| masks | **per-batch from membership** | no global N×N; 47K-safe |

2-layer support remains available (`HeteroGraphSAGE(num_layers=2)`, the code
default); the candidate is made explicit via `experiment.CANDIDATE` and `--candidate`.

## Candidate evidence (3 seeds, 25 epochs)

| metric | mean ± std | read |
|---|---|---|
| positive density | 0.031 ± 0.001 | sparse & stable (was ~74% under global threshold) |
| contrast-loss drop | 0.113 ± 0.014 | meaningful & stable (was ~0.03 flat = not learning) |
| ranking-loss drop | 0.005 ± 0.003 | tiny — ranking term sits near its margin floor |
| mean pairwise cosine | 0.887 ± 0.046 | collapse **reduced not solved** (was 0.95→0.9999) |
| tau_macro (within-dataset) | 0.128 ± 0.041 | **stable positive** ranking (reference 2-layer was −0.056) |
| tau_pool | 0.092 ± 0.105 | noisy; pooled tau conflates cross-dataset scale |
| checkpoint save/load | **OK** | num_layers=1 reloads, z matches, family_vocab matches |

This reproduces the robustness-matrix behavior for this cell, so per the decision
rule it is marked the **recommended 47K pilot config** (it did not fail).

## Mechanism gates (all pass on the real graph — `train.py`)
- gradient boundary: size/family embedding tables get nonzero grad; **frozen `x` gets none**.
- combined loss descends over epochs.
- checkpoint save → load reproduces `z` exactly; sidecar `family_vocab.csv` written; vocab/weight-row binding guard intact.

## Why the original contrast collapsed (not a code bug)
Global-threshold positives (`acc >= 0.8`) made **~74% of all model pairs positive**
— almost no negatives, so "pull positives together" ≈ "pull everything together".
Combined with MSE on the narrow [0.6,1.0] accuracy range (predicting the mean
minimizes MSE), the model minimized loss by **collapsing** embeddings (mean
pairwise cosine → ~1.0) and held-out ranking sat at ~0.

## What the scale-safe redesign changed
- **per-dataset top-fraction positives** (`topk_membership`) instead of a global
  threshold — adapts to dataset difficulty and model count (scale-safe).
- **dataset-local bounded margin ranking loss** (`perf_ranking_loss`) on the dot
  score — supervises ORDER (what retrieval needs), with capped pairs per dataset.
- **batch contrastive masks from membership + components** — never a global N×N
  (`batch_contrastive_masks(batch, M, comp)`), O(B²) per batch.
- **O(N·d) collapse metric** (‖Σz‖²) + bounded-sample over-smoothing diagnostic.
- **configurable 1/2-layer** GraphSAGE; optional off-by-default uniformity term.

## Robustness matrix summary (8 configs × 3 seeds)
- **1-layer > 2-layer on ranking**: 2-layer `tau_macro` went negative (over-smoothing).
- **raising `lambda_contrast` worsened collapse** (keep it at 1).
- **`top_frac` 0.05 = unstable zoo artifact** (density 0.01, mean_cos 0.74±0.34),
  **0.20 re-collapses** (mean_cos 0.96), **0.10 is the stable middle**.

## Why this candidate is safer for 47K than the best tiny-zoo tau
`top_frac=0.20` and `0.05` posted higher *mean* tau on the zoo, but 0.20
re-collapses embeddings and 0.05's numbers are one-seed noise (huge std). The
candidate was chosen on **reduced collapse + meaningful, stable loss descent +
no ranking degradation** — properties that should transfer — rather than a tiny,
high-variance held-out tau that will not transfer. It also avoids extreme knobs.

## Remaining risks (explicit)
- **Mean pairwise cosine is still high (~0.89)** — collapse is mitigated, not
  solved. If it persists at scale, the first lever is the (already implemented,
  off) `lambda_uniform` uniformity term, then revisit depth/temperature.
- **Zoo ranking is noisy** — small held-out sets (~13 test edges/dataset), so
  `tau_macro` has wide error bars; treat zoo tau as directional only.
- **47K may change behavior** — denser graph, more datasets/families; revalidate
  density, collapse, and ranking there before trusting this config.
- **HNSW recall NOT validated** — `hnswlib` is not installed in this environment
  (and `pyg-lib`/`torch-sparse` are absent, so training used the pure-python
  `LightLinkLoader` fallback). Install `hnswlib` + a sampler backend at 47K and
  verify recall@50 vs brute force, near-hub vs away-hub.

## Exact commands to reproduce
```
# candidate only (recommended; does NOT run the full matrix)
python -m ModelLakeFishing.stage2TrainGraphSAGE.experiment --candidate
python -m ModelLakeFishing.stage2TrainGraphSAGE.experiment --candidate --seeds 3 --epochs 25

# full robustness matrix (8 configs x 3 seeds)
python -m ModelLakeFishing.stage2TrainGraphSAGE.experiment

# mechanism gates + effect diagnostics on the real graph
python -m ModelLakeFishing.stage2TrainGraphSAGE.train

# interface contract check (Stage 1 -> Stage 2)
python -m ModelLakeFishing.stage1BuildTransferGraph.check_stage2_contract
```
Artifacts written by the candidate run: `stage2_candidate.pt` (weights + arch incl.
`num_layers` + repro metadata), `stage2_candidate.family_vocab.csv`.
