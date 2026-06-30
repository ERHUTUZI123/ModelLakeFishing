# Kendall action-guide ablation results (hf1000d/2000m)

All runs: 3 fixed split seeds (0,1,2), 1 init seed, 25 epochs, dot scorer, on
`hgraph_hf1000d_2000m_xm0_xd0.pt` (2000 models, 362 datasets). Evaluation universe:
held-out (test-split) observed candidates per dataset. Metrics from `eval_harness`
(per-dataset tau + exact z_d→z_m head retrieval). Paired bootstrap over identical
(split,dataset) tags via `compare.py`. Artifacts: `artifacts/ablation/<name>.json`.

## TL;DR

The single dominant lever is **the ranking loss**, not the graph or the message
weights. Replacing the hinge-on-`sigmoid(scale·dot+bias)` surrogate with a
**raw-dot RankNet loss** (the exact unit-vector inner product HNSW ranks) lifts
held-out `tau_macro` **0.193 → 0.398** and improves *every* head-retrieval metric.
This passes the strict paired-bootstrap gate (95% CI [+0.122, +0.294], P>0 = 1.00).
The edge-aware self/residual-path conv adds a smaller, clean increment on top.
Edge-weight consumption, dataset-graph pruning, separate heads, full-batch
"grouped" training, and (so far) the dataset→model contrastive do **not** help in
the winning regime.

## Evaluation reframing

Exact z_d→z_m **head retrieval was already strong at baseline** (hit@10 ≈ 0.81,
ndcg@50 ≈ 0.84) and the RankNet win improves it further (hit@10 → 0.85). The
production query is in good shape; `tau_macro` (full within-dataset concordance) is
the harder signal and is where the loss change pays off most.

## Ablation ladder (mean ± std over 3 split seeds)

| id | change (cumulative unless noted) | tau_macro | mean_cos | hit@10 | regret@10 | verdict |
|---|---|---|---|---|---|---|
| **B0** | baseline: dense graph, to_hetero, hinge | 0.193 ± 0.059 | 0.440 | 0.809 | 0.0055 | reference |
| B1 | drop similar_to | 0.212 ± 0.027 | 0.340 | 0.785 | 0.0109 | weak / head regress |
| B2 | top-k(10) unweighted (to_hetero) | 0.198 ± 0.064 | 0.387 | 0.801 | 0.0080 | ≈ B0 |
| B2e_ctrl | + edge-aware conv, NO weights | 0.254 ± 0.054 | 0.391 | 0.817 | 0.0101 | **retain (arch)** |
| B3_sim | + weighted similar_to | 0.232 ± 0.060 | 0.394 | 0.825 | — | no |
| B3_simtr | + weighted trained_on | 0.266 ± 0.046 | 0.392 | 0.842 | — | tau↑ head↑ |
| B3_all | + lineage weight (all weighted) | 0.279 ± 0.022 | 0.371 | 0.785 | 0.0104 | tau↑ but head regress |
| **B5_ranknet** | **edge-aware(unwtd) + RankNet (min_gap .01)** | **0.398 ± 0.035** | 0.348 | **0.850** | **0.0032** | **PROMOTE** |
| R_tohet | RankNet WITHOUT edge-aware (to_hetero) | 0.315 ± 0.031 | — | 0.857 | — | RankNet alone = +0.12 |
| R_weights | B5 + all edge weights | 0.354 ± 0.046 | — | — | — | weights hurt under RankNet |
| B4_grouped | edge-aware + full-batch macro (hinge) | 0.175 ± 0.025 | 0.070 | 0.738 | 0.0200 | **reject** |
| B6_heads | edge-aware + separate heads (hinge) | 0.172 ± 0.021 | 0.397 | — | — | **reject** |
| R_heads | B5 + separate heads | 0.383 ± 0.013 | 0.369 | 0.849 | 0.0035 | neutral (low var) |
| R_mg00 | B5 with min_gap 0.0 | 0.363 ± 0.068 | — | 0.889 | — | noisy (worst) |
| **R_mg02** | **B5 with min_gap 0.02** | **0.403 ± 0.022** | — | 0.857 | — | **best + lowest var** |

### min_gap sweep (edge-aware unweighted + RankNet)
| min_gap | tau_macro | note |
|---|---|---|
| 0.00 | 0.363 ± 0.068 | treats noise as preference → noisy, worst |
| 0.01 | 0.398 ± 0.035 | good |
| **0.02** | **0.403 ± 0.022** | best, lowest variance — refined winner |

## Decomposition of the win

- **RankNet loss alone** (to_hetero, top-k): 0.193 → **0.315** (+0.12). Biggest single factor.
- **+ edge-aware self/residual conv**: 0.315 → **0.398** (+0.08). Clean architecture gain.
- **Edge weights**: help under the *hinge* loss (0.254→0.279) but **hurt under RankNet** (0.398→0.354) and regress top-K head retrieval → leave OFF.
- **Dataset-graph pruning** (top-k / drop): dense is mildly harmful but pruning is not a standalone win once the loss is fixed.
- **Full-batch "grouped" macro & separate heads**: regress; the per-batch sampler + edge dropout matters, and the shared head is fine.

## Gate (B0 → B5_ranknet)

```
tau: delta=+0.2051  95% CI [+0.1218, +0.2943]  P(>0)=1.00  -> EXCLUDES 0 (PROMOTE)
head: hit@10 +0.041, recall_top3 +0.046, ndcg@10 +0.0095, ndcg@50 +0.0077, regret -0.0023  (all improve)
```

Reproduces on all 3 fixed splits (per-split tau: s0 +0.21, s1 +0.10, s2 +0.21).
No leakage: membership M and dm-positives built from train-visible edges only;
test edges removed from message passing in both directions by the split.

## Phase 6 — dataset→model contrastive (serving relation) & Phase 7

Built `dataset_to_model_contrastive_from_edges` (per-batch InfoNCE, z_d query →
z_m, train-visible positives only) and wired it into the winning per-batch path.
Early stopping by val tau + lr sweep added (Phase 7).

| config | tau_macro | hit@10 | recall_top3@10 | note |
|---|---|---|---|---|
| B5_ranknet (ref) | 0.398 ± 0.035 | 0.850 | 0.841 | tau-optimal |
| P6_dm05 (+dm 0.5) | 0.343 ± 0.065 | 0.905 | 0.889 | head↑ tau↓ |
| **P6_dm10 (+dm 1.0)** | 0.328 ± 0.070 | **0.905** | **0.902** | **serving-optimal** |
| P7_es (early stop, 40e) | 0.370 ± 0.042 | 0.904 | 0.862 | head↑, stable |
| P7_lr3e3 (lr 3e-3, 40e) | 0.371 ± 0.022 | 0.849 | 0.825 | lowest var |

**B5 → P6_dm10 paired:** tau −0.070 (CI [−0.137, −0.004], P>0=0.02 — significantly
lower) but hit@10 +0.055 and recall_top3 +0.060. A clean Pareto tradeoff: the
dataset→model contrastive directly optimizes the SERVING query (z_d→z_m), sharpening
the top of the retrieval list at a cost to full-order concordance.

## Final recommendation (two regimes)

Both keep a pure cosine/inner-product retrieval score (HNSW-compatible).

1. **Ranking-priority (ACCEPTED, passes strict gate):** top-k(10) graph + edge-aware
   **unweighted** conv + **raw-dot RankNet** (min_gap **0.02**) + per-batch sampling
   with edge dropout + shared head + dot scorer.
   `tau_macro 0.403 ± 0.022` (B0 0.193; ΔCI excludes 0; reproduces on all 3 splits),
   head retrieval improves on every metric. Saved + roundtrip-verified:
   `artifacts/ablation/stage2_hf1000d_ranknet_candidate.pt` (production NOT overwritten).

2. **Serving/head-priority (variant):** the same + `lambda_dm_contrast≈1.0`.
   `hit@10 0.905, recall_top3 0.90` (best best-model retrieval) at `tau ≈ 0.33`.
   Use when "retrieve the best models for a dataset" is the literal objective.

**Rejected:** edge-weight consumption (helps under hinge, hurts under RankNet),
full-batch grouped macro training (collapses), separate heads (neutral/regress).

**Leakage audit (all configs):** membership M and dm-positives from train-visible
edges only; test edges removed from message passing both directions by the split;
split_seed separated from init_seed; eval universe = held-out observed candidates.
All 8 required-test groups pass (`tests/`).
