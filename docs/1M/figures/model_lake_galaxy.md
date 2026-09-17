# The Model Lake — figure notes

Files: [`model_lake_galaxy.pdf`](model_lake_galaxy.pdf) (vector, 15.6 × 9.85 in, type
embedded, density layers rasterised) and [`model_lake_galaxy.png`](model_lake_galaxy.png).
Source: [`viz/model_lake_galaxy.py`](../../../viz/model_lake_galaxy.py) and
[`viz/galaxy_render.py`](../../../viz/galaxy_render.py). No training, evaluation, or
paper source was touched to produce it.

## Draft caption

> **The model lake and one cast into it.** (A) All 3,016,439 model embeddings of the
> frozen seed-0 held-out export, projected once and drawn as a log-density field. The
> retrieval space is a 128-dimensional unit sphere whose covariance is effectively rank
> ten, so the plate projects exactly onto that principal subspace, embeds a 250,000-row
> anchor sample with UMAP, and interpolates every remaining row from its six nearest
> anchors. Model families occupy coherent territories, named in place. The inset repeats
> the identical frame with only the 46,146 models that carry a supervision edge: 1.53% of
> the lake is observed, and the remaining 98.47% is reachable only through the learned
> representation. (B) One held-out dataset–task query, `squad · question-answering`, with
> its 1,000-candidate dense pool in the same coordinates; candidates the task prior lifts
> are warm, the ten returned models are numbered, and the held-out gold model is starred.
> The outline in A marks the stretch of lake this single cast covers. (C) The ten models
> actually returned, each drawn from where the dense stage ranked it to where it finished.
> (D) The same migration for all 835 seed-0 queries whose gold model reaches the pool.

## What each register shows

| Register | Content | Underlying quantity |
|---|---|---|
| header | 3,016,439 → 1,000 → 10 with per-stage cost | seed-0 rows of `Y4_REPORT.json` at K = 1,000 |
| A | log density of every model row; named family territories; luminous inset | `z_m_eval.npy`; `model.family_id` and `trained_on` degree from `hgraph_rf` |
| B | the dense top-1,000 pool of query 15473, its prior values, its returned ten | `exact_pool_s0.npz`, `prior_sidecar_s0.npz` |
| C | dense rank → final rank for the ten returned models | recomputed fusion, same query |
| D | dense rank → final rank for all in-pool queries | recomputed fusion, all 1,476 eligible queries |

## Numbers printed on the plate

Every figure is read from an archived report or recomputed from a frozen artifact. The
recomputation is gated: `rerank_facts` reproduces the seed-0 fusion
`r = (cos + 1)/2 + p_t(m)` with the fixed label-free tie-break and **asserts** that all
1,476 top-10 lists match `exact_pool_s0.npz::exact_top10` element for element and that
`gold@10` equals the archived `0.32859078590785906` exactly. The plate does not render if
that gate fails.

| Value | Source |
|---|---|
| 3,016,439 candidates / 18,729 query nodes / 247,803 supervision edges | `F5_runs/GRAPH_REPORT.json` |
| 46,146 luminous models, 1.530% | computed from `trained_on` edge endpoints |
| 41,056 families, 175 with ≥ 1,000 members | `hgraph_rf/meta.json` + `model.family_id` |
| recall@1000 0.9923, HNSW p50 0.406 ms, rerank p50 0.129 ms | `Y4_REPORT.json`, seed 0, K = 1,000 |
| gold@10 0.3279 (HNSW), 0.3286 (exact pool), 0.3031 (three-seed mean) | `Y4_REPORT.json` / `Y2_REPORT.json` |
| 835 in-pool, 641 not, 485 reaching the returned ten | recomputed, gated as above |

Panels B–D read the archived **exact** top-1,000 pool, whose `gold@10` is 0.3286; the
deployed HNSW path returns 0.3279 at recall@1000 0.9923. The footer states this.

## What the layout is, and what it is not

The map is a projection of the retrieval space, not the retrieval space. Neighbourhoods
are meaningful; absolute distances are not, and neither is any direction on the page.

Dataset–task nodes are deliberately **not** placed by their own embedding. Model and query
rows share one output head but occupy offset regions of the sphere, so a joint layout puts
every query in one corner and says nothing about retrieval. A query is instead drawn where
it fishes: at the similarity-weighted centroid of its exact dense top-64 models over the
full lake, which is the quantity retrieval actually uses. This is why panel B's pool is a
long stretch of the lake rather than a compact disc, and why the region is marked in A with
a density contour instead of a crop rectangle.

## Regenerating

```powershell
# from D:\research\model_lake\codes, with the ModelLakeFishing venv
python -m ModelLakeFishing.viz.model_lake_galaxy --stage project   # ~10 min, needs the GPU
python -m ModelLakeFishing.viz.model_lake_galaxy --stage render    # ~40 s
```

`--stage project` writes a cache under `data1m/figures/galaxy_cache` and skips any step
whose artifact already exists, so the render can be iterated without recomputing the
layout. Every random draw is seeded (PCA sample 0, anchor sample 11, UMAP 7, jitter
20260908).

Verified on 2026-09-08: a second `--stage project` run into an empty cache directory
reproduced every artifact bit-identically, including the 250,000-anchor UMAP embedding
and all 3,016,439 interpolated positions. Wall clock on one RTX 4060 Laptop GPU: PCA
projection 3 s, UMAP 181 s, layout interpolation 148 s, query placement 29 s, rerank
gate ~60 s.
