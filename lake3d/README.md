# lake3d — a movable 3D view of the model lake

Standalone, with one exception. `build.py` and `page.py` import nothing from
`ModelLakeFishing`; they read only frozen data artifacts and write one
self-contained HTML file. `init_space.py` does import it, because rebuilding
the encoder as it was before training needs the training code itself.

```
init_space.py the A0 seed-0 encoder at its initialisation, checked and exported
build.py      the 3D layout of both spaces, in checkpointed stages
page.py       packs the layout into lake3d.html, with a before / after switch
_harness.js   runs the page's JavaScript under stubbed THREE + DOM (node)
payload/      packed layout and metadata already embedded in the shipped pages
three.r128.min.js  vendored Three.js r128 for the offline page
```

## Rebuild the shipped pages

From the repository root, with Python and NumPy installed:

```bash
python -m pip install numpy==2.4.6
python lake3d/page.py
python -m http.server 8000 --directory lake3d
```

Open `http://127.0.0.1:8000/lake3d_offline.html`. Both HTML files are written
beside `page.py`. The included `payload/packed.npz` and `packed_meta.json`
contain the same layout and metadata as the committed pages; rebuilding the
pages needs no training data, checkpoint, sibling source directory, or network
after NumPy is installed. The offline page includes Three.js and uses local fonts.

All default paths are resolved from the scripts' location, not the terminal's
working directory. You can move or rename the repository. `build.py` and
`page.py` can also be used with just the `lake3d/` directory; `init_space.py`
requires the rest of the repository's training and evaluation code.

To use existing artifacts elsewhere, set these environment variables before
running the scripts. Absolute paths are accepted; relative values are resolved
against the repository root, independently of the working directory.

| Variable | Default relative to the repository root |
|---|---|
| `LAKE3D_DATA_ROOT` | `data/data1m` |
| `LAKE3D_A0_ROOT` | `<LAKE3D_DATA_ROOT>/a0_20260912` |
| `LAKE3D_WORK` | `lake3d/payload` |
| `LAKE3D_RUN_DIR` | `docs/1M/A0_runs/A0_4/delivery_s0` |
| `LAKE3D_DATASET_NODES` | `<LAKE3D_DATA_ROOT>/rf/canon/dataset_nodes_merged.parquet` |

## What it reads

The full layout rebuild uses frozen A0, split seed 0 artifacts. These large
graph, export and checkpoint inputs are not included in this Git checkout.
They are not needed to view or regenerate the shipped pages. A newly collected
HF dataset does not replace this frozen snapshot: the rebuild checks its hashes.

| path | what for |
|---|---|
| `a0_20260912/exports/A0GD_full_s0_e25/z_m_eval.npy` | 3,016,439 × 128 model representations |
| `a0_20260912/exports/A0GD_full_s0_e25/z_d_eval.npy` | 18,729 × 128 dataset–task queries |
| `a0_20260912/exports/A0GD_full_s0_e25/gold_cands.npz` | the held-out best model per query |
| `a0_20260912/exports/A0GD_full_s0_e25/model_ids.parquet`, `dataset_ids.parquet` | names for the model and query rows |
| `a0_20260912/exports/A0GD_full_s0_e25/EXPORT_MANIFEST.json` | export identity and hashes |
| `a0_20260912/graph` | family ids, supervision edges, lineage edges |
| `a0_20260912/metrics/a0_hnsw_s0.npz` | the measured seed-0 retrieval, for the cast |
| `docs/1M/A0_runs/A0_4/delivery_s0/ckpt/last.pt` | the trained checkpoint: its config, and the Adam state that checks the rebuilt start |
| `docs/1M/A0_runs/A0_4/delivery_s0/A0_RUN_RECORDS.json`, `MANIFEST.json` | run records and initialization seed |
| `a0_20260912/metrics/A0_EVALUATION_REPORT.json` | the recorded seed-0 scores the trained export must reproduce |
| `data1m/rf/canon/dataset_nodes_merged.parquet` | which held-out queries are scored, as A0 bound it |

The graph's `nodes.npz` and `edges.npz` are the frozen bytes. A0 changed only
the dataset feature matrix — seven performance-derived columns zeroed, with
the hashes recorded in `graph/A0_FEATURE_REPAIR.json` — so families and
lineage carry over unchanged while every representation drawn here is new.

Working files land in `LAKE3D_WORK` (by default `lake3d/payload/`). A full
rebuild replaces its packed payload; use a separate work directory to keep the
shipped payload. Nothing is written back to the export, graph, or input metrics.

## Recompute the layout from frozen A0 artifacts

Use the repository's PyTorch/PyG environment (see the root README), then install
the additional visualization dependencies. These commands run from the
repository root, after the frozen artifacts have been placed at the paths above:

```bash
python -m pip install -r lake3d/requirements.txt
python lake3d/init_space.py --stage all
python lake3d/build.py --stage all
python lake3d/page.py
```

`init_space.py` and the query-layout stage use PyTorch; the layout also uses
UMAP and FAISS. The required package versions are recorded in
`lake3d/requirements.txt` and the repository's environment files.

The included Node.js harness checks payload decoding and UI wiring with a
stubbed DOM and Three.js. It does not render WebGL:

```bash
node lake3d/_harness.js lake3d.html
node lake3d/_harness.js lake3d.html '#before'
node lake3d/_harness.js lake3d_offline.html
node lake3d/_harness.js lake3d_offline.html '#before'
```

`build.py --stage before` reruns only the untrained layout and `align`; the
trained layout's files are not touched by it.

Stage timings on one RTX 4060 laptop:

| stage | what happens | time |
|---|---|---|
| `init_space export` | rebuild the initial encoder, check it, embed 3M models and 18,729 queries | 41 s |
| `init_space eval` | the A0 exact evaluator over the trained and the untrained export | 46 s |
| `pca` | exact PCA of the unit-norm vectors, 128 → 12, then project all 3M | 17 s |
| `umap` | UMAP of 250,000 anchors, 12 → 3, seed 7 | 3.4 min |
| `extend` | 370,000 more rows by 6-anchor inverse-distance interpolation | 1.6 min |
| `queries` | exact top-64 for every query, to place them | 20 s |
| `before-pca` | the same PCA of the untrained vectors, 128 → 51 | 7 s |
| `before-umap` | UMAP of the same 250,000 anchors, 51 → 3, seed 7 | 6 min |
| `before-extend` | the same 370,000 rows, placed in the untrained space | 1.9 min |
| `before-queries` | exact top-64 per query in the untrained space | 14 s |
| `align` | the rigid turn of the untrained layout onto the trained one | < 1 s |
| `pack` | read the cast, quantise to int16, pick 26 families, cut the payload | 2 s |

## How the layout is made

1. **PCA.** The exported vectors are unit norm, so the retrieval geometry is
   angular. An exact SVD of a 300,000-row sample gives 12 components; the
   first nine carry 99.83% of the variance, so the 128-dimensional space is
   effectively nine-dimensional.
2. **Anchors.** 250,000 rows are embedded to 3D with UMAP, then centred and
   divided by their 99th-percentile radius, so the page's world unit means
   something fixed.
3. **Extension.** Every other displayed row is placed at the inverse-distance
   average of its six nearest anchors in PCA space, jittered by a fraction of
   the local anchor spacing so interpolated rows do not stack on anchors.
4. **Queries.** A dataset–task node is placed at the similarity-weighted
   centroid of its exact dense top-64, so a query sits where its candidates
   are rather than where its own vector lands. This scan places the queries;
   it is not the retrieval.
5. **Pack.** 620,000 points, quantised to int16 on a shared scale, each with
   a second position in the untrained space on its own scale, plus a family
   byte, an evidence bit, the 18,729 query positions in both spaces, the
   lineage links as pairs of point indices so either space can draw them, and
   one cast. The page is 11.2 MB, 11.8 MB offline.

## The cast is read, not recomputed

`build.py` used to run its own exact dense scan to produce the candidates it
drew, which showed what a dense scan would have retrieved rather than what the
system did. It now reads `a0_hnsw_s0.npz`: the 1,000 ids HNSW returned in
cosine order, the task prior read over exactly those ids, and the ten the
system answered with. A query that seed 0 never scored has no measured answer,
and `build.py` stops rather than drawing one.

The cast is query 15473, `squad / question-answering`. Under A0 six of its ten
are general LLMs (olmo-2, layerskip-llama2-70b, gemma-3-4b ×4), which the
layout places in the LLM region, far from the query and the gold. That spread
is the measured answer and is kept: the layout is by representation, so a
ring's distance from the query is not its retrieval distance — the second of
the ten is in fact the closest of them by cosine.

## The lake before training

The switch at the top of the panel moves every point between two places:
where the trained encoder puts the model, and where the same encoder put it
before its first training step. `#before` at the end of the address opens the
page on the untrained space.

**What "before" is.** Not a random cloud drawn for contrast. `train_eval_one`
seeds torch with the initialisation seed — 0, recorded as `init_seed` in the
run manifest — and builds the model on the CPU before moving it to the GPU, so
the starting weights can be rebuilt exactly. `init_space.py` rebuilds them
through the same `build_models`, runs them over the same seed-0 held-out
message graph through the same `chunked_forward` the export used, and writes
`before/z_m_eval.npy` and `before/z_d_eval.npy`. Only the 25 epochs differ.

**How the rebuild is checked.** Adam without weight decay leaves any element
that never received a gradient exactly where it started, and the checkpoint's
optimizer state says which ones those are: both moments exactly zero. There
are 3,216 — the 129 family rows no model in the lake belongs to, and nine
columns of `dataset_proj` whose inputs are always zero, seven of them the ones
A0 zeroed. All 3,216 match the rebuild to within 2.4e-7 and 2,613 match to the
bit; the rest differ by one unit in the last place, the two CPUs rounding the
initialisers' arithmetic differently. A rebuild from seed 1 matches none and
misses by up to 5.12. Separately, the trained weights put through the same
path give back the archived export to 1.5e-7 on four of its chunks.

**How it is laid out.** The same recipe over the same rows: its own PCA, its
own UMAP over the same 250,000 anchors, the same 370,000 interpolated rows,
queries at the centroid of their own top-64. The untrained space needs 51
components for 99% of its variance where the trained one needs 9, so it gets
51. UMAP leaves each layout in an arbitrary orientation; `align` turns the
untrained one onto the trained one by the rotation that moves the drawn points
least — rigid, so the shape is unchanged. It barely helps: the mean distance
between a model's two positions goes from 0.696 to 0.691, in units where the
99th-percentile radius is 1. The two arrangements are unrelated, and the
morph shows the whole lake being rebuilt rather than nudged.

**What the numbers say.** `init_space.py --stage eval` runs the A0 exact
evaluator (`eval_y2.evaluate_exact_seed`, protocol a0) over both exports for
the 1,476 seed-0 queries. On the trained export it reproduces all 23 recorded
values exactly, so the untrained column comes from code that is known to
agree with the record:

| seed 0, exact top-1,000 then the task prior | before | after |
|---|---:|---:|
| gold@10 | 0.000 | 0.333 |
| queries whose gold is in the first 1,000 | 0 | 795 |
| median rank of the gold by cosine, of 3,016,439 | 843,840 | 591.5 |
| the cast's gold (`squad`), rank by cosine | 2,199,758 | 445 |

In the untrained view the cast is the same 1,000 candidates, ten and gold,
placed where the untrained encoder put them. Nothing was retrieved in that
space; the page says so, and quotes only the gold's rank there.

Colours travel with each model. Depth is read once, in the trained lake, so in
the untrained view it shows where the lake's core came from.

## What the page shows

- the lake before or after training, and the morph between them
- the lake, coloured by depth, by family, by whether the model carries
  evaluation evidence, or flat
- 18,729 dataset–task queries
- 45,998 lineage links, drawn only where both ends are among the shown points
- one cast: the 1,000 candidates HNSW returned for `squad /
  question-answering`, painted by the task prior read over them, the ten the
  system returned ringed, its held-out best model as the gold star, the query
  as the crimson cross. The gold, `deepset/flan-t5-xl-squad2`, sits 445th of
  the 1,000 on cosine alone; the prior returns it first.

Camera, colouring and picking are hand-written. The online page loads Three.js
r128 from cdnjs; the offline page embeds the included `three.r128.min.js`.

`page.py` writes both pages directly into this directory. It removes the
template's CSS and script comments, while keeping the payload and the vendored
Three.js license header intact. There is no external mirrored copy to update.

## What the build turned up

- The 128-dimensional space is effectively nine-dimensional, but the nine are
  far more even than they were before the A0 feature repair: the leading
  component now holds 19.8% of the variance, and the ninth still holds 4.0%.
  Components ten through twelve hold 0.15% between them.
- The top-64 lists of all 18,729 queries land on only 22,028 distinct models.
  The retrieval space is strongly hubbed, and the query cloud inherits it:
  the crimson points sit in a thin band rather than spread over the lake.
- Of the 3,016,439 models, 46,146 carry any supervision edge — 1.53% of the
  lake is luminous, and the rest is reachable through representation alone.
- Training compresses the space. Before it, 51 components hold 99% of the
  variance and 20 hold 90%; after it, nine hold 99.8%.
- Before training the arrangement is by identity, not by use. The family and
  size tables start as N(0, 1) rows of norm near 4, against about 1 for the
  name and card features together, so each (family, size bucket) pair
  collapses to nearly one point placed at random: the median spread inside
  such a group is 0.000 before training and 0.128 after. A family is torn
  apart by size — llama's eight size groups sit a median 1.22 apart before
  training, beyond the layout's median radius of 0.74, and 0.13 apart after.
  The 409,057 models in the catch-all family make one tight blob off to one
  side, and families too small to hold a place of their own fill the diffuse
  cloud in the middle: 63% of its innermost tenth belong to families with
  fewer than 100 members, against 14% of the drawn lake.
- The untrained space is even more hubbed: the top-64 lists of all 18,729
  queries land on 6,081 distinct models, against 22,028 after training.
- The task prior is strong on its own but cannot stand in for training.
  Fused with the untrained cosine over the whole lake it reaches gold@10
  0.278; behind an untrained first stage it reaches 0.000, because the gold
  never enters the 1,000 it reranks.
