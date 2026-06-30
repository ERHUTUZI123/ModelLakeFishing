# Stage-1 dataset embedding redesign (design only — no code yet)

**Goal:** raise the *intrinsic* dimensionality of dataset embeddings `x_d^(0)` so
that `z_d` (and through the graph, `z_m`) can actually spread, instead of being
pinned to a ~3–4-dim "domain cone". This is the binding constraint identified by
the diverse-zoo experiment.

---

## 0. Diagnosis (why we are doing this)

**Current method.** A dataset is embedded by:
1. one frozen probe LM — `EleutherAI/gpt-neo-125m` (125M, autoregressive);
2. per example: `hidden_states[-1].mean(dim=1)` → one 768-d vector (mean over tokens),
   **labels discarded** (`extract_features_without_labels`);
3. `np.mean(over all examples)` → **one 768-d centroid per dataset**.

That centroid *is* `x_d^(0)`; `similar_to` edges are its pairwise distances. It is
the TransferGraph "domain similarity" embedding.

**Why it caps intrinsic rank.**
- **Single probe, single view** — one small LM gives one projection of the data.
- **Double averaging** (over tokens, then over examples) — keeps only the centroid;
  the distribution's spread, shape, and per-class structure are gone.
- **Labels discarded** — captures *text domain*, not *task*. SST-2 (sentiment) and
  CoLA (acceptability) are both single short English sentences → near-identical
  centroids, although they are different tasks.
- All our datasets are English short-text classification → centroids cluster.

**Evidence (diverse-zoo run).** Tripling diversity (177→306 models, +9 domain-distinct
datasets) left `z_d` participation **1.38 → 1.32 (flat)** and `z_m` only **1.83 → 2.04**,
while collapse crashed (mean_cos 0.89 → 0.39). Collapse was a Stage-2 optimization
artifact; **intrinsic rank did not move because it is set by `x_d`.** Measured
dataset-similarity cosine-gram effective rank stays ~3.7 regardless of dataset
selection (max-spread tops at ~4.5). **Conclusion: yes, the monolithic dataset
embedding is the dominant cause** (graph sparsity / 128-d L2 geometry / few dataset
nodes contribute too, but they are not the ceiling on intrinsic *task* rank).

---

## 1. Design principle — mirror the model side

The model node already learned this lesson:
`x_m^(0) = [e_name ‖ e_desc ‖ e_size ‖ e_fam]` = **frozen semantic views ‖ learnable
discrete descriptors**. The dataset node is still stuck at the equivalent of
"`e_name` only". The redesign makes the dataset node symmetric:

```
x_d^(0) = [ e_domain ‖ e_label ‖ e_card ‖ e_stats ]      (frozen semantic views)
          ⊕ learnable[ task_type_id , n_class_bucket_id , arity_id ]   (discrete)
```

with a new **`DatasetNodeEncoder`** that concatenates the frozen views and looks up
the learnable discrete rows — exactly parallel to `ModelNodeEncoder`. Each view below
adds a *different, complementary axis of variance*, so the concatenation has far
higher rank than any single view.

---

## 2. The views (each a step)

### Step 1 — `e_domain`: upgraded multi-statistic domain embedding (keep, but fix)
- **What:** still run dataset texts through a probe, but (a) use a **stronger text
  encoder** than gpt-neo-125m (a modern sentence/text-embedding model, e.g.
  bge/gte/e5/mpnet-class), and (b) summarize the per-example vectors with **more than
  the mean**: concatenate `[mean ‖ std ‖ a few quantiles]` (or the top-k PCA
  eigen-directions of the example cloud).
- **Why it adds rank:** the mean alone is one point; std/quantiles capture how
  *spread/shaped* the dataset is, so two datasets with the same centroid but different
  dispersion stop being identical. A better encoder also separates English text better.
- **Cost tier:** cheap (one forward pass over a sample, no labels).

### Step 2 — `e_label`: label-space embedding (the biggest task-axis win)
- **What:** semantically encode the **class label set** — embed the label strings and
  pool them, e.g. `["entailment","neutral","contradiction"]` vs
  `["positive","negative"]` vs `["world","sports","business","sci/tech"]`. Optionally
  add label-conditional text centroids (mean text vector per class), which encode how
  separable the classes are.
- **Why it adds rank:** this is the axis the current method throws away. It directly
  separates NLI / sentiment / topic / emotion — the *task* dimension — independent of
  text domain. Cheap and very discriminative.
- **Cost tier:** cheap (encode a handful of short strings).

### Step 3 — `e_card`: dataset-card / description embedding (the `e_desc` analogue)
- **What:** sentence-encode the dataset's README/card text (it usually states the task:
  "natural language inference", "sentiment of tweets"), same encoder family as the
  model-side `e_desc`.
- **Why it adds rank:** human-written task intent, orthogonal to both raw text domain
  and label strings; scalable to any HF dataset.
- **Cost tier:** cheap (one short encode), cacheable like model descriptions.

### Step 4 — `e_stats`: structural / difficulty fingerprint
- **What:** a small fixed vector of dataset statistics — sequence-length distribution
  (mean/percentiles), vocabulary richness, class balance (entropy of label
  distribution), train size bucket, and a cheap difficulty proxy (e.g. a linear probe's
  accuracy on top of frozen features, or Fisher discriminant ratio between class
  centroids).
- **Why it adds rank:** difficulty and class structure are real, task-relevant axes
  that neither domain nor label text capture.
- **Cost tier:** cheap–medium.

### Step 5 — learnable discrete descriptors (the `e_size`/`e_fam` analogue)
- **What:** discrete ids with learnable embedding tables inside `DatasetNodeEncoder`:
  - `task_type_id` — sentiment / NLI / paraphrase / topic / emotion / toxicity / QA /
    acceptability / STS / … (from HF `task_categories` + label-set heuristics), with an
    append-only vocab and an `Other` row (id 0), exactly like `family_vocab`.
  - `n_class_bucket_id` — bucketed number of classes (2 / 3 / 4–6 / 7–20 / regression).
  - `arity_id` — single-text / pair-text (NLI, paraphrase, QA) / multi-field.
- **Why it adds rank:** these are clean categorical axes the GNN can learn priors for,
  and they generalize zero-shot to new datasets (unseen task_type → Other), mirroring
  the cold-start story on the model side.
- **Cost tier:** ~free (metadata lookup).

### Step 6 (optional, expensive tier) — `e_task2vec`: Fisher/Task2Vec fingerprint
- **What:** the canonical task-aware embedding — diagonal Fisher information of a probe
  network briefly adapted to the dataset (the repo already stubs a Task2Vec path,
  `GraphAttributesWithTask2Vec` / `dataset_embed/task2vec_embed`).
- **Why it adds rank:** explicitly encodes *task* (uses labels), historically the
  strongest dataset-similarity signal; complements the cheap views.
- **Cost tier:** expensive (per-dataset adaptation). Datasets ≪ models in a lake, so
  this is affordable for the dataset axis even if not for every model.

### Step 7 (optional, use with care) — `e_perf`: collaborative performance fingerprint
- **What:** embed a dataset by the vector of how a **fixed reference panel** of models
  performs on it (two datasets are similar if the same models do well on both).
- **Why it adds rank:** maximally task-discriminative (it is the target signal).
- **Leakage caveat:** must NOT leak the perf-prediction labels — restrict to a
  held-out reference panel disjoint from train/val/test edges, or use only as an
  auxiliary view. Document exactly how it is computed. Off by default.

### Step 8 (optional) — multi-probe ensemble for `e_domain`
- **What:** compute `e_domain` from **several diverse probes** (e.g. a masked encoder,
  an instruction model, a multilingual model) and concatenate.
- **Why it adds rank:** different probes expose different axes; their concatenation is
  higher rank than any single probe. Directly answers "the embedding method is too
  single/monolithic".
- **Cost tier:** medium (k forward passes).

---

## 3. Fusion & architecture changes
- **Frozen part:** concatenate standardized (per-view L2-norm or z-score) frozen views
  `[e_domain ‖ e_label ‖ e_card ‖ e_stats (‖ e_task2vec ‖ e_perf)]` → `data['dataset'].x`.
- **Learnable part:** attach `task_type_id / n_class_bucket_id / arity_id` as int
  columns on the dataset node store; build a **`DatasetNodeEncoder`** (new, symmetric to
  `ModelNodeEncoder`) that concatenates frozen ‖ looked-up learnable rows and projects
  to the hidden dim. This replaces the current bare `dataset_proj = Linear(768→hidden)`.
- **Contract additions (symmetric to `xm0_meta`):** persist `task_type_vocab`,
  `n_class_buckets`, `arity_vocab` and bind them to the Stage-2 checkpoint (same
  orphan-row guard as `family_vocab`).
- **Keep everything else:** row-order contract, frozen/learnable boundary, no node-ID
  embeddings, reverse edges, RandomLinkSplit.

## 4. Recommended tiers (scalability)
- **Tier 0 (cheap, do first):** Steps 1–5 (better encoder + moments, label-space,
  card text, stats, learnable task_type/arity/n_class). All cacheable, lake-scalable;
  expected to lift effective rank the most per unit cost.
- **Tier 1 (richer):** add Step 6 (Task2Vec) and/or Step 8 (multi-probe).
- **Tier 2 (research):** Step 7 (collaborative perf fingerprint), leakage-guarded.

## 5. Validation (how we'll know it worked)
Re-run the same diagnostics as the diverse-zoo report and compare:
- **dataset-similarity cosine-gram effective rank** — target: from ~3.7 toward ≥8–10.
- **`z_d` participation ratio** — the headline; target: from ~1.3 toward ≥4–6.
- **`z_m` participation ratio**, **mean pairwise cos-distance**, **tau_macro** —
  expect intrinsic rank up *without* re-introducing collapse.
- **Ablation:** add views one at a time (Steps 1→2→3→…) and watch which moves
  `z_d` participation most — that isolates the true rank driver, not just "more dims".
- **Sanity:** `similar_to` neighborhoods should become *task*-coherent (NLI near NLI),
  not just domain-coherent (all English text near each other).

## 6. Risks / honesty notes
- More `x_d` dims with only ~24 dataset nodes risks GNN overfitting on the small zoo;
  judge intrinsic rank on the 47K benchmark too, not only the diagnostic graph.
- Concatenating views inflates *ambient* dim; report **effective** rank, not raw dim —
  the goal is real independent axes, not padding.
- Task2Vec / multi-probe add build cost; keep Tier 0 the default and gate the rest.
- `e_perf` is powerful but leakage-prone; off by default, documented if enabled.
