# V2 Full-Version Warm/Cold Rerun Instructions for Claude

## Mission

Rerun every historical Stage-2 version on the new v2 Stage-1 graph and reproduce
the same two-table evaluation standard used by the historical experiment:

```text
Table A -- Remaining datasets / warm held-out edges
Table B -- Completely unseen cold dataset IDs
```

Both tables must contain exactly:

```text
name | tau_macro | NDCG@1 | Hit@1 | top3_hit@1 | Rec@1 |
NDCG@10 | Hit@10 | Rec@10
```

Insert the completed v2 tables into:

```text
D:/research/model_lake/weeks/week6revieweverything/review_everything.md
```

Also save the canonical tables and machine-readable artifacts under:

```text
stage2TrainGraphSAGE/artifacts/effective_dataset_v2/full_version_rerun/
```

Do not overwrite the historical 31-warm/9-cold tables. Add a new section clearly
labelled `v2 effective-dataset graph` and state the new dataset counts.

## Fixed input

Use only the rebuilt graph:

```text
stage1BuildTransferGraph/hgraph_hf_effective_2000m_v2_xm0_xd0.pt
```

Use the corresponding canonical observation vault and manifests:

```text
stage2TrainGraphSAGE/artifacts/effective_dataset_v2/
```

The searchable universe remains the same frozen 2,000 models. Use raw normalized
dot product for ranking:

\[
s(d,m)=\operatorname{normalize}(z_d)^T\operatorname{normalize}(z_m).
\]

This request is an exact-dot semantic evaluation. HNSW ANN fidelity is not part
of either requested table.

## Versions to rerun

Use `cold_config_manifest.py` as the configuration source of truth. Do not
reconstruct flags from memory.

Run in this exact display order:

```text
Random (expected)
B0
B1
B2
B2e_ctrl
B3_sim
B3_simtr
B3_all
B4_grouped
B6_heads
BEST
B5_ranknet
R_tohet
R_weights
R_heads
R_mg00
R_mg02
P6_dm05
P6_dm10
P7_es
P7_lr3e3
G1
G2
G1dm
```

Use `init_seed=0` for this reproduction, matching the old table's `single init
seed 0` standard. Preserve the established epoch count for each named version:
normally 25 epochs, and 40 where the version manifest defines 40. Do not use the
20-epoch smoke result as a final row.

## Table A protocol — warm held-out edges

Create one dedicated warm experiment on the full v2 graph:

1. Keep every embedded dataset node available during training.
2. Use one frozen edge split with `split_seed=0`.
3. Separate train message edges, train supervision, validation edges, and warm
   test edges with no pair overlap in either direction.
4. Train each version once with `init_seed=0` on the identical split.
5. Evaluate only datasets retaining at least three positive held-out observed
   candidates with non-constant performance.
6. Use the same candidate IDs for every version.

The Table A caption must state the exact number of eligible warm datasets:

```text
Mean over N warm datasets, split seed 0, single init seed 0.
Random (expected) is analytic.
```

This table measures ranking of hidden model-performance edges for datasets that
were otherwise present during training.

## Table B protocol — completely unseen cold datasets

Run five-fold whole-dataset cold evaluation over the datasets that actually exist
in the final v2 graph and have at least three non-constant observed candidates.

Regenerate/freeze the fold manifest from final graph survivors before training.
Every eligible graph dataset must occur in exactly one cold fold. Fold sizes
should be balanced and their exact counts and names must be printed.

For each fold and version:

1. Remove every cold dataset's `trained_on` edges.
2. Remove every reverse `rev_trained_on` edge.
3. Remove all cold ranking supervision and validation labels.
4. Remove cold participation in model-model membership, dataset-model
   contrastive positives/negatives, global negatives, and hard-negative mining.
5. Remove cold dataset nodes and incident `similar_to` edges from the training
   computation.
6. Train using only the remaining datasets, with `init_seed=0`.
7. At inference, construct each cold dataset embedding only from its deployable
   raw feature and incoming deployable `similar_to` messages.
8. Do not retrain or modify model parameters during insertion.
9. Assert that every indexed `z_m` is unchanged before and after cold query
   insertion.
10. Load the cold performance vault only after training/checkpoint selection and
    use it solely for evaluation.

Pool the per-dataset results across all five folds so each cold dataset contributes
exactly once. Do not average fold means equally when fold sizes differ.

The Table B caption must state:

```text
Pooled macro mean over N completely unseen datasets from five cold folds,
single init seed 0. Each dataset is evaluated exactly once as cold.
Same frozen z_m; inductive z_d. Random (expected) is analytic.
```

## Exact metric definitions

Let `C_d` be the observed evaluation candidates for dataset `d`, `y_md` their
canonical performance, and `pi_d` the descending exact-dot prediction order.

### tau_macro

Compute Kendall tau-b between all candidate scores and performances inside each
dataset, then macro-average datasets equally. Require at least three candidates
and one non-tied pair.

### NDCG@1 and NDCG@10

Use canonical performance as graded relevance. For `k=min(K, |C_d|)`:

\[
NDCG@K_d=\frac{DCG(y_{\pi_d(1:k)})}
{DCG(\operatorname{sort}(y_d,descending)_{1:k})}.
\]

The ideal DCG must be truncated to the same `k`.

### Hit@1 and Hit@10

Let `B_d` contain every candidate tied for maximum true performance.

```text
Hit@K = 1 if predicted Top-K intersects B_d, otherwise 0.
```

### top3_hit@1

Construct the tie-aware true Top-3 set using the third-highest distinct
performance cutoff. Give one if the single predicted rank-1 model belongs to that
set, otherwise zero.

### Rec@1 and Rec@10

Use recall of that same tie-aware true Top-3 set:

\[
Rec@K_d=\frac{|TopK_s(d)\cap T_d^3|}{|T_d^3|}.
\]

`top3_hit@1` and `Rec@1` are not interchangeable. Selecting one member of a
three-model Top-3 gives `top3_hit@1=1` and `Rec@1=1/3`.

## Random baseline

Place `Random (expected)` first in both tables and use the exact same candidate
sets as learned versions.

- `tau_macro = 0` analytically;
- compute analytic expected `Hit@1`, `top3_hit@1`, `Rec@1`, `Hit@10`, and
  `Rec@10` under a uniform random permutation, including ties;
- compute expected NDCG using an exact expectation when practical, otherwise at
  least 10,000 deterministic permutations per dataset with a fixed RNG seed;
- macro-average per-dataset expectations rather than pooling edges.

## Required implementation

Create a resumable driver, for example:

```text
stage2TrainGraphSAGE/v2_full_version_rerun.py
```

It must support:

```text
--mode warm
--mode cold --fold 0..4
--version NAME
--resume
--aggregate
--render-markdown
```

Every `(mode, fold, version, seed)` run must write an independent JSON artifact
immediately after completion so an interrupted sweep resumes without repeating
finished GPU work.

Before launching the full sweep, run one B0 warm job and one B0 cold-fold job and
verify the complete artifact schema. Then launch every version.

## Required assertions

Before accepting a row, assert:

1. all versions use the same warm candidate manifest;
2. all versions use identical cold fold/candidate manifests;
3. every eligible cold dataset occurs exactly once across folds;
4. no cold performance edge appears in training messages or supervision;
5. no reverse copy of a held-out edge remains;
6. no cold label enters any contrastive/global-negative construction;
7. cold query insertion leaves indexed model embeddings unchanged;
8. all metric values are finite and within their valid range;
9. the reported `n_datasets` equals the number of saved per-dataset records;
10. no 20-epoch smoke artifact is accidentally reused as a final result.

## Required artifacts

Write:

```text
full_version_rerun/run_manifest.json
full_version_rerun/warm_candidate_manifest.json
full_version_rerun/cold_fold_manifest.json
full_version_rerun/runs/warm/<version>.json
full_version_rerun/runs/cold/fold_<f>/<version>.json
full_version_rerun/per_dataset_warm.json
full_version_rerun/per_dataset_cold.json
full_version_rerun/V2_WARM_COLD_TABLES.json
full_version_rerun/V2_WARM_COLD_TABLES.md
full_version_rerun/leakage_audit.json
```

The aggregate JSON must include raw precision values. Round to three decimals
only when rendering Markdown.

## Required Markdown insertion

Append this structure to `review_everything.md`:

```markdown
## Stage-2 full-version rerun on the v2 effective-dataset graph

Graph: 2,000 models, N dataset nodes, E distinct trained_on pairs.
These results supersede the old-materialization tables for current model
selection; the old tables remain below/above for historical comparison.

### Table A — Known datasets / warm held-out performance edges

Mean over N warm datasets, split seed 0, single init seed 0.
Random (expected) is analytic.

| name | tau_macro | NDCG@1 | Hit@1 | top3_hit@1 | Rec@1 | NDCG@10 | Hit@10 | Rec@10 |
|---|---|---|---|---|---|---|---|---|
...

### Table B — Completely unseen datasets with no training-time performance history

Pooled macro mean over N unseen datasets from five cold folds, single init seed
0. Each dataset is evaluated exactly once as cold. Same frozen z_m; inductive z_d.

| name | tau_macro | NDCG@1 | Hit@1 | top3_hit@1 | Rec@1 | NDCG@10 | Hit@10 | Rec@10 |
|---|---|---|---|---|---|---|---|---|
...
```

Below the tables, add only a short factual summary identifying:

- the best row for `tau_macro`;
- the best row for `Hit@1`;
- the best row for `top3_hit@1`;
- the best row for `Hit@10`;
- the best row for `Rec@10`;
- whether the best learned row beats random;
- warm-to-cold changes for `P6_dm10`.

Do not insert smoke-test numbers into these final tables.

## Completion report

Return:

1. files changed;
2. exact graph and manifest hashes;
3. exact warm dataset count;
4. five cold fold sizes and total unique cold datasets;
5. tests/assertions and outputs;
6. both complete tables including random;
7. failed/interrupted runs, if any;
8. exact section and line inserted into `review_everything.md`.

The task is incomplete if it reports only P6, only smoke results, only warm
results, only one cold fold, or omits any historical version listed above.
