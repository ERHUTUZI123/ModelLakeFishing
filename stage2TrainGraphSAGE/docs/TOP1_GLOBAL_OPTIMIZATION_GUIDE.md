# Top-1 and Global Retrieval Optimization Guide for Claude

## Mission

Improve one ANN-compatible dataset-to-model embedding space for both:

1. **Global candidate generation:** a good held-out model must survive competition
   from every model currently in the searchable pool and from newly added models.
2. **Local selection:** once the candidate set is small and performance-labeled,
   the same geometry must select a genuinely strong model, not merely a vaguely
   relevant one.

The serving score must remain:

\[
s(d,m)=\operatorname{normalize}(z_d)^\top\operatorname{normalize}(z_m).
\]

Do not replace HNSW candidate generation with an MLP or any score that requires
scanning every model. A later reranker may use richer features, but that is not a
substitute for improving the shared retrieval geometry here.

## Primary metrics

Every experiment must report exactly these five primary metrics on the same fixed
splits and the same eligible datasets:

| Metric | Capability measured | Direction |
|---|---|---|
| `observed_hit@1` | Exact best-model selection inside held-out observed candidates | Higher |
| `top3_hit@1` | Whether the single selected model belongs to the true observed top-3 | Higher |
| `regret@1` | Performance loss of the selected model versus the observed best | Lower |
| `full2k_gold@1` | Whether the observed gold model ranks first among all 2,000 indexed models | Higher |
| `full2k_gold@10` | Whether the observed gold model survives into the global top-10 | Higher |

Definitions for dataset \(d\):

\[
\hat m_d^{obs}=\arg\max_{m\in C_d^{test}}s(d,m),
\qquad
m_d^*=\arg\max_{m\in C_d^{test}}y_{md},
\]

\[
\operatorname{ObservedHit@1}_d=\mathbf 1[\hat m_d^{obs}=m_d^*],
\]

\[
\operatorname{Top3Hit@1}_d=
\mathbf 1[\hat m_d^{obs}\in\operatorname{TrueTop3}(C_d^{test})],
\]

\[
\operatorname{Regret@1}_d=y_{m_d^*d}-y_{\hat m_d^{obs}d},
\]

\[
\operatorname{FullPoolGold@K}_d=
\mathbf 1[m_d^*\in\operatorname{TopK}_{m\in\mathcal M_{search}}s(d,m)].
\]

`full2k_gold@K` is a **gold-survival** metric, not full-pool precision: models
without a performance label are unknown, not automatically wrong.

## Non-negotiable evaluation discipline

1. Freeze graph hash, source-data manifest, split seeds `{0,1,2}`, and init seed.
2. Use `test_data` embeddings for evaluation. Never run the original graph with
   held-out `trained_on` edges restored.
3. Materialize candidate sets once and verify that all configurations use the
   same dataset/model IDs.
4. Report per-dataset records, candidate counts, selected model ID, gold model
   ID, selected/gold accuracy, global gold rank, and all five primary metrics.
5. Report observed metrics by candidate-count strata: `3-10`, `11-20`, `21-50`,
   `51-100`, and `100+`.
6. Use paired bootstrap over datasets. Do not compare only aggregate means.
7. Report exact-dot results first. HNSW fidelity is a separate systems check and
   must not be mixed with semantic retrieval quality.
8. Resolve the current provenance mismatch (`12,205` edges in the experiment
   graph versus `9,305` in the current materialization report) before long runs.

## Baselines to preserve

Reproduce the five primary metrics for:

```text
B0
B5_ranknet
R_mg02
P6_dm10
random observed-candidate ranking
random full-pool ranking
```

Current reference values already show the central failure mode:

```text
R_mg02 observed_hit@1  = 0.364
R_mg02 full2k_gold@10  = 0.055
P6_dm10 observed_hit@1 = 0.364
P6_dm10 top3_hit@1     = 0.610
P6_dm10 regret@1       = 0.0650
P6_dm10 full2k_gold@10 = 0.056
```

The optimization target is not another observed-only gain. It is to close the
large gap between local selection and global survival.

## Acceptance rule

Treat the task as constrained multi-objective optimization, not a single weighted
leaderboard score.

Promote a change only if:

1. `full2k_gold@10` improves on all three fixed splits or its paired 95% bootstrap
   interval excludes zero;
2. `observed_hit@1` does not regress by more than 0.02 absolute;
3. `top3_hit@1` does not regress by more than 0.02 absolute;
4. `regret@1` does not increase by more than 0.005 absolute;
5. raw normalized dot product remains the exact serving score;
6. no validation/test performance values or target edges enter training.

`full2k_gold@1` is sparse at the current dataset count. Always report it, but do
not accept a configuration merely because one extra dataset changes it from zero.
Require split consistency and inspect global gold-rank movement as a secondary
diagnostic.

Maintain a Pareto table. If one configuration is best globally and another is
best locally, do not hide the tradeoff behind an arbitrary weighted average.

## Phase 0 — Correct and freeze the evaluator

Add one evaluation function that computes both observed and full-pool metrics from
the same `z_dict` and held-out labels.

Requirements:

```python
# observed selection
observed_order = argsort(-(z_m[candidates] @ z_d))

# global survival: score every searchable model
global_order = argsort(-(z_m_all @ z_d))
```

Tests must prove:

- `observed_hit@1` is not automatically one when `n_candidates <= 10`;
- the global universe contains all 2,000 model IDs exactly once;
- the held-out gold model is present in the global universe;
- changing an unlabeled distractor embedding can change global metrics without
  changing observed metrics;
- the original held-out target edge is absent in both message directions;
- exact and HNSW top-1 agree under the configured fidelity test.

Deliver a `TOP1_BASELINES.json` and `TOP1_BASELINES.md` before changing training.

## Phase 1 — Add reliable global negatives

The current ranking loss compares models only within a dataset's observed
supervision list. It never teaches the dataset query to beat most of the other
1,999 indexed models. Fix this first.

For each training-visible dataset \(d\):

- positives: its training-visible high-performing models;
- reliable local negatives: its known low-performing models;
- reliable global negatives: task/modality/schema-incompatible indexed models;
- hard reliable negatives: known-low models currently scoring highly;
- unobserved but compatible models: **unknown**, not automatically negative.

Do not label every missing model-dataset pair as negative. Missing performance is
not evidence of poor performance.

Implement a global sampled-softmax/contrastive term:

\[
\mathcal L_{global}(d)=
-\frac{1}{|P_d|}\sum_{p\in P_d}
\log\frac{\exp(s(d,p)/T_g)}
{\sum_{a\in P_d\cup N_d}\exp(s(d,a)/T_g)}.
\]

Keep `N_d` bounded and log its composition. Start with only reliable negatives.

Run one change at a time:

```text
G1: B5/R_mg02 + uniformly sampled incompatible global negatives
G2: G1 + known-low global hard-negative mining
G3: G2 + cross-batch model queue for a larger denominator
```

Primary gate: improve `full2k_gold@10` without violating the three local metric
constraints.

## Phase 2 — Preserve and improve local Top-1 quality

Keep raw-dot RankNet for broad pairwise ordering, then test one top-heavy local
loss at a time.

Preferred first experiment: a Plackett-Luce/ListNet-style likelihood over the
observed models for each dataset:

\[
q_d(m)=\frac{\exp(y_{md}/T_y)}{\sum_{j\in C_d}\exp(y_{jd}/T_y)},
\qquad
p_d(m)=\frac{\exp(s(d,m)/T_s)}{\sum_{j\in C_d}\exp(s(d,j)/T_s)},
\]

\[
\mathcal L_{local-list}=-\sum_{m\in C_d}q_d(m)\log p_d(m).
\]

This retains graded performance differences: choosing a near-best model is less
bad than choosing a poor one, which should help `regret@1`.

Run:

```text
L1: accepted global configuration + RankNet only
L2: L1 + small local listwise weight
L3: L1 + gap-weighted RankNet instead of listwise
```

Do not remove the global loss merely to recover observed metrics. Tune the local
weight under the acceptance constraints.

## Phase 3 — Train for newly added models

The final index will continuously receive models with no `trained_on` edges.
Training must simulate this condition.

Add cold-model augmentation:

1. Sample a fraction of model nodes per batch.
2. Remove their `trained_on` and reverse message edges for that forward pass.
3. Preserve frozen metadata features and, when available, lineage.
4. Continue scoring them through the same inductive encoder.

Run separately:

```text
C1: relation dropout only (current behavior)
C2: whole-model trained_on neighborhood masking
C3: C2 + lineage masking for a smaller subset
C4: family-holdout evaluation, with no training model from selected families
```

Add a cold-model split where held-out models contribute no ranking labels and no
message edges during training. Report the same five metrics, clearly suffixed
`_cold_model`.

The model must not use node-ID embeddings. New models must be encodable from the
same frozen metadata, categorical vocab with `Other/unknown` fallbacks, and
available graph relations.

## Phase 4 — Control hubness and duplicate families

Inspect the models that repeatedly outrank the gold model globally.

For each false global top-10 entry, log:

```text
model ID
family/base model
task compatibility
observed/unobserved status
number of queries for which it appears in top-10
cosine to query and gold
```

Then ablate:

```text
H1: family-balanced global negative sampling
H2: same-family hard negatives when known performance differs
H3: per-family cap used only after ANN retrieval
H4: hub-frequency regularizer or sampled query-uniformity penalty
```

`H3` is a serving policy, not an embedding improvement. Report it separately and
never let post-retrieval family deduplication disguise weak raw global geometry.

## Phase 5 — Validation selection aligned with both goals

Do not select checkpoints using validation tau alone.

Use a constrained rule:

1. discard epochs violating local metric floors;
2. among remaining epochs, maximize validation `fullpool_gold@10`;
3. break ties by lower `regret@1`, then higher `top3_hit@1`.

Do not evaluate test metrics every epoch for checkpoint selection.

Log gradient cosine between:

```text
global retrieval loss
local RankNet/listwise loss
model-model contrastive loss
```

If global and local gradients consistently conflict, first tune weights and use a
warm-up/schedule. Test gradient surgery only as an isolated later ablation.

## Phase 6 — Scale test with newly added real models

Once full-2K metrics improve, freeze the encoder and inject real HF model
distractors without retraining:

```text
2K -> 10K -> 50K -> 100K -> 500K -> 1M
```

At every scale, keep the same gold datasets/models and report the five primary
metrics where defined. For global capability, keep absolute `K={1,10}` fixed;
also record gold rank as a diagnostic.

New models must be selected independently of the evaluation datasets. Do not use
dataset tags, performance-edge counts, or gold membership to choose distractors.

After the injection-only curve, optionally retrain with a scale-matched negative
queue and compare:

```text
frozen encoder under added models
versus
scale-aware training under added models
```

This separates inductive robustness from benefits obtained by retraining.

## Minimal ablation ladder

Run sequentially and retain only accepted rows:

| ID | Single change |
|---|---|
| T0 | Reproduce B5, R_mg02, and P6 on the corrected five-metric evaluator |
| G1 | Add task/schema-incompatible sampled global negatives |
| G2 | Add known-low global hard negatives |
| G3 | Add a bounded cross-batch global model queue |
| L1 | Add a small local listwise loss for Top-1/regret |
| C1 | Whole-model message-edge masking for cold-model training |
| H1 | Family-balanced negative sampling |
| V1 | Dual-goal validation checkpoint selection |
| S1 | Inject 10K independently selected real model distractors |
| S2 | Scale accepted configuration through 50K/100K/1M |

Do not combine two unaccepted changes in one run.

## Required artifacts after every trial

Return:

1. exact hypothesis and files changed;
2. tests and outputs;
3. all five primary metrics per split and aggregate;
4. paired deltas and bootstrap intervals versus the retained control;
5. per-dataset JSON including global gold rank;
6. candidate-count-stratified local metrics;
7. negative-sampler composition and false global top-10 audit;
8. leakage audit;
9. decision: retain, reject, or investigate;
10. next single ablation.

Required summary table:

```text
name | observed_hit@1 | top3_hit@1 | regret@1 |
full2k_gold@1 | full2k_gold@10
```

## Files likely to change

- `stage2TrainGraphSAGE/eval_harness.py`
- `stage2TrainGraphSAGE/losses.py`
- `stage2TrainGraphSAGE/sampling.py`
- `stage2TrainGraphSAGE/train.py`
- `stage2TrainGraphSAGE/ablation.py`
- focused tests under `stage2TrainGraphSAGE/tests/`

## Final standard

Do not call a model production-ready because it performs well only inside a small
observed candidate list. Do not call it globally capable because a known gold
model merely lands somewhere in a generous fraction of a 2K pool.

The target is one inductive, ANN-compatible geometry that:

```text
keeps known-good models alive against the actual searchable pool,
continues to work as unseen models are inserted,
and still chooses a near-best model when the candidate set becomes small.
```

