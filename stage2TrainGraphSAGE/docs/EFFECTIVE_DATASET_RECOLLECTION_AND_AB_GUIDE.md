# Effective-Dataset Recollection and Stage-1/Stage-2 A/B Instructions for Claude

## Mission

Rebuild the data pipeline around the fixed current model universe so that Stage 2
has substantially more datasets with enough **real, distinct model-performance
observations** to train and evaluate ranking.

The current funnel is:

```text
1,000 selected dataset candidates
-> 131 with harvested single-metric performance records
-> 102 edge-bearing datasets survive embedding/materialization
-> 362 total dataset nodes, mostly node-only
-> 72 datasets have any positive trained_on edge after graph cleanup
-> 61 datasets satisfy the current ranking-evaluation rule
-> 9 datasets enter the single cold-test split
```

The objective is not to create more empty dataset nodes. The objective is to
increase the number of ranking groups for which the current 2,000 models have
real, auditable, non-constant performance measurements.

After rebuilding Stage 1, rerun Stage-2 A/B tests on frozen dataset-level splits.
Do not overwrite the current graph, source tables, checkpoints, or historical
reports.

## Definition of an effective dataset

A dataset/ranking group is effective only if all of the following hold:

1. At least three distinct models from the frozen 2,000-model universe have a
   valid observed score.
2. The scores are non-constant and contain at least one comparable non-tied pair.
3. Every observation has a traceable real source; no synthetic or inferred score
   is allowed.
4. The dataset has finite deployable input features for the Stage-2 dataset
   encoder.
5. Dataset identity, configuration, task, metric, split, and evaluation protocol
   are not incorrectly merged with a different ranking problem.
6. One canonical `(model, ranking_group)` observation appears at most once after
   deterministic conflict resolution.

Track three coverage levels separately:

```text
minimum evaluable: >= 3 distinct models
robust:            >= 10 distinct models
strong:            >= 20 distinct models
```

Never report a node-only dataset as an effective dataset.

## Frozen comparison contract

### Primary model universe

Freeze the exact current 2,000 model IDs from:

```text
stage1BuildTransferGraph/hf1000d_2000m/selected_2000_models.json
```

Write a sorted ID manifest and SHA-256 hash. The primary experiment must contain
exactly those models. A record for another model may be retained in the immutable
raw cache but must be excluded from this experiment.

Do not replace models with edge-dense alternatives merely to inflate dataset
coverage. If the fixed universe cannot support the coverage target, report that
ceiling first. A changed or expanded model universe is a separate experiment,
not the same A/B.

### Isolated output locations

Create new versioned paths, for example:

```text
stage1BuildTransferGraph/hf_effective_2000m_v2/
stage1BuildTransferGraph/dataset_embed/data_hf_effective_2000m_v2/
stage1BuildTransferGraph/hgraph_hf_effective_2000m_v2_xm0_xd0.pt
stage2TrainGraphSAGE/artifacts/effective_dataset_v2/
```

Do not mutate `hf1000d_2000m`, its data directory, or its graph.

## Phase 0 — Audit the present loss of datasets

Before downloading anything, generate a machine-readable funnel for the current
pipeline. For every candidate dataset, record whether it was lost at each step:

```text
listed by the dataset selector
has at least one raw performance observation
has >=3 / >=10 / >=20 distinct current models
has non-constant scores
can be canonicalized
can be loaded or sampled for deployable features
dataset embedding succeeds
is present in records.csv
is present in the graph before any score threshold
is present after trained_on filtering
is evaluable after deduplication
```

Write:

```text
current_funnel.json
current_funnel.md
current_dataset_failures.csv
```

The report must reconcile at least these known counts:

```text
131 raw edge-bearing selected datasets
102 edge-bearing embedded datasets
362 materialized dataset nodes
9,305 materialized records
12,205 pre-fix graph rows
7,056 distinct graph pairs after the current cleanup
72 datasets with any cleaned positive edge
61 evaluable datasets
9 cold datasets
```

If any count differs, explain the exact artifact and transformation that produced
the difference. Do not continue while the provenance cannot be reconciled.

## Phase 1 — Perform an edge-first coverage ceiling audit

The old pipeline is substantially dataset-first: it chooses many datasets and
then discovers that most have no performance records. Reverse the order for the
supervised core.

For each of the fixed 2,000 models, enumerate every available structured public
evaluation record first. Only then derive candidate ranking groups from the
observations.

The raw record schema must preserve at least:

```text
source_name
source_url_or_artifact
source_snapshot_timestamp
source_record_id_or_hash
model_id_raw
model_id_canonical
dataset_raw
dataset_canonical
dataset_config
task
metric_raw
metric_canonical
metric_direction
split
evaluation_protocol
seed_or_run, if available
value_raw
value_canonical
confidence_tier
license_or_usage_note
```

Use source adapters with explicit provenance. Start with high-confidence sources:

1. structured Hugging Face model-index/card metadata;
2. official machine-readable benchmark or leaderboard exports whose model IDs can
   be matched to the frozen universe;
3. other public structured evaluation corpora with documented metric semantics.

Free-text card parsing must be a lower confidence tier and must not be mixed into
the primary table without a separate audit. Do not run large-scale model
inference, scrape access-controlled data, or fabricate missing scores.

Before embedding datasets, produce a ceiling table over the fixed 2,000 models:

| threshold | ranking groups | distinct observations | distinct covered models |
|---|---:|---:|---:|
| >=1 model | | | |
| >=3 models and non-constant | | | |
| >=5 models and non-constant | | | |
| >=10 models and non-constant | | | |
| >=20 models and non-constant | | | |
| >=30 models and non-constant | | | |

Also report counts by task, modality, metric, source, language, and candidate-count
bucket.

### Coverage targets

Treat these as goals, not permission to manufacture data:

```text
minimum-evaluable ranking groups: target >= 150
robust groups with >=10 models:    target >= 100
strong groups with >=20 models:   target >= 60
canonical distinct observations:  target >= 20,000
fixed models with >=3 observations: target >= 1,200
```

If the ceiling audit proves these targets impossible for the frozen 2,000 models,
stop before expensive embedding and report the attainable maximum. Do not silently
change the model universe. Recommend a separately named model-universe expansion
only after reporting the fixed-universe result.

## Phase 2 — Canonicalize without merging different ranking problems

Define the ranking-group key before selecting data. The safe default is:

```text
(dataset_canonical, dataset_config, task, metric_canonical,
 split, evaluation_protocol)
```

Two records may share a dataset name and still be different ranking problems.
Do not merge different configurations, languages, subsets, metrics, splits, or
evaluation protocols merely to obtain more edges.

Normalize aliases through a versioned alias table. Every alias operation must be
reversible and retain the original string.

Normalize metric values exactly once:

- convert percentages and fractions through a declared source-specific rule;
- preserve the raw value;
- record whether higher or lower is better;
- reject impossible ranges rather than clipping silently;
- do not combine accuracy, F1, BLEU, ROUGE, loss, latency, or perplexity as if they
  were the same target;
- do not normalize a constant group by dividing by zero.

### Duplicate and conflict policy

Use a canonical observation key that includes the complete ranking-group key and
model ID. Exact duplicate source records may be collapsed. Conflicting values must
be written to `performance_conflicts.csv` with all sources retained.

Resolve conflicts only through a documented deterministic rule such as verified
source precedence, matching evaluation protocol, or a declared run aggregation.
Do not silently take `max`, since that rewards models with more reported runs.
Report a sensitivity table for unresolved conflicts.

### Prohibit the current provenance failure

`stage1BuildTransferGraph/attributes.py::get_finetuned_records()` currently
concatenates rows from `model_config_dataset.csv` with `records.csv`. That created
duplicate performance rows and allowed the same canonical pair to cross message,
train, and test roles.

For the new pipeline:

```text
records/performance_observations = the only performance-label source
model_config_dataset             = node metadata only
```

Remove the concatenation from the v2 path. A Stage-2 repair such as
`dedup_trained_on(reduce=max)` is not an acceptable substitute for correct Stage
1 provenance.

## Phase 3 — Separate evaluation observations from graph message edges

Do not throw away low-performing but valid observations merely because the
message graph retains only positive/high-performance edges.

Materialize two distinct contracts:

1. `performance_observations.parquet`: every canonical real observation used for
   ranking labels, candidate sets, and evaluation;
2. graph message edges: a training-time topology that may be filtered or weighted
   by an explicitly ablated policy.

Stage-2 train/validation/test labels must be split from the full observation
table, not reconstructed from only positive graph edges.

For a compatibility experiment that requires one metric per dataset, select it
deterministically by largest distinct-model coverage, then source confidence, then
a fixed metric-name tie-break. Do not choose the metric according to which one
gives the best model result. Preserve every unused metric in the observation
table.

If representing dataset configurations as separate nodes would change Stage-2
architecture, defer that to a separate ablation. The primary A/B should improve
data without simultaneously changing the model architecture.

## Phase 4 — Select and embed the effective supervised core

Selection priority must be:

```text
strong groups (>=20 models)
-> robust groups (>=10 models)
-> minimum-evaluable groups (>=3 models)
-> optional node-only datasets
```

Within equal coverage buckets, balance task, modality, language, and source. Do
not select datasets using predicted rankings, gold model identity, or downstream
configuration performance.

Report concentration explicitly. A larger total that consists almost entirely of
near-duplicate text-classification datasets is not sufficient.

Embed edge-bearing datasets first. For every failed dataset embedding, write:

```text
ranking_group_id
dataset name/config
candidate count
task/modality/language
load error category
exact exception summary
whether a deployable fallback is possible
```

Dataset features and `similar_to` edges must use only information available when a
new dataset is deployed: raw examples and public metadata. Performance labels,
gold models, and downstream test results must not enter dataset embeddings or
dataset similarity.

Node-only datasets may still be included for unsupervised graph coverage, but
report them in a separate count and never allow them to inflate the effective
dataset total.

## Phase 5 — Rebuild Stage 1 in an isolated v2 path

Implement a resumable driver rather than manually editing CSVs. Suggested module:

```text
stage1BuildTransferGraph/effective_dataset_rebuild.py
```

It should expose independently restartable phases similar to:

```text
freeze-models
harvest-raw
ceiling-audit
canonicalize
select-effective
embed
materialize
build-graph
audit-graph
```

Raw source responses must be immutable and content-hashed. Derived files must
contain the source snapshot hash, code commit, arguments, and creation time.

The final v2 graph must contain:

- exactly the frozen primary model universe;
- every successfully embedded effective dataset;
- a stable model and dataset ID map;
- no duplicated canonical performance pair;
- `trained_on` and reverse edges that agree exactly;
- a bounded, self-edge-free `similar_to` top-k graph, starting with `k=10`;
- deployable model/dataset node features;
- either a reference to or embedded metadata for the full canonical observation
  vault.

Do not reintroduce the old near-complete dataset-similarity graph.

### Required post-build audit

Print and save:

```text
model nodes
dataset nodes
effective datasets >=3 / >=10 / >=20
node-only datasets
canonical performance observations
message trained_on edges
reverse-edge equality
distinct models with >=1 / >=3 / >=10 observations
candidate-count quantiles
score variance by ranking group
task/modality/language/metric/source distributions
embedding failures
duplicate/conflict counts
similar_to degree distribution
graph and observation-table hashes
```

The effective count after graph construction must equal the count derived from the
canonical observation vault, except for explicitly listed embedding failures.

## Phase 6 — Freeze dataset-level evaluation folds

Do not repeat the current conclusion on one nine-dataset cold split. Use all
effective datasets through deterministic dataset-level cross-validation.

Create five cold-dataset folds, stratified as far as possible by:

```text
task
modality
language
candidate-count bucket: 3-9, 10-19, 20-49, 50+
source family
```

Every effective dataset must appear in exactly one cold fold. Fold sizes must
differ by at most one unless a documented rare stratum makes this impossible.
Use exact fold assignment rather than rounding 20% independently inside small
strata.

For fold `f`:

- remove every cold dataset's `trained_on` and reverse edge from training;
- remove cold observations from ranking supervision and all contrastive/global
  target construction;
- remove cold dataset nodes and incident `similar_to` edges during training;
- choose validation datasets only from the non-cold portion;
- insert cold datasets at inference using deployable features and incoming
  `similar_to` messages only;
- keep indexed model embeddings unchanged during query insertion.

Write a fold manifest containing exact IDs, names, counts, strata, candidate IDs,
graph hash, observation-table hash, and split seed.

Also retain a warm edge-holdout evaluation on the non-cold datasets. Warm edge
holdout and whole-dataset cold start must be reported separately.

## Phase 7 — Stage-1 A/B design

Define the data-pipeline A/B before training:

```text
A = current Stage-1 data, corrected at source so performance rows are unique
B = rebuilt edge-first v2 data under the same frozen 2,000-model universe
```

Do not compare the leaked 12,205-row graph against B.

Run two complementary analyses.

### A/B-1: common-cohort paired comparison

Use only ranking groups, models, and canonical observations shared by A and B.
Freeze identical fold assignments, candidate IDs, train/validation roles, init
seeds, and hyperparameters. This isolates changes in Stage-1 graph/features and
provenance.

### A/B-2: expanded-cohort evaluation

Evaluate B on its full new effective-dataset universe. This measures improved
coverage and diversity, but is not a direct paired claim against A. Report dataset
counts beside every metric.

Never claim a model improvement merely because B contains different or easier
datasets.

## Phase 8 — Stage-2 rerun ladder

First run a short smoke test on one fold and seed. Then run representative models:

```text
Random (expected)
Graph-free two-tower baseline using the same node features and losses
B0
B5_ranknet
R_mg02
P6_dm10
G1dm
```

The graph-free baseline is required: with more datasets, it must be possible to
tell whether heterogeneous message passing adds value beyond dataset/model
features and ranking loss.

Only after data audits pass and the representative sweep is sensible, rerun the
full historical configuration set:

```text
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

Reconstruct every version from a checked configuration manifest, not memory.
Run all versions on byte-identical fold/candidate manifests. Use at least three
initialization seeds for finalists and clearly distinguish fold variance from init
variance.

## Metrics and required tables

For warm edge holdout and cold-dataset evaluation, report:

```text
name
n_datasets
median_candidates
tau_macro (Kendall tau-b)
NDCG@1
Hit@1
top3_hit@1
Rec@1
regret@1
NDCG@10
Hit@10
Rec@10
```

Include a macro-averaged `Random (expected)` row over the exact same candidates.
Use exact normalized dot product for semantic ranking. HNSW exact-vs-ANN recall is
a separate systems metric and must not be mixed with model quality.

For `K=10`, publish both:

1. deployment-style metrics with `k=min(10, n_candidates)`;
2. a strict table restricted to datasets with at least 10 candidates.

Always print counts for candidate buckets `3-9`, `10-19`, `20-49`, and `50+` so
small candidate lists cannot silently make Hit@10 trivial.

Use paired bootstrap over dataset IDs for the common-cohort A/B. Report per-fold
values, aggregate mean, 95% interval, and the number of datasets contributing to
each metric.

## Acceptance gates

### Data-pipeline gate

B may proceed to the full sweep only if:

1. provenance and duplicate leakage counts are zero;
2. the number of effective and robust datasets materially exceeds 61 and the old
   robust count;
3. every count drop from raw observation to graph is accounted for by named IDs
   and reasons;
4. dataset embeddings and similarity use no performance labels;
5. task/source concentration is reported and has not become more pathological;
6. the frozen 2,000-model manifest matches exactly.

### Model-quality gate

Do not accept B solely because it has more datasets. On the common cohort:

- require no material regression in cold `tau_macro`, `top3_hit@1`, and
  `regret@1`;
- inspect a tolerance of `-0.02` for tau/Hit metrics and `+0.005` for regret, then
  require paired intervals and fold consistency;
- require the best graph configuration to beat random on cold datasets;
- compare it directly with the graph-free baseline;
- preserve pure normalized dot-product serving geometry.

On the expanded cohort, emphasize confidence intervals, candidate-count strata,
task strata, and per-dataset results rather than a single attractive macro mean.

## Required tests

Add focused tests proving:

1. the model-universe manifest contains exactly 2,000 unique frozen IDs;
2. source snapshots and derived artifacts are content-hashed;
3. percentage/fraction conversion and metric direction are correct;
4. aliasing does not merge different configs, languages, tasks, splits, metrics,
   or protocols;
5. exact duplicates collapse once and conflicting records enter the conflict
   report;
6. no synthetic/demo row enters the primary observation table;
7. `model_config_dataset.csv` contributes no performance labels;
8. one canonical `(model, ranking_group)` observation cannot cross split roles;
9. the full observation vault retains low-performing valid models even if the
   message graph filters edges;
10. effective-dataset counts agree before and after graph construction;
11. `trained_on` and reverse edges are exact mirrors;
12. dataset embeddings and similarity are unchanged if performance labels are
    shuffled;
13. `similar_to` has no self edges and respects top-k degree bounds;
14. all five cold folds are deterministic, disjoint, and cover every effective
    dataset exactly once;
15. no cold label, edge, node, or contrastive membership enters training;
16. cold query insertion leaves all indexed model embeddings unchanged;
17. every A/B configuration receives identical common-cohort candidates;
18. random metric expectations and tie handling are correct on toy examples.

Do not launch the full sweep until these tests pass.

## Required artifacts

Return all of the following:

```text
frozen_models.json and SHA-256
source_manifest.json
immutable raw-source cache manifest
raw_performance_observations.parquet
canonical_performance_observations.parquet
dataset_aliases.csv
performance_conflicts.csv
coverage_ceiling.json/.md
current_funnel.json/.md
new_funnel.json/.md
dataset_embedding_failures.csv
Stage-1 selection/materialization/graph reports
v2 graph and graph hash
five cold-fold manifests
common-cohort manifest
per-version per-dataset results
warm full-version table
cold five-fold full-version table
paired A/B report
leakage audit
```

Append the final dataset counts, A/B definition, warm table, cold table, and links
to these artifacts to:

```text
weeks/week6_review_graph_training/review_graph_training.md
```

Preserve historical tables and mark them as belonging to the old dataset
materialization. Do not silently replace old numbers.

## Completion report format

Claude must finish with:

1. files changed and exact commands run;
2. old and new funnel side by side;
3. exact effective counts at `>=3`, `>=10`, and `>=20` models;
4. number of cold datasets in every fold and the complete fold ID/name lists;
5. source, task, metric, modality, language, and candidate-count distributions;
6. duplicate/conflict/embedding/leakage audit counts;
7. representative A/B results before the full sweep;
8. both full-version tables including random and graph-free baselines;
9. paired common-cohort uncertainty and expanded-cohort interpretation;
10. a decision: accept v2, revise collection, or expand the model universe in a
    separately named experiment.

The work is incomplete if it merely raises the number of dataset nodes, reuses
the leaked graph, selects datasets according to downstream results, reports only
one nine-dataset split, or omits the exact reason each dataset was lost.
