# Cold-Dataset Full-Version Evaluation Instructions for Claude

## Mission

Evaluate every Stage-2 version in the deployment condition that matters:

```text
a previously unseen dataset arrives
-> it has no trained_on history
-> its query embedding is built only from deployable dataset features and
   deployable similar_to relations
-> the fixed model embeddings rank models in the candidate pool
```

Produce two comparable full-version tables:

1. **Warm/remaining datasets:** ordinary held-out performance edges belonging only
   to datasets outside the cold-test set.
2. **Cold test datasets:** entire datasets whose performance edges and graph
   presence were unavailable during training.

Both tables must contain exactly:

```text
name | tau_macro | NDCG@1 | Hit@1 | top3_hit@1 | Rec@1 |
NDCG@10 | Hit@10 | Rec@10
```

Include a `Random (expected)` row in each table. Insert both final tables into
`weeks/week6revieweverything/review_everything.md`.

Do not reuse any pre-`dedup_trained_on` metric. The 12,205-row graph contained
duplicate-pair split leakage; all learned versions must run on the 7,056-distinct-
pair graph view.

## 1. Freeze a dataset-ID split before training

Deduplicate `(model_id, dataset_id)` performance pairs first. Then build the set
of evaluable dataset IDs:

```text
at least 3 distinct observed model candidates
non-constant accuracy
all raw dataset fields required by the deployed encoder are available
```

Using one fixed `dataset_split_seed`, select 20% of these dataset IDs as the cold
test set. Stratify as far as the data permits by:

- task type;
- candidate-count bucket: `3-10`, `11-100`, `101+`;
- dataset modality, when available.

Do not choose cold IDs using any model's rank, prediction, gold identity, or
configuration result. Materialize the split once and reuse it for every version.

Write:

```text
stage2TrainGraphSAGE/artifacts/cold_dataset/cold_dataset_split.json
```

The manifest must contain:

```json
{
  "dataset_split_seed": 0,
  "graph_sha256": "...",
  "dedup_policy": "...",
  "n_evaluable_datasets": 0,
  "n_remaining_datasets": 0,
  "n_cold_test_datasets": 0,
  "cold_test_dataset_ids": [],
  "cold_test_dataset_names": [],
  "candidate_count_by_dataset": {},
  "task_count_by_partition": {}
}
```

Before any training, print and place at the top of the final report:

```text
evaluable datasets = N
remaining datasets = N_train_side
cold test datasets = N_cold
cold test dataset IDs = [...]
```

The number of cold datasets must be stated explicitly; do not leave it inferable
from a JSON file.

## 2. Construct the two evaluation partitions

Let the frozen dataset sets be:

\[
\mathcal D_{cold}\cap\mathcal D_{remain}=\varnothing.
\]

### 2.1 Remaining/warm side

On performance edges whose dataset belongs to `D_remain`, apply the existing fixed
train/validation/test edge split. This produces:

```text
remaining train message edges
remaining train ranking-supervision edges
remaining validation edges
remaining warm-test edges
```

The warm table is evaluated only on the remaining warm-test edges. All versions
must use identical candidate IDs.

### 2.2 Cold side

All deduplicated performance records whose dataset belongs to `D_cold` go directly
into an evaluation-label vault:

```text
cold_labels[dataset_id] = [(model_id, accuracy), ...]
```

They must never enter:

- the training message graph;
- reverse `rev_trained_on` messages;
- ranking supervision;
- model-model contrastive membership;
- dataset-model contrastive positives or negatives;
- global-negative construction or hard-negative mining;
- validation or checkpoint selection;
- any feature normalization fitted on training labels.

Do not pass the cold-label lookup into the training function. Load it only inside
the final evaluator after the checkpoint has been selected.

## 3. Build a genuinely inductive training graph

For every `d in D_cold`, remove from the training graph:

```text
all (model, trained_on, d) edges
all (d, rev_trained_on, model) edges
all similar_to edges incident to d
the cold dataset node from sampled training computation
```

Removing only `trained_on` is insufficient: leaving cold nodes connected through
`similar_to` exposes their features during training and is transductive rather
than a faithful new-dataset simulation.

Keep all model nodes. They are the items whose embeddings will be indexed.

Add assertions before training:

\[
E_{trained\_on}(\mathcal D_{cold})=\varnothing,
\qquad
E_{rev\_trained\_on}(\mathcal D_{cold})=\varnothing,
\]

\[
E_{similar\_to}(\mathcal D_{cold},\mathcal D)=\varnothing.
\]

Build `accuracy_lookup`, top-k membership `M`, pair samplers, and every
contrastive/global-negative structure from remaining training-visible edges only.

## 4. Insert each cold dataset only at inference

After training and checkpoint selection:

1. Compute and freeze all indexed model embeddings `z_m` from the training graph.
2. Construct the cold dataset's raw node features using exactly the production
   feature pipeline.
3. Compute its neighbors only from information available when a user submits a
   new dataset.
4. Add directed inference edges that let the cold node receive messages from
   known dataset neighbors.
5. Produce `z_d` without retraining or updating any model parameter.
6. Query the same frozen `z_m` matrix.

The current `similar_to` source is documented as GPT-Neo dataset centroids. Audit
and test that the cold centroid is computable from the new dataset's raw examples
without performance labels. If any similarity feature uses accuracy or hidden
performance records, it is illegal for this evaluation and must be rebuilt.

For PyG edge direction `source -> target`, the cold query needs incoming edges:

```text
(known_training_dataset, similar_to, cold_dataset)
```

Do not add the reverse edge merely for symmetry if doing so changes precomputed
known-dataset or model embeddings. The index is required to remain fixed when a
new query dataset arrives.

Required invariant:

```python
z_m_before = encode_indexed_models(train_graph)
z_d_cold = encode_inserted_dataset(cold_dataset, train_graph)
z_m_after = encode_indexed_models(query_graph)
assert torch.allclose(z_m_before, z_m_after, atol=1e-6)
```

If the current full-graph forward cannot satisfy this cleanly, implement a
query-only cold-dataset encoder rather than silently recomputing the index.

## 5. Candidate-pool rule

These metrics evaluate semantic ordering over candidates with observed evaluation
accuracy. They do not require an unfiltered full-2K search.

For each dataset, define the labeled evaluation candidates:

```text
warm: positive held-out edges from the remaining warm-test split
cold: all deduplicated vaulted performance records for that cold dataset
```

If a production metadata eligibility filter already exists, freeze it before the
experiment and apply it identically to all versions. It may use only deployment-
known fields such as task, modality, language, license, framework, size, download
availability, and fine-tuning support. It may not use test accuracy or gold IDs.

Report separately, outside the two requested tables:

```text
mean/median candidate-pool size
fraction with n <= 1 and n <= 10
filter gold-survival rate, if filtering is enabled
```

Never silently drop a dataset because its gold model was removed by the filter.

## 6. Exact metric definitions

For dataset `d`, let `C_d` be its evaluation candidates, `y_md` normalized
accuracy, and:

\[
s(d,m)=\operatorname{normalize}(z_d)^T
       \operatorname{normalize}(z_m).
\]

Use exact dot-product ranking for the semantic table. HNSW fidelity, if measured,
is a separate systems diagnostic.

### 6.1 Kendall tau

\[
\tau_d=\operatorname{KendallTauB}
\left(\{s(d,m)\}_{m\in C_d},\{y_{md}\}_{m\in C_d}\right),
\qquad
\tau_{macro}=\frac1{|\mathcal D|}\sum_d\tau_d.
\]

Use tau-b so accuracy ties are handled. Require at least three candidates and at
least one non-tied accuracy pair. Give every dataset equal macro weight.

### 6.2 NDCG@K

For `k_d = min(K, |C_d|)` and predicted ordering `pi_d`:

\[
DCG@K_d=\sum_{i=1}^{k_d}\frac{y_{\pi_d(i),d}}{\log_2(i+1)},
\]

\[
NDCG@K_d=\frac{DCG@K_d}{IDCG@K_d}.
\]

`IDCG@K` must use only the best `k_d` true candidates. Do not divide a truncated
`DCG@K` by the DCG of the entire candidate list. The current evaluator's
`ideal_dcg = _dcg(np.sort(a)[::-1])` must therefore be corrected to:

```python
ideal_dcg_k = _dcg(np.sort(a)[::-1][:k])
ndcg_k = _dcg(a[topk]) / ideal_dcg_k
```

### 6.3 Hit@K

Let all accuracy-tied best models be:

\[
B_d=\{m\in C_d:y_{md}=\max_{j\in C_d}y_{jd}\}.
\]

Then:

\[
Hit@K_d=\mathbf1[B_d\cap TopK_s(d)\ne\varnothing].
\]

Do not arbitrarily select one member of a tied-best set as the only gold model.

### 6.4 Rec@K

For these tables, `Rec@K` means recall of the true top-3 models, matching the
repository's existing `recall_top3@K` concept:

\[
Rec@K_d=\frac{|T_d^3\cap TopK_s(d)|}{|T_d^3|}.
\]

Define `T_d^3` as every model whose accuracy is at least the third-highest distinct
cutoff, so ties at the boundary are not broken arbitrarily. If the dataset has
fewer than three candidates, use all candidates. In artifacts, retain the explicit
key `recall_top3@K`; shorten it to `Rec@K` only in the requested presentation
tables.

### 6.5 top3_hit@1

Let the single highest-scoring selected model be:

\[
\hat m_d=\arg\max_{m\in C_d}s(d,m).
\]

Using the same tie-aware true top-3 set `T_d^3` defined above:

\[
top3\_hit@1_d=\mathbf1[\hat m_d\in T_d^3].
\]

This is not the same number as `Rec@1`. `top3_hit@1` gives full credit when the
one selected model is any true top-3 model, whereas:

\[
Rec@1_d=\frac{1}{|T_d^3|}
\]

when the selected model is relevant, and zero otherwise. Report both.

When `|C_d| < K`, use `k_d=min(K,|C_d|)` rather than failing. Because this makes
large-K Hit/Rec trivial for small lists, always publish the small-list fractions
required in Section 5 beside the tables.

## 7. Random baseline

The `Random (expected)` row must use the exact same candidates as the learned
versions, per partition.

- `tau_macro = 0` analytically;
- `top3_hit@1`: analytic probability that one uniformly selected model belongs
  to the tie-aware true top-3 set, `|T_d^3| / |C_d|`;
- `Hit@K` and `Rec@K`: compute analytic expectation under a uniform random
  permutation, respecting `k_d=min(K,n_d)` and tied relevant sets;
- `NDCG@K`: use either an exact permutation expectation or at least 10,000
  deterministic random permutations per dataset with a fixed RNG seed.

Macro-average per-dataset expectations. Do not pool edges across datasets. Store
the random seed and number of permutations in the artifact.

## 8. Versions to rerun

Rerun every distinct configuration below on the same cold split and remaining-edge
split:

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

`ACCEPTED_ranknet` is only the single-seed checkpoint form of `R_mg02`, not a
distinct algorithmic version; do not add a duplicate row unless its weights are
explicitly being audited as a checkpoint artifact.

Use one configuration manifest as the source of truth. Do not reconstruct flags
from memory. Preserve historical epoch counts where they define the version
(`P7_es` and `P7_lr3e3`: 40 epochs); otherwise use the established 25 epochs.

Each version is trained once per declared init seed, then evaluated on both warm
and cold partitions from the same checkpoint. Do not train a separate model for
the cold table.

## 9. Required tests before the sweep

Add focused tests proving:

1. cold, remaining, and validation IDs are deterministic and disjoint;
2. the manifest records the exact cold count, IDs, names, graph hash, and policy;
3. no cold `trained_on` edge exists in training messages;
4. no cold reverse edge exists;
5. no cold supervision edge exists;
6. no cold dataset contributes to `M`, contrastive positives, negatives, or hard
   mining;
7. no cold `similar_to` edge exists during training;
8. inference `similar_to` uses only deployable features;
9. insertion changes `z_d` but leaves every indexed `z_m` unchanged;
10. all versions receive byte-identical warm and cold candidate manifests;
11. `NDCG@K=1` for an ideal ordering at each `K in {1,10}`;
12. tied-best `Hit@K` accepts any tied maximum;
13. `Rec@K` handles the third-place tie cutoff correctly;
14. `top3_hit@1` gives one when rank 1 is in the true top-3 set while `Rec@1`
    gives `1 / |T_d^3|`;
15. random baseline matches analytic Hit/Recall/top3-hit expectations on toy
    examples;
16. held-out labels are loaded only after checkpoint selection;
17. test-edge leakage remains zero after deduplication in both directions.

Do not launch the full sweep until these tests pass.

## 10. Artifacts and final tables

For every version, save:

```text
config and seeds
checkpoint/state_dict hash
split-manifest hash
warm per-dataset metrics
cold per-dataset metrics
candidate IDs and counts
predicted ordering and true accuracies
leakage audit
```

Write aggregate artifacts under:

```text
stage2TrainGraphSAGE/artifacts/cold_dataset/
```

Produce these exact tables:

### Table A — Remaining datasets / warm held-out edges

```text
name | tau_macro | NDCG@1 | Hit@1 | top3_hit@1 | Rec@1 |
NDCG@10 | Hit@10 | Rec@10
```

### Table B — Completely unseen cold dataset IDs

```text
name | tau_macro | NDCG@1 | Hit@1 | top3_hit@1 | Rec@1 |
NDCG@10 | Hit@10 | Rec@10
```

Use mean over datasets; if multiple init seeds are run, show `mean +/- std` and
state whether the std is over initialization or dataset splits. Place
`Random (expected)` first.

Append one new section to:

```text
weeks/week6revieweverything/review_everything.md
```

The section must include:

1. exact cold test count and ID/name list;
2. one paragraph distinguishing warm edge holdout from whole-dataset cold start;
3. candidate-count distribution and any eligibility filter policy;
4. Table A;
5. Table B;
6. a short warm-versus-cold delta interpretation;
7. links to the split manifest and aggregate JSON.

Do not overwrite or silently edit the historical tables. Clearly mark their
pre-dedup results as superseded where appropriate.

## 11. Completion report

Return:

1. files changed;
2. tests and exact outputs;
3. cold dataset count, IDs, and names;
4. leakage audit counts, all required to be zero;
5. both complete tables including random baselines;
6. candidate-count caveats for K=10;
7. warm-to-cold deltas for every version;
8. confirmation that inserting a cold dataset did not change indexed model
   embeddings;
9. exact section/link inserted into `review_everything.md`.

The experiment is incomplete if it reports only the cold table, only aggregate
means without per-dataset artifacts, or a "cold" dataset that remained present in
the training similarity graph.
