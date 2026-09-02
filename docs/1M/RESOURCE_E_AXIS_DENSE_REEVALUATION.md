# Resource requirement: dense re-evaluation of retrieved models (the "E axis")

Companion document: [`RESOURCE_SUPERVISION_AT_3M.md`](RESOURCE_SUPERVISION_AT_3M.md).
Design rationale: [`F7.5guide.md`](F7.5guide.md). Measurements: [`F7.md`](F7.md), [`F8.md`](F8.md).

Status: the evaluation manifest for this work was built and frozen on 2026-08-21,
then the axis was descoped and its code and manifest were removed. Every number
below was measured before removal and is recomputable from artifacts that still
exist (`data1m/exports_rf/`, `rf/canon/supervision_merged.parquet`,
`datasets_full/dataset_cards_merged.parquet`). Nothing here is an estimate
unless it is labelled as one.

---

## 1. What the work would answer, and why it is not optional

The A axis measures whether the retriever recovers the model that historical
records say is best on a query dataset. On the full lake it reports
`gold@10 = 0.060` averaged over three split seeds.

That number counts every retrieved model that is not the historical winner as a
miss. An audit of what the retriever actually returns shows that this is mostly
an artefact of missing annotation rather than a record of wrong picks:

| Composition of the top-10, per split seed | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| Historical best on this dataset | 1.98% | 1.32% | 1.84% |
| Observed on this dataset, and worse | 2.78% | 4.67% | 2.34% |
| No record on this dataset, but evaluated elsewhere | 92.20% | 90.73% | 95.17% |
| No record anywhere in the lake | 3.05% | 3.28% | 0.65% |

Only about 3.3% of what the A axis counts as a miss is a model we can show to be
inferior. Roughly 93% has no record on that dataset at all, and over nine tenths
of those do carry evaluation records elsewhere, so they are not obscure.

A second, zero-cost control says the retriever is not returning noise. For each
query `(dataset, task)`, the share of returned models that have at least one
record on the same task somewhere else:

| | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| Our top-10 | 22.4% | 22.0% | 17.2% |
| 10 models drawn uniformly from the lake | 0.032% | 0.052% | 0.038% |
| 10 models drawn from the evaluated subset only | 3.36% | 2.48% | 1.95% |

That is roughly 500 times the whole-lake baseline and 7 to 9 times the harder
baseline, which rules out "it only favours models that happen to have been
evaluated". It does not show that the returned models are good; only a fresh
evaluation can show that.

So the claim the E axis would establish is: **the sparse historical gold
systematically understates retrieval quality, by a measured amount.** Without it
the paper has to report `gold@10 = 0.060` with no way to separate annotation
coverage from recommendation error.

---

## 2. Scope, and the ceiling that is not ours to choose

The protocol takes every de-duplicated test query across the three split seeds
and re-evaluates four groups of candidates per query: the retrieved models with
no record on that dataset, the query's own observed candidates as calibration, a
uniform random control, and a same-task random control. There is no query-level
sampling — picking a few cases and re-testing them cannot rule out selection
bias.

The scope has a hard ceiling that comes from the data, not from a design choice:

| | Test queries | Dataset name resolves to a real HF repo | Share |
|---|---:|---:|---:|
| ASR | 137 | 72 | 52.6% |
| Generation | 413 | 179 | 43.3% |
| Embedding (MTEB-style) | 873 | 300 | 34.4% |
| HELM | 510 | 0 | 0.0% |
| Other | 1,861 | 329 | 17.7% |
| Total | 3,794 | 880 | 23.2% |

The remaining 2,914 queries name something that cannot be downloaded: an
author's own evaluator label (`dim_128`, `dim_512`, `eval`, `custom`,
`all-nli-test-turkish`) or a benchmark name with no owner prefix. This is the
same fact that F1.5 measured from the other side, where only 18.28% of query
nodes matched an HF dataset card.

The unresolvable names are concentrated: 2,914 queries use only 1,876 distinct
names, and the top ten account for 23.9% of them.

| Name | Queries | Where it actually lives |
|---|---:|---|
| `flores200-devtest` | 245 | `facebook/flores` (gated) |
| `tatoeba-test-v2021-08-07` | 180 | `Helsinki-NLP/tatoeba_mt` |
| `ntrex128` | 172 | `davidstap/NTREX` |
| `newstest2012`, `news-test2008`, … | 10–16 each | WMT series |
| `hellaswag`, `gsm8k`, `winogrande`, `mmlu` | 9–14 each | `Rowan/hellaswag`, `openai/gsm8k`, … |

A curated alias table of roughly 100 entries would raise coverage from 23.2% to
about 50%. It is cheap, but it has to be written and frozen before any fresh
number is seen, otherwise it becomes a way of choosing datasets by their results.
A further 1,668 names occur exactly once and are not recoverable at any
reasonable effort.

---

## 3. Measured size of the frozen manifest

| | Count |
|---|---:|
| Queries in scope | 880 |
| Queries excluded, with a recorded reason | 2,914 |
| Manifest rows, all four candidate roles | 36,358 |
| Distinct models referenced | 19,735 |
| Distinct HF dataset repositories | 586 |

By role:

| Role | Rows |
|---|---:|
| Retrieved, unobserved on that dataset | 8,673 |
| Calibration (the query's own observed candidates) | 10,159 |
| Random control, whole lake | 8,800 |
| Random control, same-task pool | 8,726 |

---

## 4. What of the manifest can actually be obtained

The dataset side was checked live against the HF API on 2026-08-21. All 586
repositories are reachable; 572 are public and ungated. Fourteen are gated and
need per-repository authorisation, among them `facebook/flores`,
`ilsvrc/imagenet-1k` and `fsicoli/common_voice_18_0`.

The model side was read from the 2026-08-18 snapshot, which carries `gated`,
`private` and `disabled` for every model in it.

| Role | Obtainable | Gated | Gone from HF |
|---|---:|---:|---:|
| Random control, whole lake | 8,615 | 151 | 34 |
| Calibration | 9,466 | 160 | 533 |
| Random control, same-task pool | 6,543 | 94 | 2,089 |
| Retrieved, unobserved | 6,168 | 193 | 2,312 |

Counting a row as evaluable only when both the model and the dataset can be
obtained:

| Role | Evaluable / total | Share |
|---|---:|---:|
| Random control, whole lake | 8,459 / 8,800 | 96.1% |
| Calibration | 9,397 / 10,159 | 92.5% |
| Random control, same-task pool | 6,430 / 8,726 | 73.7% |
| Retrieved, unobserved | 6,047 / 8,673 | 69.7% |

The failure rate depends on the method, and by a large margin. 26.7% of our
top-10 no longer exists on HuggingFace against 0.4% of the weak control. The
cause is identifiable and is not a defect of the retriever: the model learns to
return models that carry supervision, and about a fifth of the supervised models
came from historical corpora and have since been deleted from the Hub. The
same-task control shows a 23.9% disappearance rate, the same order as ours, so
what correlates is membership of the supervised pool rather than selection by
our retriever. This is reported as measured, without replacement and without
averaging across methods.

At the query level, 864 of the 880 queries retain at least one evaluable
candidate, and **748 (85.0%) still have all four roles populated**, so the
controlled comparison is possible on 748 queries.

---

## 5. The resource requirement

### 5.1 Measured quantities

| Family | Evaluable rows | Queries | Distinct models | Download (measured lower bound) |
|---|---:|---:|---:|---:|
| Embedding (MTEB-style) | 11,831 | 298 | 5,047 | 16.8 TB |
| Needs fine-tuning or unclassified | 11,001 | 322 | 7,313 | 29.2 TB |
| Generation | 5,040 | 175 | 3,728 | 30.2 TB |
| ASR | 2,461 | 69 | 1,949 | 2.9 TB |
| Total | **30,333** | 864 | **16,915** | **73.5 TB** |

The download figure is a lower bound: only 43% of these models publish a
parameter count, and the rest are charged at 0.3 B parameters. 2,182 of the
models exceed 1 B parameters and 1,133 exceed 7 B.

### 5.2 Estimated compute

These are estimates. The basis for each per-evaluation figure is stated so it
can be argued with.

| Family | Evaluations | Minutes per evaluation (estimate) | Basis | GPU-hours |
|---|---:|---:|---|---:|
| Embedding | 11,831 | 8 | one `mteb` task on a sub-1 B encoder over a few-thousand-item corpus | 1,577 |
| Needs fine-tuning | 11,001 | 10 | encode the split, then fit a linear probe | 1,834 |
| Generation | 5,040 | 20 | decoding dominates; the 7 B+ tail is far worse | 1,680 |
| ASR | 2,461 | 5 | transcribe a test split, compute WER | 205 |
| Total | 30,333 | — | — | **≈ 5,300** |

The generation row is the least trustworthy. 1,133 models above 7 B parameters
sit mostly in that family, and if their decoding cost is three times the
estimate the total moves to roughly 8,700 GPU-hours.

### 5.3 What that means in wall-clock

| Allocation | Elapsed |
|---|---|
| 1 dedicated GPU, 24/7 | ≈ 7.3 months |
| 4 dedicated GPUs | ≈ 1.8 months |
| 8 dedicated GPUs | ≈ 27 days |

For reference, one GPU held continuously for a year is 8,760 GPU-hours.

Network is a separate constraint. 73.5 TB at a sustained 100 MB/s is 8.5 days of
continuous transfer, and HuggingFace rate-limits anonymous traffic. Storage can
be rolling rather than cumulative if models are deleted after use, but a
few terabytes of working space is needed throughout, plus the results table.

### 5.4 The cheapest useful subset

Not all of it has to be done at once, and the families are not equally
expensive per unit of conclusion. ASR and the embedding family both have an
off-the-shelf harness — transcribe-and-WER, and one `mteb` command per
(model, task) — and together they are:

| | Evaluations | Queries | Download | GPU-hours (estimate) |
|---|---:|---:|---:|---:|
| ASR + embedding only | 14,292 | 367 | 19.7 TB | ≈ 1,780 |

That subset alone yields a complete E-axis result on 367 queries with all four
control arms. The generation family needs per-dataset decoding parameters, and
the "needs fine-tuning" family needs a prior decision about what protocol
replaces fine-tuning — a linear probe measures a different quantity than the
historical fine-tuned accuracy, so the calibration arm would have to be
re-measured under the same protocol and the result reported as a new metric
rather than compared with the historical numbers.

---

## 6. What we have instead, at zero cost

The two audits in §1 were computed from artifacts already on disk and cost
nothing. They establish that the A axis is measuring annotation coverage as much
as recommendation error, and that the retriever is far above both random
baselines on task relevance.

What they cannot establish is the quantity the paper would most want: whether
the retrieved, unlabelled models are actually good. That requires running them,
and the cost of running them is §5.

---

## 7. Summary for a resourcing conversation

- The work is well-defined and the manifest was frozen before any result was
  seen, so the protocol is not open to being tuned after the fact.
- Its maximum reach is 880 of 3,794 test queries (23.2%), because the remaining
  queries name datasets that do not exist as downloadable repositories. About
  100 curated aliases would take that to roughly 50%.
- The full run is about **30,000 evaluations, 16,915 model downloads, 73.5 TB of
  transfer and an estimated 5,300 GPU-hours** — roughly seven months of one
  dedicated GPU, or 27 days of eight.
- A defensible partial run (ASR plus the embedding family) is about **14,300
  evaluations, 19.7 TB and an estimated 1,780 GPU-hours**, and yields a complete
  result on 367 queries with all four control arms.
- 19.1% of the models involved have already been deleted from HuggingFace, and
  the deletion rate is method-dependent (26.7% for our retrieved candidates
  against 0.4% for the uniform control). That is a permanent, unrecoverable loss
  and has to be reported rather than patched over.
