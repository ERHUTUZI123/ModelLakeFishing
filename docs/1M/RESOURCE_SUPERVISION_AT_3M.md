# Resource Requirement for Performance Supervision at 3M Scale

## Purpose and Boundary

This document estimates the resources required if collapse mitigation relies on acquiring new model–dataset performance labels across the 3.016M-model lake. It is separate from F7.2: F7.2 evaluates the quality of a frozen recommendation set, whereas this work changes the training signal available to the retriever.

No experiment has established a label-count threshold that guarantees the absence of representation collapse. The estimates below are explicit coverage scenarios, not measured acceptance thresholds. Collapse also depends on the objective, dataset representation, shared projections, and lineage propagation. Additional labels are one possible anti-collapse signal, not the only possible remedy.

## Measured Starting Point

| Quantity | Current full-lake value |
|---|---:|
| Model candidates | **3,016,439** |
| Supervision edges after six-source merging and capping | **247,803** |
| Models with nonzero supervision degree | **46,146 (1.53%)** |
| Models with zero supervision degree | **2,970,293 (98.47%)** |
| Train-visible supervised models by split seed | 35,991–39,930 (1.19%–1.32%) |
| `warm` models with supervision degree at least 10 | **2,949 (0.10%)** |
| 100K reference: models with train-visible labels | **25,379 / 100,000 (25%)** |

The current graph has approximately 5.37 supervision edges per supervised model, but the distribution is concentrated and most models receive none. Adding more labels to already warm models would increase edge count without addressing long-tail collapse. New supervision must reach currently `cold` and `frozen` models and must be balanced across tasks, datasets, families, and lineage hubs.

## Coverage Scenarios

Two planning targets are used:

1. **25% coverage:** parity with the supervised-model share of the 100K experiment;
2. **50% coverage:** a stronger scenario in which half of the full lake receives at least one performance label.

| Target coverage | Target supervised models | Additional models requiring labels |
|---|---:|---:|
| 25% | 754,110 | **707,964** |
| 50% | 1,508,220 | **1,462,074** |

One label per newly covered model is only a connectivity floor. It does not identify cross-dataset transfer quality or support robust ranking. Three labels per model provide limited cross-query evidence; ten labels approximately reach the current `warm` degree threshold.

| Coverage target | Labels per newly covered model | New model–query evaluations | Total supervision edges after addition |
|---|---:|---:|---:|
| 25% | 1 | 707,964 | 955,767 |
| 25% | 3 | **2,123,892** | 2,371,695 |
| 25% | 10 | 7,079,640 | 7,327,443 |
| 50% | 1 | 1,462,074 | 1,709,877 |
| 50% | 3 | **4,386,222** | 4,634,025 |
| 50% | 10 | 14,620,740 | 14,868,543 |

The 200-edge cap per dataset–task node must remain in force. A multi-million-edge supervision program therefore also requires many distinct dataset–task nodes; it cannot obtain useful coverage by repeatedly evaluating models on a small number of large benchmarks.

## Resource Extrapolation

The available cross-family planning baseline is the frozen F7.2 workload. Its workload counts are measured, its transfer figure is a metadata-derived lower bound, and its GPU cost is estimated from task-family runtimes:

- 30,333 evaluations are estimated to require approximately 5,300 GPU-hours, or **0.175 GPU-hours per evaluation** on average;
- 16,915 distinct models produce a **73.5 TB transfer lower bound**, or approximately **4.35 GB per distinct model**;
- missing model sizes were charged at only 0.3B parameters, so the transfer value is conservative.

Applying that measured task mix gives the following planning estimates for three labels per newly covered model:

| Coverage target | New evaluations | Model-transfer lower bound | Estimated GPU-hours | 8 dedicated GPUs | 64 dedicated GPUs |
|---|---:|---:|---:|---:|---:|
| 25% | **2,123,892** | **3.08 PB** | **371,000** | **5.3 years** | **242 days** |
| 50% | **4,386,222** | **6.35 PB** | **766,000** | **10.9 years** | **499 days** |

The sensitivity to label depth is substantial:

| Coverage target | 1 label/model | 3 labels/model | 10 labels/model |
|---|---:|---:|---:|
| 25% | 124,000 GPU-hours | **371,000 GPU-hours** | 1,237,000 GPU-hours |
| 50% | 255,000 GPU-hours | **766,000 GPU-hours** | 2,555,000 GPU-hours |

These are planning estimates, not measured full-lake costs. They assume the average F7.2 task mix, perfect hardware utilization, no queueing, and one model download amortized across its labels. They exclude failed models, retries, preprocessing, fine-tuning checkpoints, gated access delays, and engineering time. Task distributions with more generation or adaptation work would increase the estimate.

## Required Infrastructure

### Compute and Data Movement

- a sustained multi-GPU allocation measured in hundreds of thousands of GPU-hours;
- petabyte-scale cumulative model transfer, even when local storage is rolling;
- high-bandwidth network access that is not constrained by anonymous Hub rate limits;
- tens of terabytes of scratch space for concurrent models, datasets, caches, and checkpoints;
- a result store capable of tracking millions of evaluations and their provenance;
- checkpointing and retry infrastructure so partial failures do not invalidate long campaigns.

### Benchmark Coverage

- a frozen mapping from dataset aliases to canonical repositories;
- downloadable and licensed benchmark data across embedding, generation, ASR, classification, retrieval, and adaptation tasks;
- standardized train/validation/test splits and metric direction;
- per-task inference or adaptation protocols;
- calibration candidates and random controls for each protocol;
- explicit handling of datasets that cannot be resolved or legally redistributed.

Only 23.2% of the frozen F7.2 test queries currently resolve directly to downloadable Hugging Face dataset repositories. Approximately 100 curated aliases could raise that value to about 50%, but the remaining long tail requires manual provenance work or must be excluded.

### Label Quality and Sampling Design

The labeling campaign must prevent new concentration from reproducing the existing problem:

- prioritize models with zero supervision degree;
- stratify by model family, task, modality, parameter scale, age, popularity, and lineage degree;
- oversample large lineage hubs while preventing a single hub from dominating;
- preserve the 200-edge dataset–task cap;
- retain failed evaluations as explicit outcomes instead of silently replacing candidates;
- normalize only within compatible metric and protocol groups;
- audit author-reported, automatically generated, and independently reproduced labels separately;
- freeze sampling and conflict-resolution rules before results are observed.

### Personnel and Operational Support

The program requires continuing work from research engineering, benchmark-domain owners, data engineering, and cluster operations. Major tasks include harness maintenance, custom-model compatibility, gated access, metric validation, failure triage, data governance, and provenance review. GPU allocation alone is insufficient.

## What This Budget Would and Would Not Establish

A successful campaign would substantially increase the number of models receiving query-specific ranking gradients and would make same-hub and long-tail contrastive sampling possible. It could test whether supervision coverage reduces the observed 3–5-dimensional output geometry.

It would not guarantee that collapse disappears. The current audit also identifies a low-rank dataset-query space, missing dataset categorical information, biased shared projections, no active unlabeled-uniformity term, and topology-only lineage messages. A controlled retraining study is still required to isolate the effect of added labels.

Conversely, algorithmic mitigation can be tested without acquiring millions of labels. The proposed six-configuration study with 3 split seeds and 3 initialization seeds requires 54 full-lake training runs. At the measured F6 duration of approximately seven minutes per run, pure H200 training time is approximately 6–8 GPU-hours, excluding implementation, export, and evaluation. Such experiments may reduce geometric collapse, but they cannot establish the true performance of previously unobserved recommendations.

## Resource-Gap Statement

Matching only the 100K experiment's 25% supervised-model coverage with three labels per newly covered model requires approximately **2.12 million new evaluations, 3.08 PB of cumulative model transfer, and 371,000 GPU-hours** under the available measured cost model. This is an industrial-scale data-production program rather than an extension of the current experiment allocation.

The present project can afford controlled objective and architecture ablations. It cannot independently produce performance labels at the scale required to turn a 3.016M-model lake into a densely supervised benchmark. Any claim about eliminating collapse through additional supervision must therefore be presented as resource-contingent and currently unverified.
