# Resource Requirement for the Frozen F7.2 Dense Recommendation Evaluation

## Purpose

F7.2 was designed to answer a question that `gold@10` cannot resolve: whether retrieved models with no historical record on the query dataset are actually useful recommendations. The protocol re-evaluates retrieved unknown candidates, observed calibration candidates, whole-lake random controls, and same-task random controls under a common per-query evaluation procedure.

The scope was generated and frozen on 21 August 2026 before any new evaluation result was observed. The work was subsequently descoped, and the executable manifest and implementation were removed. The counts below remain reproducible from the retained F7/F8 exports, merged supervision, dataset-card table, and the detailed resource audit in `RESOURCE_E_AXIS_DENSE_REEVALUATION.md`.

## Frozen Scope

| Item | Frozen count |
|---|---:|
| De-duplicated test queries across three split seeds | 3,794 |
| Queries resolving to downloadable Hugging Face dataset repositories | 880 (23.2%) |
| Queries retaining at least one evaluable candidate | 864 |
| Queries retaining all four comparison arms | 748 |
| Manifest rows before availability filtering | 36,358 |
| Distinct models referenced | 19,735 |
| Distinct dataset repositories | 586 |
| Evaluable model–query rows | **30,333** |
| Distinct models in evaluable rows | **16,915** |

The 2,914 unresolved queries use benchmark aliases, owner-free names, or private evaluator labels. A manually frozen alias table of approximately 100 common names could raise dataset resolution from 23.2% to approximately 50%, but it would not recover the long tail of one-off identifiers.

## Required Resources

### Compute, Transfer, and Storage

| Resource | Requirement | Evidence status |
|---|---:|---|
| Evaluation jobs | **30,333** model–query evaluations | Measured frozen scope |
| Distinct model acquisitions | **16,915** | Measured frozen scope |
| Model transfer | **73.5 TB lower bound** | Derived from snapshot parameter metadata; missing sizes charged at only 0.3B parameters |
| GPU compute | **Approximately 5,300 GPU-hours** | Estimated from task-family evaluation times |
| Compute risk range | Up to approximately **8,700 GPU-hours** | Generation cost triples for the 7B+ tail |
| Sustained network time at 100 MB/s | **8.5 days minimum** | Assumes continuous full-rate transfer and no rate limiting |
| Scratch storage | Several TB continuously available | Rolling downloads can avoid storing all 73.5 TB simultaneously |
| Result and provenance storage | Tens of thousands of result rows plus logs, configs, and checksums | Small relative to model storage but mandatory for auditability |

The 73.5 TB figure is a lower bound. Only 43% of the selected models publish a parameter count; 2,182 exceed 1B parameters and 1,133 exceed 7B. Anonymous Hugging Face traffic is rate-limited, and gated repositories require separate authorization.

### Task Harnesses

| Task family | Evaluations | Required evaluation path | GPU-hours |
|---|---:|---|---:|
| Embedding / MTEB-style | 11,831 | Standardized MTEB task execution and metric normalization | 1,577 |
| Fine-tuning or unclassified | 11,001 | A frozen adaptation protocol and recalibrated controls | 1,834 |
| Generation | 5,040 | Dataset-specific prompts, decoding parameters, stopping rules, and metrics | 1,680 |
| ASR | 2,461 | Audio loading, transcription, normalization, and WER | 205 |
| Total | **30,333** | — | **Approximately 5,300** |

The fine-tuning family cannot be replaced silently with a linear probe because that would define a different metric from the historical records. The generation family requires per-dataset decoding decisions and is the largest compute uncertainty.

### Engineering and Access

The execution requires resources beyond GPU allocation:

- a resumable model and dataset acquisition service with checksum validation;
- Hugging Face credentials and approval for 14 gated dataset repositories plus gated models;
- per-family containers with pinned model, tokenizer, audio, dataset, and metric dependencies;
- failure classification for missing code, incompatible revisions, custom model implementations, and out-of-memory cases;
- result schemas that retain query, candidate role, model revision, dataset revision, metric direction, hardware, and failure reason;
- job scheduling, caching, retry limits, and rolling deletion policies;
- manual review of approximately 100 dataset aliases before evaluation begins;
- domain decisions for generation and adaptation protocols;
- quality control that verifies all four arms use the same metric and preprocessing for a query.

## Minimum Defensible Partial Scope

ASR and embedding tasks have the most standardized existing harnesses. Restricting F7.2 to those families still requires:

| Item | Requirement |
|---|---:|
| Evaluations | **14,292** |
| Queries with all four arms | **367** |
| Model transfer | **19.7 TB** |
| GPU compute | **Approximately 1,780 GPU-hours** |

This partial scope would provide a controlled result for two task families. It would not support a full-lake recommendation-quality claim for generation, fine-tuning, or unresolved datasets.

## Elapsed-Time Implication

Under ideal continuous utilization, the full estimated workload requires:

| Dedicated allocation | Compute-only elapsed time |
|---|---:|
| 1 GPU | Approximately 7.3 months |
| 4 GPUs | Approximately 1.8 months |
| 8 GPUs | Approximately 27 days |

These durations exclude model download, queueing, dependency failures, gated access, retries, and engineering time. Network and storage remain independent bottlenecks even if additional GPUs are available.

## What Existing Artifacts Establish Without This Budget

The retained zero-cost audits show that approximately 93% of top-10 results have no record on the query dataset, while only a small minority are known to be below the historical best. The share of recommendations with same-task evidence elsewhere is 17.2%–22.4%, compared with 0.032%–0.052% for whole-lake random sampling and 1.95%–3.36% for random sampling from evaluated models.

These controls establish non-random task relevance. They do not establish actual performance of the unobserved recommendations on the target query.

## Resource-Gap Statement

F7.2 is not blocked by the retrieval implementation. It is blocked by evaluation infrastructure and sustained access to **tens of terabytes of model data, thousands of GPU-hours, multiple task-specific harnesses, gated assets, and continuous engineering support**. The current project can report historical-record recovery and zero-cost relevance controls. It cannot make a controlled full-lake claim about the true quality of unobserved recommendations without the above resources.

