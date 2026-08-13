You need to redesign and rerun the Hugging Face model collection stage for the 1M Model Lake project.

The current crawl is technically correct but scientifically unsuitable. It collected the first 150,000 models under a global `downloads-desc` ordering. This caused severe distribution bias, including author concentration, quantization mirrors, dominant tasks, excessive near-duplicates, and poor family coverage.

Do not treat “successfully crawled 150K records” as success. The objective is to construct a representative, family-aware, task-balanced, English-primary HALO population suitable for graph construction, representation learning, retrieval evaluation, and eventual scaling toward 1M models.

## Existing context

Repository root:

```text
D:\research\model_lake\codes\ModelLakeFishing
```

Current raw crawl:

```text
D:\research\model_lake\data\data1m\raw\
```

Relevant files:

```text
scale1m/hf_crawl.py
scale1m/verify_raw.py
scale1m/tests/test_hf_crawl.py
docs/1M/T2.md
```

The current crawl contains:

```text
150,000 HF model records
3 frozen shards
downloads-desc ordering
0 duplicate IDs
0 ordering violations
```

However, this dataset must not be accepted as the final HALO. Preserve it as a candidate-discovery head shard or audit artifact. Do not destructively overwrite it.

## Primary objective

Replace the current:

```text
global downloads-desc
→ first 150,000 models
→ freeze
```

with:

```text
broad multi-query candidate discovery
→ metadata enrichment
→ family and near-duplicate canonicalization
→ stratified quota allocation
→ balanced selection
→ targeted deficit backfill
→ frozen final shards
```

Crawling and final sample selection must become separate stages.

The final selected population should:

1. prioritize coverage of important and sufficiently large model families;
2. retain diversity within each family;
3. prevent any family, author, quantization publisher, task, language bucket, or model format from dominating;
4. be primarily English and English-compatible multilingual;
5. maintain broad and reasonably balanced task coverage;
6. include head, mid-popularity, long-tail, and recent models;
7. preserve enough metadata and lineage structure for later graph construction;
8. remain deterministic, resumable, auditable, and reproducible.

## Step 1 — Audit before implementation

First inspect:

```text
scale1m/
docs/1M/T2.md
the current 1M plan and all sections modified by T2
the current raw shard schemas
existing CORE/HALO definitions
existing descriptor construction
existing family, lineage, task, language, size, author, and quantization utilities
```

Reuse existing utilities where correct. Do not create incompatible duplicate definitions without necessity.

Before changing code, summarize:

* why global `downloads-desc` is unsuitable;
* which observed biases exist in the current 150K;
* which metadata fields are actually available from the HF API;
* which fields require `expand[]`;
* which family and lineage signals are reliable;
* which signals are heuristic and must be marked as such.

## Step 2 — Build a broad candidate pool

Do not rely on a single global ordering.

Create a candidate-discovery stage combining multiple query sources, including where supported:

```text
global downloads-desc
global likes-desc
recently created or recently updated
task-specific downloads-desc
task-specific likes-desc
language-specific queries
major architecture or library queries
multilingual model queries
underrepresented-task queries
```

Target a candidate pool substantially larger than the final sample, preferably 400K–800K unique model IDs if API and storage costs remain reasonable.

The exact candidate count may be adjusted based on measured coverage, but justify the decision with data.

Each candidate must retain provenance such as:

```text
discovery_query
discovery_rank
discovery_source
first_seen_query
all_matching_queries
crawl_timestamp
```

A model discovered by multiple queries must appear once, with merged provenance.

The old 150K crawl may be incorporated as one candidate source, but it must not directly determine final inclusion.

## Step 3 — Normalize and annotate candidates

For every candidate, derive or retain at least:

```text
model_id
author
downloads
likes
created_at
last_modified
pipeline_tag
raw_tags
library_name
architectures
base_model
base_model_relation
safetensors metadata
parameter count where legitimately available
language metadata
model-card availability
family_id
canonical_root_id
supertask
language_bucket
source_type
quantization_type
popularity_stratum
metadata_quality_score
near_duplicate_key
mirror_or_conversion_score
selection_provenance
```

Do not use repository-name regex as an unmarked substitute for authoritative metadata.

Name-based inference may only be used as a clearly labeled heuristic when no stronger signal exists. It must never silently overwrite authoritative metadata.

## Step 4 — Model family construction

The final sampler must operate primarily at the family level rather than treating every repository as independent.

Construct `family_id` using the strongest available signals in this order:

```text
1. explicit base_model and base_model_relation metadata
2. model-card parent or derivative declarations
3. known adapter or quantization relationship
4. architecture plus normalized repository stem
5. author plus normalized repository stem
6. unresolved singleton
```

Keep the confidence and source of every family assignment:

```text
family_assignment_source
family_assignment_confidence
```

Do not collapse unrelated models merely because their names share generic tokens.

Examples of repositories that should normally be linked under a common lineage or family include:

```text
base model
official instruct/chat derivative
task-specific fine-tune
adapter or LoRA
AWQ/GPTQ/GGUF conversion
other declared quantized derivative
```

However, preserve their distinct model IDs and source types.

## Step 5 — Near-duplicate control

Create a robust near-duplicate key using fields such as:

```text
canonical root
parameter scale
task
language
quantization method
quantization level
architecture
source type
normalized repository stem
```

Do not allow hundreds of almost identical quantizations or conversions to consume the sample.

Suggested constraints:

```text
family × task × parameter scale × format:
maximum 3 retained representatives

same base model × same quantization method × same bit level:
maximum 1–2 retained representatives
```

Within equivalent groups, prefer models with:

```text
better metadata
clearer parent relationship
higher legitimate usage
official or original provenance
more complete model cards
greater relevance to an underfilled bucket
```

## Step 6 — Task normalization and balancing

Map raw HF pipeline tags and related metadata into a stable supertask taxonomy.

At minimum include:

```text
text generation / chat
embedding / retrieval / reranking
text classification
token classification / NER
question answering
summarization
translation
code
speech recognition
text-to-speech / audio generation
audio classification
image classification
object detection / segmentation
image generation
vision-language / multimodal
tabular / time series / reinforcement learning / other
unknown
```

Store both the original task metadata and the normalized supertask.

Do not make all task quotas exactly equal. Use bounded allocation:

```text
no single major supertask > 15% of final selected models
candidate-rich main tasks should normally receive at least 4%
candidate-poor tasks should be retained as fully as practical
unknown task should be tightly capped
remaining quota should scale approximately with sqrt(candidate_count)
```

A suitable baseline is:

[
q_t \propto \sqrt{N_t}
]

subject to minimum and maximum task bounds.

Explain and record the exact allocation algorithm.

Task balancing must be family-aware and author-aware. A nominally balanced task bucket is not acceptable if one family or publisher dominates it.

## Step 7 — Language normalization and balancing

Create the following language buckets:

```text
English-primary
multilingual including English
non-English-specific
language-neutral
unknown
```

Do not classify visual, pure-audio, time-series, or other non-linguistic models as language unknown when they are genuinely language-neutral.

Use authoritative metadata first:

```text
cardData.language
HF language tags
model-index metadata
README front matter
other structured model-card metadata
```

Repository-name inference may only be a last-resort labeled heuristic.

Target distribution:

```text
English-primary: approximately 55–60%
multilingual including English: approximately 18–22%
non-English-specific: approximately 10–15%
language-neutral: approximately 8–12%
unknown: no more than 3%
```

Hard requirement:

```text
English-primary + multilingual including English >= 75%
```

Task-specific exceptions are allowed where justified, especially for translation, multilingual retrieval, and language-specific evaluation coverage.

## Step 8 — Family-size stratification

Prioritize meaningful family structure without allowing large families to monopolize the lake.

Create family-size strata such as:

```text
large family: at least 50 candidate members
medium family: 10–49
small family: 2–9
singleton: 1
```

Use an initial target similar to:

```text
large families: 35%
medium families: 30%
small families: 20%
singletons: 15%
```

Adjust only if the measured candidate pool makes these targets unreasonable. Any adjustment must be documented quantitatively.

Large families should receive priority for inclusion, but selected members must represent meaningful internal variation:

```text
base models
official derivatives
different parameter sizes
important task fine-tunes
different language specializations
a limited number of representative quantizations
adapters where useful
```

Do not simply select the most downloaded 500 members of a family.

Suggested hard cap:

```text
single family maximum: 300–500 selected repositories
```

The cap should also be expressed as a maximum share of the final population.

## Step 9 — Author and publisher concentration control

The current observation that `mradermacher` alone represents approximately 20.1% of HALO is unacceptable for the final selected population.

Implement generic concentration controls rather than hard-coding one username.

Suggested gates:

```text
ordinary single author share <= 1.5%
top-10 authors combined <= 10%
high-volume mirror or quantization publisher share <= 0.5%
single author within one task bucket <= 5% of that bucket
```

Develop an auditable mirror or conversion score based on signals such as:

```text
very high repository count
large fraction of GGUF/GPTQ/AWQ conversions
highly templated model cards
many unrelated upstream base authors
low original-model ratio
repetitive repository-name patterns
```

Do not automatically delete all quantization publishers. Retain a bounded, representative sample.

Output concentration reports before and after selection.

## Step 10 — Source-type balancing

Classify models into:

```text
original/base
official derivative
community fine-tune
adapter/LoRA
quantized/conversion
unknown derivative
```

Use initial target ranges such as:

```text
original/base: 25–30%
official derivative: 20–25%
community fine-tune: 25–30%
adapter/LoRA: 8–12%
quantized/conversion: 8–12%
unknown: <= 5%
```

Quantized and conversion repositories must not exceed 15% without an explicit documented reason.

Do not force an impossible quota when metadata cannot reliably distinguish categories. In that case, report uncertainty rather than fabricating classifications.

## Step 11 — Popularity and temporal diversity

Do not use downloads as the sole ranking signal.

Within each task × language × family-size region, divide candidates into popularity strata such as:

```text
head: 25%
mid: 40%
long tail: 25%
recent or emerging: 10%
```

Compute popularity relative to a relevant task or stratum rather than globally.

A possible scoring function is:

[
p(m)=
0.55\log(1+\text{downloads})
+0.20\log(1+\text{likes})
+0.15\cdot\text{recency}
+0.10\cdot\text{metadata quality}
]

You may modify the coefficients after inspecting real distributions, but document the final formula.

Recent models must not all be low-quality noise. Apply metadata and duplication constraints before granting a recency boost.

## Step 12 — Metadata-quality gate

Create an explicit `metadata_quality_score`.

Possible positive signals:

```text
recognized task
recognized architecture or library
model card exists
structured language metadata
structured base-model relationship
safetensors or model-index metadata
valid repository contents
clear source type
```

Possible negative signals:

```text
empty repository
missing task, architecture, description, and model files
broken or deleted repository
pure mirror with no upstream attribution
unresolvable automated conversion
contradictory metadata
```

Suggested requirement:

```text
at least 95% of selected models pass the main metadata-quality threshold
```

Allow explicitly marked exceptions for rare or underrepresented tasks:

```text
low_metadata_exception = true
exception_reason
```

## Step 13 — Selection algorithm

Do not sort all models by one scalar score and take the first N.

Implement a quota-aware weighted round-robin or constrained selection algorithm.

It must continuously enforce:

```text
task quota
language quota
family-size quota
source-type quota
popularity-stratum quota
family cap
author cap
mirror-publisher cap
near-duplicate cap
metadata-quality gate
```

A model-level score may be used within eligible buckets:

[
S(m)=
w_fF(m)
+w_tT(m)
+w_lL(m)
+w_qQ(m)
+w_pP(m)
-w_aA(m)
-w_dD(m)
-w_rR(m)
]

where terms may represent:

```text
family importance
task rarity
language fit
metadata quality
popularity
author concentration penalty
duplicate penalty
mirror or conversion penalty
```

However, the hard quotas and caps must take precedence over the scalar score.

Make random tie-breaking deterministic through a recorded seed.

## Step 14 — Deficit-driven backfill

After the first selection pass, generate a quota-deficit report.

Examples:

```text
translation missing 2,000 models
audio classification missing 800
multilingual including English missing 1,500
medium families missing 3,000
original/base models below target
```

Then issue targeted HF queries for deficient regions.

Do not fill deficits by simply continuing farther down the global downloads ranking.

Repeat:

```text
discover
deduplicate
annotate
select
measure deficits
targeted backfill
```

until either:

1. all hard gates pass; or
2. the API candidate supply is demonstrably insufficient.

When insufficient, report the actual available supply and adjust the quota transparently.

## Step 15 — Required exit gates

Add or update verification gates for the final selected HALO.

At minimum include:

```text
G1  duplicate model IDs = 0
G2  near-duplicate cap violations = 0
G3  ordinary largest-author share <= 1.5%
G4  top-10 authors combined share <= 10%
G5  each high-volume mirror/conversion publisher <= 0.5%
G6  largest-family share within the approved cap
G7  quantized/conversion share <= 15%
G8  no major supertask share > 15%
G9  candidate-rich main tasks normally >= 4%
G10 English-primary + multilingual-English >= 75%
G11 unknown language <= 3%
G12 unknown task within an explicitly justified small cap
G13 task quota deviation <= 5% where supply permits
G14 language quota deviation <= 5% where supply permits
G15 family-size-stratum quota deviation <= 5% where supply permits
G16 metadata-quality pass rate >= 95%
G17 each derivative has a parent/root or an explicit unresolved flag
G18 selected model count matches the configured target
G19 deterministic rerun reproduces identical selected IDs
G20 all selected records include selection provenance
```

Global downloads ordering is no longer an exit gate for the final dataset.

Ordering may only be verified inside discovery results or popularity strata where it is meaningful.

## Step 16 — Distribution diagnostics

Produce machine-readable and Markdown reports containing:

```text
author HHI
family HHI
task entropy
language entropy
source-type entropy
effective number of authors
effective number of families
top-1, top-10, and top-100 author shares
top-1, top-10, and top-100 family shares
family-size distribution
task distribution
language distribution
source-type distribution
popularity-stratum distribution
quantization distribution
metadata-quality distribution
unresolved-family rate
near-duplicate rejection counts
quota-deficit history
```

Show side-by-side comparisons between:

```text
current downloads-desc 150K
new candidate pool
new balanced selected HALO
```

The report must make it obvious whether the redesign actually reduced concentration and improved coverage.

## Step 17 — Code structure

Refactor cleanly rather than placing all logic inside `hf_crawl.py`.

A reasonable structure is:

```text
scale1m/hf_crawl.py
    candidate discovery and raw API collection

scale1m/annotate_candidates.py
    task, language, family, source type, quality, and duplicate annotation

scale1m/select_balanced_halo.py
    quota allocation and deterministic selection

scale1m/verify_raw.py
    candidate-entry checks and selected-HALO exit gates

scale1m/report_distribution.py
    comparative reports and diagnostics
```

You may choose different filenames if they better match the repository conventions.

All stages must be:

```text
resumable
append-safe where applicable
deterministic
configurable
testable
auditable
memory-conscious
compatible with Windows paths
```

Use configuration files or explicit CLI parameters for quotas rather than scattering unexplained constants throughout the code.

## Step 18 — Tests

Add tests for at least:

```text
multi-query deduplication
provenance merging
family assignment precedence
family confidence tracking
task normalization
language normalization
language-neutral handling
source-type classification
quantization detection
near-duplicate grouping
author caps
family caps
task caps
language quotas
source-type quotas
weighted round-robin behavior
deterministic selection
deficit reporting
targeted-backfill planning
metadata-quality scoring
exit-gate failures
```

Use synthetic fixtures designed to expose domination by:

```text
one author
one family
one task
one quantization publisher
one language
many near-duplicate repositories
```

Do not merely test happy paths.

## Step 19 — Rerun policy

Preserve the existing raw 150K and its report.

Create new versioned directories, for example:

```text
data1m/candidates_v2/
data1m/selected_v2/
```

or follow the project’s existing versioning convention.

Do not call the new output “frozen” until all hard exit gates pass.

Run an initial pilot before the full candidate crawl, such as:

```text
50K–100K candidate pilot
→ annotation
→ balanced selection
→ distribution audit
```

Use the pilot to detect schema and quota failures.

Then run the full candidate discovery and final selection.

## Step 20 — Plan and documentation updates

Update `docs/1M/T2.md` with:

```text
the failure of downloads-desc as a population definition
the new candidate-discovery design
family and near-duplicate definitions
task taxonomy
language taxonomy
author and family caps
source-type quotas
popularity strata
metadata-quality rules
selection algorithm
backfill procedure
final gates
pilot and full-run results
limitations and unresolved ambiguities
```

Update all affected sections of the main 1M plan.

Add the following decision, renumbered if required:

```text
D-30 — Replace downloads-desc population selection with family-aware
stratified HALO construction.

The final HALO is not defined as the first N models returned by the Hugging
Face downloads ranking. Crawling and final population selection are separate
stages.

A broad candidate pool is collected using global, task-specific, language,
popularity, recency, architecture, and underrepresented-region queries.
Candidates are canonicalized into model families and annotated with supertask,
language bucket, source type, popularity stratum, author concentration,
metadata quality, and near-duplicate keys.

The frozen HALO is selected through bounded quotas across family size, task,
language, source type, and popularity. Large established families are
prioritized for coverage, while per-family, per-author, quantization, and
near-duplicate caps prevent any family, publisher, or conversion ecosystem
from dominating the model population.

Global downloads rank is retained only as one candidate-discovery and
within-stratum popularity signal.
```

Add further decisions if implementation reveals new evidence.

## Required final report

At completion, provide:

1. files added and modified;
2. exact commands run;
3. test results;
4. pilot candidate and selected counts;
5. full candidate and selected counts, if completed;
6. before-versus-after distribution tables;
7. all gate outcomes;
8. largest authors and families before and after;
9. exact task and language distributions;
10. quantization and source-type distributions;
11. unresolved-family and metadata-quality rates;
12. quota deficits and how they were resolved;
13. decisions added to the plan;
14. any conclusions that remain unsupported by the available metadata.

Do not report success merely because the crawler completed.

Success means that the final selected HALO passes the distribution, concentration, quality, lineage, and reproducibility gates and is materially more representative than the original downloads-desc sample.