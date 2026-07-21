# Task-type enrichment dry run

> Review artifact only. The source graph, source tables, existing vocab, and caches were not mutated.

- source graph: `ModelLakeFishing/stage1BuildTransferGraph/hgraph_hf1000d_2000m_xm0_xd0.pt`
- source SHA-256 unchanged: **TRUE**
- graph datasets: **362**; current `Other`: **312**
- proposed replacements: **210** (**32** full-graph supervised)
- unresolved conflicts: **40**; insufficient evidence: **62**

## Incompatibility-pool coverage

| protocol | supervised datasets | before | after | target >= 50 |
|---|---:|---:|---:|:--:|
| full graph | 72 | 30 | 62 | PASS |
| G2 train-visible seed 0 | 71 | 30 | 62 | PASS |
| G2 train-visible seed 1 | 72 | 30 | 62 | PASS |
| G2 train-visible seed 2 | 70 | 30 | 61 | PASS |

The implementation reproduces the reference numerator (30 pools), but reports the live denominator from the exact fixed-split code; see `reference_reconciliation` in JSON for the 30/64 caveat.

## Proposed supervised replacements

| dataset | current | proposed | evidence tier | sources |
|---|---|---|---:|---|
| `acronym_identification` | `Other` | `acronym-identification` | 2 | dataset_identity |
| `adversarial_qa` | `Other` | `question-answering` | 1 | model_index_task |
| `amazon_counterfactual` | `Other` | `counterfactual-detection` | 2 | dataset_identity |
| `amazon_massive_intent` | `Other` | `intent` | 2 | dataset_identity |
| `amazon_massive_scenario` | `Other` | `domain` | 2 | dataset_identity |
| `amazon_polarity` | `Other` | `sentiment` | 2 | dataset_card_task_id, dataset_identity, inventory_task_id |
| `banking77` | `Other` | `intent` | 2 | dataset_card_task_id, inventory_task_id |
| `biblenlp-corpus-mmteb` | `Other` | `translation` | 3 | dataset_card_task_category, inventory_task_category |
| `boolq` | `Other` | `NLI` | 2 | dataset_card_task_id, inventory_task_id |
| `bucc-bitext-mining` | `Other` | `bitext-mining` | 1 | model_index_task |
| `clinc_oos` | `Other` | `intent` | 2 | dataset_card_task_id, inventory_task_id |
| `diabla` | `Other` | `bitext-mining` | 1 | model_index_task |
| `eurlex-multilingual` | `Other` | `topic` | 2 | dataset_card_task_id, inventory_task_id |
| `few-nerd` | `Other` | `named-entity-recognition` | 2 | dataset_card_task_id, inventory_task_id |
| `language-identification` | `Other` | `language-identification` | 2 | dataset_identity |
| `mmlu` | `Other` | `question-answering` | 1 | model_index_task |
| `mrqa` | `Other` | `question-answering` | 1 | model_index_task |
| `mtop_domain` | `Other` | `domain` | 2 | dataset_identity |
| `mtop_intent` | `Other` | `intent` | 2 | dataset_identity |
| `multinerd` | `Other` | `named-entity-recognition` | 1 | model_index_task |
| `nemotron-pii` | `Other` | `pii-detection` | 2 | dataset_identity |
| `ntrex` | `Other` | `bitext-mining` | 1 | model_index_task |
| `paws-x` | `Other` | `semantic-similarity` | 2 | dataset_card_task_id, inventory_task_id |
| `scienceqa` | `Other` | `question-answering` | 2 | dataset_card_task_id, inventory_task_id |
| `sib200` | `Other` | `clustering` | 1 | model_index_task |
| `squad` | `Other` | `question-answering` | 1 | model_index_task |
| `squad_v2` | `Other` | `question-answering` | 1 | model_index_task |
| `tatoeba-bitext-mining` | `Other` | `bitext-mining` | 1 | model_index_task |
| `toxic-chat` | `Other` | `toxicity` | 2 | dataset_identity |
| `tweetner7` | `Other` | `named-entity-recognition` | 2 | dataset_card_task_id, inventory_task_id |
| `wikiann` | `Other` | `named-entity-recognition` | 2 | dataset_card_task_id, inventory_task_id |
| `yelp_review_full` | `Other` | `sentiment` | 2 | dataset_card_task_id, dataset_identity, inventory_task_id |

## Required failing query

- `amazon_massive_intent`: **propose** -> `intent`
- local evidence: dataset_identity=benchmark identity says intent classification/prediction -> intent

## Decision policy and limitations

1. Specific cached model-index task wins because it is attached to the performance edge.
2. Otherwise, a unique specific card task-id or precise benchmark identity may propose a task.
3. Otherwise, a unique specific card category may propose a task.
4. Generic classification/token-classification labels never propose a type; same-tier conflicts require review.
5. This dry run does not apply the patch, rebuild xd0, retrain G2, or establish that inferred incompatibility negatives are semantically correct.
