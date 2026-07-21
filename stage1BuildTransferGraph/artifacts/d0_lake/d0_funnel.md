# D0 intake audit -- offline funnel (Phase A)

| stage | count |
|---|---|
| cache_model_files | {'phaseA_cache': 12871, 'phaseB_cache': 3355, 'merged_distinct': 16226} |
| parse_failures | 0 |
| disabled_or_private_excluded | 0 |
| channel_has_model_index | 5270 |
| channel_has_lineage | 7052 |
| channel_family_parsable | 16072 |
| intake_union | 16117 |
| intake_strict_mi_or_lineage | 10234 |
| dropped_unnamed_dataset_obs | 135 |
| raw_primary_obs | 112162 |
| canonical_obs_dataset_model_metric | 108236 |
| distinct_node_model_pairs | 56790 |
| distinct_canonical_datasets_any_label | 560 |
| distinct_dataset_nodes_any_label | 1829 |
| nodes_ge1_bounded | 1515 |
| nodes_ge3 | 1122 |
| nodes_ge5 | 986 |
| nodes_ge10 | 897 |
| nodes_ge20 | 355 |
| gold_evaluable_nodes | 617 |
| gold_evaluable_datasets_coarse | 226 |
| distinct_roots_any_label | 544 |
| gold_evaluable_roots | 212 |
| gold_evaluable_in_dataset_cache | 605 |
| gold_evaluable_missing_from_cache | 12 |
| dataset_cache_files | 4779 |
| concentration | {'gold_bitext_like_nodes': 174, 'gold_non_bitext_nodes': 443, 'gold_distinct_roots': 195, 'gold_non_bitext_distinct_roots': 189, 'gold_top_roots': {'tatoeba-bitext-mining': 112, 'ntrex': 54, 'amazon_massive_intent': 51, 'amazon_massive_scenario': 51, 'sts22-crosslingual-sts': 18, 'xnli': 15, 'sts17-crosslingual-sts': 11, 'xcopa': 11}, 'non_bitext_ge200': True} |
| roots_by_task | {'all_labelled': {'text-classification': 96, 'token-classification': 74, 'text-generation': 68, 'classification': 54, 'sts': 30, 'question-answering': 19, 'reranking': 18, 'multiple-choice': 12, 'bitextmining': 9, 'summarization': 9, 'fill-mask': 7, 'cross-encoder-reranking': 5, 'natural-language-inference': 5, 'image-text-to-text': 5, 'image-classification': 3, 'multilabelclassification': 3, 'sentence completion': 3, 'code generation': 3, 'natural language inference': 2, 'audio-classification': 2, 'cross-encoder-correlation': 2, 'col-berttriplet': 2, 'multiple_choice': 2, 'text2text-generation': 2, 'named-entity-recognition': 2, 'pairclassification': 2, 'cross-encoder-nano-beir': 1, 'translation': 1, 'defect-detection': 1, 'paraphrase-mining': 1, 'coreference resolution': 1, 'original-capability': 1, 'vqa': 1, 'news-summarization': 1, 'relation-extraction': 1, 'math-evaluation': 1, 'math-competition': 1, 'retrieval': 1, 'sentence-similarity': 1, 'language-modeling': 1, 'feature-extraction': 1, 'scientific-reasoning': 1, 'regression': 1, 'cross-encoder-classification': 1, 'reward-modeling': 1, 'nli': 1, 'information-extraction': 1, 'text2sql': 1, 'semantic-similarity': 1}, 'gold': {'text-classification': 44, 'token-classification': 39, 'classification': 34, 'sts': 24, 'text-generation': 22, 'reranking': 12, 'question-answering': 8, 'bitextmining': 7, 'multilabelclassification': 3, 'cross-encoder-reranking': 3, 'sentence completion': 3, 'summarization': 2, 'natural language inference': 2, 'image-classification': 1, 'cross-encoder-nano-beir': 1, 'translation': 1, 'paraphrase-mining': 1, 'coreference resolution': 1, 'natural-language-inference': 1, 'original-capability': 1, 'multiple-choice': 1, 'image-text-to-text': 1}} |
| harvest_seed | {'raw_candidate_dataset_ids': 6352, 'canonical_dataset_ids': 5639, 'distinct_root_datasets': 5637, 'roots_ge2_eval_bearing_models': 475, 'datasets_kept': 530} |

targets: {'models_ge_10000': True, 'gold_nodes_ge_200': True}
intake sha256: `411625f3d70b0ec43aaf87e74971fceb62a42616b37183c51f872ced2d673e1c`
