"""Build A0.1 protocol/reference/inventory; never reads old F/X/Y/Z reports.

Run with the repository Python. This script only writes its three owned JSONs.
Historical literals are transcribed from the hash-bound English evidence source.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
DOC = OUT.parent / "EVIDENCE_SOURCE_LIBRARY_en.md"
PLAN = OUT.parent / "A0.md"
SOURCE_SHA = hashlib.sha256(DOC.read_bytes()).hexdigest()
EXPECTED_SHA = "6c7d883734fb9a05ca99c4a8409e366955c86ba2e644bd5328c9ba696ff344cd"
if SOURCE_SHA != EXPECTED_SHA:
    raise SystemExit("Authoritative evidence changed: review literals before regenerating")
SOURCE = {
    "path": str(DOC), "sha256": SOURCE_SHA,
    "role": "sole authority for historical facts and published method",
    "evidence_cutoff": "2026-09-06 America/Toronto",
    "snapshot_copy": "frozen/EVIDENCE_SOURCE_LIBRARY_en.md",
    "snapshot_copy_binding": "parent freeze confirms byte-identical SHA256; full binding in A0_SOURCE_MANIFEST.json",
}
PLAN_REF = {"path": str(PLAN), "sha256": hashlib.sha256(PLAN.read_bytes()).hexdigest(),
            "role": "authorized A0 change and execution contract, not an alternative historical source"}
SEEDS = [0, 1, 2]
PATHS = ["hnsw1000_task_prior", "exact1000_task_prior", "exact_full_lake_task_prior"]
REFS = []


def old(name, scope, seed, value, literal, section, precision, unit="fraction", note=None):
    item = {"id": f"{scope}.{name}.{seed}", "name": name, "scope": scope, "seed": seed,
            "value": value, "published_literal": literal, "unit": unit,
            "source_path": SOURCE["path"], "source_sha256": SOURCE_SHA,
            "source_section": section, "display_precision": precision,
            "status": "published_reference_only", "note": note}
    REFS.append(item)
    return item


FINAL = {
    "queries": [1476, 1101, 1545],
    "gold_at_1": ["0.1599", "0.1117", "0.1107", "0.1274"],
    "gold_at_10": ["0.3279132791", "0.3387829246", "0.2427184466", "0.3031382168"],
    "top3_at_10": ["0.3984", "0.4269", "0.2958", "0.3737"],
    "gold_gap_at_10": ["0.3814", "0.3860", "0.2777", "0.3484"],
    "root_macro_gold_at_10": ["0.2515", "0.2591", "0.1903", "0.2336"],
    "median_gold_position_when_retrieved": [6, 7, 5],
}
for metric, values in FINAL.items():
    for pos, value in enumerate(values):
        seed = pos if pos < 3 else "mean"
        integer = isinstance(value, int)
        rec = old(metric, PATHS[0], seed, value if integer else float(value), str(value),
                  "5.4", "integer" if integer else ("10 decimal places" if metric == "gold_at_10" else "4 decimal places"),
                  "query" if metric == "queries" else "1-based position" if integer else "fraction")
        if metric == "gold_at_10":
            rounded = ["0.3279", "0.3388", "0.2427", "0.3031"][pos]
            rec["other_published_display"] = {"literal": rounded, "decimal_places": 4, "source_section": "5.4"}
            rec["note"] = "The source calls these unrounded; only the ten published decimal places are preserved, with no reconstructed extra digits."
for scope, vals in [(PATHS[1], ["0.3286", "0.3388", "0.2427", "0.3034"]),
                    (PATHS[2], ["0.3408", "0.3651", "0.2589", "0.3216"])]:
    for pos, value in enumerate(vals):
        old("gold_at_10", scope, pos if pos < 3 else "mean", float(value), value, "5.4", "4 decimal places")
for metric, literal in {
    "exact_pool_retention": "94.32", "ann_retention": "99.93", "overall_retention": "94.25",
    "exact_pool_gold_coverage_at_1000": "51.88", "full_fused_top10_in_exact_pool": "89.91",
}.items():
    old(metric, "retrieval_diagnostics", "mean", float(literal), literal + "%", "5.4", "2 decimal places in percent", "percent")
for s, ef, recall, build in zip(SEEDS, [1000, 1500, 1500], ["0.9923", "0.9941", "0.9932"], ["184.6", "191.3", "181.6"]):
    old("selected_ef_search", "ann_calibration", s, ef, f"{ef:,}", "5.2", "integer", "ef_search")
    old("selected_recall_at_1000", "ann_calibration", s, float(recall), recall, "5.2", "4 decimal places")
    old("build_seconds", "index_cost", s, float(build), build, "5.4", "1 decimal place", "second")
    old("index_gib", "index_cost", s, 2.214, "about 2.214 GiB", "5.4", "approximate; 3 decimal places", "GiB",
        "Source reports approximately this size for each index; it does not publish individual exact bytes.")
for metric, values in {"total_latency_p50_ms": ["0.536", "0.847", "0.699", "0.694"],
                       "total_latency_p95_ms": ["0.905", "1.435", "1.329", "1.223"]}.items():
    for pos, value in enumerate(values):
        old(metric, "retrieval_cost", pos if pos < 3 else "mean", float(value), value, "5.4", "3 decimal places", "millisecond")
for metric, val in [("hnsw_latency_p50_ms", "0.566"), ("prior_rerank_latency_p50_ms", "0.123")]:
    old(metric, "retrieval_cost", "mean", float(val), val, "5.4", "3 decimal places", "millisecond")
old("index_gib", "index_cost", "sum", 6.643, "6.643 GiB", "5.4", "3 decimal places", "GiB")
for s, n in zip(SEEDS, [198216, 196912, 196124]):
    old("visible_prior_edges", "task_prior", s, n, f"{n:,}", "5.3", "integer", "edge")

DATA_COUNTS = {
    "snapshot_unique_models": (3003759, "2.1", "model"),
    "model_snapshot_shards": (61, "2.1", "file"),
    "model_snapshot_api_pages": (3004, "2.1", "page"),
    "model_snapshot_skipped_duplicates": (0, "2.1", "record"),
    "snapshot_dataset_repositories": (1008417, "2.1", "repository"),
    "dataset_snapshot_shards": (11, "2.1", "file"),
    "candidate_models": (3016439, "1; 3.3", "model"),
    "dataset_task_nodes": (18729, "1; 3.5", "node"),
    "historical_only_models": (12680, "3.3", "model"),
    "matched_card_nodes": (3928, "2.3", "node"),
    "native_raw_metric_rows": (2158375, "3.1", "row"),
    "native_finite_metric_rows": (2097081, "3.1", "row"),
    "native_median_deduplicated_rows": (1435162, "3.1", "row"),
    "native_primary_edges_before_cap": (143478, "3.1", "edge"),
    "native_primary_edges_after_cap": (74346, "3.1", "edge"),
    "merged_input_rows": (531958, "3.2", "row"),
    "merged_within_source_duplicates_removed": (5102, "3.2", "row"),
    "merged_cross_source_conflicts": (1502, "3.2", "pair"),
    "merged_pairs_before_cap": (525354, "3.2", "pair"),
    "supervision_edges": (247803, "3.2; 3.6", "edge"),
    "retained_modellens_v2_edges": (117898, "3.2", "edge"),
    "retained_hf_model_index_edges": (73670, "3.2", "edge"),
    "retained_d0_v1_5_edges": (45992, "3.2", "edge"),
    "retained_hf_effective_edges": (5223, "3.2", "edge"),
    "retained_a_ctrl_2000m_edges": (3670, "3.2", "edge"),
    "retained_diverse_zoo_edges": (1350, "3.2", "edge"),
    "trained_on_edges": (247803, "3.6", "edge"),
    "rev_trained_on_edges": (247803, "3.6", "edge"),
    "stored_similar_to_edges": (374580, "3.6", "edge"),
    "is_base_of_edges": (859065, "3.6", "edge"),
    "rev_is_base_of_edges": (859065, "3.6", "edge"),
    "stored_total_directed_edges": (2588316, "3.6", "edge"),
    "training_similar_to_edges": (187290, "3.6", "edge"),
    "family_vocab_rows": (41056, "3.4", "row"),
}
for name, (value, section, unit) in DATA_COUNTS.items():
    old(name, "frozen_input_audit", "shared", value,
        "zero" if name == "model_snapshot_skipped_duplicates" else f"{value:,}", section, "integer", unit)
for name, value, section in [("unknown_model_size_percent", "71.929", "3.4"),
                              ("other_model_family_percent", "13.561", "3.4"),
                              ("resolved_lineage_percent", "96.083", "3.6")]:
    old(name, "frozen_input_audit", "shared", float(value), value + "%", section, "3 decimal places in percent", "percent")
old("model_feature_file_gb", "frozen_input_audit", "shared", 5.41, "5.41 GB", "3.4", "2 decimal places", "decimal GB")

PROTOCOL = {
    "schema_version": "1.0", "stage": "A0.1", "status": "protocol_extracted_pending_input_and_implementation_reconciliation",
    "authority": SOURCE, "authorized_plan": PLAN_REF,
    "historical_facts_policy": "Use only the authoritative English evidence source. F/X/Y/Z paths may locate code/artifacts; they cannot fill unpublished historical metric digits or override this protocol.",
    "sole_final_system": "X4G+D (GD combined arm) -> held-out embeddings -> HNSW top-1000 -> task-prior fusion -> top-10",
    "official_runs": [{"split_seed": s, "initialization_seed": 0, "epochs": 25, "name": f"A0GD_full_s{s}_e25"} for s in SEEDS],
    "exact_references": [{"path": p, "purpose": "same new GD representations, queries and fusion; offline diagnostic only"} for p in PATHS[1:]],
    "out_of_scope": ["separate G or D training", "NoGraph", "BM25 or other baseline reruns", "larger K pools", "new fusion search", "new dataset inductive encoding claim", "A0.2 or later execution during A0.1"],
    "permitted_changes": {
        "feature_repair": {"authorized_by": "A0.md sections 3-4", "historical_definition": "evidence 3.5",
            "array": "x_dataset", "all_dataset_nodes": True, "zero_based_columns": [448, 449, 450, 451, 452, 453, 455],
            "operation": "set exactly to 0.0 in new graph artifact and feature builder; no columns removed",
            "old_stats": ["log(1+n)", "log(1+n)", "mean", "std", "min", "max", "log(1+root_node_count)", "gold_eligible", 0, 0],
            "new_stats": [0, 0, 0, 0, 0, 0, "log(1+root_node_count)", 0, 0, 0],
            "preserve_columns": {"454": "frozen root-node count from node table", "456,457": "original reserved zeros"},
            "preserve_dimensions": [458, 486, 128], "implemented": False,
            "gold_eligibility": "remain in independent evaluation table; never change eligibility or labels to implement input masking"},
        "derived_artifacts": "New graph digest, from-scratch checkpoints, embeddings, recomputed split priors, exact candidates, indexes, predictions and every metric/time/resource observation.",
        "ef_search": "Recalibrate on new dense IDs using the unchanged grid and recall rule; do not copy old selected values.",
        "required_evaluation_adaptations": "A0.2 may separate historical-value/effectiveness gates from completing measurement and persist missing raw outputs. Preserve formulas, precision, sorting, original gate values and all correctness gates; log deviations explicitly."
    },
    "frozen_data": {
        "source_sections": ["2.1", "2.2", "2.3", "2.4", "3.1", "3.2", "3.3", "3.4", "3.5", "3.6"],
        "snapshot_date": "2026-08-18", "candidate_models": 3016439, "dataset_task_nodes": 18729, "positive_edges": 247803,
        "candidate_identity": "lower(strip(model_id)); hub snapshot exact prefix; 12680 normalized historical-only IDs appended in sorted order",
        "dataset_identity": "(normalize(dataset), task), serialized with tab; original root mapping unchanged",
        "metric_policy": "known higher/lower directions, existing reward/unknown handling, finite-value filter, original comparable grouping and constant-group 0.5",
        "merge_priority": ["modellens_v2", "d0_v1_5", "a_ctrl_2000m", "hf_effective", "diverse_zoo", "hf_model_index"],
        "merge_policy": "median within source/node/model; min-max within node/source; fixed source conflict priority; deterministic weight-stratified cap",
        "cap_per_node": 200, "cap_numpy_seed": 0,
        "gold_rule_version": "rf-gold-2.0", "gold_rule_sha256": "be3fb05ecf92ea0112ba4de10b8191d71dfe096b8667a86911c2ebaeadb9cdb1",
        "model_ladder_sha256": "fee360d1fef9afa4c446b70af35bcbf4ead2c75387f47abc3b0d21c944ae3efa",
        "dataset_ladder_sha256": "31c027ff2eeb693aed6abaa5d235f4a6264bb4b3245bfd71cf1c8b85d32173cb",
        "model_feature_sha256": "ba1020872ddb6e90726c755f9f08f236f4c73e506013c91d8a0075fc0c0361e6",
        "family_vocab_sha256": "00d304df9bd72acb2c2c1ca5b84ed0dc9d4adb0cbc9f098f32a9e62109d24005",
        "reuse_rule": "Re-use hash-verified frozen inputs; recount all counts/coverage/file sizes. No fresh web crawl, reordering, new metadata or copied old measurements.",
    },
    "features": {
        "source_sections": ["3.4", "3.5", "4.2"], "minilm": "all-MiniLM-L6-v2", "minilm_dim": 384, "minilm_output_normalized": False,
        "model_name_seed": 42, "dataset_name_seed": 43, "name_dim": 64, "name_hash": "MD5 modulo 10000; frozen row-normalized Gaussian table; original tokenization",
        "model_descriptor": "cleaned repository identifier + canonical family + parameter-size phrase if safetensors count exists",
        "dataset_descriptor": "cleaned dataset-task name + card task categories + <=10 non-colon tags + <=400 description characters; unmatched uses cleaned name/task only",
        "model_frozen_dim": 448, "model_size_dim": 16, "model_family_dim": 16, "model_input_dim": 480,
        "dataset_frozen_dim": 458, "dataset_stats_dim": 10, "dataset_category_dims": [16, 8, 4], "dataset_input_dim": 486,
        "freeze_feature_matrices": True, "preserve_vocab_and_categorical_ids": True, "model_id_embedding": False,
    },
    "graph": {
        "source_sections": ["3.6", "4.1", "4.3", "4.7"],
        "relations": ["model/trained_on/dataset", "dataset/rev_trained_on/model", "dataset/similar_to/dataset", "model/is_base_of/model", "model/rev_is_base_of/model"],
        "stored_similarity": {"block": "384-dimensional card embeddings", "top_k": 20, "exclude_self": True, "score": "cosine"},
        "training_similarity": {"mode": "topk_unweighted", "top_k": 10, "edge_attr": 1},
        "preserve_relation_endpoints_order_and_weights": True,
        "depth": 1, "hidden_dim": 128, "output_dim": 128, "weighted_relations": [],
        "operator": "relation-specific self/neighbour linear projections and unweighted mean; scalar learned relation gates initialized to 1",
        "shared_output_head": True, "l2_normalize_output": True,
    },
    "split": {
        "source_section": "4.1", "unit": "dataset root", "split_seeds": SEEDS,
        "test_edge_fraction_approx": 0.2, "val_edge_fraction_approx": 0.1, "train_disjoint_supervision_fraction": 0.3,
        "algorithm": "original root shuffle and greedy edge quota assignment, without changing root definition or RNG",
        "training_message_edges": "E_train minus E_disjoint_supervision", "validation_message_edges": "E_train",
        "test_message_edges": "E_train union E_validation",
        "remove_reverse_performance_edges_together": True, "binary_negative_ratio": 1.0,
        "negative_exclusion": "complete positive edge set", "ranking_targets": "training-selected positive edge keys read oriented weights; no mechanism change",
    },
    "training": {
        "source_sections": ["4.4", "4.5", "4.6", "4.7", "6"],
        "optimizer": "Adam", "learning_rate": 0.01, "epochs": 25, "batch_size": 1024, "initialization_seed": 0,
        "objective_coefficients": {"ranknet": 1, "model_model_contrastive": 1, "whole_lake_sampled_softmax": 1},
        "ranknet": {"temperature": 0.1, "min_gap": 0.02, "max_pairs_per_dataset": 256, "pair_selection_if_capped": "half most inverted; half sampled remainder", "aggregation": "mean within dataset then across datasets"},
        "contrastive": {"temperature": 0.2, "n_negatives": 256, "positive_membership": "top max(1, round(0.1*n_d)) train-visible models per dataset", "sparse_membership": True, "lineage_nonpositive_negative_weight": 2},
        "global": {"temperature": 0.1, "n_negatives": 256, "n_datasets": 128, "lake_gamma": 0.5, "proposal_alpha": 0.75, "proposal_n0": 1.0,
            "proposal": "0.5*normalized((train_visible_degree+1)^0.75)+0.5*uniform_over_labeled",
            "negative_sampling": "with replacement; remove sampled positives", "logq_correction": True,
            "forward": "full training-message graph, fresh edge dropout"},
        "fanout": {"enabled": True, "hops": 2, "ordinary": 10, "lineage": 20, "fallback": "without replacement; per-node hard bound"},
        "edge_dropout": {"ordinary": 0.3, "lineage": 0.05, "targets": "message edges only"},
        "flags": {"sparse_M": True, "skip_diagnostics": True, "chunked_infer": 50000},
        "disabled": ["MSE", "embedding uniformity", "dataset-model contrastive", "hard-negative mining", "positive inverse-propensity weighting", "separate heads", "early stopping"],
        "checkpoint_policy": "start at epoch 0 in fresh formal directories; use epoch 25 last checkpoint; no old/smoke weights or optimizer state; only same-A0 valid resume",
    },
    "evaluation": {
        "source_sections": ["4.8", "5.1", "5.2", "5.3", "5.4", "6"],
        "representations": ["z_m_eval.npy", "z_d_eval.npy"], "embedding_dim": 128, "mappedID_order": True,
        "eligibility": ["known metric direction or curated oriented source", "non-RL task", "non-placeholder dataset", "at least 3 observed candidates", "nonconstant held-out values"],
        "expected_query_counts": {"0": 1476, "1": 1101, "2": 1545},
        "hnsw": {"space": "ip", "dim": 128, "M": 32, "ef_construction": 200, "K": 1000, "construction_threads_cli": 8},
        "ef_calibration": {"grid": [1000, 1500, 2000, 3000, 5000], "threshold": 0.99,
            "metric": "mean per-query intersection(ANN1000, exact_dense1000)/1000", "selection": "first passing grid value", "labels_used": False,
            "query_set_ids": None, "query_set_binding": "implementation-bound: persist exact calibration query IDs and hash; verify original selection against code",
            "A0_failure_policy": "if none passes, preserve failure and full trace, finish fixed-K measurement at ef=5000; no claim of meeting original fidelity standard"},
        "task_prior": {"visible_edges": "train+validation; exclude all scored test roots and same-root siblings", "task": "original normalized task",
            "shrink": 5, "beta": 1, "prior_mean": 0.5, "formula": "(sum_oriented+0.5*5)/(count+5) when count>0; else 0",
            "rebuild_per_split": True, "verify_rowmaps_and_root_mapping": True},
        "fusion": {"formula": "(cosine+1)/2 + task_prior", "query_minmax": False, "return_k": 10,
            "ties": "fixed label-free permutation of model row IDs; same convention for all three fused paths",
            "tie_permutation_seed": None, "tie_seed_status": "not stated in evidence; bind current original implementation during config audit"},
        "dense_rank_diagnostic": "strictly-greater-score count per evidence 4.8; do not confuse this dense convention with final deterministic fused top-10 ordering",
        "exact_compute": {"device_cli": "cuda", "query_chunk": 16, "model_chunk": 50000},
        "timing": {"start": "precomputed query embedding", "query_batch": 1, "hnsw_query_threads": 1, "warmup": True,
            "excludes": ["query encoding", "index loading", "index construction"],
            "warmup_count": None, "repetitions": None, "percentile_interpolation": None,
            "unpublished_setting_policy": "bind original implementation; record actual environment; no guessed historical values",
            "cross_seed_summary": "arithmetic mean of each seed's percentile, not pooled queries; total percentile from total per-query timings, not sum of component percentiles"},
    },
    "required_config_audit": {
        "status": "pending_parent_implementation_audit", "artifact": None,
        "rule": "Every actual resolved training/evaluation key must be enumerated and hash-bound before formal runs. Values absent from evidence are implementation-bound, not published facts.",
        "not_specified_in_authoritative_document": ["Adam betas/epsilon/weight decay and remaining optimizer keys", "all loader/worker/device flags", "tie permutation seed and exact numeric tie behavior", "HNSW construction random seed", "calibration query ID selection", "timing warmup/repetition counts and quantile convention", "chunked/full inference tolerance", "remaining resolved config keys"],
        "conflict_policy": "Record conflicts in A0_DISCREPANCIES.md; do not start affected stage or change established protocol silently."
    },
    "raw_output_contract": {
        "query_alignment": "explicit query IDs, never coincidental row position",
        "per_query": ["seed", "query_id", "root", "normalized_task", "gold_id", "top3_observed_ids", "gap_eligible_observed_ids", "heldout_model_ids_and_values", "hnsw_ids[1000]", "hnsw_cosine[1000]", "prior[1000]", "fused_score[1000]", "top10_ids[10]", "complete_pool_gold_position_or_absent", "exact_dense_ids[1000]", "exact_full_fused_top10_ids[10]", "exact_reference_quality_inputs", "hnsw_time", "prior_lookup_and_rerank_time", "total_time"],
        "bindings": ["graph", "checkpoint", "feature arrays", "model/dataset row maps", "split IDs", "query/gold IDs", "task-prior sidecar", "index", "code and resolved configuration", "all raw outputs"],
        "independent_recomputation": "A0.7 recomputes quality, counts, ratios, summaries and timing percentiles from new raw artifacts only",
    },
    "reporting": {
        "all_new_metrics_initially_null": True, "no_old_measurement_reuse": True,
        "summary_requires_all_three_seeds": True, "retention_summary": "mean of seed-specific ratios, never ratio of means",
        "zero_denominator": "undefined with numerator, denominator and reason retained", "missing_seed": "pending or missing; never silently omitted",
        "conditional_gold_position": "only queries with gold in actual pool; retain count; other quality denominators include misses",
        "no_pooled_unique_query_claim": "sum of seed query counts is observations across splits, not disjoint datasets",
        "completion_independent_of_old_score": True,
    },
}

OLD_REFERENCE = {
    "schema_version": "1.0", "stage": "A0.1", "authority": SOURCE,
    "status": "historical_reference_only", "primary_path": PATHS[0],
    "precision_policy": "Preserve exactly published precision and approximation labels. No reconstructed historical values from rounded tables or F/X/Y/Z files.",
    "historical_timing_environment": {"source_section": "5.4", "os": "Windows 11", "cpu": "Intel Family 6 Model 183", "logical_processors": 24, "single_query": True, "hnsw_query_threads": 1, "warmup": True},
    "excluded_comparisons": [{"source_section": "5.4", "item": "2.98x versus BM25", "reason": "A0 does not rerun baselines; never combine new final metrics with an old baseline as an all-recomputed claim."}],
    "unpublished_old_values": ["HNSW actual-pool gold coverage", "per-seed retention and coverage values", "exact-reference metrics besides gold@10", "root counts", "component per-seed latencies and component p95", "exact individual index bytes", "training/export/prior/exact/evaluation wall time", "peak RAM/VRAM", "unreported epoch loss numbers"],
    "records": REFS,
}
LOOKUP = {(r["name"], r["scope"], r["seed"]): r for r in REFS}
METRICS = []


def metric(name, scope, seed, definition, unit, aggregation, inputs, *, category="measurement", old_scope=None, note=None, final_status="recomputed", required=True):
    ref = LOOKUP.get((name, old_scope or scope, seed))
    item = {
        "id": f"{scope}.{name}.{seed}", "name": name, "scope": scope, "seed": seed,
        "definition": definition, "unit": unit, "aggregation": aggregation,
        "old_value": ref["value"] if ref else None,
        "old_source_section": ref["source_section"] if ref else None,
        "old_display_precision": ref["display_precision"] if ref else None,
        "old_reference_id": ref["id"] if ref else None,
        "old_source": {"path": SOURCE["path"], "sha256": SOURCE_SHA, "status": "published" if ref else "not_reported"},
        "new_value": None, "recompute_from": inputs, "recomputed_from": [], "artifact_hashes": {},
        "status": "pending", "required": required, "category": category,
        "expected_final_status": final_status, "note": note,
    }
    METRICS.append(item)


def per_seed_and_mean(name, scope, definition, unit, aggregation, inputs, **kwargs):
    for s in SEEDS:
        metric(name, scope, s, definition, unit, aggregation, inputs, **kwargs)
    metric(name, scope, "mean", definition, unit, "arithmetic mean of all three unrounded seed values; no skip-missing", [f"{scope}.{name}.{s}" for s in SEEDS], **kwargs)


QUALITY = {
    "gold_at_1": "Fraction of all eligible held-out queries whose original-rule historical gold is returned first; gold absent from pool is failure.",
    "gold_at_10": "Fraction of all eligible held-out queries whose original-rule historical gold occurs in final returned top 10; pool misses remain in denominator.",
    "top3_at_10": "Fraction of eligible queries for which ANY of the three highest original-rule observed models occurs in returned top 10; not recall of all three.",
    "gold_gap_at_10": "Fraction of eligible queries for which ANY observed model within 0.01 of the best oriented held-out value occurs in returned top 10.",
    "root_macro_gold_at_10": "Mean gold@10 within each original root, then equal-weight mean over scored roots.",
    "gold_gap_at_1": "Fraction of eligible queries for which an observed model within 0.01 of best oriented held-out value is returned first; native gold-gap@1.",
    "root_macro_gold_at_1": "Mean gold@1 within each original root, then equal-weight mean over scored roots; native root_gold@1.",
    "root_macro_top3_at_10": "Mean top3@10 within each original root, then equal-weight mean over scored roots; native root_top3@10.",
    "root_macro_gold_gap_at_10": "Mean gold-gap@10 within each original root, then equal-weight mean over scored roots; native root_gold-gap@10.",
}
for path in PATHS:
    for name, definition in QUALITY.items():
        per_seed_and_mean(name, path, definition, "fraction", "mean of per-query indicators" if not name.startswith("root_macro") else "root-macro mean",
                         ["A0 per-query query/gold/top3/gap labels and roots", f"A0 {path} deterministic fused top10"], category="final_quality" if path == PATHS[0] else "exact_reference_quality")
    for s in SEEDS:
        metric("queries", path, s, "Count of actual eligible, nonconstant held-out query IDs scored in this path.", "query", "count", ["A0 query ID and gold eligibility tables", f"A0 {path} raw query outputs"], old_scope=PATHS[0])
        for name, definition, unit, inputs in [
            ("roots", "Distinct original root IDs among scored query IDs.", "root", ["A0 scored query IDs and root map"]),
            ("candidate_models", "Actual model universe row-map count, checked against embedding rows.", "model", ["A0 model row map and heldout embeddings"]),
            ("pool_size", "Actual unique candidate count per query; 1000 for bounded paths, N for full-lake reference; retain min/max and any violations.", "candidate/query", [f"A0 {path} candidate IDs and row map"]),
            ("return_size", "Actual number of unique returned model IDs per query; retain min/max and any violations; expected 10.", "model/query", [f"A0 {path} top10 IDs"]),
        ]:
            metric(name, path, s, definition, unit, "count or per-query count distribution", inputs)
    metric("query_observations", path, "sum", "Sum of scored query counts across split seeds; these observations need not be unique datasets.", "query observation", "sum over three complete seed counts", [f"{path}.queries.{s}" for s in SEEDS])
    for name, definition, unit in [("quality_success_counts", "Per-quality-metric numerator and denominator, including per-root contributions.", "counts by metric/root"),
                                   ("extra_original_scorer_fields", "All additional gap/root/statistic fields produced by the original scoring functions, retained without silently dropping fields.", "named record")]:
        for s in SEEDS + ["mean"]:
            metric(name, path, s, definition, unit, "per-seed native fields; summary for each eligible scalar after all seeds present", [f"A0 {path} full scorer output and raw per-query records"],
                   note="Expand every discovered scorer field into a named metric before final execution/report completeness check; object placeholder alone cannot establish completeness.")

for path in PATHS[:2]:
    per_seed_and_mean("median_gold_position_when_retrieved", path,
        "Median 1-based gold position in COMPLETE deterministic reranking of the actual 1000 pool, conditioned on gold being present. This is not full-lake median gold rank.",
        "1-based position", "median among pool-retrieved gold queries only", [f"A0 {path} full 1000 candidate scores and tie ranks", "A0 gold IDs"],
        note="Retain conditional denominator separately. Summary is explicitly mean of seed medians; old summary unreported.")
    for s in SEEDS:
        metric("gold_retrieved_query_count", path, s, "Number of eligible queries with gold present in the actual candidate pool; denominator of conditional median.", "query", "count", [f"A0 {path} candidate IDs and gold IDs"])
    metric("gold_retrieved_query_count", path, "sum", "Sum of conditional query counts across seeds.", "query observation", "sum of three seed conditional counts", [f"{path}.gold_retrieved_query_count.{s}" for s in SEEDS])
for name, definition, unit in [
    ("median_full_fused_gold_rank", "Median exact gold rank across ALL N models under fixed full-lake fused score and label-free tie order; only defined for the full-lake exact diagnostic.", "1-based rank"),
    ("median_full_fused_rank_over_N", "Median full-lake fused gold rank divided by actual N; exact full-lake diagnostic only.", "normalized rank"),
    ("vs_uniform_random_top10", "Exact full-lake fused gold@10 divided by 10/N as in the existing scorer; analytic uniform-random reference, no baseline training.", "ratio"),
]:
    per_seed_and_mean(name, PATHS[2], definition, unit, "native exact full-lake scorer aggregation", ["A0 exact full-lake fused per-query rank counts", "A0 actual N"])
for path in PATHS[:2]:
    for s in SEEDS:
        metric("unavailable_full_lake_rank_fields", path, s,
               "Native median_gold_rank, median_rank_over_N and vs_random remain null because a bounded pool does not establish a whole-lake total ranking.",
               "named null record", "preserve original unavailable-field reasons", ["A0 bounded pool scorer output"],
               category="reporting_contract", final_status="undefined")
per_seed_and_mean("actual_hnsw_gold_coverage_at_1000", PATHS[0], "100 times fraction of all eligible queries with original gold in actual HNSW pool.", "percent", "100 * mean membership indicator", ["A0 actual HNSW1000 IDs", "A0 gold IDs"], note="Historical 51.88% is exact-pool coverage and cannot fill this field.")

for name, definition in {
    "exact_pool_retention": "100 * gold10_exact1000(seed) / gold10_exact_full(seed)",
    "ann_retention": "100 * gold10_hnsw1000(seed) / gold10_exact1000(seed)",
    "overall_retention": "100 * gold10_hnsw1000(seed) / gold10_exact_full(seed)",
    "exact_pool_gold_coverage_at_1000": "100 * mean over eligible queries of gold membership in exact dense1000 pool",
    "full_fused_top10_in_exact_pool": "100 * mean over queries of |exact_full_fused_top10 intersect exact_dense1000|/10",
}.items():
    per_seed_and_mean(name, "retrieval_diagnostics", definition, "percent", "seed-specific ratio or mean of per-query fraction; retain numerator and denominator",
                     ["A0 exact_full_lake_task_prior quality and full fused top10", "A0 exact_dense1000 pool and exact1000 fusion", "A0 actual HNSW1000 quality", "A0 per-query gold IDs"],
                     note="Summary is mean of per-seed ratios, never ratio of cross-seed means; zero denominator => undefined with evidence.")
for s in SEEDS + ["mean"]:
    metric("retention_numerators_denominators", "retrieval_diagnostics", s,
           "Exact numerator and denominator for every retention/coverage value; no old numerator/denominator may enter an A0 ratio.", "named count or fraction pairs", "retain raw pairs and per-seed pairs for summary", ["A0 per-query diagnostics and quality outputs"])
for s in SEEDS:
    for ef in [1000, 1500, 2000, 3000, 5000]:
        metric(f"recall_at_1000_ef_{ef}", "ann_calibration", s,
               f"Mean |ANN1000(ef={ef}) intersect exact_dense1000|/1000 over the original implementation's persisted calibration query IDs; do not use gold.",
               "fraction", "mean per-query ID recall", ["A0 calibration query ID hash", "A0 exact dense1000 IDs", f"A0 ANN1000 IDs at ef={ef}"],
               note="Follow first-passing stop rule. Later unvisited ef entries are explicitly not_applicable, with stop evidence, rather than fabricated measurements.", final_status="recomputed_or_not_applicable_after_first_pass")
    metric("selected_ef_search", "ann_calibration", s, "First ef in original ordered grid with ID recall >=0.99; if none, use 5000 for measurement and retain failure.", "ef_search", "first passing grid value or documented A0 maximum fallback", ["A0 ordered calibration trace"])
    metric("selected_recall_at_1000", "ann_calibration", s, "ID recall at newly selected ef; old ef and old recall cannot be copied.", "fraction", "selected trace entry", ["A0 ordered calibration trace and selected ef"])
    metric("calibration_passed", "ann_calibration", s, "Whether a grid entry reached recall@1000 >= 0.99.", "boolean", "any original grid entry passes threshold", ["A0 ordered calibration trace"])
    metric("calibration_query_count", "ann_calibration", s, "Number of persisted query IDs used for ID recall calibration.", "query", "count", ["A0 calibration query ID table"])
    metric("calibration_trace", "ann_calibration", s, "Full actual ef order, per-query recall, mean recall, selection, stopping and failure state.", "ordered records", "no scalar reduction", ["A0 raw calibration outputs"])
metric("selected_recall_at_1000", "ann_calibration", "mean", "Mean selected ID recall, with all seed values present.", "fraction", "mean of three seed recalls", [f"ann_calibration.selected_recall_at_1000.{s}" for s in SEEDS])
metric("selected_ef_search", "ann_calibration", "vector", "Selected ef for seeds 0/1/2; do not report a misleading average ef.", "ordered ef vector", "ordered seed values", [f"ann_calibration.selected_ef_search.{s}" for s in SEEDS])
metric("calibration_passed", "ann_calibration", "all", "All three seeds satisfy original recall threshold.", "boolean", "logical AND over all three complete seeds", [f"ann_calibration.calibration_passed.{s}" for s in SEEDS])

for component in ["hnsw", "prior_rerank", "total"]:
    for quantile in [50, 95]:
        per_seed_and_mean(f"{component}_latency_p{quantile}_ms", "retrieval_cost",
                         f"p{quantile} of recorded per-query {component} elapsed milliseconds after warmup, from precomputed embeddings with single-query/single-HNSW-thread retrieval.",
                         "millisecond", f"p{quantile} from component's raw timing array; total uses measured per-query totals",
                         [f"A0 per-query {component} timing array", "A0 timing warmup/repetition/quantile settings and environment"],
                         note="Three-seed mean is mean of seed percentiles. Never add component p50/p95 to produce total percentile.")
for name, definition, unit, inputs in [
    ("build_seconds", "Elapsed time building this new HNSW index, excluded from query latency.", "second", ["A0 index build timer log"]),
    ("index_bytes", "Actual persisted HNSW index file size from filesystem stat.", "byte", ["A0 hnsw index file stat and SHA256"]),
    ("index_gib", "Actual index_bytes / 2^30, not peak RAM and not decimal GB.", "GiB", ["A0 index_bytes"]),
]:
    per_seed_and_mean(name, "index_cost", definition, unit, "one measured value per index", inputs)
    metric(name, "index_cost", "sum", definition, unit, "sum of all three actual seed values; GiB computed from summed bytes", [f"index_cost.{name}.{s}" for s in SEEDS])
for name, definition, unit in [
    ("training_seconds", "Total measured training cost over all valid 25-epoch segments, including valid resumes.", "second"),
    ("export_seconds", "Elapsed time exporting this A0 checkpoint's new full and evaluation embeddings.", "second"),
    ("prior_build_seconds", "Elapsed time aggregating and writing the split-specific A0 task sidecar.", "second"),
    ("exact_full_reference_seconds", "Elapsed time calculating exact full-lake fusion reference from new embeddings.", "second"),
    ("exact1000_reference_seconds", "Elapsed time calculating exact dense1000 and same fusion reference; mark shared work explicitly, avoid double counting.", "second"),
    ("complete_evaluation_seconds", "Elapsed wall time for the complete seed evaluation, with exact/HNSW/calibration/final measurement stage boundaries retained.", "second"),
    ("training_peak_rss_bytes", "Peak training process resident memory with sampling method recorded.", "byte"),
    ("training_peak_vram_bytes", "Peak training device memory; specify allocated/reserved/device accounting separately if exposed.", "byte"),
    ("evaluation_peak_rss_bytes", "Peak evaluation process resident memory with measurement method and stage recorded.", "byte"),
    ("evaluation_peak_vram_bytes", "Peak evaluation device memory with measurement method and stage recorded.", "byte"),
]:
    per_seed_and_mean(name, "pipeline_cost", definition, unit, "measured stage total or peak; record segments and measurement boundary", ["A0 stage timer/resource logs", "A0 command and resume records"])
    if name.endswith("seconds"):
        metric(name, "pipeline_cost", "sum", definition, unit, "sum complete per-seed costs; do not mislabel parallel elapsed wall time", [f"pipeline_cost.{name}.{s}" for s in SEEDS])
for s in SEEDS + ["shared"]:
    metric("runtime_environment", "reproducibility", s, "Actual OS, CPU, logical processors, RAM, GPU, device, package versions, threads and timing isolation; distinguish historical same-machine and different-machine comparison.", "structured record", "record actual environment per stage/run", ["A0 machine/version probes and command logs"])

for s in SEEDS:
    for name, definition, unit, end_status in [
        ("epochs_completed", "Count of completed distinct formal epochs; expected 25.", "epoch", "verified"),
        ("initialization_seed", "Actual recorded initialization seed; expected 0.", "seed", "verified"),
        ("split_seed", "Actual root-split seed for this run.", "seed", "verified"),
        ("effective_resolved_config", "Every effective training/evaluation key with code hash and authoritative-or-implementation-bound provenance.", "structured record", "verified"),
        ("mechanism_gate", "Actual original mechanism gate outputs, including failure details; no effectiveness gate is forced true.", "structured boolean record", "recomputed"),
        ("all_existing_epoch_fields", "All 25 epoch records and every original field, including auxiliary mechanism/proposal/sampling statistics; no native field discarded.", "25 epoch records", "recomputed"),
        ("disabled_diagnostics", "Exact diagnostics disabled by original skip-diagnostics/config, with reason; not zeros and not interrupted missing measurements.", "named disabled-policy record", "disabled"),
    ]:
        metric(name, "training_records", s, definition, unit, "per run native record; no imported old losses", ["A0 resolved config", "A0 epoch history and mechanism logs"], final_status=end_status)
    for name in ["total_loss", "rank_loss", "contrast_loss", "global_loss"]:
        metric(name + "_by_epoch", "training_records", s, "Complete original training " + name + " for epochs 1..25 from this new run; preserve native step/epoch aggregation.", "loss by epoch", "original logger's epoch aggregation, explicitly bound to code", ["A0 25-epoch raw history and resolved logger configuration"])
        metric(name + "_final_epoch", "training_records", s, "Original logged " + name + " at fixed epoch 25, not selected best test epoch.", "loss", "epoch 25 value", ["A0 epoch 25 history"])
for name in ["total_loss", "rank_loss", "contrast_loss", "global_loss"]:
    for suffix in ["by_epoch", "final_epoch"]:
        metric(name + "_" + suffix, "training_records", "mean", "Mean new-run " + name + " across all seeds at corresponding fixed epochs.", "loss by epoch" if suffix == "by_epoch" else "loss", "mean of all three corresponding unrounded seed values; no skip-missing", [f"training_records.{name}_{suffix}.{s}" for s in SEEDS])
for s in SEEDS:
    metric("visible_prior_edges", "task_prior", s, "Recount edges actually retained in rebuilt split sidecar; expected same input visibility, even if priors numerically match old values.", "edge", "count sidecar source edges", ["A0 freshly aggregated sidecar", "A0 split/root maps"])
    for name, definition in [
        ("prior_root_exclusion", "No visible sidecar record belongs to any scored test root or sibling root."),
        ("prior_rowmap_alignment", "Exact equality of model/dataset rowmaps and root map with evaluation embeddings."),
        ("sidecar_rebuilt_and_hash_bound", "New aggregation output is bound to A0 graph, split and embeddings; no old sidecar measurement reuse."),
    ]:
        metric(name, "task_prior", s, definition, "boolean and evidence", "all required equality/exclusion checks", ["A0 sidecar manifest", "A0 split audit", "A0 row maps and prior aggregation log"], category="correctness_gate", final_status="verified")

INPUT_DEFINITIONS = {
    "candidate_models": "Actual row count of model row map and embeddings; preserve original mappedID order.",
    "dataset_task_nodes": "Actual dataset-task row count and contiguous mappedIDs.",
    "matched_card_nodes": "Actual exact-or-parent trustworthy card matches in frozen node/card joins.",
    "unknown_model_size_percent": "100 * actual unknown-size model count / model count; retain numerator and denominator.",
    "other_model_family_percent": "100 * actual Other-family model count / model count; retain numerator and denominator.",
    "resolved_lineage_percent": "100 * resolved declared parent relations / declared parent relations under original rule; retain both counts.",
}
for rec in [r for r in REFS if r["scope"] == "frozen_input_audit"]:
    name = rec["name"]
    metric(name, "frozen_input_audit", "shared", INPUT_DEFINITIONS.get(name, "Recount " + name.replace("_", " ") + " from actual frozen source records/arrays using the original rule; do not copy provenance report counts."),
           rec["unit"], "count/ratio/file stat using actual frozen inputs; record numerator/denominator if ratio",
           ["A0_SOURCE_MANIFEST.json actual frozen file bindings", "A0.1 actual input recount audit", "original source arrays/tables/shards"], category="input_audit")
for name, definition, unit in [
    ("matched_card_percent", "100 * exact-or-parent matched card nodes / dataset-task nodes; retain numerator and denominator.", "percent"),
    ("input_file_bytes", "Actual bytes for every frozen source file, including graph arrays, maps, features and vocabularies.", "named byte counts"),
    ("node_feature_shapes_and_dtypes", "Actual array shapes and dtypes for model/dataset features and categorical IDs.", "named structured record"),
    ("categorical_vocab_cardinalities", "Actual model size/family and dataset task/class/arity cardinalities.", "named row counts"),
    ("rowmap_identity_and_endpoint_checks", "Actual unique contiguous IDs, hub prefix, historical append order and valid graph endpoints.", "named boolean checks"),
    ("input_hash_match", "Actual file SHA256 versus bound expected hashes; archive all mismatches, not just meta.json digests.", "named checks and hashes"),
]:
    metric(name, "frozen_input_audit", "shared", definition, unit, "per actual artifact", ["A0.1 frozen input audit", "A0_SOURCE_MANIFEST.json"], category="input_audit", final_status="verified" if "checks" in name or "hash" in name else "recomputed")
for name, definition in [
    ("seven_columns_zero", "Every dataset node has zero in columns 448..453 and 455 in the newly written graph."),
    ("all_other_graph_content_unchanged", "All other feature columns, arrays, categorical IDs, vocabularies, rowmaps, relation endpoints/order/weights match frozen input."),
    ("new_graph_hash_verified", "Actual new files pass SHA256 verification, meta.json files hashes updated, new digest differs from original."),
]:
    metric(name, "feature_repair_verification", "shared", definition, "boolean and diff evidence", "all elements/files", ["A0.2 new graph diff report", "actual SHA256-verified graph artifacts"], category="correctness_gate", final_status="verified")
for s in SEEDS:
    for name, definition in [
        ("checkpoint_graph_binding", "Strict checkpoint load verifies new graph digest and all training dependencies."),
        ("embedding_shape_finite_norm_and_order", "New eval/full embeddings have expected dimensions, finite values, L2 norms and mappedID order under recorded tolerance."),
        ("chunked_inference_agreement", "Chunked inference matches whole-graph inference within original archived tolerance, bound from implementation audit."),
        ("final_output_candidate_membership", "Returned top10 IDs are unique and belong to that query's actual unique HNSW1000 pool."),
        ("fusion_and_tie_consistency", "Raw scores equal fixed fusion with no min-max; all fused references and saved rankings use original fixed label-free tie policy."),
        ("independent_metric_recomputation", "A0.7 independent recalculation from raw query/timing outputs agrees with evaluator and full inventory."),
    ]:
        metric(name, "reproducibility_checks", s, definition, "boolean and evidence", "all required per-query/per-artifact checks", ["A0 independent raw-output audit", "A0 graph/checkpoint/export/index manifests", "A0 metric artifacts"], category="correctness_gate", final_status="verified")

# The original scalar summaries retain min and max as well as the mean.
# Register these explicitly for every three-seed scalar mean already required.
for record in list(METRICS):
    if record["seed"] != "mean" or record["unit"] not in {
        "fraction", "percent", "millisecond", "second", "byte", "GiB", "loss",
        "1-based position", "1-based rank", "normalized rank", "ratio",
    }:
        continue
    for stat in ("min", "max"):
        metric(record["name"], record["scope"], stat,
               record["definition"], record["unit"],
               f"{stat} of all three complete unrounded seed values; original summary diagnostic",
               [f"{record['scope']}.{record['name']}.{s}" for s in SEEDS],
               category=record["category"], note="Historical min/max not separately published; never derive extra old values from rounded tables.")

CONFIG_AUDIT = OUT / "audit/A0_CONFIG_AUDIT.json"
if CONFIG_AUDIT.exists():
    audit = json.loads(CONFIG_AUDIT.read_text(encoding="utf-8"))
    bindings = audit["runtime_static_bindings"]
    PROTOCOL["required_config_audit"].update({
        "status": "pure_build_config_and_static_behavior_audited",
        "artifact": str(CONFIG_AUDIT), "sha256": hashlib.sha256(CONFIG_AUDIT.read_bytes()).hexdigest(),
        "resolved_config": audit["resolved_config"], "resolved_key_count": audit["resolved_key_count"],
        "all_document_mapped_values_match": audit["all_document_mapped_values_match"],
        "doc_linked_identity_checks_all_equal": all(c["exact_config_equal"] for c in audit["doc_linked_config_identity_checks"]),
        "remaining_execution_bindings": audit["remaining_execution_bindings"],
    })
    PROTOCOL["evaluation"]["fusion"].update({"tie_permutation_seed": None,
        "tie_seed_status": "not applicable: implementation uses fixed uint64 arithmetic bijection, not RNG",
        "implementation_tie_binding": bindings["tie_break"]})
    PROTOCOL["evaluation"]["timing"].update({"warmup_count": "min(50, n_queries)",
        "repetitions": 1, "percentile_interpolation": "NumPy percentile default linear; actual execution version must be recorded",
        "setting_provenance": "implementation-bound static audit, not additional published historical timing facts"})
    PROTOCOL["evaluation"]["ef_calibration"]["query_set_binding"] = "exact_pool.query from the same eligible held-out query set; persist new actual IDs/hash; static implementation binding in audit"
    PROTOCOL["evaluation"]["chunked_tolerance_binding"] = bindings["chunked_inference_check"]
    PROTOCOL["training"]["native_epoch_losses"] = bindings["active_epoch_loss_fields"]
    PROTOCOL["training"]["native_epoch_loss_aggregation"] = bindings["epoch_loss_aggregation"]
    PROTOCOL["status"] = "protocol_and_implementation_configuration_frozen_pending_parent_input_reconciliation"

INVENTORY = {
    "schema_version": "1.0", "stage": "A0.1", "authority": SOURCE, "authorized_plan": PLAN_REF,
    "status": "inventory_registered_all_new_values_pending", "primary_path": PATHS[0],
    "required_fields": ["name", "scope", "seed", "definition", "unit", "aggregation", "old_value", "old_source_section", "old_display_precision", "new_value", "recompute_from", "recomputed_from", "artifact_hashes", "status"],
    "rules": {
        "references": "A0_OLD_REFERENCE.json is read-only historical reference. All new values start null and must be computed from hash-bound A0 artifacts.",
        "old_missing": "not_reported means source does not publish this field/precision; it does not waive new measurement.",
        "summaries": "All three seeds required. Mean of unrounded seed quality values; mean of per-seed ratios; mean of seed latency quantiles. Store total counts separately.",
        "non_scalar_fields": "Persist full native records and expand extra scorer/epoch fields into independently named metrics when implementation audit enumerates them; placeholders do not satisfy final completeness.",
        "final_statuses": ["recomputed", "verified", "undefined with numerator/denominator/reason", "disabled with unchanged-policy reason", "not_applicable for unvisited ef after documented first pass"],
        "unfinished_statuses": ["pending", "missing"],
        "no_claims_from_missing": "Missing values, interrupted measurements and disabled diagnostics are distinct; never replace them by zero or historical values.",
        "historical_precision": "Keep new raw precision; comparison must respect old published precision and approximate labels.",
    },
    "metrics": METRICS,
}


def validate():
    ids = [m["id"] for m in METRICS]
    assert len(ids) == len(set(ids)), "duplicate metric ids"
    old_ids = [r["id"] for r in REFS]
    assert len(old_ids) == len(set(old_ids)), "duplicate reference ids"
    for m in METRICS:
        assert set(INVENTORY["required_fields"]) <= set(m)
        assert m["new_value"] is None and m["status"] == "pending"
        assert m["artifact_hashes"] == {} and m["recomputed_from"] == []
        if m["old_reference_id"]:
            assert m["old_reference_id"] in old_ids
            assert m["old_source_section"] and m["old_display_precision"]
    assert len(PROTOCOL["official_runs"]) == 3
    assert PROTOCOL["permitted_changes"]["feature_repair"]["zero_based_columns"] == [448,449,450,451,452,453,455]
    for p in PATHS:
        for q in QUALITY:
            for s in SEEDS + ["mean"]:
                assert f"{p}.{q}.{s}" in ids


if __name__ == "__main__":
    validate()
    outputs = {"A0_PROTOCOL.json": PROTOCOL, "A0_OLD_REFERENCE.json": OLD_REFERENCE, "A0_METRIC_INVENTORY.json": INVENTORY}
    for filename, value in outputs.items():
        target = OUT / filename
        payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        target.write_text(payload, encoding="utf-8", newline="\n")
        assert json.loads(target.read_text(encoding="utf-8")) == value
        print(json.dumps({"path": str(target), "bytes": target.stat().st_size, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}))
    print(json.dumps({"historical_reference_records": len(REFS), "metric_inventory_records": len(METRICS), "all_new_values_null_and_pending": True, "source_sha256": SOURCE_SHA}))
