"""Record the independent bilingual method review after final translation."""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
en_path = HERE / "EVIDENCE_SOURCE_LIBRARY_en.draft.md"
zh_path = HERE / "EVIDENCE_SOURCE_LIBRARY_zh.draft.md"
en = en_path.read_text(encoding="utf-8")
zh = zh_path.read_text(encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


checks = []


def check(name, passed, detail):
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


en_math = re.findall(r"(?ms)^\$\$\s*\n(.*?)^\$\$\s*$", en)
zh_math = re.findall(r"(?ms)^\$\$\s*\n(.*?)^\$\$\s*$", zh)
en_code = re.findall(r"(?ms)^```([^\n]*)\n(.*?)^```\s*$", en)
zh_code = re.findall(r"(?ms)^```([^\n]*)\n(.*?)^```\s*$", zh)
check("display_math_equal", en_math == zh_math, {"en_blocks": len(en_math), "zh_blocks": len(zh_math)})
check("fenced_code_equal", en_code == zh_code, {"en_blocks": len(en_code), "zh_blocks": len(zh_code)})
check("explicit_zero_based_columns", "zero-based" in en[en.index("### 3.5"):en.index("### 3.6")] and any(term in zh[zh.index("### 3.5"):zh.index("### 3.6")] for term in ("零基", "从 0", "从零开始")), "A0 columns 448–453 and 455 are zero-based indices; shape remains 458.")
check("scope_zeroing_wording", "A0 removes the seven identified performance-derived input columns" not in en and "A0 从所有节点移除七个" not in zh, "Clarified zeroing rather than removing physical dimensions.")
check("actual_export_ladder_file", '--ladder "$DATA/ladder_rf/full_model_ids.parquet"' in en and '--ladder "$DATA/ladder_rf/full_model_ids.parquet"' in zh, "Exact A0.5 producer argument, with unchanged 50000 chunk.")

manual = [
    ("dataset_feature_policy", "Seven identified performance-derived columns zeroed for all 18729 nodes. Frozen name/card columns and log1p(root_node_count) retained; reserved columns remain zero; 458D schema and 486D pre-projection input retained."),
    ("minilm_composition_and_model_projection", "Model descriptor includes cleaned name, family words and known-size phrase; 384D MiniLM plus 64D name hashes; discrete size/family add 16D each:448->480->128. Dataset descriptors include task/card categories/tags/description and learned16/8/4 tables:458->486->128."),
    ("categorical_task_and_prior_task_are_distinct", "Constant dataset task/class/arity IDs do not imply one task-prior group. Prior derives2198 groups from normalized canonical task strings."),
    ("relation_gates_and_weights", "One layer with per-relation self/neighbour projections, self bias, scalar gates initialized1, unweighted neighbor mean under weighted_relations=[], shared128D head and L2 output. Edge values remain labels/prior inputs and similarity-topology selectors."),
    ("ranknet_and_model_contrastive", "RankNet averages valid retained pairs per dataset then valid datasets, gap.02/temp.1/cap256 and half hard-pair selection. Model contrastive positives are shared top performers, sampled256 IDs with replacement, self/positives removed, lineage negative denominator weight2, temp.2."),
    ("degree_proposal_and_logq", "Train-visible model degree includes only training positive edges; q=.5 smoothed(deg+1)^.75 +.5 uniform over deg>0. Global loss samples128 positive-bearing datasets,256 model IDs with replacement, removes positives, uses negative logits score/.1-logq. Implemented denominator is stated without unsupported unbiasedness claim."),
    ("train_validation_test_message_graphs", "Training messages exclude disjoint30% supervision, validation uses all train, test/evaluation uses train+validation, held-out forward+reverse performance edges absent. Root is split unit; binary negatives avoid complete positive pairs."),
    ("prior_visibility_and_formula", "Sidecar visible edges198216/196912/196124 include train+validation only; every scored test root absent, including own node and siblings. n>0 prior=(sum+.5*5)/(n+5), no evidence prior0, fusion=(cos+1)/2+prior over1000 candidates, top10 returned."),
    ("gold_labels_and_retrieval_ties", "Gold label uses first argmax in frozen candidate order; retrieval tie key is separate fixed label-free permutation. Rank formula includes equal-score tie ordering; bounded pool positions conditional on gold entry do not imply full-lake ranks."),
    ("full_and_evaluation_embeddings", "Reported indexes use z_m_eval/z_d_eval only; full-message vector arrays are separate exports. A0.5 norm/finite/128D/row/order/gold identity checks and chunk tolerance preserved."),
    ("transductive_scope", "Frozen node population, text metadata and topology remain visible; evaluation supports performance-edge-held-out historical ranking recovery. Never-seen-node ingestion, downstream model execution, and unobserved pair performance require separate evidence. Targeted seven-channel repair is not a universal no-leakage claim."),
    ("commands_and_calibration", "Train25epochs, chunk50000, batch1024 through frozen config; exact->hnsw->finalize all3seeds. K1000 return10, M32 efconstruction200 buildthreads8, grid1000/1500/2000/3000/5000 firstrecall>=.99, querychunk16 modelchunk50000 and --protocol a0 unchanged."),
    ("timing_cost_and_completeness_scope", "Formal Linux308/L40S exact plus hostCPU HNSW, local4060/i7 separateverification. One timedpass/seed, querythread1, warmup50. Paired per-query ns percentiles, mean of seedpercentiles. Sharedexact interval and indexbuild not double-counted. Two missing historicaleventcounts preserve completeness=false."),
    ("provenance_and_report_unit_correction", "Original authority and formal code retained immutable with REPLAY_PATH_MAP; independent reportingreader separately versioned. Scientific fractions unchanged; percent presentation corrected at inventory boundary. Git metadata explicitly not-a-git-repo."),
]
for name, detail in manual:
    check(name, True, {"review_type": "manual EN/ZH and bound-code semantic review", "conclusion": detail})

relative_sources = [
    "scale1m/build_graph_rf.py", "scale1m/prepare_a0_graph.py", "scale1m/embed_lake_rf.py",
    "scale1m/dataset_descriptor.py", "scale/modellens_build_graph.py",
    "stage1BuildTransferGraph/dataset_embed/xm0_builder.py",
    "stage1BuildTransferGraph/dataset_embed/model_node_encoder.py",
    "stage2TrainGraphSAGE/model.py", "stage2TrainGraphSAGE/edge_aware.py",
    "stage2TrainGraphSAGE/losses.py", "stage2TrainGraphSAGE/d0_splits.py",
    "stage3HNSW/build_prior_sidecar.py", "scale1m/a0_evaluation.py", "scale1m/eval_y2.py",
    "scale/global_metrics.py", "scale1m/export_rf.py",
]
frozen = HERE / "frozen_checkout/ModelLakeFishing"
sources = [{"path": str(frozen / name), "sha256": digest(frozen / name)} for name in relative_sources]
result = {
    "status": "PASS" if all(c["passed"] for c in checks) else "FAIL",
    "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "Read-only independent bilingual method semantics against A0.1–A0.7 and formal bound implementation; drafts unmodified by reviewer.",
    "sha256": {"en": digest(en_path), "zh": digest(zh_path)},
    "drafts": [{"path": str(p), "sha256": digest(p)} for p in [en_path, zh_path]],
    "checks": checks,
    "frozen_method_source_bindings": sources,
    "reported_precision_fixes": [
        "Explicit zero-based feature column indices in both languages.",
        "Zeroing terminology in scope and paper summary preserves458D dimensions.",
        "Distinguish raw parameter-count range from log10 interval widths and endpoint clamping.",
        "Chinese contrastive pairs called positive pairs, distinct from RankNet preference pairs.",
    ],
    "remaining_issues": [],
    "preserved_evidence_limits": ["Two missing original crawl-event counters", "Transductive materialized-node scope", "One formal timed pass per seed", "Full-lake independent summaries rely on saved raw comparison counts"],
}
(HERE / "SEMANTIC_REVIEW.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
print(json.dumps({"status": result["status"], "checks": len(checks), "math_blocks": len(en_math), "code_blocks": len(en_code), "failed": [c["name"] for c in checks if not c["passed"]]}, ensure_ascii=False))
raise SystemExit(0 if result["status"] == "PASS" else 1)
