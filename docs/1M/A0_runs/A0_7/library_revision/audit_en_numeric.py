"""Check the EN draft's measured numeric table cells against A0 evidence."""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
DRAFT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "EVIDENCE_SOURCE_LIBRARY_en.draft.md"
REPORT = HERE.parent / "results/A0_REPORT.json"
report = json.loads(REPORT.read_text(encoding="utf-8"))
measured = report["new_measurements"]
lines = DRAFT.read_text(encoding="utf-8").splitlines()
checks = []

def value(scope, name, seed):
    return measured[f"{scope}.{name}.{seed}"]["value"]

def table(header):
    index = lines.index(header)
    rows = []
    for line in lines[index + 2:]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows

def check(label, cell, expected):
    text = cell.replace(",", "").replace("%", "").replace("**", "")
    actual = float(text)
    digits = len(text.split(".", 1)[1]) if "." in text else 0
    assert abs(actual - expected) <= 0.5 * 10 ** -digits + 1e-12, (label, actual, expected)
    checks.append({"label": label, "displayed": cell, "source_value": expected})

rows = table("| Path | Seed 0 gold@10 | Seed 1 gold@10 | Seed 2 gold@10 | Equal-weight mean |")
for row, scope in zip(rows, ("exact_full_lake_task_prior", "exact1000_task_prior", "hnsw1000_task_prior")):
    for cell, seed in zip(row[1:], (0, 1, 2, "mean")):
        check(f"{scope}.gold_at_10.{seed}", cell, value(scope, "gold_at_10", seed))

rows = table("| Split seed | gold@1 | gold@10 | top3@10 | gold-gap@10 | Root-macro gold@10 | Median gold position, conditional on retrieval |")
for row, seed in zip(rows, (0, 1, 2, "mean")):
    names = ("gold_at_1", "gold_at_10", "top3_at_10", "gold_gap_at_10", "root_macro_gold_at_10", "median_gold_position_when_retrieved")
    for cell, name in zip(row[1:], names):
        if cell != "—":
            check(f"hnsw1000_task_prior.{name}.{seed}", cell, value("hnsw1000_task_prior", name, seed))

rows = table("| Diagnostic | Seed 0 | Seed 1 | Seed 2 | Mean of seed ratios |")
names = ("exact_pool_retention", "ann_retention", "overall_retention", "exact_pool_gold_coverage_at_1000",
         "full_fused_top10_in_exact_pool", "actual_hnsw_gold_coverage_at_1000")
for row, name in zip(rows, names):
    scope = "hnsw1000_task_prior" if name == names[-1] else "retrieval_diagnostics"
    for cell, seed in zip(row[1:], (0, 1, 2, "mean")):
        check(f"{scope}.{name}.{seed} percent", cell, 100 * value(scope, name, seed))

rows = table("| Split seed | Selected ef_search | Recall@1000 |")
for row, seed in zip(rows, range(3)):
    for cell, name in zip(row[1:], ("selected_ef_search", "selected_recall_at_1000")):
        check(f"ann_calibration.{name}.{seed}", cell, value("ann_calibration", name, seed))

rows = table("| Seed | HNSW p50 ms | HNSW p95 ms | Prior/fusion p50 ms | Prior/fusion p95 ms | Total p50 ms | Total p95 ms |")
names = [f"{prefix}_latency_p{q}_ms" for prefix in ("hnsw", "prior_rerank", "total") for q in (50, 95)]
for row, seed in zip(rows, (0, 1, 2, "mean")):
    for cell, name in zip(row[1:], names):
        check(f"retrieval_cost.{name}.{seed}", cell, value("retrieval_cost", name, seed))

rows = table("| Seed | Index bytes | Index GiB | Build seconds |")
for row, seed in zip(rows, range(3)):
    for cell, name in zip(row[1:], ("index_bytes", "index_gib", "build_seconds")):
        check(f"index_cost.{name}.{seed}", cell, value("index_cost", name, seed))

rows = table("| Recorded interval (seconds) | Seed 0 | Seed 1 | Seed 2 |")
names = ("training_seconds", "export_seconds", "prior_build_seconds", "exact_full_reference_seconds", "complete_evaluation_seconds")
for row, name in zip(rows, names):
    for cell, seed in zip(row[1:], range(3)):
        check(f"pipeline_cost.{name}.{seed}", cell, value("pipeline_cost", name, seed))

rows = table("| Recorded resource (GiB) | Seed 0 | Seed 1 | Seed 2 |")
names = ("training_peak_rss_bytes", "training_peak_vram_bytes", "evaluation_peak_rss_bytes", "evaluation_peak_vram_bytes")
for row, name in zip(rows, names):
    for cell, seed in zip(row[1:], range(3)):
        check(f"pipeline_cost.{name}.{seed} GiB", cell, value("pipeline_cost", name, seed) / 2**30)

rows = table("| Source | Retained edges |")
for row in rows:
    name = row[0].strip("`*")
    metric = "supervision_edges" if name == "Total" else f"retained_{name}_edges"
    check(f"frozen_input_audit.{metric}.shared", row[1], value("frozen_input_audit", metric, "shared"))

rows = table("| Relation | Directed edges in the stored graph |")
for row in rows:
    name = row[0].strip("`*")
    metric = "stored_total_directed_edges" if name == "Total" else "stored_similar_to_edges" if name == "similar_to" else f"{name}_edges"
    check(f"frozen_input_audit.{metric}.shared", row[1], value("frozen_input_audit", metric, "shared"))

snapshots = json.loads((ROOT / "docs/1M/A0_runs/audit/A0_SNAPSHOT_AUDIT.json").read_text(encoding="utf-8"))["snapshots"]
rows = table("| Frozen snapshot | Parsed records | Shards | Unique raw IDs | Unique IDs after strip/lower |")
for row, snap in zip(rows, snapshots):
    for cell, name in zip(row[1:], ("parsed_records", "shards", "unique_raw_repository_ids", "unique_normalized_repository_ids")):
        check(f"snapshot.{row[0]}.{name}", cell, snap["measured"][name])

out = {"status": "PASS", "numeric_table_cells": len(checks),
       "document": str(DRAFT),
       "draft_sha256": hashlib.sha256(DRAFT.read_bytes()).hexdigest(),
       "report_sha256": hashlib.sha256(REPORT.read_bytes()).hexdigest(), "checks": checks,
       "scope": "Measured numerical tables against A0 report/snapshot audit, respecting fraction/percent and bytes/GiB conversion. Manual prose review additionally checked aggregation and timing scope."}
(HERE / ("audit_en_numeric_published.json" if len(sys.argv) > 1 else "audit_en_numeric.json")).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": "PASS", "numeric_table_cells": len(checks)}))
