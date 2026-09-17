"""Independent read-only checks of library numerical claims from A0 raw arrays."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
REPORT = ROOT / "docs/1M/A0_runs/A0_7/results/A0_REPORT.json"
RAW = Path("D:/research/model_lake/data/data1m/a0_20260912/metrics")
report = json.loads(REPORT.read_text(encoding="utf-8"))
measurements = report["new_measurements"]
checks = []

def check(key, value):
    expected = measurements[key]["value"]
    assert np.isclose(value, expected, rtol=0, atol=1e-12), (key, value, expected)
    checks.append({"metric": key, "independent_value": float(value), "matches_A0_REPORT": True})

for seed in range(3):
    with np.load(RAW / f"a0_hnsw_s{seed}.npz", allow_pickle=False) as z:
        off, ids, perf = z["observed_offsets"], z["observed_ids"], z["observed_values"]
        top, pool, roots = z["top10"], z["model"], z["root"]
        hit1, hit10, hit3, hitgap, coverage = [], [], [], [], []
        for i in range(len(top)):
            mi, vi = ids[off[i]:off[i+1]], perf[off[i]:off[i+1]]
            gold = mi[np.argmax(vi)]
            best3 = mi[np.argsort(-vi)[:3]]
            near = mi[vi >= np.max(vi) - 0.01]
            hit1.append(gold == top[i, 0])
            hit10.append(gold in top[i])
            hit3.append(bool(np.isin(top[i], best3).any()))
            hitgap.append(bool(np.isin(top[i], near).any()))
            coverage.append(gold in pool[i])
        for suffix, values in [("gold_at_1", hit1), ("gold_at_10", hit10),
                               ("top3_at_10", hit3), ("gold_gap_at_10", hitgap),
                               ("actual_hnsw_gold_coverage_at_1000", coverage)]:
            check(f"hnsw1000_task_prior.{suffix}.{seed}", np.mean(values))
        root_macro = np.mean([np.asarray(hit10)[roots == r].mean() for r in np.unique(roots)])
        check(f"hnsw1000_task_prior.root_macro_gold_at_10.{seed}", root_macro)
        assert np.array_equal(z["hnsw_ns"] + z["rerank_ns"], z["total_ns"])
        for prefix, array in [("hnsw", "hnsw_ns"), ("prior_rerank", "rerank_ns"), ("total", "total_ns")]:
            for q in (50, 95):
                check(f"retrieval_cost.{prefix}_latency_p{q}_ms.{seed}", np.percentile(z[array], q) / 1e6)
        check(f"index_cost.index_bytes.{seed}", (RAW / f"hnsw_a0_s{seed}.bin").stat().st_size)
        with np.load(RAW / f"a0_exact_s{seed}.npz", allow_pickle=False) as exact:
            # Exact and HNSW pools have unique model IDs; compute membership overlap.
            assert np.array_equal(exact["query"], z["query"])
            recalls = [np.isin(a, b, assume_unique=True).mean() for a, b in zip(z["calibration_ids"], exact["model"])]
            check(f"ann_calibration.selected_recall_at_1000.{seed}", np.mean(recalls))

out = {
    "source_report": str(REPORT),
    "source_report_sha256": hashlib.sha256(REPORT.read_bytes()).hexdigest(),
    "status": "PASS",
    "checks": checks,
    "check_count": len(checks),
    "scope": "Independent gold/top3/gap/root-macro/coverage/ANN/timing arithmetic from raw NPZ and actual index file size; no other result series read.",
}
Path(__file__).with_name("audit_numeric_check.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": "PASS", "checks": len(checks)}))
