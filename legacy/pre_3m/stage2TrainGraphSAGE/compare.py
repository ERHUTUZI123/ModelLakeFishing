"""
compare.py -- paired comparison of two ablation artifacts (the promotion gate).

Loads two {name}.json artifacts written by ablation.run and reports, over the
datasets BOTH configs scored on the SAME splits, the paired-bootstrap delta of
within-dataset tau and of head-retrieval metrics. The guide's acceptance rule:
promote only when the 95% paired interval for delta tau_macro excludes zero (or
the improvement reproduces across all fixed splits) AND head retrieval does not
materially regress.

Run:
  python -m ModelLakeFishing.stage2TrainGraphSAGE.compare --base B0 --cand B2
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.eval_harness import paired_bootstrap  # noqa: E402

ABL = os.path.join(_HERE, "artifacts", "ablation")


def _load(name):
    path = name if name.endswith(".json") else os.path.join(ABL, f"{name}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pool_per_dataset(art, block, metric):
    """Pool per-(split,init) per-dataset values into one dict keyed by
    'tag::dataset' so the pairing is exact across the SAME run tags."""
    out = {}
    for tag, per in art[block].items():
        for d, row in per.items():
            if metric in row:
                out[f"{tag}::{d}"] = {metric: row[metric]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cand", required=True)
    args = ap.parse_args()
    a, b = _load(args.base), _load(args.cand)

    print(f"BASE {a['name']}: tau_macro {a['aggregate']['tau_macro'][0]:.4f} "
          f"+/- {a['aggregate']['tau_macro'][1]:.4f}  mean_cos {a['aggregate']['mean_cos'][0]:.4f}")
    print(f"CAND {b['name']}: tau_macro {b['aggregate']['tau_macro'][0]:.4f} "
          f"+/- {b['aggregate']['tau_macro'][1]:.4f}  mean_cos {b['aggregate']['mean_cos'][0]:.4f}")

    print("\n=== paired bootstrap (cand - base), per dataset over identical run tags ===")
    pa = _pool_per_dataset(a, "per_dataset_tau", "tau")
    pb = _pool_per_dataset(b, "per_dataset_tau", "tau")
    r = paired_bootstrap(pa, pb, metric="tau")
    gate = "EXCLUDES 0 (promote)" if (r["ci_low"] > 0 or r["ci_high"] < 0) else "includes 0 (no gate)"
    print(f"  tau: delta={r['mean_delta']:+.4f}  95% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  "
          f"P(>0)={r['prob_positive']:.2f}  n={r['n_paired']}  -> {gate}")

    print("\n=== head-retrieval regression check (aggregate macro) ===")
    for k in ["hit@10", "recall_top3@10", "ndcg@10", "ndcg@50", "regret@10"]:
        av = a["head_aggregate"].get(k, [float("nan")])[0]
        bv = b["head_aggregate"].get(k, [float("nan")])[0]
        d = bv - av
        # regret: lower is better; others: higher is better
        worse = (d > 1e-4) if k.startswith("regret") else (d < -1e-4)
        print(f"  {k:<16} base={av:.4f}  cand={bv:.4f}  delta={d:+.4f}  {'REGRESS' if worse else 'ok'}")


if __name__ == "__main__":
    main()
