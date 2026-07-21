"""
w2_phase.py -- v4 W2 rows (plan §0.3): refine the ONLY working lever, the
lake objective, on top of the frozen candidate. Control + one change per row:

    L1L3 : v3 candidate replica (control)
    L1b  : L1L3 with n_neg 64 -> 256 (sampled-softmax denominator variance
           drops ~linearly in sample count; the cheapest row on the board)
    L4   : L1L3 + IPW positives, w(m) = (deg(m)+1)^-0.5 (v4's one new
           theoretical lever: the positive side of the objective is hub-biased
           too; this is the exact dual of the logQ negative correction)
    L2b  : L1L3 + one-shot ALIBI hard negatives at ep10 (L2 redesign: mined
           models survive only with a popularity or task-mismatch alibi, so
           hidden gold is never pushed down)

Gate vs L1L3 (five-metric rule + z_m PR). Run (repo root):
  ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.stage2TrainGraphSAGE.w2_phase
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

from ModelLakeFishing.stage2TrainGraphSAGE.top1_gphase import gate, PRIMARY  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.l_phase import l_configs  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.top1_baselines import run_configs, OUT  # noqa: E402

ROWS = ["L1L3", "L1b", "L4", "L2b"]
TAG = "v4_w2phase"


def w2_configs(rows):
    l1l3 = l_configs(["L1L3"])["L1L3"]
    out = {}
    if "L1L3" in rows:
        out["L1L3"] = dict(l1l3)
    if "L1b" in rows:
        out["L1b"] = dict(l1l3, global_n_neg=256)
    if "L4" in rows:
        out["L4"] = dict(l1l3, pos_ipw_beta=0.5)
    if "L2b" in rows:
        out["L2b"] = dict(l1l3, hard_mine_epoch=10, hard_k=20, n_hard=16,
                          hard_alibi=True, alibi_deg_quantile=0.9)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", default=ROWS)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfgs = w2_configs(args.rows)
    kw = {} if args.epochs is None else {"epochs": args.epochs}
    results, pooled, summary = run_configs(
        cfgs, tag=TAG,
        extra_note="v4 W2 row (lake-objective refinement; see w2_phase.py).", **kw)

    diag = {n: {k: [results[n]["splits"][s]["train_diag"].get(k)
                    for s in ("0", "1", "2")] for k in ("z_m_pr", "z_d_pr", "tau_macro")}
            for n in cfgs}
    report = {"rows": {n: summary["aggregates"][n] for n in cfgs},
              "train_diag": diag, "gates": {}}
    for n in cfgs:
        if n == "L1L3":
            continue
        verdict, details = gate(
            pooled["L1L3"], pooled[n],
            results["L1L3"]["aggregate_over_splits"], results[n]["aggregate_over_splits"],
            results["L1L3"]["splits"], results[n]["splits"])
        pr_ok = bool(np.mean(diag[n]["z_m_pr"]) >= np.mean(diag["L1L3"]["z_m_pr"]) - 1e-9)
        details["zm_pr_not_down"] = pr_ok
        if verdict == "PROMOTE" and not pr_ok:
            verdict = "INVESTIGATE"
        report["gates"][n] = {"verdict": verdict, **details}

    os.makedirs(os.path.join(OUT, TAG), exist_ok=True)
    with open(os.path.join(OUT, TAG, "W2_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    md = ["# v4 W2-phase -- lake-objective refinement (control = L1L3)", "",
          "| row | " + " | ".join(PRIMARY) + " | z_d PR | z_m PR | verdict |",
          "|---|" + "---|" * (len(PRIMARY) + 3)]
    for n in cfgs:
        ag = summary["aggregates"][n]
        cells = [f"{ag[m][0]:.3f}±{ag[m][1]:.3f}" for m in PRIMARY]
        v = report["gates"].get(n, {}).get("verdict", "control")
        md.append(f"| {n} | " + " | ".join(cells) +
                  f" | {np.mean(diag[n]['z_d_pr']):.1f} "
                  f"| {np.mean(diag[n]['z_m_pr']):.1f} | {v} |")
    with open(os.path.join(OUT, TAG, "W2_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(json.dumps({n: report["gates"].get(n, {}).get("verdict", "control")
                      for n in cfgs}, indent=1))


if __name__ == "__main__":
    main()
