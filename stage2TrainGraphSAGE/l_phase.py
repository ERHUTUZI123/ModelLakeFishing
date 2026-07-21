"""
l_phase.py -- v3 L-track rows (plan §0.3): retrain the OBJECTIVE, not the
features. Control + one change per row:

    L0   : G2 replica (control; task-incompatible + known-low pools, hard 0.5)
    L3   : L0 + dataset task_type repair applied (pools coverage 30->50 of 64;
           also fixes the z_d encoder's task_type rows -- one conceptual change:
           "repair wrong metadata")
    L1   : L0 with the global term swapped to whole-lake logQ-corrected sampled
           softmax (no pools; negatives from ALL 2000 models ~ q(m) ∝
           (deg+1)^0.75, logits corrected by -log q). The supervision blind
           spot (E9) closes: unlabeled hubs finally appear as negatives.
    L1L3 : L1 + task repair (metadata fixes the ENCODER side too)
    L2   : L1L3 + hard sets mined ONCE at epoch 10 (top-20 currently beating
           the labeled best, train graph only, never re-mined)

Same harness as G/F phases: five-metric clean eval, provenance-fixed graph,
promote-gate vs L0 plus the z_m-PR-not-down check.

Run (repo root):
  ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.stage2TrainGraphSAGE.l_phase \
      --rows L0 L3 L1 L1L3 L2
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

from ModelLakeFishing.stage2TrainGraphSAGE.top1_gphase import g_configs, gate, PRIMARY  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.top1_baselines import run_configs, OUT  # noqa: E402

ROWS = ["L0", "L3", "L1", "L1L3", "L2"]
TAG = "v3_lphase"


def l_configs(rows):
    g2 = g_configs(["G2"])["G2"]
    lake = dict(g2, global_mode="lake", lake_alpha=0.75, lake_n0=1.0)
    out = {}
    if "L0" in rows:
        out["L0"] = dict(g2)
    if "L3" in rows:
        out["L3"] = dict(g2, repair_dataset_task=True)
    if "L1" in rows:
        out["L1"] = dict(lake)
    if "L1L3" in rows:
        out["L1L3"] = dict(lake, repair_dataset_task=True)
    if "L2" in rows:
        out["L2"] = dict(lake, repair_dataset_task=True,
                         hard_mine_epoch=10, hard_k=20, n_hard=16)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", default=ROWS)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfgs = l_configs(args.rows)
    kw = {} if args.epochs is None else {"epochs": args.epochs}
    results, pooled, summary = run_configs(
        cfgs, tag=TAG,
        extra_note="v3 L-track row (objective rework; see l_phase.py).", **kw)

    zpr = {n: [results[n]["splits"][s]["train_diag"].get("z_m_pr")
               for s in ("0", "1", "2")] for n in cfgs}
    report = {"rows": {n: summary["aggregates"][n] for n in cfgs},
              "z_m_pr": zpr, "gates": {}}
    if "L0" in cfgs:
        for n in cfgs:
            if n == "L0":
                continue
            verdict, details = gate(
                pooled["L0"], pooled[n],
                results["L0"]["aggregate_over_splits"], results[n]["aggregate_over_splits"],
                results["L0"]["splits"], results[n]["splits"])
            pr_ok = bool(np.mean(zpr[n]) >= np.mean(zpr["L0"]) - 1e-9)
            details["zm_pr_not_down"] = pr_ok
            if verdict == "PROMOTE" and not pr_ok:
                verdict = "INVESTIGATE"
            report["gates"][n] = {"verdict": verdict, **details}

    os.makedirs(os.path.join(OUT, TAG), exist_ok=True)
    with open(os.path.join(OUT, TAG, "L_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    md = ["# v3 L-phase -- objective rework (control = L0/G2 replica)", "",
          "| row | " + " | ".join(PRIMARY) + " | z_m PR (mean) | verdict |",
          "|---|" + "---|" * (len(PRIMARY) + 2)]
    for n in cfgs:
        ag = summary["aggregates"][n]
        cells = [f"{ag[m][0]:.3f}±{ag[m][1]:.3f}" for m in PRIMARY]
        v = report["gates"].get(n, {}).get("verdict", "control")
        md.append(f"| {n} | " + " | ".join(cells) +
                  f" | {np.mean(zpr[n]):.1f} | {v} |")
    with open(os.path.join(OUT, TAG, "L_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(json.dumps({n: report["gates"].get(n, {}).get("verdict", "control")
                      for n in cfgs}, indent=1))


if __name__ == "__main__":
    main()
