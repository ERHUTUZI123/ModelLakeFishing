"""
z_phase.py -- v3 Z-track rows (plan §0.3): rebuild z_d conditioning.

    Z0   : G2 replica (control)
    Z1   : Z0 + learnable frozen-view projection 4618->128 (discrete share
           0.6% -> 18%; the F4 structural-down-weighting lesson applied to the
           dataset side, but LEARNABLE: training picks the surviving subspace)
    Z2   : Z0 + task repair + dataset-dataset push-apart (different-task pairs
           repel; hinge on cosine, margin 0.2). Repair is Z2's stated
           dependency (plan: same dependency as L3) -- attribution caveat is
           documented, L3-alone is known REJECT on pools mode.
    L1L3 : v3 candidate replica (second reference for the stack rows)
    Z1L  : L1L3 + Z1
    ZL   : L1L3 + Z1 + Z2  (full v3 stack: L-track + Z-track)

Gates: Z1/Z2 vs Z0; Z1L/ZL vs L1L3 (does Z add on top of the frozen
candidate?). z_d participation ratio is reported for every row -- it is the
direct target of this track (intrinsic dim ~2 today).

Run (repo root):
  ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.stage2TrainGraphSAGE.z_phase
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
from ModelLakeFishing.stage2TrainGraphSAGE.l_phase import l_configs  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.top1_baselines import run_configs, OUT  # noqa: E402

ROWS = ["Z0", "Z1", "Z2", "L1L3", "Z1L", "ZL"]
TAG = "v3_zphase"
Z1_KW = dict(dataset_frozen_proj_dim=128)
Z2_KW = dict(repair_dataset_task=True, lambda_zpush=1.0, zpush_margin=0.2)


def z_configs(rows):
    g2 = g_configs(["G2"])["G2"]
    l1l3 = l_configs(["L1L3"])["L1L3"]
    out = {}
    if "Z0" in rows:
        out["Z0"] = dict(g2)
    if "Z1" in rows:
        out["Z1"] = dict(g2, **Z1_KW)
    if "Z2" in rows:
        out["Z2"] = dict(g2, **Z2_KW)
    if "L1L3" in rows:
        out["L1L3"] = dict(l1l3)
    if "Z1L" in rows:
        out["Z1L"] = dict(l1l3, **Z1_KW)
    if "ZL" in rows:
        out["ZL"] = dict(l1l3, **Z1_KW, **Z2_KW)
    return out


CONTROL_OF = {"Z1": "Z0", "Z2": "Z0", "Z1L": "L1L3", "ZL": "L1L3"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", default=ROWS)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfgs = z_configs(args.rows)
    kw = {} if args.epochs is None else {"epochs": args.epochs}
    results, pooled, summary = run_configs(
        cfgs, tag=TAG,
        extra_note="v3 Z-track row (z_d conditioning; see z_phase.py).", **kw)

    diag = {n: {k: [results[n]["splits"][s]["train_diag"].get(k)
                    for s in ("0", "1", "2")] for k in ("z_m_pr", "z_d_pr", "tau_macro")}
            for n in cfgs}
    report = {"rows": {n: summary["aggregates"][n] for n in cfgs},
              "train_diag": diag, "gates": {}}
    for n, ctrl in CONTROL_OF.items():
        if n not in cfgs or ctrl not in cfgs:
            continue
        verdict, details = gate(
            pooled[ctrl], pooled[n],
            results[ctrl]["aggregate_over_splits"], results[n]["aggregate_over_splits"],
            results[ctrl]["splits"], results[n]["splits"])
        pr_ok = bool(np.mean(diag[n]["z_m_pr"]) >= np.mean(diag[ctrl]["z_m_pr"]) - 1e-9)
        details.update(control=ctrl, zm_pr_not_down=pr_ok)
        if verdict == "PROMOTE" and not pr_ok:
            verdict = "INVESTIGATE"
        report["gates"][n] = {"verdict": verdict, **details}

    os.makedirs(os.path.join(OUT, TAG), exist_ok=True)
    with open(os.path.join(OUT, TAG, "Z_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    md = ["# v3 Z-phase -- z_d conditioning (controls: Z0 for Z1/Z2, L1L3 for stacks)", "",
          "| row | " + " | ".join(PRIMARY) + " | z_d PR | z_m PR | verdict |",
          "|---|" + "---|" * (len(PRIMARY) + 3)]
    for n in cfgs:
        ag = summary["aggregates"][n]
        cells = [f"{ag[m][0]:.3f}±{ag[m][1]:.3f}" for m in PRIMARY]
        v = report["gates"].get(n, {}).get("verdict", "control")
        md.append(f"| {n} | " + " | ".join(cells) +
                  f" | {np.mean(diag[n]['z_d_pr']):.1f} "
                  f"| {np.mean(diag[n]['z_m_pr']):.1f} | {v} |")
    with open(os.path.join(OUT, TAG, "Z_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(json.dumps({n: report["gates"].get(n, {}).get("verdict", "control")
                      for n in cfgs}, indent=1))


if __name__ == "__main__":
    main()
