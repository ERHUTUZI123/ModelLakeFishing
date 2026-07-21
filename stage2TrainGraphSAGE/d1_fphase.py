"""
d1_fphase.py -- D1 §5.4 feature-rework ablation, rows F0-F4, one change per row.

    F0 : G2 replica (control; RankNet + global negatives config frozen)
    F1 : F0 + drop e_desc          ("desc is the noise majority" hypothesis)
    F2 : F1 + drop e_fam           ("family is carried by lineage edges")
    F3 : F2 + e_task (16-d)        (model-side task feature -- expected winner)
    F4 : F3 + name 64->16 frozen JL projection (structural down-weighting)

Everything else (splits, seeds, epochs, evaluation, candidate sets) is the
G-phase harness verbatim: five-metric clean evaluator on the provenance-fixed
graph, promote-gate vs F0 = the guide's rule (gold@10 up on all 3 splits or
paired CI>0, observed_hit@1 drop <= 0.02, regret rise <= 0.005) PLUS the G-D1
extra: z_m participation ratio must not drop (feature cuts must not collapse
the embedding space).

Run:  ../.venv/Scripts/python.exe -m ModelLakeFishing.stage2TrainGraphSAGE.d1_fphase \
          --rows F0 F1 F2 F3 F4
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

ROWS = ["F0", "F1", "F2", "F3", "F4"]


def f_configs(rows):
    g2 = g_configs(["G2"])["G2"]
    out = {}
    if "F0" in rows:
        out["F0"] = dict(g2)
    if "F1" in rows:
        out["F1"] = dict(g2, drop_desc=True)
    if "F2" in rows:
        out["F2"] = dict(g2, drop_desc=True, drop_family=True)
    if "F3" in rows:
        out["F3"] = dict(g2, drop_desc=True, drop_family=True, use_model_task=True)
    if "F4" in rows:
        out["F4"] = dict(g2, drop_desc=True, drop_family=True, use_model_task=True,
                         name_proj_dim=16)
    return out


def zm_pr_over_splits(res):
    return [res["splits"][s]["train_diag"].get("z_m_pr") for s in ("0", "1", "2")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", default=ROWS)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfgs = f_configs(args.rows)
    kw = {} if args.epochs is None else {"epochs": args.epochs}
    results, pooled, summary = run_configs(
        cfgs, tag="d1_fphase",
        extra_note="D1 feature-rework row (one change per row; see d1_fphase.py).",
        **kw)

    report = {"rows": {n: summary["aggregates"][n] for n in cfgs},
              "z_m_pr": {n: zm_pr_over_splits(results[n]) for n in cfgs},
              "gates": {}}
    if "F0" in cfgs:
        for n in cfgs:
            if n == "F0":
                continue
            verdict, details = gate(
                pooled["F0"], pooled[n],
                results["F0"]["aggregate_over_splits"], results[n]["aggregate_over_splits"],
                results["F0"]["splits"], results[n]["splits"])
            pr_ok = bool(np.mean(report["z_m_pr"][n]) >=
                         np.mean(report["z_m_pr"]["F0"]) - 1e-9)
            details["zm_pr_not_down"] = pr_ok
            if verdict == "PROMOTE" and not pr_ok:
                verdict = "INVESTIGATE"        # G-D1: PR drop blocks promotion
            report["gates"][n] = {"verdict": verdict, **details}

    out_p = os.path.join(OUT, "d1_fphase", "F_report.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md = ["# D1 F-phase -- feature-rework ablation (control = F0/G2 replica)", "",
          "| row | " + " | ".join(PRIMARY) + " | z_m PR (mean) | verdict |",
          "|---|" + "---|" * (len(PRIMARY) + 2)]
    for n in cfgs:
        ag = summary["aggregates"][n]
        cells = [f"{ag[m][0]:.3f}±{ag[m][1]:.3f}" for m in PRIMARY]
        pr = np.mean(report["z_m_pr"][n])
        v = report["gates"].get(n, {}).get("verdict", "control")
        md.append(f"| {n} | " + " | ".join(cells) + f" | {pr:.1f} | {v} |")
    with open(os.path.join(OUT, "d1_fphase", "F_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(json.dumps({n: report["gates"].get(n, {}).get("verdict", "control")
                      for n in cfgs}, indent=1))
    print("report ->", out_p)


if __name__ == "__main__":
    main()
