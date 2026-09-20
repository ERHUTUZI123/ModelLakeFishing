import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.top1_audit import configs
from ModelLakeFishing.stage2TrainGraphSAGE.top1_baselines import run_configs, OUT
from ModelLakeFishing.stage2TrainGraphSAGE.eval_harness import paired_bootstrap

PRIMARY = ["observed_hit1", "top3_hit1", "regret1", "full2k_gold@1", "full2k_gold@10"]


def g_configs(rows):
    base = configs()["R_mg02"]
    out = {}
    if "G1" in rows:
        c = dict(base); c.update(lambda_global=1.0, global_known_low=False,
                                 global_hard_frac=0.0)
        out["G1"] = c
    if "G1w" in rows:
        c = dict(base); c.update(lambda_global=0.25, global_known_low=False,
                                 global_hard_frac=0.0)
        out["G1w"] = c
    if "G2" in rows:
        c = dict(base); c.update(lambda_global=1.0, global_known_low=True,
                                 global_hard_frac=0.5)
        out["G2"] = c
    if "G2w" in rows:
        c = dict(base); c.update(lambda_global=0.25, global_known_low=True,
                                 global_hard_frac=0.5)
        out["G2w"] = c
    if "G1dm" in rows:
        c = dict(base); c.update(lambda_global=1.0, global_known_low=False,
                                 global_hard_frac=0.0, lambda_dm_contrast=1.0)
        out["G1dm"] = c
    return out


def load_pooled(tag):
    with open(os.path.join(OUT, tag, "pooled_per_dataset.json"), encoding="utf-8") as f:
        return json.load(f)


def gate(control_pooled, cand_pooled, control_agg, cand_agg, control_splits, cand_splits):
    d = {}
    bb = paired_bootstrap(control_pooled, cand_pooled, metric="full2k_gold@10")
    d["gold10_bootstrap"] = bb
    per_split_up = all(
        cand_splits[s]["aggregate"]["full2k_gold@10"] >
        control_splits[s]["aggregate"]["full2k_gold@10"] for s in ("0", "1", "2"))
    d["gold10_up_all_splits"] = per_split_up
    gold_ok = per_split_up or bb["ci_low"] > 0
    hit_ok = cand_agg["observed_hit1"][0] >= control_agg["observed_hit1"][0] - 0.02
    top3_ok = cand_agg["top3_hit1"][0] >= control_agg["top3_hit1"][0] - 0.02
    regret_ok = cand_agg["regret1"][0] <= control_agg["regret1"][0] + 0.005
    d.update(hit_ok=hit_ok, top3_ok=top3_ok, regret_ok=regret_ok, gold_gate=gold_ok)
    verdict = "PROMOTE" if (gold_ok and hit_ok and top3_ok and regret_ok) else \
              ("INVESTIGATE" if gold_ok else "REJECT")
    return verdict, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", default=["G1", "G2"])
    ap.add_argument("--tag", default="G")
    ap.add_argument("--control_tag", default="T0")
    ap.add_argument("--control_name", default="R_mg02")
    args = ap.parse_args()

    cfgs = g_configs(args.rows)
    results, pooled, summary = run_configs(cfgs, tag=args.tag, extra_note="Phase 1 global negatives.")
    with open(os.path.join(OUT, args.tag, "pooled_per_dataset.json"), "w", encoding="utf-8") as f:
        json.dump(pooled, f)

    ctrl_pooled_all = load_pooled(args.control_tag)
    ctrl_pooled = ctrl_pooled_all[args.control_name]
    with open(os.path.join(OUT, args.control_tag, f"{args.control_name}.json"), encoding="utf-8") as f:
        ctrl = json.load(f)

    report = {}
    for name in cfgs:
        verdict, details = gate(ctrl_pooled, pooled[name],
                                ctrl["aggregate_over_splits"],
                                results[name]["aggregate_over_splits"],
                                ctrl["splits"], results[name]["splits"])
        report[name] = {"verdict": verdict, **details}
        print(f"\n[{name}] verdict={verdict}")
        print(f"  gold@10 paired: delta={details['gold10_bootstrap']['mean_delta']:+.4f} "
              f"CI [{details['gold10_bootstrap']['ci_low']:+.4f}, "
              f"{details['gold10_bootstrap']['ci_high']:+.4f}] "
              f"up_all_splits={details['gold10_up_all_splits']}")
        print(f"  local gates: hit_ok={details['hit_ok']} top3_ok={details['top3_ok']} "
              f"regret_ok={details['regret_ok']}")
    with open(os.path.join(OUT, args.tag, "gate_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved {os.path.join(OUT, args.tag)}")


if __name__ == "__main__":
    main()
