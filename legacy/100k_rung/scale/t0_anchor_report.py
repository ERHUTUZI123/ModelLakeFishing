"""
t0_anchor_report.py -- T0.3 gate arithmetic (docs/1M/100kplan.md §3.3).

Reads the exports each `scale.export_ours` run leaves behind (z_m_eval.npy,
z_d_eval.npy, gold_cands.npz), recomputes the per-QUERY gold@K indicators with
the shipped harness, and produces:

  * G-A1 anchor gate : does each post-change config's gold@10 bootstrap CI
                       overlap the pre-change baseline's?
  * G-A4 / D-14      : the full-N^2 vs sampled-negative A/B table.

Why bootstrap over queries rather than over the three seeds: each seed is a
different root-aware SPLIT, so the seeds do not share a query set and a
3-point std is not a confidence interval of anything. Queries are resampled
within their own seed (a cluster bootstrap), which keeps each replicate's
composition honest while still propagating seed-to-seed variation.

Run (repo root):
  ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.scale.t0_anchor_report \
      --config baseline_pre=R0_baseline_pre_s0,R0_baseline_pre_s1,R0_baseline_pre_s2 \
      --config allon_fullneg=... --out ModelLakeFishing/docs/1M/T0_runs/anchor_report.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.top1_eval import five_metric_eval  # noqa: E402

EXPORT_ROOT = os.path.join(_REPO_ROOT, "ModelLakeFishing", "docs", "scale", "P3", "exports")
METRICS = ("full2k_gold@1", "full2k_gold@10", "gold_gap@10", "top3_hit1", "observed_hit1")


def load_run(tag, root=EXPORT_ROOT):
    """Per-query records for one export directory (held-out embeddings only)."""
    d = os.path.join(root, tag)
    zm = np.load(os.path.join(d, "z_m_eval.npy"))
    zd = np.load(os.path.join(d, "z_d_eval.npy"))
    gc = np.load(os.path.join(d, "gold_cands.npz"))
    cands = {int(k): (gc[k][0].astype(np.int64), gc[k][1]) for k in gc.files}
    per = five_metric_eval({"model": torch.from_numpy(zm),
                            "dataset": torch.from_numpy(zd)}, cands)
    report = {}
    p = os.path.join(d, "export_report.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            report = json.load(f)
    return per, report


def cluster_bootstrap(per_by_seed, metric, n_boot=10_000, seed=0, alpha=0.05):
    """Mean of `metric` pooled over seeds, with a cluster (by seed) bootstrap CI."""
    rng = np.random.default_rng(seed)
    arrays = [np.array([r[metric] for r in per.values()], dtype=float)
              for per in per_by_seed]
    point = float(np.mean(np.concatenate(arrays)))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        draw = [a[rng.integers(0, a.size, a.size)] for a in arrays]
        boots[b] = np.mean(np.concatenate(draw))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": point, "ci_lo": float(lo), "ci_hi": float(hi),
            "per_seed": [float(a.mean()) for a in arrays],
            "n_queries": int(sum(a.size for a in arrays))}


def overlap(a, b):
    return not (a["ci_hi"] < b["ci_lo"] or b["ci_hi"] < a["ci_lo"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True,
                    metavar="NAME=tag1,tag2,tag3")
    ap.add_argument("--baseline", default="baseline_pre")
    ap.add_argument("--root", default=EXPORT_ROOT)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    configs = {}
    for spec in args.config:
        name, tags = spec.split("=", 1)
        configs[name] = [t for t in tags.split(",") if t]

    loaded, meta = {}, {}
    for name, tags in configs.items():
        pers, reports = [], []
        for t in tags:
            per, rep = load_run(t, args.root)
            pers.append(per)
            reports.append(rep)
        loaded[name] = pers
        meta[name] = {
            "tags": tags,
            "train_sec": [r.get("train_sec") for r in reports],
            "switches": reports[0].get("scale_switches") if reports else None,
            "hnsw": [r.get("hnsw") for r in reports],
            "harness_parity": [r.get("harness_parity", {}).get("match") for r in reports],
        }

    out = {"metrics": {}, "meta": meta, "n_boot": args.n_boot}
    for metric in METRICS:
        out["metrics"][metric] = {n: cluster_bootstrap(p, metric, n_boot=args.n_boot)
                                  for n, p in loaded.items()}

    base = args.baseline
    out["anchor_gate"] = {}
    if base in loaded:
        for name in loaded:
            if name == base:
                continue
            g = out["metrics"]["full2k_gold@10"]
            out["anchor_gate"][name] = {
                "overlaps_baseline_ci": overlap(g[base], g[name]),
                "delta": g[name]["mean"] - g[base]["mean"],
            }

    print(f"\n{'config':<22} {'gold@10':>8} {'95% CI':>18} {'per-seed':>26} {'gate':>8}")
    print("-" * 88)
    for name in loaded:
        s = out["metrics"]["full2k_gold@10"][name]
        seeds = " ".join(f"{v:.4f}" for v in s["per_seed"])
        gate = "base" if name == base else (
            "PASS" if out["anchor_gate"].get(name, {}).get("overlaps_baseline_ci") else "FAIL")
        print(f"{name:<22} {s['mean']:>8.4f} [{s['ci_lo']:.4f},{s['ci_hi']:.4f}] "
              f"{seeds:>26} {gate:>8}")
    print()
    for metric in METRICS:
        if metric == "full2k_gold@10":
            continue
        row = "  ".join(f"{n}={out['metrics'][metric][n]['mean']:.4f}" for n in loaded)
        print(f"  {metric:<16} {row}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nsaved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
