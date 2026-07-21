"""
w1_dzero.py -- v4 W1-E2: the D0 three-column baselines, root-aware splits.

Rows (one training config per column, everything else identical):
    B0    : plain 1-layer baseline (no global term) -- the floor
    G2    : task-incompatible + known-low pools, hard 0.5 -- the v2-era best
    L1L3b : the v4 carried candidate -- whole-lake logQ sampled softmax,
            n_neg 256, native task metadata (no repair patch needed on D0:
            the task vocab was built correctly at graph-build time)

Protocol deltas vs the 2K harness, both mandated by plan v4:
    * make_root_aware_splits (same root never straddles a split boundary);
    * per-root MACRO aggregation reported alongside the flat per-dataset
      aggregate (tatoeba-scale sibling families must not dominate).
Metric keys keep the historical names (full2k_gold@K now means full-LAKE
gold@K over 9,491 models -- absolute values are NOT comparable to 2K numbers).

Run (repo root):
  ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.stage2TrainGraphSAGE.w1_dzero
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

from ModelLakeFishing.stage2TrainGraphSAGE.losses import accuracy_lookup  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.eval_harness import paired_bootstrap  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import apply_similar_to_mode  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.ablation import train_eval_one  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.top1_audit import (  # noqa: E402
    INIT_SEED, sha256_file, state_dict_sha256, model_names, candidates, configs,
)
from ModelLakeFishing.stage2TrainGraphSAGE.top1_eval import (  # noqa: E402
    five_metric_eval, aggregate, random_baselines,
)
from ModelLakeFishing.stage2TrainGraphSAGE.top1_gphase import g_configs, gate, PRIMARY  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.l_phase import l_configs  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.d0_splits import make_root_aware_splits  # noqa: E402

# v5 P4: gold-gap@10 is a REPORTED co-primary (root_macro + pooled bootstrap);
# the gate still fires on PRIMARY's strict gold@10 (goalpost unchanged).
PRIMARY_PLUS = PRIMARY + ["gold_gap@10"]

GRAPH = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage1BuildTransferGraph",
                     "hgraph_d0_v1.pt")
OUT = os.path.join(_HERE, "artifacts", "ablation", "d0")
TAG = "w1_baselines"
SPLIT_SEEDS = (0, 1, 2)
EPOCHS = 25
BATCH = 1024          # 54.8K supervision edges -> ~38 batches/epoch (2K cadence)


def d0_configs(rows):
    b0 = configs()["B0"]
    g2 = g_configs(["G2"])["G2"]
    l1l3b = l_configs(["L1L3"])["L1L3"]
    l1l3b.pop("repair_dataset_task", None)      # D0 task ids are native
    l1l3b["global_n_neg"] = 256                 # the W2/L1b upgrade
    out = {}
    if "B0" in rows:
        out["B0"] = dict(b0, batch_size=BATCH)
    if "G2" in rows:
        out["G2"] = dict(g2, batch_size=BATCH)
    if "L1L3b" in rows:
        out["L1L3b"] = dict(l1l3b, batch_size=BATCH)
    if "L1L3bP2" in rows:
        # v4 W3 hygiene row: D2-P2 mild degree cap on the message graph
        out["L1L3bP2"] = dict(l1l3b, batch_size=BATCH, degree_cap_q=0.95)
    if "L1L3bS1" in rows:
        # v6 S1: root-level positive pooling (training-side dual of S2)
        out["L1L3bS1"] = dict(l1l3b, batch_size=BATCH, root_pool_positives=True)
    if "G1" in rows:                            # historical: global negs only
        out["G1"] = dict(g_configs(["G1"])["G1"], batch_size=BATCH)
    if "P6_dm10" in rows:                       # historical: dm-contrast champion
        out["P6_dm10"] = dict(configs()["P6_dm10"], batch_size=BATCH)
    return out


def root_macro(per, root_of):
    """Per-dataset rows -> macro over roots (mean within root, then across)."""
    by_root = {}
    for d, r in per.items():
        by_root.setdefault(root_of[int(d)], []).append(r)
    keys = [k for k in PRIMARY_PLUS]
    out = {}
    for k in keys:
        vals = [float(np.mean([row[k] for row in rows])) for rows in by_root.values()]
        out[k] = float(np.mean(vals))
    out["n_roots"] = len(by_root)
    return out


def main():
    global SPLIT_SEEDS, TAG, GRAPH
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", default=["B0", "G2", "L1L3b"])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SPLIT_SEEDS))
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--graph", default=GRAPH,
                    help="v5: point at hgraph_d0_v1_content.pt for the P1×P3 "
                         "combined rebaseline (default = promoted v4 D0 graph)")
    args = ap.parse_args()
    GRAPH = args.graph
    SPLIT_SEEDS, TAG = tuple(args.seeds), args.tag
    os.makedirs(os.path.join(OUT, TAG), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    payload = torch.load(GRAPH, map_location="cpu", weights_only=False)
    udi = payload["unique_dataset_id"].sort_values("mappedID")
    root_of = udi["root"].tolist()
    graph_sha = sha256_file(GRAPH)
    cfgs = d0_configs(args.rows)
    # v6 S1: integer root code per dataset (mappedID order) for pooling
    _root_code = {r: i for i, r in enumerate(sorted(set(root_of)))}
    root_ids = [_root_code[r] for r in root_of]
    for c in cfgs.values():
        if c.get("root_pool_positives"):
            c["root_ids"] = root_ids

    results, pooled = {}, {name: {} for name in cfgs}
    for name, cfg in cfgs.items():
        res = {"config_name": name, "config": cfg, "graph": os.path.basename(GRAPH),
               "graph_sha256": graph_sha, "init_seed": INIT_SEED,
               "protocol": "root-aware splits; per-root macro reported", "splits": {}}
        for ss in SPLIT_SEEDS:
            data = torch.load(GRAPH, map_location="cpu", weights_only=False)["data"]
            xm0, xd0 = payload["xm0_meta"], payload["xd0_meta"]
            names = model_names(payload["unique_model_id"])
            data = apply_similar_to_mode(data, cfg["similar_to_mode"], k=cfg["similar_to_k"])
            split = make_root_aware_splits(data, root_of, split_seed=ss)
            if cfg.get("degree_cap_q"):
                # W3/P2: cap MESSAGE graphs of all three split views; the full
                # `data` (lookup source) and edge_label supervision stay intact
                from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import degree_cap_trained_on
                capped = []
                for part in split:
                    part, cap_stats = degree_cap_trained_on(part, quantile=cfg["degree_cap_q"])
                    capped.append(part)
                split = tuple(capped)
                print(f"    [P2 cap] {cap_stats}", flush=True)
            _tr, _val, test_data = split
            lookup = accuracy_lookup(data)

            row, _pt, _ph, model, _sc = train_eval_one(
                data, xm0, xd0, cfg, split, init_seed=INIT_SEED,
                epochs=args.epochs, device=device)
            model.eval()
            with torch.no_grad():
                z = {k: v.cpu() for k, v in model(test_data.clone().to(device)).items()}
            cands = candidates(test_data, lookup)
            per = five_metric_eval(z, cands, names=names)
            agg = aggregate(per)
            res["splits"][str(ss)] = {
                "split_seed": ss, "state_dict_sha256": state_dict_sha256(model),
                "aggregate": agg,
                "root_macro": root_macro(per, root_of),
                "train_diag": {k: row[k] for k in
                               ("tau_macro", "mean_cos", "z_m_pr", "z_d_pr") if k in row},
                "per_dataset": {str(d): r for d, r in per.items()},
            }
            for d, r in per.items():
                pooled[name][f"s{ss}::{d}"] = {k: r[k] for k in PRIMARY_PLUS}
            rm = res["splits"][str(ss)]["root_macro"]
            print(f"[{TAG}/{name} s{ss}] n_ds={agg['n_datasets']} roots={rm['n_roots']} "
                  f"hit1={agg['observed_hit1']:.3f} gold@10={agg['full2k_gold@10']:.3f} "
                  f"root_gold@10={rm['full2k_gold@10']:.3f}", flush=True)
        keys = res["splits"][str(SPLIT_SEEDS[0])]["aggregate"].keys()
        res["aggregate_over_splits"] = {
            k: [float(np.mean([res["splits"][str(s)]["aggregate"][k] for s in SPLIT_SEEDS])),
                float(np.std([res["splits"][str(s)]["aggregate"][k] for s in SPLIT_SEEDS]))]
            for k in keys}
        res["root_macro_over_splits"] = {
            k: [float(np.mean([res["splits"][str(s)]["root_macro"][k] for s in SPLIT_SEEDS])),
                float(np.std([res["splits"][str(s)]["root_macro"][k] for s in SPLIT_SEEDS]))]
            for k in PRIMARY_PLUS}
        with open(os.path.join(OUT, TAG, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        results[name] = res

    report = {"graph_sha256": graph_sha, "rows": {}, "gates": {}}
    for n in cfgs:
        report["rows"][n] = {"flat": results[n]["aggregate_over_splits"],
                             "root_macro": results[n]["root_macro_over_splits"]}
    if "G2" in cfgs and "L1L3b" in cfgs:
        verdict, details = gate(
            pooled["G2"], pooled["L1L3b"],
            results["G2"]["aggregate_over_splits"], results["L1L3b"]["aggregate_over_splits"],
            results["G2"]["splits"], results["L1L3b"]["splits"])
        report["gates"]["L1L3b_vs_G2"] = {"verdict": verdict, **details}
    with open(os.path.join(OUT, TAG, "D0_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md = ["# W1 D0 baselines -- root-aware splits, full-lake gold (9,491 models)", "",
          "| row | view | " + " | ".join(PRIMARY_PLUS) + " |",
          "|---|---|" + "---|" * len(PRIMARY_PLUS)]
    for n in cfgs:
        for view in ("flat", "root_macro"):
            ag = report["rows"][n][view]
            md.append(f"| {n} | {view} | " +
                      " | ".join(f"{ag[m][0]:.3f}±{ag[m][1]:.3f}" for m in PRIMARY_PLUS) + " |")
    if report["gates"]:
        md.append("")
        md.append(f"Gate L1L3b vs G2: **{report['gates']['L1L3b_vs_G2']['verdict']}**")
    with open(os.path.join(OUT, TAG, "D0_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(json.dumps(report.get("gates", {}).get("L1L3b_vs_G2", {}).get("verdict", "n/a")))


if __name__ == "__main__":
    main()
