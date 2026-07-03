"""
top1_baselines.py -- T0 (and later G/L rows) of the Top-1/global guide.

Runs configurations on the PROVENANCE-FIXED graph (dedup_trained_on: 12,205 rows
-> 7,056 distinct pairs; kills the duplicate-copy target leakage that put 44% of
test pairs into the train message graph) and evaluates every one with the single
five-metric evaluator (top1_eval.five_metric_eval). Adds analytic random
baselines. Writes:

  artifacts/ablation/top1/<tag>/<name>.json   (per-dataset records + hashes)
  artifacts/ablation/top1/TOP1_BASELINES.json (aggregates, T0 tag)
  artifacts/ablation/top1/TOP1_BASELINES.md

Training, losses, split logic unchanged -- training simply runs on the deduped
graph via the existing ablation.train_eval_one path.

Run:  python -m ModelLakeFishing.stage2TrainGraphSAGE.top1_baselines [--tag T0]
"""

import argparse
import hashlib
import io
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.model import load_hgraph  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.losses import accuracy_lookup  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.eval_harness import make_fixed_splits, paired_bootstrap  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import apply_similar_to_mode, dedup_trained_on  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.ablation import train_eval_one  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.top1_audit import (  # noqa: E402
    GRAPH, SPLIT_SEEDS, INIT_SEED, EPOCHS, sha256_file, state_dict_sha256,
    model_names, candidates, configs,
)
from ModelLakeFishing.stage2TrainGraphSAGE.top1_eval import (  # noqa: E402
    five_metric_eval, aggregate, strata_aggregate, random_baselines, GOLD_KS,
)

OUT = os.path.join(_HERE, "artifacts", "ablation", "top1")

PRIMARY = ["observed_hit1", "top3_hit1", "regret1", "full2k_gold@1", "full2k_gold@10"]


def run_configs(cfgs, *, tag, epochs=EPOCHS, extra_note=""):
    os.makedirs(os.path.join(OUT, tag), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graph_sha = sha256_file(GRAPH)
    xd0_full = torch.load(GRAPH, map_location="cpu", weights_only=False).get("xd0_meta")

    results = {}
    pooled = {name: {} for name in cfgs}
    cand_sig = {ss: {} for ss in SPLIT_SEEDS}
    rand_agg = None

    for name, cfg in cfgs.items():
        res = {"config_name": name, "config": cfg, "init_seed": INIT_SEED,
               "graph": os.path.basename(GRAPH), "graph_sha256": graph_sha,
               "dedup_trained_on": True, "distinct_pairs": 7056,
               "note": ("trained+evaluated on the provenance-fixed (deduped) graph; "
                        "prior 12205-row graph leaked 44% of test pairs into the train "
                        "message graph via duplicate copies. " + extra_note),
               "splits": {}}
        for ss in SPLIT_SEEDS:
            data, xm0, umi = load_hgraph(GRAPH)
            names = model_names(umi)
            data = dedup_trained_on(data)                              # provenance fix
            data = apply_similar_to_mode(data, cfg["similar_to_mode"], k=cfg["similar_to_k"])
            split = make_fixed_splits(data, split_seed=ss)
            _tr, _val, test_data = split
            lookup = accuracy_lookup(data)

            _row, _pt, _ph, model, _sc = train_eval_one(
                data, xm0, xd0_full, cfg, split, init_seed=INIT_SEED,
                epochs=epochs, device=device)
            model.eval()
            with torch.no_grad():
                z = model(test_data.clone().to(device))
            z = {k: v.cpu() for k, v in z.items()}

            cands = candidates(test_data, lookup)
            cand_sig[ss][name] = {d: sorted(c.tolist()) for d, (c, _a) in cands.items()}
            per = five_metric_eval(z, cands, names=names)
            agg = aggregate(per)
            res["splits"][str(ss)] = {
                "split_seed": ss, "state_dict_sha256": state_dict_sha256(model),
                "aggregate": agg, "strata": strata_aggregate(per),
                "per_dataset": {str(d): r for d, r in per.items()},
            }
            for d, r in per.items():
                pooled[name][f"s{ss}::{d}"] = {k: r[k] for k in PRIMARY}
            if rand_agg is None:
                rand_agg = random_baselines(cands, n_pool=data["model"].num_nodes)
            print(f"[{tag}/{name} s{ss}] n={agg['n_datasets']} hit1={agg['observed_hit1']:.3f} "
                  f"top3={agg['top3_hit1']:.3f} regret={agg['regret1']:.4f} "
                  f"gold@1={agg['full2k_gold@1']:.3f} gold@10={agg['full2k_gold@10']:.3f} "
                  f"med_rank={agg['median_gold_rank']:.0f}")
        # over-splits mean/std of every aggregate key
        keys = res["splits"][str(SPLIT_SEEDS[0])]["aggregate"].keys()
        res["aggregate_over_splits"] = {
            k: [float(np.mean([res["splits"][str(s)]["aggregate"][k] for s in SPLIT_SEEDS])),
                float(np.std([res["splits"][str(s)]["aggregate"][k] for s in SPLIT_SEEDS]))]
            for k in keys}
        with open(os.path.join(OUT, tag, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        results[name] = res

    # candidate-set identity across configs, per split
    ref = list(cfgs)[0]
    identity = {str(ss): bool(all(cand_sig[ss][n] == cand_sig[ss][ref] for n in cfgs))
                for ss in SPLIT_SEEDS}
    summary = {"tag": tag, "graph_sha256": graph_sha, "epochs": epochs,
               "split_seeds": list(SPLIT_SEEDS), "init_seed": INIT_SEED,
               "candidate_sets_identical": identity,
               "random_baselines": rand_agg,
               "aggregates": {n: results[n]["aggregate_over_splits"] for n in results}}
    return results, pooled, summary


def write_baselines_md(summary, pooled, path_md, path_json):
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    ag = summary["aggregates"]
    rnd = summary["random_baselines"]
    L = []
    L.append("# TOP1_BASELINES — five-metric T0 on the provenance-fixed graph\n")
    L.append(f"Graph sha256 `{summary['graph_sha256'][:16]}…`, **dedup_trained_on applied** "
             "(12,205 rows → 7,056 distinct pairs). Provenance resolved: Phase-3 candidate table "
             "9,530 pairs → Phase-4/5 materialize_report 9,305 records.csv rows → graph concat "
             "with model_config rows → 12,205 rows / 7,056 distinct in-graph pairs (5,143 duplicated "
             "pairs had conflicting accuracy; duplicates leaked 44% of test pairs into train "
             "message graphs pre-fix — all zero post-fix, verified both directions, all splits).\n")
    L.append(f"Split seeds {summary['split_seeds']}, init seed {summary['init_seed']}, "
             f"{summary['epochs']} epochs. Candidate sets identical across configs: "
             f"{summary['candidate_sets_identical']}.\n")
    L.append("| name | observed_hit@1 | top3_hit@1 | regret@1 | full2k_gold@1 | full2k_gold@10 | median_gold_rank |")
    L.append("|---|---|---|---|---|---|---|")
    for n, a in ag.items():
        L.append("| {} | {:.3f}±{:.3f} | {:.3f} | {:.4f} | {:.3f} | {:.3f} | {:.0f} |".format(
            n, a["observed_hit1"][0], a["observed_hit1"][1], a["top3_hit1"][0], a["regret1"][0],
            a["full2k_gold@1"][0], a["full2k_gold@10"][0], a["median_gold_rank"][0]))
    ro, rp = rnd["random_observed"], rnd["random_fullpool"]
    L.append("| random_observed | {:.3f} | {:.3f} | {:.4f} | — | — | — |".format(
        ro["observed_hit1"], ro["top3_hit1"], ro["regret1"]))
    L.append("| random_fullpool | — | — | — | {:.4f} | {:.3f} | {:.0f} |".format(
        rp["full2k_gold@1"], rp["full2k_gold@10"], rp["median_gold_rank"]))
    L.append("\nfull2k_gold@K is gold-survival, not full-pool precision (unlabeled ≠ wrong). "
             "Exact-dot only; HNSW fidelity is a separate systems check (see tests).")
    with open(path_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="T0")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    results, pooled, summary = run_configs(configs(), tag=args.tag, epochs=args.epochs)
    write_baselines_md(summary, pooled,
                       os.path.join(OUT, "TOP1_BASELINES.md"),
                       os.path.join(OUT, "TOP1_BASELINES.json"))
    with open(os.path.join(OUT, args.tag, "pooled_per_dataset.json"), "w", encoding="utf-8") as f:
        json.dump(pooled, f)
    print(f"\nwrote {os.path.join(OUT, 'TOP1_BASELINES.md')}")


if __name__ == "__main__":
    main()
