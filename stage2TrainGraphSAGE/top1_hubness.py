"""
top1_hubness.py -- Top-1/global guide Phase 4 diagnostic: audit the models that
repeatedly outrank the gold model in the global (full-2K) ranking.

Retrains ONE configuration on ONE split (exactly as the baselines runner does),
then for every scorable test dataset logs the global top-10 by exact z_d @ z_m:
model id/name, family id, train-visible task profile, observed-on-d status,
cosine to the query, and how many queries each model's top-10 appearances span
(hub frequency). No training-code changes; pure diagnostics.

Run:  python -m ModelLakeFishing.stage2TrainGraphSAGE.top1_hubness --config P6_dm10 --split 0
"""

import argparse
import collections
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.model import load_hgraph  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.losses import TRAINED_ON, accuracy_lookup, perf_supervision  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.eval_harness import make_fixed_splits  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import apply_similar_to_mode, dedup_trained_on  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.ablation import train_eval_one  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.top1_audit import (  # noqa: E402
    GRAPH, INIT_SEED, EPOCHS, model_names, candidates, configs,
)
from ModelLakeFishing.stage2TrainGraphSAGE.top1_gphase import g_configs  # noqa: E402

OUT = os.path.join(_HERE, "artifacts", "ablation", "top1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="P6_dm10")
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    all_cfgs = {**configs(), **g_configs(["G1", "G2"])}
    cfg = all_cfgs[args.config]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data, xm0, umi = load_hgraph(GRAPH)
    names = model_names(umi)
    data = dedup_trained_on(data)
    data = apply_similar_to_mode(data, cfg["similar_to_mode"], k=cfg["similar_to_k"])
    split = make_fixed_splits(data, split_seed=args.split)
    _tr, _val, test_data = split
    lookup = accuracy_lookup(data)
    xd0 = torch.load(GRAPH, map_location="cpu", weights_only=False).get("xd0_meta")

    _r, _pt, _ph, model, _s = train_eval_one(
        data, xm0, xd0, cfg, split, init_seed=INIT_SEED, epochs=EPOCHS, device=device)
    model.eval()
    with torch.no_grad():
        z = model(test_data.clone().to(device))
    z_m = F.normalize(z["model"].cpu(), dim=-1)
    z_d = F.normalize(z["dataset"].cpu(), dim=-1)

    # train-visible task profile + observed sets (for compatibility fields)
    tr_eli, _tr_t = perf_supervision(_tr[TRAINED_ON], lookup)
    ti = torch.cat([_tr[TRAINED_ON].edge_index, tr_eli], dim=1)
    tt = data["dataset"].task_type_id
    model_tasks = collections.defaultdict(set)
    for m, d in zip(ti[0].tolist(), ti[1].tolist()):
        model_tasks[m].add(int(tt[d]))
    fam = data["model"].family_id

    cands = candidates(test_data, lookup)
    hub_count = collections.Counter()
    records = {}
    for d, (cand, a) in cands.items():
        gold = int(cand[int(np.argmax(a))])
        s_all = (z_m @ z_d[int(d)]).numpy()
        top = np.argsort(-s_all)[: args.topk]
        gold_rank = int((s_all > s_all[gold]).sum()) + 1
        obs = set(cand.tolist())
        rows = []
        for m in top.tolist():
            hub_count[m] += 1
            rows.append({
                "model_id": m, "model_name": names[m], "family_id": int(fam[m]),
                "observed_on_d": m in obs,
                "train_visible_task_profile": sorted(model_tasks.get(m, [])),
                "query_task": int(tt[int(d)]),
                "cos_to_query": float(s_all[m]),
                "is_gold": m == gold,
            })
        records[int(d)] = {"gold_model_id": gold, "gold_rank": gold_rank,
                           "gold_cos": float(s_all[gold]), "top": rows}

    # hub summary: models appearing in many queries' top-10
    hubs = [{"model_id": m, "model_name": names[m], "family_id": int(fam[m]),
             "n_queries_in_top10": c,
             "task_profile": sorted(model_tasks.get(m, [])),
             "n_train_visible_tasks": len(model_tasks.get(m, []))}
            for m, c in hub_count.most_common(25)]
    n_q = len(cands)
    fam_conc = collections.Counter(int(fam[m]) for m, c in hub_count.items() for _ in range(c))
    out = {
        "config": args.config, "split": args.split, "topk": args.topk,
        "n_queries": n_q,
        "hub_models_top25": hubs,
        "family_concentration_top10_slots": dict(fam_conc.most_common(10)),
        "per_dataset": records,
    }
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"hubness_{args.config}_s{args.split}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"queries: {n_q}; distinct models occupying top-10 slots: {len(hub_count)} "
          f"of {args.topk * n_q} slots")
    print("top hubs (model, #queries, task_profile):")
    for h in hubs[:10]:
        print(f"  {h['model_name'][:60]:60} {h['n_queries_in_top10']:>3}/{n_q} tasks={h['task_profile']}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
