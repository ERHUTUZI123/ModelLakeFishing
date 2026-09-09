"""
serving_rerank.py -- v6 serving layer: the unified zero-train re-ranker adopted
in v6 (S2 sibling-prior + P2b task-prior), plugged in AFTER HNSW retrieval.

    fused(D, m) = minmax(z_d[D]·z_m)[m]  +  alpha · sibling_boost(m)
                                          +  beta  · task_boost(m)

  sibling_boost(m) = mean norm_acc of m over D's lake SIBLINGS (same root, != D)
  task_boost(m)    = shrunk mean norm_acc of m over D's SAME-TASK lake datasets
                     (!= D), shrink k=5

Both priors are read from the prior sidecar (build_prior_sidecar.py) and use only
lake-visible labels EXCLUDING D itself -- the exact legality boundary from the
offline studies (v6 §0.6-3), asserted here. Where a query has no sibling / no
same-task lake data, the corresponding boost is all-zero -> that channel silently
does nothing (no branching, no side effect). alpha=beta=1.0 are the adopted
weights (v6/S2_*, v6/P2b_*).

This is a serving re-rank on the FROZEN export: z_m / z_d / the index are never
modified. Retrieve a larger MIPS pool, re-rank by the fused score, return top-k.

Run (repo root):
  ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.stage3HNSW.serving_rerank \
      --export ModelLakeFishing/stage3HNSW/artifacts/exports/d0_L1L3b \
      --dataset xnli/en
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

SHRINK_K = 5.0
POOL = 512          # MIPS candidate pool re-ranked (full-lake gold semantics use all)


def load_serving(export_dir):
    z_m = np.load(os.path.join(export_dir, "z_m.npy"))
    z_d = np.load(os.path.join(export_dir, "z_d.npy"))
    z_m = z_m / (np.linalg.norm(z_m, axis=1, keepdims=True) + 1e-9)
    z_d = z_d / (np.linalg.norm(z_d, axis=1, keepdims=True) + 1e-9)
    did = pd.read_csv(os.path.join(export_dir, "dataset_ids.csv"))
    mid = pd.read_csv(os.path.join(export_dir, "model_ids.csv"))
    sc = np.load(os.path.join(export_dir, "prior_sidecar.npz"))
    # adjacency: dataset -> {model: acc}; root -> [datasets]; task -> [datasets]
    ds_models = defaultdict(dict)
    for m, d, a in zip(sc["edge_model"], sc["edge_dataset"], sc["edge_acc"]):
        ds_models[int(d)][int(m)] = float(a)
    root_id, task_id = sc["root_id"], sc["task_id"]
    by_root, by_task = defaultdict(list), defaultdict(list)
    for d in range(len(did)):
        by_root[int(root_id[d])].append(d)
        by_task[int(task_id[d])].append(d)
    return {"z_m": z_m, "z_d": z_d, "did": did, "mid": mid,
            "ds_models": ds_models, "root_id": root_id, "task_id": task_id,
            "by_root": by_root, "by_task": by_task, "n_models": z_m.shape[0]}


def _prior(h, d, group_datasets, shrink):
    """mean (shrunk) acc per model over `group_datasets` EXCLUDING d."""
    boost = np.zeros(h["n_models"], dtype="float32")
    acc_by_model = defaultdict(list)
    for dd in group_datasets:
        if dd == d:
            continue                       # LEGALITY: never use D's own labels
        for m, a in h["ds_models"][dd].items():
            acc_by_model[m].append(a)
    for m, accs in acc_by_model.items():
        n = len(accs)
        boost[m] = (sum(accs) + (0.5 * shrink if shrink else 0.0)) / (n + shrink) \
            if shrink else float(np.mean(accs))
    return boost


def fused_rerank(h, dataset, *, alpha=1.0, beta=1.0, k=50):
    did = h["did"]
    row = did.index[did["unique_dataset_id"] == dataset]
    assert len(row) == 1, f"dataset {dataset!r} not in export"
    d = int(row[0])
    mips = h["z_m"] @ h["z_d"][d]
    mips_n = (mips - mips.min()) / (mips.max() - mips.min() + 1e-9)
    sib = _prior(h, d, h["by_root"][int(h["root_id"][d])], shrink=0.0)
    task = _prior(h, d, h["by_task"][int(h["task_id"][d])], shrink=SHRINK_K)
    fused = mips_n + alpha * sib + beta * task
    order = np.argsort(-fused)[:k]
    mid = h["mid"]["unique_model_id"]
    out = [{"rank": i + 1, "mappedID": int(m),
            "unique_model_id": str(mid.iloc[int(m)]),
            "fused": round(float(fused[m]), 5), "mips": round(float(mips_n[m]), 5),
            "sibling_boost": round(float(sib[m]), 5),
            "task_boost": round(float(task[m]), 5)} for i, m in enumerate(order)]
    return {"dataset": dataset, "mappedID": d, "alpha": alpha, "beta": beta,
            "has_sibling": bool((sib > 0).any()), "has_task": bool((task > 0).any()),
            "results": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()
    h = load_serving(args.export)
    out = fused_rerank(h, args.dataset, alpha=args.alpha, beta=args.beta, k=args.k)
    print(f"query {out['dataset']} (mappedID {out['mappedID']}) | "
          f"has_sibling={out['has_sibling']} has_task={out['has_task']}")
    for r in out["results"][:args.k]:
        print(f"  #{r['rank']:2d} {r['unique_model_id'][:52]:52s} "
              f"fused={r['fused']:.4f} (mips {r['mips']:.3f} + sib {r['sibling_boost']:.3f} "
              f"+ task {r['task_boost']:.3f})")


if __name__ == "__main__":
    main()
