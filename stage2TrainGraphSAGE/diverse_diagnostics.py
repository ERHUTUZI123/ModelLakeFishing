"""
diverse_diagnostics.py -- Phase 5: one diagnostic suite, run identically on the
OLD zoo and the DIVERSE zoo so the report compares apples to apples.

Given a built graph (.pt) and a trained checkpoint (.pt), prints a JSON blob with:
  raw graph    : #models, #datasets, edge counts, dataset-similarity effective
                 rank (all datasets AND perf-bearing subset), family rows
                 exercised, size buckets exercised, trained_on edges/model,
                 per-dataset performance variance
  trained z    : z_m / z_d participation ratio, mean pairwise cosine, same-hub vs
                 random cosine distance, held-out tau_macro / tau_pool,
                 positive-pair density, HNSW recall (if hnswlib present)

Run:
  python -m ModelLakeFishing.stage2TrainGraphSAGE.diverse_diagnostics \
      --pt <graph.pt> --ckpt <checkpoint.pt> [--seed 0]
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.model import load_hgraph  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.learnable import load_checkpoint  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.losses import (  # noqa: E402
    TRAINED_ON, PerfScorer, accuracy_lookup, perf_supervision, split_trained_on,
    topk_membership, lineage_components, global_positive_density, per_dataset_density,
)
from ModelLakeFishing.stage2TrainGraphSAGE.train import (  # noqa: E402
    collapse_report, eval_perf, oversmoothing_report,
)
from ModelLakeFishing.stage2TrainGraphSAGE.eval_harness import (  # noqa: E402
    model_to_model_hnsw_recall, dataset_to_model_hnsw_recall, head_retrieval,
)

IS_BASE_OF = ("model", "is_base_of", "model")


def eff_rank(M):
    s = np.linalg.svd(np.asarray(M, dtype=float), compute_uv=False)
    s = s[s > 1e-12]
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def cosine_gram_eff_rank(X):
    X = np.asarray(X, dtype=float)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return eff_rank(Xn @ Xn.T)


def participation_ratio(Z):
    Z = np.asarray(Z, dtype=float)
    Z = Z - Z.mean(0, keepdims=True)
    ev = np.linalg.eigvalsh(Z.T @ Z)
    ev = ev[ev > 1e-12]
    return float((ev.sum() ** 2) / (ev ** 2).sum()) if ev.size else float("nan")


def raw_graph_metrics(data, xm0):
    out = {}
    N = data["model"].num_nodes
    D = data["dataset"].num_nodes
    out["n_models"] = int(N)
    out["n_datasets"] = int(D)
    out["edges"] = {str(et): int(data[et].edge_index.shape[1]) for et in data.edge_types}

    # dataset-similarity effective rank (cosine-gram of the dataset probe features)
    Xd = data["dataset"].x.numpy()
    out["dataset_sim_eff_rank_all"] = round(cosine_gram_eff_rank(Xd), 3)

    # perf-bearing datasets = those with >=1 trained_on edge
    to = data[TRAINED_ON].edge_index
    ds_with_edges = sorted(set(to[1].tolist()))
    out["n_perf_bearing_datasets"] = len(ds_with_edges)
    if len(ds_with_edges) >= 2:
        out["dataset_sim_eff_rank_perfbearing"] = round(
            cosine_gram_eff_rank(Xd[ds_with_edges]), 3)
    else:
        out["dataset_sim_eff_rank_perfbearing"] = float("nan")

    # family / size coverage
    fid = data["model"].family_id.numpy()
    sid = data["model"].size_bucket_id.numpy()
    out["num_families_vocab"] = int(xm0["num_families"])
    out["family_rows_exercised"] = int(len(np.unique(fid)))
    out["num_size_buckets_vocab"] = int(xm0["num_size_buckets"])
    out["size_buckets_exercised"] = int(len(np.unique(sid)))
    out["size_bucket_unknown_frac"] = round(float((sid == 0).mean()), 3)

    # trained_on edges per model
    deg = np.bincount(to[0].numpy(), minlength=N)
    out["trained_on_edges_per_model_mean"] = round(float(deg.mean()), 2)
    out["trained_on_edges_per_model_max"] = int(deg.max())
    out["models_with_zero_trained_on"] = int((deg == 0).sum())

    # per-dataset performance variance (std of normalized accuracy edge weights)
    ea = data[TRAINED_ON].edge_attr.float().numpy()
    dst = to[1].numpy()
    per_ds_std = {}
    for d in ds_with_edges:
        w = ea[dst == d]
        if w.size >= 2:
            per_ds_std[int(d)] = float(np.std(w))
    out["per_dataset_perf_std_mean"] = round(float(np.mean(list(per_ds_std.values()))), 4) if per_ds_std else float("nan")
    out["n_datasets_with_perf_variance"] = int(sum(1 for v in per_ds_std.values() if v > 1e-6))
    return out


def trained_metrics(data, ckpt_path, seed):
    model, _vocab, _repro = load_checkpoint(ckpt_path)
    model.eval()
    with torch.no_grad():
        z = model(data)
    z_m, z_d = z["model"], z["dataset"]
    out = {}
    out["num_layers"] = int(model.num_layers)
    out["z_m_participation_ratio"] = round(participation_ratio(z_m.numpy()), 3)
    out["z_d_participation_ratio"] = round(participation_ratio(z_d.numpy()), 3)
    out["mean_pairwise_cosine"] = round(collapse_report(z_m), 4)

    comp = lineage_components(data, data["model"].num_nodes)
    osm = oversmoothing_report(z_m, comp)
    out["same_hub_cos_dist"] = round(osm["same_hub_mean_dist"], 4) if osm["n_same_hub_pairs"] else None
    out["random_cos_dist"] = round(osm["random_mean_dist"], 4)
    out["n_same_hub_pairs"] = int(osm["n_same_hub_pairs"])

    # held-out tau on a seed-0 split (same protocol as training/eval)
    torch.manual_seed(seed); np.random.seed(seed)
    _tr, _v, test_data = split_trained_on(data, seed=seed)
    lookup = accuracy_lookup(data)
    scorer = PerfScorer(dim=128, mode="dot")  # untrained scorer -> dot geometry of z
    tm = eval_perf(model, scorer, test_data, lookup)
    out["tau_macro"] = round(tm["kendall_tau_macro"], 4) if tm["kendall_tau_macro"] == tm["kendall_tau_macro"] else None
    out["tau_pool"] = round(tm["kendall_tau"], 4)
    out["n_datasets_scored"] = int(tm["n_datasets_scored"])

    # positive-pair density (per-dataset top-frac membership, train-visible subset)
    eli, target = perf_supervision(_tr[TRAINED_ON], lookup)
    ti = torch.cat([_tr[TRAINED_ON].edge_index, eli], dim=1)
    ta = torch.cat([_tr[TRAINED_ON].edge_attr.float(), target], dim=0)
    M = topk_membership(data, top_frac=0.1, trained_on_index=ti, trained_on_attr=ta)
    out["positive_pair_density"] = round(global_positive_density(M), 4)
    out["per_dataset_positive_density"] = round(per_dataset_density(ti, M), 4)

    counts = torch.bincount(comp, minlength=data["model"].num_nodes)
    near_hub = counts[comp] > 1
    # z_m -> z_m: structural diagnostic only (NOT the serving relation)
    rec_mm = model_to_model_hnsw_recall(z_m, near_hub, k=50)
    out["model_to_model_hnsw_recall@50"] = rec_mm if rec_mm is not None else "skipped (hnswlib not installed)"
    # z_d -> z_m: the ACTUAL serving path -- ANN fidelity vs exact-dot top-K
    rec_dm = dataset_to_model_hnsw_recall(z, k=50)
    out["dataset_to_model_hnsw_recall@50"] = rec_dm if rec_dm is not None else "skipped (hnswlib not installed)"

    # exact-dot z_d -> z_m head retrieval over held-out observed candidates
    head_macro, _head_per = head_retrieval(z, test_data, lookup)
    out["head_retrieval"] = {k: round(v, 4) for k, v in head_macro.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="optional path to write the JSON")
    args = ap.parse_args()

    data, xm0, _umi = load_hgraph(args.pt)
    report = {
        "graph": os.path.basename(args.pt),
        "checkpoint": os.path.basename(args.ckpt),
        "raw": raw_graph_metrics(data, xm0),
        "trained": trained_metrics(data, args.ckpt, args.seed),
    }
    blob = json.dumps(report, indent=2)
    print(blob)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(blob)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
