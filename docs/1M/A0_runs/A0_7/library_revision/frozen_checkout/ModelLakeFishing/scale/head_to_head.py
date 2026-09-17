"""
head_to_head.py -- P4: the controlled head-to-head. Both systems scored on the
SAME 12K candidate universe, the SAME 517 held-out gold queries, and the SAME
global-metric harness (scale/global_metrics). A-axis = accuracy, B-axis =
latency.

HONESTY (carried from P2 §3.1 / D-5, non-negotiable disclosure):
  ModelLens's dataset_desc_matrix AND dataset2id mapping are both UNPUBLISHED.
  From published artifacts it can only run "dataset-blind" (task+metric+model
  features), which UNDER-represents the paper's full system. Its numbers here
  are the "reproducible-from-release" version, NOT proof our method wins on a
  level field. This is stated in every output.

Fairness controls:
  * identical candidate universe: our 12,000 sub-lake models, mapped to
    ModelLens's global ids for its _id_emb / model_desc_matrix (encoded with the
    REAL global ids, not arange);
  * identical query set + gold labels: the 517 test datasets from the P3 export
    (gold_cands.npz);
  * identical metric code: scale.global_metrics for both.

Run (repo root):
  ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.scale.head_to_head
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from scale.pull_corpus import data_root
from scale import global_metrics as GM
from scale import modellens_adapter as MA

EXPORT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "docs", "scale", "P3", "exports", "ml_sub_L1L3b")
LAKE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "stage1BuildTransferGraph", "artifacts", "modellens_v2_lake")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "scale", "P4", "artifacts")
NODE_SEP = "␟"


def load_export():
    # A-axis uses the HELD-OUT eval embeddings (test-split forward): the query
    # dataset does NOT see its own supervision edges. Using full-graph z_*.npy
    # here would leak a held-out query's own labels and inflate our gold@K.
    zm = np.load(os.path.join(EXPORT, "z_m_eval.npy"))
    zd = np.load(os.path.join(EXPORT, "z_d_eval.npy"))
    gc = np.load(os.path.join(EXPORT, "gold_cands.npz"))
    mid = pd.read_csv(os.path.join(EXPORT, "model_ids.csv"))
    did = pd.read_csv(os.path.join(EXPORT, "dataset_ids.csv"))
    cands = {int(k): (gc[k][0].astype(int), gc[k][1].astype(float)) for k in gc.files}
    roots = {int(r.mappedID): str(r.root) for r in did.itertuples()}
    return zm, zd, cands, roots, mid, did


def ours(zm, zd, cands, roots):
    t = time.perf_counter()
    agg, _ = GM.from_embeddings(zm, zd, cands, roots)
    agg["_wall_sec"] = round(time.perf_counter() - t, 2)
    return agg


def cache_subset(model, names, global_ids, size_ids, fam_ids, dev):
    """build_model_cache but with REAL global ids (our 12K is a SUBSET, so
    arange would mis-index _id_emb / model_desc_matrix)."""
    gid = torch.as_tensor(global_ids, dtype=torch.long, device=dev)
    with torch.no_grad():
        h_model = model.encode_model(gid, names)
        h_size = model.size_embedding(size_ids.to(dev)) if model.use_size_feature else None
        h_family = (model.family_embedding(fam_ids.to(dev))
                    if model.use_family_prior else None)
    return {"h_model": h_model, "h_size": h_size, "size_ids": size_ids.to(dev),
            "h_family": h_family, "family_ids": fam_ids.to(dev)}


def modellens(zm_len, cands, roots, mid, did):
    model, margs, dev, missing, unexpected = MA.load_modellens()
    model2id, task2id, metric2id, family2id, profile, size_bucket = MA.build_vocabs()
    pool = pd.read_csv(os.path.join(LAKE, "ml_dataset_pool.csv"))
    node2metric = dict(zip(pool["dataset_node"], pool["chosen_metric"]))

    # our 12K models in mappedID order -> their global ids / size / family
    names = mid.sort_values("mappedID")["model"].tolist()
    fam_allowed = {str(k).strip().lower(): int(v) for k, v in family2id.items()}
    UNK = {"unknown", "", "none", "null", "nan"}
    gids, size_ids, fam_ids = [], [], []
    for nm in names:
        gids.append(int(model2id.get(nm, model.unk_model_id)))
        p = profile.get(nm)
        s_id, f_id = 0, 0
        if isinstance(p, dict):
            f = str(p.get("family", "unknown")).strip().lower()
            f_id = fam_allowed.get(f, 0) if f not in UNK else 0
            s = str(p.get("size", "unknown")).strip().lower()
            if s not in UNK:
                try:
                    s_id = int(min(np.searchsorted(size_bucket, float(s), side="right"),
                                   len(size_bucket)))
                except ValueError:
                    pass
        size_ids.append(s_id); fam_ids.append(f_id)
    cache = cache_subset(model, names, np.array(gids),
                         torch.tensor(size_ids), torch.tensor(fam_ids), dev)

    node_of = dict(zip(did["mappedID"], did["dataset"]))
    unk_ds = model.unk_dataset_id
    desc_in = torch.tensor([[float(unk_ds)]], device=dev)   # blind (D-5 release version)

    scores = {}
    for d in cands:
        node = str(node_of[int(d)])
        task = node.split(NODE_SEP)[1] if NODE_SEP in node else ""
        task_id = int(task2id.get(task, 0))
        metric_id = int(metric2id.get(str(node2metric.get(node, "")), 0))
        with torch.no_grad():
            s = model.score_matrix(torch.tensor([task_id], device=dev), desc_in,
                                   cache, metric_ids=torch.tensor([metric_id], device=dev))
        scores[int(d)] = s.squeeze(0).float().cpu().numpy()
    agg, _ = GM.from_scores(scores, cands, roots)
    agg["_missing_keys"] = len(missing)
    return agg, model, cache, dev, (task2id, metric2id, node2metric, node_of)


def bench_latency(zm, zd, cands, model, cache, dev, ctx, ns=(1000, 2000, 5000, 12000)):
    """B-axis: ModelLens O(N) scan vs our HNSW, CUDA-synced, warmup + reps.
    latency-vs-N curve (the asymptotic moat; slope is what matters)."""
    import hnswlib
    task2id, metric2id, node2metric, node_of = ctx
    qds = list(cands)[:100]

    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()

    # build per-N ModelLens caches (slice the full cache tensors)
    full = cache
    zmn = zm / (np.linalg.norm(zm, axis=1, keepdims=True) + 1e-12)
    zdn = zd / (np.linalg.norm(zd, axis=1, keepdims=True) + 1e-12)
    curve = {}
    for N in ns:
        # ModelLens scan over first N candidates
        sub = {"h_model": full["h_model"][:N], "h_size": full["h_size"][:N] if full["h_size"] is not None else None,
               "size_ids": full["size_ids"][:N],
               "h_family": full["h_family"][:N] if full["h_family"] is not None else None,
               "family_ids": full["family_ids"][:N]}
        desc_in = torch.tensor([[float(model.unk_dataset_id)]], device=dev)
        # warmup
        for d in qds[:20]:
            node = str(node_of[int(d)]); task = node.split(NODE_SEP)[1] if NODE_SEP in node else ""
            with torch.no_grad():
                model.score_matrix(torch.tensor([int(task2id.get(task, 0))], device=dev),
                                   desc_in, sub,
                                   metric_ids=torch.tensor([int(metric2id.get(str(node2metric.get(node, "")), 0))], device=dev))
        sync()
        ml_lat = []
        for d in qds:
            node = str(node_of[int(d)]); task = node.split(NODE_SEP)[1] if NODE_SEP in node else ""
            ti = torch.tensor([int(task2id.get(task, 0))], device=dev)
            mi = torch.tensor([int(metric2id.get(str(node2metric.get(node, "")), 0))], device=dev)
            sync(); t = time.perf_counter_ns()
            with torch.no_grad():
                model.score_matrix(ti, desc_in, sub, metric_ids=mi)
            sync(); ml_lat.append((time.perf_counter_ns() - t) / 1e6)

        # our HNSW over first N models
        idx = hnswlib.Index(space="ip", dim=zmn.shape[1])
        idx.init_index(max_elements=N, ef_construction=200, M=32)
        idx.add_items(zmn[:N].astype(np.float32), np.arange(N))
        idx.set_ef(64)
        for d in qds[:20]:
            idx.knn_query(zdn[int(d)].astype(np.float32), k=10)
        hnsw_lat = []
        for d in qds:
            q = zdn[int(d)].astype(np.float32)
            t = time.perf_counter_ns()
            idx.knn_query(q, k=10)
            hnsw_lat.append((time.perf_counter_ns() - t) / 1e6)

        curve[N] = {
            "modellens_scan_ms_p50": float(np.percentile(ml_lat, 50)),
            "modellens_scan_ms_p99": float(np.percentile(ml_lat, 99)),
            "ours_hnsw_ms_p50": float(np.percentile(hnsw_lat, 50)),
            "ours_hnsw_ms_p99": float(np.percentile(hnsw_lat, 99)),
            "speedup_p50": float(np.percentile(ml_lat, 50) / max(np.percentile(hnsw_lat, 50), 1e-6)),
        }
    return curve


def main():
    os.makedirs(OUT, exist_ok=True)
    zm, zd, cands, roots, mid, did = load_export()
    print(f"[load] models={len(zm)} test-gold-queries={len(cands)}")

    print("[A] our system (from_embeddings) ...")
    a_ours = ours(zm, zd, cands, roots)
    print("[A] ModelLens (blind, release version, same 12K universe) ...")
    a_ml, model, cache, dev, ctx = modellens(len(zm), cands, roots, mid, did)

    print("[B] latency-vs-N (CUDA-synced) ...")
    curve = bench_latency(zm, zd, cands, model, cache, dev, ctx)

    result = {
        "universe": len(zm), "n_queries": len(cands),
        "DISCLOSURE": ("ModelLens run is the reproducible-from-release version: "
                       "its dataset_desc_matrix AND dataset2id are unpublished, so it "
                       "is dataset-blind (task+metric+model only). This UNDER-represents "
                       "the paper's full system; the gap is from unreleased artifacts, "
                       "NOT a level-field method comparison."),
        "A_axis": {"ours_L1L3b": a_ours, "modellens_blind": a_ml},
        "B_axis_latency_vs_N": curve,
    }
    with open(os.path.join(OUT, "head_to_head.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    print("\n=== A-AXIS (same 12K universe, same 517 queries, same harness) ===")
    print(f"{'metric':16s}{'ours L1L3b':>14s}{'ModelLens(blind)':>18s}")
    for k in ("gold@1", "gold@10", "top3@10", "gold-gap@10",
              "root_gold@10", "root_top3@10", "median_gold_rank"):
        o = a_ours.get(k); m = a_ml.get(k)
        print(f"{k:16s}{o:>14.4f}{m:>18.4f}")
    print("\n=== B-AXIS latency (ms/query, CUDA-synced) ===")
    print(f"{'N':>7s}{'ML scan p50':>13s}{'ours HNSW p50':>15s}{'speedup':>10s}")
    for N, c in curve.items():
        print(f"{N:>7d}{c['modellens_scan_ms_p50']:>13.4f}"
              f"{c['ours_hnsw_ms_p50']:>15.4f}{c['speedup_p50']:>10.1f}x")
    print(f"\n-> {OUT}\\head_to_head.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
