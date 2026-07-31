"""
export_ours.py -- P3: train the L1L3b champion on the ModelLens (sub)graph, then
export z_m / z_d + id tables + gold labels + an HNSW index, so P4 can score OUR
system through the SAME global-metric harness as ModelLens (scale/global_metrics)
and measure the B-axis (HNSW sublinear vs O(N) scan).

Reuses the exact champion pipeline (train_eval_one + L1L3b config from
l_phase/l_configs, unmodified) so this is the same model that won on D0.

Outputs to docs/scale/P3/exports/<tag>/:
  z_m.npy, z_d.npy          -- embeddings in mappedID row order
  model_ids.csv, dataset_ids.csv (dataset, mappedID, root, root_b)
  gold_cands.npz            -- per test-dataset (cand model ids, accs) for P4
  hnsw_index.bin            -- HNSW over z_m
  export_report.json        -- gold@K (our A-axis), HNSW fidelity + latency

Run (repo root):
  ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.scale.export_ours \
      --graph ModelLakeFishing/stage1BuildTransferGraph/hgraph_ml_v2_sub.pt --seed 0
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.losses import accuracy_lookup  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import apply_similar_to_mode  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.ablation import train_eval_one  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.top1_audit import (  # noqa: E402
    INIT_SEED, model_names, candidates)
from ModelLakeFishing.stage2TrainGraphSAGE.top1_eval import five_metric_eval, aggregate  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.l_phase import l_configs  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.d0_splits import make_root_aware_splits  # noqa: E402
from ModelLakeFishing.scale import global_metrics as GM  # noqa: E402

OUT_ROOT = os.path.join(_REPO_ROOT, "ModelLakeFishing", "docs", "scale", "P3", "exports")


def l1l3b_config(batch=1024):
    cfg = l_configs(["L1L3"])["L1L3"]
    cfg.pop("repair_dataset_task", None)     # native task ids
    cfg["global_n_neg"] = 256                # the W2/L1b upgrade
    cfg["batch_size"] = batch
    return cfg


def build_hnsw(z_m, z_d, cands, ef=200, M=32):
    """HNSW over normalized z_m; fidelity recall@50 vs brute MIPS; latency."""
    import hnswlib
    zm = z_m / (np.linalg.norm(z_m, axis=1, keepdims=True) + 1e-12)
    zd = z_d / (np.linalg.norm(z_d, axis=1, keepdims=True) + 1e-12)
    N, dim = zm.shape
    idx = hnswlib.Index(space="ip", dim=dim)
    idx.init_index(max_elements=N, ef_construction=ef, M=M)
    t0 = time.perf_counter_ns()
    idx.add_items(zm.astype(np.float32), np.arange(N))
    build_ms = (time.perf_counter_ns() - t0) / 1e6
    idx.set_ef(max(ef, 64))

    # fidelity: recall@50 vs brute-force MIPS, on the gold query datasets
    qd = list(cands)[: min(300, len(cands))]
    recalls, lat = [], []
    K = 50
    for d in qd:
        q = zd[int(d)].astype(np.float32)
        brute = np.argsort(-(zm @ q))[:K]
        t = time.perf_counter_ns()
        lab, _ = idx.knn_query(q, k=K)
        lat.append((time.perf_counter_ns() - t) / 1e6)
        recalls.append(len(set(lab[0].tolist()) & set(brute.tolist())) / K)
    return idx, dict(build_ms=round(build_ms, 1),
                     recall_at_50=float(np.mean(recalls)),
                     hnsw_query_ms_p50=float(np.percentile(lat, 50)),
                     hnsw_query_ms_p99=float(np.percentile(lat, 99)),
                     n_query=len(qd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--tag", default="ml_sub_L1L3b")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = os.path.join(OUT_ROOT, args.tag)
    os.makedirs(out, exist_ok=True)

    payload = torch.load(args.graph, map_location="cpu", weights_only=False)
    udi = payload["unique_dataset_id"].sort_values("mappedID").reset_index(drop=True)
    root_of = udi["root"].tolist()
    xm0, xd0 = payload["xm0_meta"], payload["xd0_meta"]
    names = model_names(payload["unique_model_id"])
    cfg = l1l3b_config()

    data = torch.load(args.graph, map_location="cpu", weights_only=False)["data"]
    data = apply_similar_to_mode(data, cfg["similar_to_mode"], k=cfg["similar_to_k"])
    split = make_root_aware_splits(data, root_of, split_seed=args.seed)
    _tr, _val, test_data = split
    lookup = accuracy_lookup(data)

    print(f"[train] L1L3b seed={args.seed} epochs={args.epochs} on "
          f"{os.path.basename(args.graph)}", flush=True)
    t0 = time.time()
    row, _pt, _ph, model, _sc = train_eval_one(
        data, xm0, xd0, cfg, split, init_seed=INIT_SEED,
        epochs=args.epochs, device=device)
    model.eval()

    # gold@K on held-out test (OUR A-axis; same five_metric_eval == global_metrics)
    with torch.no_grad():
        z_test = {k: v.cpu() for k, v in model(test_data.clone().to(device)).items()}
    cands = candidates(test_data, lookup)
    per = five_metric_eval(z_test, cands, names=names)
    agg = aggregate(per)
    # cross-run our own harness for parity confidence
    scores_by_q = {int(d): (F_norm(z_test["model"]) @ F_norm(z_test["dataset"])[int(d)]).numpy()
                   for d in cands}
    gm_cands = {int(d): (np.asarray(c), np.asarray(a)) for d, (c, a) in cands.items()}
    roots_q = {int(d): root_of[int(d)] for d in cands}
    gm_agg, _ = GM.from_scores(scores_by_q, gm_cands, roots_q)

    # full-graph embeddings for serving/HNSW export
    with torch.no_grad():
        z_full = {k: v.cpu() for k, v in model(data.clone().to(device)).items()}
    z_m = z_full["model"].numpy().astype(np.float32)
    z_d = z_full["dataset"].numpy().astype(np.float32)

    np.save(os.path.join(out, "z_m.npy"), z_m)
    np.save(os.path.join(out, "z_d.npy"), z_d)
    # HELD-OUT eval embeddings (test-split forward): the query dataset does NOT
    # see its own supervision edges. These are the leakage-free embeddings the
    # gold@K numbers are computed on -- P4 must use THESE, not the full-graph z
    # (full-graph z_d lets a held-out query see its own labels = inflated).
    np.save(os.path.join(out, "z_m_eval.npy"), z_test["model"].numpy().astype(np.float32))
    np.save(os.path.join(out, "z_d_eval.npy"), z_test["dataset"].numpy().astype(np.float32))
    payload["unique_model_id"].sort_values("mappedID").to_csv(
        os.path.join(out, "model_ids.csv"), index=False)
    udi.to_csv(os.path.join(out, "dataset_ids.csv"), index=False)
    # gold labels for P4
    np.savez_compressed(
        os.path.join(out, "gold_cands.npz"),
        **{str(d): np.stack([np.asarray(c, float), np.asarray(a, float)])
           for d, (c, a) in cands.items()})

    idx, hnsw = build_hnsw(z_m, z_d, cands)
    idx.save_index(os.path.join(out, "hnsw_index.bin"))

    report = dict(
        tag=args.tag, graph=os.path.basename(args.graph),
        seed=args.seed, epochs=args.epochs, device=device,
        n_models=len(z_m), n_datasets=len(z_d), n_test_datasets=len(cands),
        train_sec=round(time.time() - t0, 1),
        our_gold_at_1=agg["full2k_gold@1"], our_gold_at_10=agg["full2k_gold@10"],
        our_gold_gap_at_10=agg["gold_gap@10"], observed_hit1=agg["observed_hit1"],
        harness_parity={"five_metric_gold@10": agg["full2k_gold@10"],
                        "global_metrics_gold@10": gm_agg["gold@10"],
                        "match": abs(agg["full2k_gold@10"] - gm_agg["gold@10"]) < 1e-9},
        global_metrics=gm_agg, hnsw=hnsw,
    )
    with open(os.path.join(out, "export_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== P3 EXPORT (OUR system on ModelLens sub-lake) ===")
    print(f"  gold@1={agg['full2k_gold@1']:.4f} gold@10={agg['full2k_gold@10']:.4f} "
          f"gold-gap@10={agg['gold_gap@10']:.4f} hit1={agg['observed_hit1']:.3f}")
    print(f"  harness parity (five_metric vs global_metrics gold@10): "
          f"{report['harness_parity']['match']}")
    print(f"  HNSW: recall@50={hnsw['recall_at_50']:.3f} "
          f"p50={hnsw['hnsw_query_ms_p50']:.3f}ms p99={hnsw['hnsw_query_ms_p99']:.3f}ms "
          f"build={hnsw['build_ms']:.0f}ms")
    print(f"  -> {out}")
    return 0


def F_norm(t):
    import torch.nn.functional as F
    return F.normalize(t, dim=-1)


if __name__ == "__main__":
    sys.exit(main())
