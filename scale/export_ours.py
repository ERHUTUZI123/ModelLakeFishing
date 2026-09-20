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

from ModelLakeFishing.stage2TrainGraphSAGE.losses import accuracy_lookup
from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import apply_similar_to_mode
from ModelLakeFishing.stage2TrainGraphSAGE.ablation import train_eval_one
from ModelLakeFishing.stage2TrainGraphSAGE.top1_audit import (
    INIT_SEED, model_names, candidates)
from ModelLakeFishing.stage2TrainGraphSAGE.top1_eval import five_metric_eval, aggregate
from ModelLakeFishing.stage2TrainGraphSAGE.l_phase import l_configs
from ModelLakeFishing.stage2TrainGraphSAGE.d0_splits import make_root_aware_splits
from ModelLakeFishing.scale import global_metrics as GM

OUT_ROOT = os.path.join(_REPO_ROOT, "ModelLakeFishing", "docs", "scale", "P3", "exports")


def l1l3b_config(batch=1024):
    cfg = l_configs(["L1L3"])["L1L3"]
    cfg.pop("repair_dataset_task", None)
    cfg["global_n_neg"] = 256
    cfg["batch_size"] = batch
    return cfg


def _recall_at(idx, zm, zd, qd, K, ef, brute_cache):
    idx.set_ef(int(ef))
    recalls = []
    for d in qd:
        q = zd[int(d)].astype(np.float32)
        lab, _ = idx.knn_query(q, k=K)
        recalls.append(len(set(lab[0].tolist()) & brute_cache[d]) / K)
    return float(np.mean(recalls))


def tune_ef_for_recall(idx, zm, zd, qd, *, target=0.99, K=50, ef_max=2048,
                       brute_cache=None):
    if brute_cache is None:
        brute_cache = {d: set(np.argsort(-(zm @ zd[int(d)].astype(np.float32)))[:K].tolist())
                       for d in qd}
    trace = []
    lo, hi, hi_rec = None, None, None
    ef = K
    while ef <= ef_max:
        rec = _recall_at(idx, zm, zd, qd, K, ef, brute_cache)
        trace.append({"ef": int(ef), "recall": rec})
        if rec >= target:
            hi, hi_rec = ef, rec
            break
        lo, ef = ef, ef * 2
    if hi is None:
        return int(ef_max), trace[-1]["recall"] if trace else 0.0, trace
    while lo is not None and hi - lo > 1:
        mid = (lo + hi) // 2
        rec = _recall_at(idx, zm, zd, qd, K, mid, brute_cache)
        trace.append({"ef": int(mid), "recall": rec})
        if rec >= target:
            hi, hi_rec = mid, rec
        else:
            lo = mid
    return int(hi), float(hi_rec), trace


def build_hnsw(z_m, z_d, cands, ef=200, M=32, *, threads=None, iso_recall=None,
               n_query=300, K=50, latency_reps=1000, warmup=100):
    import hnswlib
    zm = z_m / (np.linalg.norm(z_m, axis=1, keepdims=True) + 1e-12)
    zd = z_d / (np.linalg.norm(z_d, axis=1, keepdims=True) + 1e-12)
    N, dim = zm.shape
    idx = hnswlib.Index(space="ip", dim=dim)
    idx.init_index(max_elements=N, ef_construction=ef, M=M)
    if threads:
        idx.set_num_threads(int(threads))
    t0 = time.perf_counter_ns()
    idx.add_items(zm.astype(np.float32), np.arange(N))
    build_ms = (time.perf_counter_ns() - t0) / 1e6
    idx.set_ef(max(ef, 64))

    qd = list(cands)[: min(n_query, len(cands))]
    brute_cache = {d: set(np.argsort(-(zm @ zd[int(d)].astype(np.float32)))[:K].tolist())
                   for d in qd}
    report = dict(build_ms=round(build_ms, 1), n_query=len(qd), M=int(M),
                  ef_construction=int(ef), build_threads=int(threads or 0),
                  index_mb=round(N * (dim * 4 + M * 2 * 4) / 2 ** 20, 1))

    if iso_recall is None:
        ef_used = max(ef, 64)
        recalls, lat = [], []
        for d in qd:
            q = zd[int(d)].astype(np.float32)
            t = time.perf_counter_ns()
            lab, _ = idx.knn_query(q, k=K)
            lat.append((time.perf_counter_ns() - t) / 1e6)
            recalls.append(len(set(lab[0].tolist()) & brute_cache[d]) / K)
        report.update(recall_at_50=float(np.mean(recalls)),
                      hnsw_query_ms_p50=float(np.percentile(lat, 50)),
                      hnsw_query_ms_p99=float(np.percentile(lat, 99)),
                      ef_search=int(ef_used), iso_recall=None)
        return idx, report

    ef_used, rec, trace = tune_ef_for_recall(idx, zm, zd, qd, target=iso_recall,
                                             K=K, brute_cache=brute_cache)
    idx.set_ef(ef_used)
    idx.set_num_threads(1)
    qvecs = [zd[int(d)].astype(np.float32) for d in qd]
    for i in range(warmup):
        idx.knn_query(qvecs[i % len(qvecs)], k=K)
    lat = []
    for i in range(latency_reps):
        q = qvecs[i % len(qvecs)]
        t = time.perf_counter_ns()
        idx.knn_query(q, k=K)
        lat.append((time.perf_counter_ns() - t) / 1e6)
    report.update(recall_at_50=float(rec), ef_search=int(ef_used),
                  iso_recall=float(iso_recall), ef_trace=trace,
                  hnsw_query_ms_p50=float(np.percentile(lat, 50)),
                  hnsw_query_ms_p95=float(np.percentile(lat, 95)),
                  hnsw_query_ms_p99=float(np.percentile(lat, 99)),
                  latency_reps=int(latency_reps), latency_threads=1)
    return idx, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--tag", default="ml_sub_L1L3b")
    ap.add_argument("--fanout", action="store_true",
                    help="T0-1 cap the fallback loader's neighbourhood expansion")
    ap.add_argument("--sparse-M", action="store_true",
                    help="T0-2 CSC/CSR membership instead of the dense [N_m,N_d]")
    ap.add_argument("--contrast-n-neg", type=int, default=None,
                    help="T0-3 sampled contrastive negatives (D-14; None = full N^2)")
    ap.add_argument("--contrast-max-pos-per-dataset", type=int, default=None)
    ap.add_argument("--chunked-infer", type=int, default=None, metavar="CHUNK",
                    help="T0-4 chunked exact inference, CHUNK seed nodes at a time")
    ap.add_argument("--stream-scores", action="store_true",
                    help="T0-5 stream the global-metric scoring (no {q: [N]} dict)")
    ap.add_argument("--skip-diagnostics", action="store_true", help="T0-6")
    ap.add_argument("--hnsw-threads", type=int, default=None, help="T0-7")
    ap.add_argument("--ef-construction", type=int, default=200, help="T0-7")
    ap.add_argument("--hnsw-M", type=int, default=32, help="T0-7")
    ap.add_argument("--iso-recall", type=float, default=None,
                    help="T0-7 tune ef_search to this recall@50 before timing")
    ap.add_argument("--expect-n", type=int, default=None,
                    help="IRON RULE 3: assert the candidate pool size")
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
    cfg.update(fanout=args.fanout, sparse_M=args.sparse_M,
               contrast_n_neg=args.contrast_n_neg,
               contrast_max_pos_per_dataset=args.contrast_max_pos_per_dataset,
               infer_chunk=args.chunked_infer,
               skip_diagnostics=args.skip_diagnostics)

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

    with torch.no_grad():
        if args.chunked_infer:
            from ModelLakeFishing.stage2TrainGraphSAGE.inference import chunked_forward
            z_test = chunked_forward(model, test_data, chunk_size=args.chunked_infer,
                                     device=device)
        else:
            z_test = {k: v.cpu() for k, v in model(test_data.clone().to(device)).items()}
    cands = candidates(test_data, lookup)
    per = five_metric_eval(z_test, cands, names=names, expect_n=args.expect_n)
    agg = aggregate(per)
    gm_cands = {int(d): (np.asarray(c), np.asarray(a)) for d, (c, a) in cands.items()}
    roots_q = {int(d): root_of[int(d)] for d in cands}
    if args.stream_scores:
        gm_agg, _ = GM.from_embeddings_streaming(
            z_test["model"].numpy(), z_test["dataset"].numpy(), gm_cands, roots_q)
    else:
        scores_by_q = {int(d): (F_norm(z_test["model"]) @ F_norm(z_test["dataset"])[int(d)]).numpy()
                       for d in cands}
        gm_agg, _ = GM.from_scores(scores_by_q, gm_cands, roots_q)

    with torch.no_grad():
        if args.chunked_infer:
            z_full = chunked_forward(model, data, chunk_size=args.chunked_infer,
                                     device=device)
        else:
            z_full = {k: v.cpu() for k, v in model(data.clone().to(device)).items()}
    z_m = z_full["model"].numpy().astype(np.float32)
    z_d = z_full["dataset"].numpy().astype(np.float32)
    if args.expect_n is not None:
        assert z_m.shape[0] == args.expect_n, (
            f"candidate pool is {z_m.shape[0]}, expected {args.expect_n}")

    np.save(os.path.join(out, "z_m.npy"), z_m)
    np.save(os.path.join(out, "z_d.npy"), z_d)
    np.save(os.path.join(out, "z_m_eval.npy"), z_test["model"].numpy().astype(np.float32))
    np.save(os.path.join(out, "z_d_eval.npy"), z_test["dataset"].numpy().astype(np.float32))
    payload["unique_model_id"].sort_values("mappedID").to_csv(
        os.path.join(out, "model_ids.csv"), index=False)
    udi.to_csv(os.path.join(out, "dataset_ids.csv"), index=False)
    np.savez_compressed(
        os.path.join(out, "gold_cands.npz"),
        **{str(d): np.stack([np.asarray(c, float), np.asarray(a, float)])
           for d, (c, a) in cands.items()})

    idx, hnsw = build_hnsw(z_m, z_d, cands, ef=args.ef_construction, M=args.hnsw_M,
                           threads=args.hnsw_threads, iso_recall=args.iso_recall)
    idx.save_index(os.path.join(out, "hnsw_index.bin"))
    if hnsw.get("ef_trace"):
        with open(os.path.join(out, "ef_tuning.json"), "w", encoding="utf-8") as f:
            json.dump({"target": hnsw["iso_recall"], "ef_search": hnsw["ef_search"],
                       "trace": hnsw["ef_trace"]}, f, indent=2)

    report = dict(
        tag=args.tag, graph=os.path.basename(args.graph),
        seed=args.seed, epochs=args.epochs, device=device,
        scale_switches={k: getattr(args, k) for k in (
            "fanout", "sparse_M", "contrast_n_neg", "contrast_max_pos_per_dataset",
            "chunked_infer", "stream_scores", "skip_diagnostics", "hnsw_threads",
            "ef_construction", "hnsw_M", "iso_recall", "expect_n")},
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
