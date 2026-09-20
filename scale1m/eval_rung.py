"""
eval_rung.py -- T8: the three axes (docs/1M/100kplan.md §11).

  A  accuracy      gold@K over the whole lake, from the HELD-OUT embeddings only
  B  iso-recall latency   HNSW vs an O(N) full scan, at equal recall
  C  cold-start     four degree-defined layers, and whether the frozen majority
                    is retrievable or collapsed

WHY ALL RUNGS RUN IN ONE PROCESS
    §11.2 rule 5. Latency is the B axis's whole claim, and CPU differs between
    nodes; measuring 12K on one machine and 100K on another produces a "curve"
    whose slope is partly hardware. So this takes every rung at once and refuses
    to plot a curve it did not measure together.

WHY B MEASURES OUR OWN FULL SCAN
    The competitor column is extrapolated by D-16, and an extrapolated number
    cannot carry the claim on its own. But the architectural fact -- a scoring
    function that mixes query and candidate in a hidden layer cannot be ANN
    indexed and must scan Θ(N) -- is testable on our own embeddings: brute-force
    MIPS over the same z_m, same queries, same machine, same run. That comparison
    is measured end to end and owes nothing to anyone's release.

THE ONE RULE THAT MAKES B HONEST
    Tune ef_search until recall@50 >= target, THEN time. Any ANN index is
    arbitrarily fast if it is allowed to be arbitrarily wrong, so latency at an
    unstated recall is not a claim about anything.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(_HERE), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scale1m.hf_crawl import utcnow, write_json_atomic

MODELLENS_ANCHORS = [(1_000, 1.96), (12_000, 16.6)]

WARM_MIN_DEG = 10


def model_layers(data, n_models):
    from ModelLakeFishing.stage2TrainGraphSAGE.losses import TRAINED_ON
    deg = np.zeros(n_models, dtype=np.int64)
    ei = data[TRAINED_ON].edge_index[0].numpy()
    np.add.at(deg, ei, 1)

    has_lin = np.zeros(n_models, dtype=bool)
    for et in data.edge_types:
        if et[1] in ("is_base_of", "rev_is_base_of"):
            e = data[et].edge_index.numpy()
            has_lin[e[0]] = True
            has_lin[e[1]] = True

    layer = np.full(n_models, "frozen", dtype=object)
    layer[(deg >= 1) & (deg < WARM_MIN_DEG)] = "cool"
    layer[deg >= WARM_MIN_DEG] = "warm"
    layer[(deg == 0) & has_lin] = "cold"
    return layer, deg, has_lin


def participation_ratio(Z):
    if len(Z) < 2:
        return float("nan")
    X = Z - Z.mean(0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    v = s ** 2
    return float(v.sum() ** 2 / max((v ** 2).sum(), 1e-30))


def layer_geometry(z_m, layer, rng, max_sample=4000):
    zn = z_m / (np.linalg.norm(z_m, axis=1, keepdims=True) + 1e-12)
    out = {}
    for name in ("warm", "cool", "cold", "frozen"):
        idx = np.flatnonzero(layer == name)
        rec = {"n": int(idx.size), "share": float(idx.size / len(layer))}
        if idx.size >= 2:
            take = idx if idx.size <= max_sample else rng.choice(idx, max_sample, False)
            Z = zn[take]
            G = Z @ Z.T
            iu = np.triu_indices(len(Z), k=1)
            rec["mean_cos"] = float(G[iu].mean())
            rec["effective_dim"] = participation_ratio(Z)
            rec["sampled"] = int(len(Z))
        out[name] = rec
    warm = np.flatnonzero(layer == "warm")
    if warm.size:
        c_warm = zn[warm].mean(0)
        for name in ("cool", "cold", "frozen"):
            idx = np.flatnonzero(layer == name)
            if idx.size:
                c = zn[idx].mean(0)
                out[name]["centroid_cos_to_warm"] = float(
                    c @ c_warm / (np.linalg.norm(c) * np.linalg.norm(c_warm) + 1e-12))
    return out


def sibling_vs_random_cosine(z_m, data, rng, n_pairs=20_000):
    zn = z_m / (np.linalg.norm(z_m, axis=1, keepdims=True) + 1e-12)
    pairs = None
    for et in data.edge_types:
        if et[1] == "is_base_of":
            pairs = data[et].edge_index.numpy()
            break
    if pairs is None or pairs.shape[1] == 0:
        return {"note": "no lineage edges"}
    m = min(n_pairs, pairs.shape[1])
    sel = rng.choice(pairs.shape[1], m, replace=False) if pairs.shape[1] > m else np.arange(m)
    a, b = pairs[0, sel], pairs[1, sel]
    sib = (zn[a] * zn[b]).sum(1)
    ra = rng.integers(0, len(zn), m)
    rb = rng.integers(0, len(zn), m)
    rnd = (zn[ra] * zn[rb]).sum(1)
    return {"n_pairs": int(m),
            "sibling_mean_cos": float(sib.mean()), "sibling_p95": float(np.percentile(sib, 95)),
            "random_mean_cos": float(rnd.mean()),
            "separation": float(sib.mean() - rnd.mean())}


def _recall_at(idx, qvecs, brute, K, ef):
    idx.set_ef(int(ef))
    r = []
    for i, q in enumerate(qvecs):
        lab, _ = idx.knn_query(q, k=K)
        r.append(len(set(lab[0].tolist()) & brute[i]) / K)
    return float(np.mean(r))


def bench_rung(z_m, z_d_eval, query_ids, index_path, *, target=0.99, K=50,
               warmup=100, reps=1000, n_query=300, threads=8, labels=None):
    import hnswlib
    zm = z_m / (np.linalg.norm(z_m, axis=1, keepdims=True) + 1e-12)
    zd = z_d_eval / (np.linalg.norm(z_d_eval, axis=1, keepdims=True) + 1e-12)
    N, dim = zm.shape
    idx = hnswlib.Index(space="ip", dim=dim)
    idx.load_index(index_path, max_elements=N)

    qs = list(query_ids)[:min(n_query, len(query_ids))]
    qvecs = [zd[int(d)].astype(np.float32) for d in qs]
    lb = np.arange(N) if labels is None else np.asarray(labels)
    brute = [set(lb[np.argsort(-(zm @ q))[:K]].tolist()) for q in qvecs]

    ef, trace = None, []
    e = K
    while e <= 2048:
        rec = _recall_at(idx, qvecs, brute, K, e)
        trace.append({"ef": int(e), "recall": rec})
        if rec >= target:
            ef = e
            break
        e *= 2
    if ef is None:
        ef, rec = 2048, trace[-1]["recall"]
    else:
        lo, hi = (ef // 2 if ef > K else K), ef
        while hi - lo > 1:
            mid = (lo + hi) // 2
            r = _recall_at(idx, qvecs, brute, K, mid)
            trace.append({"ef": int(mid), "recall": r})
            if r >= target:
                hi = mid
            else:
                lo = mid
        ef = hi
    rec = _recall_at(idx, qvecs, brute, K, ef)

    idx.set_ef(int(ef))
    idx.set_num_threads(1)
    for i in range(warmup):
        idx.knn_query(qvecs[i % len(qvecs)], k=K)
    lat = []
    for i in range(reps):
        q = qvecs[i % len(qvecs)]
        t = time.perf_counter_ns()
        idx.knn_query(q, k=K)
        lat.append((time.perf_counter_ns() - t) / 1e6)

    zf = zm.astype(np.float32)
    for i in range(min(warmup, 20)):
        _ = zf @ qvecs[i % len(qvecs)]
    blat = []
    for i in range(min(reps, 200)):
        q = qvecs[i % len(qvecs)]
        t = time.perf_counter_ns()
        s = zf @ q
        np.argpartition(-s, K)[:K]
        blat.append((time.perf_counter_ns() - t) / 1e6)

    idx.set_num_threads(int(threads))
    t = time.perf_counter_ns()
    for q in qvecs:
        idx.knn_query(q, k=K)
    qps = len(qvecs) / ((time.perf_counter_ns() - t) / 1e9)

    p = lambda a, q: float(np.percentile(a, q))
    return {
        "N": int(N), "ef_search": int(ef), "recall_at_50": float(rec),
        "iso_recall_target": target, "ef_trace": trace,
        "hnsw_p50": p(lat, 50), "hnsw_p95": p(lat, 95), "hnsw_p99": p(lat, 99),
        "scan_p50": p(blat, 50), "scan_p95": p(blat, 95),
        "speedup_p50": p(blat, 50) / max(p(lat, 50), 1e-9),
        "qps_threads": int(threads), "qps": float(qps),
        "index_mb": round(N * (zm.shape[1] * 4 + 32 * 2 * 4) / 2 ** 20, 1),
        "latency_reps": reps, "warmup": warmup, "n_query": len(qs),
    }


def modellens_extrapolated(N):
    (n1, t1), (n2, t2) = MODELLENS_ANCHORS
    slope = (t2 - t1) / (n2 - n1)
    return slope * N + (t1 - slope * n1)


def fit_curves(points):
    if len(points) < 3:
        return {"note": "need >= 3 rungs"}
    N = np.array([p[0] for p in points], float)
    y = np.array([p[1] for p in points], float)
    A = np.vstack([np.log(N), np.ones_like(N)]).T
    log_coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    log_res = float(((A @ log_coef - y) ** 2).sum())
    out = {"log_fit": {"a": float(log_coef[0]), "b": float(log_coef[1]),
                       "sse": log_res}}
    if (y > 0).all():
        B = np.vstack([np.log(N), np.ones_like(N)]).T
        pw, *_ = np.linalg.lstsq(B, np.log(y), rcond=None)
        out["power_fit"] = {"alpha": float(pw[0]), "c": float(np.exp(pw[1])),
                            "note": "alpha ~ 0 means flat in N (sublinear win)"}
    return out


def displacement_composition(z_m_eval, z_d_eval, gold, layer, n_core, rng,
                             max_q=400):
    zm = z_m_eval / (np.linalg.norm(z_m_eval, axis=1, keepdims=True) + 1e-12)
    zd = z_d_eval / (np.linalg.norm(z_d_eval, axis=1, keepdims=True) + 1e-12)
    qs = list(gold.keys())
    if len(qs) > max_q:
        qs = [qs[i] for i in rng.choice(len(qs), max_q, replace=False)]
    tot = {"core": 0, "halo": 0}
    bylayer = {"warm": 0, "cool": 0, "cold": 0, "frozen": 0}
    n_q, n_disp = 0, 0
    for d in qs:
        cand, acc = gold[d]
        if len(cand) == 0:
            continue
        best = int(cand[int(np.argmax(acc))])
        s = zm @ zd[int(d)]
        above = np.flatnonzero(s > s[best])
        if above.size == 0:
            n_q += 1
            continue
        n_q += 1
        n_disp += int(above.size)
        tot["core"] += int((above < n_core).sum())
        tot["halo"] += int((above >= n_core).sum())
        for nm in bylayer:
            bylayer[nm] += int((layer[above] == nm).sum())
    denom = max(tot["core"] + tot["halo"], 1)
    return {"n_queries": n_q, "total_displacers": n_disp,
            "mean_displacers_per_query": n_disp / max(n_q, 1),
            "core": tot["core"], "halo": tot["halo"],
            "halo_share": tot["halo"] / denom,
            "by_layer": bylayer,
            "by_layer_share": {k: v / denom for k, v in bylayer.items()},
            "note": "composition only; per-displacer accuracy needs a "
                    "(HALO model, dataset) accuracy join T3 did not materialise"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="T7-exported run dirs, one per rung, ALL in one call")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ladder", default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-b", action="store_true", help="A and C only")
    args = ap.parse_args(argv)

    import torch
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    report = {"written_at": utcnow(), "rungs": {}, "protocol": {
        "b_axis": "iso-recall then time; single-thread; warmup discarded",
        "modellens": "EXTRAPOLATED from P4 anchors (D-16), never measured here"}}

    for run in args.runs:
        run = os.path.abspath(run)
        exp = os.path.join(run, "exports")
        with open(os.path.join(exp, "EXPORT_MANIFEST.json"), encoding="utf-8") as fh:
            em = json.load(fh)
        rung = em["rung"]
        print(f"\n===== {rung}  ({os.path.basename(run)}) =====", flush=True)

        z_m = np.load(os.path.join(exp, "z_m.npy"))
        z_m_eval = np.load(os.path.join(exp, "z_m_eval.npy"))
        z_d_eval = np.load(os.path.join(exp, "z_d_eval.npy"))
        gz = np.load(os.path.join(exp, "gold_cands.npz"))
        gold = {int(k): (gz[k][0].astype(int), gz[k][1]) for k in gz.files}

        payload = torch.load(em["graph"], map_location="cpu", weights_only=False)
        data = payload["data"]
        n_models = int(data["model"].num_nodes)
        n_core = 30_183 if n_models > 30_183 else n_models

        layer, deg, has_lin = model_layers(data, n_models)
        rec = {
            "run": run, "run_id": em.get("run_id"), "N": n_models,
            "graph_sha256": em["graph_sha256"],
            "a_axis": em["a_axis_row"],
            "gates_from_export": {g["gate"]: g.get("ok") for g in em["gates"]},
            "c_axis": {
                "layers": layer_geometry(z_m, layer, rng),
                "lineage": sibling_vs_random_cosine(z_m, data, rng),
                "warm_min_degree": WARM_MIN_DEG,
            },
            "displacement": displacement_composition(
                z_m_eval, z_d_eval, gold, layer, n_core, rng),
        }
        gold_layer = {}
        for d, (cand, acc) in gold.items():
            if len(cand) == 0:
                continue
            best = int(cand[int(np.argmax(acc))])
            gold_layer[d] = layer[best]
        from collections import Counter
        rec["c_axis"]["gold_model_layer_counts"] = dict(Counter(gold_layer.values()))

        if not args.skip_b:
            idxp = os.path.join(exp, f"hnsw_{rung}.bin")
            rec["b_axis"] = bench_rung(z_m, z_d_eval, list(gold.keys()), idxp,
                                       threads=args.threads)
            rec["b_axis"]["modellens_p50_extrapolated"] = modellens_extrapolated(n_models)
            rec["b_axis"]["speedup_vs_modellens_extrapolated"] = (
                rec["b_axis"]["modellens_p50_extrapolated"] / max(rec["b_axis"]["hnsw_p50"], 1e-9))
            b = rec["b_axis"]
            print(f"  B: ef={b['ef_search']} recall={b['recall_at_50']:.4f} "
                  f"hnsw p50={b['hnsw_p50']:.4f}ms scan p50={b['scan_p50']:.4f}ms "
                  f"speedup={b['speedup_p50']:.1f}x qps={b['qps']:.0f}")
        c = rec["c_axis"]["layers"]
        print("  C: " + "  ".join(
            f"{k}={c[k]['n']}({c[k]['share']*100:.1f}%,cos={c[k].get('mean_cos', float('nan')):.3f},"
            f"d={c[k].get('effective_dim', float('nan')):.1f})" for k in
            ("warm", "cool", "cold", "frozen")))
        report["rungs"][rung] = rec

    if not args.skip_b:
        pts = [(r["N"], r["b_axis"]["hnsw_p50"]) for r in report["rungs"].values()]
        report["b_axis_curve"] = fit_curves(sorted(pts))
    write_json_atomic(os.path.join(args.out, "T8_REPORT.json"), report)
    print(f"\n-> {os.path.join(args.out, 'T8_REPORT.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
