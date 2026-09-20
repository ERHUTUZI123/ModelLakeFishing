import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(_HERE), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scale1m.hf_crawl import data_root, utcnow, write_json_atomic
from scale1m.eval_rung import (WARM_MIN_DEG, model_layers,
                               layer_geometry, sibling_vs_random_cosine,
                               bench_rung, fit_curves)

SEEDS = (0, 1, 2)
RUN_FMT = "RF_full_s%d_e25"


def run_dir(args, seed):
    return os.path.join(args.exports, args.run_fmt % seed)


def training_run_dir(args, seed):
    return os.path.join(args.f6_runs, args.f6_run_fmt % seed)


def default_full_sidecar(args):
    return os.path.join(run_dir(args, 0), "prior_sidecar.npz")


def curve_index_files(d0):
    sub = os.path.join(d0, "curve")
    if not os.path.isdir(sub):
        return []
    return [os.path.join(sub, f) for f in sorted(os.listdir(sub))
            if f.endswith(".bin")]


def load_manifest(args, seed):
    with open(os.path.join(run_dir(args, seed), "EXPORT_MANIFEST.json"),
              encoding="utf-8") as fh:
        return json.load(fh)


def load_cands(args, seed):
    with np.load(os.path.join(run_dir(args, seed), "gold_cands.npz")) as z:
        return {int(k): (z[k][0].astype(np.int64), z[k][1].astype(float))
                for k in z.files}


def merge(out, axis, payload, fname="F8_REPORT.json"):
    p = os.path.join(out, fname)
    rep = {}
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as fh:
            rep = json.load(fh)
    rep.setdefault("axes", {})[axis] = payload
    rep["written_at"] = utcnow()
    write_json_atomic(p, rep)
    return rep


def interval(vals):
    v = [float(x) for x in vals]
    return {"min": min(v), "mean": float(np.mean(v)), "max": max(v),
            "per_seed": v}


def curve_point(z_m_eval, z_d_eval, cands, roots, sel, device):
    from ModelLakeFishing.scale import global_metrics as GM
    pos = np.full(z_m_eval.shape[0], -1, dtype=np.int64)
    pos[sel] = np.arange(len(sel))
    keep = {}
    for d, (c, a) in cands.items():
        m = pos[c] >= 0
        if m.sum() < 3 or float(np.std(a[m])) <= 0:
            continue
        keep[int(d)] = (pos[c[m]], a[m])
    agg, _ = GM.from_embeddings_streaming(
        z_m_eval[sel], z_d_eval, keep, {d: roots[d] for d in keep}, device=device)
    n = len(sel)
    return {"N": int(n), "n_queries": agg["n_queries"], "n_roots": agg["n_roots"],
            "gold@1": agg["gold@1"], "gold@10": agg["gold@10"],
            "top3@10": agg["top3@10"], "gold-gap@10": agg["gold-gap@10"],
            "root_gold@10": agg["root_gold@10"],
            "median_gold_rank": agg["median_gold_rank"],
            "median_rank_over_N": agg["median_gold_rank"] / n,
            "vs_random": agg["gold@10"] / (10.0 / n)}


def index_labels(path, n_max):
    import hnswlib
    idx = hnswlib.Index(space="ip", dim=128)
    idx.load_index(path, max_elements=n_max)
    lab = np.sort(np.asarray(idx.get_ids_list(), dtype=np.int64))
    del idx
    gc.collect()
    return lab


def axis_a(args):
    t0 = time.time()
    rows = {}
    for s in SEEDS:
        m = load_manifest(args, s)["stages"]["metrics"]
        rows[s] = {"all_candidates": m["a_axis_row"],
                   "in_snapshot_only": m["a_axis_row_in_snapshot_only"],
                   "eval_metrics": m["eval_metrics"]}
    keys = ["gold@1", "gold@10", "top3@10", "gold-gap@10", "root_gold@10",
            "median_gold_rank", "median_rank_over_N", "vs_random", "n_queries"]
    summary = {"all_candidates":
               {k: interval([rows[s]["all_candidates"][k] for s in SEEDS]) for k in keys},
               "in_snapshot_only":
               {k: interval([rows[s]["in_snapshot_only"][k] for s in SEEDS]) for k in keys}}
    obs = {k: interval([rows[s]["eval_metrics"][k] for s in SEEDS])
           for k in ("observed_hit1", "top3_hit1", "regret1")}

    d0 = run_dir(args, 0)
    zm = np.load(os.path.join(d0, "z_m_eval.npy"), mmap_mode="r")
    zd = np.load(os.path.join(d0, "z_d_eval.npy"))
    cands = load_cands(args, 0)
    udi = pd.read_parquet(os.path.join(d0, "dataset_ids.parquet")).sort_values("mappedID")
    roots = {int(d): str(r) for d, r in zip(udi["mappedID"], udi["root"])}
    zm_mem = np.asarray(zm)

    curve = [dict(curve_point(zm_mem, zd, cands, roots,
                              np.arange(zm.shape[0]), args.device),
                  sample_seed=None, source="full lake")]
    for path in curve_index_files(d0):
        f = os.path.basename(path)
        lab = index_labels(path, zm.shape[0])
        pt = curve_point(zm_mem, zd, cands, roots, lab, args.device)
        pt["sample_seed"] = int(f.rsplit("_s", 1)[1].split(".")[0])
        pt["source"] = f
        curve.append(pt)
        print("  [A curve] N=%-9d seed=%s gold@10=%.4f rank/N=%.3e"
              % (pt["N"], pt["sample_seed"], pt["gold@10"],
                 pt["median_rank_over_N"]), flush=True)

    payload = {"seconds": round(time.time() - t0, 1),
               "per_seed": rows, "three_seed_interval": summary,
               "observed_pool_metrics": obs,
               "retrieval_curve": sorted(curve, key=lambda r: (r["N"], r["sample_seed"] or -1)),
               "note": "A uses z_*_eval only; the curve restricts the candidate "
                       "pool, never the query set"}
    merge(args.out, "a", payload)
    print(json.dumps(summary["all_candidates"]["gold@10"], indent=2))
    return payload


def axis_b(args):
    t0 = time.time()
    d0 = run_dir(args, 0)
    z_m = np.load(os.path.join(d0, "z_m.npy"))
    z_d = np.load(os.path.join(d0, "z_d_eval.npy"))
    cands = load_cands(args, 0)
    qids = list(cands)

    out = []
    jobs = [("full", None, os.path.join(d0, "hnsw_full.bin"))]
    sub = os.path.join(d0, "curve")
    for f in sorted(os.listdir(sub)):
        if f.endswith(".bin"):
            stem, seed = f[:-4].rsplit("_s", 1)
            jobs.append((stem.replace("hnsw_sub_", ""), int(seed),
                         os.path.join(sub, f)))

    for name, s, path in jobs:
        lab = None if name == "full" else index_labels(path, z_m.shape[0])
        if lab is None:
            zm_sub, remap = z_m, None
        else:
            zm_sub = z_m[lab]
            remap = lab
        r = bench_rung(zm_sub, z_d, qids, path, target=args.iso_recall,
                       threads=args.threads, reps=args.reps, warmup=args.warmup,
                       n_query=args.n_query, labels=remap)
        r["rung"] = name
        r["sample_seed"] = s
        r["index_bytes"] = os.path.getsize(path)
        r.pop("ef_trace", None)
        out.append(r)
        print("  [B] %-6s seed=%-4s N=%-9d ef=%-4d recall=%.4f  hnsw p50=%.4f ms  "
              "scan p50=%.3f ms  %.0fx  qps=%.0f"
              % (name, s, r["N"], r["ef_search"], r["recall_at_50"], r["hnsw_p50"],
                 r["scan_p50"], r["speedup_p50"], r["qps"]), flush=True)
        del zm_sub
        gc.collect()

    pts = sorted({(r["N"], r["hnsw_p50"]) for r in out})
    payload = {"seconds": round(time.time() - t0, 1),
               "machine": args.machine, "protocol":
                   "iso-recall then time; single thread; %d warmup discarded, "
                   "%d timed; full scan measured in the same process"
                   % (args.warmup, args.reps),
               "rows": out, "p50_vs_N_fit": fit_curves(pts)}
    merge(args.out, "b", payload)
    return payload


def axis_c(args):
    import torch
    from scale1m.graph_store import load_sharded
    t0 = time.time()
    payload = load_sharded(args.graph, mmap=True, verify_sha256=False)
    data = payload["data"]
    n = int(data["model"].num_nodes)
    layer, deg, has_lin = model_layers(data, n)
    ladder = pd.read_parquet(args.ladder, columns=["model", "mappedID", "in_snapshot"])
    in_snap = ladder.sort_values("mappedID")["in_snapshot"].to_numpy().astype(bool)

    rng = np.random.default_rng(args.seed)
    per_seed = {}
    for s in SEEDS:
        z_m = np.load(os.path.join(run_dir(args, s), "z_m.npy"))
        rec = {"layers": layer_geometry(z_m, layer, rng),
               "lineage": sibling_vs_random_cosine(z_m, data, rng)}
        per_seed[s] = rec
        c = rec["layers"]
        print("  [C s%d] " % s + "  ".join(
            "%s=%d(%.1f%%, cos=%.3f, d=%.1f)"
            % (k, c[k]["n"], 100 * c[k]["share"], c[k].get("mean_cos", float("nan")),
               c[k].get("effective_dim", float("nan")))
            for k in ("warm", "cool", "cold", "frozen")), flush=True)
        del z_m
        gc.collect()

    from ModelLakeFishing.stage2TrainGraphSAGE.losses import TRAINED_ON
    hubs = []
    for et in data.edge_types:
        if et[1] == "is_base_of":
            src = data[et].edge_index[0].numpy()
            cnt = np.bincount(src, minlength=n)
            top = np.argsort(-cnt)[:args.n_hubs]
            umi = payload["unique_model_id"].sort_values("mappedID")["model"].astype(str).to_numpy()
            z0 = np.load(os.path.join(run_dir(args, 0), "z_m.npy"), mmap_mode="r")
            dst = data[et].edge_index[1].numpy()
            for h in top:
                kids = dst[src == h]
                if len(kids) < 2:
                    continue
                take = kids if len(kids) <= 2000 else rng.choice(kids, 2000, False)
                Z = np.asarray(z0[np.sort(take)], dtype=np.float64)
                Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
                G = Z @ Z.T
                iu = np.triu_indices(len(Z), k=1)
                hubs.append({"parent": umi[int(h)], "n_children": int(cnt[h]),
                             "sampled": int(len(Z)),
                             "children_mean_cos": float(G[iu].mean())})
            break

    counts = {k: int((layer == k).sum()) for k in ("warm", "cool", "cold", "frozen")}
    payload_out = {
        "seconds": round(time.time() - t0, 1), "N": n,
        "warm_min_degree": WARM_MIN_DEG,
        "layer_counts": counts,
        "layer_share": {k: v / n for k, v in counts.items()},
        "layer_by_in_snapshot": {
            k: {"in_snapshot": int(((layer == k) & in_snap).sum()),
                "out_of_snapshot": int(((layer == k) & ~in_snap).sum())}
            for k in counts},
        "per_seed": per_seed,
        "largest_lineage_hubs": hubs,
    }
    merge(args.out, "c", payload_out)
    return payload_out


def axis_d(args):
    t0 = time.time()
    d0 = run_dir(args, 0)
    ex = {s: load_manifest(args, s) for s in SEEDS}
    train = load_train_costs(args)

    def dirsize(p):
        return sum(os.path.getsize(os.path.join(r, f))
                   for r, _d, fs in os.walk(p) for f in fs)

    onboarding = incremental_onboarding(d0, args)

    def report_seconds(path, key="wallclock_s"):
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as fh:
            return json.load(fh).get(key)

    payload = {
        "seconds": round(time.time() - t0, 1),
        "build_wallclock_s": {
            "features_x_m": report_seconds(
                os.path.join(os.path.dirname(args.graph), "..", "feats_rf",
                             "FEATS_REPORT.json")),
            "graph": report_seconds(os.path.join(args.graph, "GRAPH_REPORT.json")),
            "train_per_seed": train,
            "note": "crawl / dataset index / canonicalisation are recorded in "
                    "F1.md, F1.5.md and F2.md; they are wall-clock facts of "
                    "those stages, not re-measured here",
        },
        "export_seconds_per_seed": {
            s: {k: v.get("seconds") for k, v in ex[s]["stages"].items()} for s in SEEDS},
        "disk_bytes": {
            "graph": dirsize(args.graph),
            "exports_one_seed": dirsize(d0),
            "hnsw_full": os.path.getsize(os.path.join(d0, "hnsw_full.bin")),
            "curve_indexes": dirsize(os.path.join(d0, "curve")),
            "z_m": os.path.getsize(os.path.join(d0, "z_m.npy")),
        },
        "incremental_onboarding": onboarding,
    }
    merge(args.out, "d", payload)
    print(json.dumps(onboarding, indent=2))
    return payload


def load_train_costs(args):
    train = {}
    missing = []
    for s in SEEDS:
        mp = os.path.join(training_run_dir(args, s), "MANIFEST.json")
        if not os.path.isfile(mp):
            missing.append(mp)
            continue
        with open(mp, encoding="utf-8") as fh:
            m = json.load(fh)
        train[s] = {"wallclock_s": m.get("wallclock_s"),
                    "peak_gpu_mem_gb": m.get("peak_gpu_mem_gb")}
    if missing:
        raise FileNotFoundError(
            "D-axis training manifests are missing; set --f6-runs and "
            "--f6-run-fmt for the training-run family:\n  "
            + "\n  ".join(missing))
    return train


def incremental_onboarding(d0, args):
    import hnswlib
    z_m = np.load(os.path.join(d0, "z_m.npy"), mmap_mode="r")
    N, dim = z_m.shape
    idx = hnswlib.Index(space="ip", dim=dim)
    idx.load_index(os.path.join(d0, "hnsw_full.bin"), max_elements=N + args.n_insert)
    idx.set_num_threads(1)
    rng = np.random.default_rng(0)
    v = np.asarray(z_m[rng.integers(0, N, args.n_insert)], dtype=np.float32)
    v += rng.normal(0, 1e-3, v.shape).astype(np.float32)
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    lat = []
    for i in range(args.n_insert):
        t = time.perf_counter_ns()
        idx.add_items(v[i:i + 1], np.array([N + i]))
        lat.append((time.perf_counter_ns() - t) / 1e6)
    return {"n_inserted": int(args.n_insert),
            "hnsw_insert_ms_p50": float(np.percentile(lat, 50)),
            "hnsw_insert_ms_p95": float(np.percentile(lat, 95)),
            "note": "HNSW insertion only; the MiniLM descriptor pass and the "
                    "one-node GNN forward are measured in F4 and F7 and are "
                    "reported there"}


SHRINK_K = 5.0
TOPK = 10


def _prior_tables(sc, shrink=SHRINK_K):
    import numpy as np
    import pandas as pd
    em = sc["edge_model"].astype(np.int64)
    ed = sc["edge_dataset"].astype(np.int64)
    ea = sc["edge_acc"].astype(np.float64)
    task_of, root_of = sc["task_id"], sc["root_id"]

    def group(key):
        df = pd.DataFrame({"k": key, "m": em, "a": ea})
        g = df.groupby(["k", "m"], sort=True)["a"].agg(["sum", "count"])
        b = ((g["sum"] + 0.5 * shrink) / (g["count"] + shrink)).to_numpy()
        ks = g.index.get_level_values(0).to_numpy()
        ms = g.index.get_level_values(1).to_numpy()
        cut = np.searchsorted(ks, np.unique(ks))
        cut = np.append(cut, len(ks))
        out = {}
        for i, k in enumerate(np.unique(ks)):
            out[int(k)] = (ms[cut[i]:cut[i + 1]], b[cut[i]:cut[i + 1]])
        return out

    by_task = group(task_of[ed])
    by_root = group(root_of[ed])
    df = pd.DataFrame({"m": em, "a": ea}).groupby("m")["a"].agg(["sum", "count"])
    null_idx = df.index.to_numpy()
    null_val = ((df["sum"] + 0.5 * shrink) / (df["count"] + shrink)).to_numpy()
    return by_task, by_root, (null_idx, null_val)


def _minmax_pass(zm, zd, qids, device, model_chunk):
    import torch
    q = torch.as_tensor(qids, dtype=torch.long, device=device)
    zq = zd[q]
    lo = torch.full((len(qids),), float("inf"), device=device)
    hi = torch.full((len(qids),), float("-inf"), device=device)
    for ms in range(0, zm.size(0), model_chunk):
        s = zm[ms:ms + model_chunk] @ zq.t()
        lo = torch.minimum(lo, s.min(0).values)
        hi = torch.maximum(hi, s.max(0).values)
        del s
    return lo, hi


def axis_p(args):
    import gc
    import json
    import os
    import time

    import numpy as np
    import pandas as pd
    import torch

    from ModelLakeFishing.scale import global_metrics as GM
    from scale1m.fast_lever_audit import _topk_update
    from scale1m.utility_scorecard import score_source, norm_task

    t0 = time.time()
    dev = args.device
    betas = [float(b) for b in args.betas]

    meta = pd.read_parquet(args.model_meta)
    N_meta = len(meta)
    row_of = pd.Series(meta["mappedID"].to_numpy(), index=meta["model"].astype(str))
    sup = pd.read_parquet(args.supervision, columns=["node", "model"])
    sup["m_row"] = row_of.reindex(sup["model"].astype(str)).to_numpy()
    sup = sup.dropna(subset=["m_row"])
    sup["m_row"] = sup["m_row"].astype(int)
    sup["task"] = sup["node"].astype(str).str.split("\t").str[1].map(norm_task)
    task_models = {t: np.unique(g["m_row"].to_numpy()) for t, g in sup.groupby("task")}
    by_node = {k: set(v) for k, v in
               sup.groupby(sup["node"].astype(str))["m_row"].apply(set).items()}

    per_seed = {}
    for s in SEEDS:
        d0 = run_dir(args, s)
        side = os.path.join(d0, "prior_sidecar_s%d.npz" % s)
        if not os.path.isfile(side):
            raise SystemExit("missing %s -- run stage3HNSW.build_prior_sidecar "
                             "--graph-store ... --split-seed %d first" % (side, s))
        sc = np.load(side)
        with open(side.replace(".npz", "_meta.json"), encoding="utf-8") as fh:
            side_meta = json.load(fh)
        by_task, by_root, (null_idx, null_val) = _prior_tables(sc)
        task_of, root_of = sc["task_id"], sc["root_id"]

        udi = pd.read_parquet(os.path.join(d0, "dataset_ids.parquet")).sort_values("mappedID")
        node_of_all = udi["dataset"].astype(str).to_numpy()
        cands = load_cands(args, s)
        qids = sorted(cands)
        Q = len(qids)
        node_of = {q: node_of_all[q] for q in qids}
        observed = {q: by_node.get(node_of[q], set()) for q in qids}

        n_sib = sum(1 for q in qids if int(root_of[q]) in by_root)
        assert n_sib == 0, ("%d queries have a train/val-visible sibling; the "
                            "alpha term is no longer inert" % n_sib)

        zm = torch.from_numpy(np.load(os.path.join(d0, "z_m_eval.npy")))
        zd = torch.from_numpy(np.load(os.path.join(d0, "z_d_eval.npy")))
        zm = (zm / (zm.norm(dim=1, keepdim=True) + 1e-12)).to(torch.float32).to(dev)
        zd = (zd / (zd.norm(dim=1, keepdim=True) + 1e-12)).to(torch.float32).to(dev)
        N = zm.size(0)

        lo_all, hi_all = _minmax_pass(zm, zd, qids, dev, args.model_chunk)
        span_all = (hi_all - lo_all).clamp_min(1e-9)
        nullv = torch.zeros(N, dtype=torch.float32, device=dev)
        nullv[torch.as_tensor(null_idx, dtype=torch.long, device=dev)] = \
            torch.as_tensor(null_val, dtype=torch.float32, device=dev)

        variants = [("mips", None, 0.0), ("task_null_b1", "null", 1.0)]
        variants += [("task_b%g" % b, "task", b) for b in betas]
        names = [v[0] for v in variants]
        cnt = {v: np.zeros((Q, 3), dtype=np.int64) for v in names}
        top10 = {v: np.zeros((Q, TOPK), dtype=np.int64) for v in args.top10_of}

        for qs in range(0, Q, args.query_chunk):
            blk = qids[qs:qs + args.query_chunk]
            b = len(blk)
            zq = zd[torch.as_tensor(blk, dtype=torch.long, device=dev)]
            lo = lo_all[qs:qs + b]
            span = span_all[qs:qs + b]

            bi, bc, bv = [], [], []
            for j, q in enumerate(blk):
                idx, val = by_task.get(int(task_of[q]), (np.zeros(0, np.int64),
                                                         np.zeros(0)))
                bi.append(idx)
                bc.append(np.full(idx.size, j, dtype=np.int64))
                bv.append(val)
            BI = np.concatenate(bi)
            order = np.argsort(BI, kind="stable")
            BI = BI[order]
            BC = np.concatenate(bc)[order]
            BV = np.concatenate(bv)[order].astype(np.float32)
            BI_t = torch.as_tensor(BI, device=dev)
            BC_t = torch.as_tensor(BC, device=dev)
            BV_t = torch.as_tensor(BV, device=dev)

            probes = []
            for j, q in enumerate(blk):
                cand, acc = cands[q]
                gold, top3, near, ids = GM._probe_ids(cand, acc, GM.GAP_DELTA)
                probes.append((gold, np.asarray(top3, np.int64),
                               np.asarray(near, np.int64), ids))
            flat = np.concatenate([p[3] for p in probes])
            fcol = np.concatenate([np.full(p[3].size, j) for j, p in enumerate(probes)])
            ft = torch.as_tensor(flat, dtype=torch.long, device=dev)
            fc = torch.as_tensor(fcol, dtype=torch.long, device=dev)
            s_pr = (zm[ft] * zq[fc]).sum(-1)
            mm_pr = ((s_pr - lo[fc]) / span[fc]).cpu().numpy()
            nb_pr = nullv[ft].cpu().numpy()
            tb_pr = np.zeros(flat.size, dtype=np.float64)
            off = 0
            for j, q in enumerate(blk):
                idx, val = by_task.get(int(task_of[q]), (np.zeros(0, np.int64),
                                                         np.zeros(0)))
                ids = probes[j][3]
                pos = np.searchsorted(idx, ids)
                pos = np.clip(pos, 0, max(idx.size - 1, 0))
                hit = (idx.size > 0) & (idx[pos] == ids) if idx.size else np.zeros(
                    ids.size, bool)
                tb_pr[off:off + ids.size] = np.where(hit, val[pos] if idx.size else 0.0, 0.0)
                off += ids.size

            T, PID = {}, {}
            for name, kind, beta in variants:
                add = (0.0 if kind is None else
                       beta * (nb_pr if kind == "null" else tb_pr))
                fu = mm_pr + add
                t_arr = np.zeros((b, 3), dtype=np.float32)
                p_arr = np.zeros((b, 3), dtype=np.int64)
                off = 0
                for j, (gold, top3, near, ids) in enumerate(probes):
                    sl = {int(m): off + k for k, m in enumerate(ids)}
                    off += ids.size
                    for slot, group in enumerate(([gold], top3, near)):
                        best = max(group, key=lambda m: fu[sl[int(m)]])
                        t_arr[j, slot] = fu[sl[int(best)]]
                        p_arr[j, slot] = int(best)
                T[name] = torch.as_tensor(t_arr, device=dev)
                PID[name] = p_arr

            buf = {v: (None, None) for v in top10}
            for ms in range(0, N, args.model_chunk):
                sblk = zm[ms:ms + args.model_chunk] @ zq.t()
                c = sblk.size(0)
                mm = (sblk - lo) / span
                del sblk
                B = torch.zeros(c * b, dtype=torch.float32, device=dev)
                k0 = int(np.searchsorted(BI, ms))
                k1 = int(np.searchsorted(BI, ms + c))
                if k1 > k0:
                    B[(BI_t[k0:k1] - ms) * b + BC_t[k0:k1]] = BV_t[k0:k1]
                B = B.view(c, b)
                Bn = nullv[ms:ms + c].unsqueeze(1)
                for name, kind, beta in variants:
                    fused = mm if kind is None else (
                        mm + beta * (Bn if kind == "null" else B))
                    cm = fused.unsqueeze(2) > T[name].unsqueeze(0)
                    p = PID[name]
                    here = (p >= ms) & (p < ms + c)
                    if here.any():
                        jj, ss = np.nonzero(here)
                        cm[torch.as_tensor(p[jj, ss] - ms, device=dev),
                           torch.as_tensor(jj, device=dev),
                           torch.as_tensor(ss, device=dev)] = False
                    cnt[name][qs:qs + b] += cm.sum(0).cpu().numpy()
                    if name in top10:
                        buf[name] = _topk_update(buf[name][0], buf[name][1],
                                                 fused, ms, TOPK)
                    del cm
                    if kind is not None:
                        del fused
                del mm, B
            for v in top10:
                top10[v][qs:qs + b] = buf[v][1].t().cpu().numpy()
            print("    [P s%d] %d/%d queries" % (s, min(qs + b, Q), Q), flush=True)

        rows = {}
        for name in names:
            g = cnt[name][:, 0] + 1
            t3 = cnt[name][:, 1] + 1
            gp = cnt[name][:, 2] + 1
            rows[name] = {
                "n_queries": Q, "gold@1": float((g <= 1).mean()),
                "gold@10": float((g <= 10).mean()),
                "gold@50": float((g <= 50).mean()),
                "gold@100": float((g <= 100).mean()),
                "top3@10": float((t3 <= 10).mean()),
                "gold-gap@10": float((gp <= 10).mean()),
                "median_gold_rank": float(np.median(g)),
                "median_rank_over_N": float(np.median(g)) / N,
                "vs_random": float((g <= 10).mean()) / (10.0 / N)}

        layer2 = []
        for v in top10:
            ranked = {q: (top10[v][i], int(cnt[v][i, 0] + 1)) for i, q in enumerate(qids)}
            row = score_source("P axis :: " + v, ranked, cands, node_of, meta,
                               observed, task_models, N_meta)
            disagree = sum(1 for i, q in enumerate(qids)
                           if (int(cnt[v][i, 0] + 1) <= 10) !=
                           (int(cands[q][0][int(np.argmax(cands[q][1]))])
                            in set(top10[v][i].tolist())))
            row["rank_vs_top10_disagreements"] = disagree
            layer2.append(row)

        per_seed[s] = {"n_queries": Q, "sidecar": side_meta, "variants": rows,
                       "sibling_channel": {
                           "queries_with_visible_sibling": n_sib,
                           "alpha_term_inert": True,
                           "note": "root-aware holds out a whole root, so a test "
                                   "query's siblings are held out with it"},
                       "layer2": layer2}
        print("  [P s%d] mips %.4f | task_null %.4f | %s"
              % (s, rows["mips"]["gold@10"], rows["task_null_b1"]["gold@10"],
                 " | ".join("%s %.4f" % (n, rows[n]["gold@10"])
                            for n in names if n.startswith("task_b"))), flush=True)
        del zm, zd, nullv
        gc.collect()
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()

    payload = {"seconds": round(time.time() - t0, 1), "device": str(dev),
               "alpha": 1.0, "betas": betas, "shrink_k": SHRINK_K,
               "per_seed": per_seed,
               "interval": {n: interval([per_seed[s]["variants"][n]["gold@10"]
                                         for s in SEEDS])
                            for n in per_seed[SEEDS[0]]["variants"]},
               "note": "full-lake fused ranking; minmax over the whole lake as in "
                       "serving_rerank; sibling term inert under root-aware"}
    merge(args.out, "p", payload, fname="X2_PRIOR_FUSION.json")
    return payload


def beta_inf_limit(args):
    import time
    import numpy as np
    import pandas as pd

    t0 = time.time()
    per_seed = {}
    for s in SEEDS:
        d0 = run_dir(args, s)
        sc = np.load(os.path.join(d0, "prior_sidecar_s%d.npz" % s))
        by_task, _by_root, _null = _prior_tables(sc)
        task_of = sc["task_id"]
        cands = load_cands(args, s)
        qids = sorted(cands)
        zm = np.load(os.path.join(d0, "z_m_eval.npy"), mmap_mode="r")
        zd = np.load(os.path.join(d0, "z_d_eval.npy"))
        zd = zd / (np.linalg.norm(zd, axis=1, keepdims=True) + 1e-12)
        universe = np.unique(sc["edge_model"].astype(np.int64))
        pos = np.full(zm.shape[0], -1, dtype=np.int64)
        pos[universe] = np.arange(universe.size)
        ZU = np.asarray(zm[universe], dtype=np.float32)
        ZU /= (np.linalg.norm(ZU, axis=1, keepdims=True) + 1e-12)

        ks = (1, 10, 50, 100)
        hit = {k: 0 for k in ks}
        undecided = {k: 0 for k in ks}
        ranks = []
        for q in qids:
            cand, acc = cands[q]
            gold = int(cand[int(np.argmax(acc))])
            idx, val = by_task.get(int(task_of[q]), (np.zeros(0, np.int64), np.zeros(0)))
            j = int(np.searchsorted(idx, gold))
            if idx.size and j < idx.size and idx[j] == gold:
                tie = val == val[j]
                if tie.sum() > 1:
                    sub = idx[tie]
                    sco = ZU[pos[sub]] @ zd[q]
                    better_tie = int((sco > float(ZU[pos[gold]] @ zd[q])).sum())
                else:
                    better_tie = 0
                r = int((val > val[j]).sum()) + better_tie + 1
                ranks.append(r)
                for k in ks:
                    hit[k] += int(r <= k)
            else:
                ranks.append(float("inf"))
                for k in ks:
                    undecided[k] += int(idx.size < k)
        n = len(qids)
        per_seed[s] = {
            "n_queries": n,
            **{"gold@%d_lower" % k: hit[k] / n for k in ks},
            **{"gold@%d_upper" % k: (hit[k] + undecided[k]) / n for k in ks},
            "median_gold_rank_when_in_pool": float(
                np.median([r for r in ranks if np.isfinite(r)])),
            "queries_gold_in_pool": int(sum(np.isfinite(ranks)))}
        print("  [P-inf s%d] gold@10 in [%.4f, %.4f]"
              % (s, per_seed[s]["gold@10_lower"], per_seed[s]["gold@10_upper"]),
              flush=True)
        del ZU, zm

    payload = {"seconds": round(time.time() - t0, 1), "per_seed": per_seed,
               "interval": interval([per_seed[s]["gold@10_lower"] for s in SEEDS]),
               "note": "exact where the same-task pool has at least k members; "
                       "the upper bound adds the queries whose pool is smaller "
                       "than k and whose gold is outside it"}
    merge(args.out, "p_limit", payload, fname="X2_PRIOR_FUSION.json")
    return payload


NAME_JACCARD_MAX = 0.5


def _group_totals(edge_model, edge_dataset, edge_acc, key_of_dataset):
    import numpy as np
    import pandas as pd
    k = key_of_dataset[edge_dataset]
    df = pd.DataFrame({"k": k, "m": edge_model, "a": edge_acc})
    g = df.groupby(["k", "m"], sort=True)["a"].agg(["sum", "count"])
    ks = g.index.get_level_values(0).to_numpy()
    ms = g.index.get_level_values(1).to_numpy()
    sm = g["sum"].to_numpy()
    cn = g["count"].to_numpy().astype(np.int64)
    uniq = np.unique(ks)
    cut = np.append(np.searchsorted(ks, uniq), len(ks))
    totals = {int(uniq[i]): (ms[cut[i]:cut[i + 1]], sm[cut[i]:cut[i + 1]],
                             cn[cut[i]:cut[i + 1]]) for i in range(len(uniq))}
    per_ds = {}
    order = np.argsort(edge_dataset, kind="stable")
    ed = edge_dataset[order]
    uds = np.unique(ed)
    dcut = np.append(np.searchsorted(ed, uds), len(ed))
    for i, dd in enumerate(uds):
        sl = order[dcut[i]:dcut[i + 1]]
        per_ds[int(dd)] = (edge_model[sl], edge_acc[sl])
    return totals, per_ds


def _boost_minus_self(totals, per_ds, key, d, shrink):
    import numpy as np
    if key not in totals:
        return np.zeros(0, np.int64), np.zeros(0)
    ms, sm, cn = totals[key]
    sm = sm.copy()
    cn = cn.copy()
    if d in per_ds:
        om, oa = per_ds[d]
        pos = np.searchsorted(ms, om)
        pos = np.clip(pos, 0, len(ms) - 1)
        ok = ms[pos] == om
        np.subtract.at(sm, pos[ok], oa[ok])
        np.subtract.at(cn, pos[ok], 1)
    keep = cn > 0
    ms, sm, cn = ms[keep], sm[keep], cn[keep]
    if shrink:
        return ms, (sm + 0.5 * shrink) / (cn + shrink)
    return ms, sm / cn


def _boost_from_datasets(per_ds, datasets, shrink):
    import numpy as np
    from collections import defaultdict
    sm, cn = defaultdict(float), defaultdict(int)
    for dd in datasets:
        if dd not in per_ds:
            continue
        for m, a in zip(*per_ds[dd]):
            sm[int(m)] += float(a)
            cn[int(m)] += 1
    if not sm:
        return np.zeros(0, np.int64), np.zeros(0)
    ms = np.array(sorted(sm), dtype=np.int64)
    s = np.array([sm[int(m)] for m in ms])
    c = np.array([cn[int(m)] for m in ms], dtype=np.int64)
    return ms, ((s + 0.5 * shrink) / (c + shrink) if shrink else s / c)


def _scatter(BI, BC, BV, ms, c, b, dev):
    import numpy as np
    import torch
    out = torch.zeros(c * b, dtype=torch.float32, device=dev)
    k0 = int(np.searchsorted(BI[0], ms))
    k1 = int(np.searchsorted(BI[0], ms + c))
    if k1 > k0:
        out[(BI[1][k0:k1] - ms) * b + BC[k0:k1]] = BV[k0:k1]
    return out.view(c, b)


def axis_s(args):
    import gc
    import hashlib
    import json
    import os
    import re
    import time
    from collections import defaultdict

    import numpy as np
    import pandas as pd
    import torch

    from ModelLakeFishing.scale import global_metrics as GM
    from scale1m.fast_lever_audit import _topk_update, trigrams
    from scale1m.utility_scorecard import score_source, norm_task

    t0 = time.time()
    dev = args.device
    prereg = hashlib.sha256(open(args.prereg, "rb").read()).hexdigest()

    meta = pd.read_parquet(args.model_meta)
    N_meta = len(meta)
    row_of = pd.Series(meta["mappedID"].to_numpy(), index=meta["model"].astype(str))
    sup = pd.read_parquet(args.supervision, columns=["node", "model"])
    sup["m_row"] = row_of.reindex(sup["model"].astype(str)).to_numpy()
    sup = sup.dropna(subset=["m_row"])
    sup["m_row"] = sup["m_row"].astype(int)
    sup["task"] = sup["node"].astype(str).str.split("\t").str[1].map(norm_task)
    task_models = {t: np.unique(g["m_row"].to_numpy()) for t, g in sup.groupby("task")}
    by_node = {k: set(v) for k, v in
               sup.groupby(sup["node"].astype(str))["m_row"].apply(set).items()}

    full = np.load(args.full_sidecar)
    fm = full["edge_model"].astype(np.int64)
    fd = full["edge_dataset"].astype(np.int64)
    fa = full["edge_acc"].astype(np.float64)
    root_of, task_of = full["root_id"], full["task_id"]
    root_tot, root_per_ds = _group_totals(fm, fd, fa, root_of)
    task_tot, task_per_ds = _group_totals(fm, fd, fa, task_of)
    root_members = defaultdict(list)
    for dd in np.unique(fd):
        root_members[int(root_of[dd])].append(int(dd))

    SOURCES = ["sib", "sib_nd", "task_a", "task_b"]
    VARIANTS = [("mips", []), ("A_task", ["task_a"]), ("B_sib", ["sib"]),
                ("B_sib_namedistinct", ["sib_nd"]),
                ("B_sib_taskA", ["sib", "task_a"]), ("B_task", ["task_b"]),
                ("B_sib_taskB", ["sib", "task_b"])]
    names = [v[0] for v in VARIANTS]

    per_seed = {}
    for s in SEEDS:
        d0 = run_dir(args, s)
        seed_side = np.load(os.path.join(d0, "prior_sidecar_s%d.npz" % s))
        a_tot, a_per_ds = _group_totals(
            seed_side["edge_model"].astype(np.int64),
            seed_side["edge_dataset"].astype(np.int64),
            seed_side["edge_acc"].astype(np.float64), seed_side["task_id"])

        udi = pd.read_parquet(os.path.join(d0, "dataset_ids.parquet")).sort_values("mappedID")
        node_all = udi["dataset"].astype(str).to_numpy()
        cands = load_cands(args, s)
        qids = sorted(cands)
        Q = len(qids)
        node_of = {q: node_all[q] for q in qids}
        observed = {q: by_node.get(node_of[q], set()) for q in qids}

        has_sib = np.array([len([x for x in root_members[int(root_of[q])] if x != q]) > 0
                            for q in qids])

        boosts = {k: [] for k in SOURCES}
        n_nd_dropped = []
        for q in qids:
            sibs = [x for x in root_members[int(root_of[q])] if x != q]
            assert q not in sibs
            boosts["sib"].append(_boost_minus_self(root_tot, root_per_ds,
                                                   int(root_of[q]), q, 0.0))
            gq = trigrams(node_of[q].split("\t")[0])
            nd = [x for x in sibs
                  if len(gq & trigrams(node_all[x].split("\t")[0])) /
                  max(1, len(gq | trigrams(node_all[x].split("\t")[0])))
                  <= NAME_JACCARD_MAX]
            n_nd_dropped.append(len(sibs) - len(nd))
            boosts["sib_nd"].append(_boost_from_datasets(root_per_ds, nd, 0.0))
            boosts["task_a"].append(_boost_minus_self(a_tot, a_per_ds,
                                                      int(task_of[q]), q, SHRINK_K))
            boosts["task_b"].append(_boost_minus_self(task_tot, task_per_ds,
                                                      int(task_of[q]), q, SHRINK_K))

        zm = torch.from_numpy(np.load(os.path.join(d0, "z_m_eval.npy")))
        zd = torch.from_numpy(np.load(os.path.join(d0, "z_d_eval.npy")))
        zm = (zm / (zm.norm(dim=1, keepdim=True) + 1e-12)).to(torch.float32).to(dev)
        zd = (zd / (zd.norm(dim=1, keepdim=True) + 1e-12)).to(torch.float32).to(dev)
        N = zm.size(0)
        lo_all, hi_all = _minmax_pass(zm, zd, qids, dev, args.model_chunk)
        span_all = (hi_all - lo_all).clamp_min(1e-9)

        cnt = {v: np.zeros((Q, 3), dtype=np.int64) for v in names}
        top10 = {v: np.zeros((Q, TOPK), dtype=np.int64) for v in args.top10_of}

        for qs in range(0, Q, args.query_chunk):
            blk = qids[qs:qs + args.query_chunk]
            b = len(blk)
            zq = zd[torch.as_tensor(blk, dtype=torch.long, device=dev)]
            lo, span = lo_all[qs:qs + b], span_all[qs:qs + b]

            packed = {}
            for k in SOURCES:
                bi, bc, bv = [], [], []
                for j in range(b):
                    idx, val = boosts[k][qs + j]
                    bi.append(idx)
                    bc.append(np.full(idx.size, j, dtype=np.int64))
                    bv.append(val)
                BI = np.concatenate(bi)
                o = np.argsort(BI, kind="stable")
                BI = BI[o]
                packed[k] = ((BI, torch.as_tensor(BI, device=dev)),
                             torch.as_tensor(np.concatenate(bc)[o], device=dev),
                             torch.as_tensor(np.concatenate(bv)[o].astype(np.float32),
                                             device=dev))

            probes = []
            for q in blk:
                cand, acc = cands[q]
                gold, top3, near, ids = GM._probe_ids(cand, acc, GM.GAP_DELTA)
                probes.append((gold, np.asarray(top3, np.int64),
                               np.asarray(near, np.int64), ids))
            flat = np.concatenate([p[3] for p in probes])
            fcol = np.concatenate([np.full(p[3].size, j) for j, p in enumerate(probes)])
            ft = torch.as_tensor(flat, dtype=torch.long, device=dev)
            fc = torch.as_tensor(fcol, dtype=torch.long, device=dev)
            mm_pr = (((zm[ft] * zq[fc]).sum(-1) - lo[fc]) / span[fc]).cpu().numpy()
            pr_src = {}
            for k in SOURCES:
                v = np.zeros(flat.size)
                off = 0
                for j in range(b):
                    idx, val = boosts[k][qs + j]
                    ids = probes[j][3]
                    if idx.size:
                        pos = np.clip(np.searchsorted(idx, ids), 0, idx.size - 1)
                        hit = idx[pos] == ids
                        v[off:off + ids.size] = np.where(hit, val[pos], 0.0)
                    off += ids.size
                pr_src[k] = v

            T, PID = {}, {}
            for name, srcs in VARIANTS:
                fu = mm_pr + sum(pr_src[k] for k in srcs) if srcs else mm_pr
                t_arr = np.zeros((b, 3), dtype=np.float32)
                p_arr = np.zeros((b, 3), dtype=np.int64)
                off = 0
                for j, (gold, top3, near, ids) in enumerate(probes):
                    sl = {int(m): off + i for i, m in enumerate(ids)}
                    off += ids.size
                    for slot, grp in enumerate(([gold], top3, near)):
                        best = max(grp, key=lambda m: fu[sl[int(m)]])
                        t_arr[j, slot] = fu[sl[int(best)]]
                        p_arr[j, slot] = int(best)
                T[name] = torch.as_tensor(t_arr, device=dev)
                PID[name] = p_arr

            buf = {v: (None, None) for v in top10}
            for ms in range(0, N, args.model_chunk):
                sblk = zm[ms:ms + args.model_chunk] @ zq.t()
                c = sblk.size(0)
                mm = (sblk - lo) / span
                del sblk
                B = {k: _scatter(packed[k][0], packed[k][1], packed[k][2], ms, c, b, dev)
                     for k in SOURCES}
                for name, srcs in VARIANTS:
                    fused = mm if not srcs else mm + sum(B[k] for k in srcs)
                    cm = fused.unsqueeze(2) > T[name].unsqueeze(0)
                    p = PID[name]
                    here = (p >= ms) & (p < ms + c)
                    if here.any():
                        jj, ss = np.nonzero(here)
                        cm[torch.as_tensor(p[jj, ss] - ms, device=dev),
                           torch.as_tensor(jj, device=dev),
                           torch.as_tensor(ss, device=dev)] = False
                    cnt[name][qs:qs + b] += cm.sum(0).cpu().numpy()
                    if name in top10:
                        buf[name] = _topk_update(buf[name][0], buf[name][1],
                                                 fused, ms, TOPK)
                    del cm
                    if srcs:
                        del fused
                del mm, B
            for v in top10:
                top10[v][qs:qs + b] = buf[v][1].t().cpu().numpy()
            print("    [S s%d] %d/%d queries" % (s, min(qs + b, Q), Q), flush=True)

        strata = {"all": np.ones(Q, bool), "has_sibling": has_sib,
                  "no_sibling": ~has_sib}
        rows = {}
        for name in names:
            g = cnt[name][:, 0] + 1
            t3 = cnt[name][:, 1] + 1
            gp = cnt[name][:, 2] + 1
            rows[name] = {}
            for st, m in strata.items():
                if not m.any():
                    continue
                rows[name][st] = {
                    "n_queries": int(m.sum()),
                    "gold@1": float((g[m] <= 1).mean()),
                    "gold@10": float((g[m] <= 10).mean()),
                    "gold@50": float((g[m] <= 50).mean()),
                    "gold@100": float((g[m] <= 100).mean()),
                    "top3@10": float((t3[m] <= 10).mean()),
                    "gold-gap@10": float((gp[m] <= 10).mean()),
                    "median_gold_rank": float(np.median(g[m]))}

        layer2 = []
        for v in top10:
            ranked = {q: (top10[v][i], int(cnt[v][i, 0] + 1)) for i, q in enumerate(qids)}
            row = score_source("S axis :: " + v, ranked, cands, node_of, meta,
                               observed, task_models, N_meta)
            row["rank_vs_top10_disagreements"] = sum(
                1 for i, q in enumerate(qids)
                if (int(cnt[v][i, 0] + 1) <= 10) !=
                (int(cands[q][0][int(np.argmax(cands[q][1]))]) in set(top10[v][i].tolist())))
            layer2.append(row)

        sib_sizes = [int(boosts["sib"][i][0].size) for i in range(Q)]
        per_seed[s] = {
            "n_queries": Q,
            "strata": {k: int(v.sum()) for k, v in strata.items()},
            "sibling_coverage": {
                "queries_with_sibling_models": int(np.sum(np.asarray(sib_sizes) > 0)),
                "sibling_pool_p50": float(np.percentile(sib_sizes, 50)),
                "sibling_pool_p90": float(np.percentile(sib_sizes, 90)),
                "median_siblings_dropped_by_name": float(np.median(n_nd_dropped))},
            "variants": rows, "layer2": layer2}
        print("  [S s%d] has_sibling n=%d | mips %.4f | sib %.4f | sib+taskA %.4f | "
              "taskB %.4f | sib+taskB %.4f"
              % (s, int(has_sib.sum()),
                 rows["mips"]["has_sibling"]["gold@10"],
                 rows["B_sib"]["has_sibling"]["gold@10"],
                 rows["B_sib_taskA"]["has_sibling"]["gold@10"],
                 rows["B_task"]["has_sibling"]["gold@10"],
                 rows["B_sib_taskB"]["has_sibling"]["gold@10"]), flush=True)
        del zm, zd
        gc.collect()
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()

    payload = {"seconds": round(time.time() - t0, 1), "device": str(dev),
               "prereg_sha256": prereg, "prereg": os.path.basename(args.prereg),
               "alpha": 1.0, "beta": 1.0, "shrink_k": SHRINK_K,
               "name_jaccard_max": NAME_JACCARD_MAX,
               "per_seed": per_seed,
               "interval": {n: {st: interval([per_seed[s]["variants"][n][st]["gold@10"]
                                              for s in SEEDS])
                                for st in ("all", "has_sibling", "no_sibling")}
                            for n in names}}
    merge(args.out, "s", payload, fname="X3_PROTOCOL_B.json")
    return payload


ELIGIBILITY_REASONS = ("direction_unknown", "is_placeholder", "is_rl")


def query_eligibility(nodes_path, d0):
    nodes = pd.read_parquet(nodes_path,
                            columns=["node", "gold_eligible", "primary_direction",
                                     "is_placeholder", "is_rl"])
    nodes["node"] = nodes["node"].astype(str)
    nodes = nodes.set_index("node")
    udi = pd.read_parquet(os.path.join(d0, "dataset_ids.parquet")).sort_values("mappedID")
    node_of = udi["dataset"].astype(str).to_numpy()
    elig = nodes["gold_eligible"].reindex(node_of).fillna(False).to_numpy().astype(bool)
    why = {
        "direction_unknown": nodes["primary_direction"].astype(str).eq("unknown")
        .reindex(node_of).fillna(False).to_numpy().astype(bool),
        "is_placeholder": nodes["is_placeholder"].reindex(node_of).fillna(False)
        .to_numpy().astype(bool),
        "is_rl": nodes["is_rl"].reindex(node_of).fillna(False).to_numpy().astype(bool)}
    return elig, why


def a_row(zm, zd, cands, roots, device):
    from ModelLakeFishing.scale import global_metrics as GM
    if not cands:
        return None
    agg, _ = GM.from_embeddings_streaming(
        zm, zd, cands, {d: roots[d] for d in cands}, device=device)
    n = int(np.asarray(zm).shape[0])
    return {"N": n, "gold@1": agg["gold@1"], "gold@10": agg["gold@10"],
            "top3@10": agg["top3@10"], "gold-gap@10": agg["gold-gap@10"],
            "root_gold@10": agg["root_gold@10"],
            "median_gold_rank": agg["median_gold_rank"],
            "median_rank_over_N": agg["median_gold_rank"] / max(n, 1),
            "vs_random": agg["gold@10"] / (10.0 / max(n, 1)),
            "n_queries": agg["n_queries"], "n_roots": agg["n_roots"]}


def axis_e(args):
    t0 = time.time()
    per_seed, checks = {}, []
    for s in SEEDS:
        d0 = run_dir(args, s)
        elig, why = query_eligibility(args.dataset_nodes, d0)
        cands = load_cands(args, s)
        udi = pd.read_parquet(os.path.join(d0, "dataset_ids.parquet")).sort_values("mappedID")
        roots = {int(d): str(r) for d, r in zip(udi["mappedID"], udi["root"])}
        zm = np.load(os.path.join(d0, "z_m_eval.npy"), mmap_mode="r")
        zd = np.load(os.path.join(d0, "z_d_eval.npy"))

        qids = sorted(cands)
        keep = {d: cands[d] for d in qids if elig[d]}
        drop = {d: cands[d] for d in qids if not elig[d]}
        rows = {"all_queries": a_row(zm, zd, cands, roots, args.device),
                "gold_eligible_only": a_row(zm, zd, keep, roots, args.device),
                "excluded_only": a_row(zm, zd, drop, roots, args.device)}

        ref = load_manifest(args, s)["stages"]["metrics"]["a_axis_row"]
        delta = max(abs(rows["all_queries"][k] - ref[k])
                    for k in ("gold@1", "gold@10", "top3@10", "root_gold@10"))
        checks.append({"seed": s, "max_abs_delta_vs_export_row": float(delta),
                       "ok": bool(delta < 1e-9)})

        n_drop = len(drop)
        per_seed[s] = {
            "n_queries_before": len(cands), "n_queries_after": len(keep),
            "n_excluded": n_drop,
            "share_excluded": n_drop / max(len(cands), 1),
            "excluded_by_reason": {r: int(sum(1 for d in drop if why[r][d]))
                                   for r in ELIGIBILITY_REASONS},
            "rows": rows}
        print("  [E s%d] queries %d -> %d (-%d, %.2f%%) | gold@10 %.4f -> %.4f "
              "| excluded-only %.4f"
              % (s, len(cands), len(keep), n_drop, 100 * n_drop / max(len(cands), 1),
                 rows["all_queries"]["gold@10"], rows["gold_eligible_only"]["gold@10"],
                 rows["excluded_only"]["gold@10"] if rows["excluded_only"] else float("nan")),
              flush=True)
        del zm, zd
        gc.collect()

    keys = ["gold@1", "gold@10", "top3@10", "gold-gap@10", "root_gold@10",
            "median_gold_rank", "median_rank_over_N", "vs_random", "n_queries"]
    summary = {row: {k: interval([per_seed[s]["rows"][row][k] for s in SEEDS])
                     for k in keys}
               for row in ("all_queries", "gold_eligible_only", "excluded_only")}
    payload = {"seconds": round(time.time() - t0, 1),
               "exports": args.exports, "run_fmt": args.run_fmt,
               "dataset_nodes": args.dataset_nodes,
               "per_seed": per_seed, "three_seed_interval": summary,
               "recompute_matches_export_row": checks,
               "note": "all_candidates pool only; the in_snapshot_only pool is "
                       "not recomputed under this filter"}
    merge(args.out, "e", payload, fname="X5_QUERY_ELIGIBILITY.json")
    print(json.dumps(summary["gold_eligible_only"]["gold@10"], indent=2))
    return payload


def main(argv=None):
    d = os.path.join(data_root(), "data1m")
    p = argparse.ArgumentParser(description="F8: the four axes on the RF rung")
    p.add_argument("--axis", required=True,
                   choices=["a", "b", "c", "d", "e", "p", "pinf", "s"])
    p.add_argument("--exports", default=os.path.join(d, "exports_rf"))
    p.add_argument("--run-fmt", default=RUN_FMT,
                   help="printf-style export/run directory format keyed by seed")
    p.add_argument("--graph", default=os.path.join(d, "graphs", "hgraph_rf"))
    p.add_argument("--ladder", default=os.path.join(d, "ladder_rf", "full_model_ids.parquet"))
    p.add_argument("--out", default=os.path.join(d, "metrics_rf"))
    p.add_argument("--device", default=None)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--iso-recall", type=float, default=0.99)
    p.add_argument("--reps", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--n-query", type=int, default=300)
    p.add_argument("--n-hubs", type=int, default=10)
    p.add_argument("--n-insert", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--f6-runs", default=os.path.join(
        _REPO_ROOT, "ModelLakeFishing", "docs", "1M", "F6_runs"))
    p.add_argument("--f6-run-fmt", default=RUN_FMT,
                   help="printf-style training run directory format keyed by seed")
    p.add_argument("--machine", default="local RTX 4060 Laptop, 24 logical cores")
    p.add_argument("--dataset-nodes", default=os.path.join(
        d, "rf", "canon", "dataset_nodes_merged.parquet"),
        help="canonical dataset nodes; carries gold_eligible (E axis)")
    p.add_argument("--supervision",
                   default=os.path.join(d, "rf", "canon", "supervision_merged.parquet"))
    p.add_argument("--model-meta", default=os.path.join(d, "utility_rf", "model_meta.parquet"))
    p.add_argument("--betas", nargs="+", default=["0.5", "1", "2"])
    p.add_argument("--model-chunk", type=int, default=100_000)
    p.add_argument("--query-chunk", type=int, default=128)
    p.add_argument("--top10-of", nargs="+", default=["mips", "task_b1"])
    p.add_argument("--prereg", default=os.path.join(
        _REPO_ROOT, "ModelLakeFishing", "docs", "1M", "X3_runs", "PREREGISTRATION.md"))
    p.add_argument("--full-sidecar", default=None,
                   help="S-axis seed-0 sidecar (default: selected exports/run-fmt)")
    args = p.parse_args(argv)
    if args.full_sidecar is None:
        args.full_sidecar = default_full_sidecar(args)
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.axis == "s" and args.top10_of == ["mips", "task_b1"]:
        args.top10_of = ["mips", "B_sib_taskA", "B_sib_taskB"]
    os.makedirs(args.out, exist_ok=True)
    {"a": axis_a, "b": axis_b, "c": axis_c, "d": axis_d, "e": axis_e,
     "p": axis_p, "pinf": beta_inf_limit, "s": axis_s}[args.axis](args)
    report = {"p": "X2_PRIOR_FUSION.json", "pinf": "X2_PRIOR_FUSION.json",
              "s": "X3_PROTOCOL_B.json",
              "e": "X5_QUERY_ELIGIBILITY.json"}.get(args.axis, "F8_REPORT.json")
    print("[ok] axis %s -> %s" % (args.axis, os.path.join(args.out, report)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
