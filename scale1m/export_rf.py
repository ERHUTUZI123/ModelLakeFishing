"""
export_rf.py -- F7: turn an F6 checkpoint into servable embeddings and indexes.

Runbook: docs/1M/1Mplan.md §5 (F7). Record: docs/1M/F7.md.

WHY NOT export_rung.py
    That file is T7's exporter and still reproduces the 100K rung. It reads a
    single-file `.pt` graph, hashes it with `sha256_of`, reads a CSV ladder, and
    does every stage in one process. At 3,016,439 models none of those hold: the
    graph is a `graph_store` directory, the ladder is parquet, and z_m alone is
    1.54 GB while the HNSW index is another 2.3 GB -- more than this machine has
    free if the graph is still resident. So the work is split into stages that
    each exit before the next begins, and the memory-heavy ones never load the
    graph at all.

    Everything that decides a number is unchanged: the same `build_models`, the
    same `chunked_forward`, the same `make_root_aware_splits` with the split
    seed read from the checkpoint binding, the same `five_metric_eval` and
    `global_metrics`, the same `build_hnsw` with iso-recall ef tuning.

THE TWO FORWARDS
    z_*_full : whole graph. This is what gets indexed and served; at serving
               time every edge legitimately exists.
    z_*_eval : test-split forward. A held-out query dataset does not see its own
               trained_on edges.

    Only z_*_eval may produce a reported gold@K. Scoring with the full-graph z_d
    let each held-out query see its own supervision once before and read 0.61
    instead of 0.42, which is why the leakage gate is an inequality rather than
    a note.

STAGES
    embed    load graph + checkpoint, two chunked forwards, write the four
             matrices and the gold candidate sets. Peak memory is the graph.
    metrics  read the matrices, compute A-axis rows on both forwards, run the
             leakage gate and the harness parity check. Never loads the graph.
    index    read z_m/z_d, build the full-lake HNSW, tune ef to iso-recall.
    curve    the retrieval-side scaling curve: one index per (N, sampling seed),
             each containing every supervised model plus a uniform sample of the
             unlabeled ones.

Run (from ModelLakeFishing/):
    python -m scale1m.export_rf --run <F6 run dir> --stage embed
    python -m scale1m.export_rf --run <F6 run dir> --stage metrics
    python -m scale1m.export_rf --run <F6 run dir> --stage index
    python -m scale1m.export_rf --run <F6 run dir> --stage curve
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(_HERE), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scale1m import checkpoint as CK                                # noqa: E402
from scale1m.graph_store import load_sharded                        # noqa: E402
from scale1m.hf_crawl import data_root, utcnow, write_json_atomic    # noqa: E402
from scale1m.train_rung import RUNGS                                # noqa: E402

# The retrieval-side curve of 1Mplan §3.5. The full-lake point is the index the
# `index` stage already built, so it is not rebuilt here.
CURVE_NS = (100_000, 250_000, 500_000, 1_000_000)
CURVE_SEEDS = (0, 1, 2)


# ── gates ────────────────────────────────────────────────────────────────────

def gate_pool_size(n_models, expect_n):
    """G-F7a. The silent failure this catches is scoring only the supervised
    subset and reporting that gold@10 held up under 65x the distractors."""
    ok = expect_n is None or n_models == expect_n
    return {"gate": "G-F7a", "name": "candidate pool size", "ok": bool(ok),
            "n_models": int(n_models), "expect_n": expect_n}


def gate_leakage(gold10_eval, gold10_full):
    """G-F7b. Held-out embeddings must score worse than full-graph ones."""
    ok = gold10_eval < gold10_full
    return {"gate": "G-F7b", "name": "held-out < full-graph gold@10", "ok": bool(ok),
            "gold10_eval": float(gold10_eval), "gold10_full": float(gold10_full),
            "margin": float(gold10_full - gold10_eval)}


def gate_row_order(model_ids, ladder_ids, n_probe=100, seed=0):
    """G-F7c. Row order never raises on its own: z_m row i simply stops meaning
    model i. Probe random positions, because an off-by-one in a prefix survives
    a head-only check."""
    if len(model_ids) != len(ladder_ids):
        return {"gate": "G-F7c", "name": "row order vs ladder", "ok": False,
                "reason": "length %d != ladder %d" % (len(model_ids), len(ladder_ids))}
    rng = np.random.default_rng(seed)
    probe = rng.choice(len(model_ids), size=min(n_probe, len(model_ids)), replace=False)
    bad = [int(i) for i in probe if model_ids[i] != ladder_ids[i]]
    total = int(np.sum(np.asarray(model_ids) != np.asarray(ladder_ids)))
    return {"gate": "G-F7c", "name": "row order vs ladder", "ok": total == 0,
            "n_probed": int(len(probe)), "probe_mismatches": bad[:10],
            "total_mismatches": total}


# ── shared plumbing ──────────────────────────────────────────────────────────

def read_manifest(run):
    with open(os.path.join(run, "MANIFEST.json"), encoding="utf-8") as fh:
        return json.load(fh)


def resolve_ckpt(run, which):
    if which and os.path.isfile(which):
        return which
    name = {"best": CK.BEST, "last": CK.LAST}.get(which or "last", which)
    p = os.path.join(run, "ckpt", name)
    if not os.path.isfile(p):
        raise FileNotFoundError("no checkpoint at %s" % p)
    return p


def bind_or_die(ck_path, graph_path):
    """A checkpoint records the digest of the graph it was trained on. Exporting
    against a different graph is a hard stop, not a warning: the embeddings would
    be labelled with the wrong models and nothing downstream would notice."""
    ck = CK.load(ck_path)
    binding = dict(ck["binding"])
    digest = CK.graph_digest(graph_path)
    if digest != binding.get("graph_sha256"):
        raise CK.IncompatibleCheckpoint(
            "checkpoint was trained on graph %s but %s digests to %s"
            % (binding.get("graph_sha256"), graph_path, digest))
    return ck, binding, digest


def cands_path(out):
    return os.path.join(out, "gold_cands.npz")


def save_cands(out, cands):
    np.savez_compressed(cands_path(out),
                        **{str(d): np.stack([np.asarray(c, float), np.asarray(a, float)])
                           for d, (c, a) in cands.items()})


def load_cands(out):
    with np.load(cands_path(out)) as z:
        return {int(k): (z[k][0].astype(np.int64), z[k][1].astype(float))
                for k in z.files}


def merge_stage(out, name, payload):
    """One manifest per export, written stage by stage so a crashed later stage
    does not erase what an earlier one measured."""
    p = os.path.join(out, "EXPORT_MANIFEST.json")
    man = {}
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as fh:
            man = json.load(fh)
    man.setdefault("stages", {})[name] = payload
    man["written_at"] = utcnow()
    gates = []
    for st in man["stages"].values():
        gates.extend(st.get("gates", []))
    man["gates"] = gates
    man["gates_failed"] = [g["gate"] for g in gates if g.get("ok") is False]
    man["gates_passed"] = not man["gates_failed"]
    write_json_atomic(p, man)
    return man


# ── stage: embed ─────────────────────────────────────────────────────────────

def stage_embed(args, run, out):
    from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import apply_similar_to_mode
    from ModelLakeFishing.stage2TrainGraphSAGE.d0_splits import make_root_aware_splits
    from ModelLakeFishing.stage2TrainGraphSAGE.losses import accuracy_lookup
    from ModelLakeFishing.stage2TrainGraphSAGE.ablation import build_models
    from ModelLakeFishing.stage2TrainGraphSAGE.inference import chunked_forward
    from ModelLakeFishing.stage2TrainGraphSAGE.top1_audit import candidates

    man = read_manifest(run)
    graph_path = args.graph or man["graph"]
    if not os.path.exists(graph_path):                 # remote path in the manifest
        graph_path = args.graph or os.path.join(data_root(), "data1m", "graphs", "hgraph_rf")
    rung = args.rung or man["rung"]
    expect_n = RUNGS.get(rung, {}).get("expect_n")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ck_path = resolve_ckpt(run, args.ckpt)
    ck, binding, digest = bind_or_die(ck_path, graph_path)
    print("[embed] %s ckpt=%s device=%s" % (man.get("run_id"),
                                            os.path.basename(ck_path), device), flush=True)
    t0 = time.time()

    payload = load_sharded(graph_path, mmap=True, verify_sha256=False)
    data, xm0, xd0 = payload["data"], payload["xm0_meta"], payload["xd0_meta"]
    udi = payload["unique_dataset_id"].sort_values("mappedID").reset_index(drop=True)
    umi = payload["unique_model_id"].sort_values("mappedID").reset_index(drop=True)
    root_of = udi["root"].astype(str).tolist()
    n_models = int(data["model"].num_nodes)

    cfg = dict(ck["cfg"])
    split_seed = int(binding["split_seed"])
    data = apply_similar_to_mode(data, cfg["similar_to_mode"], k=cfg["similar_to_k"])
    _tr, _val, test_data = make_root_aware_splits(data, root_of, split_seed=split_seed)
    lookup = accuracy_lookup(data)

    model, scorer = build_models(data, xm0, xd0, cfg, device=device)
    model.load_state_dict(ck["model"])      # strict: a partial load would export
    scorer.load_state_dict(ck["scorer"])    # a half-random model
    model.eval()

    with torch.no_grad():
        print("[embed] held-out forward", flush=True)
        z_eval = chunked_forward(model, test_data, chunk_size=args.chunk,
                                 device=device, progress=args.progress)
        print("[embed] full-graph forward", flush=True)
        z_full = chunked_forward(model, data, chunk_size=args.chunk,
                                 device=device, progress=args.progress)

    for name, arr in (("z_m", z_full["model"]), ("z_d", z_full["dataset"]),
                      ("z_m_eval", z_eval["model"]), ("z_d_eval", z_eval["dataset"])):
        np.save(os.path.join(out, name + ".npy"), arr.numpy().astype(np.float32))
    del z_eval, z_full
    gc.collect()

    cands = candidates(test_data, lookup)
    save_cands(out, cands)
    umi.to_parquet(os.path.join(out, "model_ids.parquet"), index=False)
    udi.to_parquet(os.path.join(out, "dataset_ids.parquet"), index=False)

    # G-F7d. The whole-graph forward this compares against needs the memory of a
    # 3M-node forward with autograd off; it is run on a 100K-model subgraph, the
    # size the plan names, with the dataset side kept whole so message passing
    # is not degenerate.
    chunk_check = {"ran": False}
    if args.verify_chunked:
        k = min(args.verify_nodes, n_models)
        sub = data.subgraph({"model": torch.arange(k),
                             "dataset": torch.arange(int(data["dataset"].num_nodes))})
        with torch.no_grad():
            whole = {t: v.cpu() for t, v in model(sub.clone().to(device)).items()}
            chunked = chunked_forward(model, sub, chunk_size=args.chunk, device=device)
        deltas = {t: float((whole[t] - chunked[t]).abs().max()) for t in whole}
        chunk_check = {"ran": True, "n_models": int(k),
                       "max_abs_delta": deltas,
                       "ok": bool(max(deltas.values()) < 1e-5)}
        del whole, chunked, sub
        gc.collect()

    ladder = pd.read_parquet(args.ladder).sort_values("mappedID")
    gates = [gate_pool_size(n_models, expect_n),
             gate_row_order(umi["model"].astype(str).tolist(),
                            ladder["model"].astype(str).tolist())]
    if chunk_check["ran"]:
        gates.append({"gate": "G-F7d", "name": "chunked == whole-graph forward",
                      **chunk_check})

    rep = {"run_id": man.get("run_id"), "checkpoint": ck_path,
           "checkpoint_epoch": int(ck.get("epoch", -1)),
           "graph": os.path.abspath(graph_path), "graph_digest": digest,
           "binding": binding, "split_seed": split_seed, "rung": rung,
           "n_models": n_models, "n_datasets": int(data["dataset"].num_nodes),
           "n_test_queries": len(cands), "expect_n": expect_n,
           "device": device, "chunk": args.chunk,
           "seconds": round(time.time() - t0, 1), "gates": gates}
    merge_stage(out, "embed", rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "binding"},
                     indent=2, ensure_ascii=False))
    return rep


# ── stage: metrics ───────────────────────────────────────────────────────────

def _z(out, name, mmap=None):
    return np.load(os.path.join(out, name + ".npy"), mmap_mode=mmap)


def stage_metrics(args, run, out):
    from ModelLakeFishing.stage2TrainGraphSAGE.top1_eval import five_metric_eval, aggregate
    from ModelLakeFishing.scale import global_metrics as GM

    man = read_manifest(run)
    with open(os.path.join(out, "EXPORT_MANIFEST.json"), encoding="utf-8") as fh:
        prior = json.load(fh)["stages"]["embed"]
    expect_n = prior.get("expect_n")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    cands = load_cands(out)
    udi = pd.read_parquet(os.path.join(out, "dataset_ids.parquet"))
    root_of = udi.sort_values("mappedID")["root"].astype(str).tolist()

    def rows(tag):
        zd = {"model": torch.from_numpy(_z(out, "z_m" + tag)),
              "dataset": torch.from_numpy(_z(out, "z_d" + tag))}
        per = five_metric_eval(zd, cands, expect_n=expect_n, device=device)
        agg = aggregate(per)
        del zd
        gc.collect()
        return agg

    agg_eval = rows("_eval")
    agg_full = rows("")
    n_models = int(prior["n_models"])

    gm_cands = {int(d): (np.asarray(c), np.asarray(a)) for d, (c, a) in cands.items()}
    roots_q = {int(d): root_of[int(d)] for d in cands}
    gm_agg, _ = GM.from_embeddings_streaming(
        _z(out, "z_m_eval"), _z(out, "z_d_eval"), gm_cands, roots_q, device=device)

    parity = {"five_metric_gold@10": agg_eval["full2k_gold@10"],
              "global_metrics_gold@10": gm_agg["gold@10"],
              "match": abs(agg_eval["full2k_gold@10"] - gm_agg["gold@10"]) < 1e-9}

    a_row = {
        "N": n_models,
        "gold@1": gm_agg["gold@1"], "gold@10": gm_agg["gold@10"],
        "top3@10": gm_agg["top3@10"], "gold-gap@10": gm_agg["gold-gap@10"],
        "root_gold@10": gm_agg["root_gold@10"],
        "median_gold_rank": gm_agg["median_gold_rank"],
        "median_rank_over_N": gm_agg["median_gold_rank"] / max(n_models, 1),
        "vs_random": gm_agg["gold@10"] / (10.0 / max(n_models, 1)),
        "n_queries": gm_agg["n_queries"], "n_roots": gm_agg["n_roots"],
    }
    snap = snapshot_only_row(out, args.ladder, cands, roots_q, device)

    gates = [gate_leakage(agg_eval["full2k_gold@10"], agg_full["full2k_gold@10"]),
             {"gate": "G-F7e", "name": "five_metric == global_metrics",
              "ok": bool(parity["match"]), **parity},
             {"gate": "G-F7h", "name": "A axis reported over both candidate pools",
              "ok": bool(snap and snap.get("snapshot_is_an_exact_prefix")),
              "pools": ["all_candidates", "in_snapshot_only"]}]

    rep = {"run_id": man.get("run_id"), "device": device,
           "seconds": round(time.time() - t0, 1),
           "eval_metrics": agg_eval, "full_graph_metrics": agg_full,
           "global_metrics": gm_agg, "a_axis_row": a_row,
           "a_axis_row_in_snapshot_only": snap, "gates": gates}
    merge_stage(out, "metrics", rep)
    print(json.dumps(rep, indent=2, ensure_ascii=False, default=float))
    return rep


def snapshot_only_row(out, ladder_path, cands, roots_q, device):
    """The second A-axis pool that `rf-gold-2.0` requires.

    D-63 put 12,680 models into the lake that no longer exist on HF, so every
    A-axis number has to be reported over all candidates and over surviving
    candidates only. The ladder keeps the snapshot as an exact prefix (F3), so
    restricting the pool is a truncation of z_m -- but that also removes some
    queries' gold model, and a query whose answer is not in the pool is not a
    harder query, it is a different one. Those queries are dropped, and the
    count is reported so the two rows are read as different query sets.
    """
    from ModelLakeFishing.scale import global_metrics as GM

    ladder = pd.read_parquet(ladder_path, columns=["mappedID", "in_snapshot"])
    ladder = ladder.sort_values("mappedID")
    flag = ladder["in_snapshot"].to_numpy().astype(bool)
    n_snap = int(flag.sum())
    prefix_ok = bool(flag[:n_snap].all() and not flag[n_snap:].any())
    if not prefix_ok:
        return {"snapshot_is_an_exact_prefix": False, "n_snapshot": n_snap}

    # the same eligibility rule the candidate set was built with, re-applied to
    # the restricted pool: at least 3 surviving candidates and a non-constant
    # accuracy among them
    kept, dropped, lost_gold = {}, 0, 0
    for d, (c, a) in cands.items():
        c = np.asarray(c)
        a = np.asarray(a, float)
        m = c < n_snap
        if not bool(m[int(np.argmax(a))]):
            lost_gold += 1
        if m.sum() < 3 or float(np.std(a[m])) <= 0:
            dropped += 1
            continue
        kept[int(d)] = (c[m], a[m])
    if not kept:
        return {"snapshot_is_an_exact_prefix": True, "n_snapshot": n_snap,
                "n_queries": 0, "queries_dropped_ineligible_in_pool": dropped}

    zm = _z(out, "z_m_eval", mmap="r")[:n_snap]
    agg, _ = GM.from_embeddings_streaming(
        zm, _z(out, "z_d_eval"), kept, {d: roots_q[d] for d in kept}, device=device)
    del zm
    gc.collect()
    return {"snapshot_is_an_exact_prefix": True, "N": n_snap,
            "queries_dropped_ineligible_in_pool": dropped,
            "queries_whose_best_model_left_the_pool": lost_gold,
            "gold@1": agg["gold@1"], "gold@10": agg["gold@10"],
            "top3@10": agg["top3@10"], "gold-gap@10": agg["gold-gap@10"],
            "root_gold@10": agg["root_gold@10"],
            "median_gold_rank": agg["median_gold_rank"],
            "median_rank_over_N": agg["median_gold_rank"] / n_snap,
            "vs_random": agg["gold@10"] / (10.0 / n_snap),
            "n_queries": agg["n_queries"], "n_roots": agg["n_roots"]}


# ── stage: index ─────────────────────────────────────────────────────────────

def stage_index(args, run, out):
    from ModelLakeFishing.scale.export_ours import build_hnsw

    man = read_manifest(run)
    t0 = time.time()
    cands = load_cands(out)
    # memory-mapped: build_hnsw makes its own normalized copy, and holding a
    # second resident 1.54 GB array alongside the index is what does not fit here
    z_m, z_d = _z(out, "z_m", mmap="r"), _z(out, "z_d")
    idx, hnsw = build_hnsw(z_m, z_d, cands, ef=args.ef_construction, M=args.hnsw_M,
                           threads=args.hnsw_threads, iso_recall=args.iso_recall)
    path = os.path.join(out, "hnsw_full.bin")
    idx.save_index(path)
    if hnsw.get("ef_trace"):
        write_json_atomic(os.path.join(out, "ef_tuning.json"),
                          {"target": hnsw["iso_recall"], "ef_search": hnsw["ef_search"],
                           "trace": hnsw["ef_trace"]})
    gates = [{"gate": "G-F7f", "name": "HNSW recall@50 >= %.2f" % args.iso_recall,
              "ok": bool(hnsw["recall_at_50"] >= args.iso_recall),
              "recall_at_50": hnsw["recall_at_50"], "ef_search": hnsw["ef_search"]}]
    rep = {"run_id": man.get("run_id"), "index": path,
           "index_bytes": os.path.getsize(path),
           "seconds": round(time.time() - t0, 1), "hnsw": hnsw, "gates": gates}
    merge_stage(out, "index", rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "hnsw"}, indent=2))
    print(json.dumps({k: v for k, v in rep["hnsw"].items() if k != "ef_trace"}, indent=2))
    return rep


# ── stage: curve ─────────────────────────────────────────────────────────────

def supervised_rows(out, sup_path):
    """mappedIDs of every model carrying a supervision edge.

    §3.5 requires each subsampled universe to contain all of them: the gold for
    every query is a supervised model, so a sample that dropped one would remove
    the query's answer from the pool and measure something else."""
    ids = pd.read_parquet(os.path.join(out, "model_ids.parquet"))
    ids = ids.sort_values("mappedID")
    pos = pd.Series(ids["mappedID"].to_numpy(), index=ids["model"].astype(str))
    sup = pd.read_parquet(sup_path, columns=["model"])["model"].astype(str).unique()
    hit = pos.reindex(sup).dropna().astype(np.int64).to_numpy()
    return np.sort(hit)


def stage_curve(args, run, out):
    import hnswlib
    from ModelLakeFishing.scale.export_ours import tune_ef_for_recall

    man = read_manifest(run)
    cands = load_cands(out)
    z_m, z_d = _z(out, "z_m", mmap="r"), _z(out, "z_d")
    N = z_m.shape[0]
    zm = z_m / (np.linalg.norm(z_m, axis=1, keepdims=True) + 1e-12)
    zd = z_d / (np.linalg.norm(z_d, axis=1, keepdims=True) + 1e-12)
    zm = zm.astype(np.float32)
    del z_m, z_d
    gc.collect()

    keep = supervised_rows(out, args.supervision)
    gold_needed = np.unique(np.concatenate([np.asarray(c, np.int64)
                                            for c, _a in cands.values()]))
    keep = np.union1d(keep, gold_needed)
    rest = np.setdiff1d(np.arange(N, dtype=np.int64), keep, assume_unique=False)
    print("[curve] supervised+gold rows kept in every universe: %d" % len(keep), flush=True)

    qd = list(cands)[: min(args.curve_queries, len(cands))]
    sub_dir = os.path.join(out, "curve")
    os.makedirs(sub_dir, exist_ok=True)

    rows, gates = [], []
    for n_target in CURVE_NS:
        n = N if n_target is None else int(n_target)
        if n > N:
            continue
        for s in CURVE_SEEDS:
            t0 = time.time()
            if n < len(keep):
                raise SystemExit("N=%d is smaller than the %d rows that must be "
                                 "kept; drop this point from the curve" % (n, len(keep)))
            rng = np.random.default_rng(1000 + s)
            extra = rng.choice(rest, size=n - len(keep), replace=False)
            sel = np.sort(np.concatenate([keep, extra]))
            idx = hnswlib.Index(space="ip", dim=zm.shape[1])
            idx.init_index(max_elements=len(sel), ef_construction=args.ef_construction,
                           M=args.hnsw_M)
            if args.hnsw_threads:
                idx.set_num_threads(int(args.hnsw_threads))
            tb = time.perf_counter_ns()
            idx.add_items(zm[sel], sel)          # label == original mappedID
            build_ms = (time.perf_counter_ns() - tb) / 1e6

            # iso-recall against brute force over the SAME universe
            sub_zm = zm[sel]
            brute = {d: set(sel[np.argpartition(-(sub_zm @ zd[int(d)]), 50)[:50]].tolist())
                     for d in qd}
            ef, rec, trace = tune_ef_for_recall(idx, sub_zm, zd, qd, target=args.iso_recall,
                                                brute_cache=brute)
            name = "hnsw_sub_%dk_s%d.bin" % (n // 1000, s)
            idx.save_index(os.path.join(sub_dir, name))
            row = {"N": int(len(sel)), "sample_seed": int(s), "index": name,
                   "ef_search": int(ef), "recall_at_50": float(rec),
                   "build_ms": round(build_ms, 1),
                   "index_bytes": os.path.getsize(os.path.join(sub_dir, name)),
                   "n_query": len(qd), "seconds": round(time.time() - t0, 1)}
            rows.append(row)
            gates.append({"gate": "G-F7g", "name": "sub-index %s holds its N and all gold" % name,
                          "ok": bool(len(sel) == n and np.isin(gold_needed, sel).all()
                                     and rec >= args.iso_recall),
                          "N": int(len(sel)), "expect_N": int(n),
                          "recall_at_50": float(rec)})
            print("  [curve] N=%-9d seed=%d ef=%-4d recall=%.4f  %.1fs"
                  % (len(sel), s, ef, rec, row["seconds"]), flush=True)
            del idx, sub_zm, brute
            gc.collect()
            if trace:
                row["ef_trace"] = trace

    rep = {"run_id": man.get("run_id"), "rows": rows, "gates": gates,
           "always_kept_rows": int(len(keep))}
    merge_stage(out, "curve", rep)
    write_json_atomic(os.path.join(out, "scaling_curve_retrieval.json"), rep)
    return rep


# ── cli ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    d = os.path.join(data_root(), "data1m")
    p = argparse.ArgumentParser(description="F7: export and index the RF lake")
    p.add_argument("--run", required=True, help="an F6 run directory")
    p.add_argument("--stage", required=True,
                   choices=["embed", "metrics", "index", "curve"])
    p.add_argument("--graph", default=os.path.join(d, "graphs", "hgraph_rf"))
    p.add_argument("--ladder", default=os.path.join(d, "ladder_rf", "full_model_ids.parquet"))
    p.add_argument("--supervision",
                   default=os.path.join(d, "rf", "canon", "supervision_merged.parquet"))
    p.add_argument("--rung", default=None)
    p.add_argument("--ckpt", default="last")
    p.add_argument("--out", default=None, help="default: <run>/exports")
    p.add_argument("--chunk", type=int, default=50_000)
    p.add_argument("--device", default=None)
    p.add_argument("--progress", action="store_true")
    p.add_argument("--hnsw-M", type=int, default=32)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--hnsw-threads", type=int, default=None)
    p.add_argument("--iso-recall", type=float, default=0.99)
    p.add_argument("--curve-queries", type=int, default=300)
    p.add_argument("--verify-chunked", dest="verify_chunked", action="store_true",
                   default=True)
    p.add_argument("--no-verify-chunked", dest="verify_chunked", action="store_false")
    p.add_argument("--verify-nodes", type=int, default=100_000)
    args = p.parse_args(argv)

    run = os.path.abspath(args.run)
    out = args.out or os.path.join(run, "exports")
    os.makedirs(out, exist_ok=True)
    rep = {"embed": stage_embed, "metrics": stage_metrics,
           "index": stage_index, "curve": stage_curve}[args.stage](args, run, out)
    failed = [g["gate"] for g in rep.get("gates", []) if g.get("ok") is False]
    if failed:
        print("[FAIL] " + ", ".join(sorted(set(failed))))
        return 1
    print("[ok] stage %s -> %s" % (args.stage, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
