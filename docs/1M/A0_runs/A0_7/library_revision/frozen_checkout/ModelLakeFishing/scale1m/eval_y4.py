"""Y4: candidate-pool size curve for the frozen Y2 two-stage system.

The experiment is specified in ``docs/1M/Y4.md``.  It changes only the
first-stage pool size: X4-GD embeddings, the task prior, splits, scoring and
tie-breaking are inherited from Y2.
"""
import argparse
import gc
import json
import math
import os
import platform
import shutil
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_PARENT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_PARENT not in sys.path:
    sys.path.insert(0, _REPO_PARENT)

from ModelLakeFishing.scale1m.eval_y2 import (
    EXPECTED_DENSE,
    EXPECTED_QUERIES,
    N_TOTAL,
    SEEDS,
    _audit_seed,
    _pool_metrics,
    _sha256,
    _tie_ranks,
    _topk_update,
)
from ModelLakeFishing.scale1m.hf_crawl import data_root, utcnow, write_json_atomic


KS = (1_000, 2_000, 5_000, 10_000)
MAX_K = max(KS)
EXPECTED_Y2 = {
    "G_full_task": (0.3407859078590786, 0.3651226158038147,
                    0.2588996763754045),
    "G_exact1000_task": (0.32859078590785906, 0.3387829246139873,
                         0.24271844660194175),
    "G_hnsw1000_task": (0.32791327913279134, 0.3387829246139873,
                        0.24271844660194175),
}
EXPECTED_Y4_POOL_SHA256 = {
    0: "ac5b89e2f8bc8e3ad061b095a9df796b3ee81c1527f0d39d29d17b5b280ee9ee",
    1: "f7f0f621066917896bb97524b0b62a26c5fdd30ea1a32342aa541f7cbc5ccbd1",
    2: "dfb2b41302eb27d08e1ed2a4bfeb4e9e13d7ab6060e102f9a7ced38aba376a29",
}


def _load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _metric_equal(got, expected, label, atol=1e-12):
    if not math.isclose(float(got), float(expected), rel_tol=0.0, abs_tol=atol):
        raise AssertionError("%s %.15f != %.15f" % (label, got, expected))


def _ef_candidates(k):
    return tuple(dict.fromkeys((k, int(math.ceil(1.5 * k)), 2 * k, 3 * k, 5 * k)))


def _validate_nested(ids, scores, ks=KS):
    ids = np.asarray(ids)
    scores = np.asarray(scores)
    if ids.ndim != 2 or ids.shape != scores.shape or ids.shape[1] != max(ks):
        raise AssertionError("exact pool must be [queries, max(K)] with matching scores")
    if np.any(ids < 0) or np.any(ids >= N_TOTAL):
        raise AssertionError("exact pool has out-of-range mappedID")
    if np.any(scores[:, :-1] < scores[:, 1:] - 1e-7):
        raise AssertionError("exact pool scores are not descending")
    for row in ids:
        if len(np.unique(row)) != len(row):
            raise AssertionError("exact top-%d contains duplicate mappedID" % max(ks))
    # Smaller pools are represented only as prefixes of this one array.  Keep
    # this explicit in the returned views so a caller cannot silently rescan.
    return {k: (ids[:, :k], scores[:, :k]) for k in ks}


def _pool_eval(ids, scores, queries, candidates, roots, prior, tie_rank, k):
    row, per_query, top10 = _pool_metrics(
        ids, scores, queries, candidates, roots, prior, tie_rank, k=k)
    old_key = "gold_in_first_stage@1000"
    row["gold_in_first_stage@%d" % k] = row.pop(old_key)
    return row, per_query, top10


def _paired(base_per, new_per):
    if set(base_per) != set(new_per):
        raise AssertionError("paired query sets differ")
    rescue = harm = both_hit = both_miss = 0
    for query in sorted(base_per):
        base = int(base_per[query]["gold_rank"]) <= 10
        new = int(new_per[query]["gold_rank"]) <= 10
        if not base and new:
            rescue += 1
        elif base and not new:
            harm += 1
        elif base and new:
            both_hit += 1
        else:
            both_miss += 1
    n = len(base_per)
    base_hits = rescue * 0 + harm + both_hit
    new_hits = rescue + both_hit
    if new_hits - base_hits != rescue - harm:
        raise AssertionError("paired rescue/harm identity failed")
    return {
        "n_queries": n,
        "rescue": rescue,
        "harm": harm,
        "net": rescue - harm,
        "both_hit": both_hit,
        "both_miss": both_miss,
        "rescue_rate": rescue / n,
        "harm_rate": harm / n,
        "net_rate": (rescue - harm) / n,
    }


def _mean_intersection_recall(actual, expected, k):
    if actual.shape != expected.shape or actual.shape[1] != k:
        raise AssertionError("recall arrays have incompatible shapes")
    recalls = np.empty(actual.shape[0], dtype=np.float64)
    for i in range(actual.shape[0]):
        if len(np.unique(actual[i])) != k:
            raise AssertionError("HNSW top-%d contains duplicate mappedID" % k)
        recalls[i] = len(set(actual[i].tolist()).intersection(expected[i].tolist())) / k
    return float(recalls.mean())


def _exact_topmax(export_dir, queries, device, model_chunk, query_chunk):
    import torch

    started = time.time()
    zm_np = np.load(os.path.join(export_dir, "z_m_eval.npy"), mmap_mode="r")
    zd_np = np.load(os.path.join(export_dir, "z_d_eval.npy"), mmap_mode="r")
    if zm_np.shape[0] != N_TOTAL:
        raise AssertionError("candidate pool is %d, expected %d" %
                             (zm_np.shape[0], N_TOTAL))
    zm = torch.as_tensor(np.asarray(zm_np), dtype=torch.float32)
    zd = torch.as_tensor(np.asarray(zd_np), dtype=torch.float32)
    zm = (zm / zm.norm(dim=1, keepdim=True).clamp_min(1e-12)).to(device)
    zd = (zd / zd.norm(dim=1, keepdim=True).clamp_min(1e-12)).to(device)
    ids = np.empty((len(queries), MAX_K), dtype=np.int32)
    scores = np.empty((len(queries), MAX_K), dtype=np.float32)
    for qs in range(0, len(queries), query_chunk):
        block = queries[qs:qs + query_chunk]
        zq = zd[torch.as_tensor(block, dtype=torch.long, device=device)]
        buf = (None, None)
        for ms in range(0, N_TOTAL, model_chunk):
            raw = zm[ms:ms + model_chunk] @ zq.t()
            buf = _topk_update(*buf, raw, ms, MAX_K)
            del raw
        b = len(block)
        ids[qs:qs + b] = buf[1].t().cpu().numpy().astype(np.int32)
        scores[qs:qs + b] = buf[0].t().cpu().numpy().astype(np.float32)
        print("[Y4 exact] %d/%d queries %.1fs" %
              (min(qs + b, len(queries)), len(queries), time.time() - started),
              flush=True)
    return ids, scores, time.time() - started


def _base_report(args, y2):
    return {
        "written_at": utcnow(),
        "stage": "initialized",
        "protocol": {
            "N": N_TOTAL,
            "K": list(KS),
            "max_K": MAX_K,
            "score": "(cos+1)/2 + task_prior",
            "shrink_k": 5.0,
            "beta": 1.0,
            "eligible_queries": EXPECTED_QUERIES,
            "hnsw_threads_build": args.hnsw_threads,
            "hnsw_threads_latency": 1,
        },
        "y2_reference": {
            "report": os.path.abspath(args.y2_report),
            "G_full_task": [float(y2["per_seed"][str(s)]["rows"]
                                  ["G_full_task"]["gold@10"]) for s in SEEDS],
            "G_exact1000_task": [float(y2["per_seed"][str(s)]["rows"]
                                       ["G_exact1000_task"]["gold@10"])
                                   for s in SEEDS],
            "G_hnsw1000_task": [float(y2["hnsw"][str(s)]["rows"]
                                      ["G_hnsw1000_task"]["gold@10"])
                                  for s in SEEDS],
        },
        "exact": {},
        "hnsw": {},
        "gates": {},
    }


def _audit_y2(y2):
    if y2.get("stage") != "complete" or not all(y2.get("gates", {}).values()):
        raise AssertionError("Y2 report is not complete or has a failed gate")
    for seed in SEEDS:
        rows = y2["per_seed"][str(seed)]["rows"]
        _metric_equal(rows["G_dense"]["gold@10"], EXPECTED_DENSE[seed],
                      "Y2 G_dense seed %d" % seed)
        for name in ("G_full_task", "G_exact1000_task"):
            _metric_equal(rows[name]["gold@10"], EXPECTED_Y2[name][seed],
                          "Y2 %s seed %d" % (name, seed))
        got = y2["hnsw"][str(seed)]["rows"]["G_hnsw1000_task"]["gold@10"]
        _metric_equal(got, EXPECTED_Y2["G_hnsw1000_task"][seed],
                      "Y2 G_hnsw1000_task seed %d" % seed)


def run_exact(args):
    os.makedirs(args.out, exist_ok=True)
    y2 = _load_json(args.y2_report)
    _audit_y2(y2)
    report_path = os.path.join(args.out, "Y4_REPORT.json")
    report = _base_report(args, y2)
    tie_rank = _tie_ranks(N_TOTAL)
    first_model = first_dataset = None
    for seed in SEEDS:
        bundle = _audit_seed(args, seed, first_model, first_dataset, tie_rank)
        if first_model is None:
            first_model, first_dataset = bundle["models"], bundle["datasets"]
        queries = sorted(bundle["candidates"])
        if len(queries) != EXPECTED_QUERIES[seed]:
            raise AssertionError("eligible query count changed")
        ids, scores, seconds = _exact_topmax(
            bundle["x4"], queries, args.device, args.model_chunk, args.query_chunk)
        prefixes = _validate_nested(ids, scores)

        with np.load(os.path.join(args.y2_out, "exact_pool_s%d.npz" % seed)) as old:
            if not np.array_equal(old["query"].astype(np.int64), np.asarray(queries)):
                raise AssertionError("Y2/Y4 query order differs for seed %d" % seed)
            if not np.array_equal(old["model"].astype(np.int64), prefixes[1000][0]):
                raise AssertionError("Y4 exact top-1000 does not reproduce Y2")
            if not np.allclose(old["score"], prefixes[1000][1], rtol=0, atol=1e-6):
                raise AssertionError("Y4 exact top-1000 scores do not reproduce Y2")

        rows, paired, hits = {}, {}, {}
        base_per = None
        for k in KS:
            row, per, _top10 = _pool_eval(
                *prefixes[k], queries, bundle["candidates"], bundle["roots"],
                bundle["prior"], tie_rank, k)
            name = "G_exact%d_task" % k
            rows[name] = row
            hits[str(k)] = np.asarray(
                [int(per[q]["gold_rank"]) <= 10 for q in queries], dtype=np.bool_)
            if k == 1000:
                base_per = per
                _metric_equal(row["gold@10"], EXPECTED_Y2["G_exact1000_task"][seed],
                              "Y4 exact K=1000 seed %d" % seed)
            else:
                paired[str(k)] = _paired(base_per, per)

        artifact = os.path.join(args.out, "exact_pool_top10000_s%d.npz" % seed)
        np.savez(artifact, query=np.asarray(queries, dtype=np.int64), model=ids,
                 score=scores, **{"hit@10_K%d" % k: hits[str(k)] for k in KS})
        report["exact"][str(seed)] = {
            "rows": rows,
            "paired_vs_K1000": paired,
            "seconds": seconds,
            "artifact": artifact,
        }
        report["stage"] = "exact-partial"
        report["written_at"] = utcnow()
        write_json_atomic(report_path, report)
        del bundle, ids, scores, prefixes
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    report["stage"] = "exact-complete"
    report["gates"]["y2_reproduced_exact_K1000"] = True
    report["written_at"] = utcnow()
    write_json_atomic(report_path, report)
    return report


def _materialize_frozen_pool(source, target, expected_sha256):
    """Expose a hash-checked release artifact at the normal stage location."""
    source = os.path.abspath(source)
    target = os.path.abspath(target)
    source_hash = _sha256(source)
    if source_hash != expected_sha256:
        raise AssertionError("frozen Y4 pool sha256 %s != %s" %
                             (source_hash, expected_sha256))
    if source == target:
        return source_hash
    if os.path.exists(target):
        if (_sha256(target) == expected_sha256 and
                os.path.getsize(target) == os.path.getsize(source)):
            return source_hash
        raise FileExistsError("nonmatching replay pool already exists: %s" % target)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    if _sha256(target) != expected_sha256:
        raise AssertionError("materialized Y4 pool failed SHA-256 verification")
    return source_hash


def run_replay_exact(args):
    """Recompute the exact-pool curve from frozen, hash-bound top-10K pools.

    The expensive 3M-vector scan produced these pools.  This stage does not
    trust an archived metric report: it re-applies the current fixed fusion,
    tie-break, gold definitions, root macro aggregation, and paired tests to
    every query.  The release manifest authenticates the pool arrays.
    """
    if not args.frozen_pools:
        raise SystemExit("--frozen-pools is required for replay-exact")
    os.makedirs(args.out, exist_ok=True)
    y2 = _load_json(args.y2_report)
    _audit_y2(y2)
    report_path = os.path.join(args.out, "Y4_REPORT.json")
    report = _base_report(args, y2)
    report["protocol"]["exact_pool_source"] = "frozen hash-bound top-10000 arrays"
    tie_rank = _tie_ranks(N_TOTAL)
    first_model = first_dataset = None
    for seed in SEEDS:
        started = time.time()
        bundle = _audit_seed(args, seed, first_model, first_dataset, tie_rank)
        if first_model is None:
            first_model, first_dataset = bundle["models"], bundle["datasets"]
        queries = sorted(bundle["candidates"])
        source = os.path.join(args.frozen_pools,
                              "exact_pool_top10000_s%d.npz" % seed)
        if not os.path.isfile(source):
            raise FileNotFoundError(source)
        source_hash = _sha256(source)
        if source_hash != EXPECTED_Y4_POOL_SHA256[seed]:
            raise AssertionError("seed %d frozen Y4 pool has SHA-256 %s, expected %s" %
                                 (seed, source_hash,
                                  EXPECTED_Y4_POOL_SHA256[seed]))
        with np.load(source) as payload:
            frozen_queries = payload["query"].astype(np.int64)
            ids = payload["model"].astype(np.int64)
            scores = payload["score"].astype(np.float32)
        if not np.array_equal(frozen_queries, np.asarray(queries, dtype=np.int64)):
            raise AssertionError("frozen Y4 query order differs at seed %d" % seed)
        prefixes = _validate_nested(ids, scores)

        with np.load(os.path.join(args.y2_out, "exact_pool_s%d.npz" % seed)) as old:
            if not np.array_equal(old["query"].astype(np.int64), frozen_queries):
                raise AssertionError("Y2/Y4 query order differs for seed %d" % seed)
            if not np.array_equal(old["model"].astype(np.int64), prefixes[1000][0]):
                raise AssertionError("frozen Y4 top-1000 does not reproduce Y2")
            if not np.allclose(old["score"], prefixes[1000][1], rtol=0, atol=1e-6):
                raise AssertionError("frozen Y4 top-1000 scores do not reproduce Y2")

        rows, paired, hits = {}, {}, {}
        base_per = None
        for k in KS:
            row, per, _top10 = _pool_eval(
                *prefixes[k], queries, bundle["candidates"], bundle["roots"],
                bundle["prior"], tie_rank, k)
            name = "G_exact%d_task" % k
            rows[name] = row
            hits[str(k)] = np.asarray(
                [int(per[q]["gold_rank"]) <= 10 for q in queries], dtype=np.bool_)
            if k == 1000:
                base_per = per
                _metric_equal(row["gold@10"], EXPECTED_Y2["G_exact1000_task"][seed],
                              "Y4 replay exact K=1000 seed %d" % seed)
            else:
                paired[str(k)] = _paired(base_per, per)

        target = os.path.join(args.out, "exact_pool_top10000_s%d.npz" % seed)
        _materialize_frozen_pool(source, target, EXPECTED_Y4_POOL_SHA256[seed])
        report["exact"][str(seed)] = {
            "rows": rows,
            "paired_vs_K1000": paired,
            "seconds": time.time() - started,
            "artifact": target,
            "frozen_source": os.path.abspath(source),
            "frozen_sha256": source_hash,
            "recomputed_from_arrays": True,
        }
        report["stage"] = "exact-partial"
        report["written_at"] = utcnow()
        write_json_atomic(report_path, report)
        del bundle, ids, scores, prefixes
        gc.collect()

    report["stage"] = "exact-complete"
    report["gates"]["y2_reproduced_exact_K1000"] = True
    report["gates"]["frozen_exact_pools_recomputed"] = True
    report["written_at"] = utcnow()
    write_json_atomic(report_path, report)
    return report


def run_hnsw(args):
    import hnswlib

    report_path = os.path.join(args.out, "Y4_REPORT.json")
    report = _load_json(report_path)
    if report.get("stage") not in ("exact-complete", "hnsw-partial", "hnsw-complete"):
        raise SystemExit("Y4 exact stage is incomplete")
    tie_rank = _tie_ranks(N_TOTAL)
    first_model = first_dataset = None
    all_recall_pass = True
    for seed in SEEDS:
        bundle = _audit_seed(args, seed, first_model, first_dataset, tie_rank)
        if first_model is None:
            first_model, first_dataset = bundle["models"], bundle["datasets"]
        payload = np.load(os.path.join(args.out, "exact_pool_top10000_s%d.npz" % seed))
        queries = payload["query"].astype(np.int64)
        exact_all = payload["model"].astype(np.int64)
        zq_all = np.load(os.path.join(bundle["x4"], "z_d_eval.npy"), mmap_mode="r")
        zq = np.asarray(zq_all[queries], dtype=np.float32)
        norms = np.linalg.norm(zq, axis=1, keepdims=True)
        zq = zq / np.maximum(norms, 1e-12)
        index_path = os.path.join(bundle["x4"], "hnsw_y2_eval.bin")
        if not os.path.isfile(index_path):
            raise AssertionError("missing Y2 HNSW index: %s" % index_path)
        index = hnswlib.Index(space="ip", dim=zq.shape[1])
        index.load_index(index_path, max_elements=N_TOTAL)
        index.set_num_threads(args.hnsw_threads)

        seed_rows, seed_pairs, seed_trace, seed_latency = {}, {}, {}, {}
        base_per = None
        base_gold = None
        for k in KS:
            chosen = None
            trace = []
            for ef in _ef_candidates(k):
                index.set_ef(max(ef, k))
                ids, dist = index.knn_query(zq, k=k, num_threads=args.hnsw_threads)
                ids = ids.astype(np.int64)
                recall = _mean_intersection_recall(ids, exact_all[:, :k], k)
                trace.append({"ef_search": ef, "recall@K": recall})
                print("[Y4 hnsw] seed %d K=%d ef=%d recall=%.6f" %
                      (seed, k, ef, recall), flush=True)
                if recall >= 0.99:
                    chosen = (ef, ids, (1.0 - dist).astype(np.float32), recall)
                    break
            seed_trace[str(k)] = trace
            if chosen is None:
                all_recall_pass = False
                continue
            ef, ids, scores, recall = chosen

            index.set_ef(ef)
            for i in range(min(50, len(queries))):
                index.knn_query(zq[i:i + 1], k=k, num_threads=1)
            query_ms, rerank_ms = [], []
            for i, query in enumerate(queries):
                t0 = time.perf_counter_ns()
                one_id, one_dist = index.knn_query(zq[i:i + 1], k=k, num_threads=1)
                t1 = time.perf_counter_ns()
                one_id = one_id[0]
                one_score = 1.0 - one_dist[0]
                fused = ((one_score + 1.0) * 0.5
                         + bundle["prior"].values(int(query), one_id))
                np.lexsort((tie_rank[one_id], -fused))[:10]
                t2 = time.perf_counter_ns()
                query_ms.append((t1 - t0) / 1e6)
                rerank_ms.append((t2 - t1) / 1e6)

            row, per, _top10 = _pool_eval(
                ids, scores, queries.tolist(), bundle["candidates"], bundle["roots"],
                bundle["prior"], tie_rank, k)
            row["recall@%d" % k] = recall
            name = "G_hnsw%d_task" % k
            seed_rows[name] = row
            total_ms = np.asarray(query_ms) + np.asarray(rerank_ms)
            seed_latency[str(k)] = {
                "hnsw_p50": float(np.percentile(query_ms, 50)),
                "hnsw_p95": float(np.percentile(query_ms, 95)),
                "rerank_p50": float(np.percentile(rerank_ms, 50)),
                "rerank_p95": float(np.percentile(rerank_ms, 95)),
                "end_to_end_p50": float(np.percentile(total_ms, 50)),
                "end_to_end_p95": float(np.percentile(total_ms, 95)),
            }
            if k == 1000:
                base_per = per
                base_gold = row["gold@10"]
                _metric_equal(base_gold, EXPECTED_Y2["G_hnsw1000_task"][seed],
                              "Y4 HNSW K=1000 seed %d" % seed)
            else:
                seed_pairs[str(k)] = _paired(base_per, per)
        payload.close()
        report["hnsw"][str(seed)] = {
            "rows": seed_rows,
            "paired_vs_K1000": seed_pairs,
            "ef_trace": seed_trace,
            "latency_ms": seed_latency,
            "index": index_path,
            "index_bytes": os.path.getsize(index_path),
        }
        report["stage"] = "hnsw-partial"
        report["written_at"] = utcnow()
        write_json_atomic(report_path, report)
        del index, bundle, zq, zq_all
        gc.collect()

    report["gates"]["all_hnsw_recall_at_K_ge_0.99"] = all_recall_pass
    report["gates"]["y2_reproduced_hnsw_K1000"] = all(
        "G_hnsw1000_task" in report["hnsw"][str(s)]["rows"] for s in SEEDS)
    report["stage"] = "hnsw-complete" if all_recall_pass else "hnsw-recall-failed"
    report["written_at"] = utcnow()
    write_json_atomic(report_path, report)
    return report


def _curve_summary(report):
    full = np.asarray(report["y2_reference"]["G_full_task"], dtype=float)
    exact_base = np.asarray(report["y2_reference"]["G_exact1000_task"], dtype=float)
    hnsw_base = np.asarray(report["y2_reference"]["G_hnsw1000_task"], dtype=float)
    gaps = full - exact_base
    if np.any(gaps <= 0):
        raise AssertionError("Y2 exact K=1000 has no positive truncation gap")
    curve = {}
    for k in KS:
        exact = np.asarray([report["exact"][str(s)]["rows"]
                            ["G_exact%d_task" % k]["gold@10"] for s in SEEDS])
        hnsw = np.asarray([report["hnsw"][str(s)]["rows"]
                           ["G_hnsw%d_task" % k]["gold@10"] for s in SEEDS])
        gap_closed = (exact - exact_base) / gaps
        exact_gain = exact - exact_base
        hnsw_gain = hnsw - hnsw_base
        gain_retention = np.divide(
            hnsw_gain, exact_gain, out=np.full(3, np.nan), where=exact_gain > 0)
        p50 = np.asarray([report["hnsw"][str(s)]["latency_ms"][str(k)]
                          ["end_to_end_p50"] for s in SEEDS])
        p95 = np.asarray([report["hnsw"][str(s)]["latency_ms"][str(k)]
                          ["end_to_end_p95"] for s in SEEDS])
        curve[str(k)] = {
            "exact_gold@10": {"per_seed": exact.tolist(), "mean": float(exact.mean())},
            "hnsw_gold@10": {"per_seed": hnsw.tolist(), "mean": float(hnsw.mean())},
            "exact_gain_vs_K1000": {"per_seed": exact_gain.tolist(),
                                      "mean": float(exact_gain.mean())},
            "hnsw_gain_vs_K1000": {"per_seed": hnsw_gain.tolist(),
                                     "mean": float(hnsw_gain.mean())},
            "gap_closed": {"per_seed": gap_closed.tolist(),
                            "mean": float(gap_closed.mean())},
            "hnsw_gain_retention": {
                "per_seed": [None if np.isnan(x) else float(x) for x in gain_retention],
                "mean_positive_exact": (float(np.nanmean(gain_retention))
                                        if np.any(~np.isnan(gain_retention)) else None),
            },
            "end_to_end_p50_ms": {"per_seed": p50.tolist(), "mean": float(p50.mean())},
            "end_to_end_p95_ms": {"per_seed": p95.tolist(), "mean": float(p95.mean())},
        }
    return curve


def _deployment_decision(pool_useful, deployable):
    """Separate the preregistered K rule from the frozen actual system."""
    return {
        "pool_expansion_closes_half_gap": bool(pool_useful),
        "preregistered_rule_K": min(deployable) if deployable else None,
        "actual_system_K": 1000,
        "retain_Y2_K1000": True,
        "larger_K_role": "sensitivity_only",
    }


def finalize_report(args):
    report_path = os.path.join(args.out, "Y4_REPORT.json")
    report = _load_json(report_path)
    # Finalization is intentionally idempotent: a completed archived report can
    # be revalidated without rerunning the HNSW stage.
    if report.get("stage") not in ("hnsw-complete", "complete"):
        raise SystemExit("Y4 HNSW stage is incomplete or failed recall gate")
    curve = _curve_summary(report)
    report["curve"] = curve

    k10 = curve[str(MAX_K)]
    pool_useful = (all(x >= 0 for x in k10["gap_closed"]["per_seed"])
                   and k10["gap_closed"]["mean"] >= 0.50)
    deployable = []
    for k in KS[1:]:
        row = curve[str(k)]
        retention = row["hnsw_gain_retention"]["mean_positive_exact"]
        no_seed_worse = all(x >= -1e-15 for x in row["hnsw_gain_vs_K1000"]["per_seed"])
        preserves_gain = retention is not None and retention >= 0.90 and no_seed_worse
        latency_ok = (row["end_to_end_p50_ms"]["mean"] <= 5.0
                      and row["end_to_end_p95_ms"]["mean"] <= 10.0)
        row["preserves_exact_gain_gate"] = preserves_gain
        row["latency_gate"] = latency_ok
        if pool_useful and preserves_gain and latency_ok:
            deployable.append(k)

    # The post-experiment system decision retains the Y2 operating point:
    # +0.0069 gold@10 at K=2000 was not worth changing the actual default.
    report["decision"] = _deployment_decision(pool_useful, deployable)
    report["gates"].update({
        "exact_K10000_gap_closed_all_nonnegative":
            all(x >= 0 for x in k10["gap_closed"]["per_seed"]),
        "exact_K10000_gap_closed_mean_ge_0.50":
            k10["gap_closed"]["mean"] >= 0.50,
        "all_artifacts_exist": all(os.path.isfile(os.path.join(
            args.out, "exact_pool_top10000_s%d.npz" % s)) for s in SEEDS),
    })
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        torch_version = torch.__version__
    except Exception:
        gpu = torch_version = None
    report["environment"] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch_version,
        "gpu_for_exact_stage": gpu,
        "note": "HNSW latency is single-query/single-thread after warmup; index load excluded.",
    }
    report["stage"] = "complete"
    report["written_at"] = utcnow()
    write_json_atomic(report_path, report)
    os.makedirs(args.docs_out, exist_ok=True)
    docs_report = os.path.join(args.docs_out, "Y4_REPORT.json")
    shutil.copyfile(report_path, docs_report)
    print(json.dumps({"decision": report["decision"], "curve": curve}, indent=2),
          flush=True)
    return report


def main(argv=None):
    data = os.path.join(data_root(), "data1m")
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("exact", "replay-exact", "hnsw",
                                             "finalize", "all"),
                        default="exact")
    parser.add_argument("--exports", default=os.path.join(data, "exports_x4"))
    parser.add_argument("--run-fmt", default="X4GD_full_s%d_e25")
    parser.add_argument("--sidecar-exports", default=os.path.join(data, "exports_rf"))
    parser.add_argument("--sidecar-run-fmt", default="RF_full_s%d_e25")
    parser.add_argument("--dataset-nodes", default=os.path.join(
        data, "rf", "canon", "dataset_nodes_merged.parquet"))
    parser.add_argument("--y2-out", default=os.path.join(data, "metrics_y2"))
    parser.add_argument("--y2-report", default=os.path.join(data, "metrics_y2",
                                                             "Y2_REPORT.json"))
    parser.add_argument("--out", default=os.path.join(data, "metrics_y4"))
    parser.add_argument("--frozen-pools", default=None,
                        help="directory containing release exact_pool_top10000_s*.npz")
    parser.add_argument(
        "--docs-out", default=os.path.join(data, "metrics_y4", "archived_copy"),
        help="optional copy destination; defaults under the data root, never the checkout")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-chunk", type=int, default=50_000)
    parser.add_argument("--query-chunk", type=int, default=16)
    parser.add_argument("--hnsw-threads", type=int, default=8)
    args = parser.parse_args(argv)
    result = None
    if args.stage in ("exact", "all"):
        result = run_exact(args)
    if args.stage == "replay-exact":
        result = run_replay_exact(args)
    if args.stage in ("hnsw", "all"):
        result = run_hnsw(args)
    if args.stage in ("finalize", "all"):
        result = finalize_report(args)
    return result


if __name__ == "__main__":
    main()
