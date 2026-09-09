"""Z1: fixed calibration of dense and task-prior scores at HNSW K=1000."""
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
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_PARENT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_PARENT not in sys.path:
    sys.path.insert(0, _REPO_PARENT)

from ModelLakeFishing.scale import global_metrics as GM
from ModelLakeFishing.scale1m import checkpoint as CK
from ModelLakeFishing.scale1m.eval_rf import _prior_tables, query_eligibility
from ModelLakeFishing.scale1m.eval_y2 import (
    EXPECTED_QUERIES,
    N_TOTAL,
    SEEDS,
    _audit_seed,
    _tie_ranks,
    _topk_update,
)
from ModelLakeFishing.scale1m.graph_store import load_sharded
from ModelLakeFishing.scale1m.hf_crawl import data_root, utcnow, write_json_atomic
from ModelLakeFishing.stage2TrainGraphSAGE.ablation import build_models
from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import apply_similar_to_mode
from ModelLakeFishing.stage2TrainGraphSAGE.inference import chunked_forward
from ModelLakeFishing.stage2TrainGraphSAGE.losses import TRAINED_ON, REV_TRAINED_ON


K = 1_000
CALIBRATIONS = ("raw", "zscore", "percentile")
BETAS = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
SHRINKS = (1.0, 5.0, 20.0)
CURRENT = {"calibration": "raw", "beta": 1.0, "shrink_k": 5.0}
EXPECTED_Y2 = (0.32791327913279134, 0.3387829246139873,
               0.24271844660194175)


def _load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _split_indices(edge_index, roots, seed, num_val=0.1, num_test=0.2):
    """Exact root assignment prefix of make_root_aware_splits, without cloning x."""
    import torch

    edge_root = [roots[int(d)] for d in edge_index[1]]
    uniq = sorted(set(edge_root))
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(uniq), generator=gen).tolist()
    counts = {}
    for root in edge_root:
        counts[root] = counts.get(root, 0) + 1
    side, n_edges = {}, edge_index.shape[1]
    acc_test = acc_val = 0
    for i in perm:
        root = uniq[i]
        if acc_test < num_test * n_edges:
            side[root] = "test"
            acc_test += counts[root]
        elif acc_val < num_val * n_edges:
            side[root] = "val"
            acc_val += counts[root]
        else:
            side[root] = "train"
    masks = {name: torch.tensor([side[root] == name for root in edge_root])
             for name in ("train", "val", "test")}
    return {name: mask.nonzero().flatten() for name, mask in masks.items()}


def _candidates_from_edges(edge_index, edge_attr, eligible):
    models = edge_index[0].cpu().numpy()
    datasets = edge_index[1].cpu().numpy()
    acc = edge_attr.float().cpu().numpy()
    out = {}
    for query in np.unique(datasets):
        if not eligible[int(query)]:
            continue
        take = datasets == query
        cand, values = models[take], acc[take]
        if cand.size >= 3 and float(np.std(values)) > 0:
            out[int(query)] = (cand.astype(np.int64), values.astype(float))
    return out


class _Prior:
    def __init__(self, payload, shrink):
        self.task_id = np.asarray(payload["task_id"], dtype=np.int64)
        self.by_task, _root, _null = _prior_tables(payload, shrink=shrink)

    def matrix(self, queries, model_ids):
        out = np.zeros(model_ids.shape, dtype=np.float32)
        for row, query in enumerate(queries):
            idx, val = self.by_task.get(
                int(self.task_id[int(query)]),
                (np.zeros(0, np.int64), np.zeros(0, np.float64)))
            if idx.size:
                pos = np.searchsorted(idx, model_ids[row])
                clip = np.minimum(pos, idx.size - 1)
                hit = (pos < idx.size) & (idx[clip] == model_ids[row])
                out[row, hit] = np.asarray(val, np.float32)[clip[hit]]
        return out


def _probe_arrays(queries, candidates):
    gold = np.empty(len(queries), dtype=np.int64)
    top3 = np.empty((len(queries), 3), dtype=np.int64)
    for i, query in enumerate(queries):
        g, t3, _near, _ids = GM._probe_ids(*candidates[int(query)], GM.GAP_DELTA)
        gold[i] = g
        top3[i] = np.asarray(t3, dtype=np.int64)
    return gold, top3


def _exact_pool(z_model, z_dataset, queries, device, model_chunk=50_000,
                query_chunk=16):
    import torch

    zm = z_model.float()
    zd = z_dataset.float()
    zm = (zm / zm.norm(dim=1, keepdim=True).clamp_min(1e-12)).to(device)
    zd = (zd / zd.norm(dim=1, keepdim=True).clamp_min(1e-12)).to(device)
    ids = np.empty((len(queries), K), dtype=np.int32)
    scores = np.empty((len(queries), K), dtype=np.float32)
    for start in range(0, len(queries), query_chunk):
        block = queries[start:start + query_chunk]
        zq = zd[torch.as_tensor(block, dtype=torch.long, device=device)]
        buf = (None, None)
        for ms in range(0, zm.size(0), model_chunk):
            raw = zm[ms:ms + model_chunk] @ zq.t()
            buf = _topk_update(*buf, raw, ms, K)
            del raw
        width = len(block)
        ids[start:start + width] = buf[1].t().cpu().numpy().astype(np.int32)
        scores[start:start + width] = buf[0].t().cpu().numpy().astype(np.float32)
    return ids, scores


def _calibrate(values, method):
    values = np.asarray(values, dtype=np.float64)
    if method == "raw":
        return values
    if method == "zscore":
        mean = values.mean(axis=1, keepdims=True)
        std = values.std(axis=1, keepdims=True)
        return np.divide(values - mean, std, out=np.zeros_like(values),
                         where=std > 1e-12)
    if method == "percentile":
        from scipy.stats import rankdata
        if values.shape[1] <= 1:
            return np.zeros_like(values)
        return (rankdata(values, method="average", axis=1) - 1.0) / (values.shape[1] - 1.0)
    raise ValueError("unknown calibration %s" % method)


def _scores(dense_cos, prior, calibration, beta):
    dense = (np.asarray(dense_cos, dtype=np.float64) + 1.0) * 0.5
    return _calibrate(dense, calibration) + float(beta) * _calibrate(prior, calibration)


def _ranks_for_probes(scores, pool_ids, probes, tie_rank):
    scores = np.asarray(scores)
    out = np.full(probes.shape, K + 1, dtype=np.int64)
    rows = np.arange(len(pool_ids))
    for col in range(probes.shape[1]):
        target = probes[:, col]
        match = pool_ids == target[:, None]
        present = match.any(axis=1)
        if not present.any():
            continue
        pos = match.argmax(axis=1)
        use = rows[present]
        target_score = scores[use, pos[present]]
        greater = (scores[use] > target_score[:, None]).sum(axis=1)
        equal_better = ((scores[use] == target_score[:, None]) &
                        (tie_rank[pool_ids[use]] < tie_rank[target[present]][:, None])).sum(axis=1)
        out[present, col] = greater + equal_better + 1
    return out


def _evaluate(scores, pool_ids, gold, top3, tie_rank, return_top10=False):
    probes = np.concatenate([gold[:, None], top3], axis=1)
    ranks = _ranks_for_probes(scores, pool_ids, probes, tie_rank)
    gold_rank = ranks[:, 0]
    hit = gold_rank <= 10
    row = {
        "n_queries": int(len(gold)),
        "gold@1": float(np.mean(gold_rank == 1)),
        "gold@10": float(np.mean(hit)),
        "top3@10": float(np.mean(ranks[:, 1:].min(axis=1) <= 10)),
    }
    top10 = None
    if return_top10:
        top10 = np.empty((len(gold), 10), dtype=np.int64)
        for i in range(len(gold)):
            order = np.lexsort((tie_rank[pool_ids[i]], -scores[i]))
            top10[i] = pool_ids[i, order[:10]]
    return row, hit, top10


def _paired(base, new):
    base, new = np.asarray(base, bool), np.asarray(new, bool)
    if base.shape != new.shape:
        raise AssertionError("paired hit vectors differ")
    rescue = int(np.sum(~base & new))
    harm = int(np.sum(base & ~new))
    if int(new.sum()) - int(base.sum()) != rescue - harm:
        raise AssertionError("rescue/harm identity failed")
    return {"n_queries": int(len(base)), "rescue": rescue, "harm": harm,
            "net": rescue - harm, "rescue_rate": rescue / len(base),
            "harm_rate": harm / len(base), "net_rate": (rescue - harm) / len(base)}


def _diagnostics(dense_cos, prior):
    dense = (dense_cos.astype(np.float64) + 1.0) * 0.5

    def one(values):
        top = np.partition(values, -10, axis=1)[:, -10:]
        margin = top.max(axis=1) - top.min(axis=1)
        return {"std_mean": float(values.std(axis=1).mean()),
                "std_median": float(np.median(values.std(axis=1))),
                "top1_top10_margin_mean": float(margin.mean()),
                "top1_top10_margin_median": float(np.median(margin))}
    return {"dense": one(dense), "prior": one(prior.astype(np.float64))}


def _size_stats(top10, sizes):
    values = sizes[np.asarray(top10, dtype=np.int64)].reshape(-1)
    good = np.isfinite(values)
    return {"n": int(values.size), "n_nonmissing": int(good.sum()),
            "missing_rate": float(1.0 - good.mean()),
            "median_size_b": float(np.median(values[good])) if good.any() else None}


def _config_key(config):
    return "%s_b%g_k%g" % (config["calibration"], config["beta"], config["shrink_k"])


def _configs():
    out = []
    for calibration in CALIBRATIONS:
        for beta in BETAS:
            shrinks = (5.0,) if beta == 0 else SHRINKS
            for shrink in shrinks:
                out.append({"calibration": calibration, "beta": beta, "shrink_k": shrink})
    return out


def _selection_penalty(config):
    cal = {"raw": 0, "zscore": 1, "percentile": 2}[config["calibration"]]
    beta = 10.0 if config["beta"] == 0 else abs(math.log2(config["beta"]))
    shrink = 0 if config["shrink_k"] == 5 else 1
    return (cal + shrink, beta, _config_key(config))


def _save_pool(path, queries, ids, dense, priors, gold, top3, roots, **extra):
    payload = {"query": np.asarray(queries, np.int64), "model": ids,
               "dense_cos": dense, "gold": gold, "top3": top3,
               "root": np.asarray(roots, dtype=str)}
    payload.update({"prior_k%g" % k: v for k, v in priors.items()})
    payload.update(extra)
    np.savez(path, **payload)


def _load_pool(path):
    z = np.load(path)
    return z


def prepare(args):
    import torch

    os.makedirs(args.out, exist_ok=True)
    payload = load_sharded(args.graph, mmap=True, verify_sha256=False)
    data = payload["data"]
    udi = payload["unique_dataset_id"].sort_values("mappedID").reset_index(drop=True)
    roots = udi["root"].astype(str).tolist()
    if int(data["model"].num_nodes) != N_TOTAL:
        raise AssertionError("wrong model universe")
    graph_digest = CK.graph_digest(args.graph)
    eligible, _why = query_eligibility(args.dataset_nodes, args.first_export)
    tie_rank = _tie_ranks(N_TOTAL)

    # All X4-GD checkpoints use the same architecture/surgery configuration.
    checkpoints = []
    for seed in SEEDS:
        manifest = _load_json(os.path.join(args.exports, args.run_fmt % seed,
                                           "EXPORT_MANIFEST.json"))["stages"]["embed"]
        ck = CK.load(manifest["checkpoint"])
        if ck["binding"]["graph_sha256"] != graph_digest:
            raise AssertionError("checkpoint graph binding mismatch")
        if int(ck["binding"]["split_seed"]) != seed:
            raise AssertionError("checkpoint split seed mismatch")
        checkpoints.append((manifest, ck))
    runtime_keys = {"history0", "on_epoch_end", "resume_state"}
    cfg0 = {k: v for k, v in checkpoints[0][1]["cfg"].items()
            if k not in runtime_keys}
    if any({k: v for k, v in ck["cfg"].items() if k not in runtime_keys} != cfg0
           for _manifest, ck in checkpoints[1:]):
        raise AssertionError("X4-GD configs differ across split seeds")
    data = apply_similar_to_mode(data, cfg0["similar_to_mode"], k=cfg0["similar_to_k"])
    full_ei = data[TRAINED_ON].edge_index
    full_ea = data[TRAINED_ON].edge_attr
    original_reverse = (data[REV_TRAINED_ON].edge_index,
                        data[REV_TRAINED_ON].edge_attr)
    report = {"written_at": utcnow(), "stage": "prepare-partial",
              "protocol": {"K": K, "calibrations": CALIBRATIONS,
                           "betas": BETAS, "shrinks": SHRINKS},
              "validation": {}, "gates": {}}
    report_path = os.path.join(args.out, "Z1_REPORT.json")

    for seed, (manifest, ck) in zip(SEEDS, checkpoints):
        split = _split_indices(full_ei, roots, seed)
        tr, va, te = split["train"], split["val"], split["test"]
        side = np.load(os.path.join(args.sidecar_exports,
                                    args.sidecar_run_fmt % seed,
                                    "prior_sidecar_s%d.npz" % seed))
        expected = torch.cat([tr, va])
        if not (np.array_equal(side["edge_model"], full_ei[0, expected].cpu().numpy())
                and np.array_equal(side["edge_dataset"], full_ei[1, expected].cpu().numpy())
                and np.allclose(side["edge_acc"], full_ea[expected].cpu().numpy(),
                                rtol=0, atol=1e-7)):
            raise AssertionError("local root split does not reproduce test sidecar")

        val_candidates = _candidates_from_edges(full_ei[:, va], full_ea[va], eligible)
        queries = sorted(val_candidates)
        if not queries:
            raise AssertionError("no eligible validation queries")
        val_roots = {roots[q] for q in queries}
        test_roots = {roots[int(q)] for q in torch.unique(full_ei[1, te]).tolist()}
        train_roots = {roots[int(q)] for q in torch.unique(full_ei[1, tr]).tolist()}
        if val_roots & train_roots or val_roots & test_roots:
            raise AssertionError("validation roots overlap train/test")

        data[TRAINED_ON].edge_index = full_ei[:, tr]
        data[TRAINED_ON].edge_attr = full_ea[tr]
        data[REV_TRAINED_ON].edge_index = full_ei[:, tr].flip(0)
        data[REV_TRAINED_ON].edge_attr = full_ea[tr].clone()
        model, scorer = build_models(data, payload["xm0_meta"], payload["xd0_meta"],
                                     cfg0, device=args.device)
        model.load_state_dict(ck["model"])
        scorer.load_state_dict(ck["scorer"])
        model.eval()
        started = time.time()
        with torch.no_grad():
            z = chunked_forward(model, data, chunk_size=args.chunk, device=args.device,
                                progress=args.progress)
        ids, dense = _exact_pool(z["model"], z["dataset"], queries, args.device,
                                 args.model_chunk, args.query_chunk)
        gold, top3 = _probe_arrays(queries, val_candidates)
        prior_payload = {"edge_model": full_ei[0, tr].cpu().numpy(),
                         "edge_dataset": full_ei[1, tr].cpu().numpy(),
                         "edge_acc": full_ea[tr].cpu().numpy(),
                         "task_id": side["task_id"], "root_id": side["root_id"]}
        priors = {shrink: _Prior(prior_payload, shrink).matrix(queries, ids)
                  for shrink in SHRINKS}
        artifact = os.path.join(args.out, "val_pool_s%d.npz" % seed)
        _save_pool(artifact, queries, ids, dense, priors, gold, top3,
                   [roots[q] for q in queries])
        report["validation"][str(seed)] = {
            "n_queries": len(queries), "n_train_edges": int(len(tr)),
            "n_val_edges": int(len(va)), "n_test_edges": int(len(te)),
            "seconds": time.time() - started, "artifact": artifact,
            "checkpoint": manifest["checkpoint"]}
        write_json_atomic(report_path, report)
        del model, scorer, z, ids, dense, priors
        gc.collect()
        torch.cuda.empty_cache()

    data[TRAINED_ON].edge_index, data[TRAINED_ON].edge_attr = full_ei, full_ea
    data[REV_TRAINED_ON].edge_index, data[REV_TRAINED_ON].edge_attr = original_reverse
    report["stage"] = "prepare-complete"
    report["gates"].update({"graph_and_checkpoint_binding": True,
                            "validation_roots_disjoint": True,
                            "validation_prior_train_only": True,
                            "test_sidecars_reproduced": True})
    report["written_at"] = utcnow()
    write_json_atomic(report_path, report)
    return report


def _pool_inputs(z):
    priors = {float(k.split("prior_k", 1)[1]): z[k].astype(np.float32)
              for k in z.files if k.startswith("prior_k")}
    return (z["model"].astype(np.int64), z["dense_cos"].astype(np.float32),
            priors, z["gold"].astype(np.int64), z["top3"].astype(np.int64))


def _test_pool(args, seed, tie_rank, y2):
    import hnswlib

    bundle = _audit_seed(args, seed, None, None, tie_rank)
    queries = sorted(bundle["candidates"])
    if len(queries) != EXPECTED_QUERIES[seed]:
        raise AssertionError("test query count changed")
    zd = np.load(os.path.join(bundle["x4"], "z_d_eval.npy"), mmap_mode="r")
    zq = np.asarray(zd[queries], dtype=np.float32)
    zq /= np.maximum(np.linalg.norm(zq, axis=1, keepdims=True), 1e-12)
    index = hnswlib.Index(space="ip", dim=zq.shape[1])
    index.load_index(os.path.join(bundle["x4"], "hnsw_y2_eval.bin"),
                     max_elements=N_TOTAL)
    ef = int(y2["hnsw"][str(seed)]["ef_search"])
    index.set_ef(ef)
    index.set_num_threads(args.hnsw_threads)
    ids, dist = index.knn_query(zq, k=K, num_threads=args.hnsw_threads)
    dense = (1.0 - dist).astype(np.float32)
    query_ms = []
    index.set_num_threads(1)
    for i in range(min(50, len(queries))):
        index.knn_query(zq[i:i + 1], k=K, num_threads=1)
    for i in range(len(queries)):
        t0 = time.perf_counter_ns()
        one, _d = index.knn_query(zq[i:i + 1], k=K, num_threads=1)
        query_ms.append((time.perf_counter_ns() - t0) / 1e6)
        if not np.array_equal(one[0], ids[i]):
            raise AssertionError("batch/single HNSW results differ")
    side_path = os.path.join(args.sidecar_exports, args.sidecar_run_fmt % seed,
                             "prior_sidecar_s%d.npz" % seed)
    side = np.load(side_path)
    priors = {shrink: _Prior(side, shrink).matrix(queries, ids) for shrink in SHRINKS}
    gold, top3 = _probe_arrays(queries, bundle["candidates"])
    artifact = os.path.join(args.out, "test_pool_s%d.npz" % seed)
    _save_pool(artifact, queries, ids, dense, priors, gold, top3,
               [bundle["roots"][q] for q in queries],
               hnsw_ms=np.asarray(query_ms, np.float64), ef_search=np.asarray([ef]))
    del index, bundle
    return artifact


def _rerank_latency(ids, dense, prior, config, tie_rank):
    from scipy.stats import rankdata

    elapsed = []
    for i in range(len(ids)):
        t0 = time.perf_counter_ns()
        d = (dense[i].astype(np.float64) + 1.0) * 0.5
        p = prior[i].astype(np.float64)
        if config["calibration"] == "zscore":
            ds, ps = d.std(), p.std()
            d = (d - d.mean()) / ds if ds > 1e-12 else np.zeros_like(d)
            p = (p - p.mean()) / ps if ps > 1e-12 else np.zeros_like(p)
        elif config["calibration"] == "percentile":
            d = (rankdata(d, method="average") - 1.0) / (len(d) - 1.0)
            p = (rankdata(p, method="average") - 1.0) / (len(p) - 1.0)
        score = d + config["beta"] * p
        np.lexsort((tie_rank[ids[i]], -score))[:10]
        elapsed.append((time.perf_counter_ns() - t0) / 1e6)
    return np.asarray(elapsed)


def search(args):
    report_path = os.path.join(args.out, "Z1_REPORT.json")
    report = _load_json(report_path)
    if report.get("stage") != "prepare-complete":
        raise SystemExit("Z1 prepare stage is incomplete")
    tie_rank = _tie_ranks(N_TOTAL)
    configs = _configs()
    val_rows = {key: [] for key in map(_config_key, configs)}
    val_detail = {}
    for seed in SEEDS:
        z = _load_pool(os.path.join(args.out, "val_pool_s%d.npz" % seed))
        ids, dense, priors, gold, top3 = _pool_inputs(z)
        seed_rows = {}
        dense_raw = (dense.astype(np.float64) + 1.0) * 0.5
        dense_calibrated = {
            calibration: _calibrate(dense_raw, calibration)
            for calibration in CALIBRATIONS
        }
        prior_calibrated = {
            (calibration, shrink_k): _calibrate(priors[shrink_k], calibration)
            for calibration in CALIBRATIONS
            for shrink_k in SHRINKS
        }
        for config in configs:
            score = (dense_calibrated[config["calibration"]]
                     + config["beta"] * prior_calibrated[
                         (config["calibration"], config["shrink_k"])])
            row, hit, _top = _evaluate(score, ids, gold, top3, tie_rank)
            key = _config_key(config)
            val_rows[key].append(row["gold@10"])
            seed_rows[key] = {"row": row, "hits": hit}
        val_detail[str(seed)] = seed_rows
        z.close()

    candidates = []
    by_key = {_config_key(c): c for c in configs}
    for key, values in val_rows.items():
        config = by_key[key]
        candidates.append((-float(np.mean(values)), -float(np.min(values)),
                           _selection_penalty(config), key))
    candidates.sort()
    best_key = candidates[0][-1]
    best = by_key[best_key]
    current_key = _config_key(CURRENT)
    report["selection"] = {
        "rule": "max mean val gold@10; then max worst seed; then simplicity",
        "best": best,
        "best_key": best_key,
        "current_key": current_key,
        "validation_grid": {
            key: {"per_seed_gold@10": values, "mean_gold@10": float(np.mean(values)),
                  "min_gold@10": float(np.min(values))}
            for key, values in val_rows.items()},
    }
    report["validation_results"] = {}
    for seed in SEEDS:
        base = val_detail[str(seed)][current_key]
        chosen = val_detail[str(seed)][best_key]
        report["validation_results"][str(seed)] = {
            "current": base["row"], "selected": chosen["row"],
            "paired": _paired(base["hits"], chosen["hits"])}

    # Only after selection is frozen do we construct/read the test pools.
    y2 = _load_json(args.y2_report)
    test_results = {}
    sizes_df = pd.read_parquet(args.model_meta).sort_values("mappedID")
    if not np.array_equal(sizes_df["mappedID"].to_numpy(), np.arange(N_TOTAL)):
        raise AssertionError("model metadata row order mismatch")
    sizes = sizes_df["size_b"].to_numpy(dtype=float)
    for seed in SEEDS:
        test_path = os.path.join(args.out, "test_pool_s%d.npz" % seed)
        if not os.path.isfile(test_path):
            _test_pool(args, seed, tie_rank, y2)
        z = _load_pool(test_path)
        ids, dense, priors, gold, top3 = _pool_inputs(z)
        methods = {
            "dense_pool": _scores(dense, priors[5.0], "raw", 0.0),
            "prior_pool": priors[5.0].astype(np.float64),
            "current": _scores(dense, priors[5.0], "raw", 1.0),
            "selected": _scores(dense, priors[best["shrink_k"]],
                                best["calibration"], best["beta"]),
        }
        rows, hits, size = {}, {}, {}
        for name, score in methods.items():
            rows[name], hits[name], top10 = _evaluate(
                score, ids, gold, top3, tie_rank, return_top10=True)
            size[name] = _size_stats(top10, sizes)
        if not math.isclose(rows["current"]["gold@10"], EXPECTED_Y2[seed],
                            rel_tol=0, abs_tol=1e-12):
            raise AssertionError("current test system does not reproduce Y2")
        current_rerank = _rerank_latency(ids, dense, priors[5.0], CURRENT, tie_rank)
        selected_rerank = _rerank_latency(
            ids, dense, priors[best["shrink_k"]], best, tie_rank)
        hnsw_ms = z["hnsw_ms"].astype(float)
        test_results[str(seed)] = {
            "rows": rows,
            "paired_selected_vs_current": _paired(hits["current"], hits["selected"]),
            "size": size,
            "diagnostics_current": _diagnostics(dense, priors[5.0]),
            "latency_ms": {
                "hnsw_p50": float(np.percentile(hnsw_ms, 50)),
                "hnsw_p95": float(np.percentile(hnsw_ms, 95)),
                "current_rerank_p50": float(np.percentile(current_rerank, 50)),
                "selected_rerank_p50": float(np.percentile(selected_rerank, 50)),
                "selected_end_to_end_p50": float(np.percentile(hnsw_ms + selected_rerank, 50)),
                "selected_end_to_end_p95": float(np.percentile(hnsw_ms + selected_rerank, 95)),
            },
            "artifact": test_path,
        }
        z.close()
    report["test_results"] = test_results

    val_best = report["selection"]["validation_grid"][best_key]["mean_gold@10"]
    val_current = report["selection"]["validation_grid"][current_key]["mean_gold@10"]
    selected_test = [test_results[str(s)]["rows"]["selected"]["gold@10"] for s in SEEDS]
    current_test = [test_results[str(s)]["rows"]["current"]["gold@10"] for s in SEEDS]
    paired_ok = all(test_results[str(s)]["paired_selected_vs_current"]["rescue"] >
                    test_results[str(s)]["paired_selected_vs_current"]["harm"] for s in SEEDS)
    no_seed_down = all(selected_test[i] >= current_test[i] for i in range(3))
    latency_ok = all(test_results[str(s)]["latency_ms"]["selected_end_to_end_p50"] <= 5
                     and test_results[str(s)]["latency_ms"]["selected_end_to_end_p95"] <= 10
                     for s in SEEDS)
    size_current = np.asarray([test_results[str(s)]["size"]["current"]["median_size_b"]
                               for s in SEEDS], float)
    size_selected = np.asarray([test_results[str(s)]["size"]["selected"]["median_size_b"]
                                for s in SEEDS], float)
    missing_current = np.asarray([test_results[str(s)]["size"]["current"]["missing_rate"]
                                  for s in SEEDS])
    missing_selected = np.asarray([test_results[str(s)]["size"]["selected"]["missing_rate"]
                                   for s in SEEDS])
    size_ok = (np.all(size_selected <= 2.0 * size_current)
               and np.all(missing_selected <= missing_current + 0.05))
    gates = {
        "validation_improves_mean": val_best > val_current,
        "test_no_seed_down": no_seed_down,
        "test_rescue_gt_harm_each_seed": paired_ok,
        "test_mean_above_y2": float(np.mean(selected_test)) > float(np.mean(EXPECTED_Y2)),
        "latency_budget": latency_ok,
        "size_bias_gate": bool(size_ok),
        "y2_reproduced": True,
    }
    passed = all(gates.values())
    report["gates"].update(gates)
    report["decision"] = {
        "fixed_calibration_effective": passed,
        "selected_config": best,
        "keep_current_config": not passed,
        "validation_current_mean": val_current,
        "validation_selected_mean": val_best,
        "test_current": {"per_seed": current_test, "mean": float(np.mean(current_test))},
        "test_selected": {"per_seed": selected_test, "mean": float(np.mean(selected_test))},
    }
    report["environment"] = {"platform": platform.platform(),
                             "python": platform.python_version(),
                             "logical_cpus": os.cpu_count()}
    report["stage"] = "complete"
    report["written_at"] = utcnow()
    write_json_atomic(report_path, report)
    os.makedirs(args.docs_out, exist_ok=True)
    shutil.copyfile(report_path, os.path.join(args.docs_out, "Z1_REPORT.json"))
    print(json.dumps({"decision": report["decision"], "gates": report["gates"]},
                     indent=2), flush=True)
    return report


def main(argv=None):
    data = os.path.join(data_root(), "data1m")
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "search", "all"), default="prepare")
    parser.add_argument("--graph", default=os.path.join(data, "graphs", "hgraph_rf"))
    parser.add_argument("--exports", default=os.path.join(data, "exports_x4"))
    parser.add_argument("--run-fmt", default="X4GD_full_s%d_e25")
    parser.add_argument("--first-export", default=os.path.join(
        data, "exports_x4", "X4GD_full_s0_e25"))
    parser.add_argument("--sidecar-exports", default=os.path.join(data, "exports_rf"))
    parser.add_argument("--sidecar-run-fmt", default="RF_full_s%d_e25")
    parser.add_argument("--dataset-nodes", default=os.path.join(
        data, "rf", "canon", "dataset_nodes_merged.parquet"))
    parser.add_argument("--model-meta", default=os.path.join(
        data, "utility_rf", "model_meta.parquet"))
    parser.add_argument("--y2-report", default=os.path.join(data, "metrics_y2",
                                                             "Y2_REPORT.json"))
    parser.add_argument("--out", default=os.path.join(data, "metrics_z1"))
    parser.add_argument("--docs-out", default=os.path.join(
        _REPO_PARENT, "ModelLakeFishing", "docs", "1M", "Z1_runs"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk", type=int, default=50_000)
    parser.add_argument("--model-chunk", type=int, default=50_000)
    parser.add_argument("--query-chunk", type=int, default=16)
    parser.add_argument("--hnsw-threads", type=int, default=8)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    result = None
    if args.stage in ("prepare", "all"):
        result = prepare(args)
    if args.stage in ("search", "all"):
        result = search(args)
    return result


if __name__ == "__main__":
    main()
