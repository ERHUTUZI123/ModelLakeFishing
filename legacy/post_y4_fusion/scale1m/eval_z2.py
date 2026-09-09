"""Z2: confidence-conditioned query-level fusion at frozen HNSW K=1000.

The fit stage reads only Z1's validation-safe pools.  It freezes one gate per
split seed before the test stage is allowed to open a Z1 test pool.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import GroupKFold

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_REPO_PARENT = os.path.abspath(os.path.join(_REPO_ROOT, ".."))
if _REPO_PARENT not in sys.path:
    sys.path.insert(0, _REPO_PARENT)

from ModelLakeFishing.scale1m.eval_y2 import N_TOTAL, SEEDS, _tie_ranks
from ModelLakeFishing.scale1m.eval_z1 import (
    _evaluate,
    _load_pool,
    _paired,
    _size_stats,
    _split_indices,
)
from ModelLakeFishing.scale1m.graph_store import load_sharded
from ModelLakeFishing.scale1m.hf_crawl import data_root, utcnow, write_json_atomic
from ModelLakeFishing.stage2TrainGraphSAGE.losses import TRAINED_ON


G0 = 8.0 / 9.0
BASE_LOGIT = math.log(G0 / (1.0 - G0))
TEMPERATURE = 0.05
L2 = 1.0
N_HARD = 50
N_FOLDS = 5
FEATURE_NAMES = (
    "log_task_models",
    "log_task_roots",
    "pool_supported_fraction",
    "log_top10_support_mean",
    "prior_std",
    "prior_top1_top10_margin",
    "negative_dense_top1_top10_margin",
)
EXPECTED_Z1_FIXED = (0.34959349593495936, 0.3333333333333333,
                     0.25954692556634307)
EXPECTED_Z1_FIXED_MEAN = float(np.mean(EXPECTED_Z1_FIXED))
EXPECTED_PRODUCTION = (0.32791327913279134, 0.3387829246139873,
                       0.24271844660194175)


def _load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _mix(dense_cos, prior, gate):
    dense = (np.asarray(dense_cos, dtype=np.float64) + 1.0) * 0.5
    prior = np.asarray(prior, dtype=np.float64)
    gate = np.asarray(gate, dtype=np.float64)
    if gate.ndim == 0:
        gate = np.full(len(dense), float(gate), dtype=np.float64)
    if gate.shape != (len(dense),):
        raise AssertionError("gate must contain one scalar per query")
    if np.any((gate < 0.0) | (gate > 1.0)):
        raise AssertionError("gate escaped [0,1]")
    return (1.0 - gate[:, None]) * dense + gate[:, None] * prior


class HistoryStats:
    """Train-visible task support, with no labels from the query root."""

    def __init__(self, edge_model, edge_dataset, task_id, root_id):
        edge_model = np.asarray(edge_model, dtype=np.int64)
        edge_dataset = np.asarray(edge_dataset, dtype=np.int64)
        self.task_id = np.asarray(task_id, dtype=np.int64)
        root_id = np.asarray(root_id, dtype=np.int64)
        if edge_model.shape != edge_dataset.shape:
            raise AssertionError("history edge arrays differ")
        if edge_dataset.size and (edge_dataset.min() < 0 or
                                  edge_dataset.max() >= len(self.task_id)):
            raise AssertionError("history dataset id out of range")
        task = self.task_id[edge_dataset]
        frame = pd.DataFrame({"task": task, "model": edge_model})
        counts = frame.groupby(["task", "model"], sort=True).size()
        self.by_task = {}
        for key, values in counts.groupby(level=0, sort=True):
            self.by_task[int(key)] = (
                values.index.get_level_values(1).to_numpy(dtype=np.int64),
                values.to_numpy(dtype=np.int64),
            )
        self.n_models = {int(k): int(v) for k, v in
                         frame.groupby("task")["model"].nunique().items()}
        roots = pd.DataFrame({"task": task, "root": root_id[edge_dataset]})
        self.n_roots = {int(k): int(v) for k, v in
                        roots.groupby("task")["root"].nunique().items()}
        self.visible_root_ids = set(root_id[edge_dataset].tolist())

    def counts(self, query, model_ids):
        task = int(self.task_id[int(query)])
        models, counts = self.by_task.get(
            task, (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)))
        out = np.zeros(len(model_ids), dtype=np.int64)
        if models.size:
            pos = np.searchsorted(models, model_ids)
            clip = np.minimum(pos, models.size - 1)
            hit = (pos < models.size) & (models[clip] == model_ids)
            out[hit] = counts[clip[hit]]
        return task, out


def _top_margin(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[1] < 10:
        raise AssertionError("top-10 margin requires at least 10 candidates")
    top = np.partition(values, -10, axis=1)[:, -10:]
    return top.max(axis=1) - top.min(axis=1)


def _query_features(queries, model_ids, dense_cos, prior, history, tie_rank):
    queries = np.asarray(queries, dtype=np.int64)
    model_ids = np.asarray(model_ids, dtype=np.int64)
    dense = (np.asarray(dense_cos, dtype=np.float64) + 1.0) * 0.5
    prior = np.asarray(prior, dtype=np.float64)
    if model_ids.shape != dense.shape or prior.shape != dense.shape:
        raise AssertionError("pool matrices differ")
    out = np.empty((len(queries), len(FEATURE_NAMES)), dtype=np.float64)
    prior_margin = _top_margin(prior)
    dense_margin = _top_margin(dense)
    prior_std = prior.std(axis=1)
    for row, query in enumerate(queries):
        task, support = history.counts(query, model_ids[row])
        order = np.lexsort((tie_rank[model_ids[row]], -prior[row]))[:10]
        out[row] = (
            math.log1p(history.n_models.get(task, 0)),
            math.log1p(history.n_roots.get(task, 0)),
            float(np.mean(support > 0)),
            float(np.mean(np.log1p(support[order]))),
            prior_std[row],
            prior_margin[row],
            -dense_margin[row],
        )
    if not np.isfinite(out).all():
        raise AssertionError("non-finite gate feature")
    return out


def _scaler(features):
    median = np.median(features, axis=0)
    raw_iqr = (np.percentile(features, 75, axis=0)
               - np.percentile(features, 25, axis=0))
    iqr = np.where(raw_iqr > 1e-12, raw_iqr, 1.0)
    return median, iqr


def _standardize(features, median, iqr):
    return ((np.asarray(features, dtype=np.float64)
             - np.asarray(median, dtype=np.float64))
            / np.asarray(iqr, dtype=np.float64))


def _hard_pairs(dense_cos, prior, model_ids, gold, top3, tie_rank):
    dense = (np.asarray(dense_cos, dtype=np.float64) + 1.0) * 0.5
    prior = np.asarray(prior, dtype=np.float64)
    model_ids = np.asarray(model_ids, dtype=np.int64)
    gold = np.asarray(gold, dtype=np.int64)
    top3 = np.asarray(top3, dtype=np.int64)
    base = (1.0 - G0) * dense + G0 * prior
    a = np.zeros((len(gold), N_HARD), dtype=np.float64)
    b = np.zeros_like(a)
    usable = np.zeros(len(gold), dtype=bool)
    for row in range(len(gold)):
        found = np.flatnonzero(model_ids[row] == gold[row])
        if found.size != 1:
            continue
        gold_pos = int(found[0])
        banned = set(int(v) for v in top3[row])
        banned.add(int(gold[row]))
        order = np.lexsort((tie_rank[model_ids[row]], -base[row]))
        negatives = [int(pos) for pos in order
                     if int(model_ids[row, pos]) not in banned][:N_HARD]
        if len(negatives) != N_HARD:
            continue
        dense_delta = dense[row, gold_pos] - dense[row, negatives]
        prior_delta = prior[row, gold_pos] - prior[row, negatives]
        a[row] = dense_delta
        b[row] = prior_delta - dense_delta
        usable[row] = True
    return a, b, usable


def _objective(params, features, a, b, usable):
    x = features[usable]
    aa = a[usable]
    bb = b[usable]
    if len(x) == 0:
        raise AssertionError("no pool-present gold queries for gate fitting")
    eta = BASE_LOGIT + params[0] + x @ params[1:]
    gate = expit(np.clip(eta, -30.0, 30.0))
    delta = aa + gate[:, None] * bb
    u = -delta / TEMPERATURE
    per_query = np.logaddexp(0.0, u).mean(axis=1)
    sigmoid_u = expit(np.clip(u, -30.0, 30.0))
    dloss_dgate = (-sigmoid_u / TEMPERATURE * bb).mean(axis=1)
    dloss_deta = dloss_dgate * gate * (1.0 - gate)
    value = float(per_query.mean() + 0.5 * L2 * np.dot(params, params))
    gradient = np.concatenate((
        np.asarray([dloss_deta.mean()]),
        x.T @ dloss_deta / len(x),
    )) + L2 * params
    return value, gradient


def _fit_gate(features, a, b, usable):
    start = np.zeros(features.shape[1] + 1, dtype=np.float64)
    result = minimize(
        _objective, start, args=(features, a, b, usable), jac=True,
        method="L-BFGS-B",
        bounds=[(None, None)] + [(0.0, None)] * features.shape[1],
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise AssertionError("gate optimizer failed: %s" % result.message)
    if np.any(result.x[1:] < -1e-12):
        raise AssertionError("monotonic coefficient became negative")
    return result.x, {
        "success": bool(result.success), "message": str(result.message),
        "iterations": int(result.nit), "objective": float(result.fun),
        "gradient_inf": float(np.max(np.abs(result.jac))),
    }


def _predict(features, median, iqr, params):
    x = _standardize(features, median, iqr)
    gate = expit(np.clip(BASE_LOGIT + float(params[0]) + x @ params[1:],
                         -30.0, 30.0))
    if np.any((gate <= 0.0) | (gate >= 1.0)):
        # The clipping should keep all returned values strictly within bounds.
        raise AssertionError("invalid gate probability")
    return gate


def _gate_summary(gate):
    q = np.percentile(gate, [0, 5, 25, 50, 75, 95, 100])
    return {name: float(value) for name, value in zip(
        ("min", "p05", "p25", "p50", "p75", "p95", "max"), q)}


def _history_for_validation(graph_data, roots, seed, sidecar):
    edge_index = graph_data[TRAINED_ON].edge_index
    split = _split_indices(edge_index, roots, seed)
    train = split["train"]
    return HistoryStats(edge_index[0, train].cpu().numpy(),
                        edge_index[1, train].cpu().numpy(),
                        sidecar["task_id"], sidecar["root_id"]), split


def _history_for_test(sidecar):
    return HistoryStats(sidecar["edge_model"], sidecar["edge_dataset"],
                        sidecar["task_id"], sidecar["root_id"])


def _rows(scores, ids, gold, top3, tie_rank):
    out = {}
    hits = {}
    top10 = {}
    for name, values in scores.items():
        out[name], hits[name], top10[name] = _evaluate(
            values, ids, gold, top3, tie_rank, return_top10=True)
    return out, hits, top10


def _model_payload(median, iqr, params, optimizer, n_fit, support_thresholds):
    return {
        "feature_names": list(FEATURE_NAMES),
        "median": np.asarray(median).tolist(),
        "iqr": np.asarray(iqr).tolist(),
        "delta": float(params[0]),
        "weights": {name: float(value) for name, value in
                    zip(FEATURE_NAMES, params[1:])},
        "params": np.asarray(params).tolist(),
        "optimizer": optimizer,
        "n_fit_queries": int(n_fit),
        "support_thresholds": np.asarray(support_thresholds).tolist(),
    }


def fit(args):
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.docs_out, exist_ok=True)
    report_path = os.path.join(args.out, "Z2_REPORT.json")
    frozen_path = os.path.join(args.out, "Z2_FROZEN.json")
    if os.path.isfile(report_path) and not args.force:
        old = _load_json(report_path)
        if old.get("stage") == "complete":
            raise SystemExit("Z2 test is already complete; use --force to refit")

    z1 = _load_json(args.z1_report)
    if z1.get("stage") != "complete" or not z1["gates"].get("y2_reproduced"):
        raise AssertionError("Z1 report is not a completed, Y2-bound input")
    if z1["selection"]["best"] != {"calibration": "raw", "beta": 8.0,
                                      "shrink_k": 20.0}:
        raise AssertionError("Z1 fixed comparator changed")

    payload = load_sharded(args.graph, mmap=True, verify_sha256=False)
    graph_data = payload["data"]
    roots_frame = payload["unique_dataset_id"].sort_values("mappedID")
    roots = roots_frame["root"].astype(str).tolist()
    tie_rank = _tie_ranks(N_TOTAL)
    report = {
        "written_at": utcnow(), "stage": "fitting",
        "protocol": {
            "K": 1000, "g0": G0, "base_beta": 8.0, "shrink_k": 20.0,
            "temperature": TEMPERATURE, "l2": L2,
            "hard_negatives": N_HARD, "folds": N_FOLDS,
            "feature_names": list(FEATURE_NAMES),
        },
        "validation": {},
        "gates": {
            "z1_complete_and_bound": True,
            "test_pools_unread_during_fit": True,
            "per_seed_models_no_cross_seed_training": True,
        },
    }
    models = {}
    input_hashes = {"z1_report": _sha256(args.z1_report)}

    for seed in SEEDS:
        val_path = os.path.join(args.z1_dir, "val_pool_s%d.npz" % seed)
        input_hashes["val_pool_s%d" % seed] = _sha256(val_path)
        pool = _load_pool(val_path)
        queries = pool["query"].astype(np.int64)
        ids = pool["model"].astype(np.int64)
        dense = pool["dense_cos"].astype(np.float32)
        prior20 = pool["prior_k20"].astype(np.float32)
        prior5 = pool["prior_k5"].astype(np.float32)
        gold = pool["gold"].astype(np.int64)
        top3 = pool["top3"].astype(np.int64)
        query_roots = pool["root"].astype(str)
        side_path = os.path.join(args.sidecar_exports,
                                 args.sidecar_run_fmt % seed,
                                 "prior_sidecar_s%d.npz" % seed)
        with np.load(side_path) as side:
            history, split = _history_for_validation(graph_data, roots, seed, side)
            query_root_ids = set(side["root_id"][queries].tolist())
        if query_root_ids & history.visible_root_ids:
            raise AssertionError("validation query root entered gate history")
        if len(np.unique(query_roots)) < N_FOLDS:
            raise AssertionError("too few validation roots for grouped CV")

        features = _query_features(queries, ids, dense, prior20, history, tie_rank)
        a, b, usable = _hard_pairs(dense, prior20, ids, gold, top3, tie_rank)
        oof_gate = np.empty(len(queries), dtype=np.float64)
        fold_rows = []
        splitter = GroupKFold(n_splits=N_FOLDS)
        for fold, (train_idx, val_idx) in enumerate(
                splitter.split(features, groups=query_roots)):
            median, iqr = _scaler(features[train_idx])
            x_train = _standardize(features[train_idx], median, iqr)
            params, optimizer = _fit_gate(
                x_train, a[train_idx], b[train_idx], usable[train_idx])
            oof_gate[val_idx] = _predict(features[val_idx], median, iqr, params)
            fold_rows.append({
                "fold": fold, "n_train": int(len(train_idx)),
                "n_validation": int(len(val_idx)),
                "n_fit_queries": int(usable[train_idx].sum()),
                "optimizer": optimizer,
            })

        fixed_score = _mix(dense, prior20, G0)
        adaptive_score = _mix(dense, prior20, oof_gate)
        production_score = _mix(dense, prior5, 0.5)
        scores = {"production": production_score, "fixed_z1": fixed_score,
                  "adaptive_oof": adaptive_score}
        rows_out, hits, _top10 = _rows(scores, ids, gold, top3, tie_rank)
        median, iqr = _scaler(features)
        final_params, optimizer = _fit_gate(
            _standardize(features, median, iqr), a, b, usable)
        support_thresholds = np.quantile(features[:, 0], [1.0 / 3.0, 2.0 / 3.0])
        models[str(seed)] = _model_payload(
            median, iqr, final_params, optimizer, usable.sum(), support_thresholds)
        report["validation"][str(seed)] = {
            "n_queries": int(len(queries)),
            "n_roots": int(len(np.unique(query_roots))),
            "n_fit_queries": int(usable.sum()),
            "n_train_edges": int(len(split["train"])),
            "rows": rows_out,
            "paired_adaptive_vs_fixed": _paired(hits["fixed_z1"],
                                                  hits["adaptive_oof"]),
            "oof_gate": _gate_summary(oof_gate),
            "folds": fold_rows,
        }
        pool.close()
        write_json_atomic(report_path, report)

    fixed_values = [report["validation"][str(s)]["rows"]["fixed_z1"]["gold@10"]
                    for s in SEEDS]
    adaptive_values = [report["validation"][str(s)]["rows"]["adaptive_oof"]["gold@10"]
                       for s in SEEDS]
    report["validation_summary"] = {
        "fixed_per_seed": fixed_values, "fixed_mean": float(np.mean(fixed_values)),
        "adaptive_per_seed": adaptive_values,
        "adaptive_mean": float(np.mean(adaptive_values)),
        "mean_improves": float(np.mean(adaptive_values)) > float(np.mean(fixed_values)),
    }
    frozen = {
        "written_at": utcnow(), "status": "frozen-before-test",
        "protocol": report["protocol"], "models": models,
        "input_sha256": input_hashes,
        "validation_summary": report["validation_summary"],
    }
    write_json_atomic(frozen_path, frozen)
    frozen_sha = _sha256(frozen_path)
    report["frozen_manifest"] = frozen_path
    report["frozen_sha256"] = frozen_sha
    report["stage"] = "frozen-before-test"
    report["written_at"] = utcnow()
    write_json_atomic(report_path, report)
    shutil.copyfile(frozen_path, os.path.join(args.docs_out, "Z2_FROZEN.json"))
    shutil.copyfile(report_path, os.path.join(args.docs_out, "Z2_REPORT.json"))
    print(json.dumps({"stage": report["stage"],
                      "validation_summary": report["validation_summary"],
                      "frozen_sha256": frozen_sha}, indent=2), flush=True)
    return report


def _stratified(features, thresholds, scores, ids, gold, top3, tie_rank):
    support = features[:, 0]
    labels = np.where(support <= thresholds[0], 0,
                      np.where(support <= thresholds[1], 1, 2))
    out = {}
    for group, name in enumerate(("low", "mid", "high")):
        keep = labels == group
        if not keep.any():
            out[name] = {"n_queries": 0}
            continue
        rows, hits, _ = _rows({key: value[keep] for key, value in scores.items()},
                              ids[keep], gold[keep], top3[keep], tie_rank)
        out[name] = {
            "n_queries": int(keep.sum()), "rows": rows,
            "paired_adaptive_vs_fixed": _paired(hits["fixed_z1"],
                                                  hits["adaptive"]),
        }
    return out


def _rerank_latency(ids, dense_cos, prior, features, median, iqr, params, tie_rank):
    elapsed = []
    for row in range(len(ids)):
        started = time.perf_counter_ns()
        gate = _predict(features[row:row + 1], median, iqr, params)[0]
        dense = (dense_cos[row].astype(np.float64) + 1.0) * 0.5
        score = (1.0 - gate) * dense + gate * prior[row].astype(np.float64)
        np.lexsort((tie_rank[ids[row]], -score))[:10]
        elapsed.append((time.perf_counter_ns() - started) / 1e6)
    return np.asarray(elapsed, dtype=np.float64)


def test(args):
    report_path = os.path.join(args.out, "Z2_REPORT.json")
    frozen_path = os.path.join(args.out, "Z2_FROZEN.json")
    report = _load_json(report_path)
    if report.get("stage") != "frozen-before-test":
        raise SystemExit("Z2 must be frozen exactly once before test")
    if _sha256(frozen_path) != report.get("frozen_sha256"):
        raise AssertionError("frozen gate manifest changed before test")
    frozen = _load_json(frozen_path)
    if frozen.get("status") != "frozen-before-test":
        raise AssertionError("invalid frozen manifest")
    tie_rank = _tie_ranks(N_TOTAL)
    sizes_frame = pd.read_parquet(args.model_meta).sort_values("mappedID")
    if not np.array_equal(sizes_frame["mappedID"].to_numpy(), np.arange(N_TOTAL)):
        raise AssertionError("model size metadata is not mappedID-aligned")
    sizes = sizes_frame["size_b"].to_numpy(dtype=float)
    test_rows = {}

    for seed in SEEDS:
        pool_path = os.path.join(args.z1_dir, "test_pool_s%d.npz" % seed)
        pool = _load_pool(pool_path)
        queries = pool["query"].astype(np.int64)
        ids = pool["model"].astype(np.int64)
        dense = pool["dense_cos"].astype(np.float32)
        prior20 = pool["prior_k20"].astype(np.float32)
        prior5 = pool["prior_k5"].astype(np.float32)
        gold = pool["gold"].astype(np.int64)
        top3 = pool["top3"].astype(np.int64)
        hnsw_ms = pool["hnsw_ms"].astype(np.float64)
        side_path = os.path.join(args.sidecar_exports,
                                 args.sidecar_run_fmt % seed,
                                 "prior_sidecar_s%d.npz" % seed)
        with np.load(side_path) as side:
            history = _history_for_test(side)
            query_root_ids = set(side["root_id"][queries].tolist())
        if query_root_ids & history.visible_root_ids:
            raise AssertionError("test query root entered gate history")
        features = _query_features(queries, ids, dense, prior20, history, tie_rank)
        model = frozen["models"][str(seed)]
        median = np.asarray(model["median"], dtype=np.float64)
        iqr = np.asarray(model["iqr"], dtype=np.float64)
        params = np.asarray(model["params"], dtype=np.float64)
        gate = _predict(features, median, iqr, params)
        scores = {
            "production": _mix(dense, prior5, 0.5),
            "fixed_z1": _mix(dense, prior20, G0),
            "adaptive": _mix(dense, prior20, gate),
        }
        rows_out, hits, top10 = _rows(scores, ids, gold, top3, tie_rank)
        if not math.isclose(rows_out["fixed_z1"]["gold@10"],
                            EXPECTED_Z1_FIXED[seed], rel_tol=0, abs_tol=1e-12):
            raise AssertionError("Z2 fixed comparator does not reproduce Z1")
        if not math.isclose(rows_out["production"]["gold@10"],
                            EXPECTED_PRODUCTION[seed], rel_tol=0, abs_tol=1e-12):
            raise AssertionError("Z2 production reference does not reproduce Y2")
        latency = _rerank_latency(ids, dense, prior20, features, median, iqr,
                                  params, tie_rank)
        end_to_end = hnsw_ms + latency
        size = {name: _size_stats(top10[name], sizes) for name in top10}
        test_rows[str(seed)] = {
            "n_queries": int(len(queries)), "rows": rows_out,
            "paired_adaptive_vs_fixed": _paired(hits["fixed_z1"],
                                                  hits["adaptive"]),
            "paired_adaptive_vs_production": _paired(hits["production"],
                                                       hits["adaptive"]),
            "gate": _gate_summary(gate),
            "support_strata": _stratified(
                features, np.asarray(model["support_thresholds"]), scores,
                ids, gold, top3, tie_rank),
            "size": size,
            "latency_ms": {
                "hnsw_p50": float(np.percentile(hnsw_ms, 50)),
                "hnsw_p95": float(np.percentile(hnsw_ms, 95)),
                "gate_rerank_p50": float(np.percentile(latency, 50)),
                "gate_rerank_p95": float(np.percentile(latency, 95)),
                "end_to_end_p50": float(np.percentile(end_to_end, 50)),
                "end_to_end_p95": float(np.percentile(end_to_end, 95)),
            },
            "history_boundary_ok": True,
            "input_sha256": _sha256(pool_path),
        }
        pool.close()

    report["test"] = test_rows
    fixed = [test_rows[str(s)]["rows"]["fixed_z1"]["gold@10"] for s in SEEDS]
    adaptive = [test_rows[str(s)]["rows"]["adaptive"]["gold@10"] for s in SEEDS]
    production = [test_rows[str(s)]["rows"]["production"]["gold@10"] for s in SEEDS]
    val_improves = bool(report["validation_summary"]["mean_improves"])
    no_seed_down = all(adaptive[i] >= fixed[i] for i in range(len(SEEDS)))
    rescue_ok = all(test_rows[str(s)]["paired_adaptive_vs_fixed"]["rescue"] >
                    test_rows[str(s)]["paired_adaptive_vs_fixed"]["harm"]
                    for s in SEEDS)
    latency_ok = all(test_rows[str(s)]["latency_ms"]["end_to_end_p50"] <= 5.0
                     and test_rows[str(s)]["latency_ms"]["end_to_end_p95"] <= 10.0
                     for s in SEEDS)
    size_ok = True
    for seed in SEEDS:
        fixed_size = test_rows[str(seed)]["size"]["fixed_z1"]
        adaptive_size = test_rows[str(seed)]["size"]["adaptive"]
        size_ok &= (adaptive_size["median_size_b"] <=
                    2.0 * fixed_size["median_size_b"])
        size_ok &= (adaptive_size["missing_rate"] <=
                    fixed_size["missing_rate"] + 0.05)
    gates = {
        "validation_oof_mean_improves": val_improves,
        "test_no_seed_down_vs_fixed": no_seed_down,
        "test_rescue_gt_harm_each_seed_vs_fixed": rescue_ok,
        "test_mean_above_z1_fixed": float(np.mean(adaptive)) > EXPECTED_Z1_FIXED_MEAN,
        "latency_budget": latency_ok,
        "size_bias_gate": bool(size_ok),
        "z1_and_production_reproduced": True,
        "frozen_before_test": True,
        "test_history_train_val_only": True,
    }
    report["gates"].update(gates)
    passed = all(gates.values())
    report["decision"] = {
        "adaptive_gate_effective": passed,
        "keep_production_config": not passed,
        "stop_before_direction_three": not passed,
        "production": {"per_seed": production, "mean": float(np.mean(production))},
        "fixed_z1": {"per_seed": fixed, "mean": float(np.mean(fixed))},
        "adaptive": {"per_seed": adaptive, "mean": float(np.mean(adaptive))},
    }
    report["environment"] = {"platform": platform.platform(),
                             "python": platform.python_version(),
                             "logical_cpus": os.cpu_count()}
    report["stage"] = "complete"
    report["written_at"] = utcnow()
    write_json_atomic(report_path, report)
    shutil.copyfile(report_path, os.path.join(args.docs_out, "Z2_REPORT.json"))
    print(json.dumps({"decision": report["decision"], "gates": report["gates"]},
                     indent=2), flush=True)
    return report


def main(argv=None):
    data = os.path.join(data_root(), "data1m")
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("fit", "test"), required=True)
    parser.add_argument("--z1-dir", default=os.path.join(data, "metrics_z1"))
    parser.add_argument("--z1-report", default=os.path.join(
        data, "metrics_z1", "Z1_REPORT.json"))
    parser.add_argument("--graph", default=os.path.join(data, "graphs", "hgraph_rf"))
    parser.add_argument("--sidecar-exports", default=os.path.join(data, "exports_rf"))
    parser.add_argument("--sidecar-run-fmt", default="RF_full_s%d_e25")
    parser.add_argument("--model-meta", default=os.path.join(
        data, "utility_rf", "model_meta.parquet"))
    parser.add_argument("--out", default=os.path.join(data, "metrics_z2"))
    parser.add_argument("--docs-out", default=os.path.join(
        _REPO_ROOT, "docs", "1M", "Z2_runs"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    return fit(args) if args.stage == "fit" else test(args)


if __name__ == "__main__":
    main()
