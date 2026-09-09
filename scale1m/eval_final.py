"""Evaluate the frozen 3M retrieval system on its final online path only.

For each held-out query, the evaluator retrieves 1,000 model IDs from the
split-specific inner-product HNSW index, adds the split-safe task prior to the
rescaled dense score, and returns ten recommendations.  It intentionally does
not run alternative scorers, exhaustive retrieval, or candidate-depth sweeps.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from scale1m.paths import data_root


SEEDS = (0, 1, 2)
N_MODELS = 3_016_439
N_DATASETS = 18_729
EMBEDDING_DIM = 128
POOL_SIZE = 1_000
RETURN_SIZE = 10
SHRINKAGE = 5.0
PRIOR_WEIGHT = 1.0
EXPECTED_QUERIES = {0: 1_476, 1: 1_101, 2: 1_545}
EXPECTED_VISIBLE_EDGES = {0: 198_216, 1: 196_912, 2: 196_124}
EXPECTED_HELD_OUT_DATASETS = {0: 4_290, 1: 3_112, 2: 3_628}
EF_SEARCH = {0: 1_000, 1: 1_500, 2: 1_500}
GAP_DELTA = 0.01


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _recommendation_digest(matrix: np.ndarray) -> str:
    """Hash logical rows, independent of NumPy container metadata."""
    canonical = np.asarray(matrix, dtype="<i8", order="C")
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(canonical[:, 0]).tobytes(order="C"))
    digest.update(np.ascontiguousarray(canonical[:, 1:]).tobytes(order="C"))
    return digest.hexdigest()


def _first_existing(directory: Path, *names: str) -> Path:
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        "none of the required files exists under %s: %s"
        % (directory, ", ".join(names)))


def _row_map(path: Path, entity: str) -> tuple[np.ndarray, pd.DataFrame]:
    frame = pd.read_parquet(path).sort_values("mappedID").reset_index(drop=True)
    expected = np.arange(len(frame), dtype=np.int64)
    if not np.array_equal(frame["mappedID"].to_numpy(), expected):
        raise AssertionError("%s mappedID is not contiguous" % entity)
    if entity not in frame:
        raise AssertionError("%s row map has no %s column" % (path, entity))
    return frame[entity].astype(str).to_numpy(), frame


def _eligible_queries(dataset_nodes: Path, dataset_frame: pd.DataFrame,
                      candidate_keys: set[int], seed: int) -> list[int]:
    nodes = pd.read_parquet(dataset_nodes, columns=["node", "gold_eligible"])
    nodes["node"] = nodes["node"].astype(str)
    eligible_by_node = nodes.set_index("node")["gold_eligible"]
    names = dataset_frame["dataset"].astype(str)
    eligible = eligible_by_node.reindex(names).fillna(False).to_numpy(dtype=bool)
    queries = sorted(query for query in candidate_keys if eligible[query])
    if len(queries) != EXPECTED_QUERIES[seed]:
        raise AssertionError(
            "seed %d has %d eligible queries; expected %d"
            % (seed, len(queries), EXPECTED_QUERIES[seed]))
    return queries


def _gold_candidates(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    with np.load(path) as payload:
        candidates = {
            int(key): (payload[key][0].astype(np.int64),
                       payload[key][1].astype(np.float64))
            for key in payload.files
        }
    for query, (models, values) in candidates.items():
        if not (0 <= query < N_DATASETS):
            raise AssertionError("candidate query is outside the dataset row map")
        if not len(models) or len(models) != len(values):
            raise AssertionError("query %d has malformed candidates" % query)
        if models.min(initial=0) < 0 or models.max(initial=0) >= N_MODELS:
            raise AssertionError("query %d has an out-of-range model ID" % query)
        if not np.isfinite(values).all():
            raise AssertionError("query %d has non-finite gold values" % query)
    return candidates


def _stable_tie_keys(n: int) -> np.ndarray:
    """Unique, label-free secondary keys used by the deployed reranker."""
    ids = np.arange(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        keys = (ids * np.uint64(11400714819323198485)
                + np.uint64(0xD1B54A32D192ED03))
    if len(np.unique(keys)) != n:
        raise AssertionError("the fixed tie-break is not a permutation")
    return keys


class TaskPrior:
    """Split-specific task evidence, reduced to sorted model/boost tables."""

    def __init__(self, payload_path: Path, metadata_path: Path, seed: int,
                 query_ids: list[int], dataset_roots: np.ndarray):
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        expected = {
            "split_seed": seed,
            "n_models": N_MODELS,
            "n_datasets": N_DATASETS,
            "n_edges": EXPECTED_VISIBLE_EDGES[seed],
            "held_out_datasets": EXPECTED_HELD_OUT_DATASETS[seed],
        }
        for key, value in expected.items():
            if int(metadata[key]) != int(value):
                raise AssertionError(
                    "task-prior metadata %s=%r; expected %r"
                    % (key, metadata[key], value))

        with np.load(payload_path) as payload:
            task_id = payload["task_id"].astype(np.int64)
            root_id = payload["root_id"].astype(np.int64)
            edge_model = payload["edge_model"].astype(np.int64)
            edge_dataset = payload["edge_dataset"].astype(np.int64)
            edge_value = payload["edge_acc"].astype(np.float64)
        if len(task_id) != N_DATASETS or len(root_id) != N_DATASETS:
            raise AssertionError("task-prior dataset mapping has the wrong length")
        if (edge_model.min(initial=0) < 0
                or edge_model.max(initial=0) >= N_MODELS
                or edge_dataset.min(initial=0) < 0
                or edge_dataset.max(initial=0) >= N_DATASETS):
            raise AssertionError("task-prior edge has an out-of-range row ID")
        if not np.isfinite(edge_value).all():
            raise AssertionError("task-prior edge has a non-finite value")

        root_frame = pd.DataFrame({
            "root": dataset_roots.astype(str),
            "code": root_id,
        })
        if (root_frame.groupby("root")["code"].nunique().max() != 1
                or root_frame.groupby("code")["root"].nunique().max() != 1):
            raise AssertionError("task-prior root IDs disagree with the row map")
        visible_roots = np.unique(root_id[edge_dataset])
        held_out_roots = np.unique(root_id[np.asarray(query_ids, dtype=np.int64)])
        if np.intersect1d(visible_roots, held_out_roots).size:
            raise AssertionError("held-out query evidence leaked into task prior")

        grouped = pd.DataFrame({
            "task": task_id[edge_dataset],
            "model": edge_model,
            "value": edge_value,
        }).groupby(["task", "model"], sort=True)["value"].agg(["sum", "count"])
        boost = ((grouped["sum"] + 0.5 * SHRINKAGE)
                 / (grouped["count"] + SHRINKAGE)).to_numpy(dtype=np.float64)
        tasks = grouped.index.get_level_values(0).to_numpy(dtype=np.int64)
        models = grouped.index.get_level_values(1).to_numpy(dtype=np.int64)
        unique_tasks, starts = np.unique(tasks, return_index=True)
        ends = np.append(starts[1:], len(tasks))
        self.task_id = task_id
        self.by_task = {
            int(task): (models[start:end], boost[start:end])
            for task, start, end in zip(unique_tasks, starts, ends)
        }

    def values(self, query: int, model_ids: np.ndarray) -> np.ndarray:
        model_ids = np.asarray(model_ids, dtype=np.int64)
        known_models, known_values = self.by_task.get(
            int(self.task_id[int(query)]),
            (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)))
        result = np.zeros(model_ids.shape, dtype=np.float32)
        if known_models.size:
            positions = np.searchsorted(known_models, model_ids)
            clipped = np.minimum(positions, known_models.size - 1)
            hit = ((positions < known_models.size)
                   & (known_models[clipped] == model_ids))
            result[hit] = known_values[clipped[hit]].astype(np.float32)
        return result


def _root_macro(flags: np.ndarray, queries: list[int],
                roots: np.ndarray) -> float:
    groups: dict[str, list[float]] = defaultdict(list)
    for query, flag in zip(queries, flags):
        groups[str(roots[query])].append(float(flag))
    return float(np.mean([np.mean(values) for values in groups.values()]))


def _metrics(recommendations: np.ndarray, queries: list[int],
             candidates: dict[int, tuple[np.ndarray, np.ndarray]],
             roots: np.ndarray) -> dict[str, float | int]:
    gold_at_1 = np.zeros(len(queries), dtype=np.float64)
    gold_at_10 = np.zeros(len(queries), dtype=np.float64)
    top3_at_10 = np.zeros(len(queries), dtype=np.float64)
    gap_at_10 = np.zeros(len(queries), dtype=np.float64)
    for row, query in enumerate(queries):
        models, values = candidates[query]
        gold = int(models[int(np.argmax(values))])
        top3 = models[np.argsort(-values)[:min(3, len(models))]].astype(np.int64)
        near = models[values >= values.max() - GAP_DELTA].astype(np.int64)
        returned = recommendations[row]
        gold_at_1[row] = float(returned[0] == gold)
        gold_at_10[row] = float(np.any(returned == gold))
        top3_at_10[row] = float(np.intersect1d(returned, top3).size > 0)
        gap_at_10[row] = float(np.intersect1d(returned, near).size > 0)
    return {
        "queries": int(len(queries)),
        "gold@1": float(gold_at_1.mean()),
        "gold@10": float(gold_at_10.mean()),
        "top3@10": float(top3_at_10.mean()),
        "gold-gap@10": float(gap_at_10.mean()),
        "root-gold@10": _root_macro(gold_at_10, queries, roots),
    }


def _build_index(index_path: Path, model_embeddings: Path,
                 construction_threads: int) -> float:
    import hnswlib

    vectors = np.load(model_embeddings, mmap_mode="r")
    if vectors.shape != (N_MODELS, EMBEDDING_DIM):
        raise AssertionError("model embeddings have shape %r" % (vectors.shape,))
    started = time.time()
    index = hnswlib.Index(space="ip", dim=EMBEDDING_DIM)
    index.init_index(max_elements=N_MODELS, ef_construction=200, M=32)
    index.add_items(vectors, np.arange(N_MODELS, dtype=np.int64),
                    num_threads=construction_threads)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index.save_index(os.fspath(index_path))
    return time.time() - started


def _evaluate_seed(seed: int, artifact_root: Path, dataset_nodes: Path,
                   output_dir: Path, tie_keys: np.ndarray,
                   build_missing_index: bool, construction_threads: int,
                   warmup_queries: int) -> tuple[dict, dict]:
    import hnswlib

    directory = artifact_root / ("seed_%d" % seed)
    model_map_path = _first_existing(directory, "model_ids.parquet")
    dataset_map_path = _first_existing(directory, "dataset_ids.parquet")
    candidate_path = _first_existing(directory, "gold_candidates.npz",
                                     "gold_cands.npz")
    query_embedding_path = _first_existing(directory, "query_embeddings.npy",
                                           "z_d_eval.npy")
    prior_path = _first_existing(directory, "task_prior.npz",
                                 "prior_sidecar_s%d.npz" % seed)
    prior_metadata_path = _first_existing(directory, "task_prior_meta.json",
                                          "prior_sidecar_s%d_meta.json" % seed)
    index_path = directory / "hnsw_top1000.bin"

    models, _model_frame = _row_map(model_map_path, "model")
    datasets, dataset_frame = _row_map(dataset_map_path, "dataset")
    if len(models) != N_MODELS or len(datasets) != N_DATASETS:
        raise AssertionError(
            "row-map sizes are %d models and %d datasets"
            % (len(models), len(datasets)))
    del models, datasets, _model_frame

    candidates = _gold_candidates(candidate_path)
    queries = _eligible_queries(dataset_nodes, dataset_frame,
                                set(candidates), seed)
    roots = dataset_frame["root"].astype(str).to_numpy()
    prior = TaskPrior(prior_path, prior_metadata_path, seed, queries, roots)
    query_embeddings = np.load(query_embedding_path, mmap_mode="r")
    if query_embeddings.shape != (N_DATASETS, EMBEDDING_DIM):
        raise AssertionError(
            "query embeddings have shape %r" % (query_embeddings.shape,))

    build_seconds = 0.0
    if not index_path.is_file():
        if not build_missing_index:
            raise FileNotFoundError(
                "%s is missing; download the frozen result bundle first"
                % index_path)
        model_embedding_path = _first_existing(
            directory, "model_embeddings.npy", "z_m_eval.npy")
        build_seconds = _build_index(
            index_path, model_embedding_path, construction_threads)

    index = hnswlib.Index(space="ip", dim=EMBEDDING_DIM)
    index.load_index(os.fspath(index_path), max_elements=N_MODELS)
    if index.get_current_count() != N_MODELS:
        raise AssertionError("HNSW index contains the wrong number of models")
    index.set_num_threads(1)
    index.set_ef(EF_SEARCH[seed])
    query_matrix = np.asarray(query_embeddings[queries], dtype=np.float32)
    for row in range(min(warmup_queries, len(queries))):
        index.knn_query(query_matrix[row:row + 1], k=POOL_SIZE,
                        num_threads=1)

    recommendations = np.empty((len(queries), RETURN_SIZE), dtype=np.int64)
    retrieval_ms: list[float] = []
    reranking_ms: list[float] = []
    for row, query in enumerate(queries):
        started = time.perf_counter_ns()
        model_ids, distances = index.knn_query(
            query_matrix[row:row + 1], k=POOL_SIZE, num_threads=1)
        retrieved = time.perf_counter_ns()
        model_ids = model_ids[0].astype(np.int64, copy=False)
        if len(np.unique(model_ids)) != POOL_SIZE:
            raise AssertionError("HNSW returned duplicate candidates")
        dense_score = (1.0 - distances[0]).astype(np.float32, copy=False)
        fused_score = (((dense_score + 1.0) * 0.5)
                       + PRIOR_WEIGHT * prior.values(query, model_ids))
        order = np.lexsort((tie_keys[model_ids], -fused_score))
        recommendations[row] = model_ids[order[:RETURN_SIZE]]
        reranked = time.perf_counter_ns()
        retrieval_ms.append((retrieved - started) / 1e6)
        reranking_ms.append((reranked - retrieved) / 1e6)

    output_dir.mkdir(parents=True, exist_ok=True)
    recommendation_path = output_dir / ("recommendations.seed_%d.npy" % seed)
    result_matrix = np.column_stack((np.asarray(queries, dtype=np.int64),
                                     recommendations))
    with recommendation_path.open("wb") as handle:
        np.save(handle, result_matrix, allow_pickle=False)
    metrics = _metrics(recommendations, queries, candidates, roots)
    total_ms = np.asarray(retrieval_ms) + np.asarray(reranking_ms)
    row = {
        "seed": seed,
        "ef_search": EF_SEARCH[seed],
        "metrics": metrics,
        "recommendations_sha256": _recommendation_digest(result_matrix),
        "index_sha256": _sha256(index_path),
        "latency_ms": {
            "retrieval_p50": float(np.percentile(retrieval_ms, 50)),
            "reranking_p50": float(np.percentile(reranking_ms, 50)),
            "end_to_end_p50": float(np.percentile(total_ms, 50)),
            "end_to_end_p95": float(np.percentile(total_ms, 95)),
        },
        "index_build_seconds": float(build_seconds),
    }
    bindings = {
        "model_map_sha256": _sha256(model_map_path),
        "dataset_map_sha256": _sha256(dataset_map_path),
    }
    return row, bindings


def evaluate(args: argparse.Namespace) -> dict:
    artifact_root = Path(args.artifacts).resolve()
    dataset_nodes = (Path(args.dataset_nodes).resolve()
                     if args.dataset_nodes
                     else artifact_root / "dataset_nodes.parquet")
    output_dir = Path(args.out).resolve()
    tie_keys = _stable_tie_keys(N_MODELS)
    per_seed: list[dict] = []
    bindings: list[dict] = []
    started = time.time()
    for seed in SEEDS:
        print("[final] evaluating split seed %d" % seed, flush=True)
        row, binding = _evaluate_seed(
            seed, artifact_root, dataset_nodes, output_dir, tie_keys,
            args.build_missing_index, args.construction_threads,
            args.warmup_queries)
        per_seed.append(row)
        bindings.append(binding)
        print("[final] seed %d gold@10=%.10f" %
              (seed, row["metrics"]["gold@10"]), flush=True)

    if len({row["model_map_sha256"] for row in bindings}) != 1:
        raise AssertionError("model row maps differ across seeds")
    if len({row["dataset_map_sha256"] for row in bindings}) != 1:
        raise AssertionError("dataset row maps differ across seeds")

    metric_names = ("gold@1", "gold@10", "top3@10", "gold-gap@10",
                    "root-gold@10")
    summary = {
        name: {
            "per_seed": [float(row["metrics"][name]) for row in per_seed],
            "mean": float(np.mean([
                row["metrics"][name] for row in per_seed])),
        }
        for name in metric_names
    }
    report = {
        "schema_version": 1,
        "written_at": _utcnow(),
        "system": {
            "candidate_models": N_MODELS,
            "embedding_dimensions": EMBEDDING_DIM,
            "retrieval": "inner-product HNSW",
            "candidate_pool": POOL_SIZE,
            "reranking_score": "(cosine + 1) / 2 + split-safe task prior",
            "returned_models": RETURN_SIZE,
            "prior_shrinkage": SHRINKAGE,
            "prior_weight": PRIOR_WEIGHT,
        },
        "evaluation": {
            "split_policy": "root-disjoint held-out splits",
            "eligible_queries": [EXPECTED_QUERIES[seed] for seed in SEEDS],
            "gold@10_definition": (
                "whether the empirically best eligible model is among the "
                "ten returned models"),
        },
        "integrity": {
            "candidate_count": True,
            "contiguous_row_maps": True,
            "cross_seed_row_maps": True,
            "eligible_query_counts": True,
            "split_safe_task_prior": True,
            "unique_hnsw_candidates": True,
        },
        "per_seed": per_seed,
        "summary": summary,
        "runtime": {
            "elapsed_seconds": time.time() - started,
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "latency_boundary": (
                "single-query, one-thread HNSW plus reranking after warm-up; "
                "query encoding, index loading, and index construction excluded"),
        },
    }
    _write_json_atomic(output_dir / "FINAL_RESULTS.json", report)
    print(json.dumps({
        "gold@10": summary["gold@10"],
        "report": os.fspath(output_dir / "FINAL_RESULTS.json"),
    }, indent=2), flush=True)
    return report


def main(argv=None) -> dict:
    default_data = Path(data_root()) / "data1m"
    parser = argparse.ArgumentParser(
        description="Evaluate only the final 3M HNSW-plus-reranking path")
    parser.add_argument("--artifacts", type=Path,
                        default=default_data / "final_system")
    parser.add_argument("--dataset-nodes", type=Path, default=None)
    parser.add_argument("--out", type=Path,
                        default=default_data / "reproduced" / "final")
    parser.add_argument("--build-missing-index", action="store_true")
    parser.add_argument("--construction-threads", type=int, default=8)
    parser.add_argument("--warmup-queries", type=int, default=50)
    args = parser.parse_args(argv)
    return evaluate(args)


if __name__ == "__main__":
    main()
