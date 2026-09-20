"""Evaluate a newly collected HF lake using its own graph, splits and labels.

This reproduces the retrieval experiment on the collected evidence. It does not
assert that today's changing Hub equals the historical A0 snapshot.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import a0_evaluation as A
from . import eval_y2 as E
from . import evaluate_a0_portable as P

RUN_FMT = "LIVE_full_s%d_e25"
RESULTS = "LIVE_RESULTS.json"
MANIFEST = "LIVE_MANIFEST.json"
require = P.require


def validate_graph(directory):
    from .prepare_a0_graph import verify_files, _digest
    from .build_graph_rf import A0_ZERO_COLUMNS, dataset_stats

    directory = Path(directory).resolve()
    meta, files = verify_files(directory)
    require("a0" not in meta and "A0_FEATURE_REPAIR.json" not in files,
            "Live evaluation requires a newly built graph, not historical A0 provenance")
    nm, nd = (int(meta["num_nodes"][key]) for key in ("model", "dataset"))
    require(nm >= E.POOL_K and nd > 0, "Live graph requires >=1000 models and dataset nodes")
    for name, shape in (("x_model.npy", (nm, 448)), ("x_dataset.npy", (nd, 458))):
        values = np.load(directory / name, mmap_mode="r", allow_pickle=False)
        require(values.shape == shape and values.dtype == np.float32,
                "Graph feature shape or dtype differs from metadata: " + name)
        for start in range(0, len(values), 50_000):
            require(np.isfinite(values[start:start + 50_000]).all(), "Nonfinite graph features: " + name)
    models, _ = E._ids(directory / "unique_model_id.parquet", "live models")
    datasets, frame = E._ids(directory / "unique_dataset_id.parquet", "live datasets")
    require(len(models) == nm and len(datasets) == nd and len(np.unique(models)) == nm
            and len(np.unique(datasets)) == nd, "Graph row maps differ from dimensions or contain duplicates")
    stats, roots = dataset_stats(pd.DataFrame({"dataset": frame.dataset.astype(str).str.split("\t", n=1).str[0]}), None)
    xd = np.load(directory / "x_dataset.npy", mmap_mode="r", allow_pickle=False)
    require(np.all(xd[:, A0_ZERO_COLUMNS] == 0) and np.all(xd[:, 456:458] == 0),
            "Performance-derived or reserved dataset feature columns are nonzero")
    require(roots == frame.root.astype(str).tolist() and np.array_equal(xd[:, 448:], stats),
            "Graph root identities or structural features differ from its dataset nodes")
    with np.load(directory / "edges.npz", allow_pickle=False) as edges:
        edge = edges["model__trained_on__dataset__edge_index"]
        weights = edges["model__trained_on__dataset__edge_attr"]
        require(edge.ndim == 2 and edge.shape[0] == 2 and edge.shape[1] > 0
                and weights.shape == (edge.shape[1],) and np.issubdtype(edge.dtype, np.integer),
                "Invalid graph performance edge shape or dtype")
        require(edge[0].min() >= 0 and edge[0].max() < nm and edge[1].min() >= 0
                and edge[1].max() < nd and np.isfinite(weights).all(),
                "Graph performance edges are outside the row maps or nonfinite")
        require(np.array_equal(edges["dataset__rev_trained_on__model__edge_index"], edge[::-1])
                and np.array_equal(edges["dataset__rev_trained_on__model__edge_attr"], weights),
                "Forward and reverse performance edges differ")
    require(len(np.unique(edge[0] * nd + edge[1])) == edge.shape[1], "Duplicate graph performance edges")
    return {"directory": str(directory), "graph_digest": _digest(files), "files": files,
            "meta": meta, "models": models, "datasets": datasets, "roots": np.asarray(roots),
            "edge": edge.astype(np.int64), "weights": weights.astype(np.float32),
            "n_models": nm, "n_datasets": nd, "n_performance_edges": edge.shape[1]}


def expected_split(graph, seed):
    import torch
    from torch_geometric.data import HeteroData
    from ModelLakeFishing.stage2TrainGraphSAGE.d0_splits import make_root_aware_splits
    from ModelLakeFishing.stage2TrainGraphSAGE.losses import TRAINED_ON, REV_TRAINED_ON, accuracy_lookup
    from ModelLakeFishing.stage2TrainGraphSAGE.top1_audit import candidates

    skeleton = HeteroData()
    skeleton["model"].num_nodes = graph["n_models"]
    skeleton["dataset"].num_nodes = graph["n_datasets"]
    skeleton[TRAINED_ON].edge_index = torch.from_numpy(graph["edge"])
    skeleton[TRAINED_ON].edge_attr = torch.from_numpy(graph["weights"])
    skeleton[REV_TRAINED_ON].edge_index = skeleton[TRAINED_ON].edge_index.flip(0)
    skeleton[REV_TRAINED_ON].edge_attr = skeleton[TRAINED_ON].edge_attr.clone()
    _, _, test = make_root_aware_splits(skeleton, graph["roots"].tolist(), split_seed=seed, neg_ratio=0.)
    all_candidates = candidates(test, accuracy_lookup(skeleton))
    heldout = np.unique(test[TRAINED_ON].edge_label_index[1].numpy())
    return {"candidates": all_candidates, "heldout": heldout,
            "edge": test[TRAINED_ON].edge_index.numpy(),
            "weights": test[TRAINED_ON].edge_attr.numpy()}


class LiveTaskPrior(E.TaskPrior):

    def __init__(self, path, meta, graph, split, nodes):
        self.payload = np.load(path, allow_pickle=False)
        self.meta = meta
        try:
            P.check_prior(self.payload, meta, split["heldout"], graph["roots"],
                          graph["datasets"], nodes, graph["n_models"])
            actual = np.stack([self.payload["edge_model"], self.payload["edge_dataset"]])
            require(np.array_equal(actual, split["edge"])
                    and np.array_equal(self.payload["edge_acc"], split["weights"]),
                    "Prior is not exactly the train+validation evidence of this split")
            require(meta["held_out_datasets"] == len(split["heldout"])
                    and meta["n_edges_in_graph"] == graph["n_performance_edges"],
                    "Prior held-out or source-edge counts differ from the actual split")
            self.task_id = self.payload["task_id"].astype(np.int64)
            self.root_id = self.payload["root_id"].astype(np.int64)
            self.by_task, _, _ = E._prior_tables(self.payload)
        except BaseException:
            self.payload.close()
            raise


def validate_seed(args, seed, graph, nodes):
    folder = Path(args.exports) / (RUN_FMT % seed)
    manifest = P.read_json(folder / "EXPORT_MANIFEST.json")
    meta = manifest["stages"]["embed"]
    require(not meta.get("a0"), "Live exports cannot claim historical A0 provenance")
    nm, nd = graph["n_models"], graph["n_datasets"]
    require(meta.get("split_seed") == seed and meta.get("checkpoint_epoch") == 24
            and meta.get("n_models") == nm and meta.get("n_datasets") == nd
            and meta.get("rung") == "live" and meta.get("expect_n") == nm,
            "Export does not describe this complete 25-epoch live-lake run")
    require(meta.get("graph_digest") == graph["graph_digest"]
            == meta.get("binding", {}).get("graph_sha256"), "Export graph bindings disagree")
    checkpoint = P.validate_rebuilt_checkpoint(meta, graph, seed)
    A.verify_chunked_export(meta, n_models=nm)
    require(all(gate.get("ok") is not False for gate in meta.get("gates", [])), "An export producer gate failed")
    names = ("z_m_eval.npy", "z_d_eval.npy", "gold_cands.npz", "model_ids.parquet", "dataset_ids.parquet")
    hashes = meta.get("artifact_hashes", {})
    require(set(names) <= set(hashes), "Export producer hashes are incomplete")
    records = {name: P.file_record(folder / name, expected=hashes[name]) for name in names}
    records["EXPORT_MANIFEST.json"] = P.file_record(folder / "EXPORT_MANIFEST.json")
    models, _ = E._ids(folder / "model_ids.parquet", "export models")
    datasets, frame = E._ids(folder / "dataset_ids.parquet", "export datasets")
    require(np.array_equal(models, graph["models"]) and np.array_equal(datasets, graph["datasets"])
            and np.array_equal(frame.root.astype(str), graph["roots"]), "Export and graph row identities differ")
    P.check_embedding(folder / "z_m_eval.npy", nm)
    P.check_embedding(folder / "z_d_eval.npy", nd)
    all_queries = P.check_gold(folder / "gold_cands.npz", nm, nd)
    split = expected_split(graph, seed)
    require(np.array_equal(all_queries, sorted(split["candidates"]))
            and len(all_queries) == meta.get("n_test_queries"), "Export query cohort differs from the root-aware split")
    with np.load(folder / "gold_cands.npz", allow_pickle=False) as gold:
        for query, (ids, values) in split["candidates"].items():
            require(np.array_equal(gold[str(query)][0], ids)
                    and np.array_equal(gold[str(query)][1], values), "Export gold labels differ from graph/split evidence")
    eligible, _ = E.query_eligibility(str(args.dataset_nodes), str(folder))
    candidates = {q: v for q, v in split["candidates"].items() if eligible[q]}
    require(bool(candidates), f"Split {seed} has no eligible queries in this collection; evaluation is undefined")
    prior_path = folder / f"prior_sidecar_s{seed}.npz"
    prior_meta_path = folder / f"prior_sidecar_s{seed}_meta.json"
    prior_meta = P.read_json(prior_meta_path)
    require(not prior_meta.get("a0") and prior_meta.get("split_seed") == seed
            and prior_meta.get("n_models") == nm and prior_meta.get("n_datasets") == nd
            and prior_meta.get("graph_digest") == graph["graph_digest"], "Prior producer graph/split binding differs")
    require(isinstance(prior_meta.get("sidecar_sha256"), str), "Prior producer hash is missing")
    records[prior_path.name] = P.file_record(prior_path, expected=prior_meta["sidecar_sha256"])
    records[prior_meta_path.name] = P.file_record(prior_meta_path)
    prior = LiveTaskPrior(prior_path, prior_meta, graph, split, nodes)
    return {"folder": folder, "prior": prior, "candidates": candidates, "roots": graph["roots"]}, {
        "directory": RUN_FMT % seed, "files": records, "checkpoint": checkpoint,
        "graph_digest": graph["graph_digest"], "eligible_queries": len(candidates),
        "all_test_queries": len(all_queries), "heldout_datasets": len(split["heldout"])}


def score_seed(bundle, args, seed, nm, tie_rank):
    import hnswlib

    exact = E.evaluate_exact_seed(str(bundle["folder"]), bundle["prior"], bundle["candidates"],
        bundle["roots"], tie_rank, args.device, model_chunk=50_000, query_chunk=16,
        protocol="live", n_universe=nm)
    queries = np.asarray(exact["queries"], dtype=np.int64)
    labels = A.label_arrays(queries, bundle["candidates"], bundle["roots"], bundle["prior"], n_models=nm)
    row, arrays = A.pool_arrays(exact["exact_pool_ids"], exact["exact_pool_scores"], queries,
        bundle["candidates"], bundle["roots"], bundle["prior"], tie_rank, n_models=nm)
    exact["rows"]["G_exact1000_task"].update(row)
    independent, _ = E._top10_metrics(exact["full_top10"], queries.tolist(), bundle["candidates"],
                                     bundle["roots"], n_universe=nm)
    for metric in ("gold@1", "gold@10", "gold-gap@10", "top3@10", "root_gold@10"):
        require(independent[metric] == exact["rows"]["G_full_task"][metric], "Full exact top10/rank disagreement")
    arrays.update(labels, seed=np.asarray(seed), full_top10=exact["full_top10"],
                  full_counts=exact["full_counts"], dense_counts=exact["dense_counts"])
    A.save_npz(args.out, f"live_exact_s{seed}.npz", **arrays)
    index = hnswlib.Index(space="ip", dim=128)
    zm = np.load(bundle["folder"] / "z_m_eval.npy", mmap_mode="r", allow_pickle=False)
    start = time.perf_counter_ns()
    index.init_index(max_elements=nm, ef_construction=200, M=32)
    index.add_items(zm, np.arange(nm, dtype=np.int64), num_threads=8)
    path = Path(args.out) / f"hnsw_live_s{seed}.bin"
    index.save_index(str(path))
    build_seconds = (time.perf_counter_ns() - start) / 1e9
    require(index.get_current_count() == nm, "HNSW does not contain the full collected model universe")
    zd = np.load(bundle["folder"] / "z_d_eval.npy", mmap_mode="r", allow_pickle=False)
    zq = np.asarray(zd[queries], dtype=np.float32)

    def save_attempt(ef, ids, scores, recall):
        filename = f"live_calibration_s{seed}_ef{ef}.npz"
        A.save_npz(args.out, filename, seed=np.asarray(seed), query=queries, model=ids,
                   score=scores, recall_per_query=recall, ef_search=np.asarray(ef))
        return filename

    chosen, trace, passed = A.calibrate(index, zq, queries, exact["exact_pool_ids"], 8,
                                       save_attempt, n_models=nm)
    index.set_ef(chosen[0])
    ids, scores, timing = A.measure_queries(index, zq, queries, bundle["prior"], tie_rank)
    row, arrays = A.pool_arrays(ids, scores, queries, bundle["candidates"], bundle["roots"],
                                bundle["prior"], tie_rank, n_models=nm)
    arrays.update(labels, **timing, seed=np.asarray(seed), selected_ef=np.asarray(chosen[0]))
    A.save_npz(args.out, f"live_hnsw_s{seed}.npz", **arrays)
    exact["rows"]["G_hnsw1000_task"] = row
    return {"rows": exact["rows"], "selected_ef": chosen[0],
            "calibration": {"trace": trace, "recall@1000": chosen[3], "passed": passed,
                            "rule": "first ef in [1000,1500,2000,3000,5000] with mean recall@1000 >= 0.99"},
            "latency_ms": A.latency_summary(timing), "index_bytes": path.stat().st_size,
            "build_seconds": build_seconds, "exact_seconds": exact["seconds"]}


def evaluate(args):
    import torch

    require(args.device != "cuda" or torch.cuda.is_available(), "CUDA is unavailable")
    torch.set_float32_matmul_precision("highest")
    graph = validate_graph(args.graph)
    nodes = pd.read_parquet(args.dataset_nodes)
    require({"node", "task", "gold_eligible", "primary_direction", "is_placeholder", "is_rl"}
            <= set(nodes.columns), "Canonical dataset nodes are missing evaluation columns")
    require(nodes.node.is_unique and set(nodes.node.astype(str)) == set(graph["datasets"]),
            "Canonical dataset-node identities differ from the graph")
    P.empty_output(args.out)
    report = {"schema_version": "live.hf.evaluation.v1", "protocol": "live-hf", "status": "running",
        "n_models": graph["n_models"], "n_datasets": graph["n_datasets"],
        "n_performance_edges": graph["n_performance_edges"], "candidate_k": E.POOL_K, "return_k": 10,
        "per_seed": {}, "runtime": A.runtime_versions(), "device": args.device,
        "evaluation_setting": "transductive, root-aware held-out performance edges; actual current-HF eligible cohorts",
        "measurement_scope": "precomputed query embeddings; one-thread CPU HNSW plus prior reranking; excludes encoding, loading and network",
        "reference_scope": "new float32 exact full-lake fusion and dense top1000 on this collection",
        "tie_rule": "fixed label-free permutation for exact and fused ranking; ANN pool ties are chosen by HNSW",
        "snapshot_note": "current collected HF evidence; no equality claim to historical A0 data or metrics",
        "index_note": "eight-thread HNSW construction can vary across builds"}
    provenance = {"schema_version": "live.hf.inputs.v1", "protocol": "live-hf",
        "dataset_nodes": P.file_record(args.dataset_nodes),
        "graph": {key: graph[key] for key in ("directory", "graph_digest", "files")},
        "graph_metadata": P.file_record(Path(args.graph) / "meta.json"),
        "inputs": {}, "artifacts": {}, "code": {path.name: P.file_record(path)
            for path in (Path(__file__), Path(E.__file__), Path(A.__file__), Path(P.__file__))}}
    tie_rank = E._tie_ranks(graph["n_models"])
    for seed in E.SEEDS:
        print(f"[live evaluation] validate and score split {seed}", flush=True)
        bundle, inputs = validate_seed(args, seed, graph, nodes)
        provenance["inputs"][str(seed)] = inputs
        try:
            report["per_seed"][str(seed)] = score_seed(bundle, args, seed, graph["n_models"], tie_rank)
        finally:
            bundle["prior"].payload.close()
        for path in sorted(Path(args.out).iterdir()):
            if path.suffix in (".npz", ".bin") and path.name not in provenance["artifacts"]:
                provenance["artifacts"][path.name] = P.file_record(path)
        A.write_json_atomic(str(Path(args.out) / RESULTS), report)
        A.write_json_atomic(str(Path(args.out) / MANIFEST), provenance)
        del bundle
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
    report["summary"] = P.summarize(report["per_seed"], final_only=False)
    report["all_ann_calibration_passed"] = all(report["per_seed"][str(s)]["calibration"]["passed"] for s in E.SEEDS)
    report["status"] = "complete"
    A.write_json_atomic(str(Path(args.out) / RESULTS), report)
    provenance["results"] = P.file_record(Path(args.out) / RESULTS)
    A.write_json_atomic(str(Path(args.out) / MANIFEST), provenance)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", required=True, type=Path, help="Parent of LIVE_full_s{0,1,2}_e25")
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--dataset-nodes", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    try:
        report = evaluate(args)
    except (OSError, ValueError, KeyError, AssertionError) as exc:
        print(f"Live evaluation failed: {exc}", flush=True)
        return 1
    value = report["summary"]["quality"]["G_hnsw1000_task"]["gold@10"]["mean"]
    print(f"Current-HF mean gold@10={value:.9f}; results: {args.out / RESULTS}")
    return 0 if report["all_ann_calibration_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
