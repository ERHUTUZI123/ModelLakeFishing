"""Evaluate A0 exports without the author's filesystem or historical run scripts.

The default recomputes exact references, builds HNSW, calibrates against dense
top-1000, and measures the final retrieval path. ``--final-only`` replays a
hash-bound saved index and does not claim a new exact reference or calibration.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import a0_evaluation as A
from . import eval_y2 as E

N_DATASETS = 18_729
N_PERFORMANCE_EDGES = 247_803
RUN_FMT = "A0GD_full_s%d_e25"
RESULTS = "A0_PORTABLE_RESULTS.json"
MANIFEST = "A0_PORTABLE_MANIFEST.json"
REFERENCE_MANIFEST = "A0_EVALUATION_MANIFEST.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_record(path, *, expected=None):
    path = Path(path)
    before = path.stat()
    digest = E._sha256(path)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            f"Input changed while hashing: {path}")
    require(expected is None or digest == expected, f"SHA-256 mismatch: {path}")
    return {"sha256": digest, "bytes": after.st_size}


def check_embedding(path, rows):
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    require(matrix.shape == (rows, 128) and matrix.dtype == np.float32,
            f"Expected float32 embedding of shape ({rows}, 128): {path}")
    for start in range(0, rows, 50_000):
        block = matrix[start:start + 50_000]
        require(np.isfinite(block).all(), f"Nonfinite embedding: {path}")
        require(np.all(np.abs(np.linalg.norm(block, axis=1) - 1) <= 1e-5),
                f"Embedding is not unit length: {path}")


def check_gold(path, n_models, n_datasets):
    all_queries = []
    with np.load(path, allow_pickle=False) as labels:
        for key in labels.files:
            query = int(key)
            require(str(query) == key and 0 <= query < n_datasets,
                    f"Invalid gold query ID: {key}")
            values = labels[key]
            require(values.ndim == 2 and values.shape[0] == 2 and values.shape[1] > 0,
                    f"Invalid gold shape for query {query}")
            ids, weights = values
            require(np.isfinite(values).all() and np.equal(ids, np.floor(ids)).all(),
                    f"Invalid gold IDs or values for query {query}")
            require(ids.min() >= 0 and ids.max() < n_models and len(np.unique(ids)) == len(ids),
                    f"Duplicate or out-of-range gold IDs for query {query}")
            all_queries.append(query)
    require(len(all_queries) > 0, "Gold labels are empty")
    return np.asarray(sorted(all_queries), dtype=np.int64)


def check_prior(payload, meta, all_queries, roots, datasets, nodes, n_models):
    """Check actual arrays, including excluded queries, before any scoring."""
    em, ed, weights = (np.asarray(payload[key]) for key in ("edge_model", "edge_dataset", "edge_acc"))
    task, root = (np.asarray(payload[key]) for key in ("task_id", "root_id"))
    require(em.ndim == 1 and em.shape == ed.shape == weights.shape,
            "Prior edge arrays have inconsistent shapes")
    require(task.shape == root.shape == (len(datasets),), "Prior row-map shapes differ")
    require(all(np.issubdtype(x.dtype, np.integer) for x in (em, ed, task, root)),
            "Prior IDs must be integers")
    require(em.min(initial=0) >= 0 and em.max(initial=0) < n_models
            and ed.min(initial=0) >= 0 and ed.max(initial=0) < len(datasets),
            "Prior endpoints are outside the mapped universe")
    require(np.isfinite(weights).all(), "Prior contains nonfinite weights")
    require(meta["n_edges"] == len(em) and meta["n_datasets"] == len(datasets),
            "Prior metadata counts differ from its actual arrays")
    require(len(np.unique(em * len(datasets) + ed)) == len(em), "Duplicate prior edges")
    groups = pd.DataFrame({"root": roots.astype(str), "code": root})
    require(groups.groupby("root").code.nunique().max() == 1
            and groups.groupby("code").root.nunique().max() == 1,
            "Prior root codes disagree with dataset roots")
    require(not np.intersect1d(root[ed], root[all_queries]).size,
            "A held-out query root is visible to the prior")
    require(nodes.node.is_unique and np.isin(datasets, nodes.node.astype(str)).all(),
            "Canonical dataset-node mapping is missing or ambiguous")
    indexed = nodes.assign(node=nodes.node.astype(str)).set_index("node")
    normalized = [re.sub(r"[\s_]+", "-", str(t).strip().lower())
                  for t in indexed.task.reindex(datasets)]
    vocabulary = {name: i for i, name in enumerate(sorted(set(normalized)))}
    require(np.array_equal(task, [vocabulary[t] for t in normalized]),
            "Prior task IDs disagree with canonical task normalization")


def validate_rebuilt_graph(directory):
    """Verify new graph bytes and A0 feature policy, without an old A0 stamp."""
    from .prepare_a0_graph import verify_files, _digest
    from .build_graph_rf import A0_ZERO_COLUMNS, dataset_stats

    directory = Path(directory).resolve()
    meta, files = verify_files(directory)
    require("a0" not in meta and "A0_FEATURE_REPAIR.json" not in files,
            "--rebuilt-graph is for freshly built graphs, not historical A0 repair envelopes")
    require(meta["num_nodes"] == {"model": E.N_TOTAL, "dataset": N_DATASETS},
            "Rebuilt graph node universe differs from A0")
    for name, shape in (("x_model.npy", (E.N_TOTAL, 448)),
                        ("x_dataset.npy", (N_DATASETS, 458))):
        matrix = np.load(directory / name, mmap_mode="r", allow_pickle=False)
        require(matrix.shape == shape and matrix.dtype == np.float32,
                "Rebuilt graph feature shape/dtype differs: " + name)
        for start in range(0, len(matrix), 50_000):
            require(np.isfinite(matrix[start:start + 50_000]).all(), "Nonfinite graph feature: " + name)
    xd = np.load(directory / "x_dataset.npy", mmap_mode="r", allow_pickle=False)
    require(np.all(xd[:, A0_ZERO_COLUMNS] == 0) and np.all(xd[:, 456:458] == 0),
            "Rebuilt graph has performance-derived or reserved dataset features")
    models, _ = E._ids(directory / "unique_model_id.parquet", "rebuilt graph models")
    datasets, frame = E._ids(directory / "unique_dataset_id.parquet", "rebuilt graph datasets")
    require(len(models) == E.N_TOTAL and len(datasets) == N_DATASETS,
            "Rebuilt graph row-map counts differ")
    stats, roots = dataset_stats(pd.DataFrame({"dataset": frame.dataset.astype(str).str.split("\t", n=1).str[0]}), None)
    require(roots == frame.root.astype(str).tolist() and np.array_equal(xd[:, 448:], stats),
            "Rebuilt graph structural features or roots differ from node identities")
    with np.load(directory / "edges.npz", allow_pickle=False) as data:
        edge = data["model__trained_on__dataset__edge_index"].astype(np.int64)
        weights = data["model__trained_on__dataset__edge_attr"]
    require(edge.shape == (2, N_PERFORMANCE_EDGES) and weights.shape == (N_PERFORMANCE_EDGES,),
            "Rebuilt graph performance-edge count differs from A0")
    require(edge[0].min(initial=0) >= 0 and edge[0].max(initial=0) < E.N_TOTAL
            and edge[1].min(initial=0) >= 0 and edge[1].max(initial=0) < N_DATASETS
            and np.isfinite(weights).all(), "Invalid rebuilt performance edges")
    keys = edge[0] * N_DATASETS + edge[1]
    require(len(np.unique(keys)) == len(keys), "Duplicate rebuilt performance edges")
    return {"directory": str(directory), "graph_digest": _digest(files), "files": files,
            "meta": meta, "models": models, "datasets": datasets, "roots": np.asarray(roots),
            "edge_keys": keys, "edge_weights": weights,
            "provenance_kind": "fresh_from_source_graph; no historical A0 envelope"}


def validate_rebuilt_checkpoint(meta, graph, seed):
    from . import checkpoint as CK
    from .train_rung import build_config, reject_smoke_checkpoint

    require(Path(meta["graph"]).resolve() == Path(graph["directory"]),
            "Export graph path differs from --rebuilt-graph")
    require(meta.get("graph_digest") == graph["graph_digest"], "Export rebuilt graph digest differs")
    require(isinstance(meta.get("checkpoint_sha256"), str), "Rebuilt export lacks checkpoint hash")
    checkpoint = Path(meta["checkpoint"])
    record = file_record(checkpoint, expected=meta["checkpoint_sha256"])
    ck = CK.load(str(checkpoint))
    reject_smoke_checkpoint(ck)
    require(ck.get("epoch") == 24 and len(ck.get("history", [])) == 25,
            "Rebuilt export checkpoint is not a complete 25-epoch run")
    require(all(np.isfinite(v) for epoch in ck["history"] for v in epoch.values()
                if isinstance(v, (int, float))), "Nonfinite rebuilt checkpoint training history")
    expected = CK.make_binding(graph["meta"]["xm0_meta"], graph_path=graph["directory"],
        split_seed=seed, family_vocab_path=ck["binding"].get("family_vocab_path"),
        encoder_name=graph["meta"]["xd0_meta"].get("encoder_name", "all-MiniLM-L6-v2"))
    require(expected["family_vocab_sha256"] is not None, "Rebuilt checkpoint family vocabulary is missing")
    require(all(ck["binding"].get(k) == expected[k] and meta["binding"].get(k) == expected[k]
                for k in CK.BINDING_KEYS), "Rebuilt checkpoint/export graph, vocabulary or split binding differs")
    cfg = build_config(SimpleNamespace(fanout=True, sparse_M=True, contrast_n_neg=256,
        contrast_max_pos_per_dataset=None, chunked_infer=50_000, skip_diagnostics=True,
        batch_size=None, lake_gamma=0.5, global_n_datasets=128, num_layers=None))
    observed = {k: v for k, v in ck["cfg"].items() if k not in ("resume_state", "history0", "on_epoch_end")}
    require(observed == cfg, "Rebuilt checkpoint training configuration differs from A0 GD")
    return record


def validate_seed(args, seed, tie_rank, first_model, first_dataset, nodes):
    folder = Path(args.exports) / (RUN_FMT % seed)
    manifest = read_json(folder / "EXPORT_MANIFEST.json")
    meta = manifest["stages"]["embed"]
    envelope = meta.get("a0", {})
    rebuilt = getattr(args, "rebuilt_graph", None)
    graph = None
    checkpoint_record = None
    if rebuilt is not None:
        require(not args.final_only, "--rebuilt-graph cannot be used for saved-index replay")
        require(not envelope, "Fresh rebuilt exports must not claim historical A0 provenance")
        graph = getattr(args, "_rebuilt_graph_validation", None)
        if graph is None:
            graph = args._rebuilt_graph_validation = validate_rebuilt_graph(rebuilt)
        checkpoint_record = validate_rebuilt_checkpoint(meta, graph, seed)
    else:
        require(envelope.get("protocol") == "a0" and envelope.get("seed") == seed
                and envelope.get("run_id") == "A0_20260912", "Export is not the A0 split")
    require(meta.get("split_seed") == seed and meta.get("checkpoint_epoch") == 24
            and meta.get("n_models") == E.N_TOTAL and meta.get("n_datasets") == N_DATASETS,
            "Export does not describe the full A0 25-epoch run")
    expected_graph_digest = graph["graph_digest"] if graph is not None else envelope.get("graph_digest")
    require(expected_graph_digest == meta.get("graph_digest") == meta.get("binding", {}).get("graph_sha256")
            and meta.get("binding", {}).get("split_seed") == seed,
            "Export graph/split producer bindings disagree")
    A.verify_chunked_export(meta)
    names = ["model_ids.parquet", "dataset_ids.parquet", "gold_cands.npz", "z_d_eval.npy"]
    if not args.final_only:
        names.append("z_m_eval.npy")
    hashes = meta.get("artifact_hashes", {})
    require(set(names) <= set(hashes), "Export producer hashes are incomplete")
    files = {name: file_record(folder / name, expected=hashes[name]) for name in names}
    files["EXPORT_MANIFEST.json"] = file_record(folder / "EXPORT_MANIFEST.json")
    prior_path = folder / f"prior_sidecar_s{seed}.npz"
    prior_meta_path = folder / f"prior_sidecar_s{seed}_meta.json"
    prior_meta = read_json(prior_meta_path)
    require((not prior_meta.get("a0") if graph is not None else prior_meta.get("a0") == envelope)
            and prior_meta.get("graph_digest") == expected_graph_digest,
            "Prior and export producer bindings disagree")
    require(isinstance(prior_meta.get("sidecar_sha256"), str), "Prior producer hash is absent")
    files[prior_path.name] = file_record(prior_path, expected=prior_meta["sidecar_sha256"])
    files[prior_meta_path.name] = file_record(prior_meta_path)
    for name, rows in (("z_d_eval.npy", N_DATASETS), ("z_m_eval.npy", E.N_TOTAL)):
        if name in names:
            check_embedding(folder / name, rows)
    bundle_args = SimpleNamespace(exports=str(args.exports), sidecar_exports=str(args.exports),
                                  run_fmt=RUN_FMT, sidecar_run_fmt=RUN_FMT,
                                  dataset_nodes=str(args.dataset_nodes))
    bundle = E._audit_seed(bundle_args, seed, first_model, first_dataset, tie_rank)
    require(len(bundle["datasets"]) == N_DATASETS, "Dataset universe is not the A0 universe")
    require(len(np.unique(bundle["models"])) == E.N_TOTAL
            and len(np.unique(bundle["datasets"])) == N_DATASETS, "Duplicate row identities")
    all_queries = check_gold(folder / "gold_cands.npz", E.N_TOTAL, N_DATASETS)
    require(len(all_queries) == meta["n_test_queries"], "Gold query count differs from export")
    check_prior(bundle["prior"].payload, prior_meta, all_queries, bundle["roots"],
                bundle["datasets"], nodes, E.N_TOTAL)
    if graph is not None:
        require(np.array_equal(bundle["models"], graph["models"])
                and np.array_equal(bundle["datasets"], graph["datasets"])
                and np.array_equal(bundle["roots"], graph["roots"]), "Rebuilt graph/export row identities differ")
        payload = bundle["prior"].payload
        keys = payload["edge_model"] * N_DATASETS + payload["edge_dataset"]
        order = np.argsort(graph["edge_keys"])
        position = np.searchsorted(graph["edge_keys"][order], keys)
        require(np.all(position < len(order)) and np.array_equal(graph["edge_keys"][order[position]], keys),
                "Rebuilt prior contains edges absent from its graph")
        require(np.array_equal(payload["edge_acc"].astype(np.float32), graph["edge_weights"][order[position]]),
                "Rebuilt prior weights differ from its graph")
    inputs = {"directory": RUN_FMT % seed, "files": files,
              "eligible_queries": len(bundle["candidates"]), "all_test_queries": len(all_queries)}
    if graph is None:
        inputs["a0"] = envelope
    else:
        inputs["producer"] = {"kind": "fresh_rebuilt", "graph_digest": expected_graph_digest,
                              "split_seed": seed, "checkpoint": checkpoint_record}
    return bundle, inputs


def check_reference_index(directory, seed, inputs, dataset_record):
    """Bind index bytes and serving inputs without resolving historical paths."""
    directory = Path(directory)
    manifest = read_json(directory / REFERENCE_MANIFEST)
    require(manifest.get("protocol") == "a0" and manifest.get("run_id") == "A0_20260912",
            "Reference index manifest is not A0")
    name = f"hnsw_a0_s{seed}.bin"
    entry = manifest["seeds"][str(seed)]
    require(entry["index"] == name, "Reference index seed/name mismatch")
    record = file_record(directory / name, expected=manifest["artifacts"][name]["sha256"])
    source = manifest["binding"]["seed_inputs"][str(seed)]
    require(source["graph_digest"] == inputs["a0"]["graph_digest"],
            "Reference index and export graph bindings differ")
    for filename, current in inputs["files"].items():
        # Export/prior metadata may contain relocated path strings. Their own
        # A0 envelopes and artifact hashes were validated before reaching here.
        if filename.endswith(".json"):
            continue
        bound = {r["sha256"] for r in source["files"]
                 if r["path"].replace("\\", "/").rsplit("/", 1)[-1] == filename}
        require(bound == {current["sha256"]}, f"Reference index input mismatch: {filename}")
    node_hashes = {r["sha256"] for r in manifest["binding"]["source_files"]
                   if r["role"] == "frozen_dataset_nodes"}
    require(node_hashes == {dataset_record["sha256"]}, "Reference dataset-node table differs")
    require(entry["selected_ef"] in A.GRID, "Reference ef is outside A0 calibration grid")
    return directory / name, record, int(entry["selected_ef"])


def empty_output(path):
    path = Path(path)
    require(not path.exists() or (path.is_dir() and not any(path.iterdir())),
            f"Output must be absent or an empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def summarize(per_seed, final_only):
    require(set(per_seed) == {"0", "1", "2"}, "All three splits are required")
    summary = {"quality": A.summarize_rows(per_seed),
               "latency_ms": {key: A.numeric_summary([per_seed[str(s)]["latency_ms"][key]
                                                       for s in E.SEEDS])
                              for key in per_seed["0"]["latency_ms"]},
               "aggregation": "equal-weight mean of three split metrics and per-split latency quantiles"}
    if not final_only:
        vals = lambda row: [per_seed[str(s)]["rows"][row]["gold@10"] for s in E.SEEDS]
        summary["overall_retention"] = A.ratio_summary(vals("G_hnsw1000_task"), vals("G_full_task"))
        summary["exact_pool_retention"] = A.ratio_summary(vals("G_exact1000_task"), vals("G_full_task"))
        summary["ann_retention"] = A.ratio_summary(vals("G_hnsw1000_task"), vals("G_exact1000_task"))
    return summary


def exact_seed(bundle, args, tie_rank, seed):
    result = E.evaluate_exact_seed(bundle["x4"], bundle["prior"], bundle["candidates"],
                                  bundle["roots"], tie_rank, args.device,
                                  model_chunk=50_000, query_chunk=16, protocol="a0")
    queries = np.asarray(result["queries"], dtype=np.int64)
    row, arrays = A.pool_arrays(result["exact_pool_ids"], result["exact_pool_scores"], queries,
                               bundle["candidates"], bundle["roots"], bundle["prior"], tie_rank)
    result["rows"]["G_exact1000_task"].update(row)
    from_top, _ = E._top10_metrics(result["full_top10"], queries.tolist(),
                                  bundle["candidates"], bundle["roots"])
    for key in ("gold@1", "gold@10", "gold-gap@1", "gold-gap@10", "top3@10",
                "root_gold@1", "root_gold@10", "root_top3@10", "root_gold-gap@10"):
        require(from_top[key] == result["rows"]["G_full_task"][key],
                f"Full-lake top10/rank disagreement: {key}")
    arrays.update(A.label_arrays(queries, bundle["candidates"], bundle["roots"], bundle["prior"]))
    arrays.update(seed=np.asarray(seed), full_top10=result["full_top10"],
                  full_counts=result["full_counts"], dense_counts=result["dense_counts"])
    A.save_npz(args.out, f"a0_exact_s{seed}.npz", **arrays)
    return result


def evaluate(args):
    import hnswlib
    import torch

    require(not args.final_only or args.reference_index_dir is not None,
            "--final-only requires --reference-index-dir with its A0 manifest")
    require(args.reference_index_dir is None or args.final_only,
            "Saved indices are supported by --final-only; the full run builds fresh indices")
    require(not args.final_only or args.rebuilt_graph is None,
            "--rebuilt-graph cannot be used for saved-index replay")
    if args.device == "cuda":
        require(torch.cuda.is_available(), "CUDA is unavailable; install CUDA PyTorch or use --device cpu")
    torch.set_float32_matmul_precision("highest")
    empty_output(args.out)
    dataset_record = file_record(args.dataset_nodes)
    nodes = pd.read_parquet(args.dataset_nodes)
    require({"node", "task", "gold_eligible", "primary_direction", "is_placeholder", "is_rl"}
            <= set(nodes.columns), "Canonical dataset-node table is missing required columns")
    mode = "saved_index_final_only" if args.final_only else "rebuilt_index_full_evaluation"
    report = {"schema_version": "a0.portable.v1", "protocol": "a0", "mode": mode,
              "status": "running", "n_models": E.N_TOTAL, "n_datasets": N_DATASETS,
              "candidate_k": E.POOL_K, "return_k": 10, "per_seed": {},
              "runtime": A.runtime_versions(), "device": args.device,
              "measurement_scope": "precomputed query embeddings; HNSW plus prior reranking; excludes encoding, loading and network",
              "reference_scope": "none; frozen ef reused without recalibration" if args.final_only
              else "fresh float32 exact full-lake and dense top1000 references; original A0 scoring and ties",
              "latency_note": "New measurements on this machine; historical timings are never copied",
              "index_note": "Saved index replay" if args.final_only else
              "Eight-thread HNSW insertion is not bitwise deterministic; rebuilt-index results may differ"}
    report["representation_source"] = ("frozen_A0_export" if args.final_only else
        "fresh_training_on_rebuilt_graph" if args.rebuilt_graph is not None else
        "fresh_training_on_frozen_prepared_A0_graph")
    provenance = {"schema_version": "a0.portable.inputs.v1", "mode": mode,
                  "dataset_nodes": dataset_record, "inputs": {}, "artifacts": {},
                  "code": {path.name: file_record(path) for path in
                           (Path(__file__), Path(E.__file__), Path(A.__file__))}}
    first_model = first_dataset = None
    tie_rank = E._tie_ranks(E.N_TOTAL)
    for seed in E.SEEDS:
        print(f"[A0 portable] validate split {seed}", flush=True)
        bundle, inputs = validate_seed(args, seed, tie_rank, first_model, first_dataset, nodes)
        if first_model is None:
            first_model, first_dataset = bundle["models"], bundle["datasets"]
        provenance["inputs"][str(seed)] = inputs
        if args.rebuilt_graph is not None and "rebuilt_graph" not in provenance:
            verified = args._rebuilt_graph_validation
            provenance["rebuilt_graph"] = {k: verified[k] for k in
                ("directory", "graph_digest", "files", "provenance_kind")}
            report["rebuilt_graph"] = {"graph_digest": verified["graph_digest"],
                                       "provenance_kind": verified["provenance_kind"]}
        queries = np.asarray(sorted(bundle["candidates"]), dtype=np.int64)
        rows, exact, calibration, build_seconds = {}, None, None, None
        if not args.final_only:
            exact = exact_seed(bundle, args, tie_rank, seed)
            rows.update(exact["rows"])
        index = hnswlib.Index(space="ip", dim=128)
        name = f"hnsw_a0_s{seed}.bin"
        if args.final_only:
            target, index_record, selected_ef = check_reference_index(
                args.reference_index_dir, seed, inputs, dataset_record)
            index.load_index(str(target), max_elements=E.N_TOTAL)
            inputs["reference_index"] = dict(index_record, name=name)
            inputs["reference_manifest"] = file_record(Path(args.reference_index_dir) / REFERENCE_MANIFEST)
        else:
            target = Path(args.out) / name
            zm = np.load(Path(bundle["x4"]) / "z_m_eval.npy", mmap_mode="r", allow_pickle=False)
            start = time.perf_counter_ns()
            index.init_index(max_elements=E.N_TOTAL, ef_construction=200, M=32)
            index.add_items(zm, np.arange(E.N_TOTAL, dtype=np.int64), num_threads=8)
            index.save_index(str(target))
            build_seconds = (time.perf_counter_ns() - start) / 1e9
            del zm
        require(index.get_current_count() == E.N_TOTAL, "HNSW does not contain the full A0 universe")
        index.set_num_threads(8)
        zd = np.load(Path(bundle["x4"]) / "z_d_eval.npy", mmap_mode="r", allow_pickle=False)
        zq = np.asarray(zd[queries], dtype=np.float32)
        if exact is not None:
            def save_attempt(ef, ids, scores, recall):
                filename = f"a0_calibration_s{seed}_ef{ef}.npz"
                A.save_npz(args.out, filename, seed=np.asarray(seed), query=queries,
                           model=ids, score=scores, recall_per_query=recall, ef_search=np.asarray(ef))
                return filename
            chosen, trace, passed = A.calibrate(index, zq, queries, exact["exact_pool_ids"], 8, save_attempt)
            selected_ef = chosen[0]
            calibration = {"trace": trace, "recall@1000": chosen[3], "passed": passed,
                           "rule": "first grid ef with recall@1000 >= 0.99; otherwise measured at ef5000"}
        index.set_ef(selected_ef)
        ids, scores, timing = A.measure_queries(index, zq, queries, bundle["prior"], tie_rank)
        row, arrays = A.pool_arrays(ids, scores, queries, bundle["candidates"],
                                   bundle["roots"], bundle["prior"], tie_rank)
        rows["G_hnsw1000_task"] = row
        arrays.update(A.label_arrays(queries, bundle["candidates"], bundle["roots"], bundle["prior"]))
        arrays.update(timing, seed=np.asarray(seed), selected_ef=np.asarray(selected_ef))
        A.save_npz(args.out, f"a0_hnsw_s{seed}.npz", **arrays)
        report["per_seed"][str(seed)] = {"rows": rows, "selected_ef": selected_ef,
            "calibration": calibration, "latency_ms": A.latency_summary(timing),
            "index_bytes": target.stat().st_size, "build_seconds": build_seconds}
        for filename in sorted(Path(args.out).iterdir()):
            if filename.suffix in (".npz", ".bin") and filename.name not in provenance["artifacts"]:
                provenance["artifacts"][filename.name] = file_record(filename)
        A.write_json_atomic(str(Path(args.out) / RESULTS), report)
        A.write_json_atomic(str(Path(args.out) / MANIFEST), provenance)
        print(f"[A0 portable] split {seed} gold@10={row['gold@10']:.9f}", flush=True)
        bundle["prior"].payload.close()
        del bundle, index, zd, zq, arrays, exact
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
    report["summary"] = summarize(report["per_seed"], args.final_only)
    report["status"] = "complete"
    report["all_ann_calibration_passed"] = (None if args.final_only else
        all(report["per_seed"][str(seed)]["calibration"]["passed"] for seed in E.SEEDS))
    A.write_json_atomic(str(Path(args.out) / RESULTS), report)
    provenance["results"] = file_record(Path(args.out) / RESULTS)
    A.write_json_atomic(str(Path(args.out) / MANIFEST), provenance)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", required=True, type=Path, help="Parent of A0GD_full_s{0,1,2}_e25")
    parser.add_argument("--dataset-nodes", required=True, type=Path, help="Canonical dataset_nodes_merged.parquet")
    parser.add_argument("--out", required=True, type=Path, help="Absent or empty output directory")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu", help="Device for exact full-lake reference")
    parser.add_argument("--reference-index-dir", type=Path, help="Frozen indices plus A0_EVALUATION_MANIFEST.json")
    parser.add_argument("--final-only", action="store_true", help="Replay saved indices and frozen ef; skip exact references")
    parser.add_argument("--rebuilt-graph", type=Path,
                        help="Fresh full-profile graph: verify bytes, clean features and producer/checkpoint bindings")
    args = parser.parse_args(argv)
    try:
        report = evaluate(args)
    except (OSError, ValueError, KeyError, AssertionError) as exc:
        print(f"A0 evaluation failed: {exc}", flush=True)
        return 1
    value = report["summary"]["quality"]["G_hnsw1000_task"]["gold@10"]["mean"]
    print(f"A0 mean gold@10={value:.9f}; results: {args.out / RESULTS}")
    return 0 if report["all_ann_calibration_passed"] is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
