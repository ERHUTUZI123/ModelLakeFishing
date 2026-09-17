"""Independent raw-file correctness audit. No retrieval/quality scorer imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

ZERO_COLUMNS = [448, 449, 450, 451, 452, 453, 455]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def checked_file(record):
    path = Path(record["path"]).resolve(strict=True)
    require(sha256(path) == record["sha256"], f"correctness input SHA256 mismatch: {path}")
    return path


def result(value, sources, status="verified"):
    return {"value": value, "status": status,
            "source": "; ".join(str(Path(p).resolve()) for p in sources)}


def graph_checks(graph, input_audit, verified=None):
    """Compare retained features directly with the frozen source, including bytes."""
    from scale1m.a0_graph_validation import verify_a0_graph
    graph = Path(graph).resolve()
    verified = verify_a0_graph(graph) if verified is None else verified
    repair, files = verified["repair"], verified["files"]
    sources = [r for r in input_audit["files"] if Path(r["path"]).name == "x_dataset.npy"
               and r["sha256"] == repair["source_x_dataset_sha256"]]
    require(len(sources) == 1, "one frozen source dataset matrix is required for retained-column comparison")
    source = checked_file(sources[0])
    original = np.load(source, mmap_mode="r", allow_pickle=False)
    current_path = graph / "x_dataset.npy"
    current = np.load(current_path, mmap_mode="r", allow_pickle=False)
    keep = [i for i in range(458) if i not in ZERO_COLUMNS]
    require(current.shape == original.shape and current.dtype == original.dtype == np.float32,
            "source/repaired dataset shapes or precision differ")
    changed_retained = int(np.count_nonzero(current[:, keep] != original[:, keep]))
    nonzero = {str(c): int(np.count_nonzero(current[:, c])) for c in ZERO_COLUMNS}
    require(changed_retained == 0 and not any(nonzero.values()), "feature repair changed retained values or retained performance values")
    retained_sha = array_sha(current[:, keep])
    source_retained_sha = array_sha(original[:, keep])
    require(retained_sha == source_retained_sha, "retained feature bytes differ from frozen source")
    unchanged_files = [name for name in repair["source_files"] if name != "x_dataset.npy"]
    require(all(files[name] == repair["source_files"][name] for name in unchanged_files),
            "a non-dataset-matrix graph file changed")
    prefix = "feature_repair_verification."
    evidence = [graph / "meta.json", graph / "A0_FEATURE_REPAIR.json", current_path, source]
    return {
        prefix + "seven_columns_zero.shared": result({"passed": True, "columns": ZERO_COLUMNS,
            "dataset_rows": int(current.shape[0]), "nonzero_per_column": nonzero}, evidence),
        prefix + "all_other_graph_content_unchanged.shared": result({"passed": True,
            "retained_columns": keep, "changed_retained_elements": changed_retained,
            "retained_feature_sha256": retained_sha,
            "source_retained_feature_sha256": source_retained_sha,
            "unchanged_files": unchanged_files, "critical_metadata_sha256": repair["critical_metadata_sha256"]}, evidence),
        prefix + "new_graph_hash_verified.shared": result({"passed": True,
            "new_graph_digest": verified["graph_sha256"], "source_graph_digest": repair["source_graph_digest"],
            "actual_file_hashes": files}, evidence),
    }


def embedding_check(path, expected_rows):
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    require(matrix.shape == (expected_rows, 128) and matrix.dtype == np.float32,
            f"embedding shape/dtype mismatch: {path}")
    nonfinite = 0
    max_error = 0.0
    for start in range(0, len(matrix), 50000):
        block = matrix[start:start + 50000]
        nonfinite += int(np.count_nonzero(~np.isfinite(block)))
        norm_error = np.abs(np.linalg.norm(block, axis=1) - 1.0)
        max_error = max(max_error, float(norm_error.max(initial=0)))
    require(nonfinite == 0 and max_error <= 1e-5, f"nonfinite or non-unit embedding: {path}")
    return {"shape": list(matrix.shape), "dtype": str(matrix.dtype), "nonfinite_values": nonfinite,
            "maximum_absolute_norm_error": max_error, "norm_atol": 1e-5, "norm_rtol": 0,
            "file_sha256": sha256(path)}


def checkpoint_check(ck, graph, graph_meta, expected_cfg, seed):
    from scale1m import checkpoint as CK
    from scale1m.train_rung import reject_smoke_checkpoint
    reject_smoke_checkpoint(ck)
    expected = CK.make_binding(graph_meta["xm0_meta"], graph_path=str(graph), split_seed=seed,
        family_vocab_path=ck["binding"].get("family_vocab_path"),
        encoder_name=graph_meta["xd0_meta"].get("encoder_name", "all-MiniLM-L6-v2"))
    mismatch = {key: {"expected": expected[key], "actual": ck["binding"].get(key)}
                for key in CK.BINDING_KEYS if ck["binding"].get(key) != expected[key]}
    require(not mismatch, f"checkpoint dependency binding mismatch: {mismatch}")
    cfg = {key: value for key, value in ck["cfg"].items() if key not in ("resume_state", "history0", "on_epoch_end")}
    require(cfg == expected_cfg and ck["epoch"] == 24 and len(ck["history"]) == 25,
            "checkpoint config/25-epoch completion differs")
    require(all(np.isfinite(value) for epoch in ck["history"] for value in epoch.values()
                if isinstance(value, (int, float))), "checkpoint history has nonfinite observations")
    return {"passed": True, "actual_binding": {k: ck["binding"].get(k) for k in CK.BINDING_KEYS},
            "expected_binding": {k: expected[k] for k in CK.BINDING_KEYS}, "mismatches": mismatch,
            "stored_epoch": ck["epoch"], "completed_history_epochs": len(ck["history"]),
            "resolved_config_keys": sorted(cfg)}


def read_rowmap(path, kind):
    frame = pd.read_parquet(path)
    require("mappedID" in frame and np.array_equal(frame.mappedID.to_numpy(), np.arange(len(frame))),
            f"row map is not in actual contiguous mappedID order: {path}")
    names = [name for name in (kind, "node", "unique_" + kind + "_id") if name in frame]
    require(bool(names), f"row map has no {kind} identity column: {path}")
    return frame, frame[names[0]].astype(str).to_numpy()


def prior_checks(graph, export, sidecar_dir, seed, query_ids, dataset_nodes, input_audit,
                 envelope, model_names, dataset_names, raw_paths=()):
    graph, export, sidecar_dir = Path(graph), Path(export), Path(sidecar_dir)
    graph_meta = read_json(graph / "meta.json")
    n_models, n_datasets = len(model_names), len(dataset_names)
    sidecar = sidecar_dir / f"prior_sidecar_s{seed}.npz"
    meta_path = sidecar.with_name(sidecar.stem + "_meta.json")
    meta = read_json(meta_path)
    require(all(meta.get("a0", {}).get(k) == v for k, v in envelope.items()), "sidecar producer envelope differs")
    require(meta.get("sidecar_sha256") == sha256(sidecar) and meta.get("graph_digest") == envelope["graph_digest"],
            "sidecar bytes/graph differ from producer")
    require(meta.get("split_seed") == seed and meta.get("n_models") == n_models and meta.get("n_datasets") == n_datasets,
            "sidecar seed/node-count metadata differ")
    mapping_sources = []
    for folder in (export, sidecar_dir):
        for kind, expected in (("model", model_names), ("dataset", dataset_names)):
            path = folder / (kind + "_ids.parquet")
            _, actual = read_rowmap(path, kind)
            require(np.array_equal(actual, expected), f"prior/export/graph row map mismatch: {path}")
            mapping_sources.append(path)
    graph_udi, _ = read_rowmap(graph / "unique_dataset_id.parquet", "dataset")
    roots = graph_udi.root.astype(str).to_numpy()
    with np.load(sidecar, allow_pickle=False) as data:
        em, ed = data["edge_model"].astype(np.int64), data["edge_dataset"].astype(np.int64)
        weights = data["edge_acc"].astype(np.float64)
        task_id, root_id = data["task_id"].astype(np.int64), data["root_id"].astype(np.int64)
    require(em.shape == ed.shape == weights.shape and len(task_id) == len(root_id) == n_datasets,
            "sidecar edge/map dimensions differ")
    require(em.min(initial=0) >= 0 and em.max(initial=0) < n_models
            and ed.min(initial=0) >= 0 and ed.max(initial=0) < n_datasets and np.isfinite(weights).all(),
            "sidecar contains invalid endpoint/value")
    original_split = next(row for row in input_audit["splits"] if row["seed"] == seed)
    pairs_hash = array_sha(np.stack([em, ed]))
    require(pairs_hash == original_split["prior_visible_pairs_ordered"]["sha256_c_order_bytes"],
            "sidecar does not preserve frozen train/validation pairs and order")
    require(meta["n_edges"] == len(em), "sidecar recorded edge count differs from actual arrays")
    frame = pd.DataFrame({"root": roots, "code": root_id})
    require(frame.groupby("root").code.nunique().max() == 1 and frame.groupby("code").root.nunique().max() == 1,
            "prior integer root map differs from graph root equivalence classes")
    visible_roots, query_roots = np.unique(root_id[ed]), np.unique(root_id[query_ids])
    overlap = np.intersect1d(visible_roots, query_roots)
    require(len(overlap) == 0, "a scored query root/sibling is visible to task prior")
    nodes = pd.read_parquet(dataset_nodes, columns=["node", "task"]).set_index("node")
    tasks = nodes.task.reindex(dataset_names)
    require(np.isin(dataset_names, nodes.index).all(), "canonical node mapping is incomplete")
    normalized = [re.sub(r"[\s_]+", "-", str(task).strip().lower()) for task in tasks]
    vocabulary = {task: i for i, task in enumerate(sorted(set(normalized)))}
    require(np.array_equal(task_id, [vocabulary[task] for task in normalized]), "sidecar task normalization differs")
    with np.load(graph / "edges.npz", allow_pickle=False) as edges:
        endpoints = edges["model__trained_on__dataset__edge_index"]
        values = edges["model__trained_on__dataset__edge_attr"]
        keys = endpoints[0] * n_datasets + endpoints[1]
        order = np.argsort(keys)
        loc = np.searchsorted(keys[order], em * n_datasets + ed)
        require(np.all(loc < len(keys)) and np.array_equal(keys[order[loc]], em * n_datasets + ed),
                "sidecar pairs absent from graph")
        require(np.array_equal(weights.astype(np.float32), values[order[loc]]), "sidecar oriented values differ from graph")
    # Independent sufficient-statistic aggregation, separate from eval_rf._prior_tables.
    group_keys, inverse = np.unique(task_id[ed] * n_models + em, return_inverse=True)
    sums = np.bincount(inverse, weights=weights)
    counts = np.bincount(inverse)
    shrunk = ((sums + 2.5) / (counts + 5.0)).astype(np.float32)
    checked_raw = 0
    for raw_path in raw_paths:
        with np.load(raw_path, allow_pickle=False) as raw:
            q, models = raw["query"].astype(np.int64), raw["model"].astype(np.int64)
            require(np.array_equal(q, query_ids), "raw prior query identity differs from frozen identity")
            search = task_id[q, None] * n_models + models
            position = np.searchsorted(group_keys, search)
            present = position < len(group_keys)
            if len(group_keys):
                present &= group_keys[np.minimum(position, len(group_keys) - 1)] == search
            rebuilt = np.zeros(models.shape, dtype=np.float32)
            rebuilt[present] = shrunk[position[present]]
            require(np.array_equal(raw["prior"], rebuilt), f"raw prior differs from independent sidecar aggregation: {raw_path}")
            checked_raw += int(rebuilt.size)
    sources = [sidecar, meta_path, graph / "edges.npz", graph / "unique_dataset_id.parquet", dataset_nodes] + mapping_sources + list(raw_paths)
    prefix = "task_prior."
    return {
        prefix + f"visible_prior_edges.{seed}": result(int(len(em)), sources, "recomputed"),
        prefix + f"prior_root_exclusion.{seed}": result({"passed": True, "query_roots": len(query_roots),
            "visible_roots": len(visible_roots), "overlap_root_ids": overlap.tolist(),
            "visible_pairs_sha256": pairs_hash, "frozen_train_validation_pairs_sha256": original_split["prior_visible_pairs_ordered"]["sha256_c_order_bytes"]}, sources),
        prefix + f"prior_rowmap_alignment.{seed}": result({"passed": True, "model_rows": n_models,
            "dataset_rows": n_datasets, "root_groups": int(frame.root.nunique()), "task_groups": len(vocabulary),
            "task_id_sha256": array_sha(task_id), "root_id_sha256": array_sha(root_id)}, sources),
        prefix + f"sidecar_rebuilt_and_hash_bound.{seed}": result({"passed": True, "a0": envelope,
            "sidecar_sha256": meta["sidecar_sha256"], "visible_pairs_sha256": pairs_hash,
            "edge_values_sha256": array_sha(weights), "task_model_groups": len(group_keys),
            "shrink_k": 5.0, "neutral_value": 0.5, "unsupported_prior": 0.0,
            "aggregation": "(sum oriented values + 5*0.5)/(support+5); NumPy unique/bincount; float32 lookup",
            "raw_prior_values_independently_checked": checked_raw,
            "raw_prior_check_status": "verified" if raw_paths else "not_yet_produced"}, sources),
    }


def recompute_checks(manifest, protocol):
    """Called after the reader verifies every manifest-bound file's actual hash."""
    from scale1m import checkpoint as CK
    binding = manifest["binding"]
    input_audit_path = checked_file(protocol["input_audit_binding"]["input_audit"])
    input_audit = read_json(input_audit_path)
    identity_path = checked_file(protocol["input_audit_binding"]["query_identity"])
    identity = pd.read_json(identity_path, lines=True, dtype={"node": str, "root": str})
    node_refs = [r for r in binding["source_files"] if r["role"] == "frozen_dataset_nodes"]
    require(len(node_refs) == 1, "frozen dataset node source is missing or ambiguous")
    dataset_nodes = checked_file(node_refs[0])
    output, graph_cache = {}, {}
    for seed in (0, 1, 2):
        sb = binding.get("seed_inputs", {}).get(str(seed))
        if sb is None:
            continue
        export, sidecar = Path(sb["export_dir"]), Path(sb["sidecar_dir"])
        embed = read_json(export / "EXPORT_MANIFEST.json")["stages"]["embed"]
        graph = Path(embed["graph"]).resolve()
        if str(graph) not in graph_cache:
            checks = graph_checks(graph, input_audit)
            require(not output or all(output.get(key, value) == value for key, value in checks.items()),
                    "seeds bind to different prepared graph evidence")
            output.update(checks)
            graph_cache[str(graph)] = True
        gm = read_json(graph / "meta.json")
        _, model_names = read_rowmap(graph / "unique_model_id.parquet", "model")
        _, dataset_names = read_rowmap(graph / "unique_dataset_id.parquet", "dataset")
        checkpoint = Path(embed["checkpoint"])
        ck = CK.load(str(checkpoint))
        ck_detail = checkpoint_check(ck, graph, gm, protocol["required_config_audit"]["resolved_config"], seed)
        envelope = {"protocol": "a0", "run_id": manifest["run_id"], "seed": seed,
                    "graph_digest": sb["graph_digest"], "checkpoint_sha256": sha256(checkpoint)}
        require(all(embed.get("a0", {}).get(k) == v for k, v in envelope.items()), "export producer binding differs")
        run_records = checkpoint.parent.parent / "A0_RUN_RECORDS.json"
        records = read_json(run_records)
        require(records.get("status") == "complete" and records.get("history") == ck["history"]
                and all(records.get("a0", {}).get(k) == v for k, v in envelope.items()),
                "training records/producer completion differ")
        output[f"reproducibility_checks.checkpoint_graph_binding.{seed}"] = result(ck_detail,
            [checkpoint, run_records, graph / "meta.json"])
        embeddings = {}
        for name, rows in (("z_m.npy", len(model_names)), ("z_m_eval.npy", len(model_names)),
                           ("z_d.npy", len(dataset_names)), ("z_d_eval.npy", len(dataset_names))):
            embeddings[name] = embedding_check(export / name, rows)
            require(embeddings[name]["file_sha256"] == embed["artifact_hashes"][name], "embedding differs from export producer hash")
        for kind, names in (("model", model_names), ("dataset", dataset_names)):
            _, actual = read_rowmap(export / f"{kind}_ids.parquet", kind)
            require(np.array_equal(actual, names), "embedding row IDs differ from graph")
        output[f"reproducibility_checks.embedding_shape_finite_norm_and_order.{seed}"] = result(
            {"passed": True, "embeddings": embeddings, "model_rows": len(model_names), "dataset_rows": len(dataset_names)},
            [export / name for name in embeddings] + [export / "model_ids.parquet", export / "dataset_ids.parquet"])
        queries = identity[identity.seed == seed].sort_values("query_mappedID").query_mappedID.to_numpy(np.int64)
        raw_paths = [Path(manifest["artifacts"][manifest["seeds"][str(seed)][key]]["path"])
                     for key in ("exact", "hnsw") if key in manifest.get("seeds", {}).get(str(seed), {})]
        output.update(prior_checks(graph, export, sidecar, seed, queries, dataset_nodes, input_audit,
                                   envelope, model_names, dataset_names, raw_paths))
        del ck
    return output
