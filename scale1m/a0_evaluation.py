"""Fresh, hash-bound A0 evaluation of the fixed GD/HNSW1000/task-prior path.

The legacy evaluator remains the archived-result reproducer. This adapter keeps
its scoring functions while separating measured effectiveness from completion.
It never imports old measured quality or timing as new results.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np
import pandas as pd

from ModelLakeFishing.scale1m import eval_y2 as E
from ModelLakeFishing.scale1m.hf_crawl import write_json_atomic, utcnow

MANIFEST = "A0_EVALUATION_MANIFEST.json"
REPORT = "A0_EVALUATION_REPORT.json"
GRID = [1000, 1500, 2000, 3000, 5000]
ZERO_COLUMNS = [448, 449, 450, 451, 452, 453, 455]
RAW_SCHEMA = "a0.raw.v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def runtime_versions():
    from importlib.metadata import PackageNotFoundError, version
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "numpy": np.__version__, "processor": platform.processor(),
              "logical_cpus": os.cpu_count()}
    for name in ("torch", "hnswlib", "pandas", "torch-geometric"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "unavailable"
    return result


def _file(path, role, *, new=False, expected=None, cache=None):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    before = path.stat()
    key = (str(path), before.st_size, before.st_mtime_ns)
    digest = cache.get(key) if cache is not None else None
    if digest is None:
        digest = E._sha256(path)
        if cache is not None:
            cache[key] = digest
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"input changed while hashing: {path}")
    if expected is not None and digest != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    return {"path": str(path), "sha256": digest, "bytes": after.st_size,
            "role": role, "new_artifact": bool(new)}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_producer_envelope(record, expected, label):
    envelope = record.get("a0", {})
    for key, value in expected.items():
        _require(envelope.get(key) == value, f"{label} A0 producer binding mismatch: {key}")


def verify_chunked_export(meta, *, n_models=None):
    n_models = E.N_TOTAL if n_models is None else n_models
    gates = [gate for gate in meta.get("gates", []) if gate.get("gate") == "G-F7d"]
    _require(len(gates) == 1, "A0 export requires one newly measured G-F7d chunked agreement gate")
    gate = gates[0]
    _require(gate.get("ran") is True and gate.get("ok") is True,
             "A0 export chunked inference agreement was unmeasured or failed")
    deltas = gate.get("max_abs_delta", {})
    _require({"model", "dataset"} <= set(deltas)
             and all(isinstance(v, (float, int)) and np.isfinite(v) and 0 <= v < 1e-5 for v in deltas.values()),
             "A0 chunked agreement raw deltas fail the original threshold")
    _require(gate.get("n_models") == min(100000, n_models) and meta.get("chunk") == 50000,
             "A0 chunked agreement scope differs from the original 100K-node/50K-chunk check")
    return gate


def check_fixed_options(args):
    """Reject accidental default old paths or protocol drift before computing."""
    _require(args.stage in ("exact", "hnsw", "finalize", "all"),
             "A0 does not replay old frozen pools")
    _require(args.frozen_pools is None, "A0 cannot reuse historical frozen pools")
    _require(args.hnsw_M == 32 and args.ef_construction == 200,
             "A0 requires HNSW M=32, ef_construction=200")
    _require(list(args.ef_search) == GRID, "A0 requires original ordered ef grid")
    _require(args.hnsw_threads == 8, "A0 preserves eight HNSW construction threads")
    _require(args.model_chunk == 50000 and args.query_chunk == 16,
             "A0 preserves exact inference model/query chunks 50000/16")
    _require(bool(args.a0_run_id), "A0 run ID must be explicit and nonempty")
    _require(Path(args.out).resolve() != Path(E.data_root(), "data1m", "metrics_y2").resolve(),
             "A0 must use an isolated output directory")


def _verify_graph(path, old_digest, cache):
    graph = Path(path).resolve()
    meta = _read_json(graph / "meta.json")
    graph_check_key = ("verified_a0_graph", str(graph))
    if graph_check_key not in cache:
        from scale1m.a0_graph_validation import verify_a0_graph
        verified = verify_a0_graph(graph)
        cache[graph_check_key] = verified
    tag = meta.get("a0", {})
    _require(tag.get("protocol") == "a0", "export graph lacks A0 repair provenance")
    _require(tag.get("source_graph_digest") == old_digest,
             "A0 graph source differs from frozen A0.1 input")
    _require(tag.get("feature_repair") == "A0_FEATURE_REPAIR.json",
             "unexpected A0 repair report name")
    files = []
    for name, expected in sorted(meta["files"].items()):
        target = (graph / name).resolve()
        _require(target.parent == graph, "graph file table contains a nonlocal path")
        files.append(_file(target, "new_graph_file", new=True,
                           expected=expected, cache=cache))
    digest = _digest(meta["files"])
    _require(cache[graph_check_key]["graph_sha256"] == digest, "A0 graph validation/digest mismatch")
    _require(digest != old_digest, "A0 cannot evaluate the old graph")
    repair = _read_json(graph / tag["feature_repair"])
    _require(repair.get("protocol") == "a0" and repair.get("source_graph_digest") == old_digest,
             "repair report is not bound to original graph")
    _require(repair.get("zero_columns") == ZERO_COLUMNS,
             "repair columns differ from A0 seven-column contract")
    _require(repair.get("new_x_dataset_sha256") == meta["files"]["x_dataset.npy"],
             "repair report does not bind actual dataset features")
    _require(repair.get("root_count_preserved") is True and repair.get("builder_statistics_equal") is True,
             "A0 graph repair invariants did not pass")
    x = np.load(graph / "x_dataset.npy", mmap_mode="r", allow_pickle=False)
    _require(x.shape == (18729, 458) and x.dtype == np.float32,
             "A0 dataset feature shape/dtype changed")
    _require(np.all(x[:, ZERO_COLUMNS] == 0), "A0 performance columns remain nonzero")
    files.append(_file(graph / "meta.json", "new_graph_metadata", new=True, cache=cache))
    return graph, digest, files, meta


def _verify_embeddings(path, rows, cache):
    from ModelLakeFishing.scale1m.a0_recompute_checks import embedding_check
    role = "new_heldout_embedding" if "_eval" in Path(path).stem else "new_full_graph_embedding"
    rec = _file(path, role, new=True, cache=cache)
    rec["checked_values"] = embedding_check(path, rows)
    return rec


def _verify_query_identity(bundle, seed, identity):
    frozen = identity[identity.seed == seed].sort_values("query_mappedID")
    q = np.asarray(sorted(bundle["candidates"]), dtype=np.int64)
    _require(np.array_equal(q, frozen.query_mappedID.to_numpy(np.int64)),
             "query IDs differ from A0.1 frozen identity")
    for row in frozen.itertuples(index=False):
        d = int(row.query_mappedID)
        ids, weights = bundle["candidates"][d]
        _require(str(bundle["roots"][d]) == str(row.root), "query root differs from A0.1")
        _require(str(bundle["datasets"][d]) == str(row.node), "query node differs from A0.1")
        _require(_array_sha(ids.astype(np.int64)) == row.candidate_ids_sha256,
                 "held-out candidate IDs/order differ from A0.1")
        _require(_array_sha(weights.astype(np.float64)) == row.oriented_values_float64_sha256,
                 "held-out values differ from A0.1")


def create_binding(args):
    """Verify actual graph/checkpoint/export/sidecar inputs, without trusting filenames."""
    from ModelLakeFishing.scale1m import checkpoint as CK
    from ModelLakeFishing.scale1m.a0_recompute_checks import graph_checks, checkpoint_check, prior_checks, result as check_result

    cache = {}
    protocol_path = Path(args.a0_protocol_file).resolve()
    protocol = _read_json(protocol_path)
    source = protocol["authority"]
    source_files = [_file(protocol_path, "frozen_a0_protocol", cache=cache),
                    _file(source["path"], "sole_historical_authority", expected=source["sha256"], cache=cache)]
    input_ref = protocol["input_audit_binding"]["input_audit"]
    source_files.append(_file(input_ref["path"], "frozen_input_audit", expected=input_ref["sha256"], cache=cache))
    input_audit = _read_json(input_ref["path"])
    old_digest = input_audit["measurements"]["graph_digest"]
    query_ref = protocol["input_audit_binding"]["query_identity"]
    source_files.append(_file(args.a0_query_identity, "frozen_query_identity", expected=query_ref["sha256"], cache=cache))
    identity = pd.read_json(args.a0_query_identity, orient="records", lines=True,
                            dtype={"node": str, "root": str})
    wanted = next((r["sha256"] for r in input_audit["files"]
                   if Path(r["path"]).resolve() == Path(args.dataset_nodes).resolve()), None)
    _require(wanted is not None, "dataset node table is not the A0.1 frozen input path")
    source_files.append(_file(args.dataset_nodes, "frozen_dataset_nodes", expected=wanted, cache=cache))
    expected_cfg = protocol["required_config_audit"]["resolved_config"]
    repo = Path(__file__).resolve().parents[1]
    code_names = ["scale1m/eval_y2.py", "scale1m/a0_evaluation.py", "scale1m/eval_rf.py",
                  "scale1m/baselines.py", "scale/global_metrics.py", "scale1m/checkpoint.py",
                  "scale1m/export_rf.py", "stage3HNSW/build_prior_sidecar.py", "scale1m/train_rung.py",
                  "scale1m/a0_graph_validation.py", "scale1m/prepare_a0_graph.py", "scale1m/build_graph_rf.py",
                  "scale1m/a0_recompute_checks.py"]
    code_files = [_file(repo / name, "current_evaluation_code", cache=cache) for name in code_names]
    per_seed = {}
    audit_checks = {}
    tie_rank = E._tie_ranks(E.N_TOTAL)
    first_model = first_dataset = None
    for seed in E.SEEDS:
        bundle = E._audit_seed(args, seed, first_model, first_dataset, tie_rank)
        if first_model is None:
            first_model, first_dataset = bundle["models"], bundle["datasets"]
        _verify_query_identity(bundle, seed, identity)
        export = Path(bundle["x4"])
        export_man = export / "EXPORT_MANIFEST.json"
        meta = _read_json(export_man)["stages"]["embed"]
        _require(meta["split_seed"] == seed, "export split seed mismatch")
        verify_chunked_export(meta)
        graph, digest, graph_files, graph_meta = _verify_graph(meta["graph"], old_digest, cache)
        if not audit_checks:
            audit_checks.update(graph_checks(graph, input_audit, cache[("verified_a0_graph", str(graph))]))
            original_x = next(r for r in input_audit["files"] if Path(r["path"]).name == "x_dataset.npy")
            source_files.append(_file(original_x["path"], "frozen_pre_repair_dataset_features",
                                      expected=original_x["sha256"], cache=cache))
        _require(meta["graph_digest"] == digest and meta["binding"]["graph_sha256"] == digest,
                 "export is not bound to actual repaired graph")
        checkpoint_path = Path(meta["checkpoint"]).resolve()
        ckrec = _file(checkpoint_path, "new_epoch25_checkpoint", new=True, cache=cache)
        expected_envelope = {"protocol": "a0", "run_id": args.a0_run_id, "seed": seed,
                             "graph_digest": digest, "checkpoint_sha256": ckrec["sha256"]}
        verify_producer_envelope(meta, expected_envelope, "export")
        artifact_hashes = meta.get("artifact_hashes", {})
        required_export_files = {"z_m.npy", "z_d.npy", "z_m_eval.npy", "z_d_eval.npy",
                                 "gold_cands.npz", "model_ids.parquet", "dataset_ids.parquet"}
        _require(required_export_files <= set(artifact_hashes), "export generation hashes are incomplete")
        ck = CK.load(str(checkpoint_path))
        from scale1m.train_rung import reject_smoke_checkpoint
        reject_smoke_checkpoint(ck)
        _require(ck["binding"]["graph_sha256"] == digest and ck["binding"]["split_seed"] == seed,
                 "checkpoint graph/split binding mismatch")
        _require(int(ck["epoch"]) == 24 and int(meta["checkpoint_epoch"]) == 24 and len(ck["history"]) == 25,
                 "A0 requires completed 25 epochs (stored zero-based epoch 24)")
        _require(all(np.isfinite(v) for epoch in ck["history"] for v in epoch.values()
                     if isinstance(v, (int, float))), "checkpoint training history has nonfinite losses")
        cfg = {k: v for k, v in ck["cfg"].items() if k not in ("resume_state", "history0", "on_epoch_end")}
        _require(cfg == expected_cfg, "checkpoint effective training recipe differs from frozen GD")
        ck_detail = checkpoint_check(ck, graph, graph_meta, expected_cfg, seed)
        audit_checks[f"reproducibility_checks.checkpoint_graph_binding.{seed}"] = check_result(
            ck_detail, [checkpoint_path, graph / "meta.json"])
        run_meta_path = checkpoint_path.parent.parent / "metadata" / "resolved_config.json"
        run_meta = _read_json(run_meta_path)
        _require(run_meta["resolved_config"] == expected_cfg and run_meta["args"]["epochs"] == 25,
                 "formal run metadata is not frozen GD/25 epochs")
        _require(run_meta.get("graph_sha256") == digest, "formal run metadata graph binding mismatch")
        _require(not run_meta.get("args", {}).get("smoke_only", False), "smoke run metadata cannot evaluate")
        run_manifest_path = checkpoint_path.parent.parent / "MANIFEST.json"
        run_manifest = _read_json(run_manifest_path)
        _require(run_manifest.get("init_seed") == 0 and run_manifest.get("epochs") == 25
                 and run_manifest.get("seed") == seed and not run_manifest.get("smoke_only", False),
                 "formal run initialization/seed/epoch contract differs")
        run_records_path = checkpoint_path.parent.parent / "A0_RUN_RECORDS.json"
        run_records = _read_json(run_records_path)
        verify_producer_envelope(run_records, expected_envelope, "training records")
        _require(run_records["history"] == ck["history"], "training records differ from checkpoint history")
        files = graph_files + [ckrec, _file(export_man, "new_export_manifest", new=True, cache=cache),
                               _file(run_meta_path, "new_run_resolved_config", new=True, cache=cache),
                               _file(run_manifest_path, "new_formal_run_manifest", new=True, cache=cache),
                               _file(run_records_path, "new_training_resource_records", new=True, cache=cache)]
        for filename in sorted(required_export_files):
            files.append(_file(export / filename, "new_producer_bound_export_file", new=True,
                               expected=artifact_hashes[filename], cache=cache))
        embedding_details = {}
        for stem, rows in (("z_m_eval.npy", E.N_TOTAL), ("z_d_eval.npy", 18729),
                           ("z_m.npy", E.N_TOTAL), ("z_d.npy", 18729)):
            checked = _verify_embeddings(export / stem, rows, cache)
            files.append(checked)
            embedding_details[stem] = checked["checked_values"]
        for folder, role in ((export, "new_export"), (Path(bundle["f6"]), "new_sidecar_export")):
            for filename in ("model_ids.parquet", "dataset_ids.parquet"):
                rec = _file(folder / filename, role + "_row_map", new=True, cache=cache)
                graph_name = "unique_model_id.parquet" if filename.startswith("model") else "unique_dataset_id.parquet"
                graph_ids, _ = E._ids(graph / graph_name, "A0 graph")
                current_ids, _ = E._ids(folder / filename, role)
                E._same_ids(graph_ids, current_ids, "graph/export")
                files.append(rec)
        audit_checks[f"reproducibility_checks.embedding_shape_finite_norm_and_order.{seed}"] = check_result(
            {"passed": True, "embeddings": embedding_details, "model_rows": E.N_TOTAL, "dataset_rows": 18729},
            [export / name for name in embedding_details] + [export / "model_ids.parquet", export / "dataset_ids.parquet"])
        files.append(_file(export / "gold_cands.npz", "new_export_gold_labels", new=True, cache=cache))
        for filename in (f"prior_sidecar_s{seed}.npz", f"prior_sidecar_s{seed}_meta.json"):
            files.append(_file(Path(bundle["f6"]) / filename, "new_split_prior", new=True, cache=cache))
        prior_meta = bundle["prior"].meta
        verify_producer_envelope(prior_meta, expected_envelope, "sidecar")
        _require(prior_meta.get("graph_digest") == digest, "sidecar graph provenance mismatch")
        files.append(_file(Path(bundle["f6"]) / f"prior_sidecar_s{seed}.npz", "new_producer_bound_sidecar",
                           new=True, expected=prior_meta["sidecar_sha256"], cache=cache))
        payload = bundle["prior"].payload
        pairs = np.stack([payload["edge_model"], payload["edge_dataset"]]).astype(np.int64)
        original_split = next(s for s in input_audit["splits"] if s["seed"] == seed)
        _require(_array_sha(pairs) == original_split["prior_visible_pairs_ordered"]["sha256_c_order_bytes"],
                 "sidecar source edge order/identity differs from frozen split")
        with np.load(graph / "edges.npz", allow_pickle=False) as edges:
            idx = edges["model__trained_on__dataset__edge_index"]
            vals = edges["model__trained_on__dataset__edge_attr"]
            keys = idx[0] * 18729 + idx[1]
            order = np.argsort(keys)
            loc = np.searchsorted(keys[order], pairs[0] * 18729 + pairs[1])
            _require(np.array_equal(payload["edge_acc"].astype(np.float32), vals[order[loc]]),
                     "sidecar oriented weights differ from frozen graph")
        nodes = pd.read_parquet(args.dataset_nodes, columns=["node", "task"]).set_index("node")
        task = nodes["task"].reindex(bundle["datasets"])
        norm = [re.sub(r"[\s_]+", "-", str(t).strip().lower()) for t in task]
        vocab = {t: i for i, t in enumerate(sorted(set(norm)))}
        _require(np.array_equal(payload["task_id"], [vocab[t] for t in norm]),
                 "sidecar normalized task mapping differs from original rule")
        audit_checks.update(prior_checks(graph, export, bundle["f6"], seed,
            np.asarray(sorted(bundle["candidates"]), dtype=np.int64), args.dataset_nodes,
            input_audit, expected_envelope, bundle["models"], bundle["datasets"]))
        if ck["binding"].get("family_vocab_path"):
            files.append(_file(ck["binding"]["family_vocab_path"], "checkpoint_family_vocabulary",
                               expected=ck["binding"]["family_vocab_sha256"], cache=cache))
        per_seed[str(seed)] = {"graph_digest": digest, "files": files,
                              "export_dir": str(export.resolve()), "sidecar_dir": str(Path(bundle["f6"]).resolve())}
        bundle["prior"].payload.close()
        del ck, bundle
        gc.collect()
    return {"protocol_sha256": source_files[0]["sha256"], "old_graph_digest": old_digest,
            "run_id": args.a0_run_id, "N": E.N_TOTAL, "K": E.POOL_K, "return_k": 10,
            "code_files": code_files, "source_files": source_files, "seed_inputs": per_seed,
            "audit_checks": audit_checks,
            "runtime": runtime_versions(),
            "settings": {"M": args.hnsw_M, "ef_construction": args.ef_construction,
                         "ef_search": list(args.ef_search), "hnsw_threads": args.hnsw_threads,
                         "query_chunk": args.query_chunk, "model_chunk": args.model_chunk,
                         "device": args.device, "warmup": "min(50,Q)", "timed_passes": 1}}


def _artifact_path(out, relative):
    path = (Path(out) / relative).resolve()
    _require(path.parent == Path(out).resolve(), "A0 artifact must be directly inside its output directory")
    return path


def verify_artifacts(out, manifest):
    for name, rec in manifest["artifacts"].items():
        _require(rec.get("new_artifact") is True, "old artifact registered as new A0 output")
        _file(_artifact_path(out, name), rec["role"], expected=rec["sha256"])


def open_manifest(out, run_id, binding):
    out = Path(out)
    path = out / MANIFEST
    if path.exists():
        manifest = _read_json(path)
        _require(manifest.get("schema_version") == RAW_SCHEMA and manifest.get("protocol") == "a0",
                 "output directory is bound to another protocol")
        _require(manifest.get("run_id") == run_id and manifest.get("binding_sha256") == _digest(binding)
                 and manifest.get("binding") == binding, "A0 output cache binding changed; use an isolated new run")
        verify_artifacts(out, manifest)
        return manifest
    _require(not out.exists() or not any(out.iterdir()), "nonempty output directory lacks an A0 binding")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": RAW_SCHEMA, "protocol": "a0", "run_id": run_id,
                "binding": binding, "binding_sha256": _digest(binding),
                "artifacts": {}, "seeds": {}, "stage": "bound", "created_at": utcnow()}
    write_json_atomic(str(path), manifest)
    return manifest


def _save_manifest(out, manifest):
    manifest["written_at"] = utcnow()
    write_json_atomic(str(Path(out) / MANIFEST), manifest)


def register_artifact(out, manifest, name, role, seed):
    rec = _file(_artifact_path(out, name), role, new=True)
    rec["seed"] = seed
    manifest["artifacts"][name] = rec


def evaluation_resource_peaks(device):
    """Read real process/device high-water marks; never substitute current RSS."""
    from ModelLakeFishing.scale1m.train_rung import peak_process_rss
    import torch
    rss, rss_reason = peak_process_rss()
    try:
        if torch.device(device).type == "cuda" and torch.cuda.is_available():
            gpu, gpu_reason = int(torch.cuda.max_memory_allocated(torch.device(device))), None
        else:
            gpu, gpu_reason = None, "evaluation device is CPU or CUDA is unavailable"
    except (RuntimeError, ValueError) as exc:
        gpu, gpu_reason = None, f"{type(exc).__name__}: {exc}"
    return {"rss_bytes": rss, "vram_bytes": gpu,
            "rss_unavailable_reason": rss_reason, "vram_unavailable_reason": gpu_reason}


def _evaluation_envelope(manifest, seed):
    seed_binding = manifest["binding"]["seed_inputs"][str(seed)]
    checkpoints = [r for r in seed_binding["files"] if r["role"] == "new_epoch25_checkpoint"]
    _require(len(checkpoints) == 1, "evaluation records require one bound epoch25 checkpoint")
    return {"protocol": "a0", "run_id": manifest["run_id"], "seed": seed,
            "graph_digest": seed_binding["graph_digest"], "checkpoint_sha256": checkpoints[0]["sha256"]}


def _summarize_evaluation_record(record):
    """Unfinished process segments remain missing after a resume."""
    all_segments = record["exact_segments"] + record["hnsw_segments"]
    stage_seconds = {}
    for stage in ("exact", "hnsw"):
        segments = record[stage + "_segments"]
        completed = any(s["status"] == "completed" for s in segments)
        known = bool(segments) and all(s["end_ns"] is not None for s in segments)
        stage_seconds[stage] = sum(s["end_ns"] - s["start_ns"] for s in segments) / 1e9 if completed and known else None
    record["exact_full_reference_seconds"] = stage_seconds["exact"]
    record["exact1000_reference_seconds"] = stage_seconds["exact"]
    record["hnsw_evaluation_seconds"] = stage_seconds["hnsw"]
    record["timing_complete"] = all(v is not None for v in stage_seconds.values())
    record["complete_evaluation_seconds"] = sum(stage_seconds.values()) if record["timing_complete"] else None
    for kind, key in (("rss", "evaluation_peak_rss_bytes"), ("vram", "evaluation_peak_vram_bytes")):
        samples = [s.get("resource_peaks", {}).get(kind + "_bytes") for s in all_segments]
        available = [v for v in samples if v is not None]
        record[key + "_observed_max"] = max(available) if available else None
        complete = bool(all_segments) and all(s["end_ns"] is not None for s in all_segments) and all(v is not None for v in samples)
        record[key] = max(available) if complete else None
        record[key + "_complete"] = complete
    record["status"] = "completed" if all(any(s["status"] == "completed" for s in record[k + "_segments"])
                                          for k in ("exact", "hnsw")) else "in_progress"


def _save_evaluation_record(args, manifest, seed, record):
    name = f"A0_EVAL_RECORDS_s{seed}.json"
    _summarize_evaluation_record(record)
    record["written_at"] = utcnow()
    write_json_atomic(str(_artifact_path(args.out, name)), record)
    register_artifact(args.out, manifest, name, "new_evaluation_resource_records", seed)
    manifest["seeds"].setdefault(str(seed), {})["evaluation_records"] = name
    _save_manifest(args.out, manifest)


def begin_evaluation_record(args, manifest, seed, stage):
    name = f"A0_EVAL_RECORDS_s{seed}.json"
    path = _artifact_path(args.out, name)
    envelope = _evaluation_envelope(manifest, seed)
    if path.exists():
        _require(name in manifest["artifacts"], "unbound evaluation records cannot be reused")
        _file(path, "new_evaluation_resource_records", expected=manifest["artifacts"][name]["sha256"])
        record = _read_json(path)
        verify_producer_envelope(record, envelope, "evaluation records")
        _require(record["binding_sha256"] == manifest["binding_sha256"], "evaluation record binding differs")
    else:
        record = {"schema_version": "a0.evaluation.resources.v1", "a0": envelope,
            "binding_sha256": manifest["binding_sha256"], "exact_segments": [], "hnsw_segments": [],
            "timing_scope": {
                "exact": "per-seed input audit, one shared exact scan producing full and exact1000 references, raw scoring/serialization and cleanup",
                "exact_reference_overlap": "exact_full_reference_seconds and exact1000_reference_seconds describe the SAME shared segment; never add them",
                "hnsw": "per-seed input audit, index build or bound index load, calibration, timed queries, raw scoring/serialization and cleanup",
                "complete_evaluation_seconds": "sum exact and HNSW segments once, including measured failed attempts; excludes shared create_binding/finalize and idle gaps between invocations",
                "interruption": "null end_ns is missing evidence and is never replaced by zero; aggregate stays null after resume"},
            "evaluation_peak_rss_scope": "OS process-lifetime high-water mark observed at each seed-stage end; may include earlier seeds and shared binding; maximum over recorded processes, not isolated seed RSS",
            "evaluation_peak_vram_scope": "PyTorch max_memory_allocated for evaluation device observed at each seed-stage end; no peak reset; may include earlier seeds in this process",
            "created_at": utcnow()}
    for existing_stage in ("exact", "hnsw"):
        for segment in record[existing_stage + "_segments"]:
            if segment["status"] == "running":
                segment["status"] = "interrupted_end_unobserved"
    record[stage + "_segments"].append({"start_ns": time.perf_counter_ns(), "end_ns": None,
        "status": "running", "clock": "perf_counter_ns", "pid": os.getpid(), "device": str(args.device)})
    _save_evaluation_record(args, manifest, seed, record)
    return record


def finish_evaluation_record(args, manifest, seed, stage, record, status="completed"):
    segment = record[stage + "_segments"][-1]
    _require(segment["status"] == "running", "evaluation segment was already closed")
    segment.update(end_ns=time.perf_counter_ns(), status=status,
                   resource_peaks=evaluation_resource_peaks(args.device))
    _save_evaluation_record(args, manifest, seed, record)


def fail_active_evaluation_record(args, manifest, stage):
    """Caught failures get real endpoints; hard kills retain the saved null."""
    for seed in E.SEEDS:
        name = manifest["seeds"].get(str(seed), {}).get("evaluation_records")
        if name is None:
            continue
        record = _read_json(_artifact_path(args.out, name))
        if record[stage + "_segments"] and record[stage + "_segments"][-1]["status"] == "running":
            finish_evaluation_record(args, manifest, seed, stage, record, status="failed")


def save_npz(out, name, **arrays):
    target = _artifact_path(out, name)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, target)


def label_arrays(queries, candidates, roots, prior, *, n_models=None):
    n_models = E.N_TOTAL if n_models is None else n_models
    ids, values, offsets = [], [], [0]
    for query in queries:
        c, a = candidates[int(query)]
        ids.append(np.asarray(c, dtype=np.int64))
        values.append(np.asarray(a, dtype=np.float64))
        offsets.append(offsets[-1] + len(c))
    return {"query": np.asarray(queries, dtype=np.int64),
            "root": np.asarray([str(roots[int(q)]) for q in queries]),
            "task_id": np.asarray([prior.task_id[int(q)] for q in queries], dtype=np.int64),
            "observed_offsets": np.asarray(offsets, dtype=np.int64),
            "observed_ids": np.concatenate(ids), "observed_values": np.concatenate(values),
            "n_models": np.asarray(n_models, dtype=np.int64)}


def pool_arrays(ids, scores, queries, candidates, roots, prior, tie_rank, *, n_models=None):
    n_models = E.N_TOTAL if n_models is None else n_models
    ids, scores = np.asarray(ids, dtype=np.int64), np.asarray(scores, dtype=np.float32)
    _require(ids.shape == scores.shape == (len(queries), E.POOL_K), "invalid A0 pool shape")
    _require(ids.min(initial=0) >= 0 and ids.max(initial=0) < n_models, "pool IDs out of range")
    _require(np.isfinite(scores).all(), "pool scores must be finite")
    row, per, top10 = E._pool_metrics(ids, scores, list(map(int, queries)), candidates, roots, prior, tie_rank)
    priors = np.stack([prior.values(int(q), ids[i]) for i, q in enumerate(queries)])
    _require(np.isfinite(priors).all(), "prior has nonfinite values")
    fused = (scores + np.float32(1)) * np.float32(0.5) + priors
    counts = np.asarray([[per[int(q)][k] - 1 for k in ("gold_rank", "top3_rank", "gap_rank")]
                         for q in queries], dtype=np.int64)
    position = counts[:, 0] + 1
    position[position > E.POOL_K] = 0
    row["gold_retrieved_query_count"] = int(np.count_nonzero(position))
    row["candidate_universe_N"] = n_models
    row["rerank_K"] = E.POOL_K
    row["return_k"] = 10
    return row, {"model": ids, "score": scores, "prior": priors, "fused": fused,
                 "top10": top10, "pool_counts": counts, "gold_position": position}


def numeric_summary(values):
    if len(values) != 3:
        return {"status": "missing", "per_seed": values, "mean": None, "min": None, "max": None}
    if any(v is None for v in values):
        return {"status": "undefined", "per_seed": values, "mean": None, "min": None, "max": None}
    _require(all(np.isfinite(v) for v in values), "nonfinite summary input")
    return {"status": "recomputed", "per_seed": values, "mean": float(np.mean(values)),
            "min": float(min(values)), "max": float(max(values))}


def summarize_rows(per_seed):
    _require(set(per_seed) == set(map(str, E.SEEDS)), "summary requires all three seeds")
    names = set(per_seed["0"]["rows"])
    _require(all(set(per_seed[str(s)]["rows"]) == names for s in E.SEEDS), "seed row schemas differ")
    out = {}
    for name in sorted(names):
        keys = set(per_seed["0"]["rows"][name])
        _require(all(set(per_seed[str(s)]["rows"][name]) == keys for s in E.SEEDS), "seed metric schemas differ")
        out[name] = {key: numeric_summary([per_seed[str(s)]["rows"][name][key] for s in E.SEEDS])
                     for key in sorted(keys)}
    return out


def ratio_summary(numerators, denominators):
    ratios = [n / d if d != 0 else None for n, d in zip(numerators, denominators)]
    out = numeric_summary(ratios)
    out.update({"numerators": numerators, "denominators": denominators,
                "aggregation": "mean of three seed ratios; never ratio of means",
                "unit": "fraction", "undefined_reason": "zero denominator" if None in ratios else None})
    return out


def exact_decision(report):
    full = [report["per_seed"][str(s)]["rows"]["G_full_task"]["gold@10"] for s in E.SEEDS]
    dense = [report["per_seed"][str(s)]["rows"]["G_dense"]["gold@10"] for s in E.SEEDS]
    exact = [report["per_seed"][str(s)]["rows"]["G_exact1000_task"]["gold@10"] for s in E.SEEDS]
    retention = ratio_summary(exact, full)
    gain = bool(np.mean(full) > np.mean(dense))
    gate = gain and retention["mean"] is not None and retention["mean"] >= 0.90
    return {"prior_improves_x4_mean": gain, "exact_pool_retention": retention,
            "original_exact_effectiveness_gate": bool(gate),
            "build_hnsw": bool(gate), "a0_complete_measurement_despite_effectiveness_gate": True}


def _write_report(out, report):
    report["written_at"] = utcnow()
    # Strict JSON rejects NaN/Infinity before the existing atomic writer.
    json.dumps(report, allow_nan=False)
    write_json_atomic(str(Path(out) / REPORT), report)


def _initial_report(manifest):
    return {"schema_version": "a0.evaluator.v1", "protocol": "a0", "run_id": manifest["run_id"],
            "binding_sha256": manifest["binding_sha256"], "stage": "bound",
            "primary_path": "G_hnsw1000_task", "N": E.N_TOTAL, "K": E.POOL_K, "return_k": 10,
            "per_seed": {}, "hnsw": {}, "correctness_gates": {},
            "historical_reference": "A0_OLD_REFERENCE.json; no historical measurements used here",
            "documented_exact_corrections": {
                "full_top10_ties": "A0-only fixed label-free tie order agrees with rank counts; legacy torch.topk unchanged",
                "probe_arithmetic": "A0-only probes read same-shaped float32 block GEMM as full scan; legacy elementwise reduction unchanged",
                "precision": "float32 unchanged; no tolerance or score rounding",
                "cost": "exact reference includes extra GEMM for blocks containing observed probes"}}


def run_exact(args, manifest, report):
    tie_rank = E._tie_ranks(E.N_TOTAL)
    for seed in E.SEEDS:
        entry = manifest["seeds"].setdefault(str(seed), {})
        if entry.get("exact"):
            _require(str(seed) in report["per_seed"], "completed raw exact artifact has no matching evaluator record")
            continue
        resource_record = begin_evaluation_record(args, manifest, seed, "exact")
        bundle = E._audit_seed(args, seed, None, None, tie_rank)
        result = E.evaluate_exact_seed(bundle["x4"], bundle["prior"], bundle["candidates"],
            bundle["roots"], tie_rank, args.device, args.model_chunk, args.query_chunk, protocol="a0")
        queries = np.asarray(result["queries"], dtype=np.int64)
        row, arrays = pool_arrays(result["exact_pool_ids"], result["exact_pool_scores"], queries,
                                  bundle["candidates"], bundle["roots"], bundle["prior"], tie_rank)
        row["full_fused_top10_in_dense_top1000"] = float(np.mean([
            len(set(result["full_top10"][i]) & set(arrays["model"][i])) / 10.0 for i in range(len(queries))]))
        result["rows"]["G_exact1000_task"] = row
        # Rank sufficient statistics and actual saved full top-10 must agree.
        from_top, _ = E._top10_metrics(result["full_top10"], queries.tolist(), bundle["candidates"], bundle["roots"])
        for key in ("gold@1", "gold@10", "gold-gap@1", "gold-gap@10", "top3@10",
                    "root_gold@1", "root_gold@10", "root_top3@10", "root_gold-gap@10"):
            _require(from_top[key] == result["rows"]["G_full_task"][key],
                     f"full-fused saved top10/rank disagreement: {key}; do not silently score different ties")
        arrays.update(label_arrays(queries, bundle["candidates"], bundle["roots"], bundle["prior"]))
        arrays.update(seed=np.asarray(seed), full_top10=result["full_top10"],
                      full_counts=result["full_counts"], dense_counts=result["dense_counts"])
        name = f"a0_exact_s{seed}.npz"
        save_npz(args.out, name, **arrays)
        register_artifact(args.out, manifest, name, "exact_pool_labels_full_rank_statistics", seed)
        report["per_seed"][str(seed)] = {"rows": result["rows"], "seconds": result["seconds"], "pool_artifact": name,
            "dense_control_scope": "same GD representation; diagnostic for original prior-gain gate only",
            "checks": {"full_top10_matches_full_rank_metrics": True, "fresh_labels_and_bound_inputs": True}}
        entry["exact"] = name
        _write_report(args.out, report)
        _save_manifest(args.out, manifest)
        bundle["prior"].payload.close()
        del bundle, result, arrays
        gc.collect()
        finish_evaluation_record(args, manifest, seed, "exact", resource_record)
    report["summary"] = summarize_rows(report["per_seed"])
    report["decision"] = exact_decision(report)
    report["stage"] = manifest["stage"] = "exact-complete"
    _write_report(args.out, report)
    _save_manifest(args.out, manifest)


def recall_by_query(ids, exact):
    _require(ids.shape == exact.shape, "calibration pool/query shapes differ")
    return np.asarray([len(set(a.tolist()) & set(b.tolist())) / exact.shape[1]
                       for a, b in zip(ids, exact)], dtype=np.float64)


def calibrate(index, zq, queries, exact_ids, threads, save_attempt, *, n_models=None):
    """No gold arguments. Stop at first pass; otherwise retain maximum-ef data."""
    n_models = E.N_TOTAL if n_models is None else n_models
    trace, chosen = [], None
    for ef in GRID:
        index.set_ef(ef)
        ids, dist = index.knn_query(zq, k=E.POOL_K, num_threads=threads)
        ids = np.asarray(ids, dtype=np.int64)
        score = (np.float32(1) - dist).astype(np.float32)
        _require(ids.shape == exact_ids.shape and np.isfinite(score).all(), "invalid ANN calibration output")
        _require(ids.min(initial=0) >= 0 and ids.max(initial=0) < n_models
                 and all(len(np.unique(row)) == E.POOL_K for row in ids), "invalid calibration IDs")
        per_query = recall_by_query(ids, exact_ids)
        value = float(per_query.mean())
        artifact = save_attempt(ef, ids, score, per_query)
        trace.append({"ef_search": ef, "recall@1000": value, "artifact": artifact})
        chosen = (ef, ids, score, value)
        print(f"[A0 calibration] ef={ef} recall@1000={value:.8f}", flush=True)
        if value >= 0.99:
            break
    return chosen, trace, bool(chosen[3] >= 0.99)


def measure_queries(index, zq, queries, prior, tie_rank):
    for i in range(min(50, len(queries))):
        index.knn_query(zq[i:i + 1], k=E.POOL_K, num_threads=1)
    models, scores, hnsw, rerank, total = [], [], [], [], []
    for i, query in enumerate(queries):
        t0 = time.perf_counter_ns()
        ids, dist = index.knn_query(zq[i:i + 1], k=E.POOL_K, num_threads=1)
        t1 = time.perf_counter_ns()
        score = 1.0 - dist[0]
        fused = (score + 1.0) * 0.5 + prior.values(int(query), ids[0])
        np.lexsort((tie_rank[ids[0]], -fused))[:10]
        t2 = time.perf_counter_ns()
        models.append(ids[0]); scores.append(score)
        hnsw.append(t1-t0); rerank.append(t2-t1); total.append(t2-t0)
    return np.asarray(models, dtype=np.int64), np.asarray(scores, dtype=np.float32), {
        "hnsw_ns": np.asarray(hnsw, dtype=np.int64), "rerank_ns": np.asarray(rerank, dtype=np.int64),
        "total_ns": np.asarray(total, dtype=np.int64)}


def latency_summary(timing):
    _require(np.array_equal(timing["hnsw_ns"] + timing["rerank_ns"], timing["total_ns"]),
             "raw total time differs from same-query stage sum")
    return {f"{label}_p{q}": float(np.percentile(timing[key].astype(np.float64) / 1e6, q))
            for label, key in (("hnsw", "hnsw_ns"), ("rerank", "rerank_ns"), ("end_to_end", "total_ns"))
            for q in (50, 95)}


def run_hnsw(args, manifest, report):
    import hnswlib

    _require(all("exact" in manifest["seeds"].get(str(s), {}) for s in E.SEEDS),
             "all new exact diagnostics must complete before HNSW")
    tie_rank = E._tie_ranks(E.N_TOTAL)
    for seed in E.SEEDS:
        entry = manifest["seeds"][str(seed)]
        if entry.get("hnsw"):
            _require(str(seed) in report["hnsw"], "completed HNSW raw artifact has no evaluator record")
            continue
        resource_record = begin_evaluation_record(args, manifest, seed, "hnsw")
        bundle = E._audit_seed(args, seed, None, None, tie_rank)
        with np.load(_artifact_path(args.out, entry["exact"]), allow_pickle=False) as exact:
            queries, exact_ids = exact["query"], exact["model"]
        _require(np.array_equal(queries, np.asarray(sorted(bundle["candidates"]))), "HNSW query ID mismatch")
        zm = np.load(Path(bundle["x4"]) / "z_m_eval.npy", mmap_mode="r", allow_pickle=False)
        zd = np.load(Path(bundle["x4"]) / "z_d_eval.npy", mmap_mode="r", allow_pickle=False)
        name = f"hnsw_a0_s{seed}.bin"
        target = _artifact_path(args.out, name)
        index = hnswlib.Index(space="ip", dim=128)
        if entry.get("index"):
            _require(entry["index"] == name and name in manifest["artifacts"], "index cache is not registered")
            index.load_index(str(target), max_elements=E.N_TOTAL)
        else:
            _require(not target.exists(), "unbound preexisting HNSW index cannot be reused")
            t0 = time.perf_counter_ns()
            index.init_index(max_elements=E.N_TOTAL, ef_construction=200, M=32)
            index.add_items(zm, np.arange(E.N_TOTAL, dtype=np.int64), num_threads=args.hnsw_threads)
            temporary = str(target) + ".tmp"
            index.save_index(temporary)
            os.replace(temporary, target)
            t1 = time.perf_counter_ns()
            entry["index_build"] = {"start_ns": t0, "end_ns": t1,
                                    "build_seconds": (t1-t0)/1e9, "clock": "perf_counter_ns"}
            entry["index"] = name
            register_artifact(args.out, manifest, name, "new_hnsw_evaluation_index", seed)
            _save_manifest(args.out, manifest)
        index.set_num_threads(args.hnsw_threads)
        zq = np.asarray(zd[queries], dtype=np.float32)
        entry["calibration"] = []

        def save_attempt(ef, ids, scores, recalls):
            filename = f"a0_calibration_s{seed}_ef{ef}.npz"
            save_npz(args.out, filename, seed=np.asarray(seed), query=queries, model=ids,
                     score=scores, recall_per_query=recalls, ef_search=np.asarray(ef))
            register_artifact(args.out, manifest, filename, "new_calibration_raw_ids", seed)
            entry["calibration"].append(filename)
            _save_manifest(args.out, manifest)
            return filename

        chosen, trace, passed = calibrate(index, zq, queries, exact_ids, args.hnsw_threads, save_attempt)
        ef, calibration_ids, _calibration_scores, recall = chosen
        index.set_ef(ef)
        ids, scores, timing = measure_queries(index, zq, queries, bundle["prior"], tie_rank)
        row, arrays = pool_arrays(ids, scores, queries, bundle["candidates"], bundle["roots"], bundle["prior"], tie_rank)
        arrays.update(label_arrays(queries, bundle["candidates"], bundle["roots"], bundle["prior"]))
        arrays.update(timing)
        arrays.update(seed=np.asarray(seed), selected_ef=np.asarray(ef), calibration_ids=calibration_ids)
        raw_name = f"a0_hnsw_s{seed}.npz"
        save_npz(args.out, raw_name, **arrays)
        register_artifact(args.out, manifest, raw_name, "new_hnsw_pool_labels_and_query_timings", seed)
        entry.update(hnsw=raw_name, selected_ef=ef, calibration_passed=passed)
        report["hnsw"][str(seed)] = {"rows": {"G_hnsw1000_task": row}, "index": name,
            "index_bytes": target.stat().st_size, "index_gib": target.stat().st_size/2**30,
            "build_seconds": entry["index_build"]["build_seconds"], "ef_trace": trace,
            "ef_search": ef, "recall@1000": recall, "gate_recall@1000": passed,
            "calibration_status": "passed" if passed else "failed_measured_at_maximum_ef5000",
            "unvisited_ef": [{"ef_search": v, "status": "not_applicable", "reason": "original first-passing stop rule"}
                             for v in GRID if v > ef],
            "latency_ms": latency_summary(timing), "raw_artifact": raw_name}
        _write_report(args.out, report)
        _save_manifest(args.out, manifest)
        bundle["prior"].payload.close()
        del index, bundle, zm, zd, zq, arrays
        gc.collect()
        finish_evaluation_record(args, manifest, seed, "hnsw", resource_record)
    report["hnsw_summary"] = summarize_rows(report["hnsw"])["G_hnsw1000_task"]
    full = [report["per_seed"][str(s)]["rows"]["G_full_task"]["gold@10"] for s in E.SEEDS]
    exact = [report["per_seed"][str(s)]["rows"]["G_exact1000_task"]["gold@10"] for s in E.SEEDS]
    actual = [report["hnsw"][str(s)]["rows"]["G_hnsw1000_task"]["gold@10"] for s in E.SEEDS]
    ann, overall = ratio_summary(actual, exact), ratio_summary(actual, full)
    report["decision"].update({"ann_retention": ann, "overall_retention": overall,
        "hnsw_deployment_gate": bool(ann["mean"] is not None and ann["mean"] >= 0.98),
        "all_hnsw_recall_passed": all(report["hnsw"][str(s)]["gate_recall@1000"] for s in E.SEEDS)})
    report["stage"] = manifest["stage"] = "measurement-complete"
    _write_report(args.out, report)
    _save_manifest(args.out, manifest)


def finalize(args, manifest, report):
    _require(all("hnsw" in manifest["seeds"].get(str(s), {}) for s in E.SEEDS), "all three A0 HNSW measurements are required")
    verify_artifacts(args.out, manifest)
    report["summary"] = summarize_rows(report["per_seed"])
    report["hnsw_summary"] = summarize_rows(report["hnsw"])["G_hnsw1000_task"]
    report["latency_summary_ms"] = {
        key: numeric_summary([report["hnsw"][str(s)]["latency_ms"][key] for s in E.SEEDS])
        for key in report["hnsw"]["0"]["latency_ms"]}
    report["index_cost"] = {key: numeric_summary([report["hnsw"][str(s)][key] for s in E.SEEDS])
                            for key in ("index_bytes", "index_gib", "build_seconds")}
    for value in report["index_cost"].values():
        value["sum"] = sum(value["per_seed"])
    report["query_observations"] = sum(report["hnsw"][str(s)]["rows"]["G_hnsw1000_task"]["n_queries"] for s in E.SEEDS)
    report["correctness_gates"].update(bound_new_artifacts=True, same_query_identity=True,
        fresh_graph_epoch25_embeddings=True, fixed_K1000_prior_and_ties=True, all_three_seeds=True)
    report["effectiveness_gates"] = {key: report["decision"][key] for key in (
        "prior_improves_x4_mean", "original_exact_effectiveness_gate", "hnsw_deployment_gate", "all_hnsw_recall_passed")}
    report["environment"] = {"hostname": platform.node(), "platform": platform.platform(),
        "processor": platform.processor(), "logical_cpus": os.cpu_count(), "python": sys.version,
        "numpy": np.__version__, "timing": "one pass after min(50,Q) warmup; HNSW one query thread; precomputed query embedding",
        "percentile": "NumPy default linear; total computed from raw per-query total_ns"}
    for package in ("torch", "hnswlib", "pandas", "torch-geometric"):
        from importlib.metadata import version
        try: report["environment"][package] = version(package)
        except Exception: report["environment"][package] = "unavailable"
    report["stage"] = manifest["stage"] = "complete"
    report["ann_fidelity_status"] = "passed" if report["decision"]["all_hnsw_recall_passed"] else "failed_original_threshold_measurement_completed"
    report["computation_complete"] = True
    report["meets_all_original_effectiveness_gates"] = all(report["effectiveness_gates"].values())
    _write_report(args.out, report)
    register_artifact(args.out, manifest, REPORT, "new_evaluator_summary_not_raw_measurement", None)
    _save_manifest(args.out, manifest)


def run(args):
    check_fixed_options(args)
    binding = create_binding(args)
    manifest = open_manifest(args.out, args.a0_run_id, binding)
    path = Path(args.out) / REPORT
    report = _read_json(path) if path.exists() else _initial_report(manifest)
    _require(report.get("binding_sha256") == manifest["binding_sha256"], "evaluator report has different input binding")
    if manifest["stage"] == "complete":
        return report
    for stage, function in (("exact", run_exact), ("hnsw", run_hnsw)):
        if args.stage not in (stage, "all"):
            continue
        segment = {"stage": stage, "start_ns": time.perf_counter_ns(), "end_ns": None,
                   "status": "running", "clock": "perf_counter_ns", "includes": "this invocation only; completed cached seeds excluded"}
        report.setdefault("stage_segments", []).append(segment)
        _write_report(args.out, report)
        try:
            function(args, manifest, report)
        except BaseException:
            fail_active_evaluation_record(args, manifest, stage)
            segment.update(end_ns=time.perf_counter_ns(), status="failed")
            _write_report(args.out, report)
            raise
        else:
            segment.update(end_ns=time.perf_counter_ns(), status="completed")
            _write_report(args.out, report)
    if args.stage in ("finalize", "all"):
        finalize(args, manifest, report)
    return report
