"""Deferred read-only recount of A0 frozen source statistics, never retrieval.

Run after A0.2, when explicitly executing the missing source audit::

  python -m scale1m.a0_source_counts --protocol <A0_PROTOCOL.json>
      --out <new audit/A0_SOURCE_COUNTS.json>

Native model-index rows are streamed from the hash-verified frozen gzip shards.
The original pure build_supervision function derives dedupe/primary/cap counts
in memory; canonical outputs are never overwritten. Five original historical
graphs plus the A0.2 source-identity SHA256 bindings are required for merge counts.
The final edge, node and conflict tables must all exactly match A0.1 frozen
outputs. Current source hashes alone do not establish historical byte identity.
API page and skipped-duplicate event counts remain missing without the original
request/discard event stream: retained records and provenance totals cannot
reconstruct them. Exit 2 preserves a partial report with these explicit gaps.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scale1m import canonicalize_rf as C

HISTORICAL_NAMES = ["hgraph_ml_v2.pt", "hgraph_d05_nocontent.pt",
                    "hgraph_A_ctrl_2000m_xm0_xd0.pt", "hgraph_hf_effective_2000m_v2_xm0_xd0.pt",
                    "hgraph_diverse_xd0.pt"]
MERGE_NAMES = ["merged_input_rows", "merged_within_source_duplicates_removed",
               "merged_cross_source_conflicts", "merged_pairs_before_cap"]
EVENT_GAPS = {
    "model_snapshot_api_pages": "Missing complete original per-request event log with request identity, page/response boundaries and retry policy; retained shards contain no page boundaries.",
    "model_snapshot_skipped_duplicates": "Missing original duplicate-discard event log with occurrence identity; discarded occurrences are absent from retained snapshot records.",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_json(record):
    path = Path(record["path"])
    if digest(path) != record["sha256"]:
        raise ValueError(f"Frozen audit hash mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def native_from_records(records):
    """Use original parse/direction/primary/cap rules without writing outputs."""
    rows, raw = [], 0
    for record in records:
        model = C.normalize(record.get("id"))
        for dataset, task, metric_raw, original in C.parse_model_index(record):
            raw += 1
            value = C.to_float(original)
            if value is None:
                continue
            metric, _base, direction = C.classify(metric_raw)
            rows.append((model, dataset, task, metric, direction, value))
    frame = pd.DataFrame(rows, columns=["model", "dataset", "task", "metric", "direction", "value"])
    if frame.empty:
        counts = {name: 0 for name in ("native_finite_metric_rows", "native_median_deduplicated_rows",
                                       "native_primary_edges_before_cap", "native_primary_edges_after_cap")}
        counts["native_raw_metric_rows"] = raw
        return counts, pd.DataFrame(columns=["node", "model", "weight"])
    for key in ("dataset", "task", "metric", "direction"):
        frame[key] = frame[key].astype("category")
    capped, _nodes, _deduplicated, report = C.build_supervision(frame, cap=200)
    return {"native_raw_metric_rows": raw, "native_finite_metric_rows": len(frame),
            "native_median_deduplicated_rows": report["after_dedupe_rows"],
            "native_primary_edges_before_cap": report["edges_before_cap"],
            "native_primary_edges_after_cap": report["edges_after_cap"]}, capped


def snapshot_records(snapshot, inputs):
    directory = Path(snapshot["directory"])
    for shard in snapshot["shards"]:
        path = directory / shard["file"]
        expected = shard["sha256_after_stream"]
        if digest(path) != expected:
            raise ValueError(f"Frozen shard changed: {path}")
        inputs.append({"path": str(path.resolve()), "sha256": expected, "role": "original_frozen_model_snapshot"})
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
        if digest(path) != expected:
            raise ValueError(f"Frozen shard changed during recount: {path}")


def count_training_similarity(edge_index, n_datasets, k=10):
    """Count exact graph-surgery output: symmetric unique nonself top-k degree."""
    pairs = np.asarray(edge_index, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[0] != 2:
        raise ValueError("Invalid similarity endpoints")
    if np.any(pairs < 0) or np.any(pairs >= n_datasets):
        raise ValueError("Similarity endpoint out of range")
    neighbors = [set() for _ in range(n_datasets)]
    for left, right in pairs.T:
        if left != right:
            neighbors[int(left)].add(int(right))
            neighbors[int(right)].add(int(left))
    return sum(min(k, len(group)) for group in neighbors)


def graph_source_counts(graph, expected_files, inputs):
    graph = Path(graph)
    for name in ("nodes.npz", "edges.npz", "x_model.npy", "x_dataset.npy"):
        path = graph / name
        if str(path.resolve()) not in expected_files or digest(path) != expected_files[str(path.resolve())]:
            raise ValueError(f"Graph source is not A0.1 hash-bound: {path}")
        inputs.append({"path": str(path.resolve()), "sha256": expected_files[str(path.resolve())], "role": "original_frozen_graph"})
    with np.load(graph / "nodes.npz", allow_pickle=False) as archive:
        categories = {key: np.asarray(archive[key]) for key in archive.files if key.endswith("_id") and not key.endswith("node_id")}
        n_datasets = len(archive["dataset.node_id"])
    shapes = {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in categories.items()}
    for name in ("x_model.npy", "x_dataset.npy"):
        array = np.load(graph / name, mmap_mode="r", allow_pickle=False)
        shapes[name] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    with np.load(graph / "edges.npz", allow_pickle=False) as edges:
        similarity = edges["dataset__similar_to__dataset__edge_index"]
        weights = edges["dataset__similar_to__dataset__edge_attr"]
        if not np.isfinite(weights).all():
            raise ValueError("Similarity count requires finite original weights")
        count = count_training_similarity(similarity, n_datasets)
    # Cardinality reports both the observed categories and required index range.
    cardinalities = {key: {"observed_unique": int(len(np.unique(value))),
                          "embedding_index_cardinality": int(value.max()) + 1 if value.size else 0}
                     for key, value in categories.items()}
    result = {"training_similar_to_edges": count, "categorical_vocab_cardinalities": cardinalities,
            "node_feature_shapes_and_dtypes": shapes,
            "model_feature_file_gb": (graph / "x_model.npy").stat().st_size / 10**9}
    for name, key in [("unknown_model_size_percent", "model.size_bucket_id"),
                      ("other_model_family_percent", "model.family_id")]:
        values = categories[key]
        result[name] = {"numerator": int(np.sum(values == 0)), "denominator": int(values.size),
                        "scale": 100, "value": float(100 * np.mean(values == 0))}
    return result


def equal_edges(left, right, keys):
    """Compare actual keys/weights, permitting parquet categorical storage changes."""
    left, right = left[keys].copy(), right[keys].copy()
    for key in keys:
        if key != "weight":
            left[key], right[key] = left[key].astype(str), right[key].astype(str)
        else:
            left[key], right[key] = left[key].astype(np.float64), right[key].astype(np.float64)
    return left.sort_values(keys[:2]).reset_index(drop=True).equals(right.sort_values(keys[:2]).reset_index(drop=True))


def equal_tables(left, right, sort_keys):
    """Exact full-table semantics; only pandas/parquet storage dtypes may vary."""
    if set(left.columns) != set(right.columns) or len(left) != len(right):
        return False
    left = left.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
    right = right[left.columns].sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_categorical=False, check_exact=True)
    except AssertionError:
        return False
    return True


def audit(protocol_path, out, historical_dir=None, historical_bindings=None, run_id="A0_20260912"):
    protocol_path, out = Path(protocol_path), Path(out)
    if out.exists():
        raise FileExistsError("Source audit output already exists; preserve its provenance")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    authority = protocol["authority"]
    checked_json(protocol["input_audit_binding"]["input_audit"])
    if digest(authority["path"]) != authority["sha256"]:
        raise ValueError("Authoritative document hash changed")
    snapshots = checked_json(protocol["input_audit_binding"]["snapshot_audit"])
    frozen = checked_json(protocol["input_audit_binding"]["input_audit"])
    model_snapshot = next(snapshot for snapshot in snapshots["snapshots"] if snapshot["kind"] == "model")
    inputs = []
    measurements, native_edges = native_from_records(snapshot_records(model_snapshot, inputs))
    known = {str(Path(record["path"]).resolve()): record["sha256"] for record in frozen["files"]}
    native_path = next(Path(path) for path in known if path.replace("\\", "/").endswith("/rf/canon/supervision.parquet"))
    if digest(native_path) != known[str(native_path.resolve())]:
        raise ValueError("Frozen native supervision changed")
    actual_native = pd.read_parquet(native_path)
    keys = ["node", "model", "weight"]
    if not equal_edges(native_edges, actual_native, keys):
        raise ValueError("Recounted native supervision differs from frozen canonical table")
    inputs.append({"path": str(native_path.resolve()), "sha256": known[str(native_path.resolve())], "role": "native_output_identity_check"})
    graph = next(Path(path).parent for path in known if path.replace("\\", "/").endswith("/graphs/hgraph_rf/nodes.npz"))
    measurements.update(graph_source_counts(graph, known, inputs))
    missing = dict(EVENT_GAPS)
    historical_dir = Path(historical_dir) if historical_dir else Path(__file__).resolve().parents[1] / "stage1BuildTransferGraph"
    required = [str((historical_dir / name).resolve()) for name in HISTORICAL_NAMES]
    absent = [path for path in required if not Path(path).is_file()]
    if absent or historical_bindings is None:
        reason = "Missing original historical sources: " + ", ".join(absent) if absent else "Historical source identity bindings not supplied; pass --historical-bindings A0_HISTORICAL_SOURCE_IDENTITY.json"
        missing.update({name: reason for name in MERGE_NAMES})
    else:
        bindings = json.loads(Path(historical_bindings).read_text(encoding="utf-8"))
        if "files" in bindings:
            if bindings.get("status") != "PASS":
                raise ValueError("Historical source identity review did not pass")
            bindings = {Path(rec["path"]).name: rec["sha256"] for rec in bindings["files"]}
        inputs.append({"path": str(Path(historical_bindings).resolve()), "sha256": digest(historical_bindings),
                       "role": "historical_identity_binding_only_not_result_source"})
        for path in required:
            expected = bindings[Path(path).name]
            if digest(path) != expected:
                raise ValueError(f"Historical source SHA256 mismatch: {path}")
            inputs.append({"path": path, "sha256": expected, "role": "a02_identity_frozen_historical_source"})
        from scale1m import merge_supervision as M
        for filename in ("supervision_uncapped.parquet", "supervision_merged.parquet", "dataset_nodes_merged.parquet", "supervision_conflicts.parquet"):
            check_path = native_path.with_name(filename)
            if digest(check_path) != known.get(str(check_path.resolve())):
                raise ValueError(f"Frozen merge reference changed: {check_path}")
            inputs.append({"path": str(check_path.resolve()), "sha256": known[str(check_path.resolve())],
                           "role": "frozen_merge_input_or_output_identity_check"})
        merged, rebuilt_nodes, rebuilt_conflicts, report = M.merge(str(native_path.parent.parent), source_dir=str(historical_dir), cap=200)
        expected_merged_path = native_path.with_name("supervision_merged.parquet")
        expected_merged = pd.read_parquet(expected_merged_path)
        if not equal_tables(merged, expected_merged, ["node", "model"]):
            raise ValueError("Recounted merge differs from frozen merged supervision")
        for frame, filename, sorting in [(rebuilt_nodes, "dataset_nodes_merged.parquet", ["node"]),
                                          (rebuilt_conflicts, "supervision_conflicts.parquet", list(rebuilt_conflicts.columns))]:
            if not equal_tables(frame, pd.read_parquet(native_path.with_name(filename)), sorting):
                raise ValueError(f"Recounted merge differs from frozen {filename}")
        for name, key in zip(MERGE_NAMES, ["rows_in", "intra_source_duplicates_collapsed", "cross_source_conflicts", "edges_before_cap"]):
            measurements[name] = report[key]
    report = {"schema_version": "a0.source_counts.v1", "a0": {"protocol": "a0", "run_id": run_id,
                "scope": "frozen_input_audit"}, "protocol_sha256": digest(protocol_path), "authority": authority,
              "inputs": inputs, "implementation_sha256": {__file__: digest(__file__), C.__file__: digest(C.__file__)},
              "measurements": measurements, "missing": missing, "required_historical_paths": required,
              "status": "incomplete" if missing else "complete", "new_retrieval_metrics_computed": False,
              "scope": "New read-only recount of frozen preprocessing sources; no training, retrieval, canonical writes or historical report totals imported."}
    if not absent and historical_bindings is not None:
        report["implementation_sha256"][M.__file__] = digest(M.__file__)
        report["merge_output_identity"] = {"passed": True, "tables": ["supervision_merged.parquet",
            "dataset_nodes_merged.parquet", "supervision_conflicts.parquet"], "comparison": "All columns and rows exact after stable semantic sort; no numeric tolerance"}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run-id", default="A0_20260912")
    parser.add_argument("--historical-dir", type=Path)
    parser.add_argument("--historical-bindings", type=Path, help="A0_HISTORICAL_SOURCE_IDENTITY.json (or explicit filename:sha256 map); all reconstructed tables must also match")
    args = parser.parse_args(argv)
    try:
        result = audit(args.protocol, args.out, args.historical_dir, args.historical_bindings, args.run_id)
    except (OSError, ValueError, KeyError) as exc:
        print(f"A0 source recount failed: {exc}")
        return 1
    print(f"A0 source recount {result['status']}: {args.out}")
    return 2 if result["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
