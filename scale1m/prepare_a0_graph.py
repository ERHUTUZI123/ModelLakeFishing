"""Prepare the A0 graph from verified frozen bytes without rebuilding features/edges.

python -m scale1m.prepare_a0_graph --source <hgraph_rf> --out <a0/graph>
The CLI pins the source graph from A0.1. Existing outputs are never overwritten.
"""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
import pandas as pd

from scale1m.build_graph_rf import A0_ZERO_COLUMNS, dataset_stats, mask_performance_features
from scale1m.checkpoint import graph_digest, sha256_of
from scale1m.graph_store import load_sharded

SOURCE_GRAPH_DIGEST = "0e80b8393846dcc4b9e354218fd5139a906d569125d2e8545e17ccd01612b76c"
SOURCE_META_SHA256 = "4ed7b5d13395d3e5fe98faa3e3b546199e6c569687e2b0d27fbd278676398fa0"
SOURCE_SHAPE = (3016439, 18729)
REQUIRED_FILES = {"x_model.npy", "x_dataset.npy", "nodes.npz", "edges.npz",
                  "unique_model_id.parquet", "unique_dataset_id.parquet"}
REPAIR_FILE = "A0_FEATURE_REPAIR.json"
CRITICAL_METADATA_KEYS = ("format", "xm0_meta", "xd0_meta", "node_types", "edge_types", "num_nodes", "tensors")


def critical_metadata_digest(meta):
    return _digest({key:meta.get(key) for key in CRITICAL_METADATA_KEYS})


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def verify_files(directory):
    directory = Path(directory).resolve(strict=True)
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    files = meta["files"]
    if not REQUIRED_FILES <= set(files):
        raise ValueError("Graph manifest does not bind every required graph input")
    actual_names = {p.name for p in directory.iterdir()}
    if actual_names != set(files) | {"meta.json"}:
        raise ValueError("Graph directory membership differs from manifest")
    actual = {}
    for name, expected in files.items():
        path = directory / name
        if Path(name).name != name or not path.is_file() or path.is_symlink() or path.resolve().parent != directory:
            raise ValueError("Invalid graph file entry: " + name)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("Unbound graph file: " + name)
        actual[name] = sha256_of(path)
        if actual[name] != expected:
            raise ValueError("sha256 mismatch for " + name)
    return meta, actual


def _validate_source(source, expected_source_digest, expected_shape):
    meta, files = verify_files(source)
    if expected_source_digest == SOURCE_GRAPH_DIGEST and sha256_of(source / "meta.json") != SOURCE_META_SHA256:
        raise ValueError("Source encoder metadata differs from A0.1 frozen metadata")
    actual = _digest(files)
    if actual != expected_source_digest:
        raise ValueError("Source graph differs from A0.1 frozen graph digest")
    xm = np.load(source / "x_model.npy", mmap_mode="r", allow_pickle=False)
    xd = np.load(source / "x_dataset.npy", mmap_mode="r", allow_pickle=False)
    nm, nd = expected_shape
    if xm.shape != (nm, 448) or xd.shape != (nd, 458) or xm.dtype != np.float32 or xd.dtype != np.float32:
        raise ValueError("Frozen feature shape or dtype differs")
    if not np.isfinite(xd).all() or np.any(xd[:, 456:458] != 0):
        raise ValueError("Invalid frozen dataset features or reserved columns")
    udi = pd.read_parquet(source / "unique_dataset_id.parquet")
    if len(udi) != nd or not np.array_equal(udi.mappedID.to_numpy(), np.arange(nd)):
        raise ValueError("Dataset mappedID is not the frozen physical row order")
    nodes = pd.DataFrame({"dataset": udi.dataset.astype(str).str.split("\t", n=1).str[0]})
    stats, roots = dataset_stats(nodes, None)
    if roots != udi.root.astype(str).tolist() or not np.array_equal(stats[:, 6], xd[:, 454]):
        raise ValueError("Frozen root-count feature does not match the node table")
    return meta, files, np.array(xd, copy=True), stats


def verify_prepared(source, out, *, expected_source_digest=SOURCE_GRAPH_DIGEST, expected_shape=SOURCE_SHAPE):
    source, out = Path(source).resolve(strict=True), Path(out).resolve(strict=True)
    original_meta, original_files, old_x, clean_stats = _validate_source(source, expected_source_digest, expected_shape)
    meta, files = verify_files(out)
    repair = json.loads((out / REPAIR_FILE).read_text(encoding="utf-8"))
    if repair["source_graph_digest"] != expected_source_digest or repair["source_files"] != original_files:
        raise ValueError("A0 repair source binding differs")
    if repair.get("critical_metadata_sha256") != critical_metadata_digest(original_meta) or repair["critical_metadata_sha256"] != critical_metadata_digest(meta):
        raise ValueError("A0 critical encoder/graph metadata fingerprint differs")
    if meta.get("a0", {}).get("protocol") != "a0" or meta["a0"].get("feature_repair") != REPAIR_FILE:
        raise ValueError("Missing A0 graph repair marker")
    if set(files) != set(original_files) | {REPAIR_FILE}:
        raise ValueError("Unexpected A0 graph member changes")
    for name, digest in original_files.items():
        if name != "x_dataset.npy" and files[name] != digest:
            raise ValueError("A0 changed a frozen input: " + name)
    for key in CRITICAL_METADATA_KEYS:
        if meta.get(key) != original_meta.get(key):
            raise ValueError("A0 changed encoder/graph metadata: " + key)
    new_x = np.load(out / "x_dataset.npy", allow_pickle=False)
    if new_x.dtype != old_x.dtype or new_x.shape != old_x.shape:
        raise ValueError("Dataset shape/dtype changed")
    if not np.array_equal(new_x, mask_performance_features(old_x)):
        raise ValueError("A0 feature values do not match the seven-column repair")
    if not np.array_equal(new_x[:, 448:458], clean_stats):
        raise ValueError("Prepared and from-source builder statistics disagree")
    changed = np.flatnonzero(np.any(new_x != old_x, axis=0)).tolist()
    if not set(changed) <= set(A0_ZERO_COLUMNS) or np.any(new_x[:, A0_ZERO_COLUMNS] != 0):
        raise ValueError("A0 feature boundary violated")
    if repair["new_x_dataset_sha256"] != files["x_dataset.npy"]:
        raise ValueError("Feature repair report hash differs")
    loaded = load_sharded(str(out), mmap=True, verify_sha256=True)
    if not np.array_equal(loaded["data"]["dataset"].x.numpy(), new_x):
        raise ValueError("Loader dataset view differs from persisted features")
    digest = graph_digest(str(out))
    if digest != _digest(files) or digest == expected_source_digest:
        raise ValueError("A0 graph digest is stale or unchanged")
    return {"status": "PASS", "source_graph_digest": expected_source_digest, "new_graph_digest": digest,
            "source": str(source), "graph": str(out), "source_files": original_files, "new_files": files,
            "actual_changed_columns": changed, "zero_columns": list(A0_ZERO_COLUMNS),
            "dataset_feature_shape": list(new_x.shape), "all_nodes_zeroed": True,
            "other_graph_inputs_byte_identical": True, "builder_statistics_exact_match": True,
            "load_sharded_verify_sha256": True,
            "source_meta_sha256": sha256_of(source / "meta.json"), "new_meta_sha256": sha256_of(out / "meta.json"),
            "verification_scope": "Frozen input identity and seven-column repair; no training/export/retrieval"}


def prepare_graph(source, out, *, expected_source_digest=SOURCE_GRAPH_DIGEST, expected_shape=SOURCE_SHAPE):
    source, out = Path(source).resolve(strict=True), Path(out).resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("Source and output graph directories must be separate")
    if out.exists():
        raise FileExistsError("Graph output already exists; use --verify-only to inspect its binding")
    verification_path = out.with_name(out.name + ".verification.json")
    if verification_path.exists():
        raise FileExistsError("Existing graph verification record; refusing overwrite")
    started = time.perf_counter()
    original_meta_sha = sha256_of(source / "meta.json")
    meta, source_files, old_x, stats = _validate_source(source, expected_source_digest, expected_shape)
    clean = mask_performance_features(old_x)
    if not np.array_equal(clean[:, 448:458], stats):
        raise ValueError("Mask and rebuilt statistics differ")
    out.parent.mkdir(parents=True, exist_ok=True)
    staged = out.with_name(out.name + ".a0-staging-" + uuid.uuid4().hex)
    if staged.resolve().parent != out.parent or out.parent == source:
        raise ValueError("Unsafe staging destination")
    staged.mkdir(exist_ok=False)
    print("[A0.2] source hashes verified; copying frozen graph files", flush=True)
    for name in source_files:
        if name != "x_dataset.npy":
            shutil.copy2(source / name, staged / name)
    np.save(staged / "x_dataset.npy", clean, allow_pickle=False)
    repair = {"protocol": "a0", "source_graph_digest": expected_source_digest,
              "source_meta_sha256": original_meta_sha, "source_files": source_files,
              "critical_metadata_sha256": critical_metadata_digest(meta),
              "critical_metadata_keys": list(CRITICAL_METADATA_KEYS),
              "source_x_dataset_sha256": source_files["x_dataset.npy"],
              "new_x_dataset_sha256": sha256_of(staged / "x_dataset.npy"),
              "zero_columns": list(A0_ZERO_COLUMNS), "all_dataset_nodes": True,
              "dimensions": {"model": [expected_shape[0], 448], "dataset": list(clean.shape)},
              "changed_columns": np.flatnonzero(np.any(clean != old_x, axis=0)).tolist(),
              "per_column_changed_rows": {str(i): int(np.count_nonzero(clean[:,i] != old_x[:,i])) for i in A0_ZERO_COLUMNS},
              "root_count_preserved": bool(np.array_equal(old_x[:,454], clean[:,454])),
              "builder_statistics_equal": True,
              "implementation_sha256": {"prepare_a0_graph.py": sha256_of(Path(__file__)),
                                         "build_graph_rf.py": sha256_of(Path(__file__).with_name("build_graph_rf.py"))}}
    _json(staged / REPAIR_FILE, repair)
    new_meta = copy.deepcopy(meta)
    new_meta["written_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    new_meta["a0"] = {"protocol": "a0", "feature_repair": REPAIR_FILE,
                      "source_graph_digest": expected_source_digest}
    provenance = dict(meta.get("provenance") or {})
    provenance["a0_source_graph"] = str(source)
    new_meta["provenance"] = provenance
    new_meta["files"] = {p.name: sha256_of(p) for p in sorted(staged.iterdir())}
    _json(staged / "meta.json", new_meta)
    result = verify_prepared(source, staged, expected_source_digest=expected_source_digest, expected_shape=expected_shape)
    if sha256_of(source / "meta.json") != original_meta_sha:
        raise ValueError("Source metadata changed during preparation")
    if out.exists() or staged.resolve().parent != out.resolve().parent:
        raise FileExistsError("Output became occupied or staging parent changed")
    os.rename(staged, out)
    result["graph"] = str(out)
    result["elapsed_seconds"] = time.perf_counter() - started
    result["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _json(verification_path, result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--verify-only", action="store_true", help="Read-only verification of an existing A0 graph")
    args = p.parse_args(argv)
    result = verify_prepared(args.source,args.out) if args.verify_only else prepare_graph(args.source,args.out)
    print(json.dumps({k:v for k,v in result.items() if k not in {"source_files","new_files"}},ensure_ascii=False,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
