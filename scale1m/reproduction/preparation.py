"""Reconstruct the frozen A0 graph from its original inputs and repair binding.

This re-executes the seven-column repair, then restores the byte-identical
archived metadata.  Those metadata describe the historical reference graph;
they are not represented as a log of this execution.  A separate adjacent
``*.reconstruction.json`` records this invocation and its actual implementation.
The historical A0 graph validators and all of their fixed hashes remain intact.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import time
import uuid


PREPARED_META_SHA256 = "fd0d129c212c4d343b9ff8e7aa1887a51e1ecce558a58a3d505f2166a1239231"
FROZEN_REPAIR_SHA256 = "e1e7a7fb3dccbfe81fab3619600adcbf5ea59da096d13a014d5a8233cbce6e84"


def reconstruct_graph(source, binding_dir, out):
    import numpy as np

    from scale1m import prepare_a0_graph as preparation
    from scale1m.a0_graph_validation import PREPARED_GRAPH_DIGEST, verify_a0_graph
    from scale1m.build_graph_rf import mask_performance_features
    from scale1m.checkpoint import sha256_of

    started = time.perf_counter()
    source = Path(source).resolve(strict=True)
    binding_dir = Path(binding_dir).resolve(strict=True)
    out = Path(out).resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("Source and output graph directories must be separate")
    record_path = out.with_name(out.name + ".reconstruction.json")
    if out.exists() or record_path.exists():
        raise FileExistsError("Graph or reconstruction record already exists; refusing overwrite")

    frozen_meta = binding_dir / "meta.json"
    frozen_repair = binding_dir / preparation.REPAIR_FILE
    for path, expected in ((frozen_meta, PREPARED_META_SHA256),
                           (frozen_repair, FROZEN_REPAIR_SHA256)):
        if not path.is_file() or sha256_of(path) != expected:
            raise ValueError("Frozen A0 graph binding SHA256 mismatch: " + str(path))
    output_meta = json.loads(frozen_meta.read_text(encoding="utf-8"))
    repair = json.loads(frozen_repair.read_text(encoding="utf-8"))
    if preparation._digest(output_meta["files"]) != PREPARED_GRAPH_DIGEST:
        raise ValueError("Frozen graph binding does not identify the published A0 graph")

    source_meta, source_files, original, clean_stats = preparation._validate_source(
        source, preparation.SOURCE_GRAPH_DIGEST, preparation.SOURCE_SHAPE)
    if repair["source_files"] != source_files:
        raise ValueError("Frozen repair does not bind this exact source graph")
    if set(output_meta["files"]) != set(source_files) | {preparation.REPAIR_FILE}:
        raise ValueError("Frozen repair graph membership differs from its source")
    for key in preparation.CRITICAL_METADATA_KEYS:
        if output_meta.get(key) != source_meta.get(key):
            raise ValueError("Frozen repair changed encoder/graph metadata: " + key)
    clean = mask_performance_features(original)
    if not np.array_equal(clean[:, 448:458], clean_stats):
        raise ValueError("Reconstructed clean statistics differ from the frozen node table")

    out.parent.mkdir(parents=True, exist_ok=True)
    staged = out.with_name(out.name + ".reconstructing-" + uuid.uuid4().hex)
    if staged.resolve().parent != out.parent:
        raise ValueError("Staging path is outside the requested output parent")
    staged.mkdir(exist_ok=False)
    for name in source_files:
        if name != "x_dataset.npy":
            shutil.copy2(source / name, staged / name)
    np.save(staged / "x_dataset.npy", clean, allow_pickle=False)
    shutil.copy2(frozen_repair, staged / preparation.REPAIR_FILE)
    shutil.copy2(frozen_meta, staged / "meta.json")

    checked = preparation.verify_prepared(
        source, staged, expected_source_digest=preparation.SOURCE_GRAPH_DIGEST,
        expected_shape=preparation.SOURCE_SHAPE)
    verified = verify_a0_graph(staged)
    if verified["graph_sha256"] != PREPARED_GRAPH_DIGEST:
        raise ValueError("Reconstructed graph differs from the frozen A0 graph")
    if out.exists() or staged.resolve().parent != out.parent:
        raise FileExistsError("Output became occupied during reconstruction")
    os.rename(staged, out)

    record = {
        "schema": "a0.graph_reconstruction.v1",
        "status": "PASS",
        "source": str(source),
        "graph": str(out),
        "binding_dir": str(binding_dir),
        "source_graph_digest": preparation.SOURCE_GRAPH_DIGEST,
        "graph_digest": verified["graph_sha256"],
        "frozen_metadata_sha256": PREPARED_META_SHA256,
        "frozen_repair_sha256": FROZEN_REPAIR_SHA256,
        "zero_columns": checked["zero_columns"],
        "other_graph_inputs_byte_identical": checked["other_graph_inputs_byte_identical"],
        "implementation_sha256": sha256_of(Path(__file__)),
        "provenance_scope": (
            "meta.json and A0_FEATURE_REPAIR.json retain historical reference provenance; "
            "this record describes the fresh deterministic reconstruction, not a new crawl"
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    with record_path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Frozen pre-A0 graph directory")
    parser.add_argument("--binding-dir", required=True, help="Frozen prepared graph metadata directory")
    parser.add_argument("--out", required=True, help="New graph directory")
    args = parser.parse_args(argv)
    print(json.dumps(reconstruct_graph(args.source, args.binding_dir, args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
