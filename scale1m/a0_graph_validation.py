import json
from pathlib import Path

import numpy as np
import pandas as pd

from scale1m.build_graph_rf import A0_ZERO_COLUMNS, dataset_stats
from scale1m.prepare_a0_graph import (REPAIR_FILE, SOURCE_GRAPH_DIGEST, SOURCE_META_SHA256,
    CRITICAL_METADATA_KEYS, critical_metadata_digest, _digest, verify_files)

PREPARED_GRAPH_DIGEST = "acddeb93d926efcb636c83d452fe360736a0e6e4ac68cab3f5fef210e7cef5db"


def verify_a0_graph(directory, *, require_frozen_source=True):
    directory = Path(directory).resolve(strict=True)
    meta, files = verify_files(directory)
    if require_frozen_source and _digest(files) != PREPARED_GRAPH_DIGEST:
        raise ValueError("A0 graph differs from the prepared graph frozen in A0.2")
    marker = meta.get("a0", {})
    if marker.get("protocol") != "a0" or marker.get("feature_repair") != REPAIR_FILE:
        raise ValueError("A0 prepared graph marker/repair report missing")
    repair = json.loads((directory / REPAIR_FILE).read_text(encoding="utf-8"))
    origin = repair["source_graph_digest"]
    if (require_frozen_source and origin != SOURCE_GRAPH_DIGEST) or _digest(repair["source_files"]) != origin:
        raise ValueError("A0 source graph digest differs from frozen source")
    if marker.get("source_graph_digest") != origin or _digest(files) == origin:
        raise ValueError("A0 graph digest/metadata binding differs")
    if require_frozen_source and repair.get("source_meta_sha256") != SOURCE_META_SHA256:
        raise ValueError("A0 source encoder metadata hash differs from A0.1")
    if repair.get("critical_metadata_keys") != list(CRITICAL_METADATA_KEYS) or repair.get("critical_metadata_sha256") != critical_metadata_digest(meta):
        raise ValueError("A0 critical encoder/graph metadata fingerprint differs")
    if repair.get("zero_columns") != list(A0_ZERO_COLUMNS) or repair.get("all_dataset_nodes") is not True:
        raise ValueError("A0 seven-column policy differs")
    if set(files) != set(repair["source_files"]) | {REPAIR_FILE}:
        raise ValueError("A0 graph file membership changed")
    for name, digest in repair["source_files"].items():
        if name != "x_dataset.npy" and files[name] != digest:
            raise ValueError("A0 changed frozen graph input " + name)
    if repair.get("new_x_dataset_sha256") != files["x_dataset.npy"]:
        raise ValueError("A0 dataset feature hash differs")
    x = np.load(directory / "x_dataset.npy", mmap_mode="r", allow_pickle=False)
    shape = (meta["num_nodes"]["dataset"], 458)
    if x.shape != shape or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("A0 dataset feature shape/dtype/finite check failed")
    if np.any(x[:, A0_ZERO_COLUMNS] != 0) or np.any(x[:, 456:458] != 0):
        raise ValueError("A0 performance-derived columns or reserved columns are nonzero")
    udi = pd.read_parquet(directory / "unique_dataset_id.parquet")
    if len(udi) != len(x) or not np.array_equal(udi.mappedID.to_numpy(),np.arange(len(x))):
        raise ValueError("A0 dataset row map order differs")
    stats, roots = dataset_stats(pd.DataFrame({"dataset":udi.dataset.astype(str).str.split("\t",n=1).str[0]}),None)
    if roots != udi.root.astype(str).tolist() or not np.array_equal(x[:,448:],stats):
        raise ValueError("A0 clean statistics do not match frozen node table")
    return {"graph_sha256":_digest(files),"files":files,"repair":repair,"status":"PASS"}
