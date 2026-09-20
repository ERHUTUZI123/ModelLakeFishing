from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData

_REPO_PARENT = Path(__file__).resolve().parents[2]
if os.fspath(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, os.fspath(_REPO_PARENT))

from ModelLakeFishing.stage2TrainGraphSAGE.d0_splits import (
    make_root_aware_splits,
)
from ModelLakeFishing.stage2TrainGraphSAGE.losses import (
    TRAINED_ON,
    REV_TRAINED_ON,
)


def _skeleton(graph_dir: Path) -> tuple[HeteroData, list[str]]:
    with (graph_dir / "meta.json").open(encoding="utf-8") as handle:
        meta = json.load(handle)
    with np.load(graph_dir / "edges.npz") as arrays:
        edge_index = torch.from_numpy(
            arrays["model__trained_on__dataset__edge_index"].copy())
        edge_attr = torch.from_numpy(
            arrays["model__trained_on__dataset__edge_attr"].copy())
    data = HeteroData()
    data["model"].num_nodes = int(meta["num_nodes"]["model"])
    data["dataset"].num_nodes = int(meta["num_nodes"]["dataset"])
    data[TRAINED_ON].edge_index = edge_index
    data[TRAINED_ON].edge_attr = edge_attr
    data[REV_TRAINED_ON].edge_index = edge_index.flip(0)
    data[REV_TRAINED_ON].edge_attr = edge_attr.clone()
    import pandas as pd
    ids = pd.read_parquet(graph_dir / "unique_dataset_id.parquet")
    ids = ids.sort_values("mappedID")
    if not np.array_equal(ids["mappedID"].to_numpy(),
                          np.arange(len(ids), dtype=np.int64)):
        raise AssertionError("dataset mappedID is not contiguous")
    return data, ids["root"].astype(str).tolist()


def _fingerprint(data: HeteroData, roots: list[str], seed: int) -> dict:
    _train, _val, test = make_root_aware_splits(
        data, roots, split_seed=seed)
    labels = test[TRAINED_ON].edge_label_index.detach().cpu().numpy()
    labels = np.ascontiguousarray(labels, dtype=np.int64)
    digest = hashlib.sha256(labels.tobytes(order="C")).hexdigest()
    return {
        "test_edges": int(labels.shape[1]),
        "test_sha256": digest,
        "test_query_nodes": int(np.unique(labels[1]).size),
    }


def audit(graph_dir: Path) -> dict:
    data, roots = _skeleton(graph_dir)
    result = {
        "edges": int(data[TRAINED_ON].edge_index.shape[1]),
        "roots": len(set(roots)),
        "per_seed": {},
    }
    for seed in (0, 1, 2):
        first = _fingerprint(data, roots, seed)
        second = _fingerprint(data, roots, seed)
        result["per_seed"][str(seed)] = {
            **first,
            "reproducible": first == second,
        }
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="fingerprint the RF graph's root-aware splits")
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    graph = args.graph.resolve()
    output = args.out.resolve() if args.out else graph / "f5_splits.json"
    report = audit(graph)
    if not all(row["reproducible"] for row in report["per_seed"].values()):
        raise AssertionError("a root-aware split was not deterministic")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
