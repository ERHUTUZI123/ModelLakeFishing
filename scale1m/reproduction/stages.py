"""Input identity checks between portable from-source A0 stages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from .snapshot import sha256


def seed_vocab(source: Path, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and sha256(out) != sha256(source):
        raise ValueError("Existing vocabulary differs from the frozen A0 vocabulary")
    if not out.exists():
        shutil.copyfile(source, out)
    return {"family_vocab_sha256": sha256(out)}


def check_canonical(rf: Path, frozen: Path):
    import pandas as pd
    checked = {}
    for name, keys, expected_rows in [
        ("supervision_merged.parquet", ["node", "model"], 247803),
        ("dataset_nodes_merged.parquet", ["node"], 18729),
    ]:
        got = pd.read_parquet(rf / "canon" / name)
        expected = pd.read_parquet(frozen / name)
        if len(got) != expected_rows:
            raise ValueError(f"{name}: expected {expected_rows} rows, got {len(got)}")
        columns = sorted(expected.columns)
        if sorted(got.columns) != columns:
            raise ValueError("Canonical schema differs from frozen A0: " + name)
        # Parquet container bytes are not stable across writer versions. Check
        # every scientific value, preserving exact numeric equality instead.
        left = got.sort_values(keys).reset_index(drop=True)[columns]
        right = expected.sort_values(keys).reset_index(drop=True)[columns]
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)
        checked[name] = expected_rows
    return checked


def check_graph(graph: Path, vocab: Path, frozen_vocab: Path):
    import numpy as np
    import pandas as pd
    from scale1m.build_graph_rf import A0_ZERO_COLUMNS, dataset_stats
    from scale1m.prepare_a0_graph import verify_files

    meta, _ = verify_files(graph)
    if meta["num_nodes"] != {"model": 3016439, "dataset": 18729}:
        raise ValueError("Rebuilt graph does not have the frozen A0 node universe")
    if sha256(vocab) != sha256(frozen_vocab):
        raise ValueError("Family vocabulary IDs changed from frozen A0")
    x = np.load(graph / "x_dataset.npy", mmap_mode="r", allow_pickle=False)
    if x.shape != (18729, 458) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("Invalid dataset features")
    if np.any(x[:, A0_ZERO_COLUMNS] != 0):
        raise ValueError("Performance-derived features leaked into rebuilt graph")
    frame = pd.read_parquet(graph / "unique_dataset_id.parquet")
    stats, roots = dataset_stats(pd.DataFrame({"dataset": frame.dataset.astype(str).str.split("\t", n=1).str[0]}), None)
    if not np.array_equal(x[:, 448:], stats) or roots != frame.root.astype(str).tolist():
        raise ValueError("Dataset structural counts/root identity differ")
    report = json.loads((graph / "GRAPH_REPORT.json").read_text(encoding="utf-8"))
    if report["edges"]["trained_on"] != 247803 or not all(report["gates"].values()):
        raise ValueError("Graph construction gates or supervision count failed")
    return {"models": 3016439, "dataset_task_nodes": 18729,
            "performance_edges": 247803, "seven_performance_columns_zero": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("seed-vocab")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("canonical")
    p.add_argument("--rf", type=Path, required=True)
    p.add_argument("--frozen", type=Path, required=True)
    p = sub.add_parser("graph")
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--vocab", type=Path, required=True)
    p.add_argument("--frozen-vocab", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "seed-vocab":
        result = seed_vocab(args.source, args.out)
    elif args.command == "canonical":
        result = check_canonical(args.rf, args.frozen)
    else:
        result = check_graph(args.graph, args.vocab, args.frozen_vocab)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
