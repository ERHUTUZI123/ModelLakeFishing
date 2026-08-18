"""Unit tests for the sharded rung-graph store (RF, F0).

What is worth pinning here is not that files get written -- it is that the
round trip is EXACT. The sharded form replaces the single .pt as the input to
training at full-lake scale, so any silent dtype narrowing or column reordering
would move `x` rows out from under their mappedID without raising anything.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData

from scale1m import graph_store as G


def _tiny_ckpt(n_model=7, n_dataset=3):
    data = HeteroData()
    data["model"].node_id = torch.arange(n_model)
    data["model"].x = torch.randn(n_model, 6)
    data["model"].size_bucket_id = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    data["model"].family_id = torch.tensor([0, 0, 1, 1, 2, 2, 0])
    data["dataset"].node_id = torch.arange(n_dataset)
    data["dataset"].x = torch.randn(n_dataset, 4)
    data["model", "trained_on", "dataset"].edge_index = torch.tensor([[0, 1], [0, 2]])
    data["model", "trained_on", "dataset"].edge_attr = torch.tensor([0.5, 0.25])
    data["model", "is_base_of", "model"].edge_index = torch.tensor([[0], [3]])
    data["model", "is_base_of", "model"].edge_attr = torch.tensor([1.0])
    data["model", "is_base_of", "model"].relation_id = torch.tensor([2])
    return {
        "data": data,
        "xm0_meta": {"num_size_buckets": 3, "num_families": 3,
                     "family_vocab": {"Other": 0, "bert": 1, "llama": 2},
                     "name_dim": 2, "desc_dim": 4},
        "xd0_meta": {"num_task_types": 2, "task_type_vocab": {"a": 0, "b": 1}},
        "unique_model_id": pd.DataFrame({"model": [f"o/m{i}" for i in range(n_model)],
                                         "mappedID": list(range(n_model))}),
        "unique_dataset_id": pd.DataFrame({"dataset": ["d0", "d1", "d2"],
                                           "mappedID": [0, 1, 2]}),
        "provenance": {"rung": "tiny"},
    }


def test_round_trip_is_tensor_for_tensor_identical(tmp_path):
    ckpt = _tiny_ckpt()
    G.save_sharded(ckpt, str(tmp_path))
    back = G.load_sharded(str(tmp_path), mmap=True, verify_sha256=True)
    assert G.compare(ckpt, back) == []
    # dtypes survive: an int64 column silently becoming int32 would still
    # index correctly today and overflow at 2^31 rows later
    assert back["data"]["model"].family_id.dtype == torch.int64
    assert back["data"]["model"].x.dtype == torch.float32


def test_features_are_stored_at_full_precision(tmp_path):
    """`x` is the frozen half of the model input; storage must not downcast."""
    ckpt = _tiny_ckpt()
    G.save_sharded(ckpt, str(tmp_path))
    arr = np.load(str(tmp_path / "x_model.npy"), mmap_mode="r")
    assert arr.dtype == np.float32
    assert torch.equal(torch.from_numpy(np.asarray(arr)), ckpt["data"]["model"].x)


def test_mmap_is_the_default_and_still_reads_correctly(tmp_path):
    """The memory saving itself is measured, not asserted here: on the 100K
    graph the load-time resident cost is 50 MB mapped vs 243 MB via .pt
    (docs/1M/F0.md). This test only guards the switch and the values."""
    ckpt = _tiny_ckpt()
    G.save_sharded(ckpt, str(tmp_path))
    back = G.load_sharded(str(tmp_path))
    assert back["_mmap"] is True
    assert torch.equal(back["data"]["model"].x, ckpt["data"]["model"].x)
    eager = G.load_sharded(str(tmp_path), mmap=False)
    assert eager["_mmap"] is False
    assert torch.equal(eager["data"]["model"].x, ckpt["data"]["model"].x)


def test_a_truncated_shard_is_caught_by_sha256(tmp_path):
    ckpt = _tiny_ckpt()
    G.save_sharded(ckpt, str(tmp_path))
    path = str(tmp_path / "edges.npz")
    with open(path, "r+b") as fh:
        fh.truncate(os.path.getsize(path) - 8)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        G.load_sharded(str(tmp_path), verify_sha256=True)


def test_compare_reports_a_real_difference(tmp_path):
    """A comparison that cannot fail is not a check."""
    ckpt = _tiny_ckpt()
    G.save_sharded(ckpt, str(tmp_path))
    back = G.load_sharded(str(tmp_path))
    back["data"]["model"].family_id = back["data"]["model"].family_id.clone()
    back["data"]["model"].family_id[0] = 2
    diffs = G.compare(ckpt, back)
    assert any("family_id" in d for d in diffs)


def test_meta_json_records_every_file(tmp_path):
    ckpt = _tiny_ckpt()
    meta = G.save_sharded(ckpt, str(tmp_path))
    on_disk = {f for f in os.listdir(str(tmp_path)) if f != "meta.json"}
    assert on_disk == set(meta["files"])
    with open(str(tmp_path / "meta.json"), encoding="utf-8") as fh:
        assert json.load(fh)["xm0_meta"]["family_vocab"]["Other"] == 0
