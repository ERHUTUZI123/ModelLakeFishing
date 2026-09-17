"""Tiny fixtures only: never stream the real frozen snapshot in A0.2."""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scale1m import a0_source_counts as S


def record(model, values):
    return {"id": model, "cardData": {"model-index": [{"dataset": "Org/Data", "task": "classification",
        "metrics": [{"type": "accuracy", "value": value} for value in values]}]}}


def test_native_original_parsing_finiteness_median_and_primary_rules():
    measurements, edges = S.native_from_records([record("Org/M1", [0.2, 0.4, "NaN", True]),
        record("Org/M2", ["80%"]), record("Org/M3", [[0.1]])])
    assert measurements == {"native_raw_metric_rows": 6, "native_finite_metric_rows": 4,
        "native_median_deduplicated_rows": 3, "native_primary_edges_before_cap": 3,
        "native_primary_edges_after_cap": 3}
    assert edges.model.tolist() == ["org/m1", "org/m2", "org/m3"]
    assert edges.weight.max() == 1 and edges.weight.min() == 0


def test_all_invalid_values_preserve_raw_count():
    counts, edges = S.native_from_records([record("m", [True, "NaN", "bad", [1, 2]])])
    assert counts["native_raw_metric_rows"] == 4
    assert counts["native_finite_metric_rows"] == 0
    assert edges.empty and list(edges.columns) == ["node", "model", "weight"]


def test_training_similarity_matches_actual_graph_surgery_with_duplicates():
    import torch
    from torch_geometric.data import HeteroData
    from stage2TrainGraphSAGE.graph_surgery import topk_similar_to
    data = HeteroData()
    data["dataset"].num_nodes = 7
    endpoints = np.array([[0, 0, 0, 1, 2, 3, 4, 4, 6], [1, 1, 2, 3, 3, 4, 0, 5, 6]])
    data["dataset", "similar_to", "dataset"].edge_index = torch.tensor(endpoints)
    data["dataset", "similar_to", "dataset"].edge_attr = torch.ones(endpoints.shape[1])
    actual = topk_similar_to(data, 2)["dataset", "similar_to", "dataset"].edge_index.shape[1]
    assert S.count_training_similarity(endpoints, 7, 2) == actual
    with pytest.raises(ValueError, match="out of range"):
        S.count_training_similarity(np.array([[0], [7]]), 7)


def test_hash_bound_shard_stream_rejects_tampering(tmp_path):
    path = tmp_path / "shard-1.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(record("m", [1])) + "\n")
    snapshot = {"directory": str(tmp_path), "shards": [{"file": path.name, "sha256_after_stream": S.digest(path)}]}
    inputs = []
    assert len(list(S.snapshot_records(snapshot, inputs))) == 1
    assert inputs[0]["role"] == "original_frozen_model_snapshot"
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed"):
        list(S.snapshot_records(snapshot, []))


def test_edge_identity_gate_ignores_storage_dtype_but_not_values():
    left = pd.DataFrame({"node": pd.Series(["d", "d"], dtype="category"), "model": ["b", "a"], "weight": [1., 0.]})
    right = pd.DataFrame({"node": ["d", "d"], "model": ["a", "b"], "weight": [0., 1.]})
    assert S.equal_edges(left, right, ["node", "model", "weight"])
    right.loc[0, "weight"] = 0.1
    assert not S.equal_edges(left, right, ["node", "model", "weight"])


def test_graph_count_records_include_denominators_and_category_range(tmp_path):
    np.savez(tmp_path / "nodes.npz", **{"model.node_id": np.arange(3), "dataset.node_id": np.arange(3),
        "model.size_bucket_id": [0, 0, 2], "model.family_id": [0, 4, 4], "dataset.task_type_id": [0, 0, 0]})
    np.savez(tmp_path / "edges.npz", **{"dataset__similar_to__dataset__edge_index": [[0, 1], [1, 2]],
        "dataset__similar_to__dataset__edge_attr": [0.1, 0.2]})
    np.save(tmp_path / "x_model.npy", np.zeros((3, 448), dtype=np.float32))
    np.save(tmp_path / "x_dataset.npy", np.zeros((3, 458), dtype=np.float32))
    expected = {str(path.resolve()): S.digest(path) for path in tmp_path.iterdir()}
    result = S.graph_source_counts(tmp_path, expected, [])
    assert result["unknown_model_size_percent"]["numerator"] == 2
    assert result["unknown_model_size_percent"]["denominator"] == 3
    assert result["categorical_vocab_cardinalities"]["model.family_id"] == {"observed_unique": 2, "embedding_index_cardinality": 5}
    assert result["training_similar_to_edges"] == 4


def test_event_requirements_cannot_be_replaced_by_retained_shards_or_reports():
    assert set(S.EVENT_GAPS) == {"model_snapshot_api_pages", "model_snapshot_skipped_duplicates"}
    assert "event log" in S.EVENT_GAPS["model_snapshot_api_pages"]
    assert "discard" in S.EVENT_GAPS["model_snapshot_skipped_duplicates"]
    assert len(S.HISTORICAL_NAMES) == 5


def test_full_table_gate_rejects_non_edge_metadata_and_conflict_changes():
    actual = pd.DataFrame({"node": ["b", "a"], "gold_eligible": [True, False], "n_records": [2, 1]})
    reference = actual.iloc[::-1].reset_index(drop=True)
    assert S.equal_tables(actual, reference, ["node"])
    reference.loc[0, "gold_eligible"] = True
    assert not S.equal_tables(actual, reference, ["node"])
    assert not S.equal_tables(actual, actual.drop(columns="n_records"), ["node"])
