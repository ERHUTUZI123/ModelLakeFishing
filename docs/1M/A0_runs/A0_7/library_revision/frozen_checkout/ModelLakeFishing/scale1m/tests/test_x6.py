import os
import sys

import numpy as np
import torch
from torch_geometric.data import HeteroData

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_PARENT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_PARENT not in sys.path:
    sys.path.insert(0, _REPO_PARENT)

from ModelLakeFishing.scale import global_metrics as GM
from ModelLakeFishing.scale1m.baselines import fixed_tie_break, TEXT_LO, TEXT_HI
from ModelLakeFishing.stage2TrainGraphSAGE.model import HeteroGraphSAGE
from ModelLakeFishing.scale1m.eval_x6 import _score_static


def test_fixed_tie_break_is_unique_and_deterministic():
    first = fixed_tie_break(10_000)
    second = fixed_tie_break(10_000)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == len(first)
    assert not np.array_equal(first, np.arange(len(first), dtype=np.uint64))


def test_x6_tie_break_prevents_all_zero_scorer_from_ranking_everything_first():
    n = 100
    scores = np.zeros(n, dtype=np.float32)
    tie = fixed_tie_break(n)
    gold = int(np.argmax(tie))
    result = GM.query_ranks(scores, np.array([gold, 0, 1]),
                            np.array([1.0, 0.5, 0.1]), tie_break=tie)
    assert result["gold_rank"] == n
    historical = GM.query_ranks(scores, np.array([gold, 0, 1]),
                                np.array([1.0, 0.5, 0.1]))
    assert historical["gold_rank"] == 1


def test_semantic_baseline_slice_excludes_supervision_statistics():
    assert (TEXT_LO, TEXT_HI) == (64, 448)


def test_static_rank_optimization_matches_harness(monkeypatch):
    monkeypatch.setattr("ModelLakeFishing.scale1m.eval_x6.N_TOTAL", 8)
    scores = np.array([0, 1, 1, 0, 3, 2, 2, 0], dtype=np.float32)
    tie = fixed_tie_break(len(scores))
    cand = np.array([0, 2, 4, 6])
    acc = np.array([0.1, 1.0, 0.2, 0.8])
    expected = GM.query_ranks(scores, cand, acc, tie_break=tie)

    class Ctx:
        tie_break = tie

    bundles = {0: {"candidates": {5: (cand, acc)}, "roots": {5: "r"}}}
    got = _score_static("test", scores, Ctx(), bundles)["0"]
    assert got["gold@1"] == float(expected["gold_rank"] <= 1)
    assert got["gold@10"] == float(expected["gold_rank"] <= 10)
    assert got["median_gold_rank"] == expected["gold_rank"]


def test_zero_layer_model_is_a_no_message_passing_dual_tower():
    data = HeteroData()
    data["model"].x = torch.randn(5, 8)
    data["model"].size_bucket_id = torch.zeros(5, dtype=torch.long)
    data["model"].family_id = torch.zeros(5, dtype=torch.long)
    data["dataset"].x = torch.randn(3, 6)
    data[("model", "trained_on", "dataset")].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 2]])
    model = HeteroGraphSAGE(
        metadata=data.metadata(), frozen_dim=8, num_size_buckets=1,
        num_families=1, dataset_in_dim=6, hidden_channels=4, out_dim=4,
        num_layers=0)
    assert model.gnn is None
    output = model(data)
    assert output["model"].shape == (5, 4)
    assert output["dataset"].shape == (3, 4)
    assert torch.allclose(output["model"].norm(dim=1), torch.ones(5), atol=1e-6)
