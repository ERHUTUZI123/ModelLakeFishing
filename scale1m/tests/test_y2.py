import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_PARENT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_PARENT not in sys.path:
    sys.path.insert(0, _REPO_PARENT)

from ModelLakeFishing.scale1m.eval_y2 import _pool_metrics, _tie_ranks


class Prior:
    def __init__(self, values):
        self._values = values

    def values(self, query, model_ids):
        return self._values[int(query)][np.asarray(model_ids)]


def test_pool_rerank_can_promote_task_prior_candidate(monkeypatch):
    monkeypatch.setattr("ModelLakeFishing.scale1m.eval_y2.N_TOTAL", 5)
    ids = np.array([[0, 1, 2, 3]], dtype=np.int64)
    dense = np.array([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32)
    prior = Prior({0: np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)})
    candidates = {0: (np.array([3, 0]), np.array([1.0, 0.5]))}
    roots = np.array(["r"])
    row, _per, top10 = _pool_metrics(
        ids, dense, [0], candidates, roots, prior, _tie_ranks(5), k=4)
    assert top10[0, 0] == 3
    assert row["gold@1"] == 1.0
    assert row["gold@10"] == 1.0


def test_missing_gold_is_not_a_pool_hit(monkeypatch):
    monkeypatch.setattr("ModelLakeFishing.scale1m.eval_y2.N_TOTAL", 5)
    ids = np.array([[0, 1, 2, 3]], dtype=np.int64)
    dense = np.array([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32)
    prior = Prior({0: np.zeros(5, dtype=np.float32)})
    candidates = {0: (np.array([4, 0]), np.array([1.0, 0.5]))}
    roots = np.array(["r"])
    row, _per, _top10 = _pool_metrics(
        ids, dense, [0], candidates, roots, prior, _tie_ranks(5), k=4)
    assert row["gold_in_first_stage@1000"] == 0.0
    assert row["gold@1"] == 0.0
