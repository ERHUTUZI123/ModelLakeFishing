import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_PARENT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_PARENT not in sys.path:
    sys.path.insert(0, _REPO_PARENT)

from ModelLakeFishing.scale1m.eval_z1 import (
    _calibrate,
    _configs,
    _evaluate,
    _paired,
)


def test_percentile_is_tie_aware():
    values = np.array([[0.0, 0.0, 1.0, 2.0]])
    got = _calibrate(values, "percentile")
    assert got[0, 0] == got[0, 1]
    assert got[0, 3] == 1.0


def test_zscore_constant_row_is_zero():
    got = _calibrate(np.ones((2, 4)), "zscore")
    assert np.array_equal(got, np.zeros((2, 4)))


def test_grid_contains_current_once():
    rows = [c for c in _configs() if c == {
        "calibration": "raw", "beta": 1.0, "shrink_k": 5.0}]
    assert len(rows) == 1


def test_evaluate_obeys_label_free_tie_break(monkeypatch):
    monkeypatch.setattr("ModelLakeFishing.scale1m.eval_z1.K", 4)
    pool = np.array([[3, 2, 1, 0]])
    scores = np.zeros((1, 4))
    gold = np.array([1])
    top3 = np.array([[1, 2, 3]])
    tie = np.array([3, 2, 1, 0])
    row, hit, _top10 = _evaluate(scores, pool, gold, top3, tie)
    assert row["gold@1"] == 0.0
    assert bool(hit[0])


def test_paired_identity():
    got = _paired(np.array([False, True, True]), np.array([True, False, True]))
    assert got["rescue"] == 1
    assert got["harm"] == 1
    assert got["net"] == 0
