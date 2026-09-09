import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_PARENT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_PARENT not in sys.path:
    sys.path.insert(0, _REPO_PARENT)

from ModelLakeFishing.scale1m.eval_y4 import (
    _deployment_decision,
    _ef_candidates,
    _paired,
    _validate_nested,
)


def test_deployment_decision_keeps_k1000_separate_from_preregistered_rule():
    decision = _deployment_decision(True, [2000, 5000])
    assert decision["preregistered_rule_K"] == 2000
    assert decision["actual_system_K"] == 1000
    assert decision["retain_Y2_K1000"] is True
    assert decision["larger_K_role"] == "sensitivity_only"


def test_ef_candidates_are_fixed_multiples():
    assert _ef_candidates(1000) == (1000, 1500, 2000, 3000, 5000)
    assert _ef_candidates(5000) == (5000, 7500, 10000, 15000, 25000)


def test_validate_nested_returns_only_prefixes(monkeypatch):
    monkeypatch.setattr("ModelLakeFishing.scale1m.eval_y4.N_TOTAL", 20)
    ids = np.tile(np.arange(10), (2, 1))
    scores = np.tile(np.arange(10, 0, -1), (2, 1)).astype(np.float32)
    pools = _validate_nested(ids, scores, ks=(2, 5, 10))
    assert np.array_equal(pools[2][0], ids[:, :2])
    assert np.shares_memory(pools[5][0], ids)


def test_validate_nested_rejects_duplicates(monkeypatch):
    monkeypatch.setattr("ModelLakeFishing.scale1m.eval_y4.N_TOTAL", 20)
    ids = np.tile(np.arange(10), (1, 1))
    ids[0, -1] = ids[0, 0]
    scores = np.tile(np.arange(10, 0, -1), (1, 1)).astype(np.float32)
    try:
        _validate_nested(ids, scores, ks=(2, 5, 10))
    except AssertionError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate exact IDs were accepted")


def test_paired_rescue_harm_identity():
    base = {
        0: {"gold_rank": 11},
        1: {"gold_rank": 5},
        2: {"gold_rank": 6},
        3: {"gold_rank": 20},
    }
    new = {
        0: {"gold_rank": 4},
        1: {"gold_rank": 12},
        2: {"gold_rank": 2},
        3: {"gold_rank": 30},
    }
    got = _paired(base, new)
    assert got["rescue"] == 1
    assert got["harm"] == 1
    assert got["net"] == 0
    assert got["both_hit"] == 1
    assert got["both_miss"] == 1
