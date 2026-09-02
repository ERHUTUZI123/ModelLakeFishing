"""Unit tests for the F9 utility scorecard.

The scorecard's job is to keep three things apart that are easy to conflate: a
model with a record showing it is worse, a model with no record at all, and a
model that cannot be obtained. Each separation gets a test, plus the two ranking
helpers whose off-by-one would move every number.
"""
import numpy as np
import pandas as pd
import pytest

from scale1m import utility_scorecard as U


def test_norm_task_is_case_and_separator_insensitive():
    assert U.norm_task("Automatic Speech Recognition") == "automatic-speech-recognition"
    assert U.norm_task("automatic_speech_recognition") == "automatic-speech-recognition"
    assert U.norm_task("  Retrieval ") == "retrieval"


def test_topk_is_ordered_best_first():
    s = np.array([0.1, 0.9, 0.5, 0.7, 0.2], dtype=np.float32)
    assert U.topk_from_scores(s, 3).tolist() == [1, 3, 2]


def test_rank_of_is_one_based_and_counts_only_strictly_better():
    s = np.array([0.5, 0.9, 0.5, 0.7], dtype=np.float32)
    assert U.rank_of(s, 1) == 1          # the best
    assert U.rank_of(s, 3) == 2          # one strictly better
    assert U.rank_of(s, 0) == 3          # ties do not push it down twice
    assert U.rank_of(s, 2) == 3


def _meta(n=6):
    return pd.DataFrame({
        "mappedID": np.arange(n), "model": ["m%d" % i for i in range(n)],
        "in_snapshot": [True, True, True, False, True, True],
        "size_b": [0.1, 7.0, np.nan, 1.0, 0.5, 3.0],
        "family": ["a", "a", "b", "c", "a", "d"],
        "downloads": [5, 4, 3, 2, 1, 0], "flags": [0, 0, 1, 0, 0, 0],
        "licensed": [True, False, True, True, True, False],
        "endpoints_compatible": [True, True, False, False, True, False],
        "has_library_tag": [True, True, False, True, True, False],
        "pipeline_tag": ["retrieval", "", "retrieval", "retrieval", "text-generation", ""],
    })


def _fixture():
    meta = _meta()
    cands = {0: (np.array([0, 1, 4]), np.array([0.9, 0.5, 0.2]))}   # gold = row 0
    node_of = {0: "ds\tRetrieval"}
    observed = {0: {0, 1, 4}}
    task_models = {"retrieval": np.array([0, 2, 3])}
    return meta, cands, node_of, observed, task_models


def test_unknown_and_worse_are_counted_separately():
    """A returned model with a record showing it is worse is not the same as
    one with no record; collapsing them is the misreading the axis exists to
    prevent."""
    meta, cands, node_of, observed, task_models = _fixture()
    ranked = {0: (np.array([1, 5]), 2)}          # row 1 observed-and-worse, row 5 unknown
    r = U.score_source("t", ranked, cands, node_of, meta, observed, task_models, 6)
    assert r["record_coverage@10"] == pytest.approx(0.5)
    assert r["unknown_on_query@10"] == pytest.approx(0.5)
    assert r["recorded_gold@10"] == 0.0


def test_availability_excludes_gated_and_out_of_snapshot():
    meta, cands, node_of, observed, task_models = _fixture()
    ranked = {0: (np.array([0, 2, 3]), 1)}       # 2 is gated, 3 is out of snapshot
    r = U.score_source("t", ranked, cands, node_of, meta, observed, task_models, 6)
    assert r["available@10"] == pytest.approx(1 / 3)


def test_feasible_requires_both_obtainable_and_a_library_tag():
    meta, cands, node_of, observed, task_models = _fixture()
    assert U.score_source("t", {0: (np.array([2, 5]), 1)}, cands, node_of, meta,
                          observed, task_models, 6)["at_least_one_feasible@10"] == 0.0
    assert U.score_source("t", {0: (np.array([2, 0]), 1)}, cands, node_of, meta,
                          observed, task_models, 6)["at_least_one_feasible@10"] == 1.0


def test_family_diversity_and_duplicate_rate_are_complementary_views():
    meta, cands, node_of, observed, task_models = _fixture()
    r = U.score_source("t", {0: (np.array([0, 1, 4]), 1)}, cands, node_of, meta,
                       observed, task_models, 6)          # all family "a"
    assert r["family_diversity@10"] == pytest.approx(1 / 3)
    assert r["duplicate_family_rate@10"] == pytest.approx(2 / 3)


def test_task_evidence_uses_the_normalised_query_task():
    """The node stores `Retrieval`; the pool is keyed on `retrieval`. Losing the
    normalisation silently reports zero evidence for every query."""
    meta, cands, node_of, observed, task_models = _fixture()
    r = U.score_source("t", {0: (np.array([0, 2, 5]), 1)}, cands, node_of, meta,
                       observed, task_models, 6)
    assert r["same_task_evidence@10"] == pytest.approx(2 / 3)
