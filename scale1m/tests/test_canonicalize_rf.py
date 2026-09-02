"""Unit tests for the F2 supervision build (RF).

Every rule here is one of the knobs constraint 4 freezes: parsing, dedupe,
in-group normalisation, direction, primary-metric choice, gold eligibility and
the per-node cap. Each of them can move gold@10 on its own, so each gets a test
that would fail if the behaviour drifted.
"""
import numpy as np
import pandas as pd
import pytest

from scale1m import canonicalize_rf as C


@pytest.mark.parametrize("raw,want", [
    (0.87, 0.87), (42, 42.0), ("0.5", 0.5), (" 93.2 ", 93.2), ("88%", 88.0),
    ([0.3], 0.3),
])
def test_values_that_parse(raw, want):
    assert C.to_float(raw) == pytest.approx(want)


@pytest.mark.parametrize("raw", [
    "11.05 +/- 5.90",      # the HF RL leaderboard format, deliberately rejected
    "n/a", "", None, True, float("nan"), float("inf"), {"v": 1}, [1, 2],
])
def test_values_that_do_not_parse(raw):
    assert C.to_float(raw) is None


def test_rl_task_detection():
    assert C.is_rl_task("reinforcement-learning")
    assert C.is_rl_task("Reinforcement Learning")
    assert not C.is_rl_task("text-generation")
    assert not C.is_rl_task("")


def test_model_index_parsing_skips_entries_without_a_dataset_or_metrics():
    rec = {"cardData": {"model-index": [
        {"dataset": "Org/DS", "task": "text-classification",
         "metrics": [{"type": "accuracy", "value": 0.9}]},
        {"task": "no-dataset", "metrics": [{"type": "f1", "value": 0.5}]},
        {"dataset": "x", "task": "no-metrics", "metrics": []},
    ]}}
    got = C.parse_model_index(rec)
    assert got == [("org/ds", "text-classification", "accuracy", 0.9)]


def _frame(rows):
    df = pd.DataFrame(rows, columns=["model", "dataset", "task", "metric",
                                     "direction", "value"])
    for c in ("dataset", "task", "metric", "direction"):
        df[c] = df[c].astype("category")
    return df


def test_duplicate_records_collapse_to_the_median_not_the_best():
    """An author who submits three times must not buy a higher edge weight."""
    rows = [("m1", "d", "t", "accuracy", "higher", v) for v in (0.10, 0.50, 0.90)]
    rows += [("m2", "d", "t", "accuracy", "higher", 0.60)]
    edges, nodes, ded, rep = C.build_supervision(_frame(rows))
    assert rep["duplicate_rows_collapsed"] == 2
    m1 = ded[(ded.model == "m1")].iloc[0]
    assert m1["value"] == pytest.approx(0.50)      # median, not 0.90


def test_lower_is_better_is_flipped_so_the_best_model_wins_gold():
    rows = [("good", "d", "asr", "wer", "lower", 0.05),
            ("mid", "d", "asr", "wer", "lower", 0.30),
            ("bad", "d", "asr", "wer", "lower", 0.90)]
    edges, nodes, ded, rep = C.build_supervision(_frame(rows))
    w = dict(zip(edges.model, edges.weight))
    assert w["good"] > w["mid"] > w["bad"]
    assert w["good"] == pytest.approx(1.0) and w["bad"] == pytest.approx(0.0)


def test_normalisation_is_per_full_metric_name_not_pooled():
    """ndcg_at_1 and ndcg_at_10 are different groups (1Mplan 3.2)."""
    rows = [("m1", "d", "t", "ndcg_at_1", "higher", 0.10),
            ("m2", "d", "t", "ndcg_at_1", "higher", 0.20),
            ("m1", "d", "t", "ndcg_at_10", "higher", 0.80),
            ("m2", "d", "t", "ndcg_at_10", "higher", 0.90)]
    _e, _n, ded, _r = C.build_supervision(_frame(rows))
    per = {(r.metric, r.model): r.v_norm for r in ded.itertuples()}
    # each group spans its own [0,1]; pooling would put ndcg_at_1 near zero
    assert per[("ndcg_at_1", "m1")] == pytest.approx(0.0)
    assert per[("ndcg_at_1", "m2")] == pytest.approx(1.0)
    assert per[("ndcg_at_10", "m1")] == pytest.approx(0.0)


def test_a_constant_group_becomes_one_half_rather_than_nan():
    rows = [("m1", "d", "t", "accuracy", "higher", 0.7),
            ("m2", "d", "t", "accuracy", "higher", 0.7)]
    _e, _n, ded, rep = C.build_supervision(_frame(rows))
    assert rep["constant_groups_set_to_0.5"] == 2
    assert set(ded.v_norm) == {0.5}


def test_direction_known_metric_wins_the_primary_slot():
    rows = [("m%d" % i, "d", "t", "accuracy", "higher", i / 10) for i in range(3)]
    rows += [("m%d" % i, "d", "t", "some_custom", "unknown", i) for i in range(3)]
    _e, nodes, _d, _r = C.build_supervision(_frame(rows))
    assert nodes.iloc[0]["primary_metric"] == "accuracy"
    assert bool(nodes.iloc[0]["gold_eligible"])


def test_a_node_with_only_unverified_metrics_keeps_edges_but_loses_gold():
    """D-61: keep the edge, drop it from gold."""
    rows = [("m%d" % i, "d", "t", "mystery", "unknown", i) for i in range(4)]
    edges, nodes, _d, _r = C.build_supervision(_frame(rows))
    assert len(edges) == 4                       # edges survive
    assert not bool(nodes.iloc[0]["gold_eligible"])
    assert not bool(nodes.iloc[0]["direction_known"])


def test_reward_nodes_keep_edges_and_are_excluded_from_gold():
    rows = [("m%d" % i, "lunarlander-v2", "reinforcement-learning",
             "mean_reward", "reward", float(i)) for i in range(5)]
    edges, nodes, _d, _r = C.build_supervision(_frame(rows))
    assert len(edges) == 5
    assert bool(nodes.iloc[0]["is_rl"])
    assert not bool(nodes.iloc[0]["gold_eligible"])


def test_placeholder_dataset_names_are_kept_as_edges_but_never_queried():
    rows = [("m%d" % i, "unknown", "text-classification", "accuracy", "higher",
             i / 10) for i in range(5)]
    edges, nodes, _d, _r = C.build_supervision(_frame(rows))
    assert len(edges) == 5
    assert bool(nodes.iloc[0]["is_placeholder"])
    assert not bool(nodes.iloc[0]["gold_eligible"])


def test_fewer_than_three_models_is_not_a_query():
    rows = [("m1", "d", "t", "accuracy", "higher", 0.1),
            ("m2", "d", "t", "accuracy", "higher", 0.9)]
    _e, nodes, _d, _r = C.build_supervision(_frame(rows))
    assert int(nodes.iloc[0]["n_models"]) == 2
    assert not bool(nodes.iloc[0]["gold_eligible"])


def test_the_cap_bounds_a_node_and_keeps_the_value_range():
    rows = [("m%04d" % i, "big", "t", "accuracy", "higher", i / 500.0)
            for i in range(500)]
    edges, _n, _d, rep = C.build_supervision(_frame(rows), cap=50)
    assert rep["edges_before_cap"] == 500
    assert rep["edges_after_cap"] == 50
    assert rep["nodes_capped"] == 1
    # stratified, not head/tail: both ends of the distribution survive
    assert edges.weight.min() < 0.1 and edges.weight.max() > 0.9


def test_the_cap_is_deterministic():
    rows = [("m%04d" % i, "big", "t", "accuracy", "higher", i / 300.0)
            for i in range(300)]
    a, _n, _d, _r = C.build_supervision(_frame(rows), cap=40)
    b, _n2, _d2, _r2 = C.build_supervision(_frame(rows), cap=40)
    assert sorted(a.model) == sorted(b.model)
