import json
import os

import numpy as np
import pandas as pd
import pytest

from scale1m import export_rf as E


def test_pool_size_gate_rejects_a_short_pool():
    assert E.gate_pool_size(3_016_439, 3_016_439)["ok"] is True
    bad = E.gate_pool_size(46_146, 3_016_439)
    assert bad["ok"] is False and bad["n_models"] == 46_146


def test_pool_size_gate_is_a_pass_when_no_expectation_is_pinned():
    assert E.gate_pool_size(123, None)["ok"] is True


def test_leakage_gate_is_an_inequality_not_a_note():
    assert E.gate_leakage(0.0700, 0.1046)["ok"] is True
    assert E.gate_leakage(0.61, 0.42)["ok"] is False
    assert E.gate_leakage(0.5, 0.5)["ok"] is False


def test_row_order_gate_catches_a_single_swap_anywhere():
    ids = ["m%05d" % i for i in range(5000)]
    assert E.gate_row_order(ids, list(ids))["ok"] is True
    swapped = list(ids)
    swapped[4321], swapped[4322] = swapped[4322], swapped[4321]
    g = E.gate_row_order(ids, swapped)
    assert g["ok"] is False and g["total_mismatches"] == 2


def test_row_order_gate_catches_a_length_change_before_indexing():
    g = E.gate_row_order(["a", "b"], ["a", "b", "c"])
    assert g["ok"] is False and "length" in g["reason"]


def _exports(tmp_path, models):
    out = tmp_path / "exports"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"model": models, "mappedID": range(len(models))}).to_parquet(
        out / "model_ids.parquet", index=False)
    return str(out)


def test_supervised_rows_maps_ids_to_row_numbers_not_positions(tmp_path):
    out = _exports(tmp_path, ["a/1", "b/2", "c/3", "d/4"])
    sup = tmp_path / "sup.parquet"
    pd.DataFrame({"model": ["c/3", "a/1", "c/3"], "node": ["n"] * 3}).to_parquet(
        sup, index=False)
    got = E.supervised_rows(out, str(sup))
    assert got.tolist() == [0, 2]


def test_supervised_rows_ignores_models_absent_from_the_ladder(tmp_path):
    out = _exports(tmp_path, ["a/1", "b/2"])
    sup = tmp_path / "sup.parquet"
    pd.DataFrame({"model": ["b/2", "ghost/x"], "node": ["n", "n"]}).to_parquet(
        sup, index=False)
    assert E.supervised_rows(out, str(sup)).tolist() == [1]


def test_cands_round_trip_keeps_ids_integral(tmp_path):
    out = str(tmp_path)
    cands = {7: (np.array([3, 11, 2]), np.array([0.1, 0.9, 0.5])),
             9: (np.array([1, 4, 5, 6]), np.array([0.2, 0.3, 0.8, 0.4]))}
    E.save_cands(out, cands)
    back = E.load_cands(out)
    assert set(back) == {7, 9}
    assert back[7][0].dtype == np.int64
    assert back[7][0].tolist() == [3, 11, 2]
    assert back[9][1] == pytest.approx([0.2, 0.3, 0.8, 0.4])


def test_merge_stage_keeps_earlier_stages_and_reaggregates_gates(tmp_path):
    out = str(tmp_path)
    E.merge_stage(out, "embed", {"gates": [{"gate": "G-F7a", "ok": True}]})
    man = E.merge_stage(out, "metrics", {"gates": [{"gate": "G-F7b", "ok": False}]})
    assert set(man["stages"]) == {"embed", "metrics"}
    assert man["gates_failed"] == ["G-F7b"] and man["gates_passed"] is False
    on_disk = json.load(open(os.path.join(out, "EXPORT_MANIFEST.json"), encoding="utf-8"))
    assert on_disk["stages"]["embed"]["gates"][0]["gate"] == "G-F7a"
