"""Unit tests for the F3 ladder (RF).

The ladder is a row-order contract, and a broken one raises nothing: it just
hands every embedding to the wrong model. So each invariant the builder claims
gets a test that fails when the invariant is violated, including the two that
only matter once historical supervision is merged in -- the snapshot has to
stay an exact prefix, and every supervised model has to have a row.
"""
import os

import pandas as pd
import pytest

from scale1m import build_ladder_rf as B

COLS = ["model", "size_b", "family", "family_source", "lineage_base",
        "lineage_relation", "lineage_source", "layer"]


def _rf_dir(tmp_path, snapshot, extra, sup_models=None, nodes=None):
    canon = tmp_path / "canon"
    canon.mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate([snapshot[:2], snapshot[2:]]):
        if not chunk:
            continue
        pd.DataFrame(chunk, columns=COLS).to_parquet(
            canon / ("part-%05d.parquet" % i), index=False)
    pd.DataFrame({"model": extra, "in_snapshot": False}).to_parquet(
        canon / "models_out_of_snapshot.parquet", index=False)
    sup = pd.DataFrame({"model": sup_models if sup_models is not None
                        else [snapshot[0][0]],
                        "node": ["d\tt"] * len(sup_models or [1])})
    sup.to_parquet(canon / "supervision_merged.parquet", index=False)
    nd = nodes if nodes is not None else pd.DataFrame(
        {"node": ["d\tt"], "n_models": [3], "gold_eligible": [True]})
    nd.to_parquet(canon / "dataset_nodes_merged.parquet", index=False)
    return str(tmp_path)


def _row(mid, fam="bert"):
    return [mid, 0.11, fam, "config.model_type|new", None, None, "none", "plain"]


def test_snapshot_stays_an_exact_prefix_and_history_is_appended(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "family_from_history",
                        lambda models, source_dir=None: {m: ("llama", "history:modellens_v2")
                                        for m in models})
    rf = _rf_dir(tmp_path / "rf", [_row("a/1"), _row("a/2"), _row("a/3")],
                 ["z/old", "y/older"], sup_models=["a/1", "z/old"])
    ladder, nodes, rep = B.build(rf, str(tmp_path / "out"))
    assert ladder["model"].tolist()[:3] == ["a/1", "a/2", "a/3"]
    # appended, sorted by id, never interleaved
    assert ladder["model"].tolist()[3:] == ["y/older", "z/old"]
    assert ladder["mappedID"].tolist() == [0, 1, 2, 3, 4]
    assert all(rep["checks"].values())


def test_appended_rows_carry_no_size_and_a_recovered_family(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "family_from_history",
                        lambda models, source_dir=None: {"z/old": ("llama", "history:modellens_v2")})
    rf = _rf_dir(tmp_path / "rf", [_row("a/1")], ["z/old", "q/unknown"],
                 sup_models=["a/1"])
    ladder, _n, rep = B.build(rf, str(tmp_path / "out"))
    app = ladder[~ladder.in_snapshot].set_index("model")
    assert pd.isna(app.loc["z/old", "size_b"])          # no HF record -> no size
    assert app.loc["z/old", "family"] == "llama"
    assert app.loc["q/unknown", "family"] == "other"    # not recoverable -> Other
    assert app.loc["q/unknown", "family_source"] == "history:none"
    assert set(app["layer"]) == {B.LAYER_NO_RECORD}


def test_a_supervised_model_missing_from_the_ladder_is_caught(tmp_path, monkeypatch):
    """Edges pointing at a model with no row would index into nothing."""
    monkeypatch.setattr(B, "family_from_history", lambda models, source_dir=None: {})
    rf = _rf_dir(tmp_path / "rf", [_row("a/1")], [],
                 sup_models=["a/1", "ghost/model"])
    _l, _n, rep = B.build(rf, str(tmp_path / "out"))
    assert rep["checks"]["every_supervised_model_has_a_row"] is False


def test_duplicate_ids_between_snapshot_and_history_are_caught(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "family_from_history", lambda models, source_dir=None: {})
    rf = _rf_dir(tmp_path / "rf", [_row("a/1")], ["a/1"], sup_models=["a/1"])
    _l, _n, rep = B.build(rf, str(tmp_path / "out"))
    assert rep["checks"]["appended_rows_are_not_in_the_snapshot"] is False
    assert rep["checks"]["ids_unique_after_normalisation"] is False


def test_dataset_side_gets_its_own_contiguous_order(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "family_from_history", lambda models, source_dir=None: {})
    nodes = pd.DataFrame({"node": ["b\tt", "a\tt", "c\tt"],
                          "n_models": [3, 4, 5],
                          "gold_eligible": [True, True, False]})
    rf = _rf_dir(tmp_path / "rf", [_row("a/1")], [], sup_models=["a/1"],
                 nodes=nodes)
    _l, nd, rep = B.build(rf, str(tmp_path / "out"))
    assert nd["node"].tolist() == ["a\tt", "b\tt", "c\tt"]   # deterministic
    assert nd["mappedID"].tolist() == [0, 1, 2]
    assert rep["datasets"]["gold_eligible"] == 2


def test_report_records_a_sha256_for_both_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "family_from_history", lambda models, source_dir=None: {})
    rf = _rf_dir(tmp_path / "rf", [_row("a/1")], [], sup_models=["a/1"])
    _l, _n, rep = B.build(rf, str(tmp_path / "out"))
    assert len(rep["sha256"]["full_model_ids.parquet"]) == 64
    assert len(rep["sha256"]["full_dataset_ids.parquet"]) == 64
    assert os.path.exists(os.path.join(str(tmp_path / "out"), "LADDER_REPORT.json"))
