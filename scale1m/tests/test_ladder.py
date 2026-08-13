"""Tests for T3: canonicalization + ladder assembly.

The ladder's whole job is an integer contract -- row i is the model with
mappedID i, and rows 0..30182 are CORE in CORE's own order. Nothing downstream
raises when that is wrong, so these tests are the only thing standing between a
misalignment and a silently rewired lake.

Run (from ModelLakeFishing/):
    .\\.venv\\Scripts\\python.exe -m pytest scale1m/tests -q
"""

import json
import os

import pandas as pd
import pytest

from scale1m.hf_canonicalize import (normalize, size_b_of, lineage_base_of,
                                     family_of, layer_of)
from scale1m.build_ladder import build, LADDER_COLUMNS


# --- canonicalization ------------------------------------------------------

def test_size_b_is_safetensors_only():
    """D-26: the name regex is NOT a fallback here, however tempting."""
    assert size_b_of({"id": "a/x", "safetensors": {"total": 7_000_000_000}}) == (7.0, "safetensors")
    assert size_b_of({"id": "a/Model-7B-GGUF"}) == (None, "none")
    assert size_b_of({"id": "a/x", "safetensors": {}}) == (None, "none")


def test_lineage_prefers_basemodels_over_card_text():
    rec = {"baseModels": {"relation": "finetune", "ids": ["Meta/Llama-3"]},
           "cardData": {"base_model": "someone/else"}}
    assert lineage_base_of(rec) == ("meta/llama-3", "finetune", "baseModels")
    rec2 = {"cardData": {"base_model": ["Org/Base", "Org/Other"]}}
    assert lineage_base_of(rec2)[::2] == ("org/base", "cardData.base_model")
    assert lineage_base_of({"id": "a/x"}) == (None, None, "none")


def test_family_resolution_targets_the_core_namespace():
    """F-T3-1: CORE's families are config.model_type strings (`llama`), the
    rule table's are `LLaMA`. Landing in the wrong namespace means the same
    architecture gets two embedding rows and shares nothing."""
    used = {"llama", "bert", "vit"}
    # 1. model_type that CORE already uses wins outright
    assert family_of({"id": "meta-llama/Llama-3-8B", "config": {"model_type": "llama"}},
                     used) == ("llama", "config.model_type|core_used")
    # 2. the rule table, lowercased, when model_type is absent
    fam, src = family_of({"id": "google-bert/bert-base-uncased"}, used)
    assert fam == "bert" and src == "name_rule|core_used"
    # 3. an unknown model_type still beats a name guess -- right namespace
    fam, src = family_of({"id": "org/thing", "config": {"model_type": "brandnew"}}, used)
    assert fam == "brandnew" and src == "config.model_type|new"
    # 4. last resort is lowercased, never the capitalized rule output
    fam, src = family_of({"id": "org/mysterious-thing-7b"}, used)
    assert fam == fam.lower() and src == "name_rule|new"


def test_family_never_returns_the_rule_tables_capitalization():
    used = set()
    for mid in ("meta-llama/Llama-3", "google/vit-base", "Qwen/Qwen2-7B"):
        fam, _ = family_of({"id": mid}, used)
        assert fam == fam.lower(), mid


def test_layer_rules_including_the_dead_dropped_rule():
    assert layer_of({"cardData": {"model-index": [1]}, "tags": ["x"]}, "a/b") == "labeled"
    assert layer_of({"tags": ["x"]}, "a/b") == "lineage"
    assert layer_of({"tags": ["x"]}, None) == "plain"
    # D-27: kept verbatim so the zero is REPORTED, not silently absent
    assert layer_of({}, None) == "dropped"
    assert layer_of({"tags": ["region:us"]}, None) == "plain"


def test_normalize_is_the_one_id_rule():
    assert normalize("  Org/Model ") == "org/model"


# --- ladder assembly -------------------------------------------------------

class _FakeCore(dict):
    pass


def _write_fixture(tmp_path, core_models, halo_ids, canon_extra=None, n_halo=None):
    import torch
    core = {
        "unique_model_id": pd.DataFrame({"model": core_models,
                                         "mappedID": range(len(core_models))}),
        "data": None, "xm0_meta": {"family_vocab": {"Other": 0}},
    }
    core_path = str(tmp_path / "core.pt")
    torch.save(core, core_path)

    halo = pd.DataFrame({"id_norm": [normalize(x) for x in halo_ids],
                         "selection_order": range(len(halo_ids))})
    halo_path = str(tmp_path / "halo.parquet")
    halo.to_parquet(halo_path, index=False)

    rows = [{"model": x, "id_norm": normalize(x), "size_b": 1.0 + i,
             "family": "bert", "lineage_base": None, "layer": "plain"}
            for i, x in enumerate(halo_ids)]
    rows += (canon_extra or [])
    canon_path = str(tmp_path / "canon.parquet")
    pd.DataFrame(rows).to_parquet(canon_path, index=False)
    return core_path, halo_path, canon_path


def test_ladder_puts_core_first_in_core_order(tmp_path):
    core_models = ["c/one", "c/two", "c/three"]
    halo_ids = ["h/a", "h/b"]
    cp, hp, kp = _write_fixture(tmp_path, core_models, halo_ids)
    ladder, rep = build(cp, hp, kp, 5, str(tmp_path / "out"), "test")
    assert ladder["mappedID"].tolist() == [0, 1, 2, 3, 4]
    assert ladder["model"].tolist()[:3] == core_models
    assert ladder["layer"].tolist()[:3] == ["core"] * 3
    assert rep["core_halo_intersection"] == 0
    assert rep["distinct_models"] == 5
    assert set(pd.read_csv(rep["ladder_csv"]).columns) == set(LADDER_COLUMNS)


def test_ladder_refuses_a_halo_that_overlaps_core(tmp_path):
    """The failure mode G1-G20 cannot see: the rung silently shrinks."""
    cp, hp, kp = _write_fixture(tmp_path, ["c/one", "c/two"], ["C/ONE", "h/b"])
    with pytest.raises(AssertionError, match="already in CORE"):
        build(cp, hp, kp, 4, str(tmp_path / "out"), "test")


def test_ladder_refuses_a_wrong_sized_halo(tmp_path):
    cp, hp, kp = _write_fixture(tmp_path, ["c/one", "c/two"], ["h/a"])
    with pytest.raises(AssertionError, match="rung needs"):
        build(cp, hp, kp, 5, str(tmp_path / "out"), "test")


def test_ladder_refuses_halo_ids_missing_from_canon(tmp_path):
    import torch
    core_path = str(tmp_path / "core.pt")
    torch.save({"unique_model_id": pd.DataFrame({"model": ["c/one"], "mappedID": [0]}),
                "xm0_meta": {"family_vocab": {"Other": 0}}}, core_path)
    pd.DataFrame({"id_norm": ["h/a"], "selection_order": [0]}).to_parquet(
        tmp_path / "halo.parquet", index=False)
    pd.DataFrame([{"model": "h/zzz", "id_norm": "h/zzz", "size_b": 1.0,
                   "family": "bert", "lineage_base": None, "layer": "plain"}]
                 ).to_parquet(tmp_path / "canon.parquet", index=False)
    with pytest.raises(AssertionError, match="absent from the canon"):
        build(core_path, str(tmp_path / "halo.parquet"), str(tmp_path / "canon.parquet"),
              2, str(tmp_path / "out"), "test")


def test_ladder_halo_order_follows_selection_order(tmp_path):
    """Deterministic mappedIDs: a rerun of the selector reproduces the ladder."""
    import torch
    core_path = str(tmp_path / "core.pt")
    torch.save({"unique_model_id": pd.DataFrame({"model": ["c/one"], "mappedID": [0]}),
                "xm0_meta": {"family_vocab": {"Other": 0}}}, core_path)
    # deliberately shuffled rows with an explicit selection_order
    pd.DataFrame({"id_norm": ["h/b", "h/a"], "selection_order": [1, 0]}).to_parquet(
        tmp_path / "halo.parquet", index=False)
    pd.DataFrame([{"model": m, "id_norm": m, "size_b": 1.0, "family": "bert",
                   "lineage_base": None, "layer": "plain"} for m in ("h/a", "h/b")]
                 ).to_parquet(tmp_path / "canon.parquet", index=False)
    ladder, _ = build(core_path, str(tmp_path / "halo.parquet"),
                      str(tmp_path / "canon.parquet"), 3, str(tmp_path / "out"), "t")
    assert ladder["model"].tolist() == ["c/one", "h/a", "h/b"]


def test_ladder_report_records_the_audit_numbers(tmp_path):
    cp, hp, kp = _write_fixture(tmp_path, ["c/one"], ["h/a", "h/b"])
    _, rep = build(cp, hp, kp, 3, str(tmp_path / "out"), "test")
    for k in ("ladder_sha256", "halo_layer", "halo_size_b_known",
              "halo_family_distinct", "halo_family_other_share",
              "halo_lineage_declared", "halo_family_id_other_share_forecast"):
        assert k in rep, k
    with open(os.path.join(str(tmp_path / "out"), "LADDER_REPORT.json"),
              encoding="utf-8") as fh:
        assert json.load(fh)["n"] == 3
