"""Unit tests for the F8 evaluator (RF).

Two things here decide numbers rather than report them: restricting the
candidate pool has to remap the gold ids into the restricted row space, and an
index built over a subset returns ORIGINAL labels, so the brute-force reference
has to live in the same label space or every recall reads as zero.
"""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from scale1m import eval_rf as E
from scale1m import eval_rung as R


def test_full_sidecar_default_tracks_exports_and_run_fmt(tmp_path):
    args = SimpleNamespace(exports=str(tmp_path / "exports_x4"),
                           run_fmt="X4GD_full_s%d_e25")
    assert E.default_full_sidecar(args) == os.path.join(
        str(tmp_path / "exports_x4"), "X4GD_full_s0_e25", "prior_sidecar.npz")


def test_main_resolves_full_sidecar_after_parsing_run_fmt(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(E, "axis_s", lambda args: captured.update(vars(args)))
    exports = tmp_path / "exports_x4"
    E.main(["--axis", "s", "--device", "cpu", "--out", str(tmp_path / "out"),
            "--exports", str(exports), "--run-fmt", "X4GD_full_s%d_e25"])
    assert captured["full_sidecar"] == os.path.join(
        str(exports), "X4GD_full_s0_e25", "prior_sidecar.npz")


def test_train_costs_use_an_independent_run_format(tmp_path):
    for seed in E.SEEDS:
        run = tmp_path / ("F6RF_full_s%d_e25" % seed)
        run.mkdir()
        (run / "MANIFEST.json").write_text(json.dumps({
            "wallclock_s": 10 + seed,
            "peak_gpu_mem_gb": 20 + seed,
        }), encoding="utf-8")
    args = SimpleNamespace(
        exports=str(tmp_path / "exports_x4"),
        run_fmt="X4GD_full_s%d_e25",
        f6_runs=str(tmp_path),
        f6_run_fmt="F6RF_full_s%d_e25",
    )
    assert E.load_train_costs(args) == {
        0: {"wallclock_s": 10, "peak_gpu_mem_gb": 20},
        1: {"wallclock_s": 11, "peak_gpu_mem_gb": 21},
        2: {"wallclock_s": 12, "peak_gpu_mem_gb": 22},
    }


def test_train_costs_fail_loudly_when_the_run_family_is_wrong(tmp_path):
    args = SimpleNamespace(f6_runs=str(tmp_path),
                           f6_run_fmt="X4GD_full_s%d_e25")
    with pytest.raises(FileNotFoundError, match="--f6-run-fmt"):
        E.load_train_costs(args)


def test_curve_files_are_optional_and_sorted(tmp_path):
    d0 = tmp_path / "X4GD_full_s0_e25"
    d0.mkdir()
    assert E.curve_index_files(str(d0)) == []          # no `curve` stage
    sub = d0 / "curve"
    sub.mkdir()
    for f in ("hnsw_sub_500k_s1.bin", "hnsw_sub_100k_s0.bin", "notes.json"):
        (sub / f).write_bytes(b"")
    assert [os.path.basename(p) for p in E.curve_index_files(str(d0))] == [
        "hnsw_sub_100k_s0.bin", "hnsw_sub_500k_s1.bin"]


def test_query_eligibility_maps_flags_onto_export_rows(tmp_path):
    import pandas as pd
    nodes = tmp_path / "nodes.parquet"
    pd.DataFrame({
        "node": ["good	task", "unk	task", "##	task", "rl	task"],
        "gold_eligible": [True, False, False, False],
        "primary_direction": ["higher", "unknown", "higher", "higher"],
        "is_placeholder": [False, False, True, False],
        "is_rl": [False, False, False, True],
    }).to_parquet(nodes)
    d0 = tmp_path / "RF_full_s0_e25"
    d0.mkdir()
    # export row order is mappedID order, and it is not the parquet order
    pd.DataFrame({"mappedID": [1, 0, 2, 3],
                  "dataset": ["unk	task", "good	task", "##	task", "rl	task"],
                  "root": ["r1", "r0", "r2", "r3"]}).to_parquet(d0 / "dataset_ids.parquet")

    elig, why = E.query_eligibility(str(nodes), str(d0))
    assert elig.tolist() == [True, False, False, False]
    assert why["direction_unknown"].tolist() == [False, True, False, False]
    assert why["is_placeholder"].tolist() == [False, False, True, False]
    assert why["is_rl"].tolist() == [False, False, False, True]


def test_query_eligibility_treats_an_unknown_node_as_ineligible(tmp_path):
    import pandas as pd
    nodes = tmp_path / "nodes.parquet"
    pd.DataFrame({"node": ["a	t"], "gold_eligible": [True],
                  "primary_direction": ["higher"], "is_placeholder": [False],
                  "is_rl": [False]}).to_parquet(nodes)
    d0 = tmp_path / "RF_full_s0_e25"
    d0.mkdir()
    pd.DataFrame({"mappedID": [0, 1], "dataset": ["a	t", "missing	t"],
                  "root": ["ra", "rm"]}).to_parquet(d0 / "dataset_ids.parquet")
    elig, _ = E.query_eligibility(str(nodes), str(d0))
    assert elig.tolist() == [True, False]


def test_interval_reports_the_spread_not_just_the_mean():
    got = E.interval([0.04, 0.07, 0.0696])
    assert got["min"] == pytest.approx(0.04)
    assert got["max"] == pytest.approx(0.07)
    assert got["mean"] == pytest.approx(0.0598667)
    assert got["per_seed"] == [0.04, 0.07, 0.0696]


def test_merge_keeps_earlier_axes(tmp_path):
    out = str(tmp_path)
    E.merge(out, "a", {"x": 1})
    rep = E.merge(out, "b", {"y": 2})
    assert set(rep["axes"]) == {"a", "b"}
    on_disk = json.load(open(os.path.join(out, "F8_REPORT.json"), encoding="utf-8"))
    assert on_disk["axes"]["a"]["x"] == 1


def _toy(n=40, d=8, seed=0):
    rng = np.random.default_rng(seed)
    zm = rng.normal(size=(n, d)).astype(np.float32)
    zd = rng.normal(size=(3, d)).astype(np.float32)
    cands = {0: (np.array([1, 5, 9, 30]), np.array([0.1, 0.9, 0.5, 0.2])),
             1: (np.array([2, 6, 11]), np.array([0.3, 0.8, 0.4]))}
    roots = {0: "r0", 1: "r1"}
    return zm, zd, cands, roots


def test_curve_point_remaps_gold_into_the_restricted_row_space():
    """Without the remap the gold id indexes a different model and the metric
    is about the wrong row -- which raises nothing."""
    zm, zd, cands, roots = _toy()
    sel = np.arange(0, 40, 2)                       # keeps 5, 9 out; 2, 6 in
    keep_all = np.union1d(sel, np.concatenate([c for c, _a in cands.values()]))
    got = E.curve_point(zm, zd, cands, roots, keep_all, "cpu")
    assert got["N"] == len(keep_all)
    assert 0 < got["median_gold_rank"] <= len(keep_all)
    assert got["median_rank_over_N"] == pytest.approx(
        got["median_gold_rank"] / len(keep_all))


def test_curve_point_drops_a_query_whose_pool_lost_too_many_candidates():
    zm, zd, cands, roots = _toy()
    sel = np.array([1, 5, 9, 30, 2])                # query 1 keeps only one cand
    got = E.curve_point(zm, zd, cands, roots, sel, "cpu")
    assert got["n_queries"] == 1


def test_bench_rung_label_space_default_is_identity():
    """The historical rungs added row i under label i; the new `labels`
    argument must not disturb that path."""
    import inspect
    sig = inspect.signature(R.bench_rung)
    assert sig.parameters["labels"].default is None


def test_subindex_filename_parses_on_the_last_underscore_s():
    """`_s` also occurs inside `sub`, so splitting on the first one yields the
    literal string 'hnsw' as every rung name."""
    for name, n, seed in (("hnsw_sub_100k_s0.bin", "100k", 0),
                          ("hnsw_sub_1000k_s2.bin", "1000k", 2)):
        stem, s = name[:-4].rsplit("_s", 1)
        assert stem.replace("hnsw_sub_", "") == n
        assert int(s) == seed
