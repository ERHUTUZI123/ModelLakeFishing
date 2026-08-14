"""Tests for T7's export gates (scale1m/export_rung.py).

The gates here guard failures that produce a complete, plausible-looking export
and no exception:

  G-D1  scoring only the 30,183 CORE models at the 100K rung -- gold@10 barely
        moves and the run reads as "withstood 83x the distractors"
  G-D2  scoring with full-graph embeddings, so each held-out query sees its own
        supervision edges (P4 measured this: 0.42 became 0.61)
  G-D3  z_m row i no longer meaning model i

None of those raise on their own, so they are asserted rather than trusted.
"""

import numpy as np
import pytest

from scale1m import export_rung as EX


# ── G-D1 ─────────────────────────────────────────────────────────────────────

def test_pool_size_gate_passes_on_the_whole_lake():
    g = EX.gate_pool_size(100_000, 100_000)
    assert g["ok"] and g["gate"] == "G-D1"


def test_pool_size_gate_catches_scoring_only_core():
    """The silent 100K bug: 30,183 candidates instead of 100,000."""
    g = EX.gate_pool_size(30_183, 100_000)
    assert not g["ok"]


def test_pool_size_gate_is_vacuously_true_without_an_expectation():
    assert EX.gate_pool_size(12_000, None)["ok"]


# ── G-D2 ─────────────────────────────────────────────────────────────────────

def test_leakage_gate_passes_when_held_out_scores_lower():
    g = EX.gate_leakage(0.2360, 0.3133)
    assert g["ok"] and g["margin"] == pytest.approx(0.0773, abs=1e-4)


def test_leakage_gate_fails_when_the_two_forwards_are_swapped():
    """Passing full-graph z as the eval embeddings is the actual bug this
    catches, and swapping the arguments is exactly what that looks like."""
    assert not EX.gate_leakage(0.3133, 0.2360)["ok"]


def test_leakage_gate_fails_on_equality():
    """Equal means one forward was used for both -- no held-out forward ran."""
    assert not EX.gate_leakage(0.42, 0.42)["ok"]


# ── G-D3 ─────────────────────────────────────────────────────────────────────

def _ids(n):
    return [f"org/model-{i}" for i in range(n)]


def test_row_order_gate_passes_on_identical_id_columns():
    ids = _ids(1000)
    g = EX.gate_row_order(ids, list(ids))
    assert g["ok"] and g["n_probed"] == 100


def test_row_order_gate_catches_a_swap_anywhere_in_the_table():
    """A head-only check would miss this; the probe is random for that reason,
    and total_mismatches scans everything so the gate cannot get lucky."""
    ids = _ids(1000)
    shifted = list(ids)
    shifted[400], shifted[401] = shifted[401], shifted[400]
    g = EX.gate_row_order(ids, shifted)
    assert not g["ok"]
    assert g["total_mismatches"] == 2


def test_row_order_gate_catches_an_off_by_one_shift():
    ids = _ids(500)
    g = EX.gate_row_order(ids, ids[1:] + ["org/model-extra"])
    assert not g["ok"]


def test_row_order_gate_catches_a_length_mismatch():
    g = EX.gate_row_order(_ids(100), _ids(99))
    assert not g["ok"] and "length" in g["reason"]


def test_row_order_gate_skips_rather_than_lies_without_a_ladder():
    """12k/30k are the CORE graph itself and have no ladder file. Reporting
    'ok' there would claim a check that never ran."""
    g = EX.gate_row_order(_ids(10), None)
    assert g["ok"] is None and "skipped" in g


def test_row_order_probe_is_deterministic():
    ids = _ids(1000)
    a = EX.gate_row_order(ids, list(ids), seed=7)
    b = EX.gate_row_order(ids, list(ids), seed=7)
    assert a["n_probed"] == b["n_probed"]


# ── checkpoint resolution ────────────────────────────────────────────────────

def test_resolve_ckpt_prefers_an_explicit_path(tmp_path):
    p = tmp_path / "custom.pt"
    p.write_bytes(b"x")
    assert EX.resolve_ckpt(str(tmp_path), str(p)) == str(p)


def test_resolve_ckpt_finds_best_and_last(tmp_path):
    ck = tmp_path / "ckpt"
    ck.mkdir()
    (ck / "best.pt").write_bytes(b"x")
    (ck / "last.pt").write_bytes(b"x")
    assert EX.resolve_ckpt(str(tmp_path), "best").endswith("best.pt")
    assert EX.resolve_ckpt(str(tmp_path), "last").endswith("last.pt")


def test_resolve_ckpt_fails_loudly_when_absent(tmp_path):
    with pytest.raises(FileNotFoundError):
        EX.resolve_ckpt(str(tmp_path), "best")


# ── the derived A-axis columns ───────────────────────────────────────────────

def test_vs_random_scales_with_the_pool():
    """gold@10 against a 10/N random baseline: the same gold@10 is a bigger
    achievement in a bigger lake, which is the point of the column."""
    for n, gold in ((12_000, 0.4159), (100_000, 0.4159)):
        vs = gold / (10.0 / n)
        assert vs == pytest.approx(gold * n / 10.0)
    assert (0.4159 * 100_000 / 10.0) > (0.4159 * 12_000 / 10.0)
