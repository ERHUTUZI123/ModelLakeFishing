"""Frozen vocabulary bytes and append-only embedding row identities."""
import pandas as pd
import pytest

import scale1m  # Registers the portable dataset_embed import location.

pytest.importorskip("huggingface_hub")
from dataset_embed.xm0_builder import load_or_update_family_vocab


def frozen_csv(path, *dynamic):
    vocab = load_or_update_family_vocab([])
    for family in dynamic:
        assert family not in vocab
        vocab[family] = len(vocab)
    raw = pd.DataFrame({"family": list(vocab), "family_id": list(vocab.values())}).to_csv(
        index=False, lineterminator="\r\n").encode("utf-8")
    path.write_bytes(raw)
    return vocab, raw


def test_unchanged_frozen_mapping_keeps_crlf_bytes_without_serializing(tmp_path, monkeypatch):
    path = tmp_path / "family_vocab.csv"
    original, raw = frozen_csv(path, "nan", "null", "frozen-family")

    def unexpected_write(*args, **kwargs):
        raise AssertionError("An unchanged vocabulary must not be serialized again")

    monkeypatch.setattr(pd.DataFrame, "to_csv", unexpected_write)
    got = load_or_update_family_vocab(
        ["nan"] * 3 + ["null"] * 3 + ["frozen-family"] * 3 + ["too-rare"] * 2,
        vocab_path=path)
    assert got == original
    assert got["nan"] != got["null"]
    assert "too-rare" not in got
    assert path.read_bytes() == raw
    assert b"\r\n" in raw


def test_new_family_appends_without_changing_any_frozen_id(tmp_path):
    path = tmp_path / "family_vocab.csv"
    original, raw = frozen_csv(path, "nan", "null", "frozen-family")
    got = load_or_update_family_vocab(["new-z"] * 3 + ["new-a"] * 3, vocab_path=path)
    assert all(got[family] == identifier for family, identifier in original.items())
    assert got["new-a"] == len(original)
    assert got["new-z"] == len(original) + 1
    assert path.read_bytes() != raw
    saved = pd.read_csv(path, keep_default_na=False)
    assert dict(zip(saved.family, saved.family_id)) == got
    updated_bytes = path.read_bytes()
    assert load_or_update_family_vocab([], vocab_path=path) == got
    assert path.read_bytes() == updated_bytes


def test_missing_static_entries_are_written_even_without_new_dynamic_family(tmp_path):
    path = tmp_path / "family_vocab.csv"
    path.write_bytes(b"family,family_id\r\nOther,0\r\n")
    got = load_or_update_family_vocab([], vocab_path=path)
    saved = pd.read_csv(path, keep_default_na=False)
    assert dict(zip(saved.family, saved.family_id)) == got
    assert len(got) > 1 and got["Other"] == 0
