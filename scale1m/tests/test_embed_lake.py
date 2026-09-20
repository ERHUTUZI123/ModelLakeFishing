import numpy as np
import pytest

from scale1m import embed_lake as el


def test_halo_descriptor_is_cores_function():
    from scale.modellens_build_graph import model_descriptor
    got = el.halo_descriptors(["Org/Some-Model"], ["llama"], [7.0])
    assert got == [model_descriptor("Org/Some-Model", "llama", 7.0)]
    assert got[0] == "Org Some Model family llama 7B params"


def test_descriptor_drops_the_size_clause_when_size_is_nan():
    with_size = el.halo_descriptors(["a/b"], ["bert"], [0.11])[0]
    without = el.halo_descriptors(["a/b"], ["bert"], [float("nan")])[0]
    assert "params" in with_size and "params" not in without


def test_text_stats_counts_the_size_clause():
    st = el.text_stats(["x family bert 7B params", "y family bert"])
    assert st["n"] == 2 and st["with_size_clause"] == 0.5
    assert st["with_family_clause"] == 1.0


def _toy_core_vocab(*dynamic):
    from dataset_embed.utils.fetch_metadata import KNOWN_FAMILIES
    vocab = {"Other": 0}
    for fam in list(KNOWN_FAMILIES) + list(dynamic):
        vocab.setdefault(fam, len(vocab))
    return vocab


def test_family_vocab_appends_and_never_moves_core_rows(tmp_path):
    core_vocab = _toy_core_vocab("bert")
    path = tmp_path / "family_vocab.csv"
    halo = ["bert"] * 5 + ["newfam"] * 3 + ["rare"] * 2
    after, n_new = el.extend_family_vocab(core_vocab, halo, str(path))

    for fam, fid in core_vocab.items():
        assert after[fam] == fid
    assert "newfam" in after and after["newfam"] >= len(core_vocab)
    assert "rare" not in after
    assert n_new == len(after) - len(core_vocab)
    assert path.exists()


def test_family_vocab_rejects_non_contiguous_core_ids(tmp_path):
    with pytest.raises(AssertionError, match="contiguous"):
        el.extend_family_vocab({"Other": 0, "bert": 7}, ["bert"],
                               str(tmp_path / "v.csv"))


def test_rare_family_falls_back_to_other_id_zero(tmp_path):
    from dataset_embed.xm0_builder import build_family_ids
    core_vocab = _toy_core_vocab("bert")
    after, _ = el.extend_family_vocab(core_vocab, ["rare", "rare"],
                                      str(tmp_path / "v.csv"))
    assert list(build_family_ids(["rare", "bert"], after)) == [0, core_vocab["bert"]]


def test_row_order_gate_catches_a_shifted_halo_block():
    models = ["core/a", "core/b"] + ["halo/%d" % i for i in range(40)]
    n_core = 2
    e = el.name_embeddings(models)
    x = np.concatenate([e, np.zeros((len(models), el.DESC_DIM), np.float32)], 1)

    k, delta = el.gate_row_order(x, models, n_core, k=20)
    assert k == 20 and delta < 1e-6

    shifted = x.copy()
    shifted[n_core:] = np.roll(shifted[n_core:], 1, axis=0)
    _, bad = el.gate_row_order(shifted, models, n_core, k=20)
    assert bad > 1e-6


def test_core_verbatim_gate_is_bytewise():
    import torch
    x_core = torch.randn(5, el.X_DIM)
    x = np.concatenate([x_core.numpy(), np.zeros((3, el.X_DIM), np.float32)], 0)
    assert el.gate_core_verbatim(x, x_core)
    x[6, 0] += 1.0
    assert el.gate_core_verbatim(x, x_core)
    x[0, 0] = np.float32(x[0, 0]) + np.float32(1e-6)
    assert not el.gate_core_verbatim(x, x_core)


def test_separability_auc_is_half_on_identically_distributed_halves():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, el.X_DIM)).astype(np.float32)
    auc = el.separability_auc(x, n_core=200, seed=0)
    assert 0.3 < auc["full"] < 0.7
    assert auc["n_per_class"] == 200


def test_separability_auc_is_one_when_the_groups_are_trivially_split():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, el.X_DIM)).astype(np.float32)
    x[200:] += 20.0
    assert el.separability_auc(x, n_core=200, seed=0)["full"] > 0.99
