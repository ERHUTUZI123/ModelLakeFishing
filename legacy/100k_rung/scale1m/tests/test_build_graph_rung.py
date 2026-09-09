"""Tests for T5 graph assembly (scale1m/build_graph_rung.py + verify_rung_graph.py).

The end-to-end test builds a 5-model toy rung out of a 3-model toy CORE. It is
worth the setup cost: almost every way this stage can break is a wiring mistake
(an edge copied from the wrong store, a prefix that stops being CORE, a lineage
edge pointing at the base's row instead of the derivative's), and none of those
show up in a unit test of the join function alone.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData

from scale1m import build_graph_rung as bg
from scale1m import verify_rung_graph as vg
from scale1m.embed_lake import X_DIM

N_CORE, N = 3, 5


# ── the join ─────────────────────────────────────────────────────────────────

def _ladder(models, bases, relations=None):
    d = {"mappedID": range(len(models)), "model": models,
         "lineage_base": bases}
    if relations is not None:
        d["lineage_relation"] = relations
    return pd.DataFrame(d)


def test_lineage_join_points_base_to_derivative():
    lad = _ladder(["a/x", "b/y", "c/z", "d/w"], [None, None, "a/x", "b/y"])
    src, dst, rel, st = bg.lineage_edges(lad, n_core=2)
    assert list(zip(src, dst)) == [(0, 2), (1, 3)]
    assert st == {"declared": 2, "unresolved": 0, "self_loop": 0}


def test_lineage_join_is_case_insensitive():
    lad = _ladder(["Org/Base-Model", "x/y"], [None, "org/base-model"])
    src, dst, _, st = bg.lineage_edges(lad, n_core=1)
    assert (src, dst) == ([0], [1]) and st["unresolved"] == 0


def test_lineage_join_drops_bases_outside_the_rung():
    lad = _ladder(["a/x", "b/y"], [None, "somebody/not-in-lake"])
    src, _, _, st = bg.lineage_edges(lad, n_core=1)
    assert src == [] and st == {"declared": 1, "unresolved": 1, "self_loop": 0}


def test_lineage_join_drops_self_loops():
    lad = _ladder(["a/x", "b/y"], [None, "b/y"])
    src, _, _, st = bg.lineage_edges(lad, n_core=1)
    assert src == [] and st["self_loop"] == 1


def test_lineage_join_never_reads_core_rows():
    """CORE rows have a blank lineage_base by D-38; if one ever carried a value
    the join must still ignore it, or CORE would gain an edge it never had."""
    lad = _ladder(["a/x", "b/y", "c/z"], ["c/z", "a/x", None])
    src, dst, _, st = bg.lineage_edges(lad, n_core=2)
    assert (src, dst) == ([], []) and st["declared"] == 0


def test_unknown_relation_is_mapped_not_dropped():
    lad = _ladder(["a/x", "b/y", "c/z"], [None, "a/x", "a/x"],
                  [None, "finetune", "no-such-relation"])
    _, _, rel, _ = bg.lineage_edges(lad, n_core=1)
    assert rel == ["finetune", "unknown"]
    assert all(r in bg.RELATION_VOCAB for r in rel)


def test_components_counts_weakly_connected_groups():
    n_comp, largest = bg._components(np.array([0, 1, 5]), np.array([1, 2, 6]), 10)
    assert (n_comp, largest) == (2, 3)


def test_components_on_an_empty_lineage_graph():
    assert bg._components(np.array([], int), np.array([], int), 4) == (0, 0)


# ── end to end ───────────────────────────────────────────────────────────────

def _toy_core(tmp_path):
    torch.manual_seed(0)
    d = HeteroData()
    d["model"].node_id = torch.arange(N_CORE)
    d["model"].x = torch.randn(N_CORE, X_DIM)
    d["model"].size_bucket_id = torch.tensor([0, 1, 2])
    d["model"].family_id = torch.tensor([0, 1, 1])
    d["dataset"].node_id = torch.arange(2)
    d["dataset"].x = torch.randn(2, 458)
    d["dataset"].task_type_id = torch.tensor([0, 1])

    ei = torch.tensor([[0, 1, 2], [0, 1, 0]])
    attr = torch.tensor([0.5, 0.6, 0.7])
    d[bg.TRAINED_ON].edge_index, d[bg.TRAINED_ON].edge_attr = ei, attr
    d[bg.REV_TRAINED_ON].edge_index = ei.flip(0)
    d[bg.REV_TRAINED_ON].edge_attr = attr.clone()
    sim = torch.tensor([[0, 1], [1, 0]])
    d[bg.SIMILAR_TO].edge_index = sim
    d[bg.SIMILAR_TO].edge_attr = torch.tensor([0.9, 0.9])
    li = torch.tensor([[0], [1]])
    d[bg.IS_BASE_OF].edge_index, d[bg.IS_BASE_OF].edge_attr = li, torch.ones(1)
    d[bg.REV_IS_BASE_OF].edge_index = li.flip(0)
    d[bg.REV_IS_BASE_OF].edge_attr = torch.ones(1)

    payload = {
        "data": d,
        "xm0_meta": {"num_size_buckets": 15, "num_families": 2,
                     "family_vocab": {"Other": 0, "bert": 1},
                     "name_dim": 64, "desc_dim": 384},
        "xd0_meta": {"num_task_types": 2},
        "unique_model_id": pd.DataFrame({"model": ["c/0", "c/1", "c/2"],
                                         "mappedID": range(N_CORE)}),
        "unique_dataset_id": pd.DataFrame({"dataset": ["d0", "d1"],
                                           "mappedID": range(2),
                                           "root": ["r0", "r1"]}),
        "provenance": {"corpus": "toy"},
    }
    p = tmp_path / "core.pt"
    torch.save(payload, p)
    return p, payload


def _toy_rung(tmp_path, core_payload, halo_bases=("c/0", "h/0")):
    models = ["c/0", "c/1", "c/2", "h/0", "h/1"]
    lad = pd.DataFrame({
        "mappedID": range(N), "model": models,
        "layer": ["core"] * N_CORE + ["lineage"] * 2,
        "size_b": [None] * N_CORE + [1.0, 2.0],
        "family": [None] * N_CORE + ["bert", "newfam"],
        "lineage_base": [None] * N_CORE + list(halo_bases),
    })
    lp = tmp_path / "ladder.csv"
    lad.to_csv(lp, index=False)

    fd = tmp_path / "feats"
    fd.mkdir()
    x = np.concatenate([core_payload["data"]["model"].x.numpy(),
                        np.random.RandomState(0).randn(N - N_CORE, X_DIM)
                        ], 0).astype(np.float32)
    np.save(fd / "x_m.npy", x)
    np.save(fd / "size_bucket_id.npy", np.array([0, 1, 2, 3, 4]))
    np.save(fd / "family_id.npy", np.array([0, 1, 1, 1, 2]))
    pd.DataFrame({"family": ["Other", "bert", "newfam"],
                  "family_id": [0, 1, 2]}).to_csv(fd / "family_vocab.csv",
                                                  index=False)
    return lp, fd


def test_end_to_end_build_and_gates(tmp_path):
    core_p, core = _toy_core(tmp_path)
    lad_p, feats = _toy_rung(tmp_path, core)
    out = tmp_path / "g.pt"

    payload, rep = bg.build(str(core_p), str(lad_p), str(feats), str(out), "toy")
    d, cd = payload["data"], core["data"]

    # CORE carried through untouched
    assert torch.equal(d["model"].x[:N_CORE], cd["model"].x)
    assert torch.equal(d["dataset"].x, cd["dataset"].x)
    assert torch.equal(d[bg.TRAINED_ON].edge_index, cd[bg.TRAINED_ON].edge_index)
    assert torch.equal(d[bg.SIMILAR_TO].edge_index, cd[bg.SIMILAR_TO].edge_index)

    # CORE's 1 lineage edge plus the 2 the join found (c/0->h/0, h/0->h/1)
    assert rep["lineage"]["total_edges"] == 3
    assert rep["lineage"]["core_core"] == 1
    assert rep["lineage"]["core_halo"] == 1
    assert rep["lineage"]["halo_halo"] == 1
    assert torch.equal(d[bg.REV_IS_BASE_OF].edge_index,
                       d[bg.IS_BASE_OF].edge_index.flip(0))

    # vocab grew, CORE ids fixed
    assert payload["xm0_meta"]["num_families"] == 3
    assert payload["xm0_meta"]["family_vocab"]["bert"] == 1

    g, _ = vg.verify(str(out), str(core_p), "toy", skip_split=True,
                     skip_contract=True)
    assert not g.failed


def test_lineage_edge_attr_defaults_to_cores_value(tmp_path):
    core_p, core = _toy_core(tmp_path)
    lad_p, feats = _toy_rung(tmp_path, core)
    payload, _ = bg.build(str(core_p), str(lad_p), str(feats),
                          str(tmp_path / "g.pt"), "toy")
    attr = payload["data"][bg.IS_BASE_OF].edge_attr
    assert torch.equal(attr, torch.ones_like(attr)), \
        "relation weights must be opt-in; CORE's lineage attr is 1.0"
    assert payload["data"][bg.IS_BASE_OF].relation_id.shape == attr.shape


def test_relation_weights_are_opt_in(tmp_path):
    core_p, core = _toy_core(tmp_path)
    lad_p, feats = _toy_rung(tmp_path, core)
    pd.read_csv(lad_p).assign(
        lineage_relation=[None] * N_CORE + ["merge", "adapter"]
    ).to_csv(lad_p, index=False)

    payload, _ = bg.build(str(core_p), str(lad_p), str(feats),
                          str(tmp_path / "g.pt"), "toy", relation_weights=True)
    attr = payload["data"][bg.IS_BASE_OF].edge_attr
    assert attr[0].item() == 1.0                      # CORE's edge, untouched
    assert sorted(attr[1:].tolist()) == [0.25, 0.75]  # merge, adapter


def test_gates_catch_a_tampered_core_prefix(tmp_path):
    core_p, core = _toy_core(tmp_path)
    lad_p, feats = _toy_rung(tmp_path, core)
    out = tmp_path / "g.pt"
    bg.build(str(core_p), str(lad_p), str(feats), str(out), "toy")

    payload = torch.load(out, weights_only=False)
    payload["data"]["model"].x[0, 0] += 1.0
    torch.save(payload, out)

    g, _ = vg.verify(str(out), str(core_p), "toy", skip_split=True,
                     skip_contract=True)
    assert any("byte-identical" in m for m in g.failed)


def test_build_refuses_a_feature_matrix_whose_core_prefix_was_recomputed(tmp_path):
    core_p, core = _toy_core(tmp_path)
    lad_p, feats = _toy_rung(tmp_path, core)
    x = np.load(feats / "x_m.npy")
    x[0, 0] += 1e-6
    np.save(feats / "x_m.npy", x)

    with pytest.raises(AssertionError, match="byte-identical"):
        bg.build(str(core_p), str(lad_p), str(feats), str(tmp_path / "g.pt"), "toy")
