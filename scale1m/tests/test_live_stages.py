import json

import numpy as np
import pandas as pd
import pytest
import torch

from scale1m import build_ladder_rf as ladder_module
from scale1m import merge_supervision as merge_module
from scale1m.canonicalize_rf import build_supervision
from scale1m.reproduction.live_stages import check_canonical, check_graph, split_eligibility
from scale1m.reproduction.snapshot import sha256


def canonical_fixture(path, *, direction="higher", n_roots=15):
    canon = path / "canon"
    canon.mkdir(parents=True)
    raw = pd.DataFrame([
        {"model": f"owner/model-{m}", "dataset": f"dataset-{r}/main", "task": "text-classification",
         "metric": "accuracy" if direction == "higher" else "wer", "direction": direction,
         "value": float(m)}
        for r in range(n_roots) for m in range(5)])
    sup, _, _, _ = build_supervision(raw)
    sup.to_parquet(canon / "supervision_uncapped.parquet", index=False)
    sup.to_parquet(canon / "supervision.parquet", index=False)
    edges, nodes, conflicts, _ = merge_module.merge(path, hf_only=True, source_dir="MUST_NOT_BE_READ")
    edges.to_parquet(canon / "supervision_merged.parquet", index=False)
    nodes.to_parquet(canon / "dataset_nodes_merged.parquet", index=False)
    conflicts.to_parquet(canon / "supervision_conflicts.parquet", index=False)
    pd.DataFrame({"model": pd.Series(dtype=str), "in_snapshot": pd.Series(dtype=bool)}).to_parquet(
        canon / "models_out_of_snapshot.parquet", index=False)
    merge_module.write_rules(path, 200, hf_only=True)
    candidates = pd.DataFrame({
        "model": [f"owner/model-{i}" for i in range(1000)], "size_b": np.nan,
        "family": "other", "family_source": "name", "lineage_base": None,
        "lineage_relation": None, "lineage_source": "none", "layer": "original"})
    candidates.to_parquet(canon / "part-00000.parquet", index=False)
    return edges, nodes


def test_hf_only_preserves_lower_metric_direction_and_never_loads_history(tmp_path, monkeypatch):
    monkeypatch.setattr(merge_module, "load_source", lambda *a, **k: pytest.fail("Historical graph was read"))
    edges, nodes = canonical_fixture(tmp_path / "rf", direction="lower")
    assert set(edges.direction) == {"lower"}
    assert set(nodes.primary_direction) == {"lower"}
    assert set(edges.source) == {"hf_model_index"}
    first = edges[edges.node == edges.node.iloc[0]].set_index("model")
    assert first.loc["owner/model-0", "weight"] == 1
    assert first.loc["owner/model-4", "weight"] == 0
    report = check_canonical(tmp_path / "rf")
    assert report["performance_edges"] == 75
    assert all(x["n_eligible_queries"] > 0 for x in report["split_cohorts"].values())


def test_hf_only_ladder_never_reads_history_or_appends_models(tmp_path, monkeypatch):
    rf = tmp_path / "rf"
    canonical_fixture(rf)
    monkeypatch.setattr(ladder_module, "family_from_history", lambda *a, **k: pytest.fail("Historical family graph was read"))
    models, nodes, report = ladder_module.build(rf, tmp_path / "ladder", hf_only=True)
    assert len(models) == 1000 and models.in_snapshot.all()
    assert len(nodes) == 15 and report["models"]["historical_only"] == 0
    pd.DataFrame({"model": ["old/absent-model"], "in_snapshot": [False]}).to_parquet(
        rf / "canon/models_out_of_snapshot.parquet", index=False)
    with pytest.raises(ValueError, match="cannot append"):
        ladder_module.build(rf, tmp_path / "bad-ladder", hf_only=True)


def test_unknown_direction_is_not_promoted_to_gold(tmp_path):
    canonical_fixture(tmp_path / "rf", direction="unknown")
    nodes = pd.read_parquet(tmp_path / "rf/canon/dataset_nodes_merged.parquet")
    assert not nodes.gold_eligible.any()
    with pytest.raises(ValueError, match="no eligible held-out"):
        check_canonical(tmp_path / "rf")


def test_curated_marker_is_rejected_even_if_rules_claim_hf_only(tmp_path):
    canonical_fixture(tmp_path / "rf")
    path = tmp_path / "rf/canon/supervision_merged.parquet"
    edges = pd.read_parquet(path)
    edges["direction"] = "curated"
    edges.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="self-reports"):
        check_canonical(tmp_path / "rf")


def test_early_cohorts_match_actual_production_root_splits(tmp_path):
    from torch_geometric.data import HeteroData
    from ModelLakeFishing.stage2TrainGraphSAGE.d0_splits import make_root_aware_splits
    from ModelLakeFishing.stage2TrainGraphSAGE.losses import TRAINED_ON, REV_TRAINED_ON

    edges, nodes = canonical_fixture(tmp_path / "rf")
    nodes = nodes.sort_values("node").reset_index(drop=True)
    node_ids = dict(zip(nodes.node, range(len(nodes))))
    model_ids = {name: i for i, name in enumerate(sorted(set(edges.model)))}
    graph = HeteroData()
    graph["model"].num_nodes = 1000
    graph["dataset"].num_nodes = len(nodes)
    ei = torch.tensor([edges.model.map(model_ids).tolist(), edges.node.map(node_ids).tolist()])
    graph[TRAINED_ON].edge_index = ei
    graph[TRAINED_ON].edge_attr = torch.tensor(edges.weight.to_numpy(), dtype=torch.float32)
    graph[REV_TRAINED_ON].edge_index = ei.flip(0)
    graph[REV_TRAINED_ON].edge_attr = graph[TRAINED_ON].edge_attr.clone()
    expected = split_eligibility(nodes, edges)
    for seed in (0, 1, 2):
        _, _, test = make_root_aware_splits(graph, nodes.dataset.str.split("/").str[0].tolist(),
                                           split_seed=seed, neg_ratio=0)
        selected = sorted(nodes.iloc[test[TRAINED_ON].edge_label_index[1].unique().numpy()].node.tolist())
        assert selected == expected[str(seed)]["eligible_query_nodes"]


def test_early_gate_rejects_insufficient_root_diversity(tmp_path):
    canonical_fixture(tmp_path / "rf", n_roots=1)
    with pytest.raises(ValueError, match="nonempty train/validation/test"):
        check_canonical(tmp_path / "rf")


def graph_fixture(tmp_path, monkeypatch):
    from scale1m import build_graph_rf as builder
    rf, ladder, features, graph = (tmp_path / name for name in ("rf", "ladder", "features", "graph"))
    canonical_fixture(rf)
    _, nodes, _ = ladder_module.build(rf, ladder, hf_only=True)
    features.mkdir()
    np.save(features / "x_m.npy", np.zeros((1000, 448), dtype=np.float32))
    np.save(features / "family_id.npy", np.zeros(1000, dtype=np.int64))
    np.save(features / "size_bucket_id.npy", np.zeros(1000, dtype=np.int64))
    pd.DataFrame({"family": ["other"], "family_id": [0]}).to_csv(features / "family_vocab.csv", index=False)
    cards = tmp_path / "cards.parquet"
    pd.DataFrame({"dataset": nodes.dataset, "task": nodes.task, "card_source": "name_only"}).to_parquet(cards, index=False)
    def embedding(texts, **kwargs):
        x = np.random.default_rng(13).normal(size=(len(texts), 384)).astype(np.float32)
        return x / np.linalg.norm(x, axis=1, keepdims=True), "cpu"
    monkeypatch.setattr(builder, "minilm", embedding)
    report = builder.build(ladder, features, rf, cards, graph, device="cpu")
    assert all(report["gates"].values())
    return rf, ladder, graph


def test_dynamic_graph_validates_without_frozen_a0_universe(tmp_path, monkeypatch):
    rf, ladder, graph = graph_fixture(tmp_path, monkeypatch)
    report = check_graph(graph, rf, ladder)
    assert report["models"] == 1000 and report["dataset_task_nodes"] == 15
    assert report["seven_performance_columns_zero"]
    path = graph / "x_dataset.npy"
    x = np.load(path)
    x[0, 448] = 1
    np.save(path, x)
    meta = json.loads((graph / "meta.json").read_text(encoding="utf-8"))
    meta["files"]["x_dataset.npy"] = sha256(path)
    (graph / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="feature masking"):
        check_graph(graph, rf, ladder)


def test_export_uses_live_training_universe_and_preserves_frozen_full_gate():
    from scale1m.export_rf import expected_pool_size
    assert expected_pool_size({"rung": "live", "n_models": 4567890, "expect_n": 4567890}, "live") == 4567890
    assert expected_pool_size({"rung": "full"}, "full") == 3016439
    with pytest.raises(ValueError, match="actual candidate count"):
        expected_pool_size({"rung": "live", "n_models": 4567890, "expect_n": None}, "live")
    with pytest.raises(ValueError, match="live training manifest"):
        expected_pool_size({"rung": "full", "n_models": 3016439, "expect_n": 3016439}, "live")


def test_live_training_binds_observed_size_and_rejects_changed_resume_recipe(tmp_path, monkeypatch):
    from scale1m import checkpoint as CK, train_rung as driver
    from scale1m.tests.test_a0_smoke import mocked_driver, run_args

    graph, _ = mocked_driver(tmp_path, monkeypatch)
    out = tmp_path / "live-run"
    args = run_args(graph, out, epochs=2)
    args[args.index("--rung") + 1] = "live"
    assert driver.main(args) == 0
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["n_models"] == manifest["expect_n"] == 3
    assert manifest["evidence_mode"] == "live_hf_only"
    assert manifest["historical_a0_result"] is False
    args[args.index("--epochs") + 1] = "3"
    args[args.index("--lake-gamma") + 1] = "0.3"
    with pytest.raises(CK.IncompatibleCheckpoint, match="cfg.lake_gamma"):
        driver.main(args)
