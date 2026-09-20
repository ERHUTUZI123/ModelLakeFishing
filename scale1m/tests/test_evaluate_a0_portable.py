import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scale1m import evaluate_a0_portable as P


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_gold_rejects_fractional_duplicate_and_out_of_range_ids(tmp_path):
    path = tmp_path / "gold.npz"
    for ids in ([0.5, 1], [1, 1], [1, 99]):
        np.savez(path, **{"0": np.array([ids, [1., .5]])})
        with pytest.raises(ValueError):
            P.check_gold(path, 5, 2)
    np.savez(path, **{"1": np.array([[4, 2], [1., .5]])})
    assert P.check_gold(path, 5, 2).tolist() == [1]


def prior_fixture():
    payload = {"edge_model": np.array([0, 2]), "edge_dataset": np.array([0, 0]),
               "edge_acc": np.array([.6, .8]), "root_id": np.array([8, 9, 9]),
               "task_id": np.array([0, 0, 1])}
    meta = {"n_edges": 2, "n_datasets": 3}
    roots = np.array(["train", "test", "test"])
    datasets = np.array(["a", "b", "c"])
    nodes = pd.DataFrame({"node": datasets, "task": [" Classification ", "classification", "TEXT_GEN"]})
    return payload, meta, np.array([1]), roots, datasets, nodes, 4


def test_prior_checks_normalized_tasks_and_entire_query_root_exclusion():
    args = prior_fixture()
    P.check_prior(*args)
    args[0]["edge_dataset"][1] = 2
    with pytest.raises(ValueError, match="held-out query root"):
        P.check_prior(*args)


def test_prior_rejects_wrong_task_codes_and_missing_canonical_node():
    args = prior_fixture()
    args[0]["task_id"][2] = 0
    with pytest.raises(ValueError, match="task IDs"):
        P.check_prior(*args)
    args = list(prior_fixture())
    args[5] = args[5].iloc[:2]
    with pytest.raises(ValueError, match="missing or ambiguous"):
        P.check_prior(*args)


def test_equal_split_retention_and_latency_are_not_pooled():
    per_seed = {}
    for seed, (final, full, latency) in enumerate(((.1, .2, 1.), (.6, .8, 3.), (.9, .9, 8.))):
        per_seed[str(seed)] = {"rows": {
            "G_hnsw1000_task": {"gold@10": final}, "G_full_task": {"gold@10": full},
            "G_exact1000_task": {"gold@10": full}}, "latency_ms": {"end_to_end_p95": latency}}
    summary = P.summarize(per_seed, final_only=False)
    assert summary["overall_retention"]["mean"] == .75
    assert summary["overall_retention"]["mean"] != pytest.approx(1.6 / 1.9)
    assert summary["latency_ms"]["end_to_end_p95"]["mean"] == 4
    replay = P.summarize(per_seed, final_only=True)
    assert "overall_retention" not in replay
    with pytest.raises(ValueError, match="three splits"):
        P.summarize({"0": per_seed["0"]}, final_only=False)


def test_output_rejects_existing_results_without_changing_them(tmp_path):
    sentinel = tmp_path / "result.txt"
    sentinel.write_text("old", encoding="utf-8")
    with pytest.raises(ValueError, match="empty directory"):
        P.empty_output(tmp_path)
    assert sentinel.read_text(encoding="utf-8") == "old"


def reference_fixture(tmp_path):
    index = tmp_path / "hnsw_a0_s0.bin"
    index.write_bytes(b"index fixture")
    inputs = {"a0": {"graph_digest": "graph"}, "files": {
        "z_d_eval.npy": {"sha256": "queries"},
        "EXPORT_MANIFEST.json": {"sha256": "relocated-metadata"}}}
    manifest = {"protocol": "a0", "run_id": "A0_20260912",
        "seeds": {"0": {"index": index.name, "selected_ef": 1500}},
        "artifacts": {index.name: P.file_record(index)},
        "binding": {"seed_inputs": {"0": {"graph_digest": "graph", "files": [
            {"path": "/author/no-longer-existing/queries/z_d_eval.npy", "sha256": "queries"}]}},
            "source_files": [{"role": "frozen_dataset_nodes", "sha256": "canonical"}]}}
    write_json(tmp_path / P.REFERENCE_MANIFEST, manifest)
    return inputs, {"sha256": "canonical"}


def test_reference_binding_ignores_old_paths_but_checks_actual_bytes(tmp_path):
    inputs, nodes = reference_fixture(tmp_path)
    path, record, ef = P.check_reference_index(tmp_path, 0, inputs, nodes)
    assert ef == 1500 and record["bytes"] == len(b"index fixture")
    path.write_bytes(b"different index")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        P.check_reference_index(tmp_path, 0, inputs, nodes)


def test_reference_binding_rejects_index_from_other_export(tmp_path):
    inputs, nodes = reference_fixture(tmp_path)
    inputs["files"]["z_d_eval.npy"]["sha256"] = "other-queries"
    with pytest.raises(ValueError, match="input mismatch"):
        P.check_reference_index(tmp_path, 0, inputs, nodes)


class Prior:
    def __init__(self, n_queries=2):
        self.task_id = np.zeros(n_queries, dtype=np.int64)
        self.by_task = {0: (np.array([1]), np.array([.25]))}

    def values(self, query, ids):
        return np.where(np.asarray(ids) == 1, .25, 0).astype(np.float32)


def test_fusion_preserves_float32_and_label_free_ties(monkeypatch):
    monkeypatch.setattr(P.E, "N_TOTAL", 1024)
    ids = np.arange(1000, dtype=np.int64)[None, :]
    scores = np.ones((1, 1000), dtype=np.float32)
    ties = P.E._tie_ranks(1024)
    expected = sorted(range(1000), key=lambda m: (-float(1. + (.25 if m == 1 else 0)),
                    (m * 11400714819323198485 + 0xD1B54A32D192ED03) % 2**64))[:10]
    candidates = {0: (np.array([1, 999, 998]), np.array([1., .8, .6]))}
    row, arrays = P.A.pool_arrays(ids, scores, np.array([0]), candidates,
                                  np.array(["root"]), Prior(), ties)
    assert arrays["fused"].dtype == np.float32
    assert arrays["top10"][0].tolist() == expected
    assert row["gold@1"] == 1


def test_exact_wrapper_saves_same_tied_full_lake_rank_and_top10(tmp_path, monkeypatch):
    monkeypatch.setattr(P.E, "N_TOTAL", 1024)
    vector = np.random.default_rng(7).normal(size=128).astype(np.float32)
    vector /= np.linalg.norm(vector)
    np.save(tmp_path / "z_m_eval.npy", np.tile(vector, (1024, 1)))
    np.save(tmp_path / "z_d_eval.npy", np.tile(vector, (2, 1)))
    output = tmp_path / "output"
    output.mkdir()
    prior = Prior()
    prior.by_task = {}
    prior.values = lambda query, ids: np.zeros(np.shape(ids), dtype=np.float32)
    ties = P.E._tie_ranks(1024)
    gold = int(np.argmin(ties))
    candidates = {0: (np.array([gold, (gold + 1) % 1024]), np.array([1., .5]))}
    bundle = {"x4": str(tmp_path), "prior": prior, "candidates": candidates,
              "roots": np.array(["test", "train"])}
    args = argparse.Namespace(out=output, device="cpu")
    result = P.exact_seed(bundle, args, ties, 0)
    assert result["rows"]["G_full_task"]["gold@1"] == 1
    with np.load(output / "a0_exact_s0.npz", allow_pickle=False) as raw:
        assert raw["full_top10"][0, 0] == gold
        assert raw["full_counts"][0, 0] == 0
        assert raw["query"].tolist() == [0]


def test_embedding_shape_finite_and_norm_validation(tmp_path):
    path = tmp_path / "query.npy"
    values = np.zeros((2, 128), dtype=np.float32)
    values[:, 0] = 1
    np.save(path, values)
    P.check_embedding(path, 2)
    values[0, 0] = 2
    np.save(path, values)
    with pytest.raises(ValueError, match="unit length"):
        P.check_embedding(path, 2)
    values[0, 0] = np.nan
    np.save(path, values)
    with pytest.raises(ValueError, match="Nonfinite"):
        P.check_embedding(path, 2)


@pytest.fixture
def fresh_export(tmp_path, monkeypatch):
    import torch
    from scale1m import checkpoint as CK
    from scale1m.build_graph_rf import mask_performance_features
    from scale1m.tests.test_prepare_a0_graph import tiny_source
    from scale1m.train_rung import build_config

    monkeypatch.setattr(P.E, "N_TOTAL", 7)
    monkeypatch.setattr(P, "N_DATASETS", 3)
    monkeypatch.setattr(P, "N_PERFORMANCE_EDGES", 4)
    monkeypatch.setitem(P.E.EXPECTED_QUERIES, 0, 1)
    monkeypatch.setitem(P.E.EXPECTED_SIDECAR_EDGES, 0, 3)
    monkeypatch.setitem(P.E.EXPECTED_HELDOUT_DATASETS, 0, 1)
    graph = tmp_path / "graph"
    tiny_source(graph)
    clean = mask_performance_features(np.load(graph / "x_dataset.npy"))
    np.save(graph / "x_dataset.npy", clean)
    gm = P.read_json(graph / "meta.json")
    gm["files"]["x_dataset.npy"] = P.file_record(graph / "x_dataset.npy")["sha256"]
    write_json(graph / "meta.json", gm)
    vocab = tmp_path / "family_vocab.csv"
    vocab.write_text("family,id\nunknown,0\n", encoding="utf-8")
    binding = CK.make_binding(gm["xm0_meta"], graph_path=str(graph), split_seed=0,
                              family_vocab_path=str(vocab))
    cfg = build_config(argparse.Namespace(fanout=True, sparse_M=True, contrast_n_neg=256,
        contrast_max_pos_per_dataset=None, chunked_infer=50000, skip_diagnostics=True,
        batch_size=None, lake_gamma=.5, global_n_datasets=128, num_layers=None))
    checkpoint = tmp_path / "last.pt"
    ck = {"ckpt_version": 1, "epoch": 24, "history": [{"loss": .1} for _ in range(25)],
          "cfg": cfg, "binding": binding}
    torch.save(ck, checkpoint)
    exports = tmp_path / "exports"
    folder = exports / (P.RUN_FMT % 0)
    folder.mkdir(parents=True)
    for kind in ("model", "dataset"):
        shutil.copy2(graph / f"unique_{kind}_id.parquet", folder / f"{kind}_ids.parquet")
    for name, rows in (("z_m", 7), ("z_m_eval", 7), ("z_d", 3), ("z_d_eval", 3)):
        vectors = np.zeros((rows, 128), np.float32)
        vectors[:, 0] = 1
        np.save(folder / f"{name}.npy", vectors)
    np.savez(folder / "gold_cands.npz", **{"2": np.array([[3], [.9]])})
    names = ("z_m.npy", "z_d.npy", "z_m_eval.npy", "z_d_eval.npy",
             "gold_cands.npz", "model_ids.parquet", "dataset_ids.parquet")
    meta = {"checkpoint": str(checkpoint), "checkpoint_sha256": P.file_record(checkpoint)["sha256"],
        "checkpoint_epoch": 24, "graph": str(graph), "graph_digest": binding["graph_sha256"],
        "binding": binding, "split_seed": 0, "n_models": 7, "n_datasets": 3, "n_test_queries": 1,
        "chunk": 50000, "gates": [{"gate": "G-F7d", "ran": True, "ok": True, "n_models": 7,
                                   "max_abs_delta": {"model": 0., "dataset": 0.}}],
        "artifact_hashes": {name: P.file_record(folder / name)["sha256"] for name in names}}
    write_json(folder / "EXPORT_MANIFEST.json", {"stages": {"embed": meta}})
    prior_path = folder / "prior_sidecar_s0.npz"
    np.savez(prior_path, edge_model=np.array([0, 1, 2]), edge_dataset=np.array([0, 0, 1]),
             edge_acc=np.array([.5, .7, .2], np.float32), root_id=np.array([0, 0, 1]),
             task_id=np.array([0, 1, 0]))
    write_json(folder / "prior_sidecar_s0_meta.json", {"n_models": 7, "n_datasets": 3,
        "n_edges": 3, "n_edges_in_graph": 4, "split_seed": 0, "held_out_datasets": 1,
        "graph_digest": binding["graph_sha256"], "sidecar_sha256": P.file_record(prior_path)["sha256"]})
    nodes = pd.DataFrame({"node": ["r/a\tt1", "r/b\tt2", "s/c\tt1"], "task": ["t1", "t2", "t1"],
                          "gold_eligible": [True] * 3, "primary_direction": ["higher"] * 3,
                          "is_placeholder": [False] * 3, "is_rl": [False] * 3})
    node_path = tmp_path / "nodes.parquet"
    nodes.to_parquet(node_path)
    args = argparse.Namespace(exports=exports, dataset_nodes=node_path, final_only=False, rebuilt_graph=graph)
    return args, folder, meta, nodes


def test_fresh_graph_and_export_validate_without_historical_a0_stamp(fresh_export):
    args, folder, meta, nodes = fresh_export
    bundle, inputs = P.validate_seed(args, 0, P.E._tie_ranks(7), None, None, nodes)
    assert "a0" not in meta and "a0" not in inputs
    assert inputs["producer"]["kind"] == "fresh_rebuilt"
    assert inputs["producer"]["graph_digest"] == meta["graph_digest"]
    assert inputs["eligible_queries"] == 1
    bundle["prior"].payload.close()


def test_fresh_graph_does_not_weaken_replay_provenance(fresh_export):
    args, folder, meta, nodes = fresh_export
    args.final_only = True
    with pytest.raises(ValueError, match="cannot be used for saved-index replay"):
        P.validate_seed(args, 0, P.E._tie_ranks(7), None, None, nodes)
    args.rebuilt_graph = None
    with pytest.raises(ValueError, match="not the A0 split"):
        P.validate_seed(args, 0, P.E._tie_ranks(7), None, None, nodes)


def test_fresh_graph_rejects_rehashed_performance_feature_leak(fresh_export):
    args, folder, meta, nodes = fresh_export
    path = args.rebuilt_graph / "x_dataset.npy"
    values = np.load(path)
    values[0, 448] = .7
    np.save(path, values)
    gm = P.read_json(args.rebuilt_graph / "meta.json")
    gm["files"][path.name] = P.file_record(path)["sha256"]
    write_json(args.rebuilt_graph / "meta.json", gm)
    with pytest.raises(ValueError, match="performance-derived"):
        P.validate_rebuilt_graph(args.rebuilt_graph)


def test_fresh_checkpoint_rejects_rehashed_wrong_recipe(fresh_export):
    import torch
    args, folder, meta, nodes = fresh_export
    ck = torch.load(meta["checkpoint"], weights_only=False)
    ck["cfg"]["lake_gamma"] = .1
    torch.save(ck, meta["checkpoint"])
    meta["checkpoint_sha256"] = P.file_record(meta["checkpoint"])["sha256"]
    write_json(folder / "EXPORT_MANIFEST.json", {"stages": {"embed": meta}})
    with pytest.raises(ValueError, match="configuration differs"):
        P.validate_seed(args, 0, P.E._tie_ranks(7), None, None, nodes)
