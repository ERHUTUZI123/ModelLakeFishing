"""Actual exact/HNSW scoring of a small new lake, without historical A0 data."""
import argparse
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData

from scale1m import checkpoint as CK
from scale1m import evaluate_live as L
from scale1m.build_graph_rf import dataset_stats
from scale1m.graph_store import save_sharded
from scale1m.train_rung import build_config


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def live_lake(tmp_path):
    """Synthetic producer records are test fixtures, not a claimed training run."""
    nm, nd = 1007, 15
    rng = np.random.default_rng(31)
    nodes = pd.DataFrame({"node": [f"r{i // 2}/dataset{i}\tt{i % 2}" for i in range(nd)],
                          "dataset": [f"r{i // 2}/dataset{i}" for i in range(nd)],
                          "task": [f"t{i % 2}" for i in range(nd)],
                          "gold_eligible": True, "primary_direction": "higher",
                          "is_placeholder": False, "is_rl": False})
    stats, roots = dataset_stats(nodes, None)
    data = HeteroData()
    data["model"].x = torch.from_numpy(rng.normal(size=(nm, 448)).astype(np.float32))
    data["model"].node_id = torch.arange(nm)
    data["dataset"].x = torch.from_numpy(np.concatenate([
        rng.normal(size=(nd, 448)).astype(np.float32), stats], axis=1))
    data["dataset"].node_id = torch.arange(nd)
    forward, reverse = ("model", "trained_on", "dataset"), ("dataset", "rev_trained_on", "model")
    data[forward].edge_index = torch.from_numpy(np.stack([np.arange(nd * 4), np.repeat(np.arange(nd), 4)]))
    data[forward].edge_attr = torch.tensor(np.tile([.1, .3, .6, .9], nd), dtype=torch.float32)
    data[reverse].edge_index = data[forward].edge_index.flip(0)
    data[reverse].edge_attr = data[forward].edge_attr.clone()
    graph_path = tmp_path / "graph"
    graph_meta = save_sharded({"data": data, "xm0_meta": {"num_families": 1, "num_size_buckets": 1},
        "xd0_meta": {"view_dims": {"e_name": 64, "e_card": 384, "e_stats": 10}},
        "unique_model_id": pd.DataFrame({"model": [f"m/{i}" for i in range(nm)], "mappedID": range(nm)}),
        "unique_dataset_id": pd.DataFrame({"dataset": nodes.node, "root": roots, "mappedID": range(nd)})},
        str(graph_path))
    graph = L.validate_graph(graph_path)
    vocab = tmp_path / "family_vocab.csv"
    vocab.write_text("family,id\nunknown,0\n", encoding="utf-8")
    node_path = tmp_path / "nodes.parquet"
    nodes.to_parquet(node_path, index=False)
    cfg = build_config(argparse.Namespace(fanout=True, sparse_M=True, contrast_n_neg=256,
        contrast_max_pos_per_dataset=None, chunked_infer=50000, skip_diagnostics=True,
        batch_size=None, lake_gamma=.5, global_n_datasets=128, num_layers=None))
    exports = tmp_path / "exports"
    zm = rng.normal(size=(nm, 128)).astype(np.float32)
    zm /= np.linalg.norm(zm, axis=1, keepdims=True)
    zd = zm[np.arange(nd) * 4 + 3].copy()
    for seed in range(3):
        folder = exports / (L.RUN_FMT % seed)
        folder.mkdir(parents=True)
        binding = CK.make_binding(graph_meta["xm0_meta"], graph_path=str(graph_path), split_seed=seed,
                                  family_vocab_path=str(vocab))
        checkpoint = tmp_path / f"last_s{seed}.pt"
        torch.save({"ckpt_version": 1, "epoch": 24, "history": [{"loss": .1} for _ in range(25)],
                    "cfg": cfg, "binding": binding}, checkpoint)
        for kind in ("model", "dataset"):
            shutil.copy2(graph_path / f"unique_{kind}_id.parquet", folder / f"{kind}_ids.parquet")
        np.save(folder / "z_m_eval.npy", zm)
        np.save(folder / "z_d_eval.npy", zd)
        split = L.expected_split(graph, seed)
        np.savez(folder / "gold_cands.npz", **{str(q): np.stack([ids, values])
            for q, (ids, values) in split["candidates"].items()})
        names = ("z_m_eval.npy", "z_d_eval.npy", "gold_cands.npz", "model_ids.parquet", "dataset_ids.parquet")
        meta = {"checkpoint": str(checkpoint), "checkpoint_sha256": CK.sha256_of(checkpoint),
            "checkpoint_epoch": 24, "graph": str(graph_path), "graph_digest": graph["graph_digest"],
            "binding": binding, "split_seed": seed, "n_models": nm, "n_datasets": nd,
            "rung": "live", "expect_n": nm, "n_test_queries": len(split["candidates"]),
            "chunk": 50000, "gates": [{"gate": "G-F7d", "ran": True, "ok": True, "n_models": nm,
                                        "max_abs_delta": {"model": 0., "dataset": 0.}}],
            "artifact_hashes": {name: CK.sha256_of(folder / name) for name in names}}
        write_json(folder / "EXPORT_MANIFEST.json", {"stages": {"embed": meta}})
        sidecar = folder / f"prior_sidecar_s{seed}.npz"
        root_codes = {root: i for i, root in enumerate(sorted(set(roots)))}
        np.savez(sidecar, edge_model=split["edge"][0], edge_dataset=split["edge"][1],
            edge_acc=split["weights"], root_id=np.array([root_codes[r] for r in roots]),
            task_id=np.arange(nd) % 2)
        write_json(folder / f"prior_sidecar_s{seed}_meta.json", {"n_models": nm, "n_datasets": nd,
            "n_edges": split["edge"].shape[1], "n_edges_in_graph": 4 * nd, "split_seed": seed,
            "held_out_datasets": len(split["heldout"]), "graph_digest": graph["graph_digest"],
            "sidecar_sha256": CK.sha256_of(sidecar)})
    args = argparse.Namespace(exports=exports, graph=graph_path, dataset_nodes=node_path,
                              out=tmp_path / "evaluation", device="cpu")
    return args, graph, nodes


def test_real_exact_hnsw_three_dynamic_splits_without_a0_state(live_lake):
    args, graph, nodes = live_lake
    report = L.evaluate(args)
    assert report["status"] == "complete" and report["protocol"] == "live-hf"
    assert report["n_models"] == 1007 and report["n_datasets"] == 15
    assert report["n_performance_edges"] == 60 and report["all_ann_calibration_passed"]
    assert L.E.N_TOTAL == 3_016_439  # Dynamic run never changes historical globals.
    assert 0 < report["summary"]["quality"]["G_hnsw1000_task"]["gold@10"]["mean"] <= 1.
    assert report["summary"]["overall_retention"]["mean"] == 1.
    manifest = L.P.read_json(args.out / L.MANIFEST)
    assert manifest["results"]["sha256"] == CK.sha256_of(args.out / L.RESULTS)
    for seed in range(3):
        row = report["per_seed"][str(seed)]
        assert row["rows"]["G_full_task"]["N"] == 1007
        assert row["rows"]["G_hnsw1000_task"]["candidate_universe_N"] == 1007
        assert row["rows"]["G_hnsw1000_task"]["n_queries"] != L.E.EXPECTED_QUERIES[seed]
        assert row["latency_ms"]["end_to_end_p50"] > 0
        with np.load(args.out / f"live_hnsw_s{seed}.npz") as arrays:
            assert arrays["n_models"].item() == 1007
            assert arrays["model"].shape[1] == 1000 and arrays["top10"].shape[1] == 10
            bundle, _ = L.validate_seed(args, seed, graph, nodes)
            zm = np.load(bundle["folder"] / "z_m_eval.npy")
            zd = np.load(bundle["folder"] / "z_d_eval.npy")
            for i, query in enumerate(arrays["query"]):
                dense = zm @ zd[query]
                fused = (dense + 1) * .5 + bundle["prior"].values(query, np.arange(1007))
                expected = np.lexsort((L.E._tie_ranks(1007), -fused))[:10]
                assert np.array_equal(arrays["top10"][i], expected)
            bundle["prior"].payload.close()
    with pytest.raises(ValueError, match="empty directory"):
        L.evaluate(args)


def test_gold_tampering_rejected_even_after_producer_hash_update(live_lake):
    args, graph, nodes = live_lake
    folder = args.exports / (L.RUN_FMT % 0)
    path = folder / "gold_cands.npz"
    with np.load(path) as z:
        arrays = {key: z[key] for key in z.files}
    next(iter(arrays.values()))[1, 0] += .05
    np.savez(path, **arrays)
    manifest = L.P.read_json(folder / "EXPORT_MANIFEST.json")
    manifest["stages"]["embed"]["artifact_hashes"][path.name] = CK.sha256_of(path)
    write_json(folder / "EXPORT_MANIFEST.json", manifest)
    with pytest.raises(ValueError, match="gold labels differ"):
        L.validate_seed(args, 0, graph, nodes)


def test_prior_missing_visible_evidence_rejected_even_when_hash_and_counts_match(live_lake):
    args, graph, nodes = live_lake
    folder = args.exports / (L.RUN_FMT % 0)
    path = folder / "prior_sidecar_s0.npz"
    with np.load(path) as z:
        arrays = {key: z[key] for key in z.files}
    for key in ("edge_model", "edge_dataset", "edge_acc"):
        arrays[key] = arrays[key][:-1]
    np.savez(path, **arrays)
    meta_path = folder / "prior_sidecar_s0_meta.json"
    meta = L.P.read_json(meta_path)
    meta.update(n_edges=meta["n_edges"] - 1, sidecar_sha256=CK.sha256_of(path))
    write_json(meta_path, meta)
    with pytest.raises(ValueError, match="exactly the train\\+validation"):
        L.validate_seed(args, 0, graph, nodes)


def test_ineligible_sibling_still_cannot_enter_prior(live_lake):
    args, graph, nodes = live_lake
    split = L.expected_split(graph, 0)
    nodes.loc[nodes.index.isin(split["heldout"]), "gold_eligible"] = False
    folder = args.exports / (L.RUN_FMT % 0)
    path = folder / "prior_sidecar_s0.npz"
    with np.load(path) as z:
        arrays = {key: z[key] for key in z.files}
    arrays["edge_dataset"][0] = split["heldout"][0]
    np.savez(path, **arrays)
    meta = L.P.read_json(folder / "prior_sidecar_s0_meta.json")
    with pytest.raises(ValueError, match="held-out query root"):
        L.LiveTaskPrior(path, meta, graph, split, nodes)


def test_rehashed_performance_features_still_rejected(live_lake):
    args, _, _ = live_lake
    path = args.graph / "x_dataset.npy"
    x = np.load(path)
    x[0, 448] = .8
    np.save(path, x)
    meta = L.P.read_json(args.graph / "meta.json")
    meta["files"][path.name] = CK.sha256_of(path)
    write_json(args.graph / "meta.json", meta)
    with pytest.raises(ValueError, match="Performance-derived"):
        L.validate_graph(args.graph)


def test_live_export_rejects_historical_marker_and_wrong_dimensions(live_lake):
    args, graph, nodes = live_lake
    path = args.exports / (L.RUN_FMT % 0) / "EXPORT_MANIFEST.json"
    manifest = L.P.read_json(path)
    manifest["stages"]["embed"]["a0"] = {"protocol": "a0"}
    write_json(path, manifest)
    with pytest.raises(ValueError, match="historical A0"):
        L.validate_seed(args, 0, graph, nodes)
    del manifest["stages"]["embed"]["a0"]
    manifest["stages"]["embed"]["expect_n"] = L.E.N_TOTAL
    write_json(path, manifest)
    with pytest.raises(ValueError, match="live-lake run"):
        L.validate_seed(args, 0, graph, nodes)


def test_no_eligible_queries_is_explicit_failure(live_lake):
    args, graph, nodes = live_lake
    nodes["gold_eligible"] = False
    nodes.to_parquet(args.dataset_nodes, index=False)
    with pytest.raises(ValueError, match="no eligible queries"):
        L.validate_seed(args, 0, graph, nodes)


def test_live_exact_float32_ties_are_label_free(tmp_path):
    class EmptyPrior:
        by_task = {}
        task_id = np.array([0])

        def values(self, query, ids):
            return np.zeros(len(ids), np.float32)

    nm = 1007
    zm = np.zeros((nm, 128), np.float32)
    zm[:, 0] = 1.
    np.save(tmp_path / "z_m_eval.npy", zm)
    np.save(tmp_path / "z_d_eval.npy", zm[:1])
    ties = L.E._tie_ranks(nm)
    candidates = {0: (np.array([0, 1, 2]), np.array([.9, .6, .1]))}
    result = L.E.evaluate_exact_seed(str(tmp_path), EmptyPrior(), candidates, np.array(["root"]),
        ties, "cpu", model_chunk=401, protocol="live", n_universe=nm)
    order = np.argsort(ties)
    assert np.array_equal(result["exact_pool_ids"][0], order[:1000])
    assert np.array_equal(result["full_top10"][0], order[:10])
    assert result["dense_counts"][0, 0] == ties[0]
    assert result["full_counts"][0, 0] == ties[0]
