"""Small producer/consumer binding fixtures; no model export or real graph load."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scale1m import export_rf as X
from scale1m import a0_graph_validation as V
from stage3HNSW import build_prior_sidecar as P
from ModelLakeFishing.scale1m import a0_evaluation as A


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def producer(tmp_path, monkeypatch):
    graph, run, export = [tmp_path / name for name in ("graph", "run", "export")]
    for path in (graph, run, export):
        path.mkdir()
    checkpoint = run / "last.pt"
    checkpoint.write_bytes(b"new formal checkpoint fixture")
    digest = "a" * 64
    envelope = {"protocol": "a0", "run_id": "A0_20260912", "seed": 0,
                "graph_digest": digest, "checkpoint_sha256": X.CK.sha256_of(checkpoint)}
    ck = {"binding": {"split_seed": 0, "graph_sha256": digest},
          "epoch": 24, "history": [{"epoch": i} for i in range(25)]}
    write_json(run / "A0_RUN_RECORDS.json", {"a0": envelope, "history": ck["history"],
                                            "status": "complete", "formal_run_name": run.name})
    write_json(graph / "A0_FEATURE_REPAIR.json", {"protocol": "a0"})
    write_json(graph / "meta.json", {"a0": {"protocol": "a0"},
                                    "files": {"A0_FEATURE_REPAIR.json": "repair-hash"}})
    for name in ("z_m.npy", "z_d.npy", "z_m_eval.npy", "z_d_eval.npy", "gold_cands.npz"):
        (export / name).write_bytes(("fresh fixture " + name).encode())
    pd.DataFrame({"mappedID": [0, 1], "model": ["m0", "m1"]}).to_parquet(export / "model_ids.parquet")
    pd.DataFrame({"mappedID": [0, 1], "dataset": ["d0", "d1"]}).to_parquet(export / "dataset_ids.parquet")
    hashes = {path.name: X.CK.sha256_of(path) for path in export.iterdir()}
    embed = {"a0": envelope, "checkpoint": str(checkpoint), "artifact_hashes": hashes}
    write_json(export / "EXPORT_MANIFEST.json", {"stages": {"embed": embed}})
    monkeypatch.setattr(V, "verify_a0_graph", lambda path: {"graph_sha256": digest})
    monkeypatch.setattr(V, "PREPARED_GRAPH_DIGEST", digest)
    return argparse.Namespace(graph=graph, run=run, export=export, checkpoint=checkpoint,
                              digest=digest, envelope=envelope, ck=ck, embed=embed,
                              args=argparse.Namespace(graph_store=str(graph), export=str(export), split_seed=0))


def test_export_context_matches_consumer_and_rejects_preexisting_output(producer, tmp_path):
    p = producer
    fresh = tmp_path / "fresh"
    context = X.a0_export_context(p.run, p.checkpoint, p.ck, p.graph, p.digest, fresh)
    A.verify_producer_envelope({"a0": context}, p.envelope, "export fixture")
    with pytest.raises(FileExistsError):
        X.a0_export_context(p.run, p.checkpoint, p.ck, p.graph, p.digest, p.export)


@pytest.mark.parametrize("field,bad", [("seed", 1), ("graph_digest", "old"),
                                       ("checkpoint_sha256", "old"), ("run_id", "old")])
def test_export_rejects_wrong_producer_identity(producer, tmp_path, field, bad):
    p = producer
    record_path = p.run / "A0_RUN_RECORDS.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["a0"][field] = bad
    write_json(record_path, record)
    with pytest.raises(X.CK.IncompatibleCheckpoint):
        X.a0_export_context(p.run, p.checkpoint, p.ck, p.graph, p.digest, tmp_path / "fresh")


@pytest.mark.parametrize("field,bad", [("status", "gate_failed"), ("formal_run_name", "other_run"),
                                       ("history", [{"epoch": 24}])])
def test_export_rejects_failed_or_mismatched_native_run_record(producer, tmp_path, field, bad):
    p = producer
    path = p.run / "A0_RUN_RECORDS.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record[field] = bad
    write_json(path, record)
    with pytest.raises(X.CK.IncompatibleCheckpoint):
        X.a0_export_context(p.run, p.checkpoint, p.ck, p.graph, p.digest, tmp_path / "fresh")


def test_export_rejects_incomplete_training(producer, tmp_path):
    p = producer
    p.ck["history"].pop()
    with pytest.raises(X.CK.IncompatibleCheckpoint):
        X.a0_export_context(p.run, p.checkpoint, p.ck, p.graph, p.digest, tmp_path / "fresh")


def test_prior_context_matches_consumer(producer):
    context = P.a0_prior_context(producer.args)
    A.verify_producer_envelope({"a0": context}, producer.envelope, "prior fixture")


@pytest.mark.parametrize("field,bad", [("seed", 1), ("graph_digest", "old"),
                                       ("checkpoint_sha256", "old"), ("run_id", "old")])
def test_prior_rejects_wrong_export_identity(producer, field, bad):
    p = producer
    p.embed["a0"] = {**p.envelope, field: bad}
    write_json(p.export / "EXPORT_MANIFEST.json", {"stages": {"embed": p.embed}})
    with pytest.raises(ValueError):
        P.a0_prior_context(p.args)


@pytest.mark.parametrize("name", ["model_ids.parquet", "dataset_ids.parquet"])
def test_prior_rejects_changed_rowmap_bytes(producer, name):
    (producer.export / name).write_bytes(b"historical row map substitution")
    with pytest.raises(ValueError, match="row map"):
        P.a0_prior_context(producer.args)


def test_prior_rejects_changed_checkpoint_bytes(producer):
    producer.checkpoint.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError):
        P.a0_prior_context(producer.args)


def test_prior_rejects_stripped_export_marker(producer):
    del producer.embed["a0"]
    write_json(producer.export / "EXPORT_MANIFEST.json", {"stages": {"embed": producer.embed}})
    with pytest.raises(ValueError):
        P.a0_prior_context(producer.args)


def test_stripped_graph_marker_still_invokes_strong_validator(producer, monkeypatch):
    p = producer
    write_json(p.graph / "meta.json", {"files": {"A0_FEATURE_REPAIR.json": "repair-hash"}})
    def reject(path):
        raise ValueError("graph A0 marker stripped")
    monkeypatch.setattr(V, "verify_a0_graph", reject)
    monkeypatch.setattr(X.CK, "load", lambda path: p.ck)
    with pytest.raises(ValueError, match="marker stripped"):
        P.a0_prior_context(p.args)
    with pytest.raises(ValueError, match="marker stripped"):
        X.bind_or_die(p.checkpoint, p.graph)


def test_stripped_graph_marker_and_report_do_not_downgrade_to_legacy(producer, monkeypatch, tmp_path):
    p = producer
    write_json(p.graph / "meta.json", {"files": {"A0_FEATURE_REPAIR.json": "repair-hash"}})
    (p.graph / "A0_FEATURE_REPAIR.json").unlink()
    def reject(path):
        raise ValueError("graph A0 provenance stripped")
    monkeypatch.setattr(V, "verify_a0_graph", reject)
    monkeypatch.setattr(X.CK, "load", lambda path: p.ck)
    with pytest.raises(ValueError, match="provenance stripped"):
        P.a0_prior_context(p.args)
    with pytest.raises(ValueError, match="provenance stripped"):
        X.bind_or_die(p.checkpoint, p.graph)
    with pytest.raises(X.CK.IncompatibleCheckpoint, match="provenance"):
        X.a0_export_context(p.run, p.checkpoint, p.ck, p.graph, p.digest, tmp_path / "fresh")


def test_graph_without_any_marker_cannot_downgrade_an_a0_producer(producer, monkeypatch, tmp_path):
    p = producer
    write_json(p.graph / "meta.json", {"files": {}})
    (p.graph / "A0_FEATURE_REPAIR.json").unlink()
    def reject(path):
        raise ValueError("A0 export identity requires A0 graph")
    monkeypatch.setattr(V, "verify_a0_graph", reject)
    with pytest.raises(ValueError, match="requires A0 graph"):
        P.a0_prior_context(p.args)
    with pytest.raises(X.CK.IncompatibleCheckpoint, match="provenance"):
        X.a0_export_context(p.run, p.checkpoint, p.ck, p.graph, p.digest, tmp_path / "fresh")


def test_prior_meta_writer_is_consumable_and_rejects_overwrite(producer, monkeypatch):
    p = producer
    umi = pd.read_parquet(p.export / "model_ids.parquet")
    udi = pd.read_parquet(p.export / "dataset_ids.parquet")
    edges = np.array([[0, 1], [0, 1]], dtype=np.int64)
    values = np.array([.25, .75], dtype=np.float32)
    data = {P.TRAINED_ON: argparse.Namespace(edge_index=torch.tensor(edges))}
    monkeypatch.setattr(P, "load_graph", lambda args: (data, udi, umi))
    monkeypatch.setattr(P, "visible_edges", lambda *args: (edges, values, 0))
    monkeypatch.setattr(P, "root_ids", lambda frame: (np.array([0, 1]), 2, "fixture"))
    monkeypatch.setattr(P, "task_ids", lambda *args: (np.array([0, 0]), "fixture", 1))
    argv = ["--graph-store", str(p.graph), "--export", str(p.export), "--split-seed", "0"]
    assert P.main(argv) == 0
    meta = json.loads((p.export / "prior_sidecar_s0_meta.json").read_text())
    A.verify_producer_envelope(meta, p.envelope, "sidecar")
    assert meta["graph_digest"] == p.digest
    assert meta["sidecar_sha256"] == X.CK.sha256_of(p.export / "prior_sidecar_s0.npz")
    segment = meta["prior_build_segments"][0]
    assert isinstance(segment["start_ns"], int) and segment["end_ns"] >= segment["start_ns"]
    with np.load(p.export / "prior_sidecar_s0.npz") as arrays:
        assert np.array_equal(arrays["edge_model"], edges[0])
        assert np.array_equal(arrays["edge_acc"], values)
    with pytest.raises(FileExistsError):
        P.main(argv)
