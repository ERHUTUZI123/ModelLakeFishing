import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scale1m import checkpoint as CK
from scale1m import a0_graph_validation as V
from scale1m import a0_recompute_checks as C
from scale1m.prepare_a0_graph import prepare_graph
from scale1m.tests.test_prepare_a0_graph import tiny_source


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def ref(path, role=None):
    value = {"path": str(path), "sha256": C.sha256(path)}
    if role:
        value["role"] = role
    return value


@pytest.fixture
def raw_fixture(tmp_path, monkeypatch):
    source, graph, export, run = [tmp_path / name for name in ("source", "graph", "export", "run")]
    digest = tiny_source(source)
    prepare_graph(source, graph, expected_source_digest=digest, expected_shape=(7, 3))
    original_verify = V.verify_a0_graph
    monkeypatch.setattr(V, "verify_a0_graph", lambda path: original_verify(path, require_frozen_source=False))
    export.mkdir()
    (run / "ckpt").mkdir(parents=True)
    gm = C.read_json(graph / "meta.json")
    binding = CK.make_binding(gm["xm0_meta"], graph_path=str(graph), split_seed=0)
    cfg = {"fixture_config": 1}
    checkpoint = run / "ckpt" / "last.pt"
    ck = {"ckpt_version": CK.CKPT_VERSION, "binding": binding, "cfg": cfg,
          "epoch": 24, "history": [{"epoch": i} for i in range(25)]}
    torch.save(ck, checkpoint)
    envelope = {"protocol": "a0", "run_id": "A0_20260912", "seed": 0,
                "graph_digest": CK.graph_digest(str(graph)), "checkpoint_sha256": C.sha256(checkpoint)}
    write_json(run / "A0_RUN_RECORDS.json", {"a0": envelope, "status": "complete", "history": ck["history"]})
    for kind in ("model", "dataset"):
        pd.read_parquet(graph / f"unique_{kind}_id.parquet").to_parquet(export / f"{kind}_ids.parquet", index=False)
    for name, rows in (("z_m.npy", 7), ("z_m_eval.npy", 7), ("z_d.npy", 3), ("z_d_eval.npy", 3)):
        vectors = np.zeros((rows, 128), np.float32)
        vectors[:, 0] = 1
        np.save(export / name, vectors)
    embed = {"graph": str(graph), "checkpoint": str(checkpoint), "a0": envelope,
             "artifact_hashes": {p.name: C.sha256(p) for p in export.iterdir()}}
    write_json(export / "EXPORT_MANIFEST.json", {"stages": {"embed": embed}})
    sidecar = export / "prior_sidecar_s0.npz"
    pairs = np.array([[0, 1, 2], [0, 0, 1]], np.int64)
    np.savez(sidecar, edge_model=pairs[0], edge_dataset=pairs[1],
             edge_acc=np.array([.5, .7, .2], np.float32), root_id=np.array([0, 0, 1]), task_id=np.array([0, 1, 0]))
    sidecar_meta = {"a0": envelope, "sidecar_sha256": C.sha256(sidecar), "graph_digest": envelope["graph_digest"],
                    "split_seed": 0, "n_models": 7, "n_datasets": 3, "n_edges": 3}
    write_json(export / "prior_sidecar_s0_meta.json", sidecar_meta)
    udi = pd.read_parquet(graph / "unique_dataset_id.parquet")
    nodes = tmp_path / "nodes.parquet"
    pd.DataFrame({"node": udi.dataset, "task": ["t1", "t2", "t1"]}).to_parquet(nodes)
    identity = tmp_path / "identity.jsonl"
    identity.write_text(json.dumps({"seed": 0, "query_mappedID": 2, "node": udi.dataset.iloc[2], "root": "s"}) + "\n")
    input_audit = tmp_path / "input_audit.json"
    write_json(input_audit, {"files": [ref(source / "x_dataset.npy")], "splits": [{"seed": 0,
        "prior_visible_pairs_ordered": {"sha256_c_order_bytes": C.array_sha(pairs)}}]})
    raw = tmp_path / "raw_exact.npz"
    prior = np.zeros((1, 7), np.float32)
    prior[0, 0] = .5
    prior[0, 1] = (float(np.float32(.7)) + 2.5) / 6
    np.savez(raw, query=np.array([2]), model=np.arange(7)[None], prior=prior)
    manifest = {"run_id": "A0_20260912", "binding": {"source_files": [ref(nodes, "frozen_dataset_nodes")],
        "seed_inputs": {"0": {"graph_digest": envelope["graph_digest"], "export_dir": str(export), "sidecar_dir": str(export)}}},
        "seeds": {"0": {"exact": raw.name}}, "artifacts": {raw.name: ref(raw)}}
    protocol = {"input_audit_binding": {"input_audit": ref(input_audit), "query_identity": ref(identity)},
                "required_config_audit": {"resolved_config": cfg}}
    return manifest, protocol, {"graph": graph, "source": source, "export": export, "raw": raw,
                                "checkpoint": checkpoint, "sidecar": sidecar}


def test_raw_correctness_bridge_recomputes_nine_present_seed_metrics(raw_fixture):
    manifest, protocol, paths = raw_fixture
    got = C.recompute_checks(manifest, protocol)
    assert len(got) == 9
    assert got["task_prior.visible_prior_edges.0"]["value"] == 3
    assert got["task_prior.sidecar_rebuilt_and_hash_bound.0"]["value"]["raw_prior_values_independently_checked"] == 7
    for key, entry in got.items():
        assert entry["source"]
        if key != "task_prior.visible_prior_edges.0":
            assert entry["value"]["passed"] is True
    assert not any(key.endswith(".1") or key.endswith(".2") for key in got)


@pytest.mark.parametrize("tamper", ["embedding", "rowmap", "checkpoint_dependency", "raw_prior", "raw_prior_nan", "query_root"])
def test_raw_correctness_bridge_rejects_precise_corruptions(raw_fixture, tamper):
    manifest, protocol, paths = raw_fixture
    if tamper == "embedding":
        a = np.load(paths["export"] / "z_m.npy")
        a[0, 0] = .5
        np.save(paths["export"] / "z_m.npy", a)
    elif tamper == "rowmap":
        path = paths["export"] / "model_ids.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "model"] = "unrelated-model"
        frame.to_parquet(path)
    elif tamper == "checkpoint_dependency":
        ck = CK.load(str(paths["checkpoint"]))
        ck["binding"]["encoder_name"] = "different-encoder"
        torch.save(ck, paths["checkpoint"])
    elif tamper.startswith("raw_prior"):
        with np.load(paths["raw"]) as z:
            arrays = {name: z[name] for name in z.files}
        arrays["prior"][0, 0] = np.nan if tamper.endswith("nan") else .123
        np.savez(paths["raw"], **arrays)
    else:
        identity = Path(protocol["input_audit_binding"]["query_identity"]["path"])
        identity.write_text(json.dumps({"seed": 0, "query_mappedID": 0, "node": "r/a\tt1", "root": "r"}) + "\n")
        protocol["input_audit_binding"]["query_identity"] = ref(identity)
    with pytest.raises(ValueError):
        C.recompute_checks(manifest, protocol)


def test_embedding_audit_rejects_nan_even_when_norm_max_reduction_is_nan(tmp_path):
    a = np.zeros((2, 128), np.float32)
    a[:, 0] = 1
    a[0, 1] = np.nan
    np.save(tmp_path / "bad.npy", a)
    with pytest.raises(ValueError):
        C.embedding_check(tmp_path / "bad.npy", 2)


def test_retained_negative_zero_change_fails_even_with_consistent_rehashed_graph(tmp_path, monkeypatch):
    source, graph = tmp_path / "source", tmp_path / "graph"
    tiny_source(source)
    original = np.load(source / "x_dataset.npy")
    original[0, 0] = np.float32(0.0)
    np.save(source / "x_dataset.npy", original)
    source_meta = C.read_json(source / "meta.json")
    source_meta["files"]["x_dataset.npy"] = C.sha256(source / "x_dataset.npy")
    write_json(source / "meta.json", source_meta)
    prepare_graph(source, graph, expected_source_digest=CK.graph_digest(str(source)), expected_shape=(7, 3))
    current = np.load(graph / "x_dataset.npy")
    current[0, 0] = np.float32(-0.0)
    assert current[0, 0] == original[0, 0]
    assert current[0, 0].view(np.uint32) != original[0, 0].view(np.uint32)
    np.save(graph / "x_dataset.npy", current)
    meta = C.read_json(graph / "meta.json")
    repair = C.read_json(graph / "A0_FEATURE_REPAIR.json")
    meta["files"]["x_dataset.npy"] = C.sha256(graph / "x_dataset.npy")
    repair["new_x_dataset_sha256"] = meta["files"]["x_dataset.npy"]
    write_json(graph / "A0_FEATURE_REPAIR.json", repair)
    meta["files"]["A0_FEATURE_REPAIR.json"] = C.sha256(graph / "A0_FEATURE_REPAIR.json")
    write_json(graph / "meta.json", meta)
    original_verify = V.verify_a0_graph
    assert original_verify(graph, require_frozen_source=False)["status"] == "PASS"
    monkeypatch.setattr(V, "verify_a0_graph", lambda path: original_verify(path, require_frozen_source=False))
    with pytest.raises(ValueError, match="retained feature bytes"):
        C.graph_checks(graph, {"files": [ref(source / "x_dataset.npy")]})
