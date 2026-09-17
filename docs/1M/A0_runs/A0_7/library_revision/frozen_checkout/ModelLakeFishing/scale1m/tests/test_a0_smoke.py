"""Smoke isolation/evaluation guards using tiny tensors and mocked training only."""
import argparse
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData

from scale1m import checkpoint as CK
from scale1m import train_rung as driver
from ModelLakeFishing.stage2TrainGraphSAGE import ablation
from ModelLakeFishing.stage2TrainGraphSAGE import d0_splits, graph_surgery


def config_args(smoke=False):
    return argparse.Namespace(
        fanout=True, sparse_M=True, contrast_n_neg=256,
        contrast_max_pos_per_dataset=None, chunked_infer=50000,
        skip_diagnostics=True, batch_size=None, lake_gamma=0.5,
        global_n_datasets=128, num_layers=None, smoke_only=smoke,
    )


def tiny_pair():
    model, scorer = torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)
    opt = torch.optim.Adam(list(model.parameters()) + list(scorer.parameters()), lr=0.01)
    # Only a 2-element fixture update: no graph encoder or production trainer.
    (model(torch.ones(1, 2)).sum() + scorer(torch.ones(1, 2)).sum()).backward()
    opt.step()
    return model, scorer, opt


def test_smoke_flag_does_not_enter_or_change_the_40_key_recipe():
    formal = driver.build_config(config_args(False))
    smoke = driver.build_config(config_args(True))
    assert smoke == formal
    assert len(formal) == 40
    assert "smoke_only" not in smoke


@pytest.mark.parametrize("epochs,resume,existing", [(2, None, False), (1, "old.pt", False), (1, None, True)])
def test_smoke_requires_a_fresh_single_epoch_output(tmp_path, epochs, resume, existing):
    out = tmp_path / "smoke"
    if existing:
        out.mkdir()
    args = argparse.Namespace(smoke_only=True, epochs=epochs, resume=resume, out=str(out))
    with pytest.raises(ValueError):
        driver.validate_run_mode(args)


def test_formal_entry_rejects_smoke_directory_before_loading_graph(tmp_path):
    (tmp_path / driver.SMOKE_MARKER).write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        driver.main(["--graph", "not-loaded.pt", "--out", str(tmp_path)])
    assert exc.value.code == 2


def test_smoke_single_epoch_gate_checks_optimization_not_loss_descent():
    model, scorer, opt = tiny_pair()
    hist = [{"total": 2.0, "rank": 0.5, "contrast": 0.5, "global": 1.0}]
    gate = driver.smoke_gate(hist, model, scorer, opt)
    assert gate["passed"]
    assert gate["loss_descended"] is None
    assert gate["loss_descent_required"] is False
    assert gate["optimizer_steps_max"] == 1
    # The existing formal acceptance rule is unchanged for a one-epoch history.
    assert not driver.mechanism_gate(hist, model)["passed"]


@pytest.mark.parametrize("failure", ["loss", "parameter", "gradient", "optimizer", "empty_history", "no_update"])
def test_smoke_gate_rejects_nonfinite_or_missing_optimization(failure):
    model, scorer, opt = tiny_pair()
    hist = [{"total": 1.0}]
    if failure == "loss":
        hist[0]["total"] = float("nan")
    elif failure == "parameter":
        next(model.parameters()).data.fill_(float("inf"))
    elif failure == "gradient":
        next(model.parameters()).grad.fill_(float("nan"))
    elif failure == "optimizer":
        next(iter(opt.state.values()))["exp_avg"].fill_(float("nan"))
    elif failure == "empty_history":
        hist = []
    else:
        opt.state.clear()
    assert not driver.smoke_gate(hist, model, scorer, opt)["passed"]


@pytest.mark.parametrize("cfg,epochs", [({"resume_state": {}}, 1), ({"history0": [{"total": 1}]}, 1),
                                         ({"early_stop": True}, 1), ({"grouped": True}, 1), ({}, 2)])
def test_ablation_smoke_rejects_paths_that_could_resume_or_evaluate_during_training(cfg, epochs):
    with pytest.raises(ValueError):
        ablation.train_eval_one(None, None, None, cfg, None, init_seed=0, epochs=epochs, smoke_only=True)


def test_smoke_uses_same_training_call_and_skips_every_post_training_forward(monkeypatch):
    data = HeteroData()
    data["model"].num_nodes = 2
    data["dataset"].num_nodes = 1
    data[ablation.TRAINED_ON].edge_index = torch.tensor([[0], [0]])
    data[ablation.TRAINED_ON].edge_attr = torch.tensor([0.5])
    split = (data, data.clone(), data.clone())
    calls, forwards, evaluations = [], [], []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, graph):
            forwards.append(graph)
            return {"model": torch.ones(2, 128), "dataset": torch.ones(1, 128)}

    monkeypatch.setattr(ablation, "accuracy_lookup", lambda _: {})
    monkeypatch.setattr(ablation, "perf_supervision", lambda *a: (torch.tensor([[1], [0]]), torch.tensor([0.7])))
    monkeypatch.setattr(ablation, "topk_membership", lambda *a, **kw: torch.ones(2, 1))
    monkeypatch.setattr(ablation, "lineage_components", lambda *a: torch.arange(2))
    monkeypatch.setattr(ablation, "build_models", lambda *a, **kw: (Model(), torch.nn.Linear(1, 1)))

    def fake_train(*args, **kwargs):
        calls.append(kwargs)
        return [{"total": 1.0, "rank": 0.4, "contrast": 0.6}], None

    monkeypatch.setattr(ablation, "train", fake_train)
    monkeypatch.setattr(ablation, "per_dataset_tau", lambda *a: (evaluations.append("tau") or (0.3, {0: 0.3})))
    monkeypatch.setattr(ablation, "head_retrieval", lambda *a: (evaluations.append("head") or ({"hit@10": 1.0}, {0: {}})))
    cfg = dict(ablation.B0, skip_diagnostics=True, fanout=True, sparse_M=True,
               contrast_n_neg=256, rank_loss="ranknet", rank_temperature=0.1)
    smoke = ablation.train_eval_one(data, {}, {}, cfg, split, init_seed=0, epochs=1, smoke_only=True)
    assert len(smoke) == 5
    assert smoke[0]["evaluation_performed"] is False and smoke[1:3] == ({}, {})
    assert not forwards and not evaluations
    formal = ablation.train_eval_one(data, {}, {}, cfg, split, init_seed=0, epochs=1)
    assert len(formal) == 5 and formal[0]["tau_macro"] == 0.3
    assert len(forwards) == 2 and evaluations == ["tau", "head"]
    assert calls[0] == calls[1]


def mocked_driver(tmp_path, monkeypatch):
    graph = tmp_path / "graph.pt"
    data = HeteroData()
    data["model"].num_nodes = 3
    torch.save({"data": data, "xm0_meta": {"num_families": 2, "num_size_buckets": 3},
                "xd0_meta": {}, "unique_dataset_id": pd.DataFrame({"mappedID": [0], "root": ["root"]})}, graph)
    monkeypatch.setattr(graph_surgery, "apply_similar_to_mode", lambda data, *a, **kw: data)
    monkeypatch.setattr(d0_splits, "make_root_aware_splits", lambda data, *a, **kw: (data, data, data))
    monkeypatch.setattr(driver, "capture_metadata", lambda out, args, cfg, graph: {
        "run_id": Path(out).name, "resolved_config": CK._jsonable(cfg)})
    seen = []

    def fake_train(data, xm0, xd0, cfg, split, *, init_seed, epochs, device, smoke_only=False):
        seen.append({"smoke_only": smoke_only, "init_seed": init_seed,
                     "cfg": {k: v for k, v in cfg.items() if k not in ("resume_state", "history0", "on_epoch_end")}})
        model, scorer, opt = tiny_pair()
        start = cfg["resume_state"]["epoch"] + 1 if cfg.get("resume_state") else 0
        for epoch in range(start, epochs):
            cfg["on_epoch_end"](epoch, {"total": 2.0 / (epoch + 1), "rank": 0.5,
                                      "contrast": 0.5, "global": 1.0},
                                  {"model": model, "scorer": scorer, "opt": opt})
        return ({"evaluation_performed": False} if smoke_only else {"tau_macro": 0.3}), {}, {}, model, scorer

    monkeypatch.setattr(ablation, "train_eval_one", fake_train)
    return graph, seen


def run_args(graph, out, *, smoke=False, epochs=1):
    result = ["--rung", "12k", "--graph", str(graph), "--out", str(out), "--device", "cpu",
              "--epochs", str(epochs), "--fanout", "--sparse-M", "--contrast-n-neg", "256",
              "--chunked-infer", "50000", "--skip-diagnostics", "--lake-gamma", "0.5",
              "--global-n-datasets", "128"]
    return result + (["--smoke-only"] if smoke else [])


def test_driver_smoke_artifacts_and_checkpoint_are_separate(tmp_path, monkeypatch):
    graph, seen = mocked_driver(tmp_path, monkeypatch)
    out = tmp_path / "smoke"
    assert driver.main(run_args(graph, out, smoke=True)) == 0
    report = json.loads((out / "SMOKE_REPORT.json").read_text())
    assert report["mechanism_gate"]["checkpoint_roundtrip"]
    assert report["evaluation_performed"] is False
    assert report["formal_training_eligible"] is False
    assert report["resumed_from"] is None and report["start_epoch"] == 0
    assert (out / driver.SMOKE_MARKER).is_file()
    assert not (out / "MANIFEST.json").exists()
    assert not (out / driver.A0_RUN_RECORDS).exists()
    assert not (out / "ckpt").exists() and not (out / "exports").exists()
    assert not (out / "smoke_ckpt" / CK.BEST).exists()
    ck = CK.load(str(out / "smoke_ckpt" / CK.LAST))
    assert ck["extra"]["smoke_only"] is True
    assert seen[0]["cfg"] == driver.build_config(config_args())
    assert len(report["metadata"]["resolved_config"]) == 40


def test_copied_smoke_checkpoint_cannot_initialize_formal_run(tmp_path, monkeypatch):
    graph, seen = mocked_driver(tmp_path, monkeypatch)
    smoke = tmp_path / "smoke"
    assert driver.main(run_args(graph, smoke, smoke=True)) == 0
    ck = smoke / "smoke_ckpt" / CK.LAST
    with pytest.raises(CK.IncompatibleCheckpoint, match="smoke-only checkpoint"):
        driver.main(run_args(graph, tmp_path / "formal", epochs=25) + ["--resume", str(ck)])
    assert len(seen) == 1


def test_formal_default_outputs_and_auto_resume_remain_available(tmp_path, monkeypatch):
    graph, seen = mocked_driver(tmp_path, monkeypatch)
    out = tmp_path / "formal"
    assert driver.main(run_args(graph, out, epochs=2)) == 0
    assert (out / "MANIFEST.json").is_file() and (out / "ckpt" / CK.BEST).is_file()
    assert not (out / driver.SMOKE_MARKER).exists() and not (out / "SMOKE_REPORT.json").exists()
    assert driver.main(run_args(graph, out, epochs=3)) == 0
    report = json.loads((out / "MANIFEST.json").read_text())
    assert report["start_epoch"] == 2 and report["resume_mode"] == "last.pt"
    assert all(not call["smoke_only"] for call in seen)


@pytest.mark.parametrize("a0", [False, True])
def test_sharded_a0_uses_shared_actual_file_verifier_before_loading(tmp_path, monkeypatch, a0):
    from scale1m import a0_graph_validation, graph_store
    graph_pt, seen = mocked_driver(tmp_path, monkeypatch)
    payload = torch.load(graph_pt, weights_only=False)
    graph = tmp_path / "sharded"
    graph.mkdir()
    metadata = {"files": {"fixture": "a" * 64}}
    if a0:
        metadata["a0"] = {"protocol": "a0"}
    (graph / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    events = []

    def verify(path):
        events.append("verify")
        return {"graph_sha256": CK.graph_digest(str(path)), "status": "PASS"}

    def load(*args, **kwargs):
        events.append("load")
        # Shared verification hashes all actual bytes; no duplicate pass needed.
        assert kwargs == {"mmap": True, "verify_sha256": False}
        return payload

    monkeypatch.setattr(a0_graph_validation, "verify_a0_graph", verify)
    monkeypatch.setattr(graph_store, "load_sharded", load)
    out = tmp_path / "smoke"
    assert driver.main(run_args(graph, out, smoke=True)) == 0
    assert events == (["verify", "load"] if a0 else ["load"])
    report = json.loads((out / "SMOKE_REPORT.json").read_text())
    assert ("a0_graph_verification" in report) is a0
    assert len(seen) == 1


def test_a0_hash_or_policy_failure_prevents_loading_and_training(tmp_path, monkeypatch):
    from scale1m import a0_graph_validation, graph_store
    graph = tmp_path / "sharded"
    graph.mkdir()
    (graph / "meta.json").write_text(json.dumps({"a0": {"protocol": "a0"}}), encoding="utf-8")
    loaded = []
    monkeypatch.setattr(graph_store, "load_sharded", lambda *a, **kw: loaded.append(True))

    def invalid(*args):
        raise ValueError("sha256 mismatch or nonzero A0 statistics")

    monkeypatch.setattr(a0_graph_validation, "verify_a0_graph", invalid)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        driver.main(run_args(graph, tmp_path / "smoke", smoke=True))
    assert not loaded


def test_a0_file_table_change_after_verification_prevents_training(tmp_path, monkeypatch):
    from scale1m import a0_graph_validation, graph_store
    graph_pt, seen = mocked_driver(tmp_path, monkeypatch)
    payload = torch.load(graph_pt, weights_only=False)
    graph = tmp_path / "sharded"
    graph.mkdir()
    metadata = {"a0": {"protocol": "a0"}, "files": {"fixture": "a" * 64}}
    (graph / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(a0_graph_validation, "verify_a0_graph", lambda path: {
        "graph_sha256": CK.graph_digest(str(path)), "status": "PASS"})

    def load(*args, **kwargs):
        metadata["files"]["fixture"] = "b" * 64
        (graph / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
        return payload

    monkeypatch.setattr(graph_store, "load_sharded", load)
    with pytest.raises(ValueError, match="file table changed"):
        driver.main(run_args(graph, tmp_path / "smoke", smoke=True))
    assert not seen


@pytest.mark.parametrize("binding_present", [False, True])
def test_removing_a0_metadata_marker_does_not_skip_prepared_graph_verification(tmp_path, monkeypatch, binding_present):
    from scale1m import a0_graph_validation
    graph = tmp_path / "sharded"
    graph.mkdir()
    metadata = {"files": {"A0_FEATURE_REPAIR.json": "a" * 64} if binding_present else {}}
    (graph / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    if not binding_present:
        (graph / "A0_FEATURE_REPAIR.json").write_text("{}", encoding="utf-8")

    def invalid(*args):
        raise ValueError("prepared graph marker missing")

    monkeypatch.setattr(a0_graph_validation, "verify_a0_graph", invalid)
    with pytest.raises(ValueError, match="marker missing"):
        driver.main(run_args(graph, tmp_path / "smoke", smoke=True))


def test_native_a0_envelope_retains_interrupted_and_resumed_segments(tmp_path, monkeypatch):
    from scale1m import a0_graph_validation, graph_store
    graph_pt, _ = mocked_driver(tmp_path, monkeypatch)
    payload = torch.load(graph_pt, weights_only=False)
    graph = tmp_path / "sharded"
    graph.mkdir()
    (graph / "meta.json").write_text(json.dumps({"a0": {"protocol": "a0"}, "files": {"fixture": "a" * 64}}), encoding="utf-8")
    monkeypatch.setattr(a0_graph_validation, "verify_a0_graph", lambda path: {
        "graph_sha256": CK.graph_digest(str(path)), "status": "PASS"})
    monkeypatch.setattr(graph_store, "load_sharded", lambda *a, **kw: payload)
    monkeypatch.setattr(driver, "peak_process_rss", lambda: (123456, None))
    environments = iter([{"hostname": "first-process-host"}, {"hostname": "resumed-process-host"}])
    monkeypatch.setattr(driver, "capture_runtime_environment", lambda: next(environments))
    original_train = ablation.train_eval_one

    def interrupted(*args, **kwargs):
        cfg = args[3]
        original_callback = cfg["on_epoch_end"]

        def callback(epoch, metrics, ctx):
            original_callback(epoch, metrics, ctx)
            if epoch == 4:
                raise RuntimeError("fixture interruption after saved epoch 5")

        cfg["on_epoch_end"] = callback
        return original_train(*args, **kwargs)

    monkeypatch.setattr(ablation, "train_eval_one", interrupted)
    out = tmp_path / "A0GD_full_s0_e25"
    with pytest.raises(RuntimeError, match="fixture interruption"):
        driver.main(run_args(graph, out, epochs=25))
    first = json.loads((out / driver.A0_RUN_RECORDS).read_text())
    assert first["status"] == "failed" and len(first["history"]) == 5
    assert first["train_segments"][0]["end_ns"] >= first["train_segments"][0]["start_ns"]
    monkeypatch.setattr(ablation, "train_eval_one", original_train)
    assert driver.main(run_args(graph, out, epochs=25)) == 0
    final = json.loads((out / driver.A0_RUN_RECORDS).read_text())
    native = json.loads((out / "metrics/train_history.json").read_text())
    assert final["history"] == native and len(native) == 25
    assert final["a0"] == {"protocol": "a0", "run_id": "A0_20260912", "seed": 0,
                           "graph_digest": CK.graph_digest(str(graph)),
                           "checkpoint_sha256": CK.sha256_of(str(out / "ckpt" / CK.LAST))}
    assert final["formal_run_name"] == out.name
    assert final["status"] == "complete" and len(final["train_segments"]) == 2
    assert final["train_segments"][0] == first["train_segments"][0]
    assert final["train_segments"][1]["start_epoch"] == 5
    assert final["timing_complete"] and len(final["resolved_config"]) == 40
    assert final["peak_process_rss_bytes"] == 123456
    assert final["epochs"] == 25 and final["initialization_seed"] == 0
    assert final["runtime_environment"] == {"hostname": "resumed-process-host"}
    assert final["train_segments"][0]["runtime_environment"] == {"hostname": "first-process-host"}
    assert final["train_segments"][1]["runtime_environment"] == final["runtime_environment"]


def test_native_envelope_rejects_a_different_identity_or_missing_resume_records(tmp_path):
    cfg = driver.build_config(config_args())
    ck = tmp_path / "last.pt"
    ck.write_bytes(b"checkpoint identity fixture")
    with pytest.raises(CK.IncompatibleCheckpoint, match="existing same-run"):
        driver.begin_a0_run_records(str(tmp_path), cfg, {"graph_sha256": "a"}, "run", 0, 0, str(ck))
    records = driver.begin_a0_run_records(str(tmp_path), cfg, {"graph_sha256": "a"}, "run", 0, 0, None)
    (tmp_path / driver.A0_RUN_RECORDS).write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(CK.IncompatibleCheckpoint, match="different graph, seed, run or recipe"):
        driver.begin_a0_run_records(str(tmp_path), cfg, {"graph_sha256": "b"}, "run", 0, 0, None)


@pytest.mark.parametrize("extra", [{"smoke_only": True}, {"run_purpose": "smoke_only"},
                                   {"formal_training_eligible": False}])
def test_each_smoke_ineligibility_marker_is_sufficient(extra):
    with pytest.raises(CK.IncompatibleCheckpoint, match="smoke"):
        driver.reject_smoke_checkpoint({"extra": extra})


def test_runtime_environment_is_observed_from_current_process(monkeypatch):
    monkeypatch.setattr(driver.platform, "node", lambda: "current-host")
    monkeypatch.setattr(driver.platform, "platform", lambda: "current-platform")
    monkeypatch.setattr(driver.platform, "processor", lambda: "current-cpu")
    monkeypatch.setattr(driver.os, "cpu_count", lambda: 32)
    monkeypatch.setattr(driver.torch.cuda, "is_available", lambda: False)
    actual = driver.capture_runtime_environment()
    assert actual["hostname"] == "current-host" and actual["platform"] == "current-platform"
    assert actual["processor"] == "current-cpu" and actual["cpu_count"] == 32
    assert actual["python"] == driver.sys.version.split()[0]
    assert actual["torch"] == str(driver.torch.__version__)
    assert actual["cuda_runtime"] == driver.torch.version.cuda
    assert actual["gpu_name"] is None and actual["gpu_total_memory_bytes"] is None
