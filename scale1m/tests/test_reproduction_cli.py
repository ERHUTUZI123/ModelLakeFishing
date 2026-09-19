"""Reviewer CLI contracts, using temporary inputs and no network or author data."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scale1m.reproduction import cli, pipeline, snapshot


def manifest_for(content=b"frozen test input", name="data1m/input.bin", source=None):
    return {
        "schema_version": 1,
        "protocol": "a0",
        "source": source or {},
        "files": [{"path": name, "sha256": hashlib.sha256(content).hexdigest(),
                   "bytes": len(content), "profiles": list(snapshot.PROFILES)}],
    }


def write_manifest(tmp_path, manifest):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_clean_checkout_help_and_all_plans_have_no_side_effects(tmp_path):
    # Copy only the actual CLI package into an arbitrarily named clean checkout.
    # No ignored repro/, docs/, data/, parent package, or author PYTHONPATH exists.
    checkout = tmp_path / "reviewer-checkout"
    package = checkout / "scale1m"
    package.mkdir(parents=True)
    for name in ("__init__.py", "reproduce.py"):
        shutil.copyfile(pipeline.REPO / "scale1m" / name, package / name)
    shutil.copytree(pipeline.REPO / "scale1m/reproduction", package / "reproduction",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    before = {p.relative_to(checkout) for p in checkout.rglob("*")}
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("MLF_DATA_DIR", None)
    env.pop("MLF_RUNS_DIR", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for args in (["--help"], *(["plan", "--profile", p] for p in snapshot.PROFILES)):
        result = subprocess.run([sys.executable, "-B", "-m", "scale1m.reproduce", *args],
                                cwd=checkout, env=env, capture_output=True, text=True,
                                encoding="utf-8", timeout=30)
        assert result.returncode == 0, result.stderr
        if args[0] == "plan":
            assert "NOT CONFIGURED" in result.stdout
            assert "[evaluate" in result.stdout
    assert {p.relative_to(checkout) for p in checkout.rglob("*")} == before


def test_missing_source_fails_before_network_or_directory_creation(tmp_path, monkeypatch, capsys):
    manifest = manifest_for()
    root = tmp_path / "absent-data"
    called = []
    with pytest.raises(ValueError, match="no published Hugging Face"):
        snapshot.download(root, manifest, "full", fetch=lambda **kw: called.append(kw))
    assert not called
    assert not root.exists()
    path = write_manifest(tmp_path, manifest)
    assert cli.main(["download", "--profile", "full", "--manifest", str(path), "--data-root", str(root)]) == 1
    assert "cannot recreate" in capsys.readouterr().err
    assert not root.exists()


@pytest.mark.parametrize("name", ["../escape", "data/../escape", "/absolute", "C:/absolute",
                                  "data\\escape", "data//empty", "data/./dot", "data/end/", "",
                                  None, 123, ["not", "a", "path"]])
def test_manifest_rejects_unsafe_paths(tmp_path, name):
    with pytest.raises(ValueError, match="Unsafe manifest path"):
        snapshot.load_manifest(write_manifest(tmp_path, manifest_for(name=name)))


def test_manifest_rejects_case_aliases_on_every_platform(tmp_path):
    manifest = manifest_for(name="data1m/input.bin")
    manifest["files"].append({**manifest["files"][0], "path": "data1m/INPUT.bin"})
    with pytest.raises(ValueError, match="Duplicate manifest path"):
        snapshot.load_manifest(write_manifest(tmp_path, manifest))


@pytest.mark.parametrize("revision", ["main", "a0-v1", "2026-08-18", "a" * 7, "g" * 40])
def test_mutable_or_invalid_hf_revisions_are_rejected(tmp_path, revision):
    root = tmp_path / "absent-data"
    with pytest.raises(ValueError, match="full 40-character"):
        snapshot.download(root, manifest_for(), "full", repo_id="owner/dataset",
                          revision=revision, fetch=lambda **kw: pytest.fail("network attempted"))
    assert not root.exists()


def test_pinned_download_verifies_bytes_and_resumes_without_fetch(tmp_path):
    root = tmp_path / "data"
    content = b"frozen test input"
    manifest = manifest_for(content)
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs)
        cached = Path(kwargs["cache_dir"]) / "cached-input.bin"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(content)
        return str(cached)

    receipt = snapshot.download(root, manifest, "full", repo_id="owner/a0",
                                revision="A" * 40, fetch=fetch)
    assert calls == [{"repo_id": "owner/a0", "repo_type": "dataset", "revision": "a" * 40,
                      "filename": "data1m/input.bin", "cache_dir": str(root / ".hf_cache")}]
    assert (root / "data1m/input.bin").read_bytes() == content
    assert receipt["ok"]
    assert json.loads((root / "A0_DOWNLOAD_full.json").read_text())["source"]["revision"] == "a" * 40
    snapshot.download(root, manifest, "full", repo_id="owner/a0", revision="a" * 40,
                      fetch=lambda **kw: pytest.fail("verified input downloaded again"))
    assert snapshot.download(root, manifest, "full", offline=True)["ok"]


def test_downloaded_corruption_is_never_installed(tmp_path):
    content = b"frozen test input"
    bad = tmp_path / "bad-cache.bin"
    bad.write_bytes(b"x" * len(content))  # Same size: the SHA-256 guard must detect this.
    root = tmp_path / "data"
    with pytest.raises(ValueError, match="differs from the frozen A0 manifest"):
        snapshot.download(root, manifest_for(content), "full", repo_id="owner/a0",
                          revision="a" * 40, fetch=lambda **kw: bad)
    assert not root.exists()


def test_existing_modified_input_is_preserved_and_not_redownloaded(tmp_path):
    root = tmp_path / "data"
    target = root / "data1m/input.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"user-owned altered bytes")
    with pytest.raises(ValueError, match="Existing file has different bytes"):
        snapshot.download(root, manifest_for(), "full", repo_id="owner/a0",
                          revision="a" * 40, fetch=lambda **kw: pytest.fail("network attempted"))
    assert target.read_bytes() == b"user-owned altered bytes"
    assert not (root / "A0_DOWNLOAD_full.json").exists()


def test_offline_verify_and_run_reject_missing_inputs_without_outputs(tmp_path, capsys):
    root, work = tmp_path / "missing", tmp_path / "work"
    path = write_manifest(tmp_path, manifest_for())
    with pytest.raises(ValueError, match="Offline input verification failed"):
        snapshot.download(root, manifest_for(), "full", offline=True)
    for command in ("verify", "run"):
        assert cli.main([command, "--profile", "full", "--manifest", str(path), "--data-root", str(root),
                         "--workspace", str(work)]) == 1
    assert "Required input files are missing/corrupt" in capsys.readouterr().err
    assert not root.exists()
    assert not work.exists()


@pytest.mark.parametrize("profile", ["full", "train"])
def test_training_plans_wire_frozen_evidence_and_recorded_three_splits(tmp_path, profile):
    root, work = tmp_path / "inputs", tmp_path / "outputs"
    plan = pipeline.steps(profile, root, work, "cuda")
    by_name = {step.name: step for step in plan}
    names = list(by_name)
    if profile == "full":
        for name in ("merge", "ladder"):
            argv = by_name[name].argv
            assert argv[argv.index("--source-dir") + 1] == str(root / "data1m/source_evidence")
        assert names.index("check-canonical") < names.index("features")
        assert names.index("seed-vocabulary") < names.index("features")
        assert names.index("check-graph") < names.index("train-s0")
        features = by_name["features"].argv
        assert features[features.index("--ladder") + 1] == str(work / "ladder/full_model_ids.parquet")
    else:
        assert names[0] == "prepare-frozen-graph"
        assert "scale1m.reproduction.preparation" in by_name[names[0]].argv
    for seed in range(3):
        train = by_name[f"train-s{seed}"].argv
        assert train[train.index("--seed") + 1] == str(seed)
        assert train[train.index("--epochs") + 1] == "25"
        assert train[train.index("--lake-gamma") + 1] == "0.5"
        assert "--fanout" in train and "--sparse-M" in train
        assert names.index(f"train-s{seed}") < names.index(f"export-s{seed}") < names.index(f"prior-s{seed}")
        export = by_name[f"export-s{seed}"].argv
        assert export[export.index("--ckpt") + 1] == "last"
    assert names[-1] == "evaluate"
    assert "--final-only" not in by_name["evaluate"].argv


def test_replay_uses_saved_indexes_cpu_and_no_training(tmp_path):
    plan = pipeline.steps("replay", tmp_path / "inputs", tmp_path / "work", "cuda")
    assert len(plan) == 1
    argv = plan[0].argv
    assert "--final-only" in argv and "--reference-index-dir" in argv
    assert argv[argv.index("--device") + 1] == "cpu"


def install_fake_plan(monkeypatch, plan):
    monkeypatch.setattr(pipeline, "steps", lambda *args: plan)
    monkeypatch.setattr(pipeline, "implementation_digest", lambda: "test-code-identity")


def test_resume_verifies_latest_owner_of_shared_outputs(tmp_path, monkeypatch):
    data, work = tmp_path / "data", tmp_path / "work"
    output = work / "shared.txt"
    plan = [pipeline.Step(name, (name,), (output,)) for name in ("first", "second")]
    install_fake_plan(monkeypatch, plan)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        output.write_text(argv[0], encoding="utf-8")

    monkeypatch.setattr(pipeline.subprocess, "run", run)
    state_path = pipeline.run("full", data, work, "cuda")
    pipeline.run("full", data, work, "cuda", resume=True)
    assert calls == [["first"], ["second"]]
    assert json.loads(state_path.read_text())["status"] == "complete"
    output.write_text("externally modified", encoding="utf-8")
    with pytest.raises(ValueError, match="Completed stage output changed"):
        pipeline.run("full", data, work, "cuda", resume=True)
    assert len(calls) == 2


def test_interrupted_training_resumes_last_checkpoint(tmp_path, monkeypatch):
    data, work = tmp_path / "data", tmp_path / "work"
    run_dir = work / "runs/s0"
    checkpoint = run_dir / "ckpt/last.pt"
    plan = [pipeline.Step("train-s0", ("train",), (run_dir,))]
    install_fake_plan(monkeypatch, plan)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 1:
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"partial epoch checkpoint")
            raise subprocess.CalledProcessError(1, argv)
        assert argv[-2:] == ["--resume", str(checkpoint)]
        checkpoint.write_bytes(b"completed epoch checkpoint")

    monkeypatch.setattr(pipeline.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.run("train", data, work, "cuda")
    state_path = pipeline.run("train", data, work, "cuda", resume=True)
    assert json.loads(state_path.read_text())["status"] == "complete"
    assert len(calls) == 2


def test_workspace_cannot_overwrite_frozen_input_tree(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *a, **kw: pytest.fail("stage executed"))
    for work in (data, data / "data1m/new", tmp_path):
        with pytest.raises(ValueError, match="separate from downloaded"):
            pipeline.run("full", data, work, "cuda")
    assert not data.exists()


def test_input_digest_binds_bytes_and_paths_but_not_archive_location():
    manifest = manifest_for()
    expected = snapshot.input_digest(manifest, "full")
    manifest["source"] = {"repo_id": "another/mirror", "revision": "b" * 40}
    assert snapshot.input_digest(manifest, "full") == expected
    manifest["files"][0]["sha256"] = "c" * 64
    assert snapshot.input_digest(manifest, "full") != expected
    manifest = manifest_for(name="data1m/renamed.bin")
    assert snapshot.input_digest(manifest, "full") != expected


def test_changed_input_binding_refuses_resume_before_stage_execution(tmp_path, monkeypatch):
    data, work = tmp_path / "data", tmp_path / "work"
    output = work / "result.txt"
    install_fake_plan(monkeypatch, [pipeline.Step("stage", ("stage",), (output,))])
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        output.write_text("computed result", encoding="utf-8")

    monkeypatch.setattr(pipeline.subprocess, "run", run)
    pipeline.run("full", data, work, "cuda", input_digest="a" * 64)
    with pytest.raises(ValueError, match="changed code, inputs"):
        pipeline.run("full", data, work, "cuda", resume=True, input_digest="b" * 64)
    assert calls == [["stage"]]
    assert output.read_text(encoding="utf-8") == "computed result"


def test_cli_passes_verified_input_binding_to_pipeline(tmp_path, monkeypatch):
    data, work = tmp_path / "data", tmp_path / "work"
    manifest = manifest_for()
    manifest_path = write_manifest(tmp_path, manifest)
    target = data / "data1m/input.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"frozen test input")
    calls = []
    monkeypatch.setattr(cli, "doctor", lambda *args: {"ok": True, "warnings": []})
    monkeypatch.setattr(pipeline, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(cli, "compare_results", lambda *args: {"ok": True})
    assert cli.main(["run", "--profile", "full", "--manifest", str(manifest_path), "--data-root", str(data),
                     "--workspace", str(work)]) == 0
    assert calls == [(("full", data.resolve(), work.resolve(), "cuda"),
                      {"resume": False, "input_digest": snapshot.input_digest(manifest, "full")})]
    target.write_bytes(b"altered")
    assert cli.main(["run", "--profile", "full", "--manifest", str(manifest_path), "--data-root", str(data),
                     "--workspace", str(work)]) == 1
    assert len(calls) == 1


@pytest.fixture
def comparison(tmp_path, monkeypatch):
    expected = {"candidate_models": 100, "dataset_task_nodes": 30,
                "eligible_queries": [10, 10, 10], "gold_at_10": [0.2, 0.3, 0.4],
                "mean_gold_at_10": 0.3}
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(json.dumps(expected), encoding="utf-8")
    monkeypatch.setattr(cli, "EXPECTED", expected_path)
    work = tmp_path / "work"
    report_path = work / "evaluation/A0_PORTABLE_RESULTS.json"
    report_path.parent.mkdir(parents=True)
    report = {"protocol": "a0", "status": "complete", "n_models": 100, "n_datasets": 30,
              "mode": "saved_index_final_only", "all_ann_calibration_passed": True,
              "per_seed": {str(s): {"rows": {"G_hnsw1000_task": {"n_queries": 10, "gold@10": score}}}
                           for s, score in enumerate(expected["gold_at_10"])},
              "summary": {"latency_ms": {"p50": 5}}, "measurement_scope": "synthetic test"}

    def save():
        report_path.write_text(json.dumps(report), encoding="utf-8")

    save()
    return work, report, save


def test_replay_comparison_requires_exact_quality_but_not_identical_latency(comparison):
    work, report, save = comparison
    result = cli.compare_results(work, "replay")
    assert result["exact_quality_match"]
    assert result["latency"]["p50"] == 5
    report["per_seed"]["0"]["rows"]["G_hnsw1000_task"]["gold@10"] = 0.1
    save()
    with pytest.raises(ValueError, match="differs from recorded A0"):
        cli.compare_results(work, "replay")


def test_retrained_quality_differences_are_explicit_without_invented_tolerance(comparison):
    work, report, save = comparison
    report["mode"] = "rebuilt_index_full_evaluation"
    report["per_seed"]["0"]["rows"]["G_hnsw1000_task"]["gold@10"] = 0.1
    save()
    result = cli.compare_results(work, "full")
    assert not result["exact_quality_match"]
    assert result["per_split_delta"] == pytest.approx([-0.1, 0, 0])
    assert "not hidden by a tolerance" in result["interpretation"]
    report["all_ann_calibration_passed"] = False
    save()
    with pytest.raises(ValueError, match="pass ANN calibration"):
        cli.compare_results(work, "full")


def test_comparison_rejects_different_query_cohort(comparison):
    work, report, save = comparison
    report["per_seed"]["1"]["rows"]["G_hnsw1000_task"]["n_queries"] = 9
    save()
    with pytest.raises(ValueError, match="Eligible-query cohort differs"):
        cli.compare_results(work, "replay")
