"""Live reviewer selection, data isolation and pipeline integration contracts."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scale1m.reproduction import cli, live_snapshot, pipeline


def completed():
    return {"protocol": "live-hf", "ok": True, "counts": {"models": 1234, "datasets": 57},
            "files": [{"path": "models/input", "bytes": 3, "sha256": "a" * 64}], "errors": []}


def test_default_live_plan_in_clean_arbitrary_checkout(tmp_path):
    checkout = tmp_path / "reviewer-checkout"
    source = pipeline.REPO / "scale1m"
    target = checkout / "scale1m"
    target.mkdir(parents=True)
    for path in source.glob("*.py"):
        shutil.copy2(path, target / path.name)
    shutil.copytree(source / "reproduction", target / "reproduction",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "a0_snapshot.json", "expected.json"))
    before = {p.relative_to(checkout) for p in checkout.rglob("*")}
    env = os.environ.copy()
    for key in ("PYTHONPATH", "MLF_DATA_DIR", "MLF_RUNS_DIR"):
        env.pop(key, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run([sys.executable, "-B", "-m", "scale1m.reproduce", "plan"],
                            cwd=checkout, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "Live HF capture" in result.stdout
    assert "--hf-only" in result.stdout and "scale1m.evaluate_live" in result.stdout
    assert "frozen" not in result.stdout and "a0_snapshot" not in result.stdout
    assert {p.relative_to(checkout) for p in checkout.rglob("*")} == before


def test_capture_selection_resumes_and_refresh_is_explicit(tmp_path, monkeypatch):
    data = tmp_path / "data"
    calls = []

    def download(root):
        calls.append(root)
        root.mkdir(parents=True, exist_ok=True)
        (root / "sentinel").write_text("original", encoding="utf-8")
        return completed()

    monkeypatch.setattr(live_snapshot, "download", download)
    for _ in range(2):
        assert cli.main(["download", "--data-root", str(data)]) == 0
    assert calls[0] == calls[1]
    selected = json.loads((data / "LIVE_CURRENT.json").read_text())["snapshot_id"]
    assert calls[0] == data / "live" / selected
    assert cli.main(["download", "--data-root", str(data), "--new-snapshot"]) == 0
    assert calls[2] != calls[0]
    assert (calls[0] / "sentinel").read_text() == "original"
    assert not (data / "data1m").exists()


@pytest.mark.parametrize("name", ["../old", "a/b", "C:drive", ".", "CON", "foo."])
def test_capture_names_cannot_escape_or_alias_windows_paths(tmp_path, monkeypatch, name):
    data = tmp_path / "missing"
    monkeypatch.setattr(live_snapshot, "download", lambda *args: pytest.fail("network called"))
    assert cli.main(["download", "--data-root", str(data), "--snapshot-id", name]) == 1
    assert not data.exists()


def test_offline_verification_never_creates_capture_or_calls_network(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setattr(live_snapshot, "download", lambda *args: pytest.fail("network called"))
    monkeypatch.setattr(live_snapshot, "verify", lambda root: {"ok": False, "errors": ["missing"]})
    assert cli.main(["download", "--offline", "--snapshot-id", "review", "--data-root", str(data)]) == 1
    assert not data.exists()


def test_live_run_requires_completed_capture_and_binds_its_bytes(tmp_path, monkeypatch):
    data, work = tmp_path / "data", tmp_path / "outputs"
    calls = []
    report = {"ok": False, "errors": ["cursor not exhausted"]}
    monkeypatch.setattr(live_snapshot, "verify", lambda root: report)
    monkeypatch.setattr(live_snapshot, "input_digest", lambda record: "verified-live-inputs")
    monkeypatch.setattr(cli, "doctor", lambda *args: {"ok": True, "warnings": []})
    monkeypatch.setattr(pipeline, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(cli, "live_summary", lambda workspace: {"status": "complete"})
    argv = ["run", "--data-root", str(data), "--workspace", str(work), "--snapshot-id", "review"]
    assert cli.main(argv) == 1
    assert not calls and not work.exists()
    report = completed()
    assert cli.main(argv) == 0
    assert calls == [(("live", data, work, "cuda"), {
        "resume": False, "input_digest": "verified-live-inputs", "snapshot_root": data / "live/review"})]


def test_live_plan_uses_only_captured_hf_inputs_and_dynamic_training(tmp_path):
    data, work, capture = tmp_path / "data", tmp_path / "work", tmp_path / "data/live/review"
    plan = pipeline.steps("live", data, work, "cuda", snapshot_root=capture)
    stages = {step.name: step for step in plan}
    args = stages["canonicalize"].argv
    assert args[args.index("--candidates") + 1] == str(capture / "models")
    for name in ("merge-hf-evidence", "ladder"):
        assert "--hf-only" in stages[name].argv
        assert "--source-dir" not in stages[name].argv
    assert "seed-vocabulary" not in stages and "prepare-frozen-graph" not in stages
    assert [s.name for s in plan].index("check-canonical") < [s.name for s in plan].index("features")
    for seed in range(3):
        argv = stages[f"train-s{seed}"].argv
        assert argv[argv.index("--rung") + 1] == "live"
        assert argv[argv.index("--epochs") + 1] == "25"
        assert argv[argv.index("--seed") + 1] == str(seed)
    assert "scale1m.evaluate_live" in plan[-1].argv


def test_workspace_cannot_be_inside_live_capture(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *args, **kwargs: pytest.fail("computation called"))
    with pytest.raises(ValueError, match="separate"):
        pipeline.run("live", data, data / "live/review/outputs", "cuda", snapshot_root=data / "live/review")
    assert not data.exists()


def test_live_summary_accepts_new_counts_without_historical_score_comparison(tmp_path):
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    path = evaluation / "LIVE_RESULTS.json"
    report = {"protocol": "live-hf", "status": "complete", "n_models": 4567890,
              "n_datasets": 9876, "all_ann_calibration_passed": True,
              "summary": {"gold@10": .12}}
    path.write_text(json.dumps(report), encoding="utf-8")
    result = cli.live_summary(tmp_path)
    assert result["n_models"] == 4567890 and result["summary"] == {"gold@10": .12}
    assert "reference_gold_at_10" not in result
    report["all_ann_calibration_passed"] = False
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="calibration"):
        cli.live_summary(tmp_path)
