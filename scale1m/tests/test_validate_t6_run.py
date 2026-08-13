import copy
import json

from scale1m.validate_t6_run import validate_manifest


def _manifest(run_id="R2_12k_s0_e25", rung="12k", seed=0, n=12_000, epochs=25):
    graph_hash = "a" * 64
    return {
        "run_id": run_id,
        "rung": rung,
        "seed": seed,
        "n_models": n,
        "epochs": epochs,
        "wallclock_s": 100.0,
        "peak_gpu_mem_gb": 1.25,
        "start_epoch": 0,
        "resumed_from": None,
        "graph_sha256": graph_hash,
        "binding": {"graph_sha256": graph_hash},
        "mechanism_gate": {
            "epochs": epochs,
            "passed": True,
            "loss_descended": True,
            "no_nan": True,
        },
        "scale_switches": {
            "fanout": True,
            "sparse_M": True,
            "contrast_n_neg": 256,
            "chunked_infer": 50_000,
        },
    }


def _history(tmp_path, epochs):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    (metrics / "train_history.json").write_text(
        json.dumps([{"total": 2 - i / epochs} for i in range(epochs)]),
        encoding="utf-8",
    )


def test_validate_manifest_passes_complete_contract(tmp_path):
    run_dir = tmp_path / "R2_12k_s0_e25"
    run_dir.mkdir()
    _history(run_dir, 25)
    report = validate_manifest(
        _manifest(),
        run_dir=run_dir,
        stage="12k",
        expected_n=12_000,
        expected_epochs=25,
    )
    assert report["passed"] is True
    assert report["errors"] == []
    assert report["gold10_validated"] is False


def test_validate_manifest_rejects_failed_switch_and_bad_history(tmp_path):
    run_dir = tmp_path / "R2_12k_s0_e25"
    run_dir.mkdir()
    _history(run_dir, 24)
    manifest = copy.deepcopy(_manifest())
    manifest["scale_switches"]["sparse_M"] = False
    report = validate_manifest(
        manifest,
        run_dir=run_dir,
        stage="12k",
        expected_n=12_000,
        expected_epochs=25,
    )
    assert report["passed"] is False
    assert "sparse_M" in report["errors"]
    assert "train_history_length" in report["errors"]


def test_smoke_projection_stops_over_four_hours(tmp_path):
    run_dir = tmp_path / "R2_100k_s0_e2"
    run_dir.mkdir()
    _history(run_dir, 2)
    manifest = _manifest(
        run_id="R2_100k_s0_e2", rung="100k", n=100_000, epochs=2
    )
    manifest["wallclock_s"] = 1_201.0
    report = validate_manifest(
        manifest,
        run_dir=run_dir,
        stage="100k_smoke",
        expected_n=100_000,
        expected_epochs=2,
        full_time_seconds=14_400,
    )
    assert report["passed"] is False
    assert any("projection" in error for error in report["errors"])
