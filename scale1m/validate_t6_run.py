"""Validate a T6 training manifest for the unattended watGPU workflow.

This is deliberately independent of torch so gate jobs can run with the
cluster's small system Python instead of staging the training environment.
It validates the T6 mechanism contract only; gold@10 remains a T7/T8 gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def validate_manifest(
    manifest: dict,
    *,
    run_dir: Path,
    stage: str,
    expected_n: int,
    expected_epochs: int,
    full_time_seconds: int = 14_400,
) -> dict:
    gate = manifest.get("mechanism_gate") or {}
    switches = manifest.get("scale_switches") or {}
    binding = manifest.get("binding") or {}
    errors: list[str] = []

    expected_rung = "100k" if stage.startswith("100k") else stage
    expected_seed = int(stage[-1]) if stage in {"100k_s0", "100k_s1", "100k_s2"} else 0
    checks = {
        "run_id": manifest.get("run_id") == run_dir.name,
        "rung": manifest.get("rung") == expected_rung,
        "seed": manifest.get("seed") == expected_seed,
        "n_models": manifest.get("n_models") == expected_n,
        "epochs": manifest.get("epochs") == expected_epochs,
        "gate_epochs": gate.get("epochs") == expected_epochs,
        "mechanism_passed": gate.get("passed") is True,
        "loss_descended": gate.get("loss_descended") is True,
        "no_nan": gate.get("no_nan") is True,
        "fanout": switches.get("fanout") is True,
        "sparse_M": switches.get("sparse_M") is True,
        "contrast_n_neg": switches.get("contrast_n_neg") == 256,
        "chunked_infer": switches.get("chunked_infer") == 50_000,
        "binding_hash": bool(binding.get("graph_sha256")),
        "binding_hash_matches": (
            bool(manifest.get("graph_sha256"))
            and manifest.get("graph_sha256") == binding.get("graph_sha256")
        ),
    }
    errors.extend(name for name, passed in checks.items() if not passed)

    wall = manifest.get("wallclock_s")
    peak = manifest.get("peak_gpu_mem_gb")
    if not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall <= 0:
        errors.append("invalid wallclock_s")
    if not isinstance(peak, (int, float)) or not math.isfinite(peak) or peak <= 0:
        errors.append("invalid peak_gpu_mem_gb")

    start_epoch = manifest.get("start_epoch")
    if not isinstance(start_epoch, int) or not 0 <= start_epoch < expected_epochs:
        errors.append("invalid start_epoch")
    elif start_epoch > 0 and not manifest.get("resumed_from"):
        errors.append("start_epoch_without_resumed_from")

    history_path = run_dir / "metrics" / "train_history.json"
    if not history_path.is_file():
        errors.append("missing train_history.json")
    else:
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
            if not isinstance(history, list) or len(history) != expected_epochs:
                errors.append("train_history_length")
        except Exception as exc:  # surfaced in the gate report
            errors.append(f"cannot read train_history.json: {exc}")

    projected = None
    if stage == "100k_smoke" and isinstance(wall, (int, float)) and wall > 0:
        projected = wall / expected_epochs * 25
        if projected > full_time_seconds:
            errors.append(
                f"25-epoch projection {projected:.1f}s exceeds limit "
                f"{full_time_seconds}s"
            )

    return {
        "stage": stage,
        "run_id": run_dir.name,
        "passed": not errors,
        "errors": errors,
        "checks": checks,
        "projected_25_epoch_s": projected,
        "gold10_validated": False,
        "gold10_note": "T7/T8 are not implemented; this is a T6 mechanism gate only",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--expect-n", required=True, type=int)
    parser.add_argument("--expect-epochs", required=True, type=int)
    parser.add_argument("--train-job-id", required=True)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--full-time-seconds", type=int, default=14_400)
    parser.add_argument(
        "--gold-gate-override", type=int, default=0, choices=(0, 1),
        help="T6_ALLOW_NO_GOLD_GATE: the training chain was allowed to advance "
             "without T7/T8 gold@10 validation. Recorded in every report so the "
             "opt-in travels with the evidence, not only with jobs.tsv.")
    args = parser.parse_args(argv)

    manifest_path = args.run_dir / "MANIFEST.json"
    if not manifest_path.is_file():
        report = {
            "stage": args.stage,
            "run_id": args.run_dir.name,
            "train_job_id": args.train_job_id,
            "manifest": str(manifest_path),
            "passed": False,
            "errors": [f"missing {manifest_path}"],
            "checks": {},
            "projected_25_epoch_s": None,
            "gold10_validated": False,
            "gold10_note": "T7/T8 are not implemented; this is a T6 mechanism gate only",
            "gold_gate_override": args.gold_gate_override,
        }
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            report = validate_manifest(
                manifest,
                run_dir=args.run_dir,
                stage=args.stage,
                expected_n=args.expect_n,
                expected_epochs=args.expect_epochs,
                full_time_seconds=args.full_time_seconds,
            )
            report.update(
                train_job_id=args.train_job_id,
                manifest=str(manifest_path),
                gold_gate_override=args.gold_gate_override,
            )
        except Exception as exc:
            report = {
                "stage": args.stage,
                "run_id": args.run_dir.name,
                "train_job_id": args.train_job_id,
                "manifest": str(manifest_path),
                "passed": False,
                "errors": [f"cannot validate manifest: {exc}"],
                "checks": {},
                "projected_25_epoch_s": None,
                "gold10_validated": False,
                "gold10_note": "T7/T8 are not implemented; this is a T6 mechanism gate only",
            }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.report.with_name(args.report.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
