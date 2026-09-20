"""Collect the current public HF lake and reproduce the retrieval method."""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import uuid

from . import pipeline, snapshot

EXPECTED = Path(__file__).with_name("expected.json")


def memory_bytes():
    try:
        if os.name == "nt":
            import ctypes
            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in
                    ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]
            status = Status()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.total
        else:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError, AttributeError):
        pass
    return None


def doctor(profile: str, data: Path, device: str) -> dict:
    packages = {}
    required = ["numpy", "pandas", "pyarrow", "torch", "hnswlib", "requests"]
    if profile != "replay":
        required += ["scipy", "scikit-learn", "matplotlib", "torch-geometric"]
    if profile in ("full", "live"):
        required += ["sentence-transformers", "transformers", "huggingface-hub"]
    errors, warnings = [], []
    for name in required:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
            errors.append("Missing package: " + name)
    if sys.version_info[:2] != (3, 11):
        warnings.append("Recorded environment uses Python 3.11.4; install CPython 3.11 for the documented environment")
    info = {}
    try:
        import torch
        info = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                "cuda_available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.update(gpu=props.name, gpu_gib=props.total_memory / 2**30)
        if profile != "replay":
            if platform.system() != "Linux":
                warnings.append("Full training was validated on Linux; use Linux/WSL2 with NVIDIA GPU passthrough")
            if device != "cuda" or not torch.cuda.is_available():
                errors.append("Full-lake training requires CUDA; the documented workflow assumes a 48 GB NVIDIA GPU")
            elif info["gpu_gib"] < 40:
                errors.append("Unchanged training needs a 48 GB-class GPU (recorded peak ~38 GiB allocated plus runtime overhead); this GPU is too small")
            try:
                from torch_geometric.typing import WITH_PYG_LIB
                info["pyg_lib"] = WITH_PYG_LIB
                if not WITH_PYG_LIB:
                    errors.append("pyg-lib sampler unavailable; install the matching PyG wheel for the recorded training recipe")
            except ImportError:
                errors.append("PyG could not import; install the matching torch-geometric/pyg-lib wheels")
    except ImportError:
        errors.append("PyTorch could not import")
    try:
        import hnswlib
    except ImportError:
        errors.append("hnswlib could not import; a C++ compiler may be required when installing")
    probe = data
    while not probe.exists():
        probe = probe.parent
    free_gib = shutil.disk_usage(probe).free / 2**30
    recommended_disk = 32 if profile == "replay" else 120
    ram = memory_bytes()
    minimum_ram = 16 if profile == "replay" else 64
    if free_gib < recommended_disk:
        warnings.append(f"Recommended free disk: {recommended_disk} GiB; currently {free_gib:.1f} GiB")
    if ram is not None and ram / 2**30 < minimum_ram:
        warnings.append(f"Recommended host RAM: at least {minimum_ram} GiB; original training allocation 128 GiB")
    return {"profile": profile, "python": platform.python_version(), "system": platform.system(),
            "packages": packages, "hardware": info, "ram_gib": ram / 2**30 if ram else None,
            "free_disk_gib": free_gib, "errors": errors, "warnings": warnings, "ok": not errors}


def compare_results(work: Path, profile: str) -> dict:
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    path = work / "evaluation/A0_PORTABLE_RESULTS.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("protocol") != "a0" or report.get("status") != "complete":
        raise ValueError("Evaluation is not a completed A0 report")
    if report.get("n_models") != expected["candidate_models"] or report.get("n_datasets") != expected["dataset_task_nodes"]:
        raise ValueError("Evaluation universe differs from A0")
    rows = [report["per_seed"][str(s)]["rows"]["G_hnsw1000_task"] for s in range(3)]
    if [r["n_queries"] for r in rows] != expected["eligible_queries"]:
        raise ValueError("Eligible-query cohort differs from A0")
    values = [r["gold@10"] for r in rows]
    if not all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in values):
        raise ValueError("Invalid retrieval metrics")
    deltas = [v - e for v, e in zip(values, expected["gold_at_10"])]
    exact = all(abs(d) <= 1e-12 for d in deltas)
    if profile == "replay":
        if report.get("mode") != "saved_index_final_only":
            raise ValueError("Replay comparison requires a saved-index replay")
        if not exact:
            raise ValueError("Saved-index replay differs from recorded A0 gold@10: " + str(deltas))
    elif report.get("mode") != "rebuilt_index_full_evaluation" or not report.get("all_ann_calibration_passed"):
        raise ValueError("Fresh evaluation must include exact references and pass ANN calibration")
    result = {"profile": profile, "observed_gold_at_10": values,
              "reference_gold_at_10": expected["gold_at_10"], "per_split_delta": deltas,
              "observed_mean": sum(values) / 3, "reference_mean": expected["mean_gold_at_10"],
              "exact_quality_match": exact,
              "interpretation": "frozen-index quality matched" if profile == "replay" else
              "fresh training/index result; numerical difference is reported, not hidden by a tolerance",
              "latency": report["summary"]["latency_ms"],
              "latency_scope": report["measurement_scope"]}
    out = work / "A0_COMPARISON.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def live_summary(work: Path) -> dict:
    report = json.loads((work / "evaluation/LIVE_RESULTS.json").read_text(encoding="utf-8"))
    if report.get("protocol") != "live-hf" or report.get("status") != "complete":
        raise ValueError("Evaluation is not a completed live-HF report")
    if report.get("all_ann_calibration_passed") is not True:
        raise ValueError("Live evaluation did not pass ANN calibration")
    result = {"profile": "live", "status": "complete",
              "n_models": report["n_models"], "n_datasets": report["n_datasets"],
              "summary": report["summary"],
              "interpretation": "Method rerun on newly collected public HF metadata and model-index evidence; not an equality check against the historical A0 experiment",
              "report": str(work / "evaluation/LIVE_RESULTS.json")}
    (work / "LIVE_SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def validate_snapshot_id(value: str) -> str:
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value)
            or value.endswith(".") or value.split(".")[0].upper() in
            {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}):
        raise ValueError("Snapshot ID must be a portable directory name, not a path")
    return value


def live_paths(args, data: Path) -> tuple[str, Path, Path]:
    current = data / "LIVE_CURRENT.json"
    create = args.command == "download" and not args.offline
    if getattr(args, "new_snapshot", False) and args.snapshot_id:
        raise ValueError("Choose --new-snapshot or --snapshot-id, not both")
    name = args.snapshot_id
    if name is None and not getattr(args, "new_snapshot", False) and current.is_file():
        name = json.loads(current.read_text(encoding="utf-8"))["snapshot_id"]
    if name is None:
        if create:
            name = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
        elif args.command == "plan":
            name = "preview"
        else:
            raise ValueError("No live HF capture is selected. Run download first or supply --snapshot-id.")
    name = validate_snapshot_id(name)
    source = (data / "live" / name).resolve()
    if not source.is_relative_to(data.resolve() / "live"):
        raise ValueError("Live snapshot directory escapes the selected data root")
    work = (args.workspace or data / "work/live" / name).expanduser().resolve()
    if create:
        data.mkdir(parents=True, exist_ok=True)
        pipeline.write_state(current, {"snapshot_id": name})
    return name, source, work


def live_main(args, data: Path):
    from . import live_snapshot
    if getattr(args, "hf_repo", None) or getattr(args, "hf_revision", None):
        raise ValueError("Live downloads enumerate the Hub API; archive --hf-repo/--hf-revision require an archived profile")
    if args.command == "doctor":
        report = doctor("live", data, args.device)
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1
    name, source, work = live_paths(args, data)
    if args.command == "plan":
        print(f"Live HF capture: {name}; source: {source}")
        print("Acquire all public model and dataset metadata pages to cursor exhaustion; supervision comes from current HF model-index records.")
        for step in pipeline.steps("live", data, work, args.device, snapshot_root=source):
            print(f"[{step.name}] {subprocess.list2cmdline(step.argv)}")
        return 0
    if args.command == "download":
        if args.offline:
            report = live_snapshot.verify(source)
        else:
            report = live_snapshot.download(source)
        print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))
        return 0 if report["ok"] else 1
    if args.command == "verify":
        report = live_snapshot.verify(source)
        print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))
        return 0 if report["ok"] else 1
    if args.command == "run":
        check = live_snapshot.verify(source)
        if not check["ok"]:
            raise ValueError("Live capture is incomplete or corrupt. Run download/verify first: " + json.dumps(check.get("errors", [])[:8]))
        environment = doctor("live", data, args.device)
        if not environment["ok"]:
            raise ValueError("Environment check failed: " + "; ".join(environment["errors"]))
        for message in environment["warnings"]:
            print("[environment] " + message, flush=True)
        pipeline.run("live", data, work, args.device, resume=args.resume,
                     input_digest=live_snapshot.input_digest(check), snapshot_root=source)
    print(json.dumps(live_summary(work), indent=2))
    return 0


def parser():
    p = argparse.ArgumentParser(prog="python -m scale1m.reproduce", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("doctor", "check dependencies and GPU/memory requirements"),
        ("download", "collect the latest public HF lake, or download an explicitly selected archive"),
        ("verify", "verify downloaded inputs without running the experiment"),
        ("plan", "show the complete command sequence without data or side effects"),
        ("run", "run the full retrieval method from verified inputs"),
        ("compare", "summarize a live run, or compare an archived A0 reproduction"),
    ]:
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--profile", choices=("live", *snapshot.PROFILES), default="live")
        sp.add_argument("--data-root", type=Path, default=None,
                        help="default: MLF_DATA_DIR or <checkout>/data")
        sp.add_argument("--workspace", type=Path, default=None,
                        help="new outputs, default: <data-root>/work/live/<snapshot-id> for live runs")
        sp.add_argument("--snapshot-id", help="live capture name; default: the capture selected by download")
        sp.add_argument("--manifest", type=Path, default=snapshot.MANIFEST,
                        help="file manifest for archived profiles; unused by live collection")
        if name in ("doctor", "plan", "run"):
            sp.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        if name == "download":
            sp.add_argument("--new-snapshot", action="store_true", help="start a new live capture instead of reusing/resuming the selected capture")
            sp.add_argument("--hf-repo", help="published dataset archive, OWNER/DATASET")
            sp.add_argument("--hf-revision", help="full 40-character dataset commit SHA")
            sp.add_argument("--offline", action="store_true", help="only verify already present inputs")
        if name == "run":
            sp.add_argument("--resume", action="store_true", help="verify completed stage outputs; resume an interrupted training checkpoint")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    data = (args.data_root or Path(os.environ.get("MLF_DATA_DIR", pipeline.REPO / "data"))).expanduser().resolve()
    work = (args.workspace or data / "work" / args.profile).expanduser().resolve()
    try:
        if args.profile == "live":
            return live_main(args, data)
        if args.snapshot_id or getattr(args, "new_snapshot", False):
            raise ValueError("--snapshot-id/--new-snapshot apply only to --profile live")
        manifest = snapshot.load_manifest(args.manifest)
        if args.command == "doctor":
            report = doctor(args.profile, data, args.device)
            print(json.dumps(report, indent=2))
            return 0 if report["ok"] else 1
        if args.command == "download":
            report = snapshot.download(data, manifest, args.profile, repo_id=args.hf_repo,
                                       revision=args.hf_revision, offline=args.offline)
        elif args.command == "verify":
            report = snapshot.verify(data, manifest, args.profile)
            print(json.dumps(report, indent=2))
            return 0 if report["ok"] else 1
        elif args.command == "plan":
            files = snapshot.selected_files(manifest, args.profile)
            print(f"A0 {args.profile}: {len(files)} input files, {sum(r['bytes'] for r in files) / 1e9:.3f} GB")
            source = manifest.get("source", {})
            print("HF source: " + (f"{source['repo_id']}@{source['revision']}" if source.get("repo_id") and source.get("revision") else "NOT CONFIGURED; historical archive publication required"))
            for step in pipeline.steps(args.profile, data, work, args.device):
                print(f"[{step.name}] {subprocess.list2cmdline(step.argv)}")
            return 0
        elif args.command == "run":
            check = snapshot.verify(data, manifest, args.profile)
            if not check["ok"]:
                raise ValueError("Required input files are missing/corrupt; run download/verify first: " + json.dumps(check["errors"][:8]))
            check = doctor(args.profile, data, args.device)
            if not check["ok"]:
                raise ValueError("Environment check failed: " + "; ".join(check["errors"]))
            for message in check["warnings"]:
                print("[environment] " + message, flush=True)
            pipeline.run(args.profile, data, work, args.device, resume=args.resume,
                         input_digest=snapshot.input_digest(manifest, args.profile))
            report = compare_results(work, args.profile)
        elif args.command == "compare":
            report = compare_results(work, args.profile)
        print(json.dumps(report, indent=2))
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print("Reproduction stopped: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
