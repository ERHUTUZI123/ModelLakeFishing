"""One entry point for downloading, rebuilding, replaying, and verifying.

Public commands are exposed through ``python -m scale1m.reproduce`` so users
do not need to reconstruct the experiment DAG from research notes.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .archive import download_parts, extract_bundle
from .verify import ASSETS_PATH, REPO_ROOT, summarize, verify, verify_assets


DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs"
BUNDLES_BY_PROFILE = {
    "inputs": ("inputs-3m-v1",),
    "replay": ("eval-3m-v1", "graph-3m-v1"),
    "full": ("inputs-3m-v1",),
    "all": ("inputs-3m-v1", "eval-3m-v1", "graph-3m-v1"),
}


def _root(value: str | None, environment: str, fallback: Path) -> Path:
    raw = value or os.environ.get(environment)
    return (Path(raw).expanduser() if raw else fallback).resolve()


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _memory_bytes() -> int | None:
    if os.name == "nt":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong),
                            ("memory_load", ctypes.c_ulong),
                            ("total_physical", ctypes.c_ulonglong),
                            ("available_physical", ctypes.c_ulonglong),
                            ("total_page_file", ctypes.c_ulonglong),
                            ("available_page_file", ctypes.c_ulonglong),
                            ("total_virtual", ctypes.c_ulonglong),
                            ("available_virtual", ctypes.c_ulonglong),
                            ("available_extended_virtual", ctypes.c_ulonglong)]
            state = MemoryStatus()
            state.length = ctypes.sizeof(state)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
                return int(state.total_physical)
        except Exception:
            return None
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def doctor(profile: str, data_root: Path, runs_root: Path) -> dict:
    packages = {name: _package_version(name) for name in (
        "numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "torch",
        "torch-geometric", "pyg-lib", "hnswlib", "sentence-transformers",
        "transformers", "huggingface-hub", "requests")}
    torch_info: dict = {}
    try:
        import torch
        torch_info = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_total_gib": (torch.cuda.get_device_properties(0).total_memory /
                              (1024 ** 3)) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        torch_info = {"error": repr(exc)}
    pyg_info: dict = {}
    try:
        from torch_geometric import typing as pyg_typing
        pyg_info = {"with_pyg_lib": bool(pyg_typing.WITH_PYG_LIB),
                    "with_torch_sparse": bool(pyg_typing.WITH_TORCH_SPARSE)}
    except Exception as exc:
        pyg_info = {"error": repr(exc)}

    disk_probe = data_root
    while not disk_probe.exists() and disk_probe != disk_probe.parent:
        disk_probe = disk_probe.parent
    free = shutil.disk_usage(disk_probe).free
    report = {
        "profile": profile,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "repository": os.fspath(REPO_ROOT),
        "data_root": os.fspath(data_root),
        "runs_root": os.fspath(runs_root),
        "ram_gib": (_memory_bytes() / (1024 ** 3)) if _memory_bytes() else None,
        "free_disk_gib": free / (1024 ** 3),
        "packages": packages,
        "torch": torch_info,
        "torch_geometric": pyg_info,
        "requirements": {
            "archive": {"python": ">=3.9", "large_download": False},
            "replay": {"python": "3.11", "free_disk_gib": 30,
                       "cuda": "recommended for exact dense controls"},
            "full": {"python": "3.11", "ram_gib": 64,
                     "recommended_ram_gib": 96, "free_disk_gib": 120,
                     "gpu_gib": 48, "pyg_lib": True},
        },
    }
    failures = []
    if profile in ("replay", "full") and sys.version_info[:2] != (3, 11):
        failures.append("Python 3.11 is required for the supported environment")
    if profile == "full":
        if report["ram_gib"] is not None and report["ram_gib"] < 64:
            failures.append("full rebuild requires at least 64 GiB RAM")
        if report["free_disk_gib"] < 120:
            failures.append("full rebuild requires at least 120 GiB free disk")
        if not torch_info.get("cuda_available"):
            failures.append("full rebuild requires a CUDA GPU")
        elif (torch_info.get("gpu_total_gib") or 0) < 44:
            failures.append("full rebuild requires an approximately 48 GiB GPU")
        if not pyg_info.get("with_pyg_lib"):
            failures.append("full rebuild requires the pyg-lib sampling backend")
    if profile == "replay" and report["free_disk_gib"] < 30:
        failures.append("artifact replay requires at least 30 GiB free disk")
    report["failures"] = failures
    report["ok"] = not failures
    return report


def download(profile: str, data_root: Path, asset_dir: Path, gh: str,
             offline: bool, force: bool) -> None:
    assets = _json(ASSETS_PATH)
    if not assets.get("bundles"):
        raise SystemExit("release assets are not declared in repro/assets.json")
    repository, tag = assets["repository"], assets["release_tag"]
    for bundle_id in BUNDLES_BY_PROFILE[profile]:
        bundle = assets["bundles"].get(bundle_id)
        if bundle is None:
            raise SystemExit("release does not declare bundle %s" % bundle_id)
        manifest = _json(REPO_ROOT / Path(*bundle["manifest"].split("/")))
        print("\n[bundle] %s" % bundle_id, flush=True)
        if not offline:
            download_parts(repository, tag, bundle, asset_dir, gh=gh)
        extract_bundle(bundle, manifest, asset_dir, data_root, force=force)
        print("[verified] %s (%d logical files)" %
              (bundle_id, len(manifest["files"])), flush=True)


@dataclass(frozen=True)
class Step:
    name: str
    argv: tuple[str, ...]
    complete: Callable[[], bool]


def _complete_json(path: Path, *, stage: str | None = None,
                   all_gates: bool = False,
                   required: tuple[tuple[str, ...], ...] = ()) -> Callable[[], bool]:
    def predicate() -> bool:
        if not path.is_file():
            return False
        try:
            payload = _json(path)
        except Exception:
            return False
        if stage is not None and payload.get("stage") != stage:
            return False
        for keys in required:
            value = payload
            try:
                for key in keys:
                    value = value[key]
            except (KeyError, TypeError):
                return False
        if all_gates:
            gates = payload.get("gates", {})
            return bool(gates) and all(bool(value) for value in gates.values())
        return True
    return predicate


def _always() -> bool:
    return False


def _module(*args: str) -> tuple[str, ...]:
    return (sys.executable, "-m", *args)


def _evaluation_steps(data_root: Path, device: str, *, full: bool) -> list[Step]:
    d = data_root / "data1m"
    exports = d / "exports_x4"
    graph = d / "graphs" / "hgraph_rf"
    ladder = d / "ladder_rf" / "full_model_ids.parquet"
    nodes = d / "rf" / "canon" / "dataset_nodes_merged.parquet"
    x5 = d / "reproduced" / "x5"
    x6 = d / "reproduced" / "x6"
    y2 = d / "reproduced" / "y2"
    y4 = d / "reproduced" / "y4"
    sidecar_exports = exports
    y2_exact_stage = "exact" if full else "replay-exact"
    y4_exact_stage = "exact" if full else "replay-exact"
    y2_pool_args = (() if full else (
        "--frozen-pools", os.fspath(d / "frozen_replay" / "y2")))
    y4_pool_args = (() if full else (
        "--frozen-pools", os.fspath(d / "frozen_replay" / "y4")))
    x5_argv = (_module(
        "scale1m.eval_rf", "--axis", "e", "--exports", os.fspath(exports),
        "--run-fmt", "X4GD_full_s%d_e25", "--graph", os.fspath(graph),
        "--ladder", os.fspath(ladder), "--dataset-nodes", os.fspath(nodes),
        "--out", os.fspath(x5), "--device", device)
        if full else _module(
            "scale1m.replay_x5", "--exports", os.fspath(exports),
            "--run-fmt", "X4GD_full_s%d_e25", "--dataset-nodes",
            os.fspath(nodes), "--frozen-pools",
            os.fspath(d / "frozen_replay" / "y2"), "--out", os.fspath(x5)))
    steps = [
        Step("x5", x5_argv,
             _complete_json(x5 / "X5_QUERY_ELIGIBILITY.json")),
        Step("x6", _module("scale1m.eval_x6", "--stage", "training-free",
             "--exports", os.fspath(exports), "--dataset-nodes", os.fspath(nodes),
             "--baseline-attrs", os.fspath(d / "baseline_rf" /
                                            "baseline_attrs.parquet"),
             "--ladder", os.fspath(ladder), "--graph", os.fspath(graph),
             "--out", os.fspath(x6), "--device", device),
             _complete_json(x6 / "X6_BASELINES.json",
                            stage="training-free-complete", all_gates=True)),
        Step("y2-exact", _module("scale1m.eval_y2", "--stage", y2_exact_stage,
             "--exports", os.fspath(exports), "--sidecar-exports",
             os.fspath(sidecar_exports), "--sidecar-run-fmt",
             "X4GD_full_s%d_e25", "--dataset-nodes", os.fspath(nodes),
             "--out", os.fspath(y2), "--device", device, *y2_pool_args),
             _complete_json(y2 / "Y2_REPORT.json",
                            required=(("summary", "G_exact1000_task"),))),
        Step("y2-hnsw", _module("scale1m.eval_y2", "--stage", "hnsw",
             "--exports", os.fspath(exports), "--sidecar-exports",
             os.fspath(sidecar_exports), "--sidecar-run-fmt",
             "X4GD_full_s%d_e25", "--dataset-nodes", os.fspath(nodes),
             "--out", os.fspath(y2), "--device", device),
             _complete_json(y2 / "Y2_REPORT.json", stage="complete",
                            required=(("hnsw", "2"),))),
        Step("y2-finalize", _module("scale1m.eval_y2", "--stage", "finalize",
             "--exports", os.fspath(exports), "--sidecar-exports",
             os.fspath(sidecar_exports), "--sidecar-run-fmt",
             "X4GD_full_s%d_e25", "--dataset-nodes", os.fspath(nodes),
             "--out", os.fspath(y2), "--device", device),
             _complete_json(y2 / "Y2_REPORT.json", stage="complete", all_gates=True,
                            required=(("hnsw_summary", "gold@10"),))),
        Step("y4-exact", _module("scale1m.eval_y4", "--stage", y4_exact_stage,
             "--exports", os.fspath(exports), "--sidecar-exports",
             os.fspath(sidecar_exports), "--sidecar-run-fmt",
             "X4GD_full_s%d_e25", "--dataset-nodes", os.fspath(nodes),
             "--y2-out", os.fspath(y2), "--y2-report",
             os.fspath(y2 / "Y2_REPORT.json"), "--out", os.fspath(y4),
             "--device", device, *y4_pool_args),
             _complete_json(y4 / "Y4_REPORT.json",
                            required=(("exact", "2"),))),
        Step("y4-hnsw", _module("scale1m.eval_y4", "--stage", "hnsw",
             "--exports", os.fspath(exports), "--sidecar-exports",
             os.fspath(sidecar_exports), "--sidecar-run-fmt",
             "X4GD_full_s%d_e25", "--dataset-nodes", os.fspath(nodes),
             "--y2-out", os.fspath(y2), "--y2-report",
             os.fspath(y2 / "Y2_REPORT.json"), "--out", os.fspath(y4),
             "--device", device),
             _complete_json(y4 / "Y4_REPORT.json",
                            required=(("hnsw", "2"),))),
        Step("y4-finalize", _module("scale1m.eval_y4", "--stage", "finalize",
             "--exports", os.fspath(exports), "--sidecar-exports",
             os.fspath(sidecar_exports), "--sidecar-run-fmt",
             "X4GD_full_s%d_e25", "--dataset-nodes", os.fspath(nodes),
             "--y2-out", os.fspath(y2), "--y2-report",
             os.fspath(y2 / "Y2_REPORT.json"), "--out", os.fspath(y4),
             "--docs-out", os.fspath(y4 / "archived_copy"), "--device", device),
             _complete_json(y4 / "Y4_REPORT.json", stage="complete", all_gates=True,
                            required=(("curve", "1000"),))),
    ]
    return steps


def _full_steps(data_root: Path, runs_root: Path, device: str) -> list[Step]:
    d = data_root / "data1m"
    rf, ladder = d / "rf", d / "ladder_rf"
    feats, graph = d / "feats_rf", d / "graphs" / "hgraph_rf"
    cards = d / "datasets_full" / "dataset_cards_merged.parquet"
    candidates = d / "candidates_full"
    nodes = rf / "canon" / "dataset_nodes_merged.parquet"
    steps = [
        Step("canonicalize", _module("scale1m.canonicalize_rf", "--candidates",
             os.fspath(candidates), "--out", os.fspath(rf)),
             _complete_json(rf / "F2_REPORT.json")),
        Step("merge", _module("scale1m.merge_supervision", "--rf", os.fspath(rf),
             "--candidates", os.fspath(candidates), "--source-dir",
             os.fspath(d / "historical_graphs")),
             _complete_json(rf / "F2_MERGE_REPORT.json")),
        Step("dataset-cards", _module("scale1m.match_dataset_cards", "--nodes",
             os.fspath(nodes), "--datasets", os.fspath(d / "datasets_full"),
             "--out", os.fspath(cards)),
             _complete_json(cards.with_name(cards.stem + "_REPORT.json"))),
        Step("ladder", _module("scale1m.build_ladder_rf", "--rf", os.fspath(rf),
             "--out", os.fspath(ladder)),
             _complete_json(ladder / "LADDER_REPORT.json")),
        Step("features", _module("scale1m.embed_lake_rf", "--ladder",
             os.fspath(ladder), "--out", os.fspath(feats), "--device", device),
             _complete_json(feats / "FEATS_REPORT.json", all_gates=True)),
        Step("graph", _module("scale1m.build_graph_rf", "--ladder",
             os.fspath(ladder), "--feats", os.fspath(feats), "--rf", os.fspath(rf),
             "--cards", os.fspath(cards), "--out", os.fspath(graph),
             "--device", device),
             _complete_json(graph / "GRAPH_REPORT.json", all_gates=True)),
        Step("split-audit", _module("scale1m.audit_rf_splits", "--graph",
             os.fspath(graph), "--out", os.fspath(graph / "f5_splits.json")),
             _complete_json(graph / "f5_splits.json")),
        Step("baseline-sidecar", _module("scale1m.baseline_sidecar", "--candidates",
             os.fspath(candidates), "--ladder", os.fspath(ladder /
                                                          "full_model_ids.parquet"),
             "--out", os.fspath(d / "baseline_rf")),
             _complete_json(d / "baseline_rf" / "SIDECAR_REPORT.json")),
    ]
    for seed in range(3):
        run = runs_root / ("X4GD_full_s%d_e25" % seed)
        export = d / "exports_x4" / ("X4GD_full_s%d_e25" % seed)
        steps.append(Step(
            "train-s%d" % seed,
            _module("scale1m.train_rung", "--rung", "full", "--graph",
                    os.fspath(graph), "--out", os.fspath(run), "--seed", str(seed),
                    "--epochs", "25", "--family-vocab",
                    os.fspath(feats / "family_vocab.csv"), "--fanout", "--sparse-M",
                    "--contrast-n-neg", "256", "--chunked-infer", "50000",
                    "--skip-diagnostics", "--lake-gamma", "0.5",
                    "--global-n-datasets", "128", "--device", device),
            _complete_json(run / "MANIFEST.json")))
        steps.append(Step(
            "export-embed-s%d" % seed,
            _module("scale1m.export_rf", "--run", os.fspath(run), "--stage", "embed",
                    "--graph", os.fspath(graph), "--ladder",
                    os.fspath(ladder / "full_model_ids.parquet"), "--out",
                    os.fspath(export), "--device", device, "--progress"),
            _complete_json(export / "EXPORT_MANIFEST.json",
                           required=(("stages", "embed"),))))
        steps.append(Step(
            "export-metrics-s%d" % seed,
            _module("scale1m.export_rf", "--run", os.fspath(run), "--stage", "metrics",
                    "--graph", os.fspath(graph), "--ladder",
                    os.fspath(ladder / "full_model_ids.parquet"), "--out",
                    os.fspath(export), "--device", device),
            _complete_json(export / "EXPORT_MANIFEST.json",
                           required=(("stages", "metrics"),))))
        steps.append(Step(
            "prior-s%d" % seed,
            _module("stage3HNSW.build_prior_sidecar", "--graph-store",
                    os.fspath(graph), "--export", os.fspath(export), "--split-seed",
                    str(seed), "--task-nodes", os.fspath(nodes), "--out",
                    os.fspath(export / ("prior_sidecar_s%d.npz" % seed))),
            lambda p=export / ("prior_sidecar_s%d.npz" % seed): p.is_file()))
    return steps + _evaluation_steps(data_root, device, full=True)


def _matches(name: str, selected: set[str]) -> bool:
    if not selected:
        return True
    return name in selected or name.split("-")[0] in selected


def run(profile: str, data_root: Path, runs_root: Path, device: str,
        selected: set[str], dry_run: bool, force: bool) -> int:
    if profile == "replay":
        if not dry_run:
            input_checks = verify_assets("replay", data_root)
            if any(not row["ok"] for row in input_checks):
                print(json.dumps(input_checks, indent=2), file=sys.stderr)
                raise SystemExit("replay bundles are missing or corrupt; run download first")
        steps = _evaluation_steps(data_root, device, full=False)
    elif profile == "full":
        if not dry_run:
            input_checks = verify_assets("full", data_root)
            if any(not row["ok"] for row in input_checks):
                print(json.dumps(input_checks, indent=2), file=sys.stderr)
                raise SystemExit("full-rebuild input bundle is missing or corrupt")
        steps = _full_steps(data_root, runs_root, device)
    else:
        raise ValueError(profile)

    env = os.environ.copy()
    env["MLF_DATA_DIR"] = os.fspath(data_root)
    env["MLF_RUNS_DIR"] = os.fspath(runs_root)
    parent = os.fspath(REPO_ROOT.parent)
    env["PYTHONPATH"] = parent + (os.pathsep + env["PYTHONPATH"]
                                  if env.get("PYTHONPATH") else "")
    record = {"profile": profile, "started_at_unix": time.time(),
              "python": sys.executable, "data_root": os.fspath(data_root),
              "runs_root": os.fspath(runs_root), "steps": []}
    for step in steps:
        if not _matches(step.name, selected):
            continue
        command = subprocess.list2cmdline(step.argv)
        if not force and step.complete():
            print("[skip complete] %s" % step.name, flush=True)
            record["steps"].append({"name": step.name, "status": "already-complete",
                                    "command": command})
            continue
        print("[%s] %s" % ("dry-run" if dry_run else "run", command), flush=True)
        if dry_run:
            record["steps"].append({"name": step.name, "status": "dry-run",
                                    "command": command})
            continue
        started = time.time()
        subprocess.run(step.argv, cwd=REPO_ROOT, env=env, check=True)
        record["steps"].append({"name": step.name, "status": "completed",
                                "seconds": time.time() - started,
                                "command": command})
    record["finished_at_unix"] = time.time()
    if not dry_run:
        out = data_root / "data1m" / "reproduced" / "reproduction_commands.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scale1m.reproduce",
        description="Reproduce the fixed 3M ModelLakeFishing evidence chain")
    parser.add_argument("--data-root", default=None,
                        help="default: $MLF_DATA_DIR or <repository>/data")
    parser.add_argument("--runs-root", default=None,
                        help="default: $MLF_RUNS_DIR or <repository>/runs")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="inspect hardware and dependencies")
    p.add_argument("--profile", choices=("archive", "replay", "full"),
                   default="archive")

    p = sub.add_parser("download", help="download and hash-check frozen bundles")
    p.add_argument("--profile", choices=tuple(BUNDLES_BY_PROFILE), required=True)
    p.add_argument("--asset-dir", default=None,
                   help="part cache (default: <data-root>/.downloads/<release-tag>)")
    p.add_argument("--gh", default="gh", help="GitHub CLI executable")
    p.add_argument("--offline", action="store_true",
                   help="use already-downloaded parts; do not invoke gh")
    p.add_argument("--force", action="store_true",
                   help="replace only nonmatching files within declared bundle paths")

    p = sub.add_parser("run", help="execute the replay or full-rebuild DAG")
    p.add_argument("--profile", choices=("replay", "full"), required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--only", nargs="*", default=[],
                   help="step/group names, e.g. x6 y2 y4 or train-s0")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("verify", help="validate reports against registered results")
    p.add_argument("--profile", choices=("archive", "inputs", "replay", "full"),
                   required=True)
    p.add_argument("--output", default=None,
                   help="default: <data-root>/reproduction_report.<profile>.json")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    data_root = _root(args.data_root, "MLF_DATA_DIR", DEFAULT_DATA_ROOT)
    runs_root = _root(args.runs_root, "MLF_RUNS_DIR", DEFAULT_RUNS_ROOT)
    if args.command == "doctor":
        report = doctor(args.profile, data_root, runs_root)
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1
    if args.command == "download":
        assets = _json(ASSETS_PATH)
        asset_dir = (Path(args.asset_dir).expanduser().resolve()
                     if args.asset_dir else data_root / ".downloads" /
                     assets["release_tag"])
        download(args.profile, data_root, asset_dir, args.gh,
                 args.offline, args.force)
        return 0
    if args.command == "run":
        return run(args.profile, data_root, runs_root, args.device,
                   set(args.only), args.dry_run, args.force)
    if args.command == "verify":
        output = (Path(args.output).expanduser().resolve() if args.output else
                  data_root / ("reproduction_report.%s.json" % args.profile))
        report = verify(args.profile, data_root, runs_root, output)
        print(json.dumps(report["summary"], indent=2))
        if report["summary"]["strict_failed"]:
            for row in report["checks"]:
                if not row["ok"] and row.get("severity", "error") == "error":
                    print("[FAIL] %s: %s" % (row["id"], row.get("error", row)),
                          file=sys.stderr)
        print("report: %s" % output)
        return 0 if report["summary"]["strict_failed"] == 0 else 1
    raise AssertionError(args.command)
