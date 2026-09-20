from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .snapshot import sha256

REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Step:
    name: str
    argv: tuple[str, ...]
    outputs: tuple[Path, ...]


def module(*args) -> tuple[str, ...]:
    return (sys.executable, "-m", *(str(x) for x in args))


def live_steps(source: Path, work: Path, device: str) -> list[Step]:
    rf, ladder, feats = work / "rf", work / "ladder", work / "features"
    graph, exports = work / "graph", work / "exports"
    nodes = rf / "canon/dataset_nodes_merged.parquet"
    cards = work / "dataset_cards.parquet"
    plan = [
        Step("canonicalize", module("scale1m.canonicalize_rf", "--candidates", source / "models", "--out", rf), (rf,)),
        Step("merge-hf-evidence", module("scale1m.merge_supervision", "--rf", rf, "--candidates", source / "models", "--hf-only"), (rf,)),
        Step("check-canonical", module("scale1m.reproduction.live_stages", "canonical", "--rf", rf), ()),
        Step("dataset-cards", module("scale1m.match_dataset_cards", "--nodes", nodes, "--datasets", source / "datasets", "--out", cards), (cards,)),
        Step("ladder", module("scale1m.build_ladder_rf", "--rf", rf, "--out", ladder, "--hf-only"), (ladder,)),
        Step("features", module("scale1m.embed_lake_rf", "--ladder", ladder / "full_model_ids.parquet", "--out", feats, "--device", device), (feats,)),
        Step("graph", module("scale1m.build_graph_rf", "--ladder", ladder, "--feats", feats, "--rf", rf, "--cards", cards, "--out", graph, "--device", device), (graph,)),
        Step("check-graph", module("scale1m.reproduction.live_stages", "graph", "--graph", graph, "--rf", rf, "--ladder", ladder, "--out", work / "LIVE_UNIVERSE.json"), (work / "LIVE_UNIVERSE.json",)),
    ]
    for seed in range(3):
        name = f"LIVE_full_s{seed}_e25"
        run, export = work / "runs" / name, exports / name
        plan.extend([
            Step(f"train-s{seed}", module(
                "scale1m.train_rung", "--rung", "live", "--graph", graph,
                "--out", run, "--seed", seed, "--epochs", 25,
                "--family-vocab", feats / "family_vocab.csv", "--fanout", "--sparse-M",
                "--contrast-n-neg", 256, "--chunked-infer", 50000,
                "--skip-diagnostics", "--lake-gamma", 0.5,
                "--global-n-datasets", 128, "--device", device
            ), (run,)),
            Step(f"export-s{seed}", module(
                "scale1m.export_rf", "--run", run, "--stage", "embed", "--ckpt", "last",
                "--graph", graph, "--ladder", ladder / "full_model_ids.parquet", "--chunk", 50000,
                "--out", export, "--device", device, "--progress"
            ), (export,)),
            Step(f"prior-s{seed}", module(
                "stage3HNSW.build_prior_sidecar", "--graph-store", graph,
                "--export", export, "--split-seed", seed, "--task-nodes", nodes
            ), (export / f"prior_sidecar_s{seed}.npz", export / f"prior_sidecar_s{seed}_meta.json")),
        ])
    plan.append(Step("evaluate", module(
        "scale1m.evaluate_live", "--exports", exports, "--graph", graph,
        "--dataset-nodes", nodes, "--out", work / "evaluation", "--device", device
    ), (work / "evaluation",)))
    return plan


def steps(profile: str, data: Path, work: Path, device: str, snapshot_root=None) -> list[Step]:
    if profile == "live":
        if snapshot_root is None:
            raise ValueError("Live pipeline requires an explicit captured snapshot directory")
        return live_steps(Path(snapshot_root), work, device)
    d = data / "data1m"
    frozen = d / "frozen"
    output = work / "evaluation"
    if profile == "replay":
        return [Step("evaluate-replay", module(
            "scale1m.evaluate_a0_portable", "--exports", d / "a0_reference/exports",
            "--dataset-nodes", frozen / "dataset_nodes_merged.parquet", "--out", output,
            "--device", "cpu", "--final-only", "--reference-index-dir", d / "a0_reference/metrics"
        ), (output,))]
    graph = work / "graph"
    exports = work / "exports"
    plan = []
    if profile == "full":
        rf, ladder, feats = work / "rf", work / "ladder", work / "features"
        nodes = rf / "canon/dataset_nodes_merged.parquet"
        cards = work / "dataset_cards.parquet"
        sources = d / "source_evidence"
        plan = [
            Step("canonicalize", module("scale1m.canonicalize_rf", "--candidates", d / "candidates_full", "--out", rf), (rf,)),
            Step("merge", module("scale1m.merge_supervision", "--rf", rf, "--candidates", d / "candidates_full", "--source-dir", sources), (rf,)),
            Step("check-canonical", module("scale1m.reproduction.stages", "canonical", "--rf", rf, "--frozen", frozen), ()),
            Step("dataset-cards", module("scale1m.match_dataset_cards", "--nodes", nodes, "--datasets", d / "datasets_full", "--out", cards), (cards,)),
            Step("ladder", module("scale1m.build_ladder_rf", "--rf", rf, "--out", ladder, "--source-dir", sources), (ladder,)),
            Step("seed-vocabulary", module("scale1m.reproduction.stages", "seed-vocab", "--source", frozen / "family_vocab.csv", "--out", feats / "family_vocab.csv"), (feats / "family_vocab.csv",)),
            Step("features", module("scale1m.embed_lake_rf", "--ladder", ladder / "full_model_ids.parquet", "--out", feats, "--device", device), (feats,)),
            Step("graph", module("scale1m.build_graph_rf", "--ladder", ladder, "--feats", feats, "--rf", rf, "--cards", cards, "--out", graph, "--device", device), (graph,)),
            Step("check-graph", module("scale1m.reproduction.stages", "graph", "--graph", graph, "--vocab", feats / "family_vocab.csv", "--frozen-vocab", frozen / "family_vocab.csv"), ()),
        ]
        ladder_file, vocab_file = ladder / "full_model_ids.parquet", feats / "family_vocab.csv"
    elif profile == "train":
        plan = [Step("prepare-frozen-graph", module(
            "scale1m.reproduction.preparation", "--source", frozen / "hgraph_rf",
            "--binding-dir", frozen / "a0_graph_bindings", "--out", graph
        ), (graph, graph.with_name("graph.reconstruction.json")))]
        nodes, ladder_file, vocab_file = (frozen / "dataset_nodes_merged.parquet",
                                        frozen / "full_model_ids.parquet", frozen / "family_vocab.csv")
    else:
        raise ValueError(profile)
    for seed in range(3):
        name = f"A0GD_full_s{seed}_e25"
        run, export = work / "runs" / name, exports / name
        plan.extend([
            Step(f"train-s{seed}", module(
                "scale1m.train_rung", "--rung", "full", "--graph", graph,
                "--out", run, "--seed", seed, "--epochs", 25,
                "--family-vocab", vocab_file, "--fanout", "--sparse-M",
                "--contrast-n-neg", 256, "--chunked-infer", 50000,
                "--skip-diagnostics", "--lake-gamma", 0.5,
                "--global-n-datasets", 128, "--device", device
            ), (run,)),
            Step(f"export-s{seed}", module(
                "scale1m.export_rf", "--run", run, "--stage", "embed", "--ckpt", "last",
                "--graph", graph, "--ladder", ladder_file, "--chunk", 50000,
                "--out", export, "--device", device, "--progress"
            ), (export,)),
            Step(f"prior-s{seed}", module(
                "stage3HNSW.build_prior_sidecar", "--graph-store", graph,
                "--export", export, "--split-seed", seed, "--task-nodes", nodes
            ), (export / f"prior_sidecar_s{seed}.npz", export / f"prior_sidecar_s{seed}_meta.json")),
        ])
    plan.append(Step("evaluate", module(
        "scale1m.evaluate_a0_portable", "--exports", exports,
        "--dataset-nodes", nodes, "--out", output, "--device", device,
        *(("--rebuilt-graph", graph) if profile == "full" else ())
    ), (output,)))
    return plan


def implementation_digest() -> str:
    digest = hashlib.sha256()
    for directory in ("scale1m", "scale", "stage1BuildTransferGraph", "stage2TrainGraphSAGE", "stage3HNSW"):
        for path in sorted((REPO / directory).rglob("*.py")):
            if any(p in ("__pycache__", "tests", "data", "artifacts", "legacy") for p in path.relative_to(REPO).parts):
                continue
            digest.update(path.relative_to(REPO).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def output_hashes(paths: tuple[Path, ...], work: Path) -> dict:
    result = {}
    for path in paths:
        if not path.exists():
            raise ValueError("Stage returned success without its output: " + str(path))
        for file in sorted(path.rglob("*")) if path.is_dir() else [path]:
            if file.is_file():
                if not file.resolve().is_relative_to(work.resolve()):
                    raise ValueError("Stage output escaped workspace")
                result[file.relative_to(work).as_posix()] = sha256(file)
    return result


def write_state(path: Path, state: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(profile: str, data: Path, work: Path, device: str, *, resume=False, input_digest=None, snapshot_root=None):
    data, work = data.resolve(), work.resolve()
    if (work == data or work.is_relative_to(data / "data1m")
            or work.is_relative_to(data / "live") or data.is_relative_to(work)):
        raise ValueError("Workspace must be separate from downloaded data1m inputs and live snapshots")
    plan = (steps(profile, data, work, device, snapshot_root=snapshot_root)
            if profile == "live" else steps(profile, data, work, device))
    identity = {"profile": profile, "data_root": str(data), "workspace": str(work),
                "device": device, "implementation_sha256": implementation_digest(),
                "input_sha256": input_digest,
                "commands": [list(s.argv) for s in plan]}
    state_path = work / ("LIVE_REPRODUCTION.json" if profile == "live" else "A0_REPRODUCTION.json")
    if work.exists() and any(work.iterdir()):
        if not resume or not state_path.is_file():
            raise ValueError("Workspace is not empty. Use --resume for this run, or a new --workspace.")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["identity"] != identity:
            raise ValueError("Cannot resume with changed code, inputs, paths, or commands")
        current_outputs = {}
        for step in plan:
            previous = state["steps"].get(step.name, {})
            if previous.get("status") == "complete":
                current_outputs.update(previous["outputs"])
        for name, expected in current_outputs.items():
            path = work / name
            if not path.is_file() or sha256(path) != expected:
                raise ValueError("Completed stage output changed: " + str(path))
    else:
        work.mkdir(parents=True, exist_ok=True)
        state = {"identity": identity, "started_at_unix": time.time(), "steps": {}, "status": "running"}
    env = os.environ.copy()
    env.update(MLF_DATA_DIR=str(data), MLF_RUNS_DIR=str(work / "runs"),
               PYTHONUNBUFFERED="1", PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    env["HF_HOME"] = str(data / ".hf_home")
    for step in plan:
        previous = state["steps"].get(step.name, {})
        if previous.get("status") == "complete":
            print("[verified complete] " + step.name, flush=True)
            continue
        argv = list(step.argv)
        if resume and previous and step.name.startswith("train-"):
            last = step.outputs[0] / "ckpt/last.pt"
            if last.is_file():
                argv.extend(["--resume", str(last)])
        print("[run] " + subprocess.list2cmdline(argv), flush=True)
        state["steps"][step.name] = {"status": "running", "started_at_unix": time.time()}
        state["status"] = "running"
        write_state(state_path, state)
        try:
            subprocess.run(argv, cwd=REPO, env=env, check=True)
            hashes = output_hashes(step.outputs, work)
        except BaseException:
            state["steps"][step.name]["status"] = "failed"
            state["status"] = "failed"
            write_state(state_path, state)
            raise
        state["steps"][step.name].update(status="complete", outputs=hashes, finished_at_unix=time.time())
        write_state(state_path, state)
    state.update(status="complete", finished_at_unix=time.time())
    write_state(state_path, state)
    return state_path
