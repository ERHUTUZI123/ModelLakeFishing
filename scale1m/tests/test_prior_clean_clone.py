"""The prior command must work independently in an arbitrarily named clone."""
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import pytest

pytest.importorskip("torch_geometric", reason="Graph/prior subprocess tests require the train/full environment")
from scale1m.tests.test_prepare_a0_graph import tiny_source


def test_prior_subprocess_without_a0_marker_or_author_checkout(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    clone = tmp_path / "reviewer-chosen-name"
    # Only runtime code, with no parent checkout, docs, repro or author data.
    required = [
        "scale1m/__init__.py", "scale1m/graph_store.py",
        "stage3HNSW/build_prior_sidecar.py",
        "stage2TrainGraphSAGE/d0_splits.py", "stage2TrainGraphSAGE/losses.py",
        "stage1BuildTransferGraph/dataset_embed/utils/CustomRandomLinkSplit.py",
    ]
    for name in required:
        target = clone / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / name, target)
    graph, exports = tmp_path / "fixture-graph", tmp_path / "exports"
    tiny_source(graph)
    assert "a0" not in json.loads((graph / "meta.json").read_text(encoding="utf-8"))
    exports.mkdir()
    shutil.copy2(graph / "unique_model_id.parquet", exports / "model_ids.parquet")
    shutil.copy2(graph / "unique_dataset_id.parquet", exports / "dataset_ids.parquet")
    graph_meta = json.loads((graph / "meta.json").read_text(encoding="utf-8"))
    graph_digest = hashlib.sha256(json.dumps(graph_meta["files"], sort_keys=True,
                                            ensure_ascii=False).encode("utf-8")).hexdigest()
    export_meta = {"stages": {"embed": {"graph_digest": graph_digest, "split_seed": 0,
        "binding": {"graph_sha256": graph_digest, "split_seed": 0},
        "artifact_hashes": {name: hashlib.sha256((exports / name).read_bytes()).hexdigest()
                            for name in ("model_ids.parquet", "dataset_ids.parquet")}}}}
    (exports / "EXPORT_MANIFEST.json").write_text(json.dumps(export_meta), encoding="utf-8")
    nodes = tmp_path / "task-nodes.parquet"
    pd.DataFrame({"node": ["r/a\tt1", "r/b\tt2", "s/c\tt1"],
                  "task": ["t1", "t2", "t1"]}).to_parquet(nodes, index=False)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([
        sys.executable, "-m", "stage3HNSW.build_prior_sidecar",
        "--graph-store", str(graph), "--export", str(exports),
        "--split-seed", "0", "--task-nodes", str(nodes),
    ], cwd=clone, env=env, capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    metadata = json.loads((exports / "prior_sidecar_s0_meta.json").read_text(encoding="utf-8"))
    assert metadata["n_models"] == 7
    assert metadata["n_datasets"] == 3
    assert metadata["split_seed"] == 0
    assert metadata["held_out_datasets"] > 0
    assert metadata["n_edges"] < metadata["n_edges_in_graph"] == 4
    assert "a0" not in metadata
    assert metadata["graph_digest"] == graph_digest
    assert metadata["sidecar_sha256"] == hashlib.sha256((exports / "prior_sidecar_s0.npz").read_bytes()).hexdigest()


def test_training_deferred_imports_in_arbitrarily_named_clean_clone(tmp_path):
    for dependency in ("matplotlib", "scipy", "sklearn", "huggingface_hub"):
        pytest.importorskip(dependency, reason="Deferred training imports require the full environment")
    repo = Path(__file__).resolve().parents[2]
    clone = tmp_path / "independent-reviewer-checkout"
    for directory in ("scale1m", "scale", "stage2TrainGraphSAGE",
                      "stage1BuildTransferGraph/dataset_embed",
                      "stage1BuildTransferGraph/dataset_embed/utils"):
        for source in (repo / directory).glob("*.py"):
            target = clone / source.relative_to(repo)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    code = """
import argparse
import importlib
from pathlib import Path
import scale1m

root = Path.cwd().resolve()
modules = [
    "ModelLakeFishing.scale.export_ours",
    "ModelLakeFishing.stage2TrainGraphSAGE.ablation",
    "ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery",
    "ModelLakeFishing.stage2TrainGraphSAGE.top1_audit",
    "ModelLakeFishing.stage2TrainGraphSAGE.d0_splits",
    "ModelLakeFishing.stage2TrainGraphSAGE.inference",
    "ModelLakeFishing.stage2TrainGraphSAGE.top1_eval",
    "scale1m.embed_lake_rf", "scale.modellens_build_graph",
    "dataset_embed.xm0_builder",
    "ModelLakeFishing.stage2TrainGraphSAGE.learnable",
]
for name in modules:
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(root), name
from scale1m.train_rung import build_config
args = argparse.Namespace(fanout=True, sparse_M=True, contrast_n_neg=256,
    contrast_max_pos_per_dataset=None, chunked_infer=50000,
    skip_diagnostics=True, batch_size=None, lake_gamma=0.5,
    global_n_datasets=128, num_layers=None)
config = build_config(args)
assert len(config) == 40
assert config["lake_gamma"] == 0.5 and config["global_n_datasets"] == 128
print("training dependencies and the 40-key A0 configuration passed")
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], cwd=clone, env=env,
                            capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "40-key A0 configuration passed" in result.stdout
