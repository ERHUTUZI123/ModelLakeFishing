"""
test_export_contract.py -- Stage-3 Phase 1 gate G-A, on BOTH graphs
(diverse 306 + hf1000d 2,000 / G2):

  1. export determinism: two independent exports of the same (graph, ckpt)
     produce bitwise-identical z_m.npy / z_d.npy;
  2. ids alignment: model_ids/dataset_ids row i has mappedID i, the unique-id
     column matches the graph payload snapshot, and z row counts agree with
     the id files and the meta sidecar;
  3. normalization: every z_m / z_d row has unit L2 norm (float32, atol 1e-5);
  4. hub occupancy: deterministic int32 sidecar, row i == model mappedID i,
     counts sum to N_d * 10, and tie-breaking is mappedID ascending;
  5. manifest completeness: schema-required keys all present, every recorded
     file sha256 matches a fresh recompute, graph/ckpt hashes match disk, and
     the recorded surgery matches the checkpoint's repro metadata.

Run:  python -m ModelLakeFishing.stage3HNSW.tests.test_export_contract
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage3HNSW.export_embeddings import (  # noqa: E402
    HUB_OCCUPANCY_TOP_K, _compute_hub_occupancy, export, sha256_file,
)

MLF = os.path.join(_REPO_ROOT, "ModelLakeFishing")
CASES = {
    "diverse_candidate": (
        os.path.join(MLF, "stage1BuildTransferGraph", "hgraph_diverse_xd0.pt"),
        os.path.join(MLF, "stage2TrainGraphSAGE", "artifacts",
                     "stage2_diverse_xd0_candidate.pt")),
    "hf1000d_G2": (
        os.path.join(MLF, "stage1BuildTransferGraph", "hgraph_hf1000d_2000m_xm0_xd0.pt"),
        os.path.join(MLF, "stage2TrainGraphSAGE", "artifacts", "ablation", "top1",
                     "ckpt", "G2_s0_i0.pt")),
}

MANIFEST_SCHEMA = {
    "export_code_version": str, "created_utc": str,
    "graph": {"path": str, "sha256": str, "num_models": int, "num_datasets": int},
    "checkpoint": {"path": str, "sha256": str},
    "vocab": {"family_csv": {"path": str, "sha256": str, "rows": int}},
    "surgery": {"dedup_trained_on": bool, "similar_to_mode": str, "similar_to_k": int},
    "embedding": {"dim": int, "dtype": str, "normalized": bool},
    "hub_occupancy": {"file": str, "dtype": str, "shape": list, "top_k": int,
                      "num_dataset_queries": int, "score": str, "tie_break": str,
                      "query_batch_size": int},
    "encoders": dict, "versions": dict,
    "verification": {"forward_determinism_bitwise": bool, "ids_row_order_ok": bool,
                     "spotcheck_rows": int, "spotcheck_ok": bool},
    "files": dict,
}


def _check_schema(node, schema, crumb):
    for key, want in schema.items():
        assert key in node, f"manifest missing {crumb}{key}"
        if isinstance(want, dict):
            _check_schema(node[key], want, f"{crumb}{key}.")
        else:
            assert isinstance(node[key], want), (
                f"manifest {crumb}{key}: expected {want.__name__}, "
                f"got {type(node[key]).__name__}")


def run_case(name, graph, ckpt):
    tmp = tempfile.mkdtemp(prefix=f"export_contract_{name}_")
    try:
        dir_a, dir_b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
        manifest = export(graph, ckpt, dir_a)
        export(graph, ckpt, dir_b)

        # 1. determinism across independent exports
        for fn in ("z_m.npy", "z_d.npy", "occupancy.npy", "model_ids.csv", "dataset_ids.csv",
                   "model_meta.parquet"):
            ha, hb = sha256_file(os.path.join(dir_a, fn)), sha256_file(os.path.join(dir_b, fn))
            assert ha == hb, f"{name}: {fn} differs across exports (nondeterministic)"
        print(f"  [{name}] 1. determinism: two exports bitwise-identical")

        # 2. ids alignment + row-count agreement
        payload = torch.load(graph, map_location="cpu", weights_only=False)
        z_m = np.load(os.path.join(dir_a, "z_m.npy"))
        z_d = np.load(os.path.join(dir_a, "z_d.npy"))
        mids = pd.read_csv(os.path.join(dir_a, "model_ids.csv"))
        dids = pd.read_csv(os.path.join(dir_a, "dataset_ids.csv"))
        meta = pd.read_parquet(os.path.join(dir_a, "model_meta.parquet"))
        assert np.array_equal(mids["mappedID"].to_numpy(), np.arange(len(mids)))
        assert np.array_equal(dids["mappedID"].to_numpy(), np.arange(len(dids)))
        assert len(mids) == z_m.shape[0] == len(meta) == payload["data"]["model"].num_nodes
        assert len(dids) == z_d.shape[0] == payload["data"]["dataset"].num_nodes
        umi = payload["unique_model_id"].sort_values("mappedID")["model"].to_numpy()
        assert np.array_equal(mids["unique_model_id"].to_numpy(), umi)
        udi = payload["unique_dataset_id"].sort_values("mappedID")["dataset"].to_numpy()
        assert np.array_equal(dids["unique_dataset_id"].to_numpy(), udi)
        print(f"  [{name}] 2. ids: row i == mappedID i, snapshots match "
              f"({len(mids)} models, {len(dids)} datasets)")

        # 3. unit norms
        assert z_m.dtype == np.float32 and z_d.dtype == np.float32
        assert np.allclose(np.linalg.norm(z_m, axis=1), 1.0, atol=1e-5)
        assert np.allclose(np.linalg.norm(z_d, axis=1), 1.0, atol=1e-5)
        print(f"  [{name}] 3. normalization: all rows unit L2 (dim {z_m.shape[1]})")

        # 4. hub-occupancy sidecar contract
        occupancy = np.load(os.path.join(dir_a, "occupancy.npy"))
        effective_k = min(HUB_OCCUPANCY_TOP_K, len(mids))
        assert occupancy.dtype == np.int32 and occupancy.shape == (len(mids),)
        assert np.all(occupancy >= 0)
        assert int(occupancy.sum()) == len(dids) * effective_k
        print(f"  [{name}] 4. occupancy: int32[{len(mids)}], "
              f"sum={int(occupancy.sum())} ({len(dids)} queries x top-{effective_k})")

        # 5. manifest completeness + hash truthfulness
        with open(os.path.join(dir_a, "manifest.json"), encoding="utf-8") as f:
            m = json.load(f)
        _check_schema(m, MANIFEST_SCHEMA, "")
        assert m == manifest, "manifest on disk != manifest returned by export()"
        for fn, sha in m["files"].items():
            assert sha256_file(os.path.join(dir_a, fn)) == sha, f"{fn} hash drifted"
        assert m["graph"]["sha256"] == sha256_file(graph)
        assert m["checkpoint"]["sha256"] == sha256_file(ckpt)
        repro = torch.load(ckpt, map_location="cpu", weights_only=False)["repro"]
        assert m["surgery"]["dedup_trained_on"] == bool(repro.get("dedup_trained_on", False))
        cfg = repro.get("config") or {}
        assert m["surgery"]["similar_to_mode"] == cfg.get("similar_to_mode", "dense")
        assert m["embedding"]["normalized"] is True
        assert m["hub_occupancy"]["file"] == "occupancy.npy"
        assert m["hub_occupancy"]["shape"] == [len(mids)]
        assert m["hub_occupancy"]["top_k"] == effective_k
        assert m["hub_occupancy"]["num_dataset_queries"] == len(dids)
        assert m["hub_occupancy"]["tie_break"] == "mappedID ascending"
        print(f"  [{name}] 5. manifest: schema complete, hashes verified, "
              f"surgery == ckpt repro {m['surgery']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    # Synthetic tie fixture: all models tie for query 0, so mappedIDs 0 and 1
    # win; query 1 ranks mappedIDs 2 then 0. Batch size 1 also exercises the
    # chunked accumulation path independently of the full export cases.
    z_m = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    z_d = np.array([[0.0, 0.0], [-1.0, -2.0]], dtype=np.float32)
    occupancy, effective_k = _compute_hub_occupancy(
        z_d, z_m, top_k=2, query_batch_size=1)
    assert effective_k == 2
    assert np.array_equal(occupancy, np.array([2, 1, 1], dtype=np.int32))
    print("[synthetic] occupancy tie policy + chunking: PASS")

    for name, (graph, ckpt) in CASES.items():
        print(f"[{name}]")
        run_case(name, graph, ckpt)
    print("\nGATE G-A: PASS (export contract green on diverse 306 + hf1000d 2000/G2)")


if __name__ == "__main__":
    main()
