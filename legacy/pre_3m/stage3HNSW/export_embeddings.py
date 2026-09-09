"""
export_embeddings.py -- Stage-3 Phase 1: (graph.pt, ckpt.pt) -> serving export.

Produces the checkpoint-agnostic (z_m, ids, manifest) triple every later Stage-3
component consumes (plan D2/D4). Per the 2026-07-08 amendment the served
checkpoint is G2 only; this exporter stays generic.

Export layout  (artifacts/exports/<name>/):
    z_m.npy             float32 [N_m, dim], L2-normalized, row i == mappedID i
    z_d.npy             float32 [N_d, dim], L2-normalized, row j == dataset mappedID j
    model_ids.csv       mappedID, unique_model_id
    dataset_ids.csv     mappedID, unique_dataset_id
    model_meta.parquet  mappedID, unique_model_id, size_bucket_id, family_id
                        (the discrete-id sidecar the Stage-4 reranker needs)
    occupancy.npy       int32 [N_m], number of dataset queries whose exact
                        cosine top-10 contains model mappedID i
    manifest.json       D4 identity contract -- written LAST; an export without
                        a manifest is invalid by definition.

Contract enforcement (the Stage-2 killer risk, now at the export boundary):
  * forward runs on CPU in eval mode and is computed TWICE -- bitwise equality
    required (determinism);
  * the message graph is rebuilt exactly as the checkpoint was trained:
    dedup_trained_on + similar_to surgery are read from the checkpoint's repro
    metadata (phase0_freeze wrote them), never guessed;
  * if the repro records a graph sha256, the graph on disk must match it;
  * model row order: unique_model_id snapshot must be the permutation 0..N-1
    and data['model'].node_id must equal arange(N); same for datasets;
  * 20-row spot check: ids/meta files re-read from disk must match the graph's
    per-node vocab-bound attributes.

Run:
  python -m ModelLakeFishing.stage3HNSW.export_embeddings \
      --graph ModelLakeFishing/stage1BuildTransferGraph/hgraph_hf1000d_2000m_xm0_xd0.pt \
      --ckpt  ModelLakeFishing/stage2TrainGraphSAGE/artifacts/ablation/top1/ckpt/G2_s0_i0.pt \
      --name  hf1000d_G2
"""

import argparse
import datetime
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.learnable import load_checkpoint  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import (  # noqa: E402
    apply_similar_to_mode, dedup_trained_on,
)

EXPORT_CODE_VERSION = "1.1"
EXPORTS = os.path.join(_HERE, "artifacts", "exports")
SPOTCHECK_ROWS = 20
HUB_OCCUPANCY_TOP_K = 10
HUB_OCCUPANCY_QUERY_BATCH_SIZE = 256


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _id_snapshot(df, entity_col, n_nodes, what):
    """Validate the unique-id snapshot is the permutation 0..N-1 and return it
    sorted by mappedID (row i == mappedID i)."""
    assert len(df) == n_nodes, f"{what}: snapshot rows {len(df)} != nodes {n_nodes}"
    ids = np.sort(df["mappedID"].to_numpy())
    assert np.array_equal(ids, np.arange(n_nodes)), f"{what}: mappedID is not 0..N-1"
    out = df.sort_values("mappedID").reset_index(drop=True)
    return out[["mappedID", entity_col]].rename(columns={entity_col: f"unique_{what}_id"})


def _vocab_sidecar(ckpt_path, suffix, expected):
    """Hash the vocab CSV bound to the checkpoint and verify it matches the
    dict stored inside the checkpoint (the identity credential, D4)."""
    stem = os.path.splitext(ckpt_path)[0]
    path = f"{stem}.{suffix}.csv"
    if expected is None:
        return None
    assert os.path.exists(path), f"vocab sidecar missing: {path}"
    df = pd.read_csv(path)
    key_col, id_col = df.columns[0], df.columns[1]
    parsed = dict(zip(df[key_col], df[id_col].astype(int)))
    assert parsed == {str(k): int(v) for k, v in expected.items()}, (
        f"{path} does not match the vocab embedded in the checkpoint")
    return {"path": os.path.relpath(path, _REPO_ROOT), "sha256": sha256_file(path),
            "rows": len(parsed)}


def _compute_hub_occupancy(z_d, z_m, top_k=HUB_OCCUPANCY_TOP_K,
                           query_batch_size=HUB_OCCUPANCY_QUERY_BATCH_SIZE):
    """Count exact-cosine top-k appearances for each model.

    Both inputs are the stored unit vectors, so inner product is the serving
    cosine score. Queries are batched to bound the temporary score matrix.
    ``mergesort`` is stable: because model columns are in mappedID order,
    exactly tied scores are resolved by the lower mappedID. This makes the
    sidecar's tie policy explicit and deterministic across exports.
    """
    assert z_d.ndim == 2 and z_m.ndim == 2, "occupancy inputs must be matrices"
    assert z_d.shape[1] == z_m.shape[1], "dataset/model embedding dims differ"
    assert z_m.shape[0] > 0, "cannot compute occupancy without models"
    assert top_k > 0, "occupancy top_k must be positive"
    assert query_batch_size > 0, "occupancy query_batch_size must be positive"

    effective_k = min(int(top_k), int(z_m.shape[0]))
    occupancy = np.zeros(z_m.shape[0], dtype=np.int64)
    for start in range(0, z_d.shape[0], query_batch_size):
        scores = z_d[start:start + query_batch_size] @ z_m.T
        # A stable descending sort provides mappedID-ascending tie-breaking.
        top = np.argsort(-scores, axis=1, kind="mergesort")[:, :effective_k]
        occupancy += np.bincount(top.reshape(-1), minlength=z_m.shape[0])

    assert occupancy.max(initial=0) <= np.iinfo(np.int32).max, (
        "occupancy count exceeds int32 capacity")
    return occupancy.astype(np.int32), effective_k


def export(graph_path, ckpt_path, out_dir):
    """Run the full export. Returns the manifest dict (also written to disk)."""
    graph_path, ckpt_path = os.path.abspath(graph_path), os.path.abspath(ckpt_path)
    graph_sha, ckpt_sha = sha256_file(graph_path), sha256_file(ckpt_path)

    payload = torch.load(graph_path, map_location="cpu", weights_only=False)
    data = payload["data"]
    model, family_vocab, repro = load_checkpoint(ckpt_path)  # validates vocab binding

    # the ckpt must be served on the graph it was trained on
    if repro.get("graph_sha256"):
        assert repro["graph_sha256"] == graph_sha, (
            f"checkpoint was trained on graph {repro['graph_sha256'][:16]}... "
            f"but --graph has sha {graph_sha[:16]}...")

    # rebuild the training-time message structure from repro (never guess)
    dedup = bool(repro.get("dedup_trained_on", False))
    cfg = repro.get("config") or {}
    sim_mode = cfg.get("similar_to_mode", "dense")
    sim_k = int(cfg.get("similar_to_k", 10))
    if dedup:
        data = dedup_trained_on(data)
    if sim_mode != "dense":
        data = apply_similar_to_mode(data, sim_mode, k=sim_k)

    n_m, n_d = data["model"].num_nodes, data["dataset"].num_nodes
    assert torch.equal(data["model"].node_id, torch.arange(n_m)), "model node_id != arange"
    assert torch.equal(data["dataset"].node_id, torch.arange(n_d)), "dataset node_id != arange"

    # deterministic full-graph forward: CPU, eval, computed twice, bitwise equal
    model = model.cpu().eval()
    with torch.no_grad():
        z_a = model(data.clone())
        z_b = model(data.clone())
    det_ok = torch.equal(z_a["model"], z_b["model"]) and torch.equal(z_a["dataset"], z_b["dataset"])
    assert det_ok, "forward is not deterministic on CPU -- export aborted"

    z_m = torch.nn.functional.normalize(z_a["model"], p=2, dim=-1).numpy().astype("float32")
    z_d = torch.nn.functional.normalize(z_a["dataset"], p=2, dim=-1).numpy().astype("float32")
    assert np.allclose(np.linalg.norm(z_m, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(z_d, axis=1), 1.0, atol=1e-5)

    # id snapshots: row i == mappedID i (the iron rule)
    model_ids = _id_snapshot(payload["unique_model_id"], "model", n_m, "model")
    dataset_ids = _id_snapshot(payload["unique_dataset_id"], "dataset", n_d, "dataset")
    model_meta = model_ids.copy()
    model_meta["size_bucket_id"] = data["model"].size_bucket_id.numpy()
    model_meta["family_id"] = data["model"].family_id.numpy()
    occupancy, occupancy_top_k = _compute_hub_occupancy(z_d, z_m)
    assert int(occupancy.sum()) == n_d * occupancy_top_k

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "z_m.npy"), z_m)
    np.save(os.path.join(out_dir, "z_d.npy"), z_d)
    np.save(os.path.join(out_dir, "occupancy.npy"), occupancy)
    model_ids.to_csv(os.path.join(out_dir, "model_ids.csv"), index=False)
    dataset_ids.to_csv(os.path.join(out_dir, "dataset_ids.csv"), index=False)
    model_meta.to_parquet(os.path.join(out_dir, "model_meta.parquet"), index=False)

    # spot check FROM DISK: sampled rows must match the graph's node attributes
    rng = np.random.default_rng(0)
    rows = rng.choice(n_m, size=min(SPOTCHECK_ROWS, n_m), replace=False)
    ids_back = pd.read_csv(os.path.join(out_dir, "model_ids.csv"))
    meta_back = pd.read_parquet(os.path.join(out_dir, "model_meta.parquet"))
    umi = payload["unique_model_id"].set_index("mappedID")["model"]
    for i in map(int, rows):
        assert ids_back.loc[i, "mappedID"] == i
        assert ids_back.loc[i, "unique_model_id"] == umi.loc[i]
        assert int(meta_back.loc[i, "size_bucket_id"]) == int(data["model"].size_bucket_id[i])
        assert int(meta_back.loc[i, "family_id"]) == int(data["model"].family_id[i])
    z_back = np.load(os.path.join(out_dir, "z_m.npy"))
    assert z_back.shape == (n_m, z_m.shape[1]) and np.array_equal(z_back, z_m)
    occupancy_back = np.load(os.path.join(out_dir, "occupancy.npy"))
    assert occupancy_back.dtype == np.int32
    assert occupancy_back.shape == (n_m,) and np.array_equal(occupancy_back, occupancy)

    manifest = {
        "export_code_version": EXPORT_CODE_VERSION,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "graph": {"path": os.path.relpath(graph_path, _REPO_ROOT), "sha256": graph_sha,
                  "num_models": n_m, "num_datasets": n_d},
        "checkpoint": {"path": os.path.relpath(ckpt_path, _REPO_ROOT), "sha256": ckpt_sha,
                       "config_name": repro.get("config_name"),
                       "split_seed": repro.get("split_seed"),
                       "init_seed": repro.get("init_seed")},
        "vocab": {
            "family_csv": _vocab_sidecar(ckpt_path, "family_vocab", family_vocab),
            "task_type_csv": _vocab_sidecar(
                ckpt_path, "task_type_vocab",
                torch.load(ckpt_path, map_location="cpu",
                           weights_only=False).get("task_type_vocab")),
        },
        "surgery": {"dedup_trained_on": dedup, "similar_to_mode": sim_mode,
                    "similar_to_k": sim_k},
        "embedding": {"dim": int(z_m.shape[1]), "dtype": "float32", "normalized": True,
                      "score": "cosine == inner product of the stored unit vectors"},
        "hub_occupancy": {
            "file": "occupancy.npy", "dtype": "int32", "shape": [int(n_m)],
            "top_k": int(occupancy_top_k), "num_dataset_queries": int(n_d),
            "score": "exact cosine == inner product of the stored unit vectors",
            "tie_break": "mappedID ascending", "query_batch_size":
                HUB_OCCUPANCY_QUERY_BATCH_SIZE,
        },
        "encoders": {k: repro.get(k) for k in
                     ("e_name_seed", "e_name_token_dim", "e_name_hash_buckets",
                      "e_desc_encoder", "size_bucket_constants")},
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "verification": {"forward_determinism_bitwise": True,
                         "ids_row_order_ok": True,
                         "spotcheck_rows": int(len(rows)), "spotcheck_ok": True},
        "files": {},
    }
    for fn in ("z_m.npy", "z_d.npy", "occupancy.npy", "model_ids.csv", "dataset_ids.csv",
               "model_meta.parquet"):
        manifest["files"][fn] = sha256_file(os.path.join(out_dir, fn))
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Stage-3 Phase 1 embedding export")
    ap.add_argument("--graph", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True, help="export name -> artifacts/exports/<name>/")
    args = ap.parse_args()
    out_dir = os.path.join(EXPORTS, args.name)
    m = export(args.graph, args.ckpt, out_dir)
    print(f"[{args.name}] {m['graph']['num_models']} models x dim {m['embedding']['dim']}, "
          f"{m['graph']['num_datasets']} datasets -> {out_dir}")
    print(f"  ckpt {m['checkpoint']['config_name']} {m['checkpoint']['sha256'][:16]}... "
          f"on graph {m['graph']['sha256'][:16]}...")
    print(f"  surgery: {m['surgery']}  |  verification: {m['verification']}")


if __name__ == "__main__":
    main()
