"""
query.py -- Stage-3 Phase 3: retrieval query paths over a persisted index.

Two modes, matching how a dataset can arrive (plan §Phase 3):

  warm  the dataset's z_d row already exists in the export bound to the index:
        look it up and query. This is the evaluation replay path.

  cold  the deployment path ("target dataset embedded by one inductive forward
        pass"). For a dataset treated as cold: every edge incident to it is
        dropped (trained_on / rev_trained_on / similar_to via
        cold_graph.drop_cold_dataset_edges, plus any other dataset-incident
        edge type such as the diverse graph's transfer_to), the checkpoint's
        similar_to surgery is re-applied among the REMAINING datasets, then
        DEPLOYABLE incoming similar_to edges are attached from the e_domain
        feature slice only (cold_graph.cold_incoming_similar_to -- no
        performance labels), and one inductive forward yields z_d. The indexed
        z_m must be bitwise-unaffected by the insertion (query-only encode,
        asserted). Graph, checkpoint, surgery and e_domain dim all come from
        the export manifest / xd0_meta -- never guessed.

Output: ranked (rank, unique_model_id, cosine, optional hub_occupancy) + full
provenance (mode, ef_search, requested/effective K, index manifest hash,
cold-edge composition) as CLI table or JSON.

Run (from the repo root d:/research/model_lake/codes):
  python -m ModelLakeFishing.stage3HNSW.query --index artifacts/indexes/hf1000d_G2 \
      --dataset 42 --k 10                       # warm, by mappedID (or unique id)
  python -m ModelLakeFishing.stage3HNSW.query --index artifacts/indexes/hf1000d_G2 \
      --dataset 42 --mode cold --k 10           # same dataset via the cold path
"""

import argparse
import datetime
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
    TRAINED_ON, REV_TRAINED_ON, SIMILAR_TO, apply_similar_to_mode, dedup_trained_on,
)
from ModelLakeFishing.stage2TrainGraphSAGE.cold_graph import (  # noqa: E402
    assert_cold_absent, cold_incoming_similar_to, drop_cold_dataset_edges,
)
from ModelLakeFishing.stage3HNSW.build_index import (  # noqa: E402
    ManifestMismatch, load_index,
)
from ModelLakeFishing.stage3HNSW.export_embeddings import sha256_file  # noqa: E402


DEFAULT_HANDOFF_K = 200
MAX_RETRIEVAL_K = 500


def _validate_k(k):
    """Return K as an int, rejecting invalid or over-wide handoffs."""
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be an integer in 1..{MAX_RETRIEVAL_K}, got {k!r}")
    k = int(k)
    if not 1 <= k <= MAX_RETRIEVAL_K:
        raise ValueError(f"k must be in 1..{MAX_RETRIEVAL_K}, got {k}")
    return k


def _load_hub_occupancy(export_dir, export_manifest, n_models):
    """Load the optional, manifest-bound model hub-occupancy sidecar.

    Old exports have neither ``hub_occupancy`` metadata nor ``occupancy.npy``
    and intentionally return ``None``. A present sidecar must be hash-bound
    and obey the row-i == mappedID-i contract before it is exposed to Stage 4.
    """
    spec = export_manifest.get("hub_occupancy")
    filename = spec.get("file", "occupancy.npy") if isinstance(spec, dict) \
        else "occupancy.npy"
    path = os.path.join(export_dir, filename)
    if spec is None and not os.path.exists(path):
        return None
    if not os.path.exists(path):
        raise ManifestMismatch(f"hub occupancy sidecar missing: {path}")

    recorded_sha = export_manifest.get("files", {}).get(filename)
    if recorded_sha is None:
        raise ManifestMismatch(
            f"hub occupancy sidecar {filename} is not hash-bound in manifest.json")
    actual_sha = sha256_file(path)
    if actual_sha != recorded_sha:
        raise ManifestMismatch(
            f"hub occupancy sidecar {filename} drifted: {actual_sha[:16]}... "
            f"!= recorded {recorded_sha[:16]}...")

    occupancy = np.load(path, allow_pickle=False)
    if occupancy.shape != (n_models,):
        raise ManifestMismatch(
            f"hub occupancy shape {occupancy.shape} != expected ({n_models},)")
    if not np.issubdtype(occupancy.dtype, np.integer):
        raise ManifestMismatch(
            f"hub occupancy dtype {occupancy.dtype} is not an integer dtype")
    if np.any(occupancy < 0):
        raise ManifestMismatch("hub occupancy contains negative counts")

    if isinstance(spec, dict):
        if "shape" in spec and list(occupancy.shape) != list(spec["shape"]):
            raise ManifestMismatch(
                f"hub occupancy shape {occupancy.shape} != manifest {spec['shape']}")
        if "dtype" in spec and str(occupancy.dtype) != str(spec["dtype"]):
            raise ManifestMismatch(
                f"hub occupancy dtype {occupancy.dtype} != manifest {spec['dtype']}")
    return occupancy


# ── serving handle ──────────────────────────────────────────────────────────

def open_serving(index_dir, *, export_dir=None):
    """Manifest-verified handle: index + the export sidecars needed to query.
    The heavyweight cold context (graph + checkpoint) is loaded lazily."""
    index, man = load_index(index_dir, export_dir=export_dir)
    if export_dir is None:
        export_dir = os.path.join(_REPO_ROOT, man["export"]["path"])
    export_dir = os.path.abspath(export_dir)
    with open(os.path.join(export_dir, "manifest.json"), encoding="utf-8") as f:
        exp_man = json.load(f)
    model_ids = pd.read_csv(os.path.join(export_dir, "model_ids.csv"))
    hub_occupancy = _load_hub_occupancy(export_dir, exp_man, len(model_ids))
    return {
        "index": index, "index_dir": os.path.abspath(index_dir), "index_man": man,
        "index_manifest_sha256": sha256_file(os.path.join(index_dir, "index_manifest.json")),
        "export_dir": export_dir, "export_man": exp_man,
        "z_d": np.load(os.path.join(export_dir, "z_d.npy")),
        "model_ids": model_ids,
        "dataset_ids": pd.read_csv(os.path.join(export_dir, "dataset_ids.csv")),
        "hub_occupancy": hub_occupancy,
        "_cold_ctx": None,
    }


def _resolve_dataset(handle, dataset):
    df = handle["dataset_ids"]
    try:
        did = int(dataset)
    except (TypeError, ValueError):
        hit = df[df["unique_dataset_id"] == str(dataset)]
        assert len(hit) == 1, f"dataset {dataset!r} not found in export"
        did = int(hit["mappedID"].iloc[0])
    assert 0 <= did < len(df), f"mappedID {did} out of range 0..{len(df) - 1}"
    return did, str(df.loc[did, "unique_dataset_id"])


def _ranked(handle, z_q, k, ef_search):
    k = _validate_k(k)
    index = handle["index"]
    k = min(k, index.get_current_count())
    index.set_ef(max(ef_search, k + 16))
    labels, dists = index.knn_query(np.asarray(z_q, dtype="float32")[None, :], k=k)
    mid = handle["model_ids"]["unique_model_id"]
    occupancy = handle.get("hub_occupancy")
    ranked = []
    for r, (m, d) in enumerate(zip(labels[0], dists[0])):
        mapped_id = int(m)
        candidate = {
            "rank": r + 1,
            "mappedID": mapped_id,
            "unique_model_id": str(mid.iloc[mapped_id]),
            "cosine": round(float(1.0 - d), 6),
        }
        if occupancy is not None:
            candidate["hub_occupancy"] = int(occupancy[mapped_id])
        ranked.append(candidate)
    return ranked


def _provenance(handle, mode, ef_search, extra=None):
    p = {"mode": mode, "ef_search": ef_search,
         "hub_occupancy_available": handle.get("hub_occupancy") is not None,
         "index_dir": os.path.relpath(handle["index_dir"], _REPO_ROOT)
         if handle["index_dir"].startswith(_REPO_ROOT[:2]) else handle["index_dir"],
         "index_manifest_sha256": handle["index_manifest_sha256"],
         "index_sha256": handle["index_man"]["index"]["sha256"],
         "checkpoint_config": handle["index_man"]["export"]["checkpoint_config"],
         "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if extra:
        p.update(extra)
    return p


# ── warm path ───────────────────────────────────────────────────────────────

def warm_query(handle, dataset, *, k=50, ef_search=64):
    k = _validate_k(k)
    did, uid = _resolve_dataset(handle, dataset)
    z_q = handle["z_d"][did]
    results = _ranked(handle, z_q, k, ef_search)
    return {"query_dataset": {"mappedID": did, "unique_dataset_id": uid},
            "k": k, "results": results,
            "provenance": _provenance(
                handle, "warm", ef_search,
                extra={"requested_k": k, "effective_k": len(results)})}


# ── cold path ───────────────────────────────────────────────────────────────

def _cold_ctx(handle):
    """Lazy-load graph + checkpoint recorded in the export manifest."""
    if handle["_cold_ctx"] is None:
        m = handle["export_man"]
        graph_path = os.path.join(_REPO_ROOT, m["graph"]["path"])
        ckpt_path = os.path.join(_REPO_ROOT, m["checkpoint"]["path"])
        assert sha256_file(graph_path) == m["graph"]["sha256"], "graph drifted"
        assert sha256_file(ckpt_path) == m["checkpoint"]["sha256"], "checkpoint drifted"
        payload = torch.load(graph_path, map_location="cpu", weights_only=False)
        model, _vocab, _repro = load_checkpoint(ckpt_path)
        data = payload["data"]
        if m["surgery"]["dedup_trained_on"]:
            data = dedup_trained_on(data)
        handle["_cold_ctx"] = {
            "data": data, "model": model.cpu().eval(),
            "e_domain_dim": int(payload["xd0_meta"]["view_dims"]["e_domain"]),
            "surgery": m["surgery"],
        }
    return handle["_cold_ctx"]


def _drop_cold_incident(data, cold_ids):
    """cold_graph.drop_cold_dataset_edges + the same mask for any OTHER
    dataset-incident edge type (e.g. the diverse graph's transfer_to)."""
    data = drop_cold_dataset_edges(data, cold_ids)
    cold = {int(c) for c in cold_ids}
    for et in data.edge_types:
        if et in (TRAINED_ON, REV_TRAINED_ON, SIMILAR_TO):
            continue
        src_t, _, dst_t = et
        if "dataset" not in (src_t, dst_t):
            continue
        e = data[et].edge_index
        keep = torch.ones(e.size(1), dtype=torch.bool)
        if src_t == "dataset":
            keep &= torch.tensor([int(s) not in cold for s in e[0].tolist()])
        if dst_t == "dataset":
            keep &= torch.tensor([int(d) not in cold for d in e[1].tolist()])
        data[et].edge_index = e[:, keep]
        if getattr(data[et], "edge_attr", None) is not None:
            data[et].edge_attr = data[et].edge_attr[keep]
    return data


@torch.no_grad()
def cold_embed(handle, cold_id, *, incoming_mode=None, incoming_k=None):
    """Inductive z_d for `cold_id` treated as a NEW dataset. Returns
    (z_cold float32 unit vector, info dict). Deterministic (CPU forward)."""
    ctx = _cold_ctx(handle)
    surgery = ctx["surgery"]
    incoming_mode = incoming_mode or surgery["similar_to_mode"]
    incoming_k = incoming_k or surgery["similar_to_k"]
    cold = [int(cold_id)]

    g = _drop_cold_incident(ctx["data"], cold)
    g = apply_similar_to_mode(g, surgery["similar_to_mode"], k=surgery["similar_to_k"])
    assert_cold_absent(g, cold)

    remain = [i for i in range(ctx["data"]["dataset"].num_nodes) if i != cold[0]]
    ei, w = cold_incoming_similar_to(ctx["data"], cold, remain, mode=incoming_mode,
                                     k=incoming_k, e_domain_dim=ctx["e_domain_dim"])
    q = g.clone()
    if ei.numel():
        q[SIMILAR_TO].edge_index = torch.cat([q[SIMILAR_TO].edge_index, ei], dim=1)
        attr = getattr(q[SIMILAR_TO], "edge_attr", None)
        if attr is not None:
            q[SIMILAR_TO].edge_attr = torch.cat([attr, w])

    model = ctx["model"]
    z_train_m = model(g.clone())["model"]
    zq = model(q)
    assert torch.allclose(z_train_m, zq["model"], atol=1e-6), \
        "indexed z_m changed after cold insertion -- not a query-only encode"
    z_cold = torch.nn.functional.normalize(zq["dataset"][cold[0]], p=2, dim=-1)
    info = {"cold_mappedID": cold[0], "incoming_mode": incoming_mode,
            "incoming_k": incoming_k, "n_incoming_edges": int(ei.size(1)),
            "e_domain_dim": ctx["e_domain_dim"],
            "z_m_query_only_invariant": True}
    return z_cold.numpy().astype("float32"), info


def cold_query(handle, dataset, *, k=50, ef_search=64):
    k = _validate_k(k)
    did, uid = _resolve_dataset(handle, dataset)
    z_cold, info = cold_embed(handle, did)
    results = _ranked(handle, z_cold, k, ef_search)
    info.update({"requested_k": k, "effective_k": len(results)})
    return {"query_dataset": {"mappedID": did, "unique_dataset_id": uid},
            "k": k, "results": results,
            "provenance": _provenance(handle, "cold", ef_search, extra=info)}


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Stage-3 Phase 3 retrieval query")
    ap.add_argument("--index", required=True, help="index directory (Phase 2)")
    ap.add_argument("--dataset", required=True, help="dataset mappedID or unique id")
    ap.add_argument("--mode", default="warm", choices=["warm", "cold"])
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--ef-search", type=int, default=64)
    ap.add_argument("--json", default=None, help="also write the result to this path")
    args = ap.parse_args()

    index_dir = args.index if os.path.isdir(args.index) else os.path.join(_HERE, args.index)
    handle = open_serving(index_dir)
    fn = warm_query if args.mode == "warm" else cold_query
    out = fn(handle, args.dataset, k=args.k, ef_search=args.ef_search)

    q, p = out["query_dataset"], out["provenance"]
    print(f"[{args.mode}] dataset {q['mappedID']} ({q['unique_dataset_id']}) "
          f"-> top-{out['k']}  |  index {p['index_sha256'][:16]}... "
          f"ckpt={p['checkpoint_config']} efS={p['ef_search']}")
    if args.mode == "cold":
        print(f"  cold insertion: {p['n_incoming_edges']} deployable similar_to edges "
              f"({p['incoming_mode']}, k={p['incoming_k']}, e_domain {p['e_domain_dim']}d), "
              f"z_m query-only invariant: {p['z_m_query_only_invariant']}")
    print(f"  {'rank':<5} {'cosine':<9} {'mappedID':<9} unique_model_id")
    for r in out["results"]:
        print(f"  {r['rank']:<5} {r['cosine']:<9.4f} {r['mappedID']:<9} {r['unique_model_id']}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"  written: {args.json}")


if __name__ == "__main__":
    main()
