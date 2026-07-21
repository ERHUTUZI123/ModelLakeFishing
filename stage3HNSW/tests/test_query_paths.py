"""
test_query_paths.py -- Stage-3 Phase 3 gate G-D, on BOTH production indexes
(diverse_candidate 306 + hf1000d_G2 2,000):

  1. warm query == exact-dot top-K on the served embeddings (every dataset on
     the diverse export; 50 sampled datasets on hf1000d), and the returned
     cosine equals the exact dot within float32 tolerance;
  2. cold path determinism: two cold embeds of the same dataset are
     bitwise-identical;
  3. cold retrieval fidelity: for datasets WITH trained_on edges treated as
     cold, the HNSW top-50 of the cold embedding overlaps the exact-dot top-50
     of that same embedding >= 95% (plan smoke gate);
  4. cold insertion is query-only (z_m invariance is asserted inside
     cold_embed) and drops ALL dataset-incident edge types (incl. the diverse
     graph's transfer_to);
  5. provenance: every result carries mode / ef_search / index manifest hash,
     and the hash matches the file on disk.

Run:  python -m ModelLakeFishing.stage3HNSW.tests.test_query_paths
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage3HNSW.export_embeddings import sha256_file  # noqa: E402
from ModelLakeFishing.stage3HNSW.query import (  # noqa: E402
    cold_embed, cold_query, open_serving, warm_query,
)

S3 = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage3HNSW", "artifacts")
CASES = {  # name -> (n warm datasets to check, n cold datasets to check)
    "diverse_candidate": (None, 3),   # None = all datasets
    "hf1000d_G2": (50, 2),
}
OVERLAP_GATE = 0.95


def _cold_candidates(handle, n):
    """Datasets that actually have trained_on edges (dropping them matters)."""
    import torch
    from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import TRAINED_ON
    from ModelLakeFishing.stage3HNSW.query import _cold_ctx
    data = _cold_ctx(handle)["data"]
    counts = torch.bincount(data[TRAINED_ON].edge_index[1],
                            minlength=data["dataset"].num_nodes)
    order = torch.argsort(counts, descending=True)
    return [int(d) for d in order[:n]]


def run_case(name, n_warm, n_cold):
    handle = open_serving(os.path.join(S3, "indexes", name))
    z_m = np.load(os.path.join(handle["export_dir"], "z_m.npy"))
    z_d = handle["z_d"]
    n_d = z_d.shape[0]
    rng = np.random.default_rng(0)
    warm_ids = range(n_d) if n_warm is None else \
        sorted(rng.choice(n_d, size=min(n_warm, n_d), replace=False).tolist())

    # 1. warm == exact
    for k in (10, 50):
        kk = min(k, z_m.shape[0])
        for did in warm_ids:
            res = warm_query(handle, did, k=k, ef_search=200)
            got = [r["mappedID"] for r in res["results"]]
            exact = np.argsort(-(z_d[did] @ z_m.T), kind="stable")[:kk]
            assert set(got) == set(exact.tolist()), \
                f"{name} d{did} k={k}: warm ANN != exact set"
            for r in res["results"]:
                dot = float(z_d[did] @ z_m[r["mappedID"]])
                assert abs(r["cosine"] - dot) < 1e-4, \
                    f"{name} d{did}: cosine {r['cosine']} != dot {dot}"
    print(f"  [{name}] 1. warm == exact-dot top-K "
          f"({len(list(warm_ids))} datasets, k=10/50, scores agree)")

    # 2 + 3 + 4. cold paths
    cold_ids = _cold_candidates(handle, n_cold)
    for did in cold_ids:
        za, ia = cold_embed(handle, did)
        zb, _ = cold_embed(handle, did)
        assert np.array_equal(za, zb), f"{name} d{did}: cold embed nondeterministic"
        assert ia["z_m_query_only_invariant"] is True
        assert ia["n_incoming_edges"] > 0, f"{name} d{did}: no deployable edges attached"

        res = cold_query(handle, did, k=50, ef_search=200)
        got = {r["mappedID"] for r in res["results"]}
        kk = min(50, z_m.shape[0])
        exact = set(np.argsort(-(za @ z_m.T), kind="stable")[:kk].tolist())
        overlap = len(got & exact) / kk
        assert overlap >= OVERLAP_GATE, \
            f"{name} d{did}: cold ANN/exact overlap {overlap:.3f} < {OVERLAP_GATE}"
        print(f"  [{name}] 2-4. cold d{did}: deterministic, "
              f"{ia['n_incoming_edges']} deployable edges "
              f"(e_domain {ia['e_domain_dim']}d), z_m invariant, "
              f"ANN/exact top-{kk} overlap {overlap:.3f}")

    # 5. provenance
    res = warm_query(handle, list(warm_ids)[0], k=5)
    p = res["provenance"]
    for key in ("mode", "ef_search", "index_manifest_sha256", "index_sha256",
                "timestamp_utc"):
        assert key in p, f"provenance missing {key}"
    actual = sha256_file(os.path.join(handle["index_dir"], "index_manifest.json"))
    assert p["index_manifest_sha256"] == actual
    print(f"  [{name}] 5. provenance complete, manifest hash matches disk")


def main():
    for name, (n_warm, n_cold) in CASES.items():
        print(f"[{name}]")
        run_case(name, n_warm, n_cold)
    print("\nGATE G-D: PASS (warm==exact, cold deterministic + query-only + "
          f"overlap >= {OVERLAP_GATE:.0%}, provenance bound) on 306 + 2000")


if __name__ == "__main__":
    main()
