"""
build_index.py -- Stage-3 Phase 2: export -> persistent hnswlib index.

Consumes a Phase-1 export directory (z_m.npy + manifest.json), builds the
production hnswlib cosine index with labels == mappedID (D4), persists it, and
gates it before it may be served:

  G-B  load-roundtrip: reload from disk, re-run a fixed query batch (all z_d,
       k=50) -> labels AND distances bitwise-identical to the pre-save index;
  G-C  fidelity vs exact dot on the served vectors: recall@{1,10,50,100} with
       ef_search=200 (hard gate >= 0.99 @10; expected 1.000 at 2K).

Determinism: single-thread build with a fixed random_seed, so index.bin is
bit-reproducible from the same export.

The sidecar index_manifest.json binds the index to its export (export-manifest
sha256 + z_m sha256 + graph/ckpt hashes). `load_index()` refuses to load when
any recorded hash disagrees with what is on disk -- an index file without its
(matching) manifest is invalid by definition.

Run:
  python -m ModelLakeFishing.stage3HNSW.build_index \
      --export artifacts/exports/hf1000d_G2 [--M 16] [--efc 200]
"""

import argparse
import datetime
import json
import os
import sys
import time
from importlib.metadata import version as pkg_version

import hnswlib
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage3HNSW.export_embeddings import sha256_file  # noqa: E402

BUILD_CODE_VERSION = "1.0"
INDEXES = os.path.join(_HERE, "artifacts", "indexes")
RECALL_KS = (1, 10, 50, 100)
FIDELITY_EF = 200
FIDELITY_GATE_AT10 = 0.99
ROUNDTRIP_K = 50


class ManifestMismatch(RuntimeError):
    """Raised when an index or its export does not match the recorded hashes."""


def _repo_relpath(path):
    """Repo-relative when possible; absolute when on another drive (tmp dirs)."""
    path = os.path.abspath(path)
    try:
        return os.path.relpath(path, _REPO_ROOT)
    except ValueError:
        return path


def _load_export(export_dir):
    export_dir = os.path.abspath(export_dir)
    man_path = os.path.join(export_dir, "manifest.json")
    if not os.path.exists(man_path):
        raise ManifestMismatch(f"export manifest missing: {man_path}")
    with open(man_path, encoding="utf-8") as f:
        man = json.load(f)
    # refuse a drifted export: every file must still hash to what the export
    # manifest recorded (the export's own D4 guarantee, re-checked here)
    for fn, sha in man["files"].items():
        actual = sha256_file(os.path.join(export_dir, fn))
        if actual != sha:
            raise ManifestMismatch(f"export file {fn} drifted: {actual[:16]}... "
                                   f"!= recorded {sha[:16]}...")
    z_m = np.load(os.path.join(export_dir, "z_m.npy"))
    z_d = np.load(os.path.join(export_dir, "z_d.npy"))
    return man, sha256_file(man_path), z_m, z_d


def exact_topk(z_q, z_m, k):
    """Exact cosine ranking on unit vectors == inner-product argsort."""
    scores = z_q @ z_m.T
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def _recalls(index, z_q, z_m, ef_search):
    index.set_ef(max(ef_search, max(RECALL_KS) + 16))
    out = {}
    for k in RECALL_KS:
        kk = min(k, z_m.shape[0])
        exact = exact_topk(z_q, z_m, kk)
        ann, _ = index.knn_query(z_q, k=kk)
        out[f"recall@{k}"] = float(np.mean(
            [len(set(ann[i]) & set(exact[i])) / kk for i in range(z_q.shape[0])]))
    return out


def build(export_dir, out_dir, *, M=16, ef_construction=200, random_seed=100):
    man, man_sha, z_m, z_d = _load_export(export_dir)
    n, dim = z_m.shape

    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=n, ef_construction=ef_construction, M=M,
                     random_seed=random_seed)
    index.set_num_threads(1)  # deterministic insertion order -> reproducible .bin
    t0 = time.perf_counter()
    index.add_items(z_m, np.arange(n))
    build_s = time.perf_counter() - t0

    # G-C fidelity on the in-memory index (serving direction: z_d -> z_m)
    fidelity = _recalls(index, z_d, z_m, FIDELITY_EF)
    if fidelity["recall@10"] < FIDELITY_GATE_AT10:
        raise RuntimeError(f"fidelity gate failed: recall@10 {fidelity['recall@10']:.4f} "
                           f"< {FIDELITY_GATE_AT10}")

    # persist, then G-B load-roundtrip against the pre-save results
    os.makedirs(out_dir, exist_ok=True)
    bin_path = os.path.join(out_dir, "index.bin")
    index.save_index(bin_path)
    kk = min(ROUNDTRIP_K, n)
    index.set_ef(max(FIDELITY_EF, kk + 16))
    ref_l, ref_dist = index.knn_query(z_d, k=kk)
    re_index = hnswlib.Index(space="cosine", dim=dim)
    re_index.load_index(bin_path, max_elements=n)
    re_index.set_ef(max(FIDELITY_EF, kk + 16))
    re_l, re_dist = re_index.knn_query(z_d, k=kk)
    identical = bool(np.array_equal(ref_l, re_l) and np.array_equal(ref_dist, re_dist))
    if not identical:
        raise RuntimeError("load-roundtrip gate failed: reloaded index answers differ")

    manifest = {
        "build_code_version": BUILD_CODE_VERSION,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "backend": {"library": "hnswlib", "version": pkg_version("hnswlib"),
                    "space": "cosine"},
        "params": {"M": M, "ef_construction": ef_construction,
                   "random_seed": random_seed, "build_threads": 1},
        "export": {"path": _repo_relpath(export_dir),
                   "manifest_sha256": man_sha,
                   "z_m_sha256": man["files"]["z_m.npy"],
                   "graph_sha256": man["graph"]["sha256"],
                   "checkpoint_sha256": man["checkpoint"]["sha256"],
                   "checkpoint_config": man["checkpoint"].get("config_name"),
                   "num_models": man["graph"]["num_models"], "dim": man["embedding"]["dim"]},
        "index": {"file": "index.bin", "sha256": sha256_file(bin_path),
                  "element_count": int(index.get_current_count()),
                  "bytes": os.path.getsize(bin_path),
                  "build_seconds": round(build_s, 4)},
        "gates": {"fidelity_ef_search": FIDELITY_EF, **fidelity,
                  "fidelity_queries": f"all z_d ({z_d.shape[0]})",
                  "roundtrip_identical": identical, "roundtrip_k": kk},
    }
    with open(os.path.join(out_dir, "index_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_index(index_dir, *, export_dir=None):
    """Load a persisted index, refusing on any manifest mismatch (D4).

    Verifies: index_manifest.json exists; index.bin hashes to the recorded
    sha256; element count matches; and the bound export (default: the path
    recorded in the manifest) still exists with an unchanged manifest.json and
    z_m.npy. Returns (hnswlib.Index ready to query, index manifest dict).
    """
    index_dir = os.path.abspath(index_dir)
    man_path = os.path.join(index_dir, "index_manifest.json")
    if not os.path.exists(man_path):
        raise ManifestMismatch(f"index manifest missing: {man_path} "
                               "(an index without its manifest is invalid)")
    with open(man_path, encoding="utf-8") as f:
        man = json.load(f)
    bin_path = os.path.join(index_dir, man["index"]["file"])
    actual = sha256_file(bin_path)
    if actual != man["index"]["sha256"]:
        raise ManifestMismatch(f"index.bin drifted: {actual[:16]}... != recorded "
                               f"{man['index']['sha256'][:16]}...")
    if export_dir is None:
        export_dir = os.path.join(_REPO_ROOT, man["export"]["path"])
    exp_man = os.path.join(export_dir, "manifest.json")
    if not os.path.exists(exp_man):
        raise ManifestMismatch(f"bound export missing: {export_dir}")
    if sha256_file(exp_man) != man["export"]["manifest_sha256"]:
        raise ManifestMismatch("export manifest.json does not match the one this "
                               "index was built from")
    z_m_actual = sha256_file(os.path.join(export_dir, "z_m.npy"))
    if z_m_actual != man["export"]["z_m_sha256"]:
        raise ManifestMismatch("export z_m.npy does not match the vectors this "
                               "index was built from")
    index = hnswlib.Index(space="cosine", dim=man["export"]["dim"])
    index.load_index(bin_path, max_elements=man["index"]["element_count"])
    if index.get_current_count() != man["index"]["element_count"]:
        raise ManifestMismatch(f"element count {index.get_current_count()} != "
                               f"recorded {man['index']['element_count']}")
    return index, man


def main():
    ap = argparse.ArgumentParser(description="Stage-3 Phase 2 index build")
    ap.add_argument("--export", required=True, help="Phase-1 export directory")
    ap.add_argument("--out", default=None,
                    help="index output dir (default artifacts/indexes/<export name>)")
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()
    export_dir = args.export if os.path.isabs(args.export) else \
        os.path.normpath(os.path.join(os.getcwd(), args.export))
    if not os.path.isdir(export_dir):  # also accept paths relative to stage3HNSW/
        export_dir = os.path.join(_HERE, args.export)
    name = os.path.basename(os.path.normpath(export_dir))
    out_dir = args.out or os.path.join(INDEXES, name)
    m = build(export_dir, out_dir, M=args.M, ef_construction=args.efc,
              random_seed=args.seed)
    g = m["gates"]
    print(f"[{name}] {m['index']['element_count']} elements, dim {m['export']['dim']}, "
          f"M={m['params']['M']} efC={m['params']['ef_construction']} "
          f"-> {out_dir} ({m['index']['bytes']/1e6:.1f} MB, "
          f"{m['index']['build_seconds']:.2f}s build)")
    print(f"  G-C fidelity (efS={g['fidelity_ef_search']}, {g['fidelity_queries']}): " +
          "  ".join(f"r@{k}={g[f'recall@{k}']:.3f}" for k in RECALL_KS))
    print(f"  G-B roundtrip identical: {g['roundtrip_identical']} (k={g['roundtrip_k']})")


if __name__ == "__main__":
    main()
