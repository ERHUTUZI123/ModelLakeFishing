"""
test_index_roundtrip.py -- Stage-3 Phase 2 gates, on BOTH exports
(diverse_candidate 306 + hf1000d_G2 2,000):

  1. G-B roundtrip identity: build -> save -> load -> identical labels AND
     distances on a fixed query batch (also re-checked independently of the
     build-time assertion);
  2. G-C fidelity: recall@10 vs exact dot >= 0.99 (hard gate; expect 1.000);
  3. label contract: the index contains exactly the labels 0..N-1 (mappedIDs);
  4. manifest-mismatch refusal: (a) tampered index.bin, (b) tampered export
     z_m.npy, (c) missing index manifest -- each MUST raise ManifestMismatch;
  5. build determinism: two single-thread builds from the same export produce
     bitwise-identical index.bin files.

Run:  python -m ModelLakeFishing.stage3HNSW.tests.test_index_roundtrip
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage3HNSW.build_index import (  # noqa: E402
    ManifestMismatch, build, load_index,
)
from ModelLakeFishing.stage3HNSW.export_embeddings import sha256_file  # noqa: E402

EXPORTS = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage3HNSW",
                       "artifacts", "exports")
CASES = ["diverse_candidate", "hf1000d_G2"]


def _expect_mismatch(fn, what):
    try:
        fn()
    except ManifestMismatch:
        return
    raise AssertionError(f"{what}: expected ManifestMismatch, but load succeeded")


def run_case(name):
    export_dir = os.path.join(EXPORTS, name)
    assert os.path.isdir(export_dir), f"missing export {export_dir} (run Phase 1 first)"
    tmp = tempfile.mkdtemp(prefix=f"index_roundtrip_{name}_")
    try:
        # keep the export inside tmp so tamper tests never touch the real one
        exp = os.path.join(tmp, "export")
        shutil.copytree(export_dir, exp)
        idx_a = os.path.join(tmp, "idx_a")
        man = build(exp, idx_a)

        # 1. G-B roundtrip (independent re-check through the public loader)
        z_d = np.load(os.path.join(exp, "z_d.npy"))
        index, man_loaded = load_index(idx_a, export_dir=exp)
        index.set_ef(200)
        l1, d1 = index.knn_query(z_d, k=min(50, man["index"]["element_count"]))
        index2, _ = load_index(idx_a, export_dir=exp)
        index2.set_ef(200)
        l2, d2 = index2.knn_query(z_d, k=min(50, man["index"]["element_count"]))
        assert np.array_equal(l1, l2) and np.array_equal(d1, d2)
        assert man["gates"]["roundtrip_identical"] is True
        print(f"  [{name}] 1. G-B roundtrip: save/load answers identical")

        # 2. G-C fidelity
        r10 = man["gates"]["recall@10"]
        assert r10 >= 0.99, f"{name}: recall@10 {r10:.4f} < 0.99"
        print(f"  [{name}] 2. G-C fidelity: " +
              " ".join(f"r@{k}={man['gates'][f'recall@{k}']:.3f}" for k in (1, 10, 50, 100)))

        # 3. label contract == mappedIDs 0..N-1
        labels = np.sort(np.asarray(index.get_ids_list()))
        assert np.array_equal(labels, np.arange(man["index"]["element_count"]))
        print(f"  [{name}] 3. labels: exactly 0..{man['index']['element_count'] - 1}")

        # 4. manifest-mismatch refusal
        idx_b = os.path.join(tmp, "idx_b")
        shutil.copytree(idx_a, idx_b)
        with open(os.path.join(idx_b, "index.bin"), "r+b") as f:
            f.seek(100)
            b = f.read(1)
            f.seek(100)
            f.write(bytes([b[0] ^ 0xFF]))
        _expect_mismatch(lambda: load_index(idx_b, export_dir=exp), "tampered index.bin")

        exp_b = os.path.join(tmp, "export_b")
        shutil.copytree(exp, exp_b)
        z = np.load(os.path.join(exp_b, "z_m.npy"))
        z[0, 0] += 1e-3
        np.save(os.path.join(exp_b, "z_m.npy"), z)
        _expect_mismatch(lambda: load_index(idx_a, export_dir=exp_b), "tampered export z_m")

        idx_c = os.path.join(tmp, "idx_c")
        shutil.copytree(idx_a, idx_c)
        os.remove(os.path.join(idx_c, "index_manifest.json"))
        _expect_mismatch(lambda: load_index(idx_c, export_dir=exp), "missing manifest")
        print(f"  [{name}] 4. refusal: tampered bin / tampered z_m / missing manifest all raise")

        # 5. deterministic build
        idx_d = os.path.join(tmp, "idx_d")
        build(exp, idx_d)
        assert sha256_file(os.path.join(idx_a, "index.bin")) == \
            sha256_file(os.path.join(idx_d, "index.bin"))
        print(f"  [{name}] 5. determinism: rebuild produces bitwise-identical index.bin")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    for name in CASES:
        print(f"[{name}]")
        run_case(name)
    print("\nGATES G-B + G-C: PASS (roundtrip identity, fidelity >= 0.99@10, "
          "refusal, determinism on 306 + 2000)")


if __name__ == "__main__":
    main()
