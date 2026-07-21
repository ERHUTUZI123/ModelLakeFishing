"""Focused contract tests for the Track-B Stage-4 retrieval handoff.

The fixtures build tiny real HNSW indexes so the tests cover manifest-verified
``ModelRetriever.load`` as well as query/service ranking parity.

Run::

    python -m ModelLakeFishing.stage3HNSW.tests.test_retrieval_service
"""

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage3HNSW.build_index import (  # noqa: E402
    ManifestMismatch, build,
)
from ModelLakeFishing.stage3HNSW.export_embeddings import sha256_file  # noqa: E402
from ModelLakeFishing.stage3HNSW.query import (  # noqa: E402
    MAX_RETRIEVAL_K, open_serving, warm_query,
)
from ModelLakeFishing.stage3HNSW.retrieval_service import (  # noqa: E402
    DEFAULT_HANDOFF_K, Candidate, ModelRetriever, RetrievalOutput,
)


def _expect(exc_type, fn, text=None):
    try:
        fn()
    except exc_type as exc:
        if text is not None:
            assert text in str(exc), f"{text!r} not in {str(exc)!r}"
        return exc
    raise AssertionError(f"expected {exc_type.__name__}")


def _fixture(root, name, *, n_models, with_occupancy):
    """Create a minimal valid Phase-1 export and build its real HNSW index."""
    export_dir = os.path.join(root, f"export_{name}")
    index_dir = os.path.join(root, f"index_{name}")
    os.makedirs(export_dir)

    rng = np.random.default_rng(123 if with_occupancy else 456)
    z_m = rng.normal(size=(n_models, 16)).astype("float32")
    z_m /= np.linalg.norm(z_m, axis=1, keepdims=True)
    z_d = z_m[:min(4, n_models)].copy()
    model_ids = pd.DataFrame({
        "mappedID": np.arange(n_models),
        "unique_model_id": [f"model-{i}" for i in range(n_models)],
    })
    dataset_ids = pd.DataFrame({
        "mappedID": np.arange(len(z_d)),
        "unique_dataset_id": [f"dataset-{i}" for i in range(len(z_d))],
    })
    model_meta = model_ids.copy()
    model_meta["size_bucket_id"] = np.arange(n_models) % 15
    model_meta["family_id"] = np.arange(n_models) % 7

    np.save(os.path.join(export_dir, "z_m.npy"), z_m)
    np.save(os.path.join(export_dir, "z_d.npy"), z_d)
    model_ids.to_csv(os.path.join(export_dir, "model_ids.csv"), index=False)
    dataset_ids.to_csv(os.path.join(export_dir, "dataset_ids.csv"), index=False)
    model_meta.to_parquet(os.path.join(export_dir, "model_meta.parquet"), index=False)

    filenames = [
        "z_m.npy", "z_d.npy", "model_ids.csv", "dataset_ids.csv",
        "model_meta.parquet",
    ]
    occupancy = None
    if with_occupancy:
        occupancy = ((np.arange(n_models) * 17) % 101).astype("int32")
        np.save(os.path.join(export_dir, "occupancy.npy"), occupancy)
        filenames.append("occupancy.npy")

    manifest = {
        "export_code_version": "test",
        "graph": {
            "path": "fixture.pt", "sha256": "graph-fixture",
            "num_models": n_models, "num_datasets": len(z_d),
        },
        "checkpoint": {
            "path": "fixture.pt", "sha256": "checkpoint-fixture",
            "config_name": "fixture",
        },
        "embedding": {"dim": 16, "dtype": "float32", "normalized": True},
        "files": {
            filename: sha256_file(os.path.join(export_dir, filename))
            for filename in filenames
        },
    }
    if with_occupancy:
        manifest["hub_occupancy"] = {
            "file": "occupancy.npy",
            "dtype": "int32",
            "shape": [n_models],
            "top_k": 10,
            "num_dataset_queries": len(z_d),
            "score": "cosine",
            "tie_break": "mappedID_asc",
            "query_batch_size": 64,
        }
    with open(os.path.join(export_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # A denser fixture index makes the service/query parity check insensitive
    # to approximate misses while still exercising the production backend.
    build(export_dir, index_dir, M=min(48, max(2, n_models - 1)),
          ef_construction=400)
    return export_dir, index_dir, z_m, z_d, occupancy


def test_default_handoff_and_occupancy(tmp_path):
    export_dir, index_dir, z_m, z_d, occupancy = _fixture(
        tmp_path, "occupancy", n_models=220, with_occupancy=True)
    retriever = ModelRetriever.load(index_dir, export_dir=export_dir)
    out = retriever.retrieve(z_d[0])

    assert isinstance(out, RetrievalOutput)
    assert DEFAULT_HANDOFF_K == 200
    assert MAX_RETRIEVAL_K == 500
    assert out.requested_k == 200 and out.effective_k == 200
    assert len(out.candidates) == out.z_m.shape[0] == len(out.model_meta) == 200
    assert out.z_m.shape[1:] == (16,) and out.z_m.dtype == np.float32
    assert out.provenance["requested_k"] == 200
    assert out.provenance["effective_k"] == 200
    assert out.provenance["hub_occupancy_available"] is True
    assert retriever.has_hub_occupancy is True
    assert "hub_occupancy" in out.model_meta.columns

    for i, candidate in enumerate(out.candidates):
        assert isinstance(candidate, Candidate)
        assert candidate.rank == i + 1
        assert int(out.model_meta.iloc[i]["mappedID"]) == candidate.mapped_id
        assert out.model_meta.iloc[i]["unique_model_id"] == candidate.unique_model_id
        assert candidate.hub_occupancy == int(occupancy[candidate.mapped_id])
        assert int(out.model_meta.iloc[i]["hub_occupancy"]) == candidate.hub_occupancy
        assert np.array_equal(out.z_m[i], z_m[candidate.mapped_id])

    # The service delegates ranking to query.py; warm z_d must match exactly.
    handle = open_serving(index_dir, export_dir=export_dir)
    raw = warm_query(handle, 0, k=DEFAULT_HANDOFF_K)
    assert [c.mapped_id for c in out.candidates] == [r["mappedID"] for r in raw["results"]]
    assert [c.score for c in out.candidates] == [r["cosine"] for r in raw["results"]]
    assert all("hub_occupancy" in row for row in raw["results"])
    assert raw["provenance"]["requested_k"] == 200
    assert raw["provenance"]["effective_k"] == 200

    # K=500 is legal and truncates to the actual 220-model lake.
    wide = retriever.retrieve(z_d[0], top_k=500)
    assert wide.requested_k == 500 and wide.effective_k == 220
    assert len(wide.candidates) == 220
    assert wide.provenance["requested_k"] == 500
    assert wide.provenance["effective_k"] == 220


def test_bounds_and_old_export_compatibility(tmp_path):
    export_dir, index_dir, _z_m, z_d, _occupancy = _fixture(
        tmp_path, "legacy", n_models=17, with_occupancy=False)
    retriever = ModelRetriever.load(index_dir, export_dir=export_dir)
    out = retriever.retrieve(z_d[0])

    assert out.requested_k == 200 and out.effective_k == 17
    assert len(out.candidates) == 17
    assert retriever.has_hub_occupancy is False
    assert out.provenance["hub_occupancy_available"] is False
    assert "hub_occupancy" not in out.model_meta.columns
    assert all(candidate.hub_occupancy is None for candidate in out.candidates)

    handle = open_serving(index_dir, export_dir=export_dir)
    raw = warm_query(handle, 0, k=200)
    assert len(raw["results"]) == 17
    assert all("hub_occupancy" not in row for row in raw["results"])
    assert raw["provenance"]["requested_k"] == 200
    assert raw["provenance"]["effective_k"] == 17

    for invalid in (0, -1, 501):
        _expect(ValueError, lambda invalid=invalid: retriever.retrieve(z_d[0], k=invalid),
                "k must be in")
        _expect(ValueError, lambda invalid=invalid: warm_query(handle, 0, k=invalid),
                "k must be in")
    _expect(TypeError, lambda: retriever.retrieve(z_d[0], k=3.5), "k must be an integer")
    _expect(ValueError, lambda: retriever.retrieve(z_d[0], k=2, top_k=3),
            "either k or top_k")


def test_occupancy_integrity_is_enforced(tmp_path):
    export_dir, index_dir, _z_m, _z_d, occupancy = _fixture(
        tmp_path, "tamper", n_models=24, with_occupancy=True)
    occupancy = occupancy.copy()
    occupancy[0] += 1
    np.save(os.path.join(export_dir, "occupancy.npy"), occupancy)
    _expect(
        ManifestMismatch,
        lambda: ModelRetriever.load(index_dir, export_dir=export_dir),
        "hub occupancy sidecar occupancy.npy drifted",
    )


def main():
    with tempfile.TemporaryDirectory(prefix="stage3_service_") as root:
        test_default_handoff_and_occupancy(root)
        print("  1. default K=200, optional occupancy, candidate alignment: PASS")
        test_bounds_and_old_export_compatibility(root)
        print("  2. K bounds/truncation and old-export compatibility: PASS")
        test_occupancy_integrity_is_enforced(root)
        print("  3. occupancy manifest/hash integrity refusal: PASS")
    print("GATE G-H / TRACK B SERVICE CONTRACT: PASS")


if __name__ == "__main__":
    main()
