"""Stage-3 Phase-7 retrieval contract consumed by the Stage-4 reranker.

``ModelRetriever`` is deliberately thin: it uses the same manifest-verified
serving handle and ranking function as :mod:`query`, then gathers the model
embeddings and discrete metadata in candidate order.  Track B widens the
handoff to 200 candidates by default, with a hard ceiling of 500.

Example::

    retriever = ModelRetriever.load("artifacts/indexes/hf1000d_G2")
    out = retriever.retrieve(z_query)  # requested_k=200
    # out.candidates[i], out.z_m[i], and out.model_meta.iloc[i] are one model
"""

from dataclasses import dataclass
import os

import numpy as np
import pandas as pd

from ModelLakeFishing.stage3HNSW.build_index import ManifestMismatch
from ModelLakeFishing.stage3HNSW.export_embeddings import sha256_file
from ModelLakeFishing.stage3HNSW.query import (
    DEFAULT_HANDOFF_K,
    MAX_RETRIEVAL_K,
    _provenance,
    _ranked,
    _validate_k,
    open_serving,
)


DEFAULT_EF_SEARCH = 64


@dataclass(frozen=True, slots=True)
class Candidate:
    """One ranked model. ``hub_occupancy`` is ``None`` for old exports."""

    rank: int
    mapped_id: int
    unique_model_id: str
    score: float
    hub_occupancy: int | None = None


@dataclass(frozen=True, slots=True)
class RetrievalOutput:
    """Candidate-aligned Stage-4 input.

    Row ``i`` refers to the same model in ``candidates``, ``z_m``, and
    ``model_meta``. ``requested_k`` records the caller's handoff width while
    ``effective_k`` is safely truncated to the number of indexed models.
    """

    candidates: tuple[Candidate, ...]
    z_m: np.ndarray
    model_meta: pd.DataFrame
    provenance: dict
    requested_k: int
    effective_k: int


def _verify_export_sidecar(handle, filename):
    """Verify a non-index sidecar against the export bound to the index."""
    path = os.path.join(handle["export_dir"], filename)
    recorded = handle["export_man"].get("files", {}).get(filename)
    if recorded is None:
        raise ManifestMismatch(f"export manifest does not bind {filename}")
    if not os.path.exists(path):
        raise ManifestMismatch(f"export sidecar missing: {path}")
    actual = sha256_file(path)
    if actual != recorded:
        raise ManifestMismatch(
            f"export sidecar {filename} drifted: {actual[:16]}... "
            f"!= recorded {recorded[:16]}...")
    return path


class ModelRetriever:
    """Manifest-verified HNSW retrieval service and Stage-4 handoff."""

    default_k = DEFAULT_HANDOFF_K
    max_k = MAX_RETRIEVAL_K

    def __init__(self, handle, z_m, model_meta):
        self._handle = handle
        self._z_m = z_m
        self._dim = int(z_m.shape[1])

        n_models = len(handle["model_ids"])
        if z_m.ndim != 2 or z_m.shape[0] != n_models:
            raise ManifestMismatch(
                f"z_m shape {z_m.shape} is incompatible with {n_models} model ids")
        required = {"mappedID", "unique_model_id", "size_bucket_id", "family_id"}
        missing = required - set(model_meta.columns)
        if missing:
            raise ManifestMismatch(
                f"model_meta.parquet missing required columns: {sorted(missing)}")
        ordered = model_meta.sort_values("mappedID").reset_index(drop=True)
        mapped_ids = ordered["mappedID"].to_numpy()
        if len(ordered) != n_models or not np.array_equal(mapped_ids, np.arange(n_models)):
            raise ManifestMismatch("model_meta mappedID rows are not exactly 0..N-1")
        expected_uids = handle["model_ids"]["unique_model_id"].astype(str).to_numpy()
        if not np.array_equal(ordered["unique_model_id"].astype(str).to_numpy(), expected_uids):
            raise ManifestMismatch("model_meta unique_model_id rows disagree with model_ids.csv")
        self._model_meta = ordered.set_index("mappedID", drop=False)

    @classmethod
    def load(cls, index_dir, *, export_dir=None):
        """Load an index and all Stage-4 sidecars after manifest verification."""
        handle = open_serving(index_dir, export_dir=export_dir)
        # The current lake is intentionally small enough to keep z_m resident;
        # a regular array also avoids holding a Windows file handle forever.
        z_m = np.load(os.path.join(handle["export_dir"], "z_m.npy"))
        meta_path = _verify_export_sidecar(handle, "model_meta.parquet")
        model_meta = pd.read_parquet(meta_path)
        return cls(handle, z_m, model_meta)

    @property
    def has_hub_occupancy(self):
        return self._handle.get("hub_occupancy") is not None

    def retrieve(self, z_query, k=DEFAULT_HANDOFF_K, ef_search=DEFAULT_EF_SEARCH,
                 *, top_k=None):
        """Retrieve a candidate-aligned Stage-4 batch.

        ``k`` is the API spelling frozen in the Stage-3 plan. ``top_k`` is an
        explicit alias for configuration layers; pass only one of them. Both
        are bounded to 1..500. The default handoff is 200.
        """
        if top_k is not None:
            if k != DEFAULT_HANDOFF_K:
                raise ValueError("pass either k or top_k, not both")
            k = top_k
        k = _validate_k(k)
        if isinstance(ef_search, (bool, np.bool_)) or not isinstance(
                ef_search, (int, np.integer)):
            raise TypeError(f"ef_search must be a positive integer, got {ef_search!r}")
        ef_search = int(ef_search)
        if ef_search <= 0:
            raise ValueError(f"ef_search must be positive, got {ef_search}")

        z_query = np.asarray(z_query, dtype="float32")
        if z_query.shape != (self._dim,):
            raise ValueError(
                f"z_query shape must be ({self._dim},), got {z_query.shape}")
        if not np.all(np.isfinite(z_query)):
            raise ValueError("z_query contains NaN or infinity")
        norm = float(np.linalg.norm(z_query))
        if norm == 0.0:
            raise ValueError("z_query must be non-zero")
        query_was_unit = bool(np.isclose(norm, 1.0, atol=1e-5))
        if not query_was_unit:
            z_query = np.asarray(z_query / norm, dtype="float32")

        rows = _ranked(self._handle, z_query, k, ef_search)
        mapped_ids = np.asarray([row["mappedID"] for row in rows], dtype=np.int64)
        ranked_z_m = np.asarray(self._z_m[mapped_ids], dtype="float32").copy()
        ranked_meta = self._model_meta.loc[mapped_ids].reset_index(drop=True).copy()

        occupancy = self._handle.get("hub_occupancy")
        if occupancy is not None:
            ranked_meta["hub_occupancy"] = occupancy[mapped_ids].astype(np.int64)
        candidates = tuple(Candidate(
            rank=row["rank"],
            mapped_id=row["mappedID"],
            unique_model_id=row["unique_model_id"],
            score=row["cosine"],
            hub_occupancy=row.get("hub_occupancy"),
        ) for row in rows)

        effective_k = len(candidates)
        provenance = _provenance(
            self._handle,
            "service",
            ef_search,
            extra={
                "requested_k": k,
                "effective_k": effective_k,
                "max_k": MAX_RETRIEVAL_K,
                "query_was_unit_normalized": query_was_unit,
            },
        )
        return RetrievalOutput(
            candidates=candidates,
            z_m=ranked_z_m,
            model_meta=ranked_meta,
            provenance=provenance,
            requested_k=k,
            effective_k=effective_k,
        )


__all__ = [
    "Candidate",
    "DEFAULT_EF_SEARCH",
    "DEFAULT_HANDOFF_K",
    "MAX_RETRIEVAL_K",
    "ModelRetriever",
    "RetrievalOutput",
]
