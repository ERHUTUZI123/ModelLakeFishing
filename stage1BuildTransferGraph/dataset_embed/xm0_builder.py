"""
xm0_builder — build model node features x_m^(0) for GraphSAGE.

x_m^(0) = [e_name || e_desc || e_size || e_fam]

Row i of every output matrix corresponds to the model whose mappedID == i
(the row-order contract from unique_model_id, enforced by get_model_names).

e_name : hash-avg of model name tokens (frozen random projection, fast, no deps)
e_desc : sentence-transformer encoding of README text (frozen, pre-computed offline)
e_size : LEARNABLE — builder outputs only the discrete size_bucket_id column
         (log10 half-decade buckets); the embedding table lives in GraphSAGE
e_fam  : LEARNABLE — builder outputs only the discrete family_id column plus
         an append-only family vocab; the embedding table lives in GraphSAGE
"""

import re
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

from dataset_embed.utils.fetch_metadata import (
    get_model_names,
    get_model_descriptions,
    get_model_param_counts,
    get_model_families,
    FAMILY_OTHER,
    KNOWN_FAMILIES,
)

# ── constants ─────────────────────────────────────────────────────────────────

HASH_BUCKETS: int = 10_000

# size bucketing: log10(param_count) discretized into half-decade buckets.
# Bucket 0 is reserved for unknown (param count missing / unfetchable).
# Boundaries are FIXED constants, not data-driven quantiles, so bucket ids
# stay stable when new models are added — a cached bucket id never changes
# meaning, and the GraphSAGE nn.Embedding(NUM_SIZE_BUCKETS, size_dim) rows
# keep their identity across runs.
SIZE_BUCKET_UNKNOWN: int = 0
SIZE_LOG10_MIN: float = 5.0    # 1e5  params; anything smaller clamps to bucket 1
SIZE_LOG10_MAX: float = 12.0   # 1e12 params; anything larger clamps to the top bucket
SIZE_BUCKET_WIDTH: float = 0.5  # half a decade per bucket

# buckets 1..14 cover [1e5, 1e12); +1 for the unknown bucket -> 15 total.
# This is the num_embeddings for the learnable size embedding table.
NUM_SIZE_BUCKETS: int = int((SIZE_LOG10_MAX - SIZE_LOG10_MIN) / SIZE_BUCKET_WIDTH) + 1

# family vocab: id 0 is reserved for FAMILY_OTHER ("Other"), by the same
# convention as SIZE_BUCKET_UNKNOWN. Unlike size buckets the family vocabulary
# is NOT a fixed constant — static-rule families are seeded first, and
# dynamically extracted families join once they clear FAMILY_MIN_COUNT.
# num_embeddings for the learnable family table = len(vocab) returned by
# load_or_update_family_vocab (grows append-only across runs).
FAMILY_ID_OTHER: int = 0
# Dynamically extracted family names need >= this many models to earn a vocab
# slot; rarer ones fold into Other. Same threshold ModelLens used for its
# auto-generated families (">= 3 models"), and for the same reason: a learnable
# embedding row seen by 1-2 models is untrainable noise.
FAMILY_MIN_COUNT: int = 3


# ── name tokenizer (ModelLens strategy) ──────────────────────────────────────

def _split_model_name(name: str) -> List[str]:
    """
    Tokenize a HuggingFace model name into a deduplicated token list.

    Strategy (same as ModelLens ModelNameAvgEncoder._split):
      - keep full lowercased name
      - keep the part after '/' (repo name without org)
      - split on /, _, -, space and keep non-empty tokens
      - deduplicate while preserving order

    Examples
    --------
    "google/gemma-4-31B-it"
      -> ["google/gemma-4-31b-it", "gemma-4-31b-it", "google", "gemma", "4", "31b", "it"]
    """
    n = (name or "").strip().lower()
    if not n:
        return []

    toks = [n]
    if "/" in n:
        toks.append(n.split("/")[-1])
    toks.extend([t for t in re.split(r"[\/_\-\s]+", n) if t])

    out, seen = [], set()
    for t in toks:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _hash_token(tok: str) -> int:
    return int(hashlib.md5(tok.encode()).hexdigest(), 16) % HASH_BUCKETS


# ── name embedding builder ────────────────────────────────────────────────────

def build_name_embeddings(
    model_names: List[str],
    token_dim: int = 64,
    seed: int = 42,
    learnable: bool = False,
    emb_cache_path: Path | str | None = None,
    checkpoint_every: int = 1000,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Compute e_name for every model in model_names.

    Uses the ModelLens hash-average approach:
      1. Tokenize model name (org + repo-name + sub-tokens).
      2. Hash each token with MD5 -> bucket in [0, HASH_BUCKETS).
      3. Look up in a fixed random tok_emb table (frozen random projection).
      4. Average all token embeddings -> one vector per model.

    Supports resume-on-interrupt via emb_cache_path (.npz). Already-computed
    models are skipped; the cache is written every checkpoint_every models.
    tok_emb is always recomputed from seed (deterministic, negligible cost).

    Parameters
    ----------
    model_names      : list of model name strings in mappedID order
    token_dim        : embedding dimension; keep modest (one of 4 components)
    seed             : RNG seed for tok_emb; must stay fixed across runs so
                       cached embeddings remain valid
    learnable        : if True, also returns the tok_emb table so the caller
                       can register it as nn.Parameter during training
    emb_cache_path   : path to .npz checkpoint; created/updated in place
    checkpoint_every : save after every N newly computed models

    Returns
    -------
    embeddings : np.ndarray of shape [num_models, token_dim], dtype float32
    tok_emb    : np.ndarray of shape [HASH_BUCKETS, token_dim], dtype float32
                 (only returned when learnable=True, as a tuple)
    """
    rng = np.random.default_rng(seed)
    tok_emb = rng.standard_normal((HASH_BUCKETS, token_dim)).astype(np.float32)
    norms = np.linalg.norm(tok_emb, axis=1, keepdims=True)
    tok_emb /= norms + 1e-8

    # ── load existing cache ───────────────────────────────────────────────────
    cache: dict[str, np.ndarray] = {}
    if emb_cache_path is not None:
        emb_cache_path = Path(emb_cache_path)
        if emb_cache_path.exists():
            payload = np.load(emb_cache_path, allow_pickle=True)
            if "model_names" in payload and "embeddings" in payload:
                for n, e in zip(payload["model_names"].tolist(), payload["embeddings"]):
                    cache[str(n)] = e

    to_compute = [n for n in model_names if n not in cache]
    print(f"[build_name_embeddings] {len(cache)} cached, {len(to_compute)} to compute")

    # ── compute missing models in chunks ─────────────────────────────────────
    for chunk_start in range(0, len(to_compute), checkpoint_every):
        chunk = to_compute[chunk_start: chunk_start + checkpoint_every]
        for name in chunk:
            toks = _split_model_name(name)
            if toks:
                idxs = [_hash_token(t) for t in toks]
                cache[name] = tok_emb[idxs].mean(axis=0)
            else:
                cache[name] = np.zeros(token_dim, dtype=np.float32)

        done = min(chunk_start + checkpoint_every, len(to_compute))
        print(f"  [{done}/{len(to_compute)}] computed", end="")

        if emb_cache_path is not None:
            _save_emb_npz(emb_cache_path, cache)
            print("  -> checkpoint saved", end="")
        print()

    # ── assemble result in mappedID order ─────────────────────────────────────
    embeddings = np.zeros((len(model_names), token_dim), dtype=np.float32)
    for i, name in enumerate(model_names):
        if name in cache:
            embeddings[i] = cache[name]

    if learnable:
        return embeddings, tok_emb
    return embeddings


def build_name_component(
    unique_model_id: pd.DataFrame,
    token_dim: int = 64,
    seed: int = 42,
) -> np.ndarray:
    """
    Full pipeline: unique_model_id -> e_name matrix.

    Calls get_model_names to enforce the mappedID row-order contract, then
    calls build_name_embeddings.

    Returns
    -------
    np.ndarray of shape [num_models, token_dim], dtype float32,
    row i = e_name for the model with mappedID == i.
    """
    names = get_model_names(unique_model_id)
    return build_name_embeddings(names, token_dim=token_dim, seed=seed)


# ── desc embedding builder ────────────────────────────────────────────────────

def _save_emb_npz(path: Path, cache: dict) -> None:
    """Persist a name->embedding cache dict as a .npz checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(cache.keys())
    embs  = np.stack(list(cache.values()), axis=0)
    np.savez(path, model_names=np.array(names), embeddings=embs)


def build_desc_embeddings(
    model_names: List[str],
    model_descriptions: List[str],
    emb_cache_path: Path | str | None = None,
    encoder_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    checkpoint_every: int = 10,
) -> np.ndarray:
    """
    Encode model README texts into frozen embeddings.

    Uses sentence-transformers for semantic encoding — NOT hash-avg, because
    README text is natural language where semantic meaning matters and
    hash-avg over deduplicated tokens collapses under template pollution
    (every README shares ## Usage / ## Training tokens).

    Storage format matches ModelLens (model_desp_emb_path):
      .npz with keys 'model_names' (str array) and 'embeddings' (float32 matrix)
    so pre-computed files from both projects are interchangeable.

    Supports resume-on-interrupt: already-encoded models are loaded from the
    .npz cache and skipped; the cache is written every checkpoint_every models
    so a crash loses at most checkpoint_every entries.

    Parameters
    ----------
    model_names        : model id strings in mappedID order
    model_descriptions : README texts in the same mappedID order
                         (use get_model_descriptions(unique_model_id, ...))
    emb_cache_path     : path to .npz checkpoint file; created/updated in place
    encoder_name       : sentence-transformers model name
    batch_size         : encoding batch size passed to SentenceTransformer
    checkpoint_every   : save the .npz after every N newly encoded models

    Returns
    -------
    np.ndarray of shape [num_models, emb_dim], dtype float32,
    row i = e_desc for mappedID i. Zero vector for empty descriptions.
    """
    assert len(model_names) == len(model_descriptions), (
        f"model_names ({len(model_names)}) and model_descriptions "
        f"({len(model_descriptions)}) must have the same length and order; "
        "use build_desc_component(unique_model_id, ...) to guarantee alignment."
    )

    # ── load existing cache (name -> embedding vector) ────────────────────────
    cache: dict[str, np.ndarray] = {}
    if emb_cache_path is not None:
        emb_cache_path = Path(emb_cache_path)
        if emb_cache_path.exists():
            payload = np.load(emb_cache_path, allow_pickle=True)
            if "model_names" in payload and "embeddings" in payload:
                for n, e in zip(payload["model_names"].tolist(), payload["embeddings"]):
                    cache[str(n)] = e

    # ── identify models not yet encoded ───────────────────────────────────────
    to_encode = [
        (name, desc)
        for name, desc in zip(model_names, model_descriptions)
        if name not in cache
    ]
    print(f"[build_desc_embeddings] {len(cache)} cached, {len(to_encode)} to encode")

    if to_encode:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(encoder_name)

        for chunk_start in range(0, len(to_encode), checkpoint_every):
            chunk = to_encode[chunk_start: chunk_start + checkpoint_every]
            texts = [d if d.strip() else "[no description]" for _, d in chunk]

            embs = encoder.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)

            for (name, _), emb in zip(chunk, embs):
                cache[name] = emb

            done = min(chunk_start + checkpoint_every, len(to_encode))
            print(f"  [{done}/{len(to_encode)}] encoded", end="")

            if emb_cache_path is not None:
                _save_emb_npz(emb_cache_path, cache)
                print("  -> checkpoint saved", end="")
            print()

    # ── assemble result in mappedID order ─────────────────────────────────────
    emb_dim = next(iter(cache.values())).shape[0] if cache else 384
    result = np.zeros((len(model_names), emb_dim), dtype=np.float32)
    for i, name in enumerate(model_names):
        if name in cache:
            result[i] = cache[name]
    return result


def build_desc_component(
    unique_model_id: pd.DataFrame,
    desc_cache_path: Path | str | None = None,
    emb_cache_path: Path | str | None = None,
    max_chars: int = 2000,
    encoder_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    checkpoint_every: int = 10,
) -> np.ndarray:
    """
    Full pipeline: unique_model_id -> e_desc matrix.

    Step 1: get_model_descriptions  -> List[str] of README texts (with caching)
    Step 2: build_desc_embeddings   -> np.ndarray [num_models, emb_dim]

    Parameters
    ----------
    desc_cache_path : CSV cache for raw README texts (passed to get_model_descriptions)
    emb_cache_path  : .npz cache for computed embeddings (skips re-encoding on rerun)

    Returns
    -------
    np.ndarray of shape [num_models, emb_dim], dtype float32,
    row i = e_desc for the model with mappedID == i.
    """
    names = get_model_names(unique_model_id)
    descs = get_model_descriptions(
        unique_model_id,
        cache_path=desc_cache_path,
        max_chars=max_chars,
    )
    return build_desc_embeddings(
        model_names=names,
        model_descriptions=descs,
        emb_cache_path=emb_cache_path,
        encoder_name=encoder_name,
        batch_size=batch_size,
        checkpoint_every=checkpoint_every,
    )


# ── size bucket builder ───────────────────────────────────────────────────────

def param_count_to_size_bucket(param_count: int | None) -> int:
    """
    Map one raw parameter count to a discrete size bucket id.

    log10 transform first (param counts span ~7 orders of magnitude, so
    linear bucketing would dump every model under 1B into one bucket),
    then discretize into fixed half-decade buckets:

      None / non-positive            -> SIZE_BUCKET_UNKNOWN (0)
      log10(count) < SIZE_LOG10_MIN  -> bucket 1 (clamped)
      log10(count) >= SIZE_LOG10_MAX -> bucket NUM_SIZE_BUCKETS - 1 (clamped)
      otherwise                      -> 1 + floor((log10(count) - MIN) / WIDTH)

    Examples
    --------
    125M params -> log10 = 8.10 -> bucket 7
    7B   params -> log10 = 9.85 -> bucket 10
    70B  params -> log10 = 10.85 -> bucket 12
    """
    if param_count is None or param_count <= 0:
        return SIZE_BUCKET_UNKNOWN
    log = np.log10(param_count)
    bucket = 1 + int((log - SIZE_LOG10_MIN) // SIZE_BUCKET_WIDTH)
    return max(1, min(bucket, NUM_SIZE_BUCKETS - 1))


def build_size_bucket_ids(param_counts: List[int | None]) -> np.ndarray:
    """
    Convert raw param counts into the size_bucket_id column.

    NOTE: returns discrete indices, NOT vectors. e_size is a LEARNABLE
    embedding — the actual vectors come from nn.Embedding(NUM_SIZE_BUCKETS,
    size_dim) defined inside the GraphSAGE model and updated by gradients.
    Pre-computing fixed vectors here would silently freeze the component
    (the plan.md step-3 trap).

    Parameters
    ----------
    param_counts : raw counts in mappedID order
                   (use get_model_param_counts(unique_model_id, ...))

    Returns
    -------
    np.ndarray of shape [num_models], dtype int64,
    index i = size bucket id for mappedID i. Unknown counts -> bucket 0.
    """
    return np.array(
        [param_count_to_size_bucket(c) for c in param_counts],
        dtype=np.int64,
    )


def build_size_component(
    unique_model_id: pd.DataFrame,
    size_cache_path: Path | str | None = None,
    request_delay: float = 0.5,
) -> np.ndarray:
    """
    Full pipeline: unique_model_id -> size_bucket_id array.

    Step 1: get_model_param_counts -> List[int | None] of raw param counts
    Step 2: build_size_bucket_ids  -> np.ndarray [num_models] of bucket ids

    Caching lives at step 1: raw param counts are fetched once from
    HuggingFace and persisted to size_cache_path CSV with resume-on-interrupt
    (same mechanism as the desc pipeline). The log10-bucketing in step 2 is
    deterministic arithmetic over fixed boundaries, so it needs no cache of
    its own — rebucketing cached counts is free, and keeping the cache at
    the raw-count level means we can retune bucket boundaries later without
    re-fetching anything.

    Parameters
    ----------
    unique_model_id : DataFrame with columns ['model', 'mappedID']
    size_cache_path : CSV cache for raw param counts
                      (passed to get_model_param_counts)
    request_delay   : seconds between HuggingFace requests

    Returns
    -------
    np.ndarray of shape [num_models], dtype int64,
    row i = size_bucket_id for the model with mappedID == i.
    Pair with nn.Embedding(NUM_SIZE_BUCKETS, size_dim) at training time.
    """
    counts = get_model_param_counts(
        unique_model_id,
        cache_path=size_cache_path,
        request_delay=request_delay,
    )
    return build_size_bucket_ids(counts)


# ── family id builder ─────────────────────────────────────────────────────────

def load_or_update_family_vocab(
    families: List[str],
    vocab_path: Path | str | None = None,
    min_count: int = FAMILY_MIN_COUNT,
) -> dict[str, int]:
    """
    Load (or create) the family -> integer-id vocabulary, append-only.

    Why not ModelLens's approach: their build_model2family assigns ids by
    sorted(observed_family_set), so adding ONE new model with a new family
    reshuffles every id — and each nn.Embedding row silently changes which
    family it represents. Fine for their frozen benchmark snapshot, fatal
    for an incrementally growing model lake. Here an id, once assigned,
    NEVER changes meaning: existing entries are never re-numbered or removed,
    new families only append at the end.

    Vocab construction order (deterministic):
      id 0                  : FAMILY_OTHER, always
      next                  : KNOWN_FAMILIES (static rules) in table order,
                              always admitted regardless of count
      then                  : families loaded from vocab_path (prior runs)
      finally               : new dynamically extracted families observed
                              >= min_count times in this batch, sorted
                              alphabetically; rarer ones map to Other at
                              lookup time (they can still be promoted in a
                              later run once the lake grows)

    Parameters
    ----------
    families   : family strings for the current batch (any order)
    vocab_path : CSV with columns ['family', 'family_id']; created if absent,
                 appended to on later runs. This file IS the identity of the
                 nn.Embedding rows — keep it next to the model checkpoint.
    min_count  : admission threshold for dynamically extracted families

    Returns
    -------
    dict family -> id. len(dict) = num_embeddings for
    nn.Embedding(len(vocab), family_dim).
    """
    vocab: dict[str, int] = {FAMILY_OTHER: FAMILY_ID_OTHER}
    for fam in KNOWN_FAMILIES:
        if fam not in vocab:
            vocab[fam] = len(vocab)

    # ── load prior vocab (append-only contract: ids must match) ───────────────
    if vocab_path is not None:
        vocab_path = Path(vocab_path)
        if vocab_path.exists():
            prior = pd.read_csv(vocab_path)
            for fam, fid in zip(prior["family"], prior["family_id"]):
                fam, fid = str(fam), int(fid)
                if fam in vocab:
                    assert vocab[fam] == fid, (
                        f"family vocab conflict for '{fam}': file says {fid}, "
                        f"rebuilt says {vocab[fam]} — KNOWN_FAMILIES order changed "
                        "after the vocab file was created. Either restore the old "
                        "rule order or delete the vocab file AND retrain."
                    )
                else:
                    assert fid == len(vocab), (
                        f"family vocab file is not contiguous at '{fam}' (id {fid}, "
                        f"expected {len(vocab)}); file corrupted or hand-edited."
                    )
                    vocab[fam] = fid

    # ── admit new dynamic families that clear the threshold ───────────────────
    counts = pd.Series(families).value_counts()
    new_fams = sorted(
        fam for fam, cnt in counts.items()
        if fam not in vocab and cnt >= min_count
    )
    for fam in new_fams:
        vocab[fam] = len(vocab)
    if new_fams:
        print(f"[load_or_update_family_vocab] admitted {len(new_fams)} new families: {new_fams}")

    # ── persist ────────────────────────────────────────────────────────────────
    if vocab_path is not None:
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"family": list(vocab.keys()), "family_id": list(vocab.values())}
        ).to_csv(vocab_path, index=False)

    return vocab


def build_family_ids(families: List[str], vocab: dict[str, int]) -> np.ndarray:
    """
    Convert family strings into the family_id column.

    NOTE: returns discrete indices, NOT vectors — same contract as
    build_size_bucket_ids. e_fam is a LEARNABLE embedding; the vectors come
    from nn.Embedding(len(vocab), family_dim) inside GraphSAGE.

    Families absent from the vocab (dynamically extracted but below
    FAMILY_MIN_COUNT) map to FAMILY_ID_OTHER.

    Parameters
    ----------
    families : family strings in mappedID order
               (use get_model_families(unique_model_id, ...))
    vocab    : family -> id mapping from load_or_update_family_vocab

    Returns
    -------
    np.ndarray of shape [num_models], dtype int64,
    index i = family id for mappedID i.
    """
    return np.array(
        [vocab.get(f, FAMILY_ID_OTHER) for f in families],
        dtype=np.int64,
    )


def build_fam_component(
    unique_model_id: pd.DataFrame,
    family_cache_path: Path | str | None = None,
    vocab_path: Path | str | None = None,
    min_count: int = FAMILY_MIN_COUNT,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Full pipeline: unique_model_id -> (family_id array, family vocab).

    Step 1: get_model_families        -> List[str] (static rules + dynamic
                                         extraction, with manual-correction cache)
    Step 2: load_or_update_family_vocab -> append-only family -> id mapping
    Step 3: build_family_ids          -> np.ndarray [num_models] of ids

    Two caches, two different jobs:
      family_cache_path : model -> family strings; lets manual corrections in
                          the CSV survive reruns (fetch_metadata contract)
      vocab_path        : family -> id; pins nn.Embedding row identity across
                          runs (THIS file must be versioned with the trained
                          model — losing it orphans the trained embedding rows)

    Returns
    -------
    family_ids : np.ndarray [num_models] int64, row i = family id for mappedID i
    vocab      : family -> id dict; pass len(vocab) as num_embeddings and keep
                 it for inference-time lookup of unseen models
    """
    families = get_model_families(unique_model_id, cache_path=family_cache_path)
    vocab = load_or_update_family_vocab(families, vocab_path=vocab_path, min_count=min_count)
    return build_family_ids(families, vocab), vocab


# ── full x_m^(0) assembly ─────────────────────────────────────────────────────

def build_xm0(
    unique_model_id: pd.DataFrame,
    token_dim: int = 64,
    seed: int = 42,
    desc_cache_path: Path | str | None = None,
    desc_emb_cache_path: Path | str | None = None,
    size_cache_path: Path | str | None = None,
    family_cache_path: Path | str | None = None,
    family_vocab_path: Path | str | None = None,
    max_chars: int = 2000,
    encoder_name: str = "all-MiniLM-L6-v2",
) -> dict:
    """
    Assemble the full x_m^(0) package: frozen concat + learnable indices.

    x_m^(0) = [e_name || e_desc || e_size || e_fam] is split across two
    stages because frozen and learnable components live in different places
    (plan.md step 3):

      frozen half   : e_name || e_desc concatenated HERE into one matrix ->
                      goes into HGraph as data['model'].x via model_features
      learnable half: size_bucket_id / family_id stay DISCRETE here, attached
                      to the graph as int columns; the actual e_size / e_fam
                      vectors are looked up and concatenated by
                      ModelNodeEncoder inside the GNN at training time.

    This deliberately deviates from ModelLens MLP.py (encode_model():879-911,
    forward():950-964): their concat is split per-batch across two call sites
    because size/family are gated optional inputs of a pairwise scorer; ours
    follows the frozen-vs-learnable storage boundary of a node-feature
    pipeline. Their e_id component is dropped entirely — GraphSAGE must stay
    inductive, so no ID embedding (and no ID dropout) exists anywhere.

    Returns
    -------
    dict with keys
      frozen           : np.ndarray [N, token_dim + desc_dim] float32,
                         row i = [e_name || e_desc] for mappedID i
      size_bucket_id   : np.ndarray [N] int64
      family_id        : np.ndarray [N] int64
      family_vocab     : dict family -> id (keep next to the model checkpoint)
      num_size_buckets : num_embeddings for the size table
      num_families     : num_embeddings for the family table
      name_dim/desc_dim: slice boundaries inside frozen (for ablations)
    """
    e_name = build_name_component(unique_model_id, token_dim=token_dim, seed=seed)
    e_desc = build_desc_component(
        unique_model_id,
        desc_cache_path=desc_cache_path,
        emb_cache_path=desc_emb_cache_path,
        max_chars=max_chars,
        encoder_name=encoder_name,
    )
    size_bucket_id = build_size_component(unique_model_id, size_cache_path=size_cache_path)
    family_id, vocab = build_fam_component(
        unique_model_id,
        family_cache_path=family_cache_path,
        vocab_path=family_vocab_path,
    )

    n = len(unique_model_id)
    assert e_name.shape[0] == n and e_desc.shape[0] == n, (
        f"frozen component row mismatch: e_name {e_name.shape[0]}, "
        f"e_desc {e_desc.shape[0]}, expected {n}"
    )
    assert size_bucket_id.shape == (n,) and family_id.shape == (n,), (
        f"index column shape mismatch: size {size_bucket_id.shape}, "
        f"family {family_id.shape}, expected ({n},)"
    )

    frozen = np.concatenate([e_name, e_desc], axis=1).astype(np.float32)
    assert not np.isnan(frozen).any(), "NaN in frozen features"
    assert 0 <= size_bucket_id.min() and size_bucket_id.max() < NUM_SIZE_BUCKETS
    assert 0 <= family_id.min() and family_id.max() < len(vocab)

    return {
        "frozen": frozen,
        "size_bucket_id": size_bucket_id,
        "family_id": family_id,
        "family_vocab": vocab,
        "num_size_buckets": NUM_SIZE_BUCKETS,
        "num_families": len(vocab),
        "name_dim": e_name.shape[1],
        "desc_dim": e_desc.shape[1],
    }