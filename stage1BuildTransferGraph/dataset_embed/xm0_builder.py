"""
xm0_builder — build model node features x_m^(0) for GraphSAGE.

x_m^(0) = [e_name || e_desc || e_size || e_fam]

Row i of every output matrix corresponds to the model whose mappedID == i
(the row-order contract from unique_model_id, enforced by get_model_names).

e_name : hash-avg of model name tokens (frozen random projection, fast, no deps)
e_desc : sentence-transformer encoding of README text (frozen, pre-computed offline)
e_size / e_fam : learnable embeddings indexed by discrete bucket/family id (added later)
"""

import re
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

from dataset_embed.utils.fetch_metadata import get_model_names, get_model_descriptions

# ── constants ─────────────────────────────────────────────────────────────────

HASH_BUCKETS: int = 10_000


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
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Compute e_name for every model in model_names.

    Uses the ModelLens hash-average approach:
      1. Tokenize model name (org + repo-name + sub-tokens).
      2. Hash each token with MD5 -> bucket in [0, HASH_BUCKETS).
      3. Look up in a fixed random tok_emb table (frozen random projection).
      4. Average all token embeddings -> one vector per model.

    Parameters
    ----------
    model_names : list of model name strings in mappedID order
                  (use get_model_names(unique_model_id) to get this list)
    token_dim   : embedding dimension per token; keep modest since this is
                  one of 4 components in x_m^(0)
    seed        : RNG seed for the tok_emb table; fix this so offline-computed
                  features are reproducible across runs
    learnable   : if True, returns the raw tok_emb table alongside embeddings
                  so the caller can register it as nn.Parameter during training;
                  if False (default), treat as frozen random projection

    Returns
    -------
    embeddings : np.ndarray of shape [num_models, token_dim], dtype float32
    tok_emb    : np.ndarray of shape [HASH_BUCKETS, token_dim], dtype float32
                 (only returned when learnable=True, as a tuple)
    """
    rng = np.random.default_rng(seed)
    tok_emb = rng.standard_normal((HASH_BUCKETS, token_dim)).astype(np.float32)
    # row-normalise so frozen random projections have unit magnitude
    norms = np.linalg.norm(tok_emb, axis=1, keepdims=True)
    tok_emb /= norms + 1e-8

    embeddings = np.zeros((len(model_names), token_dim), dtype=np.float32)
    for i, name in enumerate(model_names):
        toks = _split_model_name(name)
        if not toks:
            continue
        idxs = [_hash_token(t) for t in toks]
        embeddings[i] = tok_emb[idxs].mean(axis=0)

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

def build_desc_embeddings(
    model_names: List[str],
    model_descriptions: List[str],
    emb_cache_path: Path | str | None = None,
    encoder_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
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

    Parameters
    ----------
    model_names        : model id strings in mappedID order
    model_descriptions : README texts in the same mappedID order
                         (use get_model_descriptions(unique_model_id, ...))
    emb_cache_path     : if given and the .npz exists, skip encoding and load
                         directly; otherwise encode and save
    encoder_name       : sentence-transformers model name
    batch_size         : encoding batch size

    Returns
    -------
    np.ndarray of shape [num_models, emb_dim], dtype float32,
    row i = e_desc for mappedID i. Zero vector for empty descriptions.
    """
    if emb_cache_path is not None:
        emb_cache_path = Path(emb_cache_path)
        if emb_cache_path.exists():
            payload = np.load(emb_cache_path, allow_pickle=True)
            if "model_names" in payload and "embeddings" in payload:
                lookup = {str(n): i for i, n in enumerate(payload["model_names"].tolist())}
                emb_dim = int(payload["embeddings"].shape[1])
                result = np.zeros((len(model_names), emb_dim), dtype=np.float32)
                for i, name in enumerate(model_names):
                    idx = lookup.get(name)
                    if idx is not None:
                        result[i] = payload["embeddings"][idx]
                return result

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(encoder_name)
    texts = [d if d.strip() else "[no description]" for d in model_descriptions]
    print(f"[build_desc_embeddings] encoding {len(texts)} descriptions with {encoder_name} ...")
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    if emb_cache_path is not None:
        emb_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(emb_cache_path,
                 model_names=np.array(model_names),
                 embeddings=embeddings)
        print(f"[build_desc_embeddings] saved to {emb_cache_path}")

    return embeddings


def build_desc_component(
    unique_model_id: pd.DataFrame,
    desc_cache_path: Path | str | None = None,
    emb_cache_path: Path | str | None = None,
    max_chars: int = 2000,
    encoder_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
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
    )