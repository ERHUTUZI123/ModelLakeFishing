"""
xm0_builder — build model node features x_m^(0) for GraphSAGE.

x_m^(0) = [e_name || e_desc || e_size || e_fam]

Row i of every output matrix corresponds to the model whose mappedID == i
(the row-order contract from unique_model_id, enforced by get_model_names).

This module handles the name component (e_name).
desc / size / fam components are added in subsequent steps.
"""

import re
import hashlib
import numpy as np
import pandas as pd
from typing import List

from dataset_embed.utils.fetch_metadata import get_model_names

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


# ── top-level entry point ─────────────────────────────────────────────────────

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