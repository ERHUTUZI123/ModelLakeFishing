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


HASH_BUCKETS: int = 10_000

SIZE_BUCKET_UNKNOWN: int = 0
SIZE_LOG10_MIN: float = 5.0
SIZE_LOG10_MAX: float = 12.0
SIZE_BUCKET_WIDTH: float = 0.5

NUM_SIZE_BUCKETS: int = int((SIZE_LOG10_MAX - SIZE_LOG10_MIN) / SIZE_BUCKET_WIDTH) + 1

FAMILY_ID_OTHER: int = 0
FAMILY_MIN_COUNT: int = 3


def _split_model_name(name: str) -> List[str]:
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


def build_name_embeddings(
    model_names: List[str],
    token_dim: int = 64,
    seed: int = 42,
    learnable: bool = False,
    emb_cache_path: Path | str | None = None,
    checkpoint_every: int = 1000,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tok_emb = rng.standard_normal((HASH_BUCKETS, token_dim)).astype(np.float32)
    norms = np.linalg.norm(tok_emb, axis=1, keepdims=True)
    tok_emb /= norms + 1e-8

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
    names = get_model_names(unique_model_id)
    return build_name_embeddings(names, token_dim=token_dim, seed=seed)


def _save_emb_npz(path: Path, cache: dict) -> None:
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
    assert len(model_names) == len(model_descriptions), (
        f"model_names ({len(model_names)}) and model_descriptions "
        f"({len(model_descriptions)}) must have the same length and order; "
        "use build_desc_component(unique_model_id, ...) to guarantee alignment."
    )

    cache: dict[str, np.ndarray] = {}
    if emb_cache_path is not None:
        emb_cache_path = Path(emb_cache_path)
        if emb_cache_path.exists():
            payload = np.load(emb_cache_path, allow_pickle=True)
            if "model_names" in payload and "embeddings" in payload:
                for n, e in zip(payload["model_names"].tolist(), payload["embeddings"]):
                    cache[str(n)] = e

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


def param_count_to_size_bucket(param_count: int | None) -> int:
    if param_count is None or param_count <= 0:
        return SIZE_BUCKET_UNKNOWN
    log = np.log10(param_count)
    bucket = 1 + int((log - SIZE_LOG10_MIN) // SIZE_BUCKET_WIDTH)
    return max(1, min(bucket, NUM_SIZE_BUCKETS - 1))


def build_size_bucket_ids(param_counts: List[int | None]) -> np.ndarray:
    return np.array(
        [param_count_to_size_bucket(c) for c in param_counts],
        dtype=np.int64,
    )


def build_size_component(
    unique_model_id: pd.DataFrame,
    size_cache_path: Path | str | None = None,
    request_delay: float = 0.5,
) -> np.ndarray:
    counts = get_model_param_counts(
        unique_model_id,
        cache_path=size_cache_path,
        request_delay=request_delay,
    )
    return build_size_bucket_ids(counts)


def load_or_update_family_vocab(
    families: List[str],
    vocab_path: Path | str | None = None,
    min_count: int = FAMILY_MIN_COUNT,
) -> dict[str, int]:
    vocab: dict[str, int] = {FAMILY_OTHER: FAMILY_ID_OTHER}
    for fam in KNOWN_FAMILIES:
        if fam not in vocab:
            vocab[fam] = len(vocab)

    prior_vocab = None
    if vocab_path is not None:
        vocab_path = Path(vocab_path)
        if vocab_path.exists():
            prior = pd.read_csv(vocab_path, keep_default_na=False)
            prior_vocab = {}
            for fam, fid in zip(prior["family"], prior["family_id"]):
                fam, fid = str(fam), int(fid)
                prior_vocab[fam] = fid
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

    counts = pd.Series(families).value_counts()
    new_fams = sorted(
        fam for fam, cnt in counts.items()
        if fam not in vocab and cnt >= min_count
    )
    for fam in new_fams:
        vocab[fam] = len(vocab)
    if new_fams:
        print(f"[load_or_update_family_vocab] admitted {len(new_fams)} new families: {new_fams}")

    if vocab_path is not None and prior_vocab != vocab:
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"family": list(vocab.keys()), "family_id": list(vocab.values())}
        ).to_csv(vocab_path, index=False)

    return vocab


def build_family_ids(families: List[str], vocab: dict[str, int]) -> np.ndarray:
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
    families = get_model_families(unique_model_id, cache_path=family_cache_path)
    vocab = load_or_update_family_vocab(families, vocab_path=vocab_path, min_count=min_count)
    return build_family_ids(families, vocab), vocab


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
