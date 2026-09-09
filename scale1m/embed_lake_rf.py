"""
embed_lake_rf.py -- F4: the frozen model-feature matrix for the full lake.

Runbook: docs/1M/1Mplan.md 5 (F4). Guide: docs/1M/F4GPU.md. Record: docs/1M/F4.md.

WHAT IT BUILDS
    x_m.npy            [N, 448] float32 = [e_name 64 || e_desc 384]
    size_bucket_id.npy [N] int64
    family_id.npy      [N] int64
    family_vocab.csv   the only credential for e_fam row identity
    FEATS_REPORT.json  gates, text statistics, sha256

WHY NOT scale1m/embed_lake.py
    That one builds a rung with a frozen CORE prefix: it requires --core, copies
    the first 30,183 rows verbatim, and reports the CORE/HALO separability AUC.
    This lake has no CORE and no two populations (D-56/D-63), so the prefix copy
    and the AUC have nothing to act on. The descriptor, the encoder, the name
    seed and the batch shape are identical, so the two are comparable where it
    matters.

ONE DESCRIPTOR FOR THE WHOLE LAKE (constraint 2)
    Every row goes through scale.modellens_build_graph.model_descriptor, the
    same function every earlier rung used. It drops the "<x>B params" clause
    when size_b is missing and the "family <x>" clause when family is empty, so
    clause coverage is reported rather than assumed. The 12,680 rows appended by
    F3 have no HF record and therefore no size clause; that is a property of the
    lake, not a bug, and it is in the report.

ROW ORDER IS THE CONTRACT
    x_m[i] belongs to the model at ladder mappedID i. Nothing downstream
    re-checks this, so the builder samples rows from BOTH segments -- the
    snapshot prefix and the appended tail -- recomputes e_name from the id, and
    compares. The tail is the segment F3 introduced and the one most likely to
    be mis-joined.

Run (from ModelLakeFishing/):
    python -m scale1m.embed_lake_rf --ladder <ladder.parquet> --out <feats dir>
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_S1 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "stage1BuildTransferGraph")
if _S1 not in sys.path:
    sys.path.insert(0, _S1)

from scale1m.embed_lake import (DEFAULT_BATCH, ENCODER, ENCODER_REVISION,
                                NAME_DIM, NAME_SEED, X_DIM, minilm,
                                name_embeddings, sha256_of)
from scale1m.hf_crawl import utcnow, write_json_atomic

SHARD_ROWS = 1_000_000        # one .npy part per million rows, resumable


def descriptors(models, families, sizes):
    """The single descriptor function, applied to every row."""
    from scale.modellens_build_graph import model_descriptor
    out = []
    for mid, fam, sz in zip(models, families, sizes):
        f = "" if fam is None or (isinstance(fam, float) and np.isnan(fam)) else str(fam)
        s = float(sz) if sz is not None and not pd.isna(sz) else float("nan")
        out.append(model_descriptor(str(mid), f, s))
    return out


def size_bucket_ids(sizes):
    from dataset_embed.xm0_builder import param_count_to_size_bucket
    return np.asarray(
        [param_count_to_size_bucket(None if pd.isna(s) else int(float(s) * 1e9))
         for s in sizes], dtype=np.int64)


def family_columns(families, vocab_path):
    from dataset_embed.xm0_builder import build_family_ids, load_or_update_family_vocab
    fams = [("" if f is None or pd.isna(f) else str(f)) for f in families]
    vocab = load_or_update_family_vocab(fams, vocab_path=vocab_path)
    ids = np.asarray(build_family_ids(fams, vocab), dtype=np.int64)
    return ids, vocab


def _part(out_dir, i):
    return os.path.join(out_dir, "x_desc_part-%03d.npy" % i)


def encode_desc(ladder, out_dir, batch_size, device, resume=True):
    """MiniLM over every descriptor, one .npy per SHARD_ROWS, resumable.

    Descriptors are generated per shard rather than all at once: three million
    Python strings are a few hundred MB that would sit alongside the matrix for
    no reason. Only the statistics survive each shard.
    """
    n = len(ladder)
    n_parts = (n + SHARD_ROWS - 1) // SHARD_ROWS
    dev_used = None
    lens, has_size, has_family = [], [], []
    head, tail = [], []
    for i in range(n_parts):
        path = _part(out_dir, i)
        lo, hi = i * SHARD_ROWS, min((i + 1) * SHARD_ROWS, n)
        sl = ladder.iloc[lo:hi]
        texts = descriptors(sl["model"], sl["family"], sl["size_b"])
        lens.append(np.fromiter((len(t) for t in texts), dtype=np.int32,
                                count=len(texts)))
        has_size.append(np.fromiter((" params" in t for t in texts), dtype=bool,
                                    count=len(texts)))
        has_family.append(np.fromiter(("family " in t for t in texts), dtype=bool,
                                      count=len(texts)))
        if i == 0:
            head = texts[:50]
        if i == n_parts - 1:
            tail = texts[-20:]
        if resume and os.path.exists(path):
            got = np.load(path, mmap_mode="r")
            if got.shape == (hi - lo, 384):
                print("[resume] part %d/%d already done" % (i + 1, n_parts), flush=True)
                del texts, got
                continue
            del got
            os.remove(path)
        t0 = time.time()
        emb, dev_used = minilm(texts, batch_size=batch_size, device=device,
                               tag="rf.desc.part%d" % i)
        np.save(path, emb)
        print("[part %d/%d] %d rows in %.1fs (%.0f rec/s)"
              % (i + 1, n_parts, hi - lo, time.time() - t0,
                 (hi - lo) / max(time.time() - t0, 1e-9)), flush=True)
        del emb, texts
    return n_parts, dev_used, (np.concatenate(lens), np.concatenate(has_size),
                               np.concatenate(has_family)), head, tail


def stats_from_arrays(lens, has_size, has_family, lo=0, hi=None):
    hi = len(lens) if hi is None else hi
    L = lens[lo:hi].astype(np.float64)
    if L.size == 0:
        return None
    return {"n": int(L.size),
            "char_mean": round(float(L.mean()), 3),
            "char_median": round(float(np.median(L)), 3),
            "char_std": round(float(L.std()), 3),
            "char_p10": round(float(np.percentile(L, 10)), 3),
            "char_p90": round(float(np.percentile(L, 90)), 3),
            "with_size_clause": round(float(has_size[lo:hi].mean()), 5),
            "with_family_clause": round(float(has_family[lo:hi].mean()), 5)}


def assemble(out_dir, n_parts, models, chunk=250_000):
    """[e_name || e_desc] written straight into an on-disk float32 array.

    The matrix is 5.41 GB at full scale, which does not fit in this machine's
    free RAM alongside the ladder and the descriptors. It is therefore built as
    a memmap and filled in chunks: e_name is computed 250k ids at a time, and
    e_desc is copied part by part. Nothing here ever holds more than a few
    hundred MB.
    """
    n = len(models)
    xpath = os.path.join(out_dir, "x_m.npy")
    x = np.lib.format.open_memmap(xpath, mode="w+", dtype=np.float32,
                                  shape=(n, X_DIM))
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        x[lo:hi, :NAME_DIM] = name_embeddings(models[lo:hi])
        print("[e_name] %d/%d" % (hi, n), flush=True)
    lo = 0
    for i in range(n_parts):
        part = np.load(_part(out_dir, i), mmap_mode="r")
        for a in range(0, part.shape[0], chunk):
            b = min(a + chunk, part.shape[0])
            x[lo + a:lo + b, NAME_DIM:] = part[a:b]
        lo += part.shape[0]
        del part
    assert lo == n, "parts cover %d rows, ladder has %d" % (lo, n)
    x.flush()
    return x, xpath


def finite_and_nonzero(x, chunk=250_000):
    """Chunked so the checks never materialise a second copy of the matrix."""
    finite, zero_rows = True, 0
    norms = np.empty(x.shape[0], dtype=np.float32)
    for lo in range(0, x.shape[0], chunk):
        hi = min(lo + chunk, x.shape[0])
        block = np.asarray(x[lo:hi], dtype=np.float32)
        finite &= bool(np.isfinite(block).all())
        n = np.linalg.norm(block, axis=1)
        norms[lo:hi] = n
        zero_rows += int((n == 0).sum())
    return finite, zero_rows, norms


def gate_row_order(x, models, n_snapshot, k=100, seed=0):
    """Sample from BOTH segments and recompute e_name from the id."""
    rng = np.random.default_rng(seed)
    idx = np.concatenate([
        rng.choice(n_snapshot, size=min(k, n_snapshot), replace=False),
        rng.choice(np.arange(n_snapshot, len(models)),
                   size=min(k, len(models) - n_snapshot), replace=False)
    ]) if len(models) > n_snapshot else rng.choice(len(models), size=k, replace=False)
    recomputed = name_embeddings([models[int(i)] for i in idx])
    delta = float(np.abs(recomputed - x[idx, :NAME_DIM]).max())
    return {"k": int(len(idx)), "max_delta": delta, "ok": delta < 1e-6,
            "sampled_from_tail": int((idx >= n_snapshot).sum())}


def build(ladder_path, out_dir, batch_size=DEFAULT_BATCH, device=None,
          seed=0, resume=True):
    os.makedirs(out_dir, exist_ok=True)
    ladder = pd.read_parquet(ladder_path, columns=["mappedID", "model", "family",
                                                   "size_b", "in_snapshot"])
    n = len(ladder)
    assert ladder["mappedID"].tolist() == list(range(n)), "ladder mappedID broken"
    n_snapshot = int(ladder["in_snapshot"].sum())
    models = ladder["model"].tolist()
    print("[in] %d rows (%d snapshot + %d appended) from %s"
          % (n, n_snapshot, n - n_snapshot, os.path.basename(ladder_path)), flush=True)

    t0 = time.time()
    n_parts, dev, (lens, has_size, has_family), head, tail = encode_desc(
        ladder, out_dir, batch_size, device, resume=resume)
    stats = stats_from_arrays(lens, has_size, has_family)
    stats_snapshot = stats_from_arrays(lens, has_size, has_family, 0, n_snapshot)
    stats_tail = stats_from_arrays(lens, has_size, has_family, n_snapshot, n)
    with open(os.path.join(out_dir, "descriptors_head.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(head + ["... (tail of the ladder below) ..."] + tail))
    print("[desc] %d texts | char mean %.1f | size clause %.4f | family clause %.4f"
          % (n, stats["char_mean"], stats["with_size_clause"],
             stats["with_family_clause"]), flush=True)

    x, xpath = assemble(out_dir, n_parts, models)

    size_id = size_bucket_ids(ladder["size_b"])
    vocab_path = os.path.join(out_dir, "family_vocab.csv")
    fam_id, vocab = family_columns(ladder["family"], vocab_path)
    np.save(os.path.join(out_dir, "size_bucket_id.npy"), size_id)
    np.save(os.path.join(out_dir, "family_id.npy"), fam_id)

    finite, zero_rows, norms = finite_and_nonzero(x)
    gates = {
        "shape_ok": bool(x.shape == (n, X_DIM)),
        "no_nan_inf": bool(finite),
        "no_all_zero_rows": zero_rows == 0,
        "row_order": gate_row_order(x, models, n_snapshot, seed=seed),
        "size_bucket_len": int(size_id.shape[0]) == n,
        "family_id_len": int(fam_id.shape[0]) == n,
        "family_id_in_range": bool(int(fam_id.max()) < len(vocab)),
    }
    report = {
        "artifact": "F4 feature matrix (RF)", "written_at": utcnow(),
        "ladder": {"path": os.path.abspath(ladder_path), "rows": n,
                   "snapshot_rows": n_snapshot, "appended_rows": n - n_snapshot},
        "encoder": ENCODER, "encoder_revision": ENCODER_REVISION,
        "batch_size": batch_size, "device": dev,
        "name_seed": NAME_SEED, "name_dim": NAME_DIM, "x_dim": X_DIM,
        "descriptor_fn": "scale.modellens_build_graph.model_descriptor",
        "x_m_shape": list(x.shape), "x_m_dtype": str(x.dtype),
        "x_m_sha256": sha256_of(xpath),
        "x_m_norm": {"mean": round(float(norms.mean()), 4),
                     "min": round(float(norms.min()), 4),
                     "max": round(float(norms.max()), 4)},
        "text_stats": {"all": stats, "snapshot": stats_snapshot, "appended": stats_tail},
        "size_bucket": {"unknown_share": round(float((size_id == 0).mean()), 5),
                        "distinct": int(len(np.unique(size_id)))},
        "family_vocab": {"rows": len(vocab), "path": vocab_path,
                         "sha256": sha256_of(vocab_path)
                         if os.path.exists(vocab_path) else None,
                         "other_share": round(float((fam_id == 0).mean()), 5)},
        "gates": gates,
        "wallclock_s": round(time.time() - t0, 1),
    }
    write_json_atomic(os.path.join(out_dir, "FEATS_REPORT.json"), report)
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="F4: build the RF feature matrix")
    p.add_argument("--ladder", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="debug: only the first N ladder rows")
    args = p.parse_args(argv)

    ladder = args.ladder
    if args.limit:
        df = pd.read_parquet(ladder).head(args.limit)
        ladder = os.path.join(args.out, "_ladder_head.parquet")
        os.makedirs(args.out, exist_ok=True)
        df.to_parquet(ladder, index=False)

    rep = build(ladder, args.out, batch_size=args.batch_size, device=args.device,
                seed=args.seed, resume=not args.no_resume)
    g = rep["gates"]
    print("\n[ok] x_m %s -> %s" % (rep["x_m_shape"], args.out))
    print("  sha256              %s" % rep["x_m_sha256"][:16])
    print("  shape / finite      %s / %s" % (g["shape_ok"], g["no_nan_inf"]))
    print("  row order (%d rows, %d from tail)  max|d| %.2e -> %s"
          % (g["row_order"]["k"], g["row_order"]["sampled_from_tail"],
             g["row_order"]["max_delta"], g["row_order"]["ok"]))
    print("  family_vocab        %d rows, Other %.4f"
          % (rep["family_vocab"]["rows"], rep["family_vocab"]["other_share"]))
    print("  size bucket unknown %.4f" % rep["size_bucket"]["unknown_share"])
    print("  size / family clause %.4f / %.4f"
          % (rep["text_stats"]["all"]["with_size_clause"],
             rep["text_stats"]["all"]["with_family_clause"]))
    bad = [k for k, v in g.items() if v is False] + \
          ([] if g["row_order"]["ok"] else ["row_order"])
    if bad:
        print("[FAIL] " + ", ".join(sorted(set(bad))))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
