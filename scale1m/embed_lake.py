import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

_S1 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "stage1BuildTransferGraph")
if _S1 not in sys.path:
    sys.path.insert(0, _S1)

from scale1m.hf_crawl import utcnow, write_json_atomic

NAME_DIM = 64
DESC_DIM = 384
X_DIM = NAME_DIM + DESC_DIM
NAME_SEED = 42
ENCODER = "all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DEFAULT_BATCH = 256
CORE_INTAKE = os.path.join(_S1, "artifacts", "modellens_v2_lake",
                           "ml_model_intake.csv")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pct(x):
    return round(float(x), 5)


def halo_descriptors(models, families, sizes):
    from scale.modellens_build_graph import model_descriptor
    out = []
    for mid, fam, sz in zip(models, families, sizes):
        fam = "" if fam is None or (isinstance(fam, float) and np.isnan(fam)) else str(fam)
        out.append(model_descriptor(str(mid), fam, float(sz)))
    return out


def core_descriptor_parts(intake_csv=CORE_INTAKE):
    mi = pd.read_csv(intake_csv).set_index("model_id")
    model_ids = sorted(mi.index.astype(str))
    fam = mi.loc[model_ids, "family"].fillna("").astype(str).tolist()
    size_b = mi.loc[model_ids, "size_b"].to_numpy(dtype=float)
    return model_ids, fam, size_b


def core_descriptors(intake_csv=CORE_INTAKE):
    model_ids, fam, size_b = core_descriptor_parts(intake_csv)
    return model_ids, halo_descriptors(model_ids, fam, size_b)


def text_stats(texts):
    lens = np.array([len(t) for t in texts], dtype=np.float64)
    return {
        "n": int(len(texts)),
        "char_mean": round(float(lens.mean()), 3),
        "char_median": round(float(np.median(lens)), 3),
        "char_std": round(float(lens.std()), 3),
        "char_p10": round(float(np.percentile(lens, 10)), 3),
        "char_p90": round(float(np.percentile(lens, 90)), 3),
        "with_size_clause": _pct(np.mean([" params" in t for t in texts])),
        "with_family_clause": _pct(np.mean(["family " in t for t in texts])),
    }


def minilm(texts, batch_size=DEFAULT_BATCH, device=None, tag="halo.desc"):
    import torch
    from sentence_transformers import SentenceTransformer
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    enc = SentenceTransformer(ENCODER, device=dev, revision=ENCODER_REVISION)
    emb = enc.encode(texts, batch_size=batch_size, show_progress_bar=False,
                     convert_to_numpy=True, normalize_embeddings=False)
    print("[%s] MiniLM(%s) %d texts -> %s" % (tag, dev, len(texts), emb.shape),
          flush=True)
    return emb.astype(np.float32), dev


def name_embeddings(models):
    from dataset_embed.xm0_builder import build_name_embeddings
    return build_name_embeddings(list(models), token_dim=NAME_DIM, seed=NAME_SEED)


def extend_family_vocab(core_vocab, halo_families, vocab_path):
    from dataset_embed.xm0_builder import load_or_update_family_vocab

    ordered = sorted(core_vocab.items(), key=lambda kv: kv[1])
    assert [i for _, i in ordered] == list(range(len(ordered))), \
        "CORE family_vocab ids are not contiguous 0..n-1"
    os.makedirs(os.path.dirname(os.path.abspath(vocab_path)), exist_ok=True)
    pd.DataFrame({"family": [k for k, _ in ordered],
                  "family_id": [v for _, v in ordered]}).to_csv(vocab_path,
                                                                index=False)

    vocab_after = load_or_update_family_vocab(list(halo_families),
                                              vocab_path=vocab_path)
    for fam, fid in core_vocab.items():
        assert vocab_after.get(fam) == fid, \
            "family row moved: %s %s -> %s" % (fam, fid, vocab_after.get(fam))
    return vocab_after, len(vocab_after) - len(core_vocab)


def gate_core_verbatim(x, x_core):
    import torch
    return bool(torch.equal(torch.from_numpy(x[:x_core.shape[0]]), x_core))


def gate_row_order(x, models, n_core, k=100, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(n_core, len(models)), size=min(k, len(models) - n_core),
                     replace=False)
    redone = name_embeddings([models[i] for i in idx])
    delta = np.abs(redone - x[idx, :NAME_DIM]).max()
    return int(len(idx)), float(delta)


def separability_auc(x, n_core, seed=0, max_iter=2000, subsample=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    n_halo = len(x) - n_core
    m = min(n_core, n_halo) if subsample is None else min(subsample, n_core, n_halo)
    ci = rng.choice(n_core, size=m, replace=False)
    hi = rng.choice(np.arange(n_core, len(x)), size=m, replace=False)
    idx = np.concatenate([ci, hi])
    y = np.concatenate([np.zeros(m), np.ones(m)])

    out = {}
    for tag, sl in (("full", slice(0, X_DIM)),
                    ("e_name", slice(0, NAME_DIM)),
                    ("e_desc", slice(NAME_DIM, X_DIM))):
        xs = x[idx][:, sl]
        xtr, xte, ytr, yte = train_test_split(xs, y, test_size=0.5,
                                              random_state=seed, stratify=y)
        sc = StandardScaler().fit(xtr)
        clf = LogisticRegression(max_iter=max_iter)
        clf.fit(sc.transform(xtr), ytr)
        out[tag] = round(float(roc_auc_score(
            yte, clf.decision_function(sc.transform(xte)))), 4)
        print("  [auc] %-7s %.4f" % (tag, out[tag]), flush=True)
    out["n_per_class"] = int(m)
    return out


def _auc_of(emb, m, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    y = np.concatenate([np.zeros(m), np.ones(m)])
    xtr, xte, ytr, yte = train_test_split(emb, y, test_size=0.5,
                                          random_state=seed, stratify=y)
    sc = StandardScaler().fit(xtr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(xtr), ytr)
    return round(float(roc_auc_score(yte, clf.decision_function(
        sc.transform(xte)))), 4)


def auc_ablation(core_parts, halo_parts, m, batch_size=DEFAULT_BATCH,
                 device=None, seed=0):
    rng = np.random.default_rng(seed)
    variants = {"name_only": (False, False), "name_size": (False, True),
                "name_family": (True, False), "full": (True, True)}
    out = {"n_per_class": int(m)}
    picks = {}
    for tag, (ids, fams, sizes) in (("core", core_parts), ("halo", halo_parts)):
        sel = rng.choice(len(ids), size=min(m, len(ids)), replace=False)
        picks[tag] = ([ids[i] for i in sel], [fams[i] for i in sel],
                      np.asarray(sizes)[sel])
    assert len(picks["core"][0]) == len(picks["halo"][0]), \
        "ablation groups must be balanced"

    for name, (keep_fam, keep_size) in variants.items():
        texts = []
        for tag in ("core", "halo"):
            ids, fams, sizes = picks[tag]
            texts += halo_descriptors(
                ids,
                fams if keep_fam else [""] * len(ids),
                sizes if keep_size else np.full(len(ids), np.nan))
        emb, _ = minilm(texts, batch_size=batch_size, device=device,
                        tag="ablation." + name)
        out[name] = _auc_of(emb, len(picks["core"][0]), seed=seed)
        print("  [ablation] %-12s AUC %.4f" % (name, out[name]), flush=True)
    return out


def build(core_path, ladder_path, out_dir, rung, batch_size=DEFAULT_BATCH,
          device=None, seed=0, skip_auc=False, auc_subsample=None,
          ablation=0):
    import torch

    core = torch.load(core_path, weights_only=False)
    x_core = core["data"]["model"].x
    size_core = core["data"]["model"].size_bucket_id
    fam_core = core["data"]["model"].family_id
    core_vocab = dict(core["xm0_meta"]["family_vocab"])
    umi = core["unique_model_id"].sort_values("mappedID").reset_index(drop=True)
    n_core = int(x_core.shape[0])
    assert x_core.shape[1] == X_DIM, "CORE x is %s, expected [*, %d]" % (
        tuple(x_core.shape), X_DIM)

    ladder = pd.read_csv(ladder_path)
    n = len(ladder)
    assert ladder["mappedID"].tolist() == list(range(n)), "ladder mappedID broken"
    assert ladder.iloc[:n_core]["model"].tolist() == umi["model"].tolist(), \
        "ladder CORE prefix != CORE's own mappedID order"
    assert (ladder.iloc[:n_core]["layer"] == "core").all()
    halo = ladder.iloc[n_core:]
    print("[in] CORE %d (frozen) | HALO %d | rung %s" % (n_core, len(halo), rung),
          flush=True)

    fam_raw = halo["family"].fillna("").astype(str).tolist()
    size_b = halo["size_b"].to_numpy(dtype=float)
    texts = halo_descriptors(halo["model"].tolist(), fam_raw, size_b)
    e_name = name_embeddings(halo["model"].tolist())
    e_desc, dev = minilm(texts, batch_size=batch_size, device=device)
    x_halo = np.concatenate([e_name, e_desc], axis=1).astype(np.float32)
    assert x_halo.shape == (len(halo), X_DIM)

    x = np.concatenate([x_core.numpy(), x_halo], axis=0).astype(np.float32)

    from dataset_embed.xm0_builder import (
        param_count_to_size_bucket, build_family_ids, NUM_SIZE_BUCKETS)
    counts = [(b * 1e9 if not np.isnan(b) else None) for b in size_b]
    size_halo = np.array([param_count_to_size_bucket(c) for c in counts],
                         dtype=np.int64)
    size_id = np.concatenate([size_core.numpy(), size_halo])

    vocab_path = os.path.join(out_dir, rung, "family_vocab.csv")
    fams = [f if f else "Other" for f in fam_raw]
    vocab_after, n_new = extend_family_vocab(core_vocab, fams, vocab_path)
    fam_halo = build_family_ids(fams, vocab_after)
    fam_id = np.concatenate([fam_core.numpy(), fam_halo])

    d = os.path.join(out_dir, rung)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "x_m.npy"), x)
    np.save(os.path.join(d, "size_bucket_id.npy"), size_id)
    np.save(os.path.join(d, "family_id.npy"), fam_id)
    with open(os.path.join(d, "halo_descriptors_head.txt"), "w",
              encoding="utf-8") as fh:
        for t in texts[:50]:
            fh.write(t + "\n")

    print("\n[gates]", flush=True)
    g_core = gate_core_verbatim(x, x_core)
    g_shape = (x.shape == (n, X_DIM))
    g_nan = not bool(np.isnan(x).any()) and not bool(np.isinf(x).any())
    k, name_delta = gate_row_order(x, ladder["model"].tolist(), n_core, seed=seed)
    core_ids, core_fams, core_sizes = core_descriptor_parts()
    assert core_ids == umi["model"].tolist(), \
        "reconstructed CORE descriptor order != CORE mappedID order"
    core_texts = halo_descriptors(core_ids, core_fams, core_sizes)
    st_core, st_halo = text_stats(core_texts), text_stats(texts)
    auc = None if skip_auc else separability_auc(
        x, n_core, seed=seed, subsample=auc_subsample)
    abl = None
    if ablation:
        abl = auc_ablation((core_ids, core_fams, core_sizes),
                           (halo["model"].tolist(), fam_raw, size_b),
                           ablation, batch_size=batch_size, device=device,
                           seed=seed)

    zero_rows = int((np.abs(x_halo).sum(axis=1) == 0).sum())
    norms = np.linalg.norm(x_halo, axis=1)

    report = {
        "written_at": utcnow(), "rung": rung, "n": n, "n_core": n_core,
        "n_halo": int(len(halo)),
        "core_graph": os.path.abspath(core_path),
        "ladder_csv": os.path.abspath(ladder_path),
        "ladder_sha256": sha256_of(ladder_path),
        "encoder": ENCODER, "device": dev, "batch_size": batch_size,
        "name_seed": NAME_SEED, "name_dim": NAME_DIM, "desc_dim": DESC_DIM,
        "descriptor_fn": "scale.modellens_build_graph.model_descriptor",
        "x_m_sha256": sha256_of(os.path.join(d, "x_m.npy")),
        "x_m_shape": list(x.shape),
        "gates": {
            "core_verbatim": g_core,
            "shape_ok": bool(g_shape),
            "no_nan_inf": bool(g_nan),
            "family_vocab_append_only": True,
            "row_order_k": k,
            "row_order_max_delta": name_delta,
            "row_order_ok": bool(name_delta < 1e-6),
        },
        "family_vocab": {
            "core_rows": len(core_vocab), "after_rows": len(vocab_after),
            "new_rows": n_new, "path": os.path.abspath(vocab_path),
        },
        "size_bucket": {
            "num_buckets": NUM_SIZE_BUCKETS,
            "core_unknown_share": _pct((size_core.numpy() == 0).mean()),
            "halo_unknown_share": _pct((size_halo == 0).mean()),
        },
        "family_id": {
            "core_other_share": _pct((fam_core.numpy() == 0).mean()),
            "halo_other_share": _pct((fam_halo == 0).mean()),
            "halo_distinct_ids": int(len(np.unique(fam_halo))),
        },
        "descriptor_text": {"core": st_core, "halo": st_halo,
                            "char_mean_delta": round(
                                st_halo["char_mean"] - st_core["char_mean"], 3)},
        "halo_feature_health": {
            "all_zero_rows": zero_rows,
            "l2_norm_mean": round(float(norms.mean()), 4),
            "l2_norm_min": round(float(norms.min()), 4),
            "l2_norm_max": round(float(norms.max()), 4),
        },
        "separability_auc": auc,
        "auc_ablation": abl,
    }
    write_json_atomic(os.path.join(d, "FEATS_REPORT.json"), report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description="T4: build the rung feature matrix")
    p.add_argument("--rung", default="100k")
    p.add_argument("--core", default="stage1BuildTransferGraph/hgraph_ml_v2.pt")
    p.add_argument("--ladder", required=True)
    p.add_argument("--out", required=True, help="feats root; writes <out>/<rung>/")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-auc", action="store_true")
    p.add_argument("--auc-subsample", type=int, default=None)
    p.add_argument("--ablation", type=int, default=0,
                   help="rows per group for the G-B4 clause ablation (0 = off)")
    args = p.parse_args(argv)

    rep = build(args.core, args.ladder, args.out, args.rung,
                batch_size=args.batch_size, device=args.device, seed=args.seed,
                skip_auc=args.skip_auc, auc_subsample=args.auc_subsample,
                ablation=args.ablation)

    g = rep["gates"]
    print("\n[ok] x_m %s -> %s/%s" % (rep["x_m_shape"], args.out, args.rung))
    print("  sha256                 %s" % rep["x_m_sha256"][:16])
    print("  CORE verbatim          %s" % g["core_verbatim"])
    print("  shape / no NaN         %s / %s" % (g["shape_ok"], g["no_nan_inf"]))
    print("  row-order (%d rows)    max|d| %.2e -> %s"
          % (g["row_order_k"], g["row_order_max_delta"], g["row_order_ok"]))
    print("  family_vocab           %d -> %d (+%d)"
          % (rep["family_vocab"]["core_rows"], rep["family_vocab"]["after_rows"],
             rep["family_vocab"]["new_rows"]))
    print("  descriptor chars       CORE %.1f | HALO %.1f (delta %+0.1f)"
          % (rep["descriptor_text"]["core"]["char_mean"],
             rep["descriptor_text"]["halo"]["char_mean"],
             rep["descriptor_text"]["char_mean_delta"]))
    print("  size clause            CORE %.4f | HALO %.4f"
          % (rep["descriptor_text"]["core"]["with_size_clause"],
             rep["descriptor_text"]["halo"]["with_size_clause"]))
    print("  G-B4 separability AUC  %s" % rep["separability_auc"])
    if rep["auc_ablation"]:
        print("  G-B4 clause ablation   %s" % rep["auc_ablation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
