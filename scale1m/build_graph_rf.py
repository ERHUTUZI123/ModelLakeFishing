"""
build_graph_rf.py -- F5: assemble the full-lake HGraph.

Runbook: docs/1M/1Mplan.md 5 (F5). Record: docs/1M/F5.md.

INPUTS
    ladder_rf/full_model_ids.parquet     row order, frozen by F3
    ladder_rf/full_dataset_ids.parquet   dataset node order, frozen by F3
    feats_rf/x_m.npy + size_bucket_id + family_id   the model side, from F4
    rf/canon/supervision_merged.parquet  the six-source edges, from F2
    rf/canon/part-*.parquet              lineage_base per model, from F2
    datasets_full/dataset_cards_merged.parquet      card text, from F1.5

WHAT IT BUILDS
    x_d, the dataset side, in the shape every rung has used since D0:
        x_d (458) = [e_name 64 (seed 43) || e_card 384 (MiniLM) || e_stats 10]
    five edge types:
        trained_on / rev_trained_on      the supervision, from F2
        similar_to                       k=20 cosine KNN over e_card
        is_base_of / rev_is_base_of      lineage, hashmap join on normalised id
    and writes the whole thing in the scale1m.graph_store sharded layout, so
    the 5.41 GB feature matrix is never materialised in RAM.

WHY x_m IS COPIED, NOT REBUILT
    F4 already wrote x_m.npy in exactly the .npy layout graph_store reads. The
    graph directory takes that file as-is; rebuilding it through torch would
    cost 5.41 GB of RAM this machine does not have and could only produce the
    same bytes.

Run (from ModelLakeFishing/):
    python -m scale1m.build_graph_rf --out <graph dir>
"""
import argparse
import collections
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

_S1 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "stage1BuildTransferGraph")
if _S1 not in sys.path:
    sys.path.insert(0, _S1)

from scale1m.canonicalize_rf import NODE_SEP, sha256_of
from scale1m.embed_lake import DEFAULT_BATCH, ENCODER, minilm
from scale1m.hf_crawl import data_root, utcnow, write_json_atomic
from scale1m.verify_raw import normalize

NAME_DIM_D, CARD_DIM, STAT_DIM = 64, 384, 10
XD_DIM = NAME_DIM_D + CARD_DIM + STAT_DIM
NAME_SEED_D = 43           # the dataset side has always used 43, models use 42
SIM_K = 20


def _text(v):
    """pandas hands back NaN for a null string column, and NaN is truthy."""
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def dataset_texts(nodes, cards):
    """dataset_descriptor() over the F1.5 cards, name-only where none matched."""
    from d0_build_graph import dataset_descriptor
    c = cards.set_index(["dataset", "task"])
    out, with_card = [], 0
    for r in nodes.itertuples():
        key = (r.dataset, r.task)
        rec = None
        if key in c.index:
            row = c.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if str(row.get("card_source", "")).startswith("hf_card"):
                with_card += 1
                # a missing description arrives as NaN, and `NaN or ""` is NaN
                rec = {"description": _text(row.get("description")),
                       "tags": json.loads(_text(row.get("tags")) or "[]"),
                       "cardData": {"task_categories":
                                    json.loads(_text(row.get("task_categories"))
                                               or "[]")}}
        node_str = r.dataset + (" " + r.task if r.task else "")
        out.append(dataset_descriptor(rec, node_str))
    return out, with_card


def dataset_stats(nodes, edges):
    """e_stats: the ten columns D0 defined, computed off the supervision."""
    g = edges.groupby("node", observed=True)["weight"]
    agg = g.agg(["size", "mean", "std", "min", "max"]).fillna(0.0)
    roots = nodes["dataset"].map(lambda d: str(d).split("/")[0])
    root_sizes = roots.value_counts().to_dict()
    x = np.zeros((len(nodes), STAT_DIM), dtype=np.float32)
    for i, r in enumerate(nodes.itertuples()):
        a = agg.loc[r.node] if r.node in agg.index else None
        n = float(a["size"]) if a is not None else 0.0
        x[i, 0] = np.log1p(n)
        x[i, 1] = np.log1p(n)
        x[i, 2] = float(a["mean"]) if a is not None else 0.0
        x[i, 3] = float(a["std"]) if a is not None else 0.0
        x[i, 4] = float(a["min"]) if a is not None else 0.0
        x[i, 5] = float(a["max"]) if a is not None else 0.0
        x[i, 6] = np.log1p(root_sizes.get(roots.iloc[i], 0))
        x[i, 7] = float(bool(r.gold_eligible))
        x[i, 8] = 0.0                      # has_content: the content view is off
        x[i, 9] = 0.0
    return np.nan_to_num(x), roots.tolist()


def build_xd(nodes, cards, edges, out_dir, batch_size, device):
    from dataset_embed.xm0_builder import build_name_embeddings
    texts, with_card = dataset_texts(nodes, cards)
    e_name = build_name_embeddings([r.node.replace(NODE_SEP, " ") for r in
                                    nodes.itertuples()],
                                   token_dim=NAME_DIM_D, seed=NAME_SEED_D)
    e_card, dev = minilm(texts, batch_size=batch_size, device=device, tag="rf.card")
    e_stats, roots = dataset_stats(nodes, edges)
    xd = np.concatenate([e_name, e_card, e_stats], axis=1).astype(np.float32)
    assert xd.shape == (len(nodes), XD_DIM), xd.shape
    return xd, e_card, roots, {"with_hf_card": with_card, "device": dev,
                               "char_mean": round(float(np.mean(
                                   [len(t) for t in texts])), 2)}


def lineage_edges(rf_dir, id2idx):
    """base --is_base_of--> derivative, O(N) hashmap join on the normalised id."""
    src, dst, rel = [], [], []
    rel_counter = collections.Counter()
    declared = self_ref = 0
    for part in sorted(glob.glob(os.path.join(rf_dir, "canon", "part-*.parquet"))):
        df = pd.read_parquet(part, columns=["model", "lineage_base",
                                            "lineage_relation"])
        df = df[df["lineage_base"].notna()]
        declared += len(df)
        for m, b, r in zip(df["model"], df["lineage_base"], df["lineage_relation"]):
            i = id2idx.get(normalize(m))
            j = id2idx.get(normalize(b))
            if i is None or j is None:
                continue
            if i == j:
                self_ref += 1
                continue
            src.append(j)
            dst.append(i)
            r = _text(r) or "unknown"      # NaN is truthy; _text() flattens it
            rel.append(r)
            rel_counter[r] += 1
    return (np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64),
            rel, {"declared": declared, "self_referential": self_ref,
                  "relation": dict(rel_counter)})


def lineage_stats(src, dst, n_models, meta):
    children = collections.Counter(src.tolist())
    counts = sorted(children.values())
    return {
        "edges": int(len(src)),
        "declared_base_model": meta["declared"],
        "self_referential_dropped": meta["self_referential"],
        "resolution_rate_pct": round(100.0 * len(src) / max(meta["declared"], 1), 3),
        "relation": meta["relation"],
        "distinct_parents": len(children),
        "models_with_a_parent": int(len(set(dst.tolist()))),
        "children_p50": counts[len(counts) // 2] if counts else 0,
        "children_p90": counts[int(0.9 * (len(counts) - 1))] if counts else 0,
        "children_p99": counts[int(0.99 * (len(counts) - 1))] if counts else 0,
        "parents_over_1000_children": sum(1 for c in counts if c > 1000),
        "max_children": counts[-1] if counts else 0,
        "coverage_pct": round(100.0 * len(set(dst.tolist())) / n_models, 3),
    }


def build(ladder_dir, feats_dir, rf_dir, cards_path, out_dir,
          batch_size=DEFAULT_BATCH, device=None):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    models = pd.read_parquet(os.path.join(ladder_dir, "full_model_ids.parquet"),
                             columns=["mappedID", "model", "in_snapshot"])
    nodes = pd.read_parquet(os.path.join(ladder_dir, "full_dataset_ids.parquet"))
    nodes["task"] = nodes["task"].fillna("")
    n_m, n_d = len(models), len(nodes)
    print("[in] %d models, %d dataset nodes" % (n_m, n_d), flush=True)

    edges = pd.read_parquet(os.path.join(rf_dir, "canon", "supervision_merged.parquet"))
    cards = pd.read_parquet(cards_path)
    cards["task"] = cards["task"].fillna("")

    # ---- dataset side -----------------------------------------------------
    xd, e_card, roots, card_meta = build_xd(nodes, cards, edges, out_dir,
                                            batch_size, device)
    print("[xd] %s | cards %d/%d | %.1fs"
          % (xd.shape, card_meta["with_hf_card"], n_d, time.time() - t0), flush=True)

    # ---- edges ------------------------------------------------------------
    id2idx = {normalize(m): i for i, m in zip(models["mappedID"], models["model"])}
    node2idx = {n: i for i, n in zip(nodes["mappedID"], nodes["node"])}
    miss_m = edges["model"].map(lambda m: normalize(m) not in id2idx).sum()
    miss_n = edges["node"].map(lambda n: n not in node2idx).sum()
    assert miss_m == 0 and miss_n == 0, \
        "supervision references %d unknown models / %d unknown nodes" % (miss_m, miss_n)
    tr_src = edges["model"].map(lambda m: id2idx[normalize(m)]).to_numpy(np.int64)
    tr_dst = edges["node"].map(node2idx.get).to_numpy(np.int64)
    tr_w = edges["weight"].to_numpy(np.float32)
    print("[edges] trained_on %d | %.1fs" % (len(tr_src), time.time() - t0), flush=True)

    from scale.modellens_build_graph import topk_knn
    s_src, s_dst, s_w = topk_knn(e_card, SIM_K)
    print("[edges] similar_to %d" % len(s_src), flush=True)

    l_src, l_dst, l_rel, l_meta = lineage_edges(rf_dir, id2idx)
    lstats = lineage_stats(l_src, l_dst, n_m, l_meta)
    print("[edges] is_base_of %d (resolution %.2f%%, max hub %d)"
          % (lstats["edges"], lstats["resolution_rate_pct"],
             lstats["max_children"]), flush=True)
    rel_vocab = {r: i for i, r in enumerate(
        ["unknown", "finetune", "adapter", "quantized", "merge"])}
    l_rel_id = np.asarray([rel_vocab.get(r, 0) for r in l_rel], dtype=np.int64)

    # ---- write the sharded graph -----------------------------------------
    src_x = os.path.join(feats_dir, "x_m.npy")
    dst_x = os.path.join(out_dir, "x_model.npy")
    if not os.path.exists(dst_x) or sha256_of(src_x) != sha256_of(dst_x):
        print("[write] copying x_m.npy (5.41 GB) ...", flush=True)
        shutil.copyfile(src_x, dst_x)
    np.save(os.path.join(out_dir, "x_dataset.npy"), xd)

    size_id = np.load(os.path.join(feats_dir, "size_bucket_id.npy"))
    fam_id = np.load(os.path.join(feats_dir, "family_id.npy"))
    np.savez(os.path.join(out_dir, "nodes.npz"),
             **{"model.node_id": np.arange(n_m, dtype=np.int64),
                "model.size_bucket_id": size_id,
                "model.family_id": fam_id,
                "dataset.node_id": np.arange(n_d, dtype=np.int64),
                "dataset.task_type_id": np.zeros(n_d, dtype=np.int64),
                "dataset.n_class_bucket_id": np.zeros(n_d, dtype=np.int64),
                "dataset.arity_id": np.zeros(n_d, dtype=np.int64)})
    np.savez(os.path.join(out_dir, "edges.npz"), **{
        "model__trained_on__dataset__edge_index": np.stack([tr_src, tr_dst]),
        "model__trained_on__dataset__edge_attr": tr_w,
        "dataset__rev_trained_on__model__edge_index": np.stack([tr_dst, tr_src]),
        "dataset__rev_trained_on__model__edge_attr": tr_w.copy(),
        "dataset__similar_to__dataset__edge_index": np.stack([s_src, s_dst]),
        "dataset__similar_to__dataset__edge_attr": s_w,
        "model__is_base_of__model__edge_index": np.stack([l_src, l_dst]),
        "model__is_base_of__model__edge_attr": np.ones(len(l_src), dtype=np.float32),
        "model__is_base_of__model__relation_id": l_rel_id,
        "model__rev_is_base_of__model__edge_index": np.stack([l_dst, l_src]),
        "model__rev_is_base_of__model__edge_attr": np.ones(len(l_src), dtype=np.float32),
        "model__rev_is_base_of__model__relation_id": l_rel_id.copy()})

    models.rename(columns={"model": "model"})[["model", "mappedID"]].to_parquet(
        os.path.join(out_dir, "unique_model_id.parquet"), index=False)
    du = nodes[["node", "mappedID"]].rename(columns={"node": "dataset"}).copy()
    du["root"] = roots
    du.to_parquet(os.path.join(out_dir, "unique_dataset_id.parquet"), index=False)

    from dataset_embed.xm0_builder import FAMILY_MIN_COUNT
    # keep_default_na=False: two real family strings are literally "nan" and
    # "null", and pandas would parse both as NaN and collapse them into one
    # dict key, leaving the vocab one row short of num_families.
    vocab = pd.read_csv(os.path.join(feats_dir, "family_vocab.csv"),
                        keep_default_na=False)
    meta = {
        "format": "scale1m.graph_store/1",
        "written_at": utcnow(),
        "node_types": ["model", "dataset"],
        "edge_types": [["model", "trained_on", "dataset"],
                       ["dataset", "rev_trained_on", "model"],
                       ["dataset", "similar_to", "dataset"],
                       ["model", "is_base_of", "model"],
                       ["model", "rev_is_base_of", "model"]],
        "num_nodes": {"model": n_m, "dataset": n_d},
        "xm0_meta": {"num_size_buckets": int(size_id.max()) + 1,
                     "num_families": len(vocab),
                     "family_vocab": dict(zip(vocab["family"],
                                              vocab["family_id"].astype(int))),
                     "family_min_count": FAMILY_MIN_COUNT,
                     "name_dim": 64, "desc_dim": 384},
        "xd0_meta": {"num_task_types": 1, "task_type_vocab": {"Other": 0},
                     "n_class_buckets": 1, "num_arities": 1,
                     "arity_vocab": {"unknown": 0},
                     "view_dims": {"e_name": NAME_DIM_D, "e_card": CARD_DIM,
                                   "e_stats": STAT_DIM},
                     "encoder_name": ENCODER, "has_content_view": False,
                     "probe_views": "not used (RF)"},
        "provenance": {"rung": "full", "ladder_dir": os.path.abspath(ladder_dir),
                       "feats_dir": os.path.abspath(feats_dir),
                       "supervision": "rf/canon/supervision_merged.parquet",
                       "x_m_sha256": sha256_of(src_x)},
        "tensors": {}, "files": {},
    }
    for f in os.listdir(out_dir):
        if f != "meta.json":
            meta["files"][f] = sha256_of(os.path.join(out_dir, f))
    write_json_atomic(os.path.join(out_dir, "meta.json"), meta)
    write_json_atomic(os.path.join(out_dir, "lineage_stats.json"), lstats)

    report = {
        "artifact": "F5 graph (RF)", "written_at": utcnow(),
        "models": n_m, "dataset_nodes": n_d,
        "edges": {"trained_on": int(len(tr_src)), "similar_to": int(len(s_src)),
                  "is_base_of": int(len(l_src)),
                  "total_with_reverse": int(2 * len(tr_src) + len(s_src)
                                            + 2 * len(l_src))},
        "x_dataset": {"shape": list(xd.shape), **card_meta},
        "lineage": lstats,
        "wallclock_s": round(time.time() - t0, 1),
        "gates": {
            "supervision_endpoints_resolve": bool(miss_m == 0 and miss_n == 0),
            "trained_on_matches_f2": int(len(tr_src)) == int(len(edges)),
            "x_dataset_shape": list(xd.shape) == [n_d, XD_DIM],
            "x_dataset_finite": bool(np.isfinite(xd).all()),
            "similar_to_k": int(len(s_src)) == n_d * min(SIM_K, n_d - 1),
        },
    }
    write_json_atomic(os.path.join(out_dir, "GRAPH_REPORT.json"), report)
    return report


def main(argv=None) -> int:
    d = os.path.join(data_root(), "data1m")
    p = argparse.ArgumentParser(description="F5: build the RF graph")
    p.add_argument("--ladder", default=os.path.join(d, "ladder_rf"))
    p.add_argument("--feats", default=os.path.join(d, "feats_rf"))
    p.add_argument("--rf", default=os.path.join(d, "rf"))
    p.add_argument("--cards", default=os.path.join(d, "datasets_full",
                                                   "dataset_cards_merged.parquet"))
    p.add_argument("--out", default=os.path.join(d, "graphs", "hgraph_rf"))
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    rep = build(args.ladder, args.feats, args.rf, args.cards, args.out,
                batch_size=args.batch_size, device=args.device)
    print(json.dumps({k: v for k, v in rep.items() if k != "lineage"},
                     indent=2, ensure_ascii=False))
    bad = [k for k, v in rep["gates"].items() if not v]
    if bad:
        print("[FAIL] " + ", ".join(bad))
        return 1
    print("[ok] graph in %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
