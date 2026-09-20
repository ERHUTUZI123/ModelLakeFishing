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
from scale1m.dataset_descriptor import dataset_descriptor
from scale1m.embed_lake import DEFAULT_BATCH, ENCODER, ENCODER_REVISION, minilm
from scale1m.hf_crawl import data_root, utcnow, write_json_atomic
from scale1m.verify_raw import normalize

NAME_DIM_D, CARD_DIM, STAT_DIM = 64, 384, 10
XD_DIM = NAME_DIM_D + CARD_DIM + STAT_DIM
NAME_SEED_D = 43
SIM_K = 20
A0_ZERO_COLUMNS = (448, 449, 450, 451, 452, 453, 455)


def mask_performance_features(x_dataset):
    x = np.asarray(x_dataset)
    if x.ndim != 2 or x.shape[1] != XD_DIM or x.dtype != np.float32:
        raise ValueError("A0 requires a two-dimensional float32 dataset array with 458 columns")
    if not np.isfinite(x).all():
        raise ValueError("Non-finite dataset features")
    result = x.copy()
    result[:, A0_ZERO_COLUMNS] = 0.0
    return result


def _text(v):
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def dataset_texts(nodes, cards):
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
                rec = {"description": _text(row.get("description")),
                       "tags": json.loads(_text(row.get("tags")) or "[]"),
                       "cardData": {"task_categories":
                                    json.loads(_text(row.get("task_categories"))
                                               or "[]")}}
        node_str = r.dataset + (" " + r.task if r.task else "")
        out.append(dataset_descriptor(rec, node_str))
    return out, with_card


def dataset_stats(nodes, edges):
    roots = nodes["dataset"].map(lambda d: str(d).split("/")[0])
    root_sizes = roots.value_counts().to_dict()
    x = np.zeros((len(nodes), STAT_DIM), dtype=np.float32)
    for i in range(len(nodes)):
        x[i, 6] = np.log1p(root_sizes.get(roots.iloc[i], 0))
    return x, roots.tolist()


def build_xd(nodes, cards, edges, out_dir, batch_size, device):
    from dataset_embed.xm0_builder import build_name_embeddings
    texts, with_card = dataset_texts(nodes, cards)
    e_name = build_name_embeddings([r.node.replace(NODE_SEP, " ") for r in
                                    nodes.itertuples()],
                                   token_dim=NAME_DIM_D, seed=NAME_SEED_D)
    e_card, dev = minilm(texts, batch_size=batch_size, device=device, tag="rf.card")
    e_stats, roots = dataset_stats(nodes, edges)
    xd = np.concatenate([e_name, e_card, e_stats], axis=1).astype(np.float32)
    xd = mask_performance_features(xd)
    assert xd.shape == (len(nodes), XD_DIM), xd.shape
    return xd, e_card, roots, {"with_hf_card": with_card, "device": dev,
                               "char_mean": round(float(np.mean(
                                   [len(t) for t in texts])), 2)}


def lineage_edges(rf_dir, id2idx):
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
            r = _text(r) or "unknown"
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

    xd, e_card, roots, card_meta = build_xd(nodes, cards, edges, out_dir,
                                            batch_size, device)
    print("[xd] %s | cards %d/%d | %.1fs"
          % (xd.shape, card_meta["with_hf_card"], n_d, time.time() - t0), flush=True)

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

    src_x = os.path.join(feats_dir, "x_m.npy")
    dst_x = os.path.join(out_dir, "x_model.npy")
    if not os.path.exists(dst_x) or sha256_of(src_x) != sha256_of(dst_x):
        print("[write] copying x_m.npy (%.3f GB) ..." % (os.path.getsize(src_x) / 1e9), flush=True)
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
                     "encoder_name": ENCODER,
                     "encoder_revision": ENCODER_REVISION,
                     "has_content_view": False,
                     "probe_views": "not used (RF)"},
        "provenance": {"rung": "full", "ladder_dir": os.path.abspath(ladder_dir),
                       "feats_dir": os.path.abspath(feats_dir),
                       "supervision": "rf/canon/supervision_merged.parquet",
                       "x_m_sha256": sha256_of(src_x)},
        "tensors": {}, "files": {},
    }
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
    for f in os.listdir(out_dir):
        if f != "meta.json":
            meta["files"][f] = sha256_of(os.path.join(out_dir, f))
    meta["dataset_feature_policy"] = {"name": "no_performance_derived_inputs",
                                      "zero_columns": list(A0_ZERO_COLUMNS)}
    write_json_atomic(os.path.join(out_dir, "meta.json"), meta)
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
