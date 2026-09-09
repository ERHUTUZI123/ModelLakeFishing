"""
build_graph_rung.py -- T5: assemble the rung's HGraph from the frozen CORE
graph plus the T4 feature matrix.

Runbook: docs/1M/100kplan.md §8.   Record: docs/1M/T5.md.

WHAT THIS FILE ACTUALLY DOES
    Very little arithmetic. The dataset side, the supervision edges and the
    dataset-dataset similarity edges are copied out of hgraph_ml_v2.pt without
    being touched; the model side gets the 100,000-row feature matrix T4 built;
    the only thing computed here is the HALO lineage edges, and that is one
    hashmap join.

    Copying is the point. Plan §1 requires CORE to stay byte-identical so that
    R0/R1/R2 differ in N and nothing else. Recomputing the dataset embeddings
    or the KNN would change CORE without any assertion firing, and the ladder
    would silently stop comparing to P3/P5.

WHY CORE ALREADY HAS 42 LINEAGE EDGES
    hgraph_ml_v2.pt carries 42 is_base_of edges from the ModelLens intake.
    Those stay. The HALO join adds edges whose target is a HALO model; it can
    never produce a CORE->CORE edge, because only HALO rows carry lineage_base
    in the ladder (D-38). So the two sets are disjoint by construction and the
    total is 42 + <join result>. We assert the disjointness rather than trust
    the argument.

Run (from ModelLakeFishing/):
    python -m scale1m.build_graph_rung --rung 100k `
        --core stage1BuildTransferGraph/hgraph_ml_v2.pt `
        --ladder <dir>/ladder/100k_model_ids.csv `
        --feats <dir>/feats/100k --out <dir>/graphs/hgraph_100k.pt
"""

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from scale1m.hf_crawl import utcnow, write_json_atomic
from scale1m.hf_canonicalize import normalize
from scale1m.embed_lake import sha256_of, X_DIM

TRAINED_ON = ("model", "trained_on", "dataset")
REV_TRAINED_ON = ("dataset", "rev_trained_on", "model")
SIMILAR_TO = ("dataset", "similar_to", "dataset")
IS_BASE_OF = ("model", "is_base_of", "model")
REV_IS_BASE_OF = ("model", "rev_is_base_of", "model")

# HF's structured base-model relation. T3 (D-37) pulled it off the
# authoritative `baseModels` field, so it is read rather than guessed from
# names -- this is the quantity CLAUDE.md's discrete ordered r_mm' weights
# (quantized > adapter > finetune > merge) need.
#
# It is stored as a separate `relation_id` column, and edge_attr stays 1.0 for
# every lineage edge. Reason: CORE's 42 lineage edges have attr 1.0 and cannot
# be recomputed (plan §1 rule 1), so writing relation-derived weights onto the
# HALO edges only would make edge weight mean different things in the two
# groups -- a change to the graph's semantics smuggled in at T5, when the whole
# ladder rests on N being the only thing that differs. --relation-weights turns
# the weighting on for an experiment that explicitly wants it.
RELATION_VOCAB = {"unknown": 0, "quantized": 1, "adapter": 2, "finetune": 3,
                  "merge": 4}
RELATION_WEIGHT = {"quantized": 1.0, "adapter": 0.75, "finetune": 0.5,
                   "merge": 0.25, "unknown": 0.5}
LINEAGE_ATTR = 1.0                      # what CORE uses


def load_feats(feats_dir, n, n_core, core_x):
    x = np.load(os.path.join(feats_dir, "x_m.npy"))
    size_id = np.load(os.path.join(feats_dir, "size_bucket_id.npy"))
    fam_id = np.load(os.path.join(feats_dir, "family_id.npy"))
    vocab_df = pd.read_csv(os.path.join(feats_dir, "family_vocab.csv"))

    assert x.shape == (n, X_DIM), "x_m is %s, expected (%d, %d)" % (
        x.shape, n, X_DIM)
    assert size_id.shape == (n,) and fam_id.shape == (n,)
    assert not np.isnan(x).any() and not np.isinf(x).any()
    # T4 checked this too; checking again is cheap and this is the last place
    # the CORE prefix can still be swapped for a re-embedded one.
    assert torch.equal(torch.from_numpy(x[:n_core]), core_x), \
        "feature matrix's CORE prefix is not byte-identical to the frozen graph"

    vocab = dict(zip(vocab_df["family"].astype(str), vocab_df["family_id"].astype(int)))
    assert sorted(vocab.values()) == list(range(len(vocab))), \
        "family_vocab ids are not a contiguous bijection"
    return x, size_id, fam_id, vocab


def lineage_edges(ladder, n_core):
    """HALO's declared base models, resolved against the rung's own id table.

    O(N) hashmap join on the normalized id, not O(N^2) name matching: T3 already
    established (F-T3-4) that the declarations come from HF's structured
    `baseModels`, so the only question here is whether the base is inside this
    rung's 100,000 models.
    """
    id2idx = {}
    for i, m in enumerate(ladder["model"]):
        id2idx[normalize(str(m))] = i

    src, dst, rel = [], [], []
    declared = unresolved = self_loop = 0
    relations = ladder["lineage_relation"] if "lineage_relation" in ladder else None
    for i in range(n_core, len(ladder)):
        base = ladder["lineage_base"].iloc[i]
        if base is None or (isinstance(base, float) and np.isnan(base)):
            continue
        declared += 1
        j = id2idx.get(normalize(str(base)))
        if j is None:
            unresolved += 1
            continue
        if j == i:
            self_loop += 1
            continue
        src.append(j)                       # base --is_base_of--> derivative
        dst.append(i)
        r = relations.iloc[i] if relations is not None else None
        r = "unknown" if r is None or (isinstance(r, float) and np.isnan(r)) \
            else str(r)
        rel.append(r if r in RELATION_VOCAB else "unknown")
    return src, dst, rel, dict(declared=declared, unresolved=unresolved,
                               self_loop=self_loop)


def attach_relations(ladder, canon_path):
    """Carry `lineage_relation` from the canon table onto the ladder.

    T3 pulled the relation off HF's structured `baseModels` (D-37) but the
    ladder's six columns do not carry it, so without this join every lineage
    edge lands in `unknown` and the quantity CLAUDE.md's r_mm' weights need is
    simply absent from the graph -- recoverable only by rebuilding.
    """
    canon = pd.read_parquet(canon_path, columns=["id_norm", "lineage_relation"])
    rel = dict(zip(canon["id_norm"], canon["lineage_relation"]))
    ladder["lineage_relation"] = [
        rel.get(normalize(str(m))) for m in ladder["model"]]
    return ladder


def build(core_path, ladder_path, feats_dir, out_path, rung, canon_path=None,
          relation_weights=False):
    core = torch.load(core_path, weights_only=False)
    cd = core["data"]
    n_core = int(cd["model"].num_nodes)
    n_ds = int(cd["dataset"].num_nodes)

    ladder = pd.read_csv(ladder_path)
    n = len(ladder)
    assert ladder["mappedID"].tolist() == list(range(n)), "ladder mappedID broken"
    umi_core = core["unique_model_id"].sort_values("mappedID").reset_index(drop=True)
    assert ladder.iloc[:n_core]["model"].tolist() == umi_core["model"].tolist(), \
        "ladder CORE prefix != CORE's own mappedID order"
    if canon_path:
        ladder = attach_relations(ladder, canon_path)

    x, size_id, fam_id, vocab = load_feats(feats_dir, n, n_core, cd["model"].x)
    print("[in] CORE %d | rung %d models | %d dataset nodes | vocab %d"
          % (n_core, n, n_ds, len(vocab)), flush=True)

    data = HeteroData()

    # --- model side: T4's matrix, CORE prefix included verbatim -----------
    data["model"].node_id = torch.arange(n)
    data["model"].x = torch.from_numpy(x)
    data["model"].size_bucket_id = torch.from_numpy(size_id).long()
    data["model"].family_id = torch.from_numpy(fam_id).long()

    # --- dataset side: copied, not recomputed -----------------------------
    for k in cd["dataset"].keys():
        data["dataset"][k] = cd["dataset"][k].clone()

    # --- supervision + similarity: copied ---------------------------------
    # CORE occupies mappedID 0..n_core-1 in the rung too, so the stored model
    # indices stay valid without remapping. That is the whole reason the ladder
    # pins CORE to the prefix.
    for et in (TRAINED_ON, REV_TRAINED_ON, SIMILAR_TO):
        data[et].edge_index = cd[et].edge_index.clone()
        data[et].edge_attr = cd[et].edge_attr.clone()

    # --- lineage: CORE's 42 + HALO's join ---------------------------------
    c_li = cd[IS_BASE_OF].edge_index
    c_la = cd[IS_BASE_OF].edge_attr
    h_src, h_dst, h_rel, join_stats = lineage_edges(ladder, n_core)
    h_li = torch.tensor([h_src, h_dst], dtype=torch.long) if h_src else \
        torch.zeros(2, 0, dtype=torch.long)
    h_la = torch.tensor([RELATION_WEIGHT[r] for r in h_rel], dtype=torch.float32) \
        if relation_weights else torch.full((len(h_src),), LINEAGE_ATTR)

    core_pairs = {(int(a), int(b)) for a, b in zip(*c_li.tolist())}
    halo_pairs = {(int(a), int(b)) for a, b in zip(h_src, h_dst)}
    assert not (core_pairs & halo_pairs), \
        "%d lineage edges are in both CORE and the HALO join" % len(
            core_pairs & halo_pairs)
    assert len(halo_pairs) == len(h_src), "duplicate edges inside the HALO join"

    li = torch.cat([c_li, h_li], dim=1)
    la = torch.cat([c_la, h_la])
    data[IS_BASE_OF].edge_index = li
    data[IS_BASE_OF].edge_attr = la
    data[REV_IS_BASE_OF].edge_index = li.flip(0)
    data[REV_IS_BASE_OF].edge_attr = la.clone()

    # relation ids ride along unused; CORE's 42 edges have no declared relation
    rel_id = torch.tensor(
        [RELATION_VOCAB["unknown"]] * int(c_li.shape[1])
        + [RELATION_VOCAB[r] for r in h_rel], dtype=torch.long)
    data[IS_BASE_OF].relation_id = rel_id
    data[REV_IS_BASE_OF].relation_id = rel_id.clone()

    # --- meta: CORE's, with family_vocab grown ----------------------------
    xm0 = dict(core["xm0_meta"])
    old_vocab = xm0["family_vocab"]
    for k, v in old_vocab.items():
        assert vocab.get(k) == v, "family row moved: %s %s -> %s" % (
            k, v, vocab.get(k))
    xm0["family_vocab"] = vocab
    xm0["num_families"] = len(vocab)

    umi = pd.DataFrame({"model": ladder["model"].tolist(), "mappedID": range(n)})

    payload = {
        "data": data,
        "xm0_meta": xm0,
        "xd0_meta": dict(core["xd0_meta"]),
        "unique_model_id": umi,
        "unique_dataset_id": core["unique_dataset_id"].copy(),
        "provenance": {
            "rung": rung, "n_models": n, "n_core": n_core,
            "n_halo": n - n_core, "n_dataset_nodes": n_ds,
            "built_at": utcnow(),
            "core_graph_sha256": sha256_of(core_path),
            "ladder_sha256": sha256_of(ladder_path),
            "x_m_sha256": sha256_of(os.path.join(feats_dir, "x_m.npy")),
            "family_vocab_sha256": sha256_of(
                os.path.join(feats_dir, "family_vocab.csv")),
            "core_provenance": core["provenance"],
        },
    }

    # --- plan §1 rule 1: the six assertions --------------------------------
    assert torch.equal(data["model"].x[:n_core], cd["model"].x)
    assert torch.equal(data["model"].size_bucket_id[:n_core], cd["model"].size_bucket_id)
    assert torch.equal(data["model"].family_id[:n_core], cd["model"].family_id)
    assert umi.iloc[:n_core]["model"].tolist() == umi_core["model"].tolist()
    assert torch.equal(data["dataset"].x, cd["dataset"].x)
    assert torch.equal(data[TRAINED_ON].edge_index, cd[TRAINED_ON].edge_index)
    assert torch.equal(data[TRAINED_ON].edge_attr, cd[TRAINED_ON].edge_attr)
    assert torch.equal(data[SIMILAR_TO].edge_index, cd[SIMILAR_TO].edge_index)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(payload, out_path)

    stats = lineage_stats(data, n_core, join_stats, h_rel)
    write_json_atomic(os.path.join(os.path.dirname(os.path.abspath(out_path)),
                                   "lineage_stats.json"), stats)
    report = {
        "written_at": utcnow(), "rung": rung,
        "graph": os.path.abspath(out_path),
        "graph_sha256": sha256_of(out_path),
        "graph_bytes": os.path.getsize(out_path),
        "models": n, "core": n_core, "halo": n - n_core,
        "dataset_nodes": n_ds,
        "edges": {"%s__%s__%s" % et: int(data[et].edge_index.shape[1])
                  for et in data.edge_types},
        "num_families": len(vocab),
        "relation_weights_applied": bool(relation_weights),
        "provenance": payload["provenance"],
        "lineage": stats,
    }
    write_json_atomic(os.path.join(os.path.dirname(os.path.abspath(out_path)),
                                   "GRAPH_REPORT_%s.json" % rung), report)
    return payload, report


def lineage_stats(data, n_core, join_stats, relations):
    """The numbers plan §3.2 requires reporting whatever they turn out to be."""
    ei = data[IS_BASE_OF].edge_index
    s, d = ei[0].numpy(), ei[1].numpy()
    cc = int(((s < n_core) & (d < n_core)).sum())
    ch = int(((s < n_core) & (d >= n_core)).sum())
    hc = int(((s >= n_core) & (d < n_core)).sum())
    hh = int(((s >= n_core) & (d >= n_core)).sum())

    n = int(data["model"].num_nodes)
    touched = np.unique(np.concatenate([s, d]))
    n_comp, largest = _components(s, d, n)
    rel_counts = pd.Series(relations).value_counts().to_dict() if relations else {}
    n_halo = n - n_core
    return {
        "total_edges": int(ei.shape[1]),
        "core_core": cc, "core_halo": ch, "halo_core": hc, "halo_halo": hh,
        "n_models_with_lineage": int(len(touched)),
        "n_components": n_comp, "largest_component": largest,
        "halo_declared_base": join_stats["declared"],
        "halo_base_unresolved": join_stats["unresolved"],
        "halo_self_loop_dropped": join_stats["self_loop"],
        "halo_lineage_coverage": round(
            float(len(touched[touched >= n_core]) / max(n_halo, 1)), 5),
        "relation_counts": {str(k): int(v) for k, v in rel_counts.items()},
        "relation_vocab": RELATION_VOCAB,
    }


def _components(s, d, n):
    """Connected components over the undirected lineage graph, union-find."""
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(s, d):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    touched = np.unique(np.concatenate([s, d])) if len(s) else np.array([], int)
    if not len(touched):
        return 0, 0
    roots = pd.Series([find(int(t)) for t in touched]).value_counts()
    return int(len(roots)), int(roots.iloc[0])


def main(argv=None):
    p = argparse.ArgumentParser(description="T5: build the rung HGraph")
    p.add_argument("--rung", default="100k")
    p.add_argument("--core", default="stage1BuildTransferGraph/hgraph_ml_v2.pt")
    p.add_argument("--ladder", required=True)
    p.add_argument("--feats", required=True, help="the <feats>/<rung> directory")
    p.add_argument("--canon", default=None,
                   help="hf_canon.parquet; supplies lineage_relation for r_mm'")
    p.add_argument("--out", required=True)
    p.add_argument("--relation-weights", action="store_true",
                   help="write relation-derived lineage edge_attr instead of 1.0")
    args = p.parse_args(argv)

    _, rep = build(args.core, args.ladder, args.feats, args.out, args.rung,
                   canon_path=args.canon, relation_weights=args.relation_weights)
    li = rep["lineage"]
    print("\n[ok] %s (%.1f MB) sha256 %s"
          % (rep["graph"], rep["graph_bytes"] / 2 ** 20, rep["graph_sha256"][:16]))
    print("  models %d (core %d + halo %d) | dataset nodes %d | families %d"
          % (rep["models"], rep["core"], rep["halo"], rep["dataset_nodes"],
             rep["num_families"]))
    for k, v in rep["edges"].items():
        print("  edges %-38s %d" % (k, v))
    print("  lineage total %d = core_core %d + core_halo %d + halo_halo %d"
          % (li["total_edges"], li["core_core"], li["core_halo"], li["halo_halo"]))
    print("  halo declared %d | unresolved %d | coverage %.4f"
          % (li["halo_declared_base"], li["halo_base_unresolved"],
             li["halo_lineage_coverage"]))
    print("  components %d | largest %d"
          % (li["n_components"], li["largest_component"]))
    print("  relations %s" % (li["relation_counts"] or "none (no --canon)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
