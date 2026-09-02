"""
merge_supervision.py -- F2 phase 2: six supervision sources into one edge set.

Runbook: docs/1M/1Mplan.md 3.7 (D-63 .. D-66). Record: docs/1M/F2.md.

WHY
    The snapshot alone yields 74,346 edges over 2,041 usable query nodes. Every
    earlier rung of this project measured (model, dataset, value) triples for
    real, and those measurements do not stop being true because the corpus they
    came from is no longer the lake. They are merged in, one `source` label per
    edge, so any result can be split back apart.

WHAT STAYS SEPARATE
    The candidate pool is still the snapshot. Models that appear only in
    historical supervision enter the lake as candidates carrying
    `in_snapshot=false`, and the A axis must be reported both over all
    candidates and over surviving ones only.

THE SIX SOURCES, IN PRIORITY ORDER (D-64)
    modellens_v2 > d0_v1_5 > a_ctrl_2000m > hf_effective > diverse_zoo
    > hf_model_index
    Curated corpora are measurements somebody ran; model-index is self-report.
    Two graphs are deliberately NOT sources: hgraph_zoo (99.5% inside
    diverse_zoo) and hgraph_hf1000d (all 7,056 pairs inside a_ctrl_2000m, plus
    5,149 duplicate edges of its own).

VALUES (D-65)
    Historical graphs kept only a normalised weight, and not on one scale
    ([0,1] for most, [0.60,1.00] for the zoo line). Each (node, source) group
    is min-maxed again, which preserves the within-group ordering and puts
    every source on one scale. Direction is not re-derived: these weights were
    already oriented at build time, so they carry direction class `curated`.

Run (from ModelLakeFishing/):
    python -m scale1m.merge_supervision --rf <rf dir> --out <rf dir>
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

from scale1m.canonicalize_rf import (EDGE_CAP_PER_NODE, NODE_SEP,
                                     PLACEHOLDER_DATASETS, QUERY_MIN_MODELS,
                                     DEPTHS, is_rl_task, sha256_of)
from scale1m.hf_crawl import data_root, utcnow, write_json_atomic
from scale1m.verify_raw import iter_records, normalize

RULES_VERSION = "rf-gold-2.0"
G = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "stage1BuildTransferGraph")

# key -> (file, priority). Lower priority number wins a cross-source conflict.
SOURCES = [
    ("modellens_v2", "hgraph_ml_v2.pt", 0),
    ("d0_v1_5", "hgraph_d05_nocontent.pt", 1),
    ("a_ctrl_2000m", "hgraph_A_ctrl_2000m_xm0_xd0.pt", 2),
    ("hf_effective", "hgraph_hf_effective_2000m_v2_xm0_xd0.pt", 3),
    ("diverse_zoo", "hgraph_diverse_xd0.pt", 4),
]
RF_SOURCE, RF_PRIORITY = "hf_model_index", 5
ML_SEP = "␟"          # ModelLens packs the node as `dataset<US>task`


def split_node(name, task_by_id, idx):
    """-> (dataset, task). Sources spell the node differently."""
    s = str(name).strip()
    if ML_SEP in s:
        ds, task = s.split(ML_SEP, 1)
        return normalize(ds), str(task).strip().lower()
    return normalize(s), (task_by_id.get(idx) or "")


def load_source(key, fname):
    ck = torch.load(os.path.join(G, fname), map_location="cpu", weights_only=False)
    d = ck["data"]
    umi = ck["unique_model_id"].sort_values("mappedID")
    udi = ck["unique_dataset_id"].sort_values("mappedID")
    mcol = [c for c in umi.columns if c != "mappedID"][0]
    dcol = [c for c in udi.columns if c not in ("mappedID", "root", "root_b")][0]
    models = [normalize(x) for x in umi[mcol]]

    vocab = (ck.get("xd0_meta") or {}).get("task_type_vocab")
    task_by_id = {}
    if vocab and "task_type_id" in d["dataset"]:
        inv = {v: k for k, v in vocab.items()}
        task_by_id = {i: str(inv.get(int(t), "")).strip().lower()
                      for i, t in enumerate(d["dataset"].task_type_id.tolist())}
    nodes = [split_node(n, task_by_id, i) for i, n in enumerate(udi[dcol])]

    et = [e for e in d.edge_types if e[1] == "trained_on"][0]
    ei, attr = d[et].edge_index, d[et].edge_attr
    df = pd.DataFrame({
        "model": [models[i] for i in ei[0].tolist()],
        "dataset": [nodes[i][0] for i in ei[1].tolist()],
        "task": [nodes[i][1] for i in ei[1].tolist()],
        "value": attr.tolist(),
    })
    df["source"] = key
    return df, {"models": len(models), "dataset_nodes": len(nodes),
                "edges": int(ei.shape[1]),
                "attr_min": round(float(attr.min()), 4),
                "attr_max": round(float(attr.max()), 4),
                "task_recovered": bool(task_by_id)}


def load_rf(rf_dir):
    """The snapshot-side edges F2 phase 1 produced, before its own cap."""
    sup = pd.read_parquet(os.path.join(rf_dir, "canon", "supervision_uncapped.parquet")) \
        if os.path.exists(os.path.join(rf_dir, "canon", "supervision_uncapped.parquet")) \
        else pd.read_parquet(os.path.join(rf_dir, "canon", "supervision.parquet"))
    out = pd.DataFrame({
        "model": sup["model"], "dataset": sup["dataset"], "task": sup["task"],
        "value": sup["weight"], "source": RF_SOURCE,
        "metric": sup["metric"], "direction": sup["direction"]})
    return out


def merge(rf_dir, cap=EDGE_CAP_PER_NODE):
    rep = {"sources": {}}
    frames = []
    for key, fname, _prio in SOURCES:
        df, meta = load_source(key, fname)
        rep["sources"][key] = meta
        frames.append(df)
    hist = pd.concat(frames, ignore_index=True)
    hist["metric"] = "unknown_curated"
    hist["direction"] = "curated"

    rf = load_rf(rf_dir)
    rep["sources"][RF_SOURCE] = {"edges": int(len(rf))}
    allrows = pd.concat([hist, rf], ignore_index=True)
    allrows["node"] = allrows["dataset"] + NODE_SEP + allrows["task"]
    rep["rows_in"] = int(len(allrows))

    # 1. intra-source duplicates -> median
    g = allrows.groupby(["source", "node", "model"], observed=True, sort=False)
    ded = g.agg(value=("value", "median"), n_records=("value", "size"),
                metric=("metric", "first"), direction=("direction", "first")).reset_index()
    rep["intra_source_duplicates_collapsed"] = int(len(allrows) - len(ded))

    # 2. re-normalise inside (node, source) so every source shares one scale
    grp = ded.groupby(["node", "source"], observed=True)["value"]
    lo, hi = grp.transform("min"), grp.transform("max")
    span = (hi - lo).replace(0.0, np.nan)
    ded["weight"] = ((ded["value"] - lo) / span).fillna(0.5)
    rep["constant_groups_set_to_0.5"] = int(span.isna().sum())

    # 3. cross-source conflicts -> highest priority wins, losers recorded
    prio = {k: p for k, _f, p in SOURCES}
    prio[RF_SOURCE] = RF_PRIORITY
    ded["priority"] = ded["source"].map(prio).astype(int)
    ded = ded.sort_values(["node", "model", "priority"], kind="mergesort")
    dup = ded.duplicated(["node", "model"], keep="first")
    conflicts = ded[dup].copy()
    edges = ded[~dup].copy()
    rep["cross_source_conflicts"] = int(len(conflicts))
    rep["edges_after_conflict_resolution"] = int(len(edges))

    # 4. node table before the cap
    edges[["dataset", "task"]] = edges["node"].str.split(NODE_SEP, n=1, expand=True)
    per_node = edges.groupby("node", observed=True)
    nodes = pd.DataFrame({
        "node": per_node.size().index,
        "n_models": per_node["model"].nunique().values,
        "n_sources": per_node["source"].nunique().values,
    })
    nodes[["dataset", "task"]] = nodes["node"].str.split(NODE_SEP, n=1, expand=True)
    # the source that decides gold at this node: the highest-priority one present
    gold_src = edges.sort_values("priority").drop_duplicates("node") \
                    .set_index("node")["source"]
    nodes["gold_source"] = nodes["node"].map(gold_src)
    dirs = edges.sort_values("priority").drop_duplicates("node") \
                .set_index("node")["direction"]
    nodes["primary_direction"] = nodes["node"].map(dirs)
    nodes["direction_known"] = nodes["primary_direction"].isin(
        ["higher", "lower", "curated"])
    nodes["is_rl"] = nodes["task"].map(is_rl_task)
    nodes["is_placeholder"] = nodes["dataset"].isin(PLACEHOLDER_DATASETS) \
        | nodes["dataset"].isin({"##", "#", "-", ""})
    nodes["gold_eligible"] = (~nodes["is_rl"]) & (~nodes["is_placeholder"]) \
        & nodes["direction_known"] & (nodes["n_models"] >= QUERY_MIN_MODELS)
    rep["nodes_total"] = int(len(nodes))
    rep["nodes_multi_source"] = int((nodes.n_sources > 1).sum())

    # 5. cap per node, stratified by weight (deterministic)
    keep, rng = [], np.random.default_rng(0)
    for node, sub in edges.groupby("node", observed=True, sort=False):
        if len(sub) <= cap:
            keep.append(sub)
            continue
        order = sub.sort_values("weight", kind="mergesort")
        bins = np.array_split(np.arange(len(order)), cap)
        keep.append(order.iloc[[b[rng.integers(len(b))] for b in bins if len(b)]])
    capped = pd.concat(keep, ignore_index=True)
    rep["edges_before_cap"] = int(len(edges))
    rep["edges_after_cap"] = int(len(capped))
    rep["nodes_capped"] = int((per_node.size() > cap).sum())
    rep["edges_by_source_after_cap"] = {str(k): int(v) for k, v in
                                        capped.groupby("source", observed=True)
                                        .size().items()}
    q = nodes[nodes.gold_eligible]
    rep["query_depth"] = {"gold_eligible_nodes": int(len(q)),
                          **{"nodes_ge_%d" % d: int((q.n_models >= d).sum())
                             for d in DEPTHS},
                          "by_gold_source": {str(k): int(v) for k, v in
                                             q.groupby("gold_source", observed=True)
                                             .size().items()}}
    rep["excluded"] = {
        "direction_unverified": int((~nodes.direction_known).sum()),
        "reinforcement_learning": int(nodes.is_rl.sum()),
        "placeholder_dataset_name": int(nodes.is_placeholder.sum()),
        "fewer_than_%d_models" % QUERY_MIN_MODELS:
            int((nodes.n_models < QUERY_MIN_MODELS).sum())}
    return capped, nodes, conflicts, rep


def model_table(edges, crawl_dir):
    """Every candidate, with whether the 2026-08-18 snapshot still has it."""
    prov = json.load(open(os.path.join(crawl_dir, "PROVENANCE.json"), encoding="utf-8"))
    crawl = {normalize(r.get("id")) for _, r in iter_records(crawl_dir, prov["shards"])}
    sup_models = set(edges["model"])
    extra = sorted(sup_models - crawl)
    rows = pd.DataFrame({"model": extra, "in_snapshot": False})
    return crawl, rows


def write_rules(out_dir, cap):
    rules = {
        "version": RULES_VERSION,
        "written_at": utcnow(),
        "supersedes": "rf-gold-1.1",
        "candidate_pool": "the 2026-08-18 HF snapshot; models that appear only in "
                          "historical supervision are added as candidates with "
                          "in_snapshot=false",
        "sources_in_priority_order": [k for k, _f, _p in SOURCES] + [RF_SOURCE],
        "sources_excluded_as_redundant": {
            "hgraph_zoo_xm0.pt": "99.52% of its pairs are inside diverse_zoo",
            "hgraph_hf1000d_2000m_xm0_xd0.pt":
                "all 7,056 distinct pairs are inside a_ctrl_2000m; it also "
                "carries 5,149 duplicate edges of its own"},
        "intra_source_dedupe": {"key": ["source", "node", "model"],
                                "statistic": "median"},
        "renormalisation": "min-max inside (node, source); constant group -> 0.5",
        "direction": {"curated": "historical weights are already oriented at "
                                 "build time; they may define gold",
                      "others": "see scale1m/metric_semantics.py"},
        "conflict_resolution": "highest-priority source wins; losers are written "
                               "to supervision_conflicts.parquet",
        "gold_source": "the highest-priority source present at the node",
        "edge_cap_per_node": cap,
        "cap_sampling": "stratified by weight, numpy default_rng(0)",
        "query_eligibility": {"min_models": QUERY_MIN_MODELS,
                              "exclude_reinforcement_learning": True,
                              "exclude_placeholder_dataset_names":
                                  sorted(PLACEHOLDER_DATASETS | {"##", "#", "-", ""}),
                              "require_direction_known_or_curated": True},
        "main_table_depth": QUERY_MIN_MODELS,
        "reported_depths": list(DEPTHS),
        "reporting": "every A-axis number must be reported over all candidates "
                     "AND over in_snapshot=true candidates only, and must be "
                     "splittable by source",
        "split": {"group_by": "dataset owner and base name", "seeds": [0, 1, 2]},
    }
    path = os.path.join(out_dir, "rf_gold_rules.json")
    write_json_atomic(path, rules)
    return path, sha256_of(path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="F2 phase 2: merge supervision sources")
    p.add_argument("--rf", required=True, help="the F2 phase-1 output dir")
    p.add_argument("--candidates", default=None)
    p.add_argument("--cap", type=int, default=EDGE_CAP_PER_NODE)
    args = p.parse_args(argv)
    crawl_dir = args.candidates or os.path.join(data_root(), "data1m", "candidates_full")

    print("[merge] loading six sources ...", flush=True)
    edges, nodes, conflicts, rep = merge(args.rf, cap=args.cap)

    print("[merge] model table ...", flush=True)
    crawl, extra = model_table(edges, crawl_dir)
    rep["candidates"] = {"snapshot": len(crawl), "historical_only": int(len(extra)),
                         "total": len(crawl) + int(len(extra))}

    out = os.path.join(args.rf, "canon")
    edges.to_parquet(os.path.join(out, "supervision_merged.parquet"), index=False)
    nodes.to_parquet(os.path.join(out, "dataset_nodes_merged.parquet"), index=False)
    conflicts.to_parquet(os.path.join(out, "supervision_conflicts.parquet"), index=False)
    extra.to_parquet(os.path.join(out, "models_out_of_snapshot.parquet"), index=False)
    rules_path, sha = write_rules(args.rf, args.cap)
    rep["rules"] = {"path": os.path.basename(rules_path), "sha256": sha,
                    "version": RULES_VERSION}
    write_json_atomic(os.path.join(args.rf, "F2_MERGE_REPORT.json"),
                      dict(rep, written_at=utcnow()))
    print(json.dumps({k: v for k, v in rep.items() if k != "sources"},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
