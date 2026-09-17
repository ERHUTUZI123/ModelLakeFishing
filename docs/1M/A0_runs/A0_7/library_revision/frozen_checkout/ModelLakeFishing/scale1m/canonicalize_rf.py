"""
canonicalize_rf.py -- F2: the frozen snapshot -> node attributes + supervision.

Runbook: docs/1M/1Mplan.md 5 (F2). Record: docs/1M/F2.md.

TWO OUTPUTS, ONE PASS
    canon/part-*.parquet     per model: size_b, family, lineage_base, relation,
                             layer -- the same four quantities every rung has
                             built, minus the CORE-namespace step (D-56 removed
                             the namespace to target).
    canon/supervision.parquet
                             per (model, dataset-task): the normalised edge
                             weight, its primary metric, and whether the node
                             is gold-eligible.

WHY THE FAMILY RULE IS SHORTER HERE
    100K resolved `family` against the families CORE's embedding table already
    used, so that a llama in HALO would share a row with a llama in CORE. With
    CORE gone there is no prior namespace to land in: the vocabulary is built
    from this lake alone, so the rule collapses to `config.model_type` when
    present, else the lowercased name rule. `family_source` still records which
    fired.

SUPERVISION, IN THE ORDER THE RULES APPLY (1Mplan 3.2-3.4)
    1. parse (model, dataset, task, metric, value) out of model-index
    2. drop non-finite values
    3. direction from metric_semantics: higher / lower / reward / unknown
    4. dedupe: median over (model, dataset, task, metric)
    5. min-max inside (dataset, task, full_metric_name); flip if `lower`
    6. primary metric per node = the direction-known metric with most rows
    7. edge weight = the primary metric's oriented value
    8. gold eligibility = node has a direction-known primary metric and is not
       a reinforcement-learning node
    9. cap edges per node (D-59), stratified by weight

Run (from ModelLakeFishing/):
    python -m scale1m.canonicalize_rf --candidates <dir> --out <dir>
"""
import argparse
import collections
import hashlib
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "stage1BuildTransferGraph"))

from scale1m.hf_canonicalize import (family_of, layer_of, lineage_base_of,
                                     normalize, size_b_of)
from scale1m.hf_crawl import utcnow, write_json_atomic
from scale1m.metric_semantics import classify, in_gold, orient
from scale1m.verify_raw import iter_records

RULES_VERSION = "rf-gold-1.1"

# Names that are not a dataset. Every author who writes `unknown` writes it
# about a different corpus, so pooling them into one node would ask gold to
# rank 896 models evaluated on 896 unrelated things. The edges stay in the
# graph (they still say "this model was evaluated on something"); only the
# query set drops them, the same treatment direction-unverified metrics get.
PLACEHOLDER_DATASETS = frozenset({
    "unknown", "custom", "none", "null", "n/a", "na", "private", "internal",
    "test", "dataset", "my_dataset", "mydataset", "local", "own", "self",
    "in-house", "in_house", "proprietary", "confidential",
})
EDGE_CAP_PER_NODE = 200        # D-59
QUERY_MIN_MODELS = 3           # 1Mplan 3.4; the deeper strata are reported too
DEPTHS = (3, 5, 10, 20)
NODE_SEP = "\t"


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def is_rl_task(task: str) -> bool:
    t = str(task or "").strip().lower()
    return t.startswith("reinforc") or t in ("rl", "deep-rl", "deep_rl")


def parse_model_index(rec):
    """-> list of (dataset_norm, task, metric_raw, value). Malformed rows drop."""
    out = []
    mi = (rec.get("cardData") or {}).get("model-index")
    if not mi:
        return out
    for entry in (mi if isinstance(mi, list) else [mi]):
        if not isinstance(entry, dict):
            continue
        ds = entry.get("dataset") or entry.get("dataset_name")
        if not ds:
            continue
        dsn = normalize(ds)
        task = str(entry.get("task") or "").strip()
        for m in (entry.get("metrics") or []):
            if not isinstance(m, dict):
                continue
            t, v = m.get("type"), m.get("value")
            if t is None or v is None:
                continue
            out.append((dsn, task, t, v))
    return out


def to_float(v):
    """Values arrive as numbers, numeric strings, or lists. Anything else drops."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
    elif isinstance(v, str):
        try:
            f = float(v.strip().rstrip("%"))
        except ValueError:
            return None
    elif isinstance(v, (list, tuple)) and len(v) == 1:
        return to_float(v[0])
    else:
        return None
    return f if math.isfinite(f) else None


def pass_one(cdir, out_dir, shard_rows):
    """Stream the snapshot: write canon parts, collect supervision rows."""
    prov = json.load(open(os.path.join(cdir, "PROVENANCE.json"), encoding="utf-8"))
    os.makedirs(os.path.join(out_dir, "canon"), exist_ok=True)

    sup = {"model": [], "dataset": [], "task": [], "metric": [],
           "direction": [], "value": []}
    dropped = collections.Counter()
    stats = collections.Counter()
    fam_src = collections.Counter()
    lin_src = collections.Counter()
    layers = collections.Counter()

    buf, cur_file, n_parts, n_rows = [], None, 0, 0
    for fname, rec in iter_records(cdir, prov["shards"]):
        if cur_file is None:
            cur_file = fname
        if fname != cur_file:
            n_parts += _flush(buf, out_dir, n_parts)
            buf, cur_file = [], fname
        mid = normalize(rec.get("id"))
        size_b, size_src = size_b_of(rec)
        base, relation, lsrc = lineage_base_of(rec)
        fam, fsrc = family_of(rec, used=set())     # no CORE namespace (D-56)
        layer = layer_of(rec, base)
        buf.append((mid, size_b, fam, fsrc, base, relation, lsrc, layer))
        n_rows += 1
        fam_src[fsrc] += 1
        lin_src[lsrc] += 1
        layers[layer] += 1
        stats["size_known"] += int(size_b is not None)

        for dsn, task, metric_raw, raw_v in parse_model_index(rec):
            stats["raw_metric_rows"] += 1
            v = to_float(raw_v)
            if v is None:
                dropped["non_finite_or_unparsable"] += 1
                continue
            norm, _b, direction = classify(metric_raw)
            sup["model"].append(mid)
            sup["dataset"].append(dsn)
            sup["task"].append(task)
            sup["metric"].append(norm)
            sup["direction"].append(direction)
            sup["value"].append(v)
    n_parts += _flush(buf, out_dir, n_parts)

    df = pd.DataFrame(sup)
    for c in ("dataset", "task", "metric", "direction"):
        df[c] = df[c].astype("category")
    return df, {"n_models": n_rows, "canon_parts": n_parts,
                "family_source": dict(fam_src), "lineage_source": dict(lin_src),
                "layers": dict(layers), "dropped": dict(dropped),
                "stats": dict(stats), "crawl_provenance": {
                    k: prov.get(k) for k in ("total_records", "snapshot_window_utc",
                                             "snapshot_date_utc")}}


def _flush(buf, out_dir, idx):
    if not buf:
        return 0
    cols = ["model", "size_b", "family", "family_source", "lineage_base",
            "lineage_relation", "lineage_source", "layer"]
    pd.DataFrame(buf, columns=cols).to_parquet(
        os.path.join(out_dir, "canon", "part-%05d.parquet" % idx), index=False)
    return 1


def build_supervision(df, cap=EDGE_CAP_PER_NODE):
    """Steps 4-9 of the module docstring. Returns (edges, node table, report)."""
    rep = {}
    rep["raw_rows"] = int(len(df))

    # 4. dedupe: median over (model, dataset, task, metric)
    g = df.groupby(["model", "dataset", "task", "metric", "direction"],
                   observed=True, sort=False)
    ded = g["value"].agg(["median", "size"]).reset_index()
    ded = ded.rename(columns={"median": "value", "size": "n_records"})
    rep["after_dedupe_rows"] = int(len(ded))
    rep["duplicate_rows_collapsed"] = int(rep["raw_rows"] - rep["after_dedupe_rows"])

    # 5. min-max inside (dataset, task, metric), then orient
    key = ["dataset", "task", "metric"]
    grp = ded.groupby(key, observed=True)["value"]
    lo, hi = grp.transform("min"), grp.transform("max")
    span = (hi - lo).replace(0.0, np.nan)
    ded["v_norm"] = ((ded["value"] - lo) / span).fillna(0.5)   # constant group -> 0.5
    ded["v_oriented"] = [orient(v, d) for v, d in zip(ded["v_norm"], ded["direction"])]
    rep["constant_groups_set_to_0.5"] = int((span.isna()).sum())

    # 6. primary metric per node. Direction-known metrics win; a node whose
    #    only metrics are `reward` or `unknown` still gets edges -- D-61 keeps
    #    them -- but is marked gold-ineligible, because a weight whose
    #    direction is unverified must not decide which model is best.
    ded["node"] = ded["dataset"].astype(str) + NODE_SEP + ded["task"].astype(str)
    tier = {"higher": 0, "lower": 0, "reward": 1, "unknown": 2}
    counts = ded.groupby(["node", "metric", "direction"], observed=True) \
                .size().reset_index(name="n")
    counts["tier"] = counts["direction"].map(tier).astype(int)
    counts = counts.sort_values(["node", "tier", "n", "metric"],
                                ascending=[True, True, False, True])
    first = counts.drop_duplicates("node").set_index("node")
    primary, primary_dir = first["metric"], first["direction"]
    rep["nodes_total"] = int(ded["node"].nunique())
    rep["nodes_by_primary_direction"] = {
        str(k): int(v) for k, v in first["direction"].value_counts().items()}
    rep["nodes_with_direction_known_metric"] = int(
        first["direction"].isin(["higher", "lower"]).sum())

    # 7. edge weight = the primary metric's oriented value
    ded["is_primary"] = [m == primary.get(n) for n, m in zip(ded["node"], ded["metric"])]
    edges = ded[ded["is_primary"]].groupby(["node", "model"], observed=True).agg(
        weight=("v_oriented", "median"),
        metric=("metric", "first"),
        direction=("direction", "first"),
        n_records=("n_records", "sum")).reset_index()
    edges[["dataset", "task"]] = edges["node"].str.split(NODE_SEP, n=1, expand=True)
    rep["edges_before_cap"] = int(len(edges))

    # 8. eligibility
    node_models = edges.groupby("node", observed=True)["model"].nunique()
    rl = {n: is_rl_task(n.split(NODE_SEP, 1)[1]) for n in node_models.index}
    nodes = pd.DataFrame({
        "node": node_models.index,
        "n_models": node_models.values,
        "is_rl": [rl[n] for n in node_models.index]})
    nodes[["dataset", "task"]] = nodes["node"].str.split(NODE_SEP, n=1, expand=True)
    nodes["primary_metric"] = nodes["node"].map(primary)
    nodes["primary_direction"] = nodes["node"].map(primary_dir)
    nodes["direction_known"] = nodes["primary_direction"].isin(["higher", "lower"])
    nodes["is_placeholder"] = nodes["dataset"].isin(PLACEHOLDER_DATASETS)
    nodes["gold_eligible"] = (~nodes["is_rl"]) & (~nodes["is_placeholder"]) \
        & nodes["direction_known"] & (nodes["n_models"] >= QUERY_MIN_MODELS)

    # 9. cap, stratified by weight so the value distribution keeps its shape
    edges = edges.merge(nodes[["node", "n_models"]], on="node", how="left")
    keep = []
    rng = np.random.default_rng(0)
    for node, sub in edges.groupby("node", observed=True, sort=False):
        if len(sub) <= cap:
            keep.append(sub)
            continue
        order = sub.sort_values("weight", kind="mergesort")
        bins = np.array_split(np.arange(len(order)), cap)
        pick = [b[rng.integers(len(b))] for b in bins if len(b)]
        keep.append(order.iloc[pick])
    capped = pd.concat(keep, ignore_index=True)
    rep["edges_after_cap"] = int(len(capped))
    rep["nodes_capped"] = int((node_models > cap).sum())
    rep["edge_cap_per_node"] = cap
    return capped, nodes, ded, rep


def distribution_report(edges, nodes, ded, before_edges):
    """1Mplan 3.3.1 -- by task, by dataset, by metric, before and after the cap."""
    def conc(counts):
        tot = counts.sum()
        if tot == 0:
            return {"hhi_effective": 0.0, "top1_pct": 0.0, "top10_pct": 0.0}
        sh = counts / tot
        return {"hhi_effective": round(float(1.0 / (sh ** 2).sum()), 1),
                "top1_pct": round(float(100 * sh.max()), 2),
                "top10_pct": round(float(100 * sh.nlargest(10).sum()), 2)}

    by_task_before = before_edges.groupby("task", observed=True).size()
    by_task_after = edges.groupby("task", observed=True).size()
    by_ds_before = before_edges.groupby("dataset", observed=True).size()
    by_ds_after = edges.groupby("dataset", observed=True).size()
    q = nodes[nodes.gold_eligible]

    top_ds = by_ds_after.nlargest(20)
    capped_nodes = set(nodes.loc[nodes.n_models > EDGE_CAP_PER_NODE, "dataset"])
    return {
        "by_task": {
            "before_cap": {str(k): int(v) for k, v in by_task_before.nlargest(15).items()},
            "after_cap": {str(k): int(v) for k, v in by_task_after.nlargest(15).items()},
            "reinforcement_learning_pct_before": round(
                100.0 * before_edges[before_edges.task.map(is_rl_task)].shape[0]
                / max(len(before_edges), 1), 2),
            "reinforcement_learning_pct_after": round(
                100.0 * edges[edges.task.map(is_rl_task)].shape[0] / max(len(edges), 1), 2),
            "queries_by_task": {str(k): int(v) for k, v in
                                q.groupby("task", observed=True).size().nlargest(15).items()},
        },
        "by_dataset": {
            "top20_after_cap": [{"dataset": str(k), "edges": int(v),
                                 "was_capped": str(k) in capped_nodes}
                                for k, v in top_ds.items()],
            "concentration_before_cap": conc(by_ds_before),
            "concentration_after_cap": conc(by_ds_after),
        },
        "by_metric": {
            "edges_by_primary_metric": {str(k): int(v) for k, v in
                                        edges.groupby("metric", observed=True)
                                        .size().nlargest(20).items()},
            "rows_by_direction_class": {str(k): int(v) for k, v in
                                        ded.groupby("direction", observed=True)
                                        .size().items()},
        },
        "query_depth": {
            "gold_eligible_nodes": int(q.shape[0]),
            **{"nodes_ge_%d" % d: int((q.n_models >= d).sum()) for d in DEPTHS},
            "nodes_with_edges": int(len(nodes)),
            "excluded_direction_unverified": int((~nodes.direction_known).sum()),
            "excluded_reinforcement_learning": int(nodes.is_rl.sum()),
            "excluded_placeholder_dataset_name": int(nodes.is_placeholder.sum()),
            "excluded_fewer_than_%d_models" % QUERY_MIN_MODELS:
                int((nodes.n_models < QUERY_MIN_MODELS).sum()),
            "eligible_after_all_three": int(nodes.gold_eligible.sum()),
        },
    }


def write_rules(out_dir, cap, depth_used):
    """Constraint 4: freeze every knob that can move gold@10."""
    rules = {
        "version": RULES_VERSION,
        "written_at": utcnow(),
        "metric_semantics": {
            "module": "scale1m/metric_semantics.py",
            "classes": ["higher", "lower", "reward", "unknown"],
            "unknown_and_reward_excluded_from_gold": True,
            "name_normalisation": "lowercase; @ -> _at_; [\\s\\-./]+ -> _; collapse _",
            "no_semantic_merging": "ndcg_at_1 and ndcg_at_10 stay distinct",
        },
        "grouping_key": ["dataset", "task", "full_metric_name"],
        "dedupe": {"key": ["model", "dataset", "task", "metric"], "statistic": "median"},
        "normalisation": {"method": "min-max inside the grouping key",
                          "constant_group_value": 0.5,
                          "orientation": "lower-is-better flipped as 1 - v"},
        "primary_metric": "the direction-known metric with the most rows in the node",
        "edge_cap_per_node": cap,
        "cap_sampling": "stratified by weight, numpy default_rng(0)",
        "query_eligibility": {"min_models": QUERY_MIN_MODELS,
                              "exclude_reinforcement_learning": True,
                              "require_direction_known_primary_metric": True,
                              "exclude_placeholder_dataset_names":
                                  sorted(PLACEHOLDER_DATASETS)},
        "value_parsing": {"accepted": "numbers and numeric strings, % stripped",
                          "rejected": "everything else, including the RL "
                                      "leaderboard's `mean +/- std` strings; "
                                      "no parser is added for that format"},
        "main_table_depth": depth_used,
        "reported_depths": list(DEPTHS),
        "split": {"group_by": "dataset owner and base name", "seeds": [0, 1, 2]},
    }
    path = os.path.join(out_dir, "rf_gold_rules.json")
    write_json_atomic(path, rules)
    return path, sha256_of(path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="F2: canonicalize the RF snapshot")
    p.add_argument("--candidates", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cap", type=int, default=EDGE_CAP_PER_NODE)
    p.add_argument("--main-depth", type=int, default=QUERY_MIN_MODELS)
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    print("[pass 1] streaming the snapshot ...", flush=True)
    df, canon_rep = pass_one(args.candidates, args.out, None)
    print("    %d models, %d canon parts, %d metric rows"
          % (canon_rep["n_models"], canon_rep["canon_parts"], len(df)), flush=True)

    print("[pass 2] supervision ...", flush=True)
    edges, nodes, ded, sup_rep = build_supervision(df, cap=args.cap)
    before = ded[ded["is_primary"]].groupby(["node", "model"], observed=True).size().reset_index()
    before[["dataset", "task"]] = before["node"].str.split(NODE_SEP, n=1, expand=True)
    dist = distribution_report(edges, nodes, ded, before)

    edges.to_parquet(os.path.join(args.out, "canon", "supervision.parquet"), index=False)
    # The uncapped edges are what phase 2 merges: the cap has to be applied once,
    # after all sources are in, or a node capped here would be capped again.
    ded[ded["is_primary"]].groupby(["node", "model"], observed=True).agg(
        weight=("v_oriented", "median"), metric=("metric", "first"),
        direction=("direction", "first")).reset_index().assign(
        dataset=lambda x: x["node"].str.split(NODE_SEP, n=1, expand=True)[0],
        task=lambda x: x["node"].str.split(NODE_SEP, n=1, expand=True)[1],
    ).to_parquet(os.path.join(args.out, "canon", "supervision_uncapped.parquet"),
                 index=False)
    nodes.to_parquet(os.path.join(args.out, "canon", "dataset_nodes.parquet"), index=False)
    rules_path, rules_sha = write_rules(args.out, args.cap, args.main_depth)

    report = {"artifact": "F2 canonicalization (RF)", "written_at": utcnow(),
              "rules": {"path": os.path.basename(rules_path), "sha256": rules_sha,
                        "version": RULES_VERSION},
              "canon": canon_rep, "supervision": sup_rep, "distribution": dist}
    write_json_atomic(os.path.join(args.out, "F2_REPORT.json"), report)
    print(json.dumps({"supervision": sup_rep,
                      "query_depth": dist["query_depth"],
                      "rules_sha256": rules_sha}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
