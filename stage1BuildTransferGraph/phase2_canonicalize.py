"""
phase2_canonicalize.py -- Effective-Dataset guide, Phase 2 + Phase 3.

Turns raw_performance_observations into ONE canonical observation per
(ranking_group, model) with deterministic conflict resolution, without merging
different ranking problems. Then (Phase 3) it separates the two contracts:

  * canonical_performance_observations.parquet -- the ONLY label/eval source
  * (message edges are derived later in Phase 5 from a filtered VIEW of this)

Ranking-group key stays (dataset_canonical, dataset_config, task,
metric_canonical, split). Aliasing acts ONLY on the dataset-name component and is
reversible; it never collapses config/split/metric/task.

Deterministic conflict rule for a (ranking_group, model) with disagreeing values:
  1. prefer verified=True records;
  2. among the surviving records, use the MEDIAN (declared run aggregation) --
     not max, which would reward models with more reported runs;
  3. all conflicting source rows are retained in performance_conflicts.csv.

Writes into artifacts/effective_dataset_v2/:
  canonical_performance_observations.parquet
  dataset_aliases.csv
  performance_conflicts.csv
  canonicalization_report.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
OUT = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage2TrainGraphSAGE",
                   "artifacts", "effective_dataset_v2")

# Versioned dataset-name alias table (raw_lower -> canonical). Conservative and
# reversible: only true spelling/namespace variants of the SAME dataset name.
# It must NOT map two genuinely different datasets together, and it only touches
# the dataset-name slot of the group key.
ALIAS_VERSION = "v2.0"
DATASET_ALIASES = {
    # namespace/prefix variants of the same underlying dataset name
    "mteb/amazon_massive_intent": "amazon_massive_intent",
    "mteb/amazon_massive_scenario": "amazon_massive_scenario",
    "mteb/amazon_reviews_multi": "amazon_reviews_multi",
    "mteb/amazon_counterfactual": "amazon_counterfactual",
    "mteb/tweet_sentiment_extraction": "tweet_sentiment_extraction",
    "mteb/toxic_conversations_50k": "toxic_conversations_50k",
    # common spelling variants
    "sst-2": "sst2", "sst_2": "sst2",
    "conll-2003": "conll2003", "conll_2003": "conll2003",
}


def load_raw():
    p = os.path.join(OUT, "raw_performance_observations.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    return pd.read_csv(os.path.join(OUT, "raw_performance_observations.csv"))


def apply_aliases(df):
    raw = df["dataset_canonical"].astype(str)
    canon = raw.map(lambda s: DATASET_ALIASES.get(s, s))
    alias_rows = (pd.DataFrame({"raw": raw, "canonical": canon})
                  .drop_duplicates().sort_values("raw"))
    alias_rows["changed"] = alias_rows["raw"] != alias_rows["canonical"]
    alias_rows["alias_version"] = ALIAS_VERSION
    df = df.copy()
    df["dataset_canonical_pre_alias"] = raw
    df["dataset_canonical"] = canon
    # rebuild group id after aliasing (dataset name slot only)
    gk = ["dataset_canonical", "dataset_config", "task", "metric_canonical", "split"]
    df["ranking_group_id"] = df[gk].astype(str).agg("|".join, axis=1)
    return df, alias_rows


def canonicalize(df):
    """One row per (ranking_group_id, model). Returns (canonical_df, conflicts_df)."""
    key = ["ranking_group_id", "model_id_canonical"]
    conflicts = []
    out = []
    for (gid, mid), g in df.groupby(key, sort=False):
        vals = g["value_canonical"].astype(float)
        # exact duplicates collapse silently; a conflict = >1 distinct value
        distinct = vals.round(6).nunique()
        if distinct > 1:
            # prefer verified rows for resolution
            vg = g[g["verified"]] if g["verified"].any() else g
            resolved = float(np.median(vg["value_canonical"].astype(float)))
            conflicts.append({
                "ranking_group_id": gid, "model_id_canonical": mid,
                "n_source_rows": len(g), "n_distinct_values": int(distinct),
                "values": ";".join(f"{v:.6f}" for v in sorted(vals.tolist())),
                "verified_any": bool(g["verified"].any()),
                "resolved_value": resolved,
                "resolution_rule": "verified_precedence_then_median",
                "spread": float(vals.max() - vals.min()),
            })
            resolved_verified = bool(g["verified"].any())
        else:
            resolved = float(vals.iloc[0])
            resolved_verified = bool(g["verified"].any())
        r0 = g.iloc[0]
        out.append({
            "ranking_group_id": gid,
            "model_id_canonical": mid,
            "dataset_canonical": r0["dataset_canonical"],
            "dataset_config": r0["dataset_config"],
            "task": r0["task"],
            "metric_canonical": r0["metric_canonical"],
            "metric_direction": r0["metric_direction"],
            "split": r0["split"],
            "value_canonical": resolved,
            "n_source_rows": len(g),
            "verified": resolved_verified,
            "confidence_tier": int(r0["confidence_tier"]),
            "source_name": r0["source_name"],
        })
    return pd.DataFrame(out), pd.DataFrame(conflicts)


def group_summary(canon):
    g = canon.groupby("ranking_group_id").agg(
        n_models=("model_id_canonical", "nunique"),
        task=("task", "first"),
        dataset=("dataset_canonical", "first"),
        metric=("metric_canonical", "first"),
        n_distinct_values=("value_canonical", lambda s: int(s.round(6).nunique())),
    )
    g["nonconstant"] = g["n_distinct_values"] > 1
    g["effective_ge3"] = (g["n_models"] >= 3) & g["nonconstant"]
    g["robust_ge10"] = (g["n_models"] >= 10) & g["nonconstant"]
    g["strong_ge20"] = (g["n_models"] >= 20) & g["nonconstant"]
    return g


def main():
    os.makedirs(OUT, exist_ok=True)
    raw = load_raw()
    aliased, alias_tbl = apply_aliases(raw)
    alias_tbl.to_csv(os.path.join(OUT, "dataset_aliases.csv"), index=False, encoding="utf-8")

    canon, conflicts = canonicalize(aliased)
    try:
        canon.to_parquet(os.path.join(OUT, "canonical_performance_observations.parquet"),
                         index=False)
        fmt = "parquet"
    except Exception:
        canon.to_csv(os.path.join(OUT, "canonical_performance_observations.csv"), index=False)
        fmt = "csv"
    if len(conflicts) == 0:
        conflicts = pd.DataFrame(columns=["ranking_group_id", "model_id_canonical",
                                          "n_source_rows", "n_distinct_values", "values",
                                          "verified_any", "resolved_value", "resolution_rule",
                                          "spread"])
    conflicts.to_csv(os.path.join(OUT, "performance_conflicts.csv"), index=False, encoding="utf-8")

    gsum = group_summary(canon)
    report = {
        "canonical_format": fmt,
        "n_raw_records": int(len(raw)),
        "n_canonical_observations": int(len(canon)),
        "n_collapsed_exact_or_conflict_rows": int(len(raw) - len(canon)),
        "n_conflicts": int(len(conflicts)),
        "conflict_median_spread": float(conflicts["spread"].median()) if len(conflicts) else 0.0,
        "conflict_max_spread": float(conflicts["spread"].max()) if len(conflicts) else 0.0,
        "n_ranking_groups": int(gsum.shape[0]),
        "effective_ge3": int(gsum["effective_ge3"].sum()),
        "robust_ge10": int(gsum["robust_ge10"].sum()),
        "strong_ge20": int(gsum["strong_ge20"].sum()),
        "n_distinct_datasets": int(canon["dataset_canonical"].nunique()),
        "n_distinct_models": int(canon["model_id_canonical"].nunique()),
        "alias_version": ALIAS_VERSION,
        "n_alias_rules_applied": int(alias_tbl["changed"].sum()),
    }
    with open(os.path.join(OUT, "canonicalization_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
