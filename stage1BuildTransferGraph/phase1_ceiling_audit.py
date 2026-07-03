"""
phase1_ceiling_audit.py -- Effective-Dataset guide, Phase 1 (edge-first).

Reverses the old dataset-first order: for each of the FROZEN 2,000 models, enumerate
every structured model-index evaluation record FIRST (from the immutable raw model
cache, offline), then derive candidate ranking groups from the observations.

Ranking-group key (guide Phase 2 default):
    (dataset_canonical, dataset_config, task, metric_canonical, split)

Writes into artifacts/effective_dataset_v2/:
    raw_performance_observations.parquet   (every raw model-index record, provenance kept)
    coverage_ceiling.json / coverage_ceiling.md

No network. Confidence tier 1 = structured HF model-index (verified flag preserved).
"""

import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

SRC = os.path.join(_HERE, "hf1000d_2000m")
CACHE = os.path.join(SRC, "hf_cache", "models")
OUT = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage2TrainGraphSAGE",
                   "artifacts", "effective_dataset_v2")

# metric normalisation: name -> (canonical, higher_is_better)
METRIC_CANON = {
    "accuracy": ("accuracy", True), "acc": ("accuracy", True),
    "f1": ("f1", True), "f1_score": ("f1", True), "f1-score": ("f1", True),
    "precision": ("precision", True), "recall": ("recall", True),
    "matthews_correlation": ("matthews_correlation", True), "mcc": ("matthews_correlation", True),
    "pearson": ("pearson", True), "pearsonr": ("pearson", True),
    "spearmanr": ("spearman", True), "spearman": ("spearman", True),
    "exact_match": ("exact_match", True), "em": ("exact_match", True),
    "map": ("map", True), "mrr": ("mrr", True),
    "rouge": ("rouge", True), "rouge1": ("rouge1", True), "rougel": ("rougeL", True),
    "bleu": ("bleu", True), "sacrebleu": ("bleu", True),
    "loss": ("loss", False), "wer": ("wer", False), "cer": ("cer", False),
    "perplexity": ("perplexity", False),
}


# Only these canonical metrics are treated as PRIMARY comparable targets for the
# ceiling. Everything else (per-class f1, MTEB cos_sim_*/euclidean_*/manhattan_*
# sub-scores, etc.) is a lower tier and is EXCLUDED from ranking-group construction
# so we never fabricate comparability across incommensurable metrics.
PRIMARY_METRICS = {
    "accuracy", "f1", "matthews_correlation", "pearson", "spearman",
    "exact_match", "map", "mrr", "bleu", "rougeL", "wer", "cer",
}


def canon_metric(name, mtype):
    """Return (canonical, higher_is_better) only for exact-match whitelisted metric
    ids; otherwise (None, None) so the record is dropped from the primary table."""
    for cand in (mtype, name):
        if not cand:
            continue
        k = str(cand).strip().lower().replace(" ", "_")
        if k in METRIC_CANON:
            canon, hib = METRIC_CANON[k]
            if canon in PRIMARY_METRICS:
                return canon, hib
            return None, None
    return None, None


def canon_value(value, metric_canon):
    """Normalise to a 0..1-ish comparable float where the metric is a bounded score.
    Percentages (>1 for bounded metrics) are divided by 100. Raw value is kept
    separately by the caller. Returns (canon_value, rejected_reason|None)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None, "non_numeric"
    if not np.isfinite(v):
        return None, "non_finite"
    bounded = metric_canon in {"accuracy", "f1", "precision", "recall", "map", "mrr",
                               "matthews_correlation", "pearson", "spearman", "exact_match",
                               "rouge", "rouge1", "rougeL"}
    if bounded:
        if v > 1.0 and v <= 100.0:
            v = v / 100.0
        if v < -1.01 or v > 1.01:
            return None, "out_of_range"
    return v, None


def fname(mid):
    return os.path.join(CACHE, mid.replace("/", "__") + ".json")


def main():
    os.makedirs(OUT, exist_ok=True)
    frozen = json.load(open(os.path.join(OUT, "frozen_models.json"), encoding="utf-8"))["model_ids"]

    rows = []
    rejected = 0
    for mid in frozen:
        p = fname(mid)
        if not os.path.exists(p):
            continue
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, list):
            d = d[0] if d else {}
        sha = d.get("sha", "")
        last = d.get("lastModified", "")
        mi = d.get("model-index") or []
        for entry in mi:
            for res in (entry.get("results") or []):
                task = (res.get("task") or {})
                ds = (res.get("dataset") or {})
                ds_name = ds.get("name") or ds.get("type")
                ds_type = ds.get("type") or ds.get("name")
                cfg = ds.get("config") or ds.get("args") or ""
                split = ds.get("split") or ""
                task_type = task.get("type") or task.get("name") or ""
                for met in (res.get("metrics") or []):
                    mcanon, hib = canon_metric(met.get("name"), met.get("type"))
                    if mcanon is None:      # non-primary/incommensurable metric -> drop
                        rejected += 1
                        continue
                    cv, rej = canon_value(met.get("value"), mcanon)
                    if rej is not None:
                        rejected += 1
                        continue
                    rows.append({
                        "source_name": "hf_model_index",
                        "source_record_id": f"{mid}@{sha[:12]}",
                        "source_snapshot": last,
                        "model_id_canonical": mid,
                        "dataset_raw": ds_name,
                        "dataset_canonical": str(ds_type).strip().lower(),
                        "dataset_config": str(cfg).strip(),
                        "task": str(task_type).strip().lower(),
                        "metric_raw": met.get("name") or met.get("type"),
                        "metric_canonical": mcanon,
                        "metric_direction": "higher" if hib else "lower",
                        "split": str(split).strip().lower(),
                        "value_raw": met.get("value"),
                        "value_canonical": cv,
                        "verified": bool(met.get("verified", False)),
                        "confidence_tier": 1,
                    })

    df = pd.DataFrame(rows)
    # ranking-group key
    gk = ["dataset_canonical", "dataset_config", "task", "metric_canonical", "split"]
    df["ranking_group_id"] = df[gk].astype(str).agg("|".join, axis=1)
    try:
        df.to_parquet(os.path.join(OUT, "raw_performance_observations.parquet"), index=False)
        raw_fmt = "parquet"
    except Exception:
        df.to_csv(os.path.join(OUT, "raw_performance_observations.csv"), index=False)
        raw_fmt = "csv"

    # ---- ceiling table ---------------------------------------------------------
    # per ranking group: distinct models with a valid value, and non-constant check
    def group_stats(g):
        vals = g.groupby("model_id_canonical")["value_canonical"].mean()  # collapse dup runs
        return pd.Series({"n_models": vals.index.nunique(),
                          "nonconstant": bool(vals.nunique() > 1)})
    gstats = df.groupby("ranking_group_id").apply(group_stats, include_groups=False)

    def count_groups(min_models, require_nonconst):
        m = gstats["n_models"] >= min_models
        if require_nonconst:
            m &= gstats["nonconstant"]
        return int(m.sum())

    thresholds = [
        ("ge1", 1, False), ("ge3_nonconst", 3, True), ("ge5_nonconst", 5, True),
        ("ge10_nonconst", 10, True), ("ge20_nonconst", 20, True), ("ge30_nonconst", 30, True),
    ]
    ceiling = []
    for label, k, nc in thresholds:
        gids = gstats.index[(gstats["n_models"] >= k) & (gstats["nonconstant"] if nc else True)]
        sub = df[df["ranking_group_id"].isin(gids)]
        ceiling.append({
            "threshold": label, "min_models": k, "require_nonconstant": nc,
            "ranking_groups": len(gids),
            "distinct_observations": int(sub.drop_duplicates(
                ["ranking_group_id", "model_id_canonical"]).shape[0]),
            "distinct_covered_models": int(sub["model_id_canonical"].nunique()),
        })

    # models with >=k observations (distinct ranking groups)
    obs_per_model = df.drop_duplicates(["ranking_group_id", "model_id_canonical"]) \
                      .groupby("model_id_canonical").size()
    models_ge = {f"ge{k}": int((obs_per_model >= k).sum()) for k in (1, 3, 10)}

    # breakdowns
    def dist(col, top=20):
        return df.drop_duplicates(["ranking_group_id"]).groupby(col).size() \
                 .sort_values(ascending=False).head(top).to_dict()

    targets = {
        "minimum_evaluable_ge3": 150, "robust_ge10": 100, "strong_ge20": 60,
        "canonical_observations": 20000, "models_ge3_obs": 1200,
    }
    attained = {
        "minimum_evaluable_ge3": count_groups(3, True),
        "robust_ge10": count_groups(10, True),
        "strong_ge20": count_groups(20, True),
        "canonical_observations": int(df.drop_duplicates(
            ["ranking_group_id", "model_id_canonical"]).shape[0]),
        "models_ge3_obs": models_ge["ge3"],
    }

    result = {
        "raw_observations_format": raw_fmt,
        "n_raw_records": len(df),
        "n_rejected_values": rejected,
        "n_frozen_models_with_records": int(df["model_id_canonical"].nunique()),
        "n_ranking_groups": int(gstats.shape[0]),
        "n_distinct_datasets": int(df["dataset_canonical"].nunique()),
        "ceiling_table": ceiling,
        "models_with_ge_observations": models_ge,
        "targets": targets,
        "attained": attained,
        "targets_met": {k: attained[k] >= targets[k] for k in targets},
        "by_task": dist("task"),
        "by_metric": dist("metric_canonical"),
        "by_split": dist("split"),
        "raw_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest(),
    }
    with open(os.path.join(OUT, "coverage_ceiling.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # ---- markdown --------------------------------------------------------------
    L = ["# Coverage ceiling audit (edge-first, frozen 2,000 models) — Phase 1\n"]
    L.append(f"- raw model-index records: **{len(df):,}** "
             f"(rejected values: {rejected:,})")
    L.append(f"- frozen models with >=1 record: **{result['n_frozen_models_with_records']}** / 2000")
    L.append(f"- distinct ranking groups (dataset|config|task|metric|split): "
             f"**{result['n_ranking_groups']:,}**")
    L.append(f"- distinct datasets: **{result['n_distinct_datasets']:,}**\n")
    L.append("## Ceiling table\n")
    L.append("| threshold | ranking groups | distinct observations | distinct covered models |")
    L.append("|---|---:|---:|---:|")
    for c in ceiling:
        L.append(f"| {c['threshold']} | {c['ranking_groups']:,} | "
                 f"{c['distinct_observations']:,} | {c['distinct_covered_models']:,} |")
    L.append("")
    L.append("## Targets (goals, not permission to manufacture)\n")
    L.append("| target | goal | attained | met |")
    L.append("|---|---:|---:|:--:|")
    for k in targets:
        L.append(f"| {k} | {targets[k]:,} | {attained[k]:,} | "
                 f"{'✓' if result['targets_met'][k] else '✗'} |")
    L.append("")
    L.append(f"- models with >=1 / >=3 / >=10 observations: "
             f"{models_ge['ge1']} / {models_ge['ge3']} / {models_ge['ge10']}\n")
    L.append("## By task (top ranking groups)\n")
    for k, v in list(result["by_task"].items())[:12]:
        L.append(f"- `{k or '(blank)'}`: {v}")
    L.append("\n## By metric\n")
    for k, v in list(result["by_metric"].items())[:12]:
        L.append(f"- `{k}`: {v}")
    with open(os.path.join(OUT, "coverage_ceiling.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    print(f"raw records: {len(df):,}  ranking groups: {result['n_ranking_groups']:,}  "
          f"datasets: {result['n_distinct_datasets']:,}")
    print("ceiling:")
    for c in ceiling:
        print(f"  {c['threshold']:>14}: groups={c['ranking_groups']:<6} "
              f"obs={c['distinct_observations']:<7} models={c['distinct_covered_models']}")
    print("targets met:", result["targets_met"])


if __name__ == "__main__":
    main()
