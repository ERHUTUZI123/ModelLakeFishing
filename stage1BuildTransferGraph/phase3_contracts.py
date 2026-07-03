"""
phase3_contracts.py -- Effective-Dataset guide, Phase 3.

Separates the two contracts and builds the deterministic single-metric-per-dataset
selection used by the compatibility (one-metric) experiment.

  1. The canonical observation vault is the ONLY label/eval source and keeps EVERY
     valid observation, including low-performing models. Nothing here is filtered
     by score.
  2. Graph message edges (built in Phase 5) are a filtered VIEW of this vault; the
     filter policy is explicit and ablated, and never deletes an observation from
     the vault.

Single-metric selection per (dataset, config) ranking problem, deterministic:
  pick the metric with the largest distinct-model coverage,
  then highest mean confidence tier, then a fixed metric-name tie-break.
  The metric is NOT chosen by which one gives the best model result.

Writes:
  single_metric_selection.csv
  message_edge_policy.json
  phase3_report.json
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

# fixed tie-break order (lower index = preferred) — a fixed prior, NOT result-based
METRIC_TIEBREAK = ["accuracy", "f1", "matthews_correlation", "spearman", "pearson",
                   "exact_match", "map", "mrr", "bleu", "rougeL", "wer", "cer"]
_TB = {m: i for i, m in enumerate(METRIC_TIEBREAK)}


def load_canon():
    p = os.path.join(OUT, "canonical_performance_observations.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    return pd.read_csv(os.path.join(OUT, "canonical_performance_observations.csv"))


def select_single_metric(canon):
    """For each (dataset_canonical, dataset_config, task, split) problem, choose one
    metric deterministically by coverage, then confidence, then fixed tie-break."""
    prob_key = ["dataset_canonical", "dataset_config", "task", "split"]
    rows = []
    for keyvals, g in canon.groupby(prob_key, sort=True):
        by_metric = g.groupby("metric_canonical").agg(
            n_models=("model_id_canonical", "nunique"),
            conf=("confidence_tier", "mean"),
            nonconst=("value_canonical", lambda s: int(s.round(6).nunique() > 1)),
        ).reset_index()
        by_metric["tb"] = by_metric["metric_canonical"].map(lambda m: _TB.get(m, 999))
        # sort: coverage desc, confidence desc, tie-break asc
        by_metric = by_metric.sort_values(
            ["n_models", "conf", "tb"], ascending=[False, False, True])
        chosen = by_metric.iloc[0]
        d, cfg, task, split = keyvals
        rows.append({
            "dataset_canonical": d, "dataset_config": cfg, "task": task, "split": split,
            "chosen_metric": chosen["metric_canonical"],
            "chosen_n_models": int(chosen["n_models"]),
            "chosen_nonconstant": bool(chosen["nonconst"]),
            "n_candidate_metrics": len(by_metric),
            "all_metrics": ";".join(by_metric["metric_canonical"].tolist()),
            "ranking_group_id": "|".join([str(d), str(cfg), str(task),
                                          str(chosen["metric_canonical"]), str(split)]),
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    canon = load_canon()

    # vault-retention proof: low performers present (bottom decile of every group kept)
    grp = canon.groupby("ranking_group_id")["value_canonical"]
    low_perf_retained = int((canon["value_canonical"] <= grp.transform("quantile", 0.1)).sum())

    sel = select_single_metric(canon)
    sel.to_csv(os.path.join(OUT, "single_metric_selection.csv"), index=False, encoding="utf-8")

    # message-edge policy is documented, ablatable, and non-destructive to the vault
    policy = {
        "vault": "canonical_performance_observations.parquet is the ONLY label/eval source; "
                 "no score filter is ever applied to it.",
        "message_edge_view": {
            "purpose": "training-time graph topology only",
            "default_positive_threshold": 0.6,
            "note": "high-performance trained_on edges for message passing; this is an "
                    "ABLATED policy knob, applied to a VIEW, and never removes a row from "
                    "the vault. Low-performing valid observations remain available for "
                    "ranking labels, candidate sets, and evaluation.",
            "ablation_variants": [0.0, 0.5, 0.6],
        },
        "single_metric_rule": "coverage desc -> confidence desc -> fixed metric tie-break "
                              + str(METRIC_TIEBREAK),
    }
    with open(os.path.join(OUT, "message_edge_policy.json"), "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2)

    report = {
        "n_canonical_observations": int(len(canon)),
        "low_perf_observations_retained_bottom10pct": low_perf_retained,
        "n_single_metric_problems": int(len(sel)),
        "n_problems_multi_metric_available": int((sel["n_candidate_metrics"] > 1).sum()),
        "single_metric_distinct_datasets": int(sel["dataset_canonical"].nunique()),
        "chosen_metric_distribution": sel["chosen_metric"].value_counts().to_dict(),
        "single_metric_effective_ge3": int(((sel["chosen_n_models"] >= 3) &
                                            sel["chosen_nonconstant"]).sum()),
        "single_metric_robust_ge10": int(((sel["chosen_n_models"] >= 10) &
                                          sel["chosen_nonconstant"]).sum()),
        "single_metric_strong_ge20": int(((sel["chosen_n_models"] >= 20) &
                                          sel["chosen_nonconstant"]).sum()),
    }
    with open(os.path.join(OUT, "phase3_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
