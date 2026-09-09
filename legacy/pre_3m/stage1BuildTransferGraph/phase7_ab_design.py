"""
phase7_ab_design.py -- Effective-Dataset guide, Phase 7.

Defines the data-pipeline A/B BEFORE training and materializes the common-cohort
manifest used by the paired comparison.

  A = current Stage-1 data, CORRECTED AT SOURCE so performance rows are unique
      (records.csv only, frozen models, deduped; NO model_config_dataset concat).
      This is NOT the leaked 12,205-row graph.
  B = rebuilt edge-first v2 data (canonical_performance_observations) under the SAME
      frozen 2,000-model universe.

Two analyses:
  A/B-1 common-cohort paired comparison: only ranking groups/models/observations
        shared by A and B, identical folds/candidates/seeds/hparams.
  A/B-2 expanded-cohort evaluation: B on its full effective-dataset universe;
        coverage/diversity, reported with dataset counts beside every metric, never a
        paired claim against A.

Writes:
  ab_design.md
  common_cohort_manifest.json
"""

import json
import os
import re

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
OUT = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage2TrainGraphSAGE",
                   "artifacts", "effective_dataset_v2")
RECORDS = os.path.join(_HERE, "dataset_embed", "data_hf1000d_2000m", "records.csv")


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def main():
    frozen = set(json.load(open(os.path.join(OUT, "frozen_models.json"),
                                encoding="utf-8"))["model_ids"])
    rec = pd.read_csv(RECORDS)
    A = rec[rec["model"].isin(frozen)].drop_duplicates(["model", "finetuned_dataset"]).copy()
    A["dsn"] = A["finetuned_dataset"].map(norm)

    B = pd.read_parquet(os.path.join(OUT, "canonical_performance_observations.parquet"))
    B = B.copy()
    B["dsn"] = B["dataset_canonical"].map(norm)

    A_pairs = A[["model", "dsn"]].drop_duplicates()
    B_pairs = B[["model_id_canonical", "dsn"]].rename(
        columns={"model_id_canonical": "model"}).drop_duplicates()
    common = A_pairs.merge(B_pairs, on=["model", "dsn"]).drop_duplicates()

    manifest = {
        "A_definition": "records.csv only, frozen models, deduped; NO model_config concat "
                        "(source-corrected current data; NOT the leaked 12,205-row graph)",
        "B_definition": "canonical_performance_observations.parquet (edge-first v2), same frozen universe",
        "A_distinct_pairs": int(len(A_pairs)),
        "A_distinct_datasets": int(A_pairs["dsn"].nunique()),
        "A_distinct_models": int(A_pairs["model"].nunique()),
        "B_distinct_pairs": int(len(B_pairs)),
        "B_distinct_datasets": int(B_pairs["dsn"].nunique()),
        "B_distinct_models": int(B_pairs["model"].nunique()),
        "common_cohort_pairs": int(len(common)),
        "common_cohort_datasets": int(common["dsn"].nunique()),
        "common_cohort_models": int(common["model"].nunique()),
        "match_key": "(model_id, normalized dataset name)",
        "match_caveat": "dataset-name normalization matches A's canon_key to B's model-index "
                        "dataset type; config/split granularity is coarsened to dataset for the "
                        "paired overlap. Exact per-config alignment is applied at Phase-5/8 when "
                        "both graphs share the frozen dataset id map.",
        "common_dataset_names": sorted(common["dsn"].unique().tolist()),
    }
    with open(os.path.join(OUT, "common_cohort_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    md = f"""# Stage-1 data-pipeline A/B design (Phase 7)

## Contract

- **A (corrected baseline)**: {manifest['A_distinct_pairs']:,} distinct (model,dataset) pairs
  from `records.csv` (frozen models, deduped, **no `model_config_dataset` concatenation**).
  This replaces the leaked 12,205-row graph — per the guide we never compare the leaked
  graph against B.
- **B (edge-first v2)**: {manifest['B_distinct_pairs']:,} distinct (model,dataset) pairs across
  {manifest['B_distinct_datasets']} datasets / {manifest['B_distinct_models']} models, from the
  canonical observation vault, same frozen 2,000-model universe.

## A/B-1 — common-cohort paired comparison

- Shared cohort: **{manifest['common_cohort_pairs']:,} (model,dataset) pairs**,
  **{manifest['common_cohort_datasets']} datasets**, **{manifest['common_cohort_models']} models**.
- Freeze identical fold assignments (Phase 6 folds restricted to common datasets), candidate
  IDs, train/val roles, init seeds, and hyperparameters. This isolates the effect of the
  Stage-1 graph/features + provenance fix.
- Report per-fold values, aggregate mean, 95% paired-bootstrap interval over dataset IDs, and
  the number of datasets contributing to each metric.

## A/B-2 — expanded-cohort evaluation

- Evaluate B on its full effective-dataset universe (119 effective datasets, 5 cold folds).
- Measures coverage/diversity gain; **not** a paired claim against A. Report dataset counts
  beside every metric, candidate-count strata (3-9 / 10-19 / 20-49 / 50+), and task strata.

## Guardrails

- Never claim a model win merely because B has more or easier datasets.
- Warm edge-holdout and whole-dataset cold-start reported separately.
- Exact normalized dot product for semantic ranking; HNSW ANN recall is a separate systems metric.
"""
    with open(os.path.join(OUT, "ab_design.md"), "w", encoding="utf-8") as f:
        f.write(md)

    print(json.dumps({k: v for k, v in manifest.items()
                      if k != "common_dataset_names"}, indent=2))


if __name__ == "__main__":
    main()
