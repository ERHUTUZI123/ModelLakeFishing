"""
phase6_folds.py -- Effective-Dataset guide, Phase 6.

Freeze FIVE deterministic cold-DATASET folds over the effective-dataset set, instead
of the old single 9-dataset cold split. Stratified by task and candidate-count bucket.
Every effective dataset appears in exactly one fold; fold sizes differ by at most one
within each stratum (exact assignment, no independent 20% rounding).

An effective dataset here = a distinct dataset_canonical with >=1 effective ranking
group (>=3 distinct models, non-constant) in the canonical vault. Final membership is
intersected with embedding survivors in Phase 5 (deployable-feature gate); this manifest
records the pre-embedding design and its seed so it is reproducible.

Writes:
  cold_folds/fold_manifest.json
  cold_folds/fold_{0..4}.json
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
OUT = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage2TrainGraphSAGE",
                   "artifacts", "effective_dataset_v2")
FOLD_DIR = os.path.join(OUT, "cold_folds")
N_FOLDS = 5
SPLIT_SEED = 20260703


def cand_bucket(n):
    return "3-9" if n <= 9 else ("10-19" if n <= 19 else ("20-49" if n <= 49 else "50+"))


def load_canon():
    p = os.path.join(OUT, "canonical_performance_observations.parquet")
    return pd.read_parquet(p) if os.path.exists(p) else \
        pd.read_csv(os.path.join(OUT, "canonical_performance_observations.csv"))


def effective_datasets(canon):
    """dataset_canonical -> (max distinct-model coverage over its effective groups, dominant task)."""
    g = canon.groupby("ranking_group_id").agg(
        n_models=("model_id_canonical", "nunique"),
        dataset=("dataset_canonical", "first"),
        task=("task", "first"),
        nonconst=("value_canonical", lambda s: s.round(6).nunique() > 1),
    )
    eff = g[(g["n_models"] >= 3) & g["nonconst"]]
    out = {}
    for ds, sub in eff.groupby("dataset"):
        n = int(sub["n_models"].max())
        task = sub.sort_values("n_models", ascending=False)["task"].iloc[0]
        out[ds] = (n, str(task))
    return out


def assign_folds(eff):
    """Seeded, stratified, globally balanced assignment: iterate (task, candidate-bucket)
    strata in fixed order; within each shuffled stratum assign every dataset to the
    currently least-loaded fold (tie-break by lowest fold index). Guarantees disjoint
    folds, full coverage, and GLOBAL fold sizes differing by at most one, while keeping
    each stratum spread across folds."""
    rng = np.random.default_rng(SPLIT_SEED)
    strata = {}
    for ds, (n, task) in eff.items():
        strata.setdefault((task, cand_bucket(n)), []).append(ds)
    load = [0] * N_FOLDS
    fold_of = {}
    for key in sorted(strata.keys(), key=str):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        for ds in ids:
            f = int(min(range(N_FOLDS), key=lambda k: (load[k], k)))
            fold_of[ds] = f
            load[f] += 1
    return fold_of


def main():
    os.makedirs(FOLD_DIR, exist_ok=True)
    canon = load_canon()
    eff = effective_datasets(canon)
    fold_of = assign_folds(eff)

    folds = {f: [] for f in range(N_FOLDS)}
    for ds, f in fold_of.items():
        folds[f].append(ds)

    # sanity: disjoint + full coverage
    all_assigned = [d for ds in folds.values() for d in ds]
    assert len(all_assigned) == len(set(all_assigned)) == len(eff), "fold coverage/disjoint broken"

    obs_hash = hashlib.sha256(
        pd.util.hash_pandas_object(canon, index=True).values.tobytes()).hexdigest()

    fold_summaries = []
    for f in range(N_FOLDS):
        ids = sorted(folds[f])
        tasks = {}
        buckets = {}
        for ds in ids:
            n, t = eff[ds]
            tasks[t] = tasks.get(t, 0) + 1
            buckets[cand_bucket(n)] = buckets.get(cand_bucket(n), 0) + 1
        fold_obj = {
            "fold": f, "n_cold_datasets": len(ids),
            "cold_dataset_names": ids,
            "candidate_counts": {ds: eff[ds][0] for ds in ids},
            "task_distribution": tasks,
            "candidate_bucket_distribution": buckets,
            "split_seed": SPLIT_SEED,
            "observation_table_sha256": obs_hash,
        }
        with open(os.path.join(FOLD_DIR, f"fold_{f}.json"), "w", encoding="utf-8") as fh:
            json.dump(fold_obj, fh, indent=2)
        fold_summaries.append({"fold": f, "n": len(ids),
                               "tasks": tasks, "buckets": buckets})

    manifest = {
        "n_effective_datasets": len(eff),
        "n_folds": N_FOLDS,
        "split_seed": SPLIT_SEED,
        "observation_table_sha256": obs_hash,
        "note": "Cold folds over effective datasets (>=1 effective ranking group). "
                "Final membership intersects with Phase-4 embedding survivors. Replaces "
                "the old single 9-dataset cold split. Warm edge-holdout on non-cold datasets "
                "is reported separately in Phase 8.",
        "fold_sizes": [len(folds[f]) for f in range(N_FOLDS)],
        "fold_summaries": fold_summaries,
    }
    with open(os.path.join(FOLD_DIR, "fold_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"effective datasets: {len(eff)}")
    print(f"fold sizes: {manifest['fold_sizes']} (sum={sum(manifest['fold_sizes'])})")
    for s in fold_summaries:
        print(f"  fold {s['fold']}: {s['n']} datasets, tasks={s['tasks']}")


if __name__ == "__main__":
    main()
