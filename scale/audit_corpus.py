"""
audit_corpus.py -- P0 step 3: parse the frozen corpus and establish the REAL
counts + the intake profile that P1 will have to handle.

Two jobs:
 1. Settle the headline count. The PLAN deliberately refused to pre-fill a
    model count ("42k" per the user vs "~47K" per the paper). This resolves it
    from the actual bytes, not from either claim.
 2. Profile what modellens_intake.py will face: metric vocabulary vs our
    BOUNDED whitelist, value scale (percent vs unit), per-dataset candidate
    depth (can a `gold` best-model even be defined?), and the coverage of the
    side JSONs we plan to turn into node features / lineage edges.

Streams the CSV in chunks -- data.csv is ~900 MB with a free-text column.

Run (from ModelLakeFishing/):
    .\\.venv\\Scripts\\python.exe -m scale.audit_corpus
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from scale.pull_corpus import REPOS, data_root

# Our D0 bounded-metric whitelist (stage1BuildTransferGraph/d0_build_graph.py)
BOUNDED = {"accuracy", "f1", "matthews_correlation", "pearson", "spearman",
           "exact_match", "map", "mrr", "rougeL"}

CHUNK = 250_000


def audit(corpus_key: str, root: str, out_dir: str):
    spec = REPOS[corpus_key]
    raw = os.path.join(root, spec["subdir"], "raw")
    csv_name = "data.csv" if corpus_key == "v2" else "data_clean.csv"
    csv_path = os.path.join(raw, csv_name)

    print(f"=== auditing {corpus_key}: {csv_path}")
    print(f"    revision {spec['revision']}")

    # ---- pass 1: the 5 structured columns -------------------------------
    models, datasets, tasks = set(), set(), set()
    metric_counts = Counter()
    task_counts = Counter()
    ds_model_pairs = 0
    rows = 0
    per_ds_models = defaultdict(set)
    per_model_ds = Counter()
    values = []
    value_min, value_max = np.inf, -np.inf
    n_null_value = 0

    cols = ["task", "dataset", "model", "metric", "value"]
    reader = pd.read_csv(csv_path, usecols=cols, chunksize=CHUNK,
                         low_memory=False)
    for i, ch in enumerate(reader):
        rows += len(ch)
        models.update(ch["model"].astype(str).unique())
        datasets.update(ch["dataset"].astype(str).unique())
        tasks.update(ch["task"].astype(str).unique())
        metric_counts.update(ch["metric"].astype(str).value_counts().to_dict())
        task_counts.update(ch["task"].astype(str).value_counts().to_dict())

        v = pd.to_numeric(ch["value"], errors="coerce")
        n_null_value += int(v.isna().sum())
        vv = v.dropna().to_numpy()
        if vv.size:
            value_min = min(value_min, float(vv.min()))
            value_max = max(value_max, float(vv.max()))
            # reservoir-ish sample for quantiles without holding 2M floats
            if len(values) < 400_000:
                values.append(vv[:: max(1, vv.size // 20_000)])

        for d, m in zip(ch["dataset"].astype(str), ch["model"].astype(str)):
            per_ds_models[d].add(m)
            per_model_ds[m] += 1
        ds_model_pairs += len(ch)
        if (i + 1) % 4 == 0:
            print(f"    ... {rows:,} rows")

    vsample = np.concatenate(values) if values else np.array([])

    # ---- pass 2: dataset_desp coverage ----------------------------------
    desc_seen, desc_nonempty = set(), set()
    reader = pd.read_csv(csv_path, usecols=["dataset", "dataset_desp"],
                         chunksize=CHUNK, low_memory=False)
    for ch in reader:
        ch["dataset"] = ch["dataset"].astype(str)
        for d, s in zip(ch["dataset"], ch["dataset_desp"]):
            if d in desc_seen:
                continue
            desc_seen.add(d)
            if isinstance(s, str) and s.strip():
                desc_nonempty.add(d)

    # ---- side JSONs -----------------------------------------------------
    def load(name):
        p = os.path.join(raw, name)
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)

    model2id = load("model2id.json")
    model2family = load("model2family.json")
    model_profile = load("model_profile.json")
    model_pop = load("model_popularity.json")
    task2id = load("task2id.json")
    metric2id = load("metric2id.json")
    family2id = load("family2id.json")

    # coverage of the CSV's models by each side file
    def cov(d):
        if d is None:
            return None
        keys = set(d)
        return len(models & keys)

    # size availability inside model_profile
    size_keys = ("size", "model_size", "params", "num_parameters",
                 "safetensors_total", "total")
    n_with_size = 0
    profile_field_hist = Counter()
    if model_profile:
        for k, v in model_profile.items():
            if k not in models:
                continue
            if isinstance(v, dict):
                profile_field_hist.update(v.keys())
                if any(v.get(s) not in (None, "", 0) for s in size_keys):
                    n_with_size += 1

    # candidate depth per dataset: can a gold best-model be defined?
    depth = np.array(sorted(len(v) for v in per_ds_models.values()))
    bounded_hits = sum(c for m, c in metric_counts.items()
                       if str(m).lower() in BOUNDED)

    rep = dict(
        corpus=corpus_key,
        revision=spec["revision"],
        csv=csv_name,
        rows=rows,
        n_models_csv=len(models),
        n_datasets_csv=len(datasets),
        n_tasks_csv=len(tasks),
        n_metrics_csv=len(metric_counts),
        null_value_rows=n_null_value,
        value_min=None if value_min == np.inf else value_min,
        value_max=None if value_max == -np.inf else value_max,
        value_q=({q: float(np.quantile(vsample, q))
                  for q in (0.01, 0.25, 0.5, 0.75, 0.99)} if vsample.size else {}),
        frac_value_gt_1=(float((vsample > 1.0).mean()) if vsample.size else None),
        bounded_metric_rows=bounded_hits,
        bounded_metric_frac=bounded_hits / rows if rows else None,
        top_metrics=metric_counts.most_common(20),
        top_tasks=task_counts.most_common(20),
        dataset_desp_coverage=len(desc_nonempty),
        dataset_desp_frac=len(desc_nonempty) / max(1, len(datasets)),
        candidate_depth=dict(
            min=int(depth.min()) if depth.size else 0,
            p50=int(np.quantile(depth, .5)) if depth.size else 0,
            p90=int(np.quantile(depth, .9)) if depth.size else 0,
            max=int(depth.max()) if depth.size else 0,
            n_ds_ge2=int((depth >= 2).sum()),
            n_ds_ge10=int((depth >= 10).sum()),
            n_ds_ge30=int((depth >= 30).sum()),
        ),
        sidefile_counts=dict(
            model2id=len(model2id) if model2id else None,
            model2family=len(model2family) if model2family else None,
            model_profile=len(model_profile) if model_profile else None,
            model_popularity=len(model_pop) if model_pop else None,
            task2id=len(task2id) if task2id else None,
            metric2id=len(metric2id) if metric2id else None,
            family2id=len(family2id) if family2id else None,
        ),
        sidefile_coverage_of_csv_models=dict(
            model2id=cov(model2id), model2family=cov(model2family),
            model_profile=cov(model_profile), model_popularity=cov(model_pop),
        ),
        model_profile_with_size=n_with_size,
        model_profile_fields=profile_field_hist.most_common(15),
    )

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"audit_{corpus_key}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2, default=str)

    # ---- console summary ------------------------------------------------
    print(f"\n--- {corpus_key} HEADLINE ---")
    print(f"  rows                 {rows:,}")
    print(f"  unique models        {len(models):,}")
    print(f"  unique datasets      {len(datasets):,}")
    print(f"  unique tasks         {len(tasks):,}")
    print(f"  unique metrics       {len(metric_counts):,}")
    print(f"  value range          [{rep['value_min']}, {rep['value_max']}]  "
          f"frac>1={rep['frac_value_gt_1']}")
    print(f"  bounded-metric rows  {bounded_hits:,} "
          f"({rep['bounded_metric_frac']:.1%})")
    print(f"  dataset_desp cover   {len(desc_nonempty):,}/{len(datasets):,} "
          f"({rep['dataset_desp_frac']:.1%})")
    print(f"  candidate depth      {rep['candidate_depth']}")
    print(f"  sidefile counts      {rep['sidefile_counts']}")
    print(f"  sidefile coverage    {rep['sidefile_coverage_of_csv_models']}")
    print(f"  profile w/ size      {n_with_size:,}")
    print(f"  profile fields       {rep['model_profile_fields'][:8]}")
    print(f"  -> {out}")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["v1", "v2"], default="v2")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs", "scale", "P0", "artifacts"))
    args = ap.parse_args()
    audit(args.corpus, data_root(), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
