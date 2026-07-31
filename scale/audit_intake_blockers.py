"""
audit_intake_blockers.py -- P0 step 4: correct the naive first-pass audit and
quantify the four things that actually decide P1's design.

Why this file exists: the first pass (audit_corpus.py) produced three numbers
that were WRONG in a way that would have mis-planned P1, and finding that out
is itself a P0 result:

  * `model_profile` stores the STRING "unknown" for missing size/family, so a
    naive truthiness test reported 47,242/47,242 "with size". Real coverage is
    far lower.
  * `model_popularity.json` is a WRAPPER ({fetched_at, source, num_models,
    status_counts, models}); the payload is under ["models"]. The naive read
    reported 5 entries / 0 coverage.
  * the D0 BOUNDED whitelist has no notion of `@k` suffixes or `accuracy_norm`,
    so it scored ndcg@10 / recall@100 / map@10 as unusable and reported only
    10.6% usable rows. This corpus is 48% Retrieval -- those ARE bounded.

Also answers the question that decides whether the 12.4% dataset_desp coverage
is fatal or benign: are the described datasets the DEEP ones (many candidates,
i.e. the ones that can actually serve as gold@K queries)?

Run (from ModelLakeFishing/):
    .\\.venv\\Scripts\\python.exe -m scale.audit_intake_blockers
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from scale.pull_corpus import data_root

CHUNK = 250_000
RAW = os.path.join(data_root(), "modellens_v2", "raw")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "scale", "P0", "artifacts")

# @k-aware bounded-metric rule. Strip a trailing @<int>, lowercase, then match.
AT_K = re.compile(r"@\d+$")
BOUNDED_BASE = {
    "accuracy", "accuracy_norm", "acc", "acc_norm", "f1", "micro_f1",
    "macro_f1", "weighted_f1", "exact_match", "em", "matthews_correlation",
    "mcc", "pearson", "spearman", "cosine_pearson", "cosine_spearman",
    "ndcg", "recall", "precision", "map", "mrr", "rouge1", "rouge2",
    "rougel", "rougelsum", "bleu", "chrf", "ap", "auc", "roc_auc",
    "v_measure", "ari", "nmi", "hit", "hit_rate", "success",
}


def base_metric(m: str) -> str:
    return AT_K.sub("", str(m).strip().lower())


def main():
    os.makedirs(OUT, exist_ok=True)
    rep = {}

    # ---------- 1. side-file truth ---------------------------------------
    with open(os.path.join(RAW, "model_profile.json"), encoding="utf-8") as fh:
        prof = json.load(fh)
    with open(os.path.join(RAW, "model_popularity.json"), encoding="utf-8") as fh:
        pop_wrap = json.load(fh)
    pop = pop_wrap.get("models", {})

    UNK = {"unknown", "", "none", "null", None}
    real_size = {k for k, v in prof.items()
                 if isinstance(v, dict)
                 and str(v.get("size", "unknown")).strip().lower() not in UNK}
    real_fam = {k for k, v in prof.items()
                if isinstance(v, dict)
                and str(v.get("family", "unknown")).strip().lower() not in UNK}

    rep["profile"] = dict(
        n_keys=len(prof),
        n_real_size=len(real_size),
        n_real_family=len(real_fam),
        frac_real_size=len(real_size) / len(prof),
        frac_real_family=len(real_fam) / len(prof),
        size_value_samples=[prof[k].get("size") for k in list(real_size)[:8]],
    )
    rep["popularity"] = dict(
        wrapper_keys=list(pop_wrap),
        fetched_at=pop_wrap.get("fetched_at"),
        num_models_field=pop_wrap.get("num_models"),
        n_models_payload=len(pop),
        status_counts=pop_wrap.get("status_counts"),
        sample=dict(list(pop.items())[:3]),
    )

    # ---------- 2. CSV pass: metrics, values, depth, desc ------------------
    metric_rows = Counter()
    metric_vals = defaultdict(list)          # base metric -> sampled values
    per_ds_models = defaultdict(set)
    rows = 0

    for ch in pd.read_csv(os.path.join(RAW, "data.csv"),
                          usecols=["dataset", "model", "metric", "value"],
                          chunksize=CHUNK, low_memory=False):
        rows += len(ch)
        b = ch["metric"].map(base_metric)
        metric_rows.update(b.value_counts().to_dict())
        v = pd.to_numeric(ch["value"], errors="coerce")
        for bm, grp in v.groupby(b):
            g = grp.dropna().to_numpy()
            if g.size and len(metric_vals[bm]) < 40:
                metric_vals[bm].append(g[:: max(1, g.size // 2000)])
        for d, m in zip(ch["dataset"].astype(str), ch["model"].astype(str)):
            per_ds_models[d].add(m)

    bounded_rows = sum(c for m, c in metric_rows.items() if m in BOUNDED_BASE)
    rep["metrics"] = dict(
        n_base_metrics=len(metric_rows),
        rows=rows,
        bounded_rows_atk_aware=bounded_rows,
        bounded_frac_atk_aware=bounded_rows / rows,
        top_base=metric_rows.most_common(25),
        unbounded_top=[(m, c) for m, c in metric_rows.most_common(300)
                       if m not in BOUNDED_BASE][:20],
    )

    # per-metric scale: is it [0,1] or [0,100]?
    scale = {}
    for m, parts in metric_vals.items():
        if m not in BOUNDED_BASE:
            continue
        a = np.concatenate(parts)
        if a.size < 50:
            continue
        scale[m] = dict(
            n=int(a.size), min=float(a.min()), max=float(a.max()),
            p50=float(np.median(a)),
            frac_gt_1=float((a > 1.0).mean()),
            frac_gt_100=float((a > 100.0).mean()),
        )
    rep["metric_scale"] = dict(sorted(
        scale.items(), key=lambda kv: -metric_rows[kv[0]])[:20])

    # mixed-scale detection: same metric appearing both 0-1 and 0-100
    mixed = {m: s for m, s in scale.items()
             if 0.05 < s["frac_gt_1"] < 0.95}
    rep["mixed_scale_metrics"] = mixed

    # ---------- 3. does desc coverage land on the DEEP datasets? -----------
    desc_ok = set()
    for ch in pd.read_csv(os.path.join(RAW, "data.csv"),
                          usecols=["dataset", "dataset_desp"],
                          chunksize=CHUNK, low_memory=False):
        for d, s in zip(ch["dataset"].astype(str), ch["dataset_desp"]):
            if d not in desc_ok and isinstance(s, str) and s.strip():
                desc_ok.add(d)

    depth = {d: len(ms) for d, ms in per_ds_models.items()}
    def bucket_stats(lo):
        sel = [d for d, n in depth.items() if n >= lo]
        with_desc = sum(1 for d in sel if d in desc_ok)
        return dict(n_datasets=len(sel), with_desc=with_desc,
                    frac=with_desc / max(1, len(sel)))

    rep["desc_vs_depth"] = {
        "all": bucket_stats(1), ">=2": bucket_stats(2),
        ">=10": bucket_stats(10), ">=30": bucket_stats(30),
        ">=100": bucket_stats(100),
    }

    # usable query set: depth>=10 AND has desc (what gold@10 needs)
    q10 = [d for d, n in depth.items() if n >= 10]
    rep["query_set"] = dict(
        depth_ge10=len(q10),
        depth_ge10_with_desc=sum(1 for d in q10 if d in desc_ok),
        depth_ge30=sum(1 for n in depth.values() if n >= 30),
    )

    with open(os.path.join(OUT, "audit_blockers_v2.json"), "w",
              encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2, default=str)

    # ---------- console ---------------------------------------------------
    p = rep["profile"]
    print("--- CORRECTED side-file coverage ---")
    print(f"  model_profile keys      {p['n_keys']:,}")
    print(f"  real size (not 'unknown') {p['n_real_size']:,} ({p['frac_real_size']:.1%})")
    print(f"  real family               {p['n_real_family']:,} ({p['frac_real_family']:.1%})")
    print(f"  size samples            {p['size_value_samples']}")
    pp = rep["popularity"]
    print(f"  popularity payload      {pp['n_models_payload']:,} "
          f"(wrapper keys {pp['wrapper_keys']})")
    print(f"  popularity status       {pp['status_counts']}")

    m = rep["metrics"]
    print("\n--- CORRECTED bounded-metric coverage (@k aware) ---")
    print(f"  base metrics            {m['n_base_metrics']:,}")
    print(f"  bounded rows            {m['bounded_rows_atk_aware']:,} "
          f"({m['bounded_frac_atk_aware']:.1%})   [naive whitelist said 10.6%]")
    print(f"  top base: {m['top_base'][:10]}")
    print(f"  biggest UNbounded: {m['unbounded_top'][:8]}")

    print("\n--- metric scale (0-1 vs 0-100) ---")
    for k, s in list(rep["metric_scale"].items())[:12]:
        print(f"  {k:22s} n={s['n']:>7,} range[{s['min']:.3g},{s['max']:.4g}] "
              f"p50={s['p50']:.3g} frac>1={s['frac_gt_1']:.2f}")
    if mixed:
        print(f"  !! MIXED-SCALE metrics (both 0-1 and 0-100): {list(mixed)}")

    print("\n--- dataset_desp coverage vs candidate depth ---")
    for k, s in rep["desc_vs_depth"].items():
        print(f"  depth {k:>5s}: {s['with_desc']:>5,}/{s['n_datasets']:>6,} "
              f"have desc ({s['frac']:.1%})")
    print(f"\n  gold@10-capable query set (depth>=10): {rep['query_set']}")
    print(f"\n-> {os.path.join(OUT, 'audit_blockers_v2.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
