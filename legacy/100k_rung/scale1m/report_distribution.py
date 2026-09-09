"""
report_distribution.py -- T2 v2 stage 5: comparative concentration and
coverage diagnostics.

Runbook: docs/1M/T2.md (v2), instructions Step 16.

Produces one machine-readable JSON and one Markdown table set comparing, side
by side:
    A  the v1 downloads-desc 150K head shard  (the population being replaced)
    B  the v2 candidate pool                  (discovery output)
    C  the v2 selected HALO                   (the frozen population)

The report exists to make ONE question answerable without argument: did the
redesign actually reduce concentration and improve coverage, or did it only
move the numbers around? Concentration is reported as HHI and effective count
(1/HHI) rather than only as top-k shares, because a top-1 share can fall while
the tail concentration is unchanged.

Run (from ModelLakeFishing/):
    .\\.venv\\Scripts\\python.exe -m scale1m.report_distribution `
        --v1 <raw>/annotated_v1.parquet --pool <cand>/annotated.parquet `
        --selected <cand>/selected/selected_halo.parquet --out docs/1M/T2_runs
"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

from scale1m.hf_crawl import utcnow, write_json_atomic

CATEGORICAL = ("supertask", "language_bucket", "source_type", "family_size_stratum",
               "popularity_stratum", "quantization_type")


def hhi(counts):
    """Herfindahl-Hirschman index on shares; 1/HHI = effective number."""
    tot = float(sum(counts))
    if tot <= 0:
        return 0.0
    return float(sum((c / tot) ** 2 for c in counts))


def entropy(counts, base=2.0):
    tot = float(sum(counts))
    if tot <= 0:
        return 0.0
    return float(-sum((c / tot) * math.log(c / tot, base) for c in counts if c > 0))


def concentration(df, col):
    vc = df[col].value_counts()
    n = len(df)
    h = hhi(vc.values)
    return {
        "distinct": int(len(vc)),
        "hhi": round(h, 6),
        "effective_count": round(1.0 / h, 2) if h > 0 else 0.0,
        "top1_share": round(float(vc.iloc[0]) / n, 5) if len(vc) else 0.0,
        "top1": str(vc.index[0]) if len(vc) else None,
        "top10_share": round(float(vc.iloc[:10].sum()) / n, 5),
        "top100_share": round(float(vc.iloc[:100].sum()) / n, 5),
        "top10": [[str(k), int(v), round(v / n, 5)] for k, v in vc.iloc[:10].items()],
    }


def distribution(df, col):
    vc = df[col].value_counts()
    n = len(df)
    return {
        "counts": {str(k): int(v) for k, v in vc.items()},
        "shares": {str(k): round(float(v) / n, 5) for k, v in vc.items()},
        "entropy_bits": round(entropy(vc.values), 4),
        "max_entropy_bits": round(math.log(max(len(vc), 1), 2), 4),
        "normalized_entropy": round(entropy(vc.values) / max(math.log(max(len(vc), 1), 2), 1e-9), 4)
        if len(vc) > 1 else 0.0,
    }


def profile(df, label):
    out = {"label": label, "n": int(len(df))}
    out["author"] = concentration(df, "author")
    if "family_id" in df:
        out["family"] = concentration(df, "family_id")
    for c in CATEGORICAL:
        if c in df:
            out[c] = distribution(df, c)
    if "metadata_quality_score" in df:
        q = df["metadata_quality_score"]
        out["metadata_quality"] = {
            "mean": round(float(q.mean()), 4),
            "p10": round(float(q.quantile(0.1)), 4),
            "median": round(float(q.median()), 4),
            "pass_rate": round(float((q >= 0.35).mean()), 5),
        }
    if "family_assignment_source" in df:
        out["family_assignment_source"] = distribution(df, "family_assignment_source")
        out["unresolved_family_rate"] = round(
            float((df["family_assignment_source"] == "unresolved_singleton").mean()), 5)
    if "is_mirror_publisher" in df:
        out["mirror_publisher_share"] = round(float(df["is_mirror_publisher"].mean()), 5)
    if "size_source" in df:
        out["size_known_share"] = round(float((df["size_source"] == "safetensors").mean()), 5)
    if "has_model_index" in df:
        out["model_index_share"] = round(float(df["has_model_index"].mean()), 5)
    return out


# --- markdown ---------------------------------------------------------------


def _row(label, a, b, c, fmt="%s"):
    def f(x):
        if x is None:
            return "--"
        return fmt % x
    return "| %s | %s | %s | %s |\n" % (label, f(a), f(b), f(c))


def _get(p, *path, default=None):
    cur = p
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def markdown(profiles, cfg_caps=None):
    A, B, C = profiles
    md = ["# T2 v2 distribution report\n",
          "\nGenerated %s. A = v1 downloads-desc head shard, B = v2 candidate pool, "
          "C = v2 selected HALO.\n" % utcnow(),
          "\n## Concentration\n\n",
          "| metric | A: v1 150K | B: pool | C: selected |\n|---|---|---|---|\n"]
    md.append(_row("models", A["n"], B["n"], C["n"], "%d"))
    for k, lbl in (("author", "author"), ("family", "family")):
        md.append(_row("%s distinct" % lbl, _get(A, k, "distinct"), _get(B, k, "distinct"),
                       _get(C, k, "distinct"), "%d"))
        md.append(_row("%s HHI" % lbl, _get(A, k, "hhi"), _get(B, k, "hhi"),
                       _get(C, k, "hhi"), "%.5f"))
        md.append(_row("%s effective count" % lbl, _get(A, k, "effective_count"),
                       _get(B, k, "effective_count"), _get(C, k, "effective_count"), "%.1f"))
        md.append(_row("%s top-1 share" % lbl, _get(A, k, "top1_share"),
                       _get(B, k, "top1_share"), _get(C, k, "top1_share"), "%.4f"))
        md.append(_row("%s top-10 share" % lbl, _get(A, k, "top10_share"),
                       _get(B, k, "top10_share"), _get(C, k, "top10_share"), "%.4f"))
        md.append(_row("%s top-100 share" % lbl, _get(A, k, "top100_share"),
                       _get(B, k, "top100_share"), _get(C, k, "top100_share"), "%.4f"))
    md.append(_row("largest author", _get(A, "author", "top1"), _get(B, "author", "top1"),
                   _get(C, "author", "top1")))

    for col in CATEGORICAL:
        if col not in A and col not in C:
            continue
        md.append("\n## %s\n\n" % col.replace("_", " "))
        md.append("| bucket | A share | B share | C share |\n|---|---|---|---|\n")
        keys = sorted(set(_get(A, col, "shares", default={})) |
                      set(_get(B, col, "shares", default={})) |
                      set(_get(C, col, "shares", default={})),
                      key=lambda k: -_get(C, col, "shares", default={}).get(k, 0))
        for k in keys:
            md.append(_row(k, _get(A, col, "shares", default={}).get(k),
                           _get(B, col, "shares", default={}).get(k),
                           _get(C, col, "shares", default={}).get(k), "%.4f"))
        md.append(_row("_entropy (bits)_", _get(A, col, "entropy_bits"),
                       _get(B, col, "entropy_bits"), _get(C, col, "entropy_bits"), "%.3f"))
        md.append(_row("_normalized entropy_", _get(A, col, "normalized_entropy"),
                       _get(B, col, "normalized_entropy"),
                       _get(C, col, "normalized_entropy"), "%.3f"))

    md.append("\n## Top 10 authors\n\n| rank | A: v1 150K | B: pool | C: selected |\n|---|---|---|---|\n")
    ta, tb, tc = (_get(p, "author", "top10", default=[]) for p in (A, B, C))
    for i in range(10):
        def cell(t):
            return "%s %.2f%%" % (t[i][0], 100 * t[i][2]) if i < len(t) else "--"
        md.append("| %d | %s | %s | %s |\n" % (i + 1, cell(ta), cell(tb), cell(tc)))

    md.append("\n## Quality / lineage\n\n| metric | A | B | C |\n|---|---|---|---|\n")
    for key, lbl, fmt in (("unresolved_family_rate", "unresolved family rate", "%.4f"),
                          ("mirror_publisher_share", "mirror-publisher share", "%.4f"),
                          ("size_known_share", "safetensors size known", "%.4f"),
                          ("model_index_share", "has model-index", "%.4f")):
        md.append(_row(lbl, A.get(key), B.get(key), C.get(key), fmt))
    md.append(_row("metadata quality mean", _get(A, "metadata_quality", "mean"),
                   _get(B, "metadata_quality", "mean"),
                   _get(C, "metadata_quality", "mean"), "%.4f"))
    md.append(_row("metadata quality pass rate", _get(A, "metadata_quality", "pass_rate"),
                   _get(B, "metadata_quality", "pass_rate"),
                   _get(C, "metadata_quality", "pass_rate"), "%.4f"))
    return "".join(md)


def main(argv=None):
    p = argparse.ArgumentParser(description="T2 v2 comparative distribution report")
    p.add_argument("--v1", required=True, help="annotated v1 150K parquet")
    p.add_argument("--pool", required=True, help="annotated v2 candidate parquet")
    p.add_argument("--selected", required=True, help="selected HALO parquet")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--tag", default="")
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    A = profile(pd.read_parquet(args.v1), "v1 downloads-desc 150K")
    B = profile(pd.read_parquet(args.pool), "v2 candidate pool")
    C = profile(pd.read_parquet(args.selected), "v2 selected HALO")

    suffix = ("_" + args.tag) if args.tag else ""
    write_json_atomic(os.path.join(args.out, "distribution_report%s.json" % suffix),
                      {"written_at": utcnow(), "profiles": [A, B, C]})
    md = markdown([A, B, C])
    with open(os.path.join(args.out, "distribution_report%s.md" % suffix), "w",
              encoding="utf-8") as fh:
        fh.write(md)

    print(md[:4000])
    print("\n[ok] -> %s/distribution_report%s.{json,md}" % (args.out, suffix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
