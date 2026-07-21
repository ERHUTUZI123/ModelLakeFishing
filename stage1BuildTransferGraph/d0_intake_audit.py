"""
d0_intake_audit.py -- D0 lake expansion, Phase A: OFFLINE intake audit.

Remediation plan v2 (docs/optimization/RETRIEVAL_QUALITY_REMEDIATION_PLAN.md §4)
calls for a 10,000-model x ~2,000-dataset lake. Before any network harvest, this
script audits the FULL raw cache (hf1000d_2000m/hf_cache/, 12,871 model JSONs +
4,460 dataset JSONs -- far more than the 2,000 frozen models actually selected)
against the relaxed intake rule:

    intake = has primary-metric model-index records
           OR has lineage clue (cardData.base_model / base_model:* tag)
           OR family parsable (rule/dynamic inference, not Other)

and projects the evaluable-gold-dataset count (>=3 labelled models, non-constant
value) the expanded lake would support -- the G-D0 target is >=200 (vs 61 today).

Reuses (no new parsing logic, no network):
  - phase1_ceiling_audit: METRIC_CANON whitelist, canon_metric, canon_value
  - build_hf_pilot_v3.canon_key: dataset name canonicalisation
  - effective_dataset_v2/dataset_aliases.csv: v2 alias reconciliation
  - dataset_embed.utils.fetch_metadata._infer_one_family: family inference

Anti-leakage discipline (the attributes.py 42% lesson): observations are deduped
to ONE row per (dataset_canonical, model) before any counting; the funnel reports
raw vs distinct explicitly.

Writes into artifacts/d0_lake/:
    d0_model_intake.csv        per-model channel flags + intake decision
    d0_observations.parquet    deduped primary-metric observations (provenance kept)
    d0_dataset_pool.csv        per-canonical-dataset label stats + cache cross-ref
    d0_funnel.json / d0_funnel.md
    d0_intake_models.json (+ .sha256)   frozen candidate list, deterministic order

Run (from stage1BuildTransferGraph/):
    ../.venv/Scripts/python.exe d0_intake_audit.py
"""

import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from phase1_ceiling_audit import canon_metric, canon_value  # noqa: E402
from build_hf_pilot_v3 import canon_key  # noqa: E402
from dataset_embed.utils.fetch_metadata import _infer_one_family, FAMILY_OTHER  # noqa: E402

CACHE_M = os.path.join(_HERE, "hf1000d_2000m", "hf_cache", "models")
CACHE_D = os.path.join(_HERE, "hf1000d_2000m", "hf_cache", "datasets")
# Phase-B online top-up caches (d0_online_harvest.py); merged if present
CACHE_M2 = os.path.join(_HERE, "d0_lake_cache", "models")
CACHE_D2 = os.path.join(_HERE, "d0_lake_cache", "datasets")
V2_DIR = os.path.join(_HERE, "..", "stage2TrainGraphSAGE", "artifacts", "effective_dataset_v2")
OUT = os.path.join(_HERE, "artifacts", "d0_lake")

# bounded higher-is-better metrics usable as a single comparable [0,1] score for
# the dataset-level gold projection (mirrors graph edge_attr semantics)
BOUNDED_HIGHER = {"accuracy", "f1", "matthews_correlation", "pearson", "spearman",
                  "exact_match", "map", "mrr", "rougeL"}


def _load(path):
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:
        return None
    if isinstance(d, list):
        d = d[0] if d else {}
    return d if isinstance(d, dict) else None


def _alias_map():
    p = os.path.join(V2_DIR, "dataset_aliases.csv")
    if not os.path.exists(p):
        return {}
    df = pd.read_csv(p)
    return dict(zip(df["raw"].astype(str), df["canonical"].astype(str)))


def scan_models():
    alias = _alias_map()
    files_old = sorted(glob.glob(os.path.join(CACHE_M, "*.json")))
    files_new = sorted(glob.glob(os.path.join(CACHE_M2, "*.json")))
    seen_base = {os.path.basename(p) for p in files_old}
    files = files_old + [p for p in files_new if os.path.basename(p) not in seen_base]
    rows, obs = [], []
    n_files = {"phaseA_cache": len(files_old), "phaseB_cache": len(files_new),
               "merged_distinct": len(files)}
    for path in files:
        d = _load(path)
        if d is None:
            rows.append({"model_id": os.path.basename(path)[:-5].replace("__", "/"),
                         "parse_ok": False})
            continue
        mid = d.get("id") or d.get("modelId") or os.path.basename(path)[:-5].replace("__", "/")
        cd = d.get("cardData") or {}
        if not isinstance(cd, dict):
            cd = {}
        tags = [t for t in (d.get("tags") or []) if isinstance(t, str)]
        lineage_base = cd.get("base_model") or ""
        if not lineage_base:
            for t in tags:
                if t.startswith("base_model:"):
                    lineage_base = t.split(":", 1)[1]
                    break
        if isinstance(lineage_base, list):
            lineage_base = lineage_base[0] if lineage_base else ""
        family = _infer_one_family(mid)
        pipeline_tag = d.get("pipeline_tag") or ""

        # primary-metric model-index records (same whitelist as phase1)
        n_primary = 0
        for entry in (d.get("model-index") or []):
            if not isinstance(entry, dict):
                continue
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
                    if mcanon is None:
                        continue
                    cv, rej = canon_value(met.get("value"), mcanon)
                    if rej is not None:
                        continue
                    n_primary += 1
                    ck = canon_key(ds_type, cfg, ds_name)
                    dc = alias.get(ck, ck)
                    cfg_s = str(cfg).strip().lower()
                    # node granularity = what a dataset NODE in the graph is:
                    # canonical id + config, unless canon_key already embedded it
                    node = dc if ("/" in dc or not cfg_s) else f"{dc}/{cfg_s}"
                    obs.append({
                        "model_id": mid,
                        "dataset_canonical": dc,
                        "dataset_node": node,
                        "dataset_config": cfg_s,
                        "task": str(task_type).strip().lower(),
                        "metric_canonical": mcanon,
                        "metric_direction": "higher" if hib else "lower",
                        "split": str(split).strip().lower(),
                        "value_canonical": cv,
                        "verified": bool(met.get("verified", False)),
                    })
        rows.append({
            "model_id": mid,
            "parse_ok": True,
            "disabled": bool(d.get("disabled", False)),
            "private": bool(d.get("private", False)),
            "gated": bool(d.get("gated", False)),
            "n_primary_records": n_primary,
            "has_model_index": n_primary > 0,
            "lineage_base": str(lineage_base),
            "has_lineage": bool(lineage_base),
            "family": family,
            "family_parsable": family != FAMILY_OTHER,
            "pipeline_tag": pipeline_tag,
            "downloads": int(d.get("downloads") or 0),
            "likes": int(d.get("likes") or 0),
        })
    dfm = pd.DataFrame(rows)
    dfo = pd.DataFrame(obs)
    return n_files, dfm, dfo


def dataset_cache_index():
    """canon_key(id) -> hf dataset id, for every cached dataset JSON."""
    idx = {}
    for path in sorted(glob.glob(os.path.join(CACHE_D, "*.json"))) \
            + sorted(glob.glob(os.path.join(CACHE_D2, "*.json"))):
        d = _load(path)
        if d is None:
            continue
        did = d.get("id") or os.path.basename(path)[:-5].replace("__", "/")
        idx.setdefault(canon_key(did), did)
    return idx


def main():
    os.makedirs(OUT, exist_ok=True)
    alias = _alias_map()
    n_files, dfm, dfo_raw = scan_models()

    ok = dfm[dfm["parse_ok"] == True].copy()  # noqa: E712
    eligible = ok[(~ok["disabled"]) & (~ok["private"])].copy()
    eligible["intake"] = (eligible["has_model_index"] | eligible["has_lineage"]
                          | eligible["family_parsable"])
    eligible["intake_strict"] = eligible["has_model_index"] | eligible["has_lineage"]
    intake = eligible[eligible["intake"]].copy()

    # ---- observations: restrict to intake models, dedup ----------------------
    intake_ids = set(intake["model_id"])
    dfo = dfo_raw[dfo_raw["model_id"].isin(intake_ids)].copy()
    # drop junk nodes: model-index entries whose dataset name canonicalises to
    # nothing ("" / ".") -- unnameable, unembeddable, must never count as gold
    n_unnamed = int(dfo["dataset_canonical"].isin(["", "."]).sum())
    dfo = dfo[~dfo["dataset_canonical"].isin(["", "."])]
    n_raw_obs = len(dfo)
    # canonical: one row per (node, model, metric): median over runs/splits
    dfo_m = (dfo.groupby(["dataset_canonical", "dataset_node", "model_id",
                          "metric_canonical", "metric_direction"], as_index=False)
                .agg(value=("value_canonical", "median"),
                     n_runs=("value_canonical", "size"),
                     task=("task", "first")))
    # pair-level (the trained_on granularity): one row per (node, model)
    n_distinct_pairs = dfo_m.drop_duplicates(["dataset_node", "model_id"]).shape[0]

    # ---- gold projection (both granularities) --------------------------------
    # single-metric policy per key: bounded higher-is-better metric with max
    # model coverage (mirrors phase3 single-metric selection, coarse version)
    def gold_stats(key):
        b = dfo_m[dfo_m["metric_canonical"].isin(BOUNDED_HIGHER)]
        cov = (b.groupby([key, "metric_canonical"])["model_id"]
                .nunique().reset_index(name="n_models"))
        best = cov.sort_values([key, "n_models", "metric_canonical"],
                               ascending=[True, False, True]).drop_duplicates(key)
        sel = b.merge(best[[key, "metric_canonical"]], on=[key, "metric_canonical"])
        st = (sel.groupby(key)
                 .agg(chosen_metric=("metric_canonical", "first"),
                      n_models=("model_id", "nunique"),
                      value_std=("value", "std"),
                      value_nunique=("value", "nunique"),
                      dominant_task=("task", lambda s: s.mode().iloc[0] if len(s.mode()) else ""))
                 .reset_index())
        st["nonconstant"] = st["value_nunique"] > 1
        st["gold_evaluable"] = (st["n_models"] >= 3) & st["nonconstant"]
        return st.sort_values(["n_models", key], ascending=[False, True])

    dfo_m["dataset_root"] = dfo_m["dataset_canonical"].str.split("/").str[0]
    dstats = gold_stats("dataset_node")          # graph-node granularity (primary)
    dstats_ds = gold_stats("dataset_canonical")  # coarse dataset granularity
    dstats_root = gold_stats("dataset_root")     # strictest: root datasets

    dcache = dataset_cache_index()
    dcache_alias = {alias.get(k, k): v for k, v in dcache.items()}

    def cache_hit(node):
        if node in dcache_alias:
            return dcache_alias[node]
        root = node.split("/")[0]
        return dcache_alias.get(root, "")

    dstats["hf_id_candidate"] = dstats["dataset_node"].map(cache_hit)
    dstats["in_dataset_cache"] = dstats["hf_id_candidate"] != ""

    # ---- persist -------------------------------------------------------------
    dfm.to_csv(os.path.join(OUT, "d0_model_intake.csv"), index=False)
    try:
        dfo_m.to_parquet(os.path.join(OUT, "d0_observations.parquet"), index=False)
    except Exception:
        dfo_m.to_csv(os.path.join(OUT, "d0_observations.csv"), index=False)
    dstats.to_csv(os.path.join(OUT, "d0_dataset_pool.csv"), index=False)

    model_ids = sorted(intake["model_id"])
    blob = json.dumps({"model_ids": model_ids}, indent=1)
    open(os.path.join(OUT, "d0_intake_models.json"), "w", encoding="utf-8").write(blob)
    sha = hashlib.sha256(blob.encode()).hexdigest()
    open(os.path.join(OUT, "d0_intake_models.json.sha256"), "w").write(sha)

    ge = lambda k: int((dstats["n_models"] >= k).sum())  # noqa: E731
    gold = dstats[dstats["gold_evaluable"]]
    gold_ds = dstats_ds[dstats_ds["gold_evaluable"]]
    gold_root = dstats_root[dstats_root["gold_evaluable"]]
    dstats_root.to_csv(os.path.join(OUT, "d0_root_pool.csv"), index=False)
    funnel = {
        "cache_model_files": n_files,
        "parse_failures": int((~dfm["parse_ok"]).sum()),
        "disabled_or_private_excluded": int(len(ok) - len(eligible)),
        "channel_has_model_index": int(eligible["has_model_index"].sum()),
        "channel_has_lineage": int(eligible["has_lineage"].sum()),
        "channel_family_parsable": int(eligible["family_parsable"].sum()),
        "intake_union": int(eligible["intake"].sum()),
        "intake_strict_mi_or_lineage": int(eligible["intake_strict"].sum()),
        "intake_models_sha256": sha,
        "dropped_unnamed_dataset_obs": n_unnamed,
        "raw_primary_obs": n_raw_obs,
        "canonical_obs_dataset_model_metric": int(len(dfo_m)),
        "distinct_node_model_pairs": int(n_distinct_pairs),
        "distinct_canonical_datasets_any_label": int(dfo_m["dataset_canonical"].nunique()),
        "distinct_dataset_nodes_any_label": int(dfo_m["dataset_node"].nunique()),
        "nodes_ge1_bounded": ge(1), "nodes_ge3": ge(3),
        "nodes_ge5": ge(5), "nodes_ge10": ge(10), "nodes_ge20": ge(20),
        "gold_evaluable_nodes": int(len(gold)),
        "gold_evaluable_datasets_coarse": int(len(gold_ds)),
        "distinct_roots_any_label": int(dfo_m["dataset_root"].nunique()),
        "gold_evaluable_roots": int(len(gold_root)),
        "gold_evaluable_in_dataset_cache": int(gold["in_dataset_cache"].sum()),
        "gold_evaluable_missing_from_cache": int((~gold["in_dataset_cache"]).sum()),
        "dataset_cache_files": len(dcache),
        "targets": {"models_ge_10000": int(eligible["intake"].sum()) >= 10000,
                    "gold_nodes_ge_200": int(len(gold)) >= 200},
    }

    # ---- concentration caveats (phase1 lesson: bitext near-dup inflation) ----
    bit = (gold["dataset_node"].str.contains("bitext|tatoeba|flores|bucc|ntrex",
                                             case=False)
           | gold["dominant_task"].str.contains("bitext", case=False, na=False))
    roots = gold["dataset_node"].str.split("/").str[0]
    nb = gold[~bit]
    funnel["concentration"] = {
        "gold_bitext_like_nodes": int(bit.sum()),
        "gold_non_bitext_nodes": int((~bit).sum()),
        "gold_distinct_roots": int(roots.nunique()),
        "gold_non_bitext_distinct_roots": int(nb["dataset_node"]
                                              .str.split("/").str[0].nunique()),
        "gold_top_roots": roots.value_counts().head(8).to_dict(),
        "non_bitext_ge200": int((~bit).sum()) >= 200,
    }
    # ---- root-level task-category distribution -------------------------------
    funnel["roots_by_task"] = {
        "all_labelled": dstats_root["dominant_task"].value_counts().to_dict(),
        "gold": gold_root["dominant_task"].value_counts().to_dict(),
    }

    # ---- Phase-B harvest seed summary (present only after d0_online_harvest) --
    seed_raw_p = os.path.join(OUT, "d0_harvest_seed_raw.csv")
    seed_p = os.path.join(OUT, "d0_harvest_seed.csv")
    if os.path.exists(seed_p):
        sd = pd.read_csv(seed_p)
        if "root" not in sd.columns:
            sd["root"] = sd["canon"].astype(str).str.split("/").str[0]
        n_raw = (len(pd.read_csv(seed_raw_p)) if os.path.exists(seed_raw_p)
                 else int(len(sd)))
        r2 = (sd.groupby("root")["n_eval_bearing"].max() >= 2)
        funnel["harvest_seed"] = {
            "raw_candidate_dataset_ids": n_raw,
            "canonical_dataset_ids": int(sd["canon"].nunique()),
            "distinct_root_datasets": int(sd["root"].nunique()),
            "roots_ge2_eval_bearing_models": int(r2.sum()),
            "datasets_kept": int(sd["keep"].sum()),
        }

    # gold nodes lacking cached dataset metadata -> targeted top-up harvest list
    gold[~gold["in_dataset_cache"]][["dataset_node", "n_models", "chosen_metric",
                                     "dominant_task"]] \
        .to_csv(os.path.join(OUT, "d0_missing_dataset_metadata.csv"), index=False)
    json.dump(funnel, open(os.path.join(OUT, "d0_funnel.json"), "w"), indent=2)

    md = ["# D0 intake audit -- offline funnel (Phase A)", "",
          "| stage | count |", "|---|---|"]
    for k, v in funnel.items():
        if k in ("targets", "intake_models_sha256"):
            continue
        md.append(f"| {k} | {v} |")
    md += ["", f"targets: {funnel['targets']}",
           f"intake sha256: `{sha}`", ""]
    open(os.path.join(OUT, "d0_funnel.md"), "w", encoding="utf-8").write("\n".join(md))

    for k, v in funnel.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
