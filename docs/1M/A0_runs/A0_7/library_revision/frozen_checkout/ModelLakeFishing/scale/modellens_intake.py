"""
modellens_intake.py -- P1 step 1: turn the frozen ModelLens v2 corpus into the
three D0-contract artifacts the graph builder consumes, applying ALL SIX P0
corrections and recording per-row provenance.

Output node model:
    node  = (dataset, task)              -- a coherent retrieval target
    root  = dataset                      -- split unit (a): ModelLens's own name
                                            granularity, already STRICTER than
                                            their (dataset,task,metric) triple
    root2 = normalized(dataset)          -- split unit (b): our stricter root

The six P0 corrections (see docs/scale/P0/P0_EXECUTION.md §4):
  #4  @k-aware BOUNDED whitelist            -> base_metric() strips @<int>
  #3  per-(node,metric) value scale         -> group median > 1.5 => /100
  #5a "unknown" string == missing (size)    -> parsed as NaN, not a real bucket
  #5a "unknown" string == missing (family)  -> folded to Other later
  #5b popularity read from ["models"], ok   -> weak signal, carried not gated
  #6  query set = candidate depth >= 10     -> gold_evaluable flag

Provenance: every kept observation carries `csv_rowid`, the pandas logical row
index into data.csv (NOT the byte-line number -- data.csv has embedded newlines
in dataset_desp, so byte-lines 2,067,095 != logical rows 1,807,133).

Run (from ModelLakeFishing/):
    .\\.venv\\Scripts\\python.exe -m scale.modellens_intake
"""

import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

from scale.pull_corpus import data_root

RAW = os.path.join(data_root(), "modellens_v2", "raw")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "stage1BuildTransferGraph", "artifacts", "modellens_v2_lake")
CHUNK = 300_000
NODE_SEP = "␟"          # rare symbol-for-unit-separator; joins dataset+task
GOLD_MIN_DEPTH = 10          # P0 correction #6: gold@10-capable query set

# --- P0 correction #4: @k-aware bounded-metric rule -------------------------
AT_K = re.compile(r"@\d+$")
BOUNDED_BASE = {
    "accuracy", "accuracy_norm", "acc", "acc_norm", "f1", "micro_f1",
    "macro_f1", "weighted_f1", "exact_match", "em", "matthews_correlation",
    "mcc", "pearson", "spearman", "cosine_pearson", "cosine_spearman",
    "ndcg", "recall", "precision", "map", "mrr", "rouge1", "rouge2",
    "rougel", "rougelsum", "bleu", "chrf", "ap", "auc", "roc_auc",
    "v_measure", "ari", "nmi", "hit", "hit_rate", "success",
}


def base_metric(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().str.replace(AT_K, "", regex=True)


UNK = {"unknown", "", "none", "null", "nan"}


def parse_size_billions(v):
    """model_profile size is in BILLIONS of params, or the string 'unknown'.
    Returns float billions, or np.nan for missing/garbage (P0 correction #5a)."""
    if v is None:
        return np.nan
    s = str(v).strip().lower()
    if s in UNK:
        return np.nan
    try:
        f = float(s)
        return f if 0 < f < 1e5 else np.nan   # guard absurd values
    except ValueError:
        return np.nan


def norm_root(name: str) -> str:
    """Split unit (b): conservative dataset-name root. Strips parentheticals
    like '(trained on GoPro)' and normalizes punctuation, but does NOT merge
    size variants (CoDEx Small/Medium/Large stay distinct) -- conservative."""
    r = str(name).lower()
    r = re.sub(r"\(.*?\)", " ", r)
    r = re.sub(r"[^a-z0-9]+", " ", r)
    return " ".join(r.split()) or str(name).lower()


# --- P0 correction #3: per-group scale + garbage drop -----------------------
def normalize_scale(obs: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Per-node scale decision (0-1 vs 0-100) by the node's median value, then
    DROP rows still implausible for that scale (e.g. accuracy=2317) rather than
    clip them into a fake perfect score. Vectorized (no groupby.apply, which is
    fragile about the grouping column under pandas 3.0)."""
    med = obs.groupby("node")["value"].transform("median")
    obs = obs.copy()
    obs["scale"] = np.where(med > 1.5, 100.0, 1.0)
    vn = obs["value"] / obs["scale"]
    keep = vn.between(-1e-3, 1.0 + 1e-3)
    dropped = int((~keep).sum())
    obs = obs.loc[keep].copy()
    obs["value_norm"] = (obs["value"] / obs["scale"]).clip(0.0, 1.0)
    return obs, dropped


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    print(f"raw : {RAW}\nout : {OUT}\n")

    # ---- side JSONs (P0-corrected reads) --------------------------------
    with open(os.path.join(RAW, "model2id.json"), encoding="utf-8") as fh:
        model2id = json.load(fh)
    with open(os.path.join(RAW, "model_profile.json"), encoding="utf-8") as fh:
        profile = json.load(fh)
    with open(os.path.join(RAW, "model_popularity.json"), encoding="utf-8") as fh:
        pop_wrap = json.load(fh)
    pop_models = pop_wrap.get("models", {})              # P0 #5b: real payload
    with open(os.path.join(RAW, "family2id.json"), encoding="utf-8") as fh:
        family2id = json.load(fh)
    # ModelLens's CURATED family identity (332), matching their 332-row
    # family_embedding. model_profile.family is noisier (~700 distinct values
    # incl. junk like 'output'/'test'/'a'); admit only families in this
    # canonical set, fold the rest to Other. This aligns our family vocab with
    # theirs and keeps the learnable table clean.
    fam_allowed = {str(k).strip().lower() for k in family2id}

    all_models = set(model2id)                            # 47,242 canonical
    fam_of, size_of = {}, {}
    for m in all_models:
        p = profile.get(m)
        if isinstance(p, dict):
            f = str(p.get("family", "unknown")).strip().lower()
            fam_of[m] = f if (f not in UNK and f in fam_allowed) else ""
            size_of[m] = parse_size_billions(p.get("size"))
        else:
            fam_of[m], size_of[m] = "", np.nan

    def pop_of(m):
        r = pop_models.get(m)
        if isinstance(r, dict) and r.get("status") == "ok":
            return r.get("downloads", r.get("popularity"))
        return None

    # ---- stream CSV, keep only bounded rows, preserve logical rowid -----
    print("[pass A] streaming data.csv, filtering to @k-aware bounded rows ...")
    kept = []
    n_rows = 0
    metric_hist = Counter()
    offset = 0
    for ch in pd.read_csv(os.path.join(RAW, "data.csv"),
                          usecols=["task", "dataset", "model", "metric", "value"],
                          chunksize=CHUNK, low_memory=False):
        idx = np.arange(offset, offset + len(ch))
        offset += len(ch)
        n_rows += len(ch)
        ch = ch.assign(csv_rowid=idx)
        ch["value"] = pd.to_numeric(ch["value"], errors="coerce")
        mf = ch["metric"].astype(str).str.strip().str.lower()   # FULL metric name
        bm = base_metric(ch["metric"])                          # base, for membership
        metric_hist.update(bm.value_counts().to_dict())
        mask = bm.isin(BOUNDED_BASE) & ch["value"].notna()
        sub = ch.loc[mask].copy()
        sub["base_metric"] = bm.loc[mask]
        sub["metric_full"] = mf.loc[mask]
        sub["model"] = sub["model"].astype(str)
        sub["dataset"] = sub["dataset"].astype(str)
        sub["task"] = sub["task"].astype(str)
        # only models present in the canonical vocab (should be all)
        sub = sub[sub["model"].isin(all_models)]
        kept.append(sub)
    obs = pd.concat(kept, ignore_index=True)
    del kept
    print(f"  total rows {n_rows:,} | bounded+valid+known-model {len(obs):,} "
          f"({len(obs)/n_rows:.1%})")

    # ---- choose one metric per (dataset, task) node ---------------------
    # IMPORTANT: choose by the FULL metric name (e.g. "ndcg@10"), not the base.
    # The @k-aware base rule (P0 #4) governs BOUNDED membership only; using it as
    # the node metric would blend ndcg@1 and ndcg@10 into one target and create
    # spurious (node,model) duplicates. A node's retrieval target must be ONE
    # well-defined cutoff.
    obs["node"] = obs["dataset"] + NODE_SEP + obs["task"]
    node_metric_counts = (obs.groupby(["node", "metric_full"]).size()
                          .rename("c").reset_index())
    # deterministic pick: highest count, tie -> alphabetical metric name
    node_metric_counts = node_metric_counts.sort_values(
        ["node", "c", "metric_full"], ascending=[True, False, True])
    chosen = node_metric_counts.drop_duplicates("node").set_index("node")["metric_full"]
    obs["chosen_metric"] = obs["node"].map(chosen)
    obs = obs[obs["metric_full"] == obs["chosen_metric"]].copy()

    # sanity: after choosing one FULL metric, (node, model) is unique (probe
    # showed zero dups at (ds,task,model,metric)); assert so a future corpus
    # change that breaks it fails loudly rather than silently multiplying edges.
    dup = obs.duplicated(["node", "model"]).sum()
    assert dup == 0, f"{dup} duplicate (node,model) rows -- median-fold needed"

    # ---- P0 #3: per-(node,metric) scale + garbage drop ------------------
    before = len(obs)
    obs, dropped = normalize_scale(obs)
    print(f"[scale] normalized {before:,} obs, dropped {dropped:,} "
          f"({dropped/before:.2%}) implausible-for-scale rows")

    # ---- dataset pool: depth, gold flag, roots, chosen metric -----------
    depth = obs.groupby("node")["model"].nunique().rename("depth")
    node_dt = obs.drop_duplicates("node").set_index("node")[["dataset", "task"]]
    pool = node_dt.join(depth).join(chosen.rename("chosen_metric")).reset_index()
    pool["root_a"] = pool["dataset"]
    pool["root_b"] = pool["dataset"].map(norm_root)
    pool["gold_evaluable"] = pool["depth"] >= GOLD_MIN_DEPTH

    # ---- pass B: first non-empty dataset_desp per dataset ---------------
    print("[pass B] harvesting dataset_desp (first non-empty per dataset) ...")
    desc = {}
    need = set(pool["dataset"])
    for ch in pd.read_csv(os.path.join(RAW, "data.csv"),
                          usecols=["dataset", "dataset_desp"],
                          chunksize=CHUNK, low_memory=False):
        d = ch["dataset"].astype(str)
        s = ch["dataset_desp"]
        for dd, ss in zip(d, s):
            if dd in need and dd not in desc and isinstance(ss, str) and ss.strip():
                desc[dd] = ss.strip()[:2000]
    pool["desc"] = pool["dataset"].map(desc).fillna("")

    # ---- model intake: family/size/lineage ------------------------------
    used_models = sorted(obs["model"].unique())
    lineage_base = infer_lineage(all_models)
    mi = pd.DataFrame({"model_id": sorted(all_models)})
    mi["has_model_index"] = mi["model_id"].isin(set(used_models))
    mi["family"] = mi["model_id"].map(fam_of).fillna("")
    mi["size_b"] = mi["model_id"].map(size_of)
    mi["param_count"] = (mi["size_b"] * 1e9).where(mi["size_b"].notna())
    mi["popularity"] = mi["model_id"].map(pop_of)
    mi["lineage_base"] = mi["model_id"].map(lineage_base)
    mi["has_lineage"] = mi["lineage_base"].notna()

    # keep model intake to the STRICT pool: has an observation OR has lineage
    # (mirrors D0's strict rule); family-only models are dropped.
    strict = mi[mi["has_model_index"] | mi["has_lineage"]].copy()

    # ---- write artifacts ------------------------------------------------
    obs_out = obs[["node", "dataset", "task", "model", "chosen_metric",
                   "value", "value_norm", "scale", "csv_rowid"]].rename(
        columns={"node": "dataset_node", "model": "model_id",
                 "chosen_metric": "metric_canonical"})
    obs_out.to_parquet(os.path.join(OUT, "ml_observations.parquet"), index=False)
    pool = pool.rename(columns={"node": "dataset_node"})
    pool.to_csv(os.path.join(OUT, "ml_dataset_pool.csv"), index=False)
    strict.to_csv(os.path.join(OUT, "ml_model_intake.csv"), index=False)

    # ---- reconciliation report ------------------------------------------
    n_lin_edges = int((strict["lineage_base"].isin(set(strict["model_id"]))
                       & (strict["lineage_base"] != strict["model_id"])).sum())
    report = dict(
        corpus="modellens_v2",
        raw_provenance=os.path.join(data_root(), "modellens_v2", "PROVENANCE.json"),
        node_model="(dataset, task)",
        corrections_applied=[
            "#4 @k-aware BOUNDED", "#3 per-(node,metric) scale + garbage drop",
            "#5a size/family 'unknown'->missing", "#5b popularity from ['models'] ok",
            "#6 gold_evaluable depth>=10", "dual split units root_a/root_b"],
        counts=dict(
            csv_rows=n_rows,
            bounded_valid_obs_pre_scale=int(before),
            trained_on_edges=int(len(obs_out)),
            scale_dropped=int(dropped),
            nodes=int(pool.shape[0]),
            unique_datasets=int(pool["dataset"].nunique()),
            unique_roots_b=int(pool["root_b"].nunique()),
            models_canonical=len(all_models),
            models_strict=int(strict.shape[0]),
            models_with_obs=int(strict["has_model_index"].sum()),
            models_with_family=int((strict["family"] != "").sum()),
            models_with_size=int(strict["size_b"].notna().sum()),
            models_with_popularity=int(strict["popularity"].notna().sum()),
            lineage_base_assigned=int(strict["has_lineage"].sum()),
            lineage_edges_in_pool=n_lin_edges,
            gold_nodes_depth10=int(pool["gold_evaluable"].sum()),
            gold_nodes_depth30=int((pool["depth"] >= 30).sum()),
            gold_with_desc=int((pool["gold_evaluable"] & (pool["desc"] != "")).sum()),
            desc_coverage_nodes=int((pool["desc"] != "").sum()),
        ),
        chosen_metric_top=Counter(pool["chosen_metric"]).most_common(15),
        scale_split=dict(
            unit=int((obs_out["scale"] == 1.0).sum()),
            percent=int((obs_out["scale"] == 100.0).sum())),
        runtime_sec=round(time.time() - t0, 1),
    )
    with open(os.path.join(OUT, "ml_intake_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    print("\n=== INTAKE RECONCILIATION ===")
    for k, v in report["counts"].items():
        print(f"  {k:28s} {v:>12,}" if isinstance(v, int) else f"  {k:28s} {v}")
    print(f"\n  chosen-metric top: {report['chosen_metric_top'][:8]}")
    print(f"  scale split: {report['scale_split']}")
    print(f"  runtime: {report['runtime_sec']}s  ->  {OUT}")
    return 0


def infer_lineage(all_models: set) -> dict:
    """High-precision, name-based lineage: a derivative is the SAME name minus a
    format/quantization/adapter marker whose stripped form is itself a known
    model. Cross-user finetune chains are deliberately NOT inferred (low
    precision from names alone). Reports whatever it finds -- sparse is honest.

    Marker vocabulary follows the r_mm' ordering quantized > adapter > merge."""
    markers = [
        # quantization / serialization formats (strongest, most common)
        r"[-_.]?(q[2-8]_[0-9a-z_]+)$", r"[-_.]?(iq[0-9]+_[0-9a-z]+)$",
        r"[-_.]?(gguf|ggml|awq|gptq|exl2|mlx|bnb|nf4|fp16|fp8|bf16)$",
        r"[-_.]?(int4|int8|4bit|8bit|4-bit|8-bit)$",
        r"[-_.]?(q4|q5|q6|q8)$",
        # adapters
        r"[-_.]?(lora|qlora|adapter|peft)$",
    ]
    marker_re = [re.compile(m, re.I) for m in markers]
    base = {}
    for name in all_models:
        head, _, tail = name.rpartition("/")
        seg = tail if head else name
        stripped = seg
        changed = True
        # strip repeatedly (e.g. name-q4_k_m-gguf) but bounded
        for _ in range(4):
            if not changed:
                break
            changed = False
            for rgx in marker_re:
                new = rgx.sub("", stripped)
                if new != stripped and len(new) >= 3:
                    stripped = new
                    changed = True
        if stripped == seg:
            continue
        cand = f"{head}/{stripped}" if head else stripped
        if cand in all_models and cand != name:
            base[name] = cand
    return base


if __name__ == "__main__":
    sys.exit(main())
