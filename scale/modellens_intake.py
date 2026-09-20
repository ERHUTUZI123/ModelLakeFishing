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
NODE_SEP = "␟"
GOLD_MIN_DEPTH = 10

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
    if v is None:
        return np.nan
    s = str(v).strip().lower()
    if s in UNK:
        return np.nan
    try:
        f = float(s)
        return f if 0 < f < 1e5 else np.nan
    except ValueError:
        return np.nan


def norm_root(name: str) -> str:
    r = str(name).lower()
    r = re.sub(r"\(.*?\)", " ", r)
    r = re.sub(r"[^a-z0-9]+", " ", r)
    return " ".join(r.split()) or str(name).lower()


def normalize_scale(obs: pd.DataFrame) -> tuple[pd.DataFrame, int]:
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

    with open(os.path.join(RAW, "model2id.json"), encoding="utf-8") as fh:
        model2id = json.load(fh)
    with open(os.path.join(RAW, "model_profile.json"), encoding="utf-8") as fh:
        profile = json.load(fh)
    with open(os.path.join(RAW, "model_popularity.json"), encoding="utf-8") as fh:
        pop_wrap = json.load(fh)
    pop_models = pop_wrap.get("models", {})
    with open(os.path.join(RAW, "family2id.json"), encoding="utf-8") as fh:
        family2id = json.load(fh)
    fam_allowed = {str(k).strip().lower() for k in family2id}

    all_models = set(model2id)
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
        mf = ch["metric"].astype(str).str.strip().str.lower()
        bm = base_metric(ch["metric"])
        metric_hist.update(bm.value_counts().to_dict())
        mask = bm.isin(BOUNDED_BASE) & ch["value"].notna()
        sub = ch.loc[mask].copy()
        sub["base_metric"] = bm.loc[mask]
        sub["metric_full"] = mf.loc[mask]
        sub["model"] = sub["model"].astype(str)
        sub["dataset"] = sub["dataset"].astype(str)
        sub["task"] = sub["task"].astype(str)
        sub = sub[sub["model"].isin(all_models)]
        kept.append(sub)
    obs = pd.concat(kept, ignore_index=True)
    del kept
    print(f"  total rows {n_rows:,} | bounded+valid+known-model {len(obs):,} "
          f"({len(obs)/n_rows:.1%})")

    obs["node"] = obs["dataset"] + NODE_SEP + obs["task"]
    node_metric_counts = (obs.groupby(["node", "metric_full"]).size()
                          .rename("c").reset_index())
    node_metric_counts = node_metric_counts.sort_values(
        ["node", "c", "metric_full"], ascending=[True, False, True])
    chosen = node_metric_counts.drop_duplicates("node").set_index("node")["metric_full"]
    obs["chosen_metric"] = obs["node"].map(chosen)
    obs = obs[obs["metric_full"] == obs["chosen_metric"]].copy()

    dup = obs.duplicated(["node", "model"]).sum()
    assert dup == 0, f"{dup} duplicate (node,model) rows -- median-fold needed"

    before = len(obs)
    obs, dropped = normalize_scale(obs)
    print(f"[scale] normalized {before:,} obs, dropped {dropped:,} "
          f"({dropped/before:.2%}) implausible-for-scale rows")

    depth = obs.groupby("node")["model"].nunique().rename("depth")
    node_dt = obs.drop_duplicates("node").set_index("node")[["dataset", "task"]]
    pool = node_dt.join(depth).join(chosen.rename("chosen_metric")).reset_index()
    pool["root_a"] = pool["dataset"]
    pool["root_b"] = pool["dataset"].map(norm_root)
    pool["gold_evaluable"] = pool["depth"] >= GOLD_MIN_DEPTH

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

    strict = mi[mi["has_model_index"] | mi["has_lineage"]].copy()

    obs_out = obs[["node", "dataset", "task", "model", "chosen_metric",
                   "value", "value_norm", "scale", "csv_rowid"]].rename(
        columns={"node": "dataset_node", "model": "model_id",
                 "chosen_metric": "metric_canonical"})
    obs_out.to_parquet(os.path.join(OUT, "ml_observations.parquet"), index=False)
    pool = pool.rename(columns={"node": "dataset_node"})
    pool.to_csv(os.path.join(OUT, "ml_dataset_pool.csv"), index=False)
    strict.to_csv(os.path.join(OUT, "ml_model_intake.csv"), index=False)

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
    markers = [
        r"[-_.]?(q[2-8]_[0-9a-z_]+)$", r"[-_.]?(iq[0-9]+_[0-9a-z]+)$",
        r"[-_.]?(gguf|ggml|awq|gptq|exl2|mlx|bnb|nf4|fp16|fp8|bf16)$",
        r"[-_.]?(int4|int8|4bit|8bit|4-bit|8-bit)$",
        r"[-_.]?(q4|q5|q6|q8)$",
        r"[-_.]?(lora|qlora|adapter|peft)$",
    ]
    marker_re = [re.compile(m, re.I) for m in markers]
    base = {}
    for name in all_models:
        head, _, tail = name.rpartition("/")
        seg = tail if head else name
        stripped = seg
        changed = True
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
