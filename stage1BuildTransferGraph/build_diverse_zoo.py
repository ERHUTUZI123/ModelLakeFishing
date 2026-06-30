"""
build_diverse_zoo.py -- Phase 2 of the "diverse zoo" diagnostic experiment.

Goal
----
The local 177-model zoo is intrinsically low-dimensional: its REAL performance
records cover only ~9 near-duplicate datasets (5 of them tweet_eval/*), so the
perf-bearing dataset-similarity effective rank is ~2 and z_m participation ~1.8.
This script AUGMENTS the local zoo with REAL, attributable text-classification
performance records harvested from the HuggingFace Hub, spread across
domain-distinct task types and diverse encoder families, to test whether Stage 2
"opens up" with more intrinsic diversity.

Honesty contract (every guardrail the prompt set)
--------------------------------------------------
* Every accuracy comes from a model card's `model-index`
  (cardData['model-index']) -- self-reported, dataset-attributed, source URL
  recorded in the manifest. Nothing is fabricated.
* We only AUGMENT domain-distinct datasets that are perf-EMPTY locally
  (mnli/qnli/qqp/rte/wnli/imdb/emotion/trec/hate_speech/xnli/amazon/financial),
  NEVER the perf-rich originals (cola/sst2/ag_news/rotten_tomatoes/tweet_eval/*)
  -- so a dataset's trained_on edges never mix metric scales. One metric
  (accuracy) per augmented dataset.
* A record is kept only if its dataset already has a domain-similarity embedding
  (.npy) -- the 28 existing ones are reused, so the Stage-1 pipeline can build
  the dataset node and its similar_to edges with no new embedding generation.
* Lineage edges come from each model's DECLARED base_model; the referenced base
  encoders are added as hub nodes (no trained_on edge) -- real cross-family
  lineage (roberta-base/deberta-v3-base/... hubs), not synthetic.
* Output goes into a SEPARATE data_diverse/ tree (selected via MLF_DATA_DIR) so
  data/ and every original artifact are untouched. Raw API JSON is cached under
  data_diverse/hf_cache/.

Run
---
  cd stage1BuildTransferGraph
  ../.venv/Scripts/python.exe build_diverse_zoo.py            # harvest + write CSVs
  ../.venv/Scripts/python.exe build_diverse_zoo.py --offline  # rebuild from cache only
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import pandas as pd
import requests
from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_embed.utils.fetch_metadata import _infer_one_family  # name-based family rule

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dataset_embed", "data")
DIVERSE = os.path.join(HERE, "dataset_embed", "data_diverse")
CACHE = os.path.join(DIVERSE, "hf_cache")
EMB_DIR = os.path.join(DATA, "embedded_dataset", "domain_similarity", "EleutherAI_gpt-neo-125m")

api = HfApi()

# ── augmentation targets ────────────────────────────────────────────────────
# our_name MUST normalize (replace '/'->'_') to an existing *_feature.npy stem.
# Each target: our dataset node name, a task label, and the HF dataset ids to
# query list_models with. We harvest ONLY accuracy (one metric / dataset).
AUGMENT = {
    "glue/mnli":  dict(task="NLI",            queries=["glue", "multi_nli"]),
    "glue/qnli":  dict(task="NLI/QA",         queries=["glue"]),
    "glue/qqp":   dict(task="paraphrase",     queries=["glue"]),
    "glue/rte":   dict(task="NLI",            queries=["glue"]),
    "glue/wnli":  dict(task="NLI/coref",      queries=["glue"]),
    "imdb":       dict(task="sentiment(long)",queries=["imdb"]),
    "dair-ai/emotion": dict(task="emotion",   queries=["emotion", "dair-ai/emotion"]),
    "trec":       dict(task="question-type",  queries=["trec", "CogComp/trec"]),
    "xnli_en":    dict(task="NLI(multiling)", queries=["xnli"]),
    "hate_speech":dict(task="toxicity/hate",  queries=["hate_speech", "hate_speech18", "ucberkeley-dlab/measuring-hate-speech"]),
    "mteb_amazon_reviews_multi_en": dict(task="review-sentiment", queries=["amazon_reviews_multi", "mteb/amazon_reviews_multi"]),
    "twitter-financial-news-sentiment": dict(task="finance-sentiment", queries=["zeroshot/twitter-financial-news-sentiment", "financial_phrasebank"]),
}

# selection caps (diversity over size)
PER_QUERY_FETCH   = 120   # candidate ids pulled per HF query (by downloads)
MAX_PER_DS        = 22    # keep at most this many models per augmented dataset
MAX_PER_DS_FAMILY = 4     # avoid near-duplicates: cap models per (dataset, family)
MIN_PER_DS        = 4     # drop a dataset that cannot reach this many models
MIN_DERIVS_FOR_HUB= 2     # a base model becomes a node only if >=2 derivatives kept


def embedded_stems():
    return {f[:-len("_feature.npy")] for f in os.listdir(EMB_DIR) if f.endswith("_feature.npy")}


def map_result_dataset(ds: dict):
    """Map a model-index result.dataset -> our dataset node name, or None.

    Only the perf-empty domain-distinct targets are accepted; perf-rich
    originals are deliberately rejected so edge metrics never mix."""
    t = str(ds.get("type", "")).lower()
    name = str(ds.get("name", "")).lower()
    args = str(ds.get("args", "") or ds.get("config", "")).lower()
    blob = " ".join([t, name, args])
    if "glue" in t or "glue" in name:
        if "mnli" in args or "mnli" in name: return "glue/mnli"
        if "qnli" in args or "qnli" in name: return "glue/qnli"
        if "qqp"  in args or "qqp"  in name: return "glue/qqp"
        if "rte"  in args or "rte"  in name: return "glue/rte"
        if "wnli" in args or "wnli" in name: return "glue/wnli"
        return None  # cola/sst2/mrpc/stsb -> not augmented
    if "multi_nli" in t or "multinli" in t or t == "mnli": return "glue/mnli"
    if t == "imdb" or name == "imdb": return "imdb"
    if "emotion" in t or "dair-ai/emotion" in t or t == "emotion": return "dair-ai/emotion"
    if t == "trec" or "trec" in name or "cogcomp/trec" in t: return "trec"
    if t == "xnli" or "xnli" in name: return "xnli_en"
    if "hate_speech" in t or "measuring-hate" in t or "hate_speech" in name: return "hate_speech"
    if "amazon_reviews_multi" in t or "amazon_reviews_multi" in name: return "mteb_amazon_reviews_multi_en"
    if "twitter-financial-news-sentiment" in t or "financial_phrasebank" in t or "financial" in blob and "sentiment" in blob:
        return "twitter-financial-news-sentiment"
    return None


def fetch_full(mid: str):
    """Raw /api/models/{id}?full=true (carries cardData['model-index']), cached."""
    safe = mid.replace("/", "__")
    p = os.path.join(CACHE, safe + ".json")
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    try:
        r = requests.get(f"https://huggingface.co/api/models/{mid}",
                         params={"full": "true"}, timeout=30)
        if r.status_code != 200:
            return None
        j = r.json()
    except Exception:
        return None
    os.makedirs(CACHE, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(j, f)
    time.sleep(0.05)
    return j


def extract_accuracy(j):
    """Return list of (our_dataset, accuracy_value) from this model's model-index
    (accuracy metric only; value coerced to 0..1)."""
    cd = j.get("cardData") or {}
    out = []
    for entry in (cd.get("model-index") or []):
        for r in entry.get("results", []):
            our = map_result_dataset(r.get("dataset", {}) or {})
            if not our:
                continue
            for m in r.get("metrics", []):
                if str(m.get("type", "")).lower() != "accuracy":
                    continue
                v = m.get("value")
                try:
                    v = float(v)
                except Exception:
                    continue
                if v > 1.0:           # some cards report percentages
                    v /= 100.0
                if 0.0 < v <= 1.0:
                    out.append((our, v))
    return out


def extract_base_model(j):
    cd = j.get("cardData") or {}
    bm = cd.get("base_model")
    if isinstance(bm, list):
        bm = bm[0] if bm else None
    if isinstance(bm, str) and "/" in bm or (isinstance(bm, str) and bm):
        return bm
    for t in (j.get("tags") or []):
        if isinstance(t, str) and t.startswith("base_model:"):
            cand = t.split(":", 1)[1]
            cand = cand.split(":")[-1]           # base_model:finetune:roberta-base
            if cand:
                return cand
    return None


def model_meta(j):
    cfg = j.get("config") or {}
    arch = cfg.get("architectures")
    if isinstance(arch, list):
        arch = arch[0] if arch else None
    st = j.get("safetensors") or {}
    params = st.get("total")
    nl = cfg.get("num_labels")
    if nl is None:
        id2 = cfg.get("id2label") or {}
        nl = len(id2) if id2 else None
    return arch, cfg.get("model_type"), params, nl


def harvest(offline: bool):
    os.makedirs(CACHE, exist_ok=True)
    original_models = set(pd.read_csv(os.path.join(DATA, "model_config_dataset.csv"))["model"])
    stems = embedded_stems()

    # 1) gather candidate ids per augmented dataset
    candidates = defaultdict(list)   # our_dataset -> [model_id,...]
    if not offline:
        for our, spec in AUGMENT.items():
            assert our.replace("/", "_") in stems, f"{our} has no embedding"
            seen = set()
            for q in spec["queries"]:
                try:
                    ms = list(api.list_models(filter="text-classification",
                                              trained_dataset=q, sort="downloads",
                                              limit=PER_QUERY_FETCH))
                except Exception as e:
                    print(f"  list_models({q}) failed: {e}")
                    continue
                for m in ms:
                    if m.id not in seen:
                        seen.add(m.id)
                        candidates[our].append(m.id)
            print(f"[query] {our:34s} {len(candidates[our])} candidate ids")
        with open(os.path.join(CACHE, "_candidates.json"), "w") as f:
            json.dump({k: v for k, v in candidates.items()}, f)
    else:
        with open(os.path.join(CACHE, "_candidates.json")) as f:
            candidates = defaultdict(list, json.load(f))

    # 2) fetch + parse each unique candidate, collect valid (model, dataset, acc)
    all_ids = sorted({mid for ids in candidates.values() for mid in ids})
    print(f"\n[fetch] {len(all_ids)} unique candidate models "
          f"({'cache only' if offline else 'network+cache'})")
    rows = []          # dict per (model,dataset) accuracy record
    meta_by_model = {}
    for i, mid in enumerate(all_ids):
        if mid in original_models:
            continue
        j = fetch_full(mid)
        if not j:
            continue
        accs = extract_accuracy(j)
        if not accs:
            continue
        arch, mtype, params, nl = model_meta(j)
        base = extract_base_model(j)
        fam = _infer_one_family(mid)
        meta_by_model[mid] = dict(arch=arch, model_type=mtype, params=params,
                                  num_labels=nl, base_model=base, family=fam)
        # keep the best accuracy per (model, dataset)
        best = {}
        for our, v in accs:
            if our not in best or v > best[our]:
                best[our] = v
        for our, v in best.items():
            rows.append(dict(model=mid, dataset=our, accuracy=v, family=fam,
                             arch=arch, model_type=mtype, params=params,
                             num_labels=nl, base_model=base))
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(all_ids)} scanned, {len(rows)} records so far")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no records harvested -- check network/cache")
    print(f"\n[harvest] {len(df)} raw (model,dataset) records across "
          f"{df['model'].nunique()} models, {df['dataset'].nunique()} datasets")

    # 3) balance: cap per (dataset, family) then per dataset; drop thin datasets
    kept = []
    for our, sub in df.groupby("dataset"):
        sub = sub.sort_values("accuracy", ascending=False)
        per_fam = defaultdict(int)
        chosen = []
        for _, r in sub.iterrows():
            if per_fam[r["family"]] >= MAX_PER_DS_FAMILY:
                continue
            per_fam[r["family"]] += 1
            chosen.append(r)
            if len(chosen) >= MAX_PER_DS:
                break
        if len(chosen) >= MIN_PER_DS:
            kept.extend(chosen)
        else:
            print(f"  drop {our}: only {len(chosen)} models (< {MIN_PER_DS})")
    kept = pd.DataFrame(kept)
    print(f"[balance] kept {len(kept)} records / {kept['model'].nunique()} models / "
          f"{kept['dataset'].nunique()} datasets / {kept['family'].nunique()} families")

    return kept, meta_by_model, original_models


def add_base_hubs(kept, meta_by_model):
    """Promote declared base_models with >=MIN_DERIVS_FOR_HUB kept derivatives to
    hub nodes (no trained_on edge). Returns (hub_rows_df, lineage_df)."""
    deriv_models = set(kept["model"])
    base_count = defaultdict(int)
    lineage = []
    for mid in deriv_models:
        base = meta_by_model[mid]["base_model"]
        if base and base != mid:
            base_count[base] += 1
    hubs = {b for b, c in base_count.items() if c >= MIN_DERIVS_FOR_HUB}
    # fetch metadata for hub base models (cache)
    hub_rows = []
    for b in sorted(hubs):
        j = fetch_full(b)
        if not j:
            continue
        arch, mtype, params, nl = model_meta(j)
        hub_rows.append(dict(model=b, family=_infer_one_family(b), arch=arch,
                             model_type=mtype, params=params, num_labels=nl))
    hub_ids = {h["model"] for h in hub_rows}
    for mid in deriv_models:
        base = meta_by_model[mid]["base_model"]
        if base in hub_ids:
            lineage.append(dict(model=mid, relation="finetune", base_model=base))
    print(f"[lineage] {len(hub_ids)} base hubs, {len(lineage)} is_base_of edges "
          f"across families: {sorted({h['family'] for h in hub_rows})}")
    return pd.DataFrame(hub_rows), pd.DataFrame(lineage)


def write_csvs(kept, hub_rows, lineage_new, meta_by_model):
    # ── model_config_dataset.csv (append derivatives + hubs) ──────────────────
    mc = pd.read_csv(os.path.join(DATA, "model_config_dataset.csv"))
    start = len(mc)
    new_cfg = []

    def cfg_row(idx, model, arch, mtype, params, nl, dataset, accuracy):
        return {"Unnamed: 0": idx, "model": model,
                "architectures": f"['{arch}']" if arch else None,
                "number_of_labels": nl, "labels": None, "model_type": mtype,
                "number_of_parameters": params, "memory_consumption": None,
                "dataset": dataset, "accuracy": accuracy}

    i = start
    # derivatives: one config row per (model,dataset) with the harvested accuracy
    for _, r in kept.iterrows():
        new_cfg.append(cfg_row(i, r["model"], r["arch"], r["model_type"],
                               r["params"], r["num_labels"], r["dataset"], r["accuracy"]))
        i += 1
    # hubs: a node with NO dataset (pure lineage hub) -> no trained_on edge
    for _, h in hub_rows.iterrows():
        new_cfg.append(cfg_row(i, h["model"], h["arch"], h["model_type"],
                               h["params"], h["num_labels"], None, None))
        i += 1
    mc_out = pd.concat([mc, pd.DataFrame(new_cfg)], ignore_index=True)
    mc_out.to_csv(os.path.join(DIVERSE, "model_config_dataset.csv"), index=False)
    print(f"[write] model_config_dataset.csv: {start} -> {len(mc_out)} rows")

    # ── lineage_records.csv (append real base_model edges) ────────────────────
    lin = pd.read_csv(os.path.join(DATA, "lineage_records.csv"))
    if not lineage_new.empty:
        lin_out = pd.concat([lin, lineage_new[["model", "relation", "base_model"]]],
                            ignore_index=True).drop_duplicates()
    else:
        lin_out = lin
    lin_out.to_csv(os.path.join(DIVERSE, "lineage_records.csv"), index=False)
    print(f"[write] lineage_records.csv: {len(lin)} -> {len(lin_out)} rows")
    return mc_out


def write_manifest(kept, hub_rows, original_models):
    rows = []
    for _, r in kept.iterrows():
        rows.append(dict(model=r["model"], role="derivative", family=r["family"],
                         architecture=r["arch"], model_type=r["model_type"],
                         params=r["params"], dataset=r["dataset"],
                         task=AUGMENT[r["dataset"]]["task"], metric="accuracy",
                         metric_value=round(float(r["accuracy"]), 4),
                         base_model=r["base_model"],
                         source_url=f"https://huggingface.co/{r['model']}",
                         reason=f"HF model-index accuracy on {r['dataset']} "
                                f"({AUGMENT[r['dataset']]['task']}); family {r['family']}"))
    for _, h in hub_rows.iterrows():
        rows.append(dict(model=h["model"], role="lineage_hub", family=h["family"],
                         architecture=h["arch"], model_type=h["model_type"],
                         params=h["params"], dataset=None, task="(base encoder)",
                         metric=None, metric_value=None, base_model=None,
                         source_url=f"https://huggingface.co/{h['model']}",
                         reason="declared base_model of >=2 harvested derivatives; "
                                "added as a real cross-family lineage hub (no trained_on edge)"))
    man = pd.DataFrame(rows)
    csv_p = os.path.join(HERE, "diverse_zoo_manifest.csv")
    json_p = os.path.join(HERE, "diverse_zoo_manifest.json")
    man.to_csv(csv_p, index=False)
    # dataset-level summary for the json
    ds_summary = (kept.groupby("dataset")
                  .agg(n_models=("model", "nunique"),
                       n_families=("family", "nunique"),
                       acc_min=("accuracy", "min"), acc_max=("accuracy", "max"))
                  .reset_index())
    ds_summary["task"] = ds_summary["dataset"].map(lambda d: AUGMENT[d]["task"])
    payload = {
        "description": "Diverse-zoo augmentation manifest. Harvested HF "
                       "text-classification models with model-index accuracy on "
                       "domain-distinct datasets, plus their declared base_model "
                       "lineage hubs. Augments (does not replace) the 177-model "
                       "local zoo; originals untouched.",
        "n_added_derivatives": int(kept["model"].nunique()),
        "n_added_hubs": int(hub_rows["model"].nunique()) if not hub_rows.empty else 0,
        "n_augmented_datasets": int(kept["dataset"].nunique()),
        "families_added": sorted(set(kept["family"]) | set(hub_rows["family"] if not hub_rows.empty else [])),
        "datasets": ds_summary.to_dict(orient="records"),
        "models": rows,
    }
    with open(json_p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[write] manifest: {csv_p}\n              {json_p}")
    print("\n[per-dataset summary]")
    print(ds_summary.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="rebuild from cached JSON only")
    args = ap.parse_args()
    kept, meta_by_model, original_models = harvest(args.offline)
    hub_rows, lineage_new = add_base_hubs(kept, meta_by_model)
    write_csvs(kept, hub_rows, lineage_new, meta_by_model)
    write_manifest(kept, hub_rows, original_models)
    print("\nDONE. Build the diverse graph with:")
    print('  MLF_DATA_DIR=.../dataset_embed/data_diverse '
          'python build_graph.py --contain_model_feature True --out hgraph_diverse_xm0.pt')


if __name__ == "__main__":
    main()
