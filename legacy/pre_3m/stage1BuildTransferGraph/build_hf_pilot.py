"""
build_hf_pilot.py -- Phases 1-3 of the HF diverse pilot zoo (300 models x 100 datasets).

METADATA ONLY. No large file downloads. Everything cached under
hf300m_100d/hf_cache/. Produces inventories + stratified selections + coverage
reports for the user to confirm BEFORE any graph build / training (Phases 4-6).

Phase 1: stratified-select 100 datasets across 8 task buckets.
Phase 2: stratified-select 300 models across 6 architecture buckets (BERT/RoBERTa <=50%).
Phase 3: extract self-reported performance edges (model-index, confidence=1) +
         coverage/gap report + a self-eval top-up plan (NOT run).

Run:
  cd stage1BuildTransferGraph
  ../.venv/Scripts/python.exe build_hf_pilot.py --phase all      # or datasets|models|edges
  ../.venv/Scripts/python.exe build_hf_pilot.py --phase all --offline   # cache only
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict, Counter

import pandas as pd
import requests
from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_embed.utils.fetch_metadata import _infer_one_family

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.join(HERE, "hf300m_100d")
DCACHE = os.path.join(PILOT, "hf_cache", "datasets")
MCACHE = os.path.join(PILOT, "hf_cache", "models")
api = HfApi()

# ── Phase 1: dataset task buckets (quota sums to 100) ─────────────────────────
DATASET_BUCKETS = [
    dict(name="text-classification", quota=20,
         tc=["text-classification"], search=["sentiment", "emotion", "topic", "toxic"]),
    dict(name="nli-paraphrase-sts", quota=15,
         tc=["sentence-similarity"], search=["nli", "natural language inference", "paraphrase", "semantic textual similarity"]),
    dict(name="qa-reading-comprehension", quota=10,
         tc=["question-answering"], search=["reading comprehension"]),
    dict(name="summarization", quota=10, tc=["summarization"], search=["summary"]),
    dict(name="translation-multilingual", quota=10, tc=["translation"], search=["multilingual translation"]),
    dict(name="token-classification-ner-pos", quota=10,
         tc=["token-classification"], search=["named entity recognition", "part of speech"]),
    dict(name="retrieval-ranking-similarity", quota=10,
         tc=["sentence-similarity", "text-retrieval"], search=["retrieval", "reranking"]),
    dict(name="domain-specific", quota=15, tc=[],
         search=["biomedical", "clinical", "finance", "financial", "legal", "code", "news", "social media"]),
]

# ── Phase 2: model architecture buckets (quota sums to 300) ───────────────────
ENCODER_TYPES = {"bert", "roberta", "distilbert", "deberta", "deberta-v2", "electra",
                 "albert", "modernbert", "camembert", "xlm-roberta", "mpnet", "fnet",
                 "mobilebert", "funnel", "xlnet", "ernie", "convbert"}
SEQ2SEQ_TYPES = {"t5", "mt5", "bart", "mbart", "pegasus", "marian", "led", "longt5", "bigbird_pegasus"}
DECODER_TYPES = {"gpt2", "gpt_neo", "gpt_neox", "opt", "bloom", "pythia", "gptj", "llama", "qwen2"}
MODEL_BUCKETS = [
    dict(name="encoder-classifiers", quota=120, pipeline=["text-classification"],
         search=["bert base", "roberta", "deberta", "electra", "albert", "distilbert",
                 "modernbert", "xlm-roberta classification", "finetuned sst2",
                 "finetuned emotion", "finetuned ag_news", "finetuned imdb"],
         want_types=ENCODER_TYPES),
    dict(name="sentence-transformers-embeddings", quota=40, pipeline=["sentence-similarity"],
         search=["sentence-transformers", "embedding"], want_types=None),
    dict(name="multilingual", quota=40, pipeline=["text-classification", "token-classification"],
         search=["multilingual", "xlm-roberta", "mbert", "labse"], want_types=None, multilingual=True),
    dict(name="seq2seq", quota=40, pipeline=["summarization", "translation", "text2text-generation"],
         search=[], want_types=SEQ2SEQ_TYPES),
    dict(name="domain-specific", quota=40, pipeline=["text-classification", "token-classification", "fill-mask"],
         search=["biobert", "clinicalbert", "pubmedbert", "scibert", "finbert", "legalbert", "legal-bert", "codebert"],
         want_types=None, domain=True),
    dict(name="small-causal-eval", quota=20, pipeline=["text-generation"],
         search=[], want_types=DECODER_TYPES),
]
BERTISH = {"BERT", "RoBERTa", "DistilBERT"}        # the <=50% cap family group
GLOBAL_BERTISH_CAP = 150                            # <= 50% of 300


# ── generic cached raw fetch ──────────────────────────────────────────────────
def _fetch(kind, rid, cache_dir):
    safe = rid.replace("/", "__")
    p = os.path.join(cache_dir, safe + ".json")
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    try:
        r = requests.get(f"https://huggingface.co/api/{kind}/{rid}", params={"full": "true"}, timeout=30)
        if r.status_code != 200:
            return None
        j = r.json()
    except Exception:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(j, f)
    time.sleep(0.04)
    return j


def _is_blocked(j):
    return bool(j.get("private")) or bool(j.get("disabled")) or (j.get("gated") not in (False, None))


def _name_stem(rid):
    repo = rid.split("/")[-1].lower()
    repo = re.sub(r"[-_.](v?\d+(\.\d+)*|gguf|gptq|awq|bnb|4bit|8bit|fp16|bf16|int8|q4|q8)$", "", repo)
    return repo


# ── Phase 1: datasets ─────────────────────────────────────────────────────────
def _parse_dataset(j):
    cd = j.get("cardData") or {}
    tc = cd.get("task_categories") or []
    if isinstance(tc, str):
        tc = [tc]
    tids = cd.get("task_ids") or []
    if isinstance(tids, str):
        tids = [tids]
    lang = cd.get("language") or []
    if isinstance(lang, str):
        lang = [lang]
    size = cd.get("size_categories")
    if isinstance(size, list):
        size = size[0] if size else None
    lic = cd.get("license")
    if isinstance(lic, list):
        lic = lic[0] if lic else None
    # dataset_info -> config, splits, label names, input schema
    di = cd.get("dataset_info")
    config = None; splits = []; label_names = []; input_fields = []
    if di:
        di0 = di[0] if isinstance(di, list) else di
        if isinstance(di0, dict):
            config = di0.get("config_name")
            splits = [s.get("name") for s in (di0.get("splits") or []) if isinstance(s, dict)]
            for f in (di0.get("features") or []):
                if not isinstance(f, dict):
                    continue
                if isinstance(f.get("names"), list):
                    if f.get("name", "").lower() in ("label", "labels", "class", "ner_tags", "intent"):
                        label_names = f["names"]
                    elif not label_names:
                        label_names = f["names"]
                else:
                    nm = f.get("name", "")
                    if nm and nm.lower() not in ("label", "labels", "idx", "id"):
                        input_fields.append(nm)
    n_text = len([f for f in input_fields if f.lower() not in ("id", "idx")])
    arity = "single" if n_text <= 1 else ("pair" if n_text == 2 else "multi")
    return dict(
        dataset_id=j.get("id") or j.get("_id"), config=config,
        task_categories=tc, task_ids=tids, language=lang,
        size=size, license=lic, label_names=label_names,
        input_schema=input_fields, arity=arity, splits=splits,
        gated=j.get("gated"), private=j.get("private"),
        downloads=j.get("downloads") or 0, likes=j.get("likes") or 0,
        has_card=bool((cd and (cd.get("pretty_name") or tc)) or j.get("description")),
    )


KNOWN_BENCHMARKS = {"nyu-mll/glue", "glue", "super_glue", "aps/super_glue", "squad", "rajpurkar/squad",
                    "xnli", "facebook/xnli", "conll2003", "eriktks/conll2003", "cnn_dailymail",
                    "abisee/cnn_dailymail", "wmt16", "mteb"}


def harvest_datasets(offline):
    os.makedirs(DCACHE, exist_ok=True)
    # 1) candidate ids per bucket
    cand = defaultdict(list)
    if not offline:
        for b in DATASET_BUCKETS:
            seen = set()
            for tc in b["tc"]:
                try:
                    for d in api.list_datasets(filter=f"task_categories:{tc}", sort="downloads", limit=70):
                        if d.id not in seen:
                            seen.add(d.id); cand[b["name"]].append(d.id)
                except Exception as e:
                    print(f"  list_datasets(tc={tc}) failed: {e}")
            for kw in b["search"]:
                try:
                    for d in api.list_datasets(search=kw, sort="downloads", limit=25):
                        if d.id not in seen:
                            seen.add(d.id); cand[b["name"]].append(d.id)
                except Exception as e:
                    print(f"  list_datasets(search={kw}) failed: {e}")
            print(f"[ds-query] {b['name']:32s} {len(cand[b['name']])} candidates")
        with open(os.path.join(DCACHE, "_cand.json"), "w") as f:
            json.dump(dict(cand), f)
    else:
        cand = defaultdict(list, json.load(open(os.path.join(DCACHE, "_cand.json"))))

    # 2) fetch + parse, build inventory (bucket = first bucket that proposed it)
    inv = {}
    bucket_of = {}
    for bname, ids in cand.items():
        for rid in ids:
            bucket_of.setdefault(rid, bname)
    allids = list(bucket_of.keys())
    print(f"[ds-fetch] {len(allids)} unique candidate datasets")
    for i, rid in enumerate(allids):
        j = _fetch("datasets", rid, DCACHE)
        if not j:
            continue
        rec = _parse_dataset(j)
        rec["bucket"] = bucket_of[rid]
        inv[rid] = rec
        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{len(allids)}")
    inv_df = pd.DataFrame(list(inv.values()))

    # 3) filter + stratified select per bucket with diversity caps
    def eligible(r):
        if r["gated"] not in (False, None) or r["private"]:
            return False
        if not r["has_card"] and r["dataset_id"] not in KNOWN_BENCHMARKS:
            return False
        return True

    selected = []
    seen_stem = set()
    reasons = {}
    bucket_fill = {}
    for b in DATASET_BUCKETS:
        sub = [r for r in inv.values() if r["bucket"] == b["name"] and eligible(r)]
        sub.sort(key=lambda r: -(r["downloads"] or 0))
        org_count = Counter()
        chosen = []
        for r in sub:
            stem = _name_stem(r["dataset_id"])
            org = r["dataset_id"].split("/")[0]
            if stem in seen_stem:               # dedup mirror
                continue
            if org_count[org] >= 4:             # no single org dominates a bucket
                continue
            seen_stem.add(stem); org_count[org] += 1
            why = []
            if r["splits"] and any(s in r["splits"] for s in ("validation", "test", "dev")):
                why.append("has val/test")
            if r["license"]:
                why.append(f"license={r['license']}")
            if r["label_names"]:
                why.append(f"{len(r['label_names'])} labels")
            if r["dataset_id"] in KNOWN_BENCHMARKS:
                why.append("benchmark")
            reasons[r["dataset_id"]] = f"{b['name']}; downloads={r['downloads']}; " + ", ".join(why)
            chosen.append(r)
            if len(chosen) >= b["quota"]:
                break
        bucket_fill[b["name"]] = (len(chosen), b["quota"])
        selected.extend(chosen)

    sel_df = pd.DataFrame(selected)
    if not sel_df.empty:
        sel_df["selection_reason"] = sel_df["dataset_id"].map(reasons)
        sel_df["source_url"] = "https://huggingface.co/datasets/" + sel_df["dataset_id"]
    # fetch card text for SELECTED only (cheap), cache
    _fetch_dataset_cards(sel_df, offline)

    _write_json_csv(inv_df, os.path.join(PILOT, "dataset_inventory"))
    _write_json_csv(sel_df, os.path.join(PILOT, "selected_100_datasets"))
    _dataset_coverage_report(inv_df, sel_df, bucket_fill)
    print(f"[ds] inventory {len(inv_df)} | selected {len(sel_df)}")
    return sel_df


def _fetch_dataset_cards(sel_df, offline):
    if sel_df.empty:
        return
    path = os.path.join(PILOT, "dataset_card_text.csv")
    have = {}
    if os.path.exists(path):
        prev = pd.read_csv(path)
        have = dict(zip(prev["dataset_id"].astype(str), prev["card_text"].fillna("").astype(str)))
    todo = [d for d in sel_df["dataset_id"] if d not in have]
    if todo and not offline:
        from huggingface_hub import DatasetCard
        for i, did in enumerate(todo):
            t = ""
            try:
                t = (DatasetCard.load(did, repo_type="dataset").text or "")[:1500]
            except Exception:
                t = ""
            have[did] = t
        pd.DataFrame({"dataset_id": list(have.keys()),
                      "card_text": list(have.values())}).to_csv(path, index=False)


# ── Phase 2: models ───────────────────────────────────────────────────────────
def _parse_model(j):
    cd = j.get("cardData") or {}
    cfg = j.get("config") or {}
    arch = cfg.get("architectures")
    if isinstance(arch, list):
        arch = arch[0] if arch else None
    st = j.get("safetensors") or {}
    params = st.get("total")
    lang = cd.get("language") or []
    if isinstance(lang, str):
        lang = [lang]
    lic = cd.get("license")
    if isinstance(lic, list):
        lic = lic[0] if lic else None
    base = cd.get("base_model")
    if isinstance(base, list):
        base = base[0] if base else None
    if not base:
        for t in (j.get("tags") or []):
            if isinstance(t, str) and t.startswith("base_model:"):
                base = t.split(":")[-1]; break
    mi = cd.get("model-index") or []
    rid = j.get("id") or j.get("modelId") or j.get("_id")
    return dict(
        model_id=rid, architecture=arch, model_type=cfg.get("model_type"),
        family=_infer_one_family(rid or ""), base_model=base,
        pipeline_tag=j.get("pipeline_tag"), tags=j.get("tags") or [],
        language=lang, license=lic, parameter_count=params,
        downloads=j.get("downloads") or 0, likes=j.get("likes") or 0,
        has_model_index=bool(mi), gated=j.get("gated"), private=j.get("private"),
        n_languages=len(lang),
    )


def harvest_models(offline):
    os.makedirs(MCACHE, exist_ok=True)
    cand = defaultdict(list)
    if not offline:
        for b in MODEL_BUCKETS:
            seen = set()
            for pt in b["pipeline"]:
                try:
                    for m in api.list_models(pipeline_tag=pt, sort="downloads", limit=150):
                        if m.id not in seen:
                            seen.add(m.id); cand[b["name"]].append(m.id)
                except Exception as e:
                    print(f"  list_models(pt={pt}) failed: {e}")
            for kw in b["search"]:
                try:
                    for m in api.list_models(search=kw, sort="downloads", limit=50):
                        if m.id not in seen:
                            seen.add(m.id); cand[b["name"]].append(m.id)
                except Exception as e:
                    print(f"  list_models(search={kw}) failed: {e}")
            print(f"[m-query] {b['name']:32s} {len(cand[b['name']])} candidates")
        with open(os.path.join(MCACHE, "_cand.json"), "w") as f:
            json.dump(dict(cand), f)
    else:
        cand = defaultdict(list, json.load(open(os.path.join(MCACHE, "_cand.json"))))

    bucket_of = {}
    for bname, ids in cand.items():
        for rid in ids:
            bucket_of.setdefault(rid, bname)
    allids = list(bucket_of.keys())
    print(f"[m-fetch] {len(allids)} unique candidate models")
    inv = {}
    for i, rid in enumerate(allids):
        j = _fetch("models", rid, MCACHE)
        if not j:
            continue
        if not (j.get("config") or {}).get("model_type"):     # skip unreadable config
            continue
        rec = _parse_model(j); rec["bucket"] = bucket_of[rid]
        inv[rid] = rec
        if (i + 1) % 150 == 0:
            print(f"  ...{i+1}/{len(allids)}")
    inv_df = pd.DataFrame(list(inv.values()))

    def eligible(r, b):
        if r["gated"] not in (False, None) or r["private"]:
            return False
        if b.get("want_types") and r["model_type"] not in b["want_types"]:
            return False
        if b.get("multilingual") and r["n_languages"] < 2 and "multilingual" not in [t.lower() for t in r["tags"]]:
            return False
        return True

    selected = []
    seen_stem = set()
    reasons = {}
    bertish = 0
    bucket_fill = {}
    for b in MODEL_BUCKETS:
        sub = [r for r in inv.values() if r["bucket"] == b["name"] and eligible(r, b)]
        # prefer models WITH model-index, then downloads
        sub.sort(key=lambda r: (-(1 if r["has_model_index"] else 0), -(r["downloads"] or 0)))
        fam_count = Counter()
        chosen = []
        for r in sub:
            stem = _name_stem(r["model_id"])
            if stem in seen_stem:
                continue
            if r["family"] in BERTISH and bertish >= GLOBAL_BERTISH_CAP:
                continue
            if fam_count[r["family"]] >= max(8, b["quota"] // 3):    # no single family floods a bucket
                continue
            seen_stem.add(stem); fam_count[r["family"]] += 1
            if r["family"] in BERTISH:
                bertish += 1
            why = []
            if r["has_model_index"]:
                why.append("model-index")
            if r["base_model"]:
                why.append("has base_model")
            if r["parameter_count"]:
                why.append(f"{r['parameter_count']/1e6:.0f}M params")
            reasons[r["model_id"]] = f"{b['name']}; {r['model_type']}; dl={r['downloads']}; " + ", ".join(why)
            chosen.append(r)
            if len(chosen) >= b["quota"]:
                break
        bucket_fill[b["name"]] = (len(chosen), b["quota"])
        selected.extend(chosen)

    sel_df = pd.DataFrame(selected)
    if not sel_df.empty:
        sel_df["selection_reason"] = sel_df["model_id"].map(reasons)
        sel_df["source_url"] = "https://huggingface.co/" + sel_df["model_id"]
    _write_json_csv(inv_df, os.path.join(PILOT, "model_inventory"))
    _write_json_csv(sel_df, os.path.join(PILOT, "selected_300_models"))
    _model_coverage_report(inv_df, sel_df, bucket_fill, bertish)
    print(f"[m] inventory {len(inv_df)} | selected {len(sel_df)} | bert-ish {bertish}")
    return sel_df


# ── Phase 3: performance edges from model-index ───────────────────────────────
_METRIC_KEEP = {"accuracy", "f1", "exact_match", "em", "rouge", "rouge1", "rougel",
                "bleu", "sacrebleu", "spearmanr", "pearson", "matthews_correlation",
                "map", "mrr", "ndcg", "recall", "precision"}


def _norm_ds(name):
    return re.sub(r"\s+", "", str(name).strip().lower()).replace("/", "_")


def harvest_edges(sel_models, sel_datasets, offline):
    sel_model_ids = set(sel_models["model_id"]) if sel_models is not None and not sel_models.empty else set()
    sel_ds_norm = {}
    if sel_datasets is not None and not sel_datasets.empty:
        for did in sel_datasets["dataset_id"]:
            sel_ds_norm[_norm_ds(did.split("/")[-1])] = did
            sel_ds_norm[_norm_ds(did)] = did
    rows = []
    for mid in sorted(sel_model_ids):
        j = _fetch("models", mid, MCACHE)
        if not j:
            continue
        cd = j.get("cardData") or {}
        for entry in (cd.get("model-index") or []):
            for res in entry.get("results", []):
                ds = res.get("dataset", {}) or {}
                ds_type = ds.get("type") or ds.get("name") or ""
                ds_cfg = ds.get("config") or ds.get("args") or ""
                task = (res.get("task", {}) or {}).get("type") or ""
                cand_norms = {_norm_ds(ds_type), _norm_ds(f"{ds_type}_{ds_cfg}") if ds_cfg else _norm_ds(ds_type),
                              _norm_ds(str(ds_type).split("/")[-1])}
                matched = next((sel_ds_norm[n] for n in cand_norms if n in sel_ds_norm), None)
                for m in res.get("metrics", []):
                    mt = str(m.get("type", "")).lower()
                    if mt not in _METRIC_KEEP:
                        continue
                    val = m.get("value")
                    try:
                        val = float(val)
                    except Exception:
                        continue
                    rows.append(dict(
                        model_id=mid, dataset_type=ds_type, dataset_config=ds_cfg,
                        matched_selected_dataset=matched, task_type=task,
                        metric_name=mt, metric_value=val,
                        in_selected_dataset=matched is not None,
                        confidence=1, source_name="hf_model_index",
                        source_url=f"https://huggingface.co/{mid}",
                    ))
    edf = pd.DataFrame(rows)
    # normalized score: per (matched_dataset, metric) min-max over SAME metric only; never cross-metric
    if not edf.empty:
        edf["norm_value"] = edf["metric_value"]
        for (ds, mt), idx in edf.dropna(subset=["matched_selected_dataset"]).groupby(
                ["matched_selected_dataset", "metric_name"]).groups.items():
            vals = edf.loc[idx, "metric_value"]
            lo, hi = vals.min(), vals.max()
            edf.loc[idx, "norm_value"] = 0.5 if hi - lo < 1e-9 else (vals - lo) / (hi - lo)
    _write_json_csv(edf, os.path.join(PILOT, "performance_edges"))
    _edge_coverage_report(edf, sel_models, sel_datasets)
    print(f"[edges] {len(edf)} raw model-index metrics "
          f"({int(edf['in_selected_dataset'].sum()) if not edf.empty else 0} into selected datasets)")
    return edf


# ── output helpers + reports ──────────────────────────────────────────────────
def _write_json_csv(df, stem):
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    df.to_csv(stem + ".csv", index=False)
    df.to_json(stem + ".json", orient="records", indent=1, force_ascii=False)


def _md(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  wrote {path}")


def _dataset_coverage_report(inv_df, sel_df, bucket_fill):
    lines = ["# Dataset coverage report (Phase 1)", ""]
    lines.append(f"Candidates inventoried: **{len(inv_df)}**; selected: **{len(sel_df)}** / 100 target.\n")
    lines.append("## Per-bucket fill (selected / quota)")
    for b in DATASET_BUCKETS:
        got, q = bucket_fill.get(b["name"], (0, b["quota"]))
        flag = "" if got >= q else f"  ⚠️ short by {q-got}"
        lines.append(f"- {b['name']}: {got}/{q}{flag}")
    if not sel_df.empty:
        lines.append("\n## Task-category coverage (selected)")
        tc = Counter(t for lst in sel_df["task_categories"] for t in (lst or []))
        for k, v in tc.most_common(20):
            lines.append(f"- {k}: {v}")
        lines.append("\n## Language coverage (selected, top 20)")
        lc = Counter(l for lst in sel_df["language"] for l in (lst or []))
        lines.append(", ".join(f"{k}:{v}" for k, v in lc.most_common(20)) or "(none recorded)")
        lines.append(f"\nDistinct languages: {len(lc)}")
        lines.append("\n## Arity / labels / splits")
        lines.append(f"- arity: {dict(Counter(sel_df['arity']))}")
        lines.append(f"- with label_names: {int((sel_df['label_names'].map(len)>0).sum())}/{len(sel_df)}")
        lines.append(f"- with val/test split: {int(sel_df['splits'].map(lambda s: any(x in (s or []) for x in ('validation','test','dev'))).sum())}/{len(sel_df)}")
        lines.append(f"- with license: {int(sel_df['license'].notna().sum())}/{len(sel_df)}")
        lines.append("\n## Filters applied")
        lines.append("- skipped gated/private/disabled; skipped no-card (unless known benchmark); "
                     "deduped mirrors by name-stem; capped <=4 per org per bucket.")
    _md(os.path.join(PILOT, "dataset_coverage_report.md"), "\n".join(lines))


def _model_coverage_report(inv_df, sel_df, bucket_fill, bertish):
    lines = ["# Model coverage report (Phase 2)", ""]
    lines.append(f"Candidates inventoried: **{len(inv_df)}**; selected: **{len(sel_df)}** / 300 target.\n")
    lines.append("## Per-bucket fill (selected / quota)")
    for b in MODEL_BUCKETS:
        got, q = bucket_fill.get(b["name"], (0, b["quota"]))
        flag = "" if got >= q else f"  ⚠️ short by {q-got}"
        lines.append(f"- {b['name']}: {got}/{q}{flag}")
    if not sel_df.empty:
        n = len(sel_df)
        lines.append(f"\n## Architecture / family coverage")
        lines.append(f"- BERT/RoBERTa/DistilBERT share: {bertish}/{n} = {100*bertish/n:.0f}% (cap 50%)")
        lines.append(f"- distinct families: {sel_df['family'].nunique()}; distinct model_type: {sel_df['model_type'].nunique()}")
        lines.append("- model_type histogram (top 20):")
        for k, v in Counter(sel_df["model_type"]).most_common(20):
            lines.append(f"  - {k}: {v}")
        lines.append("\n## Other coverage")
        lines.append(f"- with model-index: {int(sel_df['has_model_index'].sum())}/{n}")
        lines.append(f"- with base_model (lineage): {int(sel_df['base_model'].notna().sum())}/{n}")
        lines.append(f"- unknown parameter_count: {int(sel_df['parameter_count'].isna().sum())}/{n} "
                     f"({100*sel_df['parameter_count'].isna().mean():.0f}%)")
        lc = Counter(l for lst in sel_df["language"] for l in (lst or []))
        lines.append(f"- distinct languages: {len(lc)}; multilingual(>=2 lang) models: "
                     f"{int((sel_df['n_languages']>=2).sum())}")
        lines.append("\n## Filters applied")
        lines.append("- skipped gated/private + unreadable config; deduped near-dup checkpoints by name-stem; "
                     "BERT/RoBERTa/DistilBERT capped at 150 (<=50%); per-family-per-bucket cap; "
                     "preferred models WITH model-index.")
    _md(os.path.join(PILOT, "model_coverage_report.md"), "\n".join(lines))


def _edge_coverage_report(edf, sel_models, sel_datasets):
    lines = ["# Edge coverage report (Phase 3)", ""]
    nmodels = len(sel_models) if sel_models is not None else 0
    ndsets = len(sel_datasets) if sel_datasets is not None else 0
    if edf.empty:
        lines.append("No model-index metrics extracted.")
        _md(os.path.join(PILOT, "edge_coverage_report.md"), "\n".join(lines))
        return
    sel_edges = edf[edf["in_selected_dataset"]]
    lines.append(f"Raw model-index metrics: **{len(edf)}** (confidence=1, self-reported).")
    lines.append(f"Edges landing on a SELECTED dataset: **{len(sel_edges)}** "
                 f"(across {sel_edges['model_id'].nunique()} models, "
                 f"{sel_edges['matched_selected_dataset'].nunique()} datasets).\n")
    lines.append("## Metric-type breakdown (kept separate; NEVER mixed)")
    for k, v in Counter(edf["metric_name"]).most_common():
        lines.append(f"- {k}: {v}")
    # trained_on uses one metric class per dataset; show what would form edges
    em = sel_edges[sel_edges["metric_name"].isin(["accuracy", "f1", "matthews_correlation"])]
    lines.append("\n## trained_on coverage on selected datasets (accuracy/f1/matthews only)")
    if not em.empty:
        per_model = em.groupby("model_id")["matched_selected_dataset"].nunique()
        per_ds = em.groupby("matched_selected_dataset")["model_id"].nunique()
        import numpy as np
        lines.append(f"- usable edges: {len(em)}")
        lines.append(f"- edges/model: mean {per_model.mean():.2f}, median {per_model.median():.0f}, "
                     f"models with >=3: {(per_model>=3).sum()}/{nmodels}")
        lines.append(f"- edges/dataset: mean {per_ds.mean():.2f}, median {per_ds.median():.0f}, "
                     f"datasets with >=10: {(per_ds>=10).sum()}/{ndsets}")
        zero_models = nmodels - per_model.shape[0]
        zero_ds = ndsets - per_ds.shape[0]
        lines.append(f"- **zero-edge models: {zero_models}/{nmodels}**; zero-edge datasets: {zero_ds}/{ndsets}")
        lines.append("\n## Datasets below 10 edges (top gaps)")
        gaps = per_ds[per_ds < 10].sort_values()
        for ds, c in list(gaps.items())[:30]:
            lines.append(f"- {ds}: {c}")
    else:
        lines.append("- none (no accuracy/f1/matthews edges onto selected datasets)")
    lines.append("\n## Normalization rule")
    lines.append("- `norm_value` = per-(dataset, metric) min-max; computed ONLY within the same metric "
                 "for the same dataset. No cross-dataset or cross-metric normalization. accuracy/F1/EM/"
                 "ROUGE/BLEU/etc. are stored separately with their metric_name; trained_on edges would be "
                 "built from ONE metric class per dataset.")
    lines.append("\n## Self-eval top-up plan (NOT run — awaiting confirmation)")
    lines.append("- pick 20-30 representative selected datasets that are edge-poor;")
    lines.append("- for each, evaluate 20-50 candidate selected models on a sampled validation/test subset "
                 "(e.g. 500-1000 examples);")
    lines.append("- store with confidence=3, metric_type + raw + normalized + source='self_eval';")
    lines.append("- compute estimate: ~25 datasets x ~35 models x ~800 ex x (encoder fwd ~ tens of ms/ex on CPU) "
                 "≈ 25*35*800*0.03s ≈ ~5.8 CPU-hours (parallelizable); seq2seq/QA cost more — bound to "
                 "classification first. Exact estimate produced before any run.")
    _md(os.path.join(PILOT, "edge_coverage_report.md"), "\n".join(lines))


# ── Phase 1 FIX: robust label/split/schema enrichment + quality-weighted v2 ───
# The /api/datasets/{id} JSON rarely embeds dataset_info, so v1 got label_names
# 0/100. Here we parse ClassLabel / Sequence(ClassLabel) names from (1) the cached
# /api cardData.dataset_info, (2) the dataset CARD YAML front-matter
# (DatasetCard.load(id).data), and (3) builder_info as a bounded last resort, then
# re-select with a quality weight. v1 outputs are never overwritten.

SCHEMA_TASKS = {"question-answering", "summarization", "translation", "text-generation",
                "text2text-generation", "sentence-similarity", "text-retrieval",
                "feature-extraction", "table-question-answering", "multiple-choice"}
CLASS_TASKS = {"text-classification", "token-classification", "zero-shot-classification"}


def _names_from(obj):
    """Pull ClassLabel names out of a feature spec in any of HF's serializations:
    dtype.class_label.names | {'dtype':'ClassLabel','names':..} | sequence.class_label.names
    | {'feature':{...}} nesting. names may be a list or an idx->name dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        # direct names
        nm = obj.get("names")
        if isinstance(nm, dict):
            return [nm[k] for k in sorted(nm, key=lambda x: int(x))]
        if isinstance(nm, list) and nm:
            return list(nm)
        # dtype: {class_label: {names: ...}}
        for key in ("class_label", "ClassLabel"):
            if key in obj:
                r = _names_from(obj[key])
                if r:
                    return r
        # nested wrappers
        for key in ("dtype", "sequence", "feature", "list"):
            if key in obj:
                r = _names_from(obj[key])
                if r:
                    return r
    return None


def _parse_card_dataset_info(di):
    """di = cardData['dataset_info'] (dict or list of configs). Returns
    (label_names, splits, input_fields)."""
    if not di:
        return [], [], []
    di0 = di[0] if isinstance(di, list) else di
    if not isinstance(di0, dict):
        return [], [], []
    labels, inputs = [], []
    for f in (di0.get("features") or []):
        if not isinstance(f, dict):
            continue
        nm = (f.get("name") or "").lower()
        names = _names_from(f)
        if names:
            if nm in ("label", "labels", "class", "ner_tags", "pos_tags", "intent", "tags") or not labels:
                labels = names
        else:
            if nm and nm not in ("label", "labels", "idx", "id"):
                inputs.append(f.get("name"))
    splits = [s.get("name") for s in (di0.get("splits") or []) if isinstance(s, dict)]
    return labels, splits, inputs


def enrich_dataset(rid, offline, use_builder=False):
    """label_names / splits / input schema for one dataset, trying api-cache ->
    card YAML -> builder_info. Cached in dataset_enrichment.json."""
    labels, splits, inputs, src = [], [], [], "none"
    # (1) cached /api cardData.dataset_info
    j = _fetch("datasets", rid, DCACHE)
    if j:
        di = (j.get("cardData") or {}).get("dataset_info")
        labels, splits, inputs = _parse_card_dataset_info(di)
        if labels or splits:
            src = "api_cardData"
    # (2) card YAML front-matter
    if (not labels) and (not offline):
        try:
            from huggingface_hub import DatasetCard
            dd = DatasetCard.load(rid, repo_type="dataset").data
            di = dd.to_dict().get("dataset_info") if hasattr(dd, "to_dict") else None
            l2, s2, i2 = _parse_card_dataset_info(di)
            if l2:
                labels = l2
            if s2 and not splits:
                splits = s2
            if i2 and not inputs:
                inputs = i2
            if l2 or s2:
                src = "card_yaml"
        except Exception:
            pass
    # (3) builder_info last resort (bounded; no remote code)
    if use_builder and (not labels) and (not offline):
        try:
            from datasets import load_dataset_builder
            b = load_dataset_builder(rid)
            feats = b.info.features
            for k, v in (feats or {}).items():
                names = getattr(v, "names", None) or getattr(getattr(v, "feature", None), "names", None)
                if names:
                    labels = list(names); src = "builder_info"; break
            if b.info.splits:
                splits = splits or list(b.info.splits.keys())
        except Exception:
            pass
    return dict(label_names=labels, splits=splits, input_fields=inputs, label_source=src)


def harvest_datasets_v2(offline, use_builder=False):
    """Quality-weighted re-selection -> selected_100_datasets_v2 (+ report + v1/v2 diff).
    Targets: (label_names OR explicit task schema) >=50, val/test >=60, license >=80,
    while preserving the 8 task buckets."""
    cand = defaultdict(list, json.load(open(os.path.join(DCACHE, "_cand.json"))))
    bucket_of = {}
    for bname, ids in cand.items():
        for rid in ids:
            bucket_of.setdefault(rid, bname)

    enr_path = os.path.join(PILOT, "dataset_enrichment.json")
    enr = json.load(open(enr_path)) if os.path.exists(enr_path) else {}

    # base recs for ALL candidates (cheap: cached /api JSON, no network)
    recs = {}
    per_bucket = defaultdict(list)
    for rid in bucket_of:
        j = _fetch("datasets", rid, DCACHE)
        if not j:
            continue
        base = _parse_dataset(j); base["bucket"] = bucket_of[rid]
        recs[rid] = base
        per_bucket[bucket_of[rid]].append(rid)

    # enrich only the top-K by downloads per bucket (bounds card-YAML fetches),
    # then merge enrichment over the weak v1 fields. Cached across runs.
    ENRICH_PER_BUCKET = 70
    to_enrich = []
    for bname, ids in per_bucket.items():
        ids.sort(key=lambda r: -(recs[r]["downloads"] or 0))
        to_enrich += ids[:ENRICH_PER_BUCKET]
    print(f"[ds-v2] enriching up to {len(to_enrich)} datasets (cached {len(enr)})")
    for i, rid in enumerate(to_enrich):
        if rid not in enr:
            enr[rid] = enrich_dataset(rid, offline, use_builder)
            if (i + 1) % 40 == 0:
                json.dump(enr, open(enr_path, "w"))
                print(f"  ...enriched {i+1}/{len(to_enrich)}")
    json.dump(enr, open(enr_path, "w"))
    for rid, base in recs.items():
        e = enr.get(rid) or {}
        base["label_names"] = e.get("label_names") or base.get("label_names") or []
        base["splits"] = e.get("splits") or base.get("splits") or []
        base["input_schema"] = e.get("input_fields") or base.get("input_schema") or []
        base["label_source"] = e.get("label_source", "none")

    def has_val_test(r):
        return any(s in (r["splits"] or []) for s in ("validation", "test", "dev", "valid"))

    def has_label_or_schema(r):
        tcs = set(r["task_categories"] or [])
        if r["label_names"]:
            return True
        if tcs & SCHEMA_TASKS or r["bucket"] in (
                "qa-reading-comprehension", "summarization", "translation-multilingual",
                "retrieval-ranking-similarity"):
            return bool(r["input_schema"]) or bool(tcs & SCHEMA_TASKS)
        return False

    def eligible(r):
        if r["gated"] not in (False, None) or r["private"]:
            return False
        if not r["has_card"] and r["dataset_id"] not in KNOWN_BENCHMARKS:
            return False
        return True

    import math
    def quality(r):
        return (100 * has_label_or_schema(r) + 60 * has_val_test(r)
                + 40 * (r["license"] is not None) + 3 * math.log10((r["downloads"] or 0) + 10))

    selected, seen_stem, reasons, bucket_fill = [], set(), {}, {}
    for b in DATASET_BUCKETS:
        sub = [r for r in recs.values() if r["bucket"] == b["name"] and eligible(r)]
        sub.sort(key=lambda r: -quality(r))
        org_count = Counter(); chosen = []
        for r in sub:
            stem = _name_stem(r["dataset_id"]); org = r["dataset_id"].split("/")[0]
            if stem in seen_stem or org_count[org] >= 4:
                continue
            seen_stem.add(stem); org_count[org] += 1
            tags = []
            if has_label_or_schema(r): tags.append("label/schema")
            if has_val_test(r): tags.append("val/test")
            if r["license"]: tags.append(f"lic={r['license']}")
            reasons[r["dataset_id"]] = f"{b['name']}; q={quality(r):.0f}; dl={r['downloads']}; " + ", ".join(tags)
            chosen.append(r)
            if len(chosen) >= b["quota"]:
                break
        bucket_fill[b["name"]] = (len(chosen), b["quota"])
        selected.extend(chosen)

    sel = pd.DataFrame(selected)
    sel["has_label_or_schema"] = sel.apply(lambda r: has_label_or_schema(r), axis=1)
    sel["has_val_test"] = sel.apply(lambda r: has_val_test(r), axis=1)
    sel["selection_reason"] = sel["dataset_id"].map(reasons)
    sel["source_url"] = "https://huggingface.co/datasets/" + sel["dataset_id"]
    _write_json_csv(sel, os.path.join(PILOT, "selected_100_datasets_v2"))
    _fetch_dataset_cards(sel, offline)  # also caches card text for v2 picks

    # report + diff (never overwrite v1)
    n = len(sel)
    lab = int(sel["has_label_or_schema"].sum()); vt = int(sel["has_val_test"].sum())
    lic = int(sel["license"].notna().sum()); reallabels = int((sel["label_names"].map(len) > 0).sum())
    lines = ["# Dataset coverage report v2 (quality-weighted re-selection)", ""]
    lines.append(f"Selected: **{n}**/100. Enrichment sources: "
                 f"{dict(Counter(sel['label_source']))}.\n")
    lines.append("## Targets")
    lines.append(f"- label_names OR explicit task schema: **{lab}/100** (target >=50) "
                 f"{'OK' if lab>=50 else 'SHORT'}   [real ClassLabel names: {reallabels}]")
    lines.append(f"- val/test split: **{vt}/100** (target >=60) {'OK' if vt>=60 else 'SHORT'}")
    lines.append(f"- license: **{lic}/100** (target >=80) {'OK' if lic>=80 else 'SHORT'}")
    lines.append("\n## Per-bucket fill")
    for b in DATASET_BUCKETS:
        got, q = bucket_fill[b["name"]]; lines.append(f"- {b['name']}: {got}/{q}")
    tc = Counter(t for lst in sel["task_categories"] for t in (lst or []))
    lines.append("\n## Task-category coverage (top 15)")
    for k, v in tc.most_common(15): lines.append(f"- {k}: {v}")
    lc = Counter(l for lst in sel["language"] for l in (lst or []))
    lines.append(f"\nDistinct languages: {len(lc)}; arity: {dict(Counter(sel['arity']))}")
    _md(os.path.join(PILOT, "dataset_coverage_report_v2.md"), "\n".join(lines))

    # v1 vs v2 diff
    v1 = pd.read_csv(os.path.join(PILOT, "selected_100_datasets.csv"))
    s1, s2 = set(v1["dataset_id"]), set(sel["dataset_id"])
    d = ["# Dataset selection v1 -> v2 diff", ""]
    d.append(f"v1: {len(s1)} | v2: {len(s2)} | kept: {len(s1&s2)} | "
             f"added in v2: {len(s2-s1)} | dropped from v1: {len(s1-s2)}\n")
    v1lab = int((v1['label_names'].map(lambda x: len(eval(x)) if isinstance(x,str) and x.startswith('[') else 0)>0).sum()) if 'label_names' in v1 else 0
    d.append(f"v1 label_names>0: {v1lab}/100  ->  v2 real ClassLabel names: {reallabels}/100, "
             f"label-or-schema: {lab}/100")
    d.append(f"v1 val/test: 33/100  ->  v2 val/test: {vt}/100")
    d.append("\n## Added in v2 (sample 40)")
    for x in sorted(s2 - s1)[:40]: d.append(f"- + {x}")
    d.append("\n## Dropped from v1 (sample 40)")
    for x in sorted(s1 - s2)[:40]: d.append(f"- - {x}")
    _md(os.path.join(PILOT, "dataset_selection_v1_v2_diff.md"), "\n".join(d))
    print(f"[ds-v2] selected {n} | label-or-schema {lab}/100 (real labels {reallabels}) | "
          f"val/test {vt}/100 | license {lic}/100")
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["all", "datasets", "datasets_v2", "models", "edges"])
    ap.add_argument("--use_builder", action="store_true",
                    help="(datasets_v2) allow builder_info fallback for label names")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    os.makedirs(PILOT, exist_ok=True)
    sel_d = sel_m = None
    if args.phase == "datasets_v2":
        harvest_datasets_v2(args.offline, use_builder=args.use_builder)
        print("\nDONE datasets v2. Review selected_100_datasets_v2 + dataset_coverage_report_v2.md "
              "+ dataset_selection_v1_v2_diff.md.")
        return
    if args.phase in ("all", "datasets"):
        sel_d = harvest_datasets(args.offline)
    if args.phase in ("all", "models"):
        sel_m = harvest_models(args.offline)
    if args.phase in ("all", "edges"):
        # prefer the quality-weighted v2 dataset selection when available
        v2p = os.path.join(PILOT, "selected_100_datasets_v2.csv")
        v1p = os.path.join(PILOT, "selected_100_datasets.csv")
        if sel_d is None and os.path.exists(v2p):
            sel_d = pd.read_csv(v2p); print("[edges] matching against selected_100_datasets_v2")
        elif sel_d is None and os.path.exists(v1p):
            sel_d = pd.read_csv(v1p)
        if sel_m is None and os.path.exists(os.path.join(PILOT, "selected_300_models.csv")):
            sel_m = pd.read_csv(os.path.join(PILOT, "selected_300_models.csv"))
        harvest_edges(sel_m, sel_d, args.offline)
    print("\nDONE Phases 1-3. Review inventories + coverage reports in hf300m_100d/ before Phase 4-6.")


if __name__ == "__main__":
    main()
