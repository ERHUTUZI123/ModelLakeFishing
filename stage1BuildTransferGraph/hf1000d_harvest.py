"""
hf1000d_harvest.py -- HF-scale pilot, DATASET-FIRST (Phases 1-3).

New scale: 1000 datasets x 2000 models. Strategy change vs the 300m pilot: the
PRIMARY model-acquisition path is dataset-neighborhood retrieval, NOT a global
"list models by architecture" sweep. For every selected dataset we build an alias
set and pull the models that (a) report it in model-index, (b) tag/mention it, or
(c) match its task/pipeline/domain -- then a 30% global-diversity supplement.

METADATA ONLY. No weight downloads. Everything cached under
hf1000d_2000m/hf_cache/. Real edges only (HF self-reported model-index;
confidence=1). No fabrication, no self-eval.

Phases (resumable; each writes to hf1000d_2000m/):
  --phase datasets : >=20k listed candidates -> enrich -> select 1000
  --phase models   : dataset-neighborhood retrieval -> select 2000
  --phase edges    : real model-index edges (single-metric A + multi-metric candidate B)
  --phase all      : datasets, models, edges in order

Run:
  cd stage1BuildTransferGraph
  ../.venv/Scripts/python.exe hf1000d_harvest.py --phase datasets
  ../.venv/Scripts/python.exe hf1000d_harvest.py --phase models
  ../.venv/Scripts/python.exe hf1000d_harvest.py --phase edges
"""

import argparse
import json
import math
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
from build_hf_pilot import (
    _parse_dataset, _parse_model, _name_stem, _write_json_csv, _md,
    _parse_card_dataset_info, KNOWN_BENCHMARKS, SCHEMA_TASKS,
)
from build_hf_pilot_v3 import (
    canon_key, _GLUE, _ALIAS, KEY_TO_HFID, BENCH_BUCKET, USABLE_METRICS,
    has_val_test, has_label_or_schema,
)

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.join(HERE, "hf1000d_2000m")
DCACHE = os.path.join(PILOT, "hf_cache", "datasets")
MCACHE = os.path.join(PILOT, "hf_cache", "models")
LCACHE = os.path.join(PILOT, "hf_cache", "list")     # cached list_* query results
api = HfApi()

N_DATASETS = 1000
N_MODELS = 2000
MULTI_METRICS = {"accuracy", "f1", "matthews_correlation", "exact_match", "em",
                 "rouge", "rouge1", "rouge2", "rougel", "rougelsum", "bleu", "sacrebleu",
                 "meteor", "spearmanr", "pearson", "ndcg", "map", "mrr", "recall", "precision"}

# ── dataset task buckets (quota sums to 1000) ─────────────────────────────────
DATASET_BUCKETS = [
    dict(name="text-classification", quota=180,
         tc=["text-classification", "zero-shot-classification"],
         search=["sentiment", "emotion", "topic classification", "toxicity", "hate speech",
                 "spam", "intent classification", "news classification", "review classification",
                 "stance", "sarcasm", "offensive"]),
    dict(name="nli-paraphrase-sts", quota=120,
         tc=["sentence-similarity"],
         search=["natural language inference", "nli", "paraphrase", "semantic textual similarity",
                 "sts", "entailment", "sentence pair", "duplicate question"]),
    dict(name="qa-reading-comprehension", quota=120,
         tc=["question-answering", "multiple-choice", "table-question-answering"],
         search=["reading comprehension", "question answering", "multiple choice", "open domain qa",
                 "commonsense reasoning", "boolq"]),
    dict(name="summarization", quota=100, tc=["summarization"],
         search=["summarization", "abstractive summary", "headline generation", "dialogue summary"]),
    dict(name="translation-multilingual", quota=100, tc=["translation"],
         search=["machine translation", "multilingual translation", "parallel corpus", "opus", "wmt"]),
    dict(name="token-classification-ner-pos", quota=100, tc=["token-classification"],
         search=["named entity recognition", "ner", "part of speech", "pos tagging",
                 "chunking", "slot filling"]),
    dict(name="retrieval-ranking-similarity", quota=120,
         tc=["sentence-similarity", "text-retrieval", "feature-extraction"],
         search=["retrieval", "reranking", "passage ranking", "embedding benchmark", "mteb",
                 "information retrieval", "semantic search"]),
    dict(name="domain-specific", quota=160, tc=[],
         search=["biomedical", "clinical", "medical", "pubmed", "finance", "financial",
                 "legal", "law", "contracts", "code", "programming", "github",
                 "social media", "twitter", "scientific", "news articles", "wikipedia"]),
]

# benchmark / task-suite seed names (force-listed in the pool as high-trainability anchors)
TASK_SUITE_SEARCH = [
    "glue", "super_glue", "superglue", "squad", "xnli", "anli", "paws", "conll",
    "cnn_dailymail", "xsum", "wmt", "opus", "massive", "banking77", "trec", "emotion",
    "ag_news", "dbpedia", "imdb", "yelp", "amazon reviews", "mteb", "tweet_eval",
    "financial_phrasebank", "pubmed", "bc5cdr", "ncbi disease", "legal", "scotus",
    "lex_glue", "code_search_net", "humaneval", "mbpp", "go_emotions", "snli", "mnli",
]

# per-task pipeline tags for dataset-neighborhood model retrieval (Phase 2)
TASK_TO_PIPELINES = {
    "text-classification": ["text-classification", "zero-shot-classification"],
    "nli-paraphrase-sts": ["text-classification", "sentence-similarity"],
    "qa-reading-comprehension": ["question-answering", "text2text-generation", "multiple-choice"],
    "summarization": ["summarization", "text2text-generation"],
    "translation-multilingual": ["translation", "text2text-generation"],
    "token-classification-ner-pos": ["token-classification"],
    "retrieval-ranking-similarity": ["sentence-similarity", "feature-extraction"],
    "domain-specific": ["text-classification", "fill-mask", "token-classification"],
}
DOMAIN_MODEL_SEARCH = {
    "finance": ["finbert", "financial bert", "finance model"],
    "biomed": ["biobert", "pubmedbert", "clinicalbert", "scibert", "biomedical"],
    "legal": ["legalbert", "legal-bert", "legal model", "caselaw"],
    "code": ["codebert", "graphcodebert", "codet5", "code model"],
}
GLOBAL_SUPPLEMENT_SEARCH = [
    "sentence-transformers", "multilingual", "xlm-roberta", "labse", "mpnet",
    "t5", "bart", "pegasus", "marian", "electra", "deberta", "albert", "modernbert",
    "bloom", "opt", "gpt2", "pythia", "qwen", "distilbert multilingual", "flan-t5",
]
ENCODER_TYPES = {"bert", "roberta", "distilbert", "deberta", "deberta-v2", "electra",
                 "albert", "modernbert", "camembert", "xlm-roberta", "mpnet", "fnet",
                 "mobilebert", "funnel", "xlnet", "ernie", "convbert"}
BERTISH_TYPES = {"bert", "roberta", "distilbert"}


# ── cached raw fetch / list ───────────────────────────────────────────────────
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
    time.sleep(0.02)
    return j


def _cached_list(tag, fn):
    """Cache a list_* query's id list (+ light fields) under LCACHE by `tag`."""
    os.makedirs(LCACHE, exist_ok=True)
    p = os.path.join(LCACHE, re.sub(r"[^a-zA-Z0-9._-]", "_", tag) + ".json")
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            pass
    try:
        out = fn()
    except Exception as e:
        out = {"_error": f"{type(e).__name__}: {str(e)[:80]}"}
    json.dump(out, open(p, "w", encoding="utf-8"))
    time.sleep(0.02)
    return out


def list_datasets_q(tag, **kw):
    def fn():
        return [{"id": d.id, "downloads": getattr(d, "downloads", 0) or 0}
                for d in api.list_datasets(**kw)]
    out = _cached_list("ds__" + tag, fn)
    return [] if isinstance(out, dict) else out


def list_models_q(tag, **kw):
    def fn():
        rows = []
        for m in api.list_models(**kw):
            rows.append({"id": m.id, "downloads": getattr(m, "downloads", 0) or 0,
                         "pipeline_tag": getattr(m, "pipeline_tag", None)})
        return rows
    out = _cached_list("m__" + tag, fn)
    return [] if isinstance(out, dict) else out


# ════════════════════════════════════════════════════════════════════════════
# PHASE 1 -- datasets
# ════════════════════════════════════════════════════════════════════════════
def _enrich(rid):
    """label_names / splits / input schema for one dataset from cached /api JSON
    then card YAML front-matter (no builder exec). Cached in dataset_enrichment.json."""
    labels, splits, inputs, src = [], [], [], "none"
    j = _fetch("datasets", rid, DCACHE)
    if j:
        di = (j.get("cardData") or {}).get("dataset_info")
        labels, splits, inputs = _parse_card_dataset_info(di)
        if labels or splits:
            src = "api_cardData"
    if not labels:
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
    return dict(label_names=labels, splits=splits, input_fields=inputs, label_source=src)


def _bootstrap_trainability():
    """Predicted edge count per canonical dataset key, from a bounded model-index
    harvest over (a) dataset-filter listings of benchmark anchors and (b) top
    pipeline models. Real metrics only. Caches model JSON (shared with Phase 2)."""
    train = defaultdict(set)
    anchor_ids = set()
    # (a) benchmark anchors: list models tagged with the dataset, fetch a bounded top slice
    anchors = sorted(set(KEY_TO_HFID.values()) | {
        "glue", "super_glue", "squad", "squad_v2", "conll2003", "imdb", "ag_news",
        "emotion", "xnli", "snli", "anli", "paws", "cnn_dailymail", "xsum", "banking77",
        "trec", "go_emotions", "tweet_eval", "financial_phrasebank", "samsum",
    })
    for ds in anchors:
        for m in list_models_q(f"datasetfilter__{ds}", filter=f"dataset:{ds}",
                               sort="downloads", limit=300):
            anchor_ids.add(m["id"])
    # (b) top pipeline models (broad)
    for pt in ("text-classification", "token-classification", "question-answering",
               "summarization", "translation", "sentence-similarity"):
        for m in list_models_q(f"pipe__{pt}", pipeline_tag=pt, sort="downloads", limit=300):
            anchor_ids.add(m["id"])
    print(f"[bootstrap] fetching model-index for {len(anchor_ids)} anchor models")
    for i, mid in enumerate(sorted(anchor_ids)):
        j = _fetch("models", mid, MCACHE)
        if not j:
            continue
        for entry in ((j.get("cardData") or {}).get("model-index") or []):
            for res in entry.get("results", []):
                ds = res.get("dataset", {}) or {}
                k = canon_key(ds.get("type") or ds.get("name"), ds.get("config") or ds.get("args"),
                              ds.get("name"))
                if not k:
                    continue
                for met in res.get("metrics", []):
                    if str(met.get("type", "")).lower() in USABLE_METRICS:
                        try:
                            v = float(met.get("value"))
                        except Exception:
                            continue
                        train[k].add(mid)
                        break
        if (i + 1) % 500 == 0:
            print(f"  ...{i+1}/{len(anchor_ids)}")
    return {k: len(v) for k, v in train.items()}


def _bucket_for(rec, key):
    if key in BENCH_BUCKET:
        return BENCH_BUCKET[key]
    tcs = set(rec.get("task_categories") or [])
    if {"text-classification", "zero-shot-classification"} & tcs:
        return "text-classification"
    if {"question-answering", "multiple-choice", "table-question-answering"} & tcs:
        return "qa-reading-comprehension"
    if {"summarization"} & tcs:
        return "summarization"
    if {"translation"} & tcs:
        return "translation-multilingual"
    if {"token-classification"} & tcs:
        return "token-classification-ner-pos"
    if {"text-retrieval", "feature-extraction"} & tcs:
        return "retrieval-ranking-similarity"
    if {"sentence-similarity"} & tcs:
        return "nli-paraphrase-sts"
    return "domain-specific"


def harvest_datasets():
    os.makedirs(DCACHE, exist_ok=True)
    # 1) candidate pool (listing only; cached) ────────────────────────────────
    pool = {}   # canon_key -> dict(hf_id, listed_downloads, source, bucket_hint)
    listed = 0

    def add(key, hf_id, downloads=0, source="query", bucket_hint=None):
        if not key or not hf_id:
            return
        cur = pool.get(key)
        if cur is None:
            pool[key] = dict(canon_key=key, hf_id=hf_id, listed_downloads=downloads or 0,
                             source=source, bucket_hint=bucket_hint)
        else:
            if (downloads or 0) > (cur["listed_downloads"] or 0):
                cur["hf_id"] = hf_id
                cur["listed_downloads"] = downloads
            if source not in cur["source"]:
                cur["source"] += "+" + source
            if bucket_hint and not cur.get("bucket_hint"):
                cur["bucket_hint"] = bucket_hint

    for b in DATASET_BUCKETS:
        for tc in b["tc"]:
            rows = list_datasets_q(f"tc__{tc}", filter=f"task_categories:{tc}",
                                   sort="downloads", limit=700)
            listed += len(rows)
            for d in rows:
                add(canon_key(d["id"]), d["id"], d["downloads"], "task", b["name"])
        for kw in b["search"]:
            rows = list_datasets_q(f"search__{kw}", search=kw, sort="downloads", limit=120)
            listed += len(rows)
            for d in rows:
                add(canon_key(d["id"]), d["id"], d["downloads"], "search", b["name"])
    for kw in TASK_SUITE_SEARCH:
        rows = list_datasets_q(f"suite__{kw}", search=kw, sort="downloads", limit=120)
        listed += len(rows)
        for d in rows:
            add(canon_key(d["id"]), d["id"], d["downloads"], "suite")
    for key, hf in KEY_TO_HFID.items():
        add(key, hf, source="benchmark", bucket_hint=BENCH_BUCKET.get(key))
    for sub in ("sentiment", "emotion", "hate", "irony", "offensive"):
        add(f"tweet_eval/{sub}", "cardiffnlp/tweet_eval", source="benchmark",
            bucket_hint="text-classification")
    print(f"[ds] listed {listed} candidates -> {len(pool)} unique canonical keys")

    # 2) predicted trainability (bootstrap model-index harvest) ────────────────
    tr_path = os.path.join(PILOT, "_trainability.json")
    if os.path.exists(tr_path):
        trainability = json.load(open(tr_path))
    else:
        trainability = _bootstrap_trainability()
        json.dump(trainability, open(tr_path, "w"))
    print(f"[ds] predicted edge-bearing keys: {sum(1 for v in trainability.values() if v>0)} "
          f"(top: {sorted(trainability.items(), key=lambda x:-x[1])[:6]})")
    # any trainable canon key not yet in pool: add via KEY_TO_HFID/known id
    for key, n in trainability.items():
        if key in pool or n <= 0:
            continue
        hf = KEY_TO_HFID.get(key)
        if not hf and ("/" in key or key.replace("_", "").replace("-", "").isalnum()):
            hf = key
        if hf:
            add(key, hf, source="modelindex")

    # 3) fetch + enrich a bounded high-value set ──────────────────────────────
    by_dl = sorted(pool.values(), key=lambda r: -(r["listed_downloads"] or 0))
    fetch_set = [r for r in pool.values()
                 if trainability.get(r["canon_key"], 0) > 0 or "benchmark" in r["source"]]
    seen_hf = {r["hf_id"] for r in fetch_set}
    for r in by_dl:
        if len(fetch_set) >= 4500:
            break
        if r["hf_id"] not in seen_hf:
            fetch_set.append(r)
            seen_hf.add(r["hf_id"])
    print(f"[ds] fetching+enriching metadata for {len(fetch_set)} candidates")

    enr_path = os.path.join(PILOT, "dataset_enrichment.json")
    enr = json.load(open(enr_path)) if os.path.exists(enr_path) else {}
    recs = {}
    for i, c in enumerate(fetch_set):
        j = _fetch("datasets", c["hf_id"], DCACHE)
        if not j:
            continue
        rec = _parse_dataset(j)
        if c["hf_id"] not in enr:
            enr[c["hf_id"]] = _enrich(c["hf_id"])
        e = enr[c["hf_id"]]
        rec["label_names"] = e["label_names"] or rec.get("label_names") or []
        rec["splits"] = e["splits"] or rec.get("splits") or []
        rec["input_schema"] = e["input_fields"] or rec.get("input_schema") or []
        rec["label_source"] = e["label_source"]
        rec["canon_key"] = c["canon_key"]
        rec["hf_id"] = c["hf_id"]
        rec["listed_downloads"] = c["listed_downloads"]
        rec["source"] = c["source"]
        rec["trainability"] = trainability.get(c["canon_key"], 0)
        rec["bucket"] = c.get("bucket_hint") or _bucket_for(rec, c["canon_key"])
        # loadability proxy: has config/splits/dataset_info schema present
        rec["loadable_proxy"] = bool(rec["splits"]) or bool(rec.get("config")) or bool(rec["label_names"])
        recs[c["canon_key"]] = rec
        if (i + 1) % 300 == 0:
            json.dump(enr, open(enr_path, "w"))
            print(f"  ...fetched {i+1}/{len(fetch_set)}")
    json.dump(enr, open(enr_path, "w"))

    inv_df = pd.DataFrame(list(recs.values()))
    _write_json_csv(inv_df, os.path.join(PILOT, "dataset_inventory"))

    # 4) score + stratified select 1000 ───────────────────────────────────────
    def eligible(r):
        if r["gated"] not in (False, None) or r["private"]:
            return False
        if not (r["has_card"] or r["hf_id"] in KNOWN_BENCHMARKS or r["trainability"] > 0):
            return False
        return True

    def quality(r):
        b = r["bucket"]
        return (3 * has_label_or_schema(r, b) + 2 * has_val_test(r["splits"])
                + 1 * (r["license"] is not None))

    def score(r):
        return (10 * math.log1p(r["trainability"]) + 3 * quality(r)
                + 2 * r["loadable_proxy"] + 0.5 * math.log10((r["listed_downloads"] or 0) + 10))

    skipped = []
    cands = []
    for r in recs.values():
        if eligible(r):
            cands.append(r)
        else:
            skipped.append(dict(canon_key=r["canon_key"], hf_id=r["hf_id"],
                                reason="gated/private" if (r["gated"] not in (False, None) or r["private"])
                                else "no-card/no-edge"))
    cands.sort(key=lambda r: -score(r))

    # per-bucket quotas; text-classification capped at 25% (250)
    QUOTA = {b["name"]: b["quota"] for b in DATASET_BUCKETS}
    CAP = dict(QUOTA)
    CAP["text-classification"] = min(250, QUOTA["text-classification"])
    chosen, bcount, seen_stem, org_count = [], Counter(), set(), Counter()
    glue_n = 0
    # FORCE-INCLUDE every eligible trainable dataset first (edges are only harvestable
    # from selected datasets -- never drop a predicted edge-bearing key to a cap). glue
    # subsets still share the glue cap to avoid one benchmark family dominating.
    for r in sorted([r for r in cands if r["trainability"] > 0], key=lambda r: -r["trainability"]):
        if r["canon_key"].startswith("glue/"):
            if glue_n >= 9:
                continue
            glue_n += 1
        stem = _name_stem(r["hf_id"])
        seen_stem.add(stem)
        org_count[r["hf_id"].split("/")[0]] += 1
        bcount[r["bucket"]] += 1
        chosen.append(r)
    for r in cands:
        if r["trainability"] > 0:
            continue        # already force-included
        if len(chosen) >= N_DATASETS:
            break
        b = r["bucket"]
        if bcount[b] >= CAP.get(b, 140):
            continue
        stem = _name_stem(r["hf_id"])
        org = r["hf_id"].split("/")[0]
        if stem in seen_stem and r["trainability"] == 0:
            continue
        if org_count[org] >= 25 and r["trainability"] == 0:
            continue
        if r["canon_key"].startswith("glue/"):
            if glue_n >= 9:
                continue
            glue_n += 1
        seen_stem.add(stem)
        org_count[org] += 1
        bcount[b] += 1
        chosen.append(r)
    # backfill if caps left us short
    if len(chosen) < N_DATASETS:
        have = {r["canon_key"] for r in chosen}
        for r in cands:
            if len(chosen) >= N_DATASETS:
                break
            if r["canon_key"] not in have:
                chosen.append(r)
                have.add(r["canon_key"])

    sel = pd.DataFrame(chosen)
    sel["has_label_or_schema"] = sel.apply(lambda r: has_label_or_schema(r, r["bucket"]), axis=1)
    sel["has_val_test"] = sel["splits"].map(has_val_test)
    sel["primary_language"] = sel["language"].map(lambda L: (L or [None])[0] if isinstance(L, list) else None)
    sel["is_multilingual"] = sel["language"].map(lambda L: isinstance(L, list) and len(L) >= 2)
    sel["source_url"] = "https://huggingface.co/datasets/" + sel["hf_id"].astype(str)
    sel["selection_reason"] = sel.apply(
        lambda r: f"{r['bucket']}; pred_edges={r['trainability']}; q={quality(r)}; "
                  f"loadable_proxy={r['loadable_proxy']}; key={r['canon_key']}", axis=1)
    sel["timestamp"] = pd.Timestamp.utcnow().isoformat()
    cols = ["canon_key", "hf_id", "config", "bucket", "trainability", "task_categories",
            "task_ids", "language", "primary_language", "is_multilingual", "size", "license",
            "label_names", "input_schema", "arity", "splits", "has_label_or_schema",
            "has_val_test", "loadable_proxy", "listed_downloads", "downloads", "label_source",
            "source", "source_url", "selection_reason", "timestamp"]
    sel = sel[[c for c in cols if c in sel.columns]]
    _write_json_csv(sel, os.path.join(PILOT, "selected_1000_datasets"))
    pd.DataFrame(skipped).to_csv(os.path.join(PILOT, "dataset_skipped_report.md.csv"), index=False)
    _dataset_coverage_report(inv_df, sel, bcount, listed, len(pool), skipped)
    print(f"[ds] inventory {len(inv_df)} | selected {len(sel)} datasets")
    return sel


def _dataset_coverage_report(inv_df, sel, bcount, listed, pool_keys, skipped):
    n = len(sel)
    lab = int(sel["has_label_or_schema"].sum())
    reallab = int((sel["label_names"].map(len) > 0).sum())
    vt = int(sel["has_val_test"].sum())
    lic = int(sel["license"].notna().sum())
    load = int(sel["loadable_proxy"].sum())
    edgeful = int((sel["trainability"] > 0).sum())
    tc_share = bcount.get("text-classification", 0)
    targets = {
        "selected == 1000": (n, 1000, n == 1000),
        "label_names or task schema >= 900": (lab, 900, lab >= 900),
        "val/test split >= 700": (vt, 700, vt >= 700),
        "license >= 800": (lic, 800, lic >= 800),
        "loadable proxy >= 700": (load, 700, load >= 700),
        "predicted edge-bearing >= 200": (edgeful, 200, edgeful >= 200),
        "text-classification <= 25%": (tc_share, 250, tc_share <= 250),
        "candidate pool listed >= 20000": (listed, 20000, listed >= 20000),
        "canonical keys >= 8000": (pool_keys, 8000, pool_keys >= 8000),
    }
    lines = ["# Dataset coverage report (Phase 1, hf1000d_2000m)", ""]
    lines.append(f"Candidate pool: **{listed} listed**, **{pool_keys} unique canonical keys**.")
    lines.append(f"Inventory (fetched+enriched): {len(inv_df)}. Selected: **{n}**/1000.\n")
    lines.append("## Targets")
    for k, (got, tgt, ok) in targets.items():
        lines.append(f"- {k}: **{got}** ({'OK' if ok else 'SHORT'})")
    lines.append(f"  (real ClassLabel names among selected: {reallab})")
    lines.append("\n## Per-bucket coverage")
    for b in DATASET_BUCKETS:
        lines.append(f"- {b['name']}: {bcount.get(b['name'], 0)} / quota {b['quota']}")
    lines.append("\n## Task-category coverage (top 20)")
    tcc = Counter(t for lst in sel["task_categories"] for t in (lst or []))
    for k, v in tcc.most_common(20):
        lines.append(f"- {k}: {v}")
    lc = Counter(l for lst in sel["language"] for l in (lst or []))
    lines.append(f"\nDistinct languages: {len(lc)} (top: "
                 f"{', '.join(f'{k}:{v}' for k, v in lc.most_common(12))})")
    lines.append(f"Arity: {dict(Counter(sel['arity']))}; multilingual datasets: "
                 f"{int(sel['is_multilingual'].sum())}")
    lines.append(f"\nSkipped (gated/private/no-card-no-edge): {len(skipped)}")
    _md(os.path.join(PILOT, "dataset_coverage_report.md"), "\n".join(lines))
    sk = ["# Dataset skipped report (Phase 1)", "",
          f"Total skipped from inventory: {len(skipped)}", ""]
    for s in skipped[:400]:
        sk.append(f"- {s['canon_key']} ({s['hf_id']}): {s['reason']}")
    _md(os.path.join(PILOT, "dataset_skipped_report.md"), "\n".join(sk))


# ════════════════════════════════════════════════════════════════════════════
# PHASE 2 -- dataset-neighborhood model retrieval
# ════════════════════════════════════════════════════════════════════════════
def _aliases(row):
    """Alias strings for a dataset used to filter/search models."""
    key = row["canon_key"]
    hf = str(row["hf_id"])
    al = {key, hf, hf.split("/")[-1], key.split("/")[-1]}
    if key.startswith("glue/"):
        al |= {"glue", key.split("/")[1]}
    if key.startswith("tweet_eval/"):
        al |= {"tweet_eval", "tweeteval", key.split("/")[1]}
    # reverse the canon alias maps (so sst-2 etc. become aliases of glue/sst2)
    for raw, c in {**_GLUE, **_ALIAS}.items():
        if c == key or f"glue/{c}" == key:
            al.add(raw)
    return {a for a in al if a and len(a) >= 2}


def harvest_models():
    sel_d = pd.read_csv(os.path.join(PILOT, "selected_1000_datasets.csv"))
    print(f"[m] datasets: {len(sel_d)}")

    # 1) dataset-neighborhood candidate pool ──────────────────────────────────
    nbr = defaultdict(set)        # model_id -> set(canon_key it was retrieved near)
    nbr_source = defaultdict(set)
    for _, row in sel_d.iterrows():
        key = row["canon_key"]
        bucket = row["bucket"]
        als = _aliases(row)
        # (1) direct model-index / dataset-tag neighborhood: dataset filter
        for a in list(als)[:4]:
            for m in list_models_q(f"dsf__{a}", filter=f"dataset:{a}", sort="downloads", limit=120):
                nbr[m["id"]].add(key); nbr_source[m["id"]].add("dataset_tag")
        # (3) task/pipeline neighborhood
        for pt in TASK_TO_PIPELINES.get(bucket, []):
            for m in list_models_q(f"pipe__{pt}", pipeline_tag=pt, sort="downloads", limit=300):
                nbr[m["id"]].add(key); nbr_source[m["id"]].add("pipeline")
        # (4) domain neighborhood
        if bucket == "domain-specific":
            blob = (str(key) + " " + " ".join(row.get("task_categories") or [])
                    if isinstance(row.get("task_categories"), list) else str(key)).lower()
            for dom, kws in DOMAIN_MODEL_SEARCH.items():
                if dom[:4] in blob or (dom == "biomed" and any(w in blob for w in ("bio", "clinic", "med", "pubmed"))):
                    for kw in kws:
                        for m in list_models_q(f"msearch__{kw}", search=kw, sort="downloads", limit=60):
                            nbr[m["id"]].add(key); nbr_source[m["id"]].add("domain")
    print(f"[m] dataset-neighborhood unique candidate models: {len(nbr)}")

    # 5) global-diversity supplement pool ─────────────────────────────────────
    glob = set()
    for kw in GLOBAL_SUPPLEMENT_SEARCH:
        for m in list_models_q(f"gsearch__{kw}", search=kw, sort="downloads", limit=120):
            glob.add(m["id"])
    for pt in ("text-classification", "sentence-similarity", "summarization",
               "translation", "token-classification", "fill-mask", "text2text-generation"):
        for m in list_models_q(f"gpipe__{pt}", pipeline_tag=pt, sort="downloads", limit=120):
            glob.add(m["id"])
    print(f"[m] global-supplement candidate models: {len(glob)}")

    all_ids = set(nbr) | glob
    print(f"[m] fetching metadata for {len(all_ids)} candidate models")
    inv = {}
    for i, mid in enumerate(sorted(all_ids)):
        j = _fetch("models", mid, MCACHE)
        if not j:
            continue
        if not (j.get("config") or {}).get("model_type"):
            continue
        rec = _parse_model(j)
        rec["matched_dataset_neighbors"] = sorted(nbr.get(mid, []))
        rec["n_dataset_neighbors"] = len(nbr.get(mid, []))
        rec["neighborhood_source"] = sorted(nbr_source.get(mid, [])) or (["global"] if mid in glob else [])
        rec["is_neighborhood"] = mid in nbr
        # predicted usable edges into selected datasets
        sel_keys = set(sel_d["canon_key"])
        ne = 0
        for entry in ((j.get("cardData") or {}).get("model-index") or []):
            for res in entry.get("results", []):
                ds = res.get("dataset", {}) or {}
                k = canon_key(ds.get("type") or ds.get("name"), ds.get("config") or ds.get("args"),
                              ds.get("name"))
                if k in sel_keys and any(str(m.get("type", "")).lower() in USABLE_METRICS
                                         for m in res.get("metrics", [])):
                    ne += 1
        rec["pred_edges"] = ne
        inv[mid] = rec
        if (i + 1) % 800 == 0:
            print(f"  ...{i+1}/{len(all_ids)}")
    inv_df = pd.DataFrame(list(inv.values()))
    _write_json_csv(inv_df, os.path.join(PILOT, "model_inventory"))

    # 6) select 2000: ~70% neighborhood / ~30% global; bert-ish <=50% ──────────
    def eligible(r):
        return (r["gated"] in (False, None)) and not r["private"] and r["model_type"]

    cand = [r for r in inv.values() if eligible(r)]
    # priority: pred_edges desc, then neighborhood, then downloads
    cand.sort(key=lambda r: (-(r["pred_edges"] or 0), -(1 if r["is_neighborhood"] else 0),
                             -(r["downloads"] or 0)))
    NEI_TARGET = int(N_MODELS * 0.70)
    BERTISH_CAP = N_MODELS // 2
    chosen, seen_stem, fam_count = [], set(), Counter()
    bertish = 0
    nei = 0
    skipped = []

    def bertish_of(r):
        return (r["model_type"] in BERTISH_TYPES)

    # first pass: fill neighborhood up to NEI_TARGET (prefer edge-bearing)
    for r in cand:
        if len([c for c in chosen if c["is_neighborhood"]]) >= NEI_TARGET:
            break
        if not r["is_neighborhood"]:
            continue
        stem = _name_stem(r["model_id"])
        if stem in seen_stem and r["pred_edges"] == 0:
            continue
        if bertish_of(r) and bertish >= BERTISH_CAP:
            continue
        if fam_count[r["family"]] >= 220 and r["pred_edges"] == 0:
            continue
        seen_stem.add(stem); fam_count[r["family"]] += 1
        if bertish_of(r):
            bertish += 1
        chosen.append(r)
    # second pass: global supplement + remaining to reach 2000
    for r in cand:
        if len(chosen) >= N_MODELS:
            break
        if r in chosen:
            continue
        stem = _name_stem(r["model_id"])
        if stem in seen_stem and r["pred_edges"] == 0:
            continue
        if bertish_of(r) and bertish >= BERTISH_CAP:
            continue
        if fam_count[r["family"]] >= 220 and r["pred_edges"] == 0:
            continue
        seen_stem.add(stem); fam_count[r["family"]] += 1
        if bertish_of(r):
            bertish += 1
        chosen.append(r)

    sel = pd.DataFrame(chosen)
    sel["source_url"] = "https://huggingface.co/" + sel["model_id"]
    sel["selection_reason"] = sel.apply(
        lambda r: f"pred_edges={r['pred_edges']}; "
                  f"{'neighborhood' if r['is_neighborhood'] else 'global-supplement'}; "
                  f"{r['model_type']}; neighbors={r['n_dataset_neighbors']}; dl={r['downloads']}", axis=1)
    sel["timestamp"] = pd.Timestamp.utcnow().isoformat()
    cols = ["model_id", "architecture", "model_type", "family", "base_model", "pipeline_tag",
            "tags", "language", "license", "parameter_count", "downloads", "likes",
            "has_model_index", "pred_edges", "is_neighborhood", "n_dataset_neighbors",
            "matched_dataset_neighbors", "neighborhood_source", "source_url",
            "selection_reason", "timestamp"]
    sel = sel[[c for c in cols if c in sel.columns]]
    _write_json_csv(sel, os.path.join(PILOT, "selected_2000_models"))
    _model_coverage_report(inv_df, sel, bertish)
    print(f"[m] inventory {len(inv_df)} | selected {len(sel)} models | bert-ish {bertish}")
    return sel


def _model_coverage_report(inv_df, sel, bertish):
    n = len(sel)
    nei = int(sel["is_neighborhood"].sum())
    edgeful = int((sel["pred_edges"] > 0).sum())
    ge3 = int((sel["pred_edges"] >= 3).sum())
    nfam = sel["family"].nunique()
    targets = {
        "selected == 2000": (n, 2000, n == 2000),
        "edge-bearing models >= 500": (edgeful, 500, edgeful >= 500),
        "models with >=3 candidate edges >= 300": (ge3, 300, ge3 >= 300),
        ">= 20 families": (nfam, 20, nfam >= 20),
        "bert-ish <= 50%": (bertish, n // 2, bertish <= n // 2),
        "neighborhood ~70%": (nei, int(n * 0.6), nei >= int(n * 0.6)),
    }
    lines = ["# Model coverage report (Phase 2, hf1000d_2000m)", ""]
    lines.append(f"Inventory: {len(inv_df)}. Selected: **{n}**/2000.\n")
    lines.append("## Targets")
    for k, (got, tgt, ok) in targets.items():
        lines.append(f"- {k}: **{got}** ({'OK' if ok else 'SHORT'})")
    lines.append("\n## Acquisition path")
    lines.append(f"- dataset-neighborhood: {nei}/{n} ({100*nei/n:.0f}%); "
                 f"global-supplement: {n-nei}/{n} ({100*(n-nei)/n:.0f}%)")
    lines.append("\n## Architecture / family")
    lines.append(f"- bert-ish (bert/roberta/distilbert): {bertish}/{n} = {100*bertish/n:.0f}% (cap 50%)")
    lines.append(f"- distinct families: {nfam}; distinct model_type: {sel['model_type'].nunique()}")
    lines.append("- model_type histogram (top 25):")
    for k, v in Counter(sel["model_type"]).most_common(25):
        lines.append(f"  - {k}: {v}")
    lines.append("\n## Metadata")
    lines.append(f"- with model-index: {int(sel['has_model_index'].sum())}/{n}")
    lines.append(f"- with base_model (lineage): {int(sel['base_model'].notna().sum())}/{n}")
    unk = int(sel["parameter_count"].isna().sum())
    lines.append(f"- unknown parameter_count: {unk}/{n} ({100*unk/n:.0f}%) [reported, not hidden]")
    lc = Counter(l for lst in sel["language"] for l in (lst if isinstance(lst, list) else []))
    lines.append(f"- distinct languages: {len(lc)}")
    _md(os.path.join(PILOT, "model_coverage_report.md"), "\n".join(lines))


# ════════════════════════════════════════════════════════════════════════════
# PHASE 3 -- real performance edges
# ════════════════════════════════════════════════════════════════════════════
def harvest_edges():
    sel_m = pd.read_csv(os.path.join(PILOT, "selected_2000_models.csv"))
    sel_d = pd.read_csv(os.path.join(PILOT, "selected_1000_datasets.csv"))
    sel_keys = set(sel_d["canon_key"])
    alias_map = {}     # raw ref -> canon key (provenance)

    rows = []
    for mid in sel_m["model_id"]:
        j = _fetch("models", mid, MCACHE)
        if not j:
            continue
        for entry in ((j.get("cardData") or {}).get("model-index") or []):
            for res in entry.get("results", []):
                ds = res.get("dataset", {}) or {}
                raw_type = ds.get("type") or ds.get("name") or ""
                raw_cfg = ds.get("config") or ds.get("args") or ""
                k = canon_key(raw_type, raw_cfg, ds.get("name"))
                if not k:
                    continue
                alias_map[f"{raw_type}|{raw_cfg}"] = k
                task = (res.get("task", {}) or {}).get("type") or ""
                for met in res.get("metrics", []):
                    mt = str(met.get("type", "")).lower()
                    if mt not in MULTI_METRICS:
                        continue
                    try:
                        v = float(met.get("value"))
                    except Exception:
                        continue
                    if mt in USABLE_METRICS and v > 1.0:
                        v /= 100.0
                    rows.append(dict(
                        model_id=mid, dataset_ref=raw_type, dataset_config=raw_cfg,
                        canon_key=k, in_selected_dataset=(k in sel_keys), task_type=task,
                        metric_name=mt, metric_value=v, confidence=1,
                        verified=met.get("verified"),
                        source_name="hf_model_index", source_url=f"https://huggingface.co/{mid}"))
    edf = pd.DataFrame(rows)
    json.dump(alias_map, open(os.path.join(PILOT, "alias_map.json"), "w"), indent=1)

    # multi-metric candidate table B (into selected datasets)
    multi = edf[edf["in_selected_dataset"]].copy() if not edf.empty else edf
    _write_json_csv(multi, os.path.join(PILOT, "performance_edges_multi_metric_candidate"))

    # single-metric table A: accuracy/f1/matthews only, normalized per (key, metric)
    single = multi[multi["metric_name"].isin(list(USABLE_METRICS))].copy() if not multi.empty else multi
    if not single.empty:
        single = single[(single["metric_value"] > 0) & (single["metric_value"] <= 1.0)].copy()
        single["norm_value"] = single["metric_value"]
        for (k, mt), idx in single.groupby(["canon_key", "metric_name"]).groups.items():
            v = single.loc[idx, "metric_value"]
            lo, hi = v.min(), v.max()
            single.loc[idx, "norm_value"] = 0.5 if hi - lo < 1e-9 else (v - lo) / (hi - lo)
    _write_json_csv(single, os.path.join(PILOT, "performance_edges_single_metric"))

    _edge_reports(single, multi, edf, sel_m, sel_d)
    print(f"[edges] raw metrics {len(edf)} | single-metric(selected) {len(single)} | "
          f"multi-metric(selected) {len(multi)}")
    return single


def _edge_reports(single, multi, edf, sel_m, sel_d):
    nm, nd = len(sel_m), len(sel_d)
    # distinct trained_on pairs: one (model, dataset) pair = an edge
    if not single.empty:
        pairs = single.drop_duplicates(["model_id", "canon_key"])
        per_model = pairs.groupby("model_id")["canon_key"].nunique()
        per_ds = pairs.groupby("canon_key")["model_id"].nunique()
        n_pairs = len(pairs)
        edge_ds = per_ds.shape[0]
        ds_ge10 = int((per_ds >= 10).sum())
        models_ge3 = int((per_model >= 3).sum())
    else:
        per_model = per_ds = pd.Series(dtype=int)
        n_pairs = edge_ds = ds_ge10 = models_ge3 = 0

    targets = {
        "single-metric distinct trained_on pairs >= 3000": (n_pairs, 3000, n_pairs >= 3000),
        "edge-bearing datasets >= 100": (edge_ds, 100, edge_ds >= 100),
        "datasets with >=10 edges >= 50": (ds_ge10, 50, ds_ge10 >= 50),
        "models with >=3 edges >= 300": (models_ge3, 300, models_ge3 >= 300),
    }
    extra_multi = (len(multi.drop_duplicates(["model_id", "canon_key"])) - n_pairs) if not multi.empty else 0

    lines = ["# Edge coverage report (Phase 3, hf1000d_2000m)", ""]
    lines.append("Real HF model-index metrics only (confidence=1). Table A = single-metric "
                 "(accuracy/f1/matthews); Table B = multi-metric candidate.\n")
    lines.append("## Minimum targets before graph training (Table A)")
    allok = True
    for k, (got, tgt, ok) in targets.items():
        allok = allok and ok
        lines.append(f"- {k}: **{got}** ({'OK' if ok else 'SHORT'})")
    lines.append(f"\n## Coverage (Table A)")
    lines.append(f"- distinct trained_on pairs: **{n_pairs}**")
    if not single.empty:
        lines.append(f"- edges/model: mean {per_model.mean():.2f}, median {per_model.median():.0f}; "
                     f"models >=3: {models_ge3}/{nm}")
        lines.append(f"- edges/dataset: mean {per_ds.mean():.2f}, median {per_ds.median():.0f}; "
                     f"datasets >=10: {ds_ge10}/{nd}")
        lines.append(f"- zero-edge models: {nm - per_model.shape[0]}/{nm}; "
                     f"zero-edge datasets: {nd - edge_ds}/{nd}")
        lines.append("\n## Top datasets by edges")
        for k, c in per_ds.sort_values(ascending=False).head(25).items():
            lines.append(f"- {k}: {c}")
        lines.append("\n## Per-bucket edge coverage")
        key2bucket = dict(zip(sel_d["canon_key"], sel_d["bucket"]))
        bcov = Counter()
        for k, c in per_ds.items():
            bcov[key2bucket.get(k, "?")] += c
        for b, c in bcov.most_common():
            lines.append(f"- {b}: {c} edges")
    lines.append(f"\n## Multi-metric (Table B)")
    lines.append(f"- extra edges available under multi-metric support: {extra_multi}")
    if not edf.empty:
        lines.append("- metric histogram (all raw, selected+unselected):")
        for k, v in Counter(edf["metric_name"]).most_common():
            lines.append(f"  - {k}: {v}")
    lines.append(f"\n## VERDICT: {'TARGETS MET -> proceed to Phase 4' if allok else 'SHORT -> STOP before Phase 4 (insufficient effective graph)'}")
    _md(os.path.join(PILOT, "edge_coverage_report.md"), "\n".join(lines))

    # metric heterogeneity report
    h = ["# Metric heterogeneity report (Phase 3)", ""]
    if not edf.empty:
        sel_edf = edf[edf["in_selected_dataset"]]
        h.append(f"Selected-dataset raw metric rows: {len(sel_edf)}.")
        h.append("\n## metric_name distribution (selected datasets)")
        for k, v in Counter(sel_edf["metric_name"]).most_common():
            h.append(f"- {k}: {v}")
        h.append("\n## datasets carrying each metric (selected)")
        for k in sorted(set(sel_edf["metric_name"])):
            nds = sel_edf[sel_edf["metric_name"] == k]["canon_key"].nunique()
            h.append(f"- {k}: {nds} datasets")
        h.append("\nRule: Table A never mixes metrics. One metric class per dataset is used "
                 "for trained_on edges (accuracy/f1/matthews). Cross-metric merging only under "
                 "explicit multi-metric graph logic (not enabled this run).")
    _md(os.path.join(PILOT, "metric_heterogeneity_report.md"), "\n".join(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["all", "datasets", "models", "edges"])
    args = ap.parse_args()
    os.makedirs(PILOT, exist_ok=True)
    if args.phase in ("all", "datasets"):
        harvest_datasets()
    if args.phase in ("all", "models"):
        harvest_models()
    if args.phase in ("all", "edges"):
        harvest_edges()
    print("\nDONE.")


if __name__ == "__main__":
    main()
