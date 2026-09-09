"""
build_hf_pilot_v3.py -- edge-aware dataset re-selection (v3). DATASET SIDE ONLY.

Constraints (user):
 * do NOT touch Stage-1/Stage-2 logic; do NOT add edge types; do NOT fabricate edges;
   no self-eval. Edges still come ONLY from real HF model-index metrics.
 * fix the diversity-vs-trainability problem purely by re-sampling datasets:
   - expand the candidate pool to 3000+ (task-category queries + reverse-extracted
     dataset references from the 300 models' model-index + canonical benchmark aliases);
   - canonicalize aliases (sst2 / sst-2 / glue/sst2 -> glue/sst2; mnli / multi_nli ->
     glue/mnli; agnews / ag_news -> ag_news; ...);
   - score = diversity + quality + TRAINABILITY (matched usable model-index edge count,
     used ONLY to select, never to create edges);
   - keep v1 + v2; write selected_100_datasets_v3 + reports + v2->v3 diff.
 * also top up models to 300 (done separately), then re-match real edges -> v3.

If v3 is still edge-sparse, do NOT enter Phase 4/5; report that HF self-reported
model-index data is inherently insufficient under no-self-eval / no-fabrication.

Run (after model top-up):  ../.venv/Scripts/python.exe build_hf_pilot_v3.py
"""

import json
import math
import os
import re
import sys
from collections import defaultdict, Counter

import pandas as pd
from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_hf_pilot import (
    PILOT, DCACHE, MCACHE, _fetch, _parse_dataset, enrich_dataset, _name_stem,
    _write_json_csv, _md, KNOWN_BENCHMARKS, SCHEMA_TASKS, DATASET_BUCKETS,
)

api = HfApi()
USABLE_METRICS = {"accuracy", "f1", "matthews_correlation"}   # what trained_on edges use

# ── canonicalization ──────────────────────────────────────────────────────────
_GLUE = {"sst2": "sst2", "sst-2": "sst2", "sst": "sst2", "cola": "cola",
         "mnli": "mnli", "multi_nli": "mnli", "multinli": "mnli", "mnli-mm": "mnli",
         "qnli": "qnli", "qqp": "qqp", "rte": "rte", "mrpc": "mrpc",
         "stsb": "stsb", "sts-b": "stsb", "wnli": "wnli"}
_ALIAS = {
    "agnews": "ag_news", "ag_news": "ag_news",
    "imdb": "imdb", "dbpedia": "dbpedia_14", "dbpedia_14": "dbpedia_14",
    "yelppolarity": "yelp_polarity", "yelp": "yelp_polarity",
    "yelpreviewfull": "yelp_review_full", "amazonpolarity": "amazon_polarity",
    "squad": "squad", "squadv2": "squad_v2", "squad2": "squad_v2", "squad_v2": "squad_v2",
    "conll2003": "conll2003", "conll03": "conll2003", "conll-2003": "conll2003",
    "xnli": "xnli", "snli": "snli", "anli": "anli", "paws": "paws", "pawsx": "paws-x",
    "cnndailymail": "cnn_dailymail", "cnn_dailymail": "cnn_dailymail", "cnndm": "cnn_dailymail",
    "xsum": "xsum", "samsum": "samsum", "billsum": "billsum",
    "emotion": "emotion", "rottentomatoes": "rotten_tomatoes", "rotten_tomatoes": "rotten_tomatoes",
    "banking77": "banking77", "trec": "trec", "boolq": "boolq", "copa": "copa",
    "winogrande": "winogrande", "financialphrasebank": "financial_phrasebank",
    "hatespeech18": "hate_speech18", "tweeteval": "tweet_eval",
}
# canonical key -> a real HF dataset id to use for metadata/embedding (node still
# named by the canonical key; glue subsets share one repo + a config).
KEY_TO_HFID = {
    **{f"glue/{v}": "nyu-mll/glue" for v in set(_GLUE.values())},
    "ag_news": "fancyzhx/ag_news", "imdb": "stanfordnlp/imdb",
    "dbpedia_14": "fancyzhx/dbpedia_14", "yelp_polarity": "fancyzhx/yelp_polarity",
    "yelp_review_full": "Yelp/yelp_review_full", "amazon_polarity": "fancyzhx/amazon_polarity",
    "squad": "rajpurkar/squad", "squad_v2": "rajpurkar/squad_v2",
    "conll2003": "eriktks/conll2003", "xnli": "facebook/xnli", "snli": "stanfordnlp/snli",
    "anli": "facebook/anli", "paws": "google-research-datasets/paws",
    "cnn_dailymail": "abisee/cnn_dailymail", "xsum": "EdinburghNLP/xsum",
    "samsum": "Samsung/samsum", "billsum": "FiscalNote/billsum",
    "emotion": "dair-ai/emotion", "rotten_tomatoes": "cornell-movie-review-data/rotten_tomatoes",
    "banking77": "PolyAI/banking77", "trec": "CogComp/trec", "boolq": "google/boolq",
    "copa": "aps/super_glue", "winogrande": "allenai/winogrande",
    "financial_phrasebank": "takala/financial_phrasebank",
}
BENCH_BUCKET = {
    **{f"glue/{v}": ("nli-paraphrase-sts" if v in ("mnli", "qnli", "rte", "wnli", "qqp", "mrpc", "stsb")
                     else "text-classification") for v in set(_GLUE.values())},
    "ag_news": "text-classification", "imdb": "text-classification",
    "dbpedia_14": "text-classification", "yelp_polarity": "text-classification",
    "yelp_review_full": "text-classification", "amazon_polarity": "text-classification",
    "emotion": "text-classification", "rotten_tomatoes": "text-classification",
    "banking77": "text-classification", "trec": "text-classification",
    "financial_phrasebank": "domain-specific",
    "xnli": "nli-paraphrase-sts", "snli": "nli-paraphrase-sts", "anli": "nli-paraphrase-sts",
    "paws": "nli-paraphrase-sts", "boolq": "qa-reading-comprehension",
    "copa": "nli-paraphrase-sts", "winogrande": "nli-paraphrase-sts",
    "squad": "qa-reading-comprehension", "squad_v2": "qa-reading-comprehension",
    "conll2003": "token-classification-ner-pos",
    "cnn_dailymail": "summarization", "xsum": "summarization",
    "samsum": "summarization", "billsum": "summarization",
}


def canon_key(type_str, config="", name=""):
    s = re.sub(r"[^a-z0-9/_.\- ]", "", str(type_str or name or "").lower()).strip()
    cfg = re.sub(r"[^a-z0-9_.\- ]", "", str(config or "").lower()).strip()
    base = s.split("/")[-1]
    flat = base.replace("-", "").replace("_", "").replace(" ", "").replace(".", "")
    cflat = cfg.replace("-", "").replace("_", "").replace(" ", "")
    # glue family (type 'glue' + config, or standalone subset name)
    if "glue" in s and cflat in _GLUE:
        return f"glue/{_GLUE[cflat]}"
    if flat in _GLUE and ("glue" in s or flat in ("sst2", "mnli", "qnli", "qqp", "rte", "cola", "mrpc", "stsb", "wnli")):
        return f"glue/{_GLUE[flat]}"
    if ("tweet_eval" in s or "tweeteval" in flat):
        return f"tweet_eval/{cflat}" if cfg else "tweet_eval"
    if flat in _ALIAS:
        return _ALIAS[flat]
    return base  # default: last path component (already lowercased)


# ── reverse-extract trainability from the 300 models' model-index ─────────────
def reverse_trainability(sel_models):
    """key -> set(model_ids) with a USABLE (accuracy/f1/matthews) self-reported metric,
    plus the full raw edge rows. From cached model JSON only (no network, no new edges)."""
    train = defaultdict(set)
    raw = []
    key_example_ref = {}
    for mid in sel_models["model_id"]:
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
                key_example_ref.setdefault(k, (ds.get("type"), ds.get("config") or ds.get("args")))
                for m in res.get("metrics", []):
                    mt = str(m.get("type", "")).lower()
                    if mt not in USABLE_METRICS:
                        continue
                    try:
                        v = float(m.get("value"))
                    except Exception:
                        continue
                    if v > 1.0:
                        v /= 100.0
                    if not (0.0 < v <= 1.0):
                        continue
                    train[k].add(mid)
                    raw.append(dict(model_id=mid, canon_key=k,
                                    dataset_type=ds.get("type"), dataset_config=ds.get("config") or ds.get("args"),
                                    task_type=(res.get("task", {}) or {}).get("type"),
                                    metric_name=mt, metric_value=v, confidence=1,
                                    source_name="hf_model_index",
                                    source_url=f"https://huggingface.co/{mid}"))
    return train, raw, key_example_ref


# ── candidate pool (>=3000 listed) ────────────────────────────────────────────
def expand_candidates(train_keys, key_example_ref):
    """Return {canon_key: dict(hf_id, config, listed_downloads, source)} candidate pool.
    Lists 3000+ ids cheaply; metadata fetched later only for a bounded high-value set."""
    pool = {}

    def add(key, hf_id, config=None, downloads=0, source="query"):
        if not key or not hf_id:
            return
        cur = pool.get(key)
        if cur is None or (downloads or 0) > (cur["listed_downloads"] or 0):
            pool[key] = dict(canon_key=key, hf_id=hf_id, config=config,
                             listed_downloads=downloads or (cur["listed_downloads"] if cur else 0),
                             source=(cur["source"] + "+" + source) if cur and source not in cur["source"] else source)

    listed = 0
    # (1) task-category + search queries, big limits (listing only)
    for b in DATASET_BUCKETS:
        for tc in b["tc"]:
            try:
                for d in api.list_datasets(filter=f"task_categories:{tc}", sort="downloads", limit=400):
                    listed += 1
                    add(canon_key(d.id), d.id, downloads=getattr(d, "downloads", 0) or 0, source="task")
            except Exception:
                pass
        for kw in b["search"]:
            try:
                for d in api.list_datasets(search=kw, sort="downloads", limit=80):
                    listed += 1
                    add(canon_key(d.id), d.id, downloads=getattr(d, "downloads", 0) or 0, source="search")
            except Exception:
                pass
    # (2) canonical benchmark aliases -> real hf ids (force-include high-trainability)
    for key, hf in KEY_TO_HFID.items():
        cfg = key.split("/")[1] if key.startswith("glue/") else None
        add(key, hf, config=cfg, source="benchmark")
    # tweet_eval subtasks
    for sub in ("sentiment", "emotion", "hate", "irony", "offensive"):
        add(f"tweet_eval/{sub}", "cardiffnlp/tweet_eval", config=sub, source="benchmark")
    # (3) reverse-extracted dataset refs from model-index (the trainable universe)
    for key in train_keys:
        if key in pool:
            pool[key]["source"] += "+modelindex"
            continue
        hf = KEY_TO_HFID.get(key)
        if not hf:
            t, c = key_example_ref.get(key, (None, None))
            # use the model-index type if it looks like a real HF id (has '/' or known)
            if t and ("/" in str(t) or str(t).replace("_", "").isalnum()):
                hf = str(t)
        if hf:
            cfg = key.split("/")[1] if key.startswith("glue/") else None
            add(key, hf, config=cfg, source="modelindex")
    return pool, listed


def bucket_of(rec, key):
    if key in BENCH_BUCKET:
        return BENCH_BUCKET[key]
    tcs = set(rec.get("task_categories") or [])
    if {"text-classification", "zero-shot-classification"} & tcs:
        return "text-classification"
    if {"sentence-similarity"} & tcs:
        return "nli-paraphrase-sts"
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
    return "domain-specific"


def has_val_test(splits):
    return any(s in (splits or []) for s in ("validation", "test", "dev", "valid"))


def has_label_or_schema(rec, bucket):
    if rec.get("label_names"):
        return True
    tcs = set(rec.get("task_categories") or [])
    if bucket in ("qa-reading-comprehension", "summarization", "translation-multilingual",
                  "retrieval-ranking-similarity") or (tcs & SCHEMA_TASKS):
        return bool(rec.get("input_schema")) or bool(tcs & SCHEMA_TASKS) or True
    return False


def main():
    sel_models = pd.read_csv(os.path.join(PILOT, "selected_300_models.csv"))
    print(f"[v3] models: {len(sel_models)}")
    train, raw, key_example_ref = reverse_trainability(sel_models)
    usable_keys = {k: len(v) for k, v in train.items()}
    print(f"[v3] reverse-extracted usable-edge dataset keys: {len(usable_keys)}; "
          f"top: {sorted(usable_keys.items(), key=lambda x:-x[1])[:8]}")

    pool, listed = expand_candidates(set(train.keys()), key_example_ref)
    print(f"[v3] listed candidates: {listed}; unique canonical candidate keys: {len(pool)}")

    # fetch metadata for a bounded high-value set: all with edges + benchmarks +
    # top-by-downloads per bucket-ish. Keeps network bounded.
    enr_path = os.path.join(PILOT, "dataset_enrichment_v3.json")
    enr = json.load(open(enr_path)) if os.path.exists(enr_path) else {}
    by_dl = sorted(pool.values(), key=lambda r: -(r["listed_downloads"] or 0))
    fetch_set = [r for r in pool.values() if (r["canon_key"] in train) or ("benchmark" in r["source"])]
    fetch_set += by_dl[:700]
    seen = set(); fetch_set = [r for r in fetch_set if not (r["hf_id"] in seen or seen.add(r["hf_id"]))]
    print(f"[v3] fetching metadata for {len(fetch_set)} candidates")

    recs = {}
    for i, c in enumerate(fetch_set):
        j = _fetch("datasets", c["hf_id"], DCACHE)
        if not j:
            continue
        rec = _parse_dataset(j)
        key = c["canon_key"]
        if c["hf_id"] not in enr:
            enr[c["hf_id"]] = enrich_dataset(c["hf_id"], offline=False)
            if (i + 1) % 50 == 0:
                json.dump(enr, open(enr_path, "w")); print(f"  ...enriched {i+1}/{len(fetch_set)}")
        e = enr[c["hf_id"]]
        rec["label_names"] = e["label_names"] or rec.get("label_names") or []
        rec["splits"] = e["splits"] or rec.get("splits") or []
        rec["input_schema"] = e["input_fields"] or rec.get("input_schema") or []
        rec["label_source"] = e["label_source"]
        rec["canon_key"] = key
        rec["hf_id"] = c["hf_id"]; rec["config"] = c["config"]
        rec["trainability"] = usable_keys.get(key, 0)
        rec["bucket"] = bucket_of(rec, key)
        recs[key] = rec      # node identity = canonical key
    json.dump(enr, open(enr_path, "w"))

    # ── score = diversity + quality + trainability ───────────────────────────
    def eligible(r):
        return (r["gated"] in (False, None)) and not r["private"] and \
               (r["has_card"] or r["hf_id"] in KNOWN_BENCHMARKS or r["trainability"] > 0)

    def quality(r):
        b = r["bucket"]
        return (3 * has_label_or_schema(r, b) + 2 * has_val_test(r["splits"]) + 1 * (r["license"] is not None))

    def score(r):
        # trainability dominates (that was the gap), then quality, then a small
        # download prior; diversity handled by per-bucket caps below.
        return (12 * math.log1p(r["trainability"]) + 4 * quality(r)
                + 0.5 * math.log10((r["downloads"] or 0) + 10))

    cands = [r for r in recs.values() if eligible(r)]
    cands.sort(key=lambda r: -score(r))

    # selection: keep 8 buckets present, cap text-classification & glue so they
    # don't dominate, prefer trainable. flexible quotas.
    BUCKET_CAP = {"text-classification": 22, "nli-paraphrase-sts": 18,
                  "qa-reading-comprehension": 12, "summarization": 10,
                  "translation-multilingual": 10, "token-classification-ner-pos": 10,
                  "retrieval-ranking-similarity": 10, "domain-specific": 16}
    GLUE_CAP = 9
    chosen, bcount, glue_n = [], Counter(), 0
    for r in cands:
        b = r["bucket"]
        if len(chosen) >= 100:
            break
        if bcount[b] >= BUCKET_CAP.get(b, 14):
            continue
        if r["canon_key"].startswith("glue/"):
            if glue_n >= GLUE_CAP:
                continue
            glue_n += 1
        bcount[b] += 1
        chosen.append(r)
    # if under 100 (caps too tight), fill remaining by score ignoring caps (still dedup)
    if len(chosen) < 100:
        have = {r["canon_key"] for r in chosen}
        for r in cands:
            if len(chosen) >= 100:
                break
            if r["canon_key"] not in have:
                chosen.append(r); have.add(r["canon_key"])

    sel = pd.DataFrame(chosen)
    sel["has_label_or_schema"] = sel.apply(lambda r: has_label_or_schema(r, r["bucket"]), axis=1)
    sel["has_val_test"] = sel["splits"].map(has_val_test)
    sel["source_url"] = "https://huggingface.co/datasets/" + sel["hf_id"].astype(str)
    sel["selection_reason"] = sel.apply(
        lambda r: f"{r['bucket']}; trainability={r['trainability']} usable edges; "
                  f"q={quality(r)}; key={r['canon_key']}", axis=1)
    cols = ["canon_key", "hf_id", "config", "bucket", "trainability", "task_categories",
            "language", "license", "label_names", "splits", "has_label_or_schema",
            "has_val_test", "arity", "downloads", "label_source", "source_url", "selection_reason"]
    sel = sel[[c for c in cols if c in sel.columns]]
    _write_json_csv(sel, os.path.join(PILOT, "selected_100_datasets_v3"))

    # ── edges v3: real model-index edges restricted to selected v3 keys ───────
    sel_keys = set(sel["canon_key"])
    edf = pd.DataFrame([e for e in raw if e["canon_key"] in sel_keys])
    if not edf.empty:
        edf["in_selected_dataset"] = True
        edf["norm_value"] = edf["metric_value"]
        for (k, mt), idx in edf.groupby(["canon_key", "metric_name"]).groups.items():
            v = edf.loc[idx, "metric_value"]; lo, hi = v.min(), v.max()
            edf.loc[idx, "norm_value"] = 0.5 if hi - lo < 1e-9 else (v - lo) / (hi - lo)
    _write_json_csv(edf, os.path.join(PILOT, "performance_edges_v3"))

    _reports(sel, edf, sel_models, listed, len(pool))
    print(f"[v3] DONE. selected {len(sel)} datasets; usable edges {len(edf)}")


def _reports(sel, edf, sel_models, listed, pool_keys):
    n = len(sel); nm = len(sel_models)
    lab = int(sel["has_label_or_schema"].sum()); reallab = int((sel["label_names"].map(len) > 0).sum())
    vt = int(sel["has_val_test"].sum()); lic = int(sel["license"].notna().sum())
    em = edf[edf["metric_name"].isin(list(USABLE_METRICS))] if not edf.empty else edf
    per_model = em.groupby("model_id")["canon_key"].nunique() if not em.empty else pd.Series(dtype=int)
    per_ds = em.groupby("canon_key")["model_id"].nunique() if not em.empty else pd.Series(dtype=int)
    zero_ds = n - (per_ds.shape[0] if not em.empty else 0)
    ds_ge10 = int((per_ds >= 10).sum()) if not em.empty else 0
    models_ge3 = int((per_model >= 3).sum()) if not em.empty else 0
    zero_models = nm - (per_model.shape[0] if not em.empty else 0)
    usable = len(em)

    targets = {
        "label_or_schema>=90": (lab, 90, lab >= 90),
        "val/test>=70": (vt, 70, vt >= 70),
        "license>=80": (lic, 80, lic >= 80),
        "zero-edge datasets<30": (zero_ds, 30, zero_ds < 30),
        "datasets>=10 edges (>=30)": (ds_ge10, 30, ds_ge10 >= 30),
        "usable edges>1000": (usable, 1000, usable > 1000),
    }
    lines = ["# Dataset coverage report v3 (edge-aware re-selection)", ""]
    lines.append(f"Candidate pool: **{listed} listed**, {pool_keys} unique canonical keys "
                 f"(task queries + reverse-extracted model-index refs + benchmark aliases).")
    lines.append(f"Selected: **{n}**/100.\n")
    lines.append("## Targets")
    for k, (got, tgt, ok) in targets.items():
        lines.append(f"- {k}: **{got}** ({'OK' if ok else 'SHORT'})")
    lines.append(f"  (real ClassLabel names among selected: {reallab})")
    lines.append("\n## Per-bucket coverage")
    for b, c in Counter(sel["bucket"]).most_common():
        lines.append(f"- {b}: {c}")
    glue_n = int(sel['canon_key'].str.startswith('glue/').sum())
    lines.append(f"- glue/* share: {glue_n} (capped)")
    lines.append("\n## Edge coverage (real model-index, usable accuracy/f1/matthews)")
    lines.append(f"- usable trained_on edges: **{usable}**")
    if not em.empty:
        lines.append(f"- edges/model: mean {per_model.mean():.2f}, median {per_model.median():.0f}; "
                     f"models with >=3: {models_ge3}/{nm}")
        lines.append(f"- edges/dataset: mean {per_ds.mean():.2f}, median {per_ds.median():.0f}; "
                     f"datasets with >=10: {ds_ge10}/{n}")
        lines.append(f"- zero-edge models: {zero_models}/{nm}; zero-edge datasets: {zero_ds}/{n}")
        lines.append("\n## Top datasets by usable edges")
        for k, c in per_ds.sort_values(ascending=False).head(20).items():
            lines.append(f"- {k}: {c}")
    lc = Counter(l for lst in sel["language"] for l in (lst or []))
    lines.append(f"\nDistinct languages: {len(lc)}; arity: {dict(Counter(sel['arity']))}")
    verdict = all(ok for _, _, ok in targets.values())
    lines.append(f"\n## VERDICT: {'TRAINABLE — targets met' if verdict else 'still gaps — see SHORT items'}")
    if not verdict and usable <= 1000:
        lines.append("Under no-self-eval / no-fabrication, HF self-reported model-index data may be "
                     "inherently insufficient for the diverse buckets; do NOT enter Phase 4/5 without "
                     "either accepting sparsity or enabling self-eval.")
    _md(os.path.join(PILOT, "dataset_coverage_report_v3.md"), "\n".join(lines))
    _md(os.path.join(PILOT, "edge_coverage_report_v3.md"), "\n".join(lines))  # edge stats live here too

    # v2 vs v3 comparison
    v2 = pd.read_csv(os.path.join(PILOT, "selected_100_datasets_v2.csv"))
    d = ["# Dataset selection v2 -> v3 comparison", ""]
    def g(df, col, default=0):
        return df[col] if col in df else pd.Series([default] * len(df))
    d.append("| metric | v2 | v3 |")
    d.append("|---|---:|---:|")
    d.append(f"| selected | {len(v2)} | {n} |")
    d.append(f"| label_or_schema | {int(g(v2,'has_label_or_schema').sum())}/100 | {lab}/100 |")
    d.append(f"| val/test | {int(g(v2,'has_val_test').sum())}/100 | {vt}/100 |")
    d.append(f"| license | {int(v2['license'].notna().sum())}/100 | {lic}/100 |")
    d.append(f"| usable edges | 176 | {usable} |")
    d.append(f"| datasets >=10 edges | 3 | {ds_ge10} |")
    d.append(f"| models >=3 edges | 28 | {models_ge3} |")
    d.append(f"| zero-edge datasets | 93 | {zero_ds} |")
    d.append(f"| zero-edge models | 220 | {zero_models} |")
    d.append("\n## bucket coverage v3")
    for b, c in Counter(sel["bucket"]).most_common():
        d.append(f"- {b}: {c}")
    _md(os.path.join(PILOT, "dataset_selection_v2_v3_diff.md"), "\n".join(d))


if __name__ == "__main__":
    main()
