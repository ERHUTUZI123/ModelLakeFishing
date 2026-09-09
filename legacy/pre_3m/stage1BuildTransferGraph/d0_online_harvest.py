"""
d0_online_harvest.py -- D0 lake expansion, Phase B: ONLINE dataset-first top-up.

Phase A (d0_intake_audit.py) showed the offline pool tops out at ~130 distinct
gold ROOT datasets -- not enough task surface. User decision (2026-07-18): the
12.7K offline model pool is expendable; what must rise substantially is the
DATASET surface and the models tightly coupled to those datasets (eval-bearing
neighborhood). This script harvests exactly that, dataset-first:

  seed     : candidate datasets = (a) label-bearing roots from Phase A that lack
             cached metadata, (b) HF task-category sweeps (top-N by downloads),
             (c) benchmark name searches. For every candidate, list its model
             neighborhood (filter=dataset:<id>, cardData=true) and count models
             whose card carries a model-index -- WITHOUT full fetches.
  datasets : fetch full /api/datasets/<id> JSON for kept candidates.
  models   : fetch full /api/models/<id> JSON for the eval-bearing neighborhood
             (cap per dataset), skipping anything already in the Phase-A cache.

Everything lands in d0_lake_cache/{list,datasets,models}/ -- the frozen
hf1000d_2000m/hf_cache/ is never written to. Resumable: every list/fetch is
cached on disk; re-running skips completed work. METADATA ONLY, no weights.

Run (from stage1BuildTransferGraph/):
    ../.venv/Scripts/python.exe d0_online_harvest.py --phase seed
    ../.venv/Scripts/python.exe d0_online_harvest.py --phase datasets
    ../.venv/Scripts/python.exe d0_online_harvest.py --phase models
    ../.venv/Scripts/python.exe d0_online_harvest.py --phase all
"""

import argparse
import json
import os
import re
import sys
import time

import pandas as pd
import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from build_hf_pilot_v3 import canon_key, KEY_TO_HFID  # noqa: E402

API = "https://huggingface.co/api"
CACHE = os.path.join(_HERE, "d0_lake_cache")
LDIR = os.path.join(CACHE, "list")
DDIR = os.path.join(CACHE, "datasets")
MDIR = os.path.join(CACHE, "models")
OLD_MDIR = os.path.join(_HERE, "hf1000d_2000m", "hf_cache", "models")
OLD_DDIR = os.path.join(_HERE, "hf1000d_2000m", "hf_cache", "datasets")
ART = os.path.join(_HERE, "artifacts", "d0_lake")
SEED_CSV = os.path.join(ART, "d0_harvest_seed.csv")
REPORT = os.path.join(ART, "d0_harvest_report.json")

SLEEP = 0.05
LIST_LIMIT = 100          # models listed per dataset query
MIN_EVAL_KEEP = 2         # dataset kept if >= this many eval-bearing models listed
PER_DS_FETCH_CAP = 40     # full model fetches per dataset (eval-bearing first)
TOP_PER_CATEGORY = 300    # datasets per task-category sweep

# v5 P3 (D0.5 supervision-surface expansion): both knobs are overridable from
# the CLI so the SAME fully-cached seed scan (12,153 list calls) can be
# re-derived under looser retention — only genuinely new /api fetches cost
# network. Defaults reproduce the v4 lake exactly.
P3_SEED_CSV = os.path.join(ART, "d0_harvest_seed_p3.csv")

TASK_CATEGORIES = [
    "text-classification", "token-classification", "question-answering",
    "summarization", "translation", "sentence-similarity", "text-retrieval",
    "multiple-choice", "text-generation", "text2text-generation",
    "zero-shot-classification", "fill-mask", "feature-extraction",
    "table-question-answering",
]
BENCH_SEARCHES = [
    "glue", "superglue", "squad", "xnli", "anli", "paws", "conll", "wmt",
    "mteb", "tweet_eval", "banking77", "massive", "ag_news", "imdb", "xsum",
    "cnn_dailymail", "wikiann", "ontonotes", "hellaswag", "winogrande",
    "commonsense_qa", "openbookqa", "boolq", "race", "hotpot", "trivia_qa",
    "natural_questions", "sst2", "emotion", "hate", "sentiment", "stsb",
    "nli", "reranking", "retrieval", "ner", "pos tagging", "intent",
]


def _get(url, params=None):
    for attempt in range(3):
        try:
            r = requests.get(url, params=params or {}, timeout=30)
            if r.status_code == 200:
                time.sleep(SLEEP)
                return r.json()
            if r.status_code in (404, 401, 403):
                return None
        except Exception:
            pass
        time.sleep(1.0 + attempt)
    return None


def _cached(tag, fn, subdir=LDIR):
    os.makedirs(subdir, exist_ok=True)
    p = os.path.join(subdir, re.sub(r"[^a-zA-Z0-9._-]", "_", tag)[:150] + ".json")
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            pass
    out = fn()
    json.dump(out, open(p, "w", encoding="utf-8"))
    return out


def fetch_full(kind, rid, cache_dir):
    """Full raw JSON for one model/dataset, cached. Returns (json|None, was_new)."""
    safe = rid.replace("/", "__")
    p = os.path.join(cache_dir, safe + ".json")
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8")), False
        except Exception:
            pass
    j = _get(f"{API}/{kind}/{rid}", {"full": "true"})
    if j is None:
        return None, False
    os.makedirs(cache_dir, exist_ok=True)
    json.dump(j, open(p, "w", encoding="utf-8"))
    return j, True


def list_models_for_dataset(ds_id):
    """List model neighborhood of a dataset (cardData included), cached.
    Queries the full id and, if sparse, also the bare last component (models
    frequently tag `dataset:imdb` rather than `dataset:stanfordnlp/imdb`)."""
    def one(fid):
        return _cached("m__ds__" + fid.replace("/", "__"), lambda: _get(
            f"{API}/models",
            {"filter": f"dataset:{fid}", "sort": "downloads", "direction": -1,
             "limit": LIST_LIMIT, "cardData": "true"}) or [])
    rows = one(ds_id)
    bare = ds_id.split("/")[-1]
    if bare != ds_id and len(rows) < 20:
        seen = {m.get("id") for m in rows}
        rows = rows + [m for m in one(bare) if m.get("id") not in seen]
    return rows


def _has_mi(m):
    cd = m.get("cardData") or {}
    return bool((isinstance(cd, dict) and cd.get("model-index")) or m.get("model-index"))


# ── phase seed ────────────────────────────────────────────────────────────────
def offline_seed_roots():
    """Label-bearing dataset roots from Phase A that lack cached metadata."""
    pool = pd.read_csv(os.path.join(ART, "d0_dataset_pool.csv"))
    pool = pool[pool["dataset_node"].notna()]
    missing = pool[~pool["in_dataset_cache"]]
    return sorted(set(missing["dataset_node"].str.split("/").str[0]))


def resolve_root(root):
    """canonical root name -> HF dataset id: known mapping, direct id probes,
    then search with relaxed (exact-then-prefix) matching."""
    known = KEY_TO_HFID.get(root)
    if known:
        return known
    for probe in (root, f"mteb/{root}"):
        hit = _cached("probe__" + probe.replace("/", "__"),
                      lambda p=probe: _get(f"{API}/datasets/{p}") or {})
        if hit.get("id"):
            return hit["id"]
    hits = _cached("resolve__" + root, lambda: _get(
        f"{API}/datasets", {"search": root, "limit": 10}) or [])
    exact, prefix = None, None
    for h in hits:
        hid = h.get("id", "")
        last = hid.split("/")[-1].lower()
        dl = h.get("downloads") or 0
        if canon_key(hid) == root or last == root:
            if exact is None or dl > exact[1]:
                exact = (hid, dl)
        elif last.startswith(root) or root.startswith(last):
            if prefix is None or dl > prefix[1]:
                prefix = (hid, dl)
    return (exact or prefix or (None,))[0]


def phase_seed(min_keep=MIN_EVAL_KEEP, seed_csv=SEED_CSV):
    cands = {}  # hf_id -> source

    for cat in TASK_CATEGORIES:
        rows = _cached("ds__cat__" + cat, lambda c=cat: _get(
            f"{API}/datasets", {"filter": f"task_categories:{c}",
                                "sort": "downloads", "direction": -1,
                                "limit": TOP_PER_CATEGORY}) or [])
        for d in rows:
            cands.setdefault(d["id"], "category:" + cat)
        print(f"[seed] category {cat}: +{len(rows)} (pool {len(cands)})", flush=True)

    for q in BENCH_SEARCHES:
        rows = _cached("ds__search2__" + q.replace(" ", "_"), lambda s=q: _get(
            f"{API}/datasets", {"search": s, "sort": "downloads",
                                "direction": -1, "limit": 100}) or [])
        for d in rows:
            cands.setdefault(d["id"], "search:" + q)
    print(f"[seed] after benchmark searches: {len(cands)}", flush=True)

    for root in offline_seed_roots():
        rid = resolve_root(root)
        if rid:
            cands[rid] = "phaseA_missing"   # override: these are must-keep
    print(f"[seed] after Phase-A missing-root resolution: {len(cands)}", flush=True)

    # neighborhood scan: one cached list call per candidate
    out, t0 = [], time.time()
    for i, (rid, src) in enumerate(sorted(cands.items())):
        rows = list_models_for_dataset(rid)
        n_eval = sum(1 for m in rows if _has_mi(m))
        out.append({"hf_id": rid, "source": src, "canon": canon_key(rid),
                    "n_listed": len(rows), "n_eval_bearing": n_eval,
                    "keep": n_eval >= min_keep or src == "phaseA_missing"})
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"[seed] scanned {i+1}/{len(cands)} ({el:.0f}s) "
                  f"kept so far {sum(o['keep'] for o in out)}", flush=True)
    df_raw = pd.DataFrame(out).sort_values(["keep", "n_eval_bearing"],
                                           ascending=[False, False])
    df_raw["root"] = df_raw["canon"].str.split("/").str[0]
    df_raw.to_csv(seed_csv.replace(".csv", "_raw.csv"), index=False)
    # dedup canonically identical datasets (org/id vs bare id): keep max coverage
    df = df_raw.drop_duplicates("canon", keep="first")
    df.to_csv(seed_csv, index=False)
    print(f"[seed] DONE(min_keep={min_keep}): raw={len(df_raw)} canonical={len(df)} "
          f"roots={df['root'].nunique()} keep={int(df['keep'].sum())} "
          f"keep_roots={df[df['keep']]['root'].nunique()} -> {seed_csv}", flush=True)


# ── phase datasets ────────────────────────────────────────────────────────────
def phase_datasets(seed_csv=SEED_CSV):
    df = pd.read_csv(seed_csv)
    keep = df[df["keep"]]
    have = {f[:-5] for f in os.listdir(OLD_DDIR)} if os.path.isdir(OLD_DDIR) else set()
    n_new = n_fail = 0
    for i, rid in enumerate(keep["hf_id"]):
        if rid.replace("/", "__") in have:
            continue
        j, new = fetch_full("datasets", rid, DDIR)
        n_new += int(new)
        n_fail += int(j is None)
        if (i + 1) % 200 == 0:
            print(f"[datasets] {i+1}/{len(keep)} new={n_new} fail={n_fail}", flush=True)
    print(f"[datasets] DONE: new={n_new} fail={n_fail}", flush=True)


# ── phase models ──────────────────────────────────────────────────────────────
def phase_models(seed_csv=SEED_CSV, fetch_cap=PER_DS_FETCH_CAP):
    df = pd.read_csv(seed_csv)
    keep = df[df["keep"]].sort_values("n_eval_bearing", ascending=False)
    have_old = {f[:-5] for f in os.listdir(OLD_MDIR)} if os.path.isdir(OLD_MDIR) else set()
    planned, n_new, n_fail = set(), 0, 0
    t0 = time.time()
    for i, rid in enumerate(keep["hf_id"]):
        rows = list_models_for_dataset(rid)   # cached from seed phase
        ev = [m for m in rows if _has_mi(m)][:fetch_cap]
        for m in ev:
            mid = m.get("id")
            if not mid or mid in planned:
                continue
            planned.add(mid)
            if mid.replace("/", "__") in have_old:
                continue                       # already in Phase-A cache
            j, new = fetch_full("models", mid, MDIR)
            n_new += int(new)
            n_fail += int(j is None)
        if (i + 1) % 100 == 0:
            print(f"[models] ds {i+1}/{len(keep)} planned={len(planned)} "
                  f"new={n_new} fail={n_fail} ({time.time()-t0:.0f}s)", flush=True)
    report = {"datasets_kept": int(len(keep)), "models_planned": len(planned),
              "models_fetched_new": n_new, "model_fetch_failures": n_fail,
              "cache_dir": CACHE}
    json.dump(report, open(REPORT, "w"), indent=2)
    print(f"[models] DONE: {report}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["seed", "datasets", "models", "all"],
                    required=True)
    ap.add_argument("--min-keep", type=int, default=MIN_EVAL_KEEP,
                    help="v5 P3: dataset retention threshold (v4 default 2; P3=1)")
    ap.add_argument("--fetch-cap", type=int, default=PER_DS_FETCH_CAP,
                    help="v5 P3: full model fetches per dataset (v4 default 40; P3=100)")
    ap.add_argument("--seed-csv", default=None,
                    help="seed manifest path (P3 uses d0_harvest_seed_p3.csv)")
    a = ap.parse_args()
    seed_csv = a.seed_csv or (P3_SEED_CSV if a.min_keep != MIN_EVAL_KEEP else SEED_CSV)
    if a.phase in ("seed", "all"):
        phase_seed(min_keep=a.min_keep, seed_csv=seed_csv)
    if a.phase in ("datasets", "all"):
        phase_datasets(seed_csv=seed_csv)
    if a.phase in ("models", "all"):
        phase_models(seed_csv=seed_csv, fetch_cap=a.fetch_cap)
