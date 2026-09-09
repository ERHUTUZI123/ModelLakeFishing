"""
hf_crawl.py -- T2: HF model-metadata collection.

Runbook: docs/1M/100kplan.md 5 (T2), docs/1M/T2.md.

TWO MODES, AND WHY BOTH EXIST
    --limit N            v1 single-stream: one ordering, first N records.
                         Kept verbatim so the frozen `raw/` 150K head shard
                         stays reproducible. It is an audit artifact and a
                         candidate source -- NOT a population definition.
                         With `--sort createdAt --limit <huge>` the same mode
                         performs the RF full enumeration: it then ends by
                         exhausting the cursor rather than by hitting --limit.
    --plan {pilot,full}  v2 multi-source candidate DISCOVERY (query_plan.py).
                         Over-collects from many orthogonal orderings; the
                         final population is chosen later, under quotas, by
                         select_balanced_halo.py.

    Discovery is allowed to be biased. Selection is not. Keeping them in one
    stage is what made the v1 output unusable (one publisher at 20.1%).

WHY expand[] AND NOT full=true
    `full=true&cardData=true` does NOT return `safetensors` (the only reliable
    parameter count) and DOES return `siblings` (the per-repo file list, pure
    bloat). It also silently omits `author`, `config`, `baseModels` and
    `lastModified`, i.e. four of the fields the v2 annotation depends on.

WHAT WE DELIBERATELY DO NOT DO HERE
    Nothing in this file interprets a field. Task/language/family/source-type
    annotation lives in taxonomy.py + annotate_candidates.py. A crawl snapshot
    cannot be rebuilt after the fact, so freeze the superset once and decide
    what to use later.

Run (from ModelLakeFishing/):
    .\\.venv\\Scripts\\python.exe -m scale1m.hf_crawl --plan pilot
    .\\.venv\\Scripts\\python.exe -m scale1m.hf_crawl --plan full
    .\\.venv\\Scripts\\python.exe -m scale1m.hf_crawl --limit 150000     # v1
"""

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import random
import shutil
import sys
import time
import urllib.parse

import requests

from scale1m.query_plan import build_plan
from scale1m.paths import data_root as _portable_data_root

API = "https://huggingface.co/api/models"

# v1 field set -- frozen, because `raw/` was crawled with exactly this.
EXPAND_V1 = (
    "downloads", "likes", "pipeline_tag", "library_name", "tags",
    "createdAt", "safetensors", "cardData",
)

# v2 adds the six fields the balanced selection cannot be built without:
#   author        -> publisher concentration control (Step 9)
#   baseModels    -> AUTHORITATIVE parent + relation (Step 4/10); replaces the
#                    cardData.base_model string, which has no relation type
#   config        -> architectures / model_type (Step 3/4, task fallback)
#   lastModified  -> freshness, distinct from createdAt
#   gguf          -> authoritative quantization flag (Step 5)
#   gated/disabled/private/trendingScore -> quality gate + recency signal
EXPAND_V2 = EXPAND_V1 + (
    "author", "baseModels", "config", "lastModified", "gguf",
    "gated", "disabled", "private", "trendingScore", "downloadsAllTime",
)

KEEP_V1 = ("id", "downloads", "likes", "pipeline_tag", "library_name", "tags",
           "createdAt")
KEEP_V2 = KEEP_V1 + ("author", "lastModified", "trendingScore",
                     "downloadsAllTime", "gated", "disabled", "private")

# v1 kept only these three; v2 adds `language` -- without it the language
# bucket has no authoritative source at all (measured: the v1 shards contain
# ZERO cardData.language values, because this list dropped them).
KEEP_CARD_V1 = ("base_model", "datasets", "model-index")
KEEP_CARD_V2 = KEEP_CARD_V1 + ("language", "license", "base_model_relation",
                               "tags", "pipeline_tag", "library_name")

PAGE_LIMIT_MAX = 1000
USER_AGENT = "model-lake-fishing/scale1m.hf_crawl (research crawl; contact via repo)"


def data_root() -> str:
    """Return ``MLF_DATA_DIR`` or the repository-local ``data`` directory."""
    return _portable_data_root()


def default_out() -> str:
    return os.path.join(data_root(), "data1m", "raw")


def default_candidates_out(tag="v2") -> str:
    return os.path.join(data_root(), "data1m", "candidates_" + tag)


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# --- record trimming -------------------------------------------------------


def trim_model_index(mi):
    """Flatten cardData['model-index'] to the (task, dataset, metrics) triples.

    Kept because T8's `displacement_quality` (100kplan 11.1) needs the measured
    accuracy of a *labeled* model that outranked the gold one. Dropped:
    `verifyToken` (a multi-KB JWT per metric), `source`, `verified`.
    """
    if isinstance(mi, dict):
        mi = [mi]
    if not isinstance(mi, list):
        return None
    out = []
    for entry in mi:
        if not isinstance(entry, dict):
            continue
        results = entry.get("results")
        if not isinstance(results, list):
            continue
        for res in results:
            if not isinstance(res, dict):
                continue
            task = res.get("task") if isinstance(res.get("task"), dict) else {}
            ds = res.get("dataset") if isinstance(res.get("dataset"), dict) else {}
            metrics = []
            for m in res.get("metrics") or []:
                if isinstance(m, dict):
                    metrics.append({"type": m.get("type"), "value": m.get("value")})
            out.append({"task": task.get("type"), "dataset": ds.get("type"),
                        "dataset_name": ds.get("name"), "config": ds.get("config"),
                        "split": ds.get("split"), "metrics": metrics})
    return out or None


def trim_base_models(bm):
    """`baseModels` -> {"relation": str, "ids": [str]}.

    This is the authoritative lineage signal: unlike `cardData.base_model`
    (a bare string an author typed) it carries HF's own relation type --
    quantized / adapter / finetune / merge -- which is exactly the ordering
    CLAUDE.md's lineage edge weights already use.
    """
    if not isinstance(bm, dict):
        return None
    ids = [m.get("id") for m in (bm.get("models") or []) if isinstance(m, dict) and m.get("id")]
    if not ids and not bm.get("relation"):
        return None
    return {"relation": bm.get("relation"), "ids": ids}


def trim_config(cfg):
    """Only the two structured fields we use; drops tokenizer_config bloat."""
    if not isinstance(cfg, dict):
        return None
    out = {}
    if cfg.get("architectures"):
        out["architectures"] = cfg["architectures"][:4]
    if cfg.get("model_type"):
        out["model_type"] = cfg["model_type"]
    return out or None


def trim(rec: dict, v2: bool = True) -> dict:
    """Full API record -> the shard record. Keys absent upstream stay absent."""
    keep = KEEP_V2 if v2 else KEEP_V1
    keep_card = KEEP_CARD_V2 if v2 else KEEP_CARD_V1
    out = {k: rec[k] for k in keep if k in rec and rec[k] is not None}

    st = rec.get("safetensors")
    if isinstance(st, dict) and st.get("total") is not None:
        out["safetensors"] = {"total": st["total"]}

    card = rec.get("cardData")
    if isinstance(card, dict):
        kept = {}
        for k in keep_card:
            if k not in card or card[k] is None:
                continue
            kept[k] = trim_model_index(card[k]) if k == "model-index" else card[k]
            if kept[k] is None:
                del kept[k]
        if kept:
            out["cardData"] = kept

    if v2:
        bm = trim_base_models(rec.get("baseModels"))
        if bm:
            out["baseModels"] = bm
        cfg = trim_config(rec.get("config"))
        if cfg:
            out["config"] = cfg
        if rec.get("gguf"):
            out["gguf"] = True
    return out


# --- shard / state io ------------------------------------------------------


def shard_stem(idx: int) -> str:
    return "hf_models_%05d" % idx


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: str, obj) -> None:
    tmp = path + ".tmp"
    if os.path.exists(path):
        os.chmod(path, 0o644)
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def finalize_shard(out_dir: str, idx: int, n_records: int, shards: list) -> None:
    """Plain .jsonl -> .jsonl.gz, sha256, read-only. Idempotent per shard."""
    plain = os.path.join(out_dir, shard_stem(idx) + ".jsonl")
    gz = os.path.join(out_dir, shard_stem(idx) + ".jsonl.gz")
    with open(plain, "rb") as src, gzip.open(gz, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst, length=1 << 20)
    os.remove(plain)
    os.chmod(gz, 0o444)
    shards.append({"file": os.path.basename(gz), "n_records": n_records,
                   "bytes": os.path.getsize(gz), "sha256": sha256_of(gz),
                   "finalized_at": utcnow()})


def replay_seen_ids(out_dir: str, shards: list, shard_idx: int, partial_lines: int):
    """Rebuild the dedupe set on resume, from the shards already on disk."""
    seen = set()
    for sh in shards:
        with gzip.open(os.path.join(out_dir, sh["file"]), "rt", encoding="utf-8") as fh:
            for line in fh:
                seen.add(json.loads(line)["id"])
    plain = os.path.join(out_dir, shard_stem(shard_idx) + ".jsonl")
    if os.path.exists(plain):
        kept = []
        with open(plain, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= partial_lines:
                    break
                kept.append(line)
        with open(plain, "w", encoding="utf-8") as fh:
            fh.writelines(kept)
        for line in kept:
            seen.add(json.loads(line)["id"])
    return seen


# --- http ------------------------------------------------------------------


def parse_ratelimit(headers):
    """`RateLimit: "api";r=493;t=67` -> (remaining, reset_seconds)."""
    raw = headers.get("RateLimit")
    if not raw:
        return None, None
    rem = reset = None
    for part in raw.split(";"):
        part = part.strip()
        if part.startswith("r="):
            rem = int(part[2:])
        elif part.startswith("t="):
            reset = int(part[2:])
    return rem, reset


def next_url(resp) -> str:
    """Cursor pagination lives in the Link header, rel="next"."""
    link = resp.headers.get("Link") or ""
    for part in link.split(","):
        if 'rel="next"' in part:
            return part.split(">")[0].split("<")[-1].strip()
    return ""


def build_url(params: dict, expand) -> str:
    pairs = []
    for k, v in params.items():
        if isinstance(v, (list, tuple)):
            pairs.extend((k, x) for x in v)
        else:
            pairs.append((k, v))
    pairs.extend(("expand[]", f) for f in expand)
    return API + "?" + urllib.parse.urlencode(pairs)


def fetch(session, url, timeout, max_retries, stats, min_remaining):
    """One page, with backoff. Returns (records, next_url)."""
    delay = 2.0
    for attempt in range(max_retries + 1):
        try:
            resp = session.get(url, timeout=timeout)
        except requests.RequestException as exc:
            if attempt == max_retries:
                raise
            stats["retries"] += 1
            stats["retry_reasons"][type(exc).__name__] = \
                stats["retry_reasons"].get(type(exc).__name__, 0) + 1
            time.sleep(delay + random.uniform(0, 1))
            delay = min(delay * 2, 60)
            continue

        if resp.status_code == 200:
            rem, reset = parse_ratelimit(resp.headers)
            stats["last_ratelimit"] = {"remaining": rem, "reset_s": reset}
            if rem is not None and reset is not None and rem < min_remaining:
                stats["ratelimit_sleeps"] += 1
                stats["ratelimit_sleep_s"] = stats.get("ratelimit_sleep_s", 0) + reset + 1
                print("      [ratelimit] %d left, sleeping %ds" % (rem, reset + 1), flush=True)
                time.sleep(reset + 1)
            return resp.json(), next_url(resp)

        if resp.status_code in (429, 500, 502, 503, 504):
            if attempt == max_retries:
                resp.raise_for_status()
            stats["retries"] += 1
            key = "http_%d" % resp.status_code
            stats["retry_reasons"][key] = stats["retry_reasons"].get(key, 0) + 1
            wait = delay
            ra = resp.headers.get("Retry-After")
            if ra:
                try:
                    wait = float(ra)
                except ValueError:
                    pass
            else:
                _, reset = parse_ratelimit(resp.headers)
                if resp.status_code == 429 and reset:
                    wait = reset + 1
            time.sleep(wait + random.uniform(0, 1))
            delay = min(delay * 2, 60)
            continue

        # 400 on an unsupported filter is a plan bug, not a transient error.
        resp.raise_for_status()
    raise RuntimeError("unreachable")


def make_session(token):
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    if token:
        session.headers["Authorization"] = "Bearer " + token
    return session


def new_stats():
    return {"pages": 0, "retries": 0, "retry_reasons": {}, "ratelimit_sleeps": 0,
            "duplicates_skipped": 0, "started_at": utcnow()}


# =========================================================================
# mode 1: v1 single-stream (frozen behaviour)
# =========================================================================


def crawl_single(args) -> int:
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    cursor_path = os.path.join(out_dir, "CURSOR.json")
    shards_path = os.path.join(out_dir, "SHARDS.json")
    prov_path = os.path.join(out_dir, "PROVENANCE.json")

    state = load_json(cursor_path) if not args.restart else None
    shards = load_json(shards_path, []) if not args.restart else []
    if args.restart:
        stale = [f for f in os.listdir(out_dir) if f.startswith("hf_models_")]
        if stale:
            print("[abort] --restart but %d shard file(s) still in %s; they are "
                  "read-only frozen snapshots -- move or delete them by hand."
                  % (len(stale), out_dir), file=sys.stderr)
            return 2

    expand = EXPAND_V2 if args.v2_fields else EXPAND_V1
    sort, direction = args.sort, str(args.direction)
    start_url = build_url({"limit": str(min(args.page_size, PAGE_LIMIT_MAX)),
                           "sort": sort, "direction": direction}, expand)

    if state:
        # The stream order is baked into the stored cursor. Resuming a
        # createdAt enumeration with --sort downloads would silently splice two
        # different orderings into one shard set.
        prev = (state.get("sort", "downloads"), str(state.get("direction", "-1")))
        if prev != (sort, direction):
            print("[abort] %s holds a %s/%s stream; this run asks for %s/%s. "
                  "Use a different --out or --restart."
                  % (out_dir, prev[0], prev[1], sort, direction), file=sys.stderr)
            return 2
        url, n_written, n_in_shard = state["next_url"], state["n_written"], state["n_in_shard"]
        stats = state.get("stats", new_stats())
        print("[resume] %d records already written, shard %d, %d in flight"
              % (n_written, len(shards), n_in_shard))
    else:
        url, n_written, n_in_shard, stats = start_url, 0, 0, new_stats()
    stats.setdefault("retry_reasons", {})

    if n_written >= args.limit:
        print("[done] already have %d >= --limit %d; nothing to do" % (n_written, args.limit))
        write_provenance(args, out_dir, prov_path, shards, stats, n_written,
                         expand, mode="single-stream")
        return 0

    seen = (replay_seen_ids(out_dir, shards, state["shard_index"], n_in_shard)
            if state else set())
    session = make_session(args.token or os.environ.get("HF_TOKEN"))
    stats["authenticated"] = bool(args.token or os.environ.get("HF_TOKEN"))

    shard_idx = len(shards)
    plain_path = os.path.join(out_dir, shard_stem(shard_idx) + ".jsonl")
    fh = open(plain_path, "a", encoding="utf-8")
    t0 = time.time()
    try:
        while n_written < args.limit and url:
            recs, url = fetch(session, url, args.timeout, args.max_retries,
                              stats, args.min_ratelimit_remaining)
            stats["pages"] += 1
            if not recs:
                print("[end] API returned an empty page after %d records" % n_written)
                break
            for rec in recs:
                mid = rec.get("id")
                if mid is None:
                    continue
                if mid in seen:
                    stats["duplicates_skipped"] += 1
                    continue
                seen.add(mid)
                fh.write(json.dumps(trim(rec, v2=args.v2_fields), ensure_ascii=False) + "\n")
                n_written += 1
                n_in_shard += 1
                if "first_id" not in stats:
                    # The newest record in the stream is the snapshot's upper
                    # boundary: anything created after this instant is, by
                    # construction, not in this crawl.
                    stats["first_id"] = mid
                    stats["first_created_at"] = rec.get("createdAt")
                    stats["first_record_at"] = utcnow()
                stats["last_id"] = mid
                stats["last_downloads"] = rec.get("downloads")
                stats["last_created_at"] = rec.get("createdAt")
                stats["last_sort_value"] = rec.get(sort)
                if n_in_shard >= args.shard_size or n_written >= args.limit:
                    fh.close()
                    finalize_shard(out_dir, shard_idx, n_in_shard, shards)
                    write_json_atomic(shards_path, shards)
                    shard_idx += 1
                    n_in_shard = 0
                    plain_path = os.path.join(out_dir, shard_stem(shard_idx) + ".jsonl")
                    fh = open(plain_path, "a", encoding="utf-8")
                    if n_written >= args.limit:
                        break
            fh.flush()
            os.fsync(fh.fileno())
            write_json_atomic(cursor_path, {"next_url": url, "n_written": n_written,
                                            "n_in_shard": n_in_shard, "shard_index": shard_idx,
                                            "sort": sort, "direction": direction,
                                            "stats": stats, "updated_at": utcnow()})
            print("[page %5d] n=%7d  shard=%d  last=%s (%s=%s)  %.0f rec/s"
                  % (stats["pages"], n_written, shard_idx, stats.get("last_id"),
                     sort, stats.get("last_sort_value"),
                     n_written / max(time.time() - t0, 1e-9)),
                  flush=True)
    finally:
        if not fh.closed:
            fh.flush()
            fh.close()

    if n_in_shard > 0 and os.path.exists(plain_path):
        finalize_shard(out_dir, shard_idx, n_in_shard, shards)
        write_json_atomic(shards_path, shards)
        n_in_shard, shard_idx = 0, shard_idx + 1
    elif os.path.exists(plain_path) and os.path.getsize(plain_path) == 0:
        os.remove(plain_path)

    stats["wallclock_s"] = round(time.time() - t0, 1)
    stats["finished_at"] = utcnow()
    stats["exhausted_cursor"] = not url
    write_json_atomic(cursor_path, {"next_url": url, "n_written": n_written,
                                    "n_in_shard": n_in_shard, "shard_index": shard_idx,
                                    "sort": sort, "direction": direction,
                                    "stats": stats, "updated_at": utcnow()})
    write_provenance(args, out_dir, prov_path, shards, stats, n_written, expand,
                     mode="single-stream")
    print("\n[ok] %d records in %d shards -> %s%s"
          % (n_written, len(shards), out_dir,
             "  (cursor exhausted)" if not url else ""))
    # A full enumeration ends by running out of cursor, not by hitting --limit;
    # both are successful terminations.
    return 0 if (n_written >= args.limit or not url) else 1


# =========================================================================
# mode 2: v2 multi-query candidate discovery
# =========================================================================


def membership_path(out_dir, qid, gz=False):
    safe = qid.replace("/", "__").replace(":", "_")
    return os.path.join(out_dir, "membership", safe + (".tsv.gz" if gz else ".tsv"))


def crawl_plan(args) -> int:
    """Run every query in the plan; write each unique model ONCE, with merged
    provenance recorded per query in membership/.

    Resume granularity is per query AND per page: PLAN_STATE.json holds each
    query's cursor and committed hit count, so a killed run loses at most the
    records of one page.
    """
    out_dir = args.out
    os.makedirs(os.path.join(out_dir, "membership"), exist_ok=True)
    state_path = os.path.join(out_dir, "PLAN_STATE.json")
    shards_path = os.path.join(out_dir, "SHARDS.json")
    prov_path = os.path.join(out_dir, "PROVENANCE.json")

    if args.backfill_deficits:
        from scale1m.query_plan import build_backfill_plan
        with open(args.backfill_deficits, "r", encoding="utf-8") as fh:
            deficits = json.load(fh)
        plan = build_backfill_plan(deficits.get("deficits", deficits), scale=args.scale)
    elif args.plan == "deep":
        from scale1m.query_plan import build_deep_plan
        plan = build_deep_plan(scale=args.scale)
    else:
        plan = build_plan(name=args.plan, scale=args.scale)

    state = load_json(state_path, {}) or {}
    shards = load_json(shards_path, [])
    queries_done = state.get("queries_done", {})
    n_written = state.get("n_written", 0)
    n_in_shard = state.get("n_in_shard", 0)
    shard_idx = state.get("shard_index", len(shards))
    stats = state.get("stats", new_stats())
    stats.setdefault("retry_reasons", {})
    stats.setdefault("by_source", {})

    seen = replay_seen_ids(out_dir, shards, shard_idx, n_in_shard) if state else set()
    if state:
        print("[resume] %d unique records, %d/%d queries done"
              % (n_written, len(queries_done), len(plan.queries)))

    session = make_session(args.token or os.environ.get("HF_TOKEN"))
    stats["authenticated"] = bool(args.token or os.environ.get("HF_TOKEN"))
    expand = EXPAND_V2

    plain_path = os.path.join(out_dir, shard_stem(shard_idx) + ".jsonl")
    fh = open(plain_path, "a", encoding="utf-8")
    t0 = time.time()

    def flush_state(qid=None, cursor=None, n_hits=None):
        st = dict(state)
        st.update({"n_written": n_written, "n_in_shard": n_in_shard,
                   "shard_index": shard_idx, "stats": stats,
                   "queries_done": queries_done, "plan": plan.name,
                   "scale": args.scale, "updated_at": utcnow()})
        if qid is not None:
            st.setdefault("in_flight", {})[qid] = {"cursor": cursor, "n_hits": n_hits}
        write_json_atomic(state_path, st)
        return st

    try:
        for qi, q in enumerate(plan.queries):
            if q.qid in queries_done:
                continue
            inflight = (state.get("in_flight") or {}).get(q.qid) or {}
            url = inflight.get("cursor") or build_url(q.url_params(), expand)
            n_hits = inflight.get("n_hits", 0) if inflight.get("cursor") else 0

            mpath = membership_path(out_dir, q.qid)
            if n_hits == 0 and os.path.exists(mpath):
                os.remove(mpath)          # restart this query cleanly
            elif os.path.exists(mpath):   # truncate a torn tail
                with open(mpath, "r", encoding="utf-8") as m:
                    lines = [l for i, l in enumerate(m) if i < n_hits]
                with open(mpath, "w", encoding="utf-8") as m:
                    m.writelines(lines)
            mfh = open(mpath, "a", encoding="utf-8")

            q_new = 0
            try:
                while url and n_hits < q.cap:
                    recs, url = fetch(session, url, args.timeout, args.max_retries,
                                      stats, args.min_ratelimit_remaining)
                    stats["pages"] += 1
                    if not recs:
                        break
                    for rec in recs:
                        mid = rec.get("id")
                        if mid is None:
                            continue
                        n_hits += 1
                        mfh.write("%d\t%s\n" % (n_hits, mid))
                        if mid in seen:
                            stats["duplicates_skipped"] += 1
                        else:
                            seen.add(mid)
                            row = trim(rec, v2=True)
                            row["_prov"] = {"first_seen_query": q.qid,
                                            "discovery_rank": n_hits,
                                            "discovery_source": q.source,
                                            "crawl_timestamp": utcnow()}
                            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                            n_written += 1
                            n_in_shard += 1
                            q_new += 1
                            stats["by_source"][q.source] = stats["by_source"].get(q.source, 0) + 1
                            if n_in_shard >= args.shard_size:
                                fh.close()
                                finalize_shard(out_dir, shard_idx, n_in_shard, shards)
                                write_json_atomic(shards_path, shards)
                                shard_idx += 1
                                n_in_shard = 0
                                fh = open(os.path.join(out_dir, shard_stem(shard_idx) + ".jsonl"),
                                          "a", encoding="utf-8")
                        if n_hits >= q.cap:
                            break
                    fh.flush()
                    mfh.flush()
                    flush_state(q.qid, url, n_hits)
            finally:
                mfh.close()

            queries_done[q.qid] = {"hits": n_hits, "new": q_new, "source": q.source,
                                   "cap": q.cap, "exhausted": not url,
                                   "finished_at": utcnow()}
            if (state.get("in_flight") or {}).pop(q.qid, None) is not None:
                pass
            state = flush_state()
            print("[%3d/%3d] %-46s hits=%6d new=%6d uniq=%7d %s"
                  % (qi + 1, len(plan.queries), q.qid, n_hits, q_new, n_written,
                     "(exhausted)" if not url else ""), flush=True)
    finally:
        if not fh.closed:
            fh.flush()
            fh.close()

    if n_in_shard > 0 and os.path.exists(os.path.join(out_dir, shard_stem(shard_idx) + ".jsonl")):
        finalize_shard(out_dir, shard_idx, n_in_shard, shards)
        write_json_atomic(shards_path, shards)
        n_in_shard, shard_idx = 0, shard_idx + 1
    else:
        stale = os.path.join(out_dir, shard_stem(shard_idx) + ".jsonl")
        if os.path.exists(stale) and os.path.getsize(stale) == 0:
            os.remove(stale)

    # membership -> gz, read-only (they are part of the frozen snapshot)
    for qid in queries_done:
        p, g = membership_path(out_dir, qid), membership_path(out_dir, qid, gz=True)
        if os.path.exists(p):
            with open(p, "rb") as src, gzip.open(g, "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
            os.remove(p)
            os.chmod(g, 0o444)

    stats["wallclock_s"] = round(time.time() - t0, 1)
    stats["finished_at"] = utcnow()
    flush_state()
    write_provenance(args, out_dir, prov_path, shards, stats, n_written, expand,
                     mode="plan:" + plan.name, queries_done=queries_done,
                     plan_size=len(plan.queries))
    total_hits = sum(v["hits"] for v in queries_done.values())
    print("\n[ok] %d UNIQUE records from %d queries (%d raw hits, %.1fx overlap)"
          % (n_written, len(queries_done), total_hits, total_hits / max(n_written, 1)))
    print("     %d shards, %d pages, %.1fs, %d retries, %d ratelimit sleeps"
          % (len(shards), stats["pages"], stats["wallclock_s"], stats["retries"],
             stats["ratelimit_sleeps"]))
    return 0


def write_provenance(args, out_dir, prov_path, shards, stats, n_written, expand,
                     mode, queries_done=None, plan_size=None) -> None:
    """The snapshot date is the one irreproducible input -- record it hard."""
    import huggingface_hub
    prov = {
        "artifact": "hf model metadata crawl (T2)",
        "mode": mode,
        "runbook": "docs/1M/100kplan.md 5",
        "snapshot_date_utc": (stats.get("finished_at") or utcnow())[:10],
        "started_at": stats.get("started_at"),
        "finished_at": stats.get("finished_at"),
        "endpoint": API,
        "expand": list(expand),
        "kept_top_level": list(KEEP_V2 if args.v2_fields or queries_done else KEEP_V1),
        "kept_card_data": list(KEEP_CARD_V2 if args.v2_fields or queries_done else KEEP_CARD_V1),
        "page_size": min(args.page_size, PAGE_LIMIT_MAX),
        "shard_size": args.shard_size,
        "authenticated": stats.get("authenticated"),
        "total_records": n_written,
        "pages": stats.get("pages"),
        "retries": stats.get("retries"),
        "retry_reasons": stats.get("retry_reasons"),
        "ratelimit_sleeps": stats.get("ratelimit_sleeps"),
        "ratelimit_sleep_s": stats.get("ratelimit_sleep_s", 0),
        "duplicates_skipped": stats.get("duplicates_skipped"),
        "wallclock_s": stats.get("wallclock_s"),
        "shards": shards,
        "python": sys.version.split()[0],
        "requests": requests.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "written_at": utcnow(),
    }
    if queries_done is not None:
        prov["plan_queries"] = plan_size
        prov["plan_scale"] = args.scale
        prov["queries"] = queries_done
        prov["new_by_source"] = stats.get("by_source")
        prov["raw_hits"] = sum(v["hits"] for v in queries_done.values())
    else:
        prov["requested_limit"] = args.limit
        prov["sort"] = args.sort
        prov["direction"] = str(args.direction)
        prov["exhausted_cursor"] = stats.get("exhausted_cursor")
        prov["first_id"] = stats.get("first_id")
        prov["first_created_at"] = stats.get("first_created_at")
        prov["first_record_at"] = stats.get("first_record_at")
        prov["last_id"] = stats.get("last_id")
        prov["last_downloads"] = stats.get("last_downloads")
        prov["last_created_at"] = stats.get("last_created_at")
        # The snapshot is a window, not an instant: the stream is ordered by
        # createdAt desc, so models created after `first_created_at` are absent
        # and models deleted during the window simply vanish from it.
        prov["snapshot_window_utc"] = {
            "crawl_started_at": stats.get("started_at"),
            "crawl_finished_at": stats.get("finished_at"),
            "newest_model_created_at": stats.get("first_created_at"),
            "oldest_model_created_at": stats.get("last_created_at"),
        }
    write_json_atomic(prov_path, prov)
    os.chmod(prov_path, 0o444)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HF model metadata collection (T2)")
    p.add_argument("--plan", choices=("pilot", "full", "deep"), default=None,
                   help="v2 multi-source candidate discovery (see query_plan.py)")
    p.add_argument("--scale", type=float, default=None,
                   help="shrink every plan cap (default 0.08 for pilot, 1.0 for full)")
    p.add_argument("--backfill-deficits", default=None,
                   help="deficit JSON from select_balanced_halo -> targeted plan")
    p.add_argument("--limit", type=int, default=150_000,
                   help="v1 single-stream mode: total records to hold after this run")
    p.add_argument("--sort", default="downloads",
                   choices=("downloads", "createdAt", "likes", "lastModified",
                            "trendingScore"),
                   help="single-stream sort key. Default `downloads` keeps the v1 "
                        "snapshot reproducible. Use `createdAt` for full "
                        "enumeration: its cursor is keyed on the immutable _id, "
                        "so nothing is skipped mid-crawl (see docs/1M/F0.md)")
    p.add_argument("--direction", default="-1", choices=("-1", "1"),
                   help="-1 descending (default), 1 ascending")
    p.add_argument("--out", default=None)
    p.add_argument("--v2-fields", action="store_true",
                   help="single-stream mode: use the richer v2 expand/KEEP sets")
    p.add_argument("--shard-size", type=int, default=50_000)
    p.add_argument("--page-size", type=int, default=PAGE_LIMIT_MAX)
    p.add_argument("--token", default=None, help="defaults to $HF_TOKEN; optional")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--max-retries", type=int, default=8)
    p.add_argument("--min-ratelimit-remaining", type=int, default=25)
    p.add_argument("--restart", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.plan or args.backfill_deficits:
        if args.scale is None:
            args.scale = 0.08 if args.plan == "pilot" else 1.0
        if args.out is None:
            args.out = default_candidates_out("pilot" if args.plan == "pilot" else "v2")
        args.plan = args.plan or "backfill"
        args.v2_fields = True
        return crawl_plan(args)

    if args.out is None:
        args.out = default_out()
    args.scale = args.scale or 1.0
    return crawl_single(args)


if __name__ == "__main__":
    raise SystemExit(main())
