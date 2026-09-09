"""
hf_crawl_datasets.py -- F1.5: the dataset-side snapshot.

WHY THIS EXISTS
    `x_d` has been `[e_name 64 || e_card 384 || e_stats 10]` since D0, and
    `e_card` is MiniLM over a dataset descriptor built from the HF dataset
    card -- name, task_categories, tags, description (see
    scale1m/dataset_descriptor.py). The
    ModelLens rungs took that text from the corpus's own `desc` column. With
    the corpus gone (D-56), the same text has to come from where D0 originally
    took it: the HF datasets API.

WHY ENUMERATE INSTEAD OF FETCHING THE 7,700 BY ID
    Anonymous rate limit is 500 requests / 300 s. 7,700 per-id GETs cost about
    16 windows (over an hour); enumerating the whole dataset index at 1000 per
    page costs about 980 requests (2 windows, ~10 min) and additionally tells
    us which of our names are not HF repos at all. It also keeps the
    closed-world property: one snapshot, one source.

    Ordering is `createdAt` desc for the same reason as the model crawl: its
    pagination cursor is keyed on the immutable `_id`, so nothing already in
    the index is skipped mid-crawl (docs/1M/F0.md 2.2).

Run (from ModelLakeFishing/):
    python -m scale1m.hf_crawl_datasets --out $MLF_DATA_DIR/data1m/datasets_full
"""
import argparse
import json
import os
import sys
import time
import urllib.parse

import requests

from scale1m.hf_crawl import (PAGE_LIMIT_MAX, data_root, fetch, finalize_shard,
                              load_json, make_session, new_stats, replay_seen_ids,
                              shard_stem, utcnow, write_json_atomic)

API = "https://huggingface.co/api/datasets"

# Exactly the fields dataset_descriptor() reads, plus the popularity/recency
# signals the stats view may use. No file listings.
EXPAND = ("author", "createdAt", "lastModified", "downloads", "likes",
          "tags", "description", "cardData")

KEEP = ("id", "author", "createdAt", "lastModified", "downloads", "likes", "tags")
KEEP_CARD = ("task_categories", "task_ids", "language", "size_categories",
             "license", "multilinguality", "source_datasets")
DESC_CHARS = 1000          # the descriptor uses 400; keep headroom, drop the rest


def build_url(params):
    pairs = list(params.items()) + [("expand[]", f) for f in EXPAND]
    return API + "?" + urllib.parse.urlencode(pairs)


def trim(rec):
    out = {k: rec[k] for k in KEEP if rec.get(k) is not None}
    d = rec.get("description")
    if isinstance(d, str) and d.strip():
        out["description"] = d[:DESC_CHARS]
    card = rec.get("cardData")
    if isinstance(card, dict):
        kept = {k: card[k] for k in KEEP_CARD
                if card.get(k) is not None and card.get(k) != []}
        if kept:
            out["cardData"] = kept
    return out


def crawl(args) -> int:
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    cursor_path = os.path.join(out_dir, "CURSOR.json")
    shards_path = os.path.join(out_dir, "SHARDS.json")
    prov_path = os.path.join(out_dir, "PROVENANCE.json")

    state = load_json(cursor_path) if not args.restart else None
    shards = load_json(shards_path, []) if not args.restart else []
    start = build_url({"limit": str(min(args.page_size, PAGE_LIMIT_MAX)),
                       "sort": "createdAt", "direction": "-1"})
    if state:
        url, n_written, n_in_shard = state["next_url"], state["n_written"], state["n_in_shard"]
        stats = state.get("stats", new_stats())
        seen = replay_seen_ids(out_dir, shards, state["shard_index"], n_in_shard)
        print("[resume] %d records already written" % n_written)
    else:
        url, n_written, n_in_shard, stats, seen = start, 0, 0, new_stats(), set()
    stats.setdefault("retry_reasons", {})

    session = make_session(args.token or os.environ.get("HF_TOKEN"))
    stats["authenticated"] = bool(args.token or os.environ.get("HF_TOKEN"))
    shard_idx = len(shards)
    plain = os.path.join(out_dir, shard_stem(shard_idx) + ".jsonl")
    fh = open(plain, "a", encoding="utf-8")
    t0 = time.time()
    try:
        while n_written < args.limit and url:
            recs, url = fetch(session, url, args.timeout, args.max_retries,
                              stats, args.min_ratelimit_remaining)
            stats["pages"] += 1
            if not recs:
                print("[end] empty page after %d records" % n_written)
                break
            for rec in recs:
                did = rec.get("id")
                if did is None:
                    continue
                if did in seen:
                    stats["duplicates_skipped"] += 1
                    continue
                seen.add(did)
                fh.write(json.dumps(trim(rec), ensure_ascii=False) + "\n")
                n_written += 1
                n_in_shard += 1
                if "first_id" not in stats:
                    stats["first_id"] = did
                    stats["first_created_at"] = rec.get("createdAt")
                stats["last_id"] = did
                stats["last_created_at"] = rec.get("createdAt")
                if n_in_shard >= args.shard_size or n_written >= args.limit:
                    fh.close()
                    finalize_shard(out_dir, shard_idx, n_in_shard, shards)
                    write_json_atomic(shards_path, shards)
                    shard_idx += 1
                    n_in_shard = 0
                    plain = os.path.join(out_dir, shard_stem(shard_idx) + ".jsonl")
                    fh = open(plain, "a", encoding="utf-8")
                    if n_written >= args.limit:
                        break
            fh.flush()
            os.fsync(fh.fileno())
            write_json_atomic(cursor_path, {
                "next_url": url, "n_written": n_written, "n_in_shard": n_in_shard,
                "shard_index": shard_idx, "sort": "createdAt", "direction": "-1",
                "stats": stats, "updated_at": utcnow()})
            if stats["pages"] % 50 == 0:
                print("[page %5d] n=%7d  shard=%d  last=%s (%s)  %.0f rec/s"
                      % (stats["pages"], n_written, shard_idx, stats.get("last_id"),
                         stats.get("last_created_at"),
                         n_written / max(time.time() - t0, 1e-9)), flush=True)
    finally:
        if not fh.closed:
            fh.flush()
            fh.close()

    if n_in_shard > 0 and os.path.exists(plain):
        finalize_shard(out_dir, shard_idx, n_in_shard, shards)
        write_json_atomic(shards_path, shards)
    elif os.path.exists(plain) and os.path.getsize(plain) == 0:
        os.remove(plain)

    stats["wallclock_s"] = round(time.time() - t0, 1)
    stats["finished_at"] = utcnow()
    stats["exhausted_cursor"] = not url
    write_json_atomic(cursor_path, {
        "next_url": url, "n_written": n_written, "n_in_shard": 0,
        "shard_index": len(shards), "sort": "createdAt", "direction": "-1",
        "stats": stats, "updated_at": utcnow()})

    prov = {
        "artifact": "hf dataset metadata crawl (F1.5)",
        "runbook": "docs/1M/1Mplan.md F1.5",
        "endpoint": API, "expand": list(EXPAND),
        "kept_top_level": list(KEEP), "kept_card_data": list(KEEP_CARD),
        "description_chars_kept": DESC_CHARS,
        "sort": "createdAt", "direction": "-1",
        "snapshot_date_utc": (stats.get("finished_at") or utcnow())[:10],
        "snapshot_window_utc": {
            "crawl_started_at": stats.get("started_at"),
            "crawl_finished_at": stats.get("finished_at"),
            "newest_dataset_created_at": stats.get("first_created_at"),
            "oldest_dataset_created_at": stats.get("last_created_at")},
        "total_records": n_written, "pages": stats.get("pages"),
        "exhausted_cursor": stats.get("exhausted_cursor"),
        "authenticated": stats.get("authenticated"),
        "retries": stats.get("retries"), "retry_reasons": stats.get("retry_reasons"),
        "ratelimit_sleeps": stats.get("ratelimit_sleeps"),
        "ratelimit_sleep_s": stats.get("ratelimit_sleep_s", 0),
        "duplicates_skipped": stats.get("duplicates_skipped"),
        "wallclock_s": stats.get("wallclock_s"),
        "shards": shards,
        "python": sys.version.split()[0], "requests": requests.__version__,
        "written_at": utcnow(),
    }
    write_json_atomic(prov_path, prov)
    os.chmod(prov_path, 0o444)
    print("\n[ok] %d datasets in %d shards -> %s%s"
          % (n_written, len(shards), out_dir,
             "  (cursor exhausted)" if not url else ""))
    return 0 if (n_written >= args.limit or not url) else 1


def build_parser():
    p = argparse.ArgumentParser(description="HF dataset metadata snapshot (F1.5)")
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=100_000_000)
    p.add_argument("--shard-size", type=int, default=100_000)
    p.add_argument("--page-size", type=int, default=PAGE_LIMIT_MAX)
    p.add_argument("--token", default=None, help="defaults to $HF_TOKEN; optional")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--max-retries", type=int, default=8)
    p.add_argument("--min-ratelimit-remaining", type=int, default=25)
    p.add_argument("--restart", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.out is None:
        args.out = os.path.join(data_root(), "data1m", "datasets_full")
    return crawl(args)


if __name__ == "__main__":
    raise SystemExit(main())
