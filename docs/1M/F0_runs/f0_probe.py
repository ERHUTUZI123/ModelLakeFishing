"""F0 enumeration-feasibility probe.

Two independent questions, two parts:

  A. depth   -- does the keyset cursor keep advancing for hundreds of pages
                under `sort=createdAt&direction=-1`?  Uses a minimal field set,
                because the cursor does not depend on `expand[]`.
  B. payload -- how many bytes and seconds does ONE page cost with the real
                EXPAND_V2 field set?  Needed for the F1 time/storage estimate.

Writes a JSON record of every page so the numbers in F0.md are auditable.
"""
import base64
import json
import os
import sys
import time
import urllib.parse

sys.path.insert(0, r"D:\research\model_lake\codes\ModelLakeFishing")

from scale1m.hf_crawl import (API, EXPAND_V2, build_url, make_session,
                              next_url, parse_ratelimit, utcnow)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f0_probe.json")
DEPTH_PAGES = int(os.environ.get("DEPTH_PAGES", "200"))
PAYLOAD_PAGES = int(os.environ.get("PAYLOAD_PAGES", "5"))


def decode_cursor(url):
    """The `Link: rel=next` URL carries a base64 keyset cursor."""
    if not url:
        return None
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    raw = (q.get("cursor") or [None])[0]
    if raw is None:
        return None
    try:
        pad = raw + "=" * (-len(raw) % 4)
        return json.loads(base64.b64decode(pad).decode("utf-8"))
    except Exception as exc:                      # noqa: BLE001 - diagnostic only
        return {"_undecodable": repr(raw)[:200], "_error": str(exc)}


def get(session, url, tries=6):
    delay = 2.0
    for attempt in range(tries):
        t0 = time.time()
        resp = session.get(url, timeout=60)
        dt = time.time() - t0
        if resp.status_code == 200:
            return resp, dt
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < tries - 1:
            wait = float(resp.headers.get("Retry-After") or delay)
            print("    [retry] http %d, sleeping %.0fs" % (resp.status_code, wait))
            time.sleep(wait)
            delay = min(delay * 2, 60)
            continue
        resp.raise_for_status()
    raise RuntimeError("unreachable")


def run_depth(session):
    url = build_url({"limit": "1000", "sort": "createdAt", "direction": "-1"}, ())
    seen, pages = set(), []
    dup_total = 0
    prev_created = None
    order_violations = 0
    t_start = time.time()

    for i in range(DEPTH_PAGES):
        resp, dt = get(session, url)
        body = resp.content
        recs = resp.json()
        nxt = next_url(resp)
        rem, reset = parse_ratelimit(resp.headers)

        ids = [r.get("id") for r in recs]
        dups = sum(1 for m in ids if m in seen)
        dup_total += dups
        seen.update(ids)

        created = [r.get("createdAt") for r in recs if r.get("createdAt")]
        if created:
            if prev_created is not None and created[0] > prev_created:
                order_violations += 1
            for a, b in zip(created, created[1:]):
                if b > a:
                    order_violations += 1
            prev_created = created[-1]

        pages.append({
            "page": i,
            "n_records": len(recs),
            "bytes": len(body),
            "seconds": round(dt, 3),
            "dups_in_page": dups,
            "has_next": bool(nxt),
            "first_id": ids[0] if ids else None,
            "last_id": ids[-1] if ids else None,
            "first_createdAt": created[0] if created else None,
            "last_createdAt": created[-1] if created else None,
            "ratelimit_remaining": rem,
        })
        if i in (0, 1, DEPTH_PAGES // 2, DEPTH_PAGES - 1):
            pages[-1]["cursor_decoded"] = decode_cursor(nxt)

        if (i + 1) % 20 == 0 or i == 0:
            print("  page %3d  n=%4d  %6.1f KB  %.2fs  uniq=%d  next=%s  rl=%s"
                  % (i, len(recs), len(body) / 1024, dt, len(seen),
                     bool(nxt), rem), flush=True)

        if not nxt:
            print("  [end] no rel=next after page %d" % i)
            break
        if not recs:
            print("  [end] empty page at %d" % i)
            break
        url = nxt
        if rem is not None and rem < 25:
            print("    [ratelimit] %d left, sleeping %ds" % (rem, (reset or 60) + 1))
            time.sleep((reset or 60) + 1)

    wall = time.time() - t_start
    got = sum(p["n_records"] for p in pages)
    return {
        "pages_requested": DEPTH_PAGES,
        "pages_fetched": len(pages),
        "records_fetched": got,
        "unique_ids": len(seen),
        "duplicate_records": dup_total,
        "pages_full_1000": sum(1 for p in pages if p["n_records"] == 1000),
        "createdAt_order_violations": order_violations,
        "terminated_by": ("no_next" if pages and not pages[-1]["has_next"]
                          else "page_budget"),
        "wallclock_s": round(wall, 1),
        "bytes_total": sum(p["bytes"] for p in pages),
        "pages": pages,
    }


def run_payload(session):
    url = build_url({"limit": "1000", "sort": "createdAt", "direction": "-1"},
                    EXPAND_V2)
    out = []
    for i in range(PAYLOAD_PAGES):
        resp, dt = get(session, url)
        recs = resp.json()
        out.append({"page": i, "n_records": len(recs), "bytes": len(resp.content),
                    "seconds": round(dt, 3),
                    "bytes_per_record": round(len(resp.content) / max(1, len(recs)), 1)})
        print("  payload page %d  n=%d  %.1f MB  %.2fs  %.0f B/rec"
              % (i, len(recs), len(resp.content) / 1e6, dt,
                 out[-1]["bytes_per_record"]), flush=True)
        nxt = next_url(resp)
        if not nxt:
            break
        url = nxt
    n = sum(p["n_records"] for p in out)
    b = sum(p["bytes"] for p in out)
    s = sum(p["seconds"] for p in out)
    return {"pages": out, "records": n, "bytes": b, "seconds": round(s, 2),
            "bytes_per_record": round(b / max(1, n), 1),
            "seconds_per_page": round(s / max(1, len(out)), 3)}


def main():
    token = os.environ.get("HF_TOKEN")
    session = make_session(token)
    result = {"started_at": utcnow(), "api": API, "authenticated": bool(token),
              "sort": "createdAt", "direction": "-1"}

    # does any header carry a total count?
    resp, _ = get(session, build_url({"limit": "1"}, ()))
    result["headers_of_limit1_request"] = dict(resp.headers)

    print("[A] depth probe: %d pages, minimal fields" % DEPTH_PAGES)
    result["depth"] = run_depth(session)
    print("[B] payload probe: %d pages, EXPAND_V2" % PAYLOAD_PAGES)
    result["payload"] = run_payload(session)
    result["finished_at"] = utcnow()

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print("\nwrote", OUT)

    d = result["depth"]
    print("depth: %d pages, %d records, %d unique, %d dups, order violations %d, %s, %.1fs"
          % (d["pages_fetched"], d["records_fetched"], d["unique_ids"],
             d["duplicate_records"], d["createdAt_order_violations"],
             d["terminated_by"], d["wallclock_s"]))


if __name__ == "__main__":
    main()
