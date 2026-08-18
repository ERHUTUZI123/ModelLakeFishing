"""Estimate N_full without enumerating it.

The probe showed that under `sort=createdAt&direction=-1` the pagination cursor
is `{"_id": {"$lt": <ObjectId>}}`.  A Mongo ObjectId begins with a 4-byte unix
timestamp, so a cursor can be SYNTHESISED for any date: hex(ts) + 16 zeros.

That turns the API into a random-access probe over the creation timeline.  For
each sample date we pull one page of 1000 consecutive models and measure the
time span they cover; 1000 / span_days is the local creation rate.  Integrating
the rate over the timeline estimates how many models exist.

This is an ESTIMATE, not a count.  It is biased downward by models that were
created and later deleted (they are gone from the timeline but were never
counted anyway) and it interpolates between sample points.
"""
import base64
import datetime as dt
import json
import os
import sys
import time

sys.path.insert(0, r"D:\research\model_lake\codes\ModelLakeFishing")
from scale1m.hf_crawl import API, build_url, make_session, next_url  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f0_size_estimate.json")


def cursor_for(ts):
    oid = "%08x" % int(ts) + "0" * 16
    raw = json.dumps({"_id": {"$lt": oid}}, separators=(",", ":")).encode()
    return base64.b64encode(raw).decode().rstrip("=")


def page_at(session, ts):
    url = build_url({"limit": "1000", "sort": "createdAt", "direction": "-1",
                     "cursor": cursor_for(ts)}, ())
    r = session.get(url, timeout=60)
    r.raise_for_status()
    recs = r.json()
    got = [x["createdAt"] for x in recs if x.get("createdAt")]
    return recs, got, bool(next_url(r))


def parse(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def main():
    session = make_session(os.environ.get("HF_TOKEN"))
    now = dt.datetime.now(dt.timezone.utc)
    samples, cursor_ok = [], True

    # monthly sample points back to before the Hub had models
    month = dt.datetime(2022, 1, 1, tzinfo=dt.timezone.utc)
    points = []
    while month < now:
        points.append(month)
        # step one month
        month = (month.replace(day=28) + dt.timedelta(days=8)).replace(day=1)
    points.append(now)

    for p in points:
        recs, created, has_next = page_at(session, p.timestamp())
        if not created:
            samples.append({"at": p.isoformat(), "n": len(recs), "rate_per_day": 0.0,
                            "span_days": None, "note": "empty page"})
            continue
        newest, oldest = parse(created[0]), parse(created[-1])
        span = (newest - oldest).total_seconds() / 86400.0
        rate = len(created) / span if span > 0 else None
        samples.append({"at": p.isoformat(), "n": len(created),
                        "newest": created[0], "oldest": created[-1],
                        "span_days": round(span, 4),
                        "rate_per_day": round(rate, 1) if rate else None,
                        "has_next": has_next})
        print("  %s  1000 models span %8.3f d  -> %9.1f /day"
              % (p.date(), span, rate if rate else -1), flush=True)
        time.sleep(0.05)

    # trapezoidal integration of rate(t) over the timeline
    usable = [s for s in samples if s.get("rate_per_day")]
    total = 0.0
    for a, b in zip(usable, usable[1:]):
        ta, tb = parse(a["at"] + "" if a["at"].endswith("+00:00") else a["at"]), None
        ta = dt.datetime.fromisoformat(a["at"])
        tb = dt.datetime.fromisoformat(b["at"])
        days = (tb - ta).total_seconds() / 86400.0
        total += days * (a["rate_per_day"] + b["rate_per_day"]) / 2.0

    result = {"method": "synthesised ObjectId cursor + trapezoidal rate integration",
              "sampled_points": len(samples), "usable_points": len(usable),
              "estimated_models": int(total),
              "note": "estimate only; ignores deleted models and interpolates "
                      "between monthly samples",
              "samples": samples}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print("\nestimated N_full ~ %.2f M  (%d sample points)" % (total / 1e6, len(usable)))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
