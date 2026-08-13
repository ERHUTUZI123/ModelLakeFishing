"""
verify_raw.py -- T2 exit gate / T3 entry guard: prove the frozen raw crawl is
still byte-identical to what PROVENANCE.json recorded, and report the field
coverage that decides whether T3 can even build the rung.

Runbook: docs/1M/100kplan.md 5.4 (T2 gate) and 6.3 (T3 gate).

Same contract as scale/verify_corpus.py:
    exit 0 = snapshot intact, safe to proceed
    exit 1 = MISMATCH / MISSING / SHORT -- stop, do not canonicalize

Beyond checksums it answers the three questions T3 cannot start without:
    * safetensors.total coverage  -> the size_b missing rate (100kplan 6.3)
    * cardData.base_model coverage-> the lineage-edge ceiling (100kplan 8.1)
    * CORE overlap                -> how many HALO candidates actually remain
The last one is the one that can silently sink the rung: the ladder needs
100000 - 30183 = 69817 HALO rows, and every CORE id sitting inside the crawled
prefix is one candidate fewer.

Run (from ModelLakeFishing/):
    .\\.venv\\Scripts\\python.exe -m scale1m.verify_raw
    .\\.venv\\Scripts\\python.exe -m scale1m.verify_raw --core stage1BuildTransferGraph/hgraph_ml_v2.pt --sample 20
"""

import argparse
import gzip
import json
import os
import random
import re
import statistics
import sys
from collections import Counter

from scale1m.hf_crawl import default_out, sha256_of

# Same shape T3 will look for in a model name when safetensors is absent
# ("...-7b-...", "...-350M-..."). Used here only to SIZE the decision, not to
# make it: see D-26 in docs/1M/100kplan.md 16.
NAME_SIZE_RE = re.compile(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s*([bmBM])(?![a-z0-9])")


def normalize(mid) -> str:
    """The single id-normalization rule shared with T3 (100kplan 6.1)."""
    return str(mid).strip().lower()


def iter_records(out_dir: str, shards: list):
    for sh in shards:
        with gzip.open(os.path.join(out_dir, sh["file"]), "rt", encoding="utf-8") as fh:
            for line in fh:
                yield sh["file"], json.loads(line)


def base_model_of(rec):
    """cardData.base_model, normalized. str or list (take the first)."""
    b = (rec.get("cardData") or {}).get("base_model")
    if isinstance(b, list):
        b = b[0] if b else None
    return normalize(b) if isinstance(b, str) and b.strip() else None


def clean_name(mid: str) -> str:
    """The first clause of scale.modellens_build_graph.model_descriptor."""
    return mid.replace("/", " ").replace("-", " ").replace("_", " ")


def _pct(a, b):
    return 100.0 * a / max(b, 1)


def _quant(vals, p):
    s = sorted(vals)
    return s[int(p * (len(s) - 1))]


def forecast(recs, core, core_ids, rung_n):
    """The T2 -> T3/T5/T8 handoff: everything downstream can be bounded now.

    None of this is a gate -- it is the set of numbers whose surprise value is
    highest *before* the work, so that T3/T5/T8 cannot be quietly re-narrated
    afterwards. Every line here is a prediction with a named owner gate.
    """
    print("\n" + "=" * 72)
    print("FORECAST for rung N=%d  (predictions, not gates)" % rung_n)
    print("=" * 72)

    halo = [r for r in recs if normalize(r["id"]) not in core_ids]
    need = rung_n - len(core_ids)
    halo = halo[:need]
    print("  CORE %d (frozen prefix) + HALO %d = %d" % (len(core_ids), len(halo), rung_n))
    if len(halo) < need:
        print("  !! only %d HALO candidates -- crawl deeper before T3" % len(halo))
        return

    # -- T3 6.3: layer distribution ------------------------------------
    lab = lin = plain = dropped = 0
    for r in halo:
        card = r.get("cardData") or {}
        if not r.get("pipeline_tag") and not r.get("tags") \
                and not (r.get("safetensors") or {}).get("total"):
            dropped += 1
        elif card.get("model-index"):
            lab += 1
        elif base_model_of(r):
            lin += 1
        else:
            plain += 1
    print("\n  [T3 6.3] HALO layer: labeled %d (%.2f%%) | lineage %d (%.2f%%) | "
          "plain %d (%.2f%%) | dropped %d (%.2f%%)"
          % (lab, _pct(lab, len(halo)), lin, _pct(lin, len(halo)),
             plain, _pct(plain, len(halo)), dropped, _pct(dropped, len(halo))))
    print("           -> T8 11.1 displacement_quality has %d labeled HALO models to work with" % lab)

    # -- T5 8.1: the is_base_of edge count, i.e. D-8's first real test --
    lake = core_ids | {normalize(r["id"]) for r in halo}
    declared = hit = hit_core = 0
    for r in halo:
        b = base_model_of(r)
        if not b:
            continue
        declared += 1
        if b in lake and b != normalize(r["id"]):
            hit += 1
            if b in core_ids:
                hit_core += 1
    print("\n  [T5 8.1] HALO declaring base_model: %d (%.2f%%)" % (declared, _pct(declared, len(halo))))
    print("           resolving INSIDE the rung: %d  (core_halo %d | halo_halo %d | core_core 0)"
          % (hit, hit_core, hit - hit_core))
    print("           P1's name-matching baseline was 42 edges -> ~%.0fx" % (hit / 42.0))

    # -- T8 11.3: the cold-start strata ---------------------------------
    key = [e for e in core["data"].edge_types if e[0] == "model" and "trained" in e[1]][0]
    ei = core["data"][key].edge_index
    import torch
    deg = torch.zeros(core["data"]["model"].num_nodes, dtype=torch.long)
    deg.scatter_add_(0, ei[0], torch.ones_like(ei[0]))
    warm, cool, core_d0 = int((deg >= 10).sum()), int(((deg >= 1) & (deg < 10)).sum()), int((deg == 0).sum())
    cold = hit                 # HALO with a resolvable base; all HALO have degree 0
    frozen = len(halo) - hit + core_d0
    print("\n  [T8 11.3] predicted strata: warm %d (%.2f%%) | cool %d (%.2f%%) | "
          "cold %d (%.2f%%) | frozen %d (%.2f%%)"
          % (warm, _pct(warm, rung_n), cool, _pct(cool, rung_n),
             cold, _pct(cold, rung_n), frozen, _pct(frozen, rung_n)))

    # -- T4 7.3 / G-B4: is the descriptor information content aligned? ---
    sb = core["data"]["model"].size_bucket_id
    core_size_cov = _pct(int((sb != 0).sum()), sb.numel())
    halo_st = sum(1 for r in halo if (r.get("safetensors") or {}).get("total"))
    halo_regex = sum(1 for r in halo
                     if not (r.get("safetensors") or {}).get("total")
                     and NAME_SIZE_RE.search(r["id"].split("/")[-1].replace("_", "-")))
    print("\n  [T4 7.3 / G-B4] size clause in the descriptor ('<x>B params'):")
    print("           CORE has it          : %.2f%%" % core_size_cov)
    print("           HALO, safetensors    : %.2f%%   (gap %+.2f pp)"
          % (_pct(halo_st, len(halo)), _pct(halo_st, len(halo)) - core_size_cov))
    print("           HALO, +name regex    : %.2f%%   (gap %+.2f pp)"
          % (_pct(halo_st + halo_regex, len(halo)),
             _pct(halo_st + halo_regex, len(halo)) - core_size_cov))

    core_names = [str(m) for m in core["unique_model_id"]["model"]]
    halo_names = [r["id"] for r in halo]
    for label, names in (("CORE", core_names), ("HALO", halo_names)):
        chars = [len(clean_name(n)) for n in names]
        toks = [len(clean_name(n).split()) for n in names]
        print("           %s name: chars mean %.1f p10/50/90 %d/%d/%d | tokens mean %.2f | "
              "org-prefixed %.1f%%"
              % (label, statistics.mean(chars), _quant(chars, .1), _quant(chars, .5),
                 _quant(chars, .9), statistics.mean(toks),
                 _pct(sum(1 for n in names if "/" in n), len(names))))
    top = Counter(n.split("/")[0] for n in halo_names if "/" in n).most_common(5)
    print("           HALO org concentration: " + ", ".join("%s %d (%.1f%%)"
          % (o, c, _pct(c, len(halo_names))) for o, c in top))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="verify the T2 raw crawl")
    p.add_argument("--raw", default=default_out())
    p.add_argument("--expect", type=int, default=150_000,
                   help="minimum total_records the gate demands")
    p.add_argument("--core", default=None,
                   help="CORE graph (.pt) -- enables the HALO-headroom check")
    p.add_argument("--sample", type=int, default=20,
                   help="records to print for the manual eyeball (gate 3)")
    p.add_argument("--forecast", type=int, default=100_000,
                   help="rung N to forecast T3/T5/T8 quantities for (0 = off, needs --core)")
    p.add_argument("--selected", default=None,
                   help="selected_halo.parquet -> run the T2 v2 exit gates G1-G20 instead")
    p.add_argument("--config", default=None, help="halo_quotas.json (with --selected)")
    p.add_argument("--annotated", default=None,
                   help="candidate pool parquet, for the supply-permitting gates")
    p.add_argument("--rerun-check", default=None,
                   help="a second selected_halo parquet -> G19 determinism check")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.selected:
        from scale1m.gates import run_exit_gates
        return run_exit_gates(args.selected, args.config, args.annotated,
                              args.rerun_check)

    fail = []
    prov_path = os.path.join(args.raw, "PROVENANCE.json")
    if not os.path.exists(prov_path):
        print("MISSING PROVENANCE.json in %s" % args.raw, file=sys.stderr)
        return 1
    with open(prov_path, "r", encoding="utf-8") as fh:
        prov = json.load(fh)

    print("=" * 72)
    print("T2 RAW CRAWL VERIFY  --  %s" % args.raw)
    print("=" * 72)
    print("snapshot_date_utc : %s   (the one irreproducible input)" % prov.get("snapshot_date_utc"))
    print("sort              : %s %s" % (prov.get("sort"), prov.get("direction")))
    print("total_records     : %s" % prov.get("total_records"))
    print("authenticated     : %s" % prov.get("authenticated"))
    print("pages/retries     : %s / %s   dupes skipped: %s"
          % (prov.get("pages"), prov.get("retries"), prov.get("duplicates_skipped")))
    print("wallclock_s       : %s" % prov.get("wallclock_s"))

    # --- gate 1: checksums + read-only ------------------------------------
    print("\n-- shards --")
    n_declared = 0
    for sh in prov.get("shards", []):
        path = os.path.join(args.raw, sh["file"])
        if not os.path.exists(path):
            print("  MISSING  %s" % sh["file"])
            fail.append("missing shard %s" % sh["file"])
            continue
        got = sha256_of(path)
        ok = got == sh["sha256"]
        writable = bool(os.stat(path).st_mode & 0o222)
        print("  %s %-26s n=%6d  %10d B  sha=%s%s"
              % ("OK  " if ok else "BAD ", sh["file"], sh["n_records"], sh["bytes"],
                 sh["sha256"][:16], "  [WRITABLE]" if writable else ""))
        if not ok:
            fail.append("sha256 mismatch %s" % sh["file"])
        if writable:
            fail.append("shard not read-only: %s" % sh["file"])
        n_declared += sh["n_records"]

    if n_declared != prov.get("total_records"):
        fail.append("shard record counts sum to %d, PROVENANCE says %s"
                    % (n_declared, prov.get("total_records")))
    if (prov.get("total_records") or 0) < args.expect:
        fail.append("total_records %s < --expect %d" % (prov.get("total_records"), args.expect))

    # --- gate 2: one full pass -- order, dupes, field coverage -------------
    n = 0
    ids_norm = set()
    dupes = 0
    order_violations = 0
    prev_dl = None
    cov = dict(safetensors=0, base_model=0, model_index=0, pipeline_tag=0,
               library_name=0, tags=0, card_datasets=0)
    zero_info = 0          # the T3 `dropped` layer, exactly as 100kplan 6.1 defines it
    reservoir = []
    rng = random.Random(args.seed)
    want_forecast = bool(args.core) and args.forecast > 0
    recs = [] if want_forecast else None

    for _, rec in iter_records(args.raw, prov.get("shards", [])):
        n += 1
        if want_forecast:
            recs.append(rec)
        nid = normalize(rec.get("id"))
        if nid in ids_norm:
            dupes += 1
        ids_norm.add(nid)

        dl = rec.get("downloads")
        if prev_dl is not None and dl is not None and dl > prev_dl:
            order_violations += 1
        if dl is not None:
            prev_dl = dl

        card = rec.get("cardData") or {}
        has_size = bool((rec.get("safetensors") or {}).get("total"))
        if has_size:
            cov["safetensors"] += 1
        if card.get("base_model"):
            cov["base_model"] += 1
        if card.get("model-index"):
            cov["model_index"] += 1
        if card.get("datasets"):
            cov["card_datasets"] += 1
        if rec.get("pipeline_tag"):
            cov["pipeline_tag"] += 1
        if rec.get("library_name"):
            cov["library_name"] += 1
        if rec.get("tags"):
            cov["tags"] += 1
        if not rec.get("pipeline_tag") and not rec.get("tags") and not has_size:
            zero_info += 1

        if len(reservoir) < args.sample:
            reservoir.append(rec)
        else:
            j = rng.randrange(n)
            if j < args.sample:
                reservoir[j] = rec

    print("\n-- integrity --")
    print("  records read            : %d" % n)
    print("  unique normalized ids   : %d   (duplicates: %d)" % (len(ids_norm), dupes))
    print("  downloads-desc violations: %d" % order_violations)
    if n != n_declared:
        fail.append("read %d records but shards declare %d" % (n, n_declared))
    if dupes:
        fail.append("%d duplicate normalized ids in the crawl" % dupes)
    if order_violations:
        fail.append("%d downloads-descending violations (ladder order is the artifact order)"
                    % order_violations)

    print("\n-- field coverage (drives T3/T5 gates) --")
    for k, v in cov.items():
        print("  %-16s %7d  %6.2f%%" % (k, v, 100.0 * v / max(n, 1)))
    print("  %-16s %7d  %6.2f%%   <- T3 layer='dropped'" % ("zero-info", zero_info,
                                                            100.0 * zero_info / max(n, 1)))
    print("  %-16s %7d  %6.2f%%   <- size_b will be NaN -> bucket 0"
          % ("size_b MISSING", n - cov["safetensors"],
             100.0 * (n - cov["safetensors"]) / max(n, 1)))

    # --- gate 3: HALO headroom against CORE -------------------------------
    if args.core:
        import torch
        core = torch.load(args.core, weights_only=False)
        core_ids = set(normalize(m) for m in core["unique_model_id"]["model"])
        overlap = len(core_ids & ids_norm)
        halo_pool = len(ids_norm) - overlap - zero_info  # upper bound: zero-info may overlap
        need = 100_000 - len(core_ids)
        print("\n-- HALO headroom (CORE = %s) --" % os.path.basename(args.core))
        print("  CORE models                 : %d" % len(core_ids))
        print("  CORE ids inside the crawl   : %d  (%.2f%% of CORE)"
              % (overlap, 100.0 * overlap / max(len(core_ids), 1)))
        print("  HALO candidates (>= approx) : %d" % halo_pool)
        print("  R2 needs                    : %d" % need)
        if halo_pool < need:
            fail.append("only ~%d HALO candidates for a rung that needs %d -- crawl deeper"
                        % (halo_pool, need))
        else:
            print("  headroom                    : %+d  OK" % (halo_pool - need))

        if want_forecast:
            forecast(recs, core, core_ids, args.forecast)

    # --- gate 4: the manual eyeball ---------------------------------------
    print("\n-- random sample of %d (gate: id / safetensors.total / cardData.base_model) --"
          % args.sample)
    for rec in reservoir:
        card = rec.get("cardData") or {}
        bm = card.get("base_model")
        if isinstance(bm, list):
            bm = bm[0] if bm else None
        print("  %-58s dl=%-9s size=%-12s base=%s"
              % (rec.get("id")[:58], rec.get("downloads"),
                 (rec.get("safetensors") or {}).get("total"), bm))

    print("\n" + "=" * 72)
    if fail:
        print("FAIL (%d):" % len(fail))
        for f in fail:
            print("  - %s" % f)
        return 1
    print("PASS -- T2 gate green, T3 may proceed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
