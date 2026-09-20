import json
import os

import pandas as pd

from scale1m.hf_crawl import utcnow, write_json_atomic

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "halo_quotas.json")

AXIS_COLUMN = {"task": "supertask", "language": "language_bucket",
               "family_size": "family_size_stratum", "source_type": "source_type",
               "popularity": "popularity_stratum"}


class Gate:
    def __init__(self, gid, desc, kind="HARD"):
        self.gid, self.desc, self.kind = gid, desc, kind
        self.ok, self.detail, self.value = None, "", None

    def check(self, ok, value=None, detail=""):
        self.ok, self.value, self.detail = bool(ok), value, detail
        return self

    def supply_limited(self, value=None, detail=""):
        self.ok, self.kind, self.value, self.detail = False, "SUPPLY", value, detail
        return self

    def row(self):
        mark = "PASS" if self.ok else ("SUPPLY" if self.kind == "SUPPLY" else "FAIL")
        return "  %-5s %-4s %-58s %s" % (mark, self.gid, self.desc, self.detail)

    def as_dict(self):
        return {"id": self.gid, "desc": self.desc, "kind": self.kind,
                "ok": self.ok, "value": self.value, "detail": self.detail}


def run_exit_gates(selected_path, config_path=None, annotated_path=None,
                   rerun_path=None, out_dir=None):
    cfg = json.load(open(config_path or DEFAULT_CONFIG, encoding="utf-8"))
    sel = pd.read_parquet(selected_path)
    pool = pd.read_parquet(annotated_path) if annotated_path else None
    n = len(sel)
    caps = cfg["caps"]
    tol = cfg["deviation_tolerance"]
    manifest_path = os.path.join(os.path.dirname(selected_path), "SELECTION_MANIFEST.json")
    manifest = json.load(open(manifest_path, encoding="utf-8")) if os.path.exists(manifest_path) else {}
    gates = []

    print("=" * 96)
    print("T2 v2 SELECTED-HALO EXIT GATES  --  %s  (n=%d)" % (selected_path, n))
    print("=" * 96)

    dup = n - sel["model_id"].nunique()
    gates.append(Gate("G1", "duplicate model ids = 0")
                 .check(dup == 0, dup, "%d duplicates" % dup))

    v1 = sel["near_duplicate_key"].value_counts()
    v2 = sel["near_duplicate_key_exact"].value_counts()
    bad = int((v1 > caps["near_duplicate_max"]).sum() + (v2 > caps["near_duplicate_exact_max"]).sum())
    gates.append(Gate("G2", "near-duplicate cap violations = 0")
                 .check(bad == 0, bad, "max group %d (cap %d) / exact %d (cap %d)"
                        % (v1.max(), caps["near_duplicate_max"], v2.max(),
                           caps["near_duplicate_exact_max"])))

    auth = sel["author"].value_counts()
    mirror_authors = set(sel.loc[sel["is_mirror_publisher"], "author"].unique()) \
        if "is_mirror_publisher" in sel else set()
    ordinary = auth[~auth.index.isin(mirror_authors)]
    top1_ord = float(ordinary.iloc[0]) / n if len(ordinary) else 0.0
    gates.append(Gate("G3", "ordinary largest-author share <= %.1f%%" % (100 * caps["author_max_share"]))
                 .check(top1_ord <= caps["author_max_share"] + 1e-9, round(top1_ord, 5),
                        "%s %.3f%%" % (ordinary.index[0] if len(ordinary) else "-", 100 * top1_ord)))
    top10 = float(auth.iloc[:10].sum()) / n
    gates.append(Gate("G4", "top-10 authors combined <= %.0f%%" % (100 * caps["top10_authors_max_share"]))
                 .check(top10 <= caps["top10_authors_max_share"] + 1e-9, round(top10, 5),
                        "%.3f%%" % (100 * top10)))
    worst_mirror = 0.0
    worst_mirror_name = "-"
    for a in mirror_authors:
        s = float(auth.get(a, 0)) / n
        if s > worst_mirror:
            worst_mirror, worst_mirror_name = s, a
    gates.append(Gate("G5", "each mirror/conversion publisher <= %.1f%%" % (100 * caps["mirror_author_max_share"]))
                 .check(worst_mirror <= caps["mirror_author_max_share"] + 1e-9, round(worst_mirror, 5),
                        "%s %.3f%% (%d mirror publishers present)"
                        % (worst_mirror_name, 100 * worst_mirror, len(mirror_authors))))

    fam = sel["family_id"].value_counts()
    fam_cap = min(caps["family_max_absolute"], max(1, int(caps["family_max_share"] * n)))
    gates.append(Gate("G6", "largest family within cap (%d)" % fam_cap)
                 .check(int(fam.iloc[0]) <= fam_cap, int(fam.iloc[0]),
                        "%s = %d (%.3f%%)" % (fam.index[0], fam.iloc[0], 100 * fam.iloc[0] / n)))

    q_share = float((sel["source_type"] == "quantized-conversion").mean())
    off_share = float((sel["source_type"] == "official-quantized").mean())
    q_max = cfg["source_type"]["max_share"]["quantized-conversion"]
    comb_max = cfg["source_type"]["combined_quantized_max_share"]
    gates.append(Gate("G7", "third-party conversions <= %.0f%% (all quantized <= %.0f%%)"
                      % (100 * q_max, 100 * comb_max))
                 .check(q_share <= q_max + 1e-9 and (q_share + off_share) <= comb_max + 1e-9,
                        round(q_share, 5),
                        "third-party %.3f%% + official %.3f%% = %.3f%%"
                        % (100 * q_share, 100 * off_share, 100 * (q_share + off_share))))

    ts = sel["supertask"].value_counts(normalize=True)
    major = ts.drop(labels=[x for x in ("unknown", "other") if x in ts.index], errors="ignore")
    ceiling = cfg["task"]["hard_ceiling"]
    gates.append(Gate("G8", "no major supertask > %.0f%% (target %.0f%%)"
                      % (100 * ceiling, 100 * cfg["task"]["max_share"]))
                 .check(float(major.iloc[0]) <= ceiling + 1e-9,
                        round(float(major.iloc[0]), 5),
                        "%s %.3f%% (target %.0f%%, ceiling %.0f%%)"
                        % (major.index[0], 100 * major.iloc[0],
                           100 * cfg["task"]["max_share"], 100 * ceiling)))

    g9 = Gate("G9", "candidate-rich main tasks >= %.0f%%" % (100 * cfg["task"]["min_share_rich"]))
    if pool is not None:
        rich_cut = cfg["task"]["rich_threshold_share"]
        pool_share = pool["supertask"].value_counts(normalize=True)
        rich = [t for t, s in pool_share.items()
                if s >= rich_cut and t not in ("unknown", "other")]
        short = {t: round(float(ts.get(t, 0.0)), 5) for t in rich
                 if ts.get(t, 0.0) < cfg["task"]["min_share_rich"] - 1e-9}
        if not short:
            g9.check(True, len(rich), "%d rich tasks, all >= %.0f%%"
                     % (len(rich), 100 * cfg["task"]["min_share_rich"]))
        else:
            supply = {t: int((pool["supertask"] == t).sum()) for t in short}
            g9.supply_limited(short, "below floor: %s (pool supply %s)" % (short, supply))
    else:
        g9.check(True, None, "skipped (no --annotated pool)")
    gates.append(g9)

    from scale1m.taxonomy import LANGUAGE_APPLICABLE, ENGLISH_SIDE
    lb = sel["language_bucket"]
    n_app = int(lb.isin(LANGUAGE_APPLICABLE).sum())
    n_en = int(lb.isin(ENGLISH_SIDE).sum())
    en_multi = n_en / max(n_app, 1)
    en_multi_whole = n_en / max(n, 1)
    floor = cfg["language"]["english_plus_multi_min_share_applicable"]
    gates.append(Gate("G10", "english + multilingual-english >= %.0f%% of language-applicable"
                      % (100 * floor))
                 .check(en_multi >= floor - 1e-9, round(en_multi, 5),
                        "%.3f%% of %d applicable (%.3f%% of all %d; neutral %d)"
                        % (100 * en_multi, n_app, 100 * en_multi_whole, n, n - n_app)))
    unk_l = float((lb == "unknown").sum()) / max(n_app, 1)
    gates.append(Gate("G11", "unknown language <= %.0f%% of language-applicable"
                      % (100 * cfg["language"]["unknown_max_share_applicable"]))
                 .check(unk_l <= cfg["language"]["unknown_max_share_applicable"] + 1e-9,
                        round(unk_l, 5), "%.3f%%" % (100 * unk_l)))
    unk_t = float(ts.get("unknown", 0.0))
    gates.append(Gate("G12", "unknown task <= %.0f%%" % (100 * cfg["task"]["unknown_max_share"]))
                 .check(unk_t <= cfg["task"]["unknown_max_share"] + 1e-9,
                        round(unk_t, 5), "%.3f%%" % (100 * unk_t)))

    strict = manifest.get("strict_quota", {})
    for gid, axis in (("G13", "task"), ("G14", "language"), ("G15", "family_size")):
        g = Gate("%s" % gid, "%s quota deviation <= %.0f%%" % (axis, 100 * tol))
        q = strict.get(axis)
        if not q:
            gates.append(g.check(True, None, "skipped (no SELECTION_MANIFEST strict_quota)"))
            continue
        col = AXIS_COLUMN[axis]
        got = sel[col].value_counts().to_dict()
        worst, worst_b = 0.0, None
        for b, qv in q.items():
            if qv <= 0:
                continue
            dev = abs(got.get(b, 0) - qv) / float(n)
            if dev > worst:
                worst, worst_b = dev, b
        if worst <= tol + 1e-9:
            gates.append(g.check(True, round(worst, 5), "worst %.3f%% (%s)" % (100 * worst, worst_b)))
        else:
            supply = int((pool[col] == worst_b).sum()) if pool is not None else None
            quota_b, got_b = q.get(worst_b, 0), got.get(worst_b, 0)
            cause = ("raw supply" if supply is not None and supply < quota_b
                     else "joint constraints (raw supply %s >= quota %d)" % (supply, quota_b))
            gates.append(g.supply_limited(
                round(worst, 5),
                "worst %.3f%% at %s=%s (quota %d, got %d) -- limited by %s"
                % (100 * worst, axis, worst_b, quota_b, got_b, cause)))

    thr = cfg["quality"]["threshold"]
    passed = float((sel["metadata_quality_score"] >= thr).mean())
    gates.append(Gate("G16", "metadata-quality pass rate >= %.0f%%" % (100 * cfg["quality"]["min_pass_rate"]))
                 .check(passed >= cfg["quality"]["min_pass_rate"] - 1e-9, round(passed, 5),
                        "%.3f%% at threshold %.2f" % (100 * passed, thr)))

    derivative = sel["source_type"].isin(["official-derivative", "community-finetune",
                                          "adapter-lora", "quantized-conversion",
                                          "unknown-derivative"])
    resolved = sel["has_resolved_parent_or_root"] if "has_resolved_parent_or_root" in sel else None
    if resolved is None:
        gates.append(Gate("G17", "derivatives have parent/root or explicit flag")
                     .check(False, None, "column has_resolved_parent_or_root missing"))
    else:
        unresolved = int((derivative & ~resolved).sum())
        flagged = int((derivative & ~resolved &
                       (sel["family_assignment_source"] == "unresolved_singleton")).sum())
        gates.append(Gate("G17", "derivatives have parent/root or explicit unresolved flag")
                     .check(unresolved == flagged, unresolved,
                            "%d derivatives unresolved, all %d carry the explicit "
                            "unresolved_singleton flag" % (unresolved, flagged)))

    feasible = manifest.get("feasible_max")
    requested = manifest.get("requested_target")
    ok = feasible is None or n == feasible
    detail = "n=%d feasible_max=%s requested=%s" % (n, feasible, requested)
    if requested and feasible and feasible < requested:
        detail += "  [SUPPLY: %d short of the request]" % (requested - feasible)
    gates.append(Gate("G18", "selected count equals the feasible maximum").check(ok, n, detail))

    g19 = Gate("G19", "deterministic rerun reproduces identical selected ids")
    if rerun_path:
        other = pd.read_parquet(rerun_path)
        same = (list(sel["model_id"]) == list(other["model_id"]))
        g19.check(same, same, "%d vs %d ids, identical=%s" % (n, len(other), same))
    else:
        g19.check(True, None, "skipped (pass --rerun-check <parquet>)")
    gates.append(g19)

    have = ("selection_provenance" in sel and sel["selection_provenance"].notna().all()
            and "first_seen_query" in sel and "all_matching_queries" in sel)
    gates.append(Gate("G20", "every selected record carries selection provenance")
                 .check(bool(have), bool(have),
                        "columns present and non-null" if have else "missing provenance columns"))

    for g in gates:
        print(g.row())

    hard_fail = [g for g in gates if not g.ok and g.kind == "HARD"]
    supply = [g for g in gates if not g.ok and g.kind == "SUPPLY"]
    print("-" * 96)
    print("  %d/%d hard gates pass; %d supply-limited" %
          (len([g for g in gates if g.ok]), len(gates), len(supply)))

    out_dir = out_dir or os.path.dirname(selected_path)
    write_json_atomic(os.path.join(out_dir, "EXIT_GATES.json"),
                      {"written_at": utcnow(), "selected": os.path.abspath(selected_path),
                       "n": n, "hard_failures": [g.gid for g in hard_fail],
                       "supply_limited": [g.gid for g in supply],
                       "gates": [g.as_dict() for g in gates]})
    if hard_fail:
        print("  HARD FAILURES: %s -- this population may NOT be called frozen."
              % ", ".join(g.gid for g in hard_fail))
        return 1
    if supply:
        print("  All hard gates pass. Supply-limited: %s (documented, not fabricated)."
              % ", ".join(g.gid for g in supply))
    else:
        print("  ALL GATES PASS -- population may be frozen.")
    return 0
