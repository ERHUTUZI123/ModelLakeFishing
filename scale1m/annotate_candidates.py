"""
annotate_candidates.py -- T2 v2 stage 2: turn raw discovery shards into the
annotated candidate table the balanced selector consumes.

Runbook: docs/1M/T2.md (v2), instructions Steps 3-5, 6-7, 10-12.

Everything here is derived, nothing is fetched. Every derived label carries the
signal that produced it (`*_source`) and, where meaningful, a confidence, so a
distribution report can always answer "is this metadata or is this a guess?".

Order matters and is not arbitrary:
    1. per-record structural annotation (task, language, quant, size, arch)
    2. lineage closure  -> canonical_root, family_id       (needs all records)
    3. author aggregates -> mirror score                   (needs step 2)
    4. source type       -> needs parent author + quant    (needs steps 2,3)
    5. popularity strata -> percentiles within supertask   (needs step 1)
    6. near-duplicate key-> needs root + size + quant       (needs steps 1,2)

Run (from ModelLakeFishing/):
    .\\.venv\\Scripts\\python.exe -m scale1m.annotate_candidates --candidates <dir>
"""

import argparse
import gzip
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "stage1BuildTransferGraph"))

from scale1m import taxonomy as TX
from scale1m.hf_crawl import default_candidates_out, utcnow, write_json_atomic

# The EXISTING xm0 architecture-family rules -- reused, not re-implemented.
# This is the same function that produced CORE's family_id, so `arch_family`
# stays comparable across CORE and HALO.
from dataset_embed.utils.fetch_metadata import _infer_one_family, FAMILY_OTHER

RECENT_DAYS = 180


def normalize(mid) -> str:
    return str(mid).strip().lower()


# --- load ------------------------------------------------------------------


def load_candidates(cdir):
    prov = json.load(open(os.path.join(cdir, "PROVENANCE.json"), encoding="utf-8"))
    recs = []
    for sh in prov["shards"]:
        with gzip.open(os.path.join(cdir, sh["file"]), "rt", encoding="utf-8") as fh:
            for line in fh:
                recs.append(json.loads(line))
    return recs, prov


def load_membership(cdir):
    """id -> sorted list of query ids that returned it (merged provenance)."""
    mdir = os.path.join(cdir, "membership")
    memb = defaultdict(list)
    if not os.path.isdir(mdir):
        return memb
    for fn in sorted(os.listdir(mdir)):
        path = os.path.join(mdir, fn)
        qid = fn[:-7] if fn.endswith(".tsv.gz") else fn[:-4]
        opener = gzip.open if fn.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2:
                    memb[parts[1]].append(qid)
    return memb


# --- per-record structural annotation --------------------------------------

_NAME_SIZE_RE = re.compile(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s*([bmBM])(?![a-z0-9])")


def size_b_of(rec):
    """-> (size_b, source). safetensors is authoritative; the name regex is a
    LABELED heuristic and is never allowed to overwrite it (Step 3).

    NOTE the split from D-26: for the CORE-comparable descriptor the graph
    stage uses safetensors ONLY. Here both are kept as separate columns so the
    selector can stratify on parameter scale without the graph stage ever
    seeing the heuristic value.
    """
    total = (rec.get("safetensors") or {}).get("total")
    if total:
        return float(total) / 1e9, "safetensors"
    m = _NAME_SIZE_RE.search((rec.get("id") or "").split("/")[-1].replace("_", "-"))
    if m:
        v = float(m.group(1))
        return (v if m.group(2).lower() == "b" else v / 1000.0), "name_heuristic"
    return None, "none"


def base_ids_of(rec):
    """-> (list of normalized parent ids, relation, source)."""
    bm = rec.get("baseModels")
    if isinstance(bm, dict) and bm.get("ids"):
        return [normalize(x) for x in bm["ids"]], bm.get("relation"), "baseModels"
    card = rec.get("cardData") or {}
    b = card.get("base_model")
    rel = card.get("base_model_relation")
    if isinstance(b, str):
        b = [b]
    if isinstance(b, list) and b:
        return [normalize(x) for x in b if isinstance(x, str) and x.strip()], rel, "cardData.base_model"
    return [], None, "none"


def annotate_records(recs, memb):
    rows = []
    for r in recs:
        mid = r.get("id") or ""
        nid = normalize(mid)
        task, task_src, task_conf = TX.supertask_of(r)
        lang_bucket, langs, lang_src, lang_conf = TX.language_bucket_of(r, task)
        quant, bits, quant_src = TX.quantization_of(r)
        size_b, size_src = size_b_of(r)
        bases, relation, base_src = base_ids_of(r)
        qual, flags = TX.metadata_quality_of(r, task, task_src)
        prov = r.get("_prov") or {}
        cfg = r.get("config") or {}
        rows.append({
            "model_id": mid,
            "id_norm": nid,
            "author": (r.get("author") or mid.split("/")[0] if "/" in mid else ""),
            "downloads": r.get("downloads") or 0,
            "downloads_all_time": r.get("downloadsAllTime") or 0,
            "likes": r.get("likes") or 0,
            "trending": r.get("trendingScore") or 0,
            "created_at": r.get("createdAt"),
            "last_modified": r.get("lastModified"),
            "pipeline_tag": r.get("pipeline_tag"),
            "library_name": r.get("library_name"),
            "n_tags": len(r.get("tags") or []),
            "architectures": ",".join(cfg.get("architectures") or []),
            "model_type": cfg.get("model_type"),
            "arch_family": _infer_one_family(mid),
            "supertask": task, "supertask_source": task_src, "supertask_conf": task_conf,
            "language_bucket": lang_bucket, "languages": ",".join(langs),
            "language_source": lang_src, "language_conf": lang_conf,
            "quantization_type": quant, "quantization_bits": bits,
            "quantization_source": quant_src,
            "size_b": size_b, "size_source": size_src,
            "base_ids": bases, "base_relation": relation, "base_source": base_src,
            "has_model_index": bool((r.get("cardData") or {}).get("model-index")),
            "has_card": bool(r.get("cardData")),
            "gated": bool(r.get("gated")), "disabled": bool(r.get("disabled")),
            "metadata_quality_score": qual, "metadata_quality_flags": ";".join(flags),
            "repo_stem": TX.repo_stem(mid),
            # the only raw tags source_type_of needs; kept out of the parquet
            "_adapter_tags": [t for t in (r.get("tags") or [])
                              if str(t).lower() in ("lora", "peft", "adapter")],
            "first_seen_query": prov.get("first_seen_query"),
            "discovery_rank": prov.get("discovery_rank"),
            "discovery_source": prov.get("discovery_source"),
            "crawl_timestamp": prov.get("crawl_timestamp"),
            "all_matching_queries": ";".join(memb.get(mid, [])),
            "n_matching_queries": len(memb.get(mid, [])),
        })
    return pd.DataFrame(rows)


# --- lineage closure -> canonical_root / family_id -------------------------


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra
        return ra


# Generic stems that must never be used to merge unrelated repositories
# (Step 4: "do not collapse unrelated models merely because their names share
# generic tokens").
GENERIC_STEMS = frozenset({
    "", "model", "models", "test", "tmp", "temp", "demo", "checkpoint", "ckpt",
    "output", "results", "finetune", "finetuned", "train", "training", "run",
    "my", "new", "final", "best", "custom", "example", "sample", "untitled",
    "merged", "merge", "lora", "adapter", "peft", "base", "chat", "instruct",
})
MIN_STEM_LEN = 4


def build_families(df):
    """Step 4, in precedence order. Returns columns added in place.

    1-3. declared parent (baseModels relation > cardData.base_model). All
         declared edges whose parent is inside the pool are unioned, so an
         adapter of a quantization of a finetune lands in one family and the
         canonical root is the member with no in-pool parent.
    4.   architecture + normalized stem, only when the group has >= 2 members
         and the stem is neither generic nor a 3-char fragment.
    5.   author + normalized stem, same guard.
    6.   unresolved singleton.
    """
    in_pool = set(df["id_norm"])
    dsu = DSU()
    has_parent = {}
    edge_src = {}
    for nid, bases, bsrc in zip(df["id_norm"], df["base_ids"], df["base_source"]):
        dsu.find(nid)
        linked = [b for b in bases if b in in_pool and b != nid]
        if linked:
            has_parent[nid] = linked[0]
            edge_src[nid] = bsrc
            for b in linked:
                dsu.union(b, nid)

    # The FAMILY is the DSU component; the canonical ROOT is one member of it.
    # Walking each node's declared chain independently is not enough: authors
    # do declare cycles (a says its base is b, b says its base is a), and then
    # two members of one component walk to two different "roots" and the family
    # silently splits. So the root is decided once per component: the
    # lexicographically smallest parentless member, or -- if the component is
    # entirely cyclic -- its smallest member.
    comp_members = defaultdict(list)
    for nid in df["id_norm"]:
        comp_members[dsu.find(nid)].append(nid)
    comp_root = {}
    for comp, members in comp_members.items():
        parentless = sorted(m for m in members if m not in has_parent)
        comp_root[comp] = parentless[0] if parentless else sorted(members)[0]
    root_of = {nid: comp_root[dsu.find(nid)] for nid in df["id_norm"]}

    fam_id, fam_src, fam_conf, canon = [], [], [], []
    comp_size = {c: len(m) for c, m in comp_members.items()}

    stem_groups = Counter(zip(df["model_type"].fillna(""), df["repo_stem"]))
    author_groups = Counter(zip(df["author"], df["repo_stem"]))

    for nid, mt, stem, author in zip(df["id_norm"], df["model_type"].fillna(""),
                                     df["repo_stem"], df["author"]):
        comp = dsu.find(nid)
        if comp_size[comp] > 1:
            fam_id.append("lin:" + root_of[nid])
            fam_src.append(edge_src.get(nid, "declared_descendant"))
            fam_conf.append(0.95 if edge_src.get(nid) == "baseModels" else 0.8)
            canon.append(root_of[nid])
            continue
        usable = stem and stem not in GENERIC_STEMS and len(stem) >= MIN_STEM_LEN
        if usable and mt and stem_groups[(mt, stem)] > 1:
            fam_id.append("arch:%s|%s" % (mt, stem))
            fam_src.append("arch+stem")
            fam_conf.append(0.5)
            canon.append(None)
        elif usable and author_groups[(author, stem)] > 1:
            fam_id.append("auth:%s|%s" % (author, stem))
            fam_src.append("author+stem")
            fam_conf.append(0.35)
            canon.append(None)
        else:
            fam_id.append("single:" + nid)
            fam_src.append("unresolved_singleton")
            fam_conf.append(0.0)
            canon.append(None)

    df["family_id"] = fam_id
    df["family_assignment_source"] = fam_src
    df["family_assignment_confidence"] = fam_conf
    df["canonical_root_id"] = canon
    df["parent_id"] = [has_parent.get(n) for n in df["id_norm"]]
    df["has_resolved_parent_or_root"] = [
        (n in has_parent) or (s != "unresolved_singleton")
        for n, s in zip(df["id_norm"], df["family_assignment_source"])]
    return df


# --- author aggregates -> mirror score -------------------------------------


def build_author_stats(df):
    stats = {}
    by_author = df.groupby("author", observed=True)
    for author, g in by_author:
        n = len(g)
        quant_frac = float((g["quantization_type"] != "none").mean())
        upstream = set()
        for bases in g["base_ids"]:
            for b in bases:
                if "/" in b:
                    upstream.add(b.split("/")[0])
        original_frac = float((g["base_ids"].map(len) == 0).mean())
        stem_counts = g["repo_stem"].value_counts()
        stem_repeat = float((stem_counts[stem_counts > 1].sum()) / max(n, 1))
        s = {"n_repos": n, "quant_frac": quant_frac,
             "distinct_upstream_authors": len(upstream),
             "original_frac": original_frac, "stem_repeat_frac": stem_repeat}
        score, why = TX.mirror_score_of(s)
        s["mirror_score"] = score
        s["mirror_reasons"] = why
        s["is_mirror_publisher"] = score >= TX.MIRROR_SCORE_THRESHOLD
        stats[author] = s
    return stats


# --- popularity -------------------------------------------------------------


def _minmax(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    return np.zeros_like(x) if hi <= lo else (x - lo) / (hi - lo)


def add_popularity(df, now=None):
    """Step 11's score, then strata computed WITHIN supertask (not globally).

    Global percentiles would hand every "head" slot to text-generation, which
    is exactly the failure being fixed.
    """
    now = now or pd.Timestamp.utcnow()
    created = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    age_days = (now - created).dt.total_seconds() / 86400.0
    df["age_days"] = age_days
    recency = np.exp(-age_days.fillna(3650) / 365.0)
    w = TX.POPULARITY_WEIGHTS
    df["pop_score"] = (
        w["downloads"] * _minmax(np.log1p(df["downloads"]))
        + w["likes"] * _minmax(np.log1p(df["likes"]))
        + w["recency"] * recency.to_numpy()
        + w["metadata"] * df["metadata_quality_score"].to_numpy()
    )
    df["is_recent"] = (age_days.fillna(9999) <= RECENT_DAYS)

    stratum = pd.Series(["long-tail"] * len(df), index=df.index, dtype=object)
    for task, g in df.groupby("supertask", observed=True):
        older = g.index[~g["is_recent"]]
        if len(older) == 0:
            continue
        s = df.loc[older, "pop_score"]
        hi, mid = s.quantile(0.75), s.quantile(0.35)
        stratum.loc[older] = np.where(s >= hi, "head", np.where(s >= mid, "mid", "long-tail"))
    stratum.loc[df.index[df["is_recent"]]] = "recent"
    df["popularity_stratum"] = stratum
    return df


# --- near duplicates --------------------------------------------------------


def add_duplicate_keys(df):
    df["near_duplicate_key"] = [
        TX.near_duplicate_key(root, task, size, lang, q, b, st, stem)
        for root, task, size, lang, q, b, st, stem in zip(
            df["canonical_root_id"], df["supertask"], df["size_b"],
            df["language_bucket"], df["quantization_type"], df["quantization_bits"],
            df["source_type"], df["repo_stem"])]
    df["near_duplicate_key_exact"] = [
        TX.near_duplicate_key_exact(k, q, b, m)
        for k, q, b, m in zip(df["near_duplicate_key"], df["quantization_type"],
                              df["quantization_bits"], df["model_id"])]
    return df


# --- main -------------------------------------------------------------------


def annotate(cdir, out_path, report_path):
    recs, prov = load_candidates(cdir)
    memb = load_membership(cdir)
    print("[load] %d candidate records, %d with membership rows" % (len(recs), len(memb)))

    df = annotate_records(recs, memb)
    print("[annotate] structural pass done")

    df = build_families(df)
    print("[families] %d distinct family_id, %d resolved by declared lineage"
          % (df["family_id"].nunique(),
             int((df["family_assignment_source"].isin(["baseModels", "cardData.base_model",
                                                       "declared_descendant"])).sum())))

    author_stats = build_author_stats(df)
    df["author_n_repos"] = df["author"].map(lambda a: author_stats[a]["n_repos"])
    df["author_mirror_score"] = df["author"].map(lambda a: author_stats[a]["mirror_score"])
    df["is_mirror_publisher"] = df["author"].map(lambda a: author_stats[a]["is_mirror_publisher"])

    # source type needs the parent's author, so it runs after the closure
    author_of = dict(zip(df["id_norm"], df["author"]))
    st, st_src, st_conf = [], [], []
    for rec_author, bases, relation, lib, qm, qs, adapter_tag in zip(
            df["author"], df["base_ids"], df["base_relation"], df["library_name"],
            df["quantization_type"], df["quantization_source"], df["_adapter_tags"]):
        parent_author = None
        for b in bases:
            parent_author = author_of.get(b) or (b.split("/")[0] if "/" in b else None)
            if parent_author:
                break
        same = bool(parent_author) and str(parent_author).lower() == str(rec_author).lower()
        a, b_, c = TX.source_type_of(qm, qs, lib, adapter_tag, bases, relation, same)
        st.append(a)
        st_src.append(b_)
        st_conf.append(c)
    df["source_type"] = st
    df["source_type_source"] = st_src
    df["source_type_confidence"] = st_conf

    df = add_popularity(df)
    df = add_duplicate_keys(df)

    fam_sizes = df["family_id"].value_counts()
    df["family_size"] = df["family_id"].map(fam_sizes)
    df["family_size_stratum"] = pd.cut(
        df["family_size"], bins=[0, 1, 9, 49, 10**9],
        labels=["singleton", "small", "medium", "large"]).astype(str)

    df["passes_quality"] = df["metadata_quality_score"] >= TX.METADATA_QUALITY_THRESHOLD
    df["taxonomy_version"] = TX.TAXONOMY_VERSION

    out = df.drop(columns=["base_ids", "_adapter_tags"]).copy()
    out["base_ids"] = df["base_ids"].map(lambda x: ";".join(x))
    out.to_parquet(out_path, index=False)

    report = {
        "written_at": utcnow(),
        "taxonomy_version": TX.TAXONOMY_VERSION,
        "candidates_dir": cdir,
        "snapshot_date_utc": prov.get("snapshot_date_utc"),
        "n_candidates": len(df),
        "n_authors": int(df["author"].nunique()),
        "n_families": int(df["family_id"].nunique()),
        "supertask": df["supertask"].value_counts().to_dict(),
        "supertask_source": df["supertask_source"].value_counts().to_dict(),
        "language_bucket": df["language_bucket"].value_counts().to_dict(),
        "language_source": df["language_source"].value_counts().to_dict(),
        "source_type": df["source_type"].value_counts().to_dict(),
        "quantization_type": df["quantization_type"].value_counts().to_dict(),
        "family_assignment_source": df["family_assignment_source"].value_counts().to_dict(),
        "family_size_stratum": df["family_size_stratum"].value_counts().to_dict(),
        "popularity_stratum": df["popularity_stratum"].value_counts().to_dict(),
        "size_source": df["size_source"].value_counts().to_dict(),
        "quality_pass_rate": float(df["passes_quality"].mean()),
        "unresolved_family_rate": float((df["family_assignment_source"] == "unresolved_singleton").mean()),
        "mirror_publishers": sorted(
            [{"author": a, **{k: v for k, v in s.items() if k != "mirror_reasons"}}
             for a, s in author_stats.items() if s["is_mirror_publisher"]],
            key=lambda x: -x["n_repos"])[:40],
        "unmapped_pipeline_tags": dict(TX.unmapped_pipeline_tags(recs).most_common(20)),
        "discovery_source_of_first_sight": df["discovery_source"].value_counts().to_dict(),
        "mean_queries_per_model": float(df["n_matching_queries"].mean()),
    }
    write_json_atomic(report_path, report)
    write_json_atomic(os.path.join(os.path.dirname(out_path), "AUTHOR_STATS.json"),
                      {a: {k: v for k, v in s.items()} for a, s in author_stats.items()
                       if s["n_repos"] >= 5})
    return df, report


def main(argv=None):
    p = argparse.ArgumentParser(description="annotate T2 v2 candidates")
    p.add_argument("--candidates", default=default_candidates_out("v2"))
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    out = args.out or os.path.join(args.candidates, "annotated.parquet")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df, report = annotate(args.candidates, out,
                          os.path.join(os.path.dirname(out), "ANNOTATE_REPORT.json"))
    print("\n[ok] %d annotated -> %s" % (len(df), out))
    for k in ("supertask", "language_bucket", "source_type", "family_size_stratum"):
        print("  %-20s %s" % (k, dict(sorted(report[k].items(), key=lambda kv: -kv[1])[:8])))
    print("  quality pass rate    %.3f" % report["quality_pass_rate"])
    print("  unresolved families  %.3f" % report["unresolved_family_rate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
