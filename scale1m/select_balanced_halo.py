"""
select_balanced_halo.py -- T2 v2 stage 3: choose the frozen HALO population
from the annotated candidate pool under hard quotas and caps.

Runbook: docs/1M/T2.md (v2), instructions Steps 6, 8-14; gates in Step 15.

THE ONE RULE THIS FILE EXISTS TO ENFORCE
    Do NOT sort by a scalar and take the top N. That is precisely what
    `sort=downloads` did, and it is why one publisher owned 24.2% of the v1
    population. Here a scalar score only ORDERS candidates *inside* a bucket
    the quota machinery has already decided to fill; the quotas and caps win.

DENOMINATOR RULE (the v1.1 correction)
    Every percentage cap is a share of the FINAL SELECTED POPULATION, never of
    the requested `--target`. Enforcing `text_generation <= 0.15 * target`
    during selection is unstable: passing a larger --target silently buys a
    looser absolute cap, and if the run then falls short of that target the
    cap is violated as a share of what was actually produced.

    So the caps are parameterized by a working size N, and the run searches for
    the largest N at which a population of exactly N is achievable
    (`find_feasible_max`). At that fixed point every cap is, by construction, a
    correct share of the actual output -- and the exit gates re-verify it
    against the real count rather than trusting the search.

    achieved(N) is non-decreasing in N (bigger N -> looser absolute caps) and
    achieved(N) - N is decreasing, so `achieved(N) >= N` is monotone in N and a
    binary search finds the exact maximum.

THE LANGUAGE DENOMINATOR
    ">= 75% English" is a claim about the language mix, so its denominator is
    the language-applicable population -- everything except `language-neutral`.
    Counting a ViT against the English share measures nothing and puts the rule
    in direct conflict with task balance.

ALGORITHM (quota-aware weighted round-robin, per working size N)
    Five quota axes tracked at once -- task, language, family size, source
    type, popularity. Each step: take the most deficient bucket across all
    axes, walk its score-ordered list until a candidate satisfies EVERY cap,
    admit it. Exhausted buckets close and their quota is redistributed. A
    second, relaxed pass fills the remainder against HARD CEILINGS only; caps
    are never relaxed.

DETERMINISM
    Ties break on sha1(model_id + seed); no set/dict iteration order is ever
    load-bearing. A rerun reproduces identical ids (gate G19).

Run (from ModelLakeFishing/):
    .\\.venv\\Scripts\\python.exe -m scale1m.select_balanced_halo `
        --annotated <dir>/annotated.parquet --target 69817
"""

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from scale1m.hf_crawl import utcnow, write_json_atomic
from scale1m.taxonomy import LANGUAGE_APPLICABLE, ENGLISH_SIDE

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "halo_quotas.json")

AXES = ("task", "language", "family_size", "source_type", "popularity")
AXIS_COLUMN = {"task": "supertask", "language": "language_bucket",
               "family_size": "family_size_stratum", "source_type": "source_type",
               "popularity": "popularity_stratum"}
QUANT_TYPES = ("quantized-conversion", "official-quantized")


def load_config(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def tiebreak(model_id, seed):
    return hashlib.sha1(("%s|%d" % (model_id, seed)).encode()).hexdigest()


# --- quota allocation ------------------------------------------------------


def sqrt_task_quota(counts, n_work, cfg):
    """q_t proportional to sqrt(N_t), bounded, with iterative redistribution.

    sqrt (not linear) is the point: it compresses the head. Linear allocation
    would hand text-generation ~26% of the lake because that is its share of
    the candidate pool -- the very concentration being removed.
    """
    max_n = int(cfg["max_share"] * n_work)
    min_rich = int(cfg["min_share_rich"] * n_work)
    rich_cut = cfg["rich_threshold_share"] * n_work
    caps = {}
    for t, n in counts.items():
        cap = min(n, max_n)
        if t == "unknown":
            cap = min(cap, int(cfg["unknown_max_share"] * n_work))
        elif t == "other":
            cap = min(cap, int(cfg["other_max_share"] * n_work))
        caps[t] = cap

    quota = {t: 0 for t in counts}
    remaining = n_work
    active = set(counts)
    for _ in range(64):
        if remaining <= 0 or not active:
            break
        w = {t: math.sqrt(counts[t]) for t in active}
        tot = sum(w.values()) or 1.0
        progressed = False
        for t in sorted(active):
            add = int(remaining * w[t] / tot)
            take = max(0, min(add, caps[t] - quota[t]))
            if take:
                quota[t] += take
                progressed = True
        for t in list(active):
            if quota[t] >= caps[t]:
                active.discard(t)
        used = sum(quota.values())
        if used >= n_work:
            break
        if not progressed:
            for t in sorted(active, key=lambda x: (-(caps[x] - quota[x]), x)):
                if sum(quota.values()) >= n_work:
                    break
                if quota[t] < caps[t]:
                    quota[t] += 1
            break
        remaining = n_work - used

    for t in sorted(counts):
        if counts[t] >= rich_cut and quota[t] < min(min_rich, caps[t]):
            quota[t] = min(min_rich, caps[t])
    return quota


def share_quota(counts, n_work, targets, max_share=None):
    """Fixed-share axis with supply clipping and deterministic redistribution."""
    quota = {}
    for b, share in targets.items():
        quota[b] = min(int(round(share * n_work)), counts.get(b, 0))
    for b in counts:
        quota.setdefault(b, 0)
    if max_share:
        for b, ms in max_share.items():
            if b in quota:
                quota[b] = min(quota[b], int(ms * n_work))
    short = n_work - sum(quota.values())
    if short > 0:
        for b in sorted(quota, key=lambda b: (-targets.get(b, 0.0), b)):
            if short <= 0:
                break
            room = counts.get(b, 0) - quota[b]
            if max_share and b in max_share:
                room = min(room, int(max_share[b] * n_work) - quota[b])
            take = max(0, min(short, room))
            quota[b] += take
            short -= take
    return quota


def language_quota(counts, n_work, cfg):
    """Language quota, expressed on the language-applicable sub-population.

    `language-neutral` gets its own share of the WHOLE population; the four
    language-bearing buckets split what remains, in the proportions the config
    states for the applicable population.
    """
    neutral = min(int(round(cfg["language_neutral_target_share"] * n_work)),
                  counts.get("language-neutral", 0),
                  int(cfg["language_neutral_max_share"] * n_work))
    applicable = max(0, n_work - neutral)
    q = share_quota({k: v for k, v in counts.items() if k != "language-neutral"},
                    applicable, cfg["targets_applicable"],
                    {"unknown": cfg["unknown_max_share_applicable"]})
    q["language-neutral"] = neutral
    return q


# --- scoring ----------------------------------------------------------------


def model_scores(df, cfg, task_counts):
    w = cfg["score_weights"]
    n = len(df)
    fam_imp = np.log1p(df["family_size"].to_numpy()) / max(math.log1p(df["family_size"].max()), 1e-9)
    task_rar = df["supertask"].map(lambda t: 1.0 - task_counts.get(t, 0) / max(n, 1)).to_numpy()
    lang_fit = df["language_bucket"].map(
        {"english-primary": 1.0, "multilingual-with-english": 0.9,
         "language-neutral": 0.6, "non-english": 0.5, "unknown": 0.0}).fillna(0.0).to_numpy()
    auth_pen = np.log1p(df["author_n_repos"].to_numpy()) / max(
        math.log1p(df["author_n_repos"].max()), 1e-9)
    dup_pen = np.log1p(
        df.groupby("near_duplicate_key")["model_id"].transform("size").to_numpy() - 1) / 6.0
    return (w["family_importance"] * fam_imp
            + w["task_rarity"] * task_rar
            + w["language_fit"] * lang_fit
            + w["metadata_quality"] * df["metadata_quality_score"].to_numpy()
            + w["popularity"] * df["pop_score"].to_numpy()
            - w["author_concentration_penalty"] * auth_pen
            - w["duplicate_penalty"] * np.clip(dup_pen, 0, 1)
            - w["mirror_penalty"] * df["author_mirror_score"].to_numpy())


# --- the selector -----------------------------------------------------------


class Selector:
    """Selects a population of size `n_work`. Every cap is a share of n_work,
    so a run that actually reaches n_work has caps that are correct shares of
    its own output. `find_feasible_max` is what guarantees that it does."""

    def __init__(self, df, cfg, n_work, precomputed=None):
        self.df = df
        self.cfg = cfg
        self.target = n_work
        self.caps = cfg["caps"]
        self.seed = cfg.get("seed", 0)

        # Hot-path columns as numpy: the round robin touches these millions of
        # times and pandas .iat is ~50x slower than an ndarray index.
        pre = precomputed or precompute(df, cfg)
        self.axis_val = pre["axis_val"]
        self.col = pre["col"]
        self.order = pre["order"]
        self.task_counts = pre["task_counts"]

        self.quota = {
            "task": sqrt_task_quota(self.task_counts, n_work, cfg["task"]),
            "language": language_quota(pre["counts"]["language"], n_work, cfg["language"]),
            "family_size": share_quota(pre["counts"]["family_size"], n_work,
                                       cfg["family_size"]["targets"]),
            "source_type": share_quota(pre["counts"]["source_type"], n_work,
                                       cfg["source_type"]["targets"],
                                       cfg["source_type"].get("max_share")),
            "popularity": share_quota(pre["counts"]["popularity"], n_work,
                                      cfg["popularity"]["targets"]),
        }
        self.taken = {a: Counter() for a in AXES}
        self.lists, self.ptr = {}, {}
        for a in AXES:
            vals = self.axis_val[a]
            by = defaultdict(list)
            for i in self.order:
                by[vals[i]].append(i)
            self.lists[a] = {k: np.asarray(v, dtype=np.int64) for k, v in by.items()}
            self.ptr[a] = {k: 0 for k in by}

        self.selected = []
        self.sel_mask = np.zeros(len(df), dtype=bool)
        self.family_ct, self.author_ct = Counter(), Counter()
        self.dup_ct, self.dup_exact_ct = Counter(), Counter()
        self.author_task_ct = Counter()
        self.reject = Counter()

        # absolute caps, all shares of n_work
        self.family_cap = min(self.caps["family_max_absolute"],
                              max(1, int(self.caps["family_max_share"] * n_work)))
        self.author_cap = max(1, int(self.caps["author_max_share"] * n_work))
        self.mirror_cap = max(1, int(self.caps["mirror_author_max_share"] * n_work))
        self.top10_cap = int(self.caps["top10_authors_max_share"] * n_work)
        self.task_ceiling = int(cfg["task"]["hard_ceiling"] * n_work)
        self.quant_combined_cap = int(
            cfg["source_type"]["combined_quantized_max_share"] * n_work)

        # running language-mix invariant (dynamic denominator)
        self.en_floor = cfg["language"]["english_plus_multi_min_share_applicable"]
        self.n_applicable = 0
        self.n_nonen_applicable = 0
        self.n_unknown_lang = 0
        self.unknown_lang_max = cfg["language"]["unknown_max_share_applicable"]
        self.n_quant = 0
        self._phase = "strict"

    # -- constraint checks -------------------------------------------------
    def _top10_ok(self, author):
        counts = self.author_ct.copy()
        counts[author] += 1
        return sum(c for _, c in counts.most_common(10)) <= self.top10_cap

    def eligible(self, i):
        if self.sel_mask[i]:
            return "already"
        author = self.col["author"][i]
        if self.col["is_mirror"][i]:
            if self.author_ct[author] >= self.mirror_cap:
                return "mirror_author_cap"
        elif self.author_ct[author] >= self.author_cap:
            return "author_cap"
        if self.family_ct[self.col["family_id"][i]] >= self.family_cap:
            return "family_cap"
        if self.dup_ct[self.col["dup"][i]] >= self.caps["near_duplicate_max"]:
            return "near_duplicate_cap"
        if self.dup_exact_ct[self.col["dup_exact"][i]] >= self.caps["near_duplicate_exact_max"]:
            return "near_duplicate_exact_cap"

        task = self.axis_val["task"][i]
        if self.taken["task"][task] >= self.task_ceiling:
            return "task_hard_ceiling"
        task_quota = self.quota["task"].get(task, 0)
        if task_quota and self.author_task_ct[(author, task)] >= max(
                1, int(self.caps["author_within_task_max_share"] * task_quota)):
            return "author_within_task_cap"

        for a in AXES:
            b = self.axis_val[a][i]
            if self.taken[a][b] >= self.quota[a].get(b, 0):
                return "axis_full:" + a

        lang = self.axis_val["language"][i]
        if lang in LANGUAGE_APPLICABLE and lang not in ENGLISH_SIDE:
            # Keep the >=75% English floor true at EVERY prefix, with the
            # language-applicable count as the denominator. Checking it only at
            # the end would mean discovering the violation once it is too late
            # to fix without discarding work.
            if (self.n_nonen_applicable + 1) > (1.0 - self.en_floor) * (self.n_applicable + 1):
                return "english_floor"
        if lang == "unknown":
            # Same dynamic denominator as the English floor (G11). Capping it
            # at a share of the WHOLE population instead lets it drift above
            # the gate: 3% of 69,817 is 3.57% of the 58,647 that actually have
            # a language to be unknown about.
            if (self.n_unknown_lang + 1) > self.unknown_lang_max * (self.n_applicable + 1):
                return "unknown_language_cap"

        if self.col["is_quant"][i] and self.n_quant >= self.quant_combined_cap:
            return "combined_quant_cap"
        if not self._top10_ok(author):
            return "top10_author_cap"
        return None

    def admit(self, i, trigger):
        self.sel_mask[i] = True
        self.selected.append((i, trigger))
        author = self.col["author"][i]
        self.author_ct[author] += 1
        self.family_ct[self.col["family_id"][i]] += 1
        self.dup_ct[self.col["dup"][i]] += 1
        self.dup_exact_ct[self.col["dup_exact"][i]] += 1
        self.author_task_ct[(author, self.axis_val["task"][i])] += 1
        for a in AXES:
            self.taken[a][self.axis_val[a][i]] += 1
        lang = self.axis_val["language"][i]
        if lang in LANGUAGE_APPLICABLE:
            self.n_applicable += 1
            if lang not in ENGLISH_SIDE:
                self.n_nonen_applicable += 1
            if lang == "unknown":
                self.n_unknown_lang += 1
        if self.col["is_quant"][i]:
            self.n_quant += 1

    # -- capacity ----------------------------------------------------------
    def duplicate_capacity(self):
        """How many models the near-duplicate caps allow AT MOST, pool-wide.

        Measured before selecting: it is an upper bound on any achievable N and
        the bracket for the feasibility search.
        """
        d = self.df
        keep = d.groupby("near_duplicate_key_exact", observed=True).head(
            self.caps["near_duplicate_exact_max"])
        keep = keep.groupby("near_duplicate_key", observed=True).head(
            self.caps["near_duplicate_max"])
        return int(len(keep))

    def relax_quota(self):
        """Phase-2 quotas: axis targets become HARD CEILINGS only.

        Preserved exactly where the instructions make it a hard gate (task
        <= hard_ceiling, third-party quantized <= 15%, combined quantized
        <= 20%, unknown language/task capped, the English floor). Released
        everywhere a bounded deviation is allowed (G13-G15, "where supply
        permits"). Caps -- author, mirror, family, near-duplicate, top-10 --
        are NEVER relaxed; they are the point of the redesign.
        """
        n = self.target
        cfg = self.cfg
        q = {a: dict(self.quota[a]) for a in AXES}
        for b in q["task"]:
            if b == "unknown":
                q["task"][b] = int(cfg["task"]["unknown_max_share"] * n)
            elif b == "other":
                q["task"][b] = int(cfg["task"]["other_max_share"] * n)
            else:
                q["task"][b] = self.task_ceiling
        lang_cfg = cfg["language"]
        for b in q["language"]:
            if b == "language-neutral":
                q["language"][b] = int(lang_cfg["language_neutral_max_share"] * n)
            elif b == "unknown":
                q["language"][b] = int(lang_cfg["unknown_max_share_applicable"] * n)
            else:
                q["language"][b] = n           # the running floor governs these
        for b in q["family_size"]:
            q["family_size"][b] = n
        st_max = cfg["source_type"].get("max_share", {})
        for b in q["source_type"]:
            q["source_type"][b] = int(st_max[b] * n) if b in st_max else n
        for b in q["popularity"]:
            q["popularity"][b] = n
        return q

    # -- the round robin ---------------------------------------------------
    def run(self, verbose=True):
        self._phase = "strict"
        n_strict = self._round_robin(verbose)
        self.n_strict = n_strict
        self.strict_quota = {a: dict(self.quota[a]) for a in AXES}
        if len(self.selected) < self.target:
            if verbose:
                print("   [phase 2] strict quotas gave %d/%d; relaxing axis targets to "
                      "hard ceilings (caps stay absolute)" % (n_strict, self.target))
            self.quota = self.relax_quota()
            self._phase = "relaxed"
            self._round_robin(verbose)
        self.n_relaxed = len(self.selected) - n_strict
        return self.selected

    def _round_robin(self, verbose=True):
        prio = self.cfg["axis_priority"]
        closed = set()
        start = len(self.selected)
        while len(self.selected) < self.target:
            best, best_key = None, None
            for a in AXES:
                pa = prio.get(a, 1.0)
                for b, q in self.quota[a].items():
                    if q <= 0 or (a, b) in closed:
                        continue
                    deficit = (q - self.taken[a][b]) / q
                    if deficit <= 0:
                        continue
                    key = (deficit * pa, a, str(b))
                    if best_key is None or key > best_key:
                        best, best_key = (a, b), key
            if best is None:
                break
            a, b = best
            lst = self.lists[a].get(b)
            if lst is None:
                closed.add((a, b))
                continue
            p = self.ptr[a][b]
            admitted = False
            while p < len(lst):
                i = int(lst[p])
                p += 1
                why = self.eligible(i)
                if why is None:
                    self.ptr[a][b] = p
                    self.admit(i, "%s:%s=%s" % (self._phase, a, b))
                    admitted = True
                    break
                self.reject[why] += 1
            if not admitted:
                self.ptr[a][b] = p
                closed.add((a, b))
            elif verbose and len(self.selected) % 10000 == 0:
                print("   ... %d / %d selected" % (len(self.selected), self.target), flush=True)
        return len(self.selected) - start

    # -- outputs -----------------------------------------------------------
    def deficits(self):
        """Reported against the STRICT quota -- the relaxed pass fills the
        population, it does not make the strict target met."""
        out = []
        strict = getattr(self, "strict_quota", self.quota)
        for a in AXES:
            for b, q in strict[a].items():
                miss = q - self.taken[a][b]
                if miss > 0:
                    out.append({"axis": a, "bucket": str(b), "target": int(q),
                                "taken": int(self.taken[a][b]), "missing": int(miss),
                                "supply": int((self.axis_val[a] == b).sum())})
        return sorted(out, key=lambda d: -d["missing"])

    def result(self):
        idx = [i for i, _ in self.selected]
        out = self.df.iloc[idx].copy()
        out["selection_trigger"] = [t for _, t in self.selected]
        out["selection_phase"] = [t.split(":")[0] for _, t in self.selected]
        out["selection_order"] = range(len(idx))
        out["selection_provenance"] = [
            "trigger=%s;score=%.5f;tb=%s;taxonomy=%s" % (t, s, tb[:12], tx)
            for (_, t), s, tb, tx in zip(self.selected, out["score"], out["_tb"],
                                         out["taxonomy_version"])]
        return out.drop(columns=["_tb"])


def precompute(df, cfg):
    """Shared, size-independent setup so the feasibility search does not redo
    the scoring and sorting once per probe."""
    task_counts = df["supertask"].value_counts().to_dict()
    df["score"] = model_scores(df, cfg, task_counts)
    df["_tb"] = [tiebreak(m, cfg.get("seed", 0)) for m in df["model_id"]]
    order = df.sort_values(["score", "_tb"], ascending=[False, True]).index.to_numpy()
    pos = {ix: k for k, ix in enumerate(df.index)}
    order = np.asarray([pos[i] for i in order], dtype=np.int64)
    axis_val = {a: df[AXIS_COLUMN[a]].to_numpy() for a in AXES}
    col = {
        "author": df["author"].to_numpy(),
        "family_id": df["family_id"].to_numpy(),
        "dup": df["near_duplicate_key"].to_numpy(),
        "dup_exact": df["near_duplicate_key_exact"].to_numpy(),
        "is_mirror": df["is_mirror_publisher"].to_numpy(),
        "is_quant": df["source_type"].isin(QUANT_TYPES).to_numpy(),
    }
    counts = {a: df[AXIS_COLUMN[a]].value_counts().to_dict() for a in AXES}
    return {"axis_val": axis_val, "col": col, "order": order,
            "task_counts": task_counts, "counts": counts}


def find_feasible_max(pool, cfg, upper, verbose=True):
    """Largest N with achieved(N) >= N -- the exact feasible maximum.

    Monotone because a larger N loosens every absolute cap, so `achieved(N)>=N`
    flips exactly once. Reported as `feasibility_probe` so the search itself is
    auditable rather than a number that appeared from nowhere.
    """
    pre = precompute(pool, cfg)
    tol = cfg["feasible_search"]["rel_tolerance"]
    max_it = cfg["feasible_search"]["max_iterations"]
    probes, best = [], None

    def probe(n):
        sel = Selector(pool, cfg, n, precomputed=pre)
        sel.run(verbose=False)
        got = len(sel.selected)
        probes.append({"n_work": int(n), "achieved": int(got), "feasible": bool(got >= n)})
        if verbose:
            print("   [probe] N=%7d -> achieved %7d  %s"
                  % (n, got, "FEASIBLE" if got >= n else "short by %d" % (n - got)),
                  flush=True)
        return sel, got

    lo, hi = 1, int(upper)
    sel_hi, got_hi = probe(hi)
    if got_hi >= hi:
        return sel_hi, hi, probes
    best_sel, best_n = None, 0
    for _ in range(max_it):
        if hi - lo <= max(1, int(tol * hi)):
            break
        mid = (lo + hi) // 2
        sel, got = probe(mid)
        if got >= mid:
            lo, best_sel, best_n = mid, sel, mid
        else:
            hi = mid
    if best_sel is None:
        best_sel, got = probe(lo)
        best_n = lo if got >= lo else got
    return best_sel, best_n, probes


# --- deficit -> backfill plan ----------------------------------------------

SUPERTASK_TO_PIPELINE = defaultdict(list)


def _init_reverse_map():
    from scale1m.taxonomy import PIPELINE_TO_SUPERTASK
    for pt, st in PIPELINE_TO_SUPERTASK.items():
        SUPERTASK_TO_PIPELINE[st].append(pt)


_init_reverse_map()

LANG_BUCKET_TO_CODES = {
    "non-english": ["zh", "es", "fr", "de", "ja", "ru", "ar", "pt", "ko", "it",
                    "hi", "nl", "tr", "pl", "vi", "id", "fa", "th", "bn", "ta"],
    "multilingual-with-english": ["multilingual", "en"],
    "english-primary": ["en"],
    "language-neutral": [],
}
SOURCE_TYPE_TO_FILTERS = {
    "adapter-lora": ["lora", "peft", "adapter"],
    "original-base": ["base_model:finetune"],
    "official-derivative": ["trl", "sft"],
    "community-finetune": ["dpo", "orpo", "grpo", "mergekit"],
    "official-quantized": ["gguf", "awq"],
}


def enrich_deficits(deficits, df):
    """Attach the query handles the backfill planner needs (Step 14)."""
    out = []
    for d in deficits:
        d = dict(d)
        if d["axis"] == "task":
            d["pipeline_tags"] = SUPERTASK_TO_PIPELINE.get(d["bucket"], [])[:6]
        elif d["axis"] == "language":
            d["language_codes"] = LANG_BUCKET_TO_CODES.get(d["bucket"], [])
        elif d["axis"] == "source_type":
            d["filters"] = SOURCE_TYPE_TO_FILTERS.get(d["bucket"], [])
        elif d["axis"] == "family_size":
            sub = df[df["family_size_stratum"] == d["bucket"]]
            d["authors"] = list(sub["author"].value_counts().head(8).index)
        out.append(d)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="select the balanced HALO population")
    p.add_argument("--annotated", required=True)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--target", type=int, default=None,
                   help="desired population; the run selects min(target, feasible max)")
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--exclude-core", default=None,
                   help="CORE graph (.pt): its models are NOT HALO candidates. "
                        "The rung needs CORE + HALO to be DISJOINT -- 1,892 of a "
                        "69,817 selection were CORE members before this was added, "
                        "which would have made the 100K lake 98,108 distinct models.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    df = pd.read_parquet(args.annotated)
    target = args.target or cfg["target"]
    out_dir = args.out or os.path.join(os.path.dirname(args.annotated), "selected")
    os.makedirs(out_dir, exist_ok=True)

    n_all = len(df)
    n_core_excluded = 0
    if args.exclude_core:
        import torch
        core = torch.load(args.exclude_core, weights_only=False)
        core_ids = {str(m).strip().lower() for m in core["unique_model_id"]["model"]}
        before = len(df)
        df = df[~df["id_norm"].isin(core_ids)].copy()
        n_core_excluded = before - len(df)
        print("[exclude] dropped %d CORE members from the candidate pool "
              "(CORE and HALO must be disjoint)" % n_core_excluded)
    pool = df[df["passes_quality"] & ~df["disabled"]].copy()
    print("[pool] %d candidates -> %d pass the quality gate (%.1f%%)"
          % (n_all, len(pool), 100.0 * len(pool) / max(n_all, 1)))
    if len(pool) < target:
        print("[warn] quality-passing supply %d < target %d; admitting flagged "
              "low-metadata exceptions" % (len(pool), target))
        extra = df[~df.index.isin(pool.index) & ~df["disabled"]].copy()
        extra = extra.sort_values("metadata_quality_score", ascending=False)
        room = int(cfg["quality"]["low_metadata_exception_max_share"] * target)
        extra = extra.head(max(0, min(room, target - len(pool))))
        extra["low_metadata_exception"] = True
        extra["exception_reason"] = "quality-passing supply below target"
        pool["low_metadata_exception"] = False
        pool["exception_reason"] = ""
        pool = pd.concat([pool, extra], ignore_index=True)
    else:
        pool["low_metadata_exception"] = False
        pool["exception_reason"] = ""
    pool = pool.reset_index(drop=True)

    capacity = Selector(pool, cfg, max(target, 1)).duplicate_capacity()
    print("[capacity] near-duplicate caps allow at most %d of %d pool records (%.1f%%)"
          % (capacity, len(pool), 100.0 * capacity / max(len(pool), 1)))

    upper = min(target, capacity)
    print("[feasible] searching for the largest N whose caps hold as shares of N, "
          "upper bound %d" % upper)
    sel, feasible_n, probes = find_feasible_max(pool, cfg, upper, verbose=not args.quiet)
    result = sel.result()
    deficits = enrich_deficits(sel.deficits(), pool)

    result.to_parquet(os.path.join(out_dir, "selected_halo.parquet"), index=False)
    result[["model_id", "author", "supertask", "language_bucket", "family_id",
            "family_size_stratum", "source_type", "popularity_stratum",
            "selection_provenance"]].to_csv(
        os.path.join(out_dir, "selected_halo.csv"), index=False)

    manifest = {
        "written_at": utcnow(),
        "config_version": cfg.get("config_version"),
        "config_path": os.path.abspath(args.config),
        "annotated": os.path.abspath(args.annotated),
        "seed": cfg["seed"],
        "requested_target": target,
        "feasible_max": int(feasible_n),
        "target": int(len(result)),
        "n_selected": len(result),
        "n_strict_phase": int(getattr(sel, "n_strict", len(result))),
        "n_relaxed_phase": int(getattr(sel, "n_relaxed", 0)),
        "near_duplicate_capacity": capacity,
        "feasibility_probe": probes,
        "n_pool": len(pool),
        "n_candidates": n_all,
        "n_core_excluded": n_core_excluded,
        "exclude_core": os.path.abspath(args.exclude_core) if args.exclude_core else None,
        "strict_quota": {a: {str(k): int(v) for k, v in sel.strict_quota[a].items()}
                         for a in AXES},
        "final_quota": {a: {str(k): int(v) for k, v in sel.quota[a].items()} for a in AXES},
        "taken": {a: {str(k): int(v) for k, v in sel.taken[a].items()} for a in AXES},
        "caps_resolved": {"basis_n": sel.target, "family": sel.family_cap,
                          "author": sel.author_cap, "mirror_author": sel.mirror_cap,
                          "top10": sel.top10_cap, "task_hard_ceiling": sel.task_ceiling,
                          "combined_quantized": sel.quant_combined_cap,
                          "near_duplicate": cfg["caps"]["near_duplicate_max"],
                          "near_duplicate_exact": cfg["caps"]["near_duplicate_exact_max"]},
        "rejections": dict(sel.reject.most_common()),
        "deficits": deficits,
        "score_weights": cfg["score_weights"],
    }
    write_json_atomic(os.path.join(out_dir, "SELECTION_MANIFEST.json"), manifest)
    write_json_atomic(os.path.join(out_dir, "DEFICITS.json"), {"deficits": deficits})

    print("\n[ok] selected %d (requested %d, feasible max %d) -> %s"
          % (len(result), target, feasible_n, out_dir))
    print("     strict %d + relaxed %d" % (getattr(sel, "n_strict", 0),
                                           getattr(sel, "n_relaxed", 0)))
    if deficits:
        print("  strict-quota deficits (%d buckets):" % len(deficits))
        for d in deficits[:8]:
            print("    %-13s %-28s missing %6d of %6d (supply %d)"
                  % (d["axis"], d["bucket"], d["missing"], d["target"], d["supply"]))
    print("  top rejections:", dict(sel.reject.most_common(6)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
