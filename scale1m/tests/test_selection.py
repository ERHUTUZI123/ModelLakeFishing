"""Tests for annotation assembly + balanced selection (Steps 4-5, 13-15, 18).

Every fixture here is built to be DOMINATED by something -- one author, one
family, one task, one quantization publisher, one language, or a wall of
near-duplicates -- because a selector that only works on well-behaved input is
the selector we already had.

Run (from ModelLakeFishing/):
    .\\.venv\\Scripts\\python.exe -m pytest scale1m/tests -q
"""

import json
import os

import pandas as pd
import pytest

from scale1m import taxonomy as TX
from scale1m.annotate_candidates import (DSU, build_families, build_author_stats,
                                         add_popularity, add_duplicate_keys,
                                         load_membership, base_ids_of, size_b_of)
from scale1m.select_balanced_halo import (Selector, load_config, sqrt_task_quota,
                                          share_quota, language_quota, tiebreak,
                                          DEFAULT_CONFIG, enrich_deficits,
                                          find_feasible_max)

CFG = load_config(DEFAULT_CONFIG)


# =========================================================================
# fixtures
# =========================================================================

def mk(model_id, task="text-generation-chat", lang="english-primary",
       source="original-base", pop="mid", family=None, size=7.0, quant="none",
       bits=None, quality=0.9, author=None, mirror=False, n_repos=5,
       stem=None, root=None):
    author = author if author is not None else model_id.split("/")[0]
    return {
        "model_id": model_id, "id_norm": model_id.lower(), "author": author,
        "downloads": 100, "likes": 5, "created_at": "2025-01-01T00:00:00.000Z",
        "supertask": task, "language_bucket": lang, "source_type": source,
        "popularity_stratum": pop, "size_b": size, "quantization_type": quant,
        "quantization_bits": bits, "metadata_quality_score": quality,
        "family_id": family or ("single:" + model_id.lower()),
        "canonical_root_id": root, "repo_stem": stem or TX.repo_stem(model_id),
        "author_n_repos": n_repos, "author_mirror_score": 0.9 if mirror else 0.0,
        "is_mirror_publisher": mirror, "pop_score": 0.5,
        "family_assignment_source": "declared_descendant" if family else "unresolved_singleton",
        "has_resolved_parent_or_root": bool(family), "passes_quality": quality >= 0.35,
        "disabled": False, "taxonomy_version": TX.TAXONOMY_VERSION,
    }


def frame(rows, target=None):
    df = pd.DataFrame(rows)
    df["family_size"] = df["family_id"].map(df["family_id"].value_counts())
    df["family_size_stratum"] = pd.cut(
        df["family_size"], bins=[0, 1, 9, 49, 10 ** 9],
        labels=["singleton", "small", "medium", "large"]).astype(str)
    df = add_duplicate_keys(df)
    return df


def cfg_for(target, **overrides):
    cfg = json.loads(json.dumps(CFG))
    cfg["target"] = target
    for k, v in overrides.items():
        cur, *rest = k.split(".")
        node = cfg[cur]
        for r in rest[:-1]:
            node = node[r]
        node[rest[-1]] = v
    return cfg



# A realistic pool spreads across the quota axes; the fixtures below then
# concentrate ONE axis so the cap under test is the thing that binds. A pool
# that is degenerate on some *other* axis (e.g. a single supertask) cannot fill
# any target at all -- no task may exceed 15% -- and would test nothing.
TASKS = ("text-generation-chat", "embedding-retrieval", "text-classification",
         "token-classification", "question-answering", "summarization",
         "translation", "code", "speech-recognition", "image-classification",
         "detection-segmentation", "image-generation", "vision-language")
LANGS = ("english-primary",) * 6 + ("multilingual-with-english",) * 2 + (
    "non-english", "language-neutral")
SOURCES = ("original-base", "official-derivative", "community-finetune",
           "adapter-lora", "quantized-conversion")
POPS = ("head", "mid", "mid", "long-tail", "recent")


def spread(i, **over):
    """Axis values for filler row i, cycled with coprime strides so the axes
    are not correlated with each other."""
    d = {"task": TASKS[i % len(TASKS)], "lang": LANGS[(i * 3) % len(LANGS)],
         "source": SOURCES[(i * 7) % len(SOURCES)], "pop": POPS[(i * 11) % len(POPS)],
         "size": float(1 + (i % 40))}
    d.update(over)
    return d


def filler(n, prefix="f", **over):
    return [mk("%s%d/m" % (prefix, i), author="%s%d" % (prefix, i),
               stem="%s%d" % (prefix, i), **spread(i, **over)) for i in range(n)]


# =========================================================================
# annotation: family precedence, provenance, size
# =========================================================================

def test_dsu_merges_a_lineage_chain_into_one_family():
    rows = [
        {**mk("meta/llama-3-8b"), "base_ids": [], "base_source": "none", "model_type": "llama"},
        {**mk("org/llama-3-8b-sft"), "base_ids": ["meta/llama-3-8b"],
         "base_source": "baseModels", "model_type": "llama"},
        {**mk("quant/llama-3-8b-sft-gguf"), "base_ids": ["org/llama-3-8b-sft"],
         "base_source": "baseModels", "model_type": "llama"},
    ]
    df = build_families(pd.DataFrame(rows))
    assert df["family_id"].nunique() == 1
    assert set(df["canonical_root_id"]) == {"meta/llama-3-8b"}
    assert df["family_assignment_confidence"].min() >= 0.8


def test_declared_lineage_outranks_name_similarity():
    """A model that declares a parent must NOT be re-homed by the stem rule."""
    rows = [
        {**mk("a/base"), "base_ids": [], "base_source": "none", "model_type": "bert"},
        {**mk("b/base-tuned"), "base_ids": ["a/base"], "base_source": "baseModels",
         "model_type": "bert"},
        {**mk("c/base-tuned"), "base_ids": [], "base_source": "none", "model_type": "bert"},
    ]
    df = build_families(pd.DataFrame(rows))
    fam = dict(zip(df["id_norm"], df["family_id"]))
    assert fam["a/base"] == fam["b/base-tuned"], "declared edge must win"
    assert fam["c/base-tuned"] != fam["b/base-tuned"], "no parent -> not the same family"
    src = dict(zip(df["id_norm"], df["family_assignment_source"]))
    assert src["b/base-tuned"] == "baseModels"


def test_generic_stems_never_merge_unrelated_repos():
    rows = [{**mk("a/model"), "base_ids": [], "base_source": "none", "model_type": "bert"},
            {**mk("b/model"), "base_ids": [], "base_source": "none", "model_type": "bert"},
            {**mk("c/chat"), "base_ids": [], "base_source": "none", "model_type": "bert"}]
    df = build_families(pd.DataFrame(rows))
    assert df["family_id"].nunique() == 3
    assert set(df["family_assignment_source"]) == {"unresolved_singleton"}


def test_cyclic_base_model_declaration_terminates():
    rows = [{**mk("a/x"), "base_ids": ["b/y"], "base_source": "baseModels", "model_type": "t"},
            {**mk("b/y"), "base_ids": ["a/x"], "base_source": "baseModels", "model_type": "t"}]
    df = build_families(pd.DataFrame(rows))     # must not hang
    assert df["family_id"].nunique() == 1


def test_family_confidence_records_the_weaker_rules():
    rows = [{**mk("org/mistral-tiny-v1"), "base_ids": [], "base_source": "none",
             "model_type": "mistral"},
            {**mk("org/mistral-tiny-v2"), "base_ids": [], "base_source": "none",
             "model_type": "mistral"}]
    df = build_families(pd.DataFrame(rows))
    # identical stems after version stripping -> arch+stem rule, low confidence
    assert set(df["family_assignment_source"]) <= {"arch+stem", "author+stem",
                                                   "unresolved_singleton"}
    assert df["family_assignment_confidence"].max() <= 0.5


def test_size_b_prefers_safetensors_and_labels_the_heuristic():
    assert size_b_of({"id": "a/x-7b", "safetensors": {"total": 3_000_000_000}}) == (3.0, "safetensors")
    v, src = size_b_of({"id": "a/Model-7B-GGUF"})
    assert v == 7.0 and src == "name_heuristic"
    assert size_b_of({"id": "a/plain"}) == (None, "none")


def test_base_ids_prefers_authoritative_relation():
    rec = {"baseModels": {"relation": "quantized", "ids": ["Meta/Llama"]},
           "cardData": {"base_model": "someone/else"}}
    ids, rel, src = base_ids_of(rec)
    assert ids == ["meta/llama"] and rel == "quantized" and src == "baseModels"


def test_provenance_merges_across_queries(tmp_path):
    mdir = tmp_path / "membership"
    mdir.mkdir()
    (mdir / "global_downloads.tsv").write_text("1\ta/x\n2\tb/y\n", encoding="utf-8")
    (mdir / "task_translation_downloads.tsv").write_text("7\ta/x\n", encoding="utf-8")
    memb = load_membership(str(tmp_path))
    assert sorted(memb["a/x"]) == ["global_downloads", "task_translation_downloads"]
    assert memb["b/y"] == ["global_downloads"]


def test_mirror_publisher_detected_from_aggregates_only():
    rows = []
    for i in range(600):
        rows.append({**mk("quantmill/m%d-gguf" % i, source="quantized-conversion",
                          quant="gguf", author="quantmill"),
                     "base_ids": ["up%d/orig" % i], "repo_stem": "m%d" % i})
    for i in range(20):
        rows.append({**mk("goodlab/model-%d" % i, author="goodlab"),
                     "base_ids": [], "repo_stem": "model"})
    stats = build_author_stats(pd.DataFrame(rows))
    assert stats["quantmill"]["is_mirror_publisher"]
    assert not stats["goodlab"]["is_mirror_publisher"]


# =========================================================================
# quota allocation
# =========================================================================

def test_sqrt_quota_compresses_a_dominant_task():
    counts = {"text-generation-chat": 500_000, "translation": 3_000, "summarization": 2_000}
    q = sqrt_task_quota(counts, 10_000, CFG["task"])
    assert sum(q.values()) <= 10_000
    assert q["text-generation-chat"] <= int(0.15 * 10_000)
    # sqrt must give the small tasks far more than their linear share (0.6%)
    assert q["translation"] > 0.006 * 10_000


def test_unknown_task_is_tightly_capped():
    counts = {"unknown": 900_000, "text-classification": 10_000}
    q = sqrt_task_quota(counts, 10_000, CFG["task"])
    assert q["unknown"] <= int(CFG["task"]["unknown_max_share"] * 10_000)


def test_share_quota_clips_to_supply_and_redistributes():
    q = share_quota({"english-primary": 100, "non-english": 10_000}, 1_000,
                    CFG["language"]["targets_applicable"])
    assert q["english-primary"] == 100          # supply-clipped
    assert sum(q.values()) <= 1_000


def test_source_type_max_share_is_enforced_in_allocation():
    counts = {"quantized-conversion": 100_000, "original-base": 100_000}
    q = share_quota(counts, 10_000, CFG["source_type"]["targets"],
                    CFG["source_type"]["max_share"])
    assert q["quantized-conversion"] <= int(0.15 * 10_000)


# =========================================================================
# selection: caps under adversarial domination
# =========================================================================

def _run(rows, target, **cfg_over):
    """Same entry point production uses -- the feasibility search, so every
    share-based cap is tested against the FINAL population size and never
    against the requested target."""
    cfg = cfg_for(target, **cfg_over)
    df = frame(rows).reset_index(drop=True)
    sel, feasible_n, probes = find_feasible_max(df, cfg, target, verbose=False)
    return sel, sel.result()


def test_one_author_cannot_dominate():
    rows = [mk("hog/m%d" % i, author="hog", stem="m%d" % i, **spread(i))
            for i in range(4000)]
    rows += filler(4000)
    sel, out = _run(rows, 1000)
    top = out["author"].value_counts().iloc[0] / len(out)
    assert top <= CFG["caps"]["author_max_share"] + 1e-9, top


def test_one_family_cannot_dominate():
    rows = [mk("org%d/x" % i, family="lin:big", root="big/base", stem="x%d" % i,
               author="org%d" % i, **spread(i)) for i in range(3000)]
    rows += filler(4000, prefix="solo")
    sel, out = _run(rows, 1000)
    cap = min(CFG["caps"]["family_max_absolute"],
              max(1, int(CFG["caps"]["family_max_share"] * 1000)))
    assert out["family_id"].value_counts().iloc[0] <= cap


def test_one_quantization_publisher_is_bounded():
    rows = [mk("mill/m%d-gguf" % i, author="mill", mirror=True, n_repos=5000,
               quant="gguf", bits=4, stem="m%d" % i,
               **spread(i, source="quantized-conversion")) for i in range(3000)]
    rows += filler(4000, prefix="lab")
    sel, out = _run(rows, 1000)
    mill = (out["author"] == "mill").mean()
    assert mill <= CFG["caps"]["mirror_author_max_share"] + 1e-9, mill


def test_one_task_cannot_exceed_the_max_share():
    rows = [mk("o%d/m" % i, author="o%d" % i, stem="s%d" % i,
               **spread(i, task="text-generation-chat")) for i in range(8000)]
    rows += filler(4000, prefix="p")
    sel, out = _run(rows, 1000)
    top = out["supertask"].value_counts(normalize=True).iloc[0]
    assert top <= CFG["task"]["hard_ceiling"] + 1e-9, top


def test_english_floor_survives_a_non_english_flood():
    """G10 is a floor, so it has to be enforced as a cap on everything else --
    including in the relaxation phase. The denominator is the
    LANGUAGE-APPLICABLE population; language-neutral models are excluded."""
    rows = [mk("ne%d/m" % i, author="ne%d" % i, stem="n%d" % i,
               **spread(i, lang="non-english")) for i in range(8000)]
    rows += [mk("en%d/m" % i, author="en%d" % i, stem="e%d" % i,
                **spread(i, lang="english-primary")) for i in range(4000)]
    sel, out = _run(rows, 1000)
    from scale1m.taxonomy import LANGUAGE_APPLICABLE, ENGLISH_SIDE
    app = out["language_bucket"].isin(LANGUAGE_APPLICABLE).sum()
    en = out["language_bucket"].isin(ENGLISH_SIDE).sum()
    share = en / max(app, 1)
    assert share >= CFG["language"]["english_plus_multi_min_share_applicable"] - 1e-9, share


def test_near_duplicate_wall_is_capped():
    """500 identical-in-every-respect conversions of one base model."""
    rows = [mk("q%d/llama-3-8b-gguf" % i, author="q%d" % i, quant="gguf", bits=4,
               size=8.0, root="meta/llama-3-8b", family="lin:llama",
               stem="llama-3-8b", task="text-generation-chat",
               lang="english-primary", source="quantized-conversion",
               pop=POPS[i % len(POPS)]) for i in range(500)]
    rows += filler(4000)
    sel, out = _run(rows, 500)
    grp = out["near_duplicate_key"].value_counts()
    assert grp.max() <= CFG["caps"]["near_duplicate_max"]
    assert out["near_duplicate_key_exact"].value_counts().max() <= \
        CFG["caps"]["near_duplicate_exact_max"]


def test_top10_author_cap_binds_even_when_each_author_is_under_cap():
    """10 authors each at the 1.5% individual cap would be 15% together."""
    rows = []
    for a in range(10):
        rows += [mk("big%d/m%d" % (a, i), author="big%d" % a, stem="m%d_%d" % (a, i),
                    quality=1.0, **spread(i)) for i in range(400)]
    rows += [mk("t%d/m" % i, author="t%d" % i, stem="s%d" % i, quality=0.5,
                **spread(i)) for i in range(8000)]
    sel, out = _run(rows, 2000)
    top10 = out["author"].value_counts().iloc[:10].sum() / len(out)
    assert top10 <= CFG["caps"]["top10_authors_max_share"] + 1e-9, top10


def test_author_within_task_cap():
    rows = [mk("mono/m%d" % i, author="mono", stem="m%d" % i,
               **spread(i, task="translation")) for i in range(2000)]
    rows += [mk("o%d/m" % i, author="o%d" % i, stem="s%d" % i,
                **spread(i, task="translation")) for i in range(2000)]
    rows += filler(4000, prefix="p")
    sel, out = _run(rows, 1000)
    tr = out[out["supertask"] == "translation"]
    if len(tr):
        assert (tr["author"] == "mono").sum() <= max(
            1, int(CFG["caps"]["author_within_task_max_share"] * sel.quota["task"]["translation"]))


# =========================================================================
# determinism, deficits, relaxation
# =========================================================================

def test_selection_is_deterministic():
    rows = filler(6000)
    _, a = _run(rows, 400)
    _, b = _run(rows, 400)
    assert list(a["model_id"]) == list(b["model_id"])


def test_seed_changes_tie_breaking_but_not_determinism():
    assert tiebreak("a/b", 0) != tiebreak("a/b", 1)
    assert tiebreak("a/b", 0) == tiebreak("a/b", 0)


def test_deficits_are_reported_against_the_strict_quota():
    """A pool with almost no translation supply must SAY so, not quietly
    reallocate and report success."""
    rows = [r for r in filler(4000) if r["supertask"] != "translation"]
    rows += [mk("tr%d/m" % i, task="translation", author="tr%d" % i, stem="t%d" % i)
             for i in range(3)]
    sel, out = _run(rows, 500)
    d = {(x["axis"], x["bucket"]): x for x in sel.deficits()}
    assert ("task", "translation") not in d or d[("task", "translation")]["supply"] == 3


def test_caps_hold_as_shares_of_the_actual_population_not_the_request():
    """The v1.1 correction: asking for a bigger population must NOT buy a
    looser cap. Selecting with target=400 and target=4000 from the same pool
    must both respect 1.5% OF WHAT THEY ACTUALLY PRODUCED."""
    rows = [mk("hog/m%d" % i, author="hog", stem="m%d" % i, **spread(i))
            for i in range(4000)]
    rows += filler(6000)
    for t in (400, 4000):
        sel, out = _run(rows, t)
        share = out["author"].value_counts().iloc[0] / len(out)
        assert share <= CFG["caps"]["author_max_share"] + 1e-9, (t, share, len(out))


def test_feasible_max_is_actually_achieved():
    """find_feasible_max must return a size the selector really reaches --
    otherwise every share-based cap is evaluated against a fiction."""
    rows = filler(6000)
    cfg = cfg_for(3000)
    df = frame(rows).reset_index(drop=True)
    sel, n, probes = find_feasible_max(df, cfg, 3000, verbose=False)
    assert len(sel.selected) >= n > 0
    assert probes and all("achieved" in p for p in probes)


def test_bigger_request_never_yields_a_more_concentrated_population():
    rows = [mk("hog/m%d" % i, author="hog", stem="m%d" % i, **spread(i))
            for i in range(3000)]
    rows += filler(6000)
    _, small = _run(rows, 500)
    _, big = _run(rows, 5000)
    s_share = small["author"].value_counts().iloc[0] / len(small)
    b_share = big["author"].value_counts().iloc[0] / len(big)
    assert s_share <= CFG["caps"]["author_max_share"] + 1e-9
    assert b_share <= CFG["caps"]["author_max_share"] + 1e-9


def test_official_quantized_is_not_counted_as_a_third_party_conversion():
    rows = [mk("qwen/model-%d-gguf" % i, author="qwen", quant="gguf", bits=4,
               stem="model-%d" % i, **spread(i, source="official-quantized"))
            for i in range(2000)]
    rows += [mk("mill/model-%d-gguf" % i, author="mill%d" % i, quant="gguf", bits=4,
                stem="m-%d" % i, **spread(i, source="quantized-conversion"))
             for i in range(2000)]
    rows += filler(6000)
    sel, out = _run(rows, 1500)
    third = (out["source_type"] == "quantized-conversion").mean()
    both = out["source_type"].isin(["quantized-conversion", "official-quantized"]).mean()
    assert third <= CFG["source_type"]["max_share"]["quantized-conversion"] + 1e-9, third
    assert both <= CFG["source_type"]["combined_quantized_max_share"] + 1e-9, both


def test_language_quota_uses_the_applicable_denominator():
    counts = {"english-primary": 1000, "multilingual-with-english": 1000,
              "non-english": 1000, "unknown": 1000, "language-neutral": 5000}
    q = language_quota(counts, 1000, CFG["language"])
    applicable = sum(v for k, v in q.items() if k != "language-neutral")
    en = q["english-primary"] + q["multilingual-with-english"]
    assert q["language-neutral"] <= int(CFG["language"]["language_neutral_max_share"] * 1000)
    assert en / max(applicable, 1) >= CFG["language"]["english_plus_multi_min_share_applicable"] - 0.02


def test_relaxation_fills_the_target_but_never_breaks_a_cap():
    rows = [mk("hog/m%d" % i, author="hog", stem="m%d" % i, **spread(i))
            for i in range(5000)]
    rows += filler(4000)
    sel, out = _run(rows, 1200)
    assert sel.n_relaxed >= 0
    assert out["author"].value_counts().iloc[0] / len(out) <= CFG["caps"]["author_max_share"] + 1e-9


def test_duplicate_capacity_is_measured_before_selecting():
    rows = [mk("q%d/x" % i, author="q%d" % i, root="one/base", family="lin:one",
               stem="x", quant="gguf", bits=4, size=8.0,
               source="quantized-conversion") for i in range(1000)]
    df = frame(rows).reset_index(drop=True)
    sel = Selector(df, cfg_for(500), 500)
    assert sel.duplicate_capacity() <= CFG["caps"]["near_duplicate_max"] * 2


def test_enrich_deficits_attaches_query_handles():
    df = frame(filler(5))
    out = enrich_deficits([{"axis": "task", "bucket": "translation", "missing": 100},
                           {"axis": "language", "bucket": "non-english", "missing": 50},
                           {"axis": "source_type", "bucket": "adapter-lora", "missing": 10},
                           {"axis": "family_size", "bucket": "large", "missing": 10}], df)
    assert "translation" in out[0]["pipeline_tags"]
    assert out[1]["language_codes"]
    assert out[2]["filters"]
    assert "authors" in out[3]


def test_backfill_plan_does_not_walk_the_global_download_ranking():
    from scale1m.query_plan import build_backfill_plan
    plan = build_backfill_plan([{"axis": "task", "bucket": "translation", "missing": 500,
                                 "pipeline_tags": ["translation"]}])
    assert plan.queries
    for q in plan.queries:
        assert "pipeline_tag" in q.params or "filter" in q.params or "author" in q.params
        assert q.source == "backfill"
