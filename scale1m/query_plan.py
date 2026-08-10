"""
query_plan.py -- the multi-source candidate-discovery plan for T2 v2.

Runbook: docs/1M/T2.md (v2), instructions Step 2.

WHY A PLAN INSTEAD OF ONE ORDERING
    A single `sort=downloads&direction=-1` stream defines the population as
    "whatever HF's 30-day download counter ranks highest". Measured on the v1
    150K head shard that produces: one publisher at 20.1%, 47% of records with
    no pipeline_tag, 65% with no parameter count, and a task histogram where
    text-generation alone is 24% while translation is 0.6%. That is a sampling
    artifact, not a property of the model ecosystem.

    So discovery is deliberately over-collected from many orthogonal orderings
    (popularity, appreciation, recency, freshness, trend, task, language,
    ecosystem) and the FINAL population is chosen later, under quotas, by
    select_balanced_halo.py. Discovery is allowed to be biased; selection is
    not.

WHAT THE API ACTUALLY SUPPORTS (probed 2026-08-03, recorded in T2.md)
    sort   : downloads | likes | createdAt | lastModified | trendingScore
             (downloadsAllTime is NOT a valid sort key)
    filter : repeated -> AND. Matches any HF tag: language codes, library
             names, architecture names, licence, task.
    pipeline_tag=<task>, author=<org>, search=<str>, minNumParameters=<int>
    library=<x> is accepted but silently IGNORED -- use filter=<x>.
"""

from dataclasses import dataclass, field

# --- the task axis --------------------------------------------------------
# All 56 pipeline tags observed on HF. Discovery queries every one of them so
# that a rare task is limited by real supply, not by where it happens to fall
# in a global download ranking.
PIPELINE_TAGS = (
    "text-generation", "text-classification", "token-classification",
    "question-answering", "table-question-answering", "summarization",
    "translation", "fill-mask", "sentence-similarity", "feature-extraction",
    "text-ranking", "text-retrieval", "zero-shot-classification",
    "multiple-choice", "automatic-speech-recognition", "audio-classification",
    "audio-to-audio", "text-to-speech", "text-to-audio", "voice-activity-detection",
    "audio-text-to-text", "image-classification", "zero-shot-image-classification",
    "object-detection", "zero-shot-object-detection", "image-segmentation",
    "mask-generation", "keypoint-detection", "depth-estimation",
    "image-feature-extraction", "text-to-image", "image-to-image",
    "text-to-video", "image-to-video", "video-to-video", "image-text-to-image",
    "image-text-to-video", "unconditional-image-generation", "text-to-3d",
    "image-to-3d", "image-text-to-text", "image-to-text", "visual-question-answering",
    "video-text-to-text", "video-classification", "visual-document-retrieval",
    "document-question-answering", "any-to-any", "tabular-classification",
    "tabular-regression", "time-series-forecasting", "reinforcement-learning",
    "robotics", "graph-ml", "other",
)

# --- the language axis ----------------------------------------------------
# English first (it is the target majority), then the languages with real HF
# supply. Queried explicitly so the language quota is filled from models that
# actually declare a language, rather than from name guessing.
LANGUAGE_FILTERS = (
    "en", "multilingual", "zh", "es", "fr", "de", "ja", "ru", "ar", "pt",
    "ko", "it", "hi", "nl", "tr", "pl", "vi", "id", "fa", "th", "sv", "cs",
    "uk", "he", "ro", "el", "da", "fi", "no", "hu", "bn", "ta", "ur", "sw",
)

# --- the ecosystem axis ---------------------------------------------------
# Library / architecture tags. These reach populations that popularity
# ordering never surfaces (timm's 1k vision backbones, espnet's speech zoo).
ECOSYSTEM_FILTERS = (
    "transformers", "sentence-transformers", "diffusers", "timm", "peft",
    "espnet", "nemo", "open_clip", "ultralytics", "stable-baselines3",
    "ml-agents", "sample-factory", "keras", "tf-keras", "spacy", "flair",
    "fastai", "speechbrain", "gliner", "setfit", "adapter-transformers",
    "paddlepaddle", "openvino", "onnx", "coreml", "tensorboard",
    "bertopic", "pythae", "unity-sentis", "mlx",
)

# --- source types deliberately hunted -------------------------------------
# Step 10 wants a source-type mix. Adapters and base models are structurally
# rare in a download ranking, so they get their own discovery queries.
STRUCTURAL_FILTERS = ("lora", "adapter", "base_model:finetune", "merge",
                      "mergekit", "trl", "sft", "dpo", "orpo", "grpo")


@dataclass
class Query:
    """One discovery stream. `qid` is the stable identity used for resume."""
    qid: str
    source: str                       # global | task | language | ecosystem | structural | backfill
    params: dict
    cap: int
    note: str = ""
    membership_only: bool = False     # True -> record hits but do not add new records

    def url_params(self):
        p = {"limit": "1000"}
        p.update({k: str(v) for k, v in self.params.items()})
        return p


@dataclass
class Plan:
    name: str
    queries: list = field(default_factory=list)

    def total_cap(self):
        return sum(q.cap for q in self.queries)


def build_deep_plan(scale=1.0, tasks=PIPELINE_TAGS, languages=LANGUAGE_FILTERS,
                    suffix=":deep"):
    """Second pass: go DEEPER on the axes that still had supply, not wider.

    Why this exists (measured, not anticipated): after the `full` plan the pool
    held 359,388 unique candidates, but the near-duplicate caps allow at most
    63,069 of them to be selected -- below the 69,817 HALO target. Capacity,
    not raw pool size, is the binding quantity, and capacity grows with the
    number of DISTINCT (root, task, scale, language, format) combinations. The
    global orderings are already exhausted at their head; per-task and
    per-language streams still had supply, so those are the ones deepened.

    New `qid`s (suffixed) so this appends to the same candidate directory
    instead of being skipped as already-done.
    """
    def c(n):
        return max(1000, int(n * scale))

    qs = []
    for t in tasks:
        for sort_key, cap in (("downloads", 12_000), ("likes", 6_000),
                              ("createdAt", 6_000)):
            qs.append(Query(qid="task:%s:%s%s" % (t, sort_key, suffix), source="task-deep",
                            params={"pipeline_tag": t, "sort": sort_key, "direction": "-1"},
                            cap=c(cap), note="deep task supply"))
    for lg in languages:
        for sort_key, cap in (("downloads", 6_000), ("createdAt", 4_000)):
            qs.append(Query(qid="lang:%s:%s%s" % (lg, sort_key, suffix), source="language-deep",
                            params={"filter": lg, "sort": sort_key, "direction": "-1"},
                            cap=c(cap), note="deep language supply"))
    return Plan(name="deep", queries=qs)


def build_plan(name="full", scale=1.0, tasks=PIPELINE_TAGS,
               languages=LANGUAGE_FILTERS, ecosystems=ECOSYSTEM_FILTERS,
               structural=STRUCTURAL_FILTERS):
    """Assemble the discovery plan. `scale` shrinks every cap for the pilot.

    Cap sizing rationale (Step 2 asks for a candidate pool "substantially
    larger than the final sample", 400K-800K unique for a 69,817 target):
      global streams        ~300K fetched -- five orthogonal orderings
      per-task streams      ~500K fetched -- the widest axis, and the one the
                                             v1 crawl was worst at
      per-language streams  ~100K fetched
      ecosystem/structural  ~120K fetched
    Overlap between streams is large and expected; unique yield is measured
    and reported, not assumed.
    """
    # Floor scales with `scale` too, otherwise a 0.08 pilot degenerates into
    # "every one of the 189 queries fetches the floor" and costs as much as
    # the full run.
    floor = max(200, int(1000 * scale))

    def c(n):
        return max(floor, int(n * scale))

    qs = []

    # -- global orderings ------------------------------------------------
    for sort_key, cap, note in (
        ("downloads", 120_000, "30-day popularity (the v1 population)"),
        ("likes", 60_000, "appreciation, decorrelated from download bots"),
        ("createdAt", 60_000, "recency -- the emerging stratum"),
        ("lastModified", 40_000, "freshness / actively maintained"),
        ("trendingScore", 20_000, "HF trending signal"),
    ):
        qs.append(Query(qid="global:%s" % sort_key, source="global",
                        params={"sort": sort_key, "direction": "-1"},
                        cap=c(cap), note=note))

    # -- per-task: two orderings each ------------------------------------
    for t in tasks:
        qs.append(Query(qid="task:%s:downloads" % t, source="task",
                        params={"pipeline_tag": t, "sort": "downloads", "direction": "-1"},
                        cap=c(6_000), note="task supply, popularity order"))
        qs.append(Query(qid="task:%s:likes" % t, source="task",
                        params={"pipeline_tag": t, "sort": "likes", "direction": "-1"},
                        cap=c(3_000), note="task supply, appreciation order"))

    # -- per-language ----------------------------------------------------
    for lg in languages:
        qs.append(Query(qid="lang:%s:downloads" % lg, source="language",
                        params={"filter": lg, "sort": "downloads", "direction": "-1"},
                        cap=c(3_000), note="language-declared supply"))

    # -- ecosystem / structural ------------------------------------------
    for f in ecosystems:
        qs.append(Query(qid="eco:%s" % f, source="ecosystem",
                        params={"filter": f, "sort": "downloads", "direction": "-1"},
                        cap=c(3_000), note="library/architecture ecosystem"))
    for f in structural:
        qs.append(Query(qid="struct:%s" % f, source="structural",
                        params={"filter": f, "sort": "downloads", "direction": "-1"},
                        cap=c(3_000), note="source-type structure"))

    return Plan(name=name, queries=qs)


def build_backfill_plan(deficits, scale=1.0):
    """Step 14: targeted queries for the regions the first selection missed.

    `deficits` is the list emitted by select_balanced_halo.py:
        [{"axis": "task", "bucket": "translation", "missing": 2000}, ...]
    Deficits are NOT filled by walking further down the global download
    ranking -- that is what produced the v1 distribution in the first place.
    """
    qs = []
    for d in deficits:
        axis, bucket, missing = d["axis"], d["bucket"], d["missing"]
        cap = max(2_000, int(missing * 4 * scale))
        if axis == "task":
            for pt in d.get("pipeline_tags") or []:
                for sort_key in ("downloads", "likes", "createdAt"):
                    qs.append(Query(
                        qid="backfill:task:%s:%s:%s" % (bucket, pt, sort_key),
                        source="backfill",
                        params={"pipeline_tag": pt, "sort": sort_key, "direction": "-1"},
                        cap=cap, note="deficit %d in supertask %s" % (missing, bucket)))
        elif axis == "language":
            for lg in d.get("language_codes") or []:
                qs.append(Query(
                    qid="backfill:lang:%s:%s" % (bucket, lg), source="backfill",
                    params={"filter": lg, "sort": "downloads", "direction": "-1"},
                    cap=cap, note="deficit %d in language bucket %s" % (missing, bucket)))
        elif axis == "source_type":
            for f in d.get("filters") or []:
                qs.append(Query(
                    qid="backfill:src:%s:%s" % (bucket, f), source="backfill",
                    params={"filter": f, "sort": "downloads", "direction": "-1"},
                    cap=cap, note="deficit %d in source type %s" % (missing, bucket)))
        elif axis == "family_size":
            for a in d.get("authors") or []:
                qs.append(Query(
                    qid="backfill:fam:%s:%s" % (bucket, a), source="backfill",
                    params={"author": a, "sort": "downloads", "direction": "-1"},
                    cap=cap, note="deficit %d in family stratum %s" % (missing, bucket)))
    return Plan(name="backfill", queries=qs)
