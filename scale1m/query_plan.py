from dataclasses import dataclass, field

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

LANGUAGE_FILTERS = (
    "en", "multilingual", "zh", "es", "fr", "de", "ja", "ru", "ar", "pt",
    "ko", "it", "hi", "nl", "tr", "pl", "vi", "id", "fa", "th", "sv", "cs",
    "uk", "he", "ro", "el", "da", "fi", "no", "hu", "bn", "ta", "ur", "sw",
)

ECOSYSTEM_FILTERS = (
    "transformers", "sentence-transformers", "diffusers", "timm", "peft",
    "espnet", "nemo", "open_clip", "ultralytics", "stable-baselines3",
    "ml-agents", "sample-factory", "keras", "tf-keras", "spacy", "flair",
    "fastai", "speechbrain", "gliner", "setfit", "adapter-transformers",
    "paddlepaddle", "openvino", "onnx", "coreml", "tensorboard",
    "bertopic", "pythae", "unity-sentis", "mlx",
)

STRUCTURAL_FILTERS = ("lora", "adapter", "base_model:finetune", "merge",
                      "mergekit", "trl", "sft", "dpo", "orpo", "grpo")


@dataclass
class Query:
    qid: str
    source: str
    params: dict
    cap: int
    note: str = ""
    membership_only: bool = False

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
    floor = max(200, int(1000 * scale))

    def c(n):
        return max(floor, int(n * scale))

    qs = []

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

    for t in tasks:
        qs.append(Query(qid="task:%s:downloads" % t, source="task",
                        params={"pipeline_tag": t, "sort": "downloads", "direction": "-1"},
                        cap=c(6_000), note="task supply, popularity order"))
        qs.append(Query(qid="task:%s:likes" % t, source="task",
                        params={"pipeline_tag": t, "sort": "likes", "direction": "-1"},
                        cap=c(3_000), note="task supply, appreciation order"))

    for lg in languages:
        qs.append(Query(qid="lang:%s:downloads" % lg, source="language",
                        params={"filter": lg, "sort": "downloads", "direction": "-1"},
                        cap=c(3_000), note="language-declared supply"))

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
