"""
taxonomy.py -- the shared, versioned vocabulary every T2-v2 stage annotates
against: supertask, language bucket, source type, quantization, family,
near-duplicate key, metadata quality.

Runbook: docs/1M/T2.md (v2 redesign), docs/1M/100kplan.md 5-6.

DESIGN RULE (Step 3 of the T2 v2 instructions, and the reason this file exists
at all): **authoritative metadata first, heuristics second, and every value
carries the name of the signal that produced it.** Every `*_of()` function
returns `(value, source, confidence)`, never a bare value. A downstream report
that cannot say WHERE a label came from cannot be audited, and a heuristic that
cannot be told apart from metadata will silently be treated as ground truth.

REUSED, NOT REDEFINED
    Architecture family (`arch_family`) is the EXISTING xm0 notion and comes
    from `dataset_embed.utils.fetch_metadata._infer_one_family` -- the same
    rules that produced CORE's `family_id`. It is NOT the same thing as this
    module's lineage `family_id` / `canonical_root`, which is a base_model
    closure. Two different questions ("what architecture is this?" vs "what
    is this a derivative of?"), so two different fields, both kept.
"""

import re
from collections import Counter

# --- version -------------------------------------------------------------
# Bump when any mapping below changes; annotated artifacts record it so a
# selection can never be silently compared against a different taxonomy.
TAXONOMY_VERSION = "t2v2.1"


# =========================================================================
# supertask
# =========================================================================

SUPERTASKS = (
    "text-generation-chat",
    "embedding-retrieval",
    "text-classification",
    "token-classification",
    "question-answering",
    "summarization",
    "translation",
    "code",
    "speech-recognition",
    "audio-generation",
    "audio-classification",
    "image-classification",
    "detection-segmentation",
    "image-generation",
    "vision-language",
    "tabular-timeseries-rl",
    "other",
    "unknown",
)

# HF pipeline_tag is the authoritative signal when present. Every one of the
# 56 pipeline tags observed in the 150K head shard is mapped here; an
# unmapped tag falls through to "other" and is COUNTED (see unmapped_tags()).
PIPELINE_TO_SUPERTASK = {
    "text-generation": "text-generation-chat",
    "text2text-generation": "text-generation-chat",
    "fill-mask": "text-classification",
    "sentence-similarity": "embedding-retrieval",
    "feature-extraction": "embedding-retrieval",
    "text-ranking": "embedding-retrieval",
    "text-retrieval": "embedding-retrieval",
    "image-feature-extraction": "embedding-retrieval",
    "visual-document-retrieval": "embedding-retrieval",
    "text-classification": "text-classification",
    "zero-shot-classification": "text-classification",
    "multiple-choice": "text-classification",
    "token-classification": "token-classification",
    "question-answering": "question-answering",
    "table-question-answering": "question-answering",
    "document-question-answering": "vision-language",
    "summarization": "summarization",
    "translation": "translation",
    "automatic-speech-recognition": "speech-recognition",
    "voice-activity-detection": "audio-classification",
    "audio-classification": "audio-classification",
    "text-to-speech": "audio-generation",
    "text-to-audio": "audio-generation",
    "audio-to-audio": "audio-generation",
    "audio-text-to-text": "vision-language",
    "image-classification": "image-classification",
    "zero-shot-image-classification": "image-classification",
    "video-classification": "image-classification",
    "object-detection": "detection-segmentation",
    "zero-shot-object-detection": "detection-segmentation",
    "image-segmentation": "detection-segmentation",
    "mask-generation": "detection-segmentation",
    "keypoint-detection": "detection-segmentation",
    "depth-estimation": "detection-segmentation",
    "text-to-image": "image-generation",
    "image-to-image": "image-generation",
    "text-to-video": "image-generation",
    "image-to-video": "image-generation",
    "video-to-video": "image-generation",
    "image-text-to-image": "image-generation",
    "image-text-to-video": "image-generation",
    "unconditional-image-generation": "image-generation",
    "text-to-3d": "image-generation",
    "image-to-3d": "image-generation",
    "image-text-to-text": "vision-language",
    "image-to-text": "vision-language",
    "visual-question-answering": "vision-language",
    "video-text-to-text": "vision-language",
    "any-to-any": "vision-language",
    "tabular-classification": "tabular-timeseries-rl",
    "tabular-regression": "tabular-timeseries-rl",
    "time-series-forecasting": "tabular-timeseries-rl",
    "reinforcement-learning": "tabular-timeseries-rl",
    "robotics": "tabular-timeseries-rl",
    "graph-ml": "other",
    "other": "other",
}

# Library is a weaker but still structured signal, used only when pipeline_tag
# is absent. `transformers` is deliberately absent -- it says nothing.
LIBRARY_TO_SUPERTASK = {
    "sentence-transformers": "embedding-retrieval",
    "pylate": "embedding-retrieval",
    "diffusers": "image-generation",
    "diffusion-single-file": "image-generation",
    "timm": "image-classification",
    "open_clip": "image-classification",
    "ultralytics": "detection-segmentation",
    "paddleocr": "vision-language",
    "ml-agents": "tabular-timeseries-rl",
    "lerobot": "tabular-timeseries-rl",
    "sample-factory": "tabular-timeseries-rl",
    "stable-baselines3": "tabular-timeseries-rl",
    "espnet": "speech-recognition",
    "nemo": "speech-recognition",
    "transcribe.cpp": "speech-recognition",
    "mlx-audio": "audio-generation",
    "gliner": "token-classification",
    "llama.cpp": "text-generation-chat",
    "mlc-llm": "text-generation-chat",
    "vllm": "text-generation-chat",
    "mesh-llm": "text-generation-chat",
    "litert-lm": "text-generation-chat",
}

# config.model_type is authoritative structured metadata (it comes from
# config.json, not from a name), so it outranks tag guessing.
MODEL_TYPE_TO_SUPERTASK = {
    "llama": "text-generation-chat", "qwen2": "text-generation-chat",
    "qwen3": "text-generation-chat", "mistral": "text-generation-chat",
    "gemma": "text-generation-chat", "gemma2": "text-generation-chat",
    "gemma3": "text-generation-chat", "phi": "text-generation-chat",
    "phi3": "text-generation-chat", "gpt2": "text-generation-chat",
    "gpt_neox": "text-generation-chat", "falcon": "text-generation-chat",
    "mpt": "text-generation-chat", "bloom": "text-generation-chat",
    "opt": "text-generation-chat", "mixtral": "text-generation-chat",
    "bert": "text-classification", "roberta": "text-classification",
    "distilbert": "text-classification", "deberta": "text-classification",
    "deberta-v2": "text-classification", "electra": "text-classification",
    "albert": "text-classification", "xlm-roberta": "text-classification",
    "modernbert": "text-classification",
    "t5": "translation", "mt5": "translation", "marian": "translation",
    "m2m_100": "translation", "nllb": "translation", "bart": "summarization",
    "whisper": "speech-recognition", "wav2vec2": "speech-recognition",
    "hubert": "speech-recognition",
    "vit": "image-classification", "swin": "image-classification",
    "convnext": "image-classification", "resnet": "image-classification",
    "clip": "image-classification", "siglip": "image-classification",
    "detr": "detection-segmentation", "yolos": "detection-segmentation",
    "segformer": "detection-segmentation", "sam": "detection-segmentation",
    "llava": "vision-language", "qwen2_vl": "vision-language",
    "idefics": "vision-language", "blip": "vision-language",
}

# Coarse tag hints, last structured resort before the name heuristic.
TAG_TO_SUPERTASK = {
    "text-generation-inference": "text-generation-chat",
    "conversational": "text-generation-chat",
    "stable-diffusion": "image-generation",
    "lora": None,           # says source type, not task -- deliberately not a task hint
    "colpali": "embedding-retrieval",
    "mteb": "embedding-retrieval",
    "sentence-transformers": "embedding-retrieval",
    "espnet": "speech-recognition",
    "asr": "speech-recognition",
    "tts": "audio-generation",
}

CODE_TOKENS = ("code", "coder", "codegen", "starcoder", "codellama", "codeqwen",
               "deepseek-coder", "santacoder", "replit-code", "codegemma",
               "codestral", "sql", "text-to-sql")


def _tokens(model_id: str):
    return set(re.split(r"[\/_\-.\s]+", (model_id or "").lower()))


def supertask_of(rec: dict):
    """-> (supertask, source, confidence in [0,1]).

    Precedence: pipeline_tag > config.model_type > library_name > tags > name.
    `code` is a cross-cut: it overrides a text-generation label only, because a
    code model IS a text generator and the distinction is what we want to
    balance on. It never overrides a vision/audio label.
    """
    pt = str(rec.get("pipeline_tag") or "").strip().lower()
    task, source, conf = None, None, 0.0
    if pt:
        task = PIPELINE_TO_SUPERTASK.get(pt)
        if task:
            task, source, conf = task, "pipeline_tag", 1.0
        else:
            task, source, conf = "other", "pipeline_tag_unmapped", 0.5

    if task is None:
        mt = str((rec.get("config") or {}).get("model_type") or "").strip().lower()
        if mt in MODEL_TYPE_TO_SUPERTASK:
            task, source, conf = MODEL_TYPE_TO_SUPERTASK[mt], "config.model_type", 0.85

    if task is None:
        lib = str(rec.get("library_name") or "").strip().lower()
        if lib in LIBRARY_TO_SUPERTASK:
            task, source, conf = LIBRARY_TO_SUPERTASK[lib], "library_name", 0.7

    if task is None:
        for t in rec.get("tags") or []:
            hint = TAG_TO_SUPERTASK.get(str(t).strip().lower())
            if hint:
                task, source, conf = hint, "tags", 0.55
                break

    if task is None:
        task, source, conf = "unknown", "none", 0.0

    if task == "text-generation-chat":
        toks = _tokens(rec.get("id", ""))
        if toks & set(CODE_TOKENS):
            return "code", (source or "none") + "+name_heuristic", min(conf, 0.5)
    return task, source, conf


# =========================================================================
# language
# =========================================================================

LANGUAGE_BUCKETS = ("english-primary", "multilingual-with-english",
                    "non-english", "language-neutral", "unknown")

# The ">= 75% English" rule is a statement about the LANGUAGE MIX, so its
# denominator is the models a language can be attributed to. A ViT or a
# time-series forecaster has no language; counting it against the English
# share does not measure anything, and it puts the rule in direct conflict
# with task balance (the language-neutral supertasks alone want >25% of the
# population under the task quota). Measured on the 537K pool, that conflict
# is what capped the balanced HALO at ~60K.
LANGUAGE_APPLICABLE = frozenset({"english-primary", "multilingual-with-english",
                                 "non-english", "unknown"})
ENGLISH_SIDE = frozenset({"english-primary", "multilingual-with-english"})

# Supertasks whose models are genuinely language-neutral. Step 7 is explicit:
# do NOT label these "unknown" just because they carry no language tag.
LANGUAGE_NEUTRAL_TASKS = frozenset({
    "image-classification", "detection-segmentation", "image-generation",
    "audio-classification", "tabular-timeseries-rl",
})

# ISO-639-1 (all) plus the ISO-639-3 codes HF actually uses heavily. Kept as an
# explicit allowlist because a bare `^[a-z]{2,3}$` regex over `tags` matches
# junk: the 150K head shard produced "trl", "sft", "tf", "jax", "mms", "mtp".
ISO_639_1 = frozenset("""
aa ab ae af ak am an ar as av ay az ba be bg bh bi bm bn bo br bs ca ce ch co
cr cs cu cv cy da de dv dz ee el en eo es et eu fa ff fi fj fo fr fy ga gd gl
gn gu gv ha he hi ho hr ht hu hy hz ia id ie ig ii ik io is it iu ja jv ka kg
ki kj kk kl km kn ko kr ks ku kv kw ky la lb lg li ln lo lt lu lv mg mh mi mk
ml mn mr ms mt my na nb nd ne ng nl nn no nr nv ny oc oj om or os pa pi pl ps
pt qu rm rn ro ru rw sa sc sd se sg si sk sl sm sn so sq sr ss st su sv sw ta
te tg th ti tk tl tn to tr ts tt tw ty ug uk ur uz ve vi vo wa wo xh yi yo za
zh zu
""".split())

ISO_639_3_COMMON = frozenset("""
ace ada afr amh ara asm aym aze bak bam bel ben bod bos bul cat ceb ces cmn cym
dan deu ell eng epo est eus fao fas fin fra ful gla gle glg grn guj hat hau heb
hin hrv hun hye ibo ind isl ita jav jpn kan kat kaz khm kin kir kor kur lao lat
lav lit ltz lug mal mar mkd mlg mlt mon mri msa mya nld nno nob npi nya ori orm
pan pol por pus que ron run rus sin slk slv sna snd som spa sqi srp sun swa swe
tam tat tel tgk tgl tha tir ton tsn tuk tur ukr urd uzb vie wol xho yor yue zho
zul nan wuu hak gan hsn
""".split())

LANG_CODES = ISO_639_1 | ISO_639_3_COMMON
ENGLISH_CODES = frozenset({"en", "eng"})

# `multilingual` is a real HF tag, not a language code.
MULTILINGUAL_TAGS = frozenset({"multilingual", "multi"})

_LANG_TAG_RE = re.compile(r"^([a-z]{2,3})(?:[-_][A-Za-z]{2,4})?$")


def languages_of(rec: dict):
    """-> (sorted language codes, source).

    Authoritative order: cardData.language > bare language tags > `lang:xx`
    tags. Repository-name inference is NOT used at all here -- Step 7 allows it
    only as a labeled last resort, and in practice it produces more false
    positives (every "-de-" is not German) than it resolves.
    """
    out, src = set(), None
    card = (rec.get("cardData") or {}).get("language")
    if isinstance(card, str):
        card = [card]
    if isinstance(card, list):
        for v in card:
            code = str(v).strip().lower()
            m = _LANG_TAG_RE.match(code)
            if m and m.group(1) in LANG_CODES:
                out.add(m.group(1))
            elif code in MULTILINGUAL_TAGS:
                out.add("multilingual")
        if out:
            src = "cardData.language"

    if not out:
        for t in rec.get("tags") or []:
            t = str(t).strip()
            low = t.lower()
            if low.startswith("lang:"):
                low = low[5:]
            m = _LANG_TAG_RE.match(low)
            if m and m.group(1) in LANG_CODES:
                out.add(m.group(1))
            elif low in MULTILINGUAL_TAGS:
                out.add("multilingual")
        if out:
            src = "tags"

    return sorted(out), (src or "none")


def language_bucket_of(rec: dict, supertask: str):
    """-> (bucket, languages, source, confidence)."""
    langs, src = languages_of(rec)
    has_multi_tag = "multilingual" in langs
    codes = [l for l in langs if l != "multilingual"]
    has_en = bool(set(codes) & ENGLISH_CODES)

    if codes or has_multi_tag:
        if has_multi_tag or len(codes) > 1:
            bucket = "multilingual-with-english" if (has_en or has_multi_tag) else "non-english"
        elif has_en:
            bucket = "english-primary"
        else:
            bucket = "non-english"
        return bucket, langs, src, 0.9

    if supertask in LANGUAGE_NEUTRAL_TASKS:
        # Genuinely language-neutral, not missing metadata (Step 7, explicit).
        return "language-neutral", [], "supertask_neutral", 0.7
    return "unknown", [], "none", 0.0


# =========================================================================
# quantization / source type
# =========================================================================

QUANT_METHODS = {
    "gguf": ("gguf", ("gguf", "ggml", "q4_k_m", "q5_k_m", "q8_0", "imatrix", "i1")),
    "gptq": ("gptq", ("gptq",)),
    "awq": ("awq", ("awq",)),
    "bnb": ("bitsandbytes", ("bnb", "nf4", "4bit", "8bit", "bitsandbytes")),
    "mlx": ("mlx", ("mlx",)),
    "onnx": ("onnx", ("onnx", "onnxruntime")),
    "openvino": ("openvino", ("openvino", "ov")),
    "fp8": ("fp8", ("fp8", "w8a8", "w4a16", "int8", "int4")),
    "exl2": ("exl2", ("exl2", "exllama")),
    "torchao": ("torchao", ("torchao",)),
    "tensorrt": ("tensorrt", ("tensorrt", "trt")),
}
_BITS_RE = re.compile(r"(?:^|[^a-z0-9])(?:q|int|w|fp)(\d{1,2})(?:[^0-9]|$)", re.I)


def quantization_of(rec: dict):
    """-> (method, bits, source). ('none', None, ...) when not a quantization.

    Authoritative signals first: the `gguf` expand block and library_name are
    repository facts; tags are curated by the author; the repo name is the
    labeled last resort and marks itself as such in `source`.
    """
    if rec.get("gguf"):
        return "gguf", _bits(rec.get("id", "")), "gguf_block"
    lib = str(rec.get("library_name") or "").strip().lower()
    if lib in ("gguf", "llama.cpp", "ggml"):
        return "gguf", _bits(rec.get("id", "")), "library_name"
    if lib == "mlx":
        return "mlx", _bits(rec.get("id", "")), "library_name"

    tags = {str(t).strip().lower() for t in (rec.get("tags") or [])}
    for _, (method, pats) in QUANT_METHODS.items():
        if tags & set(pats):
            return method, _bits(rec.get("id", "")), "tags"

    name = str(rec.get("id") or "").lower()
    for _, (method, pats) in QUANT_METHODS.items():
        for p in pats:
            if p in name:
                return method, _bits(name), "name_heuristic"
    return "none", None, "none"


def _bits(text: str):
    m = _BITS_RE.search(text or "")
    if not m:
        return None
    b = int(m.group(1))
    return b if 1 <= b <= 32 else None


SOURCE_TYPES = ("original-base", "official-derivative", "community-finetune",
                "adapter-lora", "official-quantized", "quantized-conversion",
                "unknown-derivative")

# `official-quantized` and `quantized-conversion` are BOTH quantizations, and
# separating them is not cosmetic. `Qwen/Qwen3-8B-GGUF` is a first-party
# release of a model its publisher trained; `somebody/Qwen3-8B-i1-GGUF` is a
# third-party repackaging of someone else's weights. Lumping them together
# either throws away legitimate first-party releases or lets a conversion mill
# ride in under the same quota. The distinguishing fact is structural and
# available: does the quantization's author equal the base model's author?
THIRD_PARTY_QUANT = "quantized-conversion"
OFFICIAL_QUANT = "official-quantized"

# HF `baseModels.relation` values, mapped to our source types. This is
# AUTHORITATIVE metadata -- the whole point of pulling the `baseModels` expand.
RELATION_TO_SOURCE = {
    "quantized": THIRD_PARTY_QUANT,
    "adapter": "adapter-lora",
    "finetune": "community-finetune",
    "merge": "community-finetune",
}


def source_type_of(quant_method, quant_src, library_name, tags,
                   base_ids, relation, same_author_as_base):
    """-> (source_type, source, confidence).

    Takes the already-derived quantization verdict rather than a raw record:
    the caller has computed it once, and re-deriving it here is how the two
    could silently disagree.

    `official-*` is defined structurally: a derivative whose author is the same
    as its base model's author. That is a fact, not a judgement call.
    """
    if quant_method != "none" and quant_src != "name_heuristic":
        if same_author_as_base:
            return OFFICIAL_QUANT, "quant:" + quant_src + "+author", 0.9
        return THIRD_PARTY_QUANT, "quant:" + quant_src, 0.9

    lib = str(library_name or "").strip().lower()
    tags = {str(t).strip().lower() for t in (tags or [])}
    if lib == "peft" or {"lora", "peft", "adapter"} & tags:
        return "adapter-lora", "library/tags", 0.85

    if relation:
        st = RELATION_TO_SOURCE.get(str(relation).strip().lower())
        if st == "community-finetune" and same_author_as_base:
            return "official-derivative", "baseModels.relation+author", 0.85
        if st == THIRD_PARTY_QUANT and same_author_as_base:
            return OFFICIAL_QUANT, "baseModels.relation+author", 0.9
        if st:
            return st, "baseModels.relation", 0.9

    if base_ids:
        if same_author_as_base:
            return "official-derivative", "base_model+author", 0.7
        return "community-finetune", "base_model", 0.7

    if quant_method != "none":
        if same_author_as_base:
            return OFFICIAL_QUANT, "quant:name_heuristic+author", 0.4
        return THIRD_PARTY_QUANT, "quant:name_heuristic", 0.4
    return "original-base", "no_declared_parent", 0.5


# =========================================================================
# metadata quality
# =========================================================================

def metadata_quality_of(rec: dict, supertask: str, task_source: str):
    """-> (score in [0,1], flags list).

    Positive: recognized task, structured architecture, model card, structured
    language, declared parent, safetensors/model-index. Negative: gated/
    disabled/private, no task AND no architecture AND no card.
    """
    flags, score = [], 0.0
    card = rec.get("cardData") or {}
    cfg = rec.get("config") or {}

    if supertask not in ("unknown", "other") and task_source == "pipeline_tag":
        score += 0.25
        flags.append("task:authoritative")
    elif supertask not in ("unknown",):
        score += 0.12
        flags.append("task:inferred")

    if cfg.get("architectures") or cfg.get("model_type"):
        score += 0.20
        flags.append("arch")
    if card:
        score += 0.15
        flags.append("card")
    if card.get("language"):
        score += 0.10
        flags.append("lang")
    if card.get("base_model") or rec.get("baseModels"):
        score += 0.10
        flags.append("parent")
    if (rec.get("safetensors") or {}).get("total"):
        score += 0.12
        flags.append("safetensors")
    if card.get("model-index"):
        score += 0.08
        flags.append("model-index")

    if rec.get("disabled"):
        score -= 0.60
        flags.append("NEG:disabled")
    if rec.get("private"):
        score -= 0.60
        flags.append("NEG:private")
    if rec.get("gated"):
        score -= 0.10
        flags.append("NEG:gated")
    if not rec.get("pipeline_tag") and not cfg and not card:
        score -= 0.25
        flags.append("NEG:bare")

    return max(0.0, min(1.0, score)), flags


METADATA_QUALITY_THRESHOLD = 0.35


# =========================================================================
# near-duplicate key / repo stem
# =========================================================================

# Tokens stripped when reducing a repo name to its "stem": they describe the
# CONVERSION, not the model. Deliberately a superset of xm0's _SKIP_TOKENS on
# the quantization side, because that is what floods this lake.
_STEM_STRIP = frozenset("""
gguf ggml gptq awq bnb nf4 int4 int8 fp8 fp16 bf16 fp32 4bit 8bit 2bit 3bit
6bit q2 q3 q4 q5 q6 q8 k m s l xs xxs i1 imatrix mlx onnx openvino ov exl2
exl3 exllama trt tensorrt torchao quantized quant safetensors gptqmodel
w8a8 w4a16 w8a16 a16 hqq marlin autoround smashed pruned
""".split())
_SIZE_TOK_RE = re.compile(r"^\d+(\.\d+)?[bmk]$")
_VER_TOK_RE = re.compile(r"^v?\d+(\.\d+)*$")


def repo_stem(model_id: str) -> str:
    """Normalized repo name with conversion/quantization tokens removed."""
    repo = (model_id or "").split("/")[-1].lower()
    keep = []
    for tok in re.split(r"[-_.\s]+", repo):
        if not tok or tok in _STEM_STRIP or _SIZE_TOK_RE.match(tok):
            continue
        keep.append(tok)
    return "-".join(keep) or repo


def param_scale_bucket(size_b):
    """Coarse parameter band for duplicate grouping (NOT the xm0 size bucket).

    Deliberately coarser than xm0's half-decade buckets: two 7B and 8B
    conversions of the same base are near-duplicates for sampling purposes.

    The `!= size_b` test catches float('nan'), which is what a missing size
    looks like once the column has been through pandas. Without it every
    unknown-size model falls through every `<` comparison and is labelled
    ">=90B" -- measured: that silently merged 16,267 unrelated models into one
    duplicate group and capped the whole selection.
    """
    if size_b is None or size_b != size_b:
        return "unknown"
    for hi, name in ((0.1, "<0.1B"), (0.5, "0.1-0.5B"), (1.5, "0.5-1.5B"),
                     (4, "1.5-4B"), (9, "4-9B"), (16, "9-16B"), (40, "16-40B"),
                     (90, "40-90B")):
        if size_b < hi:
            return name
    return ">=90B"


def near_duplicate_key(canonical_root, supertask, size_b, language_bucket,
                       quant_method, quant_bits, source_type, stem):
    """Coarse group: 'the same thing, packaged again'.

    Two levels are used by the selector (Step 5):
      * this key                 -> max 3 representatives
      * this key + quant level   -> max 2 (same base, same method, same level)

    A model with no resolved lineage root falls back to its own repo stem, so
    unrelated singletons never share a group. `canonical_root or stem` is NOT
    enough: after a pandas round-trip a missing root is float('nan'), which is
    truthy, and every singleton then collapses onto the literal string "nan".
    """
    root = canonical_root if (isinstance(canonical_root, str) and canonical_root) else stem
    return "|".join([
        str(root), supertask, param_scale_bucket(size_b), language_bucket,
        quant_method, source_type,
    ])


def near_duplicate_key_exact(base_key, quant_method, quant_bits, model_id):
    """The tighter Step-5 rule: same base, same quantization METHOD, same bit
    level -> at most 1-2 representatives.

    It applies ONLY to quantizations. An unquantized model gets a key unique to
    itself, so the tighter cap cannot bind on it: appending a constant "na"
    instead would silently lower every unquantized model's cap from 3 to 2,
    which is not what Step 5 says and cost ~8,000 of measured capacity.
    """
    if quant_method == "none":
        return "%s|__unq__|%s" % (base_key, model_id)
    return "%s|%s|%s" % (base_key, quant_method, quant_bits)


# =========================================================================
# mirror / conversion publisher score
# =========================================================================

MIRROR_SCORE_THRESHOLD = 0.60


def mirror_score_of(author_stats: dict):
    """-> (score in [0,1], reasons).

    `author_stats` (built by annotate_candidates over the whole pool):
        n_repos, quant_frac, distinct_upstream_authors, original_frac,
        stem_repeat_frac
    Generic by construction -- no username is hard-coded anywhere (Step 9).
    """
    n = author_stats.get("n_repos", 0)
    quant_frac = author_stats.get("quant_frac", 0.0)
    upstream = author_stats.get("distinct_upstream_authors", 0)
    original_frac = author_stats.get("original_frac", 1.0)
    stem_repeat = author_stats.get("stem_repeat_frac", 0.0)

    score, why = 0.0, []
    if n >= 500:
        score += 0.30
        why.append("n_repos>=500")
    elif n >= 100:
        score += 0.18
        why.append("n_repos>=100")
    elif n >= 30:
        score += 0.08
        why.append("n_repos>=30")

    if quant_frac >= 0.8:
        score += 0.30
        why.append("quant_frac>=0.8")
    elif quant_frac >= 0.5:
        score += 0.18
        why.append("quant_frac>=0.5")

    if upstream >= 100:
        score += 0.20
        why.append("upstream_authors>=100")
    elif upstream >= 20:
        score += 0.10
        why.append("upstream_authors>=20")

    if original_frac <= 0.05:
        score += 0.15
        why.append("original_frac<=0.05")
    elif original_frac <= 0.2:
        score += 0.07
        why.append("original_frac<=0.2")

    if stem_repeat >= 0.5:
        score += 0.05
        why.append("stem_repeat>=0.5")

    return min(1.0, score), why


# =========================================================================
# popularity
# =========================================================================

POPULARITY_STRATA = ("head", "mid", "long-tail", "recent")

# Step 11's scoring function. Coefficients kept here (not scattered) and
# recorded into the run manifest.
POPULARITY_WEIGHTS = {"downloads": 0.55, "likes": 0.20, "recency": 0.15,
                      "metadata": 0.10}


def unmapped_pipeline_tags(records):
    """Audit helper: which pipeline tags fell through to 'other'."""
    c = Counter()
    for r in records:
        pt = (r.get("pipeline_tag") or "").strip().lower()
        if pt and pt not in PIPELINE_TO_SUPERTASK:
            c[pt] += 1
    return c
