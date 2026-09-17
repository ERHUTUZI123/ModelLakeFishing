"""Tests for the T2 v2 annotation vocabulary (scale1m/taxonomy.py).

These are not happy-path tests. Each one pins a rule that, if it silently
regressed, would produce a population that still *looks* balanced in the
report: a name heuristic overwriting authoritative metadata, a vision model
labelled "unknown language", a quantization mirror classified as an original
base model, or two unrelated repos merged because their names share a token.

Run (from ModelLakeFishing/):
    .\\.venv\\Scripts\\python.exe -m pytest scale1m/tests -q
"""

import pytest

from scale1m import taxonomy as TX


# --- supertask -------------------------------------------------------------

def test_pipeline_tag_beats_every_weaker_signal():
    rec = {"id": "org/x", "pipeline_tag": "translation",
           "config": {"model_type": "llama"}, "library_name": "diffusers",
           "tags": ["stable-diffusion"]}
    task, src, conf = TX.supertask_of(rec)
    assert (task, src, conf) == ("translation", "pipeline_tag", 1.0)


def test_supertask_precedence_chain_degrades_in_order():
    base = {"id": "org/x"}
    assert TX.supertask_of({**base, "config": {"model_type": "whisper"},
                            "library_name": "diffusers"})[:2] == \
        ("speech-recognition", "config.model_type")
    assert TX.supertask_of({**base, "library_name": "timm"})[:2] == \
        ("image-classification", "library_name")
    assert TX.supertask_of({**base, "tags": ["mteb"]})[:2] == \
        ("embedding-retrieval", "tags")
    assert TX.supertask_of(base)[:2] == ("unknown", "none")


def test_every_known_pipeline_tag_maps_to_a_declared_supertask():
    for pt, st in TX.PIPELINE_TO_SUPERTASK.items():
        assert st in TX.SUPERTASKS, "%s -> %s is not a declared supertask" % (pt, st)


def test_unmapped_pipeline_tag_is_other_and_is_counted():
    rec = {"id": "org/x", "pipeline_tag": "brand-new-task-2027"}
    task, src, conf = TX.supertask_of(rec)
    assert task == "other" and src == "pipeline_tag_unmapped" and conf < 1.0
    assert TX.unmapped_pipeline_tags([rec])["brand-new-task-2027"] == 1


def test_code_only_overrides_text_generation_never_vision():
    assert TX.supertask_of({"id": "org/CodeLlama-7b", "pipeline_tag": "text-generation"})[0] == "code"
    # a vision model whose name happens to contain "code" must stay vision
    assert TX.supertask_of({"id": "org/qr-code-detector",
                            "pipeline_tag": "object-detection"})[0] == "detection-segmentation"


# --- language ---------------------------------------------------------------

def test_card_language_beats_tags():
    rec = {"id": "org/x", "cardData": {"language": ["fr"]}, "tags": ["en"]}
    langs, src = TX.languages_of(rec)
    assert langs == ["fr"] and src == "cardData.language"


def test_language_junk_tags_are_rejected():
    """The bare `^[a-z]{2,3}$` regex matches trl/sft/tf/jax/mms -- measured on
    the v1 150K shard. The ISO allowlist is what stops those."""
    rec = {"id": "org/x", "tags": ["trl", "sft", "tf", "jax", "mms", "mtp", "en"]}
    langs, _ = TX.languages_of(rec)
    assert langs == ["en"]


def test_vision_model_without_language_is_neutral_not_unknown():
    rec = {"id": "org/x", "pipeline_tag": "image-classification"}
    task = TX.supertask_of(rec)[0]
    bucket, langs, src, _ = TX.language_bucket_of(rec, task)
    assert bucket == "language-neutral" and src == "supertask_neutral"


def test_text_model_without_language_is_unknown():
    rec = {"id": "org/x", "pipeline_tag": "text-generation"}
    assert TX.language_bucket_of(rec, "text-generation-chat")[0] == "unknown"


@pytest.mark.parametrize("langs,expected", [
    (["en"], "english-primary"),
    (["en", "fr"], "multilingual-with-english"),
    (["multilingual"], "multilingual-with-english"),
    (["fr"], "non-english"),
    (["fr", "de"], "non-english"),
])
def test_language_buckets(langs, expected):
    rec = {"id": "org/x", "cardData": {"language": langs}, "pipeline_tag": "text-generation"}
    assert TX.language_bucket_of(rec, "text-generation-chat")[0] == expected


# --- quantization -----------------------------------------------------------

def test_quantization_authoritative_before_name():
    assert TX.quantization_of({"id": "org/plain", "gguf": True})[::2] == ("gguf", "gguf_block")
    assert TX.quantization_of({"id": "org/x", "library_name": "mlx"})[::2] == ("mlx", "library_name")
    assert TX.quantization_of({"id": "org/x", "tags": ["gptq"]})[::2] == ("gptq", "tags")
    m, bits, src = TX.quantization_of({"id": "org/Model-Q4_K_M-GGUF"})
    assert m == "gguf" and src == "name_heuristic"


def test_plain_model_is_not_a_quantization():
    assert TX.quantization_of({"id": "google-bert/bert-base-uncased"})[0] == "none"


# --- source type ------------------------------------------------------------

def test_relation_metadata_drives_source_type():
    assert TX.source_type_of("none", "none", None, [], ["a/base"], "adapter", False)[0] == "adapter-lora"
    assert TX.source_type_of("none", "none", None, [], ["a/base"], "merge", False)[0] == "community-finetune"


def test_official_derivative_is_same_author_finetune():
    assert TX.source_type_of("none", "none", None, [], ["qwen/base"], "finetune", True)[0] \
        == "official-derivative"
    assert TX.source_type_of("none", "none", None, [], ["qwen/base"], "finetune", False)[0] \
        == "community-finetune"


def test_quantization_publisher_is_never_original_base():
    st, src, _ = TX.source_type_of("gguf", "gguf_block", "gguf", [], [], None, False)
    assert st == "quantized-conversion" and src.startswith("quant:")


def test_no_parent_no_quant_is_original_base():
    assert TX.source_type_of("none", "none", "transformers", [], [], None, False)[0] == "original-base"


# --- metadata quality -------------------------------------------------------

def test_quality_rewards_structure_and_punishes_bare_repos():
    rich = {"id": "org/x", "pipeline_tag": "text-classification",
            "config": {"model_type": "bert"}, "safetensors": {"total": 1},
            "cardData": {"language": ["en"], "base_model": "a/b", "model-index": [1]}}
    bare = {"id": "org/y"}
    q_rich, _ = TX.metadata_quality_of(rich, "text-classification", "pipeline_tag")
    q_bare, flags = TX.metadata_quality_of(bare, "unknown", "none")
    assert q_rich > 0.8 and q_bare <= 0.0
    assert "NEG:bare" in flags


def test_disabled_repo_fails_quality():
    rec = {"id": "org/x", "pipeline_tag": "text-generation", "disabled": True,
           "config": {"model_type": "llama"}, "cardData": {"language": ["en"]}}
    q, flags = TX.metadata_quality_of(rec, "text-generation-chat", "pipeline_tag")
    assert q < TX.METADATA_QUALITY_THRESHOLD and "NEG:disabled" in flags


# --- stems / duplicates -----------------------------------------------------

def test_repo_stem_strips_conversion_tokens_only():
    assert TX.repo_stem("mradermacher/Meta-Llama-3-8B-Instruct-i1-GGUF") == \
        TX.repo_stem("TheBloke/Meta-Llama-3-8B-Instruct-AWQ")
    assert TX.repo_stem("org/llama-3-8b-instruct") != TX.repo_stem("org/llama-3-8b-base")


def test_near_duplicate_key_separates_meaningful_variation():
    common = dict(canonical_root="meta/llama-3-8b", supertask="text-generation-chat",
                  language_bucket="english-primary", source_type="quantized-conversion",
                  stem="llama-3-8b")
    k_q4 = TX.near_duplicate_key(quant_method="gguf", quant_bits=4, size_b=8.0, **common)
    k_awq = TX.near_duplicate_key(quant_method="awq", quant_bits=4, size_b=8.0, **common)
    k_70b = TX.near_duplicate_key(quant_method="gguf", quant_bits=4, size_b=70.0, **common)
    assert k_q4 != k_awq, "different quantization methods are different variation"
    assert k_q4 != k_70b, "different parameter scale is different variation"
    # same base, same method, same scale -> the same duplicate group
    assert k_q4 == TX.near_duplicate_key(quant_method="gguf", quant_bits=5,
                                         size_b=8.4, **common)


def test_param_scale_bucket_is_coarser_than_xm0_buckets():
    assert TX.param_scale_bucket(7.0) == TX.param_scale_bucket(8.0)
    assert TX.param_scale_bucket(None) == "unknown"
    assert TX.param_scale_bucket(200.0) == ">=90B"


def test_missing_size_is_unknown_not_the_top_bucket():
    """float('nan') survives every `<` comparison. Without an explicit NaN
    test it lands in ">=90B" and merges thousands of unrelated models into one
    duplicate group -- measured on the 419K pool: 16,267 in a single group."""
    assert TX.param_scale_bucket(float("nan")) == "unknown"


def test_missing_root_falls_back_to_stem_not_the_string_nan():
    a = TX.near_duplicate_key(float("nan"), "text-classification", 1.0,
                              "english-primary", "none", None, "original-base", "alpha")
    b = TX.near_duplicate_key(float("nan"), "text-classification", 1.0,
                              "english-primary", "none", None, "original-base", "beta")
    assert a != b and "nan" not in a.split("|")[0]


def test_exact_key_only_binds_on_quantizations():
    base = "root|task|7B|english-primary|none|original-base"
    k1 = TX.near_duplicate_key_exact(base, "none", None, "a/x")
    k2 = TX.near_duplicate_key_exact(base, "none", None, "a/y")
    assert k1 != k2, "the tighter cap must not apply to unquantized models"
    q1 = TX.near_duplicate_key_exact(base, "gguf", 4, "a/x")
    q2 = TX.near_duplicate_key_exact(base, "gguf", 4, "a/y")
    assert q1 == q2, "same base + method + bit level is one exact group"
    assert q1 != TX.near_duplicate_key_exact(base, "gguf", 8, "a/z")


# --- mirror score -----------------------------------------------------------

def test_mirror_score_is_generic_not_a_username_list():
    quant_mill = {"n_repos": 14061, "quant_frac": 0.98,
                  "distinct_upstream_authors": 4200, "original_frac": 0.01,
                  "stem_repeat_frac": 0.6}
    real_lab = {"n_repos": 120, "quant_frac": 0.05, "distinct_upstream_authors": 2,
                "original_frac": 0.8, "stem_repeat_frac": 0.1}
    s_mill, why = TX.mirror_score_of(quant_mill)
    s_lab, _ = TX.mirror_score_of(real_lab)
    assert s_mill >= TX.MIRROR_SCORE_THRESHOLD and s_lab < TX.MIRROR_SCORE_THRESHOLD
    assert why, "reasons must be auditable"
    # nothing in the module keys off a specific account name
    import inspect
    src = inspect.getsource(TX)
    for name in ("mradermacher", "RichardErkhov", "TheBloke", "bartowski"):
        assert name not in src


def test_small_publisher_never_scores_as_mirror():
    s, _ = TX.mirror_score_of({"n_repos": 3, "quant_frac": 1.0,
                               "distinct_upstream_authors": 3, "original_frac": 0.0,
                               "stem_repeat_frac": 0.0})
    assert s < TX.MIRROR_SCORE_THRESHOLD
