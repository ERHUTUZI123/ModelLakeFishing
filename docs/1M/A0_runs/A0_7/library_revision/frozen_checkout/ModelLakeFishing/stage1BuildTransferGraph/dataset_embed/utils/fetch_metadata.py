import re
import time
import pandas as pd
from typing import List
from pathlib import Path
from huggingface_hub import HfApi, ModelCard

api = HfApi()


# lineage-fetch script: guarded so that importing this module stays
# side-effect-free (no network, no file writes) — run directly to refresh
# dataset_embed/data/lineage_records.csv
def _fetch_lineage_records():
    model_ids = \
    ["google/gemma-4-31B-it",
     "gghfez/gemma-4-31b-it-control-vectors",
     "nvidia/Gemma-4-31B-IT-NVFP4",
     "virtuous7373/Gemma-4-Harmonia-31B",
     "unsloth/gemma-4-31B-it-GGUF"]

    # model lineage records
    df = pd.DataFrame(columns=["model", "relation", "base_model"])

    for model_id in model_ids:
        child_model_id = model_id
        info = api.model_info(
            repo_id=model_id,
            expand=["baseModels"]
        )

        print(info.base_models)
        relation = info.base_models.get('relation')
        print(relation)
        models_list = info.base_models.get('models', [])
        if models_list:
            for model in models_list:
                parent_model_id = model.get('id')
                df.loc[len(df)] = [child_model_id, relation, parent_model_id]
        else:
            continue
    print(df)

    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir.parent / "data"
    output_file = output_dir / "lineage_records.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"File successfully saved to: {output_file}")


if __name__ == "__main__":
    _fetch_lineage_records()

# model name records
def get_model_names(unique_model_id: pd.DataFrame) -> List[str]:
    """
    Return model name strings in strict mappedID order.

    Row i of the returned list corresponds to the model whose mappedID == i,
    which is the ordering contract for xm0_builder's feature matrix.

    Parameters
    ----------
    unique_model_id : DataFrame with columns ['model', 'mappedID']

    Returns
    -------
    List[str] of length num_models, sorted by mappedID ascending.
    """
    ids = unique_model_id["mappedID"]
    assert ids.min() == 0 and ids.max() == len(ids) - 1 and ids.nunique() == len(ids), (
        "mappedID must be a 0-based consecutive range; "
        "did you call this after the homo-mode offset shift?"
    )
    return (
        unique_model_id
        .sort_values("mappedID")["model"]
        .tolist()
    )

# model description
def _fetch_one_description(repo_id: str, max_chars: int) -> str:
    """Fetch README text for one model; return '' on any failure."""
    try:
        card = ModelCard.load(repo_id)
        return (card.text or "")[:max_chars]
    except Exception:
        return ""


def get_model_descriptions(
    unique_model_id: pd.DataFrame,
    cache_path: Path | str | None = None,
    max_chars: int = 2000,
    request_delay: float = 0.5,
) -> List[str]:
    """
    Fetch README text for every model in unique_model_id from HuggingFace.

    Returns descriptions in strict mappedID order (same contract as
    get_model_names), so index i = description of the model with mappedID i.

    Supports resume-on-interrupt: already-fetched models are loaded from
    cache_path CSV and skipped; the CSV is updated every 10 new fetches.

    Parameters
    ----------
    unique_model_id : DataFrame with columns ['model', 'mappedID']
    cache_path      : path to a CSV with columns ['model', 'description'];
                      created if absent, appended to on resume
    max_chars       : README is truncated to this length before storing
    request_delay   : seconds to sleep between HuggingFace requests

    Returns
    -------
    List[str] of length num_models, index i = description for mappedID i.
    Empty string for models whose README could not be fetched.
    """
    ids = unique_model_id["mappedID"]
    assert ids.min() == 0 and ids.max() == len(ids) - 1 and ids.nunique() == len(ids), (
        "mappedID must be a 0-based consecutive range; "
        "did you call this after the homo-mode offset shift?"
    )

    ordered = unique_model_id.sort_values("mappedID").reset_index(drop=True)

    # ── load or initialise cache ──────────────────────────────────────────────
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_df = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=["model", "description"])
    else:
        cache_df = pd.DataFrame(columns=["model", "description"])

    fetched = set(cache_df["model"].tolist())
    to_fetch = [m for m in ordered["model"].tolist() if m not in fetched]
    print(f"[get_model_descriptions] {len(fetched)} cached, {len(to_fetch)} to fetch")

    # ── fetch missing models ──────────────────────────────────────────────────
    new_rows: list[dict] = []
    for i, model_id in enumerate(to_fetch):
        desc = _fetch_one_description(model_id, max_chars)
        status = f"[{len(desc)} chars]" if desc else "[empty]"
        print(f"  [{i+1}/{len(to_fetch)}] {model_id} {status}")
        new_rows.append({"model": model_id, "description": desc})

        # periodic save every 10 fetches so a crash loses at most 10 entries
        if cache_path is not None and len(new_rows) % 10 == 0:
            cache_df = pd.concat([cache_df, pd.DataFrame(new_rows)], ignore_index=True)
            new_rows = []
            cache_df.to_csv(cache_path, index=False)

        time.sleep(request_delay)

    # ── final save ────────────────────────────────────────────────────────────
    if new_rows:
        cache_df = pd.concat([cache_df, pd.DataFrame(new_rows)], ignore_index=True)
        if cache_path is not None:
            cache_df.to_csv(cache_path, index=False)

    # ── align to mappedID order ───────────────────────────────────────────────
    result = ordered.merge(cache_df[["model", "description"]], on="model", how="left")
    result["description"] = result["description"].fillna("")
    return result["description"].tolist()


# model param count (for size embedding)
def _fetch_one_param_count(repo_id: str) -> int | None:
    """Fetch total parameter count for one model via safetensors metadata; return None on failure."""
    try:
        info = api.model_info(repo_id, expand=["safetensors"])
        if info.safetensors is not None:
            total = getattr(info.safetensors, "total", None)
            if isinstance(total, int) and total > 0:
                return total
        return None
    except Exception:
        return None


def get_model_param_counts(
    unique_model_id: pd.DataFrame,
    cache_path: Path | str | None = None,
    request_delay: float = 0.5,
) -> List[int | None]:
    """
    Fetch total parameter count for every model in unique_model_id from HuggingFace.

    Returns param counts in strict mappedID order (same contract as
    get_model_names), so index i = param count of the model with mappedID i.

    Reads from the safetensors metadata field — available for most models
    stored in safetensors format. Returns None for models where this metadata
    is absent (GGUF-only, old PT checkpoints, private models, API errors).

    Supports resume-on-interrupt: already-fetched models are loaded from
    cache_path CSV and skipped; the CSV is updated every 10 new fetches.
    Models whose count could not be determined are stored with param_count=NaN
    so they are not retried on resume (same contract as get_model_descriptions).

    Parameters
    ----------
    unique_model_id : DataFrame with columns ['model', 'mappedID']
    cache_path      : path to a CSV with columns ['model', 'param_count'];
                      created if absent, appended to on resume
    request_delay   : seconds to sleep between HuggingFace requests

    Returns
    -------
    List[int | None] of length num_models, index i = param count for mappedID i.
    None for models whose parameter count could not be fetched.
    """
    ids = unique_model_id["mappedID"]
    assert ids.min() == 0 and ids.max() == len(ids) - 1 and ids.nunique() == len(ids), (
        "mappedID must be a 0-based consecutive range; "
        "did you call this after the homo-mode offset shift?"
    )

    ordered = unique_model_id.sort_values("mappedID").reset_index(drop=True)

    # ── load or initialise cache ──────────────────────────────────────────────
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_df = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=["model", "param_count"])
    else:
        cache_df = pd.DataFrame(columns=["model", "param_count"])

    fetched = set(cache_df["model"].tolist())
    to_fetch = [m for m in ordered["model"].tolist() if m not in fetched]
    print(f"[get_model_param_counts] {len(fetched)} cached, {len(to_fetch)} to fetch")

    # ── fetch missing models ──────────────────────────────────────────────────
    new_rows: list[dict] = []
    for i, model_id in enumerate(to_fetch):
        count = _fetch_one_param_count(model_id)
        status = f"[{count:,} params]" if count is not None else "[unknown]"
        print(f"  [{i+1}/{len(to_fetch)}] {model_id} {status}")
        new_rows.append({"model": model_id, "param_count": count})

        # periodic save every 10 fetches so a crash loses at most 10 entries
        if cache_path is not None and len(new_rows) % 10 == 0:
            cache_df = pd.concat([cache_df, pd.DataFrame(new_rows)], ignore_index=True)
            new_rows = []
            cache_df.to_csv(cache_path, index=False)

        time.sleep(request_delay)

    # ── final save ────────────────────────────────────────────────────────────
    if new_rows:
        cache_df = pd.concat([cache_df, pd.DataFrame(new_rows)], ignore_index=True)
        if cache_path is not None:
            cache_df.to_csv(cache_path, index=False)

    # ── align to mappedID order ───────────────────────────────────────────────
    result = ordered.merge(cache_df[["model", "param_count"]], on="model", how="left")
    return [
        int(v) if pd.notna(v) else None
        for v in result["param_count"].tolist()
    ]


# ── model family inference ────────────────────────────────────────────────────

FAMILY_OTHER: str = "Other"

# Ordered rules: first match wins. More-specific patterns are listed BEFORE any
# shorter string they contain as a substring:
#   "codellama" before "llama", "wizardcoder" before "wizard",
#   "openhermes" before "hermes", "gpt-neox" before "gpt-neo" before "gpt",
#   "distilgpt2" before "gpt2", "speecht5" before "t5", "mbart" before "bart",
#   "olmoe" before "olmo", "recurrentgemma" before "gemma",
#   "minicpm-v" before "minicpm", "deberta"/"roberta"/"albert" before "bert",
#   "chatglm" before "glm".
# Patterns are matched as substrings of the full lowercased model id ("org/repo-name").
_FAMILY_RULES: list[tuple[str, str]] = [
    # ── LLaMA family (most specific before bare "llama") ─────────────────────
    ("codellama",        "CodeLLaMA"),
    ("code-llama",       "CodeLLaMA"),
    ("meta-llama",       "LLaMA"),
    ("llama",            "LLaMA"),
    # LLaMA-derivative fine-tune families
    ("wizardcoder",      "WizardCoder"),    # before "wizard"
    ("wizard",           "WizardLM"),
    ("vicuna",           "Vicuna"),
    ("alpaca",           "Alpaca"),
    ("openhermes",       "Hermes"),         # before "hermes"
    ("hermes",           "Hermes"),         # NousResearch Hermes
    ("orca",             "Orca"),
    ("zephyr",           "Zephyr"),
    ("openchat",         "OpenChat"),
    ("platypus",         "Platypus"),
    ("guanaco",          "Guanaco"),
    ("beluga",           "Beluga"),
    ("samantha",         "Samantha"),
    ("tulu",             "Tulu"),
    ("neural-chat",      "NeuralChat"),     # Intel fine-tune
    ("notus",            "Notus"),
    ("yarn-",            "Yarn"),           # NousResearch long-context; dash avoids "yarn" substrings
    ("capybara",         "Capybara"),
    # ── Mistral family ────────────────────────────────────────────────────────
    ("mathstral",        "Mistral"),        # Mistral math variant
    ("mixtral",          "Mistral"),
    ("mistral",          "Mistral"),
    # ── Qwen family ───────────────────────────────────────────────────────────
    ("qwen",             "Qwen"),
    # ── Gemma family (recurrentgemma before gemma) ────────────────────────────
    ("recurrentgemma",   "Gemma"),
    ("gemma",            "Gemma"),
    # ── Phi family ────────────────────────────────────────────────────────────
    ("phi",              "Phi"),
    # ── DeepSeek family ───────────────────────────────────────────────────────
    ("deepseek",         "DeepSeek"),
    # ── Solar / Upstage ───────────────────────────────────────────────────────
    ("solar",            "Solar"),
    # ── Multimodal LLMs ───────────────────────────────────────────────────────
    ("llava",            "LLaVA"),
    ("cogvlm",           "CogVLM"),
    ("internvl",         "InternVL"),
    ("idefics",          "Idefics"),
    ("minicpm-v",        "MiniCPM-V"),      # before "minicpm"
    ("minigpt",          "MiniGPT"),
    ("blip",             "BLIP"),
    ("clip",             "CLIP"),
    ("florence",         "Florence"),       # Microsoft Florence
    ("fuyu",             "Fuyu"),           # Adept Fuyu
    ("kosmos",           "KOSMOS"),         # Microsoft KOSMOS
    ("flamingo",         "Flamingo"),       # DeepMind Flamingo
    ("pali",             "PaLI"),           # Google PaLI / PaLIGemma
    ("emu",              "Emu"),            # BAAI Emu / Emu2
    # ── Image generation / Diffusion ──────────────────────────────────────────
    ("stable-diffusion", "StableDiffusion"),
    ("stable_diffusion", "StableDiffusion"),
    ("sdxl",             "StableDiffusion"),
    ("flux",             "Flux"),           # Black Forest Labs Flux
    ("controlnet",       "ControlNet"),
    # ── Vision / Detection ────────────────────────────────────────────────────
    ("segment-anything", "SAM"),
    ("yolo",             "YOLO"),
    ("detr",             "DETR"),
    ("dino",             "DINO"),
    ("swin",             "Swin"),
    ("beit",             "BEiT"),
    ("deit",             "DeiT"),
    ("convnext",         "ConvNeXt"),
    ("resnet",           "ResNet"),
    ("efficientnet",     "EfficientNet"),
    ("mobilenet",        "MobileNet"),
    ("vit",              "ViT"),            # broad; listed after all vit-containing names above
    # ── Other major LLM families ──────────────────────────────────────────────
    ("falcon",           "Falcon"),
    ("mpt",              "MPT"),
    ("olmoe",            "OLMoE"),          # before "olmo"
    ("olmo",             "OLMo"),
    ("stablelm",         "StableLM"),
    ("dolly",            "Dolly"),
    ("pythia",           "Pythia"),
    ("rwkv",             "RWKV"),
    ("mamba",            "Mamba"),
    ("jamba",            "Jamba"),
    ("dbrx",             "DBRX"),
    ("nemotron",         "Nemotron"),
    ("granite",          "Granite"),
    ("smollm",           "SmolLM"),
    ("command",          "Command"),        # Cohere Command / Command-R
    ("aya",              "Aya"),            # Cohere Aya
    ("amber",            "Amber"),          # LLM360 Amber
    ("persimmon",        "Persimmon"),      # Adept Persimmon
    ("minicpm",          "MiniCPM"),        # OpenBMB MiniCPM
    ("bloom",            "BLOOM"),
    # ── Yi / 01-ai family ─────────────────────────────────────────────────────
    ("01-ai",            "Yi"),
    ("yi-",              "Yi"),             # dash avoids matching "yiyan" etc.
    # ── Chinese LLM families ──────────────────────────────────────────────────
    ("baichuan",         "Baichuan"),
    ("chatglm",          "ChatGLM"),        # before "glm"
    ("glm",              "ChatGLM"),        # THUDM/glm-*
    ("internlm",         "InternLM"),
    ("moss",             "MOSS"),           # Fudan MOSS
    ("xverse",           "XVERSE"),
    ("tigerbot",         "TigerBot"),
    ("aquila",           "Aquila"),         # BAAI Aquila
    ("bluelm",           "BlueLM"),
    ("skywork",          "Skywork"),
    ("orion",            "Orion"),          # OrionStarAI
    ("seallm",           "SeaLLM"),
    # ── GPT family (specific before generic) ──────────────────────────────────
    ("gpt-neox",         "GPT-NeoX"),       # before "gpt-neo"
    ("gpt-j",            "GPT-J"),
    ("gpt-neo",          "GPT-Neo"),        # before "gpt"
    ("distilgpt2",       "GPT-2"),          # before "gpt2"
    ("gpt2",             "GPT-2"),
    ("gpt-2",            "GPT-2"),
    ("gpt",              "GPT"),
    # ── BERT family (specific before generic) ─────────────────────────────────
    ("distilbert",       "DistilBERT"),
    ("deberta",          "DeBERTa"),        # "bert" ⊂ "deberta"
    ("roberta",          "RoBERTa"),        # "bert" ⊂ "roberta"
    ("albert",           "ALBERT"),         # "bert" ⊂ "albert"
    ("electra",          "ELECTRA"),
    ("xlnet",            "XLNet"),
    ("camembert",        "CamemBERT"),      # "bert" ⊂ "camembert"
    ("flaubert",         "FlauBERT"),       # "bert" ⊂ "flaubert"
    ("ernie",            "ERNIE"),          # Baidu ERNIE
    ("macbert",          "MacBERT"),        # "bert" ⊂ "macbert"
    ("bert",             "BERT"),
    # ── Seq2Seq / Encoder-Decoder ──────────────────────────────────────────────
    ("flan",             "T5"),             # flan-t5, flan-ul2
    ("speecht5",         "SpeechT5"),       # before "t5" ("t5" ⊂ "speecht5")
    ("t5",               "T5"),
    ("mbart",            "mBART"),          # before "bart" ("bart" ⊂ "mbart")
    ("bart",             "BART"),
    ("pegasus",          "PEGASUS"),
    ("m2m",              "M2M-100"),        # Facebook M2M multilingual translation
    ("nllb",             "NLLB"),           # No Language Left Behind
    ("opus-mt",          "OpusMT"),         # Helsinki-NLP OpusMT
    # ── Multilingual / Cross-lingual ───────────────────────────────────────────
    ("xlm",              "XLM"),
    # ── OPT ───────────────────────────────────────────────────────────────────
    ("opt",              "OPT"),
    # ── Code-specific models ───────────────────────────────────────────────────
    ("starcoder",        "StarCoder"),
    ("santacoder",       "SantaCoder"),
    ("codegen",          "CodeGen"),
    ("incoder",          "InCoder"),
    ("polycoder",        "PolyCoder"),
    ("replit-code",      "Replit"),
    ("magicoder",        "Magicoder"),
    # ── Audio / Speech ─────────────────────────────────────────────────────────
    ("whisper",          "Whisper"),
    ("wavlm",            "WavLM"),          # before "wav2vec"
    ("wav2vec",          "Wav2Vec"),
    ("hubert",           "HuBERT"),
    ("seamless",         "SeamlessM4T"),    # Meta SeamlessM4T
    ("musicgen",         "MusicGen"),       # Meta MusicGen
    ("encodec",          "EnCodec"),        # Meta EnCodec
    ("bark",             "Bark"),           # Suno Bark TTS
    ("clap",             "CLAP"),           # LAION CLAP
    # ── Text Embedding models ──────────────────────────────────────────────────
    ("e5-",              "E5"),             # Microsoft E5; dash avoids false positives
    ("bge-",             "BGE"),            # BAAI BGE
    ("gte-",             "GTE"),            # Alibaba GTE
    ("nomic-embed",      "NomicEmbed"),
    ("instructor",       "INSTRUCTOR"),     # INSTRUCTOR embedding
    ("jina",             "Jina"),           # Jina AI embeddings
]

# Canonical list of named families from static rules (insertion-ordered, deduplicated).
# This is the BACKBONE — at runtime the actual family vocabulary may be larger, because
# models that match no rule get an auto-extracted family name (see _extract_candidate_family).
# xm0_builder derives the full id mapping from the returned strings, not from this list.
KNOWN_FAMILIES: list[str] = list(dict.fromkeys(fam for _, fam in _FAMILY_RULES))

# ── dynamic family extraction (fallback for rule misses) ─────────────────────

# Tokens that appear in repo names but carry no architecture identity.
_SKIP_TOKENS: frozenset[str] = frozenset({
    # task / tuning stage
    "base", "chat", "instruct", "instruction", "sft", "rlhf", "dpo", "ppo",
    "it", "ift", "hf", "finetune", "finetuned",
    # quantisation / precision
    "gguf", "gptq", "awq", "bnb", "nf4", "int4", "int8",
    "bf16", "fp16", "fp32", "4bit", "8bit", "q4", "q8",
    # common descriptor adjectives
    "small", "mini", "large", "medium", "xl", "xxl", "tiny",
    "fast", "lite", "light", "plus", "pro", "ultra", "turbo",
    "open", "free", "best", "new", "the", "model", "lm", "llm",
    "v1", "v2", "v3", "v4", "v5",
})
# Patterns that look like parameter-count suffixes ("7b", "1.3b", "125m", "70b")
# or version strings ("v1.0", "v2.1").
_SIZE_TOKEN_RE = re.compile(r"^\d+\.?\d*[bBmMkKtT]$")
_VER_TOKEN_RE  = re.compile(r"^v\d")


def _extract_candidate_family(model_name: str) -> str:
    """
    Heuristic fallback: extract a candidate family name from the repo portion
    of a model id when no static rule matches.

    Strategy:
      1. Take the part after '/' (repo name), lowercase it.
      2. Split on [-_. ].
      3. Walk tokens left-to-right; skip size suffixes, version strings,
         single/double-char tokens, and tokens in _SKIP_TOKENS.
      4. Return the first surviving token, capitalised; or FAMILY_OTHER if none.

    Examples
    --------
    "togethercomputer/RedPajama-INCITE-7B-Base"  -> "Redpajama"
    "myorg/NewArch-13b-chat"                     -> "Newarch"
    "someorg/7b-finetuned"                       -> FAMILY_OTHER  (no useful token)
    """
    n = (model_name or "").strip()
    repo = n.split("/")[-1].lower() if "/" in n else n.lower()
    for tok in re.split(r"[-_.\s]+", repo):
        if not tok:
            continue
        if len(tok) <= 2:
            continue
        if _SIZE_TOKEN_RE.match(tok):
            continue
        if _VER_TOKEN_RE.match(tok):
            continue
        if tok in _SKIP_TOKENS:
            continue
        if not any(c.isalpha() for c in tok):
            continue
        return tok.capitalize()
    return FAMILY_OTHER


def _infer_one_family(model_name: str) -> str:
    """
    Infer model family from model name string.

    Step 1 — static rules (_FAMILY_RULES): checked in order, first match wins.
    Step 2 — dynamic extraction (_extract_candidate_family): applied when no
              rule matches; extracts the first meaningful token from the repo name.
    Step 3 — FAMILY_OTHER: returned only if extraction also yields nothing.

    The two-step design means "Other" is a genuine last resort, not a catch-all
    for every model that happens to lack a static rule entry.
    """
    n = (model_name or "").lower()
    for pattern, family in _FAMILY_RULES:
        if pattern in n:
            return family
    return _extract_candidate_family(model_name)


def get_model_families(
    unique_model_id: pd.DataFrame,
    cache_path: Path | str | None = None,
) -> List[str]:
    """
    Infer model family for every model in unique_model_id from the model name.

    Family assignment is rule-based (no HuggingFace network calls) — fast even
    at 47K-model scale. The cache serves a specific purpose here: models already
    in the cache are NEVER re-inferred, which lets you manually correct
    auto-assigned families in the CSV and have those corrections persist across
    runs. New models not in the cache are inferred and appended.

    Returns families in strict mappedID order (same contract as get_model_names),
    so index i = family of the model with mappedID i.

    Parameters
    ----------
    unique_model_id : DataFrame with columns ['model', 'mappedID']
    cache_path      : path to a CSV with columns ['model', 'family'];
                      created if absent, appended to on resume.
                      Edit rows in this file to override auto-inferred families.

    Returns
    -------
    List[str] of length num_models, index i = family for mappedID i.
    FAMILY_OTHER ("Other") for models whose family could not be inferred.
    """
    ids = unique_model_id["mappedID"]
    assert ids.min() == 0 and ids.max() == len(ids) - 1 and ids.nunique() == len(ids), (
        "mappedID must be a 0-based consecutive range; "
        "did you call this after the homo-mode offset shift?"
    )

    ordered = unique_model_id.sort_values("mappedID").reset_index(drop=True)

    # ── load or initialise cache ──────────────────────────────────────────────
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_df = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=["model", "family"])
    else:
        cache_df = pd.DataFrame(columns=["model", "family"])

    already_done = set(cache_df["model"].tolist())
    to_infer = [m for m in ordered["model"].tolist() if m not in already_done]
    print(f"[get_model_families] {len(already_done)} cached, {len(to_infer)} to infer")

    # ── infer missing models (local, no network) ──────────────────────────────
    if to_infer:
        new_rows = [{"model": m, "family": _infer_one_family(m)} for m in to_infer]
        cache_df = pd.concat([cache_df, pd.DataFrame(new_rows)], ignore_index=True)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_df.to_csv(cache_path, index=False)
        print(f"  inferred {len(to_infer)} families -> saved")

    # ── align to mappedID order ───────────────────────────────────────────────
    result = ordered.merge(cache_df[["model", "family"]], on="model", how="left")
    result["family"] = result["family"].fillna(FAMILY_OTHER)
    return result["family"].tolist()