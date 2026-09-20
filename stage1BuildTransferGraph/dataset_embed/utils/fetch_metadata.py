import re
import time
import pandas as pd
from typing import List
from pathlib import Path
from huggingface_hub import HfApi, ModelCard

api = HfApi()


def _fetch_lineage_records():
    model_ids = \
    ["google/gemma-4-31B-it",
     "gghfez/gemma-4-31b-it-control-vectors",
     "nvidia/Gemma-4-31B-IT-NVFP4",
     "virtuous7373/Gemma-4-Harmonia-31B",
     "unsloth/gemma-4-31B-it-GGUF"]

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

def get_model_names(unique_model_id: pd.DataFrame) -> List[str]:
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

def _fetch_one_description(repo_id: str, max_chars: int) -> str:
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
    ids = unique_model_id["mappedID"]
    assert ids.min() == 0 and ids.max() == len(ids) - 1 and ids.nunique() == len(ids), (
        "mappedID must be a 0-based consecutive range; "
        "did you call this after the homo-mode offset shift?"
    )

    ordered = unique_model_id.sort_values("mappedID").reset_index(drop=True)

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_df = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=["model", "description"])
    else:
        cache_df = pd.DataFrame(columns=["model", "description"])

    fetched = set(cache_df["model"].tolist())
    to_fetch = [m for m in ordered["model"].tolist() if m not in fetched]
    print(f"[get_model_descriptions] {len(fetched)} cached, {len(to_fetch)} to fetch")

    new_rows: list[dict] = []
    for i, model_id in enumerate(to_fetch):
        desc = _fetch_one_description(model_id, max_chars)
        status = f"[{len(desc)} chars]" if desc else "[empty]"
        print(f"  [{i+1}/{len(to_fetch)}] {model_id} {status}")
        new_rows.append({"model": model_id, "description": desc})

        if cache_path is not None and len(new_rows) % 10 == 0:
            cache_df = pd.concat([cache_df, pd.DataFrame(new_rows)], ignore_index=True)
            new_rows = []
            cache_df.to_csv(cache_path, index=False)

        time.sleep(request_delay)

    if new_rows:
        cache_df = pd.concat([cache_df, pd.DataFrame(new_rows)], ignore_index=True)
        if cache_path is not None:
            cache_df.to_csv(cache_path, index=False)

    result = ordered.merge(cache_df[["model", "description"]], on="model", how="left")
    result["description"] = result["description"].fillna("")
    return result["description"].tolist()


def _fetch_one_param_count(repo_id: str) -> int | None:
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
    ids = unique_model_id["mappedID"]
    assert ids.min() == 0 and ids.max() == len(ids) - 1 and ids.nunique() == len(ids), (
        "mappedID must be a 0-based consecutive range; "
        "did you call this after the homo-mode offset shift?"
    )

    ordered = unique_model_id.sort_values("mappedID").reset_index(drop=True)

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_df = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=["model", "param_count"])
    else:
        cache_df = pd.DataFrame(columns=["model", "param_count"])

    fetched = set(cache_df["model"].tolist())
    to_fetch = [m for m in ordered["model"].tolist() if m not in fetched]
    print(f"[get_model_param_counts] {len(fetched)} cached, {len(to_fetch)} to fetch")

    new_rows: list[dict] = []
    for i, model_id in enumerate(to_fetch):
        count = _fetch_one_param_count(model_id)
        status = f"[{count:,} params]" if count is not None else "[unknown]"
        print(f"  [{i+1}/{len(to_fetch)}] {model_id} {status}")
        new_rows.append({"model": model_id, "param_count": count})

        if cache_path is not None and len(new_rows) % 10 == 0:
            cache_df = pd.concat([cache_df, pd.DataFrame(new_rows)], ignore_index=True)
            new_rows = []
            cache_df.to_csv(cache_path, index=False)

        time.sleep(request_delay)

    if new_rows:
        cache_df = pd.concat([cache_df, pd.DataFrame(new_rows)], ignore_index=True)
        if cache_path is not None:
            cache_df.to_csv(cache_path, index=False)

    result = ordered.merge(cache_df[["model", "param_count"]], on="model", how="left")
    return [
        int(v) if pd.notna(v) else None
        for v in result["param_count"].tolist()
    ]


FAMILY_OTHER: str = "Other"

_FAMILY_RULES: list[tuple[str, str]] = [
    ("codellama",        "CodeLLaMA"),
    ("code-llama",       "CodeLLaMA"),
    ("meta-llama",       "LLaMA"),
    ("llama",            "LLaMA"),
    ("wizardcoder",      "WizardCoder"),
    ("wizard",           "WizardLM"),
    ("vicuna",           "Vicuna"),
    ("alpaca",           "Alpaca"),
    ("openhermes",       "Hermes"),
    ("hermes",           "Hermes"),
    ("orca",             "Orca"),
    ("zephyr",           "Zephyr"),
    ("openchat",         "OpenChat"),
    ("platypus",         "Platypus"),
    ("guanaco",          "Guanaco"),
    ("beluga",           "Beluga"),
    ("samantha",         "Samantha"),
    ("tulu",             "Tulu"),
    ("neural-chat",      "NeuralChat"),
    ("notus",            "Notus"),
    ("yarn-",            "Yarn"),
    ("capybara",         "Capybara"),
    ("mathstral",        "Mistral"),
    ("mixtral",          "Mistral"),
    ("mistral",          "Mistral"),
    ("qwen",             "Qwen"),
    ("recurrentgemma",   "Gemma"),
    ("gemma",            "Gemma"),
    ("phi",              "Phi"),
    ("deepseek",         "DeepSeek"),
    ("solar",            "Solar"),
    ("llava",            "LLaVA"),
    ("cogvlm",           "CogVLM"),
    ("internvl",         "InternVL"),
    ("idefics",          "Idefics"),
    ("minicpm-v",        "MiniCPM-V"),
    ("minigpt",          "MiniGPT"),
    ("blip",             "BLIP"),
    ("clip",             "CLIP"),
    ("florence",         "Florence"),
    ("fuyu",             "Fuyu"),
    ("kosmos",           "KOSMOS"),
    ("flamingo",         "Flamingo"),
    ("pali",             "PaLI"),
    ("emu",              "Emu"),
    ("stable-diffusion", "StableDiffusion"),
    ("stable_diffusion", "StableDiffusion"),
    ("sdxl",             "StableDiffusion"),
    ("flux",             "Flux"),
    ("controlnet",       "ControlNet"),
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
    ("vit",              "ViT"),
    ("falcon",           "Falcon"),
    ("mpt",              "MPT"),
    ("olmoe",            "OLMoE"),
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
    ("command",          "Command"),
    ("aya",              "Aya"),
    ("amber",            "Amber"),
    ("persimmon",        "Persimmon"),
    ("minicpm",          "MiniCPM"),
    ("bloom",            "BLOOM"),
    ("01-ai",            "Yi"),
    ("yi-",              "Yi"),
    ("baichuan",         "Baichuan"),
    ("chatglm",          "ChatGLM"),
    ("glm",              "ChatGLM"),
    ("internlm",         "InternLM"),
    ("moss",             "MOSS"),
    ("xverse",           "XVERSE"),
    ("tigerbot",         "TigerBot"),
    ("aquila",           "Aquila"),
    ("bluelm",           "BlueLM"),
    ("skywork",          "Skywork"),
    ("orion",            "Orion"),
    ("seallm",           "SeaLLM"),
    ("gpt-neox",         "GPT-NeoX"),
    ("gpt-j",            "GPT-J"),
    ("gpt-neo",          "GPT-Neo"),
    ("distilgpt2",       "GPT-2"),
    ("gpt2",             "GPT-2"),
    ("gpt-2",            "GPT-2"),
    ("gpt",              "GPT"),
    ("distilbert",       "DistilBERT"),
    ("deberta",          "DeBERTa"),
    ("roberta",          "RoBERTa"),
    ("albert",           "ALBERT"),
    ("electra",          "ELECTRA"),
    ("xlnet",            "XLNet"),
    ("camembert",        "CamemBERT"),
    ("flaubert",         "FlauBERT"),
    ("ernie",            "ERNIE"),
    ("macbert",          "MacBERT"),
    ("bert",             "BERT"),
    ("flan",             "T5"),
    ("speecht5",         "SpeechT5"),
    ("t5",               "T5"),
    ("mbart",            "mBART"),
    ("bart",             "BART"),
    ("pegasus",          "PEGASUS"),
    ("m2m",              "M2M-100"),
    ("nllb",             "NLLB"),
    ("opus-mt",          "OpusMT"),
    ("xlm",              "XLM"),
    ("opt",              "OPT"),
    ("starcoder",        "StarCoder"),
    ("santacoder",       "SantaCoder"),
    ("codegen",          "CodeGen"),
    ("incoder",          "InCoder"),
    ("polycoder",        "PolyCoder"),
    ("replit-code",      "Replit"),
    ("magicoder",        "Magicoder"),
    ("whisper",          "Whisper"),
    ("wavlm",            "WavLM"),
    ("wav2vec",          "Wav2Vec"),
    ("hubert",           "HuBERT"),
    ("seamless",         "SeamlessM4T"),
    ("musicgen",         "MusicGen"),
    ("encodec",          "EnCodec"),
    ("bark",             "Bark"),
    ("clap",             "CLAP"),
    ("e5-",              "E5"),
    ("bge-",             "BGE"),
    ("gte-",             "GTE"),
    ("nomic-embed",      "NomicEmbed"),
    ("instructor",       "INSTRUCTOR"),
    ("jina",             "Jina"),
]

KNOWN_FAMILIES: list[str] = list(dict.fromkeys(fam for _, fam in _FAMILY_RULES))


_SKIP_TOKENS: frozenset[str] = frozenset({
    "base", "chat", "instruct", "instruction", "sft", "rlhf", "dpo", "ppo",
    "it", "ift", "hf", "finetune", "finetuned",
    "gguf", "gptq", "awq", "bnb", "nf4", "int4", "int8",
    "bf16", "fp16", "fp32", "4bit", "8bit", "q4", "q8",
    "small", "mini", "large", "medium", "xl", "xxl", "tiny",
    "fast", "lite", "light", "plus", "pro", "ultra", "turbo",
    "open", "free", "best", "new", "the", "model", "lm", "llm",
    "v1", "v2", "v3", "v4", "v5",
})
_SIZE_TOKEN_RE = re.compile(r"^\d+\.?\d*[bBmMkKtT]$")
_VER_TOKEN_RE  = re.compile(r"^v\d")


def _extract_candidate_family(model_name: str) -> str:
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
    n = (model_name or "").lower()
    for pattern, family in _FAMILY_RULES:
        if pattern in n:
            return family
    return _extract_candidate_family(model_name)


def get_model_families(
    unique_model_id: pd.DataFrame,
    cache_path: Path | str | None = None,
) -> List[str]:
    ids = unique_model_id["mappedID"]
    assert ids.min() == 0 and ids.max() == len(ids) - 1 and ids.nunique() == len(ids), (
        "mappedID must be a 0-based consecutive range; "
        "did you call this after the homo-mode offset shift?"
    )

    ordered = unique_model_id.sort_values("mappedID").reset_index(drop=True)

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_df = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=["model", "family"])
    else:
        cache_df = pd.DataFrame(columns=["model", "family"])

    already_done = set(cache_df["model"].tolist())
    to_infer = [m for m in ordered["model"].tolist() if m not in already_done]
    print(f"[get_model_families] {len(already_done)} cached, {len(to_infer)} to infer")

    if to_infer:
        new_rows = [{"model": m, "family": _infer_one_family(m)} for m in to_infer]
        cache_df = pd.concat([cache_df, pd.DataFrame(new_rows)], ignore_index=True)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_df.to_csv(cache_path, index=False)
        print(f"  inferred {len(to_infer)} families -> saved")

    result = ordered.merge(cache_df[["model", "family"]], on="model", how="left")
    result["family"] = result["family"].fillna(FAMILY_OTHER)
    return result["family"].tolist()