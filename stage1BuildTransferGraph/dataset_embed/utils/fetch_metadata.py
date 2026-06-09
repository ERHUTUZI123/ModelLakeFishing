import time
import pandas as pd
from typing import List
from pathlib import Path
from huggingface_hub import HfApi, ModelCard

api = HfApi()

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