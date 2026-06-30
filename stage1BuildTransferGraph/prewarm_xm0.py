"""prewarm_xm0.py -- pre-fetch the 2000 selected models' README descriptions into
the exact cache build_graph (xm0) reads in Phase 5, so the build skips that network
I/O. Safe to run in parallel with dataset embedding (network vs CPU)."""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_embed.utils.fetch_metadata import get_model_descriptions

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dataset_embed", "data_hf1000d_2000m")
os.makedirs(DATA, exist_ok=True)
cache = os.path.join(DATA, "model_descriptions.csv")

sel = pd.read_csv(os.path.join(HERE, "hf1000d_2000m", "selected_2000_models.csv"))
uid = pd.DataFrame({"model": sel["model_id"].tolist(),
                    "mappedID": range(len(sel))})
print(f"prewarming {len(uid)} model descriptions -> {cache}")
descs = get_model_descriptions(uid, cache_path=cache)
nonempty = sum(1 for d in descs if d and d.strip())
print(f"done: {len(descs)} models, {nonempty} non-empty descriptions cached")
