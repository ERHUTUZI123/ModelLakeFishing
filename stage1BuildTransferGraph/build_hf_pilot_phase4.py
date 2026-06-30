"""
build_hf_pilot_phase4.py -- materialize the v3 HF pilot into a Stage-1 data dir and
generate the dataset domain embeddings, WITHOUT changing Stage-1/Stage-2 logic and
WITHOUT fabricating edges.

Inputs (from Phases 1-3):
  hf300m_100d/selected_300_models.csv
  hf300m_100d/selected_100_datasets_v3.csv
  hf300m_100d/performance_edges_v3.csv   (real model-index accuracy/f1 only)

Outputs into dataset_embed/data_hf300m_100d/ (isolated; selected via MLF_DATA_DIR):
  embedded_dataset/domain_similarity/EleutherAI_gpt-neo-125m/<name>_feature.npy   (--phase embed)
  model_config_dataset.csv, records.csv, lineage_records.csv                      (--phase materialize)

Then build (separate command):
  MLF_DATA_DIR=.../data_hf300m_100d python build_graph.py \
    --contain_model_feature True --contain_rich_dataset_feature True \
    --gnn_method SAGEConv_without_transfer --out hgraph_hf300m_100d_xm0_xd0.pt

Embedding mirrors the existing pipeline (gpt-neo-125m, hidden_states[-1].mean(dim=1));
text datasets only. Multimodal / script-only / gated datasets are SKIPPED and reported
(they simply won't be graph nodes). Resumable (skips existing .npy).
"""

import argparse
import ast
import json
import os
import sys
from itertools import islice

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.join(HERE, "hf300m_100d")
DATA = os.path.join(HERE, "dataset_embed", "data_hf300m_100d")
EMB_DIR = os.path.join(DATA, "embedded_dataset", "domain_similarity", "EleutherAI_gpt-neo-125m")
PROBE = "EleutherAI/gpt-neo-125m"
N_SAMPLES = 256
MAX_LEN = 128

_TEXT_PRIORITY = ["text", "sentence", "sentence1", "premise", "question", "document",
                  "content", "tweet", "review", "context", "passage", "query", "title",
                  "comment_text", "sentence2", "hypothesis", "answer"]


def _sanitize(name):
    return str(name).replace("/", "_").replace(" ", "-")


def _example_text(ex):
    """Concatenate the string fields of one example into a probe input."""
    parts = []
    for k in _TEXT_PRIORITY:
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v)
    if not parts:  # fallback: any string field
        for k, v in ex.items():
            if isinstance(v, str) and v.strip():
                parts.append(v)
            elif isinstance(v, dict):  # e.g. translation: {'en':..,'fr':..}
                parts += [x for x in v.values() if isinstance(x, str)]
    return " ".join(parts)[:2000] if parts else ""


def embed_datasets(limit=None):
    os.makedirs(EMB_DIR, exist_ok=True)
    sel = pd.read_csv(os.path.join(PILOT, "selected_100_datasets_v3.csv"))
    import torch
    from transformers import AutoTokenizer, AutoModel
    from datasets import load_dataset
    print(f"loading probe {PROBE} ...")
    tok = AutoTokenizer.from_pretrained(PROBE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(PROBE)
    model.eval()
    dev = "cpu"

    status = []
    rows = sel.itertuples()
    done = 0
    for r in rows:
        key = r.canon_key
        hf_id = r.hf_id
        config = None if pd.isna(r.config) else r.config
        out = os.path.join(EMB_DIR, _sanitize(key) + "_feature.npy")
        if os.path.exists(out):
            status.append((key, "cached")); continue
        if limit is not None and done >= limit:
            break
        try:
            try:
                ds = load_dataset(hf_id, config, split="train", streaming=True)
            except Exception:
                # try without config / other split
                ds = load_dataset(hf_id, split="train", streaming=True)
            texts = []
            for ex in islice(ds, N_SAMPLES * 2):
                t = _example_text(ex)
                if t:
                    texts.append(t)
                if len(texts) >= N_SAMPLES:
                    break
            if len(texts) < 8:
                status.append((key, "skip:no-text")); continue
            feats = []
            with torch.no_grad():
                for i in range(0, len(texts), 16):
                    batch = tok(texts[i:i+16], return_tensors="pt", truncation=True,
                                max_length=MAX_LEN, padding=True)
                    out_h = model(**{k: v.to(dev) for k, v in batch.items()},
                                  output_hidden_states=True)
                    feats.append(out_h.hidden_states[-1].mean(dim=1).cpu().numpy())
            arr = np.concatenate(feats, axis=0).astype(np.float32)
            np.save(out, arr)
            status.append((key, f"ok:{arr.shape[0]}x{arr.shape[1]}"))
            done += 1
            print(f"  [{done}] {key} <- {hf_id} {config or ''}  {arr.shape}")
        except Exception as e:
            status.append((key, f"skip:{type(e).__name__}"))
            print(f"  skip {key} <- {hf_id}: {type(e).__name__} {str(e)[:80]}")
    # summary
    sdf = pd.DataFrame(status, columns=["canon_key", "status"])
    sdf.to_csv(os.path.join(PILOT, "embedding_status.csv"), index=False)
    ok = sdf["status"].str.startswith(("ok", "cached")).sum()
    print(f"\n[embed] {ok}/{len(sel)} datasets embedded; "
          f"skipped {len(sel)-ok}. status -> hf300m_100d/embedding_status.csv")
    from collections import Counter
    print("  reasons:", dict(Counter(s.split(':')[0] for s in sdf['status'])))


def materialize():
    """Write Stage-1 CSVs from the v3 selection + real edges. Only datasets with a
    successfully generated embedding become nodes."""
    sel_m = pd.read_csv(os.path.join(PILOT, "selected_300_models.csv"))
    edges = pd.read_csv(os.path.join(PILOT, "performance_edges_v3.csv"))
    embedded = {f[:-len("_feature.npy")] for f in os.listdir(EMB_DIR) if f.endswith("_feature.npy")}
    print(f"[materialize] {len(embedded)} embedded datasets available as nodes")

    # canon_key -> sanitized embedding stem; keep only edges whose dataset embedded
    edges = edges[edges["canon_key"].map(lambda k: _sanitize(k) in embedded)].copy()
    # ONE metric per dataset (don't mix): pick the dataset's dominant usable metric
    keep_rows = []
    for key, sub in edges.groupby("canon_key"):
        mt = sub["metric_name"].value_counts().idxmax()
        s = sub[sub["metric_name"] == mt].drop_duplicates(["model_id"])
        for _, e in s.iterrows():
            keep_rows.append(dict(model=e["model_id"], dataset=key,
                                  accuracy=float(e["metric_value"]), metric=mt))
    edf = pd.DataFrame(keep_rows)
    print(f"[materialize] usable trained_on edges kept: {len(edf)} on {edf['dataset'].nunique()} datasets")

    # model_config_dataset.csv: one row per edge (model,dataset,accuracy) + node-only rows
    mc_rows = []
    i = 0
    arch_map = dict(zip(sel_m["model_id"], sel_m.get("architecture", pd.Series(dtype=str))))
    type_map = dict(zip(sel_m["model_id"], sel_m.get("model_type", pd.Series(dtype=str))))
    param_map = dict(zip(sel_m["model_id"], sel_m.get("parameter_count", pd.Series(dtype=float))))

    def row(model, dataset, acc):
        nonlocal i
        a = arch_map.get(model)
        r = {"Unnamed: 0": i, "model": model,
             "architectures": f"['{a}']" if isinstance(a, str) and a else None,
             "number_of_labels": None, "labels": None, "model_type": type_map.get(model),
             "number_of_parameters": param_map.get(model), "memory_consumption": None,
             "dataset": dataset, "accuracy": acc}
        i += 1
        return r

    edge_models = set(edf["model"])
    for _, e in edf.iterrows():
        mc_rows.append(row(e["model"], e["dataset"], e["accuracy"]))
    for m in sel_m["model_id"]:
        if m not in edge_models:                       # node-only model (no trained_on edge)
            mc_rows.append(row(m, None, None))
    mc = pd.DataFrame(mc_rows)

    # lineage_records.csv (kept for provenance/manifest; NOT installed under without_transfer)
    lin_rows = []
    if "base_model" in sel_m.columns:
        node_models = set(sel_m["model_id"])
        for _, m in sel_m.iterrows():
            b = m["base_model"]
            if isinstance(b, str) and b and b in node_models and b != m["model_id"]:
                lin_rows.append(dict(model=m["model_id"], relation="finetune", base_model=b))
    lin = pd.DataFrame(lin_rows or [{"model": None, "relation": None, "base_model": None}])

    os.makedirs(DATA, exist_ok=True)
    mc.to_csv(os.path.join(DATA, "model_config_dataset.csv"), index=False)
    # records.csv: minimal real finetune rows mirroring edges (eval_accuracy)
    rec = edf.rename(columns={"dataset": "finetuned_dataset", "accuracy": "eval_accuracy"})
    rec["model_name"] = rec["model"]; rec["task_type"] = "text-classification"
    rec.to_csv(os.path.join(DATA, "records.csv"), index=False)
    lin.to_csv(os.path.join(DATA, "lineage_records.csv"), index=False)
    # empty transferability file (no fabrication); without_transfer will skip it
    pd.DataFrame(columns=["model", "target_dataset", "score"]).to_csv(
        os.path.join(DATA, "transferability_score_records.csv"))

    print(f"[materialize] wrote {len(mc)} model_config rows "
          f"({len(edge_models)} models with >=1 edge, {len(sel_m)-len(edge_models)} node-only), "
          f"{len(lin_rows)} lineage rows (NOT installed under without_transfer).")
    print(f"  data dir: {DATA}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["embed", "materialize"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.phase == "embed":
        embed_datasets(args.limit)
    else:
        materialize()


if __name__ == "__main__":
    main()
