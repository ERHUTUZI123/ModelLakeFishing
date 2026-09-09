"""
phase4_embed.py -- Effective-Dataset guide, Phase 4.

Embed the effective supervised core (the datasets that carry >=1 effective ranking
group) with the SAME xd0-compatible gpt-neo-125m probe, into an ISOLATED v2 dir:

  dataset_embed/data_hf_effective_2000m_v2/embedded_dataset/domain_similarity/EleutherAI_gpt-neo-125m/

Selection priority is coverage (strong -> robust -> minimum-evaluable); we embed every
effective dataset. Already-embedded datasets are reused (copied) from the old dir by
normalized-name match; the rest are embedded fresh on the GPU. Every failure is logged
with a category, per the guide.

Dataset features use ONLY deployable info (raw examples); no performance label ever
enters the embedding. Writes:
  dataset_embedding_failures.csv
  phase4_embed_report.md
  effective_dataset_selection.csv
"""

import json
import os
import re
import shutil
import sys
from collections import Counter

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import ModelLakeFishing.stage1BuildTransferGraph.hf1000d_phase4 as p4  # helpers

ART = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage2TrainGraphSAGE",
                   "artifacts", "effective_dataset_v2")
OLD_EMB = os.path.join(_HERE, "dataset_embed", "data_hf1000d_2000m",
                       "embedded_dataset", "domain_similarity", "EleutherAI_gpt-neo-125m")
V2_DATA = os.path.join(_HERE, "dataset_embed", "data_hf_effective_2000m_v2")
V2_EMB = os.path.join(V2_DATA, "embedded_dataset", "domain_similarity", "EleutherAI_gpt-neo-125m")


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def build_selection():
    B = pd.read_parquet(os.path.join(ART, "raw_performance_observations.parquet"))
    eff = []
    for f in range(5):
        eff += json.load(open(os.path.join(ART, "cold_folds", f"fold_{f}.json"),
                              encoding="utf-8"))["cold_dataset_names"]
    eff = set(eff)
    sub = B[B["dataset_canonical"].isin(eff)]
    rows = []
    for ds, g in sub.groupby("dataset_canonical"):
        cfg = g["dataset_config"].mode()
        rows.append({
            "dataset_canonical": ds,
            "hf_id": ds,                        # model-index type is the loadable id
            "config": (cfg.iloc[0] if len(cfg) else ""),
            "task": g["task"].mode().iloc[0],
            "n_models": int(g["model_id_canonical"].nunique()),
        })
    sel = pd.DataFrame(rows).sort_values("n_models", ascending=False)  # strong first
    sel.to_csv(os.path.join(ART, "effective_dataset_selection.csv"), index=False, encoding="utf-8")
    return sel


def main(limit=None):
    os.makedirs(V2_EMB, exist_ok=True)
    p4.EMB_DIR = V2_EMB                          # redirect helper aux writes to v2
    sel = build_selection()

    # index old embeddings by normalized key for reuse
    old = {}
    if os.path.isdir(OLD_EMB):
        for fn in os.listdir(OLD_EMB):
            if fn.endswith("_feature.npy"):
                old[_norm(fn[:-len("_feature.npy")])] = os.path.join(OLD_EMB, fn)

    import torch
    from transformers import AutoTokenizer, AutoModel
    from datasets import load_dataset
    import datasets as _hfds
    try:
        _hfds.config.STREAMING_READ_MAX_RETRIES = 2
        _hfds.config.STREAMING_READ_RETRY_INTERVAL = 1
    except Exception:
        pass

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading probe {p4.PROBE} on {dev} ...")
    tok = AutoTokenizer.from_pretrained(p4.PROBE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(p4.PROBE).to(dev).eval()

    status, failures = [], []
    done = 0
    for r in sel.itertuples():
        key = r.dataset_canonical
        out = os.path.join(V2_EMB, p4._sanitize(key) + "_feature.npy")
        if os.path.exists(out):
            status.append((key, r.hf_id, r.n_models, "cached"))
            continue
        nk = _norm(key)
        if nk in old:                           # reuse existing embedding
            shutil.copy(old[nk], out)
            aux = old[nk].replace("_feature.npy", "_aux.npz")
            if os.path.exists(aux):
                shutil.copy(aux, out.replace("_feature.npy", "_aux.npz"))
            status.append((key, r.hf_id, r.n_models, "reused"))
            continue
        if limit is not None and done >= limit:
            break
        cfg = None if (not r.config or str(r.config) == "nan") else str(r.config)
        try:
            res = p4._load_sample_with_timeout(load_dataset, r.hf_id, cfg,
                                               p4.LOAD_TIMEOUT, canon_key=key)
            if res == "TIMEOUT":
                status.append((key, r.hf_id, r.n_models, "skip:timeout"))
                failures.append((key, r.hf_id, r.n_models, r.task, "timeout", "streaming stall"))
                continue
            texts, raw_labels, word_lens, n_fields, n_train_hint = res
            if len(texts) < 8:
                status.append((key, r.hf_id, r.n_models, "skip:no-text"))
                failures.append((key, r.hf_id, r.n_models, r.task, "no-text", f"{len(texts)} texts"))
                continue
            feats = []
            with torch.no_grad():
                for i in range(0, len(texts), 32):
                    batch = tok(texts[i:i + 32], return_tensors="pt", truncation=True,
                                max_length=p4.MAX_LEN, padding=True)
                    oh = model(**{k: v.to(dev) for k, v in batch.items()},
                               output_hidden_states=True)
                    feats.append(oh.hidden_states[-1].mean(dim=1).cpu().numpy())
            arr = np.concatenate(feats, axis=0).astype(np.float32)
            np.save(out, arr)
            try:
                p4._save_aux(key, r.hf_id, raw_labels, word_lens, n_fields, texts, n_train_hint)
            except Exception:
                pass
            status.append((key, r.hf_id, r.n_models, f"ok:{arr.shape[0]}x{arr.shape[1]}"))
            done += 1
            print(f"  [{done}] {key} <- {r.hf_id} n_models={r.n_models} {arr.shape}")
        except Exception as e:
            status.append((key, r.hf_id, r.n_models, f"skip:{type(e).__name__}"))
            failures.append((key, r.hf_id, r.n_models, r.task,
                             type(e).__name__, str(e)[:200]))
        if len(status) % 20 == 0:
            pd.DataFrame(status, columns=["dataset_canonical", "hf_id", "n_models", "status"]) \
              .to_csv(os.path.join(ART, "phase4_embedding_status.csv"), index=False)

    sdf = pd.DataFrame(status, columns=["dataset_canonical", "hf_id", "n_models", "status"])
    sdf.to_csv(os.path.join(ART, "phase4_embedding_status.csv"), index=False)
    fdf = pd.DataFrame(failures, columns=["ranking_group_dataset", "hf_id", "candidate_count",
                                          "task", "error_category", "exception_summary"])
    fdf.to_csv(os.path.join(ART, "dataset_embedding_failures.csv"), index=False)

    ok = sdf["status"].str.startswith(("ok", "cached", "reused"))
    reasons = dict(Counter(s.split(":")[0] for s in sdf["status"]))
    lines = ["# Phase 4 embedding report (effective supervised core, v2)", "",
             f"- effective datasets: **{len(sel)}**",
             f"- embedded (ok+reused+cached): **{int(ok.sum())}**",
             f"- failed: **{int((~ok).sum())}**",
             f"- status breakdown: {reasons}", "",
             "## Failure categories"]
    for k, v in Counter(fdf["error_category"]).most_common():
        lines.append(f"- {k}: {v}")
    with open(os.path.join(ART, "phase4_embed_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[phase4] {int(ok.sum())}/{len(sel)} embedded; failed {int((~ok).sum())}; reasons={reasons}")


if __name__ == "__main__":
    lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    main(limit=lim)
