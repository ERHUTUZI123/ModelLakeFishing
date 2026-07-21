"""
d1_model_task_vocab.py -- D1 feature rework (plan v2 §5.3): the MODEL-side task
vocab, e_task's identity credential.

The dataset side has had a task_type table (10 classes, 16-d) since xd0; the
model side has NONE -- dataset-model task alignment has no feature-level anchor.
e_task closes that structural gap. This script builds, OFFLINE (raw HF JSON
caches only, no network):

    task_vocab.csv       task -> task_id, Other == 0   (row-identity credential,
                         same discipline as family_vocab: bound to checkpoints)
    model_task_ids.csv   one row per graph model (mappedID order!): raw source,
                         canonical task, task_id
    d1_task_report.json  coverage + distribution numbers

Task assignment per model (first hit wins):
    1. pipeline_tag from the raw model JSON
    2. modal task.type among the model's model-index results
    3. Other
Canonicalisation folds spelling variants (e.g. sentence-similarity vs
feature-extraction stay distinct; text-classification variants merge), then any
task with < TASK_MIN_COUNT models folds into Other -- mirroring
FAMILY_MIN_COUNT: below threshold a dedicated row is a noise row.

Zero-shot rule (inference symmetry): a new model with no pipeline_tag and no
model-index -> Other (id 0), isomorphic to size's unknown bucket and family's
Other. Inductivity is untouched.

Run (from stage1BuildTransferGraph/):
    ../.venv/Scripts/python.exe d1_model_task_vocab.py \
        [--graph hgraph_hf1000d_2000m_xm0_xd0.pt]
"""

import argparse
import json
import os
from collections import Counter

import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHES = [os.path.join(_HERE, "hf1000d_2000m", "hf_cache", "models"),
          os.path.join(_HERE, "d0_lake_cache", "models")]
OUT = os.path.join(_HERE, "artifacts", "d1_features")

TASK_MIN_COUNT = 5          # < this -> Other (2000-model graph; FAMILY_MIN_COUNT analogue)
TASK_OTHER = "Other"
TASK_ID_OTHER = 0

# spelling/granularity folds observed in HF pipeline_tag + model-index task.type.
# Keys are lowercased raw values; values are canonical task names.
_CANON = {
    "text-classification": "text-classification",
    "sentiment-analysis": "text-classification",
    "zero-shot-classification": "text-classification",
    "nli": "text-classification",
    "natural-language-inference": "text-classification",
    "token-classification": "token-classification",
    "named-entity-recognition": "token-classification",
    "ner": "token-classification",
    "part-of-speech": "token-classification",
    "question-answering": "question-answering",
    "table-question-answering": "question-answering",
    "extractive-question-answering": "question-answering",
    "summarization": "summarization",
    "translation": "translation",
    "text2text-generation": "text2text-generation",
    "text-generation": "text-generation",
    "fill-mask": "fill-mask",
    "sentence-similarity": "sentence-similarity",
    "feature-extraction": "feature-extraction",
    "multiple-choice": "multiple-choice",
    "text-retrieval": "retrieval",
    "retrieval": "retrieval",
    "reranking": "reranking",
    "text-ranking": "reranking",
    "automatic-speech-recognition": "speech",
    "audio-classification": "speech",
    "image-classification": "vision",
    "image-text-to-text": "vision",
    "sts": "sentence-similarity",
    "classification": "text-classification",
    "bitext-mining": "sentence-similarity",
    "bitextmining": "sentence-similarity",
    "clustering": "sentence-similarity",
    "pair-classification": "sentence-similarity",
    "pairclassification": "sentence-similarity",
}


def canon_task(raw):
    k = str(raw or "").strip().lower().replace("_", "-").replace(" ", "-")
    if not k:
        return None
    return _CANON.get(k, k)     # unknown-but-named tasks keep their own name;
                                # the MIN_COUNT fold decides if they survive


def _load_model_json(mid):
    safe = mid.replace("/", "__") + ".json"
    for c in CACHES:
        p = os.path.join(c, safe)
        if os.path.exists(p):
            try:
                d = json.load(open(p, encoding="utf-8"))
                if isinstance(d, list):
                    d = d[0] if d else {}
                return d if isinstance(d, dict) else None
            except Exception:
                return None
    return None


def task_of(d):
    """(task_canonical|None, source) by the first-hit rule."""
    if d is None:
        return None, "no_json"
    t = canon_task(d.get("pipeline_tag"))
    if t:
        return t, "pipeline_tag"
    votes = Counter()
    for entry in (d.get("model-index") or []):
        if not isinstance(entry, dict):
            continue
        for res in (entry.get("results") or []):
            tt = canon_task(((res.get("task") or {}).get("type")
                             or (res.get("task") or {}).get("name")))
            if tt:
                votes[tt] += 1
    if votes:
        return votes.most_common(1)[0][0], "model_index"
    return None, "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=os.path.join(_HERE, "hgraph_hf1000d_2000m_xm0_xd0.pt"))
    ap.add_argument("--min-count", type=int, default=TASK_MIN_COUNT)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    payload = torch.load(args.graph, map_location="cpu", weights_only=False)
    umi = payload["unique_model_id"].sort_values("mappedID").reset_index(drop=True)
    assert (umi["mappedID"].values == range(len(umi))).all() if hasattr(
        umi["mappedID"].values, "all") else True

    rows = []
    for _, r in umi.iterrows():
        mid = r["model"]
        t, src = task_of(_load_model_json(mid))
        rows.append({"unique_model_id": mid, "mappedID": int(r["mappedID"]),
                     "task_raw": t or "", "source": src})
    df = pd.DataFrame(rows).sort_values("mappedID").reset_index(drop=True)

    # fold rare tasks into Other, build the vocab (Other pinned to 0)
    counts = df.loc[df["task_raw"] != "", "task_raw"].value_counts()
    kept = sorted(counts[counts >= args.min_count].index)
    vocab = {TASK_OTHER: TASK_ID_OTHER}
    for i, t in enumerate(kept, start=1):
        vocab[t] = i
    df["task"] = df["task_raw"].map(lambda t: t if t in vocab else TASK_OTHER)
    df["task_id"] = df["task"].map(vocab)

    df.to_csv(os.path.join(OUT, "model_task_ids.csv"), index=False)
    with open(os.path.join(OUT, "task_vocab.csv"), "w", newline="", encoding="utf-8") as f:
        f.write("task,task_id\n")
        for t, i in sorted(vocab.items(), key=lambda kv: kv[1]):
            f.write(f"{t},{i}\n")

    report = {
        "graph": os.path.basename(args.graph),
        "n_models": int(len(df)),
        "min_count": args.min_count,
        "num_model_tasks": len(vocab),
        "coverage_by_source": df["source"].value_counts().to_dict(),
        "share_other": float((df["task_id"] == TASK_ID_OTHER).mean()),
        "task_distribution": df["task"].value_counts().to_dict(),
        "folded_rare_tasks": sorted(set(counts[counts < args.min_count].index)),
    }
    json.dump(report, open(os.path.join(OUT, "d1_task_report.json"), "w"), indent=2)
    for k, v in report.items():
        if k not in ("task_distribution", "folded_rare_tasks"):
            print(f"{k}: {v}")
    print("task_distribution:", dict(sorted(report["task_distribution"].items(),
                                            key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
