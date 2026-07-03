"""
cold_dataset_split.py -- Cold-Dataset guide §1: freeze a DATASET-ID split before
training. Deduplicate pairs, pick the evaluable dataset IDs (>=3 distinct observed
candidates, non-constant accuracy, deployable features present), then hold out 20%
as the cold test set, stratified by task type and candidate-count bucket, using a
fixed dataset_split_seed. Cold IDs never depend on any model output.

Writes artifacts/cold_dataset/cold_dataset_split.json.
"""

import hashlib
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.model import load_hgraph  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import dedup_trained_on, TRAINED_ON  # noqa: E402

GRAPH = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage1BuildTransferGraph",
                     "hgraph_hf1000d_2000m_xm0_xd0.pt")
OUT = os.path.join(_HERE, "artifacts", "cold_dataset")
SPLIT_PATH = os.path.join(OUT, "cold_dataset_split.json")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def count_bucket(n):
    return "3-10" if n <= 10 else ("11-100" if n <= 100 else "101+")


def evaluable_datasets(dedup_data):
    """dataset_id -> (candidate_count, task_type) for datasets with >=3 candidates,
    non-constant accuracy, and dataset features present."""
    ei = dedup_data[TRAINED_ON].edge_index
    ea = dedup_data[TRAINED_ON].edge_attr.float()
    tt = dedup_data["dataset"].task_type_id
    by_d = {}
    for m, d, a in zip(ei[0].tolist(), ei[1].tolist(), ea.tolist()):
        by_d.setdefault(int(d), []).append(float(a))
    out = {}
    x = dedup_data["dataset"].x
    for d, accs in by_d.items():
        if len(accs) >= 3 and np.std(accs) > 0 and bool(torch.isfinite(x[d]).all()):
            out[d] = (len(accs), int(tt[d]))
    return out


def build_split(dataset_split_seed=0, cold_frac=0.20, graph=GRAPH):
    data, _xm0, umi = load_hgraph(graph)
    dedup = dedup_trained_on(data)
    ev = evaluable_datasets(dedup)
    names = umi.sort_values("mappedID")["model"].tolist() if hasattr(umi, "sort_values") else None
    dnames = None
    ud = torch.load(graph, map_location="cpu", weights_only=False).get("unique_dataset_id")
    if ud is not None and hasattr(ud, "sort_values"):
        dnames = ud.sort_values("mappedID")["dataset"].tolist()

    # stratify by (task_type, candidate-count bucket); seeded per-stratum sampling
    rng = np.random.default_rng(dataset_split_seed)
    strata = {}
    for d, (n, tt) in ev.items():
        strata.setdefault((tt, count_bucket(n)), []).append(d)
    cold = []
    for key, ids in sorted(strata.items(), key=lambda kv: (str(kv[0]))):
        ids = sorted(ids)
        rng.shuffle(ids)
        n_cold = int(round(cold_frac * len(ids)))
        cold.extend(ids[:n_cold])
    cold = sorted(set(cold))
    remain = sorted(set(ev) - set(cold))

    task_by_part = {"cold": {}, "remaining": {}}
    for part, ids in (("cold", cold), ("remaining", remain)):
        for d in ids:
            tt = ev[d][1]
            task_by_part[part][str(tt)] = task_by_part[part].get(str(tt), 0) + 1

    manifest = {
        "dataset_split_seed": dataset_split_seed,
        "graph_sha256": _sha256(graph),
        "dedup_policy": "dedup_trained_on(reduce=max): 12205 rows -> 7056 distinct "
                        "(model,dataset) pairs; rev_trained_on mirrored",
        "cold_frac": cold_frac,
        "n_evaluable_datasets": len(ev),
        "n_remaining_datasets": len(remain),
        "n_cold_test_datasets": len(cold),
        "cold_test_dataset_ids": cold,
        "cold_test_dataset_names": [dnames[d] if dnames else str(d) for d in cold],
        "remaining_dataset_ids": remain,
        "candidate_count_by_dataset": {str(d): ev[d][0] for d in ev},
        "task_count_by_partition": task_by_part,
        "count_bucket_by_dataset": {str(d): count_bucket(ev[d][0]) for d in ev},
    }
    return manifest


def main():
    os.makedirs(OUT, exist_ok=True)
    m = build_split()
    with open(SPLIT_PATH, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    print(f"evaluable datasets = {m['n_evaluable_datasets']}")
    print(f"remaining datasets = {m['n_remaining_datasets']}")
    print(f"cold test datasets = {m['n_cold_test_datasets']}")
    print(f"cold test dataset IDs = {m['cold_test_dataset_ids']}")
    print(f"task_count_by_partition = {m['task_count_by_partition']}")
    print(f"wrote {SPLIT_PATH}")


if __name__ == "__main__":
    main()
