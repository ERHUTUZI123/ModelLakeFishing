"""
build_prior_sidecar.py -- v6 serving: emit the sibling/task prior sidecar next to
a Stage-3 export, so the serving re-ranker (serving_rerank.py) can compute the
S2 sibling-prior and P2b task-prior at query time WITHOUT the training graph.

The sidecar carries exactly what the two adopted serving fusions need:
  trained_on : (model_mappedID, dataset_mappedID, norm_acc) lake supervision
  root_id    : per dataset mappedID -> integer root code
  task_id    : per dataset mappedID -> task_type_id
All keyed by the SAME mappedID row order as z_m / z_d in the export (the iron
rule), verified against the export's id snapshots.

Run (repo root):
  ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.stage3HNSW.build_prior_sidecar \
      --graph ModelLakeFishing/stage1BuildTransferGraph/hgraph_d0_v1.pt \
      --export ModelLakeFishing/stage3HNSW/artifacts/exports/d0_L1L3b
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--export", required=True)
    args = ap.parse_args()

    payload = torch.load(args.graph, map_location="cpu", weights_only=False)
    data = payload["data"]
    udi = payload["unique_dataset_id"].sort_values("mappedID").reset_index(drop=True)
    umi = payload["unique_model_id"].sort_values("mappedID").reset_index(drop=True)

    # iron-rule check: sidecar row order must match the export's id snapshots
    exp_d = pd.read_csv(os.path.join(args.export, "dataset_ids.csv"))
    exp_m = pd.read_csv(os.path.join(args.export, "model_ids.csv"))
    assert (exp_d["unique_dataset_id"].tolist() == udi["dataset"].tolist()), \
        "dataset id order != export"
    assert (exp_m["unique_model_id"].tolist() == umi["model"].tolist()), \
        "model id order != export"

    TRAINED_ON = ("model", "trained_on", "dataset")
    ei = data[TRAINED_ON].edge_index.numpy()
    acc = data[TRAINED_ON].edge_attr.numpy().astype("float32")
    roots = udi["dataset"].str.split("/").str[0]
    root_code = {r: i for i, r in enumerate(sorted(set(roots)))}
    root_id = np.array([root_code[r] for r in roots], dtype="int64")
    task_id = data["dataset"].task_type_id.numpy().astype("int64")

    out = os.path.join(args.export, "prior_sidecar.npz")
    np.savez(out, edge_model=ei[0].astype("int64"), edge_dataset=ei[1].astype("int64"),
             edge_acc=acc, root_id=root_id, task_id=task_id)
    meta = {"n_edges": int(ei.shape[1]), "n_models": int(len(umi)),
            "n_datasets": int(len(udi)), "n_roots": len(root_code),
            "n_tasks": int(task_id.max()) + 1,
            "graph": os.path.basename(args.graph)}
    json.dump(meta, open(os.path.join(args.export, "prior_sidecar_meta.json"), "w"), indent=2)
    print(f"sidecar -> {out}"); print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
