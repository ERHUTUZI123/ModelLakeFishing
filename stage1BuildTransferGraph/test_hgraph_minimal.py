"""
Minimal HGraph validation — all raw data lives in dataset_embed/data/.

CSV files
---------
  mini_records.csv          model, dataset, accuracy
  mini_transferability.csv  model, target_dataset, score
  mini_lineage_records.csv  model, relation, base_model
  mini_dataset_features.csv dataset, f0, f1, f2, f3

Run
---
  cd stage1BuildTransferGraph
  python test_hgraph_minimal.py
  -> prints graph summary + writes hgraph_dump.txt
"""

import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
import torch

sys.path.insert(0, os.path.dirname(__file__))
from dataset_embed.utils.graph import HGraph

# ── paths ─────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "dataset_embed", "data")

records_df = pd.read_csv(os.path.join(DATA_DIR, "mini_records.csv"))
tran_df    = pd.read_csv(os.path.join(DATA_DIR, "mini_transferability.csv"))
lineage_df = pd.read_csv(os.path.join(DATA_DIR, "mini_lineage_records.csv"))
feat_df    = pd.read_csv(os.path.join(DATA_DIR, "mini_dataset_features.csv"),
                         index_col="dataset")

# ── 1. Node ID tables (derived from records) ──────────────────────────────────

model_list   = sorted(records_df["model"].unique())
dataset_list = sorted(records_df["dataset"].unique())

unique_model_id = pd.DataFrame({
    "model":    model_list,
    "mappedID": range(len(model_list)),
})
unique_dataset_id = pd.DataFrame({
    "dataset":  dataset_list,
    "mappedID": range(len(dataset_list)),
})

# ── 2. Dataset feature dict ───────────────────────────────────────────────────

feat_cols = list(feat_df.columns)
FEAT_DIM  = len(feat_cols)
dataset_features = {
    ds: feat_df.loc[ds, feat_cols].values.astype(np.float32)
    for ds in dataset_list
}

# ── 3. Edge: model --trained_on--> dataset  (accuracy) ───────────────────────
# normalize per dataset so values are in [0, 1]

accu = records_df.copy()
accu["accuracy"] = accu.groupby("dataset")["accuracy"].transform(
    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
)
accu = (accu
        .merge(unique_model_id, on="model", how="inner")
        .rename(columns={"mappedID": "model_id"})
        .merge(unique_dataset_id, on="dataset", how="inner")
        .rename(columns={"mappedID": "dataset_id"}))

edge_index_accu = torch.tensor([accu["model_id"].values,
                                 accu["dataset_id"].values], dtype=torch.long)
edge_attr_accu  = torch.tensor(accu["accuracy"].values, dtype=torch.float)

# ── 4. Edge: model --transfer_to--> dataset  (transferability) ───────────────

tran = (tran_df
        .merge(unique_model_id, on="model", how="inner")
        .rename(columns={"mappedID": "model_id"})
        .merge(unique_dataset_id,
               left_on="target_dataset", right_on="dataset", how="inner")
        .rename(columns={"mappedID": "dataset_id"}))

edge_index_tran = torch.tensor([tran["model_id"].values,
                                 tran["dataset_id"].values], dtype=torch.long)
edge_attr_tran  = torch.tensor(tran["score"].values, dtype=torch.float)

# ── 5. Edge: dataset --similar_to--> dataset  (cosine sim from features) ──────

names    = unique_dataset_id["dataset"].tolist()
feat_mat = np.array([dataset_features[n] for n in names])
srcs, dsts, sims = [], [], []
for (i, n1), (j, n2) in combinations(enumerate(names), 2):
    similarity = 1.0 - dist.cosine(feat_mat[i], feat_mat[j])
    srcs.append(i); dsts.append(j); sims.append(similarity)

edge_index_dd = torch.tensor([srcs, dsts], dtype=torch.long)
edge_attr_dd  = torch.tensor(sims, dtype=torch.float)

# ── 6. Edge: model --is_base_of--> model  (lineage) ──────────────────────────
# edge_index[0] = base_model (is_base_of)  edge_index[1] = derived_model

RELATION_WEIGHTS = {"quantized": 0.9, "adapter": 0.7, "finetune": 0.5, "merge": 0.3}

lin = (lineage_df
       .merge(unique_model_id.rename(columns={"model": "base_model",
                                               "mappedID": "base_id"}),
              on="base_model", how="inner")
       .merge(unique_model_id.rename(columns={"mappedID": "derived_id"}),
              on="model", how="inner"))
lin["weight"] = lin["relation"].map(RELATION_WEIGHTS).fillna(0.0)

edge_index_mm = torch.tensor([lin["base_id"].values,
                               lin["derived_id"].values], dtype=torch.long)
edge_attr_mm  = torch.tensor(lin["weight"].values, dtype=torch.float)

# ── 7. Build HGraph ───────────────────────────────────────────────────────────

max_dataset_idx = int(unique_dataset_id["mappedID"].max())
model_idx       = unique_model_id["mappedID"].values

graph = HGraph(
    gnn_method="SAGEConv",        # no "without_accuracy" / "without_transfer"
    max_dataset_idx=max_dataset_idx,
    model_idx=model_idx,
    unique_model_id=unique_model_id,
    model_features=[],            # contain_model_feature=False -> random x
    unique_dataset_id=unique_dataset_id,
    dataset_features=dataset_features,
    edge_index_accu_model_to_dataset=edge_index_accu,
    edge_attr_accu_model_to_dataset=edge_attr_accu,
    edge_index_dataset_to_dataset=edge_index_dd,
    edge_attr_dataset_to_dataset=edge_attr_dd,
    edge_index_tran_model_to_dataset=edge_index_tran,
    edge_attr_tran_model_to_dataset=edge_attr_tran,
    edge_index_model_to_model=edge_index_mm,
    edge_attr_model_to_model=edge_attr_mm,
    negative_pairs=None,
    contain_data_similarity=True,
    contain_dataset_feature=False,
    contain_model_feature=False,
)

# ── 8. Dump readable ──────────────────────────────────────────────────────────

model_names   = dict(zip(unique_model_id["mappedID"],   unique_model_id["model"]))
dataset_names = dict(zip(unique_dataset_id["mappedID"], unique_dataset_id["dataset"]))

out_path = os.path.join(os.path.dirname(__file__), "hgraph_dump.txt")
graph.dump_readable(model_names=model_names, dataset_names=dataset_names,
                    out_path=out_path)
