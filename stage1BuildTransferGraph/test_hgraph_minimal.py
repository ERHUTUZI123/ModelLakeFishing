"""
Minimal HGraph validation — all raw data lives in dataset_embed/data/.

CSV files
---------
  mini_records.csv          model, dataset, accuracy
  mini_transferability.csv  model, target_dataset, score
  mini_lineage_records.csv  model, relation, base_model
  mini_dataset_features.csv dataset, f0, f1, f2, f3

Offline x_m^(0) fixtures (written by this script — synthetic models do not
exist on HuggingFace, so the raw-data caches are pre-seeded and the builders
never touch the network or load sentence-transformers):
  mini_desc_cache.csv       model, description       (README text cache)
  mini_desc_emb_cache.npz   model_names, embeddings  (desc embedding cache)
  mini_size_cache.csv       model, param_count       (NaN -> unknown bucket)
  mini_family_vocab.csv     family, family_id        (append-only vocab)

Validates (plan.md step 6): row-order contract, frozen concat dims, no NaN,
size/family index ranges, HGraph integration of x + index columns, and that
gradients reach ModelNodeEncoder's learnable tables but not the frozen x.

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
from dataset_embed.xm0_builder import build_xm0, build_name_embeddings
from dataset_embed.TRAINING_model_node_encoder import ModelNodeEncoder

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

# ── 7. x_m^(0) node features: seed offline caches, build, sanity-check ───────

DESC_DIM = 32
mini_descs = {
    "resnet18":    "ResNet-18 trained on ImageNet. Baseline vision backbone.",
    "resnet18_ft": "ResNet-18 fine-tuned on flowers and cifar10.",
    "resnet18_q":  "Quantized adapter variant of resnet18_ft.",
}
pd.DataFrame(
    {"model": list(mini_descs), "description": list(mini_descs.values())}
).to_csv(os.path.join(DATA_DIR, "mini_desc_cache.csv"), index=False)

rng = np.random.default_rng(7)
mini_desc_embs = {m: rng.standard_normal(DESC_DIM).astype(np.float32)
                  for m in mini_descs}
# stored in REVERSED name order on purpose: cache rows are keyed by model
# name, so the builder must realign them to mappedID order — the row-order
# spot check below catches it if alignment ever silently breaks
rev = list(reversed(list(mini_desc_embs)))
np.savez(os.path.join(DATA_DIR, "mini_desc_emb_cache.npz"),
         model_names=np.array(rev),
         embeddings=np.stack([mini_desc_embs[m] for m in rev]))

pd.DataFrame({
    "model":       ["resnet18", "resnet18_ft", "resnet18_q"],
    "param_count": [11689512,   11689512,      np.nan],  # NaN -> unknown bucket
}).to_csv(os.path.join(DATA_DIR, "mini_size_cache.csv"), index=False)

xm0 = build_xm0(
    unique_model_id,
    token_dim=64,
    desc_cache_path=os.path.join(DATA_DIR, "mini_desc_cache.csv"),
    desc_emb_cache_path=os.path.join(DATA_DIR, "mini_desc_emb_cache.npz"),
    size_cache_path=os.path.join(DATA_DIR, "mini_size_cache.csv"),
    family_vocab_path=os.path.join(DATA_DIR, "mini_family_vocab.csv"),
)

N = len(unique_model_id)
assert xm0["frozen"].shape == (N, 64 + DESC_DIM)
assert not np.isnan(xm0["frozen"]).any()

# row-order contract spot check: row 0 must be mappedID 0's name || desc
e_name_0 = build_name_embeddings([model_list[0]], token_dim=64)[0]
assert np.allclose(xm0["frozen"][0, :xm0["name_dim"]], e_name_0)
assert np.allclose(xm0["frozen"][0, xm0["name_dim"]:], mini_desc_embs[model_list[0]])

# 11.7M params -> log10 = 7.07 -> bucket 5; NaN -> unknown bucket 0
assert xm0["size_bucket_id"].tolist() == [5, 5, 0], xm0["size_bucket_id"]
fam_resnet = xm0["family_vocab"]["ResNet"]
assert xm0["family_id"].tolist() == [fam_resnet] * N, xm0["family_id"]

print(f"\n[xm0] frozen: {xm0['frozen'].shape}  "
      f"(name {xm0['name_dim']} || desc {xm0['desc_dim']})")
print(f"[xm0] size_bucket_id: {xm0['size_bucket_id'].tolist()}  "
      f"family_id: {xm0['family_id'].tolist()} (ResNet={fam_resnet})")
print("[xm0] sanity checks passed")

# ── 8. Build HGraph ───────────────────────────────────────────────────────────

max_dataset_idx = int(unique_dataset_id["mappedID"].max())
model_idx       = unique_model_id["mappedID"].values

graph = HGraph(
    gnn_method="SAGEConv",        # no "without_accuracy" / "without_transfer"
    max_dataset_idx=max_dataset_idx,
    model_idx=model_idx,
    unique_model_id=unique_model_id,
    model_features=xm0["frozen"],          # frozen half -> data['model'].x
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
    contain_model_feature=True,
    model_size_bucket_id=xm0["size_bucket_id"],   # learnable half: indices only
    model_family_id=xm0["family_id"],
)

assert graph.data["model"].x.shape == (N, xm0["frozen"].shape[1])
assert torch.equal(graph.data["model"].size_bucket_id,
                   torch.from_numpy(xm0["size_bucket_id"]))
assert torch.equal(graph.data["model"].family_id,
                   torch.from_numpy(xm0["family_id"]))

# ── 9. ModelNodeEncoder: training-time concat completing x_m^(0) ─────────────

torch.manual_seed(0)
encoder = ModelNodeEncoder(
    frozen_dim=xm0["frozen"].shape[1],
    num_size_buckets=xm0["num_size_buckets"],
    num_families=xm0["num_families"],
    size_dim=16,
    family_dim=16,
)
x_m0 = encoder(
    graph.data["model"].x,
    graph.data["model"].size_bucket_id,
    graph.data["model"].family_id,
)
assert x_m0.shape == (N, encoder.out_dim)

# frozen/learnable separation: gradients must reach the two embedding tables,
# while the graph's x stays a plain non-learnable tensor
x_m0.sum().backward()
assert encoder.size_embedding.weight.grad.abs().sum() > 0
assert encoder.family_embedding.weight.grad.abs().sum() > 0
assert not graph.data["model"].x.requires_grad

print(f"[encoder] x_m^(0): {tuple(x_m0.shape)}  "
      f"(frozen {encoder.frozen_dim} || size 16 || family 16)")
print("[encoder] gradient flow checks passed")

# ── 10. Dump readable ─────────────────────────────────────────────────────────

model_names   = dict(zip(unique_model_id["mappedID"],   unique_model_id["model"]))
dataset_names = dict(zip(unique_dataset_id["mappedID"], unique_dataset_id["dataset"]))

out_path = os.path.join(os.path.dirname(__file__), "hgraph_dump.txt")
graph.dump_readable(model_names=model_names, dataset_names=dataset_names,
                    out_path=out_path)
