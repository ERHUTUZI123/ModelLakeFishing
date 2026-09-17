"""
modellens_subsample.py -- P3 D-3 fallback: subsample hgraph_ml_v2.pt to a model
count that fits an 8 GiB GPU, WITHOUT touching the champion training code.

Why: the L1L3b champion uses a full-batch supervised contrastive loss whose
[N_model, N_model] similarity/mask tensors are O(N^2). At 30,183 models that is
~4 x 3.6 GB, impossible on an 8 GiB laptop GPU, and pyg-lib (which would give
bounded neighbor-sampled subgraphs instead of the full-graph fallback) has no
wheel for torch 2.12. PLAN §3.2 sanctions matched-scale root/degree-stratified
subsampling as the fallback -- "对照仍受控,论点不弱". Both systems are
restricted to this same universe in P4, so the head-to-head stays controlled.

Selection (deterministic), keeps the metric well-defined:
  1. keep every gold node's gold + observed top-3 model (eval-critical);
  2. fill by DESCENDING trained_on degree up to --budget (hubs matter most for
     message passing and are the real candidates);
  3. keep every dataset node that still has >=1 supervision edge.

Everything is reindexed; vocabs (family/task) are unchanged (ids still valid).
Output honors the identical Stage-2 contract as the parent graph.

Run (from ModelLakeFishing/):
    .\\.venv\\Scripts\\python.exe -m scale.modellens_subsample --budget 13000
"""

import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd
import torch

_S1 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "stage1BuildTransferGraph")
GRAPH = os.path.join(_S1, "hgraph_ml_v2.pt")
LAKE = os.path.join(_S1, "artifacts", "modellens_v2_lake")
NODE_SEP = "␟"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=12000, help="target model count")
    ap.add_argument("--max-datasets", type=int, default=3000,
                    help="cap dataset nodes (gold always kept); bounds the "
                         "global_positive_density CPU diagnostic (~n_sample x N_d)")
    ap.add_argument("--out", default=os.path.join(_S1, "hgraph_ml_v2_sub.pt"))
    args = ap.parse_args()

    payload = torch.load(GRAPH, map_location="cpu", weights_only=False)
    data = payload["data"]
    umi = payload["unique_model_id"].sort_values("mappedID").reset_index(drop=True)
    udi = payload["unique_dataset_id"].sort_values("mappedID").reset_index(drop=True)
    n_m, n_d = len(umi), len(udi)
    name_of = umi["model"].tolist()
    mid_of = {m: i for i, m in enumerate(name_of)}

    # gold + top-3 models (eval-critical) from intake labels
    obs = pd.read_parquet(os.path.join(LAKE, "ml_observations.parquet"))
    pool = pd.read_csv(os.path.join(LAKE, "ml_dataset_pool.csv"))
    gold_nodes = set(pool[pool["gold_evaluable"]]["dataset_node"])
    og = obs[obs["dataset_node"].isin(gold_nodes)]
    critical = set()
    for _, g in og.groupby("dataset_node"):
        critical.update(g.nlargest(3, "value_norm")["model_id"].tolist())
    critical_ids = {mid_of[m] for m in critical if m in mid_of}
    print(f"[keep] eval-critical (gold+top3) models: {len(critical_ids)}")

    # trained_on degree per model (mappedID order)
    eidx = data["model", "trained_on", "dataset"].edge_index
    deg = torch.bincount(eidx[0], minlength=n_m).numpy()

    budget = min(args.budget, n_m)
    keep = set(critical_ids)
    # fill by descending degree
    for mi in np.argsort(-deg):
        if len(keep) >= budget:
            break
        keep.add(int(mi))
    keep_m = np.array(sorted(keep), dtype=np.int64)
    print(f"[keep] models: {len(keep_m)}/{n_m} (budget {budget})")

    m_new = -np.ones(n_m, dtype=np.int64)
    m_new[keep_m] = np.arange(len(keep_m))

    # dataset nodes retaining >=1 edge among kept models
    em, ed = eidx[0].numpy(), eidx[1].numpy()
    emask = m_new[em] >= 0
    ed_kept = ed[emask]
    with_edge = np.unique(ed_kept)
    # cap datasets: all gold nodes + top by retained-edge-count, up to max
    gold_idx = {i for i, nd in enumerate(udi["dataset"]) if nd in gold_nodes}
    edeg = np.bincount(ed_kept, minlength=n_d)
    keep_dset = set(int(d) for d in with_edge if d in gold_idx)   # gold first
    for di in np.argsort(-edeg):
        if len(keep_dset) >= args.max_datasets:
            break
        if edeg[di] > 0:
            keep_dset.add(int(di))
    keep_d = np.array(sorted(keep_dset), dtype=np.int64)
    d_new = -np.ones(n_d, dtype=np.int64)
    d_new[keep_d] = np.arange(len(keep_d))
    print(f"[keep] dataset nodes: {len(keep_d)}/{n_d} "
          f"(gold {sum(1 for d in keep_d if d in gold_idx)}, cap {args.max_datasets})")

    # ---- rebuild HeteroData -------------------------------------------------
    from torch_geometric.data import HeteroData
    g = HeteroData()
    km = torch.as_tensor(keep_m)
    kd = torch.as_tensor(keep_d)
    g["model"].node_id = torch.arange(len(keep_m))
    g["model"].x = data["model"].x[km]
    g["model"].size_bucket_id = data["model"].size_bucket_id[km]
    g["model"].family_id = data["model"].family_id[km]
    g["dataset"].node_id = torch.arange(len(keep_d))
    g["dataset"].x = data["dataset"].x[kd]
    g["dataset"].task_type_id = data["dataset"].task_type_id[kd]
    g["dataset"].n_class_bucket_id = data["dataset"].n_class_bucket_id[kd]
    g["dataset"].arity_id = data["dataset"].arity_id[kd]

    def remap_edges(et, src_map, dst_map):
        ei = data[et].edge_index.numpy()
        ea = data[et].edge_attr
        m = (src_map[ei[0]] >= 0) & (dst_map[ei[1]] >= 0)
        ni = np.stack([src_map[ei[0][m]], dst_map[ei[1][m]]])
        return torch.as_tensor(ni, dtype=torch.long), ea[torch.as_tensor(m)]

    for et, sm, dm in [
        (("model", "trained_on", "dataset"), m_new, d_new),
        (("dataset", "rev_trained_on", "model"), d_new, m_new),
        (("dataset", "similar_to", "dataset"), d_new, d_new),
        (("model", "is_base_of", "model"), m_new, m_new),
        (("model", "rev_is_base_of", "model"), m_new, m_new),
    ]:
        ei, ea = remap_edges(et, sm, dm)
        g[et].edge_index = ei
        g[et].edge_attr = ea

    # contracts
    assert g["model"].x.shape[0] == len(keep_m)
    assert int(g["model", "trained_on", "dataset"].edge_index[0].max()) < len(keep_m)
    assert len(g.edge_types) == 5

    umi2 = pd.DataFrame({"model": [name_of[i] for i in keep_m],
                         "mappedID": range(len(keep_m))})
    udi2 = udi.iloc[keep_d].copy()
    udi2["mappedID"] = range(len(keep_d))

    out_payload = dict(payload)
    out_payload["data"] = g
    out_payload["unique_model_id"] = umi2
    out_payload["unique_dataset_id"] = udi2.reset_index(drop=True)
    out_payload.setdefault("provenance", {})
    out_payload["provenance"] = {**payload.get("provenance", {}),
                                 "subsampled_from": "hgraph_ml_v2.pt",
                                 "budget": budget,
                                 "selection": "gold+top3 then descending degree",
                                 "n_models": len(keep_m), "n_datasets": len(keep_d)}
    torch.save(out_payload, args.out)
    sha = hashlib.sha256(open(args.out, "rb").read()).hexdigest()

    n_edges = int(g["model", "trained_on", "dataset"].edge_index.shape[1])
    n_gold_kept = sum(1 for nd in udi2["dataset"] if nd in gold_nodes)
    n_lin = int(g["model", "is_base_of", "model"].edge_index.shape[1])
    print(f"\n=== SUBSAMPLE ===")
    print(f"  models {len(keep_m)} | datasets {len(keep_d)} | "
          f"trained_on {n_edges} | similar_to "
          f"{int(g['dataset','similar_to','dataset'].edge_index.shape[1])} | "
          f"lineage {n_lin}")
    print(f"  gold nodes retained: {n_gold_kept}")
    print(f"  sha256 {sha[:16]} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
