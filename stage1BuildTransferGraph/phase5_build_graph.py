"""
phase5_build_graph.py -- Effective-Dataset guide, Phase 5.

Assemble CLEAN, isolated v2 graphs. One builder, two feature-controlled specs so the
A/B differs only in Stage-1 data/provenance/coverage (not architecture or dataset
feature construction):

  B (v2)     = effective datasets (Phase 4 embeddings) + canonical-vault trained_on edges
  A (ctrl)   = old datasets (existing embeddings)      + records.csv edges (NO model_config
               concat; the source-corrected current data, NOT the leaked 12,205-row graph)

Both graphs:
  * reuse the frozen 2,000 model nodes + lineage edges from the shipped graph unchanged;
  * use 768-d gpt-neo dataset centroids (deployable features only; no performance label);
  * carry exact mirror trained_on / rev_trained_on;
  * carry a self-edge-free similar_to top-k=10 dataset graph;
  * dedup canonical (model,dataset) pairs (reduce=mean); per-dataset min-max normalized
    edge weight.

Outputs:
  hgraph_hf_effective_2000m_v2_xm0_xd0.pt   (B)
  hgraph_A_ctrl_2000m_xm0_xd0.pt            (A)
  effective_dataset_v2/phase5_build_report.json
"""

import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from torch_geometric.data import HeteroData

ART = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage2TrainGraphSAGE",
                   "artifacts", "effective_dataset_v2")
SHIPPED = os.path.join(_HERE, "hgraph_hf1000d_2000m_xm0_xd0.pt")
V2_EMB = os.path.join(_HERE, "dataset_embed", "data_hf_effective_2000m_v2",
                      "embedded_dataset", "domain_similarity", "EleutherAI_gpt-neo-125m")
OLD_EMB = os.path.join(_HERE, "dataset_embed", "data_hf1000d_2000m",
                       "embedded_dataset", "domain_similarity", "EleutherAI_gpt-neo-125m")
RECORDS = os.path.join(_HERE, "dataset_embed", "data_hf1000d_2000m", "records.csv")
TOPK = 10


def _sanitize(name):
    return re.sub(r"[^0-9a-zA-Z_.-]", "_", str(name))


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _centroid(emb_dir, name):
    p = os.path.join(emb_dir, _sanitize(name) + "_feature.npy")
    if not os.path.exists(p):
        # try normalized-name match
        for fn in os.listdir(emb_dir):
            if fn.endswith("_feature.npy") and _norm(fn[:-12]) == _norm(name):
                p = os.path.join(emb_dir, fn)
                break
        else:
            return None
    arr = np.load(p).astype(np.float32)
    c = arr.mean(axis=0)
    n = np.linalg.norm(c)
    return (c / n if n > 1e-9 else c).astype(np.float32)


def _similar_to_topk(dfeat, k=TOPK):
    """Directed cosine top-k dataset graph: each dataset points to its k nearest
    neighbours (out-degree exactly min(k, D-1)), self-edge-free. Bounded by design."""
    X = torch.tensor(dfeat)
    X = torch.nn.functional.normalize(X, dim=1)
    S = X @ X.t()
    D = S.size(0)
    S.fill_diagonal_(-1e9)
    kk = min(k, D - 1)
    idx = S.topk(kk, dim=1).indices
    src = torch.arange(D).view(-1, 1).expand(-1, kk).reshape(-1)
    dst = idx.reshape(-1)
    return torch.stack([src, dst])


def build_graph(spec, out_path):
    ship = torch.load(SHIPPED, map_location="cpu", weights_only=False)
    sdata = ship["data"]
    umi = ship["unique_model_id"].sort_values("mappedID").reset_index(drop=True)
    model2id = dict(zip(umi["model"], umi["mappedID"]))

    # ---- dataset nodes (only those with an available embedding) ----------------
    names, feats = [], []
    for ds in spec["datasets"]:
        c = _centroid(spec["emb_dir"], ds)
        if c is not None:
            names.append(ds)
            feats.append(c)
    dfeat = np.stack(feats).astype(np.float32)
    ds2id = {n: i for i, n in enumerate(names)}

    # ---- trained_on edges from the edge table ----------------------------------
    e = spec["edges"].copy()
    e = e[e["model_id"].isin(model2id) & e["dataset"].isin(ds2id)]
    # per-dataset min-max normalize the selected value -> edge weight in [0,1]
    e["value"] = e["value"].astype(float)
    lo = e.groupby("dataset")["value"].transform("min")
    hi = e.groupby("dataset")["value"].transform("max")
    span = (hi - lo)
    e["w"] = np.where(span <= 0, 0.5, (e["value"] - lo) / span.replace(0, np.nan)).astype(float)
    e["w"] = e["w"].fillna(0.5)
    # dedup canonical (model,dataset): reduce=mean (documented, not max)
    e = e.groupby(["model_id", "dataset"], as_index=False).agg(w=("w", "mean"))
    m_idx = e["model_id"].map(model2id).to_numpy()
    d_idx = e["dataset"].map(ds2id).to_numpy()
    ti = torch.tensor(np.stack([m_idx, d_idx]), dtype=torch.long)
    tw = torch.tensor(e["w"].to_numpy(), dtype=torch.float32)   # 1-D, matches shipped convention

    # ---- assemble HeteroData ---------------------------------------------------
    data = HeteroData()
    data["model"].x = sdata["model"].x.clone()
    data["model"].size_bucket_id = sdata["model"].size_bucket_id.clone()
    data["model"].family_id = sdata["model"].family_id.clone()
    data["model"].node_id = torch.arange(data["model"].x.size(0))
    data["dataset"].x = torch.tensor(dfeat)
    data["dataset"].node_id = torch.arange(len(names))
    # discrete task_type_id (0 = Other/unknown) — needed by the G* global-negative
    # loss (ablation.py). n_class_bucket_id/arity_id are only read by the xd0
    # DatasetNodeEncoder (off here) but attached as zeros for contract completeness.
    tmap = spec.get("task_of", {})
    task_vocab = spec.get("task_vocab", {})
    tt = torch.tensor([task_vocab.get(tmap.get(n, "other"), 0) for n in names], dtype=torch.long)
    data["dataset"].task_type_id = tt
    data["dataset"].n_class_bucket_id = torch.zeros(len(names), dtype=torch.long)
    data["dataset"].arity_id = torch.zeros(len(names), dtype=torch.long)

    data["model", "trained_on", "dataset"].edge_index = ti
    data["model", "trained_on", "dataset"].edge_attr = tw
    data["dataset", "rev_trained_on", "model"].edge_index = ti.flip(0)
    data["dataset", "rev_trained_on", "model"].edge_attr = tw.clone()

    sim = _similar_to_topk(dfeat)
    data["dataset", "similar_to", "dataset"].edge_index = sim

    # reuse lineage (model-model) unchanged
    for rel in [("model", "is_base_of", "model"), ("model", "rev_is_base_of", "model")]:
        if rel in sdata.edge_types:
            data[rel].edge_index = sdata[rel].edge_index.clone()
            if "edge_attr" in sdata[rel]:
                data[rel].edge_attr = sdata[rel].edge_attr.clone()

    unique_dataset_id = pd.DataFrame({"dataset": names, "mappedID": range(len(names))})

    payload = {
        "data": data,
        "unique_model_id": ship["unique_model_id"],
        "unique_dataset_id": unique_dataset_id,
        "xm0_meta": ship["xm0_meta"],
        "spec_name": spec["name"],
    }
    torch.save(payload, out_path)

    # audit
    ti_np = ti.numpy()
    distinct_pairs = len({(int(a), int(b)) for a, b in zip(ti_np[0], ti_np[1])})
    by_d = pd.Series(ti_np[1]).value_counts()
    eff3 = int((by_d >= 3).sum())
    mirror_ok = bool(torch.equal(
        data["model", "trained_on", "dataset"].edge_index,
        data["dataset", "rev_trained_on", "model"].edge_index.flip(0)))
    self_loops = int((sim[0] == sim[1]).sum())
    audit = {
        "name": spec["name"], "out": os.path.relpath(out_path, _REPO_ROOT),
        "model_nodes": int(data["model"].x.size(0)),
        "dataset_nodes": len(names),
        "trained_on_edges": int(ti.size(1)),
        "distinct_pairs": distinct_pairs,
        "datasets_with_ge3_edges": eff3,
        "similar_to_edges": int(sim.size(1)),
        "similar_to_self_loops": self_loops,
        "reverse_mirror_exact": mirror_ok,
        "dataset_feat_dim": int(dfeat.shape[1]),
        "graph_sha256": hashlib.sha256(open(out_path, "rb").read()).hexdigest()[:16],
    }
    return audit


def spec_B():
    sel = pd.read_csv(os.path.join(ART, "single_metric_selection.csv"))
    canon = pd.read_parquet(os.path.join(ART, "canonical_performance_observations.parquet"))
    # effective datasets = those in the folds
    eff = []
    for f in range(5):
        eff += json.load(open(os.path.join(ART, "cold_folds", f"fold_{f}.json"),
                              encoding="utf-8"))["cold_dataset_names"]
    eff = set(eff)
    # one selected metric per (dataset) -> pick the highest-coverage group per dataset
    selr = sel[sel["dataset_canonical"].isin(eff)].sort_values(
        "chosen_n_models", ascending=False).drop_duplicates("dataset_canonical")
    gids = set(selr["ranking_group_id"])
    edf = canon[canon["ranking_group_id"].isin(gids)][
        ["model_id_canonical", "dataset_canonical", "value_canonical"]].rename(
        columns={"model_id_canonical": "model_id", "dataset_canonical": "dataset",
                 "value_canonical": "value"})
    task_of = dict(zip(canon["dataset_canonical"], canon["task"]))
    task_vocab = {t: i + 1 for i, t in enumerate(sorted(set(task_of.values())))}
    return {"name": "B_v2", "emb_dir": V2_EMB, "datasets": sorted(eff), "edges": edf,
            "task_of": task_of, "task_vocab": task_vocab}


def spec_A():
    frozen = set(json.load(open(os.path.join(ART, "frozen_models.json"),
                                encoding="utf-8"))["model_ids"])
    rec = pd.read_csv(RECORDS)
    rec = rec[rec["model"].isin(frozen)].copy()
    edf = rec.rename(columns={"model": "model_id", "finetuned_dataset": "dataset",
                              "eval_accuracy": "value"})[["model_id", "dataset", "value"]]
    edf = edf.dropna(subset=["value"])
    task_of = dict(zip(rec["finetuned_dataset"], rec.get("task_type", "other"))) \
        if "task_type" in rec.columns else {}
    task_vocab = {t: i + 1 for i, t in enumerate(sorted(set(str(v) for v in task_of.values())))}
    task_of = {k: str(v) for k, v in task_of.items()}
    return {"name": "A_ctrl", "emb_dir": OLD_EMB,
            "datasets": sorted(edf["dataset"].unique().tolist()), "edges": edf,
            "task_of": task_of, "task_vocab": task_vocab}


def main():
    audits = {}
    audits["B"] = build_graph(spec_B(),
                              os.path.join(_HERE, "hgraph_hf_effective_2000m_v2_xm0_xd0.pt"))
    audits["A"] = build_graph(spec_A(),
                              os.path.join(_HERE, "hgraph_A_ctrl_2000m_xm0_xd0.pt"))
    with open(os.path.join(ART, "phase5_build_report.json"), "w", encoding="utf-8") as f:
        json.dump(audits, f, indent=2)
    print(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
