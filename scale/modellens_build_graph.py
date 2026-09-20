import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

_S1 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "stage1BuildTransferGraph")
sys.path.insert(0, _S1)
from dataset_embed.xm0_builder import (
    build_name_embeddings, param_count_to_size_bucket,
    load_or_update_family_vocab, build_family_ids, NUM_SIZE_BUCKETS,
    FAMILY_OTHER,
)

LAKE = os.path.join(_S1, "artifacts", "modellens_v2_lake")
OUT_GRAPH = os.path.join(_S1, "hgraph_ml_v2.pt")
OUT_REPORT = os.path.join(LAKE, "ml_graph_report.json")
ENCODER = "all-MiniLM-L6-v2"
SIM_K = 20
TASK_MIN_COUNT = 5
NODE_SEP = "␟"


def _minilm(texts, tag):
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer(ENCODER, device=dev)
    emb = enc.encode(texts, batch_size=256, show_progress_bar=False,
                     convert_to_numpy=True, normalize_embeddings=False)
    print(f"[{tag}] MiniLM({dev}) {len(texts)} texts -> {emb.shape}", flush=True)
    return emb.astype(np.float32)


def model_descriptor(mid, family, size_b):
    name = mid.replace("/", " ").replace("-", " ").replace("_", " ")
    parts = [name]
    if family:
        parts.append(f"family {family}")
    if not np.isnan(size_b):
        parts.append(f"{size_b:g}B params" if size_b >= 1 else
                     f"{size_b*1000:g}M params")
    return " ".join(parts)


def topk_knn(emb, k):
    c = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    n = c.shape[0]
    k = min(k, n - 1)
    src = np.repeat(np.arange(n), k)
    dst = np.empty(n * k, dtype=np.int64)
    wt = np.empty(n * k, dtype=np.float32)
    block = 2048
    for s in range(0, n, block):
        e = min(s + block, n)
        sims = c[s:e] @ c.T
        for i in range(e - s):
            sims[i, s + i] = -1.0
        part = np.argpartition(-sims, k, axis=1)[:, :k]
        rows = np.arange(e - s)[:, None]
        pw = sims[rows, part]
        order = np.argsort(-pw, axis=1)
        nbr = part[rows, order]
        w = pw[rows, order]
        base = s * k
        dst[base:base + (e - s) * k] = nbr.ravel()
        wt[base:base + (e - s) * k] = w.ravel().astype(np.float32)
    return src, dst, wt


def main():
    obs = pd.read_parquet(os.path.join(LAKE, "ml_observations.parquet"))
    pool = pd.read_csv(os.path.join(LAKE, "ml_dataset_pool.csv"))
    mi = pd.read_csv(os.path.join(LAKE, "ml_model_intake.csv"))
    print(f"[load] obs {len(obs):,} | nodes {len(pool):,} | models {len(mi):,}",
          flush=True)

    model_ids = sorted(mi["model_id"].astype(str))
    m_of = {m: i for i, m in enumerate(model_ids)}
    mi = mi.set_index("model_id")

    ds_nodes = sorted(pool["dataset_node"].astype(str))
    d_of = {n: i for i, n in enumerate(ds_nodes)}
    pool = pool.set_index("dataset_node")
    roots = pool.loc[ds_nodes, "root_a"].astype(str).tolist()
    roots_b = pool.loc[ds_nodes, "root_b"].astype(str).tolist()

    obs = obs[obs["model_id"].isin(m_of) & obs["dataset_node"].isin(d_of)]
    src = torch.tensor([m_of[m] for m in obs["model_id"]], dtype=torch.long)
    dst = torch.tensor([d_of[n] for n in obs["dataset_node"]], dtype=torch.long)
    attr = torch.tensor(obs["value_norm"].to_numpy(), dtype=torch.float32)
    print(f"[edges] trained_on {len(src):,}", flush=True)

    lb, ld = [], []
    lin = mi[mi["lineage_base"].notna()]
    for mid, base in lin["lineage_base"].items():
        base = str(base)
        if base in m_of and str(mid) in m_of and base != mid:
            lb.append(m_of[base]); ld.append(m_of[str(mid)])
    print(f"[edges] is_base_of {len(lb):,}", flush=True)

    e_name = build_name_embeddings(model_ids, token_dim=64, seed=42)
    fam_raw = mi.loc[model_ids, "family"].fillna("").astype(str).tolist()
    size_b = mi.loc[model_ids, "size_b"].to_numpy(dtype=float)
    e_desc = _minilm([model_descriptor(m, f, s)
                      for m, f, s in zip(model_ids, fam_raw, size_b)], "xm0.desc")
    xm = np.concatenate([e_name, e_desc], axis=1)

    param_counts = [(b * 1e9 if not np.isnan(b) else None) for b in size_b]
    size_id = torch.tensor([param_count_to_size_bucket(c) for c in param_counts],
                           dtype=torch.long)
    fams = [f if f else FAMILY_OTHER for f in fam_raw]
    family_vocab = load_or_update_family_vocab(fams, vocab_path=None)
    fam_id = torch.as_tensor(build_family_ids(fams, family_vocab), dtype=torch.long)
    print(f"[xm0] frozen {xm.shape} | size unknown "
          f"{float((size_id==0).float().mean()):.2f} | families {len(family_vocab)}",
          flush=True)

    dn_emb = build_name_embeddings(ds_nodes, token_dim=64, seed=43)
    card_texts = []
    for n in ds_nodes:
        d, t = n.split(NODE_SEP) if NODE_SEP in n else (n, "")
        desc = str(pool.loc[n, "desc"]) if pd.notna(pool.loc[n, "desc"]) else ""
        card_texts.append(desc if desc else f"{d} {t}")
    dc_emb = _minilm(card_texts, "xd0.card")

    g = obs.groupby("dataset_node")
    stat = {}
    for n, gr in g:
        v = gr["value_norm"]
        stat[n] = (np.log1p(len(gr)), np.log1p(len(gr)), v.mean(),
                   float(v.std() or 0.0), v.min(), v.max())
    root_sizes = pd.Series(roots).value_counts().to_dict()
    gold_ev = pool.loc[ds_nodes, "gold_evaluable"].astype(bool).tolist()
    xd_stats = np.nan_to_num(np.array(
        [[*stat[n], np.log1p(root_sizes[roots[i]]), float(gold_ev[i]), 0.0, 0.0]
         for i, n in enumerate(ds_nodes)], dtype=np.float32))
    xd = np.concatenate([dn_emb, dc_emb, xd_stats], axis=1)

    node_task = {n: (n.split(NODE_SEP)[1] if NODE_SEP in n else "") for n in ds_nodes}
    tcounts = pd.Series([node_task[n].lower() for n in ds_nodes if node_task[n]]
                        ).value_counts()
    kept = sorted(tcounts[tcounts >= TASK_MIN_COUNT].index)
    task_vocab = {"Other": 0, **{t: i for i, t in enumerate(kept, 1)}}
    tt_id = torch.tensor([task_vocab.get(node_task[n].lower(), 0) for n in ds_nodes],
                         dtype=torch.long)
    print(f"[xd0] frozen {xd.shape} | task vocab {len(task_vocab)} | "
          f"Other share {float((tt_id==0).float().mean()):.2f}", flush=True)

    s_src, s_dst, s_attr = topk_knn(dc_emb, SIM_K)
    print(f"[edges] similar_to {len(s_src):,}", flush=True)

    data = HeteroData()
    data["model"].node_id = torch.arange(len(model_ids))
    data["model"].x = torch.from_numpy(xm)
    data["model"].size_bucket_id = size_id
    data["model"].family_id = fam_id
    data["dataset"].node_id = torch.arange(len(ds_nodes))
    data["dataset"].x = torch.from_numpy(xd)
    data["dataset"].task_type_id = tt_id
    data["dataset"].n_class_bucket_id = torch.zeros(len(ds_nodes), dtype=torch.long)
    data["dataset"].arity_id = torch.zeros(len(ds_nodes), dtype=torch.long)

    data["model", "trained_on", "dataset"].edge_index = torch.stack([src, dst])
    data["model", "trained_on", "dataset"].edge_attr = attr
    data["dataset", "rev_trained_on", "model"].edge_index = torch.stack([dst, src])
    data["dataset", "rev_trained_on", "model"].edge_attr = attr.clone()
    data["dataset", "similar_to", "dataset"].edge_index = torch.tensor(
        np.stack([s_src, s_dst]), dtype=torch.long)
    data["dataset", "similar_to", "dataset"].edge_attr = torch.from_numpy(s_attr)
    li = torch.tensor([lb, ld], dtype=torch.long) if lb else torch.zeros(2, 0, dtype=torch.long)
    la = torch.ones(li.shape[1])
    data["model", "is_base_of", "model"].edge_index = li
    data["model", "is_base_of", "model"].edge_attr = la
    data["model", "rev_is_base_of", "model"].edge_index = li.flip(0)
    data["model", "rev_is_base_of", "model"].edge_attr = la.clone()

    assert data["model"].x.shape == (len(model_ids), 448)
    assert data["dataset"].x.shape == (len(ds_nodes), 458)
    assert not torch.isnan(data["model"].x).any()
    assert not torch.isnan(data["dataset"].x).any()
    assert int(src.max()) < len(model_ids) and int(dst.max()) < len(ds_nodes)
    assert int(fam_id.max()) < len(family_vocab) and family_vocab["Other"] == 0
    assert int(tt_id.max()) < len(task_vocab) and task_vocab["Other"] == 0
    assert attr.dim() == 1 and (attr >= 0).all() and (attr <= 1).all()
    assert len(data.edge_types) == 5

    umi = pd.DataFrame({"model": model_ids, "mappedID": range(len(model_ids))})
    udi = pd.DataFrame({"dataset": ds_nodes, "mappedID": range(len(ds_nodes)),
                        "root": roots, "root_b": roots_b})

    payload = {
        "data": data,
        "xm0_meta": {"num_size_buckets": NUM_SIZE_BUCKETS,
                     "num_families": len(family_vocab),
                     "family_vocab": family_vocab, "name_dim": 64, "desc_dim": 384},
        "xd0_meta": {"num_task_types": len(task_vocab),
                     "task_type_vocab": task_vocab,
                     "n_class_buckets": 7, "num_arities": 5,
                     "arity_vocab": {"unknown": 0, "single": 1, "pair": 2,
                                     "multi": 3, "span_or_qa": 4},
                     "view_dims": {"e_name": 64, "e_card": 384, "e_stats": 10},
                     "encoder_name": ENCODER,
                     "has_content_view": False,
                     "probe_views": "DEFERRED (D0-v1 scope note)"},
        "unique_model_id": umi,
        "unique_dataset_id": udi,
        "provenance": {"corpus": "modellens_v2",
                       "raw": os.path.join(data_root_note(), "modellens_v2"),
                       "node_model": "(dataset, task)",
                       "split_units": {"a": "root (dataset name)",
                                       "b": "root_b (normalized dataset)"}},
    }
    torch.save(payload, OUT_GRAPH)
    sha = hashlib.sha256(open(OUT_GRAPH, "rb").read()).hexdigest()

    n_gold = int(pool.loc[ds_nodes, "gold_evaluable"].astype(bool).sum())
    report = dict(
        graph=os.path.basename(OUT_GRAPH), sha256=sha,
        models=len(model_ids), dataset_nodes=len(ds_nodes),
        roots_a=len(set(roots)), roots_b=len(set(roots_b)),
        trained_on_edges=int(attr.numel()), lineage_edges=len(lb),
        similar_to_edges=int(len(s_src)),
        gold_nodes=n_gold, task_vocab=len(task_vocab),
        task_other_share=float((tt_id == 0).float().mean()),
        size_unknown_share=float((size_id == 0).float().mean()),
        num_families=len(family_vocab),
        model_x_dim=448, dataset_x_dim=458,
    )
    with open(OUT_REPORT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\n=== GRAPH REPORT ===")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"  -> {OUT_GRAPH}")
    return 0


def data_root_note():
    from scale.pull_corpus import data_root
    return data_root()


if __name__ == "__main__":
    sys.exit(main())
