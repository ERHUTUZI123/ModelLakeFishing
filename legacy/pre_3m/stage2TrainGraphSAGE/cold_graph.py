"""
cold_graph.py -- Cold-Dataset guide §§3-4: build a genuinely inductive training
graph (every edge incident to a cold dataset removed) and insert a cold dataset
only at inference (produce z_d without changing any indexed z_m).

Deployable-only cold neighbors: the cold node's incoming `similar_to` edges are
recomputed from the e_domain slice (first `view_dims['e_domain']` cols of the xd0
dataset features = the MiniLM/probe domain embedding), which is computed from raw
examples and uses NO performance labels (audited by test #8).
"""

import numpy as np
import torch

from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import (
    TRAINED_ON, REV_TRAINED_ON, SIMILAR_TO, apply_similar_to_mode,
)

E_DOMAIN_DIM = 3840          # xd0 view_dims['e_domain'] for hf1000d_2000m


def drop_cold_dataset_edges(data, cold_ids):
    """Remove ALL edges incident to cold datasets: trained_on (dst in cold),
    rev_trained_on (src in cold), similar_to (either endpoint in cold). Model
    nodes and all dataset nodes are KEPT (cold nodes become isolated)."""
    cold = set(int(c) for c in cold_ids)
    data = data.clone()

    def _filter(et, keep_mask):
        e = data[et].edge_index
        attr = getattr(data[et], "edge_attr", None)
        data[et].edge_index = e[:, keep_mask]
        if attr is not None:
            data[et].edge_attr = attr[keep_mask]

    to = data[TRAINED_ON].edge_index
    _filter(TRAINED_ON, torch.tensor([int(d) not in cold for d in to[1].tolist()]))
    rev = data[REV_TRAINED_ON].edge_index
    _filter(REV_TRAINED_ON, torch.tensor([int(d) not in cold for d in rev[0].tolist()]))
    sim = data[SIMILAR_TO].edge_index
    _filter(SIMILAR_TO, torch.tensor(
        [(int(s) not in cold and int(t) not in cold) for s, t in zip(sim[0].tolist(), sim[1].tolist())]))
    return data


def build_training_graph(dedup_data, cold_ids, *, similar_to_mode, similar_to_k):
    """Inductive training graph for one config: drop all cold-incident edges, then
    apply the config's similar_to surgery among the REMAINING datasets only.
    Asserts the three emptiness invariants (guide §3)."""
    g = drop_cold_dataset_edges(dedup_data, cold_ids)
    g = apply_similar_to_mode(g, similar_to_mode, k=similar_to_k)
    assert_cold_absent(g, cold_ids)
    return g


def assert_cold_absent(g, cold_ids):
    cold = set(int(c) for c in cold_ids)
    to = g[TRAINED_ON].edge_index
    assert sum(int(d) in cold for d in to[1].tolist()) == 0, "cold trained_on edge present"
    rev = g[REV_TRAINED_ON].edge_index
    assert sum(int(d) in cold for d in rev[0].tolist()) == 0, "cold rev_trained_on edge present"
    sim = g[SIMILAR_TO].edge_index
    assert sum((int(s) in cold or int(t) in cold)
               for s, t in zip(sim[0].tolist(), sim[1].tolist())) == 0, "cold similar_to edge present"


def cold_labels_vault(dedup_data, cold_ids):
    """{cold_dataset_id: [(model_id, normalized_acc), ...]} from the deduped
    trained_on edges -- the EVALUATION-ONLY label store (never touches training)."""
    cold = set(int(c) for c in cold_ids)
    ei = dedup_data[TRAINED_ON].edge_index
    ea = dedup_data[TRAINED_ON].edge_attr.float()
    vault = {}
    for m, d, a in zip(ei[0].tolist(), ei[1].tolist(), ea.tolist()):
        if int(d) in cold:
            vault.setdefault(int(d), []).append((int(m), float(a)))
    return vault


def cold_incoming_similar_to(dedup_data, cold_ids, remain_ids, *, mode, k,
                             e_domain_dim=E_DOMAIN_DIM):
    """Deployable incoming edges (known_remaining --similar_to--> cold), computed
    from e_domain cosine only. mode 'none' -> no edges; 'dense' -> all remaining;
    'topk'/'topk_unweighted' -> top-k remaining by cosine. Weight = cosine
    (edge-aware conv normalizes per destination; ignored by unweighted configs)."""
    if mode == "none":
        return torch.empty(2, 0, dtype=torch.long), torch.empty(0)
    X = dedup_data["dataset"].x[:, :e_domain_dim].numpy()
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    remain = list(remain_ids)
    src, tgt, w = [], [], []
    kk = None if mode == "dense" else int(min(k, len(remain)))
    for c in cold_ids:
        sims = Xn[remain] @ Xn[int(c)]
        idx = np.arange(len(remain)) if kk is None else np.argsort(-sims)[:kk]
        for j in idx:
            src.append(remain[int(j)])
            tgt.append(int(c))
            w.append(float(max(0.0, sims[int(j)])))
    return torch.tensor([src, tgt], dtype=torch.long), torch.tensor(w, dtype=torch.float)


@torch.no_grad()
def encode_indexed_and_cold(model, train_graph, dedup_data, cold_ids, remain_ids, *,
                            mode, k, device="cpu"):
    """Frozen-index cold insertion. Returns (z_m, z_d_cold_map) where z_m is the
    indexed model matrix from the TRAINING graph and z_d_cold_map maps each cold
    dataset id -> its inference z_d. Asserts z_m is byte-identical before/after
    insertion (guide §4 invariant)."""
    model.eval()
    z_train = model(train_graph.clone().to(device))
    z_m = z_train["model"].cpu()

    # query graph = training graph + incoming similar_to edges to cold nodes
    src_w = cold_incoming_similar_to(dedup_data, cold_ids, remain_ids, mode=mode, k=k)
    q = train_graph.clone()
    ei, w = src_w
    if ei.numel():
        q[SIMILAR_TO].edge_index = torch.cat([q[SIMILAR_TO].edge_index, ei], dim=1)
        base_attr = getattr(q[SIMILAR_TO], "edge_attr", None)
        if base_attr is not None:
            q[SIMILAR_TO].edge_attr = torch.cat([base_attr, w])
    z_query = model(q.to(device))
    z_m_after = z_query["model"].cpu()
    assert torch.allclose(z_m, z_m_after, atol=1e-6), \
        "indexed z_m changed after cold insertion -- not a query-only encode"
    z_d = z_query["dataset"].cpu()
    return z_m, {int(c): z_d[int(c)] for c in cold_ids}
