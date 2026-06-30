"""
graph_surgery.py -- Phase 1: derive dataset-graph variants from a BUILT graph.

The hf1000d/2000m graph was built with the (now-fixed) bug that forced
threshold=1, so its `similar_to` relation is nearly complete (~130k edges,
out-degree ~361) and its edge_attr already holds the full normalized
dataset-dataset similarity. That means the Phase 1 ablations -- remove
`similar_to`, top-k unweighted, weighted top-k -- can be produced by SURGERY on
the loaded graph, identical to what the fixed stage-1 builder (attributes.py
get_dataset_edge_index with top_k) would emit, WITHOUT re-running the heavy
stage-1 embedding pipeline.

Surgery operates ONLY on the dataset-dataset `similar_to` relation (message
structure); it never touches trained_on supervision or lineage. Top-k keeps, per
source dataset, its `k` highest-similarity targets (directed, self excluded),
preserving similarity as edge_attr. A degree assertion fails loudly if a
"sparse" variant comes out nearly complete.
"""

import torch

SIMILAR_TO = ("dataset", "similar_to", "dataset")


def _set_similar_to(data, edge_index, edge_attr):
    store = data[SIMILAR_TO]
    store.edge_index = edge_index
    if edge_attr is not None:
        store.edge_attr = edge_attr
    elif hasattr(store, "edge_attr"):
        del store.edge_attr
    return data


def drop_similar_to(data):
    """B1: remove the `similar_to` relation entirely (empty edge set)."""
    n = data["dataset"].num_nodes
    data = data.clone()
    _set_similar_to(data, torch.empty(2, 0, dtype=torch.long), torch.empty(0))
    return data


def topk_similar_to(data, k, *, weighted=True):
    """B2 (weighted=False) / B3 (weighted=True): keep each dataset's top-k
    highest-similarity neighbours.

    weighted=True keeps similarity in edge_attr; weighted=False sets all kept
    edges to weight 1.0 (topology only). NOTE: until Phase 2 makes the GNN
    edge-aware, weighted and unweighted are numerically identical in message
    passing -- the distinction matters only once edge_attr is consumed.
    """
    data = data.clone()
    n = data["dataset"].num_nodes
    ei = data[SIMILAR_TO].edge_index
    ea = getattr(data[SIMILAR_TO], "edge_attr", None)
    if ea is None:
        ea = torch.ones(ei.size(1))
    # symmetric weight matrix from the existing (symmetrized) edges
    W = torch.full((n, n), float("-inf"))
    W[ei[0], ei[1]] = ea.float()
    W[ei[1], ei[0]] = ea.float()
    W.fill_diagonal_(float("-inf"))               # never keep self edges

    kk = int(min(k, n - 1))
    src, tgt, w = [], [], []
    for i in range(n):
        vals, idx = torch.topk(W[i], kk)
        for v, j in zip(vals.tolist(), idx.tolist()):
            if v != float("-inf"):
                src.append(i); tgt.append(j); w.append(v)
    edge_index = torch.tensor([src, tgt], dtype=torch.long)
    edge_attr = torch.tensor(w, dtype=torch.float) if weighted else torch.ones(len(w))

    # degree assertions (guide Phase 1 step 5)
    if edge_index.numel():
        deg = torch.bincount(edge_index[0], minlength=n)
        assert int(deg.max()) <= kk, f"top-k degree {int(deg.max())} exceeds k={kk}"
        assert int(deg.max()) < n - 1, (
            f"top-k graph nearly complete (deg {int(deg.max())} of {n-1}); check k")
    return _set_similar_to(data, edge_index, edge_attr)


def apply_similar_to_mode(data, mode, *, k=10):
    """Dispatch used by the ablation driver.
      mode='dense' : leave the built (near-complete) graph as-is
      mode='none'  : drop similar_to (B1)
      mode='topk'  : top-k weighted (B3; numerically == B2 until Phase 2)
      mode='topk_unweighted' : top-k topology only (B2)
    """
    if mode == "dense":
        return data
    if mode == "none":
        return drop_similar_to(data)
    if mode == "topk":
        return topk_similar_to(data, k, weighted=True)
    if mode == "topk_unweighted":
        return topk_similar_to(data, k, weighted=False)
    raise ValueError(f"unknown similar_to mode: {mode}")
