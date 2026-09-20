import torch

SIMILAR_TO = ("dataset", "similar_to", "dataset")
TRAINED_ON = ("model", "trained_on", "dataset")
REV_TRAINED_ON = ("dataset", "rev_trained_on", "model")


def dedup_trained_on(data, *, reduce="max", verbose=False):
    assert reduce in ("max", "mean")
    data = data.clone()
    ei = data[TRAINED_ON].edge_index
    ea = data[TRAINED_ON].edge_attr.float()
    key = ei[0].to(torch.int64) * (int(ei[1].max()) + 1) + ei[1].to(torch.int64)
    uniq, inv = torch.unique(key, return_inverse=True)
    n = uniq.numel()
    if reduce == "max":
        val = torch.full((n,), float("-inf"))
        val.scatter_reduce_(0, inv, ea, reduce="amax")
    else:
        s = torch.zeros(n).scatter_add_(0, inv, ea)
        c = torch.zeros(n).scatter_add_(0, inv, torch.ones_like(ea))
        val = s / c
    base = int(ei[1].max()) + 1
    m = (uniq // base).to(torch.long)
    d = (uniq % base).to(torch.long)
    new_ei = torch.stack([m, d])
    data[TRAINED_ON].edge_index = new_ei
    data[TRAINED_ON].edge_attr = val
    data[REV_TRAINED_ON].edge_index = torch.stack([d, m])
    data[REV_TRAINED_ON].edge_attr = val.clone()
    if verbose:
        print(f"dedup_trained_on[{reduce}]: {ei.size(1)} rows -> {n} distinct pairs")
    assert new_ei.size(1) == n
    return data


def _set_similar_to(data, edge_index, edge_attr):
    store = data[SIMILAR_TO]
    store.edge_index = edge_index
    if edge_attr is not None:
        store.edge_attr = edge_attr
    elif hasattr(store, "edge_attr"):
        del store.edge_attr
    return data


def drop_similar_to(data):
    n = data["dataset"].num_nodes
    data = data.clone()
    _set_similar_to(data, torch.empty(2, 0, dtype=torch.long), torch.empty(0))
    return data


def topk_similar_to(data, k, *, weighted=True):
    data = data.clone()
    n = data["dataset"].num_nodes
    ei = data[SIMILAR_TO].edge_index
    ea = getattr(data[SIMILAR_TO], "edge_attr", None)
    if ea is None:
        ea = torch.ones(ei.size(1))
    W = torch.full((n, n), float("-inf"))
    W[ei[0], ei[1]] = ea.float()
    W[ei[1], ei[0]] = ea.float()
    W.fill_diagonal_(float("-inf"))

    kk = int(min(k, n - 1))
    src, tgt, w = [], [], []
    for i in range(n):
        vals, idx = torch.topk(W[i], kk)
        for v, j in zip(vals.tolist(), idx.tolist()):
            if v != float("-inf"):
                src.append(i); tgt.append(j); w.append(v)
    edge_index = torch.tensor([src, tgt], dtype=torch.long)
    edge_attr = torch.tensor(w, dtype=torch.float) if weighted else torch.ones(len(w))

    if edge_index.numel():
        deg = torch.bincount(edge_index[0], minlength=n)
        assert int(deg.max()) <= kk, f"top-k degree {int(deg.max())} exceeds k={kk}"
        assert int(deg.max()) < n - 1, (
            f"top-k graph nearly complete (deg {int(deg.max())} of {n-1}); check k")
    return _set_similar_to(data, edge_index, edge_attr)


def apply_similar_to_mode(data, mode, *, k=10):
    if mode == "dense":
        return data
    if mode == "none":
        return drop_similar_to(data)
    if mode == "topk":
        return topk_similar_to(data, k, weighted=True)
    if mode == "topk_unweighted":
        return topk_similar_to(data, k, weighted=False)
    raise ValueError(f"unknown similar_to mode: {mode}")


def degree_cap_trained_on(data, *, quantile=0.95):
    out = data.clone()
    ei = out[TRAINED_ON].edge_index
    ea = out[TRAINED_ON].edge_attr.float().flatten()
    deg = torch.bincount(ei[0], minlength=int(ei[0].max()) + 1 if ei.numel() else 1)
    lab = deg[deg > 0].float()
    tau = int(torch.quantile(lab, quantile).ceil()) if lab.numel() else 0
    keep = torch.ones(ei.shape[1], dtype=torch.bool)
    n_capped_models = 0
    for m in (deg > tau).nonzero().flatten().tolist():
        idx = (ei[0] == m).nonzero().flatten()
        order = idx[torch.argsort(-ea[idx], stable=True)]
        keep[order[tau:]] = False
        n_capped_models += 1
    out[TRAINED_ON].edge_index = ei[:, keep]
    out[TRAINED_ON].edge_attr = data[TRAINED_ON].edge_attr[keep]
    out[REV_TRAINED_ON].edge_index = out[TRAINED_ON].edge_index.flip(0)
    out[REV_TRAINED_ON].edge_attr = out[TRAINED_ON].edge_attr.clone()
    stats = {"tau": tau, "n_capped_models": n_capped_models,
             "edges_before": int(ei.shape[1]), "edges_after": int(keep.sum())}
    return out, stats
