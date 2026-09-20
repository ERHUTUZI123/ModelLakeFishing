import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_STAGE1 = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage1BuildTransferGraph")
for _p in (_REPO_ROOT, _STAGE1):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ModelLakeFishing.stage1BuildTransferGraph.dataset_embed.utils.CustomRandomLinkSplit import RandomLinkSplit


class _RandomLinkSplit(RandomLinkSplit):

    def forward(self, data):
        return RandomLinkSplit.__call__(self, data)


TRAINED_ON = ("model", "trained_on", "dataset")
REV_TRAINED_ON = ("dataset", "rev_trained_on", "model")
IS_BASE_OF = ("model", "is_base_of", "model")


class PerfScorer(nn.Module):

    def __init__(self, dim: int, mode: str = "dot", hidden: int = 128):
        super().__init__()
        assert mode in ("dot", "mlp")
        self.mode = mode
        if mode == "dot":
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.bias = nn.Parameter(torch.tensor(0.0))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(2 * dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
            )

    def forward(self, z_model, z_dataset, edge_label_index):
        zm = z_model[edge_label_index[0]]
        zd = z_dataset[edge_label_index[1]]
        if self.mode == "dot":
            s = (zm * zd).sum(dim=-1)
            return torch.sigmoid(self.scale * s + self.bias)
        return torch.sigmoid(self.mlp(torch.cat([zm, zd], dim=-1)).squeeze(-1))


def accuracy_lookup(data) -> dict:
    ei = data[TRAINED_ON].edge_index
    ea = data[TRAINED_ON].edge_attr.float()
    return {(int(s), int(d)): float(a) for s, d, a in zip(ei[0], ei[1], ea)}


def perf_supervision(split_store, lookup: dict):
    eli = split_store.edge_label_index
    pos = split_store.edge_label == 1
    eli_pos = eli[:, pos]
    target = torch.tensor(
        [lookup[(int(eli_pos[0, k]), int(eli_pos[1, k]))] for k in range(eli_pos.size(1))],
        dtype=torch.float32,
    )
    return eli_pos, target


def perf_loss(scorer, z_dict, edge_label_index, target) -> torch.Tensor:
    pred = scorer(z_dict["model"], z_dict["dataset"], edge_label_index)
    return F.mse_loss(pred, target.to(pred.device))


def perf_ranking_loss(scorer, z_dict, edge_label_index, edge_label, *,
                      margin: float = 0.05, max_pairs_per_dataset: int = 256,
                      min_models_per_dataset: int = 2, min_gap: float = 1e-3,
                      generator=None, return_stats: bool = False):
    z_m, z_d = z_dict["model"], z_dict["dataset"]
    src, dst, acc = edge_label_index[0], edge_label_index[1], edge_label.float()
    hi, lo = [], []
    n_datasets = 0
    for d in torch.unique(dst).tolist():
        sel = (dst == d).nonzero().flatten()
        n = int(sel.numel())
        if n < min_models_per_dataset:
            continue
        a = acc[sel]
        cap = min(max_pairs_per_dataset, n * (n - 1))
        p = torch.randint(n, (cap,), generator=generator, device=z_m.device)
        q = torch.randint(n, (cap,), generator=generator, device=z_m.device)
        good = a[p] > a[q] + min_gap
        if bool(good.any()):
            hi.append(sel[p[good]])
            lo.append(sel[q[good]])
            n_datasets += 1
    if not hi:
        loss = z_m.new_zeros(())
        return (loss, {"n_pairs": 0, "n_datasets": 0}) if return_stats else loss
    hi, lo = torch.cat(hi), torch.cat(lo)
    s_hi = scorer(z_m, z_d, torch.stack([src[hi], dst[hi]]))
    s_lo = scorer(z_m, z_d, torch.stack([src[lo], dst[lo]]))
    loss = torch.relu(margin - (s_hi - s_lo)).mean()
    if return_stats:
        return loss, {"n_pairs": int(hi.numel()), "n_datasets": n_datasets}
    return loss


def raw_dot_ranknet_loss(z_dict, edge_label_index, edge_label, *,
                         temperature: float = 0.1, min_gap: float = 0.0,
                         max_pairs_per_dataset: int = 256, min_models_per_dataset: int = 2,
                         gap_weighted: bool = False, hard_frac: float = 0.5,
                         generator=None, return_stats: bool = False):
    assert temperature > 0, "temperature must be positive"
    z_m, z_d = z_dict["model"], z_dict["dataset"]
    src, dst, acc = edge_label_index[0], edge_label_index[1], edge_label.float()
    s_all = (z_m[src] * z_d[dst]).sum(-1)

    losses, n_pairs, n_datasets = [], 0, 0
    for d in torch.unique(dst).tolist():
        sel = (dst == d).nonzero().flatten()
        n = int(sel.numel())
        if n < min_models_per_dataset:
            continue
        a = acc[sel]
        s = s_all[sel]
        ai, aj = a.unsqueeze(1), a.unsqueeze(0)
        better = (ai - aj) > min_gap
        ii, jj = better.nonzero(as_tuple=True)
        if ii.numel() == 0:
            continue
        if ii.numel() > max_pairs_per_dataset:
            margin = (s[ii] - s[jj]).detach()
            n_hard = int(round(hard_frac * max_pairs_per_dataset))
            order = torch.argsort(margin)
            hard = order[:n_hard]
            rest = order[n_hard:]
            if rest.numel() > 0:
                perm = torch.randperm(rest.numel(), generator=generator, device=rest.device)
                rand = rest[perm[:max_pairs_per_dataset - n_hard]]
                keep = torch.cat([hard, rand])
            else:
                keep = hard
            ii, jj = ii[keep], jj[keep]
        diff = s[ii] - s[jj]
        pair_loss = F.softplus(-diff / temperature)
        if gap_weighted:
            w = (a[ii] - a[jj]).clamp(max=1.0)
            pair_loss = pair_loss * w
        losses.append(pair_loss.mean())
        n_pairs += int(ii.numel())
        n_datasets += 1

    if not losses:
        loss = z_m.new_zeros(())
        return (loss, {"n_pairs": 0, "n_datasets": 0}) if return_stats else loss
    loss = torch.stack(losses).mean()
    if return_stats:
        return loss, {"n_pairs": n_pairs, "n_datasets": n_datasets}
    return loss


def split_trained_on(data, *, num_val=0.1, num_test=0.2, neg_ratio=1.0,
                     disjoint_train_ratio=0.3, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    transform = _RandomLinkSplit(
        num_val=num_val,
        num_test=num_test,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=neg_ratio,
        disjoint_train_ratio=disjoint_train_ratio,
        edge_types=TRAINED_ON,
        rev_edge_types=REV_TRAINED_ON,
        custom_negative_sampling=False,
        negative_pairs=[],
    )
    return transform(data)


def _union_find_components(edge_index, num_nodes):
    parent = list(range(num_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        rs, rd = find(s), find(d)
        if rs != rd:
            parent[rd] = rs
    return torch.tensor([find(i) for i in range(num_nodes)], dtype=torch.long)


def _hub_mask(data, num_models):
    comp = _union_find_components(data[IS_BASE_OF].edge_index, num_models)
    hub_mask = comp.unsqueeze(0) == comp.unsqueeze(1)
    hub_mask.fill_diagonal_(False)
    counts = torch.bincount(comp, minlength=num_models)
    singleton = counts[comp] <= 1
    hub_mask[singleton, :] = False
    hub_mask[:, singleton] = False
    return hub_mask


def _resolve_trained_on(data, trained_on_index, trained_on_attr):
    if trained_on_index is None:
        return data[TRAINED_ON].edge_index, data[TRAINED_ON].edge_attr.float()
    return trained_on_index, trained_on_attr.float()


def contrastive_supervision(data, *, acc_thresh: float = 0.8,
                            trained_on_index=None, trained_on_attr=None):
    num_models = data["model"].num_nodes
    num_datasets = data["dataset"].num_nodes
    idx, attr = _resolve_trained_on(data, trained_on_index, trained_on_attr)

    M = torch.zeros(num_models, num_datasets)
    keep = attr >= acc_thresh
    M[idx[0][keep], idx[1][keep]] = 1.0
    pos_mask = (M @ M.t()) > 0
    pos_mask.fill_diagonal_(False)
    return pos_mask, _hub_mask(data, num_models)


class SparseMembership:

    def __init__(self, rows, cols, shape, dtype=torch.float32):
        rows = torch.as_tensor(rows, dtype=torch.long)
        cols = torch.as_tensor(cols, dtype=torch.long)
        assert rows.numel() == cols.numel()
        self.shape = (int(shape[0]), int(shape[1]))
        self.dtype = dtype
        n_m, n_d = self.shape
        if rows.numel():
            keys = torch.unique(rows * n_d + cols)
            rows, cols = keys // n_d, keys % n_d
        o = torch.argsort(cols * n_m + rows)
        self.col_rows = rows[o].contiguous()
        self.col_ptr = torch.zeros(n_d + 1, dtype=torch.long)
        if cols.numel():
            self.col_ptr[1:] = torch.cumsum(torch.bincount(cols[o], minlength=n_d), 0)
        o = torch.argsort(rows * n_d + cols)
        self.row_cols = cols[o].contiguous()
        self.row_ptr = torch.zeros(n_m + 1, dtype=torch.long)
        if rows.numel():
            self.row_ptr[1:] = torch.cumsum(torch.bincount(rows[o], minlength=n_m), 0)

    @classmethod
    def from_dense(cls, M):
        rows, cols = (M > 0).nonzero(as_tuple=True)
        return cls(rows.cpu(), cols.cpu(), M.shape, dtype=M.dtype)

    @property
    def device(self):
        return self.col_rows.device

    @property
    def nnz(self):
        return int(self.col_rows.numel())

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]

    def to(self, device):
        for k in ("col_rows", "col_ptr", "row_cols", "row_ptr"):
            setattr(self, k, getattr(self, k).to(device))
        return self

    def float(self):
        self.dtype = torch.float32
        return self

    def to_dense(self):
        out = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        rows = torch.repeat_interleave(
            torch.arange(self.shape[0], device=self.device),
            self.row_ptr[1:] - self.row_ptr[:-1])
        out[rows, self.row_cols] = 1
        return out

    def col_ids(self, d):
        return self.col_rows[self.col_ptr[d]:self.col_ptr[d + 1]]

    def col_dense(self, d):
        v = torch.zeros(self.shape[0], dtype=self.dtype, device=self.device)
        v[self.col_ids(d)] = 1
        return v

    def row_ids(self, m):
        return self.row_cols[self.row_ptr[m]:self.row_ptr[m + 1]]

    def rows_dense(self, n_id):
        n_id = torch.as_tensor(n_id, dtype=torch.long, device=self.device)
        starts, ends = self.row_ptr[n_id], self.row_ptr[n_id + 1]
        counts = ends - starts
        out = torch.zeros((n_id.numel(), self.shape[1]),
                          dtype=self.dtype, device=self.device)
        total = int(counts.sum())
        if total == 0:
            return out
        local = torch.repeat_interleave(
            torch.arange(n_id.numel(), device=self.device), counts)
        offs = (torch.arange(total, device=self.device)
                - torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
                + torch.repeat_interleave(starts, counts))
        out[local, self.row_cols[offs]] = 1
        return out

    def rows_coo(self, n_id):
        n_id = torch.as_tensor(n_id, dtype=torch.long, device=self.device)
        starts, ends = self.row_ptr[n_id], self.row_ptr[n_id + 1]
        counts = ends - starts
        total = int(counts.sum())
        if total == 0:
            empty = torch.zeros(0, dtype=torch.long, device=self.device)
            return empty, empty
        local = torch.repeat_interleave(
            torch.arange(n_id.numel(), device=self.device), counts)
        offs = (torch.arange(total, device=self.device)
                - torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
                + torch.repeat_interleave(starts, counts))
        return local, self.row_cols[offs]

    def sum(self, dim=None):
        if dim in (0, -2):
            return (self.col_ptr[1:] - self.col_ptr[:-1]).to(self.dtype)
        if dim in (1, -1):
            return (self.row_ptr[1:] - self.row_ptr[:-1]).to(self.dtype)
        return torch.tensor(float(self.nnz), dtype=self.dtype, device=self.device)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            r, c = key
            if isinstance(r, slice) and r == slice(None):
                return self.col_dense(int(c))
            if isinstance(r, int) and isinstance(c, int):
                col = self.col_ids(c)
                return (col == r).any().to(self.dtype)
            return self.col_dense(int(c))[torch.as_tensor(r, dtype=torch.long)]
        return self.rows_dense(key)

    def __repr__(self):
        return (f"SparseMembership(shape={self.shape}, nnz={self.nnz}, "
                f"density={self.nnz / max(1, self.shape[0] * self.shape[1]):.2e})")


def high_performer_membership(trained_on_index, trained_on_attr,
                              num_models, num_datasets, *, top_frac=0.1, top_k=None,
                              sparse=False):
    src, dst, acc = trained_on_index[0], trained_on_index[1], trained_on_attr.float()
    rows, cols = [], []
    for d in torch.unique(dst).tolist():
        sel = dst == d
        models_d, accs_d = src[sel], acc[sel]
        n = int(models_d.numel())
        if n == 0:
            continue
        kk = min(int(top_k), n) if top_k is not None else max(1, int(round(top_frac * n)))
        top = torch.topk(accs_d, kk).indices
        rows.append(models_d[top])
        cols.append(torch.full((int(top.numel()),), d, dtype=torch.long))
    rows = torch.cat(rows) if rows else torch.zeros(0, dtype=torch.long)
    cols = torch.cat(cols) if cols else torch.zeros(0, dtype=torch.long)
    if sparse:
        return SparseMembership(rows, cols, (num_models, num_datasets))
    M = torch.zeros(num_models, num_datasets)
    M[rows, cols] = 1.0
    return M


def contrastive_supervision_topk(data, *, top_frac=0.1, top_k=None,
                                 trained_on_index=None, trained_on_attr=None):
    num_models = data["model"].num_nodes
    num_datasets = data["dataset"].num_nodes
    idx, attr = _resolve_trained_on(data, trained_on_index, trained_on_attr)
    M = high_performer_membership(idx, attr, num_models, num_datasets,
                                  top_frac=top_frac, top_k=top_k)
    pos_mask = (M @ M.t()) > 0
    pos_mask.fill_diagonal_(False)
    return pos_mask, _hub_mask(data, num_models)


def pair_density(pos_mask) -> float:
    n = pos_mask.size(0)
    return float(pos_mask.sum()) / (n * (n - 1))


def lineage_components(data, num_models):
    if IS_BASE_OF not in data.edge_types or data[IS_BASE_OF].num_edges == 0:
        return torch.arange(num_models)
    return _union_find_components(data[IS_BASE_OF].edge_index, num_models)


def topk_membership(data, *, top_frac=0.1, top_k=None,
                    trained_on_index=None, trained_on_attr=None, sparse=False):
    num_models = data["model"].num_nodes
    num_datasets = data["dataset"].num_nodes
    idx, attr = _resolve_trained_on(data, trained_on_index, trained_on_attr)
    return high_performer_membership(idx, attr, num_models, num_datasets,
                                     top_frac=top_frac, top_k=top_k, sparse=sparse)


def pool_membership_by_root(M, root_ids):
    root_ids = torch.as_tensor(root_ids, dtype=torch.long)
    if isinstance(M, SparseMembership):
        root_ids = root_ids.to(M.device)
        rows = torch.repeat_interleave(
            torch.arange(M.shape[0], device=M.device),
            M.row_ptr[1:] - M.row_ptr[:-1])
        roots = root_ids[M.row_cols]
        pairs = torch.unique(rows * (int(root_ids.max()) + 1) + roots)
        base = int(root_ids.max()) + 1
        p_rows, p_roots = pairs // base, pairs % base
        order = torch.argsort(root_ids)
        cnt = torch.bincount(root_ids, minlength=base)
        ptr = torch.zeros(base + 1, dtype=torch.long, device=M.device)
        ptr[1:] = torch.cumsum(cnt, 0)
        reps = cnt[p_roots]
        out_rows = torch.repeat_interleave(p_rows, reps)
        starts = ptr[p_roots]
        offs = (torch.arange(int(reps.sum()), device=M.device)
                - torch.repeat_interleave(torch.cumsum(reps, 0) - reps, reps)
                + torch.repeat_interleave(starts, reps))
        return SparseMembership(out_rows.cpu(), order[offs].cpu(), M.shape, dtype=M.dtype)
    M = M.float()
    uniq, inv = torch.unique(root_ids, return_inverse=True)
    inv = inv.to(M.device)
    pooled = torch.zeros(M.size(0), uniq.numel(), dtype=M.dtype, device=M.device)
    pooled.scatter_reduce_(1, inv.unsqueeze(0).expand(M.size(0), -1), M,
                           reduce="amax", include_self=True)
    return pooled[:, inv]


def per_dataset_density(trained_on_index, M, *, min_models=2):
    dst = trained_on_index[1]
    col_sum = M.sum(0)
    ds = []
    for d in torch.unique(dst).tolist():
        n = int((dst == d).sum())
        if n < min_models:
            continue
        k = int(col_sum[d])
        ds.append((k * (k - 1)) / (n * (n - 1)) if n > 1 else 0.0)
    return float(np.mean(ds)) if ds else float("nan")


def global_positive_density(M, *, max_dense=2000, n_sample=200_000, generator=None,
                            pair_chunk=4096):
    N = M.size(0)
    if N <= max_dense and not isinstance(M, SparseMembership):
        P = (M @ M.t()) > 0
        P.fill_diagonal_(False)
        return float(P.sum()) / (N * (N - 1))
    i = torch.randint(N, (n_sample,), generator=generator)
    j = torch.randint(N, (n_sample,), generator=generator)
    keep = i != j
    i, j = i[keep], j[keep]
    hits, total = 0, int(i.numel())
    for s in range(0, total, pair_chunk):
        a, b = i[s:s + pair_chunk], j[s:s + pair_chunk]
        Ma = M.rows_dense(a) if isinstance(M, SparseMembership) else M[a]
        Mb = M.rows_dense(b) if isinstance(M, SparseMembership) else M[b]
        hits += int(((Ma * Mb).sum(-1) > 0).sum())
    return float(hits) / max(1, total)


def uniformity_loss(z, *, n_pairs: int = 4096, t: float = 2.0, generator=None):
    N = z.size(0)
    if N < 2:
        return z.new_zeros(())
    i = torch.randint(N, (n_pairs,), generator=generator, device=z.device)
    j = torch.randint(N, (n_pairs,), generator=generator, device=z.device)
    keep = i != j
    if not bool(keep.any()):
        return z.new_zeros(())
    sq = (z[i[keep]] - z[j[keep]]).pow(2).sum(-1)
    return torch.log(torch.exp(-t * sq).mean() + 1e-12)


def contrastive_loss(z_model, pos_mask, hub_mask, *,
                     temperature: float = 0.2, hard_neg_weight: float = 2.0) -> torch.Tensor:
    N = z_model.size(0)
    sim = (z_model @ z_model.t()) / temperature
    sim.fill_diagonal_(float("-inf"))

    weight = torch.ones_like(sim)
    weight[hub_mask & ~pos_mask] = hard_neg_weight
    logits = sim + torch.log(weight.clamp_min(1e-12))
    denom = torch.logsumexp(logits, dim=1, keepdim=True)
    log_prob = sim - denom

    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0
    if not bool(valid.any()):
        return z_model.new_zeros(())
    pos_log_prob = log_prob.masked_fill(~pos_mask, 0.0)
    loss_i = -pos_log_prob.sum(dim=1) / pos_count.clamp_min(1).float()
    return loss_i[valid].mean()


def contrastive_loss_sampled(z_model, pos_pairs, comp_b, *,
                             temperature: float = 0.2, hard_neg_weight: float = 2.0,
                             n_neg: int = 256, generator=None):
    ai, bi = pos_pairs
    z = z_model
    B = z.size(0)
    if ai.numel() == 0:
        return z.new_zeros(())
    exact = n_neg is None
    anchors, inv = torch.unique(ai, return_inverse=True)
    A = int(anchors.numel())
    sim_pos = (z[ai] * z[bi]).sum(-1) / temperature
    za = z[anchors]

    if exact:
        neg = torch.arange(B, device=z.device).expand(A, B)
        sim_neg = (za @ z.t()) / temperature
    else:
        neg = torch.randint(B, (A, int(n_neg)), generator=generator, device=z.device)
        sim_neg = torch.bmm(z[neg], za.unsqueeze(2)).squeeze(2) / temperature

    pos_key = torch.sort(ai * B + bi).values
    is_pos = torch.isin(anchors.unsqueeze(1) * B + neg, pos_key)
    is_self = neg == anchors.unsqueeze(1)
    hub = comp_b[anchors].unsqueeze(1) == comp_b[neg]

    w = torch.ones_like(sim_neg)
    w[hub & ~is_pos & ~is_self] = hard_neg_weight
    logits = sim_neg + torch.log(w.clamp_min(1e-12))
    if exact:
        denom = torch.logsumexp(logits.masked_fill(is_self, float("-inf")), dim=1)
    else:
        logits = logits.masked_fill(is_self | is_pos, float("-inf"))
        mx = torch.full((A,), float("-inf"), device=z.device, dtype=sim_pos.dtype)
        mx = mx.index_reduce(0, inv, sim_pos, "amax", include_self=False)
        sp = torch.zeros(A, device=z.device, dtype=sim_pos.dtype).index_add(
            0, inv, torch.exp(sim_pos - mx[inv]))
        lse_pos = mx + torch.log(sp)
        denom = torch.logaddexp(torch.logsumexp(logits, dim=1), lse_pos)

    cnt = torch.zeros(A, device=z.device, dtype=sim_pos.dtype).index_add(
        0, inv, torch.ones_like(sim_pos))
    mean_pos = torch.zeros(A, device=z.device, dtype=sim_pos.dtype).index_add(
        0, inv, sim_pos) / cnt
    return (denom - mean_pos).mean()


def dataset_to_model_contrastive(z_dict, trained_on_index, M, *,
                                 temperature: float = 0.1, hard_neg_weight: float = 1.0,
                                 min_pos: int = 1, return_stats: bool = False):
    z_m, z_d = z_dict["model"], z_dict["dataset"]
    dst = trained_on_index[1]
    src = trained_on_index[0]
    losses, n_datasets, n_pos_total = [], 0, 0
    for d in torch.unique(dst).tolist():
        cand = torch.unique(src[dst == d])
        if cand.numel() < 2:
            continue
        pos_flag = M[cand, d] > 0
        if int(pos_flag.sum()) < min_pos or int(pos_flag.sum()) == cand.numel():
            continue
        s = (z_m[cand] @ z_d[d]) / temperature
        w = torch.ones_like(s)
        w[~pos_flag] = hard_neg_weight
        logits = s + torch.log(w.clamp_min(1e-12))
        denom = torch.logsumexp(logits, dim=0)
        num = torch.logsumexp(s[pos_flag], dim=0)
        losses.append(denom - num)
        n_datasets += 1
        n_pos_total += int(pos_flag.sum())
    if not losses:
        loss = z_m.new_zeros(())
        return (loss, {"n_datasets": 0, "n_pos": 0}) if return_stats else loss
    loss = torch.stack(losses).mean()
    if return_stats:
        return loss, {"n_datasets": n_datasets, "n_pos": n_pos_total}
    return loss


def dataset_to_model_contrastive_from_edges(z_dict, edge_label_index, edge_label, *,
                                            top_frac=0.10, temperature=0.1,
                                            min_models=2, return_stats=False):
    z_m, z_d = z_dict["model"], z_dict["dataset"]
    src, dst, acc = edge_label_index[0], edge_label_index[1], edge_label.float()
    losses, n_datasets = [], 0
    for d in torch.unique(dst).tolist():
        sel = (dst == d).nonzero().flatten()
        n = int(sel.numel())
        if n < min_models:
            continue
        models_d = src[sel]
        a = acc[sel]
        kk = max(1, int(round(top_frac * n)))
        if kk >= n:
            continue
        pos_idx = torch.topk(a, kk).indices
        pos_flag = torch.zeros(n, dtype=torch.bool, device=z_m.device)
        pos_flag[pos_idx] = True
        s = (z_m[models_d] @ z_d[d]) / temperature
        denom = torch.logsumexp(s, dim=0)
        num = torch.logsumexp(s[pos_flag], dim=0)
        losses.append(denom - num)
        n_datasets += 1
    if not losses:
        loss = z_m.new_zeros(())
        return (loss, {"n_datasets": 0}) if return_stats else loss
    loss = torch.stack(losses).mean()
    return (loss, {"n_datasets": n_datasets}) if return_stats else loss


def _pos_ids(M, d):
    if isinstance(M, SparseMembership):
        return M.col_ids(int(d))
    return (M[:, d] > 0).nonzero().flatten()


def build_global_negative_pools(task_type_id, trained_on_index, trained_on_attr, M, *,
                                include_known_low=False, low_frac=0.3):
    src, dst, acc = trained_on_index[0], trained_on_index[1], trained_on_attr.float()
    model_tasks = {}
    for m, d in zip(src.tolist(), dst.tolist()):
        model_tasks.setdefault(m, set()).add(int(task_type_id[d]))
    observed = sorted(model_tasks)

    pools, stats = {}, {"datasets": 0, "with_incompat": 0, "with_known_low": 0,
                        "incompat_total": 0, "known_low_total": 0}
    for d in torch.unique(dst).tolist():
        td = int(task_type_id[d])
        neg = []
        n_inc = 0
        if td != 0:
            for m in observed:
                ts = model_tasks[m]
                if ts and td not in ts:
                    neg.append(m)
            n_inc = len(neg)
        n_low = 0
        if include_known_low:
            sel = (dst == d).nonzero().flatten()
            if sel.numel() >= 3:
                a = acc[sel]
                thresh = torch.quantile(a, low_frac)
                low = src[sel][a <= thresh]
                low = [int(m) for m in low.tolist() if M[int(m), d] == 0]
                n_low = len(low)
                neg.extend(low)
        if neg:
            pools[int(d)] = torch.tensor(sorted(set(neg)), dtype=torch.long)
            stats["datasets"] += 1
            stats["with_incompat"] += int(n_inc > 0)
            stats["with_known_low"] += int(n_low > 0)
            stats["incompat_total"] += n_inc
            stats["known_low_total"] += n_low
    return pools, stats


def global_retrieval_loss(z_dict, M, pools, *, temperature=0.1, n_neg=64,
                          n_datasets=16, hard_frac=0.0, generator=None,
                          return_stats=False):
    z_m, z_d = z_dict["model"], z_dict["dataset"]
    col_sum = M.sum(0)
    ds_all = [d for d in pools if int(col_sum[d]) > 0]
    if not ds_all:
        loss = z_m.new_zeros(())
        return (loss, {"n_datasets": 0}) if return_stats else loss
    if len(ds_all) > n_datasets:
        idx = torch.randperm(len(ds_all), generator=generator)[:n_datasets]
        ds_batch = [ds_all[i] for i in idx.tolist()]
    else:
        ds_batch = ds_all

    losses, n_negs_used = [], 0
    for d in ds_batch:
        pos = _pos_ids(M, d).to(z_m.device)
        pool = pools[d].to(z_m.device)
        if pool.numel() > n_neg:
            if hard_frac > 0:
                with torch.no_grad():
                    s_pool = z_m[pool] @ z_d[d]
                n_hard = int(round(hard_frac * n_neg))
                hard = pool[torch.argsort(-s_pool)[:n_hard]]
                rest = pool[torch.randperm(pool.numel(), generator=generator)[:n_neg - n_hard]]
                neg = torch.unique(torch.cat([hard, rest]))
            else:
                neg = pool[torch.randperm(pool.numel(), generator=generator)[:n_neg]]
        else:
            neg = pool
        cand = torch.cat([pos, neg])
        s = (z_m[cand] @ z_d[d]) / temperature
        denom = torch.logsumexp(s, dim=0)
        losses.append((denom - s[: pos.numel()]).mean())
        n_negs_used += int(neg.numel())
    loss = torch.stack(losses).mean()
    if return_stats:
        return loss, {"n_datasets": len(ds_batch), "n_negs_used": n_negs_used}
    return loss


def build_lake_logq(trained_on_index, num_models, *, alpha=0.75, n0=1.0,
                    gamma=0.0):
    deg = torch.bincount(trained_on_index[0], minlength=num_models).float()
    w = (deg + float(n0)) ** float(alpha)
    q = w / w.sum()
    if gamma > 0.0:
        lab = (deg > 0).to(q.dtype)
        n_lab = lab.sum()
        if n_lab > 0:
            q = (1.0 - float(gamma)) * q + float(gamma) * lab / n_lab
    return q, q.log()


def global_lake_loss(z_dict, M, q, logq, *, temperature=0.1, n_neg=64,
                     n_datasets=16, hard_sets=None, n_hard=16,
                     pos_ipw=None, generator=None, return_stats=False):
    z_m, z_d = z_dict["model"], z_dict["dataset"]
    dev = z_m.device
    ds_all = (M.sum(0) > 0).nonzero().flatten().tolist()
    if not ds_all:
        loss = z_m.new_zeros(())
        return (loss, {"n_datasets": 0}) if return_stats else loss
    if len(ds_all) > n_datasets:
        idx = torch.randperm(len(ds_all), generator=generator)[:n_datasets]
        ds_batch = [ds_all[i] for i in idx.tolist()]
    else:
        ds_batch = ds_all

    q_dev, logq_dev = q.to(dev), logq.to(dev)
    losses, n_neg_used, n_hard_used = [], 0, 0
    for d in ds_batch:
        pos = _pos_ids(M, d).to(dev)
        samp = torch.multinomial(q_dev, n_neg, replacement=True, generator=generator)
        samp = samp[~torch.isin(samp, pos)]
        logits = [z_m[pos] @ z_d[d] / temperature,
                  z_m[samp] @ z_d[d] / temperature - logq_dev[samp]]
        n_neg_used += int(samp.numel())
        if hard_sets and d in hard_sets and n_hard > 0:
            h = hard_sets[d].to(dev)
            h = h[~torch.isin(h, pos)]
            if h.numel() > n_hard:
                h = h[torch.randperm(h.numel(), generator=generator)[:n_hard]]
            if h.numel():
                logits.append(z_m[h] @ z_d[d] / temperature)
                n_hard_used += int(h.numel())
        s = torch.cat(logits)
        denom = torch.logsumexp(s, dim=0)
        per_pos = denom - s[: pos.numel()]
        if pos_ipw is not None:
            w = pos_ipw.to(dev)[pos]
            losses.append((w * per_pos).sum() / w.sum())
        else:
            losses.append(per_pos.mean())
    loss = torch.stack(losses).mean()
    if return_stats:
        return loss, {"n_datasets": len(ds_batch), "n_negs_used": n_neg_used,
                      "n_hard_used": n_hard_used}
    return loss


def mine_hard_negative_sets(z_dict, M, *, hard_k=20):
    z_m, z_d = z_dict["model"].detach(), z_dict["dataset"].detach()
    hard = {}
    for d in (M.sum(0) > 0).nonzero().flatten().tolist():
        pos = _pos_ids(M, d).to(z_m.device)
        s = z_m @ z_d[d]
        s[pos] = float("-inf")
        k = min(hard_k, int(s.numel() - pos.numel()))
        hard[d] = torch.topk(s, k).indices.cpu()
    return hard


def mine_alibi_hard_negative_sets(z_dict, M, deg, dataset_task_id, model_tasks, *,
                                  hard_k=20, deg_quantile=0.9, overshoot=3):
    z_m, z_d = z_dict["model"].detach(), z_dict["dataset"].detach()
    lab = deg[deg > 0].float()
    thresh = torch.quantile(lab, deg_quantile) if lab.numel() else torch.tensor(1.0)
    hard = {}
    for d in (M.sum(0) > 0).nonzero().flatten().tolist():
        pos = _pos_ids(M, d).to(z_m.device)
        s = z_m @ z_d[d]
        s[pos] = float("-inf")
        k_mine = min(hard_k * overshoot, int(s.numel() - pos.numel()))
        cand = torch.topk(s, k_mine).indices.cpu()
        td = int(dataset_task_id[d])
        keep = []
        for m in cand.tolist():
            hubby = bool(deg[m] >= thresh)
            ts = model_tasks.get(m, set())
            mismatch = bool(ts) and td != 0 and td not in ts
            if hubby or mismatch:
                keep.append(m)
            if len(keep) == hard_k:
                break
        if keep:
            hard[d] = torch.tensor(keep, dtype=torch.long)
    return hard


def build_model_task_profiles(trained_on_index, task_type_id):
    prof = {}
    for m, d in zip(trained_on_index[0].tolist(), trained_on_index[1].tolist()):
        t = int(task_type_id[d])
        if t != 0:
            prof.setdefault(m, set()).add(t)
    return prof


def dataset_push_apart_loss(z_d, task_type_id, *, margin=0.2, n_anchor=32,
                            n_neg=16, generator=None, return_stats=False):
    known = (task_type_id > 0).nonzero().flatten()
    if known.numel() < 2:
        loss = z_d.new_zeros(())
        return (loss, {"n_pairs": 0}) if return_stats else loss
    if known.numel() > n_anchor:
        pick = torch.randperm(known.numel(), generator=generator)[:n_anchor]
        anchors = known[pick.to(known.device)]
    else:
        anchors = known
    tt = task_type_id
    terms, n_pairs = [], 0
    for a in anchors.tolist():
        diff = known[tt[known] != tt[a]]
        if diff.numel() == 0:
            continue
        if diff.numel() > n_neg:
            pick = torch.randperm(diff.numel(), generator=generator)[:n_neg]
            diff = diff[pick.to(diff.device)]
        cos = z_d[diff] @ z_d[a]
        terms.append(torch.relu(cos - margin).mean())
        n_pairs += int(diff.numel())
    if not terms:
        loss = z_d.new_zeros(())
        return (loss, {"n_pairs": 0}) if return_stats else loss
    loss = torch.stack(terms).mean()
    if return_stats:
        return loss, {"n_pairs": n_pairs, "n_anchors": len(terms)}
    return loss


def combined_loss(l_perf, l_contrast, *, lambda_perf=1.0, lambda_contrast=1.0):
    total = lambda_perf * l_perf + lambda_contrast * l_contrast
    return total, {
        "perf": float(l_perf.detach()),
        "contrast": float(l_contrast.detach()),
        "total": float(total.detach()),
    }
