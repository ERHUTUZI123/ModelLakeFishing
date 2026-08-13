"""
sampling.py -- Stage 2, Step 4: subgraph sampling + edge dropout.

LinkNeighborLoader samples a subgraph around each batch of supervised `trained_on`
edges. The node index columns (size_bucket_id / family_id) and the frozen x ride
on the model node store, so the loader slices and re-aligns them with the
subgraph automatically -- there is never any global-id gather in user code. The
supervision edges live in `edge_label_index` / `edge_label` (the regression
target), which the loader also remaps to local node ids.

Edge dropout randomly drops MESSAGE edges (those in `edge_index`) during training
to simulate the sparse neighbourhoods of long-tail / cold-start models, forcing
the GNN to recover signal from a node's own four-component x_m^(0) when neighbours
go missing. It is our structural dual of ModelLens's ID-dropout. Two invariants:

  * supervision is never dropped -- it lives in edge_label_index, not edge_index,
    so touching edge_index alone protects it by construction;
  * lineage edges (is_base_of / rev_is_base_of) get a LOWER drop rate -- they are
    the only edge a cold-start model has, so dropping them out is self-sabotage.

This file provides the loader factory, the edge-dropout transform, and a helper
to slice the global contrastive masks down to a batch. The full training loop
(optimizer, epochs, val/test, checkpoint export) is Step 5.

Run the mechanism smoke test:
    python -m ModelLakeFishing.stage2TrainGraphSAGE.sampling
"""

import os
import sys
import warnings

import torch
from torch_geometric.loader import LinkNeighborLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.losses import TRAINED_ON  # noqa: E402

LINEAGE_RELS = {
    ("model", "is_base_of", "model"),
    ("model", "rev_is_base_of", "model"),
}


# ── CSR neighbour index (T0 item 1) ─────────────────────────────────────────
# The legacy closure asked `torch.isin(src, tensor(sorted(cur)))` per relation
# per hop -- an O(E) scan of every edge for every frontier. Sorting each relation
# by source once turns a neighbour lookup into a slice: nbr[ptr[v]:ptr[v+1]].

def build_csr(edge_index, num_src):
    """(ptr[num_src+1], nbr[E]) with node v's neighbours at nbr[ptr[v]:ptr[v+1]]."""
    src, dst = edge_index[0], edge_index[1]
    order = torch.argsort(src)
    nbr = dst[order].contiguous()
    ptr = torch.zeros(num_src + 1, dtype=torch.long)
    if src.numel():
        ptr[1:] = torch.cumsum(torch.bincount(src[order], minlength=num_src), 0)
    return ptr, nbr


def _sample_neighbours(ptr, nbr, nodes, k, generator=None):
    """Union of <= k neighbours per node in `nodes` (all of them when deg <= k).

    k=None means "no cap" (the legacy full-neighbour closure). Sampling is
    WITHOUT replacement inside a node's own adjacency slice, so the fan-out is a
    hard per-node bound, not an expectation.
    """
    out = set()
    for v in nodes:
        lo, hi = int(ptr[v]), int(ptr[v + 1])
        deg = hi - lo
        if deg == 0:
            continue
        if k is None or deg <= k:
            out.update(nbr[lo:hi].tolist())
        else:
            pick = torch.randperm(deg, generator=generator)[:k]
            out.update(nbr[lo + pick].tolist())
    return out


def _has_sampler_backend() -> bool:
    """LinkNeighborLoader's neighbour sampler needs pyg-lib or torch-sparse."""
    for mod in ("pyg_lib", "torch_sparse"):
        try:
            __import__(mod)
            return True
        except Exception:
            pass
    return False


class LightLinkLoader:
    """
    Pure-python fallback for environments without pyg-lib / torch-sparse.

    Yields the SAME batch interface LinkNeighborLoader does -- an induced
    subgraph (built with HeteroData.subgraph, which slices node attrs and
    relabels edges, so size_bucket_id / family_id / x stay aligned with no manual
    gather), carrying `edge_label_index` (local) + `edge_label` (regression
    target) on the trained_on store and `n_id` per node type.

    Two expansion modes:

      fanout=None  : FULL-neighbour k-hop closure (the legacy path -- correct and
                     cheap on the zoo, but the subgraph is unbounded, so on a
                     100K lake a single batch can pull in most of the graph).
      fanout=(k1,k2): at most k_hop neighbours per frontier node per relation
                     direction, drawn without replacement. The subgraph size is
                     then bounded by the seeds and the fan-out, independent of N
                     -- this is what makes the loader usable at 100K/1M.

    Lineage relations take `fanout_lineage` (higher by default): they are the
    only edge a cold-start model has, the same reasoning that gives them a lower
    edge-dropout rate.
    """

    def __init__(self, data, edge_label_index, edge_label, *,
                 num_hops=2, batch_size=128, shuffle=True,
                 fanout=None, fanout_lineage=None, generator=None):
        self.data = data
        self.eli = edge_label_index
        self.elabel = edge_label
        self.num_hops = num_hops
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator
        # cache message edges per relation for neighbour expansion
        self._edges = {et: data[et].edge_index for et in data.edge_types}
        self.fanout = list(fanout) if fanout is not None else None
        if self.fanout is not None:
            assert len(self.fanout) >= num_hops, "fanout needs one entry per hop"
            self.fanout_lineage = (list(fanout_lineage) if fanout_lineage is not None
                                   else [2 * k for k in self.fanout])
            n_nodes = {t: data[t].num_nodes for t in data.node_types}
            # forward (src -> dst) and backward (dst -> src) CSR per relation:
            # the closure walks both directions, exactly as the legacy path did.
            self._csr, self._csc = {}, {}
            for et, ei in self._edges.items():
                st, _rel, dt = et
                self._csr[et] = build_csr(ei, n_nodes[st])
                self._csc[et] = build_csr(ei.flip(0), n_nodes[dt])
        else:
            self.fanout_lineage = None

    def _khop_closure(self, seed_m, seed_d):
        if self.fanout is not None:
            return self._khop_closure_fanout(seed_m, seed_d)
        cur = {"model": set(seed_m), "dataset": set(seed_d)}
        for _ in range(self.num_hops):
            add = {"model": set(), "dataset": set()}
            for (st, _rel, dt), ei in self._edges.items():
                src, dst = ei[0], ei[1]
                if cur[st]:
                    sel = torch.isin(src, torch.tensor(sorted(cur[st])))
                    add[dt].update(dst[sel].tolist())
                if cur[dt]:
                    sel = torch.isin(dst, torch.tensor(sorted(cur[dt])))
                    add[st].update(src[sel].tolist())
            grew = False
            for t in cur:
                before = len(cur[t])
                cur[t] |= add[t]
                grew |= len(cur[t]) > before
            if not grew:
                break
        return cur

    def _khop_closure_fanout(self, seed_m, seed_d):
        """Fan-out-capped closure: expand only the CURRENT frontier, and only up
        to fanout[hop] neighbours per node per relation direction."""
        visited = {"model": set(seed_m), "dataset": set(seed_d)}
        frontier = {"model": set(seed_m), "dataset": set(seed_d)}
        for hop in range(self.num_hops):
            nxt = {"model": set(), "dataset": set()}
            for et in self._edges:
                st, _rel, dt = et
                k = (self.fanout_lineage[hop] if et in LINEAGE_RELS
                     else self.fanout[hop])
                if k is not None and k <= 0:
                    continue
                if frontier[st]:
                    ptr, nbr = self._csr[et]
                    nxt[dt] |= _sample_neighbours(ptr, nbr, frontier[st], k,
                                                  generator=self.generator)
                if frontier[dt]:
                    ptr, nbr = self._csc[et]
                    nxt[st] |= _sample_neighbours(ptr, nbr, frontier[dt], k,
                                                  generator=self.generator)
            frontier = {t: nxt[t] - visited[t] for t in visited}
            for t in visited:
                visited[t] |= frontier[t]
            if not any(frontier.values()):
                break
        return visited

    def __iter__(self):
        P = self.eli.size(1)
        order = torch.randperm(P) if self.shuffle else torch.arange(P)
        for start in range(0, P, self.batch_size):
            idx = order[start:start + self.batch_size]
            seed = self.eli[:, idx]
            closure = self._khop_closure(seed[0].tolist(), seed[1].tolist())
            model_ids = torch.tensor(sorted(closure["model"]), dtype=torch.long)
            dataset_ids = torch.tensor(sorted(closure["dataset"]), dtype=torch.long)

            sub = self.data.subgraph({"model": model_ids, "dataset": dataset_ids})
            sub["model"].n_id = model_ids
            sub["dataset"].n_id = dataset_ids

            g2l_m = {int(g): l for l, g in enumerate(model_ids.tolist())}
            g2l_d = {int(g): l for l, g in enumerate(dataset_ids.tolist())}
            local = torch.tensor(
                [[g2l_m[int(s)] for s in seed[0].tolist()],
                 [g2l_d[int(d)] for d in seed[1].tolist()]],
                dtype=torch.long,
            )
            sub[TRAINED_ON].edge_label_index = local
            sub[TRAINED_ON].edge_label = self.elabel[idx]
            yield sub


def make_link_loader(train_data, edge_label_index, edge_label, *,
                     num_neighbors=(10, 10), batch_size=128, shuffle=True,
                     fanout=False, fanout_lineage=None, generator=None):
    """
    Loader over the supervised `trained_on` edges.

    train_data       : the TRAIN split from losses.split_trained_on. Its
                       trained_on.edge_index holds MESSAGE edges only (the
                       disjoint split already moved supervision out), so sampled
                       neighbourhoods never see the edges being predicted.
    edge_label_index : [2, P] positive supervision edges (model_idx, dataset_idx)
    edge_label       : [P] regression target (normalized accuracy) for each
    num_neighbors    : fan-out per hop; length should match GNN depth (2).
    fanout           : T0 item 1. False (default) keeps the historical
                       full-neighbour fallback closure bit-for-bit. True caps the
                       fallback loader's expansion at `num_neighbors` per hop
                       (lineage relations at `fanout_lineage`, default 2x), which
                       is what bounds subgraph size at 100K/1M. Ignored when a
                       real sampler backend is present -- LinkNeighborLoader
                       always fans out.

    Returns a real LinkNeighborLoader when a sampler backend is installed,
    otherwise the pure-python LightLinkLoader (same batch interface).
    """
    if _has_sampler_backend():
        return LinkNeighborLoader(
            train_data,
            num_neighbors=list(num_neighbors),
            edge_label_index=(TRAINED_ON, edge_label_index),
            edge_label=edge_label,
            batch_size=batch_size,
            shuffle=shuffle,
        )
    if not fanout:
        warnings.warn(
            "pyg-lib / torch-sparse not installed; falling back to LightLinkLoader "
            "(full-neighbour, fine for the zoo). Install a backend, or pass "
            "fanout=True, before scaling past ~30K models.",
            RuntimeWarning,
        )
    return LightLinkLoader(
        train_data, edge_label_index, edge_label,
        num_hops=len(num_neighbors), batch_size=batch_size, shuffle=shuffle,
        fanout=list(num_neighbors) if fanout else None,
        fanout_lineage=fanout_lineage, generator=generator,
    )


@torch.no_grad()
def apply_edge_dropout(batch, *, p: float = 0.3, p_lineage: float = 0.05):
    """
    Drop message edges in place, per edge type. Supervision (edge_label_index)
    is untouched because it is not an edge_index. Lineage relations use the lower
    `p_lineage`. A relation is never emptied entirely (keeps message passing and
    to_hetero happy on small subgraphs). Returns a {relation: (kept, total)} dict.
    """
    info = {}
    for et in batch.edge_types:
        store = batch[et]
        if "edge_index" not in store:
            continue
        total = store.edge_index.size(1)
        if total == 0:
            info[et] = (0, 0)
            continue
        rate = p_lineage if et in LINEAGE_RELS else p
        if rate <= 0:
            info[et] = (total, total)
            continue
        keep = torch.rand(total, device=store.edge_index.device) >= rate
        if int(keep.sum()) == 0:                 # never zero out a relation
            keep[torch.randint(total, (1,))] = True
        store.edge_index = store.edge_index[:, keep]
        if getattr(store, "edge_attr", None) is not None and store.edge_attr.size(0) == total:
            store.edge_attr = store.edge_attr[keep]
        info[et] = (int(keep.sum()), total)
    return info


def batch_contrastive_masks(batch, M, comp):
    """
    Build the batch's [B, B] contrastive masks from the membership matrix M
    [N_model, N_dataset] and the lineage component vector comp [N_model], using
    batch['model'].n_id. This is 47K-safe: it touches only the batch's rows of M
    and comp -- it NEVER materializes a global [N_model, N_model] mask.

      pos_mask_b[i, j] : models i, j co-selected as top performers on >=1 dataset
      hub_mask_b[i, j] : models i, j in the same lineage component

    Returns (pos_mask_b, hub_mask_b) over the batch's local model ordering.

    Still [B, B]: fine up to a few thousand batch nodes, ~6.4 GB of live blocks
    at B = 20,000. Past that use batch_positive_pairs + losses.contrastive_loss_sampled.
    """
    n_id = batch["model"].n_id
    Mb = M.rows_dense(n_id) if hasattr(M, "rows_dense") else M[n_id]  # [B, num_datasets]
    pos = (Mb @ Mb.t()) > 0                        # [B, B]
    pos.fill_diagonal_(False)
    cb = comp[n_id]
    hub = cb.unsqueeze(0) == cb.unsqueeze(1)       # [B, B]; singletons never match
    hub.fill_diagonal_(False)
    return pos, hub


def batch_positive_pairs(batch, M, *, max_per_dataset=None, generator=None):
    """
    T0 item 3: the SPARSE form of batch_contrastive_masks' pos_mask.

    Returns (ai, bi) local index pairs -- de-duplicated (a pair co-selected on
    three datasets is ONE positive, exactly as the boolean mask says) and
    diagonal-free -- so a [B, B] block is never allocated. Cost is
    sum_d k_d^2 over the batch's datasets, where k_d is how many of d's high
    performers landed in this batch; `max_per_dataset` caps k_d by subsampling
    when a hub dataset would otherwise dominate.
    """
    n_id = batch["model"].n_id
    if hasattr(M, "rows_coo"):
        loc, col = M.rows_coo(n_id)
    else:
        loc, col = (M[n_id] > 0).nonzero(as_tuple=True)
    B = int(n_id.numel())
    empty = torch.zeros(0, dtype=torch.long, device=loc.device)
    if loc.numel() == 0:
        return empty, empty

    order = torch.argsort(col)
    loc, col = loc[order], col[order]
    uniq, counts = torch.unique_consecutive(col, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts

    if max_per_dataset is not None and bool((counts > max_per_dataset).any()):
        keep = []
        for s, c in zip(starts.tolist(), counts.tolist()):
            if c <= max_per_dataset:
                keep.append(torch.arange(s, s + c, device=loc.device))
            else:
                pick = torch.randperm(c, generator=generator)[:max_per_dataset]
                keep.append(s + pick.to(loc.device))
        sel = torch.cat(keep)
        loc, col = loc[sel], col[sel]
        uniq, counts = torch.unique_consecutive(col, return_counts=True)
        starts = torch.cumsum(counts, 0) - counts

    sq = counts * counts
    total = int(sq.sum())
    if total == 0:
        return empty, empty
    gid = torch.repeat_interleave(torch.arange(counts.numel(), device=loc.device), sq)
    within = (torch.arange(total, device=loc.device)
              - torch.repeat_interleave(torch.cumsum(sq, 0) - sq, sq))
    k = counts[gid]
    base = starts[gid]
    ai = loc[base + within // k]
    bi = loc[base + within % k]
    keep = ai != bi
    ai, bi = ai[keep], bi[keep]
    if ai.numel() == 0:
        return empty, empty
    key = torch.unique(ai * B + bi)                 # one pair, however many datasets
    return key // B, key % B


# ── mechanism smoke test ────────────────────────────────────────────────────
if __name__ == "__main__":
    from ModelLakeFishing.stage2TrainGraphSAGE.model import HeteroGraphSAGE, load_hgraph
    from ModelLakeFishing.stage2TrainGraphSAGE.losses import (
        PerfScorer, accuracy_lookup, perf_supervision, perf_ranking_loss,
        split_trained_on, topk_membership, lineage_components, contrastive_loss,
    )

    data, xm0, _umi = load_hgraph()
    model = HeteroGraphSAGE(
        metadata=data.metadata(),
        frozen_dim=data["model"].x.shape[1],
        num_size_buckets=xm0["num_size_buckets"],
        num_families=xm0["num_families"],
        dataset_in_dim=data["dataset"].x.shape[1],
    )
    scorer = PerfScorer(dim=128, mode="dot")

    failures = []

    def check(cond, msg):
        print(f"  [{'OK  ' if cond else 'FAIL'}] {msg}")
        if not cond:
            failures.append(msg)

    train_data, _val, _test = split_trained_on(data, seed=0)
    lookup = accuracy_lookup(data)
    eli, target = perf_supervision(train_data[TRAINED_ON], lookup)
    # scale-safe supervision: membership M + components, masks built PER BATCH
    ti = torch.cat([train_data[TRAINED_ON].edge_index, eli], dim=1)
    ta = torch.cat([train_data[TRAINED_ON].edge_attr.float(), target], dim=0)
    M = topk_membership(data, top_frac=0.1, trained_on_index=ti, trained_on_attr=ta)
    comp = lineage_components(data, data["model"].num_nodes)

    loader = make_link_loader(train_data, eli, target, num_neighbors=(10, 10), batch_size=128)
    print(f"\nactive loader: {type(loader).__name__} "
          f"(sampler backend {'present' if _has_sampler_backend() else 'MISSING -> fallback'})")

    # --- one batch: structure, auto-aligned node attrs, supervision present ---
    print("\n=== loader batch structure ===")
    batch = next(iter(loader))
    m = batch["model"]
    print(f"      batch model nodes={m.num_nodes}, dataset nodes={batch['dataset'].num_nodes}, "
          f"supervision edges={batch[TRAINED_ON].edge_label_index.size(1)}")
    check(m.x.size(0) == m.num_nodes == m.size_bucket_id.size(0) == m.family_id.size(0),
          "model x / size_bucket_id / family_id auto-sliced to the same node count (no gather)")
    check(hasattr(batch[TRAINED_ON], "edge_label") and
          batch[TRAINED_ON].edge_label.size(0) == batch[TRAINED_ON].edge_label_index.size(1),
          "supervision edge_label (regression target) rides along, remapped to local ids")
    check(hasattr(m, "n_id"), "batch carries global n_id for masks/export alignment")

    # --- edge dropout: message edges drop, supervision preserved, lineage spared ---
    print("\n=== edge dropout (supervision protected, lineage lower rate) ===")
    sup_before = batch[TRAINED_ON].edge_label_index.size(1)
    msg_before = {et: batch[et].edge_index.size(1) for et in batch.edge_types}
    info = apply_edge_dropout(batch, p=0.5, p_lineage=0.0)
    sup_after = batch[TRAINED_ON].edge_label_index.size(1)
    check(sup_after == sup_before, "supervision edge count unchanged by dropout")
    non_lineage_dropped = any(
        info[et][0] < msg_before[et] for et in batch.edge_types
        if et not in LINEAGE_RELS and msg_before[et] > 0
    )
    check(non_lineage_dropped, "non-lineage message edges were dropped (p=0.5)")
    lineage_kept = all(
        info.get(et, (0, 0))[0] == info.get(et, (0, 0))[1]
        for et in LINEAGE_RELS if et in batch.edge_types
    )
    check(lineage_kept, "lineage edges fully kept at p_lineage=0.0 (cold-start lifeline)")

    # --- forward on the dropped subgraph + per-batch loss (ranking + contrast) ---
    print("\n=== forward + per-batch loss (47K-safe masks) ===")
    z = model(batch)
    lp = perf_ranking_loss(scorer, z, batch[TRAINED_ON].edge_label_index, batch[TRAINED_ON].edge_label)
    pos_b, hub_b = batch_contrastive_masks(batch, M, comp)
    check(pos_b.shape == (m.num_nodes, m.num_nodes),
          "per-batch masks are B x B (built from membership, no global N x N)")
    lc = contrastive_loss(z["model"], pos_b, hub_b)
    total = lp + lc
    check(torch.isfinite(total) and total.item() > 0, f"per-batch loss finite ({total.item():.4f})")

    # --- a few sampled+dropped steps descend ---
    print("\n=== mini-batch training descends ===")
    params = list(model.parameters()) + list(scorer.parameters())
    opt = torch.optim.Adam(params, lr=1e-2)
    first = last = None
    for epoch in range(4):
        for b in loader:
            apply_edge_dropout(b, p=0.3, p_lineage=0.05)
            opt.zero_grad()
            zb = model(b)
            lpb = perf_ranking_loss(scorer, zb, b[TRAINED_ON].edge_label_index, b[TRAINED_ON].edge_label)
            pb, hb = batch_contrastive_masks(b, M, comp)
            lcb = contrastive_loss(zb["model"], pb, hb)
            tot = lpb + lcb
            tot.backward()
            opt.step()
            val = float(tot.detach())
            if first is None:
                first = val
            last = val
    print(f"      first batch total={first:.4f}   last batch total={last:.4f}")
    check(last < first, "per-batch loss decreased over sampled+dropped batches")

    print("\n" + "=" * 52)
    if failures:
        print(f"SMOKE TEST FAILED -- {len(failures)} check(s):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("SMOKE TEST OK -- LinkNeighborLoader auto-alignment + edge dropout verified.")
