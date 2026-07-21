"""
losses.py -- Stage 2, Step 3: the two training losses.

    L = lambda_perf * L_perf + lambda_contrast * L_contrast

L_perf (performance regression)
    Supervised on `trained_on` edges. A scorer reads z_m[src] and z_d[dst] and
    predicts the normalized accuracy edge weight (already in ~[0.6, 1.0] on the
    zoo); MSE against that weight. This is the main task: it makes z_m . z_d
    encode "who performs well on which kind of data", which is exactly the inner
    product HNSW will rank by. The train/val/test edge split reuses Stage 1's
    RandomLinkSplit, and -- critically -- it is handed the reverse edge type
    (`rev_trained_on`) so held-out edges are removed in BOTH directions from the
    message-passing graph (otherwise the reverse edge leaks the label back in).

L_contrast (task-structure contrast)
    Supervised-contrastive (InfoNCE) over model embeddings. Positives: models
    that both perform well (accuracy >= acc_thresh) on a common dataset.
    Negatives: every other model, with same-hub lineage derivatives that are NOT
    co-performers deliberately up-weighted in the denominator (hard negatives).
    Without that pressure the lineage edges collapse a family's derivatives into
    one point near the hub and HNSW recall there dies; this term is the only
    source of task-sensitivity under a single index.

lambda_perf : lambda_contrast starts at 1:1 -- read both descent curves before
tuning, do not pre-tune.

This file defines the losses and the supervision builders only. The training
loop, sampling (LinkNeighborLoader) and edge dropout are Step 4/5.

Run the mechanism smoke test:
    python -m ModelLakeFishing.stage2TrainGraphSAGE.losses
"""

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

from ModelLakeFishing.stage1BuildTransferGraph.dataset_embed.utils.CustomRandomLinkSplit import RandomLinkSplit  # noqa: E402


class _RandomLinkSplit(RandomLinkSplit):
    """CustomRandomLinkSplit overrides __call__ but newer torch_geometric marks
    BaseTransform.forward abstract; provide a concrete forward (delegating to the
    custom __call__) so the class is instantiable. forward is never hit because
    transform(data) dispatches to the overridden __call__."""

    def forward(self, data):
        return RandomLinkSplit.__call__(self, data)


TRAINED_ON = ("model", "trained_on", "dataset")
REV_TRAINED_ON = ("dataset", "rev_trained_on", "model")
IS_BASE_OF = ("model", "is_base_of", "model")


# ── L_perf: scorer + regression ─────────────────────────────────────────────

class PerfScorer(nn.Module):
    """
    Predict normalized accuracy for (model, dataset) pairs from z_m, z_d.

    mode="dot" : sigmoid(scale * <z_m, z_d> + bias). Keeps the trained geometry
                 aligned with the HNSW inner-product metric (z's are unit norm,
                 so <z_m, z_d> is cosine); scale/bias are the only free params.
    mode="mlp" : sigmoid(MLP([z_m || z_d])). More flexible, but the score is no
                 longer a pure inner product, so prefer "dot" unless evidence
                 says otherwise.
    """

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
    """(model_idx, dataset_idx) -> normalized accuracy, from the full graph."""
    ei = data[TRAINED_ON].edge_index
    ea = data[TRAINED_ON].edge_attr.float()
    return {(int(s), int(d)): float(a) for s, d, a in zip(ei[0], ei[1], ea)}


def perf_supervision(split_store, lookup: dict):
    """
    Pull the POSITIVE supervision edges out of a RandomLinkSplit store and attach
    their regression target (accuracy). RandomLinkSplit writes binary edge_label
    (1 = real edge, 0 = sampled negative); regression uses only the positives.

    Returns (edge_label_index[2, P], target[P]).
    """
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
    """
    Margin ranking on `trained_on` supervision, WITHIN each dataset.

    For a dataset d, two models with different accuracy form a pair; the higher
    accuracy one must score higher (same z_m . z_d dot geometry as PerfScorer, so
    the trained space stays the one HNSW ranks by -- NOT an MLP score) by margin:

        hinge = relu(margin - (score(hi, d) - score(lo, d))).

    This supervises ORDER ("who is better on d"), which is what Kendall's tau
    measures -- unlike MSE on the narrow [0.6, 1.0] target range, where predicting
    the mean already looks good and the model is rewarded for collapsing.

    Scale-safe: at most `max_pairs_per_dataset` pairs are SAMPLED per dataset
    (never all-pairs), and datasets with < `min_models_per_dataset` models are
    skipped. Works unchanged on mini-batches -- it reads the (model, dataset,
    accuracy) triples the batch already carries. With return_stats=True also
    returns {"n_pairs", "n_datasets"} for reporting.
    """
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
        good = a[p] > a[q] + min_gap                # p strictly better than q
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
    """
    RankNet-style logistic ranking on the RAW dot score HNSW ranks (Phase 4).

        s(d, m) = <z_m, z_d>            # raw cosine of unit vectors -- NO scale/bias/sigmoid
        L_d     = mean softplus(-(s(d, hi) - s(d, lo)) / temperature)   over pairs in d
        L       = mean_d L_d                                            # macro over datasets

    Unlike perf_ranking_loss (hinge on sigmoid(scale*dot+bias)), this trains the
    EXACT geometry HNSW serves (the unit-vector inner product), so there is no
    monotone surrogate between the loss and the retrieval score.

    Pairs are formed WITHIN each dataset from the supervision triples the batch
    carries. Small candidate lists enumerate all unique non-tied pairs; large ones
    sample, biased toward HARD pairs (currently-inverted or near-tie under the
    model's own prediction) via `hard_frac`. `min_gap` treats |acc_hi-acc_lo| <=
    min_gap as a tie (skip) so measurement noise is not a strict preference.
    `gap_weighted` optionally scales each pair by its (capped) accuracy gap; the
    default is unweighted because Kendall gives every non-tied pair equal weight.
    `temperature` must be > 0.
    """
    assert temperature > 0, "temperature must be positive"
    z_m, z_d = z_dict["model"], z_dict["dataset"]
    src, dst, acc = edge_label_index[0], edge_label_index[1], edge_label.float()
    # raw dot per supervision edge (unit vectors -> cosine == inner product)
    s_all = (z_m[src] * z_d[dst]).sum(-1)

    losses, n_pairs, n_datasets = [], 0, 0
    for d in torch.unique(dst).tolist():
        sel = (dst == d).nonzero().flatten()
        n = int(sel.numel())
        if n < min_models_per_dataset:
            continue
        a = acc[sel]
        s = s_all[sel]
        # all ordered pairs (i better than j) with a strict accuracy gap
        ai, aj = a.unsqueeze(1), a.unsqueeze(0)
        better = (ai - aj) > min_gap            # [n, n]; i strictly better than j
        ii, jj = better.nonzero(as_tuple=True)
        if ii.numel() == 0:
            continue
        if ii.numel() > max_pairs_per_dataset:
            # hard-pair mining: rank candidate pairs by how INVERTED they are now
            # (s_hi - s_lo small or negative == hard); take a hard_frac head + random tail
            margin = (s[ii] - s[jj]).detach()
            n_hard = int(round(hard_frac * max_pairs_per_dataset))
            order = torch.argsort(margin)                  # most inverted first
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
    loss = torch.stack(losses).mean()              # macro over datasets (equal weight)
    if return_stats:
        return loss, {"n_pairs": n_pairs, "n_datasets": n_datasets}
    return loss


def split_trained_on(data, *, num_val=0.1, num_test=0.2, neg_ratio=1.0,
                     disjoint_train_ratio=0.3, seed=None):
    """
    Train/val/test split of `trained_on`, with `rev_trained_on` handed in so the
    reverse edges of held-out links are removed too (no label leakage). Returns
    (train_data, val_data, test_data); each carries reduced message-passing
    edges plus edge_label_index / edge_label supervision on the trained_on store.
    """
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


# ── L_contrast: supervision builders + InfoNCE ──────────────────────────────

def _union_find_components(edge_index, num_nodes):
    """Connected-component id per node over an (undirected) edge set."""
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
    """Same-hub mask: models in the same lineage component (>=2 members)."""
    comp = _union_find_components(data[IS_BASE_OF].edge_index, num_models)
    hub_mask = comp.unsqueeze(0) == comp.unsqueeze(1)
    hub_mask.fill_diagonal_(False)
    counts = torch.bincount(comp, minlength=num_models)
    singleton = counts[comp] <= 1            # a lone model is not a "shared hub"
    hub_mask[singleton, :] = False
    hub_mask[:, singleton] = False
    return hub_mask


def _resolve_trained_on(data, trained_on_index, trained_on_attr):
    if trained_on_index is None:
        return data[TRAINED_ON].edge_index, data[TRAINED_ON].edge_attr.float()
    return trained_on_index, trained_on_attr.float()


def contrastive_supervision(data, *, acc_thresh: float = 0.8,
                            trained_on_index=None, trained_on_attr=None):
    """
    GLOBAL-THRESHOLD positives (the original, kept as the experiment baseline).

    pos_mask[i, j] : models i and j both score >= acc_thresh on a common dataset.
    hub_mask[i, j] : models i and j share a lineage hub.

    On the zoo this makes ~74% of all model pairs positive at acc_thresh=0.8 --
    nearly no negatives, which is what drives embedding collapse. Prefer
    contrastive_supervision_topk for sparse, discriminative positives.
    """
    num_models = data["model"].num_nodes
    num_datasets = data["dataset"].num_nodes
    idx, attr = _resolve_trained_on(data, trained_on_index, trained_on_attr)

    M = torch.zeros(num_models, num_datasets)
    keep = attr >= acc_thresh
    M[idx[0][keep], idx[1][keep]] = 1.0
    pos_mask = (M @ M.t()) > 0
    pos_mask.fill_diagonal_(False)
    return pos_mask, _hub_mask(data, num_models)


def high_performer_membership(trained_on_index, trained_on_attr,
                              num_models, num_datasets, *, top_frac=0.1, top_k=None):
    """
    Per-dataset high-performer membership M[model, dataset].

    For each dataset, rank its models by trained_on accuracy and mark only the
    top fraction (top_frac) or top_k as high performers. This is RELATIVE to each
    dataset, so a dataset where everyone scores ~0.9 still yields a small, sharp
    positive set instead of admitting (almost) everyone the way a global
    accuracy threshold does.
    """
    M = torch.zeros(num_models, num_datasets)
    src, dst, acc = trained_on_index[0], trained_on_index[1], trained_on_attr.float()
    for d in torch.unique(dst).tolist():
        sel = dst == d
        models_d, accs_d = src[sel], acc[sel]
        n = int(models_d.numel())
        if n == 0:
            continue
        kk = min(int(top_k), n) if top_k is not None else max(1, int(round(top_frac * n)))
        top = torch.topk(accs_d, kk).indices
        M[models_d[top], d] = 1.0
    return M


def contrastive_supervision_topk(data, *, top_frac=0.1, top_k=None,
                                 trained_on_index=None, trained_on_attr=None):
    """
    PER-DATASET top-k/top-fraction positives (the redesign).

    pos_mask[i, j] : models i and j are BOTH among the top performers of a common
                     dataset -- a sparse, discriminative co-selection rather than
                     "both above a global threshold". hub_mask as before.
    """
    num_models = data["model"].num_nodes
    num_datasets = data["dataset"].num_nodes
    idx, attr = _resolve_trained_on(data, trained_on_index, trained_on_attr)
    M = high_performer_membership(idx, attr, num_models, num_datasets,
                                  top_frac=top_frac, top_k=top_k)
    pos_mask = (M @ M.t()) > 0
    pos_mask.fill_diagonal_(False)
    return pos_mask, _hub_mask(data, num_models)


def pair_density(pos_mask) -> float:
    """Fraction of off-diagonal model pairs that are positive (sparsity check)."""
    n = pos_mask.size(0)
    return float(pos_mask.sum()) / (n * (n - 1))


# ── 47K-safe supervision: carry membership M + components, build masks per batch ──
# A dense [N_model, N_model] pos/hub mask is fine on the 177-model zoo but is
# 2.2e9 entries at 47K. So the training path NEVER materializes it: it keeps the
# membership matrix M [N_model, N_dataset] and the lineage-component vector, and
# sampling.batch_contrastive_masks builds the small [B, B] masks for each batch.

def lineage_components(data, num_models):
    """Component id per model over is_base_of (singletons get unique roots, so
    comp[i]==comp[j] for i!=j is true ONLY for a real shared hub). A graph with no
    lineage edges yields all-singleton components (no same-hub structure)."""
    if IS_BASE_OF not in data.edge_types or data[IS_BASE_OF].num_edges == 0:
        return torch.arange(num_models)
    return _union_find_components(data[IS_BASE_OF].edge_index, num_models)


def topk_membership(data, *, top_frac=0.1, top_k=None,
                    trained_on_index=None, trained_on_attr=None):
    """Per-dataset top-fraction high-performer membership M [N_model, N_dataset].
    Pass TRAIN-visible trained_on_index/attr (never test edges)."""
    num_models = data["model"].num_nodes
    num_datasets = data["dataset"].num_nodes
    idx, attr = _resolve_trained_on(data, trained_on_index, trained_on_attr)
    return high_performer_membership(idx, attr, num_models, num_datasets,
                                     top_frac=top_frac, top_k=top_k)


def pool_membership_by_root(M, root_ids):
    """v6 S1: root-level positive pooling. M_pooled[:, d] = OR over all datasets
    d' sharing d's root of M[:, d'] -- so the global objective treats "a model
    good on ANY sibling of the root" as a positive of every root member. Teaches
    z_d a root-invariant model preference (the training-side dual of S2's
    serving-time sibling prior).

    Leakage-safe under root-aware splits: M is built from TRAIN-visible edges, so
    a test root's columns are all-zero and stay all-zero after pooling (their
    siblings are test too). Only train roots' positive sets grow.

    root_ids : LongTensor [N_dataset] integer root code per dataset (mappedID order).
    Returns M_pooled (same shape/dtype as M).
    """
    M = M.float()
    out = torch.zeros_like(M)
    for r in torch.unique(root_ids).tolist():
        cols = (root_ids == r).nonzero().flatten()
        if cols.numel() == 1:
            out[:, cols] = M[:, cols]                 # singleton root: unchanged
        else:
            pooled = (M[:, cols].sum(dim=1, keepdim=True) > 0).float()
            out[:, cols] = pooled                     # broadcast root-union to all members
    return (out > 0).to(M.dtype) if M.dtype != torch.float32 else out


def per_dataset_density(trained_on_index, M, *, min_models=2):
    """Mean within-dataset positive-pair density: for each dataset, C(k,2)/C(n,2)
    where n = models evaluated on it and k = those selected as top performers.
    Scale-safe (no N x N)."""
    dst = trained_on_index[1]
    ds = []
    for d in torch.unique(dst).tolist():
        n = int((dst == d).sum())
        if n < min_models:
            continue
        k = int(M[:, d].sum())
        ds.append((k * (k - 1)) / (n * (n - 1)) if n > 1 else 0.0)
    return float(np.mean(ds)) if ds else float("nan")


def global_positive_density(M, *, max_dense=2000, n_sample=200_000, generator=None):
    """Fraction of model pairs co-selected on >=1 dataset. Exact for small N,
    else estimated from `n_sample` random pairs (47K-safe)."""
    N = M.size(0)
    if N <= max_dense:
        P = (M @ M.t()) > 0
        P.fill_diagonal_(False)
        return float(P.sum()) / (N * (N - 1))
    i = torch.randint(N, (n_sample,), generator=generator)
    j = torch.randint(N, (n_sample,), generator=generator)
    keep = i != j
    co = (M[i[keep]] * M[j[keep]]).sum(-1) > 0
    return float(co.float().mean())


def uniformity_loss(z, *, n_pairs: int = 4096, t: float = 2.0, generator=None):
    """Wang-Isola uniformity on SAMPLED pairs (bounded -> 47K-safe). Lower = more
    spread. OPTIONAL anti-collapse regularizer; off by default (lambda_uniform=0)
    -- reported as a diagnostic, not tuned for zoo tau."""
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
    """
    Supervised-contrastive (InfoNCE) over model embeddings.

    For each anchor i with >= 1 positive, pull its co-high-performers together
    and push everything else away; same-hub non-positives get their denominator
    logit multiplied by `hard_neg_weight` (>1), so a family's derivatives are
    actively separated unless they are genuine co-performers.

    z_model assumed L2-normalized (it is -- the model's output head normalizes).
    Returns a scalar; 0 if no anchor has a positive (nothing to contrast).
    """
    N = z_model.size(0)
    sim = (z_model @ z_model.t()) / temperature
    sim.fill_diagonal_(float("-inf"))                      # exclude self

    weight = torch.ones_like(sim)
    weight[hub_mask & ~pos_mask] = hard_neg_weight         # hard negatives
    logits = sim + torch.log(weight.clamp_min(1e-12))      # w*exp(s) = exp(s+log w)
    denom = torch.logsumexp(logits, dim=1, keepdim=True)   # log sum_a w_ia exp(sim_ia)
    log_prob = sim - denom                                 # positives carry weight 1

    # sum log-prob over positives only; mask (not multiply) so the diagonal's
    # -inf never hits a 0 * -inf = NaN.
    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0
    if not bool(valid.any()):
        return z_model.new_zeros(())
    pos_log_prob = log_prob.masked_fill(~pos_mask, 0.0)
    loss_i = -pos_log_prob.sum(dim=1) / pos_count.clamp_min(1).float()
    return loss_i[valid].mean()


# ── combined ─────────────────────────────────────────────────────────────────

def dataset_to_model_contrastive(z_dict, trained_on_index, M, *,
                                 temperature: float = 0.1, hard_neg_weight: float = 1.0,
                                 min_pos: int = 1, return_stats: bool = False):
    """
    Phase 6: contrastive loss on the SERVING relation z_d -> z_m (InfoNCE per
    dataset query). The shipped contrastive_loss trains z_m -> z_m only; it never
    pulls a dataset query toward its good models. This does exactly that.

    For each dataset d:
      candidates = models with a TRAIN-VISIBLE trained_on edge to d (observed);
      positives  = those marked top-performer in M[:, d];
      negatives  = the remaining OBSERVED candidates (reliable negatives) --
                   unobserved pairs are NOT treated as negatives (missing != neg).
      loss_d = -log( sum_{p in pos} e^{s_dp/τ} / sum_{c in cand} w_c e^{s_dc/τ} )
    with s = <z_d, z_m> (the exact retrieval score). Same-candidate high-scoring
    negatives dominate the denominator naturally; hard_neg_weight (>1) optionally
    up-weights all negatives. Macro-averaged over datasets (equal weight).

    Leakage-safe: pass TRAIN-VISIBLE trained_on_index and a TRAIN-VISIBLE M.
    """
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
            continue                                   # need both positives and negatives
        s = (z_m[cand] @ z_d[d]) / temperature         # [n_cand]
        w = torch.ones_like(s)
        w[~pos_flag] = hard_neg_weight
        logits = s + torch.log(w.clamp_min(1e-12))
        denom = torch.logsumexp(logits, dim=0)
        num = torch.logsumexp(s[pos_flag], dim=0)      # positives carry weight 1
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
    """Phase 6, batch-friendly: dataset->model InfoNCE built directly from a
    batch's supervision triples (model, dataset, accuracy) -- works in the winning
    per-batch RankNet regime (no full-graph M needed).

    Per dataset d in the batch: candidates = its supervised models; positives =
    the top ceil(top_frac*n) by accuracy; negatives = the rest (observed only --
    missing pairs are never negatives). InfoNCE on s=<z_d,z_m>; macro over datasets.
    """
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
            continue                                   # need at least one negative
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


def build_global_negative_pools(task_type_id, trained_on_index, trained_on_attr, M, *,
                                include_known_low=False, low_frac=0.3):
    """Top-1/global guide Phase 1: RELIABLE global negative pools per dataset.

    task_type_id     : data['dataset'].task_type_id (0 = Other/unknown)
    trained_on_index : TRAIN-VISIBLE trained_on edges [2, E] (never val/test)
    trained_on_attr  : their normalized accuracies [E]
    M                : train-visible top-performer membership [N_model, N_dataset]

    Sources (both are *reliable*; unobserved-but-compatible models are UNKNOWN and
    never enter any pool -- missing performance is not evidence of poor performance):

      incompatible : models whose train-visible task-type profile is non-empty and
                     does not include dataset d's task type. Datasets with unknown
                     task (id 0) get NO incompatible negatives (unreliable).
      known_low    : (include_known_low=True, the G2 ingredient) models with a
                     train-visible edge to d whose accuracy is in the bottom
                     `low_frac` of d's list AND that are not top-performer
                     positives of d.

    Returns (pools, stats): pools = {dataset_idx: LongTensor of model ids};
    stats logs composition per source (guide: log the sampler composition).
    """
    src, dst, acc = trained_on_index[0], trained_on_index[1], trained_on_attr.float()
    # model -> set of observed task types (train-visible only)
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
        if td != 0:                                       # unknown task -> skip source
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
                low = [int(m) for m in low.tolist() if M[int(m), d] == 0]  # never positives
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
    """Sampled-softmax global retrieval term (guide Phase 1):

        L_global(d) = -(1/|P_d|) sum_{p in P_d} log
                       exp(s(d,p)/T) / sum_{a in P_d ∪ N_d} exp(s(d,a)/T)

    s = <z_d, z_m> on the FULL-graph z (the batch subgraph lacks global
    negatives). P_d = train-visible top performers (M[:, d]); N_d = a bounded
    sample from the reliable pool: `hard_frac` picked by highest current score
    (hard mining, G2) and the rest uniform (G1: hard_frac=0). Macro over sampled
    datasets. Unobserved models never appear (they are not in any pool).
    """
    z_m, z_d = z_dict["model"], z_dict["dataset"]
    ds_all = [d for d in pools if int(M[:, d].sum()) > 0]
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
        pos = (M[:, d] > 0).nonzero().flatten().to(z_m.device)
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
        # -(1/|P|) sum_p log softmax(p) == mean over positives of (denom - s_p)
        losses.append((denom - s[: pos.numel()]).mean())
        n_negs_used += int(neg.numel())
    loss = torch.stack(losses).mean()
    if return_stats:
        return loss, {"n_datasets": len(ds_batch), "n_negs_used": n_negs_used}
    return loss


def build_lake_logq(trained_on_index, num_models, *, alpha=0.75, n0=1.0):
    """v3 L1: the whole-lake proposal distribution q(m) for logQ-corrected
    sampled softmax.

        deg(m) = train-visible trained_on label degree of model m
        q(m)  ∝ (deg(m) + n0)^alpha

    alpha=0.75 tempers the head (word2vec convention); n0 keeps zero-degree
    models sampleable (the lake majority is unlabeled -- they ARE legitimate
    retrieval negatives under InfoNCE semantics, which is the v3 break from the
    G-phase 'reliable negatives only' rule: sampled-softmax negatives assert
    'less relevant than the positive on average', not 'known bad').

    Returns (q [N], logq [N]) on CPU.
    """
    deg = torch.bincount(trained_on_index[0], minlength=num_models).float()
    w = (deg + float(n0)) ** float(alpha)
    q = w / w.sum()
    return q, q.log()


def global_lake_loss(z_dict, M, q, logq, *, temperature=0.1, n_neg=64,
                     n_datasets=16, hard_sets=None, n_hard=16,
                     pos_ipw=None, generator=None, return_stats=False):
    """v3 L1/L2: full-lake sampled softmax with logQ correction.

        L(d) = -(1/|P_d|) sum_{p in P_d} log
               exp(s_p/T) / [ sum_{p' in P_d} exp(s_{p'}/T)
                              + sum_{n in S_d} exp(s_n/T - log q(n))
                              + sum_{h in H_d} exp(s_h/T) ]

    S_d ~ q over the ENTIRE lake (with replacement), positives filtered out.
    The -log q(n) logit correction (Bengio & Senecal sampled softmax) makes the
    denominator an unbiased estimate of the FULL-lake softmax: high-degree hubs
    are sampled often but individually down-weighted -- net effect, every
    dataset's loss pushes down every non-positive direction, including the
    unlabeled hubs the supervision blind spot (plan v3 §0.1 E9) never touched.

    H_d (optional, L2) = mined 'currently-beats-gold' negatives, uncorrected
    logits (mixed negative sampling convention): they are a targeted extra set,
    not draws from q; mixing breaks strict unbiasedness in exchange for putting
    the serving failure pairs directly into the objective.

    pos_ipw (optional, v4 L4) = per-model inverse-propensity weights [N_m],
    e.g. (deg(m)+1)^-beta. The POSITIVE side of the loss is hub-biased too:
    top-frac positive sets over-contain high-degree models, and every positive
    term drags the query direction toward them. Weighting the positive average
    by 1/(deg+1)^beta is the exact dual of the logQ negative correction --
    the last known popularity-bias source inside this objective. None = uniform
    (L1L3 behaviour, bit-identical).
    """
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
        pos = (M[:, d] > 0).nonzero().flatten().to(dev)
        samp = torch.multinomial(q_dev, n_neg, replacement=True, generator=generator)
        samp = samp[~torch.isin(samp, pos)]               # positives never negatives
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
    """v3 L2, run ONCE mid-training (anti-feedback-loop discipline, same as the
    plan's P3 rule): for every supervised dataset, the current top-`hard_k`
    scoring models on the TRAIN message graph, minus the train-visible
    positives -- i.e. exactly 'the models currently beating the labeled best'.
    Returns {dataset_idx: LongTensor[<=hard_k]} (CPU)."""
    z_m, z_d = z_dict["model"].detach(), z_dict["dataset"].detach()
    hard = {}
    for d in (M.sum(0) > 0).nonzero().flatten().tolist():
        pos = (M[:, d] > 0)
        s = z_m @ z_d[d]
        s[pos] = float("-inf")                    # positives can't be hard negatives
        k = min(hard_k, int((~pos).sum()))
        hard[d] = torch.topk(s, k).indices.cpu()
    return hard


def mine_alibi_hard_negative_sets(z_dict, M, deg, dataset_task_id, model_tasks, *,
                                  hard_k=20, deg_quantile=0.9, overshoot=3):
    """v4 L2b: hard negatives WITH AN ALIBI -- the fix for L2's self-sabotage.

    L2 mined 'models currently beating the labeled best' and excluded only the
    labeled positives; when the TRUE gold (unlabeled, low-degree, task-matching)
    had already been pushed into the top-k by L1, L2 pushed it back down.

    L2b keeps a mined model m for dataset d only if there is EVIDENCE it is a
    popularity artifact rather than hidden gold:

        alibi(m, d) = deg(m) >= P{deg_quantile}(deg | labeled models)   (hub suspect)
                    ∨ (model_tasks[m] non-empty ∧ task(d) known
                       ∧ task(d) ∉ model_tasks[m])                      (task mismatch)

    A low-degree model whose observed task profile is compatible with d -- the
    exact signature of hidden gold -- is never mined. Mining goes `overshoot`×
    deeper than hard_k so the filter has candidates to reject.

    deg          : [N_m] train-visible label degree (build once from ti)
    dataset_task_id : [N_d] (repaired) task ids, 0 = unknown
    model_tasks  : {model_idx: set(task_id)} train-visible task profile
    Returns {dataset_idx: LongTensor[<=hard_k]} (CPU).
    """
    z_m, z_d = z_dict["model"].detach(), z_dict["dataset"].detach()
    lab = deg[deg > 0].float()
    thresh = torch.quantile(lab, deg_quantile) if lab.numel() else torch.tensor(1.0)
    hard = {}
    for d in (M.sum(0) > 0).nonzero().flatten().tolist():
        pos = (M[:, d] > 0)
        s = z_m @ z_d[d]
        s[pos] = float("-inf")
        k_mine = min(hard_k * overshoot, int((~pos).sum()))
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
    """{model_idx: set of observed dataset task ids (known only)} from
    train-visible trained_on edges -- the same profile build_global_negative_pools
    uses, exposed for L2b's alibi filter."""
    prof = {}
    for m, d in zip(trained_on_index[0].tolist(), trained_on_index[1].tolist()):
        t = int(task_type_id[d])
        if t != 0:
            prof.setdefault(m, set()).add(t)
    return prof


def dataset_push_apart_loss(z_d, task_type_id, *, margin=0.2, n_anchor=32,
                            n_neg=16, generator=None, return_stats=False):
    """v3 Z2: dataset-dataset repulsion for z_d conditioning.

        L = mean_{a, n : task(a) != task(n), both known}  relu(<z_a, z_n> - margin)

    Today NOTHING pushes two datasets apart: similar_to edges only pull
    together, trained_on supervision shapes z_d only through shared models --
    z_d intrinsic dim ~= 2 is the direct consequence of this one-directional
    force (plan v3 §0.3). This term makes different-TASK dataset pairs repel
    until their cosine falls below `margin`; same-task pairs and datasets with
    unknown task (id 0, post-L3 ~28%) are never touched (no false repulsion).
    Hinge rather than InfoNCE: repulsion-only by design -- we assert different
    tasks should be separable, NOT that same-task datasets are interchangeable.
    """
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