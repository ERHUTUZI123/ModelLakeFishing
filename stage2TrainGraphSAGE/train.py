"""
train.py -- Stage 2, Step 5: training loop + validation on the REAL graph.

Runs on Stage 1's hgraph_zoo_xm0.pt (true x_m^(0): frozen [e_name||e_desc] plus
the size/family index columns) -- NOT a random smoke graph. load_hgraph already
points there; __main__ asserts the real-feature signature (frozen dim 448,
num_families 136) before training.

Two gates, in the plan's order:

  MECHANISM (must pass): combined loss descends; both learnable tables get
    nonzero gradient; frozen x gets none; checkpoint saves and reloads to the
    same z. These are hard-asserted.

  EFFECT (reported): held-out L_perf + Kendall's tau on test trained_on edges;
    same-hub cosine distance (over-smoothing watch -- distances trending to 0
    means a family collapsed to a point); HNSW recall@50 vs brute force, split
    near-hub / away-hub (needs hnswlib, else skipped with a note); a task-coloured
    2D scatter (UMAP if installed, else PCA fallback).

Run:  python -m ModelLakeFishing.stage2TrainGraphSAGE.train
"""

import os
import sys

import numpy as np
import torch
from scipy.stats import kendalltau

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch.nn.functional as F  # noqa: E402

from ModelLakeFishing.stage2TrainGraphSAGE.model import HeteroGraphSAGE, load_hgraph  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.losses import (  # noqa: E402
    TRAINED_ON, PerfScorer, accuracy_lookup, perf_supervision, perf_loss,
    perf_ranking_loss, raw_dot_ranknet_loss, split_trained_on, topk_membership,
    lineage_components, per_dataset_density, global_positive_density,
    contrastive_loss, dataset_to_model_contrastive,
    dataset_to_model_contrastive_from_edges, uniformity_loss,
    global_retrieval_loss, global_lake_loss, mine_hard_negative_sets,
    mine_alibi_hard_negative_sets, dataset_push_apart_loss,
)
from ModelLakeFishing.stage2TrainGraphSAGE.sampling import (  # noqa: E402
    make_link_loader, apply_edge_dropout, batch_contrastive_masks,
)
from ModelLakeFishing.stage2TrainGraphSAGE.learnable import (  # noqa: E402
    save_checkpoint, load_checkpoint,
)

ARTIFACTS = os.path.join(_HERE, "artifacts")


# ── training ─────────────────────────────────────────────────────────────────

def _val_tau_macro(model, scorer, val_data, lookup, device):
    """Within-dataset macro Kendall tau on a val split (for early stopping)."""
    from ModelLakeFishing.stage2TrainGraphSAGE.eval_harness import per_dataset_tau
    model.eval()
    with torch.no_grad():
        z = {k: v for k, v in model(val_data.clone().to(device)).items()}
    macro, _ = per_dataset_tau(scorer, z, val_data, lookup)
    return macro if macro == macro else -1.0          # NaN -> -1


def train(model, scorer, train_data, eli, target, M, comp, *,
          epochs=40, lr=1e-2, lambda_mse=0.0, lambda_rank=1.0, lambda_contrast=1.0,
          lambda_uniform=0.0, rank_margin=0.05, p=0.3, p_lineage=0.05,
          num_neighbors=(10, 10), batch_size=128, device=None,
          rank_loss="hinge", rank_temperature=0.1, rank_min_gap=0.0,
          rank_gap_weighted=False,
          val_data=None, val_lookup=None, patience=0, eval_every=1,
          lambda_dm_contrast=0.0, dm_temperature=0.1, dm_top_frac=0.10,
          global_ctx=None, zpush_ctx=None):
    """
    Per batch: lambda_rank * ranking + lambda_contrast * contrast (+ optional
    lambda_mse * MSE, + optional lambda_uniform * uniformity). Ranking (not MSE)
    is the default perf term -- it supervises ORDER (what Kendall's tau scores);
    MSE on the narrow [0.6,1.0] target range rewards predicting the mean
    (collapse). Contrastive masks are built PER BATCH from (M, comp) -- 47K-safe,
    no global N x N. lambda_uniform defaults to 0 (anti-collapse regularizer is a
    diagnostic, not tuned here). One optimizer; epochs/lambda kept modest.
    """
    if device is None:
        device = next(model.parameters()).device
    opt = torch.optim.Adam(list(model.parameters()) + list(scorer.parameters()), lr=lr)
    loader = make_link_loader(train_data, eli, target,
                              num_neighbors=num_neighbors, batch_size=batch_size)
    history = []
    # ── Top-1 guide Phase 1: global sampled-softmax negatives ────────────────
    # global_ctx = {pools, M, lambda_g, temperature, n_neg, n_datasets, hard_frac}
    # The global term scores EVERY indexed model, so it needs a full-graph
    # forward per step (the batch subgraph lacks the global negatives). The full
    # train message graph gets its own edge dropout each step.
    use_global = global_ctx is not None and global_ctx.get("lambda_g", 0.0) > 0
    # v3 Z2: dataset-dataset push-apart also needs a full-graph z per step
    use_zpush = zpush_ctx is not None and zpush_ctx.get("lambda_zp", 0.0) > 0
    if use_global or use_zpush:
        g_base = train_data.clone().to(device)
    if use_global:
        g_M = global_ctx["M"].to(device)
        # v3 L1: "lake" mode replaces the reliable pools with whole-lake logQ
        # sampling; L2 adds hard sets mined ONCE at hard_mine_epoch (>=1).
        use_lake = "lake" in global_ctx
        hard_sets = None
        hard_mine_epoch = int(global_ctx.get("hard_mine_epoch", 0) or 0)
    use_val = val_data is not None and val_lookup is not None
    best_val, best_state, bad_epochs = -2.0, None, 0
    import copy
    for epoch in range(epochs):
        model.train()
        # v3 L2: mine 'currently-beats-the-labeled-best' negatives EXACTLY once,
        # from a clean (no-dropout) train-graph forward -- never re-mined, so no
        # self-reinforcing feedback loop (plan v3 §0.3, P3-style discipline).
        if (use_global and use_lake and hard_mine_epoch > 0
                and epoch == hard_mine_epoch and hard_sets is None):
            model.eval()
            with torch.no_grad():
                z_mine = model(g_base.clone())
            alibi = global_ctx.get("alibi")
            if alibi is not None:
                # v4 L2b: only mined models with a popularity/task-mismatch
                # alibi survive -- hidden gold (low-degree, task-compatible)
                # is never pushed down (the L2 self-sabotage fix)
                hard_sets = mine_alibi_hard_negative_sets(
                    z_mine, g_M, alibi["deg"], alibi["dataset_task_id"],
                    alibi["model_tasks"], hard_k=global_ctx.get("hard_k", 20),
                    deg_quantile=alibi.get("deg_quantile", 0.9))
                tag = "alibi hard-mine"
            else:
                hard_sets = mine_hard_negative_sets(
                    z_mine, g_M, hard_k=global_ctx.get("hard_k", 20))
                tag = "lake hard-mine"
            model.train()
            print(f"    [{tag} @ep{epoch}] datasets={len(hard_sets)} "
                  f"k={global_ctx.get('hard_k', 20)}")
        ep = []
        for batch in loader:
            batch = batch.to(device)                    # GPU: sample on CPU, train on device
            apply_edge_dropout(batch, p=p, p_lineage=p_lineage)
            opt.zero_grad()
            z = model(batch)
            eli_b, el_b = batch[TRAINED_ON].edge_label_index, batch[TRAINED_ON].edge_label
            pb, hb = batch_contrastive_masks(batch, M, comp)

            parts, total = {}, z["model"].new_zeros(())
            if lambda_mse > 0:
                lmse = perf_loss(scorer, z, eli_b, el_b)
                total = total + lambda_mse * lmse
                parts["mse"] = float(lmse.detach())
            if lambda_rank > 0:
                if rank_loss == "ranknet":
                    lrank = raw_dot_ranknet_loss(
                        z, eli_b, el_b, temperature=rank_temperature,
                        min_gap=rank_min_gap, gap_weighted=rank_gap_weighted)
                else:
                    lrank = perf_ranking_loss(scorer, z, eli_b, el_b, margin=rank_margin)
                total = total + lambda_rank * lrank
                parts["rank"] = float(lrank.detach())
            lc = contrastive_loss(z["model"], pb, hb)
            total = total + lambda_contrast * lc
            parts["contrast"] = float(lc.detach())
            if lambda_dm_contrast > 0:
                ldm = dataset_to_model_contrastive_from_edges(
                    z, eli_b, el_b, top_frac=dm_top_frac, temperature=dm_temperature)
                total = total + lambda_dm_contrast * ldm
                parts["dm_contrast"] = float(ldm.detach())
            if use_global or use_zpush:
                g_graph = g_base.clone()
                apply_edge_dropout(g_graph, p=p, p_lineage=p_lineage)
                z_full = model(g_graph)
            if use_zpush:
                lzp = dataset_push_apart_loss(
                    z_full["dataset"], g_base["dataset"].task_type_id,
                    margin=zpush_ctx.get("margin", 0.2),
                    n_anchor=zpush_ctx.get("n_anchor", 32),
                    n_neg=zpush_ctx.get("n_neg", 16))
                total = total + zpush_ctx["lambda_zp"] * lzp
                parts["zpush"] = float(lzp.detach())
            if use_global:
                if use_lake:
                    q, logq = global_ctx["lake"]
                    lg = global_lake_loss(
                        z_full, g_M, q, logq,
                        temperature=global_ctx.get("temperature", 0.1),
                        n_neg=global_ctx.get("n_neg", 64),
                        n_datasets=global_ctx.get("n_datasets", 16),
                        hard_sets=hard_sets,
                        n_hard=global_ctx.get("n_hard", 16),
                        pos_ipw=global_ctx.get("pos_ipw"))
                else:
                    lg = global_retrieval_loss(
                        z_full, g_M, global_ctx["pools"],
                        temperature=global_ctx.get("temperature", 0.1),
                        n_neg=global_ctx.get("n_neg", 64),
                        n_datasets=global_ctx.get("n_datasets", 16),
                        hard_frac=global_ctx.get("hard_frac", 0.0))
                total = total + global_ctx["lambda_g"] * lg
                parts["global"] = float(lg.detach())
            if lambda_uniform > 0:
                lu = uniformity_loss(z["model"])
                total = total + lambda_uniform * lu
                parts["uniform"] = float(lu.detach())
            parts["total"] = float(total.detach())

            total.backward()
            opt.step()
            ep.append(parts)
        history.append({k: float(np.mean([d[k] for d in ep])) for k in ep[0]})
        # ── Phase 7: select the best-validation epoch (early stopping) ──────────
        if use_val and (epoch % eval_every == 0 or epoch == epochs - 1):
            v = _val_tau_macro(model, scorer, val_data, val_lookup, device)
            history[-1]["val_tau"] = v
            if v > best_val:
                best_val, bad_epochs = v, 0
                best_state = (copy.deepcopy(model.state_dict()),
                              copy.deepcopy(scorer.state_dict()))
            else:
                bad_epochs += 1
                if patience and bad_epochs >= patience:
                    break
            model.train()
    if use_val and best_state is not None:
        model.load_state_dict(best_state[0])
        scorer.load_state_dict(best_state[1])
    return history, loader


def _full_masks(M, comp):
    """Dense [N,N] pos/hub masks from membership M and components comp (used by
    full-batch macro training; fine at the 2000-model scale, NOT at 47K)."""
    pos = (M @ M.t()) > 0
    pos.fill_diagonal_(False)
    hub = comp.unsqueeze(0) == comp.unsqueeze(1)
    hub.fill_diagonal_(False)
    counts = torch.bincount(comp, minlength=comp.numel())
    singleton = counts[comp] <= 1
    hub[singleton, :] = False
    hub[:, singleton] = False
    return pos, hub


def train_grouped(model, scorer, train_data, eli, target, M, comp, *,
                  epochs=40, lr=1e-2, lambda_mse=0.0, lambda_rank=1.0, lambda_contrast=1.0,
                  lambda_uniform=0.0, rank_margin=0.05, p=0.3, p_lineage=0.05, device=None,
                  rank_loss="ranknet", rank_temperature=0.1, rank_min_gap=0.0,
                  rank_gap_weighted=False, log_every=0,
                  ti=None, lambda_dm_contrast=0.0, dm_temperature=0.1,
                  dm_hard_neg_weight=1.0, dm_warmup=0):
    """Phase 3: dataset-grouped, MACRO-BALANCED training (every dataset equal weight).

    Instead of globally-shuffled edge batches (where pair-rich datasets dominate
    the gradient while tau_macro weights datasets equally), this does ONE full-graph
    step per epoch over ALL supervision edges, and the ranking loss macro-averages
    over datasets (raw_dot_ranknet_loss / perf_ranking_loss both compute one mean
    per dataset, then mean over datasets). Edge dropout is applied to a fresh clone
    of the message graph each epoch (cold-start hardening preserved). Exact macro
    balance at the 2000-model scale; for 47K, switch to a sampled dataset-grouped
    loader (same per-dataset-then-mean reduction).

    Returns (history, stats) where stats logs datasets/pairs contributing per epoch.
    """
    if device is None:
        device = next(model.parameters()).device
    opt = torch.optim.Adam(list(model.parameters()) + list(scorer.parameters()), lr=lr)
    base = train_data.clone().to(device)
    eli, target = eli.to(device), target.to(device)
    M, comp = M.to(device), comp.to(device)
    if ti is not None:
        ti = ti.to(device)
    pos_mask, hub_mask = _full_masks(M, comp)
    history, last_stats = [], {}
    for epoch in range(epochs):
        model.train()
        g = base.clone()
        apply_edge_dropout(g, p=p, p_lineage=p_lineage)
        opt.zero_grad()
        z = model(g)
        parts, total = {}, z["model"].new_zeros(())
        if lambda_mse > 0:
            lmse = perf_loss(scorer, z, eli, target)
            total = total + lambda_mse * lmse
            parts["mse"] = float(lmse.detach())
        if lambda_rank > 0:
            if rank_loss == "ranknet":
                lrank, st = raw_dot_ranknet_loss(
                    z, eli, target, temperature=rank_temperature, min_gap=rank_min_gap,
                    gap_weighted=rank_gap_weighted, return_stats=True)
            else:
                lrank, st = perf_ranking_loss(scorer, z, eli, target, margin=rank_margin,
                                              return_stats=True)
            total = total + lambda_rank * lrank
            parts["rank"] = float(lrank.detach())
            last_stats = st
        lc = contrastive_loss(z["model"], pos_mask, hub_mask)
        total = total + lambda_contrast * lc
        parts["contrast"] = float(lc.detach())
        if lambda_dm_contrast > 0 and ti is not None:
            # ranking warm-up: anneal the dataset->model contrastive weight from 0
            ramp = min(1.0, (epoch + 1) / dm_warmup) if dm_warmup > 0 else 1.0
            ldm = dataset_to_model_contrastive(
                z, ti, M, temperature=dm_temperature, hard_neg_weight=dm_hard_neg_weight)
            total = total + (lambda_dm_contrast * ramp) * ldm
            parts["dm_contrast"] = float(ldm.detach())
        if lambda_uniform > 0:
            lu = uniformity_loss(z["model"])
            total = total + lambda_uniform * lu
            parts["uniform"] = float(lu.detach())
        parts["total"] = float(total.detach())
        total.backward()
        opt.step()
        history.append(parts)
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"      [grouped ep{epoch}] total={parts['total']:.4f} "
                  f"rank={parts.get('rank', 0):.4f} datasets={last_stats.get('n_datasets')} "
                  f"pairs={last_stats.get('n_pairs')}")
    return history, last_stats


def collapse_report(z_model) -> float:
    """Mean off-diagonal pairwise cosine. ~1.0 = collapsed. O(N*d), not O(N^2):
    sum_{i!=j}<zi,zj> = ||sum_i zi||^2 - N (z unit-norm), so it scales to 47K."""
    z = F.normalize(z_model, p=2, dim=-1)
    N = z.size(0)
    s = z.sum(0)
    return float((s @ s - N) / (N * (N - 1)))


def eval_perf(model, scorer, split_data, lookup, *, min_per_dataset=3):
    """Held-out metrics on a split's trained_on edges.

    kendall_tau       : pooled over ALL (model, dataset) pairs (cross-dataset
                        scale conflated -- a blunt number).
    kendall_tau_macro : mean of WITHIN-dataset Kendall's tau (datasets with
                        >= min_per_dataset scorable edges). This is what the
                        ranking loss optimizes and what retrieval needs: for a
                        given dataset, are the models ranked correctly.
    """
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        z = model(split_data.clone().to(dev))           # clone: .to() is in-place in PyG
        eli, target = perf_supervision(split_data[TRAINED_ON], lookup)
        eli, target = eli.to(dev), target.to(dev)
        pred = scorer(z["model"], z["dataset"], eli)
        mse = torch.mean((pred - target) ** 2).item()
    pr, tg, ds = pred.cpu().numpy(), target.cpu().numpy(), eli[1].cpu().numpy()
    tau, _ = kendalltau(pr, tg)
    taus = []
    for d in np.unique(ds):
        m = ds == d
        if m.sum() >= min_per_dataset and np.std(tg[m]) > 0 and np.std(pr[m]) > 0:
            t, _ = kendalltau(pr[m], tg[m])
            if not np.isnan(t):
                taus.append(t)
    tau_macro = float(np.mean(taus)) if taus else float("nan")
    return {"mse": mse, "kendall_tau": float(tau), "kendall_tau_macro": tau_macro,
            "n": int(eli.size(1)), "n_datasets_scored": len(taus)}


# ── effect diagnostics ───────────────────────────────────────────────────────

def oversmoothing_report(z_model, comp, *, max_pairs=20000, generator=None):
    """Mean cosine DISTANCE among same-hub derivatives vs a random-pair baseline.

    Same-hub distance trending to ~0 (much below random) is the over-smoothing
    signature: a family's derivatives collapsed onto a point. Same-hub pairs come
    from the lineage components; random pairs are SAMPLED (bounded) so this is
    47K-safe rather than dense N x N.
    """
    import collections
    z = F.normalize(z_model, p=2, dim=-1)
    N = z.size(0)
    groups = collections.defaultdict(list)
    for i, c in enumerate(comp.tolist()):
        groups[c].append(i)
    same = [(g[a], g[b]) for g in groups.values() if len(g) > 1
            for a in range(len(g)) for b in range(a + 1, len(g))]
    if len(same) > max_pairs:
        idx = torch.randperm(len(same), generator=generator)[:max_pairs]
        same = [same[k] for k in idx.tolist()]
    if same:
        sp = torch.tensor(same)
        same_dist = float((1.0 - (z[sp[:, 0]] * z[sp[:, 1]]).sum(-1)).mean())
    else:
        same_dist = float("nan")
    i = torch.randint(N, (max_pairs,), generator=generator)
    j = torch.randint(N, (max_pairs,), generator=generator)
    keep = i != j
    rand_dist = float((1.0 - (z[i[keep]] * z[j[keep]]).sum(-1)).mean())
    return {"n_same_hub_pairs": len(same), "same_hub_mean_dist": same_dist,
            "random_mean_dist": rand_dist}


def hnsw_recall(z_model, near_hub, k=50):
    """HNSW top-k vs brute-force top-k recall, split near-hub / away. Needs hnswlib."""
    try:
        import hnswlib
    except Exception:
        return None
    z = torch.nn.functional.normalize(z_model, p=2, dim=-1).numpy().astype("float32")
    N = z.shape[0]
    k = min(k, N - 1)
    index = hnswlib.Index(space="cosine", dim=z.shape[1])
    index.init_index(max_elements=N, ef_construction=200, M=16)
    index.add_items(z, np.arange(N))
    index.set_ef(max(64, k + 16))
    sim = z @ z.T
    np.fill_diagonal(sim, -np.inf)
    brute = np.argsort(-sim, axis=1)[:, :k]
    labels, _ = index.knn_query(z, k=k + 1)
    out = {}
    for name, mask in (("near_hub", near_hub.numpy()), ("away_hub", (~near_hub).numpy())):
        if mask.sum() == 0:
            out[name] = float("nan")
            continue
        recs = []
        for i in np.where(mask)[0]:
            ann = set(int(x) for x in labels[i] if int(x) != i)
            recs.append(len(ann & set(brute[i].tolist())) / k)
        out[name] = float(np.mean(recs))
    return out


def task_scatter(z_model, model_task, path):
    """2D scatter of model embeddings coloured by best-performing dataset (task).
    UMAP if available, else PCA fallback."""
    z = torch.nn.functional.normalize(z_model, p=2, dim=-1).numpy()
    try:
        import umap
        xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine").fit_transform(z)
        method = "UMAP"
    except Exception:
        from sklearn.decomposition import PCA
        xy = PCA(n_components=2).fit_transform(z)
        method = "PCA(fallback)"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(xy[:, 0], xy[:, 1], c=model_task, cmap="tab20", s=18)
    plt.title(f"model z_m by best task  ({method})")
    plt.colorbar(sc, label="best-task dataset id (-1 = none)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return method


def best_task_per_model(trained_index, trained_attr, num_models):
    """For each model, the dataset id it performs best on (-1 if none seen)."""
    best = np.full(num_models, -1, dtype=int)
    best_acc = np.full(num_models, -np.inf)
    for m, d, a in zip(trained_index[0].tolist(), trained_index[1].tolist(),
                       trained_attr.tolist()):
        if a > best_acc[m]:
            best_acc[m] = a
            best[m] = d
    return best


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    data, xm0, umi = load_hgraph()
    # guard: this must be the REAL xm0 graph, not a random smoke graph
    assert "xm0_meta" not in (None,) and xm0.get("num_families") == 136, "expected real xm0_meta"
    assert data["model"].x.shape[1] == 448, "expected frozen dim 448 (e_name||e_desc)"
    print(f"graph: {data['model'].num_nodes} models, {data['dataset'].num_nodes} datasets, "
          f"frozen dim {data['model'].x.shape[1]}, families {xm0['num_families']}")

    failures = []

    def check(cond, msg):
        print(f"  [{'OK  ' if cond else 'FAIL'}] {msg}")
        if not cond:
            failures.append(msg)

    # supervision split (with reverse edges) + targets + contrastive masks built
    # from TRAIN-VISIBLE performance only (no val/test leakage into geometry)
    train_data, val_data, test_data = split_trained_on(data, seed=0)
    lookup = accuracy_lookup(data)
    eli, target = perf_supervision(train_data[TRAINED_ON], lookup)
    ti = torch.cat([train_data[TRAINED_ON].edge_index, eli], dim=1)
    ta = torch.cat([train_data[TRAINED_ON].edge_attr.float(), target], dim=0)
    # redesigned supervision (47K-safe): per-dataset top-fraction membership M
    # + lineage components comp; per-batch masks are built from these, never a
    # global N x N. Built from TRAIN-visible edges only (no test leakage).
    M = topk_membership(data, top_frac=0.1, trained_on_index=ti, trained_on_attr=ta)
    comp = lineage_components(data, data["model"].num_nodes)
    print(f"positive-pair density: global {100 * global_positive_density(M):.1f}%  "
          f"per-dataset {100 * per_dataset_density(ti, M):.1f}%  (global-threshold baseline ~74%)")

    model = HeteroGraphSAGE(
        metadata=data.metadata(),
        frozen_dim=data["model"].x.shape[1],
        num_size_buckets=xm0["num_size_buckets"],
        num_families=xm0["num_families"],
        dataset_in_dim=data["dataset"].x.shape[1],
    )
    scorer = PerfScorer(dim=128, mode="dot")

    # ── MECHANISM: one batch -> grad boundary ────────────────────────────────
    print("\n=== MECHANISM: gradient boundary (one batch) ===")
    loader0 = make_link_loader(train_data, eli, target, batch_size=128)
    b0 = next(iter(loader0))
    apply_edge_dropout(b0, p=0.3, p_lineage=0.05)
    z0 = model(b0)
    lp0 = perf_loss(scorer, z0, b0[TRAINED_ON].edge_label_index, b0[TRAINED_ON].edge_label)
    pb0, hb0 = batch_contrastive_masks(b0, M, comp)
    (lp0 + contrastive_loss(z0["model"], pb0, hb0)).backward()
    se = model.model_encoder.size_embedding.weight.grad
    fe = model.model_encoder.family_embedding.weight.grad
    check(se is not None and se.abs().sum() > 0, "size_embedding got nonzero grad")
    check(fe is not None and fe.abs().sum() > 0, "family_embedding got nonzero grad")
    check(b0["model"].x.grad is None, "frozen model.x got NO grad")

    # ── MECHANISM: training descends (ranking + contrast, modest epochs) ─────
    print("\n=== MECHANISM: combined loss descends ===")
    history, _ = train(model, scorer, train_data, eli, target, M, comp,
                       epochs=40, lambda_rank=1.0, lambda_contrast=1.0)
    print(f"      epoch 0 : {history[0]}")
    print(f"      epoch 39: {history[-1]}")
    check(history[-1]["total"] < history[0]["total"], "combined loss decreased over 40 epochs")
    # rank / contrast descent are EFFECT signals (reported, not gated): the
    # supervision redesign is meant to make them MEANINGFUL, not to pass a bar.
    print(f"      (rank {history[0]['rank']:.4f}->{history[-1]['rank']:.4f}, "
          f"contrast {history[0]['contrast']:.3f}->{history[-1]['contrast']:.3f})")

    # ── MECHANISM: checkpoint save / load reproduces z ───────────────────────
    print("\n=== MECHANISM: checkpoint save / load ===")
    model.eval()
    with torch.no_grad():
        z_ref = model(test_data)["model"].clone()
    ckpt = os.path.join(ARTIFACTS, "stage2_zoo.pt")
    save_checkpoint(model, xm0["family_vocab"], ckpt)
    model2, vocab2, repro = load_checkpoint(ckpt)
    with torch.no_grad():
        z_re = model2(test_data)["model"]
    check(torch.allclose(z_ref, z_re, atol=1e-6), "reloaded model reproduces z exactly")
    stem = os.path.splitext(os.path.basename(ckpt))[0]
    check(os.path.exists(os.path.join(ARTIFACTS, f"{stem}.family_vocab.csv")),
          "sidecar family_vocab.csv written next to checkpoint")
    check(vocab2 == xm0["family_vocab"], "reloaded family_vocab matches xm0")

    # ── EFFECT: held-out perf + Kendall's tau ────────────────────────────────
    print("\n=== EFFECT: held-out trained_on (val / test) ===")
    val_m = eval_perf(model, scorer, val_data, lookup)
    test_m = eval_perf(model, scorer, test_data, lookup)
    print(f"      val : tau_pool={val_m['kendall_tau']:.3f}  tau_macro={val_m['kendall_tau_macro']:.3f}  (n={val_m['n']})")
    print(f"      test: tau_pool={test_m['kendall_tau']:.3f}  tau_macro={test_m['kendall_tau_macro']:.3f}  (n={test_m['n']})")
    print("      (MSE not reported as quality: ranking objective does not calibrate absolute scale)")

    # ── EFFECT: structural diagnostics on the full-graph embeddings ──────────
    print("\n=== EFFECT: collapse + over-smoothing + retrieval ===")
    with torch.no_grad():
        z_full = model(data)["model"]
    mean_cos = collapse_report(z_full)
    print(f"      mean pairwise cosine={mean_cos:.4f}  (target <0.9 for well-spread; "
          f"collapse if ~1.0 -- reduced from ~0.95/0.9999 baselines but NOT yet solved)")
    counts = torch.bincount(comp, minlength=data["model"].num_nodes)
    near_hub = counts[comp] > 1
    osm = oversmoothing_report(z_full, comp)
    print(f"      same-hub mean cos-dist={osm['same_hub_mean_dist']:.4f} vs "
          f"random={osm['random_mean_dist']:.4f}  ({osm['n_same_hub_pairs']} pairs)")
    if osm["n_same_hub_pairs"] > 0:
        check(osm["same_hub_mean_dist"] > 0.05 * osm["random_mean_dist"],
              "same-hub embeddings not collapsed (no severe over-smoothing)")

    rec = hnsw_recall(z_full, near_hub, k=50)
    if rec is None:
        print("      HNSW recall@50: SKIPPED (hnswlib missing; `pip install hnswlib`)")
    else:
        print(f"      HNSW recall@50  near_hub={rec['near_hub']:.3f}  away_hub={rec['away_hub']:.3f}")

    # ── EFFECT: task-coloured 2D scatter ─────────────────────────────────────
    best = best_task_per_model(ti, ta, data["model"].num_nodes)
    png = os.path.join(ARTIFACTS, "task_scatter.png")
    method = task_scatter(z_full, best, png)
    print(f"      wrote {png}  ({method})")

    print("\n" + "=" * 56)
    if failures:
        print(f"MECHANISM FAILED -- {len(failures)} check(s):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("MECHANISM OK on real xm0 graph; EFFECT metrics reported above.")
