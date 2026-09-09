"""
visualize_trained.py -- Stage-2 counterpart of stage1/visualize_hgraph.py.

stage1's visualizer draws the RAW transfer graph on a synthetic circle. This one
draws the TRAINED RESULT: node positions are the GraphSAGE embeddings z_m / z_d
(the locked candidate, trained on the PRODUCTION diverse graph hgraph_diverse_xd0.pt
-- xm0 model features + xd0 dataset features) projected
to 2D. So the figure shows what training actually did to the geometry -- which
models cluster, whether lineage families stay together, how near each model lands
to the dataset it performs best on, and whether the space collapsed.

It REUSES stage1 visualize_hgraph helpers (dataset palette, primary-dataset
anchoring, bezier arcs, dark theme) so the two figures read consistently; only the
node coordinates change (synthetic circle -> trained embedding projection).

Embeddings come from artifacts/stage2_diverse_xd0_candidate.pt (1-layer candidate).
Pass --train to (re)train the candidate first, or --pt/--ckpt for legacy baselines.

Run with the venv python that has torch, e.g.:
  ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.stage2TrainGraphSAGE.visualize_trained
  ... visualize_trained --show
"""

import argparse
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_STAGE1 = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage1BuildTransferGraph")
for _p in (_REPO_ROOT, _STAGE1):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ModelLakeFishing.stage1BuildTransferGraph import visualize_hgraph as vh  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.model import load_hgraph  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.learnable import load_checkpoint  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.losses import lineage_components  # noqa: E402

ARTIFACTS = os.path.join(_HERE, "artifacts")
# PRODUCTION DEFAULT: diverse zoo + xm0 + xd0 (see artifacts/PRODUCTION.md).
# Legacy graphs/checkpoints remain runnable via --pt / --ckpt.
DEFAULT_PT = os.path.normpath(os.path.join(_STAGE1, "hgraph_diverse_xd0.pt"))
DEFAULT_CKPT = os.path.join(ARTIFACTS, "stage2_diverse_xd0_candidate.pt")


def parse_args():
    p = argparse.ArgumentParser(description="Visualize Stage-2 TRAINED embeddings")
    p.add_argument("--pt", default=DEFAULT_PT, help="xm0 graph the model was trained on")
    p.add_argument("--ckpt", default=DEFAULT_CKPT, help="trained candidate checkpoint")
    p.add_argument("--out", default=os.path.join(ARTIFACTS, "stage2_diverse_xd0_embedding_viz.png"))
    p.add_argument("--show", action="store_true")
    p.add_argument("--train", action="store_true",
                   help="train the candidate (seed 0) and save the checkpoint first")
    p.add_argument("--epochs", type=int, default=25)
    return p.parse_args()


# ── trained embeddings ──────────────────────────────────────────────────────

def _train_candidate(pt, ckpt, epochs):
    """Train the locked candidate (1-layer, top_frac=0.10, lambda_contrast=1) and
    save it. Lazy imports so the default (load-only) path stays light."""
    from ModelLakeFishing.stage2TrainGraphSAGE.model import HeteroGraphSAGE
    from ModelLakeFishing.stage2TrainGraphSAGE.losses import (
        TRAINED_ON, PerfScorer, accuracy_lookup, perf_supervision, split_trained_on,
        topk_membership)
    from ModelLakeFishing.stage2TrainGraphSAGE.train import train
    from ModelLakeFishing.stage2TrainGraphSAGE.learnable import save_checkpoint
    from ModelLakeFishing.stage2TrainGraphSAGE.experiment import CANDIDATE as C

    data, xm0, _ = load_hgraph(pt)
    torch.manual_seed(0); np.random.seed(0)
    train_data, _v, _t = split_trained_on(data, seed=0)
    lookup = accuracy_lookup(data)
    eli, target = perf_supervision(train_data[TRAINED_ON], lookup)
    ti = torch.cat([train_data[TRAINED_ON].edge_index, eli], dim=1)
    ta = torch.cat([train_data[TRAINED_ON].edge_attr.float(), target], dim=0)
    M = topk_membership(data, top_frac=C["top_frac"], trained_on_index=ti, trained_on_attr=ta)
    comp = lineage_components(data, data["model"].num_nodes)
    torch.manual_seed(0); np.random.seed(0)
    model = HeteroGraphSAGE(metadata=data.metadata(), frozen_dim=data["model"].x.shape[1],
                            num_size_buckets=xm0["num_size_buckets"], num_families=xm0["num_families"],
                            dataset_in_dim=data["dataset"].x.shape[1], num_layers=C["num_layers"])
    scorer = PerfScorer(dim=128, mode="dot")
    train(model, scorer, train_data, eli, target, M, comp, epochs=epochs,
          lambda_mse=C["lambda_mse"], lambda_rank=C["lambda_rank"],
          lambda_contrast=C["lambda_contrast"], lambda_uniform=C["lambda_uniform"])
    save_checkpoint(model, xm0["family_vocab"], ckpt)


def trained_embeddings(pt, ckpt):
    data, xm0, _umi = load_hgraph(pt)
    model, _vocab, _repro = load_checkpoint(ckpt)
    model.eval()
    with torch.no_grad():
        z = model(data)
    return data, z["model"].numpy(), z["dataset"].numpy(), model.num_layers


def project_2d(Z):
    try:
        import umap
        xy = umap.UMAP(n_neighbors=15, min_dist=0.2, metric="cosine",
                       random_state=0).fit_transform(Z)
        return xy, "UMAP(cosine)"
    except Exception:
        from sklearn.decomposition import PCA
        return PCA(n_components=2).fit_transform(Z), "PCA(fallback)"


def _unit(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


def participation_ratio(Z):
    """Participation ratio of the (centered) embedding covariance spectrum:
    (sum lambda)^2 / sum(lambda^2). ~1 = one dominant direction (collapsed cone);
    higher = the embedding actually uses more of its dimensions."""
    Z = np.asarray(Z, dtype=float)
    Z = Z - Z.mean(0, keepdims=True)
    ev = np.linalg.eigvalsh(Z.T @ Z)
    ev = ev[ev > 1e-12]
    if ev.size == 0:
        return float("nan")
    return float((ev.sum() ** 2) / (ev ** 2).sum())


def collapse_metrics(z_m, comp):
    """Mean pairwise cosine + same-hub vs random cosine distance (bounded sample)."""
    z = _unit(z_m)
    N = z.shape[0]
    s = z.sum(0)
    mean_cos = float((s @ s - N) / (N * (N - 1)))
    import collections
    groups = collections.defaultdict(list)
    for i, c in enumerate(comp.tolist()):
        groups[c].append(i)
    same = [(g[a], g[b]) for g in groups.values() if len(g) > 1
            for a in range(len(g)) for b in range(a + 1, len(g))]
    same_d = (float(np.mean([1 - z[a] @ z[b] for a, b in same])) if same else float("nan"))
    rng = np.random.default_rng(0)
    ii = rng.integers(0, N, 4000); jj = rng.integers(0, N, 4000); keep = ii != jj
    rand_d = float(np.mean([1 - z[a] @ z[b] for a, b in zip(ii[keep], jj[keep])]))
    return mean_cos, same_d, rand_d, len(same)


# ── drawing (mimics stage1 visualize_hgraph aesthetic) ──────────────────────

def draw(args):
    data, z_m, z_d, num_layers = trained_embeddings(args.pt, args.ckpt)
    g = vh.load_graph(args.pt)
    n_m, n_d = len(g["model_names"]), len(g["dataset_names"])
    assert z_m.shape[0] == n_m and z_d.shape[0] == n_d, "embedding/name row mismatch"

    best_d, best_w = vh.primary_dataset(g)
    order = vh.circle_order(vh.similarity_matrix(g))
    colors = vh.dataset_palette(order)
    comp = lineage_components(data, n_m)
    mean_cos, same_d, rand_d, n_same = collapse_metrics(z_m, comp)
    pr = participation_ratio(z_m)

    lin_ei = g["lineage"][0]
    lin_deg = (np.bincount(np.concatenate([lin_ei[0], lin_ei[1]]), minlength=n_m)
               if lin_ei.size else np.zeros(n_m, dtype=int))

    xy, method = project_2d(np.vstack([z_m, z_d]))
    mxy, dxy = xy[:n_m], xy[n_m:]
    # normalize coordinates into a tidy square
    c = xy.mean(0); span = np.abs(xy - c).max() + 1e-9
    mxy = (mxy - c) / span * 10.0
    dxy = (dxy - c) / span * 10.0

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(18, 11), facecolor=vh.BG)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.5, 1], height_ratios=[1, 1],
                          left=0.02, right=0.97, top=0.90, bottom=0.06, wspace=0.12, hspace=0.28)
    ax = fig.add_subplot(gs[:, 0]); ax.set_facecolor(vh.BG); ax.set_aspect("equal"); ax.axis("off")

    # trained_on chords: each model -> the dataset it performs best on, in EMBED space
    segs, cols = [], []
    for m in range(n_m):
        d = best_d[m]
        if d < 0:
            continue
        segs.append(vh.bezier(mxy[m], dxy[d], pull=0.15))
        r, gg, b = colors.get(d, (0.5, 0.5, 0.5))
        cols.append((r, gg, b, 0.20))
    if segs:
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.5, zorder=1))

    # lineage (is_base_of): gold arcs between trained model positions
    if lin_ei.size:
        lsegs = [vh.bezier(mxy[s], mxy[d], pull=0.12) for s, d in zip(lin_ei[0], lin_ei[1])]
        ax.add_collection(LineCollection(lsegs, colors=[(1.0, 0.78, 0.25, 0.7)],
                                         linewidths=1.6, zorder=2.5))

    # model dots: colour = best dataset, size ~ best accuracy, gold ring = lineage
    nw = vh.norm01(best_w)
    mc = [colors.get(d, (0.5, 0.5, 0.5)) for d in best_d]
    ec = ["#ffc740" if lin_deg[m] > 0 else vh.BG for m in range(n_m)]
    lw = [1.0 if lin_deg[m] > 0 else 0.4 for m in range(n_m)]
    ax.scatter(mxy[:, 0], mxy[:, 1], s=14 + 70 * nw ** 2, c=mc, edgecolors=ec,
               linewidths=lw, zorder=4)

    # dataset hubs at their trained z_d position
    counts = np.bincount(best_d[best_d >= 0], minlength=n_d)
    for d in range(n_d):
        x, y = dxy[d]
        ax.scatter([x], [y], s=150 + 30 * counts[d], color=colors.get(d, (.5, .5, .5)),
                   edgecolors="#f5f5f5", linewidths=1.1, zorder=5)
        ax.text(x, y + 0.35, f"{vh.short(g['dataset_names'][d], 16)}·{counts[d]}",
                fontsize=7.5, fontweight="bold", color=colors.get(d, (.7, .7, .7)),
                ha="center", va="bottom", zorder=6,
                path_effects=[pe.withStroke(linewidth=2.0, foreground=vh.BG)])

    # label the strongest models + lineage family hubs
    call = list(dict.fromkeys(sorted(range(n_m), key=lambda m: -best_w[m])[:5]
                              + [m for m in range(n_m) if lin_deg[m] >= 3]))
    for m in call:
        ax.text(mxy[m, 0], mxy[m, 1] + 0.25, vh.short(g["model_names"][m], 18),
                fontsize=6.0, color="#f3f3f3", ha="center", va="bottom", zorder=6.5,
                path_effects=[pe.withStroke(linewidth=1.8, foreground=vh.BG)])

    lim = 11.5
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

    handles = [
        Line2D([], [], marker="o", ls="none", markerfacecolor="#cccccc", markeredgecolor="none",
               markersize=11, label="dataset hub · position = trained z_d · ·N = models anchored"),
        Line2D([], [], marker="o", ls="none", markerfacecolor="#cccccc", markeredgecolor="none",
               markersize=5, label="model · position = trained z_m · colour = best dataset · size ∝ accuracy"),
        Line2D([], [], marker="o", ls="none", markerfacecolor="#888888", markeredgecolor="#ffc740",
               markersize=6, markeredgewidth=1.2, label="gold ring = appears in a lineage relation"),
        Line2D([], [], color="#ffc740", lw=2.0, label="is_base_of lineage (in embedding space)"),
        Line2D([], [], color="#8fa0c0", lw=1.0, label="trained_on → best dataset (embedding-space chord)"),
    ]
    leg = ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=7.4,
                    labelcolor=vh.FG, borderaxespad=0.2, handlelength=1.6)
    leg.set_zorder(7)

    # ── right top: metrics readout ───────────────────────────────────────────
    axm = fig.add_subplot(gs[0, 1]); axm.set_facecolor(vh.BG); axm.axis("off")
    mean_dist = 1.0 - mean_cos                                   # mean pairwise cosine DISTANCE
    collapse_flag = "COLLAPSED" if mean_dist < 0.03 else ("still tight" if mean_dist < 0.2 else "spread")
    txt = (
        "TRAINED-EMBEDDING DIAGNOSTICS\n"
        f"(candidate: {num_layers}-layer, top_frac=0.10, λ_contrast=1, dot scorer)\n"
        "\n"
        f"mean pairwise cos-distance : {mean_dist:.3f}   ({collapse_flag}; 0.0 = collapsed)\n"
        f"z_m participation ratio    : {pr:.2f} / {z_m.shape[1]} dims\n"
        f"same-hub cos-distance: {same_d:.3f}   ({n_same} lineage pairs)\n"
        f"random cos-distance  : {rand_d:.3f}\n"
        "\n"
        "Same-hub < random means lineage families stay closer than\n"
        "average (expected). Both small means the whole space is still\n"
        "tight -- collapse is reduced, NOT solved (see\n"
        "stage2_candidate_report.md)."
    )
    axm.text(0.0, 0.98, txt, transform=axm.transAxes, va="top", ha="left",
             fontsize=8.4, color=vh.FG, linespacing=1.5, family="monospace")

    # ── right bottom: how to read ─────────────────────────────────────────────
    axg = fig.add_subplot(gs[1, 1]); axg.set_facecolor(vh.BG); axg.axis("off")
    guide = (
        "HOW TO READ THIS FIGURE\n"
        "\n"
        "Unlike the Stage-1 figure (synthetic circle), positions here are\n"
        "the TRAINED GraphSAGE embeddings z_m / z_d, projected to 2D\n"
        f"({method}). Distance ≈ learned (cosine) similarity.\n"
        "\n"
        "What to look for if training worked:\n"
        " • models cluster by the dataset/task they perform best on\n"
        "   (matching dot colours grouping together);\n"
        " • each model sits near its best dataset hub (short chords);\n"
        " • gold-ringed lineage families stay near their base, but NOT\n"
        "   collapsed onto a single point (that would be over-smoothing).\n"
        "\n"
        "Embeddings come from the xm0-trained candidate checkpoint; no\n"
        "synthetic layout is used."
    )
    axg.text(0.0, 0.98, guide, transform=axg.transAxes, va="top", ha="left",
             fontsize=8.2, color=vh.FG, linespacing=1.45, family="sans-serif")

    fig.suptitle(f"Stage-2 TRAINED embeddings — {n_m} models × {n_d} datasets "
                 f"(GraphSAGE z_m / z_d)", fontsize=16, fontweight="bold",
                 color="white", x=0.02, ha="left", y=0.975)
    fig.text(0.02, 0.935,
             f"ckpt {os.path.basename(args.ckpt)} · graph {os.path.basename(args.pt)} · "
             f"candidate {num_layers}-layer top_frac=0.10 λ_contrast=1 (dot) · "
             f"projection {method} · {g['lineage'][0].shape[1]} lineage edges · "
             f"mean cos-dist {mean_dist:.3f} · z_m participation {pr:.2f}",
             fontsize=9.5, color=vh.MUTED, ha="left")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=220, facecolor=vh.BG)
    print(f"saved {args.out}  ({n_m} models, {n_d} datasets, projection {method})")
    print(f"mean_cos={mean_cos:.3f}  same_hub_dist={same_d:.3f}  random_dist={rand_d:.3f}")
    if args.show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()
    if args.train or not os.path.exists(args.ckpt):
        if not os.path.exists(args.ckpt):
            print(f"checkpoint {args.ckpt} not found -> training candidate ({args.epochs} epochs)")
        _train_candidate(args.pt, args.ckpt, args.epochs)
    draw(args)


if __name__ == "__main__":
    main()
