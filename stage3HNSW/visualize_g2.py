"""
visualize_g2.py -- presentation view of the G2 SERVING embedding space.

Same aesthetic as stage1/visualize_hgraph.py and stage2/visualize_trained.py
(dark theme, bezier chords, dataset-anchored colouring), but built for an
external audience: ONE full-bleed panel, light annotations, no diagnostics
column. Vectors come from the hash-bound Phase-1 export (the geometry actually
indexed by HNSW), never recomputed here.

Reading the figure:
  * every dot is a model at its trained z_m (UMAP of the 128-d cosine space);
  * colour = the dataset the model is anchored to (comparative-advantage
    argmax, vh.primary_dataset); only the top-N anchor datasets get colours,
    the long tail is muted;
  * big labelled markers = those anchor datasets at their trained z_d;
  * faint chords connect each coloured model to its anchor dataset.

Run (repo root):
  python -m ModelLakeFishing.stage3HNSW.visualize_g2 [--show]
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_STAGE1 = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage1BuildTransferGraph")
for _p in (_REPO_ROOT, _STAGE1):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ModelLakeFishing.stage1BuildTransferGraph import visualize_hgraph as vh  # noqa: E402

EXPORT = os.path.join(_HERE, "artifacts", "exports", "hf1000d_G2")
GRAPH = os.path.join(_STAGE1, "hgraph_hf1000d_2000m_xm0_xd0.pt")
OUT = os.path.join(_HERE, "artifacts", "g2_embedding_viz.png")
TOP_DATASETS = 14           # anchor datasets that get a colour + a label


def parse_args():
    p = argparse.ArgumentParser(description="G2 serving-embedding presentation figure")
    p.add_argument("--export", default=EXPORT)
    p.add_argument("--graph", default=GRAPH)
    p.add_argument("--out", default=OUT)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def project_2d(Z):
    try:
        import umap
        xy = umap.UMAP(n_neighbors=15, min_dist=0.25, metric="cosine",
                       random_state=0).fit_transform(Z)
        return xy, "UMAP(cosine)"
    except Exception:
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, metric="cosine",
                    random_state=0).fit_transform(Z), "t-SNE(fallback)"


def main():
    args = parse_args()
    z_m = np.load(os.path.join(args.export, "z_m.npy"))
    z_d = np.load(os.path.join(args.export, "z_d.npy"))
    g = vh.load_graph(args.graph)
    n_m, n_d = len(g["model_names"]), len(g["dataset_names"])
    assert z_m.shape[0] == n_m and z_d.shape[0] == n_d, "export/graph row mismatch"

    best_d, best_w = vh.primary_dataset(g)
    counts = np.bincount(best_d[best_d >= 0], minlength=n_d)
    top = [int(d) for d in np.argsort(-counts)[:TOP_DATASETS] if counts[d] > 0]
    colors = vh.dataset_palette(top)                    # dataset -> RGB, in hub order
    muted = (0.42, 0.46, 0.55)

    xy, method = project_2d(np.vstack([z_m, z_d]).astype(np.float64))
    c = xy.mean(0); span = np.abs(xy - c).max() + 1e-9
    mxy = (xy[:n_m] - c) / span * 10.0
    dxy = (xy[n_m:] - c) / span * 10.0

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(16, 11), facecolor=vh.BG)
    ax = fig.add_axes([0.015, 0.02, 0.97, 0.88])
    ax.set_facecolor(vh.BG); ax.set_aspect("equal"); ax.axis("off")

    # chords: coloured models -> their anchor dataset (embedding space)
    segs, cols = [], []
    for m in range(n_m):
        d = int(best_d[m])
        if d in colors:
            r, gg, b = colors[d]
            segs.append(vh.bezier(mxy[m], dxy[d], pull=0.15))
            cols.append((r, gg, b, 0.10))
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.45, zorder=1))

    # model dots: colour = anchor dataset (top-N) else muted; size ~ anchor weight
    nw = vh.norm01(best_w)
    mc = [colors.get(int(d), muted) for d in best_d]
    al = [0.95 if int(d) in colors else 0.45 for d in best_d]
    ax.scatter(mxy[:, 0], mxy[:, 1], s=10 + 55 * nw ** 2, c=mc, alpha=al,
               edgecolors="none", zorder=4)

    # anchor-dataset hubs at their trained z_d, labels repelled off the pile-up
    # (the top z_d positions sit close together, so naive placement overlaps)
    labels = [f"{vh.short(g['dataset_names'][d], 20)} · {counts[d]}" for d in top]
    anchors = np.array([dxy[d] for d in top])
    half_w = np.array([0.075 * len(t) for t in labels])
    half_h = np.full(len(top), 0.30)
    lpos = vh.repel_labels(anchors, half_w, half_h,
                           obstacles=[tuple(a) for a in anchors], obs_r=0.55,
                           lim=10.8)
    for i, d in enumerate(top):
        x, y = dxy[d]
        ax.scatter([x], [y], s=210 + 25 * np.sqrt(counts[d]), color=colors[d],
                   edgecolors="#f5f5f5", linewidths=1.2, zorder=5, marker="o")
        lx, ly = lpos[i]
        if np.hypot(lx - x, ly - y) > 0.55:                  # leader line if moved
            ax.plot([x, lx], [y, ly], color=colors[d], lw=0.7, alpha=0.7, zorder=5.5)
        ax.text(lx, ly, labels[i],
                fontsize=8.5, fontweight="bold", color=colors[d],
                ha="center", va="center", zorder=6,
                path_effects=[pe.withStroke(linewidth=2.6, foreground=vh.BG)])

    lim = 11.2
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

    # title block (annotation only -- no diagnostics column)
    fig.text(0.03, 0.965, "Model Lake · trained embedding space",
             fontsize=19, fontweight="bold", color=vh.FG, ha="left", va="top")
    fig.text(0.03, 0.925,
             f"2,000 models · 362 datasets · inductive heterogeneous GraphSAGE (G2) · "
             f"{method} of the 128-d cosine serving geometry",
             fontsize=10.5, color="#9aa6ba", ha="left", va="top")
    fig.text(0.97, 0.965,
             "one dot = one model at its learned position\n"
             "search = nearest neighbours of a dataset's position (HNSW, O(log N))",
             fontsize=9.5, color="#9aa6ba", ha="right", va="top", linespacing=1.5)

    handles = [
        Line2D([], [], marker="o", ls="none", markerfacecolor="#cccccc",
               markeredgecolor="#f5f5f5", markersize=11,
               label="dataset · position = its learned embedding z_d · N = models anchored to it"),
        Line2D([], [], marker="o", ls="none", markerfacecolor="#cccccc",
               markeredgecolor="none", markersize=5,
               label="model · position = its learned embedding z_m · colour = anchor dataset"),
        Line2D([], [], marker="o", ls="none", markerfacecolor="#6b7382",
               markeredgecolor="none", markersize=4, alpha=0.6,
               label="model anchored to a long-tail dataset (muted)"),
        Line2D([], [], color="#8fa0c0", lw=1.0, alpha=0.6,
               label="model — anchor-dataset link (in embedding space)"),
    ]
    leg = ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8.2,
                    labelcolor=vh.FG, borderaxespad=0.4, handlelength=1.5)
    leg.set_zorder(7)
    fig.text(0.985, 0.015, "checkpoint G2 · hf1000d_2000m · Stage-3 serving export",
             fontsize=7.5, color=vh.MUTED, ha="right", va="bottom")

    fig.savefig(args.out, dpi=220, facecolor=vh.BG)
    print(f"saved: {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
