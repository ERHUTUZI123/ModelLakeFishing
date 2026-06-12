"""
visualize_hgraph.py — pretty one-shot visualization of the stage-1 HGraph.

Reads the payload saved by build_graph.py (HeteroData + id tables + args) and
renders a single figure:

  * main panel — radial "hub galaxy": the 23 datasets sit on a circle,
    ordered by domain similarity (hierarchical clustering, so similar
    datasets are neighbours); every model is fanned out behind the dataset
    where it scores best relative to the field (z-scored accuracy, so one
    universally easy dataset cannot swallow them all). trained_on edges
    are drawn as bezier chords coloured by their target dataset,
    transfer_to (LogME) edges as faint
    grey chords, the strongest similar_to pairs as white arcs through the
    middle, and is_base_of lineage relations as gold arcs between models
    (lineage-involved models carry a gold ring).
  * right panels — the clustered dataset-similarity heatmap and a
    "how to read this figure" guide.

Run
---
  cd stage1BuildTransferGraph
  python visualize_hgraph.py                       # -> hgraph_zoo_viz.png
  python visualize_hgraph.py --pt hgraph_zoo_xm0.pt --show
"""

import argparse
import colorsys
import math
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

BG = '#0b0e14'
FG = '#dde3ee'
MUTED = '#5c6677'


def parse_args():
    p = argparse.ArgumentParser(description="Visualize the stage-1 HGraph")
    p.add_argument('--pt', default=os.path.join(HERE, 'hgraph_zoo.pt'),
                   help='payload saved by build_graph.py')
    p.add_argument('--out', default=None,
                   help='output image (default: <pt-stem>_viz.png next to the .pt)')
    p.add_argument('--show', action='store_true', help='open an interactive window too')
    p.add_argument('--top-sim', type=int, default=2,
                   help='similar_to arcs kept per dataset (the circle order already encodes the rest)')
    return p.parse_args()


# ---------------------------------------------------------------- data prep

def load_graph(pt_path):
    payload = torch.load(pt_path, map_location='cpu', weights_only=False)
    data = payload['data']
    model_names = payload['unique_model_id'].sort_values('mappedID')['model'].tolist()
    dataset_names = payload['unique_dataset_id'].sort_values('mappedID')['dataset'].tolist()

    def edges(et):
        if et not in data.edge_types:
            return np.zeros((2, 0), dtype=int), np.zeros(0)
        ei = data[et].edge_index.numpy()
        w = data[et].edge_attr.numpy().reshape(-1) if 'edge_attr' in data[et] else np.ones(ei.shape[1])
        return ei, w

    return {
        'args': payload.get('args', {}),
        'model_names': model_names,
        'dataset_names': dataset_names,
        'accu': edges(('model', 'trained_on', 'dataset')),
        'tran': edges(('model', 'transfer_to', 'dataset')),
        'sim': edges(('dataset', 'similar_to', 'dataset')),
        'lineage': edges(('model', 'is_base_of', 'model')),
    }


def similarity_matrix(g):
    n = len(g['dataset_names'])
    S = np.zeros((n, n))
    ei, w = g['sim']
    S[ei[0], ei[1]] = w
    S = np.maximum(S, S.T)
    np.fill_diagonal(S, 1.0)
    return S


def circle_order(S):
    """Order datasets so similar ones sit next to each other on the circle."""
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
        from scipy.spatial.distance import squareform
        D = np.clip(1.0 - S, 0.0, None)
        np.fill_diagonal(D, 0.0)
        cond = squareform(D, checks=False)
        Z = optimal_leaf_ordering(linkage(cond, method='average'), cond)
        return list(leaves_list(Z))
    except Exception:
        return list(range(S.shape[0]))


def primary_dataset(g):
    """Anchor each model to its comparative-advantage dataset.

    Weights are z-scored per dataset before the argmax: anchoring on raw
    accuracy degenerates (one universally easy dataset is the maximum for
    136 of 164 zoo models). LogME and accuracy live on different scales,
    so the fallback pass only places models with no accuracy edge at all.
    """
    n_m = len(g['model_names'])
    best_d = np.full(n_m, -1, dtype=int)
    best_z = np.zeros(n_m)
    best_w = np.zeros(n_m)          # raw weight, kept for dot sizes / call-outs
    for key in ('accu', 'tran'):
        frozen = best_d >= 0
        ei, w = g[key]
        if ei.shape[1] == 0:
            continue
        z = np.zeros_like(w)
        for d in np.unique(ei[1]):
            mask = ei[1] == d
            z[mask] = (w[mask] - w[mask].mean()) / (w[mask].std() + 1e-9)
        for (m, d, zz, ww) in zip(ei[0], ei[1], z, w):
            if not frozen[m] and (best_d[m] == -1 or zz > best_z[m]):
                best_d[m], best_z[m], best_w[m] = d, zz, ww
    return best_d, best_w


# ------------------------------------------------------------------ layout

def radial_layout(g, order, best_d, best_w, R=10.0):
    """Dataset hubs on a circle; each dataset's models packed in concentric
    arcs just outside its segment. Segment width grows with model count."""
    n_d = len(g['dataset_names'])
    counts = np.bincount(best_d[best_d >= 0], minlength=n_d)

    widths = counts[order] + 3.0
    widths = widths / widths.sum() * 2 * math.pi
    starts = np.concatenate([[0.0], np.cumsum(widths)[:-1]])
    theta = {}                     # dataset id -> hub angle
    for pos, d in enumerate(order):
        theta[d] = math.pi / 2 - (starts[pos] + widths[pos] / 2)   # start at 12 o'clock, clockwise

    hub_pos = {d: (R * math.cos(t), R * math.sin(t)) for d, t in theta.items()}

    # models per dataset, strongest first (they get the innermost ring)
    models_of = {d: [] for d in range(n_d)}
    for m, d in enumerate(best_d):
        if d >= 0:
            models_of[d].append(m)
    for d in models_of:
        models_of[d].sort(key=lambda m: -best_w[m])

    model_pos = np.zeros((len(g['model_names']), 2))
    seg_outer = {}                 # dataset id -> outermost model radius (for label placement)
    spacing, row_step, eff = 0.62, 0.95, 0.80
    for pos, d in enumerate(order):
        half = widths[pos] * eff / 2
        queue, row = list(models_of[d]), 0
        seg_outer[d] = R
        while queue:
            r = R + 1.5 + row * row_step
            cap = max(1, int(2 * half * r / spacing))
            batch, queue = queue[:cap], queue[cap:]
            k = len(batch)
            offs = [0.0] if k == 1 else np.linspace(-half * k / cap, half * k / cap, k)
            for m, off in zip(batch, offs):
                t = theta[d] + off
                model_pos[m] = (r * math.cos(t), r * math.sin(t))
            seg_outer[d] = r
            row += 1

    # edge-less models (shouldn't happen, but never lose nodes silently)
    lost = np.where(best_d < 0)[0]
    for k, m in enumerate(lost):
        t, r = k * 2.39996, 0.6 * math.sqrt(k + 1)
        model_pos[m] = (r * math.cos(t), r * math.sin(t))

    return theta, hub_pos, model_pos, counts, seg_outer, lost


def bezier(p0, p1, pull=0.45, n=24):
    """Quadratic bezier whose midpoint is pulled toward the circle centre."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    c = (p0 + p1) / 2 * (1.0 - pull)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p1


def norm01(w):
    w = np.asarray(w, float)
    if w.size == 0:
        return w
    lo, hi = w.min(), w.max()
    return (w - lo) / (hi - lo) if hi > lo else np.ones_like(w)


def repel_labels(anchors, half_w, half_h, obstacles, obs_r, lim,
                 iters=240, step=0.35):
    """Force-directed label placement (a tiny adjustText).

    Each label box is tethered to its anchor by a weak spring, repels every
    other label box on whichever axis they overlap least, and is pushed out
    of a disc around each fixed obstacle (the dataset-label anchor points).
    Returns the resolved text centres; leader lines back to the anchors are
    drawn by the caller.
    """
    pos = anchors.astype(float).copy()
    # nudge each label slightly outward so it starts off its own dot
    norm = np.linalg.norm(pos, axis=1, keepdims=True)
    pos += pos / np.clip(norm, 1e-6, None) * 0.6
    n = len(pos)
    for _ in range(iters):
        disp = (anchors - pos) * 0.015                       # spring home
        for i in range(n):
            for j in range(i + 1, n):
                dx, dy = pos[i] - pos[j]
                ox = (half_w[i] + half_w[j]) - abs(dx)
                oy = (half_h[i] + half_h[j]) - abs(dy)
                if ox > 0 and oy > 0:                        # boxes overlap
                    if ox < oy:
                        s = (ox / 2 + 1e-3) * (1 if dx >= 0 else -1)
                        disp[i, 0] += s; disp[j, 0] -= s
                    else:
                        s = (oy / 2 + 1e-3) * (1 if dy >= 0 else -1)
                        disp[i, 1] += s; disp[j, 1] -= s
        for i in range(n):                                   # clear obstacles
            for (ox_, oy_) in obstacles:
                dx, dy = pos[i, 0] - ox_, pos[i, 1] - oy_
                d = math.hypot(dx, dy)
                rr = obs_r + half_h[i]
                if 1e-6 < d < rr:
                    f = (rr - d) / d * 0.5
                    disp[i, 0] += dx * f; disp[i, 1] += dy * f
        pos += np.clip(disp, -step, step)
        pos = np.clip(pos, -lim + 0.5, lim - 0.5)
    return pos


def dataset_palette(order):
    """One hue per dataset, flowing rainbow-style around the circle."""
    cols = {}
    for pos, d in enumerate(order):
        h = pos / len(order)
        cols[d] = colorsys.hls_to_rgb(h, 0.62, 0.72)
    return cols


def short(name, n=22):
    name = name.split('/')[-1]
    return name if len(name) <= n else name[:n - 1] + '…'


# ----------------------------------------------------------------- drawing

def draw(g, args, out_path, show):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    S = similarity_matrix(g)
    order = circle_order(S)
    best_d, best_w = primary_dataset(g)
    theta, hub_pos, model_pos, counts, seg_outer, lost = radial_layout(g, order, best_d, best_w)
    colors = dataset_palette(order)
    n_m, n_d = len(g['model_names']), len(g['dataset_names'])
    held_out = g['args'].get('test_dataset')

    fig = plt.figure(figsize=(19, 11), facecolor=BG)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.55, 1], height_ratios=[1.55, 1],
                          left=0.015, right=0.965, top=0.90, bottom=0.07,
                          wspace=0.16, hspace=0.30)
    ax = fig.add_subplot(gs[:, 0])
    ax.set_facecolor(BG)
    ax.set_aspect('equal')
    ax.axis('off')

    # guide circle under everything
    ax.add_patch(plt.Circle((0, 0), 10.0, fill=False, color='#222a38', lw=0.8, ls=':', zorder=0))

    def chords(ei, w, pull, color_of, a_lo, a_hi, lw_lo, lw_hi, src_pos, dst_pos, zorder):
        if ei.shape[1] == 0:
            return
        nw = norm01(w)
        idx = np.argsort(nw)                      # strong edges drawn last (on top)
        segs, cols, lws = [], [], []
        for i in idx:
            s, d = ei[0, i], ei[1, i]
            p0, p1 = src_pos(s), dst_pos(d)
            pl = pull(p0, p1) if callable(pull) else pull
            segs.append(bezier(p0, p1, pl))
            r, gg, b = color_of(s, d)
            cols.append((r, gg, b, a_lo + (a_hi - a_lo) * nw[i]))
            lws.append(lw_lo + (lw_hi - lw_lo) * nw[i])
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws,
                                         capstyle='round', zorder=zorder))

    m_pos = lambda m: model_pos[m]
    d_pos = lambda d: hub_pos[d]

    # transfer_to (LogME): faint grey underlayer
    ei, w = g['tran']
    chords(ei, w, 0.45, lambda s, d: (0.50, 0.57, 0.66), 0.04, 0.20, 0.30, 0.80,
           m_pos, d_pos, zorder=1)

    # trained_on (accuracy): coloured by target dataset
    ei, w = g['accu']
    chords(ei, w, 0.45, lambda s, d: colors[d], 0.10, 0.55, 0.40, 1.80,
           m_pos, d_pos, zorder=2)

    # similar_to: keep top-k arcs per dataset, white through the middle
    ei, w = g['sim']
    keep = {}
    for i in range(ei.shape[1]):
        a, b = int(ei[0, i]), int(ei[1, i])
        if a != b:
            keep.setdefault(a, []).append((w[i], (min(a, b), max(a, b))))
    pairs = {}
    for a, lst in keep.items():
        for ww, pair in sorted(lst, reverse=True)[:args.top_sim]:
            pairs[pair] = max(ww, pairs.get(pair, 0.0))
    if pairs:
        pe_ = np.array([[a, b] for a, b in pairs]).T
        pw = np.array(list(pairs.values()))
        chords(pe_, pw, 0.72, lambda s, d: (1.0, 1.0, 1.0), 0.08, 0.40, 0.50, 2.60,
               d_pos, d_pos, zorder=1.5)

    # lineage (is_base_of): gold arcs between models — bow gently for nearby
    # family members, dive through the circle for cross-segment relations
    ei, w = g['lineage']
    lin_pull = lambda p0, p1: min(0.65, 0.08 + np.hypot(*(np.asarray(p0) - p1)) / 20.0 * 0.55)
    chords(ei, w, lin_pull, lambda s, d: (1.0, 0.78, 0.25), 0.35, 0.85, 0.70, 2.20,
           m_pos, m_pos, zorder=2.5)
    lin_deg = (np.bincount(np.concatenate([ei[0], ei[1]]), minlength=n_m)
               if ei.size else np.zeros(n_m, dtype=int))

    # model dots — lineage-involved models get a gold ring
    nw = norm01(best_w)
    mc = [colors.get(d, (0.5, 0.5, 0.5)) for d in best_d]
    ec = ['#ffc740' if lin_deg[m] > 0 else BG for m in range(n_m)]
    lw = [1.0 if lin_deg[m] > 0 else 0.4 for m in range(n_m)]
    ax.scatter(model_pos[:, 0], model_pos[:, 1], s=14 + 70 * nw ** 2, c=mc,
               edgecolors=ec, linewidths=lw, zorder=4)

    # dataset hubs, sized by how many models call them home
    for d in range(n_d):
        x, y = hub_pos[d]
        ax.scatter([x], [y], s=140 + 30 * counts[d], color=colors[d],
                   edgecolors='#f5f5f5', linewidths=1.1, zorder=5)
        if g['dataset_names'][d] == held_out:
            ax.add_patch(plt.Circle((x, y), 0.78, fill=False, color='#ff5566',
                                    lw=1.3, ls='--', zorder=5))

    # dataset labels, radial like a circos plot
    for d in range(n_d):
        t = theta[d]
        deg = math.degrees(t) % 360
        r = seg_outer[d] + 1.15
        x, y = r * math.cos(t), r * math.sin(t)
        flip = 90 < deg < 270
        label = f"{g['dataset_names'][d]}  ·{counts[d]}"
        if g['dataset_names'][d] == held_out:
            label += '  (held-out)'
        ax.text(x, y, label, rotation=deg + 180 if flip else deg,
                rotation_mode='anchor', ha='right' if flip else 'left', va='center',
                fontsize=8.5, fontweight='bold', color=colors[d], zorder=6,
                path_effects=[pe.withStroke(linewidth=2.2, foreground=BG)])

    lim = max(seg_outer.values()) + 3.4

    # call out the strongest models plus the lineage family hubs, with a
    # force-directed pass so the labels neither pile on each other nor land
    # on the radial dataset labels
    top = sorted(range(n_m), key=lambda m: -best_w[m])[:5]
    hubs = [m for m in range(n_m) if lin_deg[m] >= 3]
    call = list(dict.fromkeys(top + hubs))
    if call:
        labels = [short(g['model_names'][m], 20) for m in call]
        anchors = model_pos[call]
        cw, lh = 0.0085 * lim, 0.052 * lim          # ~data units per char / line
        half_w = np.array([max(2, len(t)) * cw for t in labels])
        half_h = np.full(len(labels), lh)
        obstacles = [(seg_outer[d] + 1.15) * np.array([math.cos(theta[d]), math.sin(theta[d])])
                     for d in range(n_d)]
        lp = repel_labels(anchors, half_w, half_h, obstacles, obs_r=0.9 * lh * 6, lim=lim)
        for m, (lx, ly), txt in zip(call, lp, labels):
            ax.plot([model_pos[m, 0], lx], [model_pos[m, 1], ly],
                    color='#9aa6b8', lw=0.5, alpha=0.7, zorder=5.5)
            ax.text(lx, ly, txt, fontsize=6.2, color='#f3f3f3', ha='center', va='center',
                    zorder=6.5, path_effects=[pe.withStroke(linewidth=2.0, foreground=BG)])

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    handles = [
        Line2D([], [], color='#e8a87c', lw=2.0,
               label='model ─trained_on→ dataset · colour = target dataset · opacity & width ∝ fine-tune accuracy'),
        Line2D([], [], color='#8091a8', lw=1.2,
               label='model ─transfer_to→ dataset · grey · opacity ∝ LogME transferability score'),
        Line2D([], [], color='white', lw=2.0, alpha=0.7,
               label=f'dataset ─similar_to─ dataset · white inner arc · top {args.top_sim} per dataset · width ∝ similarity'),
        Line2D([], [], color='#ffc740', lw=2.0,
               label='model ─is_base_of→ model · gold arc · width ∝ relation strength:'
                     ' quantized .9 > adapter .7 > finetune .5 > merge .3'),
        Line2D([], [], marker='o', ls='none', markerfacecolor='#cccccc', markeredgecolor='none',
               markersize=11, label='dataset hub · size & ·N in label = models anchored to it'),
        Line2D([], [], marker='o', ls='none', markerfacecolor='#cccccc', markeredgecolor='none',
               markersize=5, label='model · colour = anchor dataset · size ∝ best raw accuracy'),
        Line2D([], [], marker='o', ls='none', markerfacecolor='#888888', markeredgecolor='#ffc740',
               markersize=6, markeredgewidth=1.2, label='gold ring = model appears in a lineage relation'),
        Line2D([], [], color='#ff5566', lw=1.2, ls='--',
               label=f'dashed red ring = held-out dataset ({held_out}): accuracy edges hidden from the graph'),
    ]
    leg = ax.legend(handles=handles, loc='lower left', frameon=False, fontsize=7.2,
                    labelcolor=FG, borderaxespad=0.2, handlelength=1.6)
    leg.set_zorder(7)

    # ---- right top: clustered similarity heatmap
    axh = fig.add_subplot(gs[0, 1])
    axh.set_facecolor(BG)
    So = S[np.ix_(order, order)]
    im = axh.imshow(So, cmap='magma', vmin=0, vmax=1)
    names = [g['dataset_names'][d] for d in order]
    axh.set_xticks(range(n_d), names, rotation=90, fontsize=5.6)
    axh.set_yticks(range(n_d), names, fontsize=5.6)
    for tick, d in zip(axh.get_xticklabels(), order):
        tick.set_color(colors[d])
    for tick, d in zip(axh.get_yticklabels(), order):
        tick.set_color(colors[d])
    axh.tick_params(length=0)
    for sp in axh.spines.values():
        sp.set_visible(False)
    axh.set_title('dataset–dataset domain similarity  (clustered = circle order)',
                  fontsize=9, color=FG, pad=8)
    cb = fig.colorbar(im, ax=axh, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=6, colors=MUTED)
    cb.outline.set_visible(False)

    # ---- right bottom: how to read this figure
    axg = fig.add_subplot(gs[1, 1])
    axg.set_facecolor(BG)
    axg.axis('off')
    guide = (
        "HOW TO READ THIS FIGURE\n"
        "\n"
        "Layout — the datasets sit on a circle, ordered by hierarchical\n"
        "clustering of the similarity matrix: neighbouring hubs are\n"
        "similar domains, and the rainbow hue follows the circle.\n"
        "\n"
        "Anchoring — each model is drawn in the fan behind the dataset\n"
        "where its fine-tune accuracy is highest relative to the other\n"
        "models (z-scored per dataset); its chords to every other\n"
        "dataset it was evaluated on cross the circle. Strongest models\n"
        "sit on the innermost ring of each fan.\n"
        "\n"
        "Lineage — gold arcs link derivative models to their bases\n"
        "(real HuggingFace base_model relations: the gemma-2,\n"
        "llama-3.1 and gpt-2 families); family hubs are labelled.\n"
        "\n"
        "Held-out — the dashed-red dataset keeps only similarity and\n"
        "LogME edges; its fine-tune accuracy edges are hidden so it\n"
        "can serve as the link-prediction test target.\n"
        "\n"
        "NOTE — the family models' eval accuracies are synthetic demo\n"
        "values (added by add_family_pack.py); lineage relations are real."
    )
    axg.text(0.0, 0.98, guide, transform=axg.transAxes, va='top', ha='left',
             fontsize=8.2, color=FG, linespacing=1.45, family='sans-serif')

    # ---- titles
    n_lin = g['lineage'][0].shape[1]
    fig.suptitle(f'HGraph zoo — transfer graph of {n_m} models × {n_d} datasets',
                 fontsize=17, fontweight='bold', color='white', x=0.015, ha='left', y=0.975)
    sub = (f"{g['accu'][0].shape[1]} accuracy edges · {g['tran'][0].shape[1]} LogME edges · "
           f"{g['sim'][0].shape[1] // 2} similarity pairs · {n_lin} lineage edges"
           + (f" · held-out: {held_out}" if held_out else "")
           + f" · models anchored where they score best relative to the field")
    fig.text(0.015, 0.935, sub, fontsize=9.5, color=MUTED, ha='left')
    ga = g['args']
    foot = (f"task: {ga.get('task_type', '?')} · gnn: {ga.get('gnn_method', '?')} · "
            f"accuracy thresholds [{ga.get('accu_neg_thres', '?')}, {ga.get('accu_pos_thres', '?')}] · "
            f"source: {os.path.basename(args.pt)}")
    fig.text(0.015, 0.022, foot, fontsize=8, color=MUTED, ha='left')
    if lost.size:
        fig.text(0.015, 0.045, f'{lost.size} edge-less models shown grey at the centre',
                 fontsize=8, color='#aa6666', ha='left')

    fig.savefig(out_path, dpi=220, facecolor=BG)
    print(f'saved {out_path}  ({n_m} models, {n_d} datasets)')
    if show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()
    out = args.out or os.path.splitext(args.pt)[0] + '_viz.png'
    draw(load_graph(args.pt), args, out, args.show)


if __name__ == '__main__':
    main()
