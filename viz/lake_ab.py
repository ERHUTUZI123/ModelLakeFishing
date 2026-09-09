"""lake_ab.py -- the lake, and one query answered, as paper figures.

A stripped rebuild of panels A and B of `galaxy_render.py`'s plate.  No poster
header, no cascade, no rank panels, no footer: two boxes, a short title over
each, and a legend strip underneath, the way a figure like this is normally
set.

    A   the whole lake, complete, with the family territories named and the
        outline of the region B enlarges
    B   one held-out query, its 1,000 pooled candidates, the ten that came
        back, and where the gold model landed

They are two independent figures, not two halves of one, so each is drawn at
its own natural aspect ratio rather than squeezed into a shared row.  That
matters most for A: the lake is markedly taller than it is wide, and the
square frame it used to get is what left it small and structureless in the
middle of white paper.  A is therefore portrait, and shows the whole lake --
only a few hundred rows of three million fall outside the frame, and the
subtitle says how many.

Legends and notes live in a strip *below* the panel, never inside the data
area, so nothing can collide with the numbered discs.

Everything is read from the same cache and archived products the plate uses,
so no number here is typed by hand.  Outputs, into docs/1M/figures/:

    lake_a      the lake
    lake_b      one query, answered
    lake_ab     the same two panels side by side, for a two-column float

each as PDF, SVG and PNG.

    python -m ModelLakeFishing.viz.lake_ab
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from ModelLakeFishing.viz import model_lake_galaxy as G
from ModelLakeFishing.viz.galaxy_render import (
    PAPER, INK, SUBINK, FAINT, HAIR, GOLD, GOLD_LT, CRIMSON, AMBER,
    NEBULA, PRIOR_CMAP, SANS, MONO,
    density_image, place_labels, _bounds, _fit, _to_axes,
    _clean, _frame, _style,
)

OUTDIR = G.OUTDIR

# Panel A shows the whole lake, so it is not cropped: the frame is the lake's
# own percentile box and the panel is given that box's aspect ratio.  The lake
# is markedly taller than it is wide, and forcing it into a square frame is
# what used to leave it small and structureless in the middle of white paper.
LAKE_LO, LAKE_HI, LAKE_PAD = 0.10, 99.90, 0.04

NAMED = [("llama", "llama"), ("qwen", "qwen"), ("gemma", "gemma"),
         ("mistral", "mistral"), ("bert", "bert"), ("roberta", "roberta"),
         ("xlm-roberta", "xlm-roberta"), ("t5", "t5"), ("gpt2", "gpt2"),
         ("distilbert", "distilbert"), ("whisper", "whisper"),
         ("wav2vec2", "wav2vec2"), ("vit", "vit"), ("marian", "marian"),
         ("stablediffusion", "stable-diffusion"), ("flux", "flux"),
         ("blockassist", "blockassist")]

FS_LETTER, FS_TITLE, FS_SUB = 9.4, 8.8, 6.4
FS_FAMILY, FS_NOTE, FS_LEGEND = 6.4, 6.1, 6.3
FS_CASE, FS_CASE_SUB = 7.2, 6.2


# ------------------------------------------------------------------- inputs --
def load(cache=None):
    cache = cache or G.CACHE
    xy = np.asarray(np.load(os.path.join(cache, "layout_xy.npy"), mmap_mode="r"))
    qxy = np.load(os.path.join(cache, "query_xy.npz"))["xy"]
    gf = np.load(os.path.join(cache, "graph_facts.npz"))
    ginfo = json.load(open(os.path.join(cache, "graph_facts.json"),
                           encoding="utf-8"))
    rr = np.load(os.path.join(cache, "rerank.npz"))
    qi = int(np.flatnonzero(rr["query"] == G.CASE_QUERY)[0])
    case = {"pool": rr["pool_ids"][qi].astype(np.int64),
            "prior": rr["case_prior"].astype(np.float32),
            "top10": rr["top10"][qi].astype(np.int64),
            "gold": int(rr["gold_id"][qi]),
            "dense_rank": int(rr["dense_rank"][qi]),
            "fused_rank": int(rr["fused_rank"][qi])}
    model_name = (pd.read_parquet(os.path.join(G.EXPORT, "model_ids.parquet"))
                  .sort_values("mappedID")["model"].to_numpy())
    ds_name = (pd.read_parquet(os.path.join(G.EXPORT, "dataset_ids.parquet"))
               .sort_values("mappedID")["dataset"].to_numpy())
    return dict(xy=xy, qxy=qxy, family_id=gf["family_id"], ginfo=ginfo,
                case=case, model_name=model_name, ds_name=ds_name)


def lake_box(xy):
    """The lake's own bounding box, before any panel aspect is imposed."""
    return _bounds(xy[:, 0], xy[:, 1], lo=LAKE_LO, hi=LAKE_HI, pad=LAKE_PAD)


def lake_aspect(xy):
    b = lake_box(xy)
    return (b[1] - b[0]) / (b[3] - b[2])


def ext_lake(xy, aspect):
    return _fit(lake_box(xy), aspect)


def cast_box(xy, pool):
    return _bounds(xy[pool, 0], xy[pool, 1], lo=0.8, hi=99.2, pad=0.24)


def cast_aspect(xy, pool):
    b = cast_box(xy, pool)
    return (b[1] - b[0]) / (b[3] - b[2])


def ext_cast(xy, pool, aspect):
    return _fit(cast_box(xy, pool), aspect)


def _peaks_in_frame(xy, family_id, vocab, wanted, ext, res=260, min_n=400):
    """Densest in-frame cell of each named family.

    `galaxy_render.family_peaks` histograms inside the extent but does not
    check that anything landed there, so a family cropped out by the tighter
    zoom would be labelled at an arbitrary corner.  Here a family is named only
    if enough of it is actually inside the frame.
    """
    x, y = np.asarray(xy[:, 0]), np.asarray(xy[:, 1])
    x0, x1, y0, y1 = ext
    out = []
    for key, label in wanted:
        fid = vocab.get(key)
        if fid is None:
            continue
        m = family_id == fid
        if int(m.sum()) < min_n:
            continue
        fx, fy = x[m], y[m]
        inside = (fx >= x0) & (fx <= x1) & (fy >= y0) & (fy <= y1)
        if int(inside.sum()) < min_n:
            continue
        H, xe, ye = np.histogram2d(fx[inside], fy[inside], bins=res,
                                   range=[[x0, x1], [y0, y1]])
        S = gaussian_filter(H, 2.2)
        i, j = np.unravel_index(np.argmax(S), S.shape)
        out.append({"key": key, "label": label, "n": int(inside.sum()),
                    "x": float((xe[i] + xe[i + 1]) / 2),
                    "y": float((ye[j] + ye[j + 1]) / 2)})
    return out


def _title(ax, letter, title, sub=None):
    ax.text(0.0, 1.016, letter, transform=ax.transAxes, fontsize=FS_LETTER,
            weight="bold", color=INK, ha="left", va="bottom")
    ax.text(0.040, 1.016, title, transform=ax.transAxes, fontsize=FS_TITLE,
            color=INK, ha="left", va="bottom")
    if sub:
        ax.text(1.0, 1.020, sub, transform=ax.transAxes, fontsize=FS_SUB,
                color=FAINT, ha="right", va="bottom")


def _strip(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _clean(ax)
    return ax


# ------------------------------------------------------------------ panel A --
def panel_a(ax, D, ext):
    xy, qxy = D["xy"], D["qxy"]
    x, y = xy[:, 0], xy[:, 1]
    img = density_image(x, y, ext, res=(2400, 2400), smooth=1.35, sharp=0.40,
                        clip=99.80)
    ax.imshow(img, origin="lower", extent=ext, cmap=NEBULA, aspect="auto",
              interpolation="bilinear", zorder=1, rasterized=True,
              vmin=0, vmax=1)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    _clean(ax)
    _frame(ax)

    outside = int((~((x >= ext[0]) & (x <= ext[1]) &
                     (y >= ext[2]) & (y <= ext[3]))).sum())
    _title(ax, "A", "the model lake",
           "%s models, %s of them beyond this frame"
           % (f"{len(xy):,}", f"{outside:,}"))

    qx, qy = _to_axes(qxy[G.CASE_QUERY][0], qxy[G.CASE_QUERY][1], ext)
    blocked = [[qx - 0.055, qy - 0.040, qx + 0.080, qy + 0.040]]
    peaks = _peaks_in_frame(xy, D["family_id"], D["ginfo"]["family_vocab"],
                            NAMED, ext)
    peaks = place_labels(peaks, ext, blocked=blocked, min_sep=0.082,
                         margin=0.028, text_room=0.118)
    for p in peaks:
        ax.plot([p["ax"], p["lx"]], [p["ay"], p["ly"]], transform=ax.transAxes,
                color=INK, lw=0.45, alpha=0.42, zorder=8, solid_capstyle="round")
        ax.scatter([p["ax"]], [p["ay"]], transform=ax.transAxes, s=3.0,
                   color=INK, alpha=0.55, lw=0, zorder=9)
        ha = "left" if p["lx"] >= p["ax"] else "right"
        off = 0.006 if ha == "left" else -0.006
        ax.text(p["lx"] + off, p["ly"], p["label"], transform=ax.transAxes,
                fontsize=FS_FAMILY, color=INK, ha=ha, va="center", zorder=10,
                bbox=dict(boxstyle="round,pad=0.14", fc=PAPER, ec="none",
                          alpha=0.86))

    # where B's cast lands: the contour holding most of that query's pool
    pool_xy = xy[D["case"]["pool"]]
    ph, pxe, pye = np.histogram2d(pool_xy[:, 0], pool_xy[:, 1], bins=110,
                                  range=[[ext[0], ext[1]], [ext[2], ext[3]]])
    ps = gaussian_filter(ph, 3.0)
    flat = np.sort(ps.ravel())[::-1]
    lvl = flat[np.searchsorted(np.cumsum(flat), 0.86 * flat.sum())]
    ax.contour((pxe[:-1] + pxe[1:]) / 2, (pye[:-1] + pye[1:]) / 2, ps.T,
               levels=[lvl], colors=[AMBER], linewidths=1.0, zorder=6, alpha=0.9)
    ax.scatter([qx], [qy], transform=ax.transAxes, s=40, marker="P",
               facecolor=CRIMSON, edgecolor=PAPER, lw=0.9, zorder=13)
    ax.text(qx + 0.017, qy - 0.006, "B", transform=ax.transAxes,
            fontsize=FS_TITLE, weight="bold", color=AMBER, ha="left", va="top",
            zorder=13)


# ------------------------------------------------------------------ panel B --
def panel_b(ax, D, ext, strip=None):
    xy, qxy, case = D["xy"], D["qxy"], D["case"]
    pxy = xy[case["pool"]]
    inbox = ((xy[:, 0] > ext[0]) & (xy[:, 0] < ext[1]) &
             (xy[:, 1] > ext[2]) & (xy[:, 1] < ext[3]))
    sub = xy[inbox]
    img = density_image(sub[:, 0], sub[:, 1], ext, res=(1500, 1500),
                        smooth=1.55, sharp=0.28)
    ax.imshow(img * 0.55, origin="lower", extent=ext, cmap=NEBULA,
              aspect="auto", interpolation="bilinear", zorder=1,
              rasterized=True, vmin=0, vmax=1)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    _clean(ax)
    _frame(ax, color=AMBER, lw=0.8)
    _title(ax, "B", "one query, answered")

    pnorm = case["prior"] / max(case["prior"].max(), 1e-9)
    o = np.argsort(pnorm)
    ax.scatter(pxy[o, 0], pxy[o, 1], s=4.5 + 26 * pnorm[o] ** 1.4, c=pnorm[o],
               cmap=PRIOR_CMAP, vmin=0, vmax=1, lw=0, alpha=0.85, zorder=4,
               rasterized=True)

    gp = xy[case["gold"]]
    qp = qxy[G.CASE_QUERY]

    # The ten returned models are ranks, not measurements, so a disc may be
    # nudged off its model to keep it readable -- but never silently: anything
    # that moves keeps a hairline back to where it belongs.  Two of the ten sit
    # close enough to collide at any printed size.
    anchors, ranks = [], []
    for r, m in enumerate(case["top10"], 1):
        p = gp if int(m) == case["gold"] else xy[int(m)]
        anchors.append(_to_axes(p[0], p[1], ext))
        ranks.append(r)
    anchors = np.array(anchors, float)
    pos = anchors.copy()
    gi = ranks.index(1)
    pos[gi] += (0.052, 0.040)              # rank 1 sits beside its star
    fixed = np.zeros(len(pos), bool)
    fixed[gi] = True

    SEP, PULL = 0.036, 0.30
    for _ in range(400):
        moved = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                d = pos[i] - pos[j]
                n = float(np.hypot(*d))
                if n >= SEP:
                    continue
                if n < 1e-9:
                    d, n = np.array([1.0, 0.0]), 1.0
                push = (SEP - n) / 2.0 * (d / n)
                if not fixed[i]:
                    pos[i] += push
                if not fixed[j]:
                    pos[j] -= push
                moved = True
        for i in range(len(pos)):
            if not fixed[i]:
                pos[i] += PULL * (anchors[i] - pos[i]) * 0.10
        if not moved:
            break
    pos = np.clip(pos, 0.018, 0.982)

    for k, r in enumerate(ranks):
        ax_, ay_ = anchors[k]
        lx, ly = pos[k]
        if np.hypot(lx - ax_, ly - ay_) > 0.012:
            ax.plot([ax_, lx], [ay_, ly], transform=ax.transAxes, color=INK,
                    lw=0.45, alpha=0.55, zorder=13, solid_capstyle="round")
            ax.scatter([ax_], [ay_], transform=ax.transAxes, s=3.0, color=INK,
                       alpha=0.7, lw=0, zorder=13)
        gold_badge = r == 1
        ax.scatter([lx], [ly], transform=ax.transAxes,
                   s=62 if gold_badge else 46,
                   facecolor=GOLD if gold_badge else PAPER,
                   edgecolor=PAPER if gold_badge else INK,
                   lw=0.8 if gold_badge else 0.65,
                   zorder=18 if gold_badge else 14, alpha=1.0)
        ax.text(lx, ly, str(r), transform=ax.transAxes,
                fontsize=5.2 if gold_badge else 4.9,
                color=PAPER if gold_badge else INK, ha="center", va="center",
                zorder=19 if gold_badge else 15, weight="bold")
    ax.annotate("", xy=(gp[0], gp[1]), xytext=(qp[0], qp[1]), zorder=5,
                arrowprops=dict(arrowstyle="-", color=GOLD, lw=1.0, alpha=0.8,
                                connectionstyle="arc3,rad=0.24"))
    ax.scatter([gp[0]], [gp[1]], s=250, marker="*", facecolor=GOLD_LT,
               edgecolor=INK, lw=0.7, zorder=16)
    # the star carries its badge like the other nine, so the rank the figure is
    # about does not depend on the caption
    bx = gp[0] + 0.052 * (ext[1] - ext[0])
    by = gp[1] + 0.040 * (ext[3] - ext[2])
    ax.scatter([bx], [by], s=62, facecolor=GOLD, edgecolor=PAPER, lw=0.8,
               zorder=18)
    ax.text(bx, by, "1", fontsize=5.2, color=PAPER, ha="center", va="center",
            zorder=19, weight="bold")
    ax.scatter([qp[0]], [qp[1]], s=80, marker="P", facecolor=CRIMSON,
               edgecolor=PAPER, lw=1.0, zorder=17)

    if strip is not None:
        items = [("*", GOLD_LT, INK, 74,
                  "held-out gold model:  %s" % D["model_name"][case["gold"]]),
                 ("P", CRIMSON, PAPER, 40, "query, at its dense top-64 centroid"),
                 ("o", AMBER, "none", 22, "candidate the task prior lifts"),
                 ("o", "#a4abbb", "none", 15, "candidate the prior never touches")]
        for i, (mk, fc, ec, s, txt) in enumerate(items):
            col, row = i % 2, i // 2
            xx, yy = 0.012 + col * 0.500, 0.72 - row * 0.46
            strip.scatter([xx], [yy], s=s, marker=mk, facecolor=fc,
                          edgecolor=ec, lw=0.6, clip_on=False)
            strip.text(xx + 0.022, yy, txt, fontsize=FS_LEGEND, color=SUBINK,
                       ha="left", va="center")



# ----------------------------------------------------------------- builders --
def _save(fig, name, dpi=600):
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(os.path.join(OUTDIR, "%s.%s" % (name, ext)),
                    facecolor=PAPER, dpi=dpi, bbox_inches="tight",
                    pad_inches=0.014)
    plt.close(fig)
    print("  wrote", name, "(pdf, svg, png)")


def _ratio(rect, w, h):
    return (rect[2] * w) / (rect[3] * h)


def build_a(D):
    """Portrait, because the lake is: the panel takes the lake's own aspect."""
    panel = [0.034, 0.018, 0.948, 0.930]
    PW = 5.90                                    # printed panel width, inches
    W = PW / panel[2]
    H = (PW / lake_aspect(D["xy"])) / panel[3]
    fig = plt.figure(figsize=(W, H))
    panel_a(fig.add_axes(panel), D, ext_lake(D["xy"], _ratio(panel, W, H)))
    _save(fig, "lake_a")


def build_b(D):
    """The cast at its own aspect, with the legend in a strip underneath."""
    PW = 6.30                                    # printed panel width, inches
    ph = PW / cast_aspect(D["xy"], D["case"]["pool"])
    lpad, rpad, top, strip_h, bot = 0.10, 0.10, 0.28, 0.46, 0.10
    W = lpad + PW + rpad
    H = bot + strip_h + ph + top
    panel = [lpad / W, (bot + strip_h) / H, PW / W, ph / H]
    fig = plt.figure(figsize=(W, H))
    panel_b(fig.add_axes(panel), D,
            ext_cast(D["xy"], D["case"]["pool"], _ratio(panel, W, H)),
            strip=_strip(fig, [panel[0], 0.06 / H, panel[2], (strip_h - 0.10) / H]))
    _save(fig, "lake_b")


def build_pair(D):
    """The same two tuned panels, side by side at equal height.

    Neither panel is reshaped to fit the row: each keeps the aspect ratio its
    own data asked for, and the row is sized around them.  A is portrait
    because the lake is, B is landscape because the cast is.
    """
    ph = 4.55                                    # shared panel height, inches
    wa = ph * lake_aspect(D["xy"])
    wb = ph * cast_aspect(D["xy"], D["case"]["pool"])
    lpad, gap, rpad = 0.10, 0.42, 0.10
    top, strip_h, bot = 0.30, 0.46, 0.10
    W = lpad + wa + gap + wb + rpad
    H = bot + strip_h + ph + top
    fa = [lpad / W, (bot + strip_h) / H, wa / W, ph / H]
    fb = [(lpad + wa + gap) / W, (bot + strip_h) / H, wb / W, ph / H]
    fig = plt.figure(figsize=(W, H))
    panel_a(fig.add_axes(fa), D, ext_lake(D["xy"], _ratio(fa, W, H)))
    panel_b(fig.add_axes(fb), D,
            ext_cast(D["xy"], D["case"]["pool"], _ratio(fb, W, H)),
            strip=_strip(fig, [fb[0], 0.06 / H, fb[2], (strip_h - 0.10) / H]))
    _save(fig, "lake_ab")


def main():
    _style()
    plt.rcParams.update({"font.sans-serif": SANS, "svg.fonttype": "path"})
    print("reading", G.CACHE)
    D = load()
    build_a(D)
    build_b(D)
    build_pair(D)
    print("figures in", OUTDIR)


if __name__ == "__main__":
    main()
