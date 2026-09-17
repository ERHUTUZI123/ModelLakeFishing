"""lake_ab.py -- the lake, and one query answered, as paper figures.

A stripped rebuild of panels A and B of `galaxy_render.py`'s plate.  No poster
header, no cascade, no rank panels, no footer: two boxes, a short title over
each, and a legend strip underneath, the way a figure like this is normally
set.

    A   the whole lake, complete, with the family territories named and the
        outline of the region B enlarges
    B   one held-out query, the 1,000 candidates HNSW returned for it, the
        ten that came back, and where the gold model landed

They are two independent figures, not two halves of one, so each is drawn at
its own natural aspect ratio rather than squeezed into a shared row.  That
matters most for A: it takes whatever shape the lake's own percentile box
turns out to have, because a frame of the figure's choosing is what left the
lake small and structureless in the middle of white paper.  A shows the whole
lake, and the rows its frame leaves out -- 64,086 of three million under A0,
where the pre-A0 layout left out 4,416 -- are counted in the subtitle rather
than quietly dropped.

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
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
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
# own percentile box and the panel is given that box's aspect ratio, whatever
# that turns out to be.  The point is that no shape is imposed -- forcing the
# lake into a box of the figure's choosing is what used to leave it small and
# structureless in the middle of white paper.  Under A0 the box comes out very
# nearly square (0.97); the pre-A0 layout was markedly portrait.
LAKE_LO, LAKE_HI, LAKE_PAD = 0.10, 99.90, 0.04
# The lake trails off downward with nothing but white between, and at plate
# size that emptiness is most of the page.  The frame therefore starts at this
# percentile of y: what it leaves out is counted under the figure, never
# quietly dropped.  Under A0 the frame leaves out 64,086 rows, 2.1% -- more
# than the pre-A0 layout's 4,416, because A0's lake has a longer sparse fringe.
LAKE_Y_LO = 2.5

NAMED = [("llama", "llama"), ("qwen", "qwen"), ("gemma", "gemma"),
         ("mistral", "mistral"), ("bert", "bert"), ("roberta", "roberta"),
         ("xlm-roberta", "xlm-roberta"), ("t5", "t5"), ("gpt2", "gpt2"),
         ("distilbert", "distilbert"), ("whisper", "whisper"),
         ("wav2vec2", "wav2vec2"), ("vit", "vit"), ("marian", "marian"),
         ("stablediffusion", "stable-diffusion"), ("flux", "flux"),
         ("blockassist", "blockassist")]

# The lake is coloured the way the 3-D viewer colours it in "families" mode,
# and by the same table: the palette is read out of lake3d/page.py and the
# family order out of the pack the viewer ships, so a family is the same
# colour here as it is there.  Copying the list by hand is what once left the
# viewer painting bert the query's crimson, so nothing is copied.
LAKE3D = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "lake3d")
LAKE3D_PACK = r"D:/research/model_lake/data/data1m/lake3d_a0/packed_meta.json"
OTHER_COLOR = "#7b8398"
N_FAMILIES = 26


def lake3d_families():
    """name -> colour, in the viewer's own order (biggest family first)."""
    src = open(os.path.join(LAKE3D, "page.py"), encoding="utf-8").read()
    pal = re.findall(r'"(#[0-9a-f]{6})"',
                     src[src.index("FAMILY_COLORS = ["):
                         src.index("]" + chr(10) + "OTHER_COLOR")])
    meta = json.load(open(LAKE3D_PACK, encoding="utf-8"))
    return [(f["name"], int(f["id"]), pal[i])
            for i, f in enumerate(meta["families"][:len(pal)])]

FS_LETTER, FS_TITLE, FS_SUB = 16.0, 16.0, 16.0
FS_FAMILY, FS_NOTE, FS_LEGEND = 13.0, 16.0, 16.0
FS_CASE, FS_CASE_SUB = 16.0, 16.0
# The numbered disc, as matplotlib wants it (marker area in points squared) and
# as the page sees it (printed diameter in inches).  The second is derived from
# the first so the size the discs are drawn at and the distance they are held
# apart at cannot drift away from one another.
DISC_PT2 = 470.0
DISC_IN = 2.0 * np.sqrt(DISC_PT2 / np.pi) / 72.0


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
    b = _bounds(xy[:, 0], xy[:, 1], lo=LAKE_LO, hi=LAKE_HI, pad=LAKE_PAD)
    y0 = float(np.percentile(xy[:, 1], LAKE_Y_LO))
    return [b[0], b[1], y0 - (b[3] - y0) * LAKE_PAD, b[3]]


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


def frame_families(D, ext, k=N_FAMILIES, min_n=1):
    """The viewer's families, in the viewer's order, that this frame shows.

    A family the frame barely holds is left to the context layer rather than
    given a swatch nobody can find, but the colour of the ones that stay is
    the viewer's, not one this figure chose.
    """
    x, y = D["xy"][:, 0], D["xy"][:, 1]
    inside = (x >= ext[0]) & (x <= ext[1]) & (y >= ext[2]) & (y <= ext[3])
    u, c = np.unique(D["family_id"][inside], return_counts=True)
    seen = dict(zip(u.tolist(), c.tolist()))
    out = []
    for name, fid, color in lake3d_families()[:k]:
        n = int(seen.get(fid, 0))
        if n >= min_n:
            out.append({"id": fid, "label": name, "n": n, "color": color})
    return out


def outside_frame(xy, ext):
    """Rows the frame leaves out -- each figure reports its own count."""
    x, y = np.asarray(xy[:, 0]), np.asarray(xy[:, 1])
    return int((~((x >= ext[0]) & (x <= ext[1]) &
                  (y >= ext[2]) & (y <= ext[3]))).sum())


def _hist(x, y, ext, res):
    H, _, _ = np.histogram2d(x, y, bins=res,
                             range=[[ext[0], ext[1]], [ext[2], ext[3]]])
    return H


def family_image(D, ext, fams, res=(2000, 2000), smooth=1.35, sharp=0.40,
                 clip=99.80, blur=1.7):
    """The lake, painted by family: hue says who lives there, value says how many.

    Ink is the total density -- the same image the one-colour plate drew -- so
    the shape of the lake is untouched.  Colour is decided per cell by which
    family leads it, with cells no named family leads left in the context grey.
    Territories are therefore readable without any cell being invented.
    """
    x, y = D["xy"][:, 0], D["xy"][:, 1]
    ink = density_image(x, y, ext, res=res, smooth=smooth, sharp=sharp,
                        clip=clip)                      # (H, W), 0..1
    fid = D["family_id"]
    named = np.stack([gaussian_filter(np.log1p(_hist(x[fid == f["id"]],
                                                     y[fid == f["id"]],
                                                     ext, res)), blur).T
                      for f in fams])                   # (F, H, W)
    rest = ~np.isin(fid, [f["id"] for f in fams])
    other = gaussian_filter(np.log1p(_hist(x[rest], y[rest], ext, res)), blur).T

    lead = named.argmax(0)
    top = named.max(0)
    cols = np.array([to_rgb(f["color"]) for f in fams])
    rgb = np.where((top > other * 0.35)[..., None], cols[lead],
                   to_rgb(OTHER_COLOR))

    # a named territory carries a little more ink than the context it sits in
    a = np.clip(ink * np.where(top > other * 0.35, 1.50, 1.00), 0, 1)[..., None]
    return 1.0 - a * (1.0 - rgb)


def family_peaks(D, ext, fams, res=300):
    """Densest in-frame cell of each named family, for its label."""
    x, y = D["xy"][:, 0], D["xy"][:, 1]
    out = []
    for f in fams:
        m = D["family_id"] == f["id"]
        H = gaussian_filter(_hist(x[m], y[m], ext, res), 2.4)
        i, j = np.unravel_index(np.argmax(H), H.shape)
        xe = np.linspace(ext[0], ext[1], res + 1)
        ye = np.linspace(ext[2], ext[3], res + 1)
        out.append(dict(f, x=float((xe[i] + xe[i + 1]) / 2),
                        y=float((ye[j] + ye[j + 1]) / 2)))
    return out


def _title(ax, letter, title, sub=None, sub_below=False):
    """Letter and title on one line, with the count, if any, set beside them.

    The count is as large and as black as the title -- it is a headline
    number, not a footnote -- which makes it too wide to share the title's
    line, so it goes on lines of its own: stacked above the title (bottom
    line first), or, with `sub_below`, hung under the panel (top line first).
    """
    if letter:
        ax.text(0.0, 1.016, letter, transform=ax.transAxes, fontsize=FS_LETTER,
                weight="bold", color=INK, ha="left", va="bottom")
    ax.text(0.075 if letter else 0.0, 1.016, title, transform=ax.transAxes,
            fontsize=FS_TITLE, color=INK, ha="left", va="bottom")
    if sub:
        lines = [sub] if isinstance(sub, str) else list(sub)
        for i, line in enumerate(lines):
            if sub_below:
                ax.text(0.0, -0.026 - 0.062 * i, line, transform=ax.transAxes,
                        fontsize=FS_SUB, color=INK, ha="left", va="top")
            else:
                ax.text(1.0, 1.072 + 0.062 * i, line, transform=ax.transAxes,
                        fontsize=FS_SUB, color=INK, ha="right", va="bottom")


def _axes_size_in(ax):
    """Printed size of an axes, in inches -- the pills are sized in points."""
    fw, fh = ax.figure.get_size_inches()
    bb = ax.get_position()
    return bb.width * fw, bb.height * fh


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

    outside = outside_frame(xy, ext)
    _title(ax, "A", "the model lake",
           ["%s models" % f"{len(xy):,}",
            "%s of them beyond this frame" % f"{outside:,}"],
           sub_below=True)

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
    ax.text(qx + 0.022, qy - 0.006, "B", transform=ax.transAxes,
            fontsize=FS_TITLE, weight="bold", color=AMBER, ha="left", va="top",
            zorder=13)


# ------------------------------------------------------------------ panel B --
def panel_b(ax, D, ext, strip=None, letter="B", title="one query, answered",
            sub=None, fams=None, cols=None):
    xy, qxy, case = D["xy"], D["qxy"], D["case"]
    pxy = xy[case["pool"]]
    fams = frame_families(D, ext) if fams is None else fams
    ax.imshow(family_image(D, ext, fams), origin="lower", extent=ext,
              aspect="auto", interpolation="bilinear", zorder=1,
              rasterized=True)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    _clean(ax)
    _frame(ax, color=HAIR, lw=0.8)
    _title(ax, letter, title, sub, sub_below=True)

    pnorm = case["prior"] / max(case["prior"].max(), 1e-9)
    o = np.argsort(pnorm)
    ax.scatter(pxy[o, 0], pxy[o, 1], s=4.5 + 26 * pnorm[o] ** 1.4, c=pnorm[o],
               cmap=PRIOR_CMAP, vmin=0, vmax=1, lw=0, alpha=0.85, zorder=4,
               rasterized=True)

    gp = xy[case["gold"]]
    qp = qxy[G.CASE_QUERY]

    # The ten that came back are ranks, not measurements, so a disc may be
    # nudged off its model to keep it readable -- but never silently: anything
    # that moves keeps a hairline back to where it belongs.  Which family each
    # one belongs to is read off the colour of the water it sits in, named in
    # the legend, so the disc carries only its depth in the list.
    anchors = np.array([_to_axes(*(gp if int(m) == case["gold"] else xy[int(m)]),
                                 ext) for m in case["top10"]], float)
    ranks = list(range(1, len(anchors) + 1))
    pos = anchors.copy()
    where = np.flatnonzero(case["top10"] == case["gold"])
    if not where.size:
        raise AssertionError("query %d: the gold model is not among the ten "
                             "returned, so panel B has no star to draw"
                             % G.CASE_QUERY)
    gi = int(where[0])
    # The gold is drawn once, as the star, with no numbered disc of its own;
    # the legend states where it came back.  Its slot stays fixed exactly on
    # the star, so the slot's only job is to keep the other discs off it.
    #
    # This reads cleanly only while the gold comes back first, as it does for
    # the squad case.  If the case is ever changed to one whose gold is
    # returned lower, the plate will show a star, a separate "1", and no disc
    # for the gold's own rank -- give the gold its own numbered disc back then.
    fixed = np.zeros(len(pos), bool)
    fixed[gi] = True

    # Discs are pushed apart in printed inches, not in axes units.  A single
    # number in axes units is a different printed distance on every panel: it
    # held the discs 1.44 of their own widths apart on the wide pair plate,
    # which turned the five near-coincident models of a returned group into a
    # chain marching off toward the frame's corner, and only 0.71 of a width
    # apart on the narrower B, where they could still touch.  Separating by
    # the disc's own diameter keeps a tight group tight and legible on both.
    w_in, h_in = _axes_size_in(ax)
    inch = np.array([w_in, h_in], float)
    SEP = DISC_IN * 1.12                   # the disc, plus a hair of air
    PULL = 0.30
    P, Anc = pos * inch, anchors * inch    # everything below is in inches
    for _ in range(500):
        moved = False
        for i in range(len(P)):
            for j in range(i + 1, len(P)):
                d = P[i] - P[j]
                n = float(np.hypot(*d))
                if n >= SEP:
                    continue
                if n < 1e-9:
                    d, n = np.array([1.0, 0.0]), 1.0
                push = (SEP - n) / 2.0 * (d / n)
                if not fixed[i]:
                    P[i] += push
                if not fixed[j]:
                    P[j] -= push
                moved = True
        for i in range(len(P)):
            if not fixed[i]:
                P[i] += PULL * (Anc[i] - P[i]) * 0.10
        if not moved:
            break
    pos = np.clip(P / inch, 0.014, 0.986)

    ax.annotate("", xy=(gp[0], gp[1]), xytext=(qp[0], qp[1]), zorder=5,
                arrowprops=dict(arrowstyle="-", color=GOLD, lw=1.2, alpha=0.85,
                                connectionstyle="arc3,rad=0.24"))
    ax.scatter([gp[0]], [gp[1]], s=430, marker="*", facecolor=GOLD_LT,
               edgecolor=INK, lw=0.9, zorder=16)

    gold_rank = ranks[gi]
    for k, r in enumerate(ranks):
        if k == gi:                            # drawn once, as the star
            continue
        ax_, ay_ = anchors[k]
        lx, ly = pos[k]
        if np.hypot(lx - ax_, ly - ay_) > 0.010:
            ax.plot([ax_, lx], [ay_, ly], transform=ax.transAxes, color=INK,
                    lw=0.6, alpha=0.55, zorder=13, solid_capstyle="round")
            ax.scatter([ax_], [ay_], transform=ax.transAxes, s=6.0, color=INK,
                       alpha=0.75, lw=0, zorder=13)
        ax.scatter([lx], [ly], transform=ax.transAxes, s=DISC_PT2,
                   facecolor=PAPER, edgecolor=INK, lw=1.1, zorder=14)
        ax.text(lx, ly, str(r), transform=ax.transAxes, fontsize=FS_CASE,
                color=INK, ha="center", va="center", weight="bold", zorder=15)
    ax.scatter([qp[0]], [qp[1]], s=180, marker="P", facecolor=CRIMSON,
               edgecolor=PAPER, lw=1.2, zorder=17)

    if strip is not None:
        legend_strip(strip, D, case, fams, gold_rank,
                     cols or legend_cols(_axes_size_in(strip)[0], fams))


LEGEND_COLS_MAX = 5
# A legend column has to hold a swatch, its gap, and the longest name in the
# list.  The legend is set at one size in one font, so a per-character advance
# is enough to pick a column count that does not overrun.  Five columns fit
# the wide pair plate; on the narrower single-panel B they do not, and
# "stablediffusion" was running into "test".
LEGEND_CHAR_IN = FS_LEGEND / 72.0 * 0.55     # advance of one character, inches
LEGEND_SWATCH_IN = 0.26                      # the swatch and its gap


def _ordinal(n):
    return {1: "first", 2: "second", 3: "third"}.get(n, "%dth" % n)


def legend_cols(width_in, fams):
    """As many columns as the widest entry will actually fit in."""
    longest = max([len(f["label"]) for f in fams] + [len("no named family")])
    need = LEGEND_SWATCH_IN + longest * LEGEND_CHAR_IN
    return max(1, min(LEGEND_COLS_MAX, int(width_in / need)))


LEGEND_MARKS = 3        # star, query, numbered disc


def legend_rows(fams, cols):
    """How many lines the legend needs: the marks, then the families."""
    return LEGEND_MARKS + int(np.ceil((len(fams) + 1) / float(cols)))


def legend_strip(strip, D, case, fams, gold_rank, cols):
    """The marks, one per line, then the family colours in a swatch grid.

    The families are named here rather than on the water: at lake scale a name
    laid over its own territory either covers it or points at it from far
    away, and there are two dozen of them.  The last swatch is the water that
    belongs to no named family.

    `cols` is passed in rather than chosen here: the caller sized the strip
    from it, and a grid laid out on a different count than the strip was cut
    for is how the rows end up on top of one another.
    """
    n = legend_rows(fams, cols)
    step = 1.0 / n
    y = lambda r: 1.0 - step * (r + 0.5)

    strip.scatter([0.012], [y(0)], s=230, marker="*", facecolor=GOLD_LT,
                  edgecolor=INK, lw=0.8, clip_on=False)
    strip.text(0.030, y(0), "held-out gold model, returned %s:  %s"
               % (_ordinal(gold_rank), D["model_name"][case["gold"]]),
               fontsize=FS_LEGEND, color=INK, ha="left", va="center")

    strip.scatter([0.012], [y(1)], s=130, marker="P", facecolor=CRIMSON,
                  edgecolor=PAPER, lw=0.8, clip_on=False)
    strip.text(0.030, y(1), "query, at its dense top-64 centroid",
               fontsize=FS_LEGEND, color=INK, ha="left", va="center")

    strip.scatter([0.012], [y(2)], s=DISC_PT2, facecolor=PAPER, edgecolor=INK,
                  lw=1.1, clip_on=False, zorder=4)
    strip.text(0.012, y(2), "n", fontsize=FS_LEGEND, color=INK, ha="center",
               va="center", weight="bold", zorder=5, clip_on=False)
    strip.text(0.030, y(2), "the n-th model returned, 1 the highest ranked",
               fontsize=FS_LEGEND, color=INK, ha="left", va="center")

    w = 1.0 / cols
    swatches = [(f["label"], f["color"]) for f in fams]
    swatches.append(("no named family", OTHER_COLOR))
    for i, (label, color) in enumerate(swatches):
        r, c = LEGEND_MARKS + i // cols, i % cols
        x = 0.012 + c * w
        strip.add_patch(plt.Rectangle((x, y(r) - step * 0.26), 0.017,
                                      step * 0.52, facecolor=color,
                                      edgecolor="none", clip_on=False))
        strip.text(x + 0.026, y(r), label, fontsize=FS_LEGEND, color=INK,
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
    """The panel takes the lake's own aspect, whatever shape that is."""
    panel = [0.034, 0.130, 0.948, 0.830]
    PW = 5.90                                    # printed panel width, inches
    W = PW / panel[2]
    H = (PW / lake_aspect(D["xy"])) / panel[3]
    fig = plt.figure(figsize=(W, H))
    ext = ext_lake(D["xy"], _ratio(panel, W, H))
    panel_a(fig.add_axes(panel), D, ext)
    _save(fig, "lake_a")
    print("    frame leaves out %s rows" % f"{outside_frame(D['xy'], ext):,}")


def build_b(D):
    """The cast at its own aspect, with the legend in a strip underneath.

    The strip is sized from the number of rows the legend actually needs, the
    way `build_pair` sizes its own.  It used to be a fixed 1.45 inches, which
    held while the cast frame was tight enough to show only a handful of
    families; A0's cast covers most of the lake, all 26 families fall inside
    it, and nine rows of legend piled into that fixed strip on top of one
    another.
    """
    PW = 6.30                                    # printed panel width, inches
    ph = PW / cast_aspect(D["xy"], D["case"]["pool"])
    lpad, rpad, top, bot = 0.10, 0.10, 0.45, 0.10
    # the same list is measured and then drawn, so the strip cannot be sized
    # for one legend and filled with another
    fams = frame_families(D, cast_box(D["xy"], D["case"]["pool"]))
    cols = legend_cols(PW, fams)
    strip_h = 0.30 + 0.36 * legend_rows(fams, cols)
    W = lpad + PW + rpad
    H = bot + strip_h + ph + top
    panel = [lpad / W, (bot + strip_h) / H, PW / W, ph / H]
    fig = plt.figure(figsize=(W, H))
    panel_b(fig.add_axes(panel), D,
            ext_cast(D["xy"], D["case"]["pool"], _ratio(panel, W, H)),
            strip=_strip(fig, [panel[0], 0.06 / H, panel[2], (strip_h - 0.10) / H]),
            fams=fams, cols=cols)
    _save(fig, "lake_b")


def build_pair(D):
    """One plate: the whole lake, painted by family, with a query answered on it.

    Panel A is gone -- this is the big figure, so it takes the lake's own
    frame and its own aspect, the ten answers keep their depth in the list,
    and the family colours are named in the legend underneath.
    """
    PW = 10.60                                   # printed panel width, inches
    ph = PW / lake_aspect(D["xy"])
    lpad, rpad, top, bot = 0.16, 0.16, 0.10, 0.16
    W = lpad + PW + rpad

    fams = frame_families(D, lake_box(D["xy"]))  # only what the frame shows
    cols = legend_cols(PW, fams)
    strip_h = 0.30 + 0.36 * legend_rows(fams, cols)
    H = bot + strip_h + ph + top
    panel = [lpad / W, (bot + strip_h) / H, PW / W, ph / H]
    ext = ext_lake(D["xy"], _ratio(panel, W, H))
    fig = plt.figure(figsize=(W, H))
    panel_b(fig.add_axes(panel), D, ext,
            strip=_strip(fig, [panel[0], 0.10 / H, panel[2],
                               (strip_h - 0.16) / H]),
            letter="", title="", fams=fams, cols=cols)
    _save(fig, "lake_ab")
    print("    frame leaves out %s rows, legend in %d columns"
          % (f"{outside_frame(D['xy'], ext):,}", cols))


def caption_facts(D):
    """The numbers `lake_ab_captions.tex` quotes, so the prose can be checked.

    The captions are written by hand and the figures are not, which is exactly
    how a caption ends up a run behind its picture.  Printing the figure's own
    numbers here is what lets the two be reconciled after every rebuild.
    """
    xy, case = D["xy"], D["case"]
    named = _peaks_in_frame(xy, D["family_id"], D["ginfo"]["family_vocab"],
                            NAMED, ext_lake(xy, lake_aspect(xy)))
    gold_rank = int(np.flatnonzero(case["top10"] == case["gold"])[0]) + 1
    print("\ncaption facts   (each figure's own frame count is printed above)")
    print("  models                     %s" % f"{len(xy):,}")
    print("  families, named in A       %s, %d"
          % (f"{D['ginfo']['n_families']:,}", len(named)))
    print("  query                      %s"
          % D["ds_name"][G.CASE_QUERY].replace("\t", " / "))
    print("  candidate pool             %s, from HNSW" % f"{len(case['pool']):,}")
    print("  gold                       %s" % D["model_name"][case["gold"]])
    print("  gold rank by cosine alone  %d of %d"
          % (case["dense_rank"], len(case["pool"])))
    print("  gold after the task prior  returned %s" % _ordinal(gold_rank))
    print("  panel aspects  A %.2f  B %.2f"
          % (lake_aspect(xy), cast_aspect(xy, case["pool"])))


def main():
    _style()
    plt.rcParams.update({"font.sans-serif": SANS, "svg.fonttype": "path"})
    print("reading", G.CACHE)
    D = load()
    build_a(D)
    build_b(D)
    build_pair(D)
    print("figures in", OUTDIR)
    caption_facts(D)


if __name__ == "__main__":
    main()
