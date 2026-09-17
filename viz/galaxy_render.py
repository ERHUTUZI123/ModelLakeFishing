"""galaxy_render.py -- draws the plate from the cache built by model_lake_galaxy.

The plate is one landscape figure on white:

    header   title, and the cascade 3,016,439 -> 1,000 -> 10 with its costs
    A        the lake: log-density of all 3,016,439 model rows, family
             territories named in place, inset of the 46,146 luminous rows,
             and the outline of the cast that panel B enlarges
    B        one cast: the dense top-1,000 pool of a single held-out query
             drawn in the same coordinates
    C        what came back: the ten returned models, dense rank -> final rank
    D        rank migration for every seed-0 query whose gold reached the pool

Only matplotlib is used.  The million-row layers are rasterised inside an
otherwise vector PDF, so the file stays small and the type stays sharp.
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

from ModelLakeFishing.viz import model_lake_galaxy as G

# ------------------------------------------------------------------ palette --
PAPER = "#ffffff"
INK = "#12172a"
SUBINK = "#5d6579"
FAINT = "#8b93a6"
HAIR = "#ccd1dc"
GHOST = "#eef1f7"
GOLD = "#b0730f"
GOLD_LT = "#edb75a"
CRIMSON = "#a8264a"
AMBER = "#d9541b"
DEEP = "#2f3470"
TEAL = "#166f7c"

NEBULA = LinearSegmentedColormap.from_list("lakeink", [
    (0.000, "#ffffff"), (0.055, "#f1f3fc"), (0.150, "#d8dff3"),
    (0.300, "#adb9e4"), (0.470, "#7f8bca"), (0.640, "#535ca9"),
    (0.800, "#343783"), (1.000, "#15163c")])
LUMEN = LinearSegmentedColormap.from_list("lumen", [
    (0.000, "#ffffff"), (0.070, "#fdf2dd"), (0.260, "#f6d99b"),
    (0.500, "#e7ae4f"), (0.740, "#cb7f16"), (1.000, "#82470a")])
PRIOR_CMAP = LinearSegmentedColormap.from_list("prior", [
    (0.0, "#a4abbb"), (0.30, "#95a3c1"), (0.60, "#cd9a4c"), (1.0, "#d9541b")])

SANS = ["Segoe UI", "Calibri", "Arial", "DejaVu Sans"]
MONO = ["Consolas", "DejaVu Sans Mono"]

# panel rectangles, in figure coordinates
FIGSIZE = (15.6, 9.85)
R_HEADER = [0.000, 0.893, 1.000, 0.107]
R_A = [0.030, 0.052, 0.512, 0.795]
R_B = [0.586, 0.520, 0.394, 0.327]
R_C = [0.586, 0.272, 0.394, 0.190]
R_D = [0.586, 0.066, 0.394, 0.132]
R_FOOT = [0.000, 0.000, 1.000, 0.030]


def _style():
    plt.rcParams.update({
        "figure.facecolor": PAPER, "savefig.facecolor": PAPER,
        "axes.facecolor": PAPER, "font.family": "sans-serif",
        "font.sans-serif": SANS, "text.color": INK,
        "axes.edgecolor": HAIR, "axes.labelcolor": SUBINK,
        "xtick.color": SUBINK, "ytick.color": SUBINK,
        "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _clean(ax):
    """Strip every default axis decoration, minor log ticks included."""
    from matplotlib.ticker import NullLocator
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(which="both", length=0)


def _frame(ax, lw=0.7, color=HAIR):
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                               ec=color, lw=lw, zorder=40, clip_on=False))


def _tag(ax, letter, title, sub=None, dy=1.022):
    ax.text(0.0, dy, letter, transform=ax.transAxes, fontsize=10.5, weight="bold",
            color=INK, ha="left", va="bottom")
    ax.text(0.030, dy, title, transform=ax.transAxes, fontsize=9.6, color=INK,
            ha="left", va="bottom")
    if sub:
        ax.text(1.0, dy + 0.004, sub, transform=ax.transAxes, fontsize=7.4,
                color=FAINT, ha="right", va="bottom", linespacing=1.55,
                multialignment="right")


def _ax_ratio(rect):
    return (rect[2] * FIGSIZE[0]) / (rect[3] * FIGSIZE[1])


# ------------------------------------------------------------------ density --
def density_image(x, y, extent, res, smooth=1.35, sharp=0.34, clip=99.94):
    """Log-density on a fixed grid, softened once and mixed back with itself.

    The soft copy makes the diffuse outskirts legible on white; the sharp copy
    keeps dense knots from dissolving.  Normalisation is against a high
    percentile of occupied cells so one hub does not consume the whole ramp.
    """
    x0, x1, y0, y1 = extent
    H, _xe, _ye = np.histogram2d(x, y, bins=res, range=[[x0, x1], [y0, y1]])
    L = np.log1p(H)
    soft = gaussian_filter(L, smooth)
    occupied = L[L > 0]
    hi = np.percentile(occupied, clip) if occupied.size else 1.0
    scale = L.max() / max(hi, 1e-9)
    img = (1.0 - sharp) * (soft / max(soft.max(), 1e-9) * scale) + sharp * (L / max(hi, 1e-9))
    return np.clip(img, 0.0, 1.0).T


def _bounds(x, y, lo=0.04, hi=99.96, pad=0.10):
    x0, x1 = np.percentile(x, [lo, hi])
    y0, y1 = np.percentile(y, [lo, hi])
    mx, my = (x1 - x0) * pad, (y1 - y0) * pad
    return [x0 - mx, x1 + mx, y0 - my, y1 + my]


def _fit(extent, aspect):
    """Grow an extent so it exactly fills a panel of the given width/height."""
    x0, x1, y0, y1 = extent
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    return [cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2]


def _to_axes(px, py, extent):
    return ((px - extent[0]) / (extent[1] - extent[0]),
            (py - extent[2]) / (extent[3] - extent[2]))


# -------------------------------------------------------------- family peaks --
def family_peaks(xy, family_id, vocab, wanted, extent, res=260, min_n=300):
    """The densest grid cell of each named family, used to anchor its label."""
    x, y = np.asarray(xy[:, 0]), np.asarray(xy[:, 1])
    x0, x1, y0, y1 = extent
    out = []
    for key, label in wanted:
        fid = vocab.get(key)
        if fid is None:
            continue
        m = family_id == fid
        if int(m.sum()) < min_n:
            continue
        H, xe, ye = np.histogram2d(x[m], y[m], bins=res, range=[[x0, x1], [y0, y1]])
        S = gaussian_filter(H, 2.2)
        i, j = np.unravel_index(np.argmax(S), S.shape)
        out.append({"key": key, "label": label, "n": int(m.sum()),
                    "x": float((xe[i] + xe[i + 1]) / 2),
                    "y": float((ye[j] + ye[j + 1]) / 2)})
    return out


def place_labels(peaks, extent, blocked=(), min_sep=0.085, margin=0.045,
                 text_room=0.105):
    """Slide each label outward along its ray until it is clear of everything.

    Labels sit outside their own territory and are joined back to it by a
    leader, so no word ever covers the region it names.  `blocked` holds axes-
    coordinate rectangles (the inset, the caption, the marked query) that
    labels must avoid.  A label pushed leftward is set right of its anchor and
    grows to the left, so the usable margin on that side is `text_room`, not
    `margin`; the same applies mirrored on the right.
    """
    cx = cy = 0.5
    taken = []

    def free(lx, ly, leftward):
        lo = text_room if leftward else margin
        hi = 1 - (margin if leftward else text_room)
        if not (lo < lx < hi and margin < ly < 1 - margin):
            return False
        for bx0, by0, bx1, by1 in blocked:
            if bx0 - 0.02 < lx < bx1 + 0.02 and by0 - 0.02 < ly < by1 + 0.02:
                return False
        return all(np.hypot(lx - a, ly - b) > min_sep for a, b in taken)

    for p in sorted(peaks, key=lambda d: -d["n"]):
        ax_, ay_ = _to_axes(p["x"], p["y"], extent)
        ux, uy = ax_ - cx, ay_ - cy
        n = max(np.hypot(ux, uy), 1e-6)
        ux, uy = ux / n, uy / n
        best = None
        for rot in (0.0, 0.16, -0.16, 0.34, -0.34, 0.55, -0.55, 0.85, -0.85,
                    1.15, -1.15, 1.5, -1.5, 1.9, -1.9, 2.4, -2.4, np.pi):
            c, s = np.cos(rot), np.sin(rot)
            vx, vy = ux * c - uy * s, ux * s + uy * c
            push = 0.085
            while push < 0.58:
                lx, ly = ax_ + vx * push, ay_ + vy * push
                if free(lx, ly, lx < ax_):
                    best = (lx, ly)
                    break
                push += 0.010
            if best:
                break
        if best is None:
            best = (min(max(ax_, text_room), 1 - text_room),
                    min(max(ay_, margin), 1 - margin))
        taken.append(best)
        p["lx"], p["ly"] = best
        p["ax"], p["ay"] = ax_, ay_
    return peaks


# ================================================================== the plate =
def stage_render(cache, outdir, dpi=460):
    # Panels C and D still print the Y2/Y4 archive, while the cache A and B
    # are laid out from is now A0.  Drawing them on one plate would put A0
    # water under pre-A0 numbers, which is the kind of quiet mismatch the rest
    # of this file is written to prevent.  `lake_ab` is the figure that has
    # been moved over; this one needs its C and D repointed first.
    raise SystemExit(
        "the four-panel plate is not on A0 yet: A and B would be laid out "
        "from %s while C and D still print the Y2/Y4 archive.  Repoint them "
        "at %s first, or render lake_ab, which is on A0."
        % (os.path.basename(G.CACHE), os.path.basename(G.A0_REPORT)))

    _style()
    os.makedirs(outdir, exist_ok=True)

    xy = np.asarray(np.load(os.path.join(cache, "layout_xy.npy"), mmap_mode="r"))
    qxy = np.load(os.path.join(cache, "query_xy.npz"))["xy"]
    gf = np.load(os.path.join(cache, "graph_facts.npz"))
    ginfo = json.load(open(os.path.join(cache, "graph_facts.json"), encoding="utf-8"))
    rr = np.load(os.path.join(cache, "rerank.npz"))
    rinfo = json.load(open(os.path.join(cache, "rerank.json"), encoding="utf-8"))
    y2 = json.load(open(G.Y2_REPORT, encoding="utf-8"))
    y4 = json.load(open(G.Y4_REPORT, encoding="utf-8"))

    model_name = (pd.read_parquet(os.path.join(G.EXPORT, "model_ids.parquet"))
                  .sort_values("mappedID")["model"].to_numpy())
    ds_name = (pd.read_parquet(os.path.join(G.EXPORT, "dataset_ids.parquet"))
               .sort_values("mappedID")["dataset"].to_numpy())

    lum = gf["sup_deg"] > 0
    family_id = gf["family_id"]
    vocab = ginfo["family_vocab"]

    ext_A = _fit(_bounds(xy[:, 0], xy[:, 1]), _ax_ratio(R_A))
    case = _case_data(rr, xy)
    ext_B = _fit(_bounds(xy[case["pool"], 0], xy[case["pool"], 1],
                         lo=0.8, hi=99.2, pad=0.24), _ax_ratio(R_B))

    fig = plt.figure(figsize=FIGSIZE)

    hdr = fig.add_axes(R_HEADER)
    hdr.set_xlim(0, 1)
    hdr.set_ylim(0, 1)
    _clean(hdr)
    hdr.text(0.030, 0.72, "The Model Lake", fontsize=25.5, weight="bold",
             color=INK, ha="left", va="center")
    hdr.text(0.030, 0.40,
             "3,016,439 model embeddings, 18,729 dataset–task queries,",
             fontsize=10.0, color=SUBINK, ha="left", va="center")
    hdr.text(0.030, 0.22,
             "and the two-stage retrieval that fishes one from the other",
             fontsize=10.0, color=SUBINK, ha="left", va="center")
    _draw_cascade(hdr, y4)
    hdr.plot([0.030, 0.980], [0.020, 0.020], color=HAIR, lw=0.7, clip_on=False)

    axA = fig.add_axes(R_A)
    _panel_lake(axA, xy, qxy, lum, family_id, vocab, ext_A, ginfo, case)

    axB = fig.add_axes(R_B)
    _panel_cast(axB, xy, qxy, case, model_name, ds_name, ext_B)

    axC = fig.add_axes(R_C)
    _panel_returned(axC, case, model_name)

    axD = fig.add_axes(R_D)
    _panel_migration(axD, rr, rinfo, case)

    foot = fig.add_axes(R_FOOT)
    foot.set_xlim(0, 1)
    foot.set_ylim(0, 1)
    _clean(foot)
    foot.text(0.030, 0.45,
              "graph hgraph_rf sha256 %s…   export X4GD_full_s0_e25, "
              "z_m_eval / z_d_eval, seed 0" % G.GRAPH_DIGEST[:14],
              fontsize=6.4, color=FAINT, ha="left", va="center", family=MONO)
    foot.text(0.360, 0.45,
              "B–D read the archived exact top-1,000 pool, whose gold@10 is "
              "%.4f; the deployed HNSW path returns %.4f"
              % (y2["per_seed"]["0"]["rows"]["G_exact1000_task"]["gold@10"],
                 y4["hnsw"]["0"]["rows"]["G_hnsw1000_task"]["gold@10"]),
              fontsize=6.4, color=FAINT, ha="left", va="center", family=MONO)
    foot.text(0.980, 0.45,
              "all %s archived seed-0 top-10 lists reproduced exactly"
              % f"{rinfo['n_queries']:,}",
              fontsize=6.4, color=FAINT, ha="right", va="center", family=MONO)

    pdf = os.path.join(outdir, "model_lake_galaxy.pdf")
    png = os.path.join(outdir, "model_lake_galaxy.png")
    fig.savefig(pdf, dpi=dpi)
    fig.savefig(png, dpi=190)
    plt.close(fig)
    G._say("wrote %s" % pdf)
    G._say("wrote %s" % png)
    return pdf


def _case_data(rr, xy):
    qi = int(np.flatnonzero(rr["query"] == G.CASE_QUERY)[0])
    return {"qi": qi,
            "pool": rr["pool_ids"][qi].astype(np.int64),
            "cos": rr["pool_cos"][qi].astype(np.float32),
            "prior": rr["case_prior"].astype(np.float32),
            "top10": rr["top10"][qi].astype(np.int64),
            "gold": int(rr["gold_id"][qi]),
            "dense_rank": int(rr["dense_rank"][qi]),
            "fused_rank": int(rr["fused_rank"][qi])}


# ------------------------------------------------------------------ header --
def _draw_cascade(ax, y4):
    """3,016,439 -> 1,000 -> 10, as a chain rather than a bar chart.

    Every figure printed here is read out of the archived Y4 report for the
    same seed the rest of the plate uses, so nothing on the header is typed in
    by hand.
    """
    h0 = y4["hnsw"]["0"]
    row = h0["rows"]["G_hnsw1000_task"]
    lat = h0["latency_ms"]["1000"]
    mean10 = float(np.mean([y4["hnsw"][s]["rows"]["G_hnsw1000_task"]["gold@10"]
                            for s in ("0", "1", "2")]))
    cols = [DEEP, TEAL, AMBER]
    heads = ["3,016,439", "1,000", "10"]
    subs = ["candidates", "pooled", "returned"]
    notes = [["128-D unit sphere", "one shared retrieval space"],
             ["HNSW inner product",
              "recall@1000 %.4f · p50 %.3f ms" % (row["recall@1000"],
                                                  lat["hnsw_p50"])],
             ["task-prior rerank",
              "p50 %.3f ms · gold@10 %.4f" % (lat["rerank_p50"],
                                              row["gold@10"])]]
    ax.text(0.980, 0.075,
            "seed 0 throughout;  three-seed mean gold@10 %.4f" % mean10,
            fontsize=6.8, color=FAINT, ha="right", va="center", family=MONO)
    xs = [0.545, 0.712, 0.879]
    for k, x0 in enumerate(xs):
        ax.add_patch(plt.Rectangle((x0, 0.30), 0.003, 0.56, color=cols[k],
                                   lw=0, clip_on=False))
        ax.text(x0 + 0.010, 0.80, heads[k], fontsize=15.5, weight="bold",
                color=cols[k], ha="left", va="center")
        ax.text(x0 + 0.010, 0.585, subs[k], fontsize=8.6, color=INK,
                ha="left", va="center")
        for j, note in enumerate(notes[k]):
            ax.text(x0 + 0.010, 0.435 - j * 0.115, note, fontsize=6.5,
                    color=FAINT, ha="left", va="center", family=MONO)
        if k < 2:
            ax.annotate("", xy=(xs[k + 1] - 0.014, 0.79),
                        xytext=(x0 + 0.108, 0.79),
                        arrowprops=dict(arrowstyle="-|>", color=FAINT, lw=0.9,
                                        shrinkA=0, shrinkB=0,
                                        mutation_scale=8), clip_on=False)


# ------------------------------------------------------------------ panel A --
def _panel_lake(ax, xy, qxy, lum, family_id, vocab, ext, ginfo, case):
    x, y = xy[:, 0], xy[:, 1]
    img = density_image(x, y, ext, res=(2200, 2200), smooth=1.45, sharp=0.36,
                       clip=99.80)
    ax.imshow(img, origin="lower", extent=ext, cmap=NEBULA, aspect="auto",
              interpolation="bilinear", zorder=1, rasterized=True, vmin=0, vmax=1)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    _clean(ax)
    _frame(ax)
    outside = int((~((x >= ext[0]) & (x <= ext[1]) &
                     (y >= ext[2]) & (y <= ext[3]))).sum())
    _tag(ax, "A", "the lake, laid out once",
         "%s model rows, %d of them beyond this frame  ·  log density  ·  "
         "neighbourhoods are meaningful, distances are not"
         % (f"{len(xy):,}", outside))

    qx, qy = _to_axes(qxy[G.CASE_QUERY][0], qxy[G.CASE_QUERY][1], ext)
    inset_rect = [0.606, 0.014, 0.986, 0.394]
    # a label anchored just left of the inset still runs into it, so the region
    # labels must keep clear of is wider than the inset itself
    inset_keepout = [0.500, 0.000, 1.000, 0.410]
    caption_rect = [0.010, 0.010, 0.46, 0.090]
    blocked = [inset_keepout, caption_rect,
               [qx - 0.045, qy - 0.030, qx + 0.055, qy + 0.030]]

    peaks = family_peaks(xy, family_id, vocab, G.FAMILY_LABELS, ext)
    peaks = place_labels(peaks, ext, blocked=blocked)
    for p in peaks:
        ax.plot([p["ax"], p["lx"]], [p["ay"], p["ly"]], transform=ax.transAxes,
                color=INK, lw=0.45, alpha=0.40, zorder=8, solid_capstyle="round")
        ax.scatter([p["ax"]], [p["ay"]], transform=ax.transAxes, s=3.4,
                   color=INK, alpha=0.55, lw=0, zorder=9)
        ha = "left" if p["lx"] >= p["ax"] else "right"
        off = 0.007 if ha == "left" else -0.007
        ax.text(p["lx"] + off, p["ly"], p["label"], transform=ax.transAxes,
                fontsize=7.3, color=INK, ha=ha, va="center", zorder=10,
                bbox=dict(boxstyle="round,pad=0.15", fc="#ffffff", ec="none",
                          alpha=0.82))

    # Where panel B's cast lands.  A rectangle would be wrong here: the dense
    # top-1,000 of one query is not a small neighbourhood, it is a long stretch
    # of the lake, so the region is drawn as the contour that holds most of the
    # pool rather than as a crop box.
    pool_xy = xy[case["pool"]]
    ph, pxe, pye = np.histogram2d(pool_xy[:, 0], pool_xy[:, 1], bins=110,
                                  range=[[ext[0], ext[1]], [ext[2], ext[3]]])
    ps = gaussian_filter(ph, 3.0)
    flat = np.sort(ps.ravel())[::-1]
    lvl = flat[np.searchsorted(np.cumsum(flat), 0.86 * flat.sum())]
    ax.contour((pxe[:-1] + pxe[1:]) / 2, (pye[:-1] + pye[1:]) / 2, ps.T,
               levels=[lvl], colors=[AMBER], linewidths=1.0, zorder=6,
               alpha=0.85)
    ax.scatter([qx], [qy], transform=ax.transAxes, s=44, marker="P",
               facecolor=CRIMSON, edgecolor="#ffffff", lw=1.0, zorder=13)
    ax.text(qx + 0.014, qy - 0.004, "B", transform=ax.transAxes, fontsize=8.0,
            weight="bold", color=AMBER, ha="left", va="top", zorder=13)

    ax.text(0.014, 0.048,
            "%s model families, %d of them with 1,000 or more members"
            % (f"{ginfo['n_families']:,}", ginfo["families_ge_1000"]),
            transform=ax.transAxes, fontsize=7.1, color=SUBINK,
            ha="left", va="bottom", zorder=12)
    ax.text(0.014, 0.018,
            "layout: exact PCA-10 of z_m, UMAP over 250,000 anchors, every "
            "other row by 6-anchor inverse-distance interpolation",
            transform=ax.transAxes, fontsize=6.5, color=FAINT,
            ha="left", va="bottom", zorder=12)

    # inset: the identical frame, luminous rows only
    ins = ax.inset_axes(inset_rect[:2] + [inset_rect[2] - inset_rect[0],
                                          inset_rect[3] - inset_rect[1]])
    outline = density_image(x, y, ext, res=(820, 820), smooth=2.4, sharp=0.0)
    ins.contourf(np.linspace(ext[0], ext[1], 820),
                 np.linspace(ext[2], ext[3], 820), outline,
                 levels=[0.030, 10.0], colors=[GHOST], zorder=1)
    li = density_image(x[lum], y[lum], ext, res=(820, 820), smooth=1.15,
                       sharp=0.42, clip=99.5)
    ins.imshow(li, origin="lower", extent=ext, cmap=LUMEN, aspect="auto",
               interpolation="bilinear", zorder=2, rasterized=True,
               vmin=0, vmax=1)
    ins.set_xlim(ext[0], ext[1])
    ins.set_ylim(ext[2], ext[3])
    _clean(ins)
    _frame(ins, lw=0.6)
    ins.text(0.045, 0.955, "%s models carry evidence" % f"{ginfo['n_luminous']:,}",
             transform=ins.transAxes, fontsize=7.6, weight="bold", color=GOLD,
             ha="left", va="top")
    ins.text(0.045, 0.885,
             "%.2f%% of the lake.  The other %.2f%% is\nreachable only through "
             "the representation." % (ginfo["luminous_pct"],
                                      100 - ginfo["luminous_pct"]),
             transform=ins.transAxes, fontsize=6.5, color=SUBINK,
             ha="left", va="top", linespacing=1.40)
    ins.text(0.045, 0.048,
             "Supervision is %s model–dataset edges over\n"
             "%s dataset–task query nodes.  Panel B follows one."
             % (f"{G.N_EVID_EDGES:,}", f"{G.N_QUERIES:,}"),
             transform=ins.transAxes, fontsize=6.5, color=SUBINK,
             ha="left", va="bottom", linespacing=1.40)


# ------------------------------------------------------------------ panel B --
def _panel_cast(ax, xy, qxy, case, model_name, ds_name, ext):
    pool = case["pool"]
    pxy = xy[pool]
    inbox = ((xy[:, 0] > ext[0]) & (xy[:, 0] < ext[1]) &
             (xy[:, 1] > ext[2]) & (xy[:, 1] < ext[3]))
    sub = xy[inbox]
    img = density_image(sub[:, 0], sub[:, 1], ext, res=(1350, 1350),
                        smooth=1.55, sharp=0.28)
    ax.imshow(img * 0.55, origin="lower", extent=ext, cmap=NEBULA,
              aspect="auto", interpolation="bilinear", zorder=1,
              rasterized=True, vmin=0, vmax=1)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    _clean(ax)
    _frame(ax, color=AMBER, lw=0.7)
    shown = int(((pxy[:, 0] >= ext[0]) & (pxy[:, 0] <= ext[1]) &
                 (pxy[:, 1] >= ext[2]) & (pxy[:, 1] <= ext[3])).sum())
    _tag(ax, "B", "one cast",
         ("all 1,000 pooled candidates, enlarged from the outline in A"
          if shown == len(pxy) else
          "%d of the 1,000 pooled candidates, enlarged from the outline in A"
          % shown))

    pnorm = case["prior"] / max(case["prior"].max(), 1e-9)
    o = np.argsort(pnorm)
    ax.scatter(pxy[o, 0], pxy[o, 1], s=5.5 + 30 * pnorm[o] ** 1.4, c=pnorm[o],
               cmap=PRIOR_CMAP, vmin=0, vmax=1, lw=0, alpha=0.85, zorder=4,
               rasterized=True)

    for r, m in enumerate(case["top10"], 1):
        if int(m) == case["gold"]:
            continue                      # the star already marks that rank
        p = xy[int(m)]
        ax.scatter([p[0]], [p[1]], s=52, facecolor="#ffffff", edgecolor=INK,
                   lw=0.7, zorder=14, alpha=0.95)
        ax.text(p[0], p[1], str(r), fontsize=5.0, color=INK, ha="center",
                va="center", zorder=15, weight="bold")
    gp = xy[case["gold"]]
    qp = qxy[G.CASE_QUERY]
    ax.annotate("", xy=(gp[0], gp[1]), xytext=(qp[0], qp[1]), zorder=5,
                arrowprops=dict(arrowstyle="-", color=GOLD, lw=1.0, alpha=0.8,
                                connectionstyle="arc3,rad=0.24"))
    ax.scatter([gp[0]], [gp[1]], s=230, marker="*", facecolor=GOLD_LT,
               edgecolor=INK, lw=0.7, zorder=16)
    ax.scatter([qp[0]], [qp[1]], s=90, marker="P", facecolor=CRIMSON,
               edgecolor="#ffffff", lw=1.0, zorder=17)

    label = ds_name[G.CASE_QUERY].replace("\t", "   ·   ")
    ax.add_patch(plt.Rectangle((0.0, 0.752), 0.455, 0.248,
                               transform=ax.transAxes, fc="#ffffff", ec=HAIR,
                               lw=0.5, alpha=1.0, zorder=10))
    ax.text(0.022, 0.966, label, transform=ax.transAxes, fontsize=8.4,
            color=CRIMSON, weight="bold", ha="left", va="top", zorder=11)
    ax.text(0.022, 0.897,
            "gold  %s\ndense rank %d of 1,000   →   returned at %d"
            % (model_name[case["gold"]], case["dense_rank"], case["fused_rank"]),
            transform=ax.transAxes, fontsize=7.0, color=INK, ha="left",
            va="top", linespacing=1.5, zorder=11)

    legend = [("*", GOLD_LT, INK, 90, "held-out gold model"),
              ("P", CRIMSON, "#ffffff", 42, "query, at the centroid of its dense top-64"),
              ("o", AMBER, "none", 22, "candidate the task prior lifts"),
              ("o", "#a4abbb", "none", 16, "candidate the prior never touches")]
    ax.add_patch(plt.Rectangle((0.573, 0.0), 0.427, 0.183,
                               transform=ax.transAxes, fc="#ffffff", ec=HAIR,
                               lw=0.5, alpha=1.0, zorder=10))
    for i, (mk, fc, ec, s, txt) in enumerate(legend):
        yy = 0.150 - i * 0.037
        ax.scatter([0.601], [yy], transform=ax.transAxes, s=s, marker=mk,
                   facecolor=fc, edgecolor=ec, lw=0.6, zorder=12, clip_on=False)
        ax.text(0.629, yy, txt, transform=ax.transAxes, fontsize=6.4,
                color=SUBINK, ha="left", va="center", zorder=12)


# ------------------------------------------------------------------ panel C --
def _panel_returned(ax, case, model_name):
    dpos = {int(m): i + 1 for i, m in enumerate(case["pool"])}
    ax.set_xscale("log")
    ax.set_xlim(0.80, 3.0e4)
    ax.set_ylim(10.75, 0.25)
    _clean(ax)
    ax.axvspan(0.80, 10, color="#f4f6fb", zorder=-1)
    for gx, lab in ((1, "1"), (10, "10"), (100, "100"), (1000, "1,000")):
        ax.plot([gx, gx], [0.25, 10.75], color=HAIR, lw=0.45, zorder=0)
        ax.text(gx, 11.25, lab, fontsize=6.4, color=SUBINK, ha="center", va="top")
    ax.plot([1300, 1300], [0.25, 10.75], color=HAIR, lw=0.5, zorder=0)
    ax.text(np.sqrt(0.8 * 1000), 11.95, "rank inside the 1,000-candidate pool",
            fontsize=6.8, color=SUBINK, ha="center", va="top")

    for r, m in enumerate(case["top10"], 1):
        m = int(m)
        d = dpos[m]
        is_gold = m == case["gold"]
        c = GOLD if is_gold else INK
        ax.plot([d, r], [r, r], color=c, lw=1.6 if is_gold else 0.9,
                alpha=0.9, zorder=3, solid_capstyle="round")
        ax.scatter([d], [r], s=13, color=FAINT, lw=0, zorder=4)
        ax.scatter([r], [r], s=34 if is_gold else 15, color=c, lw=0, zorder=5,
                   marker="*" if is_gold else "o")
        nm = model_name[m]
        nm = nm if len(nm) <= 44 else nm[:42] + "…"
        ax.text(1550, r, nm, fontsize=6.4, color=c, ha="left", va="center",
                zorder=6, family=MONO, weight="bold" if is_gold else "normal")
    _tag(ax, "C", "what came back",
         "grey dot: where the dense stage had it  ·  bar: how far the prior moved it")


# ------------------------------------------------------------------ panel D --
def _panel_migration(ax, rr, rinfo, case):
    dense = rr["dense_rank"].astype(np.int64)
    fused = rr["fused_rank"].astype(np.int64)
    keep = dense > 0
    d, f = dense[keep], fused[keep]
    ax.set_xscale("log")
    ax.set_xlim(0.80, 3.0e4)
    ax.set_ylim(-0.06, 1.06)
    _clean(ax)
    ax.axvspan(0.80, 10, color="#f4f6fb", zorder=-1)
    for gx, lab in ((1, "1"), (10, "10"), (100, "100"), (1000, "1,000")):
        ax.plot([gx, gx], [0, 1], color=HAIR, lw=0.45, zorder=0)
        ax.text(gx, -0.13, lab, fontsize=6.4, color=SUBINK, ha="center", va="top")

    won = f <= 10
    for sel, col, alpha, lw, z in ((~won, "#9aa2b4", 0.26, 0.38, 2),
                                   (won, GOLD, 0.32, 0.5, 3)):
        segs = [((a, 1.0), (b, 0.0)) for a, b in zip(d[sel], f[sel])]
        ax.add_collection(LineCollection(segs, colors=col, linewidths=lw,
                                         alpha=alpha, zorder=z, capstyle="round",
                                         rasterized=True))
    ax.plot([case["dense_rank"], case["fused_rank"]], [1, 0], color=AMBER,
            lw=1.5, zorder=6)
    ax.scatter([case["dense_rank"], case["fused_rank"]], [1, 0], s=16,
               color=AMBER, lw=0, zorder=7)

    ax.text(1550, 1.0, "rank after the dense stage", fontsize=6.8, color=SUBINK,
            ha="left", va="center")
    ax.text(1550, 0.0, "rank after the task-prior rerank", fontsize=6.8,
            color=INK, ha="left", va="center", weight="bold")
    ax.text(1550, 0.50,
            "gold reaches the returned ten\nfor %s of them"
            % f"{int(won.sum()):,}",
            fontsize=6.8, color=GOLD, ha="left", va="center", linespacing=1.5)
    _tag(ax, "D", "rank migration",
         "%s of the %s eligible seed-0 queries.  For the other %s the gold\n"
         "model never reaches the pool, so the loss at this depth is pool\n"
         "truncation, not approximate search."
         % (f"{int(keep.sum()):,}", f"{rinfo['n_queries']:,}",
            f"{int((~keep).sum()):,}"))
