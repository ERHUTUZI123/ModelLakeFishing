"""logo_render.py -- the ModelLakeFishing mark, drawn from the real lake.

The nebula inside the disc is not decoration.  It is the log-density of all
3,016,439 model rows, in the same layout, palette and rendering path that
`galaxy_render.py` uses for the plate, so the mark and the figure are the same
picture at two scales.  On top of it sits the system's whole story in one
gesture: a query lands on the lake (crimson dot, amber ripples), a line goes
down, and one model comes back up (gold star, breaking the rim).

Variants written into release_assets/logo/, each as SVG, PDF and PNG:

    mark              the disc on its own, full nebula
    mark_mono         same geometry, single ink
    mark_compact      solid silhouette and heavy strokes, survives 32 px
    mark_compact_mono compact, single ink
    lockup            mark + wordmark, horizontal
    lockup_tagline    the same with the one-line caption
    lockup_mono       single ink

Only matplotlib is used, and every raster layer is confined to the nebula, so
the PDF and SVG stay vector everywhere the type and strokes are.

    python -m ModelLakeFishing.viz.logo_render
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter

from ModelLakeFishing.viz.galaxy_render import (
    PAPER, INK, SUBINK, HAIR, GOLD, GOLD_LT, CRIMSON, AMBER,
    NEBULA, SANS, MONO, density_image,
)

CACHE = r"d:/research/model_lake/data/data1m/figures/galaxy_cache"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "release_assets", "logo")

# ------------------------------------------------------------------ geometry --
# All of it in the mark's own coordinates, where the disc is the unit circle.
FOCUS = (-0.25, -0.05)                       # where the cast lands
RIPPLE = ((0.145, 1.00), (0.275, 0.78), (0.425, 0.58))   # radius, line weight
_TH = np.deg2rad(50.0)
STAR = (float(np.cos(_TH)), float(np.sin(_TH)))          # the catch, on the rim
CTRL = (0.34, 0.06)                          # bezier control for the line
DAMP = 0.92                                  # hold the nebula back under the ink
BODY = "#3f4695"                             # compact silhouette fill

_LAKE = None


def lake(res=1000):
    """Log-density of the whole lake, on a square window centred on its mass."""
    global _LAKE
    if _LAKE is not None:
        return _LAKE
    xy = np.load(os.path.join(CACHE, "layout_xy.npy"))
    x, y = xy[:, 0].astype(np.float64), xy[:, 1].astype(np.float64)
    keep = ((x > np.percentile(x, 0.5)) & (x < np.percentile(x, 99.5)) &
            (y > np.percentile(y, 0.5)) & (y < np.percentile(y, 99.5)))
    cx, cy = x[keep].mean(), y[keep].mean()
    half = 0.98 * max(np.percentile(np.abs(x[keep] - cx), 98.5),
                      np.percentile(np.abs(y[keep] - cy), 98.5))
    _LAKE = density_image(x, y, (cx - half, cx + half, cy - half, cy + half),
                          res=(res, res), smooth=1.5, sharp=0.30)
    return _LAKE


def _bezier(p0, p1, p2, n=220):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return ((1 - t) ** 2 * np.array(p0, float)
            + 2 * (1 - t) * t * np.array(p1, float)
            + t ** 2 * np.array(p2, float))


def _halo(s, on=True):
    if not on:
        return []
    return [pe.withStroke(linewidth=2.6 * s, foreground=PAPER, alpha=0.75)]


def _axes(ax, pad=1.17):
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    ax.set_aspect("equal")
    ax.axis("off")


# --------------------------------------------------------------------- marks --
def draw_mark(ax, s=1.0, mono=False):
    """Full mark: the million rows, the cast, the catch."""
    _axes(ax)
    ring = INK if mono else AMBER
    im = ax.imshow(lake() * DAMP, origin="lower", extent=(-1, 1, -1, 1),
                   cmap=("Greys" if mono else NEBULA), vmin=0.0,
                   vmax=(1.15 if mono else 1.0), interpolation="bilinear",
                   zorder=2, rasterized=True)
    im.set_clip_path(Circle((0, 0), 1.0, transform=ax.transData))

    ax.add_patch(Circle((0, 0), 1.0, fill=False, lw=1.0 * s,
                        ec=(INK if mono else HAIR), zorder=6))

    fx = _halo(s)
    for rad, lw in RIPPLE:
        c = Circle(FOCUS, rad, fill=False, lw=lw * s, ec=ring, zorder=7)
        c.set_path_effects(fx)
        ax.add_patch(c)

    arc = _bezier(FOCUS, CTRL, STAR)
    ax.plot(arc[:, 0], arc[:, 1], lw=1.25 * s, color=ring,
            solid_capstyle="round", zorder=8, path_effects=fx)

    ax.scatter([FOCUS[0]], [FOCUS[1]], s=36 * s * s, marker="o",
               facecolor=(INK if mono else CRIMSON), edgecolor=PAPER,
               lw=0.95 * s, zorder=9)
    ax.scatter([STAR[0]], [STAR[1]], s=250 * s * s, marker="*",
               facecolor=(PAPER if mono else GOLD_LT),
               edgecolor=(INK if mono else GOLD), lw=1.2 * s, zorder=10)


def draw_compact(ax, mono=False):
    """Small-size build: one solid body, two ripples, heavy strokes."""
    _axes(ax, pad=1.16)
    ring = INK if mono else AMBER
    img = gaussian_filter(lake(), 13.0)
    lvl = float(np.percentile(img[img > 0], 60))
    cs = ax.contourf(np.linspace(-1, 1, img.shape[1]),
                     np.linspace(-1, 1, img.shape[0]), img,
                     levels=[lvl, img.max() * 2], colors=[INK if mono else BODY],
                     zorder=2)
    cs.set_clip_path(Circle((0, 0), 1.0, transform=ax.transData))

    ax.add_patch(Circle((0, 0), 1.0, fill=False, lw=2.4,
                        ec=(INK if mono else "#9aa2bb"), zorder=6))

    fx = [pe.withStroke(linewidth=4.6, foreground=PAPER, alpha=0.95)]
    for rad, lw in ((0.18, 3.0), (0.36, 2.1)):
        c = Circle(FOCUS, rad, fill=False, lw=lw, ec=ring, zorder=7)
        c.set_path_effects(fx)
        ax.add_patch(c)
    arc = _bezier(FOCUS, CTRL, STAR)
    ax.plot(arc[:, 0], arc[:, 1], lw=2.8, color=ring, solid_capstyle="round",
            zorder=8, path_effects=fx)
    ax.scatter([STAR[0]], [STAR[1]], s=900, marker="*",
               facecolor=(PAPER if mono else GOLD_LT),
               edgecolor=(INK if mono else GOLD), lw=2.4, zorder=10)


# ------------------------------------------------------------------ builders --
def _style():
    plt.rcParams.update({
        "figure.facecolor": PAPER, "savefig.facecolor": PAPER,
        "axes.facecolor": PAPER, "font.family": "sans-serif",
        "font.sans-serif": SANS, "text.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",   # glyphs as outlines, so the SVG needs no font
    })


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext, kw in (("svg", {}), ("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(os.path.join(OUT, "%s.%s" % (name, ext)), facecolor=PAPER,
                    bbox_inches="tight", pad_inches=0.02, **kw)
    plt.close(fig)
    print("  wrote", name)


def build_mark(mono=False):
    fig = plt.figure(figsize=(2.2, 2.2))
    draw_mark(fig.add_axes([0, 0, 1, 1]), s=1.0, mono=mono)
    _save(fig, "mark_mono" if mono else "mark")


def build_compact(mono=False):
    fig = plt.figure(figsize=(1.5, 1.5))
    draw_compact(fig.add_axes([0, 0, 1, 1]), mono=mono)
    _save(fig, "mark_compact_mono" if mono else "mark_compact")


def build_lockup(tagline=False, mono=False):
    fig = plt.figure(figsize=(7.7 if tagline else 6.8, 1.9))
    draw_mark(fig.add_axes([0.006, 0.05, 0.245, 0.90]), s=0.92, mono=mono)

    tx = fig.add_axes([0.262, 0.0, 0.735, 1.0])
    tx.axis("off")
    tx.set_xlim(0, 1)
    tx.set_ylim(0, 1)

    y = 0.60 if tagline else 0.50
    size = 34
    a = tx.text(0.0, y, "ModelLake", fontsize=size, weight="bold", color=INK,
                ha="left", va="center")
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    w = (a.get_window_extent(r).width /
         tx.get_window_extent(r).width)
    tx.text(w + 0.002, y, "Fishing", fontsize=size, weight="bold",
            color=(INK if mono else AMBER), ha="left", va="center")

    if tagline:
        tx.text(0.005, 0.255,
                "3,016,439 models  ·  one shared space  ·  a bounded cast  "
                "·  ten returned",
                fontsize=8.0, color=(INK if mono else SUBINK), ha="left",
                va="center", family=MONO)

    _save(fig, "lockup" + ("_tagline" if tagline else "") +
          ("_mono" if mono else ""))


def build_sheet():
    """One contact sheet so the variants can be judged together."""
    fig = plt.figure(figsize=(11.5, 6.4))
    fig.text(0.035, 0.945, "ModelLakeFishing", fontsize=17, weight="bold",
             color=INK, ha="left", va="center")
    fig.text(0.035, 0.898,
             "the mark is the lake: log-density of all 3,016,439 model rows, "
             "same layout and palette as the plate",
             fontsize=8.2, color=SUBINK, ha="left", va="center")

    draw_mark(fig.add_axes([0.035, 0.44, 0.235, 0.40]))
    draw_mark(fig.add_axes([0.300, 0.44, 0.235, 0.40]), mono=True)
    draw_compact(fig.add_axes([0.565, 0.44, 0.235, 0.40]))
    draw_compact(fig.add_axes([0.760, 0.505, 0.150, 0.27]), mono=True)
    for x, lab in ((0.035, "mark"), (0.300, "mark_mono"),
                   (0.565, "mark_compact"), (0.775, "compact_mono")):
        fig.text(x + 0.004, 0.415, lab, fontsize=7.6, color=SUBINK,
                 family=MONO, ha="left", va="center")

    draw_mark(fig.add_axes([0.035, 0.075, 0.150, 0.255]), s=0.9)
    tx = fig.add_axes([0.190, 0.075, 0.560, 0.255])
    tx.axis("off"); tx.set_xlim(0, 1); tx.set_ylim(0, 1)
    a = tx.text(0.0, 0.60, "ModelLake", fontsize=30, weight="bold", color=INK,
                ha="left", va="center")
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    w = a.get_window_extent(r).width / tx.get_window_extent(r).width
    tx.text(w + 0.002, 0.60, "Fishing", fontsize=30, weight="bold", color=AMBER,
            ha="left", va="center")
    tx.text(0.005, 0.235,
            "3,016,439 models  ·  one shared space  ·  a bounded cast  ·  ten returned",
            fontsize=7.6, color=SUBINK, ha="left", va="center", family=MONO)

    sw = [("INK", INK), ("DEEP body", BODY), ("AMBER", AMBER),
          ("CRIMSON", CRIMSON), ("GOLD", GOLD), ("GOLD_LT", GOLD_LT),
          ("HAIR", HAIR)]
    for i, (lab, col) in enumerate(sw):
        ax = fig.add_axes([0.800, 0.300 - i * 0.036, 0.026, 0.026])
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, fc=col, ec=HAIR, lw=0.5))
        fig.text(0.834, 0.313 - i * 0.036, "%-10s %s" % (lab, col), fontsize=6.6,
                 color=SUBINK, family=MONO, ha="left", va="center")
    _save(fig, "sheet")


def main():
    _style()
    print("building from", CACHE)
    build_mark()
    build_mark(mono=True)
    build_compact()
    build_compact(mono=True)
    build_lockup()
    build_lockup(tagline=True)
    build_lockup(mono=True)
    build_sheet()
    print("variants in", OUT)


if __name__ == "__main__":
    main()
