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

LAKE_LO, LAKE_HI, LAKE_PAD = 0.10, 99.90, 0.04
LAKE_Y_LO = 2.5

NAMED = [("llama", "llama"), ("qwen", "qwen"), ("gemma", "gemma"),
         ("mistral", "mistral"), ("bert", "bert"), ("roberta", "roberta"),
         ("xlm-roberta", "xlm-roberta"), ("t5", "t5"), ("gpt2", "gpt2"),
         ("distilbert", "distilbert"), ("whisper", "whisper"),
         ("wav2vec2", "wav2vec2"), ("vit", "vit"), ("marian", "marian"),
         ("stablediffusion", "stable-diffusion"), ("flux", "flux"),
         ("blockassist", "blockassist")]

LAKE3D = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "lake3d")
LAKE3D_PACK = r"D:/research/model_lake/data/data1m/lake3d_a0/packed_meta.json"
OTHER_COLOR = "#7b8398"
N_FAMILIES = 26


def lake3d_families():
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
DISC_PT2 = 470.0
DISC_IN = 2.0 * np.sqrt(DISC_PT2 / np.pi) / 72.0


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
    x, y = np.asarray(xy[:, 0]), np.asarray(xy[:, 1])
    return int((~((x >= ext[0]) & (x <= ext[1]) &
                  (y >= ext[2]) & (y <= ext[3]))).sum())


def _hist(x, y, ext, res):
    H, _, _ = np.histogram2d(x, y, bins=res,
                             range=[[ext[0], ext[1]], [ext[2], ext[3]]])
    return H


def family_image(D, ext, fams, res=(2000, 2000), smooth=1.35, sharp=0.40,
                 clip=99.80, blur=1.7):
    x, y = D["xy"][:, 0], D["xy"][:, 1]
    ink = density_image(x, y, ext, res=res, smooth=smooth, sharp=sharp,
                        clip=clip)
    fid = D["family_id"]
    named = np.stack([gaussian_filter(np.log1p(_hist(x[fid == f["id"]],
                                                     y[fid == f["id"]],
                                                     ext, res)), blur).T
                      for f in fams])
    rest = ~np.isin(fid, [f["id"] for f in fams])
    other = gaussian_filter(np.log1p(_hist(x[rest], y[rest], ext, res)), blur).T

    lead = named.argmax(0)
    top = named.max(0)
    cols = np.array([to_rgb(f["color"]) for f in fams])
    rgb = np.where((top > other * 0.35)[..., None], cols[lead],
                   to_rgb(OTHER_COLOR))

    a = np.clip(ink * np.where(top > other * 0.35, 1.50, 1.00), 0, 1)[..., None]
    return 1.0 - a * (1.0 - rgb)


def family_peaks(D, ext, fams, res=300):
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
    fw, fh = ax.figure.get_size_inches()
    bb = ax.get_position()
    return bb.width * fw, bb.height * fh


def _strip(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _clean(ax)
    return ax


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
    fixed = np.zeros(len(pos), bool)
    fixed[gi] = True

    w_in, h_in = _axes_size_in(ax)
    inch = np.array([w_in, h_in], float)
    SEP = DISC_IN * 1.12
    PULL = 0.30
    P, Anc = pos * inch, anchors * inch
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
        if k == gi:
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
LEGEND_CHAR_IN = FS_LEGEND / 72.0 * 0.55
LEGEND_SWATCH_IN = 0.26


def _ordinal(n):
    return {1: "first", 2: "second", 3: "third"}.get(n, "%dth" % n)


def legend_cols(width_in, fams):
    longest = max([len(f["label"]) for f in fams] + [len("no named family")])
    need = LEGEND_SWATCH_IN + longest * LEGEND_CHAR_IN
    return max(1, min(LEGEND_COLS_MAX, int(width_in / need)))


LEGEND_MARKS = 3


def legend_rows(fams, cols):
    return LEGEND_MARKS + int(np.ceil((len(fams) + 1) / float(cols)))


def legend_strip(strip, D, case, fams, gold_rank, cols):
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
    panel = [0.034, 0.130, 0.948, 0.830]
    PW = 5.90
    W = PW / panel[2]
    H = (PW / lake_aspect(D["xy"])) / panel[3]
    fig = plt.figure(figsize=(W, H))
    ext = ext_lake(D["xy"], _ratio(panel, W, H))
    panel_a(fig.add_axes(panel), D, ext)
    _save(fig, "lake_a")
    print("    frame leaves out %s rows" % f"{outside_frame(D['xy'], ext):,}")


def build_b(D):
    PW = 6.30
    ph = PW / cast_aspect(D["xy"], D["case"]["pool"])
    lpad, rpad, top, bot = 0.10, 0.10, 0.45, 0.10
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
    PW = 10.60
    ph = PW / lake_aspect(D["xy"])
    lpad, rpad, top, bot = 0.16, 0.16, 0.10, 0.16
    W = lpad + PW + rpad

    fams = frame_families(D, lake_box(D["xy"]))
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
