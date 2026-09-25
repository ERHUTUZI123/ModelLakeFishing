"""build.py -- a 3D layout of the model lake, built from the A0 products only.

Standalone: this file imports nothing from ModelLakeFishing.  It reads the A0
graph and the seed-0 A0 export, and writes a small payload that `page.py`
turns into an interactive WebGL viewer.

    a0_20260912/graph                       family ids, supervision, lineage
    .../exports/A0GD_full_s0_e25
        z_m_eval.npy       3,016,439 x 128  model representations
        z_d_eval.npy          18,729 x 128  dataset-task query representations
        gold_cands.npz                      the held-out best model per query
    .../metrics/a0_hnsw_s0.npz              the measured seed-0 retrieval
    lake3d_a0/before                        the same encoder before training,
                                            written by init_space.py

The graph's nodes and edges are the frozen bytes; A0 changed only the dataset
feature matrix -- seven performance-derived columns zeroed, recorded with its
hashes in graph/A0_FEATURE_REPAIR.json -- so families and lineage carry over
unchanged while every representation drawn here is new.

The layout follows the same idea as the printed plate but in three dimensions
and computed here from scratch:

    1  pca      exact PCA of the unit-norm model vectors, 128 -> 12
    2  anchors  UMAP of 250,000 randomly chosen rows, 12 -> 3
    3  extend   every other displayed row placed by inverse-distance
                interpolation over its 6 nearest anchors in PCA space
    4  queries  each dataset-task node placed at the centroid of its exact
                dense top-64 models, so a query sits where its candidates are
    5  pack     subsample, quantise to int16, and write the viewer payload

The cast the viewer shows is not recomputed here.  It is read out of the
measured seed-0 evaluation: the 1,000 candidates HNSW actually returned, the
task prior read over exactly those, and the ten the system answered with.

The viewer can also show the lake before training.  `init_space.py` rebuilds
the seed-0 encoder at its initialisation and runs it through the export path;
stages 1-4 then lay that space out over the same anchors and the same drawn
rows, so every point on screen is one model in either space.  UMAP leaves each
layout in an arbitrary orientation, so the untrained layout is turned onto the
trained one by the rotation -- or reflection, if that fits better -- that moves
the drawn points least.  The map is rigid, so nothing about the shape changes;
the page then morphs between the two.

Stages checkpoint to disk and can be re-run individually.

    python build.py --stage all
    python build.py --stage pca|umap|extend|queries           the trained space
    python build.py --stage before                            the untrained one
    python build.py --stage before-pca|before-umap|before-extend|before-queries|align
    python build.py --stage pack
"""
import argparse
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, os.environ.get("LAKE3D_DATA_ROOT", "data/data1m"))
A0 = os.path.join(REPO, os.environ.get("LAKE3D_A0_ROOT",
                                     os.path.join(DATA, "a0_20260912")))
EXPORT = os.path.join(A0, "exports", "A0GD_full_s0_e25")
GRAPH = os.path.join(A0, "graph")
HNSW = os.path.join(A0, "metrics", "a0_hnsw_s0.npz")
WORK = os.path.join(REPO, os.environ.get("LAKE3D_WORK",
                                       os.path.join(HERE, "payload")))
BEFORE = os.path.join(WORK, "before")       # init_space.py writes here
OUT = os.path.dirname(os.path.abspath(__file__))

SPLIT_SEED = 0
N_MODELS = 3_016_439
N_QUERIES = 18_729
PCA_DIM = 12
# The untrained space is not nine-dimensional.  Nine components hold 99.8% of
# the trained space's variance and the twelve kept hold 99.98%; the untrained
# one needs 51 for 99% (20 for 90%), so twelve would flatten it.  Each space
# keeps at least 99% of its own variance.
PCA_DIM_BEFORE = 51
PCA_SAMPLE = 300_000
N_ANCHOR = 250_000
UMAP_NEIGHBORS = 25
UMAP_MIN_DIST = 0.0
UMAP_SEED = 7
KNN_K = 6
JITTER = 0.28              # fraction of local anchor spacing
N_DISPLAY = 620_000        # points shipped to the browser
QUERY_TOPK = 64
SEED = 20260909

# The two spaces a stage can run over.  Both read the same kind of export, so
# stages 1-4 are one code path; only the untrained space borrows its anchors
# and its drawn rows from the trained one.
SPACES = {
    "after": dict(z_m=os.path.join(EXPORT, "z_m_eval.npy"),
                  z_d=os.path.join(EXPORT, "z_d_eval.npy"),
                  work=WORK, pca_dim=PCA_DIM),
    "before": dict(z_m=os.path.join(BEFORE, "z_m_eval.npy"),
                   z_d=os.path.join(BEFORE, "z_d_eval.npy"),
                   work=BEFORE, pca_dim=PCA_DIM_BEFORE),
}

# The cast the viewer can show.  15473 is the plate's case, and it is one of
# the 1,476 queries seed 0 actually scored, so the cast is read rather than
# recomputed.  A query that was never scored has no measured answer to show.
# The plate and this viewer are meant to show the same cast.
CASTS = [15473]


def say(*a):
    print("[%7.1fs]" % (time.time() - say.t0), *a, flush=True)


say.t0 = time.time()


def _rng():
    return np.random.default_rng(SEED)


def _pca_path(sp):
    s = SPACES[sp]
    return os.path.join(s["work"], "pca%d.npy" % s["pca_dim"])


# ------------------------------------------------------------------ stage 1 --
def stage_pca(sp="after"):
    s = SPACES[sp]
    dim = s["pca_dim"]
    if not os.path.exists(s["z_m"]):
        raise SystemExit("no %s; the untrained export comes from "
                         "`python init_space.py --stage all`" % s["z_m"])
    os.makedirs(s["work"], exist_ok=True)
    zm = np.load(s["z_m"], mmap_mode="r")
    assert zm.shape[0] == N_MODELS, zm.shape
    idx = np.sort(_rng().choice(N_MODELS, PCA_SAMPLE, replace=False))
    say("pca: reading %s sample rows" % f"{PCA_SAMPLE:,}")
    S = np.array(zm[idx], dtype=np.float32, copy=True)
    S /= np.linalg.norm(S, axis=1, keepdims=True) + 1e-12
    mu = S.mean(0)
    S -= mu
    say("pca: svd")
    # 128 columns, so an exact SVD of the sample is cheap and deterministic
    _u, sv, vt = np.linalg.svd(S, full_matrices=False)
    V = np.ascontiguousarray(vt[:dim].astype(np.float32))
    spectrum = sv ** 2 / (sv ** 2).sum()
    evr = spectrum[:dim]
    say("pca: explained variance of the kept %d:" % dim, np.round(evr, 4).tolist(),
        "sum %.4f" % float(evr.sum()))
    cum = np.cumsum(spectrum)
    say("pca: components for 90/95/99%% of the variance: %d / %d / %d"
        % tuple(int(np.searchsorted(cum, f)) + 1 for f in (0.90, 0.95, 0.99)))
    np.savez(os.path.join(s["work"], "pca.npz"), mu=mu, V=V, evr=evr, sample=idx,
             spectrum=spectrum)

    P = np.lib.format.open_memmap(_pca_path(sp), mode="w+",
                                  dtype=np.float32, shape=(N_MODELS, dim))
    step = 200_000
    for i in range(0, N_MODELS, step):
        B = np.array(zm[i:i + step], dtype=np.float32, copy=True)
        B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
        B -= mu
        P[i:i + step] = B @ V.T
        if (i // step) % 5 == 0:
            say("pca: projected %s / %s" % (f"{min(i + step, N_MODELS):,}",
                                            f"{N_MODELS:,}"))
    P.flush()
    say("pca: done ->", _pca_path(sp))


# ------------------------------------------------------------------ stage 2 --
def stage_umap(sp="after"):
    import umap
    s = SPACES[sp]
    P = np.load(_pca_path(sp), mmap_mode="r")
    if sp == "after":
        anchors = np.sort(_rng().choice(N_MODELS, N_ANCHOR, replace=False))
    else:
        # the rows the trained layout was anchored on, not a fresh draw
        anchors = np.load(os.path.join(WORK, "anchors.npy"))
    A = np.array(P[anchors], dtype=np.float32, copy=True)
    say("umap: %s anchors, %d -> 3" % (f"{N_ANCHOR:,}", s["pca_dim"]))
    reducer = umap.UMAP(n_components=3, n_neighbors=UMAP_NEIGHBORS,
                        min_dist=UMAP_MIN_DIST, metric="euclidean",
                        random_state=UMAP_SEED, verbose=True,
                        low_memory=True)
    xyz = reducer.fit_transform(A).astype(np.float32)
    xyz -= xyz.mean(0)
    scale = np.percentile(np.linalg.norm(xyz, axis=1), 99.0)
    xyz /= max(scale, 1e-9)                     # unit-ish radius, for the page
    np.save(os.path.join(s["work"], "anchors.npy"), anchors)
    np.save(os.path.join(s["work"], "anchor_xyz.npy"), xyz)
    say("umap: done, radius p50 %.3f p99 %.3f"
        % (np.percentile(np.linalg.norm(xyz, axis=1), 50),
           np.percentile(np.linalg.norm(xyz, axis=1), 99)))


# ------------------------------------------------------------------ stage 3 --
def _knn_index(A):
    import faiss
    ix = faiss.IndexFlatL2(A.shape[1])
    ix.add(np.ascontiguousarray(A))
    try:
        faiss.omp_set_num_threads(os.cpu_count() or 8)
    except Exception:
        pass
    return ix


def _interpolate(ix, axyz, P, rows, rng=None, jitter=JITTER, chunk=100_000):
    """Place rows by inverse-distance weighting over their 6 nearest anchors."""
    out = np.empty((len(rows), 3), np.float32)
    spacing = np.empty(len(rows), np.float32)
    for i in range(0, len(rows), chunk):
        r = rows[i:i + chunk]
        Q = np.ascontiguousarray(np.array(P[r], dtype=np.float32, copy=True))
        d2, nn = ix.search(Q, KNN_K)
        d = np.sqrt(np.maximum(d2, 0.0))
        w = 1.0 / (d + 1e-6)
        w /= w.sum(1, keepdims=True)
        out[i:i + chunk] = np.einsum("nk,nkd->nd", w, axyz[nn])
        # local anchor spacing, used to size the jitter that keeps interpolated
        # rows from stacking exactly on top of their anchors
        spacing[i:i + chunk] = np.linalg.norm(
            axyz[nn[:, 1:]] - axyz[nn[:, :1]], axis=2).mean(1)
    if rng is not None and jitter > 0:
        out += (rng.standard_normal(out.shape).astype(np.float32)
                * (jitter * spacing)[:, None] / np.sqrt(3.0))
    return out


def _space_index(sp):
    """The anchors of one space and a kNN index over them in its PCA."""
    s = SPACES[sp]
    P = np.load(_pca_path(sp), mmap_mode="r")
    anchors = np.load(os.path.join(s["work"], "anchors.npy"))
    axyz = np.load(os.path.join(s["work"], "anchor_xyz.npy"))
    return P, anchors, axyz, _knn_index(np.array(P[anchors], dtype=np.float32, copy=True))


def stage_extend(sp="after"):
    s = SPACES[sp]
    P, anchors, axyz, ix = _space_index(sp)
    is_anchor = np.zeros(N_MODELS, bool)
    is_anchor[anchors] = True
    if sp == "after":
        rng = _rng()
        extra_n = max(N_DISPLAY - len(anchors), 0)
        pool = np.flatnonzero(~is_anchor)
        extra = np.sort(rng.choice(pool, extra_n, replace=False))
    else:
        # the rows the trained layout drew, so each point is one model in both;
        # only the jitter is drawn afresh
        shown = np.load(os.path.join(WORK, "display.npz"))["idx"]
        extra = shown[~is_anchor[shown]]
        rng = np.random.default_rng(SEED + 1)
    say("extend: interpolating %s non-anchor rows" % f"{len(extra):,}")
    exyz = _interpolate(ix, axyz, P, extra, rng=rng)

    idx = np.concatenate([anchors, extra])
    xyz = np.concatenate([axyz, exyz]).astype(np.float32)
    order = np.argsort(idx)
    np.savez(os.path.join(s["work"], "display.npz"), idx=idx[order], xyz=xyz[order])
    say("extend: display set %s rows" % f"{len(idx):,}")


# ------------------------------------------------------------------ stage 4 --
def stage_queries(sp="after"):
    import torch
    s = SPACES[sp]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    zd = np.load(s["z_d"]).astype(np.float32)
    zd /= np.linalg.norm(zd, axis=1, keepdims=True) + 1e-12
    zm = np.load(s["z_m"], mmap_mode="r")
    nq = zd.shape[0]
    assert nq == N_QUERIES, nq
    Q = torch.as_tensor(zd, device=dev)

    def scan(K, qsel, step=200_000, qchunk=512):
        """Exact top-K over the whole lake for the selected queries."""
        nq_ = len(qsel)
        Qs = Q[torch.as_tensor(qsel, device=dev)]
        bv = torch.full((nq_, K), -2.0, device=dev)
        bi = torch.zeros((nq_, K), dtype=torch.long, device=dev)
        for i in range(0, N_MODELS, step):
            B = np.array(zm[i:i + step], dtype=np.float32, copy=True)
            B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
            Bt = torch.as_tensor(B, device=dev)
            for a in range(0, nq_, qchunk):
                sc = Qs[a:a + qchunk] @ Bt.T
                k = min(K, sc.shape[1])
                v, j = torch.topk(sc, k, dim=1)
                cv = torch.cat([bv[a:a + qchunk], v], 1)
                ci = torch.cat([bi[a:a + qchunk], j + i], 1)
                v2, sel = torch.topk(cv, K, dim=1)
                bv[a:a + qchunk] = v2
                bi[a:a + qchunk] = torch.gather(ci, 1, sel)
                del sc, v, j, cv, ci, v2, sel
            del Bt
            if dev == "cuda":
                torch.cuda.empty_cache()
            if (i // step) % 3 == 0:
                say("  scanned %s / %s" % (f"{min(i + step, N_MODELS):,}",
                                           f"{N_MODELS:,}"))
        return bi.cpu().numpy().astype(np.int64), bv.cpu().numpy().astype(np.float32)

    # This scan places the queries; it is not the retrieval.  A query is drawn
    # where it fishes -- at the centroid of its dense top-64 -- because model
    # and query rows share one output head but occupy offset regions of the
    # sphere, so a query's own vector lands nowhere near its candidates.
    say("queries: exact top-%d for all %s queries on %s"
        % (QUERY_TOPK, f"{nq:,}", dev))
    top, val = scan(QUERY_TOPK, np.arange(nq))
    np.savez(os.path.join(s["work"], "query_top.npz"), top=top, val=val)

    P, _anchors, axyz, ix = _space_index(sp)
    flat = np.unique(top[:, :QUERY_TOPK].ravel())
    say("queries: placing %s distinct top-64 models" % f"{len(flat):,}")
    pos = _interpolate(ix, axyz, P, flat, rng=None, jitter=0.0)
    lut = {int(m): k for k, m in enumerate(flat)}
    qxyz = np.empty((nq, 3), np.float32)
    for q in range(nq):
        rows = np.array([lut[int(m)] for m in top[q, :QUERY_TOPK]])
        w = np.maximum(val[q, :QUERY_TOPK], 0.0) + 1e-6
        w /= w.sum()
        qxyz[q] = (pos[rows] * w[:, None]).sum(0)
    np.save(os.path.join(s["work"], "query_xyz.npy"), qxyz)
    say("queries: done")


# ------------------------------------------------------------------- align --
def stage_align():
    """Turn the untrained layout onto the trained one without changing it.

    Orthogonal Procrustes over the 620,000 drawn points: the rotation, or
    rotation with a mirror, that brings them closest to where the same models
    sit after training.  No scaling -- both layouts are already sized by the
    99th-percentile radius of their anchors -- and no shift, since both are
    centred on their anchors and the page orbits the origin.
    """
    a = np.load(os.path.join(WORK, "display.npz"))
    b = np.load(os.path.join(BEFORE, "display.npz"))
    if not np.array_equal(a["idx"], b["idx"]):
        raise SystemExit("the two layouts do not draw the same rows")
    A = a["xyz"].astype(np.float64)
    B = b["xyz"].astype(np.float64)
    U, _sv, Vt = np.linalg.svd((B - B.mean(0)).T @ (A - A.mean(0)))
    R = U @ Vt
    det = float(np.linalg.det(R))
    raw = np.linalg.norm(B - A, axis=1)
    fit = np.linalg.norm(B @ R - A, axis=1)
    np.savez(os.path.join(BEFORE, "align.npz"), R=R, det=det,
             shift_raw=raw.mean(), shift_fit=fit.mean())
    say("align: %s, mean distance between a model's two positions %.3f -> %.3f"
        % ("rotation with a mirror" if det < 0 else "rotation", raw.mean(), fit.mean()))


# ------------------------------------------------------------------ stage 5 --
def _families():
    meta = json.load(open(os.path.join(GRAPH, "meta.json"), encoding="utf-8"))
    vocab = meta["xm0_meta"]["family_vocab"]
    inv = {int(v): k for k, v in vocab.items()}
    fid = np.load(os.path.join(GRAPH, "nodes.npz"))["model.family_id"]
    return fid, inv


def _measured_cast(q, names, dsn):
    """The seed-0 answer to one query, read from the archived evaluation.

    Nothing here is re-retrieved.  `model` holds the 1,000 ids HNSW returned
    in cosine order, `prior` the task prior read over exactly those ids,
    `fused` the score the system ranked by, and `top10` the ten it answered
    with.  The gold model is the held-out label, not a retrieval output.
    """
    h = np.load(HNSW, allow_pickle=True)
    rows = np.flatnonzero(h["query"] == q)
    if not rows.size:
        raise SystemExit(
            "query %d is not one of the %d queries seed %d scored, so it has "
            "no measured answer to show" % (q, len(h["query"]), SPLIT_SEED))
    i = int(rows[0])
    pool = h["model"][i].astype(np.int64)
    prior = h["prior"][i].astype(np.float32)
    top10 = h["top10"][i].astype(np.int64)

    gold_c = np.load(os.path.join(EXPORT, "gold_cands.npz"), allow_pickle=True)
    g = gold_c[str(q)]
    gold = int(g[0][int(np.argmax(g[1]))])

    where = np.flatnonzero(pool == gold)
    dense_rank = int(where[0]) + 1 if where.size else 0     # 0 = not in the pool
    returned = np.flatnonzero(top10 == gold)
    got = int(returned[0]) + 1 if returned.size else 0
    # the evaluator's own record of where the gold came back; if the two ever
    # disagreed the page would be drawing a rank nobody scored
    if got and got != int(h["gold_position"][i]):
        raise AssertionError("query %d: gold is %d-th in the archived top10 "
                             "but the archive records position %d"
                             % (q, got, int(h["gold_position"][i])))
    return {
        "row": i, "pool": pool, "prior": prior, "top10": top10, "gold": gold,
        "dense_rank": dense_rank, "gold_position": got,
        "n_cand": int(g[0].shape[0]),
        "label": str(dsn[q]).replace("\t", "  \u00b7  "),
        "root": str(h["root"][i]), "task_id": int(h["task_id"][i]),
        "ef_search": int(h["selected_ef"]),
        "hnsw_ms": float(h["hnsw_ns"][i]) / 1e6,
        "rerank_ms": float(h["rerank_ns"][i]) / 1e6,
        "gold_name": str(names[gold]),
        "top10_names": [str(names[m]) for m in top10],
    }


def _eval_row(space):
    """The few numbers the page quotes for one space, from init_space's eval."""
    rows = space["rows"]
    return {"gold10": rows["G_exact1000_task"]["gold@10"],
            "gold1": rows["G_exact1000_task"]["gold@1"],
            "firstStage": rows["G_exact1000_task"]["gold_in_first_stage@1000"],
            "gold10Dense": rows["G_dense"]["gold@10"],
            "denseMedian": space["dense_gold_rank"]["median"],
            "within1000": space["dense_gold_rank"]["within_1000"]}


def _before_meta(lim):
    """What the page says about the untrained space, and how it was checked."""
    man = json.load(open(os.path.join(BEFORE, "INIT_MANIFEST.json"), encoding="utf-8"))
    ev = json.load(open(os.path.join(BEFORE, "EVAL.json"), encoding="utf-8"))
    pca = np.load(os.path.join(BEFORE, "pca.npz"))
    al = np.load(os.path.join(BEFORE, "align.npz"))
    ic, cc = man["init_check"], man["control_check"]
    rec = ev["after_reproduces_record"]
    return {
        "initSeed": int(man["init_seed"]), "controlSeed": int(man["control_seed"]),
        "checkpoint": man["checkpoint_sha256"], "run": man["run_id"],
        "check": {"neverUpdated": ic["never_updated"], "bitIdentical": ic["bit_identical"],
                  "tolerance": ic["tolerance"], "maxDiff": ic["max_abs_diff"],
                  "controlMaxDiff": cc["max_abs_diff"],
                  "pathMaxDiff": man["path_check"]["max_abs_diff"]},
        "pcaDim": int(PCA_DIM_BEFORE), "pcaVariance": float(pca["evr"].sum()),
        "align": {"mirror": bool(float(al["det"]) < 0),
                  "shiftRaw": float(al["shift_raw"]), "shiftFit": float(al["shift_fit"])},
        "eval": {"nQueries": ev["spaces"]["after"]["dense_gold_rank"]["n_queries"],
                 "after": _eval_row(ev["spaces"]["after"]),
                 "before": _eval_row(ev["spaces"]["before"]),
                 "reproducesRecord": all(m["equal"] for m in rec.values()),
                 "castDenseRank": {k: ev["spaces"][k]["cast"]["dense_gold_rank"]
                                   for k in ("after", "before")}},
    }


def stage_pack():
    import pandas as pd
    disp = np.load(os.path.join(WORK, "display.npz"))
    idx, xyz = disp["idx"], disp["xyz"]
    fid, finv = _families()
    edges = np.load(os.path.join(GRAPH, "edges.npz"))
    sup = np.zeros(N_MODELS, np.int32)
    np.add.at(sup, edges["model__trained_on__dataset__edge_index"][0], 1)

    # the families the viewer can colour and fly to: biggest first, skipping
    # the catch-all bucket, and only those actually present in the display set
    on_screen = np.zeros(len(finv) + 1, np.int64)
    df = fid[idx]
    u, c = np.unique(df, return_counts=True)
    on_screen[u] = c
    cand = [(int(f), int(n)) for f, n in zip(u, c)
            if int(f) != 0 and n >= 400
            and finv.get(int(f), "").lower() != "other"]
    cand.sort(key=lambda t: -t[1])
    fam_ids = [f for f, _ in cand[:26]]
    fam_names = [finv[f] for f in fam_ids]
    fam_slot = np.zeros(len(finv) + 1, np.uint8)     # 0 = other
    for s, f in enumerate(fam_ids, 1):
        fam_slot[f] = s
    slot = fam_slot[df]

    # quantise the cloud to int16 on a shared scale
    lim = float(np.percentile(np.abs(xyz), 99.9)) * 1.02
    def q16(a, scale=lim):
        return np.clip(np.rint(a / scale * 32000.0), -32767, 32767).astype(np.int16)
    qm = q16(xyz)

    qxyz = np.load(os.path.join(WORK, "query_xyz.npy"))
    qq = q16(qxyz)

    ev = (sup[idx] > 0).astype(np.uint8)
    ev_bits = np.packbits(ev)

    # lineage, restricted to pairs where both ends are on screen.  Sent as the
    # two ends' positions in the drawn set, so the page can place a link in
    # either space from the points it already has.
    pos_of = -np.ones(N_MODELS, np.int64)
    pos_of[idx] = np.arange(len(idx))
    li = edges["model__is_base_of__model__edge_index"]
    keep = (pos_of[li[0]] >= 0) & (pos_of[li[1]] >= 0)
    src, dst = pos_of[li[0][keep]], pos_of[li[1][keep]]
    rng = _rng()
    if len(src) > 60_000:
        sel = rng.choice(len(src), 60_000, replace=False)
        src, dst = src[sel], dst[sel]
    lineage = np.empty(len(src) * 2, np.uint32)
    lineage[0::2] = src
    lineage[1::2] = dst

    # the untrained layout of the same rows, turned onto this one by `align`
    # and quantised on its own scale; the page scales it into this one's units
    d0 = np.load(os.path.join(BEFORE, "display.npz"))
    if not np.array_equal(d0["idx"], idx):
        raise SystemExit("the untrained layout draws different rows; rerun --stage before")
    R = np.load(os.path.join(BEFORE, "align.npz"))["R"]
    xyz0 = (d0["xyz"].astype(np.float64) @ R).astype(np.float32)
    lim0 = float(np.percentile(np.abs(xyz0), 99.9)) * 1.02
    qm0 = q16(xyz0, lim0)
    qq0 = q16((np.load(os.path.join(BEFORE, "query_xyz.npy")).astype(np.float64) @ R)
              .astype(np.float32), lim0)

    # one cast, read from the measured seed-0 evaluation: the 1,000 candidates
    # HNSW returned, the ten the system answered with, and the held-out gold
    names = (pd.read_parquet(os.path.join(EXPORT, "model_ids.parquet"))
             .sort_values("mappedID")["model"].to_numpy())
    dsn = (pd.read_parquet(os.path.join(EXPORT, "dataset_ids.parquet"))
           .sort_values("mappedID")["dataset"].to_numpy())
    P, _anchors, axyz, ix = _space_index("after")
    P0, _anchors0, axyz0, ix0 = _space_index("before")

    def place0(rows):
        """Where rows sit in the untrained layout, in the page's frame."""
        p = _interpolate(ix0, axyz0, P0, rows, rng=None, jitter=0.0)
        return q16((p.astype(np.float64) @ R).astype(np.float32), lim0)

    casts = []
    for q in CASTS:
        c = _measured_cast(q, names, dsn)
        ppos = _interpolate(ix, axyz, P, c["pool"], rng=None, jitter=0.0)
        tpos = _interpolate(ix, axyz, P, c["top10"], rng=None, jitter=0.0)
        gpos = _interpolate(ix, axyz, P, np.array([c["gold"]]), rng=None,
                            jitter=0.0)
        pri = c["prior"]
        casts.append({
            "query": int(q),
            "label": c["label"],
            "root": c["root"],
            "gold": c["gold_name"],
            "goldXYZ": q16(gpos)[0].tolist(),
            "queryXYZ": qq[q].tolist(),
            "poolXYZ": q16(ppos).ravel().tolist(),
            # the prior over exactly those candidates, as a byte of its own
            # maximum, so the page can show which water the prior lit up
            "poolPrior": np.rint(np.clip(pri / max(float(pri.max()), 1e-9),
                                         0, 1) * 255).astype(np.uint8).tolist(),
            "topXYZ": q16(tpos).ravel().tolist(),
            "topNames": c["top10_names"],
            "poolN": int(len(c["pool"])),
            "denseRank": c["dense_rank"],
            "goldPosition": c["gold_position"],
            "nCand": c["n_cand"],
            "priorNonzero": int((pri > 0).sum()),
            "efSearch": c["ef_search"],
            "hnswMs": round(c["hnsw_ms"], 3),
            "rerankMs": round(c["rerank_ms"], 3),
            # the same models where the untrained encoder put them.  Nothing
            # was retrieved in that space; these are positions, not an answer
            "goldXYZ0": place0(np.array([c["gold"]]))[0].tolist(),
            "queryXYZ0": qq0[q].tolist(),
            "poolXYZ0": place0(c["pool"]).ravel().tolist(),
            "topXYZ0": place0(c["top10"]).ravel().tolist(),
        })

    repair = json.load(open(os.path.join(GRAPH, "A0_FEATURE_REPAIR.json"),
                            encoding="utf-8"))
    emani = json.load(open(os.path.join(EXPORT, "EXPORT_MANIFEST.json"),
                           encoding="utf-8"))["stages"]["embed"]
    hz = np.load(HNSW, allow_pickle=True)
    before = _before_meta(lim)
    before["scale"] = lim0 / lim

    meta = {
        "nModels": int(N_MODELS), "nShown": int(len(idx)),
        "nQueries": int(len(qxyz)), "nFamilies": int(len(finv)),
        "nEvidence": int((sup > 0).sum()), "nEvidenceShown": int(ev.sum()),
        "nEvidenceEdges": int(edges["model__trained_on__dataset__edge_index"].shape[1]),
        "nLineageEdges": int(len(src)), "nLineageTotal": int(li.shape[1]),
        "families": [{"name": n, "count": int(on_screen[f]), "id": int(f)}
                     for f, n in zip(fam_ids, fam_names)],
        "scale": lim, "anchors": int(N_ANCHOR),
        "umap": {"neighbors": UMAP_NEIGHBORS, "min_dist": UMAP_MIN_DIST,
                 "seed": UMAP_SEED, "pca_dim": PCA_DIM, "knn": KNN_K},
        "source": {
            "run": str(emani["run_id"]),
            "splitSeed": int(SPLIT_SEED),
            "graphDigest": str(emani["graph_digest"]),
            "sourceGraphDigest": str(repair["source_graph_digest"]),
            "zeroedColumns": [int(z) for z in repair["zero_columns"]],
            "nScored": int(len(hz["query"])),
            "efSearch": int(hz["selected_ef"]),
        },
        "casts": casts,
        "before": before,
    }
    np.savez(os.path.join(WORK, "packed.npz"), xyz=qm, slot=slot,
             ev=ev_bits, qxyz=qq, lineage=lineage, xyz0=qm0, qxyz0=qq0)
    json.dump(meta, open(os.path.join(WORK, "packed_meta.json"), "w",
                         encoding="utf-8"))
    say("pack: %s points, %s queries, %s lineage segments"
        % (f"{len(idx):,}", f"{len(qxyz):,}", f"{len(src):,}"))
    for c in casts:
        say("pack: cast %d  %s  gold %s  rank %d by cosine alone -> returned "
            "at %d" % (c["query"], c["label"], c["gold"], c["denseRank"],
                       c["goldPosition"]))
    b = before["eval"]
    say("pack: before training  gold@10 %.4f  gold in the first 1,000 %.4f  "
        "median gold rank %s;  after  %.4f  %.4f  %s"
        % (b["before"]["gold10"], b["before"]["firstStage"],
           f"{b['before']['denseMedian']:,.1f}", b["after"]["gold10"],
           b["after"]["firstStage"], f"{b['after']['denseMedian']:,.1f}"))
    say("pack: payload bytes  xyz %s  xyz0 %s  slot %s  ev %s  q %s  q0 %s  lineage %s"
        % (f"{qm.nbytes:,}", f"{qm0.nbytes:,}", f"{slot.nbytes:,}",
           f"{ev_bits.nbytes:,}", f"{qq.nbytes:,}", f"{qq0.nbytes:,}",
           f"{lineage.nbytes:,}"))


def _before(stage):
    return lambda: stage("before")


STAGES = {"pca": stage_pca, "umap": stage_umap, "extend": stage_extend,
          "queries": stage_queries,
          "before-pca": _before(stage_pca), "before-umap": _before(stage_umap),
          "before-extend": _before(stage_extend),
          "before-queries": _before(stage_queries), "align": stage_align,
          "pack": stage_pack}
GROUPS = {"all": list(STAGES),
          "before": ["before-pca", "before-umap", "before-extend",
                     "before-queries", "align"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=list(GROUPS) + list(STAGES))
    a = ap.parse_args()
    todo = GROUPS.get(a.stage, [a.stage])
    for s in todo:
        say("=== stage", s, "===")
        STAGES[s]()
    say("done")


if __name__ == "__main__":
    main()
