"""model_lake_galaxy.py -- a paper figure built from the A0 3M-scale artifacts.

WHAT THIS DRAWS
    One plate, "The Model Lake", in four registers:

      A  the lake itself: all 3,016,439 trained model embeddings of the A0
         seed-0 held-out export, laid out once and rendered as a log-density
         nebula, with the model families that own each region named in place.
         An inset repeats the identical frame showing only the models that
         carry any supervision edge -- a small fraction of the lake is
         luminous, the rest is reachable through representation alone.
      B  one cast: the 1,000 candidates HNSW returned for a single held-out
         dataset--task query, drawn in the same coordinates, with the ten
         models the system actually returned and the held-out gold marked.
      C  rank migration: for every seed-0 query whose gold model reaches the
         pool, its dense rank on the left and its rank after the deterministic
         task-prior rerank on the right.
      D  the two-stage cascade: 3,016,439 -> 1,000 -> 10, with the recall,
         latency, and gold@10 of each step.

    Nothing here trains, evaluates, or edits anything.  The cast and the ranks
    are read out of the archived A0 evaluation rather than re-retrieved, and
    the gold@10 they imply is checked against the archived A0.7 value before
    anything is drawn.  Only the layout is computed here.

THE LAYOUT, AND WHAT IT IS AND IS NOT
    Model rows are L2-normalised 128-vectors whose covariance is effectively
    rank 10 (the first ten principal directions carry 99.96% of the variance
    and are near-balanced).  The plate therefore projects to that 10-D
    principal subspace exactly, runs UMAP on a 250,000-row anchor sample, and
    places the remaining rows by inverse-distance interpolation over their six
    nearest anchors in the same 10-D space, plus a jitter proportional to the
    local anchor spacing so that 3M rows do not stack onto 250K positions.
    The map is a projection of the retrieval space, not the retrieval space:
    neighbourhoods are meaningful, absolute distances are not.

    Dataset--task nodes are NOT placed by their own embedding.  Model and query
    rows share one output head but occupy offset regions of the sphere, so a
    direct joint layout puts every query in one corner and says nothing.  A
    query is instead drawn where it fishes: at the similarity-weighted centroid
    of its exact dense top-64 models over the full lake.  That is the quantity
    retrieval actually uses.

Run (from the repository parent, with the ModelLakeFishing venv):
    python -m ModelLakeFishing.viz.model_lake_galaxy --stage project
    python -m ModelLakeFishing.viz.model_lake_galaxy --stage render
    python -m ModelLakeFishing.viz.model_lake_galaxy --stage all
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
_REPO_PARENT = os.path.abspath(os.path.join(_HERE, "..", ".."))
# scale1m modules import each other both as `ModelLakeFishing.scale1m.x` and as
# bare `scale1m.x`, so both roots have to be importable.
for _p in (_REPO_PARENT, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------- artifacts --
# A0: the graph's nodes and edges are the frozen bytes; only the dataset
# feature matrix changed -- seven performance-derived columns zeroed, recorded
# with its hashes in graph/A0_FEATURE_REPAIR.json -- so families and lineage
# carry over unchanged while every representation drawn here is new.
DATA = r"D:\research\model_lake\data\data1m"
A0 = os.path.join(DATA, "a0_20260912")
EXPORT = os.path.join(A0, "exports", "A0GD_full_s0_e25")
GRAPH = os.path.join(A0, "graph")
SIDECAR = os.path.join(EXPORT, "prior_sidecar_s0.npz")
# the measured seed-0 evaluation: HNSW top-1,000, the task prior read over
# exactly those, the fused score, and the ten the system answered with
HNSW_POOL = os.path.join(A0, "metrics", "a0_hnsw_s0.npz")
EXACT_POOL = os.path.join(A0, "metrics", "a0_exact_s0.npz")
A0_REPORT = os.path.abspath(os.path.join(_HERE, "..", "docs", "1M", "A0_runs",
                                         "A0_7", "results", "A0_REPORT.json"))
A0_EVAL_REPORT = os.path.join(A0, "metrics", "A0_EVALUATION_REPORT.json")
# The four-panel plate's C and D panels still print these pre-A0 archives, so
# they are kept resolvable; `galaxy_render.stage_render` refuses to draw until
# they are repointed rather than setting A0 water under pre-A0 numbers.
Y2_REPORT = os.path.abspath(os.path.join(_HERE, "..", "docs", "1M", "Y2_runs", "Y2_REPORT.json"))
Y4_REPORT = os.path.abspath(os.path.join(_HERE, "..", "docs", "1M", "Y4_runs", "Y4_REPORT.json"))
CACHE = os.path.join(DATA, "figures", "galaxy_cache_a0")
OUTDIR = os.path.abspath(os.path.join(_HERE, "..", "docs", "1M", "figures"))

# ------------------------------------------------------------- frozen facts --
N_MODELS = 3_016_439
N_QUERIES = 18_729
N_EVID_EDGES = 247_803
SPLIT_SEED = 0
POOL_K = 1_000
RETURN_K = 10
# the A0 graph, and the frozen graph it was repaired from
GRAPH_DIGEST = "acddeb93d926efcb636c83d452fe360736a0e6e4ac68cab3f5fef210e7cef5db"
SOURCE_GRAPH_DIGEST = "0e80b8393846dcc4b9e354218fd5139a906d569125d2e8545e17ccd01612b76c"

# ------------------------------------------------------- projection settings --
PCA_SAMPLE = 200_000        # rows used to fit the principal subspace
PCA_DIM = 12                # kept; the layout uses the leading 10
LAYOUT_DIM = 10
N_ANCHOR = 250_000          # rows UMAP actually embeds
UMAP_NEIGHBORS = 25
UMAP_MIN_DIST = 0.0
UMAP_EPOCHS = 200
UMAP_SEED = 7
KNN_K = 6                   # anchors averaged per interpolated row
JITTER = 0.30               # fraction of local anchor spacing
QUERY_TOPK = 64             # models whose centroid places a query
RNG_SEED = 20260908

# The query whose cast panel B draws: squad / question-answering, one of the
# 1,476 queries seed 0 scored.  Chosen because it is what the second stage is
# for -- its gold sits 445th in the pool on cosine alone, and the task prior
# brings it back first -- and because squad is the benchmark a reader knows.
#
# Under A0 six of its ten (olmo-2, layerskip-llama2-70b, gemma-3-4b x4) are
# general LLMs that the map places in the LLM region, far from the query and
# the gold in the extractive-QA cluster.  That spread is kept on purpose: it
# is the measured answer.  The map is laid out by representation, so a disc's
# distance from the query is not its retrieval distance -- the second of the
# ten is in fact the closest of them by cosine.
CASE_QUERY = 15473

# Families named on the map: frozen-vocabulary key -> label drawn on the plate.
FAMILY_LABELS = [
    ("llama", "llama"), ("qwen", "qwen"), ("qwen3", "qwen3"),
    ("gemma", "gemma"), ("mistral", "mistral"), ("bert", "bert"),
    ("distilbert", "distilbert"), ("roberta", "roberta"), ("t5", "t5"),
    ("gpt2", "gpt2"), ("whisper", "whisper"), ("wav2vec2", "wav2vec2"),
    ("vit", "vit"), ("flux", "flux"), ("stablediffusion", "stable-diffusion"),
    ("marian", "marian"), ("blockassist", "blockassist"),
    ("lunarlander", "lunarlander"), ("xlm-roberta", "xlm-roberta"),
]


def _say(msg):
    print("[galaxy] " + msg, flush=True)


def _load_zm():
    return np.load(os.path.join(EXPORT, "z_m_eval.npy"), mmap_mode="r")


def _unit(a):
    a = np.asarray(a, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)


# ============================================================ stage: project ==
def principal_subspace(cache):
    """Leading PCA_DIM directions of the model rows, fitted on a fixed sample."""
    path = os.path.join(cache, "pca.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["V"], z["mu"], z["evr"]
    zm = _load_zm()
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(zm.shape[0], PCA_SAMPLE, replace=False))
    S = _unit(np.array(zm[idx], copy=True))
    mu = S.mean(0)
    _u, s, Vt = np.linalg.svd((S - mu).astype(np.float64), full_matrices=False)
    evr = (s ** 2 / (s ** 2).sum()).astype(np.float32)
    V = Vt[:PCA_DIM].astype(np.float32)
    np.savez(path, V=V, mu=mu, evr=evr)
    _say("principal subspace: rank-10 holds %.4f of the variance" % evr[:10].sum())
    return V, mu, evr


def project_all(cache, V, mu):
    path = os.path.join(cache, "pca_coords.npy")
    if os.path.exists(path):
        return np.load(path, mmap_mode="r")
    zm = _load_zm()
    P = np.empty((zm.shape[0], PCA_DIM), dtype=np.float32)
    t = time.time()
    for i in range(0, zm.shape[0], 500_000):
        b = _unit(np.array(zm[i:i + 500_000], copy=True))
        P[i:i + 500_000] = (b - mu) @ V.T
    np.save(path, P)
    _say("projected %d rows to %d-D in %.1fs" % (P.shape[0], PCA_DIM, time.time() - t))
    return P


def umap_anchors(cache, P):
    ip = os.path.join(cache, "anchor_idx.npy")
    yp = os.path.join(cache, "anchor_xy.npy")
    if os.path.exists(ip) and os.path.exists(yp):
        return np.load(ip), np.load(yp)
    import umap
    rng = np.random.default_rng(11)
    idx = np.sort(rng.choice(P.shape[0], N_ANCHOR, replace=False))
    A = np.ascontiguousarray(P[idx][:, :LAYOUT_DIM])
    t = time.time()
    Y = umap.UMAP(n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST,
                  metric="euclidean", random_state=UMAP_SEED,
                  n_epochs=UMAP_EPOCHS).fit_transform(A).astype(np.float32)
    _say("UMAP embedded %d anchors in %.0fs" % (N_ANCHOR, time.time() - t))
    np.save(ip, idx)
    np.save(yp, Y)
    return idx, Y


def extend_layout(cache, P, anchor_idx, anchor_xy):
    """Place every model row from its KNN_K nearest anchors in the 10-D space."""
    path = os.path.join(cache, "layout_xy.npy")
    if os.path.exists(path):
        return np.load(path, mmap_mode="r")
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.as_tensor(np.ascontiguousarray(P[anchor_idx][:, :LAYOUT_DIM]), device=dev)
    YA = torch.as_tensor(anchor_xy, device=dev)
    An = (A * A).sum(1)
    N = P.shape[0]
    xy = np.empty((N, 2), dtype=np.float32)
    spread = np.empty(N, dtype=np.float32)
    t = time.time()
    chunk = 2048 if dev == "cuda" else 512
    for i in range(0, N, chunk):
        b = torch.as_tensor(np.ascontiguousarray(P[i:i + chunk, :LAYOUT_DIM]), device=dev)
        d = An[None, :] - 2.0 * (b @ A.T)
        v, j = torch.topk(d, KNN_K, dim=1, largest=False)
        v = (v + (b * b).sum(1, keepdim=True)).clamp_min(0).sqrt()
        w = 1.0 / (v + 1e-3)
        w = w / w.sum(1, keepdim=True)
        y = (YA[j] * w[..., None]).sum(1)
        spread[i:i + chunk] = (YA[j] - y[:, None, :]).pow(2).sum(-1).sqrt().mean(1).cpu().numpy()
        xy[i:i + chunk] = y.cpu().numpy()
    rng = np.random.default_rng(RNG_SEED)
    xy += rng.normal(size=(N, 2)).astype(np.float32) * (JITTER * spread)[:, None]
    xy[anchor_idx] = anchor_xy
    np.save(path, xy)
    _say("placed %d rows in %.0fs (median local anchor spacing %.3f)"
         % (N, time.time() - t, float(np.median(spread))))
    return xy


def query_positions(cache, xy):
    """A query sits at the similarity-weighted centroid of its dense top-64."""
    path = os.path.join(cache, "query_xy.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["xy"], z["top_idx"], z["top_val"]
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    zm = _load_zm()
    zd = _unit(np.load(os.path.join(EXPORT, "z_d_eval.npy")))
    Q = zd.shape[0]
    zdT = torch.as_tensor(zd, device=dev)
    best_v = torch.full((Q, QUERY_TOPK), -2.0, device=dev)
    best_i = torch.zeros((Q, QUERY_TOPK), dtype=torch.long, device=dev)
    t = time.time()
    mch, qch = (400_000, 1500) if dev == "cuda" else (200_000, 400)
    for i in range(0, zm.shape[0], mch):
        bt = torch.as_tensor(_unit(np.array(zm[i:i + mch], copy=True)), device=dev)
        for qs in range(0, Q, qch):
            qe = min(qs + qch, Q)
            v, j = torch.topk(zdT[qs:qe] @ bt.T, QUERY_TOPK, dim=1)
            cv = torch.cat([best_v[qs:qe], v], 1)
            ci = torch.cat([best_i[qs:qe], j + i], 1)
            nv, pos = torch.topk(cv, QUERY_TOPK, dim=1)
            best_v[qs:qe] = nv
            best_i[qs:qe] = torch.gather(ci, 1, pos)
        del bt
        if dev == "cuda":
            torch.cuda.empty_cache()
    top_idx = best_i.cpu().numpy().astype(np.int64)
    top_val = best_v.cpu().numpy().astype(np.float32)
    w = np.exp((top_val - top_val[:, :1]) / 0.02)
    w /= w.sum(1, keepdims=True)
    qxy = (np.asarray(xy)[top_idx] * w[..., None]).sum(1).astype(np.float32)
    np.savez(path, xy=qxy, top_idx=top_idx, top_val=top_val)
    _say("placed %d dataset-task queries by their dense top-%d in %.0fs"
         % (Q, QUERY_TOPK, time.time() - t))
    return qxy, top_idx, top_val


def graph_facts(cache):
    """Family ids, supervision degree, lineage -- read from the frozen graph."""
    path = os.path.join(cache, "graph_facts.npz")
    meta_path = os.path.join(cache, "graph_facts.json")
    if os.path.exists(path) and os.path.exists(meta_path):
        return np.load(path), json.load(open(meta_path, encoding="utf-8"))
    meta = json.load(open(os.path.join(GRAPH, "meta.json"), encoding="utf-8"))
    fam_vocab = meta["xm0_meta"]["family_vocab"]
    with np.load(os.path.join(GRAPH, "nodes.npz")) as z:
        family_id = z["model.family_id"].astype(np.int32)
        size_bucket = z["model.size_bucket_id"].astype(np.int8)
    with np.load(os.path.join(GRAPH, "edges.npz")) as z:
        sup = z["model__trained_on__dataset__edge_index"]
        lin = z["model__is_base_of__model__edge_index"]
    if sup.shape[1] != N_EVID_EDGES or len(family_id) != N_MODELS:
        raise AssertionError("frozen graph does not match the expected shape: "
                             "%d supervision edges over %d models, expected "
                             "%d over %d" % (sup.shape[1], len(family_id),
                                             N_EVID_EDGES, N_MODELS))
    sup_deg = np.bincount(sup[0], minlength=N_MODELS).astype(np.int32)
    child_count = np.bincount(lin[0], minlength=N_MODELS).astype(np.int32)
    has_parent = np.zeros(N_MODELS, dtype=bool)
    has_parent[np.unique(lin[1])] = True
    info = {
        "family_vocab": fam_vocab,
        "n_luminous": int((sup_deg > 0).sum()),
        "luminous_pct": float(100.0 * (sup_deg > 0).mean()),
        "n_with_parent": int(has_parent.sum()),
        "n_parents": int((child_count > 0).sum()),
        "unknown_size_pct": float(100.0 * (size_bucket == 0).mean()),
        "n_families": len(fam_vocab),
        "families_ge_1000": int((np.bincount(family_id,
                                             minlength=len(fam_vocab)) >= 1000).sum()),
    }
    np.savez(path, family_id=family_id, sup_deg=sup_deg,
             child_count=child_count, has_parent=has_parent)
    json.dump(info, open(meta_path, "w", encoding="utf-8"))
    _say("graph facts: %d luminous models (%.3f%%), %d lineage parents"
         % (info["n_luminous"], info["luminous_pct"], info["n_parents"]))
    return np.load(path), info


def rerank_facts(cache):
    """The measured seed-0 answer, read out of the archived A0 evaluation.

    Nothing here is re-retrieved.  `a0_hnsw_s0.npz` holds, for each of the
    1,476 queries seed 0 scored, the 1,000 ids HNSW returned in cosine order,
    the task prior read over exactly those ids, the fused score the system
    ranked by, and the ten ids it answered with.  The only quantity derived
    here is the held-out gold label, and the gold@10 it implies is checked
    against the archived A0.7 value before any of it reaches a figure.

    `dense_rank` is where the gold sat in the pool by cosine alone and
    `fused_rank` where it came back after the task prior; both are 0 when the
    gold never reached the pool at all.
    """
    path = os.path.join(cache, "rerank.npz")
    meta_path = os.path.join(cache, "rerank.json")
    if os.path.exists(path) and os.path.exists(meta_path):
        info = json.load(open(meta_path, encoding="utf-8"))
        # `case_prior` is the prior for one query, so a cache built under a
        # different CASE_QUERY holds the wrong one -- and nothing downstream
        # would notice, it would just draw the last case's prior under this
        # case's name.  Rebuild instead of trusting it.
        if info.get("case_query") == CASE_QUERY:
            return np.load(path), info
        _say("cache holds the rerank for query %s, not %d -- rebuilding"
             % (info.get("case_query"), CASE_QUERY))
    pool = np.load(HNSW_POOL, allow_pickle=True)
    queries = pool["query"].astype(np.int64)
    pool_ids = pool["model"].astype(np.int64)
    pool_cos = pool["score"].astype(np.float32)
    pool_prior = pool["prior"].astype(np.float32)
    top10 = pool["top10"].astype(np.int64)
    fused_rank = pool["gold_position"].astype(np.int32)
    if pool_ids.shape[1] != POOL_K or top10.shape[1] != RETURN_K:
        raise AssertionError("archived pool is %s, expected %d candidates and "
                             "%d returned" % (pool_ids.shape, POOL_K, RETURN_K))

    gold_cands = np.load(os.path.join(EXPORT, "gold_cands.npz"))
    nq = len(queries)
    gold_id = np.zeros(nq, np.int64)
    dense_rank = np.zeros(nq, np.int32)
    prior_case = None
    for i, q in enumerate(queries):
        q = int(q)
        cand, acc = gold_cands[str(q)]
        gold = int(cand[int(np.argmax(acc))])
        gold_id[i] = gold
        where = np.flatnonzero(pool_ids[i] == gold)
        dense_rank[i] = (where[0] + 1) if where.size else 0
        if q == CASE_QUERY:
            prior_case = pool_prior[i].copy()
    if prior_case is None:
        raise AssertionError("query %d is not one of the %d queries seed %d "
                             "scored, so it has no measured answer to draw"
                             % (CASE_QUERY, nq, SPLIT_SEED))

    # the two ways the archive can be read have to agree: membership of the
    # returned ten, and the recorded position of the gold in the fused order
    by_member = float(np.mean([gold_id[i] in set(top10[i].tolist())
                               for i in range(nq)]))
    by_rank = float(((fused_rank > 0) & (fused_rank <= RETURN_K)).mean())
    want = json.load(open(A0_REPORT, encoding="utf-8"))[
        "native_recomputed"][str(SPLIT_SEED)]["hnsw1000_task_prior"]["gold@10"]
    if abs(by_member - want) > 1e-12 or abs(by_rank - want) > 1e-12:
        raise AssertionError("the archived seed-%d answer does not carry the "
                             "reported gold@10: %.12f by membership, %.12f by "
                             "recorded position, %.12f reported"
                             % (SPLIT_SEED, by_member, by_rank, want))
    info = {"n_queries": nq, "gold@10": want, "case_query": CASE_QUERY,
            "gold_in_pool": float((dense_rank > 0).mean()),
            "dense_gold_at10": float(((dense_rank > 0)
                                      & (dense_rank <= RETURN_K)).mean()),
            "median_fused_rank_if_retrieved":
                float(np.median(fused_rank[fused_rank > 0])),
            "source": os.path.basename(HNSW_POOL)}
    np.savez(path, query=queries, pool_ids=pool_ids, pool_cos=pool_cos,
             dense_rank=dense_rank, fused_rank=fused_rank, gold_id=gold_id,
             top10=top10, case_prior=prior_case)
    json.dump(info, open(meta_path, "w", encoding="utf-8"))
    _say("archived rerank read: %d queries, gold@10=%.10f, gold reaches the "
         "pool for %.4f of them" % (nq, want, info["gold_in_pool"]))
    return np.load(path), info


def stage_project(cache):
    os.makedirs(cache, exist_ok=True)
    V, mu, _evr = principal_subspace(cache)
    P = project_all(cache, V, mu)
    ai, ay = umap_anchors(cache, P)
    xy = extend_layout(cache, P, ai, ay)
    query_positions(cache, xy)
    graph_facts(cache)
    rerank_facts(cache)
    _say("projection cache complete: %s" % cache)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", choices=("project", "render", "all"), default="all")
    p.add_argument("--cache", default=CACHE)
    p.add_argument("--out", default=OUTDIR)
    p.add_argument("--dpi", type=int, default=460)
    args = p.parse_args(argv)
    if args.stage in ("project", "all"):
        stage_project(args.cache)
    if args.stage in ("render", "all"):
        from ModelLakeFishing.viz.galaxy_render import stage_render
        stage_render(args.cache, args.out, dpi=args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
