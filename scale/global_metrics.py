"""
global_metrics.py -- P2: the SINGLE global-metric harness for the head-to-head.

Both systems are scored on OUR global metrics only (user ruling 2026-07-22):
    gold@1, gold@10, top3@10        (+ root-macro aggregation, gold-gap@10)
ModelLens's local metrics (tau_w / NDCG / Hit / Rec) are NOT used here.

Definitions (r_d(m) = m's rank in the full-lake ranking for query node d;
g_d = labeled best model; T_d^(3) = labeled top-3; near_delta = models within
delta of the best labeled accuracy):

    gold@K     = 1[ r_d(g_d) <= K ]
    top3@10    = 1[ min_{m in T_d^(3)} r_d(m) <= 10 ]        (week8 definition)
    gold-gap@K = 1[ min_{m in near_delta} r_d(m) <= K ]       (delta = 0.01)

gold@K is a GOLD-SURVIVAL metric identical to top1_eval.five_metric_eval's
`full2k_gold@K` (cross-checked in --selftest); unlabeled models are unknown,
not automatically wrong.

Two entry points, one core, so both systems funnel through identical math:
    from_embeddings(z_m, z_d, cands, roots)  -- MIPS/dot-product (our system)
    from_scores(scores_by_query, cands, roots) -- arbitrary scorer (ModelLens)

Run self-test (from ModelLakeFishing/):
    .\\.venv\\Scripts\\python.exe -m scale.global_metrics --selftest
"""

import argparse
import sys

import numpy as np

GAP_DELTA = 0.01
KS = (1, 10)


def query_ranks(s_all: np.ndarray, cand: np.ndarray, acc: np.ndarray,
                gap_delta: float = GAP_DELTA) -> dict:
    """Full-lake ranks for one query. s_all: [M] scores over the candidate
    universe; cand: [n] indices into that universe; acc: [n] labeled accuracy.
    Ranks are 1-indexed and tie-safe (rank = #strictly-better + 1)."""
    cand = np.asarray(cand)
    acc = np.asarray(acc, dtype=float)
    gold = int(cand[int(np.argmax(acc))])
    gold_rank = int((s_all > s_all[gold]).sum()) + 1

    order = np.argsort(-acc)[: min(3, len(acc))]
    top3 = cand[order]
    top3_rank = int(min((s_all > s_all[int(m)]).sum() + 1 for m in top3))

    near = cand[acc >= acc.max() - gap_delta]
    gap_rank = int(min((s_all > s_all[int(m)]).sum() + 1 for m in near))
    return {"gold_rank": gold_rank, "top3_rank": top3_rank,
            "gap_rank": gap_rank, "n_candidates": int(cand.size)}


def _flag(rank, k):
    return float(rank <= k)


def aggregate(per: dict, roots: dict | None = None, ks=KS) -> dict:
    """per: {query -> ranks dict}. roots: {query -> root key} for root-macro.
    Returns micro (per-query mean) + root-macro (mean over roots of per-root
    mean) for gold@k, top3@10, gold-gap@k."""
    q = list(per)
    out = {"n_queries": len(q)}

    def micro(fn):
        return float(np.mean([fn(per[d]) for d in q])) if q else 0.0

    for k in ks:
        out[f"gold@{k}"] = micro(lambda r, k=k: _flag(r["gold_rank"], k))
        out[f"gold-gap@{k}"] = micro(lambda r, k=k: _flag(r["gap_rank"], k))
    out["top3@10"] = micro(lambda r: _flag(r["top3_rank"], 10))
    out["median_gold_rank"] = float(np.median([per[d]["gold_rank"] for d in q])) if q else 0.0

    if roots is not None:
        from collections import defaultdict
        by_root = defaultdict(list)
        for d in q:
            by_root[roots.get(d, d)].append(d)

        def rootmacro(fn):
            per_root = [np.mean([fn(per[d]) for d in ds]) for ds in by_root.values()]
            return float(np.mean(per_root)) if per_root else 0.0

        out["n_roots"] = len(by_root)
        for k in ks:
            out[f"root_gold@{k}"] = rootmacro(lambda r, k=k: _flag(r["gold_rank"], k))
        out["root_top3@10"] = rootmacro(lambda r: _flag(r["top3_rank"], 10))
        out["root_gold-gap@10"] = rootmacro(lambda r: _flag(r["gap_rank"], 10))
    return out


def from_scores(scores_by_query: dict, cands: dict, roots: dict | None = None,
                ks=KS) -> tuple[dict, dict]:
    """scores_by_query[d] = [M] scores over the shared candidate universe.
    cands[d] = (cand_idx[n], acc[n]). Returns (aggregate, per_query)."""
    per = {}
    for d, (cand, acc) in cands.items():
        if d not in scores_by_query:
            continue
        per[d] = query_ranks(np.asarray(scores_by_query[d]), cand, acc)
    return aggregate(per, roots, ks), per


def from_embeddings(z_m: np.ndarray, z_d: np.ndarray, cands: dict,
                    roots: dict | None = None, ks=KS,
                    normalize: bool = True) -> tuple[dict, dict]:
    """MIPS scorer: s(d,m) = <norm(z_d), norm(z_m)>. cands as in from_scores."""
    zm = z_m.astype(np.float64)
    zd = z_d.astype(np.float64)
    if normalize:
        zm = zm / (np.linalg.norm(zm, axis=1, keepdims=True) + 1e-12)
        zd = zd / (np.linalg.norm(zd, axis=1, keepdims=True) + 1e-12)
    per = {}
    for d, (cand, acc) in cands.items():
        s_all = zm @ zd[int(d)]
        per[d] = query_ranks(s_all, cand, acc)
    return aggregate(per, roots, ks), per


# --------------------------------------------------------------------------
# T0 item 5: streaming scorer.
#
# The caller side of this harness used to materialize a {query: [N] scores}
# dict -- 517 x 12,000 is 25 MB, 517 x 100,000 is 207 MB, 517 x 1,000,000 is
# 2 GB, and every one of those vectors exists only to be reduced to three
# integers (gold / top3 / gap rank). This computes the same three integers
# without ever holding a full score vector: score the handful of PROBE models a
# query actually needs, then stream the candidate universe in blocks counting
# how many models beat each probe.
#
# Ranks are integers, so this is not "close to" from_embeddings -- it is equal,
# and the tests assert per-query equality rather than aggregate agreement.
# --------------------------------------------------------------------------
def _probe_ids(cand, acc, gap_delta):
    """The only models whose score a query's three ranks depend on."""
    cand = np.asarray(cand)
    acc = np.asarray(acc, dtype=float)
    gold = int(cand[int(np.argmax(acc))])
    top3 = cand[np.argsort(-acc)[: min(3, len(acc))]].astype(int)
    near = cand[acc >= acc.max() - gap_delta].astype(int)
    ids = np.unique(np.concatenate([[gold], top3, near]))
    return gold, top3, near, ids


def from_embeddings_streaming(z_m, z_d, cands: dict, roots: dict | None = None,
                              ks=KS, normalize: bool = True, device: str = "cpu",
                              model_chunk: int = 50_000, query_chunk: int = 64,
                              gap_delta: float = GAP_DELTA,
                              dtype=None) -> tuple[dict, dict]:
    """Same output as from_embeddings, in O(n_probes) memory per query block.

    device="cuda" moves the (single) resident z_m to the GPU and streams the
    matmul there; dtype defaults to float64 on CPU (bit-identical to
    from_embeddings) and float32 on GPU.
    """
    import torch

    if dtype is None:
        dtype = torch.float32 if str(device).startswith("cuda") else torch.float64
    zm = torch.as_tensor(np.asarray(z_m)).to(dtype)
    zd = torch.as_tensor(np.asarray(z_d)).to(dtype)
    if normalize:
        zm = zm / (zm.norm(dim=1, keepdim=True) + 1e-12)
        zd = zd / (zd.norm(dim=1, keepdim=True) + 1e-12)
    zm = zm.to(device)
    zd = zd.to(device)
    N = zm.size(0)

    queries = list(cands)
    per = {}
    for qs in range(0, len(queries), query_chunk):
        block = queries[qs:qs + query_chunk]
        meta, flat_ids, flat_q = [], [], []
        for qi, d in enumerate(block):
            cand, acc = cands[d]
            gold, top3, near, ids = _probe_ids(cand, acc, gap_delta)
            meta.append((d, gold, top3, near, ids, int(np.asarray(cand).size)))
            flat_ids.append(ids)
            flat_q.append(np.full(ids.size, qi))
        flat_ids = torch.as_tensor(np.concatenate(flat_ids), dtype=torch.long, device=device)
        flat_q = torch.as_tensor(np.concatenate(flat_q), dtype=torch.long, device=device)
        zq = zd[torch.as_tensor([int(d) for d in block], dtype=torch.long, device=device)]
        s_probe = (zm[flat_ids] * zq[flat_q]).sum(-1)                    # [P]
        greater = torch.zeros_like(s_probe, dtype=torch.long)
        pcols = torch.arange(flat_ids.numel(), device=device)
        for ms in range(0, N, model_chunk):
            sc = zm[ms:ms + model_chunk] @ zq.t()                        # [C, Q]
            cmp = sc[:, flat_q] > s_probe.unsqueeze(0)
            # A probe scored by the gather path and by the matmul path can differ
            # in the last bit, which would let a probe out-rank ITSELF and shift
            # the rank by one. A rank counts STRICTLY better models and nothing is
            # strictly better than itself, so mask the diagonal explicitly.
            here = (flat_ids >= ms) & (flat_ids < ms + sc.size(0))
            if bool(here.any()):
                cmp[flat_ids[here] - ms, pcols[here]] = False
            greater += cmp.sum(0)
        greater = greater.cpu().numpy()

        off = 0
        for (d, gold, top3, near, ids, n_cand) in meta:
            rank = {int(m): int(greater[off + k]) + 1 for k, m in enumerate(ids)}
            off += ids.size
            per[d] = {"gold_rank": rank[gold],
                      "top3_rank": min(rank[int(m)] for m in top3),
                      "gap_rank": min(rank[int(m)] for m in near),
                      "n_candidates": n_cand}
    return aggregate(per, roots, ks), per


# --------------------------------------------------------------------------
# self-test: cross-check gold@K against the established top1_eval, plus hand
# assertions for top3@10 / gold-gap@10 that top1_eval does not compute.
# --------------------------------------------------------------------------
def _selftest() -> int:
    import os
    rng = np.random.default_rng(0)
    N, D, dim = 500, 40, 16
    z_m = rng.standard_normal((N, dim)).astype(np.float32)
    z_d = rng.standard_normal((D, dim)).astype(np.float32)
    cands = {}
    for d in range(D):
        n = int(rng.integers(3, 30))
        idx = rng.choice(N, size=n, replace=False)
        acc = rng.random(n).astype(np.float32)
        cands[d] = (idx, acc)

    agg_ours, per_ours = from_embeddings(z_m, z_d, cands)

    # cross-check gold@K against top1_eval.five_metric_eval on identical inputs
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "stage2TrainGraphSAGE"))
    import torch
    from top1_eval import five_metric_eval
    z_dict = {"model": torch.from_numpy(z_m), "dataset": torch.from_numpy(z_d)}
    per_ref = five_metric_eval(z_dict, cands)

    ref_g1 = np.mean([per_ref[d]["full2k_gold@1"] for d in per_ref])
    ref_g10 = np.mean([per_ref[d]["full2k_gold@10"] for d in per_ref])
    print(f"gold@1  ours={agg_ours['gold@1']:.4f}  top1_eval={ref_g1:.4f}")
    print(f"gold@10 ours={agg_ours['gold@10']:.4f}  top1_eval={ref_g10:.4f}")
    assert abs(agg_ours["gold@1"] - ref_g1) < 1e-9, "gold@1 mismatch vs top1_eval"
    assert abs(agg_ours["gold@10"] - ref_g10) < 1e-9, "gold@10 mismatch vs top1_eval"
    # per-query gold_rank must match exactly (tie-safe formula identical)
    for d in per_ref:
        assert per_ours[d]["gold_rank"] == per_ref[d]["gold_rank"], f"rank {d}"

    # hand cases: top3@10 and gold-gap@10 (not in top1_eval)
    # universe of 5, query with 3 candidates; craft scores so ranks are known.
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5])          # model 0 best ... 4 worst
    cand = np.array([4, 2, 0])                        # acc-sorted below
    acc = np.array([0.10, 0.50, 0.90])               # best=model0(acc .9)
    r = query_ranks(s, cand, acc)
    # gold = model 0 -> rank 1; top3 = {0,2,4} -> min rank 1; near(.01)= {0} rank1
    assert r["gold_rank"] == 1 and r["top3_rank"] == 1 and r["gap_rank"] == 1
    # now make gold poorly ranked but a near-tie well ranked
    s2 = np.array([0.1, 0.2, 0.99, 0.3, 0.4])        # model 2 top-scored
    cand2 = np.array([0, 2])
    acc2 = np.array([0.90, 0.895])                   # gold=model0(rank5), near incl model2
    r2 = query_ranks(s2, cand2, acc2)
    assert r2["gold_rank"] == 5, r2
    assert r2["gap_rank"] == 1, r2   # model2 within 0.01 of best and top-scored
    print("top3@10 / gold-gap@10 hand cases: OK")
    print("\n*** SELFTEST PASSED: harness matches top1_eval on gold@K ***")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
