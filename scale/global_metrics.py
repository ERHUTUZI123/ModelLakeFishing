import argparse
import sys

import numpy as np

GAP_DELTA = 0.01
KS = (1, 10)


def query_ranks(s_all: np.ndarray, cand: np.ndarray, acc: np.ndarray,
                gap_delta: float = GAP_DELTA,
                tie_break: np.ndarray | None = None) -> dict:
    s_all = np.asarray(s_all)
    cand = np.asarray(cand)
    acc = np.asarray(acc, dtype=float)
    if tie_break is not None:
        tie_break = np.asarray(tie_break)
        if tie_break.shape != s_all.shape:
            raise ValueError("tie_break shape %r != scores shape %r"
                             % (tie_break.shape, s_all.shape))

    def rank_of(m):
        m = int(m)
        score = s_all[m]
        rank = int((s_all > score).sum()) + 1
        if tie_break is not None:
            rank += int(((s_all == score) & (tie_break < tie_break[m])).sum())
        return rank

    gold = int(cand[int(np.argmax(acc))])
    gold_rank = rank_of(gold)

    order = np.argsort(-acc)[: min(3, len(acc))]
    top3 = cand[order]
    top3_rank = int(min(rank_of(m) for m in top3))

    near = cand[acc >= acc.max() - gap_delta]
    gap_rank = int(min(rank_of(m) for m in near))
    return {"gold_rank": gold_rank, "top3_rank": top3_rank,
            "gap_rank": gap_rank, "n_candidates": int(cand.size)}


def _flag(rank, k):
    return float(rank <= k)


def aggregate(per: dict, roots: dict | None = None, ks=KS) -> dict:
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
    per = {}
    for d, (cand, acc) in cands.items():
        if d not in scores_by_query:
            continue
        per[d] = query_ranks(np.asarray(scores_by_query[d]), cand, acc)
    return aggregate(per, roots, ks), per


def from_embeddings(z_m: np.ndarray, z_d: np.ndarray, cands: dict,
                    roots: dict | None = None, ks=KS,
                    normalize: bool = True) -> tuple[dict, dict]:
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


def _probe_ids(cand, acc, gap_delta):
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
        s_probe = (zm[flat_ids] * zq[flat_q]).sum(-1)
        greater = torch.zeros_like(s_probe, dtype=torch.long)
        pcols = torch.arange(flat_ids.numel(), device=device)
        for ms in range(0, N, model_chunk):
            sc = zm[ms:ms + model_chunk] @ zq.t()
            cmp = sc[:, flat_q] > s_probe.unsqueeze(0)
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
    for d in per_ref:
        assert per_ours[d]["gold_rank"] == per_ref[d]["gold_rank"], f"rank {d}"

    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    cand = np.array([4, 2, 0])
    acc = np.array([0.10, 0.50, 0.90])
    r = query_ranks(s, cand, acc)
    assert r["gold_rank"] == 1 and r["top3_rank"] == 1 and r["gap_rank"] == 1
    s2 = np.array([0.1, 0.2, 0.99, 0.3, 0.4])
    cand2 = np.array([0, 2])
    acc2 = np.array([0.90, 0.895])
    r2 = query_ranks(s2, cand2, acc2)
    assert r2["gold_rank"] == 5, r2
    assert r2["gap_rank"] == 1, r2
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
