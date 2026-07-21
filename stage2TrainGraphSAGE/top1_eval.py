"""
top1_eval.py -- Phase 0 of the Top-1/global guide: ONE evaluation function that
computes BOTH observed-candidate and full-pool metrics from the same z_dict and
held-out labels.

Five primary metrics per dataset d (candidates C_d = positive held-out
trained_on edges of test_data; >=3 candidates, non-constant accuracy):

  observed_hit@1  : argmax_{m in C_d} s(d,m) achieves the max observed accuracy
  top3_hit@1      : the selected model is within the true observed top-3
  regret@1        : y(best) - y(selected)
  full2k_gold@1   : gold (highest-acc candidate) ranks 1st among ALL indexed z_m
  full2k_gold@10  : gold survives into the global top-10

full-pool gold@K is a GOLD-SURVIVAL metric, not full-pool precision: unlabeled
models are unknown, not automatically wrong. s(d,m) = <normalize(z_d),
normalize(z_m)> throughout — the exact HNSW serving score.

Also provides analytic random baselines (observed uniform pick / uniform full-pool
ranking) and candidate-count strata aggregation.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

STRATA = [(3, 10), (11, 20), (21, 50), (51, 100), (101, 10 ** 9)]
GOLD_KS = (1, 10, 50, 100)
# v5 P4: gold-gap@K co-primary. Strict gold@K asks "does THE single best-acc
# model survive into top-K"; gold-gap@K relaxes THE to "a model within delta of
# the best". delta = 0.01 is measured, not chosen -- P0 (v5/P0_DIAGNOSIS.md §2.3)
# found 70% of datasets have their 2nd-best within 0.01 and a median gold/2nd gap
# of 0.0017, so strict gold under-counts real serving utility ~3x. This adds a
# metric; it never moves gold@K (the goalpost stays).
GOLD_GAP_DELTA = 0.01


def five_metric_eval(z_dict, cands, *, names=None):
    """Observed + full-pool metrics from ONE z_dict.

    z_dict : {"model": [N,dim], "dataset": [D,dim]} embeddings (from test_data)
    cands  : {dataset_idx: (candidate_model_idx array, normalized_acc array)}
             — materialized once and shared by every configuration.

    Returns per_dataset dict with the five primary metrics + audit fields
    (selected/gold ids & accuracies, global gold rank, reciprocal rank).
    """
    z_m = F.normalize(z_dict["model"], dim=-1)
    z_d = F.normalize(z_dict["dataset"], dim=-1)
    per = {}
    for d, (cand, a) in cands.items():
        ct = torch.as_tensor(cand, dtype=torch.long)
        # observed selection: rank ONLY the held-out observed candidates
        s_obs = (z_m[ct] @ z_d[int(d)]).numpy()
        sel = int(np.argmax(s_obs))
        max_acc = float(a.max())
        top3 = set(np.argsort(-a)[: min(3, len(a))].tolist())
        gold = int(cand[int(np.argmax(a))])
        # global survival: score EVERY indexed model with the same z_dict
        s_all = (z_m @ z_d[int(d)]).numpy()
        gold_rank = int((s_all > s_all[gold]).sum()) + 1     # 1-indexed, tie-safe
        # v5 P4: gold-gap rank = best (lowest) global rank achieved by ANY
        # candidate whose acc is within GOLD_GAP_DELTA of the best. gold itself
        # (gap 0) is always in this set, so gap_rank <= gold_rank -> gold-gap@K
        # is a strict relaxation of gold@K. "good-enough model in top-K".
        near = cand[a >= (max_acc - GOLD_GAP_DELTA)]
        gap_rank = int(min((s_all > s_all[int(m)]).sum() + 1 for m in near))
        rec = {
            "n_candidates": int(cand.size),
            "observed_hit1": float(a[sel] == max_acc),
            "top3_hit1": float(sel in top3),
            "regret1": float(max_acc - a[sel]),
            "selected_model_id": int(cand[sel]),
            "selected_acc": float(a[sel]),
            "gold_model_id": gold,
            "gold_acc": max_acc,
            "gold_rank": gold_rank,
            "gold_gap_rank": gap_rank,
            "n_within_gap_delta": int((a >= max_acc - GOLD_GAP_DELTA).sum()),
            "reciprocal_rank": 1.0 / gold_rank,
            **{f"full2k_gold@{K}": float(gold_rank <= K) for K in GOLD_KS},
            **{f"gold_gap@{K}": float(gap_rank <= K) for K in GOLD_KS},
        }
        if names is not None:
            rec["selected_model_name"] = names[rec["selected_model_id"]]
            rec["gold_model_name"] = names[gold]
        per[int(d)] = rec
    return per


def aggregate(per):
    keys = ["observed_hit1", "top3_hit1", "regret1", "reciprocal_rank"] + \
           [f"full2k_gold@{K}" for K in GOLD_KS] + \
           [f"gold_gap@{K}" for K in GOLD_KS]       # v5 P4 co-primary
    out = {k: float(np.mean([r[k] for r in per.values()])) for k in keys}
    ranks = [r["gold_rank"] for r in per.values()]
    out["median_gold_rank"] = float(np.median(ranks))
    out["mean_gold_rank"] = float(np.mean(ranks))
    out["median_gold_gap_rank"] = float(np.median([r["gold_gap_rank"] for r in per.values()]))
    out["n_datasets"] = len(per)
    return out


def strata_aggregate(per):
    out = {}
    for lo, hi in STRATA:
        rows = [r for r in per.values() if lo <= r["n_candidates"] <= hi]
        key = f"{lo}-{hi if hi < 10**9 else 'inf'}"
        out[key] = ({"n_datasets": len(rows),
                     "observed_hit1": float(np.mean([r["observed_hit1"] for r in rows])),
                     "top3_hit1": float(np.mean([r["top3_hit1"] for r in rows])),
                     "regret1": float(np.mean([r["regret1"] for r in rows]))}
                    if rows else {"n_datasets": 0})
    return out


def random_baselines(cands, n_pool):
    """Analytic expectations under uniform ranking (no model, no variance).

    observed: uniform pick among candidates -> hit1 = E[#argmax ties / n];
    top3 = min(3,n)/n; regret = best - mean(acc).
    full-pool: uniform ranking of n_pool models -> gold@K = K/n_pool,
    E[rank] = (n_pool+1)/2, E[1/rank] = H(n_pool)/n_pool.
    """
    h1, t3, rg = [], [], []
    gap = {K: [] for K in GOLD_KS}       # v5 P4: per-dataset near-set expectation
    for _, (cand, a) in cands.items():
        n = cand.size
        h1.append(float((a == a.max()).sum()) / n)
        t3.append(min(3, n) / n)
        rg.append(float(a.max() - a.mean()))
        s = int((a >= a.max() - GOLD_GAP_DELTA).sum())    # near-set size
        for K in GOLD_KS:
            # P(>=1 of s uniformly-ranked near-models in top-K) via hypergeometric
            miss = 1.0
            for j in range(K):
                miss *= max(n_pool - s - j, 0) / (n_pool - j)
            gap[K].append(1.0 - miss)
    harm = float(np.sum(1.0 / np.arange(1, n_pool + 1)))
    obs = {"observed_hit1": float(np.mean(h1)), "top3_hit1": float(np.mean(t3)),
           "regret1": float(np.mean(rg)), "n_datasets": len(h1)}
    pool = {**{f"full2k_gold@{K}": K / n_pool for K in GOLD_KS},
            **{f"gold_gap@{K}": float(np.mean(gap[K])) for K in GOLD_KS},
            "median_gold_rank": (n_pool + 1) / 2.0,
            "mean_gold_rank": (n_pool + 1) / 2.0,
            "reciprocal_rank": harm / n_pool}
    return {"random_observed": obs, "random_fullpool": pool}
