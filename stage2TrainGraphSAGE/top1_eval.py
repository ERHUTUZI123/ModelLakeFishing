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
GOLD_GAP_DELTA = 0.01
_RANK_CHUNK = 1 << 20


def five_metric_eval(z_dict, cands, *, names=None, device=None, expect_n=None):
    z_m = F.normalize(z_dict["model"], dim=-1)
    z_d = F.normalize(z_dict["dataset"], dim=-1)
    if expect_n is not None:
        assert z_m.shape[0] == expect_n, (
            f"candidate pool is {z_m.shape[0]}, expected {expect_n}")
    if device is not None:
        z_m, z_d = z_m.to(device), z_d.to(device)
    per = {}
    for d, (cand, a) in cands.items():
        ct = torch.as_tensor(cand, dtype=torch.long, device=z_m.device)
        s_obs = (z_m[ct] @ z_d[int(d)]).cpu().numpy()
        sel = int(np.argmax(s_obs))
        max_acc = float(a.max())
        top3 = set(np.argsort(-a)[: min(3, len(a))].tolist())
        gold = int(cand[int(np.argmax(a))])
        s_all_t = z_m @ z_d[int(d)]
        near = cand[a >= (max_acc - GOLD_GAP_DELTA)]
        probe = torch.as_tensor(np.concatenate([[gold], np.asarray(near, dtype=np.int64)]),
                                dtype=torch.long, device=z_m.device)
        s_probe = s_all_t[probe]
        counts = torch.zeros(probe.numel(), dtype=torch.long, device=z_m.device)
        for s in range(0, s_all_t.numel(), _RANK_CHUNK):
            counts += (s_all_t[s:s + _RANK_CHUNK].unsqueeze(1) > s_probe.unsqueeze(0)).sum(0)
        ranks = counts.cpu().numpy() + 1
        del s_all_t
        gold_rank = int(ranks[0])
        gap_rank = int(ranks.min())
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
           [f"gold_gap@{K}" for K in GOLD_KS]
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
    h1, t3, rg = [], [], []
    gap = {K: [] for K in GOLD_KS}
    for _, (cand, a) in cands.items():
        n = cand.size
        h1.append(float((a == a.max()).sum()) / n)
        t3.append(min(3, n) / n)
        rg.append(float(a.max() - a.mean()))
        s = int((a >= a.max() - GOLD_GAP_DELTA).sum())
        for K in GOLD_KS:
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
