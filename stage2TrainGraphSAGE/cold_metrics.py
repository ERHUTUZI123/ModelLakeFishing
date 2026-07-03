"""
cold_metrics.py -- Cold-Dataset guide §6 semantic metrics + §7 random baseline.

All metrics rank candidates by exact s(d,m) = <normalize(z_d), normalize(z_m)>.
Corrected per the guide:

  tau       : Kendall tau-b over candidates (scipy default), needs >=3 candidates
              and >=2 distinct accuracies (a non-tied pair).
  NDCG@K    : DCG@K(pred) / IDCG@K  where IDCG@K uses the best k=min(K,n) true
              candidates (NOT the full-list DCG).
  Hit@K     : tie-aware -- the tied-best set B_d = {m : y_m = max y} intersects
              the predicted top-k.
  Rec@K     : recall of the tie-aware true top-3 set T3_d (accuracy >= third-
              highest DISTINCT cutoff), |T3 ∩ topk| / |T3|.
  top3_hit@1: the single argmax_s model is in T3_d (distinct from Rec@1).

Random (expected) is analytic per dataset, macro-averaged (never pooled):
  tau=0; top3_hit@1=|T3|/n; Hit@1=|B|/n; Hit@K=1-C(n-|B|,k)/C(n,k);
  Rec@K=k/n; NDCG@K = (mean(y)*Z_k)/IDCG@K  (exact permutation expectation, since
  IDCG@K is a per-dataset constant and E[y at any rank]=mean(y)); Z_k=sum 1/log2(i+1).
"""

import math

import numpy as np
from scipy.stats import kendalltau

KS = (1, 10)


def _dcg(gains):
    gains = np.asarray(gains, dtype=float)
    return float((gains / np.log2(np.arange(2, gains.size + 2))).sum())


def true_top3_set(a):
    """T3_d: indices whose accuracy >= the third-highest DISTINCT value (tie-aware;
    all candidates if < 3 distinct values / < 3 candidates)."""
    distinct = sorted(set(np.asarray(a).tolist()), reverse=True)
    cutoff = distinct[min(2, len(distinct) - 1)]
    return set(int(i) for i, v in enumerate(a) if v >= cutoff)


def best_set(a):
    mx = float(np.max(a))
    return set(int(i) for i, v in enumerate(a) if v == mx)


def dataset_metrics(score, a, *, ks=KS):
    """Per-dataset semantic metrics from predicted scores `score` and true
    normalized accuracies `a` over the SAME candidate list. Returns a dict or None
    if the dataset is not tau-eligible (needs >=3 candidates, >=2 distinct acc)."""
    a = np.asarray(a, dtype=float)
    score = np.asarray(score, dtype=float)
    n = a.size
    if n < 3 or np.unique(a).size < 2:
        return None
    order = np.argsort(-score)                       # predicted best-first
    a_desc = np.sort(a)[::-1]
    B = best_set(a)
    T3 = true_top3_set(a)
    sel = int(order[0])
    tau, _ = kendalltau(score, a)                    # tau-b
    out = {"n_candidates": n, "tau": float(tau) if not np.isnan(tau) else 0.0,
           "top3_hit@1": float(sel in T3), "n_true_top3": len(T3), "n_best": len(B)}
    for K in ks:
        k = min(K, n)
        topk = set(order[:k].tolist())
        idcg = _dcg(a_desc[:k])
        out[f"NDCG@{K}"] = (_dcg(a[list(order[:k])]) / idcg) if idcg > 0 else 0.0
        out[f"Hit@{K}"] = float(len(B & topk) > 0)
        out[f"Rec@{K}"] = len(T3 & topk) / len(T3)
    return out


def _log_comb(n, k):
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def dataset_random_expected(a, *, ks=KS):
    """Analytic random-permutation expectation for one dataset (guide §7)."""
    a = np.asarray(a, dtype=float)
    n = a.size
    if n < 3 or np.unique(a).size < 2:
        return None
    a_desc = np.sort(a)[::-1]
    nB = len(best_set(a))
    nT3 = len(true_top3_set(a))
    mean_y = float(a.mean())
    out = {"tau": 0.0, "top3_hit@1": nT3 / n}
    for K in ks:
        k = min(K, n)
        # Hit@K: P(>=1 of nB best in top-k) = 1 - C(n-nB,k)/C(n,k)
        out[f"Hit@{K}"] = 1.0 - math.exp(_log_comb(n - nB, k) - _log_comb(n, k))
        out[f"Rec@{K}"] = k / n                       # E[recall of T3] = k/n
        Zk = sum(1.0 / math.log2(i + 1) for i in range(1, k + 1))
        idcg = _dcg(a_desc[:k])
        out[f"NDCG@{K}"] = (mean_y * Zk / idcg) if idcg > 0 else 0.0
    return out


AGG_KEYS = ["tau", "NDCG@1", "Hit@1", "top3_hit@1", "Rec@1", "NDCG@10", "Hit@10", "Rec@10"]


def macro(per):
    """Macro-average the presentation keys over datasets. `per` maps id->metrics
    dict; tau uses the key 'tau' (renamed to tau_macro on presentation)."""
    if not per:
        return {k: float("nan") for k in AGG_KEYS} | {"n_datasets": 0}
    vals = list(per.values())
    out = {k: float(np.mean([v[k] for v in vals])) for k in AGG_KEYS}
    out["tau_macro"] = out.pop("tau")
    out["n_datasets"] = len(per)
    return out
