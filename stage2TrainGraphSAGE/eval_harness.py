"""
eval_harness.py -- Phase 0: the ONE evaluation path every ablation reuses.

The Kendall action guide demands a single, fixed, leakage-free evaluation path so
that configs are compared on identical splits with a paired uncertainty estimate,
and so the SERVING relation (z_d -> z_m exact-dot retrieval) is measured directly
-- not the z_m -> z_m diagnostic that train.hnsw_recall used to report.

What this module provides:

  * make_fixed_splits         -- materialize train/val/test ONCE per split_seed and
                                 reuse for every config. split_seed is separate from
                                 the model init_seed (a config never silently changes
                                 its own test set).
  * per_dataset_tau           -- within-dataset Kendall tau + candidate/pair counts,
                                 returned PER DATASET so configs can be compared with
                                 a paired test (not just an aggregate).
  * head_retrieval            -- exact z_d . z_m retrieval over the OBSERVED held-out
                                 candidates of each test dataset: Hit@K, Recall@K for
                                 true top-3 / top-10%, NDCG@K, regret@K. Per dataset
                                 then macro-averaged. This is the production query.
  * dataset_to_model_hnsw_recall -- ANN fidelity: index ALL z_m, query with z_d,
                                 overlap of HNSW top-K with exact-dot top-K. (The old
                                 train.hnsw_recall queried z_m with z_m -- a different
                                 relation; kept under model_to_model_hnsw_recall.)
  * paired_bootstrap          -- 95% paired bootstrap CI for delta of a per-dataset
                                 metric between two configs over the SAME datasets.

Evaluation universe (state it plainly): for a test dataset d, the candidate set is
the set of models that have a HELD-OUT (test-split) trained_on edge to d -- i.e.
models whose true performance on d we observe but did not train on. Metrics over
this observed set are NOT full-lake recall; they are leakage-free held-out ranking.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.losses import (  # noqa: E402
    TRAINED_ON, perf_supervision, split_trained_on,
)

DEFAULT_KS = (10, 50, 100, 200)


# ── split discipline ─────────────────────────────────────────────────────────

def make_fixed_splits(data, *, split_seed, num_val=0.1, num_test=0.2,
                      neg_ratio=1.0, disjoint_train_ratio=0.3):
    """Materialize (train, val, test) ONCE for a split_seed. Reuse the returned
    objects for every config so paired comparisons are on identical test edges.

    Returns (train_data, val_data, test_data). The split is a pure function of
    split_seed (split_trained_on seeds torch's RNG internally); the caller sets a
    SEPARATE init_seed right before model construction.
    """
    return split_trained_on(
        data, num_val=num_val, num_test=num_test, neg_ratio=neg_ratio,
        disjoint_train_ratio=disjoint_train_ratio, seed=split_seed)


# ── per-dataset within-dataset Kendall tau ───────────────────────────────────

def per_dataset_tau(scorer, z_dict, split_data, lookup, *, min_per_dataset=3):
    """Within-dataset Kendall tau on a split's POSITIVE trained_on edges.

    Returns (macro_tau, per_dataset) where per_dataset maps dataset_idx ->
    {tau, n_candidates, n_comparable_pairs}. n_comparable_pairs counts non-tied
    pairs (the pairs Kendall actually scores). Datasets with < min_per_dataset
    scorable candidates or zero score/target variance are excluded (tau undefined).
    """
    dev = z_dict["model"].device
    eli, target = perf_supervision(split_data[TRAINED_ON], lookup)
    eli = eli.to(dev)
    pred = scorer(z_dict["model"], z_dict["dataset"], eli).detach().cpu().numpy()
    tg = target.numpy()
    ds = eli[1].cpu().numpy()

    per = {}
    taus = []
    for d in np.unique(ds):
        m = ds == d
        n = int(m.sum())
        if n < min_per_dataset:
            continue
        tg_d, pr_d = tg[m], pred[m]
        # non-tied comparable pairs in the TRUTH (what Kendall can score)
        diffs = tg_d[:, None] - tg_d[None, :]
        n_pairs = int((np.triu(diffs, 1) != 0).sum())
        if np.std(tg_d) == 0 or np.std(pr_d) == 0:
            continue
        t, _ = kendalltau(pr_d, tg_d)
        if np.isnan(t):
            continue
        per[int(d)] = {"tau": float(t), "n_candidates": n, "n_comparable_pairs": n_pairs}
        taus.append(t)
    macro = float(np.mean(taus)) if taus else float("nan")
    return macro, per


# ── exact-dot z_d -> z_m head retrieval (the production query) ────────────────

def _dcg(gains):
    gains = np.asarray(gains, dtype=float)
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    return float((gains * discounts).sum())


def head_retrieval(z_dict, split_data, lookup, *, ks=DEFAULT_KS,
                   min_per_dataset=3, top_frac=0.10):
    """Exact z_d . z_m retrieval over each test dataset's OBSERVED held-out models.

    For dataset d: candidates = models with a held-out trained_on edge to d;
    true relevance = their normalized accuracy. Rank candidates by exact dot
    score(d, m) = <z_d, z_m> (the score HNSW will rank by) and compute, per K:

      hit@K        : true single best candidate is within the top-K retrieved
      recall_top3  : fraction of the true top-3 candidates within top-K
      recall_top10pct : fraction of the true top-ceil(top_frac*n) within top-K
      ndcg@K       : DCG of retrieved order / ideal DCG, gain = accuracy
      regret@K     : best true accuracy  -  best true accuracy among top-K retrieved

    Per-dataset values are returned plus their macro average. Datasets with
    < min_per_dataset candidates or no accuracy variance are skipped.
    """
    dev = z_dict["model"].device
    z_m = F.normalize(z_dict["model"], p=2, dim=-1)
    z_d = F.normalize(z_dict["dataset"], p=2, dim=-1)

    eli, target = perf_supervision(split_data[TRAINED_ON], lookup)
    models = eli[0].numpy()
    ds = eli[1].numpy()
    acc = target.numpy()

    per = {}
    for d in np.unique(ds):
        sel = ds == d
        cand = models[sel]
        a = acc[sel]
        n = int(cand.size)
        if n < min_per_dataset or np.std(a) == 0:
            continue
        zd = z_d[int(d)].to(dev)
        zm = z_m[torch.as_tensor(cand, dtype=torch.long, device=dev)]
        score = (zm @ zd).detach().cpu().numpy()
        order = np.argsort(-score)                 # retrieved ranking (best first)
        truth_order = np.argsort(-a)               # ideal ranking by accuracy

        best_acc = float(a.max())
        true_best = int(truth_order[0])
        n_top3 = min(3, n)
        top3 = set(truth_order[:n_top3].tolist())
        n_top10 = max(1, int(np.ceil(top_frac * n)))
        top10 = set(truth_order[:n_top10].tolist())
        ideal_dcg = _dcg(np.sort(a)[::-1])

        row = {"n_candidates": n}
        for K in ks:
            k = min(K, n)
            topk = order[:k]
            topk_set = set(topk.tolist())
            row[f"hit@{K}"] = float(true_best in topk_set)
            row[f"recall_top3@{K}"] = len(top3 & topk_set) / len(top3)
            row[f"recall_top10pct@{K}"] = len(top10 & topk_set) / len(top10)
            row[f"ndcg@{K}"] = (_dcg(a[topk]) / ideal_dcg) if ideal_dcg > 0 else 0.0
            row[f"regret@{K}"] = best_acc - float(a[topk].max())
        per[int(d)] = row

    keys = [k for k in (per[next(iter(per))].keys() if per else []) if k != "n_candidates"]
    macro = {k: float(np.mean([per[d][k] for d in per])) for k in keys} if per else {}
    macro["n_datasets"] = len(per)
    return macro, per


# ── ANN fidelity: index z_m, query z_d (the serving path) ─────────────────────

def dataset_to_model_hnsw_recall(z_dict, *, k=50, query_datasets=None):
    """Index ALL z_m in HNSW, query with z_d, compare HNSW top-K to exact-dot
    top-K. This validates the SERVING path (z_d -> z_m). Returns mean ANN recall
    over the queried datasets, or None if hnswlib is missing.

    query_datasets : optional list of dataset indices to query (default: all).
    """
    try:
        import hnswlib
    except Exception:
        return None
    z_m = F.normalize(z_dict["model"], p=2, dim=-1).detach().cpu().numpy().astype("float32")
    z_d = F.normalize(z_dict["dataset"], p=2, dim=-1).detach().cpu().numpy().astype("float32")
    N, dim = z_m.shape
    k = min(k, N)
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=N, ef_construction=200, M=16)
    index.add_items(z_m, np.arange(N))
    index.set_ef(max(64, k + 16))

    q = z_d if query_datasets is None else z_d[np.asarray(query_datasets)]
    exact = np.argsort(-(q @ z_m.T), axis=1)[:, :k]
    ann, _ = index.knn_query(q, k=k)
    recs = [len(set(ann[i].tolist()) & set(exact[i].tolist())) / k for i in range(q.shape[0])]
    return float(np.mean(recs))


def model_to_model_hnsw_recall(z_model, near_hub, k=50):
    """The ORIGINAL train.hnsw_recall, renamed: index z_m, query z_m, split
    near-hub / away-hub. Kept as a structural diagnostic ONLY -- it does NOT
    validate the z_d -> z_m serving path. Needs hnswlib."""
    try:
        import hnswlib
    except Exception:
        return None
    z = F.normalize(z_model, p=2, dim=-1).detach().cpu().numpy().astype("float32")
    N = z.shape[0]
    k = min(k, N - 1)
    index = hnswlib.Index(space="cosine", dim=z.shape[1])
    index.init_index(max_elements=N, ef_construction=200, M=16)
    index.add_items(z, np.arange(N))
    index.set_ef(max(64, k + 16))
    sim = z @ z.T
    np.fill_diagonal(sim, -np.inf)
    brute = np.argsort(-sim, axis=1)[:, :k]
    labels, _ = index.knn_query(z, k=k + 1)
    out = {}
    for name, mask in (("near_hub", near_hub.numpy()), ("away_hub", (~near_hub).numpy())):
        if mask.sum() == 0:
            out[name] = float("nan")
            continue
        recs = []
        for i in np.where(mask)[0]:
            a = set(int(x) for x in labels[i] if int(x) != i)
            recs.append(len(a & set(brute[i].tolist())) / k)
        out[name] = float(np.mean(recs))
    return out


# ── paired uncertainty over datasets ──────────────────────────────────────────

def paired_bootstrap(per_a, per_b, metric="tau", *, n_boot=10000, seed=0):
    """95% paired bootstrap CI for (b - a) of a per-dataset metric, over the
    datasets BOTH configs scored. per_a / per_b map dataset_idx -> {metric: val}.

    Returns {n_paired, mean_delta, ci_low, ci_high, prob_positive}. An interval
    excluding zero is the guide's promotion gate.
    """
    common = sorted(set(per_a) & set(per_b))
    if not common:
        return {"n_paired": 0, "mean_delta": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"),
                "prob_positive": float("nan")}
    da = np.array([per_a[d][metric] for d in common])
    db = np.array([per_b[d][metric] for d in common])
    diff = db - da
    rng = np.random.default_rng(seed)
    n = diff.size
    boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return {
        "n_paired": n,
        "mean_delta": float(diff.mean()),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "prob_positive": float((boots > 0).mean()),
    }
