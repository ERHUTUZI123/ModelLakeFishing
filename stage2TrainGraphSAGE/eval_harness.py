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

from ModelLakeFishing.stage2TrainGraphSAGE.losses import (
    TRAINED_ON, perf_supervision, split_trained_on,
)

DEFAULT_KS = (10, 50, 100, 200)


def make_fixed_splits(data, *, split_seed, num_val=0.1, num_test=0.2,
                      neg_ratio=1.0, disjoint_train_ratio=0.3):
    return split_trained_on(
        data, num_val=num_val, num_test=num_test, neg_ratio=neg_ratio,
        disjoint_train_ratio=disjoint_train_ratio, seed=split_seed)


def per_dataset_tau(scorer, z_dict, split_data, lookup, *, min_per_dataset=3):
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


def _dcg(gains):
    gains = np.asarray(gains, dtype=float)
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    return float((gains * discounts).sum())


def head_retrieval(z_dict, split_data, lookup, *, ks=DEFAULT_KS,
                   min_per_dataset=3, top_frac=0.10):
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
        order = np.argsort(-score)
        truth_order = np.argsort(-a)

        best_acc = float(a.max())
        true_best = int(truth_order[0])
        n_top3 = min(3, n)
        top3 = set(truth_order[:n_top3].tolist())
        n_top10 = max(1, int(np.ceil(top_frac * n)))
        top10 = set(truth_order[:n_top10].tolist())
        a_sorted_desc = np.sort(a)[::-1]

        row = {"n_candidates": n}
        for K in ks:
            k = min(K, n)
            topk = order[:k]
            topk_set = set(topk.tolist())
            ideal_dcg_k = _dcg(a_sorted_desc[:k])
            row[f"hit@{K}"] = float(true_best in topk_set)
            row[f"recall_top3@{K}"] = len(top3 & topk_set) / len(top3)
            row[f"recall_top10pct@{K}"] = len(top10 & topk_set) / len(top10)
            row[f"ndcg@{K}"] = (_dcg(a[topk]) / ideal_dcg_k) if ideal_dcg_k > 0 else 0.0
            row[f"regret@{K}"] = best_acc - float(a[topk].max())
        per[int(d)] = row

    keys = [k for k in (per[next(iter(per))].keys() if per else []) if k != "n_candidates"]
    macro = {k: float(np.mean([per[d][k] for d in per])) for k in keys} if per else {}
    macro["n_datasets"] = len(per)
    return macro, per


def dataset_to_model_hnsw_recall(z_dict, *, k=50, query_datasets=None):
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


def paired_bootstrap(per_a, per_b, metric="tau", *, n_boot=10000, seed=0):
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
