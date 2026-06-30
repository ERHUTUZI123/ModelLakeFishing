"""
kendall_lab.py -- Phase 0 of KENDALL_ACTION_GUIDE: a dedicated, reusable evaluation
harness for the dataset->model retrieval ladder. NOT production: it imports the
existing graph/model/loss UNCHANGED and only adds fixed-split discipline + the
serving-path metrics every ablation must reuse.

What it provides (guide Phase 0):
  * fixed edge splits with split_seed SEPARATE from init_seed (a model seed never
    changes the test set);
  * baseline reproduction on those splits;
  * per-dataset held-out Kendall tau WITH comparable non-tied pair counts;
  * EXACT z_d->z_m head retrieval on the RAW cosine score HNSW will rank by:
    Hit@K (true best), Recall@K (true top-3 / top-10%), NDCG@K, regret@K;
  * paired bootstrap over datasets (delta vs a baseline artifact);
  * a leakage audit (held-out edges + their reverse absent from the message graph);
  * a reproducible run manifest.

Geometry note: serving score = <normalize(z_d), normalize(z_m)>. The dot PerfScorer
is sigmoid(scale*<z_m,z_d>+bias), monotone in the cosine, so the ORDER it learns is
the cosine order -- head metrics are therefore computed on the raw cosine directly.

Run B0 (reproduce baseline + emit serving metrics):
  python -m ModelLakeFishing.stage2TrainGraphSAGE.kendall_lab --config B0 \
      --pt ModelLakeFishing/stage1BuildTransferGraph/hgraph_hf1000d_2000m_xm0_xd0.pt \
      --split_seeds 0 1 2 --epochs 25 --device cuda
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.model import HeteroGraphSAGE, load_hgraph  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.losses import (  # noqa: E402
    TRAINED_ON, REV_TRAINED_ON, PerfScorer, accuracy_lookup, perf_supervision,
    split_trained_on, topk_membership, lineage_components,
)
from ModelLakeFishing.stage2TrainGraphSAGE.train import train, collapse_report  # noqa: E402

ARTIFACTS = os.path.join(_HERE, "artifacts", "kendall_lab")
DEFAULT_PT = os.path.join(_HERE, "..", "stage1BuildTransferGraph", "hgraph_hf1000d_2000m_xm0_xd0.pt")
CANDIDATE = dict(num_layers=1, top_frac=0.10, lambda_contrast=1.0, lambda_rank=1.0,
                 lambda_mse=0.0, lambda_uniform=0.0, scorer="dot")
KS = (10, 50, 100, 200)
MIN_PAIRS = 1     # a dataset is "scorable" for tau if it has >=1 comparable non-tied pair


# ── splits (fixed; split_seed independent of init_seed) ───────────────────────
def make_split(data, split_seed):
    """Transductive RandomLinkSplit of trained_on (+ reverse). Deterministic in
    split_seed only."""
    return split_trained_on(data, seed=split_seed)


def leakage_audit(train_data, test_data):
    """Assert every held-out POSITIVE test edge (and its reverse) is absent from the
    train message graph. Returns a dict; raises on leak."""
    tr = set(map(tuple, train_data[TRAINED_ON].edge_index.t().tolist()))
    te_pos = test_data[TRAINED_ON].edge_label_index[:, test_data[TRAINED_ON].edge_label == 1]
    leaked = [tuple(e) for e in te_pos.t().tolist() if tuple(e) in tr]
    # reverse direction
    rev_tr = set(map(tuple, train_data[REV_TRAINED_ON].edge_index.t().tolist())) \
        if REV_TRAINED_ON in train_data.edge_types else set()
    rev_leaked = [(d, m) for (m, d) in (tuple(e) for e in te_pos.t().tolist()) if (d, m) in rev_tr]
    audit = dict(train_msg_edges=len(tr), test_pos_edges=int(te_pos.size(1)),
                 leaked_fwd=len(leaked), leaked_rev=len(rev_leaked))
    assert not leaked and not rev_leaked, f"LEAKAGE: {audit}"
    return audit


# ── train one config on one split ─────────────────────────────────────────────
def train_one(data, xm0, xd0, cfg, split_seed, init_seed, epochs, device):
    device = torch.device(device)
    train_data, val_data, test_data = make_split(data, split_seed)
    lookup = accuracy_lookup(data)
    eli, target = perf_supervision(train_data[TRAINED_ON], lookup)
    ti = torch.cat([train_data[TRAINED_ON].edge_index, eli], dim=1)
    ta = torch.cat([train_data[TRAINED_ON].edge_attr.float(), target], dim=0)
    M = topk_membership(data, top_frac=cfg["top_frac"], trained_on_index=ti, trained_on_attr=ta)
    comp = lineage_components(data, data["model"].num_nodes)

    torch.manual_seed(init_seed); np.random.seed(init_seed)
    ds_kw = dict(num_task_types=xd0["num_task_types"], n_class_buckets=xd0["n_class_buckets"],
                 num_arities=xd0["num_arities"]) if xd0 else {}
    model = HeteroGraphSAGE(
        metadata=data.metadata(), frozen_dim=data["model"].x.shape[1],
        num_size_buckets=xm0["num_size_buckets"], num_families=xm0["num_families"],
        dataset_in_dim=data["dataset"].x.shape[1], num_layers=cfg["num_layers"], **ds_kw).to(device)
    scorer = PerfScorer(dim=128, mode=cfg["scorer"]).to(device)
    train(model, scorer, train_data, eli, target, M.to(device), comp.to(device), epochs=epochs,
          lambda_mse=cfg["lambda_mse"], lambda_rank=cfg["lambda_rank"],
          lambda_contrast=cfg["lambda_contrast"], lambda_uniform=cfg["lambda_uniform"], device=device)
    audit = leakage_audit(train_data, test_data)
    return model, scorer, train_data, test_data, lookup, audit, device


# ── serving-path embeddings + head metrics (raw cosine, z_d -> z_m) ───────────
def serving_embeddings(model, message_data, device):
    """z on the split's REDUCED message graph (held-out edges already removed).
    Returns L2-normalized z_m, z_d as cpu numpy (the exact HNSW geometry)."""
    model.eval()
    with torch.no_grad():
        z = model(message_data.clone().to(device))
    zm = torch.nn.functional.normalize(z["model"], dim=-1).cpu().numpy()
    zd = torch.nn.functional.normalize(z["dataset"], dim=-1).cpu().numpy()
    return zm, zd


def _test_candidates(test_data, lookup):
    """{dataset_idx -> [(model_idx, true_acc)]} from held-out POSITIVE test edges."""
    eli = test_data[TRAINED_ON].edge_label_index
    pos = test_data[TRAINED_ON].edge_label == 1
    eli = eli[:, pos]
    by_d = defaultdict(list)
    for k in range(eli.size(1)):
        m, d = int(eli[0, k]), int(eli[1, k])
        if (m, d) in lookup:
            by_d[d].append((m, lookup[(m, d)]))
    return by_d


def _ndcg(scores, rel, K):
    order = np.argsort(-scores)[:K]
    gains = (2 ** rel[order] - 1)
    discounts = 1.0 / np.log2(np.arange(2, len(order) + 2))
    dcg = float((gains * discounts).sum())
    ideal = np.sort(rel)[::-1][:K]
    idcg = float(((2 ** ideal - 1) * (1.0 / np.log2(np.arange(2, len(ideal) + 2)))).sum())
    return dcg / idcg if idcg > 1e-12 else np.nan


def per_dataset_metrics(zm, zd, cand_by_d, Ks=KS):
    rows = []
    for d, cands in cand_by_d.items():
        if len(cands) < 2:
            continue
        m_idx = np.array([m for m, _ in cands])
        acc = np.array([a for _, a in cands], dtype=float)
        scores = (zd[d] * zm[m_idx]).sum(-1)        # RAW cosine, serving geometry
        if np.std(acc) < 1e-9:
            continue                                 # no rankable signal -> skip (tied)
        tau, _ = kendalltau(scores, acc)
        # comparable non-tied pairs
        n = len(acc)
        nontied = int(sum(1 for i in range(n) for j in range(i + 1, n) if abs(acc[i] - acc[j]) > 1e-9))
        if np.isnan(tau) or nontied < MIN_PAIRS:
            continue
        best_acc = acc.max()
        order = np.argsort(-scores)
        true_best = int(np.argmax(acc))
        top3 = set(np.argsort(-acc)[:3].tolist())
        n_top10 = max(1, int(np.ceil(0.10 * n)))
        top10 = set(np.argsort(-acc)[:n_top10].tolist())
        row = dict(dataset=d, n_cand=n, n_nontied_pairs=nontied, tau=float(tau))
        for K in Ks:
            kk = min(K, n)
            topk = set(order[:kk].tolist())
            row[f"hit@{K}"] = float(true_best in topk)
            row[f"recall_top3@{K}"] = len(top3 & topk) / len(top3)
            row[f"recall_top10pct@{K}"] = len(top10 & topk) / len(top10)
            row[f"ndcg@{K}"] = _ndcg(scores, acc, kk)
            row[f"regret@{K}"] = float(best_acc - acc[list(topk)].max())
        rows.append(row)
    return pd.DataFrame(rows)


def macro(df, col="tau"):
    return float(df[col].mean()) if len(df) else float("nan")


def paired_bootstrap(df_base, df_new, col="tau", n_boot=10000, seed=0):
    """Paired bootstrap over the datasets common to both. Returns (delta_mean, lo, hi)."""
    j = df_base[["dataset", col]].merge(df_new[["dataset", col]], on="dataset",
                                        suffixes=("_b", "_n"))
    if len(j) == 0:
        return float("nan"), float("nan"), float("nan")
    d = (j[f"{col}_n"] - j[f"{col}_b"]).to_numpy()
    rng = np.random.default_rng(seed)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ── run a config across split seeds, write artifacts ──────────────────────────
def run_config(name, data, xm0, xd0, cfg, split_seeds, init_seed, epochs, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    all_rows, summary = [], []
    for ss in split_seeds:
        model, scorer, train_data, test_data, lookup, audit, dev = train_one(
            data, xm0, xd0, cfg, ss, init_seed, epochs, device)
        zm, zd = serving_embeddings(model, test_data, dev)
        cand = _test_candidates(test_data, lookup)
        df = per_dataset_metrics(zm, zd, cand)
        df["split_seed"] = ss
        all_rows.append(df)
        with torch.no_grad():
            zfull = model(data.clone().to(dev))["model"]
        srow = dict(split_seed=ss, n_datasets_scored=len(df),
                    tau_macro=macro(df, "tau"), mean_cos=collapse_report(zfull),
                    leak_fwd=audit["leaked_fwd"], leak_rev=audit["leaked_rev"])
        for K in KS:
            for m in ("hit", "recall_top3", "recall_top10pct", "ndcg", "regret"):
                srow[f"{m}@{K}"] = macro(df, f"{m}@{K}")
        summary.append(srow)
        print(f"[{name} split={ss}] datasets={len(df)} tau_macro={srow['tau_macro']:.4f} "
              f"hit@10={srow['hit@10']:.3f} ndcg@10={srow['ndcg@10']:.3f} regret@10={srow['regret@10']:.4f} "
              f"leak={audit['leaked_fwd']+audit['leaked_rev']}")
    per_ds = pd.concat(all_rows, ignore_index=True)
    per_ds.to_csv(os.path.join(out_dir, f"{name}_per_dataset.csv"), index=False)
    sdf = pd.DataFrame(summary)
    agg = {c: (float(sdf[c].mean()), float(sdf[c].std())) for c in sdf.columns if c != "split_seed"}
    json.dump(dict(config=name, cfg=cfg, split_seeds=list(split_seeds), init_seed=init_seed,
                   epochs=epochs, aggregate=agg),
              open(os.path.join(out_dir, f"{name}_summary.json"), "w"), indent=2)
    return per_ds, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="B0")
    ap.add_argument("--pt", default=os.path.normpath(DEFAULT_PT))
    ap.add_argument("--split_seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--init_seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", default=ARTIFACTS)
    args = ap.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    data, xm0, _umi = load_hgraph(args.pt)
    payload = torch.load(args.pt, map_location="cpu", weights_only=False)
    xd0 = payload.get("xd0_meta")
    print(f"graph {os.path.basename(args.pt)}: {data['model'].num_nodes} models, "
          f"{data['dataset'].num_nodes} datasets, dataset.x {tuple(data['dataset'].x.shape)}")
    print(f"config={args.config} cfg={CANDIDATE} split_seeds={args.split_seeds} "
          f"init_seed={args.init_seed} device={args.device}\n")

    # manifest
    os.makedirs(args.out, exist_ok=True)
    json.dump(dict(graph=os.path.abspath(args.pt), config=args.config, cfg=CANDIDATE,
                   split_seeds=args.split_seeds, init_seed=args.init_seed, epochs=args.epochs,
                   xm0_num_families=xm0["num_families"], device=args.device, Ks=list(KS)),
              open(os.path.join(args.out, f"{args.config}_manifest.json"), "w"), indent=2)

    per_ds, agg = run_config(args.config, data, xm0, xd0, CANDIDATE,
                             args.split_seeds, args.init_seed, args.epochs, args.device, args.out)
    print("\n" + "=" * 64)
    print(f"{args.config} aggregate over splits {args.split_seeds} (mean ± std):")
    for k in ("tau_macro", "n_datasets_scored", "mean_cos", "hit@10", "recall_top3@10",
              "recall_top10pct@10", "ndcg@10", "regret@10", "hit@50", "ndcg@50", "regret@50"):
        if k in agg:
            print(f"  {k:<20} {agg[k][0]:>8.4f} ± {agg[k][1]:.4f}")
    print(f"  leakage (fwd+rev)    {agg['leak_fwd'][0]+agg['leak_rev'][0]:.0f}")
    print("=" * 64)
    print(f"artifacts -> {args.out}/{args.config}_per_dataset.csv, _summary.json, _manifest.json")


if __name__ == "__main__":
    main()
