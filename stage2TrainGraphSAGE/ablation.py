"""
ablation.py -- the single experiment driver for the Kendall action guide.

Runs ONE config (a point on the ablation ladder B0..B9) over fixed splits with a
separated init_seed, evaluates it with the shared eval_harness (per-dataset tau,
exact z_d->z_m head retrieval, collapse / participation), and writes a per-dataset
metric artifact plus an aggregate. Two configs run here can be compared with a
paired bootstrap over the identical test datasets (eval_harness.paired_bootstrap).

Design rules enforced (guide, "Split discipline"):
  * fixed splits materialized once per split_seed, reused for every config;
  * split_seed is SEPARATE from init_seed -- a config never changes its test set;
  * contrastive membership M is built from TRAIN-VISIBLE edges only (no leakage);
  * the evaluation universe (held-out observed candidates) is stated in the report.

A config is a plain dict. Phase flags are added as each phase lands; unknown keys
fall back to the B0 baseline behaviour so old configs keep reproducing.

Run a single config (defaults reproduce the B0 baseline on hf1000d/2000m):
  python -m ModelLakeFishing.stage2TrainGraphSAGE.ablation \
      --pt ModelLakeFishing/stage1BuildTransferGraph/hgraph_hf1000d_2000m_xm0_xd0.pt \
      --name B0 --seeds 3 --epochs 25
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.model import HeteroGraphSAGE, load_hgraph  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.losses import (  # noqa: E402
    TRAINED_ON, PerfScorer, accuracy_lookup, perf_supervision,
    topk_membership, lineage_components, global_positive_density, per_dataset_density,
)
from ModelLakeFishing.stage2TrainGraphSAGE.train import train, train_grouped, collapse_report  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.learnable import save_checkpoint, load_checkpoint  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.eval_harness import (  # noqa: E402
    make_fixed_splits, per_dataset_tau, head_retrieval, dataset_to_model_hnsw_recall,
)
from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import apply_similar_to_mode  # noqa: E402

ARTIFACTS = os.path.join(_HERE, "artifacts", "ablation")

# B0 baseline config (current production candidate: 1-layer, top_frac 0.10,
# rank+contrast both 1, dot scorer, hinge ranking, shared head, edge-attr ignored).
B0 = dict(
    num_layers=1, top_frac=0.10, lambda_rank=1.0, lambda_contrast=1.0,
    lambda_mse=0.0, lambda_uniform=0.0, scorer="dot",
    # Phase 1 graph-construction knobs (default = the built dense graph = B0)
    similar_to_mode="dense", similar_to_k=10,
)


def _ds_kwargs(xd0):
    if not xd0:
        return {}
    return dict(num_task_types=xd0["num_task_types"],
                n_class_buckets=xd0["n_class_buckets"], num_arities=xd0["num_arities"])


def participation_ratio(Z):
    Z = np.asarray(Z, dtype=float)
    Z = Z - Z.mean(0, keepdims=True)
    ev = np.linalg.eigvalsh(Z.T @ Z)
    ev = ev[ev > 1e-12]
    return float((ev.sum() ** 2) / (ev ** 2).sum()) if ev.size else float("nan")


def train_eval_one(data, xm0, xd0, cfg, split, *, init_seed, epochs, device="cpu"):
    """Train one config on a FIXED split, evaluate with the shared harness.

    split : (train_data, val_data, test_data) from make_fixed_splits.
    Returns (agg_row, per_dataset_tau, per_dataset_head).
    """
    train_data, _val, test_data = split
    device = torch.device(device)
    lookup = accuracy_lookup(data)

    # supervision + TRAIN-VISIBLE membership (no test leakage into geometry)
    eli, target = perf_supervision(train_data[TRAINED_ON], lookup)
    ti = torch.cat([train_data[TRAINED_ON].edge_index, eli], dim=1)
    ta = torch.cat([train_data[TRAINED_ON].edge_attr.float(), target], dim=0)
    M = topk_membership(data, top_frac=cfg["top_frac"], trained_on_index=ti, trained_on_attr=ta)
    if cfg.get("root_pool_positives") and cfg.get("root_ids") is not None:
        # v6 S1: pool the global-term positive sets within each root (train-visible
        # only; test roots stay all-zero). z_d learns root-invariant preference.
        from ModelLakeFishing.stage2TrainGraphSAGE.losses import pool_membership_by_root
        root_ids = torch.as_tensor(cfg["root_ids"], dtype=torch.long)
        M = pool_membership_by_root(M, root_ids)
    comp = lineage_components(data, data["model"].num_nodes)

    # init_seed is separate from the split: only the model/scorer init + training
    # RNG depend on it; the test edges are already fixed by split_seed.
    torch.manual_seed(init_seed); np.random.seed(init_seed)
    # D1 §5.3 feature-variant kwargs (one change per row; absent keys = legacy)
    fkw = {}
    if cfg.get("drop_desc"):
        fkw.update(use_desc=False, name_dim=xm0["name_dim"])
    if cfg.get("drop_family"):
        fkw["use_family"] = False
    if cfg.get("name_proj_dim"):
        fkw.update(name_proj_dim=cfg["name_proj_dim"], name_dim=xm0["name_dim"])
    if cfg.get("use_model_task"):
        assert "num_model_tasks" in xm0 and getattr(data["model"], "task_id", None) is not None, (
            "use_model_task needs attach_model_task_ids() run on this graph "
            "(see d1_features.py) and xm0['num_model_tasks'] set")
        fkw["num_model_tasks"] = xm0["num_model_tasks"]
    if cfg.get("dataset_frozen_proj_dim"):
        # v3 Z1: learnable projection of the frozen xd0 views inside the encoder
        fkw["dataset_frozen_proj_dim"] = cfg["dataset_frozen_proj_dim"]
    model = HeteroGraphSAGE(
        metadata=data.metadata(), frozen_dim=data["model"].x.shape[1],
        num_size_buckets=xm0["num_size_buckets"], num_families=xm0["num_families"],
        dataset_in_dim=data["dataset"].x.shape[1], num_layers=cfg["num_layers"],
        edge_aware=cfg.get("edge_aware", False),
        weighted_relations=cfg.get("weighted_relations", None),
        separate_heads=cfg.get("separate_heads", False),
        **fkw, **_ds_kwargs(xd0)).to(device)
    scorer = PerfScorer(dim=128, mode=cfg["scorer"]).to(device)

    grouped = cfg.get("grouped", False)
    common = dict(epochs=epochs, lr=cfg.get("lr", 1e-2),
                  batch_size=cfg.get("batch_size", 128),
                  lambda_mse=cfg["lambda_mse"], lambda_rank=cfg["lambda_rank"],
                  lambda_contrast=cfg["lambda_contrast"], lambda_uniform=cfg["lambda_uniform"],
                  rank_loss=cfg.get("rank_loss", "hinge"),
                  rank_temperature=cfg.get("rank_temperature", 0.1),
                  rank_min_gap=cfg.get("rank_min_gap", 0.0),
                  rank_gap_weighted=cfg.get("rank_gap_weighted", False), device=device)
    if grouped:
        # the dataset->model contrastive (Phase 6) needs the train-visible ti + full M
        common.update(ti=ti, lambda_dm_contrast=cfg.get("lambda_dm_contrast", 0.0),
                      dm_temperature=cfg.get("dm_temperature", 0.1),
                      dm_hard_neg_weight=cfg.get("dm_hard_neg_weight", 1.0),
                      dm_warmup=cfg.get("dm_warmup", 0))
        hist, _ = train_grouped(model, scorer, train_data, eli, target, M.to(device),
                                comp.to(device), **common)
    else:
        if cfg.get("early_stop", False):
            common.update(val_data=_val, val_lookup=lookup,
                          patience=cfg.get("patience", 0), eval_every=cfg.get("eval_every", 1))
        if cfg.get("lambda_dm_contrast", 0.0) > 0:
            common.update(lambda_dm_contrast=cfg["lambda_dm_contrast"],
                          dm_temperature=cfg.get("dm_temperature", 0.1),
                          dm_top_frac=cfg.get("top_frac", 0.10))
        if cfg.get("lambda_global", 0.0) > 0:
            gctx = dict(M=M, lambda_g=cfg["lambda_global"],
                        temperature=cfg.get("global_temperature", 0.1),
                        n_neg=cfg.get("global_n_neg", 64),
                        n_datasets=cfg.get("global_n_datasets", 16))
            if cfg.get("global_mode", "pools") == "lake":
                # v3 L1: whole-lake logQ-corrected sampled softmax (+L2 mining)
                from ModelLakeFishing.stage2TrainGraphSAGE.losses import build_lake_logq
                q, logq = build_lake_logq(
                    ti, data["model"].num_nodes,
                    alpha=cfg.get("lake_alpha", 0.75), n0=cfg.get("lake_n0", 1.0))
                deg = torch.bincount(ti[0], minlength=data["model"].num_nodes)
                print(f"    [lake logQ] models={q.numel()} labeled={(deg > 0).sum().item()} "
                      f"max_deg={int(deg.max())} q_head={q.max():.4f} "
                      f"alpha={cfg.get('lake_alpha', 0.75)}")
                gctx.update(lake=(q, logq),
                            hard_mine_epoch=cfg.get("hard_mine_epoch", 0),
                            hard_k=cfg.get("hard_k", 20),
                            n_hard=cfg.get("n_hard", 16))
                if cfg.get("pos_ipw_beta", 0.0) > 0:
                    # v4 L4: inverse-propensity positive weights (logQ's dual)
                    w = (deg.float() + 1.0) ** (-cfg["pos_ipw_beta"])
                    gctx["pos_ipw"] = w
                    print(f"    [pos IPW] beta={cfg['pos_ipw_beta']} "
                          f"w_range=[{w.min():.3f},{w.max():.3f}]")
                if cfg.get("hard_alibi"):
                    # v4 L2b: alibi context for the one-shot miner
                    from ModelLakeFishing.stage2TrainGraphSAGE.losses import build_model_task_profiles
                    gctx["alibi"] = dict(
                        deg=deg,
                        dataset_task_id=data["dataset"].task_type_id,
                        model_tasks=build_model_task_profiles(
                            ti, data["dataset"].task_type_id),
                        deg_quantile=cfg.get("alibi_deg_quantile", 0.9))
            else:
                # Top-1 guide Phase 1: reliable global negatives (train-visible only)
                from ModelLakeFishing.stage2TrainGraphSAGE.losses import build_global_negative_pools
                pools, pool_stats = build_global_negative_pools(
                    data["dataset"].task_type_id, ti, ta, M,
                    include_known_low=cfg.get("global_known_low", False),
                    low_frac=cfg.get("global_low_frac", 0.3))
                print(f"    [global negs] {pool_stats}")
                gctx.update(pools=pools, hard_frac=cfg.get("global_hard_frac", 0.0))
            common.update(global_ctx=gctx)
        if cfg.get("lambda_zpush", 0.0) > 0:
            # v3 Z2: dataset-dataset push-apart (uses train_data task_type_id,
            # repaired upstream when cfg['repair_dataset_task'] is set)
            known = int((data["dataset"].task_type_id > 0).sum())
            print(f"    [zpush] known-task datasets={known} "
                  f"margin={cfg.get('zpush_margin', 0.2)}")
            common.update(zpush_ctx=dict(
                lambda_zp=cfg["lambda_zpush"],
                margin=cfg.get("zpush_margin", 0.2),
                n_anchor=cfg.get("zpush_n_anchor", 32),
                n_neg=cfg.get("zpush_n_neg", 16)))
        hist, _ = train(model, scorer, train_data, eli, target, M.to(device), comp.to(device),
                        **common)

    # ── evaluate on the FIXED test split (z computed on the test message graph) ──
    model.eval()
    with torch.no_grad():
        z_test = {k: v.cpu() for k, v in model(test_data.clone().to(device)).items()}
        z_full = model(data.clone().to(device))["model"].cpu()
    scorer_cpu = scorer.cpu()

    macro_tau, per_tau = per_dataset_tau(scorer_cpu, z_test, test_data, lookup)
    head_macro, per_head = head_retrieval(z_test, test_data, lookup)
    dm_recall = dataset_to_model_hnsw_recall(z_test, k=50)

    row = {
        "tau_macro": macro_tau,
        "n_datasets_scored": len(per_tau),
        "mean_cos": collapse_report(z_full),
        "z_m_pr": participation_ratio(z_full.numpy()),
        "z_d_pr": participation_ratio(z_test["dataset"].numpy()),
        "density": global_positive_density(M),
        "pd_density": per_dataset_density(ti, M),
        "rank_drop": hist[0].get("rank", 0.0) - hist[-1].get("rank", 0.0),
        "contrast_drop": hist[0].get("contrast", 0.0) - hist[-1].get("contrast", 0.0),
        "head": head_macro,
        "dm_hnsw_recall@50": dm_recall,
    }
    return row, per_tau, per_head, model, scorer


def run(graph_path, cfg, *, name, split_seeds, init_seeds, epochs, device="cpu",
        out_dir=ARTIFACTS, save_ckpt=None):
    data, xm0, _umi = load_hgraph(graph_path)
    xd0 = torch.load(graph_path, map_location="cpu", weights_only=False).get("xd0_meta")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Phase 1 graph surgery on the dataset `similar_to` relation (message structure
    # only; trained_on supervision + lineage untouched). Default 'dense' = B0.
    mode = cfg.get("similar_to_mode", "dense")
    SIM = ("dataset", "similar_to", "dataset")
    before = data[SIM].edge_index.size(1)
    data = apply_similar_to_mode(data, mode, k=cfg.get("similar_to_k", 10))
    after = data[SIM].edge_index.size(1)
    print(f"[{name}] similar_to mode={mode} (k={cfg.get('similar_to_k', 10)}): "
          f"{before} -> {after} edges")

    print(f"[{name}] graph {os.path.basename(graph_path)}: {data['model'].num_nodes} models, "
          f"{data['dataset'].num_nodes} datasets, families {xm0['num_families']}")
    print(f"[{name}] config {cfg}")
    print(f"[{name}] split_seeds {split_seeds}, init_seeds {init_seeds}, {epochs} epochs, device {device}\n")

    rows, per_tau_all, per_head_all = [], {}, {}
    ckpt_status = "not saved"
    for ss in split_seeds:
        split = make_fixed_splits(data, split_seed=ss)
        for isd in init_seeds:
            row, per_tau, per_head, model, _scorer = train_eval_one(
                data, xm0, xd0, cfg, split, init_seed=isd, epochs=epochs, device=device)
            # save the accepted candidate as a NEW artifact (never overwrite the
            # production checkpoint) + verify checkpoint roundtrip (guide test #10)
            if save_ckpt and ss == split_seeds[0] and isd == init_seeds[0]:
                tt_vocab = xd0["task_type_vocab"] if xd0 else None
                save_checkpoint(model.to("cpu"), xm0["family_vocab"], save_ckpt, task_type_vocab=tt_vocab)
                _, _, test_data = split
                with torch.no_grad():
                    z_ref = model(test_data.clone())["model"].clone()
                m2, vocab2, _r = load_checkpoint(save_ckpt)
                with torch.no_grad():
                    z_re = m2(test_data.clone())["model"]
                ok = bool(torch.allclose(z_ref, z_re, atol=1e-6)) and vocab2 == xm0["family_vocab"]
                ckpt_status = f"{'OK' if ok else 'FAILED'} (roundtrip {'matches' if ok else 'MISMATCH'})"
                print(f"  [ckpt] saved {os.path.basename(save_ckpt)} -> {ckpt_status}")
                model.to(device)
            tag = f"s{ss}_i{isd}"
            rows.append({"split_seed": ss, "init_seed": isd, **{k: v for k, v in row.items() if k != "head"}})
            rows[-1]["head"] = row["head"]
            per_tau_all[tag] = per_tau
            per_head_all[tag] = per_head
            print(f"  [{tag}] tau_macro={row['tau_macro']:.4f} (n={row['n_datasets_scored']}) "
                  f"mean_cos={row['mean_cos']:.4f} z_m_pr={row['z_m_pr']:.2f} "
                  f"hit@10={row['head'].get('hit@10', float('nan')):.3f} "
                  f"ndcg@50={row['head'].get('ndcg@50', float('nan')):.3f}")

    # aggregate
    def agg(key):
        vals = [r[key] for r in rows if not (isinstance(r[key], float) and np.isnan(r[key]))]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))

    scalar_keys = ["tau_macro", "mean_cos", "z_m_pr", "z_d_pr", "density", "pd_density",
                   "rank_drop", "contrast_drop"]
    aggregate = {k: agg(k) for k in scalar_keys}
    head_keys = sorted({k for r in rows for k in r["head"]})
    head_agg = {k: (float(np.mean([r["head"][k] for r in rows if k in r["head"]])),
                    float(np.std([r["head"][k] for r in rows if k in r["head"]]))) for k in head_keys}

    os.makedirs(out_dir, exist_ok=True)
    artifact = {
        "name": name, "graph": os.path.basename(graph_path), "config": cfg,
        "split_seeds": list(split_seeds), "init_seeds": list(init_seeds), "epochs": epochs,
        "runs": rows, "aggregate": aggregate, "head_aggregate": head_agg,
        "per_dataset_tau": per_tau_all, "per_dataset_head": per_head_all,
    }
    out_path = os.path.join(out_dir, f"{name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    print("\n" + "=" * 64)
    for k in scalar_keys:
        print(f"  {k:<16} {aggregate[k][0]:>8.4f} +/- {aggregate[k][1]:.4f}")
    print("  --- exact z_d->z_m head retrieval (macro over held-out candidates) ---")
    for k in ["hit@10", "recall_top3@10", "recall_top10pct@50", "ndcg@10", "ndcg@50", "regret@10"]:
        if k in head_agg:
            print(f"  {k:<16} {head_agg[k][0]:>8.4f} +/- {head_agg[k][1]:.4f}")
    print("=" * 64)
    print(f"saved: {out_path}")
    return artifact


def main():
    ap = argparse.ArgumentParser(description="Stage-2 ablation driver (fixed-split, paired-comparable)")
    ap.add_argument("--pt", default=os.path.normpath(os.path.join(
        _HERE, "..", "stage1BuildTransferGraph", "hgraph_hf1000d_2000m_xm0_xd0.pt")))
    ap.add_argument("--name", default="B0")
    ap.add_argument("--seeds", type=int, default=3, help="number of split seeds (0..N-1)")
    ap.add_argument("--init_seeds", type=int, default=1, help="init seeds per split (0..N-1)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    # config overrides (default = B0)
    ap.add_argument("--num_layers", type=int, default=B0["num_layers"])
    ap.add_argument("--top_frac", type=float, default=B0["top_frac"])
    ap.add_argument("--lambda_rank", type=float, default=B0["lambda_rank"])
    ap.add_argument("--lambda_contrast", type=float, default=B0["lambda_contrast"])
    ap.add_argument("--lambda_mse", type=float, default=B0["lambda_mse"])
    ap.add_argument("--lambda_uniform", type=float, default=B0["lambda_uniform"])
    ap.add_argument("--scorer", default=B0["scorer"], choices=["dot", "mlp"])
    ap.add_argument("--similar_to_mode", default=B0["similar_to_mode"],
                    choices=["dense", "none", "topk", "topk_unweighted"])
    ap.add_argument("--similar_to_k", type=int, default=B0["similar_to_k"])
    ap.add_argument("--edge_aware", action="store_true")
    ap.add_argument("--rank_loss", default="hinge", choices=["hinge", "ranknet"])
    ap.add_argument("--rank_temperature", type=float, default=0.1)
    ap.add_argument("--rank_min_gap", type=float, default=0.0)
    ap.add_argument("--rank_gap_weighted", action="store_true")
    ap.add_argument("--separate_heads", action="store_true")
    ap.add_argument("--grouped", action="store_true",
                    help="Phase 3 dataset-grouped macro-balanced full-batch training")
    ap.add_argument("--lambda_dm_contrast", type=float, default=0.0,
                    help="Phase 6 dataset->model contrastive weight (needs --grouped)")
    ap.add_argument("--dm_temperature", type=float, default=0.1)
    ap.add_argument("--dm_hard_neg_weight", type=float, default=1.0)
    ap.add_argument("--dm_warmup", type=int, default=0, help="epochs to anneal dm-contrast from 0")
    ap.add_argument("--early_stop", action="store_true",
                    help="Phase 7 select best-val-tau epoch (non-grouped path)")
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--save_ckpt", default=None,
                    help="path to save the accepted candidate checkpoint (new artifact)")
    ap.add_argument("--weighted_relations", default=None,
                    help="comma-separated relation middle-names to weight (e.g. "
                         "'similar_to,trained_on,rev_trained_on'); omit=all; "
                         "'none'=weight nothing (edge-aware architecture control)")
    args = ap.parse_args()

    if args.weighted_relations is None:
        wr = None                                   # all relations weighted
    elif args.weighted_relations.strip().lower() == "none":
        wr = []                                     # weight nothing (control)
    else:
        wr = [r for r in args.weighted_relations.split(",") if r]
    cfg = dict(num_layers=args.num_layers, top_frac=args.top_frac, lr=args.lr,
               lambda_rank=args.lambda_rank, lambda_contrast=args.lambda_contrast,
               lambda_mse=args.lambda_mse, lambda_uniform=args.lambda_uniform,
               scorer=args.scorer, similar_to_mode=args.similar_to_mode,
               similar_to_k=args.similar_to_k, edge_aware=args.edge_aware,
               weighted_relations=wr, rank_loss=args.rank_loss,
               rank_temperature=args.rank_temperature, rank_min_gap=args.rank_min_gap,
               rank_gap_weighted=args.rank_gap_weighted, separate_heads=args.separate_heads,
               grouped=args.grouped, lambda_dm_contrast=args.lambda_dm_contrast,
               dm_temperature=args.dm_temperature, dm_hard_neg_weight=args.dm_hard_neg_weight,
               dm_warmup=args.dm_warmup, early_stop=args.early_stop, patience=args.patience)
    run(args.pt, cfg, name=args.name, split_seeds=tuple(range(args.seeds)),
        init_seeds=tuple(range(args.init_seeds)), epochs=args.epochs, device=args.device,
        save_ckpt=args.save_ckpt)


if __name__ == "__main__":
    main()
