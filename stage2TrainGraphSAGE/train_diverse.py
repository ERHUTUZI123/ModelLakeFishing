"""
train_diverse.py -- Phase 4: train the Stage-2 candidate on the DIVERSE graph.

A thin, fully-parameterized driver around the existing Stage-2 modules. It does
NOT replace experiment.py (which hardcodes load_hgraph(), the stage2_candidate.pt
output, and an assert num_families==136 that only fits the original zoo). Here:

  * the graph path, output checkpoint, seeds and epochs are CLI args;
  * the pilot config defaults to experiment.CANDIDATE (the single source of truth:
    1-layer, top_frac=0.10, lambda_contrast=1, lambda_rank=1, lambda_mse=0,
    lambda_uniform=0, dot scorer, batch contrastive masks only) but EVERY
    hyperparameter is overridable -- nothing is hardcoded as a universal default;
  * num_families is whatever the diverse xm0 produced (no ==136 assert);
  * a checkpoint save/load roundtrip verifies the diverse 1-layer config reloads
    to the same z and the (now larger) family vocab binding still holds.

Run (venv python):
  python -m ModelLakeFishing.stage2TrainGraphSAGE.train_diverse \
      --pt .../stage1BuildTransferGraph/hgraph_diverse_xm0.pt \
      --out artifacts/stage2_diverse_candidate.pt --seeds 3 --epochs 25
"""

import argparse
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
    TRAINED_ON, PerfScorer, accuracy_lookup, perf_supervision, split_trained_on,
    topk_membership, lineage_components, global_positive_density, per_dataset_density,
)
from ModelLakeFishing.stage2TrainGraphSAGE.train import train, eval_perf, collapse_report  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.learnable import save_checkpoint, load_checkpoint  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.experiment import CANDIDATE  # noqa: E402

ARTIFACTS = os.path.join(_HERE, "artifacts")
METRICS = ("density", "pd_density", "contrast_drop", "rank_drop", "mean_cos", "tau_pool", "tau_macro")


def _ds_kwargs(xd0):
    """Dataset-encoder kwargs from xd0_meta (empty -> plain dataset_proj)."""
    if not xd0:
        return {}
    return dict(num_task_types=xd0["num_task_types"],
                n_class_buckets=xd0["n_class_buckets"],
                num_arities=xd0["num_arities"])


def run(data, xm0, cfg, seeds, epochs, out_ckpt, xd0=None, device="cpu"):
    rows, ckpt_status = [], "not run"
    ds_kw = _ds_kwargs(xd0)
    device = torch.device(device)
    for si, seed in enumerate(seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        train_data, _val, test_data = split_trained_on(data, seed=seed)
        lookup = accuracy_lookup(data)
        eli, target = perf_supervision(train_data[TRAINED_ON], lookup)
        ti = torch.cat([train_data[TRAINED_ON].edge_index, eli], dim=1)
        ta = torch.cat([train_data[TRAINED_ON].edge_attr.float(), target], dim=0)
        M = topk_membership(data, top_frac=cfg["top_frac"], trained_on_index=ti, trained_on_attr=ta)
        comp = lineage_components(data, data["model"].num_nodes)

        torch.manual_seed(seed); np.random.seed(seed)
        model = HeteroGraphSAGE(
            metadata=data.metadata(), frozen_dim=data["model"].x.shape[1],
            num_size_buckets=xm0["num_size_buckets"], num_families=xm0["num_families"],
            dataset_in_dim=data["dataset"].x.shape[1], num_layers=cfg["num_layers"],
            **ds_kw).to(device)
        scorer = PerfScorer(dim=128, mode=cfg["scorer"]).to(device)
        # train on `device`: loader samples on CPU, batches move per-step; masks on device
        hist, _ = train(model, scorer, train_data, eli, target, M.to(device), comp.to(device),
                        epochs=epochs, lambda_mse=cfg["lambda_mse"], lambda_rank=cfg["lambda_rank"],
                        lambda_contrast=cfg["lambda_contrast"], lambda_uniform=cfg["lambda_uniform"],
                        device=device)
        model.eval()
        with torch.no_grad():
            z_full = model(data.clone().to(device))["model"].cpu()
        tm = eval_perf(model, scorer, test_data, lookup)
        rows.append({
            "density": global_positive_density(M),
            "pd_density": per_dataset_density(ti, M),
            "contrast_drop": hist[0]["contrast"] - hist[-1]["contrast"],
            "rank_drop": hist[0]["rank"] - hist[-1]["rank"],
            "mean_cos": collapse_report(z_full),
            "tau_pool": tm["kendall_tau"],
            "tau_macro": tm["kendall_tau_macro"],
        })
        print(f"  seed {seed}: mean_cos={rows[-1]['mean_cos']:.4f} "
              f"tau_macro={rows[-1]['tau_macro']:.4f} contrast_drop={rows[-1]['contrast_drop']:.4f}")
        if si == 0:                                    # checkpoint roundtrip + SAVE the deliverable
            tt_vocab = xd0["task_type_vocab"] if xd0 else None
            save_checkpoint(model, xm0["family_vocab"], out_ckpt, task_type_vocab=tt_vocab)
            with torch.no_grad():
                z_ref = model(test_data.clone().to(device))["model"].clone().cpu()
            model2, vocab2, repro2 = load_checkpoint(out_ckpt)   # reloads on CPU
            with torch.no_grad():
                z_re = model2(test_data)["model"]
            ok = bool(torch.allclose(z_ref, z_re, atol=1e-6)) and vocab2 == xm0["family_vocab"]
            if tt_vocab is not None:
                ok = ok and (repro2.get("task_type_vocab") == tt_vocab) and model2.use_dataset_encoder
            ds_tag = (f", dataset_encoder=ON num_task_types={xd0['num_task_types']}"
                      if xd0 else ", dataset_encoder=OFF")
            ckpt_status = (f"OK (num_layers={model2.num_layers}, num_families={xm0['num_families']}"
                           f"{ds_tag}, z matches, vocab matches)") if ok else "FAILED (z or vocab mismatch)"
    return rows, ckpt_status


def main():
    ap = argparse.ArgumentParser(description="Train Stage-2 candidate on the diverse graph")
    # PRODUCTION DEFAULT: diverse zoo + xm0 + xd0 (see artifacts/PRODUCTION.md).
    # Pass --pt/--out to reproduce legacy baselines (e.g. hgraph_diverse_xm0.pt).
    ap.add_argument("--pt", default=os.path.normpath(os.path.join(
        _HERE, "..", "stage1BuildTransferGraph", "hgraph_diverse_xd0.pt")))
    ap.add_argument("--out", default=os.path.join(ARTIFACTS, "stage2_diverse_xd0_candidate.pt"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=25)
    # candidate config -- explicit & overridable (defaults = experiment.CANDIDATE)
    ap.add_argument("--num_layers", type=int, default=CANDIDATE["num_layers"])
    ap.add_argument("--top_frac", type=float, default=CANDIDATE["top_frac"])
    ap.add_argument("--lambda_contrast", type=float, default=CANDIDATE["lambda_contrast"])
    ap.add_argument("--lambda_rank", type=float, default=CANDIDATE["lambda_rank"])
    ap.add_argument("--lambda_mse", type=float, default=CANDIDATE["lambda_mse"])
    ap.add_argument("--lambda_uniform", type=float, default=CANDIDATE["lambda_uniform"])
    ap.add_argument("--scorer", default="dot", choices=["dot", "mlp"])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="auto -> cuda if available else cpu")
    args = ap.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = dict(num_layers=args.num_layers, top_frac=args.top_frac,
               lambda_contrast=args.lambda_contrast, lambda_rank=args.lambda_rank,
               lambda_mse=args.lambda_mse, lambda_uniform=args.lambda_uniform,
               scorer=args.scorer)

    data, xm0, _umi = load_hgraph(args.pt)
    xd0 = torch.load(args.pt, map_location="cpu", weights_only=False).get("xd0_meta")
    assert data["model"].x.shape[1] == 448, "expected real xm0 frozen dim 448 (e_name||e_desc)"
    assert bool((data["model"].x.min() < 0).item()), "model.x looks like smoke random, not real xm0"
    if xd0 is not None:
        # using the xd0 graph -> DatasetNodeEncoder must engage on the discrete cols
        for c in ("task_type_id", "n_class_bucket_id", "arity_id"):
            assert hasattr(data["dataset"], c), f"xd0_meta present but dataset.{c} missing"
    print(f"graph {os.path.basename(args.pt)}: {data['model'].num_nodes} models, "
          f"{data['dataset'].num_nodes} datasets, model.x {tuple(data['model'].x.shape)}, "
          f"dataset.x {tuple(data['dataset'].x.shape)}, families {xm0['num_families']}")
    print(f"dataset side: {'xd0 DatasetNodeEncoder (num_task_types=%d)' % xd0['num_task_types'] if xd0 else 'plain Linear (no xd0)'}")
    print(f"candidate config: {cfg}")
    print(f"device: {args.device}")
    print(f"seeds {tuple(range(args.seeds))}, {args.epochs} epochs\n")

    rows, ckpt_status = run(data, xm0, cfg, tuple(range(args.seeds)), args.epochs, args.out,
                            xd0=xd0, device=args.device)
    agg = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))) for k in METRICS}
    print("\n" + "=" * 60)
    for k in METRICS:
        print(f"  {k:<16} {agg[k][0]:>8.4f} +/- {agg[k][1]:.4f}")
    print(f"  {'checkpoint':<16} {ckpt_status}")
    print("=" * 60)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
