"""
experiment.py -- Stage 2, Step 5: scale-safe robustness matrix.

All configs use the SCALE-SAFE family (per-dataset top-fraction positives +
within-dataset margin ranking on the z_m . z_d dot score + supervised contrast,
masks built per batch from membership -- never a global N x N). We vary the axes
that should matter at 47K, NOT zoo-specific knobs, and average over seeds because
single runs on the 177-model zoo are high-variance:

  axes : num_layers {1, 2}, lambda_contrast {1, 2, 4}, top_frac {0.05, 0.10, 0.20}
  seeds: 0, 1, 2          epochs: modest (25)

Reported per config (mean +/- std over seeds): positive density, contrast-loss
drop, ranking-loss drop, mean pairwise cosine (collapse), pooled tau, tau_macro
(within-dataset -- the retrieval-relevant ranking metric).

Recommendation is chosen on reduced collapse + meaningful loss descent + NO
ranking degradation -- explicitly not on max zoo tau.

Run:  python -m ModelLakeFishing.stage2TrainGraphSAGE.experiment
"""

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
    topk_membership, lineage_components, global_positive_density,
)
from ModelLakeFishing.stage2TrainGraphSAGE.train import (  # noqa: E402
    train, eval_perf, collapse_report,
)
from ModelLakeFishing.stage2TrainGraphSAGE.learnable import (  # noqa: E402
    save_checkpoint, load_checkpoint,
)

EPOCHS = 25
SEEDS = (0, 1, 2)
ARTIFACTS = os.path.join(_HERE, "artifacts")

# ── the recommended 47K pilot candidate (explicit, single source of truth) ──────
# Conservative on the robustness matrix: 1 layer (less over-smoothing), per-dataset
# top-10% positives (healthy density, not the unstable 0.05), lambda_contrast=1
# (raising it worsened collapse), ranking is the perf objective (MSE off so it
# cannot dominate), uniformity off, dot scorer. NOT the max-zoo-tau config.
CANDIDATE = dict(num_layers=1, top_frac=0.10, lambda_contrast=1.0,
                 lambda_rank=1.0, lambda_mse=0.0, lambda_uniform=0.0)

# one-axis-at-a-time from a center (L2, c1, f0.10) -- covers every axis without a
# full 18-cell grid. name -> (num_layers, lambda_contrast, top_frac)
CONFIGS = [
    ("L1 c1 f.10", 1, 1.0, 0.10),
    ("L2 c1 f.10", 2, 1.0, 0.10),   # center / reference
    ("L1 c2 f.10", 1, 2.0, 0.10),
    ("L2 c2 f.10", 2, 2.0, 0.10),
    ("L1 c4 f.10", 1, 4.0, 0.10),
    ("L2 c4 f.10", 2, 4.0, 0.10),
    ("L2 c1 f.05", 2, 1.0, 0.05),
    ("L2 c1 f.20", 2, 1.0, 0.20),
]
REFERENCE = "L2 c1 f.10"
METRICS = ("density", "contrast_drop", "rank_drop", "mean_cos", "tau_pool", "tau_macro")


def run_one(data, xm0, num_layers, lambda_contrast, top_frac, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_data, _val, test_data = split_trained_on(data, seed=seed)
    lookup = accuracy_lookup(data)
    eli, target = perf_supervision(train_data[TRAINED_ON], lookup)
    ti = torch.cat([train_data[TRAINED_ON].edge_index, eli], dim=1)
    ta = torch.cat([train_data[TRAINED_ON].edge_attr.float(), target], dim=0)
    M = topk_membership(data, top_frac=top_frac, trained_on_index=ti, trained_on_attr=ta)
    comp = lineage_components(data, data["model"].num_nodes)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = HeteroGraphSAGE(
        metadata=data.metadata(), frozen_dim=data["model"].x.shape[1],
        num_size_buckets=xm0["num_size_buckets"], num_families=xm0["num_families"],
        dataset_in_dim=data["dataset"].x.shape[1], num_layers=num_layers)
    scorer = PerfScorer(dim=128, mode="dot")
    hist, _ = train(model, scorer, train_data, eli, target, M, comp,
                    epochs=EPOCHS, lambda_rank=1.0, lambda_contrast=lambda_contrast)
    model.eval()
    with torch.no_grad():
        z_full = model(data)["model"]
    tm = eval_perf(model, scorer, test_data, lookup)
    return {
        "density": global_positive_density(M),
        "contrast_drop": hist[0]["contrast"] - hist[-1]["contrast"],
        "rank_drop": hist[0]["rank"] - hist[-1]["rank"],
        "mean_cos": collapse_report(z_full),
        "tau_pool": tm["kendall_tau"],
        "tau_macro": tm["kendall_tau_macro"],
    }


def run_candidate(data, xm0, seeds, epochs):
    """Train ONLY the recommended candidate (CANDIDATE) over `seeds`. Also does a
    checkpoint save/load roundtrip on the first seed (verifies the 1-layer config
    reloads to the same z and the vocab binding holds). Returns (rows, ckpt_status)."""
    c = CANDIDATE
    lookup = accuracy_lookup(data)
    rows, ckpt_status = [], "not run"
    for si, seed in enumerate(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_data, _val, test_data = split_trained_on(data, seed=seed)
        eli, target = perf_supervision(train_data[TRAINED_ON], lookup)
        ti = torch.cat([train_data[TRAINED_ON].edge_index, eli], dim=1)
        ta = torch.cat([train_data[TRAINED_ON].edge_attr.float(), target], dim=0)
        M = topk_membership(data, top_frac=c["top_frac"], trained_on_index=ti, trained_on_attr=ta)
        comp = lineage_components(data, data["model"].num_nodes)

        torch.manual_seed(seed)
        np.random.seed(seed)
        model = HeteroGraphSAGE(
            metadata=data.metadata(), frozen_dim=data["model"].x.shape[1],
            num_size_buckets=xm0["num_size_buckets"], num_families=xm0["num_families"],
            dataset_in_dim=data["dataset"].x.shape[1], num_layers=c["num_layers"])
        scorer = PerfScorer(dim=128, mode="dot")
        hist, _ = train(model, scorer, train_data, eli, target, M, comp, epochs=epochs,
                        lambda_mse=c["lambda_mse"], lambda_rank=c["lambda_rank"],
                        lambda_contrast=c["lambda_contrast"], lambda_uniform=c["lambda_uniform"])
        model.eval()
        with torch.no_grad():
            z_full = model(data)["model"]
        tm = eval_perf(model, scorer, test_data, lookup)
        rows.append({
            "density": global_positive_density(M),
            "contrast_drop": hist[0]["contrast"] - hist[-1]["contrast"],
            "rank_drop": hist[0]["rank"] - hist[-1]["rank"],
            "mean_cos": collapse_report(z_full),
            "tau_pool": tm["kendall_tau"],
            "tau_macro": tm["kendall_tau_macro"],
        })
        if si == 0:                                    # checkpoint roundtrip once
            ckpt = os.path.join(ARTIFACTS, "stage2_candidate.pt")
            save_checkpoint(model, xm0["family_vocab"], ckpt)
            with torch.no_grad():
                z_ref = model(test_data)["model"].clone()
            model2, vocab2, _repro = load_checkpoint(ckpt)
            with torch.no_grad():
                z_re = model2(test_data)["model"]
            ok = bool(torch.allclose(z_ref, z_re, atol=1e-6)) and vocab2 == xm0["family_vocab"]
            ckpt_status = (f"OK (num_layers={model2.num_layers} reloaded, z matches, "
                           f"vocab matches)") if ok else "FAILED (z or vocab mismatch)"
    return rows, ckpt_status


def recommend(agg):
    """Safest config to carry to 47K. Criteria, in spirit of 'do not cherry-pick
    a zoo artifact':
      - meaningful contrast descent (mean drop > 0.05)
      - STABLE across seeds (mean_cos std < 0.15 AND contrast_drop std bounded) --
        excludes wildly seed-dependent configs like an over-sparse top_frac whose
        low collapse is one lucky seed, not robustness
      - no ranking degradation (tau_macro positive and >= reference floor)
    Among the survivors, pick the lowest collapse (mean_cos). NOT max tau."""
    ref = agg[REFERENCE]
    ref_floor = ref["tau_macro"][0] - ref["tau_macro"][1]
    cands = []
    for name, m in agg.items():
        cd_mean, cd_std = m["contrast_drop"]
        mc_mean, mc_std = m["mean_cos"]
        tau_mean, _ = m["tau_macro"]
        meaningful = cd_mean > 0.05
        stable = mc_std < 0.15 and cd_std < max(0.05, 0.6 * cd_mean)
        no_degrade = tau_mean > 0 and tau_mean >= ref_floor
        if meaningful and stable and no_degrade:
            cands.append((mc_mean, name))
    if not cands:
        return None
    cands.sort()
    return cands[0][1]


def _agg(rows):
    return {k: (float(np.mean([r[k] for r in rows])),
               float(np.std([r[k] for r in rows]))) for k in METRICS}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Stage-2 scale-safe configs")
    ap.add_argument("--candidate", action="store_true",
                    help="run ONLY the recommended 47K pilot candidate (not the full matrix)")
    ap.add_argument("--seeds", type=int, default=len(SEEDS), help="number of seeds (0..N-1)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    data, xm0, _umi = load_hgraph()
    assert xm0.get("num_families") == 136 and data["model"].x.shape[1] == 448, "expected real xm0 graph"
    seeds = tuple(range(args.seeds))

    # ── candidate-only path ──────────────────────────────────────────────────
    if args.candidate:
        print(f"CANDIDATE (47K pilot): {CANDIDATE}")
        print(f"seeds {seeds}, {args.epochs} epochs, dot scorer, batch masks only\n")
        rows, ckpt_status = run_candidate(data, xm0, seeds, args.epochs)
        m = _agg(rows)
        print("=" * 60)
        for k in METRICS:
            print(f"  {k:<16} {m[k][0]:>8.4f} +/- {m[k][1]:.4f}")
        print(f"  {'checkpoint':<16} {ckpt_status}")
        print("=" * 60)
        print("note: tau_macro (within-dataset) is the ranking metric; MSE is NOT "
              "used as a success signal. Collapse (mean_cos) still high -- see "
              "stage2_candidate_report.md for remaining risks.")
        sys.exit(0)

    # ── full robustness matrix ───────────────────────────────────────────────
    print(f"scale-safe robustness matrix: {len(CONFIGS)} configs x {len(SEEDS)} seeds, {EPOCHS} epochs\n")
    agg = {}
    for name, nl, lc, tf in CONFIGS:
        runs = [run_one(data, xm0, nl, lc, tf, s) for s in SEEDS]
        agg[name] = {k: (float(np.mean([r[k] for r in runs])),
                         float(np.std([r[k] for r in runs]))) for k in METRICS}

    print("=" * 96)
    print(f"{'config':<12}{'density':>9}{'contr_drop':>13}{'rank_drop':>12}"
          f"{'mean_cos':>14}{'tau_pool':>13}{'tau_macro':>14}")
    print("-" * 96)
    for name, _, _, _ in CONFIGS:
        m = agg[name]
        print(f"{name:<12}{m['density'][0]:>8.2f} "
              f"{m['contrast_drop'][0]:>6.3f}+/-{m['contrast_drop'][1]:<4.3f}"
              f"{m['rank_drop'][0]:>6.3f}+/-{m['rank_drop'][1]:<4.3f}"
              f"{m['mean_cos'][0]:>7.3f}+/-{m['mean_cos'][1]:<4.3f}"
              f"{m['tau_pool'][0]:>6.3f}+/-{m['tau_pool'][1]:<4.3f}"
              f"{m['tau_macro'][0]:>7.3f}+/-{m['tau_macro'][1]:<4.3f}")
    print("=" * 96)

    rec = recommend(agg)
    print(f"\nreference config: {REFERENCE}  (tau_macro {agg[REFERENCE]['tau_macro'][0]:.3f}"
          f"+/-{agg[REFERENCE]['tau_macro'][1]:.3f}, mean_cos {agg[REFERENCE]['mean_cos'][0]:.3f})")
    if rec:
        m = agg[rec]
        print(f"scale-safe recommendation: {rec}")
        print(f"  -> mean_cos {m['mean_cos'][0]:.3f} (collapse), contrast_drop {m['contrast_drop'][0]:.3f} "
              f"(meaningful), tau_macro {m['tau_macro'][0]:.3f}+/-{m['tau_macro'][1]:.3f} (no degradation vs ref)")
        print("  chosen on reduced collapse + meaningful descent + no ranking loss, NOT on max zoo tau.")
    else:
        print("no config beats the reference on collapse without ranking degradation -- keep reference.")
