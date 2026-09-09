"""
phase8_smoke.py -- Effective-Dataset guide, Phase 8 (smoke + representative warm A/B).

Proves the v2 pipeline end-to-end: load each Stage-1 graph, warm edge-holdout split,
train representative configs, evaluate with the shared five-metric harness, and print
A (source-corrected current) vs B (edge-first v2) side by side with the random baseline.

This is the guide's "short smoke test ... then representative models" rung, NOT the full
25/40-config × 5-fold × 3-seed sweep (that is the remaining compute-bound follow-up).

Run:
  python -m ModelLakeFishing.stage2TrainGraphSAGE.phase8_smoke --epochs 20
"""

import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.model import load_hgraph
from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import dedup_trained_on, apply_similar_to_mode
from ModelLakeFishing.stage2TrainGraphSAGE.eval_harness import make_fixed_splits
from ModelLakeFishing.stage2TrainGraphSAGE.ablation import train_eval_one
from ModelLakeFishing.stage2TrainGraphSAGE.losses import accuracy_lookup, TRAINED_ON
from ModelLakeFishing.stage2TrainGraphSAGE.top1_audit import configs, model_names, candidates
from ModelLakeFishing.stage2TrainGraphSAGE.top1_eval import (
    five_metric_eval, aggregate, random_baselines)

S1 = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage1BuildTransferGraph")
GRAPHS = {
    "A_ctrl": os.path.join(S1, "hgraph_A_ctrl_2000m_xm0_xd0.pt"),
    "B_v2": os.path.join(S1, "hgraph_hf_effective_2000m_v2_xm0_xd0.pt"),
}
OUT = os.path.join(_HERE, "artifacts", "effective_dataset_v2")


def run(graph_path, cfgs, epochs, split_seed=0, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data, xm0, umi = load_hgraph(graph_path)
    names = model_names(umi)
    data = dedup_trained_on(data)
    lookup = accuracy_lookup(data)
    out = {}
    rand = None
    for name, cfg in cfgs.items():
        d2 = apply_similar_to_mode(data.clone(), cfg["similar_to_mode"], k=cfg["similar_to_k"])
        split = make_fixed_splits(d2, split_seed=split_seed)
        _tr, _val, test_data = split
        _row, _pt, _ph, model, _sc = train_eval_one(
            d2, xm0, None, cfg, split, init_seed=0, epochs=epochs, device=device)
        model.eval()
        with torch.no_grad():
            z = {k: v.cpu() for k, v in model(test_data.clone().to(device)).items()}
        cands = candidates(test_data, lookup)
        agg = aggregate(five_metric_eval(z, cands, names=names))
        if rand is None:
            rand = random_baselines(cands, n_pool=data["model"].num_nodes)
        out[name] = agg
        print(f"  [{name}] n={agg['n_datasets']} hit1={agg['observed_hit1']:.3f} "
              f"top3={agg['top3_hit1']:.3f} regret={agg['regret1']:.4f} "
              f"gold@10={agg['full2k_gold@10']:.3f}")
    return {"aggregates": out, "random": rand,
            "n_dataset_nodes": int(data["dataset"].num_nodes),
            "trained_on_edges": int(data[TRAINED_ON].edge_index.size(1))}


def main():
    epochs = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 20
    cfgs = {k: configs()[k] for k in ("B0", "P6_dm10")}
    report = {"epochs": epochs, "split_seed": 0, "configs": list(cfgs)}
    for tag, path in GRAPHS.items():
        if not os.path.exists(path):
            print(f"SKIP {tag}: {path} missing"); continue
        print(f"=== {tag} ({os.path.basename(path)}) ===")
        report[tag] = run(path, cfgs, epochs)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "phase8_smoke_warm.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nwrote", os.path.relpath(os.path.join(OUT, "phase8_smoke_warm.json"), _REPO_ROOT))


if __name__ == "__main__":
    main()
