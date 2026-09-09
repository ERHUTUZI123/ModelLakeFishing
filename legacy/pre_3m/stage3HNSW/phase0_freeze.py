"""
phase0_freeze.py -- Stage-3 Phase 0: put the serving checkpoints on disk.

The Top-1 G-phase / T0 runs (top1_baselines.run_configs) recorded per-split
state_dict sha256 + five-metric aggregates but never persisted weights. This
script re-runs the recorded (config, split_seed=0, init_seed=0) training via
the SAME code path (dedup_trained_on -> apply_similar_to_mode ->
make_fixed_splits -> ablation.train_eval_one), then

  1. compares state_dict sha256 with the recorded value (exact bit
     reproduction; may fail under CUDA nondeterminism -- reported, not fatal),
  2. re-runs the five-metric evaluation on split 0 and compares every
     aggregate key with the recorded numbers (the decisive check),
  3. saves the checkpoint bound to family/task_type vocabs
     (learnable.save_checkpoint) and verifies the save/load roundtrip.

Checkpoints land in  stage2TrainGraphSAGE/artifacts/ablation/top1/ckpt/
(they are Stage-2 artifacts; Stage 3 only consumes exports built from them).
A machine-readable report lands in  stage3HNSW/artifacts/phase0_freeze_report.json.

Run:  python -m ModelLakeFishing.stage3HNSW.phase0_freeze [--rows G2 G1 P6_dm10]
"""

import argparse
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.model import load_hgraph  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.losses import accuracy_lookup  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.eval_harness import make_fixed_splits  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import (  # noqa: E402
    apply_similar_to_mode, dedup_trained_on,
)
from ModelLakeFishing.stage2TrainGraphSAGE.ablation import train_eval_one  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.learnable import save_checkpoint, load_checkpoint  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.top1_audit import (  # noqa: E402
    GRAPH, EPOCHS, sha256_file, state_dict_sha256, model_names, candidates,
)
from ModelLakeFishing.stage2TrainGraphSAGE.top1_eval import five_metric_eval, aggregate  # noqa: E402

TOP1 = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage2TrainGraphSAGE",
                    "artifacts", "ablation", "top1")
# name -> the run_configs record holding its config + recorded split-0 facts
RECORDS = {
    "G2": os.path.join(TOP1, "G", "G2.json"),
    "G1": os.path.join(TOP1, "G", "G1.json"),
    "P6_dm10": os.path.join(TOP1, "T0", "P6_dm10.json"),
}
CKPT_DIR = os.path.join(TOP1, "ckpt")
SPLIT_SEED = 0  # canonical serving split: first of the recorded (0, 1, 2)


def freeze_one(name, device):
    with open(RECORDS[name], encoding="utf-8") as f:
        rec = json.load(f)
    assert rec.get("dedup_trained_on") is True, f"{name}: record is not deduped-protocol"
    cfg = rec["config"]
    init_seed = rec["init_seed"]
    recorded = rec["splits"][str(SPLIT_SEED)]

    graph_sha = sha256_file(GRAPH)
    assert graph_sha == rec["graph_sha256"], (
        f"{name}: graph on disk ({graph_sha[:16]}...) != recorded "
        f"({rec['graph_sha256'][:16]}...) -- the input graph changed, do not retrain blindly")

    # identical prep to top1_baselines.run_configs
    xd0_full = torch.load(GRAPH, map_location="cpu", weights_only=False).get("xd0_meta")
    data, xm0, umi = load_hgraph(GRAPH)
    names = model_names(umi)
    data = dedup_trained_on(data)
    data = apply_similar_to_mode(data, cfg["similar_to_mode"], k=cfg["similar_to_k"])
    split = make_fixed_splits(data, split_seed=SPLIT_SEED)
    _tr, _val, test_data = split
    lookup = accuracy_lookup(data)

    print(f"\n[{name}] retraining split_seed={SPLIT_SEED} init_seed={init_seed} "
          f"epochs={EPOCHS} device={device}")
    _row, _pt, _ph, model, _scorer = train_eval_one(
        data, xm0, xd0_full, cfg, split, init_seed=init_seed, epochs=EPOCHS, device=device)
    model.eval()

    # (1) exact-bit check, model still on the training device like run_configs
    sha = state_dict_sha256(model)
    sha_match = sha == recorded["state_dict_sha256"]

    # (2) five-metric replay on the same split
    with torch.no_grad():
        z = {k: v.cpu() for k, v in model(test_data.clone().to(device)).items()}
    agg = aggregate(five_metric_eval(z, candidates(test_data, lookup), names=names))
    deltas = {k: agg[k] - recorded["aggregate"][k]
              for k in recorded["aggregate"] if k in agg}
    max_abs_delta = max(abs(v) for v in deltas.values())

    # (3) save bound checkpoint + roundtrip
    out_path = os.path.join(CKPT_DIR, f"{name}_s{SPLIT_SEED}_i{init_seed}.pt")
    tt_vocab = xd0_full["task_type_vocab"] if xd0_full else None
    save_checkpoint(model.cpu(), xm0["family_vocab"], out_path,
                    task_type_vocab=tt_vocab,
                    extra_repro={
                        "config_name": name, "config": cfg,
                        "source_record": os.path.relpath(RECORDS[name], _REPO_ROOT),
                        "split_seed": SPLIT_SEED, "init_seed": init_seed, "epochs": EPOCHS,
                        "graph": os.path.basename(GRAPH), "graph_sha256": graph_sha,
                        "dedup_trained_on": True,
                        "recorded_state_dict_sha256": recorded["state_dict_sha256"],
                        "rerun_state_dict_sha256": sha,
                        "state_dict_sha256_match": sha_match,
                        "rerun_split0_aggregate": agg,
                        "recorded_split0_aggregate": recorded["aggregate"],
                    })
    with torch.no_grad():
        z_ref = model(test_data.clone())["model"]
    m2, vocab2, _repro = load_checkpoint(out_path)
    with torch.no_grad():
        z_re = m2(test_data.clone())["model"]
    roundtrip_ok = bool(torch.allclose(z_ref, z_re, atol=1e-6)) and vocab2 == xm0["family_vocab"]

    print(f"  sha256 match: {sha_match}  ({sha[:16]}... vs recorded "
          f"{recorded['state_dict_sha256'][:16]}...)")
    print(f"  five-metric max |delta| vs recorded split-0 aggregate: {max_abs_delta:.6g}")
    for k in ("observed_hit1", "top3_hit1", "regret1", "full2k_gold@1", "full2k_gold@10",
              "median_gold_rank"):
        print(f"    {k:<18} rerun {agg[k]:.4f}  recorded {recorded['aggregate'][k]:.4f}")
    print(f"  checkpoint: {out_path}  roundtrip_ok={roundtrip_ok}")

    return {
        "ckpt": os.path.relpath(out_path, _REPO_ROOT),
        "ckpt_sha256": sha256_file(out_path),
        "state_dict_sha256_match": sha_match,
        "rerun_state_dict_sha256": sha,
        "recorded_state_dict_sha256": recorded["state_dict_sha256"],
        "five_metric_max_abs_delta": max_abs_delta,
        "five_metric_deltas": deltas,
        "roundtrip_ok": roundtrip_ok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", default=["G2", "G1", "P6_dm10"],
                    choices=list(RECORDS))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" \
        else args.device

    os.makedirs(CKPT_DIR, exist_ok=True)
    report = {"graph": os.path.basename(GRAPH), "graph_sha256": sha256_file(GRAPH),
              "split_seed": SPLIT_SEED, "epochs": EPOCHS, "device": device,
              "torch": torch.__version__, "rows": {}}
    for name in args.rows:
        report["rows"][name] = freeze_one(name, device)

    out = os.path.join(_HERE, "artifacts", "phase0_freeze_report.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nreport: {out}")
    ok = all(r["roundtrip_ok"] for r in report["rows"].values())
    print("PHASE0 FREEZE:", "OK" if ok else "FAILED (roundtrip)")


if __name__ == "__main__":
    main()
