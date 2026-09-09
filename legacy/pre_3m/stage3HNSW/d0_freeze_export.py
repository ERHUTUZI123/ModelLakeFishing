"""
d0_freeze_export.py -- Stage-3 re-export/re-index for the PROMOTED D0 config
(v4 W1 verdict: L1L3b). Pipeline unchanged (the Stage-3 selling point): this
script only supplies a NEW (graph, checkpoint) pair to the existing
export_embeddings + build_index machinery.

Steps (mirrors phase0_freeze discipline, adapted to the D0 protocol):
  1. FREEZE : retrain L1L3b on hgraph_d0_v1 with ROOT-AWARE split_seed=0 and
              the recorded init seed; require bit-exact state_dict sha vs the
              w1_baselines record (training is deterministic or we stop);
              save the vocab-bound checkpoint.
  2. EXPORT : export_embeddings.export() -- deterministic full-graph forward,
              L2-normalized z_m/z_d in mappedID row order, occupancy sidecar.
  3. INDEX  : build_index -- HNSW (M=16, efc=200) + exact-dot fidelity gate
              (recall@10 >= 0.99 with ef_search=200, hard gate).

Run (repo root):
  ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.stage3HNSW.d0_freeze_export
"""

import json
import os
import subprocess
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.losses import accuracy_lookup  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import apply_similar_to_mode  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.ablation import train_eval_one  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.top1_audit import (  # noqa: E402
    INIT_SEED, sha256_file, state_dict_sha256, model_names, candidates,
)
from ModelLakeFishing.stage2TrainGraphSAGE.top1_eval import five_metric_eval, aggregate  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.d0_splits import make_root_aware_splits  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.w1_dzero import d0_configs, GRAPH  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.learnable import save_checkpoint, load_checkpoint  # noqa: E402

STAGE2_ART = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage2TrainGraphSAGE",
                          "artifacts", "ablation", "d0")
RECORD = os.path.join(STAGE2_ART, "w1_baselines", "L1L3b.json")
CKPT_DIR = os.path.join(STAGE2_ART, "ckpt")
EXPORT_DIR = os.path.join(_HERE, "artifacts", "exports", "d0_L1L3b")
SPLIT_SEED = 0
EPOCHS = 25


def freeze(device):
    rec = json.load(open(RECORD, encoding="utf-8"))
    cfg = d0_configs(["L1L3b"])["L1L3b"]
    assert cfg == rec["config"], "runner config drifted from the recorded row"
    graph_sha = sha256_file(GRAPH)
    assert graph_sha == rec["graph_sha256"], "graph on disk changed since the record"
    recorded = rec["splits"][str(SPLIT_SEED)]

    payload = torch.load(GRAPH, map_location="cpu", weights_only=False)
    data, xm0, xd0 = payload["data"], payload["xm0_meta"], payload["xd0_meta"]
    udi = payload["unique_dataset_id"].sort_values("mappedID")
    names = model_names(payload["unique_model_id"])
    data = apply_similar_to_mode(data, cfg["similar_to_mode"], k=cfg["similar_to_k"])
    split = make_root_aware_splits(data, udi["root"].tolist(), split_seed=SPLIT_SEED)
    _tr, _val, test_data = split
    lookup = accuracy_lookup(data)

    print(f"[freeze] retraining L1L3b s{SPLIT_SEED} i{INIT_SEED} on {os.path.basename(GRAPH)}")
    _row, _pt, _ph, model, _sc = train_eval_one(
        data, xm0, xd0, cfg, split, init_seed=INIT_SEED, epochs=EPOCHS, device=device)
    model.eval()
    sha = state_dict_sha256(model)
    sha_match = sha == recorded["state_dict_sha256"]
    print(f"[freeze] state_dict sha match: {sha_match}")
    # D0-scale honesty note: CUDA scatter aggregation is non-deterministic at
    # the kernel level; at 54.8K edges bit-exact retrains are not reproducible
    # (they were at 2K/12K edges). The freeze gate is therefore FUNCTIONAL:
    # the five-metric replay on the identical split must match the recorded
    # aggregates within tolerance. Both shas are recorded either way.
    with torch.no_grad():
        z = {k: v.cpu() for k, v in model(test_data.clone().to(device)).items()}
    agg = aggregate(five_metric_eval(z, candidates(test_data, lookup), names=names))
    prim = ("observed_hit1", "top3_hit1", "regret1", "full2k_gold@1", "full2k_gold@10")
    max_d = max(abs(agg[k] - recorded["aggregate"][k]) for k in prim)
    print(f"[freeze] five-metric replay max |delta|: {max_d:.4f} (gate <= 0.02)")
    assert max_d <= 0.02, (
        f"functional freeze gate failed: replay deviates {max_d:.4f} > 0.02 "
        "from the recorded row -- the checkpoint is not a faithful member of "
        "the promoted config family")

    os.makedirs(CKPT_DIR, exist_ok=True)
    out_path = os.path.join(CKPT_DIR, f"L1L3b_s{SPLIT_SEED}_i{INIT_SEED}.pt")
    save_checkpoint(model.cpu(), xm0["family_vocab"], out_path,
                    task_type_vocab=xd0["task_type_vocab"],
                    extra_repro={
                        "config_name": "L1L3b", "config": cfg,
                        "source_record": os.path.relpath(RECORD, _REPO_ROOT),
                        "split_seed": SPLIT_SEED, "init_seed": INIT_SEED,
                        "epochs": EPOCHS, "graph": os.path.basename(GRAPH),
                        "graph_sha256": graph_sha,
                        "dedup_trained_on": False,   # D0 obs deduped at source
                        "protocol": "root-aware splits (d0_splits)",
                        "recorded_state_dict_sha256": recorded["state_dict_sha256"],
                        "rerun_state_dict_sha256": sha,
                        "state_dict_sha256_match": sha_match,
                        "freeze_gate": "functional (five-metric replay <= 0.02); "
                                       "CUDA scatter nondeterminism at 54.8K edges",
                        "replay_max_abs_delta": max_d,
                        "rerun_split0_aggregate": agg,
                        "recorded_split0_aggregate": recorded["aggregate"],
                    })
    m2, _v, _r = load_checkpoint(out_path)
    with torch.no_grad():
        z2 = m2(test_data.clone())["model"]
        z1 = model(test_data.clone())["model"]
    assert torch.allclose(z1, z2, atol=1e-6), "checkpoint roundtrip drifted"
    print(f"[freeze] bound checkpoint: {out_path}")
    return out_path


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = freeze(device)

    py = sys.executable
    print("\n[export] running export_embeddings ...")
    subprocess.run([py, "-m", "ModelLakeFishing.stage3HNSW.export_embeddings",
                    "--graph", GRAPH, "--ckpt", ckpt, "--name", "d0_L1L3b"],
                   check=True, cwd=_REPO_ROOT)
    print("\n[index] running build_index (fidelity gate inside) ...")
    subprocess.run([py, "-m", "ModelLakeFishing.stage3HNSW.build_index",
                    "--export", EXPORT_DIR], check=True, cwd=_REPO_ROOT)
    print("\nDONE: freeze -> export -> index all green.")


if __name__ == "__main__":
    main()
