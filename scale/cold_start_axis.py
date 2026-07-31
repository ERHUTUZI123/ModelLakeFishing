"""
cold_start_axis.py -- P5 C-axis: cold-start robustness, both systems, same 12K
universe / same 517 held-out gold queries / same harness. No extra training --
stratifies the A-axis result along two cold-start dimensions:

  (1) new-MODEL proxy: bin queries by the GOLD model's TRAINING evidence
      (# distinct train datasets it was observed on). A gold model seen on few
      train datasets is closer to cold. Our GNN is inductive (embeds a model
      from features + neighbourhood); ModelLens leans on a learned per-model id
      embedding that is weak/[UNK] for thinly-seen models.
  (2) new-DATASET dimension: bin queries by SIBLING availability -- whether the
      query dataset's root has OTHER (dataset,task) nodes in the training split.
      Isolated roots are the hard cold-start; sibling-rich roots are where our
      graph structure (similar_to) and sibling evidence can help.

Both systems are the SAME as P5 A-axis: ours = held-out z (from_embeddings);
ModelLens = the retrained NON-BLIND model (retrained_modellens.pt), re-scored.

Run (repo root):
  ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.scale.cold_start_axis \
     --graph ModelLakeFishing/stage1BuildTransferGraph/hgraph_ml_v2_sub.pt
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
from scale import global_metrics as GM  # noqa: E402
from scale.retrain_modellens import build_modellens, LAKE, EXPORT, OUT, NODE_SEP  # noqa: E402


def gold_ranks_ours(zm, zd, cands):
    zm = zm / (np.linalg.norm(zm, axis=1, keepdims=True) + 1e-12)
    zd = zd / (np.linalg.norm(zd, axis=1, keepdims=True) + 1e-12)
    out = {}
    for d, (c, a) in cands.items():
        s = zm @ zd[int(d)]
        gold = int(c[int(np.argmax(a))])
        out[int(d)] = int((s > s[gold]).sum()) + 1
    return out


def gold_ranks_modellens(graph, cands, dev):
    payload = torch.load(graph, map_location="cpu", weights_only=False)
    data = payload["data"]
    umi = payload["unique_model_id"].sort_values("mappedID").reset_index(drop=True)
    udi = payload["unique_dataset_id"].sort_values("mappedID").reset_index(drop=True)
    names = umi["model"].tolist()
    n_m, n_d = len(names), len(udi)
    xm, xd = data["model"].x.numpy(), data["dataset"].x.numpy()
    model_desc = torch.tensor(xm[:, 64:448], dtype=torch.float32)
    dataset_desc = torch.tensor(xd[:, 64:448], dtype=torch.float32).to(dev)
    size_ids = data["model"].size_bucket_id.to(dev)
    fam_ids = data["model"].family_id.to(dev)
    task_ids_d = data["dataset"].task_type_id
    n_size = int(payload["xm0_meta"]["num_size_buckets"])
    n_fam = int(payload["xm0_meta"]["num_families"])
    n_task = int(payload["xd0_meta"]["num_task_types"])
    pool = pd.read_csv(os.path.join(LAKE, "ml_dataset_pool.csv"))
    node2metric = dict(zip(pool["dataset_node"], pool["chosen_metric"]))
    metrics = sorted(set(str(node2metric.get(nd, "unknown")) for nd in udi["dataset"]))
    metric2id = {"unknown_metric": 0, **{m: i for i, m in enumerate(metrics, 1)}}
    metric_ids_d = torch.tensor(
        [metric2id.get(str(node2metric.get(nd, "unknown")), 0) for nd in udi["dataset"]],
        dtype=torch.long)
    n_metric = len(metric2id)

    model, _ = build_modellens(n_m, n_d, n_task, n_metric, n_size, n_fam, dev)
    del model.model_desc_matrix
    model.register_buffer("model_desc_matrix", model_desc.clone().to(dev))
    model.load_state_dict(torch.load(os.path.join(OUT, "retrained_modellens.pt"),
                                     map_location=dev), strict=False)
    model.eval()
    unk = model.unk_dataset_id
    cache = model.build_model_cache(names, size_ids, all_model_family_ids=fam_ids, device=dev)
    out = {}
    with torch.no_grad():
        for d, (c, a) in cands.items():
            model.dataset_desc_matrix[min(unk, model.dataset_desc_matrix.shape[0]-1)] = dataset_desc[int(d)]
            s = model.score_matrix(task_ids_d[int(d)].view(1).to(dev),
                                   torch.tensor([[float(unk)]], device=dev), cache,
                                   metric_ids=metric_ids_d[int(d)].view(1).to(dev))
            s = s.squeeze(0).float().cpu().numpy()
            gold = int(c[int(np.argmax(a))])
            out[int(d)] = int((s > s[gold]).sum()) + 1
    return out, udi, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    zm = np.load(os.path.join(EXPORT, "z_m_eval.npy"))
    zd = np.load(os.path.join(EXPORT, "z_d_eval.npy"))
    gc = np.load(os.path.join(EXPORT, "gold_cands.npz"))
    cands = {int(k): (gc[k][0].astype(int), gc[k][1].astype(float)) for k in gc.files}

    r_ours = gold_ranks_ours(zm, zd, cands)
    r_ml, udi, data = gold_ranks_modellens(args.graph, cands, dev)

    # per-query gold model + its TRAIN degree (train = non-test datasets)
    ei = data["model", "trained_on", "dataset"].edge_index.numpy()
    test_d = set(cands)
    train_deg = np.zeros(len(udi) if False else int(ei[0].max()) + 1, dtype=int)
    seen = {}
    for k in range(ei.shape[1]):
        m, d = int(ei[0, k]), int(ei[1, k])
        if d in test_d:
            continue
        seen.setdefault(m, set()).add(d)
    gold_deg = {}
    for d, (c, a) in cands.items():
        g = int(c[int(np.argmax(a))])
        gold_deg[d] = len(seen.get(g, set()))

    # same-TASK train-peer availability (P2b task-fusion dimension). NOTE the
    # same-ROOT sibling axis is degenerate here: root-aware splitting removes
    # every same-root sibling from training BY DESIGN (leakage-free), so all
    # queries are root-isolated. Task peers are the non-degenerate cold signal.
    task_d = data["dataset"].task_type_id.numpy()
    train_task = {}
    for nd_id in range(len(udi)):
        if nd_id in test_d:
            continue
        t = int(task_d[nd_id]); train_task[t] = train_task.get(t, 0) + 1
    sib = {d: train_task.get(int(task_d[int(d)]), 0) for d in cands}

    def gold10(ranks, keys):
        return float(np.mean([ranks[k] <= 10 for k in keys])) if keys else float("nan")

    def stratum(name, val_of, bins):
        rows = []
        for lo, hi, lab in bins:
            keys = [d for d in cands if lo <= val_of[d] <= hi]
            rows.append({"stratum": lab, "n": len(keys),
                         "ours_gold@10": gold10(r_ours, keys),
                         "modellens_gold@10": gold10(r_ml, keys)})
        return {"dimension": name, "strata": rows}

    result = {
        "universe": len(zm), "n_queries": len(cands),
        "overall": {"ours_gold@10": gold10(r_ours, list(cands)),
                    "modellens_retrained_gold@10": gold10(r_ml, list(cands))},
        "C1_new_model_proxy_gold_model_train_degree": stratum(
            "gold model # train datasets (cold->warm)", gold_deg,
            [(0, 2, "cold (<=2)"), (3, 10, "mid (3-10)"), (11, 10**9, "warm (>=11)")]),
        "C2_new_dataset_same_task_peers": stratum(
            "query # same-task train datasets (root-siblings excluded by split)", sib,
            [(0, 5, "rare task (<=5)"), (6, 50, "mid (6-50)"), (51, 10**9, "common task (>=51)")]),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "cold_start.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    print("=== C-AXIS cold-start (gold@10, same 12K/517/harness) ===")
    print(f"overall: ours {result['overall']['ours_gold@10']:.3f}  "
          f"ModelLens(retrained) {result['overall']['modellens_retrained_gold@10']:.3f}")
    for key in ("C1_new_model_proxy_gold_model_train_degree",
                "C2_new_dataset_same_task_peers"):
        print(f"\n{result[key]['dimension']}:")
        for r in result[key]["strata"]:
            print(f"  {r['stratum']:18s} n={r['n']:4d}  ours {r['ours_gold@10']:.3f}  "
                  f"ML {r['modellens_gold@10']:.3f}")
    print(f"\n-> {OUT}\\cold_start.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
