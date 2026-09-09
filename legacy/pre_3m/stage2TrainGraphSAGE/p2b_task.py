"""
p2b_task.py -- v6 P2b: serving-time TASK-prior fusion (zero-train).

P0 found a "predictable" stratum (9% of roots) where the task heuristic O-task@10
= 0.87 but L1L3b only reaches 0.31 -- the model under-uses task metadata. P2b is
the serving-side fix, orthogonal to S2's sibling channel: fuse the MIPS score with
a TASK prior (a model's shrunk mean accuracy on SAME-TASK train datasets). Task is
metadata the query D carries (task_type_id), NOT a held-out label -- legitimate.

Same node-level protocol/harness as S2 (reuses node_level_dataset_split). This
run decides whether the serving layer fuses TWO channels (sibling + task) or one:
it reports MIPS baseline, MIPS+task (P2b), and -- with S2's sibling boost held at
alpha=1 -- MIPS+sibling+task (do they stack?). Stratified by whether the query has
a same-task train prior.

LEGALITY: task-prior uses only TRAIN-visible same-task datasets, excluding D
itself; D's own held-out labels never enter.

Run (repo root):
  ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.stage2TrainGraphSAGE.p2b_task
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.losses import TRAINED_ON, accuracy_lookup  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import apply_similar_to_mode  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.ablation import train_eval_one  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.top1_audit import candidates, model_names  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.top1_eval import GOLD_GAP_DELTA  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.w1_dzero import GRAPH, d0_configs  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.s2_sibling import (  # noqa: E402
    node_level_dataset_split, BATCH,
)

OUT = os.path.join(_HERE, "artifacts", "ablation", "d0", "p2b_task")
SEEDS = list(range(8))
EPOCHS = 25
SHRINK_K = 5.0
BETAS = (0.0, 0.5, 1.0)          # task-fusion weight (0 = pure MIPS)
SIB_ALPHA = 1.0                  # adopted S2 sibling weight for the stacking test


def build_task_prior(train_data, lookup, task_id, num_models):
    """(model, task) -> shrunk mean acc over TRAIN datasets of that task, plus
    the set of train nodes each drew from (to exclude the query itself)."""
    tr = torch.cat([train_data[TRAINED_ON].edge_index,
                    train_data[TRAINED_ON].edge_label_index[
                        :, train_data[TRAINED_ON].edge_label == 1]], dim=1)
    s = defaultdict(float); c = defaultdict(int); nodes = defaultdict(set)
    all_acc = []
    for m, d in zip(tr[0].tolist(), tr[1].tolist()):
        a = lookup.get((int(m), int(d)))
        if a is None:
            continue
        t = int(task_id[d]); s[(m, t)] += a; c[(m, t)] += 1
        nodes[(m, t)].add(d); all_acc.append(a)
    prior = float(np.mean(all_acc)) if all_acc else 0.5
    return s, c, nodes, prior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    payload = torch.load(GRAPH, map_location="cpu", weights_only=False)
    udi = payload["unique_dataset_id"].sort_values("mappedID")
    root_of = udi["root"].tolist()
    task_id = payload["data"]["dataset"].task_type_id
    names = model_names(payload["unique_model_id"])
    cfg = d0_configs(["L1L3b"])["L1L3b"]; cfg["batch_size"] = BATCH

    rows = []
    for ss in args.seeds:
        data = payload["data"].clone()
        data = apply_similar_to_mode(data, cfg["similar_to_mode"], k=cfg["similar_to_k"])
        train_data, val_data, test_data, _tn = node_level_dataset_split(
            data, split_seed=ss, test_frac=0.2)
        lookup = accuracy_lookup(data)
        _r, _pt, _ph, model, _sc = train_eval_one(
            data, payload["xm0_meta"], payload["xd0_meta"], cfg,
            (train_data, val_data, test_data), init_seed=0, epochs=args.epochs, device=device)
        model.eval()
        with torch.no_grad():
            z = {k: v.cpu() for k, v in model(test_data.clone().to(device)).items()}
        z_m = torch.nn.functional.normalize(z["model"], dim=-1)
        z_d = torch.nn.functional.normalize(z["dataset"], dim=-1)
        cands = candidates(test_data, lookup)
        nm = data["model"].num_nodes

        # task-prior tables (train-visible) + flat acc_pair for sibling reuse
        ts, tc, tnodes, tprior = build_task_prior(train_data, lookup, task_id, nm)
        acc_pair, seen_nodes = {}, defaultdict(set)
        tr = torch.cat([train_data[TRAINED_ON].edge_index,
                        train_data[TRAINED_ON].edge_label_index[
                            :, train_data[TRAINED_ON].edge_label == 1]], dim=1)
        for m, d in zip(tr[0].tolist(), tr[1].tolist()):
            a = lookup.get((int(m), int(d)))
            if a is not None:
                acc_pair[(m, d)] = a
                seen_nodes[(m, root_of[d])].add(d)
        train_ds = set(train_data[TRAINED_ON].edge_index[1].tolist())

        for d, (cand, a) in cands.items():
            gold = int(cand[int(np.argmax(a))]); gold_acc = float(a.max())
            t = int(task_id[d])
            # task boost: shrunk same-task mean per model, excluding d
            same_task_train = {dd for dd in train_ds if int(task_id[dd]) == t} - {d}
            has_task = len(same_task_train) > 0
            tboost = np.zeros(nm)
            if has_task:
                for (m, tt), cnt in tc.items():
                    if tt != t:
                        continue
                    sib = tnodes[(m, tt)] - {d}
                    if not sib:
                        continue
                    su = sum(acc_pair[(m, dd)] for dd in sib)
                    tboost[m] = (su + tprior * SHRINK_K) / (len(sib) + SHRINK_K)
            mips = (z_m @ z_d[d]).numpy()
            mips_n = (mips - mips.min()) / (mips.max() - mips.min() + 1e-9)
            # sibling boost (same as S2) for the stacking test
            sib_boost = np.zeros(nm)
            r = root_of[d]
            for (m, rr), nodes_ in seen_nodes.items():
                if rr != r:
                    continue
                sib = nodes_ - {d}
                if sib:
                    sib_boost[m] = float(np.mean([acc_pair[(m, dd)] for dd in sib]))
            near = cand[a >= gold_acc - GOLD_GAP_DELTA]
            has_sib = bool((sib_boost > 0).any())
            rec = {"seed": ss, "dataset": d, "root": r, "task": t,
                   "has_task_prior": has_task, "has_sibling": has_sib,
                   "n_same_task": len(same_task_train)}

            def score_metrics(s_all, tag):
                grank = int((s_all > s_all[gold]).sum()) + 1
                gap = int(min((s_all > s_all[int(m)]).sum() + 1 for m in near))
                rec[f"{tag}_gold@10"] = float(grank <= 10)
                rec[f"{tag}_gold@1"] = float(grank <= 1)
                rec[f"{tag}_gap@10"] = float(gap <= 10)

            score_metrics(mips_n, "mips")
            for beta in BETAS:
                if beta == 0.0:
                    continue
                score_metrics(mips_n + beta * tboost, f"t{beta}")
            # stacking: sibling(alpha=1) + task(beta=1)
            score_metrics(mips_n + SIB_ALPHA * sib_boost, "sib")
            score_metrics(mips_n + SIB_ALPHA * sib_boost + 1.0 * tboost, "sibtask")
            rows.append(rec)
        print(f"[p2b s{ss}] test {len(cands)} | with task-prior "
              f"{sum(r['has_task_prior'] for r in rows if r['seed']==ss)}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "p2b_rows.csv"), index=False)

    def rootmac(sub, col):
        ps = [sub[sub.seed == ss].groupby("root")[col].mean().mean()
              for ss in sub.seed.unique() if not sub[sub.seed == ss].empty]
        return float(np.mean(ps))

    tags = ["mips"] + [f"t{b}" for b in BETAS if b > 0] + ["sib", "sibtask"]
    report = {"protocol": "node-level; task-prior fusion", "seeds": len(args.seeds),
              "betas": BETAS, "strata": {}}
    for name, sub in (("all", df), ("has_sibling", df[df.has_sibling]),
                      ("no_sibling", df[~df.has_sibling])):
        report["strata"][name] = {"n_per_seed": int(len(sub) / len(args.seeds))}
        for tag in tags:
            report["strata"][name][tag] = {
                m: rootmac(sub, f"{tag}_{m}") for m in ("gold@10", "gold@1", "gap@10")}
    json.dump(report, open(os.path.join(OUT, "p2b_report.json"), "w"), indent=2)
    print(json.dumps(report["strata"], indent=1))


if __name__ == "__main__":
    main()
