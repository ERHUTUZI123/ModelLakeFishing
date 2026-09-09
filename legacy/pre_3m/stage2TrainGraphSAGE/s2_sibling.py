"""
s2_sibling.py -- v6 S2: serving-time sibling-prior fusion (headline, zero-train).

The deployable realization of the O-sibling signal (P0: O-sibling@10 = 0.563 vs
incumbent 0.100). At query time for a NEW dataset D whose sibling configs are
ALREADY in the lake (labeled), borrow those siblings' known-good models as a
prior and fuse with the MIPS score z_d . z_m. NO new training -- it re-ranks the
frozen L1L3b embeddings.

LEGALITY BOUNDARY (v6 §0.6-3, asserted below): the prior may use only
TRAIN-VISIBLE sibling labels (a DIFFERENT dataset node, same root, whose edges
are in the training graph). It must NEVER use D's own held-out labels -- that
is the O-sibling *oracle* (leakage), not a deployable signal.

Protocol note (why NOT root-aware): root-aware splits hold out a whole root, so
a test dataset's siblings are all in test too -> zero train-visible siblings ->
S2 can't fire (correctly). That is the ISOLATED-root scenario (E19), where
content was the only hope and failed. S2 targets the SIBLING-RICH scenario, so
we evaluate under a NODE-LEVEL dataset split: D's own edges are held out (cold
for D) while D's sibling nodes stay in train (labeled) -- exactly "new config of
a known benchmark family."

Run (repo root):
  ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.stage2TrainGraphSAGE.s2_sibling
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

from ModelLakeFishing.stage2TrainGraphSAGE.losses import (  # noqa: E402
    TRAINED_ON, REV_TRAINED_ON, accuracy_lookup,
)
from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import apply_similar_to_mode  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.ablation import train_eval_one  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.top1_audit import candidates, model_names  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.top1_eval import GOLD_KS, GOLD_GAP_DELTA  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.d0_splits import _neg_sample  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.w1_dzero import GRAPH, d0_configs  # noqa: E402

OUT = os.path.join(_HERE, "artifacts", "ablation", "d0", "s2_sibling")
SEEDS = list(range(8))
EPOCHS = 25
BATCH = 1024
ALPHAS = (0.0, 0.5, 1.0, 2.0)     # 0.0 = pure-MIPS baseline
TEST_FRAC = 0.2


def node_level_dataset_split(data, *, split_seed, test_frac=0.2, val_frac=0.1,
                             neg_ratio=1.0):
    """Hold out `test_frac` of DATASET NODES; ALL their trained_on edges become
    test supervision, the rest stay in the train message+supervision graph. A
    held-out dataset D is cold (its own edges gone) but its sibling nodes remain
    in train -> the S2 signal is legitimately available. rev mirrors message."""
    gen = torch.Generator().manual_seed(split_seed)
    nd = data["dataset"].num_nodes
    perm = torch.randperm(nd, generator=gen)
    n_test = int(round(test_frac * nd)); n_val = int(round(val_frac * nd))
    test_nodes = set(perm[:n_test].tolist())
    val_nodes = set(perm[n_test:n_test + n_val].tolist())

    ei = data[TRAINED_ON].edge_index
    attr = data[TRAINED_ON].edge_attr
    dcol = ei[1]
    side = np.array(["train"] * ei.shape[1], dtype=object)
    for i, d in enumerate(dcol.tolist()):
        if d in test_nodes:
            side[i] = "test"
        elif d in val_nodes:
            side[i] = "val"
    mask = {s: torch.tensor(side == s) for s in ("train", "val", "test")}
    pos_set = {(int(m), int(d)) for m, d in zip(ei[0].tolist(), ei[1].tolist())}
    nm = data["model"].num_nodes

    def _mk(msg_idx, label_idx):
        out = data.clone()
        me = ei[:, msg_idx]
        out[TRAINED_ON].edge_index = me
        out[TRAINED_ON].edge_attr = attr[msg_idx]
        out[REV_TRAINED_ON].edge_index = me.flip(0)
        out[REV_TRAINED_ON].edge_attr = attr[msg_idx].clone()
        pos = ei[:, label_idx]
        neg = _neg_sample(pos_set, nm, nd, int(round(neg_ratio * pos.shape[1])), gen)
        out[TRAINED_ON].edge_label_index = torch.cat([pos, neg], dim=1)
        out[TRAINED_ON].edge_label = torch.cat(
            [torch.ones(pos.shape[1]), torch.zeros(neg.shape[1])])
        return out

    tr_idx = mask["train"].nonzero().flatten()
    val_idx = mask["val"].nonzero().flatten()
    test_idx = mask["test"].nonzero().flatten()
    # disjoint train supervision (30%) removed from the message graph
    trp = tr_idx[torch.randperm(tr_idx.numel(), generator=gen)]
    n_dis = int(round(0.3 * tr_idx.numel()))
    train_data = _mk(trp[n_dis:], trp[:n_dis])
    val_data = _mk(tr_idx, val_idx)
    test_data = _mk(torch.cat([tr_idx, val_idx]), test_idx)
    # leakage guard: no test dataset's OWN edge in the train message graph
    train_msg_ds = set(train_data[TRAINED_ON].edge_index[1].tolist())
    assert not (train_msg_ds & test_nodes), "cold-D violated: test node edge in train msg"
    return train_data, val_data, test_data, test_nodes


def s2_scores(z_m, z_d, d, acc_pair, seen_nodes, root_of, alpha):
    """Full-lake fused score for query dataset d: fused = minmax(MIPS) +
    alpha * sibling_boost. boost[m] = mean train-visible acc of m over d's
    SIBLING nodes (same root, != d). LEGALITY: d itself is excluded from every
    mean, so d's own held-out label never enters the score."""
    mips = (z_m @ z_d[d]).numpy()
    mips_n = (mips - mips.min()) / (mips.max() - mips.min() + 1e-9)
    r = root_of[d]
    boost = np.zeros_like(mips_n)
    for (m, rr), nodes in seen_nodes.items():
        if rr != r:
            continue
        sib = nodes - {d}                       # exclude the query's own node
        if sib:
            boost[m] = float(np.mean([acc_pair[(m, dd)] for dd in sib]))
    return mips_n + alpha * boost, boost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--s1", action="store_true",
                    help="v6 S1: train with root-level positive pooling (the "
                         "training-side dual). Combined with the S2 fusion at "
                         "eval this measures S1 alone (a=0) and S1xS2 (a>0).")
    args = ap.parse_args()
    global OUT
    if args.s1:
        OUT = OUT + "_s1"
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    payload = torch.load(GRAPH, map_location="cpu", weights_only=False)
    udi = payload["unique_dataset_id"].sort_values("mappedID")
    root_of = udi["root"].tolist()
    names = model_names(payload["unique_model_id"])
    cfg = d0_configs(["L1L3b"])["L1L3b"]; cfg["batch_size"] = BATCH
    if args.s1:
        _rc = {r: i for i, r in enumerate(sorted(set(root_of)))}
        cfg["root_pool_positives"] = True
        cfg["root_ids"] = [_rc[r] for r in root_of]

    rows = []
    for ss in args.seeds:
        data = payload["data"].clone()
        data = apply_similar_to_mode(data, cfg["similar_to_mode"], k=cfg["similar_to_k"])
        train_data, val_data, test_data, test_nodes = node_level_dataset_split(
            data, split_seed=ss, test_frac=TEST_FRAC)
        lookup = accuracy_lookup(data)
        _row, _pt, _ph, model, _sc = train_eval_one(
            data, payload["xm0_meta"], payload["xd0_meta"], cfg,
            (train_data, val_data, test_data), init_seed=0, epochs=args.epochs, device=device)
        model.eval()
        with torch.no_grad():
            z = {k: v.cpu() for k, v in model(test_data.clone().to(device)).items()}
        z_m = torch.nn.functional.normalize(z["model"], dim=-1)
        z_d = torch.nn.functional.normalize(z["dataset"], dim=-1)
        cands = candidates(test_data, lookup)

        # train-visible (model, node) -> acc and seen_nodes per (model, root)
        acc_pair, seen_nodes = {}, {}
        from collections import defaultdict
        seen_nodes = defaultdict(set)
        tr = torch.cat([train_data[TRAINED_ON].edge_index,
                        train_data[TRAINED_ON].edge_label_index[
                            :, train_data[TRAINED_ON].edge_label == 1]], dim=1)
        for m, d in zip(tr[0].tolist(), tr[1].tolist()):
            a = lookup.get((int(m), int(d)))
            if a is not None:
                acc_pair[(m, d)] = a
                seen_nodes[(m, root_of[d])].add(d)

        # which roots have train-visible sibling datasets at all
        train_ds = set(train_data[TRAINED_ON].edge_index[1].tolist())
        n_sib = 0
        for d, (cand, a) in cands.items():
            gold = int(cand[int(np.argmax(a))]); gold_acc = float(a.max())
            r = root_of[d]
            sib_nodes = {dd for dd in train_ds if root_of[dd] == r} - {d}
            has_sib = len(sib_nodes) > 0
            n_sib += has_sib
            rec = {"seed": ss, "dataset": d, "root": r, "has_sibling": has_sib,
                   "n_cand": int(cand.size), "n_sib_nodes": len(sib_nodes)}
            near = cand[a >= gold_acc - GOLD_GAP_DELTA]
            for alpha in ALPHAS:
                s_all, boost = s2_scores(z_m, z_d, d, acc_pair, seen_nodes, root_of, alpha)
                grank = int((s_all > s_all[gold]).sum()) + 1
                gap_rank = int(min((s_all > s_all[int(m)]).sum() + 1 for m in near))
                for K in (1, 10):
                    rec[f"a{alpha}_gold@{K}"] = float(grank <= K)
                    rec[f"a{alpha}_gap@{K}"] = float(gap_rank <= K)
            rows.append(rec)
        print(f"[s2 s{ss}] test datasets {len(cands)} | with train-sibling {n_sib}",
              flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "s2_rows.csv"), index=False)

    def root_macro(sub, col):
        per_seed = []
        for ss in df["seed"].unique():
            s = sub[sub["seed"] == ss]
            if s.empty:
                continue
            per_seed.append(s.groupby("root")[col].mean().mean())
        return float(np.mean(per_seed)), float(np.std(per_seed))

    report = {"protocol": "node-level dataset split (cold-D, warm siblings)",
              "seeds": len(args.seeds), "test_frac": TEST_FRAC, "alphas": ALPHAS,
              "strata": {}}
    for name, sub in (("all", df), ("has_sibling", df[df["has_sibling"]]),
                      ("no_sibling", df[~df["has_sibling"]])):
        report["strata"][name] = {"n_per_seed": int(len(sub) / len(args.seeds))}
        for alpha in ALPHAS:
            report["strata"][name][f"alpha{alpha}"] = {
                m: root_macro(sub, f"a{alpha}_{m}")[0]
                for m in ("gold@10", "gap@10", "gold@1")}
    json.dump(report, open(os.path.join(OUT, "s2_report.json"), "w"), indent=2)
    print(json.dumps(report["strata"], indent=1))


if __name__ == "__main__":
    main()
