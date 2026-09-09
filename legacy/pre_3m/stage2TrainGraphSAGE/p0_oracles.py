"""
p0_oracles.py -- v5 P0: the diagnostic funnel. ZERO training.

For every root-aware seed (0..7) and every evaluable test dataset, rank ALL
9,491 lake models with three oracles and measure where the gold model lands
(same gold / tie-safe-rank semantics as five_metric_eval):

  O-static  : shrunk global mean accuracy over TRAIN-visible edges
              (popularity/quality prior; unlabeled models unrankable = -inf)
  O-task    : shrunk mean accuracy on SAME-TASK train datasets
              (task(d) from the native vocab; task unknown -> falls back to
              O-static; models with no same-task history = -inf)
  O-sibling : mean accuracy on the test root's OTHER datasets (deliberate
              leak, leave-one-out) -- the upper bound "if we knew this root"

Plus the structural quantities v5 §0.3-P0 asks for:
  cold-gold rate      : gold has ZERO train-visible edges (z_m has no
                        supervision signal at all)
  near-tie structure  : |{candidates with acc >= gold - delta}| for several
                        deltas (the P4 gold-gap@K target-set size)
  root reachability   : roots stratified by O-task performance; the recorded
                        L1L3b per-root gold@10 reported inside each stratum
  gold-gap@10 (model) : for seed 0 via the frozen L1L3b checkpoint (clean
                        test-graph forward), delta in {0.01, 0.02}

Run (repo root):
  ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.stage2TrainGraphSAGE.p0_oracles
Outputs: stage2TrainGraphSAGE/artifacts/ablation/d0/p0_funnel/p0_report.{json,md}
"""

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
from ModelLakeFishing.stage2TrainGraphSAGE.top1_audit import candidates  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.d0_splits import make_root_aware_splits  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.w1_dzero import GRAPH  # noqa: E402

OUT = os.path.join(_HERE, "artifacts", "ablation", "d0", "p0_funnel")
SEEDS = list(range(8))
SHRINK_K = 5.0
DELTAS = (0.005, 0.01, 0.02, 0.05)
KS = (1, 10, 50)


def _rank(score, gold):
    """tie-safe 1-indexed rank, identical to five_metric_eval."""
    return int((score > score[gold]).sum()) + 1


def seed_funnel(payload, ss, l1l3b_per):
    data = payload["data"]
    udi = payload["unique_dataset_id"].sort_values("mappedID")
    roots = udi["root"].tolist()
    n_m = data["model"].num_nodes
    tt = data["dataset"].task_type_id

    split = make_root_aware_splits(data, roots, split_seed=ss)
    train_data, _val, test_data = split
    lookup = accuracy_lookup(data)
    cands = candidates(test_data, lookup)

    # TRAIN-visible edges = the train roots' full edge set (message + disjoint)
    tr_msg = train_data[TRAINED_ON].edge_index
    tr_sup = train_data[TRAINED_ON].edge_label_index[
        :, train_data[TRAINED_ON].edge_label == 1]
    tr = torch.cat([tr_msg, tr_sup], dim=1)
    tr_acc = torch.tensor([lookup[(int(m), int(d))] for m, d in tr.T.tolist()])

    # O-static: shrunk global mean per model
    m_idx = tr[0].numpy()
    sums = np.bincount(m_idx, weights=tr_acc.numpy(), minlength=n_m)
    cnts = np.bincount(m_idx, minlength=n_m).astype(float)
    prior = float(tr_acc.mean())
    s_static = np.where(cnts > 0, (sums + prior * SHRINK_K) / (cnts + SHRINK_K), -np.inf)
    deg_train = cnts

    # O-task: shrunk per-(model, task) mean
    task_of_edge = tt[tr[1]].numpy()
    s_task = {}
    for t in np.unique(task_of_edge):
        sel = task_of_edge == t
        su = np.bincount(m_idx[sel], weights=tr_acc.numpy()[sel], minlength=n_m)
        ct = np.bincount(m_idx[sel], minlength=n_m).astype(float)
        pr = float(tr_acc.numpy()[sel].mean())
        s_task[int(t)] = np.where(ct > 0, (su + pr * SHRINK_K) / (ct + SHRINK_K), -np.inf)

    # O-sibling: per test root, per-model sum/count over the root's TEST edges
    te = test_data[TRAINED_ON].edge_label_index[
        :, test_data[TRAINED_ON].edge_label == 1]
    te_acc = np.array([lookup[(int(m), int(d))] for m, d in te.T.tolist()])
    root_of = np.array([roots[int(d)] for d in te[1]])
    sib_sum, sib_cnt = defaultdict(lambda: np.zeros(n_m)), defaultdict(lambda: np.zeros(n_m))
    ds_sum, ds_cnt = defaultdict(lambda: np.zeros(n_m)), defaultdict(lambda: np.zeros(n_m))
    for k in range(te.shape[1]):
        m, d = int(te[0, k]), int(te[1, k])
        r = root_of[k]
        sib_sum[r][m] += te_acc[k]; sib_cnt[r][m] += 1
        ds_sum[d][m] += te_acc[k]; ds_cnt[d][m] += 1

    rows = []
    for d, (cand, a) in cands.items():
        gold = int(cand[int(np.argmax(a))])
        gold_acc = float(a.max())
        r = roots[int(d)]
        t = int(tt[int(d)])
        rec = {"seed": ss, "dataset": int(d), "root": r, "task": t,
               "n_candidates": int(cand.size), "gold": gold,
               "cold_gold": bool(deg_train[gold] == 0)}
        for name, score in (("static", s_static),
                            ("task", s_task.get(t, s_static) if t != 0 else s_static)):
            rk = _rank(score, gold)
            rec[f"{name}_rank"] = rk
            for K in KS:
                rec[f"{name}@{K}"] = float(rk <= K)
        lo_sum, lo_cnt = sib_sum[r] - ds_sum[d], sib_cnt[r] - ds_cnt[d]
        if lo_cnt.sum() > 0:
            s_sib = np.where(lo_cnt > 0, lo_sum / np.maximum(lo_cnt, 1), -np.inf)
            rk = _rank(s_sib, gold)
            rec["sibling_defined"] = True
            rec["sibling_rank"] = rk
            for K in KS:
                rec[f"sibling@{K}"] = float(rk <= K)
        else:
            rec["sibling_defined"] = False
        srt = np.sort(a)[::-1]
        rec["gap_to_2nd"] = float(srt[0] - srt[1]) if len(srt) > 1 else float("nan")
        for dl in DELTAS:
            rec[f"n_within_{dl}"] = int((a >= gold_acc - dl).sum())
        key = f"s{ss}::{d}"
        if key in l1l3b_per:
            rec["l1l3b_gold10"] = l1l3b_per[key]
        rows.append(rec)
    return rows


def load_l1l3b_per():
    per = {}
    for tag in ("w1_baselines", "w1_ext"):
        p = os.path.join(_HERE, "artifacts", "ablation", "d0", tag, "L1L3b.json")
        d = json.load(open(p, encoding="utf-8"))
        for ss, sp in d["splits"].items():
            for k, r in sp["per_dataset"].items():
                per[f"s{ss}::{k}"] = float(r["full2k_gold@10"])
    return per


def root_macro(rows, key, defined_key=None):
    by_root = defaultdict(list)
    for r in rows:
        if defined_key and not r.get(defined_key):
            continue
        if key in r:
            by_root[(r["seed"], r["root"])].append(r[key])
    per_seed = defaultdict(list)
    for (ss, _root), vals in by_root.items():
        per_seed[ss].append(float(np.mean(vals)))
    means = [float(np.mean(v)) for v in per_seed.values()]
    return float(np.mean(means)), float(np.std(means))


def main():
    os.makedirs(OUT, exist_ok=True)
    payload = torch.load(GRAPH, map_location="cpu", weights_only=False)
    l1l3b_per = load_l1l3b_per()
    rows = []
    for ss in SEEDS:
        rows += seed_funnel(payload, ss, l1l3b_per)
        print(f"[p0] seed {ss}: {sum(r['seed'] == ss for r in rows)} datasets", flush=True)

    rep = {"n_rows": len(rows), "seeds": SEEDS, "oracles_root_macro": {}}
    for name in ("static", "task", "sibling"):
        dk = "sibling_defined" if name == "sibling" else None
        for K in KS:
            mu, sd = root_macro(rows, f"{name}@{K}", dk)
            rep["oracles_root_macro"][f"{name}@{K}"] = [mu, sd]
    mu, sd = root_macro(rows, "l1l3b_gold10")
    rep["oracles_root_macro"]["L1L3b@10 (recorded)"] = [mu, sd]

    rep["cold_gold_rate"] = float(np.mean([r["cold_gold"] for r in rows]))
    rep["sibling_undefined_share"] = float(np.mean(
        [not r["sibling_defined"] for r in rows]))
    gaps = [r["gap_to_2nd"] for r in rows if not np.isnan(r["gap_to_2nd"])]
    rep["gap_to_2nd"] = {"median": float(np.median(gaps)),
                         "p25": float(np.percentile(gaps, 25)),
                         "share_le_0.01": float(np.mean(np.array(gaps) <= 0.01)),
                         "share_le_0.02": float(np.mean(np.array(gaps) <= 0.02))}
    rep["near_tie_mean_set_size"] = {str(dl): float(np.mean([r[f"n_within_{dl}"] for r in rows]))
                                     for dl in DELTAS}

    # reachability strata: per (seed, root), O-task mean -> predictable if >= 0.5
    strat = defaultdict(lambda: {"otask": [], "l1l3b": []})
    by_root = defaultdict(lambda: {"otask": [], "l1l3b": []})
    for r in rows:
        by_root[(r["seed"], r["root"])]["otask"].append(r["task@10"])
        if "l1l3b_gold10" in r:
            by_root[(r["seed"], r["root"])]["l1l3b"].append(r["l1l3b_gold10"])
    for (_ss, _root), v in by_root.items():
        stratum = "predictable" if np.mean(v["otask"]) >= 0.5 else "unpredictable"
        strat[stratum]["otask"].append(float(np.mean(v["otask"])))
        if v["l1l3b"]:
            strat[stratum]["l1l3b"].append(float(np.mean(v["l1l3b"])))
    rep["reachability"] = {
        s: {"n_root_seed_pairs": len(v["otask"]),
            "otask@10": float(np.mean(v["otask"])) if v["otask"] else None,
            "L1L3b@10": float(np.mean(v["l1l3b"])) if v["l1l3b"] else None}
        for s, v in strat.items()}

    json.dump(rep, open(os.path.join(OUT, "p0_report.json"), "w"), indent=2)
    import csv
    with open(os.path.join(OUT, "p0_rows.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
