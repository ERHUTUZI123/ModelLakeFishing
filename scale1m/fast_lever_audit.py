import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(_HERE), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from torch_geometric.data import HeteroData

from scale1m.hf_crawl import data_root, utcnow, write_json_atomic
from scale1m.utility_scorecard import PIPELINE_TAGS, norm_task
from ModelLakeFishing.stage2TrainGraphSAGE.losses import (
    TRAINED_ON, REV_TRAINED_ON, build_lake_logq)
from ModelLakeFishing.stage2TrainGraphSAGE.d0_splits import (
    make_root_aware_splits)
from ModelLakeFishing.scale import global_metrics as GM

SEEDS = (0, 1, 2)
RUN_FMT = "RF_full_s%d_e25"
SHRINK_K = 5.0
BATCH_SIZE = 1024
EPOCHS = 25
N_DATASETS_PER_STEP = 16
N_NEG = 256
RANK_KS = (1, 3, 10, 50, 100, 200, 500, 1000, 2000, 10000)
PRIOR_KS = (1, 3, 10, 50, 100)


def reduced_graph(gdir):
    with np.load(os.path.join(gdir, "edges.npz")) as z:
        ei = torch.from_numpy(z["model__trained_on__dataset__edge_index"].copy())
        ea = torch.from_numpy(z["model__trained_on__dataset__edge_attr"].copy())
    with open(os.path.join(gdir, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    d = HeteroData()
    d["model"].num_nodes = int(meta["num_nodes"]["model"])
    d["dataset"].num_nodes = int(meta["num_nodes"]["dataset"])
    d[TRAINED_ON].edge_index = ei
    d[TRAINED_ON].edge_attr = ea
    d[REV_TRAINED_ON].edge_index = ei.flip(0)
    d[REV_TRAINED_ON].edge_attr = ea.clone()
    udi = pd.read_parquet(os.path.join(gdir, "unique_dataset_id.parquet"))
    udi = udi.sort_values("mappedID").reset_index(drop=True)
    return d, udi["root"].astype(str).tolist(), udi


def split_facts(gdir, seed):
    d, roots, udi = reduced_graph(gdir)
    tr, _val, te = make_root_aware_splits(d, roots, split_seed=seed)
    pos = tr[TRAINED_ON].edge_label == 1
    sup = tr[TRAINED_ON].edge_label_index[:, pos]
    msg = tr[TRAINED_ON].edge_index
    ti = torch.cat([msg, sup], dim=1)
    return {"data": d, "roots": roots, "udi": udi, "ti": ti,
            "n_sup_rows": int(sup.shape[1]),
            "n_label_rows": int(tr[TRAINED_ON].edge_label.numel()),
            "train_visible_datasets": int(torch.unique(ti[1]).numel()),
            "visible": te[TRAINED_ON].edge_index,
            "visible_attr": te[TRAINED_ON].edge_attr,
            "test_pos": te[TRAINED_ON].edge_label_index[
                :, te[TRAINED_ON].edge_label == 1],
            "n_models": int(d["model"].num_nodes),
            "n_datasets": int(d["dataset"].num_nodes),
            "n_edges": int(d[TRAINED_ON].edge_index.shape[1])}


def load_cands(exports, seed, run_fmt=RUN_FMT):
    p = os.path.join(exports, run_fmt % seed, "gold_cands.npz")
    with np.load(p) as z:
        return {int(k): (z[k][0].astype(np.int64), z[k][1].astype(float))
                for k in z.files}


def merge(out, stage, payload):
    p = os.path.join(out, "X1_FAST_LEVERS.json")
    rep = {}
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as fh:
            rep = json.load(fh)
    rep.setdefault("stages", {})[stage] = payload
    rep["written_at"] = utcnow()
    write_json_atomic(p, rep)
    return rep


def interval(vals):
    v = [float(x) for x in vals]
    return {"min": min(v), "mean": float(np.mean(v)), "max": max(v), "per_seed": v}


def mixture_mass(deg, alpha, n0, gamma):
    w = (deg.double() + n0) ** alpha
    base = w / w.sum()
    lab = deg > 0
    if int(lab.sum()) == 0:
        return 0.0
    return float((1 - gamma) * base[lab].sum() + gamma)


def stage_q(args):
    t0 = time.time()
    out = {}
    for name, gdir in _rungs(args).items():
        per_seed = {}
        for s in args.seeds:
            f = split_facts(gdir, s)
            N = f["n_models"]
            deg_tr = torch.bincount(f["ti"][0], minlength=N)
            deg_all = torch.bincount(f["data"][TRAINED_ON].edge_index[0], minlength=N)
            row = {"n_models": N,
                   "labeled_train_visible": int((deg_tr > 0).sum()),
                   "labeled_full_graph": int((deg_all > 0).sum()),
                   "alpha_sweep": {}, "gamma_sweep": {}}
            for a in (0.0, 0.5, 0.75, 1.0):
                q, _ = build_lake_logq(f["ti"], N, alpha=a, n0=1.0)
                m = float(q[deg_tr > 0].sum())
                row["alpha_sweep"]["%.2f" % a] = {
                    "mass_on_supervised": m, "expected_supervised_in_256": m * N_NEG}
            qa, _ = build_lake_logq(f["data"][TRAINED_ON].edge_index, N,
                                    alpha=0.75, n0=1.0)
            row["audit_full_graph_deg_mass_alpha075"] = float(qa[deg_all > 0].sum())
            for g in (0.0, 0.25, 0.5, 0.75):
                m = mixture_mass(deg_tr, 0.75, 1.0, g)
                row["gamma_sweep"]["%.2f" % g] = {
                    "mass_on_supervised": m, "expected_supervised_in_256": m * N_NEG}
            per_seed[s] = row
            print("  [M1] %-8s seed=%d labeled(train-visible)=%d mass@a=0.75 %.4f "
                  "-> %.1f of %d negatives"
                  % (name, s, row["labeled_train_visible"],
                     row["alpha_sweep"]["0.75"]["mass_on_supervised"],
                     row["alpha_sweep"]["0.75"]["expected_supervised_in_256"], N_NEG),
                  flush=True)
            del f
        out[name] = {"per_seed": per_seed, "interval_mass_alpha075": interval(
            [per_seed[s]["alpha_sweep"]["0.75"]["mass_on_supervised"] for s in per_seed])}
    payload = {"seconds": round(time.time() - t0, 1), "n_neg": N_NEG, "rungs": out,
               "note": "ablation.py builds q from ti = train message edges + the "
                       "disjoint supervision part, i.e. the TRAIN-VISIBLE degree; "
                       "the audit's 0.6841 / 0.0464 used the full-graph degree"}
    merge(args.out, "q", payload)
    return payload


def _rungs(args):
    r = {}
    for name, sub in (("full_3m", "hgraph_rf"), ("100k", "hgraph_100k_sharded")):
        p = os.path.join(args.graphs, sub)
        if os.path.isdir(p):
            r[name] = p
    return r


def stage_splits(args):
    t0 = time.time()
    out = {}
    key = "dataset_touches_in_%d_epochs" % EPOCHS
    for name, gdir in _rungs(args).items():
        per_seed = {}
        for s in args.seeds:
            f = split_facts(gdir, s)
            steps = math.ceil(f["n_sup_rows"] / BATCH_SIZE)
            vis = f["train_visible_datasets"]
            per_seed[s] = {
                "n_edges": f["n_edges"], "n_supervision_rows": f["n_sup_rows"],
                "n_label_rows_incl_negatives": f["n_label_rows"],
                "steps_per_epoch": steps,
                "steps_per_epoch_if_negatives_counted":
                    math.ceil(f["n_label_rows"] / BATCH_SIZE),
                "train_visible_datasets": vis,
                key: steps * EPOCHS * N_DATASETS_PER_STEP / vis,
                "n_test_query_datasets_raw": int(torch.unique(f["test_pos"][1]).numel())}
            print("  [M2] %-8s seed=%d steps/epoch=%d train-visible datasets=%d "
                  "touches=%.2f" % (name, s, steps, vis, per_seed[s][key]), flush=True)
            del f
        out[name] = {"per_seed": per_seed,
                     "interval_touches": interval([per_seed[s][key] for s in per_seed])}
    payload = {"seconds": round(time.time() - t0, 1), "batch_size": BATCH_SIZE,
               "epochs": EPOCHS, "global_n_datasets": N_DATASETS_PER_STEP,
               "rungs": out}
    merge(args.out, "splits", payload)
    return payload


_NONWORD = re.compile(r"[^0-9a-z]+")


def trigrams(name):
    s = re.sub(r"\s+", " ", _NONWORD.sub(" ", str(name).lower()).strip())
    return {s[i:i + 3] for i in range(max(0, len(s) - 2))} or {s}


def rank_with_ties(score, pool, gold):
    sg = score[gold]
    sub = score[pool]
    return int((sub > sg).sum()) + 1, int((sub == sg).sum()) - 1


def _at(ranks, ks=PRIOR_KS):
    a = np.asarray(ranks, dtype=float)
    return {"gold@%d" % k: float((a <= k).mean()) for k in ks}


def stage_task(args):
    t0 = time.time()
    nodes = pd.read_parquet(args.dataset_nodes,
                            columns=["node", "task", "gold_eligible",
                                     "primary_direction", "is_placeholder", "is_rl"])
    nodes["node"] = nodes["node"].astype(str)
    task_by_node = nodes.set_index("node")["task"]
    elig_by_node = nodes.set_index("node")["gold_eligible"]
    reason_by_node = {
        "direction_unknown": nodes.set_index("node")["primary_direction"]
        .astype(str).eq("unknown"),
        "is_placeholder": nodes.set_index("node")["is_placeholder"].fillna(False),
        "is_rl": nodes.set_index("node")["is_rl"].fillna(False)}

    per_seed = {}
    for s in args.seeds:
        f = split_facts(os.path.join(args.graphs, "hgraph_rf"), s)
        exp = os.path.join(args.exports, args.run_fmt % s)
        udi = pd.read_parquet(os.path.join(exp, "dataset_ids.parquet")).sort_values("mappedID")
        node_of = udi["dataset"].astype(str).to_numpy()
        root_of = udi["root"].astype(str).to_numpy()
        task_of = np.array([norm_task(t) for t in
                            task_by_node.reindex(node_of).to_numpy()], dtype=object)
        elig_of = elig_by_node.reindex(node_of).fillna(False).to_numpy().astype(bool)
        reason_of = {k: v.reindex(node_of).fillna(False).to_numpy().astype(bool)
                     for k, v in reason_by_node.items()}

        cands = load_cands(args.exports, s, args.run_fmt)
        qids = sorted(cands)
        gold_of = {q: int(cands[q][0][int(np.argmax(cands[q][1]))]) for q in qids}

        vis_m = f["visible"][0].numpy()
        vis_d = f["visible"][1].numpy()
        vis_a = f["visible_attr"].numpy().astype(np.float64)
        assert not np.isin(vis_d, np.asarray(qids, dtype=np.int64)).any(), \
            "a train/val-visible edge lands on a query dataset -- the prior would leak"

        by_td = defaultdict(lambda: defaultdict(list))
        for m, d, a in zip(vis_m, vis_d, vis_a):
            by_td[task_of[d]][int(d)].append((int(m), float(a)))
        by_task = {}
        for t, dd in by_td.items():
            sm, cn = defaultdict(float), defaultdict(int)
            for lst in dd.values():
                for m, a in lst:
                    sm[m] += a
                    cn[m] += 1
            pool = np.array(sorted(sm), dtype=np.int64)
            boost = np.array([(sm[int(m)] + 0.5 * SHRINK_K) / (cn[int(m)] + SHRINK_K)
                              for m in pool], dtype=np.float64)
            by_task[t] = (pool, boost)

        zm = np.load(os.path.join(exp, "z_m_eval.npy"), mmap_mode="r")
        zd = np.load(os.path.join(exp, "z_d_eval.npy"))
        universe = np.unique(vis_m)
        pos_in_u = np.full(zm.shape[0], -1, dtype=np.int64)
        pos_in_u[universe] = np.arange(universe.size)
        ZU = np.asarray(zm[universe], dtype=np.float32)
        ZU /= (np.linalg.norm(ZU, axis=1, keepdims=True) + 1e-12)
        ZD = zd / (np.linalg.norm(zd, axis=1, keepdims=True) + 1e-12)

        col = defaultdict(list)
        root_count = pd.Series(root_of).value_counts()
        vis_ds = set(int(x) for x in np.unique(vis_d))
        root_members = defaultdict(list)
        for i, r in enumerate(root_of):
            root_members[r].append(i)
        n_root_ge2, n_sib_visible = 0, 0

        for q in qids:
            t = task_of[q]
            gold = gold_of[q]
            r = root_of[q]
            if root_count.get(r, 0) >= 2:
                n_root_ge2 += 1
            if any(i in vis_ds for i in root_members[r] if i != q):
                n_sib_visible += 1

            pool, boost = by_task.get(t, (np.zeros(0, np.int64), np.zeros(0)))
            col["pool_size"].append(int(pool.size))
            gpos = np.searchsorted(pool, gold)
            in_pool = bool(gpos < pool.size and pool[gpos] == gold)
            col["in_pool"].append(float(in_pool))
            col["uniform10"].append(min(10.0, pool.size) / pool.size if in_pool else 0.0)
            if in_pool:
                pr, ties = rank_with_ties(boost, np.arange(pool.size), int(gpos))
                col["prior_rank"].append(pr)
                col["prior_ties"].append(ties)
                sc = ZU[pos_in_u[pool]] @ ZD[q]
                col["mips_pool_rank"].append(int((sc > sc[gpos]).sum()) + 1)
            else:
                col["prior_rank"].append(np.inf)
                col["prior_ties"].append(0)
                col["mips_pool_rank"].append(np.inf)

            gq = trigrams(node_of[q].split("\t")[0])
            sm, cn, dropped = defaultdict(float), defaultdict(int), 0
            for dd, lst in by_td.get(t, {}).items():
                gn = trigrams(node_of[dd].split("\t")[0])
                if len(gq & gn) / max(1, len(gq | gn)) > 0.5:
                    dropped += 1
                    continue
                for m, a in lst:
                    sm[m] += a
                    cn[m] += 1
            col["n_task_datasets"].append(len(by_td.get(t, {})))
            col["n_dropped_nodup"].append(dropped)
            if gold in sm:
                p2 = np.array(sorted(sm), dtype=np.int64)
                b2 = np.array([(sm[int(m)] + 0.5 * SHRINK_K) / (cn[int(m)] + SHRINK_K)
                               for m in p2], dtype=np.float64)
                g2 = int(np.searchsorted(p2, gold))
                pr2, _ = rank_with_ties(b2, np.arange(p2.size), g2)
                col["prior_rank_nodup"].append(pr2)
                col["in_pool_nodup"].append(1.0)
            else:
                col["prior_rank_nodup"].append(np.inf)
                col["in_pool_nodup"].append(0.0)

        pr = np.asarray(col["prior_rank"], dtype=float)
        ties = np.asarray(col["prior_ties"], dtype=float)
        exp_rank = pr + ties / 2.0
        per_seed[s] = {
            "n_queries": len(qids),
            "n_visible_edges": int(vis_m.size),
            "n_visible_models": int(universe.size),
            "prior_only": _at(col["prior_rank"]),
            "prior_only_expected_tie_break": {
                "gold@%d" % k: float((exp_rank <= k).mean()) for k in PRIOR_KS},
            "median_prior_ties": float(np.median(ties[np.isfinite(pr)])),
            "same_task_pool_mips": _at(col["mips_pool_rank"]),
            "pool_uniform_10": float(np.mean(col["uniform10"])),
            "ceiling_gold_in_pool": float(np.mean(col["in_pool"])),
            "pool_size": {"p10": float(np.percentile(col["pool_size"], 10)),
                          "p50": float(np.percentile(col["pool_size"], 50)),
                          "p90": float(np.percentile(col["pool_size"], 90)),
                          "mean": float(np.mean(col["pool_size"]))},
            "queries_with_task_prior": int(np.sum(np.asarray(col["pool_size"]) > 0)),
            "nodup": {**_at(col["prior_rank_nodup"]),
                      "ceiling_gold_in_pool": float(np.mean(col["in_pool_nodup"])),
                      "median_datasets_dropped": float(np.median(col["n_dropped_nodup"])),
                      "queries_gold_left_pool": int(np.sum(
                          (np.asarray(col["in_pool"]) == 1) &
                          (np.asarray(col["in_pool_nodup"]) == 0)))},
            "sibling": {
                "queries_in_root_ge2": n_root_ge2,
                "share_in_root_ge2": n_root_ge2 / len(qids),
                "queries_with_train_visible_sibling": n_sib_visible,
                "dataset_nodes_in_root_ge2": int(
                    root_count[root_count >= 2].sum()),
                "share_dataset_nodes_in_root_ge2": float(
                    root_count[root_count >= 2].sum() / len(root_of))},
            "gold_eligible": {
                "n_false": int((~elig_of[qids]).sum()),
                "share_false": float((~elig_of[qids]).mean()),
                "by_reason": {k: int((v[qids] & ~elig_of[qids]).sum())
                              for k, v in reason_of.items()}},
        }
        print("  [M3] seed=%d prior-only gold@10=%.4f (tie-broken %.4f) | pool+MIPS "
              "%.4f | uniform10 %.4f | ceiling %.4f | siblings visible %d"
              % (s, per_seed[s]["prior_only"]["gold@10"],
                 per_seed[s]["prior_only_expected_tie_break"]["gold@10"],
                 per_seed[s]["same_task_pool_mips"]["gold@10"],
                 per_seed[s]["pool_uniform_10"],
                 per_seed[s]["ceiling_gold_in_pool"], n_sib_visible), flush=True)
        del f, ZU, zm, zd

    payload = {"seconds": round(time.time() - t0, 1), "shrink_k": SHRINK_K,
               "per_seed": per_seed,
               "interval": {
                   "prior_only_gold@10": interval(
                       [per_seed[s]["prior_only"]["gold@10"] for s in per_seed]),
                   "prior_only_tie_broken_gold@10": interval(
                       [per_seed[s]["prior_only_expected_tie_break"]["gold@10"]
                        for s in per_seed]),
                   "same_task_pool_mips_gold@10": interval(
                       [per_seed[s]["same_task_pool_mips"]["gold@10"] for s in per_seed]),
                   "pool_uniform_10": interval(
                       [per_seed[s]["pool_uniform_10"] for s in per_seed]),
                   "ceiling": interval(
                       [per_seed[s]["ceiling_gold_in_pool"] for s in per_seed])}}
    merge(args.out, "task", payload)
    return payload


def _topk_update(bv, bi, sc, offset, k):
    v, i = torch.topk(sc, min(k, sc.size(0)), dim=0)
    i = i + offset
    if bv is None:
        return v, i
    cv = torch.cat([bv, v], 0)
    ci = torch.cat([bi, i], 0)
    v2, pos = torch.topk(cv, k, dim=0)
    return v2, torch.gather(ci, 0, pos)


def stage_ranks(args):
    t0 = time.time()
    meta = pd.read_parquet(args.model_meta,
                           columns=["in_snapshot", "pipeline_tag", "family"])
    ptag = np.array([norm_task(x) if isinstance(x, str) and x else ""
                     for x in meta["pipeline_tag"].to_numpy()], dtype=object)
    tag_vocab = {t: i + 1 for i, t in enumerate(sorted(set(ptag) & PIPELINE_TAGS))}
    tag_of_model = np.array([tag_vocab.get(t, 0) for t in ptag], dtype=np.int64)
    insnap = meta["in_snapshot"].to_numpy().astype(bool)
    fam = meta["family"].astype(str).to_numpy()

    nodes = pd.read_parquet(args.dataset_nodes, columns=["node", "task"])
    task_by_node = nodes.set_index(nodes["node"].astype(str))["task"]
    dev = args.device

    variants = ["mips", "sup_pool_all", "sup_pool_visible",
                "csls_test", "csls_train", "pipeline_tag"]
    per_seed = {}
    for s in args.seeds:
        exp = os.path.join(args.exports, args.run_fmt % s)
        f = split_facts(os.path.join(args.graphs, "hgraph_rf"), s)
        N = f["n_models"]
        deg_all = torch.bincount(f["data"][TRAINED_ON].edge_index[0],
                                 minlength=N).numpy()
        deg_vis = torch.bincount(f["visible"][0], minlength=N).numpy()
        udi = pd.read_parquet(os.path.join(exp, "dataset_ids.parquet")).sort_values("mappedID")
        node_of = udi["dataset"].astype(str).to_numpy()
        task_of = np.array([norm_task(t) for t in
                            task_by_node.reindex(node_of).to_numpy()], dtype=object)

        cands = load_cands(args.exports, s, args.run_fmt)
        qids = sorted(cands)
        Q = len(qids)
        gold_of = np.array([int(cands[q][0][int(np.argmax(cands[q][1]))]) for q in qids])

        zm = torch.from_numpy(np.load(os.path.join(exp, "z_m_eval.npy")))
        zd = torch.from_numpy(np.load(os.path.join(exp, "z_d_eval.npy")))
        zm = (zm / (zm.norm(dim=1, keepdim=True) + 1e-12)).to(torch.float32).to(dev)
        zd = (zd / (zd.norm(dim=1, keepdim=True) + 1e-12)).to(torch.float32).to(dev)

        rng = np.random.default_rng(s)
        train_ds = np.unique(f["ti"][1].numpy())
        samp = np.sort(rng.choice(train_ds, size=min(Q, train_ds.size), replace=False))
        rbar = {}
        for tag, qset in (("test", np.asarray(qids)), ("train", samp)):
            acc = torch.zeros(N, dtype=torch.float32, device=dev)
            zq = zd[torch.as_tensor(qset, dtype=torch.long, device=dev)]
            for ms in range(0, N, args.model_chunk):
                acc[ms:ms + args.model_chunk] += (
                    zm[ms:ms + args.model_chunk] @ zq.t()).sum(1)
            rbar[tag] = acc / float(len(qset))
        print("  [M6] seed=%d rbar(test) mean=%.4f max=%.4f | rbar(train) mean=%.4f"
              % (s, float(rbar["test"].mean()), float(rbar["test"].max()),
                 float(rbar["train"].mean())), flush=True)

        sup_all = torch.from_numpy(deg_all > 0).to(dev)
        sup_vis = torch.from_numpy(deg_vis > 0).to(dev)
        tag_m = torch.from_numpy(tag_of_model).to(dev)
        tag_q_np = np.array([tag_vocab.get(task_of[q], 0) for q in qids], dtype=np.int64)
        tag_ok = (tag_q_np > 0) & (tag_of_model[gold_of] == tag_q_np)
        tag_q = torch.from_numpy(np.where(tag_q_np > 0, tag_q_np, -1)).to(dev)

        gr = {v: np.zeros(Q, dtype=np.int64) for v in variants}
        t3 = {v: np.zeros(Q, dtype=np.int64) for v in variants}
        gp = {v: np.zeros(Q, dtype=np.int64) for v in variants}
        top10_i = np.zeros((Q, 10), dtype=np.int64)

        for qs in range(0, Q, args.query_chunk):
            blk = qids[qs:qs + args.query_chunk]
            b = len(blk)
            metas, flat_ids, flat_q = [], [], []
            for j, q in enumerate(blk):
                cand, acc = cands[q]
                gold, top3, near, ids = GM._probe_ids(cand, acc, GM.GAP_DELTA)
                metas.append((q, gold, top3, near, ids))
                flat_ids.append(ids)
                flat_q.append(np.full(ids.size, j))
            fi = torch.as_tensor(np.concatenate(flat_ids), dtype=torch.long, device=dev)
            fq = torch.as_tensor(np.concatenate(flat_q), dtype=torch.long, device=dev)
            zq = zd[torch.as_tensor(blk, dtype=torch.long, device=dev)]
            base = (zm[fi] * zq[fq]).sum(-1)
            probe = {"csls_test": 2 * base - rbar["test"][fi],
                     "csls_train": 2 * base - rbar["train"][fi]}
            tag_qb = tag_q[torch.as_tensor(
                np.arange(qs, qs + b), dtype=torch.long, device=dev)][fq]
            cnt = {v: torch.zeros_like(base, dtype=torch.long) for v in variants}
            pcols = torch.arange(fi.numel(), device=dev)
            bv = bi = None
            for ms in range(0, N, args.model_chunk):
                sc = zm[ms:ms + args.model_chunk] @ zq.t()
                c = sc.size(0)
                bv, bi = _topk_update(bv, bi, sc, ms, 10)
                scq = sc[:, fq]
                here = (fi >= ms) & (fi < ms + c)
                hi, hc = fi[here] - ms, pcols[here]
                cb = scq > base.unsqueeze(0)
                cb[hi, hc] = False
                cnt["mips"] += cb.sum(0)
                cnt["sup_pool_all"] += (cb & sup_all[ms:ms + c].unsqueeze(1)).sum(0)
                cnt["sup_pool_visible"] += (cb & sup_vis[ms:ms + c].unsqueeze(1)).sum(0)
                cnt["pipeline_tag"] += (
                    cb & (tag_m[ms:ms + c].unsqueeze(1) == tag_qb.unsqueeze(0))).sum(0)
                for v in ("csls_test", "csls_train"):
                    sv = 2 * scq - rbar[v.split("_")[1]][ms:ms + c].unsqueeze(1)
                    cv = sv > probe[v].unsqueeze(0)
                    cv[hi, hc] = False
                    cnt[v] += cv.sum(0)
                    del sv, cv
                del sc, scq, cb
            top10_i[qs:qs + b] = bi.t().cpu().numpy()
            host = {v: cnt[v].cpu().numpy() for v in variants}
            off = 0
            for j, (q, gold, top3, near, ids) in enumerate(metas):
                idx = {int(m): off + k for k, m in enumerate(ids)}
                for v in variants:
                    c = host[v]
                    gr[v][qs + j] = c[idx[gold]] + 1
                    t3[v][qs + j] = min(c[idx[int(m)]] for m in top3) + 1
                    gp[v][qs + j] = min(c[idx[int(m)]] for m in near) + 1
                off += ids.size
            print("    [M6] seed=%d %d/%d queries" % (s, min(qs + b, Q), Q), flush=True)

        def summarize(v, mask=None):
            m = np.ones(Q, dtype=bool) if mask is None else mask
            r = gr[v].astype(float)[m]
            row = {"n_queries": int(m.sum()), "median_gold_rank": float(np.median(r))}
            for k in RANK_KS:
                row["gold@%d" % k] = float((r <= k).mean())
            row["top3@10"] = float((t3[v][m] <= 10).mean())
            row["gold-gap@10"] = float((gp[v][m] <= 10).mean())
            return row

        rows = {v: summarize(v) for v in variants if v != "pipeline_tag"}
        rows["pipeline_tag"] = summarize("pipeline_tag", tag_ok)
        rows["mips_on_pipeline_tag_subset"] = summarize("mips", tag_ok)

        flat = top10_i.reshape(-1)
        counts = pd.Series(flat).value_counts()
        slots = {"n_slots": int(flat.size), "distinct_models": int(counts.size),
                 "top100_share": float(counts.iloc[:100].sum() / flat.size),
                 "max_appearances": int(counts.iloc[0]),
                 "max_appearance_share_of_queries": float(counts.iloc[0] / Q),
                 "supervised_share": float((deg_all[flat] > 0).mean()),
                 "in_snapshot_share": float(insnap[flat].mean()),
                 "mean_distinct_family_per_query": float(np.mean(
                     [len(set(fam[top10_i[i]])) for i in range(Q)]))}
        per_seed[s] = {"n_queries": Q, "variants": rows, "top10": slots,
                       "pipeline_tag_eligible_queries": int(tag_ok.sum()),
                       "queries_whose_task_is_an_hf_tag": int((tag_q_np > 0).sum())}
        print("  [M6] seed=%d mips gold@10=%.4f | sup-pool %.4f | csls(test) %.4f | "
              "csls(train) %.4f | %d distinct models in %d slots"
              % (s, rows["mips"]["gold@10"], rows["sup_pool_all"]["gold@10"],
                 rows["csls_test"]["gold@10"], rows["csls_train"]["gold@10"],
                 slots["distinct_models"], slots["n_slots"]), flush=True)

        if args.check:
            sub = {q: cands[q] for q in qids[:args.check_n]}
            agg, per = GM.from_embeddings_streaming(
                np.load(os.path.join(exp, "z_m_eval.npy")),
                np.load(os.path.join(exp, "z_d_eval.npy")), sub, device=dev,
                model_chunk=args.model_chunk)
            bad = [q for i, q in enumerate(qids[:args.check_n])
                   if per[q]["gold_rank"] != int(gr["mips"][i])]
            assert not bad, "rank mismatch vs global_metrics on %d queries" % len(bad)
            per_seed[s]["global_metrics_crosscheck"] = {
                "n_queries": len(sub), "exact_match": True,
                "gold@10_subset": agg["gold@10"]}
            print("  [M6] seed=%d cross-check vs global_metrics: %d/%d gold ranks "
                  "exactly equal" % (s, len(sub), len(sub)), flush=True)

        del zm, zd, rbar, f
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()

    payload = {"seconds": round(time.time() - t0, 1), "device": str(dev),
               "per_seed": per_seed,
               "interval": {v: interval([per_seed[s]["variants"][v]["gold@10"]
                                         for s in per_seed])
                            for v in per_seed[args.seeds[0]]["variants"]},
               "note": "every rank is #strictly-better + 1 with the probe's own "
                       "diagonal masked, identical to "
                       "global_metrics.from_embeddings_streaming"}
    merge(args.out, "ranks", payload)
    return payload


def main(argv=None):
    d = os.path.join(data_root(), "data1m")
    p = argparse.ArgumentParser(description="X1: the six fast-lever measurements")
    p.add_argument("--stage", required=True,
                   choices=["q", "splits", "task", "ranks", "all"])
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--exports", default=os.path.join(d, "exports_rf"))
    p.add_argument("--run-fmt", default=RUN_FMT,
                   help="printf-style export directory format keyed by seed")
    p.add_argument("--graphs", default=os.path.join(d, "graphs"))
    p.add_argument("--dataset-nodes",
                   default=os.path.join(d, "rf", "canon", "dataset_nodes_merged.parquet"))
    p.add_argument("--model-meta",
                   default=os.path.join(d, "utility_rf", "model_meta.parquet"))
    p.add_argument("--out", default=os.path.join(d, "metrics_rf"))
    p.add_argument("--device", default=None)
    p.add_argument("--model-chunk", type=int, default=100_000)
    p.add_argument("--query-chunk", type=int, default=128)
    p.add_argument("--check", action="store_true")
    p.add_argument("--check-n", type=int, default=200)
    args = p.parse_args(argv)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    stages = ["q", "splits", "task", "ranks"] if args.stage == "all" else [args.stage]
    for st in stages:
        {"q": stage_q, "splits": stage_splits, "task": stage_task,
         "ranks": stage_ranks}[st](args)
        print("[ok] stage %s" % st, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
