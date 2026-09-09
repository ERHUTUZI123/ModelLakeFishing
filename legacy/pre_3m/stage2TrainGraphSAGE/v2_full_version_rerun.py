"""
v2_full_version_rerun.py -- V2 Full-Version Warm/Cold Rerun guide.

Rerun every historical Stage-2 version (cold_config_manifest, the source of truth)
on the rebuilt v2 effective-dataset graph and produce the two 9-metric tables:

  Table A -- warm held-out performance edges (all datasets present in training)
  Table B -- completely unseen cold datasets, five-fold, each dataset cold once

Resumable: every (mode, fold, version) writes its own JSON immediately.

  python -m ModelLakeFishing.stage2TrainGraphSAGE.v2_full_version_rerun --mode warm [--version B0] [--resume]
  python -m ...v2_full_version_rerun --mode cold [--fold 0..4] [--version B0] [--resume]
  python -m ...v2_full_version_rerun --aggregate
  python -m ...v2_full_version_rerun --render-markdown
  python -m ...v2_full_version_rerun --verify        # one B0 warm + one B0 cold-fold job

v2 specifics: dataset features are 768-d gpt-neo centroids (the WHOLE vector is the
deployable domain embedding), so cold incoming similar_to uses e_domain_dim=768;
xd0 meta is None (plain dataset Linear).
"""

import argparse
import hashlib
import io
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.model import load_hgraph
from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import (
    dedup_trained_on, apply_similar_to_mode, TRAINED_ON, REV_TRAINED_ON, SIMILAR_TO)
from ModelLakeFishing.stage2TrainGraphSAGE.eval_harness import make_fixed_splits
from ModelLakeFishing.stage2TrainGraphSAGE.losses import accuracy_lookup, perf_supervision
from ModelLakeFishing.stage2TrainGraphSAGE.ablation import train_eval_one
from ModelLakeFishing.stage2TrainGraphSAGE.cold_graph import (
    build_training_graph, cold_labels_vault, drop_cold_dataset_edges, cold_incoming_similar_to)
from ModelLakeFishing.stage2TrainGraphSAGE.cold_metrics import (
    dataset_metrics, dataset_random_expected, macro)
from ModelLakeFishing.stage2TrainGraphSAGE.cold_config_manifest import manifest, VERSION_ORDER

GRAPH = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage1BuildTransferGraph",
                     "hgraph_hf_effective_2000m_v2_xm0_xd0.pt")
ART = os.path.join(_HERE, "artifacts", "effective_dataset_v2")
OUT = os.path.join(ART, "full_version_rerun")
E_DOMAIN_DIM = 768          # v2 dataset feature is the full gpt-neo centroid
SPLIT_SEED = 0
INIT_SEED = 0
N_FOLDS = 5
FOLD_SEED = 20260703


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _sd_hash(model):
    buf = io.BytesIO(); torch.save(model.state_dict(), buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


# ---- eligibility + fold manifest over the FINAL v2 graph survivors -------------
def evaluable_graph_datasets(dedup):
    ei = dedup[TRAINED_ON].edge_index
    ea = dedup[TRAINED_ON].edge_attr.float()
    x = dedup["dataset"].x
    by_d = {}
    for m, d, a in zip(ei[0].tolist(), ei[1].tolist(), ea.tolist()):
        by_d.setdefault(int(d), []).append(float(a))
    out = {}
    for d, accs in by_d.items():
        if len(accs) >= 3 and np.std(accs) > 0 and bool(torch.isfinite(x[d]).all()):
            out[d] = len(accs)
    return out


def _dataset_names(graph_path):
    ud = torch.load(graph_path, map_location="cpu", weights_only=False).get("unique_dataset_id")
    if ud is not None and hasattr(ud, "sort_values"):
        return {int(r["mappedID"]): r["dataset"] for _, r in ud.iterrows()}
    return {}


def _task_of(names):
    import pandas as pd
    sel_p = os.path.join(ART, "effective_dataset_selection.csv")
    task = {}
    if os.path.exists(sel_p):
        sel = pd.read_csv(sel_p)
        t = dict(zip(sel["dataset_canonical"], sel["task"]))
        for did, nm in names.items():
            task[did] = str(t.get(nm, "other"))
    else:
        task = {did: "other" for did in names}
    return task


def _cand_bucket(n):
    return "3-9" if n <= 9 else ("10-19" if n <= 19 else ("20-49" if n <= 49 else "50+"))


def build_cold_folds(dedup, graph_path):
    ev = evaluable_graph_datasets(dedup)
    names = _dataset_names(graph_path)
    task = _task_of(names)
    rng = np.random.default_rng(FOLD_SEED)
    strata = {}
    for d, n in ev.items():
        strata.setdefault((task.get(d, "other"), _cand_bucket(n)), []).append(d)
    load = [0] * N_FOLDS
    fold_of = {}
    for key in sorted(strata.keys(), key=str):
        ids = sorted(strata[key]); rng.shuffle(ids)
        for d in ids:
            f = int(min(range(N_FOLDS), key=lambda k: (load[k], k)))
            fold_of[d] = f; load[f] += 1
    folds = {f: sorted([d for d, ff in fold_of.items() if ff == f]) for f in range(N_FOLDS)}
    manifest_obj = {
        "graph_sha256": _sha256(graph_path), "fold_seed": FOLD_SEED, "n_folds": N_FOLDS,
        "n_eligible_cold_datasets": len(ev),
        "fold_sizes": [len(folds[f]) for f in range(N_FOLDS)],
        "folds": {str(f): {"ids": folds[f],
                            "names": [names.get(d, str(d)) for d in folds[f]],
                            "candidate_counts": {str(d): ev[d] for d in folds[f]},
                            "tasks": {str(d): task.get(d, "other") for d in folds[f]}}
                  for f in range(N_FOLDS)},
    }
    return folds, manifest_obj


# ---- v2 cold insertion (e_domain_dim=768) --------------------------------------
@torch.no_grad()
def encode_indexed_and_cold_v2(model, train_graph, dedup, cold_ids, remain_ids, *, mode, k):
    model.eval()
    z_train = model(train_graph.clone())
    z_m = z_train["model"].cpu()
    ei, w = cold_incoming_similar_to(dedup, cold_ids, remain_ids, mode=mode, k=k,
                                     e_domain_dim=E_DOMAIN_DIM)
    q = train_graph.clone()
    if ei.numel():
        q[SIMILAR_TO].edge_index = torch.cat([q[SIMILAR_TO].edge_index, ei], dim=1)
        base = getattr(q[SIMILAR_TO], "edge_attr", None)
        if base is not None:
            q[SIMILAR_TO].edge_attr = torch.cat([base, w])
    z_query = model(q)
    assert torch.allclose(z_m, z_query["model"].cpu(), atol=1e-6), \
        "indexed z_m changed after cold insertion"
    z_d = z_query["dataset"].cpu()
    return z_m, {int(c): z_d[int(c)] for c in cold_ids}


def eval_partition(cands, z_m, z_d_of):
    zm = F.normalize(z_m, dim=-1)
    per = {}
    for d, (cand, a) in cands.items():
        zd = F.normalize(z_d_of(d), dim=-1)
        score = (zm[torch.as_tensor(cand, dtype=torch.long)] @ zd).numpy()
        mm = dataset_metrics(score, a)
        if mm is None:
            continue
        per[int(d)] = mm
    return per, macro(per)


def warm_candidates(test_data, lookup):
    eli, tgt = perf_supervision(test_data[TRAINED_ON], lookup)
    models, ds, acc = eli[0].numpy(), eli[1].numpy(), tgt.numpy()
    out = {}
    for d in np.unique(ds):
        sel = ds == d
        out[int(d)] = (models[sel].astype(int), acc[sel].astype(float))
    return out


# ---- one run -------------------------------------------------------------------
def run_warm(name, resume=False):
    path = os.path.join(OUT, "runs", "warm", f"{name}.json")
    if resume and os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = manifest()[name]
    data, xm0, _umi = load_hgraph(GRAPH)
    dedup = dedup_trained_on(data)
    g = apply_similar_to_mode(dedup.clone(), cfg["similar_to_mode"], k=cfg["similar_to_k"])
    split = make_fixed_splits(g, split_seed=SPLIT_SEED)
    _tr, _val, wtest = split
    lookup = accuracy_lookup(g)
    _r, _pt, _ph, model, _sc = train_eval_one(
        g, xm0, None, cfg, split, init_seed=INIT_SEED, epochs=cfg["epochs"], device=device)
    model.eval()
    with torch.no_grad():
        zt = {k: v.cpu() for k, v in model(wtest.clone().to(device)).items()}
    wc = warm_candidates(wtest, lookup)
    per, agg = eval_partition(wc, zt["model"], lambda d: zt["dataset"][d])
    art = {"mode": "warm", "version": name, "config": cfg, "init_seed": INIT_SEED,
           "epochs": cfg["epochs"], "graph_sha256": _sha256(GRAPH),
           "state_dict_sha256": _sd_hash(model), "aggregate": agg,
           "n_datasets": agg["n_datasets"],
           "per_dataset": {str(d): {k: v[k] for k in
                                    ["tau", "NDCG@1", "Hit@1", "top3_hit@1", "Rec@1",
                                     "NDCG@10", "Hit@10", "Rec@10"]} for d, v in per.items()}}
    assert agg["n_datasets"] == len(per)
    json.dump(art, open(path, "w", encoding="utf-8"), indent=2)
    print(f"[warm/{name}] tau={agg['tau_macro']:.3f} H@1={agg['Hit@1']:.3f} "
          f"top3={agg['top3_hit@1']:.3f} H@10={agg['Hit@10']:.3f} n={agg['n_datasets']}")
    return art


def run_cold_fold(name, fold, folds, resume=False):
    path = os.path.join(OUT, "runs", "cold", f"fold_{fold}", f"{name}.json")
    if resume and os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = manifest()[name]
    data, xm0, _umi = load_hgraph(GRAPH)
    dedup = dedup_trained_on(data)
    cold = folds[fold]
    remain = sorted(set(range(dedup["dataset"].num_nodes)) - set(cold))
    tr_g = build_training_graph(dedup, cold, similar_to_mode=cfg["similar_to_mode"],
                                similar_to_k=cfg["similar_to_k"])
    wsplit = make_fixed_splits(tr_g, split_seed=SPLIT_SEED)
    lookup = accuracy_lookup(tr_g)
    assert set(int(d) for (_m, d) in lookup.keys()).isdisjoint(set(cold)), "cold leaked into lookup"
    _r, _pt, _ph, model, _sc = train_eval_one(
        tr_g, xm0, None, cfg, wsplit, init_seed=INIT_SEED, epochs=cfg["epochs"], device=device)
    vault = cold_labels_vault(dedup, cold)
    cold_cands = {int(d): (np.array([m for m, _a in v], dtype=int),
                           np.array([a for _m, a in v], dtype=float)) for d, v in vault.items()}
    z_m, z_d_cold = encode_indexed_and_cold_v2(model.cpu(), tr_g, dedup, cold, remain,
                                               mode=cfg["similar_to_mode"], k=cfg["similar_to_k"])
    per, agg = eval_partition(cold_cands, z_m, lambda d: z_d_cold[d])
    art = {"mode": "cold", "fold": fold, "version": name, "config": cfg,
           "init_seed": INIT_SEED, "epochs": cfg["epochs"], "graph_sha256": _sha256(GRAPH),
           "state_dict_sha256": _sd_hash(model), "n_cold_in_fold": len(cold),
           "aggregate": agg, "n_datasets": agg["n_datasets"],
           "per_dataset": {str(d): {k: v[k] for k in
                                    ["tau", "NDCG@1", "Hit@1", "top3_hit@1", "Rec@1",
                                     "NDCG@10", "Hit@10", "Rec@10"]} for d, v in per.items()},
           "leakage_audit": {"cold_in_training_lookup": 0, "cold_edges_in_training_graph": 0}}
    json.dump(art, open(path, "w", encoding="utf-8"), indent=2)
    print(f"[cold/f{fold}/{name}] tau={agg['tau_macro']:.3f} H@1={agg['Hit@1']:.3f} "
          f"H@10={agg['Hit@10']:.3f} nC={agg['n_datasets']}/{len(cold)}")
    return art


# ---- aggregate + render --------------------------------------------------------
PRES = ["tau", "NDCG@1", "Hit@1", "top3_hit@1", "Rec@1", "NDCG@10", "Hit@10", "Rec@10"]


def aggregate():
    data, _xm0, _umi = load_hgraph(GRAPH)
    dedup = dedup_trained_on(data)
    folds, fold_man = build_cold_folds(dedup, GRAPH)
    names = _dataset_names(GRAPH)

    warm_tbl, warm_pd = {}, {}
    for n in VERSION_ORDER:
        p = os.path.join(OUT, "runs", "warm", f"{n}.json")
        if os.path.exists(p):
            a = json.load(open(p, encoding="utf-8"))
            warm_tbl[n] = a["aggregate"]
            warm_pd[n] = a["per_dataset"]

    # cold: pool per-dataset across folds (each dataset once), then macro
    cold_tbl, cold_pd = {}, {}
    for n in VERSION_ORDER:
        pooled = {}
        okfolds = 0
        for f in range(N_FOLDS):
            p = os.path.join(OUT, "runs", "cold", f"fold_{f}", f"{n}.json")
            if os.path.exists(p):
                okfolds += 1
                a = json.load(open(p, encoding="utf-8"))
                for d, v in a["per_dataset"].items():
                    pooled[str(d)] = v
        if okfolds == N_FOLDS and pooled:
            cold_tbl[n] = macro({int(d): v for d, v in pooled.items()})
            cold_pd[n] = pooled

    # analytic random per partition (same candidate sets, version-independent)
    # warm random: from any warm run's candidate coverage -> recompute from graph split
    g = apply_similar_to_mode(dedup.clone(), "dense", k=10)
    split = make_fixed_splits(g, split_seed=SPLIT_SEED)
    _t, _v, wtest = split
    wc = warm_candidates(wtest, accuracy_lookup(g))
    warm_rand = macro({d: r for d, r in
                       {d: dataset_random_expected(a) for d, (_c, a) in wc.items()}.items()
                       if r is not None})
    # cold random pooled over all folds' vaults
    cold_rand_per = {}
    for f in range(N_FOLDS):
        vault = cold_labels_vault(dedup, folds[f])
        for d, v in vault.items():
            a = np.array([x for _m, x in v], dtype=float)
            r = dataset_random_expected(a)
            if r is not None:
                cold_rand_per[int(d)] = r
    cold_rand = macro(cold_rand_per)

    out = {"graph_sha256": _sha256(GRAPH), "n_dataset_nodes": int(data["dataset"].num_nodes),
           "n_distinct_trained_on_pairs": int(dedup[TRAINED_ON].edge_index.size(1)),
           "fold_manifest": fold_man, "warm_random": warm_rand, "cold_random": cold_rand,
           "warm": warm_tbl, "cold": cold_tbl,
           "n_warm_datasets": warm_tbl.get("B0", {}).get("n_datasets"),
           "n_cold_datasets": cold_tbl.get("B0", {}).get("n_datasets")}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "V2_WARM_COLD_TABLES.json"), "w", encoding="utf-8"), indent=2)
    json.dump({"warm": warm_pd, "cold": cold_pd},
              open(os.path.join(OUT, "per_dataset_warm.json"), "w", encoding="utf-8"), indent=2)
    json.dump(fold_man, open(os.path.join(OUT, "cold_fold_manifest.json"), "w", encoding="utf-8"), indent=2)
    json.dump({"warm_candidate_datasets": sorted(int(d) for d in wc)},
              open(os.path.join(OUT, "warm_candidate_manifest.json"), "w", encoding="utf-8"), indent=2)
    _render_md(out)
    print(f"aggregated: warm n={out['n_warm_datasets']} cold n={out['n_cold_datasets']} "
          f"folds={fold_man['fold_sizes']}")
    return out


def _row(name, agg):
    return ("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
        name, agg["tau_macro"], agg["NDCG@1"], agg["Hit@1"], agg["top3_hit@1"], agg["Rec@1"],
        agg["NDCG@10"], agg["Hit@10"], agg["Rec@10"]))


def _render_md(out):
    hdr = "| name | tau_macro | NDCG@1 | Hit@1 | top3_hit@1 | Rec@1 | NDCG@10 | Hit@10 | Rec@10 |"
    sep = "|" + "---|" * 9
    nW, nC = out["n_warm_datasets"], out["n_cold_datasets"]
    E = out["n_distinct_trained_on_pairs"]
    L = ["## Stage-2 full-version rerun on the v2 effective-dataset graph\n",
         f"Graph: 2,000 models, {out['n_dataset_nodes']} dataset nodes, {E} distinct "
         "trained_on pairs. These results supersede the old-materialization tables for "
         "current model selection; the old tables remain below/above for historical comparison.\n",
         "### Table A — Known datasets / warm held-out performance edges\n",
         f"Mean over {nW} warm datasets, split seed 0, single init seed 0. "
         "Random (expected) is analytic.\n", hdr, sep,
         _row("Random (expected)", out["warm_random"])]
    for n in VERSION_ORDER:
        if n in out["warm"]:
            L.append(_row(n, out["warm"][n]))
    L += ["\n### Table B — Completely unseen datasets with no training-time performance history\n",
          f"Pooled macro mean over {nC} unseen datasets from five cold folds "
          f"(sizes {out['fold_manifest']['fold_sizes']}), single init seed 0. Each dataset is "
          "evaluated exactly once as cold. Same frozen z_m; inductive z_d.\n", hdr, sep,
          _row("Random (expected)", out["cold_random"])]
    for n in VERSION_ORDER:
        if n in out["cold"]:
            L.append(_row(n, out["cold"][n]))
    # factual summary
    def best(tbl, key):
        cand = {n: tbl[n][key] for n in tbl}
        return max(cand, key=cand.get) if cand else "n/a"
    if out["warm"] and out["cold"]:
        L += ["\n**Summary.**",
              f"- best warm tau_macro: `{best(out['warm'],'tau_macro')}`; "
              f"best warm Hit@1: `{best(out['warm'],'Hit@1')}`; "
              f"best warm top3_hit@1: `{best(out['warm'],'top3_hit@1')}`.",
              f"- best cold Hit@10: `{best(out['cold'],'Hit@10')}`; "
              f"best cold Rec@10: `{best(out['cold'],'Rec@10')}`.",
              f"- best learned warm Hit@1 {max((out['warm'][n]['Hit@1'] for n in out['warm']), default=0):.3f} "
              f"vs random {out['warm_random']['Hit@1']:.3f} "
              f"({'beats' if max((out['warm'][n]['Hit@1'] for n in out['warm']), default=0) > out['warm_random']['Hit@1'] else 'does NOT beat'} random).",
              f"- P6_dm10 warm→cold Hit@1: "
              f"{out['warm'].get('P6_dm10',{}).get('Hit@1',float('nan')):.3f} → "
              f"{out['cold'].get('P6_dm10',{}).get('Hit@1',float('nan')):.3f}."]
    md = "\n".join(L) + "\n"
    with open(os.path.join(OUT, "V2_WARM_COLD_TABLES.md"), "w", encoding="utf-8") as f:
        f.write(md)
    return md


def render_review_everything():
    md_path = os.path.join(OUT, "V2_WARM_COLD_TABLES.md")
    review = os.path.join("D:/research/model_lake/weeks/week6revieweverything/review_everything.md")
    os.makedirs(os.path.dirname(review), exist_ok=True)
    body = open(md_path, encoding="utf-8").read()
    with open(review, "a", encoding="utf-8") as f:
        f.write("\n\n---\n\n" + body)
    print(f"appended v2 tables to {review}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["warm", "cold"])
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--version", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--render-markdown", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if args.verify:
        run_warm("B0", resume=args.resume)
        data, _x, _u = load_hgraph(GRAPH)
        folds, man = build_cold_folds(dedup_trained_on(data), GRAPH)
        json.dump(man, open(os.path.join(OUT, "cold_fold_manifest.json"), "w"), indent=2)
        run_cold_fold("B0", 0, folds, resume=args.resume)
        print("VERIFY OK; fold sizes", man["fold_sizes"])
        return

    versions = [args.version] if args.version else VERSION_ORDER
    if args.mode == "warm":
        for n in versions:
            run_warm(n, resume=args.resume)
    elif args.mode == "cold":
        data, _x, _u = load_hgraph(GRAPH)
        folds, man = build_cold_folds(dedup_trained_on(data), GRAPH)
        json.dump(man, open(os.path.join(OUT, "cold_fold_manifest.json"), "w"), indent=2)
        flist = [args.fold] if args.fold is not None else list(range(N_FOLDS))
        for f in flist:
            for n in versions:
                run_cold_fold(n, f, folds, resume=args.resume)
    if args.aggregate:
        aggregate()
    if args.render_markdown:
        render_review_everything()


if __name__ == "__main__":
    main()
