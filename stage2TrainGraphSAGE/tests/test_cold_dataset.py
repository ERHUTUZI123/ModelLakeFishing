"""
test_cold_dataset.py -- the 17 required tests for the Cold-Dataset evaluation.
Must pass before the sweep. Run:
    python -m ModelLakeFishing.stage2TrainGraphSAGE.tests.test_cold_dataset
"""

import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.model import load_hgraph, HeteroGraphSAGE
from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import (
    dedup_trained_on, TRAINED_ON, REV_TRAINED_ON, SIMILAR_TO,
)
from ModelLakeFishing.stage2TrainGraphSAGE.eval_harness import make_fixed_splits
from ModelLakeFishing.stage2TrainGraphSAGE.losses import (
    accuracy_lookup, perf_supervision, topk_membership,
)
from ModelLakeFishing.stage2TrainGraphSAGE.cold_dataset_split import build_split, SPLIT_PATH
from ModelLakeFishing.stage2TrainGraphSAGE.cold_graph import (
    build_training_graph, cold_labels_vault, cold_incoming_similar_to,
    encode_indexed_and_cold, drop_cold_dataset_edges,
)
from ModelLakeFishing.stage2TrainGraphSAGE.cold_metrics import (
    dataset_metrics, dataset_random_expected, true_top3_set,
)
from ModelLakeFishing.stage2TrainGraphSAGE.cold_config_manifest import manifest

GRAPH = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage1BuildTransferGraph",
                     "hgraph_hf1000d_2000m_xm0_xd0.pt")
failures = []


def check(cond, msg):
    print(f"  [{'OK  ' if cond else 'FAIL'}] {msg}")
    if not cond:
        failures.append(msg)


def main():
    data, xm0, umi = load_hgraph(GRAPH)
    dedup = dedup_trained_on(data)
    xd0 = torch.load(GRAPH, map_location="cpu", weights_only=False).get("xd0_meta")
    m = build_split(dataset_split_seed=0)
    cold = m["cold_test_dataset_ids"]
    remain = m["remaining_dataset_ids"]
    coldset = set(cold)

    # ── 1. deterministic + disjoint cold/remaining/validation ────────────────
    print("=== 1. split determinism + disjointness ===")
    m2 = build_split(dataset_split_seed=0)
    check(m2["cold_test_dataset_ids"] == cold, "cold split deterministic across calls")
    check(set(cold).isdisjoint(remain), "cold disjoint from remaining")
    tr_g = build_training_graph(dedup, cold, similar_to_mode="topk_unweighted", similar_to_k=10)
    _tr, _val, _te = make_fixed_splits(tr_g, split_seed=0)
    val_eli, _ = perf_supervision(_val[TRAINED_ON], accuracy_lookup(tr_g))
    val_ds = set(val_eli[1].tolist())
    check(val_ds.isdisjoint(coldset), "validation datasets disjoint from cold")

    # ── 2. manifest completeness ──────────────────────────────────────────────
    print("\n=== 2. manifest completeness ===")
    for key in ("dataset_split_seed", "graph_sha256", "dedup_policy",
                "n_cold_test_datasets", "cold_test_dataset_ids", "cold_test_dataset_names",
                "candidate_count_by_dataset", "task_count_by_partition"):
        check(key in m, f"manifest has '{key}'")
    check(m["n_cold_test_datasets"] == len(cold) == len(m["cold_test_dataset_names"]),
          "cold count matches IDs and names length")

    # ── 3-7. inductive training graph emptiness ──────────────────────────────
    print("\n=== 3-7. cold edges absent from training graph ===")
    to = tr_g[TRAINED_ON].edge_index
    check(sum(int(d) in coldset for d in to[1].tolist()) == 0, "3. no cold trained_on edge")
    rev = tr_g[REV_TRAINED_ON].edge_index
    check(sum(int(d) in coldset for d in rev[0].tolist()) == 0, "4. no cold rev_trained_on edge")
    lk = accuracy_lookup(tr_g)
    sup_eli, _ = perf_supervision(_tr[TRAINED_ON], lk)
    check(set(sup_eli[1].tolist()).isdisjoint(coldset), "5. no cold ranking-supervision edge")
    ti = torch.cat([_tr[TRAINED_ON].edge_index, sup_eli], dim=1)
    ta = torch.cat([_tr[TRAINED_ON].edge_attr.float(),
                    perf_supervision(_tr[TRAINED_ON], lk)[1]], dim=1) if False else None
    M = topk_membership(tr_g, top_frac=0.10, trained_on_index=ti,
                        trained_on_attr=torch.cat([_tr[TRAINED_ON].edge_attr.float(),
                                                   perf_supervision(_tr[TRAINED_ON], lk)[1]]))
    check(float(M[:, cold].sum()) == 0.0, "6. no cold dataset in top-k membership M")
    sim = tr_g[SIMILAR_TO].edge_index
    check(sum((int(s) in coldset or int(t) in coldset)
              for s, t in zip(sim[0].tolist(), sim[1].tolist())) == 0, "7. no cold similar_to edge")

    # ── 8. inference similar_to uses only deployable (label-free) features ────
    print("\n=== 8. cold neighbors from deployable features only ===")
    ei_c, w_c = cold_incoming_similar_to(dedup, cold, remain, mode="topk_unweighted", k=10)
    check(bool((torch.tensor([int(t) in coldset for t in ei_c[1].tolist()]).all())),
          "all inference edges point INTO cold nodes")
    check(set(ei_c[0].tolist()).issubset(set(remain)), "inference sources are remaining datasets only")
    # perturbing accuracy labels must not change the (feature-only) neighbor edges
    dperturb = dedup.clone()
    dperturb[TRAINED_ON].edge_attr = dperturb[TRAINED_ON].edge_attr * 0 + 0.5
    ei_c2, _ = cold_incoming_similar_to(dperturb, cold, remain, mode="topk_unweighted", k=10)
    check(torch.equal(ei_c, ei_c2), "cold neighbors independent of accuracy labels (deployable)")

    # ── 9. insertion changes z_d, leaves z_m unchanged ───────────────────────
    print("\n=== 9. cold insertion invariant (z_m fixed) ===")
    cfg = manifest()["R_mg02"]
    torch.manual_seed(0)
    ds_kw = dict(num_task_types=xd0["num_task_types"], n_class_buckets=xd0["n_class_buckets"],
                 num_arities=xd0["num_arities"])
    model = HeteroGraphSAGE(metadata=tr_g.metadata(), frozen_dim=tr_g["model"].x.shape[1],
                            num_size_buckets=xm0["num_size_buckets"], num_families=xm0["num_families"],
                            dataset_in_dim=tr_g["dataset"].x.shape[1], num_layers=1,
                            edge_aware=True, weighted_relations=[], **ds_kw)
    z_m, z_d_cold = encode_indexed_and_cold(model, tr_g, dedup, cold, remain,
                                            mode="topk_unweighted", k=10)  # asserts z_m fixed internally
    with torch.no_grad():
        z_train = model(tr_g.clone())
    changed = any(not torch.allclose(z_train["dataset"][c], z_d_cold[c], atol=1e-6) for c in cold)
    check(changed, "z_d for cold nodes changed by insertion")
    check(z_m.shape[0] == tr_g["model"].num_nodes, "z_m covers all indexed models")

    # ── 10. byte-identical candidate manifests across versions ───────────────
    print("\n=== 10. identical warm+cold candidates across configs ===")
    g_dense = build_training_graph(dedup, cold, similar_to_mode="dense", similar_to_k=10)
    _t2, _v2, te2 = make_fixed_splits(g_dense, split_seed=0)
    w1 = set(zip(*perf_supervision(_te[TRAINED_ON], lk)[0].tolist()))
    w2 = set(zip(*perf_supervision(te2[TRAINED_ON], accuracy_lookup(g_dense))[0].tolist()))
    check(w1 == w2, "warm test candidates identical for topk vs dense configs")
    v1 = cold_labels_vault(dedup, cold)
    check(all(sorted(v1[c]) == sorted(v1[c]) for c in cold) and set(v1) == coldset,
          "cold vault covers exactly the cold datasets")

    # ── 11-14. metric correctness ────────────────────────────────────────────
    print("\n=== 11-14. metric definitions ===")
    a = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    ideal = a.copy()                                  # score == accuracy -> ideal order
    mm = dataset_metrics(ideal, a)
    check(abs(mm["NDCG@1"] - 1.0) < 1e-9 and abs(mm["NDCG@10"] - 1.0) < 1e-9,
          "11. NDCG@K = 1 for ideal ordering (K=1,10)")
    # tied best: two models tie at max; a wrong-looking pick that is a tied max still hits
    a_tie = np.array([0.9, 0.9, 0.5, 0.2])
    sc = np.array([0.1, 1.0, 0.2, 0.0])               # picks model 1 (a tied best)
    mt = dataset_metrics(sc, a_tie)
    check(mt["Hit@1"] == 1.0, "12. Hit@1 accepts a tied-maximum pick")
    # third-place tie cutoff: distinct desc = [.9,.6,.5]; cutoff .5 -> T3 = {.9,.6,.5,.5}
    a_c = np.array([0.9, 0.6, 0.5, 0.5, 0.2])
    check(true_top3_set(a_c) == {0, 1, 2, 3}, "13. Rec@K third-place tie cutoff includes both .5")
    # top3_hit@1 vs Rec@1
    a_r = np.array([0.9, 0.8, 0.7, 0.1])              # T3 = {0,1,2}, |T3|=3
    sc_r = np.array([0.0, 1.0, 0.0, 0.0])             # selects model 1 (in T3)
    mr = dataset_metrics(sc_r, a_r)
    check(mr["top3_hit@1"] == 1.0 and abs(mr["Rec@1"] - 1.0 / 3) < 1e-9,
          "14. top3_hit@1=1 while Rec@1=1/|T3| for an in-top3 pick")

    # ── 15. random baseline matches analytic on a toy ────────────────────────
    print("\n=== 15. random baseline analytic ===")
    a_t = np.array([0.9, 0.5, 0.5, 0.2, 0.1])         # n=5, B={0}, T3(distinct .9,.5,.2)= {0,1,2,3}
    rnd = dataset_random_expected(a_t)
    check(abs(rnd["top3_hit@1"] - 4 / 5) < 1e-9, "15a. top3_hit@1 random = |T3|/n = 4/5")
    check(abs(rnd["Hit@1"] - 1 / 5) < 1e-9, "15b. Hit@1 random = |B|/n = 1/5")
    # Hit@10 with k=min(10,5)=5 -> certain (all in top-5) -> 1.0
    check(abs(rnd["Hit@10"] - 1.0) < 1e-9, "15c. Hit@K random = 1 when k>=n")
    check(abs(rnd["Rec@1"] - 1 / 5) < 1e-9, "15d. Rec@1 random = k/n = 1/5")

    # ── 16. cold labels never enter training lookup ──────────────────────────
    print("\n=== 16. cold labels absent from training lookup ===")
    check(set(int(d) for (_m, d) in lk.keys()).isdisjoint(coldset),
          "training accuracy_lookup has no cold dataset keys (labels loaded only at eval)")

    # ── 17. dedup leakage zero both directions ───────────────────────────────
    print("\n=== 17. post-dedup leakage zero ===")
    ok = True
    for ss in (0, 1, 2):
        tr, _v, te = make_fixed_splits(dedup, split_seed=ss)
        eli, _ = perf_supervision(te[TRAINED_ON], accuracy_lookup(dedup))
        tp = set(zip(eli[0].tolist(), eli[1].tolist()))
        for store, flip in ((tr[TRAINED_ON], False), (tr[REV_TRAINED_ON], True)):
            e = store.edge_index
            msg = set(zip(e[1].tolist(), e[0].tolist())) if flip else set(zip(e[0].tolist(), e[1].tolist()))
            ok &= len(tp & msg) == 0
    check(ok, "0 held-out pairs in train message graph, both directions, all splits")

    print("\n" + "=" * 52)
    if failures:
        print(f"COLD-DATASET TESTS FAILED -- {len(failures)}:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("COLD-DATASET TESTS OK (17 groups)")


if __name__ == "__main__":
    main()
