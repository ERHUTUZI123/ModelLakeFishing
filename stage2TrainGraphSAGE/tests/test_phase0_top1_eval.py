"""
test_phase0_top1_eval.py -- Top-1/global guide Phase 0 required tests:

  1. observed_hit@1 is NOT automatically 1 when n_candidates <= 10;
  2. the global universe contains all 2,000 model IDs exactly once;
  3. the held-out gold model is present in the global universe;
  4. perturbing an unlabeled distractor embedding changes global metrics
     WITHOUT changing observed metrics;
  5. the held-out target edge is absent in both message directions
     (on the DEDUPED graph -- fails on the raw graph, which is the provenance
     bug this phase resolves);
  6. exact and HNSW top-1 agree under the configured fidelity test.

Run:  python -m ModelLakeFishing.stage2TrainGraphSAGE.tests.test_phase0_top1_eval
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.model import load_hgraph
from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import (
    dedup_trained_on, TRAINED_ON, REV_TRAINED_ON,
)
from ModelLakeFishing.stage2TrainGraphSAGE.eval_harness import make_fixed_splits
from ModelLakeFishing.stage2TrainGraphSAGE.losses import accuracy_lookup, perf_supervision
from ModelLakeFishing.stage2TrainGraphSAGE.top1_eval import five_metric_eval, aggregate
from ModelLakeFishing.stage2TrainGraphSAGE.top1_audit import candidates

GRAPH = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage1BuildTransferGraph",
                     "hgraph_hf1000d_2000m_xm0_xd0.pt")
failures = []


def check(cond, msg):
    print(f"  [{'OK  ' if cond else 'FAIL'}] {msg}")
    if not cond:
        failures.append(msg)


def synthetic(n_pool=20, dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    z_m = F.normalize(torch.randn(n_pool, dim, generator=g), dim=-1)
    z_d = F.normalize(torch.randn(1, dim, generator=g), dim=-1)
    return {"model": z_m, "dataset": z_d}


def main():
    # ── 1. observed_hit@1 not automatically 1 for small candidate lists ──────
    print("=== 1. small-candidate hit@1 can be 0 ===")
    z = synthetic()
    # 4 candidates; force the WORST-accuracy candidate to align best with z_d
    zd = z["dataset"][0]
    cand = np.array([0, 1, 2, 3])
    z["model"][0] = zd.clone()                        # model 0 scores highest
    acc = np.array([0.1, 0.5, 0.7, 0.9])              # ...but is the worst
    per = five_metric_eval(z, {0: (cand, acc)})
    check(per[0]["observed_hit1"] == 0.0 and per[0]["n_candidates"] <= 10,
          "hit@1 == 0 with 4 candidates when selection is wrong")
    check(per[0]["top3_hit1"] == 0.0, "top3_hit@1 == 0 (selected is rank-4 by accuracy)")
    check(abs(per[0]["regret1"] - 0.8) < 1e-9, "regret@1 = 0.9 - 0.1 = 0.8")

    # ── 2+3. global universe = every model exactly once, gold present ────────
    print("\n=== 2+3. global universe integrity ===")
    zz = synthetic(n_pool=2000)
    per = five_metric_eval(zz, {0: (np.array([5, 6, 7]), np.array([0.2, 0.9, 0.5]))})
    r = per[0]
    check(zz["model"].shape[0] == 2000, "global universe has exactly 2000 embeddings")
    check(1 <= r["gold_rank"] <= 2000, "gold rank within [1, 2000] (gold present)")
    check(r["gold_model_id"] == 6, "gold = highest-accuracy candidate")
    # rank recomputed independently
    s = (F.normalize(zz["model"], dim=-1) @ F.normalize(zz["dataset"], dim=-1)[0]).numpy()
    check(r["gold_rank"] == int((s > s[6]).sum()) + 1, "gold rank matches independent recompute")

    # ── 4. unlabeled distractor moves global, not observed ───────────────────
    print("\n=== 4. distractor perturbation ===")
    z1 = synthetic(n_pool=50, seed=1)
    cand = np.array([1, 2, 3, 4])
    acc = np.array([0.3, 0.9, 0.6, 0.2])
    p1 = five_metric_eval(z1, {0: (cand, acc)})[0]
    # pick the LOWEST-scoring non-candidate as the distractor, so lifting it to
    # the top adds exactly one model above gold (deterministic rank shift)
    s0 = (F.normalize(z1["model"], dim=-1) @ F.normalize(z1["dataset"], dim=-1)[0]).numpy()
    non_cand = [i for i in range(50) if i not in set(cand.tolist())]
    distractor = int(min(non_cand, key=lambda i: s0[i]))
    z2 = {"model": z1["model"].clone(), "dataset": z1["dataset"].clone()}
    z2["model"][distractor] = z1["dataset"][0].clone()   # unlabeled distractor -> top
    p2 = five_metric_eval(z2, {0: (cand, acc)})[0]
    check(p1["observed_hit1"] == p2["observed_hit1"] and p1["regret1"] == p2["regret1"]
          and p1["selected_model_id"] == p2["selected_model_id"],
          "observed metrics unchanged by distractor")
    check(p2["gold_rank"] == p1["gold_rank"] + 1, "global gold rank worsened by exactly 1")

    # ── 5. held-out target edge absent both directions (deduped graph) ──────
    print("\n=== 5. leakage: held-out pair absent from message graphs (dedup) ===")
    data, _xm0, _umi = load_hgraph(GRAPH)
    raw_rows = data[TRAINED_ON].edge_index.size(1)
    dd = dedup_trained_on(data)
    n_pairs = dd[TRAINED_ON].edge_index.size(1)
    check(raw_rows == 12205 and n_pairs == 7056,
          f"dedup: {raw_rows} rows -> {n_pairs} distinct pairs")
    ok_all = True
    for ss in (0, 1, 2):
        tr, _v, te = make_fixed_splits(dd, split_seed=ss)
        lookup = accuracy_lookup(dd)
        eli, _ = perf_supervision(te[TRAINED_ON], lookup)
        tp = set(zip(eli[0].tolist(), eli[1].tolist()))
        for store, flip in ((tr[TRAINED_ON], False), (te[TRAINED_ON], False),
                            (tr[REV_TRAINED_ON], True), (te[REV_TRAINED_ON], True)):
            e = store.edge_index
            msg = set(zip(e[1].tolist(), e[0].tolist())) if flip else set(zip(e[0].tolist(), e[1].tolist()))
            ok_all &= len(tp & msg) == 0
    check(ok_all, "0 held-out pairs in any message graph, both directions, all splits")

    # ── 6. exact vs HNSW top-1 agreement ─────────────────────────────────────
    print("\n=== 6. exact vs HNSW top-1 ===")
    try:
        import faiss
        zz = synthetic(n_pool=2000, seed=2)
        zm = F.normalize(zz["model"], dim=-1).numpy().astype("float32")
        qs = F.normalize(torch.randn(40, zm.shape[1]), dim=-1).numpy().astype("float32")
        index = faiss.IndexHNSWFlat(zm.shape[1], 32)
        index.hnsw.efConstruction = 200
        index.add(zm)
        index.hnsw.efSearch = 256
        _, ann = index.search(qs, 1)
        exact = np.argmax(qs @ zm.T, axis=1)
        agree = float(np.mean(ann[:, 0] == exact))
        check(agree == 1.0, f"HNSW top-1 == exact top-1 for all queries (agree={agree:.3f})")
    except ImportError:
        check(False, "faiss not installed (required for fidelity test)")

    print("\n" + "=" * 52)
    if failures:
        print(f"PHASE 0 TOP1 EVAL TESTS FAILED -- {len(failures)}:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("PHASE 0 TOP1 EVAL TESTS OK")


if __name__ == "__main__":
    main()
