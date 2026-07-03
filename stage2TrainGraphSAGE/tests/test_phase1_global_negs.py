"""
test_phase1_global_negs.py -- Top-1/global guide Phase 1 required tests:

  - incompatible pools exclude compatible AND unobserved models;
  - datasets with unknown task type (id 0) get no incompatible negatives;
  - known-low pools exclude the dataset's top-performer positives;
  - unobserved-but-compatible models never appear in ANY pool
    (missing performance is not evidence of poor performance);
  - the sampled-softmax loss raises s(d, positive) above s(d, negative).

Run:  python -m ModelLakeFishing.stage2TrainGraphSAGE.tests.test_phase1_global_negs
"""

import os
import sys

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.losses import (
    build_global_negative_pools, global_retrieval_loss,
)

failures = []


def check(cond, msg):
    print(f"  [{'OK  ' if cond else 'FAIL'}] {msg}")
    if not cond:
        failures.append(msg)


def main():
    torch.manual_seed(0)
    # synthetic universe: 8 models, 4 datasets
    #   d0: task 1   d1: task 2   d2: task 0 (unknown)   d3: task 1
    task = torch.tensor([1, 2, 0, 1])
    # train-visible edges (model, dataset, acc):
    #   m0->d0 .9  m1->d0 .2  m2->d0 .5      (d0: m0 top, m1 low)
    #   m3->d1 .8  m4->d1 .3                 (d1: task 2)
    #   m5->d2 .7                            (d2: unknown task)
    #   m6, m7: UNOBSERVED anywhere
    ti = torch.tensor([[0, 1, 2, 3, 4, 5],
                       [0, 0, 0, 1, 1, 2]])
    ta = torch.tensor([0.9, 0.2, 0.5, 0.8, 0.3, 0.7])
    M = torch.zeros(8, 4)
    M[0, 0] = 1.0                                  # d0 positive: m0
    M[3, 1] = 1.0                                  # d1 positive: m3

    # ── pools: incompatible only (G1) ────────────────────────────────────────
    print("=== pool construction (incompatible only) ===")
    pools, stats = build_global_negative_pools(task, ti, ta, M)
    # d0 (task 1): incompatible = models observed ONLY on other tasks = {m3,m4 (task2), m5 (task0)}
    check(set(pools[0].tolist()) == {3, 4, 5},
          f"d0 pool = observed models w/o task-1 exposure (got {pools[0].tolist()})")
    # unobserved m6, m7 never appear anywhere
    all_pool = set(x for p in pools.values() for x in p.tolist())
    check(6 not in all_pool and 7 not in all_pool,
          "unobserved models never appear in any pool (missing != negative)")
    # compatible observed models excluded: m1, m2 observed on d0 (task 1) -> not in d0 pool
    check(1 not in pools[0].tolist() and 2 not in pools[0].tolist(),
          "task-compatible observed models excluded from incompatible pool")
    # d2 (unknown task) -> no incompatible pool
    check(2 not in pools, "unknown-task dataset gets no incompatible negatives")

    # ── pools: + known-low (G2 ingredient) ───────────────────────────────────
    print("\n=== pool construction (+ known-low) ===")
    pools2, stats2 = build_global_negative_pools(task, ti, ta, M, include_known_low=True)
    # d0 known-low: bottom 30% of {.9,.2,.5} -> m1 (acc .2); m0 is positive -> excluded
    check(1 in pools2[0].tolist(), "known-low model (m1) joins d0's pool")
    check(0 not in pools2[0].tolist(), "top-performer positive (m0) never a negative")
    check(stats2["known_low_total"] >= 1, f"composition logged: {stats2}")

    # ── loss direction: positives rise above negatives ───────────────────────
    print("\n=== global loss orders positive above negatives ===")
    dim = 16
    z_m_raw = torch.randn(8, dim, requires_grad=True)
    z_d = F.normalize(torch.randn(4, dim), dim=-1)
    opt = torch.optim.Adam([z_m_raw], lr=0.05)
    for _ in range(200):
        opt.zero_grad()
        z = {"model": F.normalize(z_m_raw, dim=-1), "dataset": z_d}
        loss = global_retrieval_loss(z, M, pools, temperature=0.2, n_neg=8, n_datasets=4)
        loss.backward()
        opt.step()
    with torch.no_grad():
        zm = F.normalize(z_m_raw, dim=-1)
        s0 = zm @ z_d[0]
    check(bool(s0[0] > s0[3] and s0[0] > s0[4] and s0[0] > s0[5]),
          f"after training, s(d0, pos m0) > s(d0, every pool negative) ({s0.tolist()})")

    print("\n" + "=" * 52)
    if failures:
        print(f"PHASE 1 GLOBAL-NEG TESTS FAILED -- {len(failures)}:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("PHASE 1 GLOBAL-NEG TESTS OK")


if __name__ == "__main__":
    main()
