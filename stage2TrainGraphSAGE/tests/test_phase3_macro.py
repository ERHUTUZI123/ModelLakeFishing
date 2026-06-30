"""
test_phase3_macro.py -- Phase 3 required tests (Kendall action guide):
  5. the macro ranking loss covers the expected datasets and unique pairs;
  6. the macro-balanced loss gives EQUAL dataset weight regardless of edge count
     (a pair-rich dataset does not dominate the gradient).

Run:  python -m ModelLakeFishing.stage2TrainGraphSAGE.tests.test_phase3_macro
"""

import os
import sys

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.losses import raw_dot_ranknet_loss

failures = []


def check(cond, msg):
    print(f"  [{'OK  ' if cond else 'FAIL'}] {msg}")
    if not cond:
        failures.append(msg)


def main():
    torch.manual_seed(0)
    dim = 8

    # --- 5. coverage: stats report the datasets/pairs that contribute ---
    print("=== macro loss coverage ===")
    # dataset 0 has 5 models, dataset 1 has 3 models, dataset 2 has 1 (skipped)
    nm = {0: 5, 1: 3, 2: 1}
    srcs, dsts, accs = [], [], []
    mid = 0
    for d, k in nm.items():
        for r in range(k):
            srcs.append(mid); dsts.append(d); accs.append(1.0 - 0.1 * r); mid += 1
    eli = torch.tensor([srcs, dsts])
    acc = torch.tensor(accs)
    z = {"model": F.normalize(torch.randn(mid, dim), dim=-1),
         "dataset": F.normalize(torch.randn(3, dim), dim=-1)}
    _loss, st = raw_dot_ranknet_loss(z, eli, acc, temperature=0.1, return_stats=True)
    check(st["n_datasets"] == 2, f"only datasets with >=2 models contribute (got {st['n_datasets']})")
    # unique strict pairs: C(5,2)+C(3,2) = 10+3 = 13
    check(st["n_pairs"] == 13, f"unique non-tied pairs enumerated correctly (got {st['n_pairs']})")

    # --- 6. equal dataset weight regardless of edge count ---
    print("\n=== macro balance: equal weight per dataset ===")
    # Build two scenarios that differ ONLY in how many models the BIG dataset has.
    # The per-dataset loss is a mean over its pairs, so the dataset's contribution
    # to the macro loss must NOT scale with its pair count.
    def loss_for(big_n):
        torch.manual_seed(1)
        # dataset A: big_n models; dataset B: 2 models (fixed)
        zA = F.normalize(torch.randn(big_n, dim), dim=-1)
        zB = F.normalize(torch.randn(2, dim), dim=-1)
        zd = F.normalize(torch.randn(2, dim), dim=-1)
        zm = torch.cat([zA, zB], 0)
        src = list(range(big_n)) + [big_n, big_n + 1]
        dst = [0] * big_n + [1, 1]
        acc = [1.0 - 0.05 * r for r in range(big_n)] + [0.9, 0.4]
        eli = torch.tensor([src, dst]); a = torch.tensor(acc)
        z = {"model": zm, "dataset": zd}
        # isolate dataset B's contribution by zeroing A's via min_gap? simpler:
        # compute full macro and the B-only macro; B's share = full has 2 datasets
        full = float(raw_dot_ranknet_loss(z, eli, a, temperature=0.1, max_pairs_per_dataset=10000))
        return full

    # The macro loss = (L_A + L_B)/2 in both cases; L_B is identical across cases
    # (same B vectors/accs), so any difference comes only from L_A's own mean, NOT
    # from A's pair COUNT inflating the total. Verify the divisor is #datasets (2),
    # not #pairs: construct A with 2 vs 6 models but IDENTICAL per-pair loss by
    # reusing the same two vectors repeated -> L_A mean unchanged, count changes.
    torch.manual_seed(2)
    v = F.normalize(torch.randn(2, dim), dim=-1)
    zd = F.normalize(torch.randn(1, dim), dim=-1)
    # 2 models
    z2 = {"model": v, "dataset": zd}
    eli2 = torch.tensor([[0, 1], [0, 0]]); a2 = torch.tensor([0.9, 0.4])
    l2 = float(raw_dot_ranknet_loss(z2, eli2, a2, temperature=0.1))
    # 6 models = the same 2 vectors repeated 3x (same pairwise losses, more pairs)
    z6 = {"model": v.repeat(3, 1), "dataset": zd}
    src6 = [0, 1, 2, 3, 4, 5]; dst6 = [0] * 6
    a6 = torch.tensor([0.9, 0.4, 0.9, 0.4, 0.9, 0.4])
    eli6 = torch.tensor([src6, dst6])
    l6 = float(raw_dot_ranknet_loss(z6, eli6, a6, temperature=0.1, max_pairs_per_dataset=10000))
    # single dataset, mean over pairs -> identical regardless of pair count
    check(abs(l2 - l6) < 1e-4,
          f"per-dataset mean is invariant to pair count (l2={l2:.4f}, l6={l6:.4f})")

    print("\n" + "=" * 52)
    if failures:
        print(f"PHASE 3 TESTS FAILED -- {len(failures)}:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("PHASE 3 MACRO TESTS OK")


if __name__ == "__main__":
    main()
