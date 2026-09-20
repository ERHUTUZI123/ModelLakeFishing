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

    print("=== macro loss coverage ===")
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
    check(st["n_pairs"] == 13, f"unique non-tied pairs enumerated correctly (got {st['n_pairs']})")

    print("\n=== macro balance: equal weight per dataset ===")
    def loss_for(big_n):
        torch.manual_seed(1)
        zA = F.normalize(torch.randn(big_n, dim), dim=-1)
        zB = F.normalize(torch.randn(2, dim), dim=-1)
        zd = F.normalize(torch.randn(2, dim), dim=-1)
        zm = torch.cat([zA, zB], 0)
        src = list(range(big_n)) + [big_n, big_n + 1]
        dst = [0] * big_n + [1, 1]
        acc = [1.0 - 0.05 * r for r in range(big_n)] + [0.9, 0.4]
        eli = torch.tensor([src, dst]); a = torch.tensor(acc)
        z = {"model": zm, "dataset": zd}
        full = float(raw_dot_ranknet_loss(z, eli, a, temperature=0.1, max_pairs_per_dataset=10000))
        return full

    torch.manual_seed(2)
    v = F.normalize(torch.randn(2, dim), dim=-1)
    zd = F.normalize(torch.randn(1, dim), dim=-1)
    z2 = {"model": v, "dataset": zd}
    eli2 = torch.tensor([[0, 1], [0, 0]]); a2 = torch.tensor([0.9, 0.4])
    l2 = float(raw_dot_ranknet_loss(z2, eli2, a2, temperature=0.1))
    z6 = {"model": v.repeat(3, 1), "dataset": zd}
    src6 = [0, 1, 2, 3, 4, 5]; dst6 = [0] * 6
    a6 = torch.tensor([0.9, 0.4, 0.9, 0.4, 0.9, 0.4])
    eli6 = torch.tensor([src6, dst6])
    l6 = float(raw_dot_ranknet_loss(z6, eli6, a6, temperature=0.1, max_pairs_per_dataset=10000))
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
