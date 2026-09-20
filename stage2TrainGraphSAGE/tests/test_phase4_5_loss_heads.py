import os
import sys

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.losses import raw_dot_ranknet_loss
from ModelLakeFishing.stage2TrainGraphSAGE.model import HeteroGraphSAGE, load_hgraph

GRAPH = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage1BuildTransferGraph",
                     "hgraph_hf1000d_2000m_xm0_xd0.pt")
failures = []


def check(cond, msg):
    print(f"  [{'OK  ' if cond else 'FAIL'}] {msg}")
    if not cond:
        failures.append(msg)


def main():
    torch.manual_seed(0)

    print("=== RankNet orders raw dot by accuracy ===")
    dim = 8
    z_d = F.normalize(torch.randn(1, dim), dim=-1)
    z_m_raw = torch.randn(4, dim, requires_grad=True)
    acc = torch.tensor([0.9, 0.7, 0.5, 0.3])
    eli = torch.stack([torch.arange(4), torch.zeros(4, dtype=torch.long)])
    opt = torch.optim.Adam([z_m_raw], lr=0.1)
    for _ in range(300):
        opt.zero_grad()
        z = {"model": F.normalize(z_m_raw, dim=-1), "dataset": z_d}
        loss = raw_dot_ranknet_loss(z, eli, acc, temperature=0.1)
        loss.backward(); opt.step()
    with torch.no_grad():
        s = (F.normalize(z_m_raw, dim=-1) @ z_d.t()).squeeze(-1)
    order_ok = bool((s[0] > s[1] > s[2] > s[3]).item())
    check(order_ok, f"raw dot order matches accuracy order after training (s={s.tolist()})")

    near = torch.tensor([0.5000, 0.5005, 0.4998, 0.5002])
    z = {"model": F.normalize(torch.randn(4, dim), dim=-1), "dataset": z_d}
    l_tie = raw_dot_ranknet_loss(z, eli, near, temperature=0.1, min_gap=0.01)
    check(float(l_tie) == 0.0, "min_gap=0.01 skips near-tie pairs (zero loss)")
    l_notie = raw_dot_ranknet_loss(z, eli, near, temperature=0.1, min_gap=0.0)
    check(float(l_notie) > 0.0, "min_gap=0 keeps the same near-tie pairs (nonzero loss)")

    try:
        raw_dot_ranknet_loss(z, eli, acc, temperature=-1.0)
        check(False, "negative temperature should raise")
    except AssertionError:
        check(True, "negative temperature raises (positivity enforced)")

    print("\n=== separate heads: normalized + score == HNSW geometry ===")
    data, xm0, _ = load_hgraph(GRAPH)
    xd0 = torch.load(GRAPH, map_location="cpu", weights_only=False).get("xd0_meta")
    ds_kw = dict(num_task_types=xd0["num_task_types"], n_class_buckets=xd0["n_class_buckets"],
                 num_arities=xd0["num_arities"]) if xd0 else {}
    torch.manual_seed(0)
    model = HeteroGraphSAGE(
        metadata=data.metadata(), frozen_dim=data["model"].x.shape[1],
        num_size_buckets=xm0["num_size_buckets"], num_families=xm0["num_families"],
        dataset_in_dim=data["dataset"].x.shape[1], num_layers=1,
        separate_heads=True, **ds_kw)
    model.eval()
    with torch.no_grad():
        z = model(data.clone())
    zm, zd = z["model"], z["dataset"]
    check(torch.allclose(zm.norm(dim=-1), torch.ones(zm.size(0)), atol=1e-5),
          "z_m unit-normalized under separate heads")
    check(torch.allclose(zd.norm(dim=-1), torch.ones(zd.size(0)), atol=1e-5),
          "z_d unit-normalized under separate heads")
    d0 = 0
    train_score = (zm[:5] * zd[d0]).sum(-1)
    hnsw_score = zm[:5] @ zd[d0]
    check(torch.allclose(train_score, hnsw_score, atol=1e-6),
          "z_d . z_m training score == HNSW inner-product geometry (exact)")

    print("\n" + "=" * 52)
    if failures:
        print(f"PHASE 4/5 TESTS FAILED -- {len(failures)}:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("PHASE 4/5 TESTS OK")


if __name__ == "__main__":
    main()
