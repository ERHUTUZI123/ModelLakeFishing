import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.model import HeteroGraphSAGE, load_hgraph
from ModelLakeFishing.stage2TrainGraphSAGE.edge_aware import WeightedSAGEConv, EdgeAwareHetero
from ModelLakeFishing.stage2TrainGraphSAGE.learnable import save_checkpoint, load_checkpoint
from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import topk_similar_to

GRAPH = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage1BuildTransferGraph",
                     "hgraph_hf1000d_2000m_xm0_xd0.pt")
ARTIFACTS = os.path.join(_HERE, "..", "artifacts")

failures = []


def check(cond, msg):
    print(f"  [{'OK  ' if cond else 'FAIL'}] {msg}")
    if not cond:
        failures.append(msg)


def main():
    torch.manual_seed(0)

    print("=== WeightedSAGEConv: weights change output, zero weight = self only ===")
    conv = WeightedSAGEConv(4, 4)
    x = torch.randn(5, 4)
    ei = torch.tensor([[0, 1, 2, 3], [4, 4, 4, 4]])
    w1 = torch.tensor([1.0, 1.0, 1.0, 1.0])
    w2 = torch.tensor([5.0, 0.1, 0.1, 0.1])
    out1 = conv(x, ei, w1)
    out2 = conv(x, ei, w2)
    check(not torch.allclose(out1[4], out2[4], atol=1e-6),
          "changing edge weights (fixed topology) changes destination output")

    w0 = torch.zeros(4)
    out0 = conv(x, ei, w0)
    self_only = conv.lin_self(x[4])
    check(torch.allclose(out0[4], self_only, atol=1e-5),
          "zeroing a node's in-edge weights leaves only the self path")

    print("\n=== EdgeAwareHetero: relation weight zeroing ===")
    meta = (["a"], [("a", "rel", "a")])
    het = EdgeAwareHetero(4, meta, num_layers=1)
    xd = {"a": torch.randn(5, 4)}
    eid = {("a", "rel", "a"): ei}
    full = het(xd, eid, {("a", "rel", "a"): torch.ones(4)})["a"]
    zeroed = het(xd, eid, {("a", "rel", "a"): torch.zeros(4)})["a"]
    check(not torch.allclose(full[4], zeroed[4], atol=1e-6),
          "relation contribution vanishes when its edge weights are zeroed")

    print("\n=== HeteroGraphSAGE(edge_aware=True) on top-k graph ===")
    data, xm0, _ = load_hgraph(GRAPH)
    data = topk_similar_to(data, 10, weighted=True)
    xd0 = torch.load(GRAPH, map_location="cpu", weights_only=False).get("xd0_meta")
    ds_kw = dict(num_task_types=xd0["num_task_types"], n_class_buckets=xd0["n_class_buckets"],
                 num_arities=xd0["num_arities"]) if xd0 else {}
    torch.manual_seed(0)
    model = HeteroGraphSAGE(
        metadata=data.metadata(), frozen_dim=data["model"].x.shape[1],
        num_size_buckets=xm0["num_size_buckets"], num_families=xm0["num_families"],
        dataset_in_dim=data["dataset"].x.shape[1], num_layers=1,
        edge_aware=True, weighted_relations=["similar_to", "trained_on", "rev_trained_on"],
        **ds_kw)
    model.eval()
    with torch.no_grad():
        z_a = model(data.clone())["model"]
    zd_a = model(data.clone())["dataset"]
    data2 = data.clone()
    st = ("dataset", "similar_to", "dataset")
    data2[st].edge_attr = (data2[st].edge_attr * 0 + 1.0)
    with torch.no_grad():
        zd_b = model(data2)["dataset"]
    check(not torch.allclose(zd_a, zd_b, atol=1e-6),
          "z_d changes when only similar_to edge_attr changes (weights are consumed)")

    print("\n=== edge-aware checkpoint round-trip ===")
    os.makedirs(ARTIFACTS, exist_ok=True)
    ckpt = os.path.join(ARTIFACTS, "test_edge_aware.pt")
    save_checkpoint(model, xm0["family_vocab"], ckpt,
                    task_type_vocab=(xd0["task_type_vocab"] if xd0 else None))
    model2, vocab2, _repro = load_checkpoint(ckpt)
    model2.eval()
    with torch.no_grad():
        z_re = model2(data.clone())["model"]
    check(getattr(model2, "edge_aware", False), "reloaded model is edge_aware")
    check(torch.allclose(z_a, z_re, atol=1e-6), "reloaded edge-aware model reproduces z")

    print("\n" + "=" * 52)
    if failures:
        print(f"PHASE 2 TESTS FAILED -- {len(failures)}:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("PHASE 2 EDGE-AWARE TESTS OK")


if __name__ == "__main__":
    main()
