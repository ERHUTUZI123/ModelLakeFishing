"""
check_stage2_contract.py -- verify the Stage-1 -> Stage-2 interface contract on a
built HGraph (.pt). Re-creation of the checker described in CLAUDE.md; works on
ANY graph via --pt (default: the original hgraph_zoo_xm0.pt).

Checks (all hard):
  1. payload carries data + unique_model_id + unique_dataset_id + xm0_meta
  2. data['model'].x is the REAL frozen [e_name||e_desc] (dim == name_dim+desc_dim,
     has negative entries) -- NOT a torch.rand smoke feature (all in [0,1))
  3. data['model'].size_bucket_id / family_id present, shape [N], in range
  4. xm0_meta: num_size_buckets, num_families, family_vocab; vocab is a contiguous
     bijection with Other->0 and len == num_families
  5. edge types present include trained_on, similar_to, the auto reverse edges,
     and -- if lineage exists -- BOTH is_base_of and rev_is_base_of
  6. ModelNodeEncoder gradient boundary: after a forward+backward, both learnable
     tables get nonzero grad and the frozen x gets NONE
  7. z_m row-order: a full-graph forward yields exactly N model rows (mappedID order)

Run:
  ../.venv/Scripts/python.exe -m ModelLakeFishing.stage1BuildTransferGraph.check_stage2_contract --pt <graph.pt>
or  cd stage1BuildTransferGraph && ../.venv/Scripts/python.exe check_stage2_contract.py --pt hgraph_diverse_xm0.pt
"""

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_REPO_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ModelLakeFishing.stage2TrainGraphSAGE.model import HeteroGraphSAGE  # noqa: E402

BASE_EDGES = {
    ("model", "trained_on", "dataset"),
    ("dataset", "rev_trained_on", "model"),
    ("dataset", "similar_to", "dataset"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default=os.path.join(_HERE, "hgraph_zoo_xm0.pt"))
    ap.add_argument("--sharded", default=None,
                    help="a scale1m.graph_store directory instead of a .pt; the "
                         "feature matrices arrive memory-mapped, so this also "
                         "checks that a read-only x survives forward+backward")
    args = ap.parse_args()

    if args.sharded:
        from ModelLakeFishing.scale1m.graph_store import load_sharded
        payload = load_sharded(args.sharded, mmap=True, verify_sha256=True)
        args.pt = args.sharded
    else:
        payload = torch.load(args.pt, map_location="cpu", weights_only=False)
    fails = []

    def check(cond, msg):
        print(f"  [{'OK  ' if cond else 'FAIL'}] {msg}")
        if not cond:
            fails.append(msg)

    print(f"=== Stage-2 contract check on {os.path.basename(args.pt)} ===")

    # 1. payload completeness
    for k in ("data", "unique_model_id", "unique_dataset_id", "xm0_meta"):
        check(k in payload, f"payload has '{k}'")
    data = payload["data"]
    xm0 = payload.get("xm0_meta", {})
    N = data["model"].num_nodes
    D = data["dataset"].num_nodes
    print(f"      models={N}  datasets={D}")

    # 2. real frozen features, not smoke random
    x = data["model"].x
    name_dim = xm0.get("name_dim", 64)
    desc_dim = xm0.get("desc_dim", 384)
    check(x.shape[0] == N, f"model.x has N={N} rows (got {x.shape[0]})")
    check(x.shape[1] == name_dim + desc_dim,
          f"model.x dim == name_dim+desc_dim ({name_dim}+{desc_dim}={name_dim+desc_dim}); got {x.shape[1]}")
    check(bool((x.min() < 0).item()),
          f"model.x has negative entries (real e_name||e_desc, not torch.rand smoke); min={float(x.min()):.4f}")

    # 3. learnable index columns
    check(hasattr(data["model"], "size_bucket_id") and data["model"].size_bucket_id.shape == (N,),
          "model.size_bucket_id present, shape [N]")
    check(hasattr(data["model"], "family_id") and data["model"].family_id.shape == (N,),
          "model.family_id present, shape [N]")
    if hasattr(data["model"], "family_id"):
        check(int(data["model"].family_id.max()) < xm0.get("num_families", 0),
              "family_id values < num_families")
        check(int(data["model"].size_bucket_id.max()) < xm0.get("num_size_buckets", 0),
              "size_bucket_id values < num_size_buckets")

    # 4. xm0 vocab bijection
    vocab = xm0.get("family_vocab", {})
    nf = xm0.get("num_families")
    check(len(vocab) == nf, f"family_vocab len ({len(vocab)}) == num_families ({nf})")
    check(set(vocab.values()) == set(range(nf or 0)), "family_vocab ids are a contiguous bijection 0..num_families-1")
    check(vocab.get("Other") == 0, "family_vocab['Other'] == 0 (zero-shot degradation row)")

    # 5. edge types
    ets = set(data.edge_types)
    for e in BASE_EDGES:
        check(e in ets, f"edge type present: {e}")
    has_lineage = ("model", "is_base_of", "model") in ets
    if has_lineage:
        check(("model", "rev_is_base_of", "model") in ets,
              "lineage is a DIRECTED pair: rev_is_base_of present alongside is_base_of")
        nlin = data[("model", "is_base_of", "model")].edge_index.shape[1]
        print(f"      lineage is_base_of edges: {nlin}")
    else:
        print("      (no is_base_of edges in this graph)")
    for et in sorted(ets):
        print(f"        {et}: {data[et].edge_index.shape[1]}")

    # 6. gradient boundary via ModelNodeEncoder inside HeteroGraphSAGE
    print("  -- gradient boundary (one forward/backward) --")
    model = HeteroGraphSAGE(
        metadata=data.metadata(), frozen_dim=x.shape[1],
        num_size_buckets=xm0["num_size_buckets"], num_families=xm0["num_families"],
        dataset_in_dim=data["dataset"].x.shape[1], num_layers=1)
    data["model"].x.requires_grad_(False)
    z = model(data)
    check(z["model"].shape[0] == N, f"forward returns N={N} model rows (z_m row-order contract)")
    z["model"].sum().backward()
    se = model.model_encoder.size_embedding.weight.grad
    fe = model.model_encoder.family_embedding.weight.grad
    check(se is not None and se.abs().sum() > 0, "size_embedding got nonzero grad")
    check(fe is not None and fe.abs().sum() > 0, "family_embedding got nonzero grad")
    check(data["model"].x.grad is None, "frozen model.x got NO grad")

    print("=" * 56)
    if fails:
        print(f"CONTRACT FAILED -- {len(fails)} check(s):")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("CONTRACT OK")


if __name__ == "__main__":
    main()
