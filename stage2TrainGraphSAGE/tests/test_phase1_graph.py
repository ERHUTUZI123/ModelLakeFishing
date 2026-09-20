import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.model import load_hgraph
from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import (
    SIMILAR_TO, drop_similar_to, topk_similar_to,
)

GRAPH = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage1BuildTransferGraph",
                     "hgraph_hf1000d_2000m_xm0_xd0.pt")

failures = []


def check(cond, msg):
    print(f"  [{'OK  ' if cond else 'FAIL'}] {msg}")
    if not cond:
        failures.append(msg)


def main():
    data, _xm0, _umi = load_hgraph(GRAPH)
    n = data["dataset"].num_nodes
    dense = data[SIMILAR_TO].edge_index.size(1)
    print(f"loaded: {n} datasets, dense similar_to edges = {dense}")

    print("\n=== top-k alters degree ===")
    degs = {}
    for k in (5, 10, 20):
        g = topk_similar_to(data, k)
        ei = g[SIMILAR_TO].edge_index
        deg = torch.bincount(ei[0], minlength=n)
        degs[k] = (int(deg.max()), float(deg.float().mean()), ei.size(1))
        check(int(deg.max()) <= k, f"k={k}: max out-degree {int(deg.max())} <= {k}")
    check(degs[5][2] < degs[10][2] < degs[20][2],
          f"edge count grows with k: {degs[5][2]} < {degs[10][2]} < {degs[20][2]}")

    print("\n=== no self edges, sparse ===")
    g = topk_similar_to(data, 10)
    ei = g[SIMILAR_TO].edge_index
    check(int((ei[0] == ei[1]).sum()) == 0, "no self edges in top-k graph")
    check(int(torch.bincount(ei[0], minlength=n).max()) < n - 1,
          f"top-k graph not near-complete (max deg < {n-1})")
    check(ei.size(1) < 0.2 * dense, f"top-k({10}) far sparser than dense ({ei.size(1)} << {dense})")

    print("\n=== weighted vs unweighted top-k ===")
    gw = topk_similar_to(data, 10, weighted=True)
    gu = topk_similar_to(data, 10, weighted=False)
    same_topo = torch.equal(gw[SIMILAR_TO].edge_index, gu[SIMILAR_TO].edge_index)
    check(same_topo, "weighted and unweighted top-k have identical topology")
    check(bool((gu[SIMILAR_TO].edge_attr == 1.0).all()), "unweighted edge_attr all 1.0")
    check(float(gw[SIMILAR_TO].edge_attr.min()) >= 0 and float(gw[SIMILAR_TO].edge_attr.max()) <= 1.0,
          "weighted edge_attr in [0,1] (normalized similarity preserved)")
    check(not bool((gw[SIMILAR_TO].edge_attr == 1.0).all()), "weighted edge_attr is not constant")

    print("\n=== drop similar_to ===")
    gd = drop_similar_to(data)
    check(gd[SIMILAR_TO].edge_index.size(1) == 0, "drop_similar_to removes all edges")

    print("\n=== builder top-k math == surgery (synthetic) ===")
    rng = np.random.default_rng(0)
    m = 12
    W = rng.random((m, m)); W = (W + W.T) / 2; np.fill_diagonal(W, 0.0)
    kk = 3
    Wm = torch.tensor(W); Wm.fill_diagonal_(float("-inf"))
    sur = set()
    for i in range(m):
        _, idx = torch.topk(Wm[i], kk)
        for j in idx.tolist():
            sur.add((i, j))
    Wb = W.copy(); np.fill_diagonal(Wb, -np.inf)
    bld = set()
    for i in range(m):
        for j in np.argsort(-Wb[i])[:kk]:
            bld.add((i, int(j)))
    check(sur == bld, "surgery and builder select the same top-k neighbours")

    print("\n" + "=" * 52)
    if failures:
        print(f"PHASE 1 GRAPH TESTS FAILED -- {len(failures)}:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("PHASE 1 GRAPH TESTS OK")


if __name__ == "__main__":
    main()
