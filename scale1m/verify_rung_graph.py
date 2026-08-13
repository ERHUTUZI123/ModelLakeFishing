"""
verify_rung_graph.py -- T5 exit gates (docs/1M/100kplan.md §8.3 = §12 G-B).

Separate from build_graph_rung.py on purpose. The builder asserts as it goes,
but a builder that asserts its own output only proves the code was internally
consistent this run. This file reopens the saved .pt from disk and checks it
against the frozen CORE graph, so a graph that was built correctly and then
overwritten, truncated or swapped still fails.

The split-invariance check is the one worth explaining. HALO models have no
supervised edges, so the root-aware split is a function of CORE's trained_on
edges alone. Running it on the rung graph and on CORE must therefore select the
identical test edges. If they differ, either CORE's edge order was disturbed or
the split is reading something it should not -- both are silent bugs that would
only surface as an unexplained metric shift at T8.

Run (from ModelLakeFishing/):
    python -m scale1m.verify_rung_graph --rung 100k `
        --graph <dir>/graphs/hgraph_100k.pt `
        --core stage1BuildTransferGraph/hgraph_ml_v2.pt
"""

import argparse
import hashlib
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scale1m.build_graph_rung import (  # noqa: E402
    TRAINED_ON, REV_TRAINED_ON, SIMILAR_TO, IS_BASE_OF, REV_IS_BASE_OF)

EXPECT = {"100k": {"models": 100_000, "core": 30_183},
          "500k": {"models": 500_000, "core": 30_183},
          "1m": {"models": 1_000_000, "core": 30_183}}


class Gates:
    def __init__(self):
        self.rows = []

    def check(self, ok, msg):
        self.rows.append((bool(ok), msg))
        print("  [%s] %s" % ("OK  " if ok else "FAIL", msg), flush=True)
        return bool(ok)

    @property
    def failed(self):
        return [m for ok, m in self.rows if not ok]


def test_edge_sha(data, roots, split_seed=0):
    """sha256 of the test split's (model, dataset) pairs, order-independent."""
    from ModelLakeFishing.stage2TrainGraphSAGE.d0_splits import make_root_aware_splits
    _, _, test_data = make_root_aware_splits(data, roots, split_seed=split_seed)
    lab = test_data[TRAINED_ON].edge_label
    idx = test_data[TRAINED_ON].edge_label_index[:, lab == 1]
    pairs = sorted((int(m), int(d)) for m, d in zip(*idx.tolist()))
    h = hashlib.sha256()
    for m, d in pairs:
        h.update(b"%d,%d;" % (m, d))
    return h.hexdigest(), len(pairs)


def verify(graph_path, core_path, rung, split_seed=0, skip_split=False,
           skip_contract=False):
    g = Gates()
    payload = torch.load(graph_path, map_location="cpu", weights_only=False)
    core = torch.load(core_path, map_location="cpu", weights_only=False)
    data, cd = payload["data"], core["data"]
    xm0, cxm0 = payload["xm0_meta"], core["xm0_meta"]
    n_core = int(cd["model"].num_nodes)
    exp = EXPECT.get(rung, {})

    print("=== T5 gates on %s ===" % os.path.basename(graph_path))

    # --- shapes and types (§8.3) ------------------------------------------
    n = int(data["model"].num_nodes)
    if exp:
        g.check(n == exp["models"], "model nodes == %d (got %d)" % (exp["models"], n))
        g.check(n_core == exp["core"], "CORE prefix == %d" % exp["core"])
    g.check(int(data["dataset"].num_nodes) == int(cd["dataset"].num_nodes),
            "dataset nodes == CORE's %d" % int(cd["dataset"].num_nodes))
    g.check(tuple(data["model"].x.shape) == (n, 448), "model.x is [%d, 448]" % n)
    g.check(tuple(data["dataset"].x.shape) == tuple(cd["dataset"].x.shape),
            "dataset.x is %s" % (tuple(cd["dataset"].x.shape),))
    g.check(len(data.edge_types) == 5, "5 edge types (got %d)" % len(data.edge_types))
    g.check(not torch.isnan(data["model"].x).any()
            and not torch.isinf(data["model"].x).any(), "model.x has no NaN/Inf")
    g.check(not torch.isnan(data["dataset"].x).any(), "dataset.x has no NaN")

    # --- plan §1 rule 1: CORE unchanged -----------------------------------
    g.check(torch.equal(data["model"].x[:n_core], cd["model"].x),
            "model.x[:%d] byte-identical to CORE" % n_core)
    g.check(torch.equal(data["model"].size_bucket_id[:n_core],
                        cd["model"].size_bucket_id), "size_bucket_id prefix identical")
    g.check(torch.equal(data["model"].family_id[:n_core], cd["model"].family_id),
            "family_id prefix identical")
    g.check(payload["unique_model_id"].sort_values("mappedID")["model"].tolist()[:n_core]
            == core["unique_model_id"].sort_values("mappedID")["model"].tolist(),
            "unique_model_id prefix identical to CORE's order")
    g.check(torch.equal(data["dataset"].x, cd["dataset"].x),
            "dataset.x untouched")
    g.check(torch.equal(data[TRAINED_ON].edge_index, cd[TRAINED_ON].edge_index)
            and torch.equal(data[TRAINED_ON].edge_attr, cd[TRAINED_ON].edge_attr),
            "trained_on edge_index+attr identical to CORE")
    g.check(torch.equal(data[REV_TRAINED_ON].edge_index, cd[REV_TRAINED_ON].edge_index),
            "rev_trained_on identical to CORE")
    g.check(torch.equal(data[SIMILAR_TO].edge_index, cd[SIMILAR_TO].edge_index),
            "similar_to identical to CORE")

    # --- meta contract -----------------------------------------------------
    g.check(xm0["num_size_buckets"] == cxm0["num_size_buckets"],
            "num_size_buckets unchanged (%d)" % cxm0["num_size_buckets"])
    g.check(xm0["name_dim"] == cxm0["name_dim"] and xm0["desc_dim"] == cxm0["desc_dim"],
            "name_dim/desc_dim unchanged")
    v, cv = xm0["family_vocab"], cxm0["family_vocab"]
    g.check(len(v) >= len(cv), "family_vocab only grew (%d -> %d)" % (len(cv), len(v)))
    g.check(all(v.get(k) == fid for k, fid in cv.items()),
            "no CORE family row moved")
    g.check(sorted(v.values()) == list(range(len(v))),
            "family_vocab ids contiguous 0..%d" % (len(v) - 1))
    g.check(xm0["num_families"] == len(v), "num_families == len(family_vocab)")
    g.check(int(data["model"].family_id.max()) < len(v), "family_id < num_families")
    g.check(int(data["model"].size_bucket_id.max()) < xm0["num_size_buckets"],
            "size_bucket_id < num_size_buckets")
    g.check(payload["xd0_meta"] == core["xd0_meta"], "xd0_meta identical to CORE")
    g.check(payload["unique_dataset_id"].equals(core["unique_dataset_id"]),
            "unique_dataset_id identical to CORE")

    # --- lineage -----------------------------------------------------------
    li = data[IS_BASE_OF].edge_index
    g.check(torch.equal(data[REV_IS_BASE_OF].edge_index, li.flip(0)),
            "rev_is_base_of is the exact flip of is_base_of")
    g.check(int(li.shape[1]) >= int(cd[IS_BASE_OF].edge_index.shape[1]),
            "lineage edges >= CORE's %d" % int(cd[IS_BASE_OF].edge_index.shape[1]))
    g.check(int(li.max()) < n if li.numel() else True, "lineage indices < N")
    g.check(int((li[0] == li[1]).sum()) == 0, "no lineage self-loops")
    stats_path = os.path.join(os.path.dirname(os.path.abspath(graph_path)),
                              "lineage_stats.json")
    g.check(os.path.exists(stats_path), "lineage_stats.json exists")

    # --- split invariance (§8.3, the one that catches reordering) ----------
    split_info = None
    if not skip_split:
        roots = core["unique_dataset_id"].sort_values("mappedID")["root"].astype(str).tolist()
        sha_rung, n_rung = test_edge_sha(data, roots, split_seed)
        sha_core, n_c = test_edge_sha(cd, roots, split_seed)
        split_info = {"rung_sha256": sha_rung, "core_sha256": sha_core,
                      "test_edges": n_rung, "split_seed": split_seed}
        g.check(sha_rung == sha_core and n_rung == n_c,
                "test-edge set identical to CORE's (%d edges, sha %s)"
                % (n_rung, sha_rung[:16]))

    # --- Stage-2 contract --------------------------------------------------
    if not skip_contract:
        import subprocess
        r = subprocess.run(
            [sys.executable, os.path.join(_REPO_ROOT, "ModelLakeFishing",
                                          "stage1BuildTransferGraph",
                                          "check_stage2_contract.py"),
             "--pt", os.path.abspath(graph_path)],
            capture_output=True, text=True)
        print(r.stdout[-2000:] if r.stdout else r.stderr[-2000:])
        g.check(r.returncode == 0, "check_stage2_contract passed")

    print("=" * 60)
    if g.failed:
        print("T5 GATES FAILED -- %d check(s):" % len(g.failed))
        for m in g.failed:
            print("  -", m)
    else:
        print("T5 GATES OK -- %d checks" % len(g.rows))
    return g, split_info


def main(argv=None):
    p = argparse.ArgumentParser(description="T5 exit gates")
    p.add_argument("--rung", default="100k")
    p.add_argument("--graph", required=True)
    p.add_argument("--core", default="stage1BuildTransferGraph/hgraph_ml_v2.pt")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--skip-split", action="store_true")
    p.add_argument("--skip-contract", action="store_true")
    p.add_argument("--out", default=None, help="write the gate table as JSON")
    args = p.parse_args(argv)

    g, split_info = verify(args.graph, args.core, args.rung, args.split_seed,
                           args.skip_split, args.skip_contract)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"graph": os.path.abspath(args.graph), "rung": args.rung,
                       "passed": not g.failed,
                       "checks": [{"ok": ok, "check": m} for ok, m in g.rows],
                       "split": split_info}, fh, indent=2, ensure_ascii=False)
    return 1 if g.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
