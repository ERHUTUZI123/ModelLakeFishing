"""Validate a newly crawled HF lake without any frozen A0 input or dimensions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scale1m.canonicalize_rf import (NODE_SEP, PLACEHOLDER_DATASETS,
    QUERY_MIN_MODELS, is_rl_task)
from scale1m.hf_crawl import utcnow, write_json_atomic
from scale1m.merge_supervision import RF_SOURCE
from .snapshot import sha256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def split_eligibility(nodes, edges):
    """Match the production root assignment without materializing graph clones.

    The production split greedily fills edge-count quotas after a seeded
    permutation of sorted dataset roots. Only the subsequent negative sampling
    and disjoint edge permutation consume additional random numbers.
    """
    import torch

    root_by_node = nodes.set_index("node")["dataset"].astype(str).str.split("/").str[0]
    edge_roots = edges["node"].map(root_by_node)
    require(not edge_roots.isna().any(), "Performance evidence references unknown dataset nodes")
    counts = edge_roots.value_counts().to_dict()
    roots = sorted(counts)
    eligible = nodes.loc[nodes["gold_eligible"].astype(bool), "node"]
    cohorts = {}
    for seed in (0, 1, 2):
        order = torch.randperm(len(roots), generator=torch.Generator().manual_seed(seed)).tolist()
        assigned = {}
        test_count = val_count = 0
        for position in order:
            root = roots[position]
            if test_count < 0.2 * len(edges):
                assigned[root] = "test"
                test_count += counts[root]
            elif val_count < 0.1 * len(edges):
                assigned[root] = "val"
                val_count += counts[root]
            else:
                assigned[root] = "train"
        sides = edge_roots.map(assigned)
        n_train = int((sides == "train").sum())
        n_supervised = int(round(0.3 * n_train))
        test_nodes = sorted(eligible[root_by_node.reindex(eligible).map(assigned).to_numpy() == "test"].astype(str))
        require(n_train > n_supervised > 0 and val_count > 0 and test_count > 0,
                f"HF evidence cannot support nonempty train/validation/test root split {seed}")
        require(bool(test_nodes), f"HF evidence has no eligible held-out queries in split {seed}")
        cohorts[str(seed)] = {
            "n_train_edges": n_train, "n_train_supervision_edges": n_supervised,
            "n_validation_edges": val_count, "n_test_edges": test_count,
            "n_eligible_queries": len(test_nodes), "eligible_query_nodes": test_nodes,
        }
    return cohorts


def check_canonical(rf):
    rf = Path(rf)
    edges = pd.read_parquet(rf / "canon" / "supervision_merged.parquet")
    nodes = pd.read_parquet(rf / "canon" / "dataset_nodes_merged.parquet")
    rules = json.loads((rf / "rf_gold_rules.json").read_text(encoding="utf-8"))
    require(rules.get("evidence_mode") == "live_hf_only"
            and rules.get("sources_in_priority_order") == [RF_SOURCE],
            "Live pipeline requires explicit HF-only evidence provenance")
    require(len(edges) > 0 and len(nodes) > 0, "No usable HF model-index performance evidence")
    require(set(edges.source.astype(str)) == {RF_SOURCE}
            and set(nodes.gold_source.astype(str)) == {RF_SOURCE},
            "Live evidence contains historical/non-HF supervision")
    require(not edges.direction.astype(str).eq("curated").any()
            and not nodes.primary_direction.astype(str).eq("curated").any(),
            "HF self-reports cannot be classified as curated historical measurements")
    require(nodes.node.is_unique and not edges.duplicated(["node", "model"]).any(),
            "Duplicate dataset nodes or performance pairs")
    require(set(edges.node) == set(nodes.node), "Canonical node/edge universe differs")
    require(np.isfinite(edges.weight.to_numpy(dtype=float)).all()
            and edges.weight.between(0, 1).all(), "Invalid normalized performance weights")
    require((nodes.node == nodes.dataset.astype(str) + NODE_SEP + nodes.task.fillna("").astype(str)).all(),
            "Dataset-task node identity differs from its components")
    expected_eligibility = (nodes.primary_direction.isin(["higher", "lower"])
        & ~nodes.task.map(is_rl_task)
        & ~nodes.dataset.isin(PLACEHOLDER_DATASETS | {"##", "#", "-", ""})
        & (nodes.n_models >= QUERY_MIN_MODELS))
    require(np.array_equal(nodes.gold_eligible.to_numpy(dtype=bool), expected_eligibility.to_numpy()),
            "HF-only query eligibility differs from metric-direction/minimum-evidence rules")
    return {
        "protocol": "live_hf", "evidence_mode": "live_hf_only",
        "dataset_task_nodes": len(nodes), "performance_edges": len(edges),
        "eligible_queries": int(nodes.gold_eligible.sum()),
        "split_cohorts": split_eligibility(nodes, edges),
        "input_hashes": {name: sha256(rf / name) for name in (
            "canon/supervision_merged.parquet", "canon/dataset_nodes_merged.parquet", "rf_gold_rules.json")},
    }


def check_graph(graph, rf, ladder):
    from scale1m.build_graph_rf import A0_ZERO_COLUMNS, dataset_stats
    from scale1m.prepare_a0_graph import verify_files, _digest

    graph, rf, ladder = Path(graph), Path(rf), Path(ladder)
    result = check_canonical(rf)
    meta, files = verify_files(graph)
    require("a0" not in meta and "A0_FEATURE_REPAIR.json" not in files,
            "Live graph cannot claim a historical A0 repair envelope")
    nm, nd = meta["num_nodes"]["model"], meta["num_nodes"]["dataset"]
    require(nm >= 1000, "Full-lake top-1000 retrieval requires at least 1,000 observed models")
    require(nd == result["dataset_task_nodes"], "Graph dataset count differs from observed evidence")
    models = pd.read_parquet(graph / "unique_model_id.parquet")
    datasets = pd.read_parquet(graph / "unique_dataset_id.parquet")
    candidate_table = pd.read_parquet(ladder / "full_model_ids.parquet")
    node_table = pd.read_parquet(ladder / "full_dataset_ids.parquet")
    require(candidate_table.in_snapshot.astype(bool).all(), "Live candidates include models outside the HF crawl")
    for table, count, label in ((models, nm, "graph models"), (datasets, nd, "graph datasets"),
                                (candidate_table, nm, "candidate ladder"), (node_table, nd, "dataset ladder")):
        require(len(table) == count and np.array_equal(table.mappedID, np.arange(count)),
                "Noncontiguous physical row map: " + label)
    require(models.model.is_unique and datasets.dataset.is_unique, "Duplicate graph row IDs")
    require(np.array_equal(models.model.astype(str), candidate_table.model.astype(str))
            and np.array_equal(datasets.dataset.astype(str), node_table.node.astype(str)),
            "Graph row maps differ from the candidate/dataset ladder")
    for name, shape in (("x_model.npy", (nm, 448)), ("x_dataset.npy", (nd, 458))):
        matrix = np.load(graph / name, mmap_mode="r", allow_pickle=False)
        require(matrix.shape == shape and matrix.dtype == np.float32, "Invalid graph feature schema: " + name)
        for start in range(0, len(matrix), 50_000):
            require(np.isfinite(matrix[start:start + 50_000]).all(), "Nonfinite graph features: " + name)
    xd = np.load(graph / "x_dataset.npy", mmap_mode="r", allow_pickle=False)
    stats, roots = dataset_stats(pd.DataFrame({"dataset": datasets.dataset.astype(str).str.split(NODE_SEP, n=1).str[0]}), None)
    require(np.all(xd[:, A0_ZERO_COLUMNS] == 0) and np.array_equal(xd[:, 448:], stats)
            and roots == datasets.root.astype(str).tolist(),
            "Dataset feature masking, structural counts or roots differ from the node universe")
    supervision = pd.read_parquet(rf / "canon" / "supervision_merged.parquet")
    mid = dict(zip(models.model, models.mappedID))
    did = dict(zip(datasets.dataset, datasets.mappedID))
    source, target = supervision.model.map(mid), supervision.node.map(did)
    require(not source.isna().any() and not target.isna().any(), "Unmapped performance endpoints")
    expected_edges = np.stack([source.to_numpy(np.int64), target.to_numpy(np.int64)])
    expected_weights = supervision.weight.to_numpy(np.float32)
    with np.load(graph / "edges.npz", allow_pickle=False) as arrays:
        for prefix, expected in (("model__trained_on__dataset", expected_edges),
                                 ("dataset__rev_trained_on__model", expected_edges[::-1])):
            require(np.array_equal(arrays[prefix + "__edge_index"], expected)
                    and np.array_equal(arrays[prefix + "__edge_attr"], expected_weights),
                    "Graph performance edges/reverse weights differ from canonical evidence")
    with np.load(graph / "nodes.npz", allow_pickle=False) as arrays:
        require(np.array_equal(arrays["model.node_id"], np.arange(nm))
                and np.array_equal(arrays["dataset.node_id"], np.arange(nd)), "Graph node IDs differ")
        for key, count, limit in (("model.family_id", nm, meta["xm0_meta"]["num_families"]),
                                  ("model.size_bucket_id", nm, meta["xm0_meta"]["num_size_buckets"])):
            ids = arrays[key]
            require(ids.shape == (count,) and np.issubdtype(ids.dtype, np.integer)
                    and ids.min(initial=0) >= 0 and ids.max(initial=0) < limit,
                    "Invalid categorical feature IDs: " + key)
    report = json.loads((graph / "GRAPH_REPORT.json").read_text(encoding="utf-8"))
    require(report["edges"]["trained_on"] == result["performance_edges"] and all(report["gates"].values()),
            "Graph construction gates or performance counts failed")
    result.update(models=nm, graph_digest=_digest(files), graph_files=files,
                  graph_meta_sha256=sha256(graph / "meta.json"), seven_performance_columns_zero=True,
                  ladder_hashes={name: sha256(ladder / name) for name in (
                      "full_model_ids.parquet", "full_dataset_ids.parquet")})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    canonical = commands.add_parser("canonical")
    canonical.add_argument("--rf", type=Path, required=True)
    canonical.add_argument("--out", type=Path)
    graph = commands.add_parser("graph")
    graph.add_argument("--graph", type=Path, required=True)
    graph.add_argument("--rf", type=Path, required=True)
    graph.add_argument("--ladder", type=Path, required=True)
    graph.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = (check_canonical(args.rf) if args.command == "canonical"
              else check_graph(args.graph, args.rf, args.ladder))
    result["checked_at"] = utcnow()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(str(args.out), result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("graph_files", "input_hashes", "split_cohorts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
