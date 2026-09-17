"""Build the split-safe task-prior sidecar used by the 3M retrieval path.

The current consumer is ``scale1m.eval_y2``. It retrieves a dense HNSW
top-1,000 pool and looks up task-level historical evidence without loading the
training graph. The sidecar carries:

  trained_on : (model_mappedID, dataset_mappedID, norm_acc) lake supervision
  root_id    : per dataset mappedID -> integer root code
  task_id    : per dataset mappedID -> task group code
All arrays use the exact mappedID row order of z_m / z_d and are checked against
the export's ID snapshots.

CURRENT FULL-LAKE SOURCE
  --graph-store  a scale1m.graph_store directory. ID tables are parquet and
                 node features do not need to be loaded.

HISTORICAL COMPATIBILITY
  --graph        a single-file .pt graph used by pre-3M experiments. The old
                 consumer is archived under legacy/pre_3m/stage3HNSW.

WHY --split-seed IS NOT OPTIONAL AT SCALE
  Excluding only the query node is insufficient: its same-task group may contain
  other test datasets. With --split-seed, the sidecar contains train+validation
  edges only, recomputes the export's root-aware split, and asserts that no
  surviving edge touches a test-side dataset.

WHY task_id MAY NOT COME FROM THE GRAPH
  Earlier code read `data["dataset"].task_type_id`. On the full-lake graph that
  column is identically 0 -- build_graph_rf.py writes `np.zeros(n_d)` and
  declares `num_task_types: 1`, because the RF dataset features are
  [e_name || e_card || e_stats] with no probe views and no task vocabulary. A
  literal port would therefore put all 18,729 datasets in ONE task group and the
  "task prior" would degenerate into a global mean-accuracy prior. When the
  graph's task ids are degenerate, --task-nodes supplies the grouping from the
  canonical dataset table's `task` column instead, and prior_sidecar_meta.json
  records which source was used.

Run from the parent of the repository, once per split seed:
  ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.stage3HNSW.build_prior_sidecar \
      --graph-store <DATA>/data1m/graphs/hgraph_rf \
      --export <DATA>/data1m/exports_rf/RF_full_s0_e25 --split-seed 0 \
      --task-nodes <DATA>/data1m/rf/canon/dataset_nodes_merged.parquet
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(_HERE), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_PACKAGE_ROOT = os.path.dirname(_HERE)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

TRAINED_ON = ("model", "trained_on", "dataset")


def a0_prior_context(args):
    """Verify producer graph/export bindings before creating an A0 sidecar."""
    if not args.graph_store:
        return None
    with open(os.path.join(args.graph_store,"meta.json"),encoding="utf-8") as handle:
        graph_meta=json.load(handle)
    manifest_path=os.path.join(args.export,"EXPORT_MANIFEST.json")
    embed={}
    if os.path.isfile(manifest_path):
        with open(manifest_path,encoding="utf-8") as handle:
            embed=json.load(handle).get("stages",{}).get("embed",{})
    is_a0=("a0" in graph_meta or "A0_FEATURE_REPAIR.json" in graph_meta.get("files", {})
           or os.path.isfile(os.path.join(args.graph_store,"A0_FEATURE_REPAIR.json")) or "a0" in embed)
    if not is_a0:
        return None
    from scale1m.a0_graph_validation import verify_a0_graph
    from scale1m.checkpoint import sha256_of
    verified=verify_a0_graph(args.graph_store)
    context=embed.get("a0",{})
    if (context.get("protocol")!="a0" or context.get("run_id")!="A0_20260912"
            or context.get("seed")!=args.split_seed
            or context.get("graph_digest")!=verified["graph_sha256"]
            or context.get("checkpoint_sha256")!=sha256_of(embed["checkpoint"])):
        raise ValueError("A0 sidecar graph/seed/final checkpoint differs from export producer")
    for name in ("model_ids.parquet","dataset_ids.parquet"):
        if sha256_of(os.path.join(args.export,name))!=embed["artifact_hashes"][name]:
            raise ValueError("A0 export row map bytes changed")
    return dict(context)


def _norm_task(t):
    return re.sub(r"[\s_]+", "-", str(t).strip().lower())


def _read_ids(export_dir, stem, col):
    """The export's id snapshot, csv (D0) or parquet (graph_store rungs)."""
    csv = os.path.join(export_dir, stem + ".csv")
    if os.path.isfile(csv):
        return pd.read_csv(csv)[col].astype(str).tolist()
    pq = pd.read_parquet(os.path.join(export_dir, stem + ".parquet"))
    if "mappedID" in pq.columns:
        pq = pq.sort_values("mappedID")
    name = col if col in pq.columns else col.replace("unique_", "").replace("_id", "")
    return pq[name].astype(str).tolist()


def load_graph(args):
    """Returns (data, unique_dataset_id df, unique_model_id df)."""
    if args.graph_store:
        from ModelLakeFishing.scale1m.graph_store import load_sharded
        payload = load_sharded(args.graph_store, mmap=True, verify_sha256=False)
    else:
        payload = torch.load(args.graph, map_location="cpu", weights_only=False)
    udi = payload["unique_dataset_id"].sort_values("mappedID").reset_index(drop=True)
    umi = payload["unique_model_id"].sort_values("mappedID").reset_index(drop=True)
    return payload["data"], udi, umi


def dataset_names(udi):
    col = "dataset" if "dataset" in udi.columns else "node"
    return udi[col].astype(str)


def root_ids(udi):
    """The root of every dataset node, as an integer code.

    graph_store rungs carry an explicit `root` column and make_root_aware_splits
    splits on exactly that column, so the sidecar must use it too or the sibling
    groups and the split units would disagree. D0 has no such column and its
    roots are the part before the first '/', which is what v6 used.
    """
    if "root" in udi.columns:
        roots = udi["root"].astype(str)
        source = "unique_dataset_id.root"
    else:
        roots = dataset_names(udi).str.split("/").str[0]
        source = "dataset name before the first '/'"
    code = {r: i for i, r in enumerate(sorted(set(roots)))}
    return np.array([code[r] for r in roots], dtype="int64"), len(code), source


def task_ids(data, udi, task_nodes):
    """Task group per dataset node, and where the grouping came from."""
    graph_ids = None
    if "task_type_id" in data["dataset"]:
        graph_ids = data["dataset"].task_type_id.numpy().astype("int64")
        if len(np.unique(graph_ids)) > 1:
            return graph_ids, "graph task_type_id", int(graph_ids.max()) + 1
    if not task_nodes:
        raise SystemExit(
            "the graph's task_type_id has %d distinct value(s); pass --task-nodes "
            "to supply the grouping, or the task prior degenerates to one group"
            % (0 if graph_ids is None else len(np.unique(graph_ids))))
    tn = pd.read_parquet(task_nodes, columns=["node", "task"])
    tn["node"] = tn["node"].astype(str)
    task = tn.set_index("node")["task"].reindex(dataset_names(udi)).to_numpy()
    norm = np.array([_norm_task(t) for t in task], dtype=object)
    code = {t: i for i, t in enumerate(sorted(set(norm)))}
    return (np.array([code[t] for t in norm], dtype="int64"),
            "%s :: normalised `task` column" % os.path.basename(task_nodes),
            len(code))


def visible_edges(data, udi, split_seed):
    """train + val trained_on edges under the export's own root-aware split.

    test_data's MESSAGE graph is train + val by construction (d0_splits), which
    is precisely the edge set a serving prior may read for a test query.
    """
    from ModelLakeFishing.stage2TrainGraphSAGE.d0_splits import make_root_aware_splits
    from ModelLakeFishing.stage2TrainGraphSAGE.losses import REV_TRAINED_ON
    from torch_geometric.data import HeteroData

    roots = (udi["root"].astype(str).tolist() if "root" in udi.columns
             else dataset_names(udi).str.split("/").str[0].tolist())
    # a skeleton: the split reads only these four things, and cloning the real
    # graph would copy a 5.4 GB feature matrix three times to learn nothing
    sk = HeteroData()
    sk["model"].num_nodes = int(data["model"].num_nodes)
    sk["dataset"].num_nodes = int(data["dataset"].num_nodes)
    ei, ea = data[TRAINED_ON].edge_index, data[TRAINED_ON].edge_attr
    sk[TRAINED_ON].edge_index, sk[TRAINED_ON].edge_attr = ei, ea
    sk[REV_TRAINED_ON].edge_index, sk[REV_TRAINED_ON].edge_attr = ei.flip(0), ea.clone()
    _tr, _val, te = make_root_aware_splits(sk, roots, split_seed=split_seed)
    vis = te[TRAINED_ON].edge_index.numpy()
    vis_attr = te[TRAINED_ON].edge_attr.numpy().astype("float32")
    test_pos = te[TRAINED_ON].edge_label_index[:, te[TRAINED_ON].edge_label == 1]
    test_ds = np.unique(test_pos[1].numpy())
    assert not np.isin(vis[1], test_ds).any(), \
        "a surviving edge touches a test-side dataset -- the prior would leak"
    return vis, vis_attr, int(test_ds.size)


def main(argv=None):
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--graph", help="single-file .pt checkpoint")
    src.add_argument("--graph-store", help="scale1m.graph_store directory")
    ap.add_argument("--export", required=True)
    ap.add_argument("--split-seed", type=int, default=None,
                    help="restrict the sidecar to train+val edges of this split")
    ap.add_argument("--task-nodes", default=None,
                    help="parquet with node/task columns, used when the graph's "
                         "task_type_id is degenerate")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    prior_start_ns=time.perf_counter_ns()
    a0_context=a0_prior_context(args)
    if a0_context is not None:
        expected_name="prior_sidecar_s%d.npz" % args.split_seed
        proposed=args.out or os.path.join(args.export,expected_name)
        if os.path.exists(proposed) or os.path.exists(os.path.join(args.export,os.path.basename(proposed).replace(".npz","_meta.json"))):
            raise FileExistsError("A0 sidecar output already exists; preserve its provenance before regeneration")

    data, udi, umi = load_graph(args)

    # iron-rule check: sidecar row order must match the export's id snapshots
    assert _read_ids(args.export, "dataset_ids", "unique_dataset_id") == \
        dataset_names(udi).tolist(), "dataset id order != export"
    assert _read_ids(args.export, "model_ids", "unique_model_id") == \
        umi[umi.columns[0] if "model" not in umi.columns else "model"].astype(str).tolist(), \
        "model id order != export"

    if args.split_seed is None:
        ei = data[TRAINED_ON].edge_index.numpy()
        acc = data[TRAINED_ON].edge_attr.numpy().astype("float32")
        n_test_ds = 0
    else:
        ei, acc, n_test_ds = visible_edges(data, udi, args.split_seed)

    root_id, n_roots, root_src = root_ids(udi)
    task_id, task_src, n_tasks = task_ids(data, udi, args.task_nodes)

    name = ("prior_sidecar.npz" if args.split_seed is None
            else "prior_sidecar_s%d.npz" % args.split_seed)
    out = args.out or os.path.join(args.export, name)
    np.savez(out, edge_model=ei[0].astype("int64"), edge_dataset=ei[1].astype("int64"),
             edge_acc=acc, root_id=root_id, task_id=task_id)
    meta = {"n_edges": int(ei.shape[1]),
            "n_edges_in_graph": int(data[TRAINED_ON].edge_index.shape[1]),
            "n_models": int(len(umi)), "n_datasets": int(len(udi)),
            "n_roots": n_roots, "n_tasks": n_tasks,
            "split_seed": args.split_seed,
            "held_out_datasets": n_test_ds,
            "root_source": root_src, "task_source": task_src,
            "graph": os.path.basename(args.graph_store or args.graph)}
    if a0_context is not None:
        from scale1m.checkpoint import sha256_of
        meta.update(a0=a0_context,graph_digest=a0_context["graph_digest"],sidecar_sha256=sha256_of(out),
                    created_at=datetime.now(timezone.utc).isoformat(),
                    prior_build_segments=[{"start_ns":prior_start_ns,"end_ns":time.perf_counter_ns(),
                                           "clock":"perf_counter_ns","scope":"prior build including input verification"}])
    json.dump(meta, open(os.path.join(
        args.export, os.path.basename(out).replace(".npz", "_meta.json")), "w"),
        indent=2)
    print("sidecar -> %s" % out)
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
