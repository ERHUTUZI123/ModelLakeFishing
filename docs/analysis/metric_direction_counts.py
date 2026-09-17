"""Read-only recount of metric directions in frozen A0 inputs and splits.

Run from ModelLakeFishing:
  .venv/Scripts/python.exe docs/analysis/metric_direction_counts.py --data-root D:/research/model_lake/data/data1m --out docs/analysis/metric_direction_counts.json

Writes only the requested report. Does not alter frozen inputs or run training.
Percentages count forward performance edges, not loss weights or model impact.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch_geometric.data import HeteroData

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO.parent))
from ModelLakeFishing.stage2TrainGraphSAGE.d0_splits import make_root_aware_splits
from ModelLakeFishing.stage2TrainGraphSAGE.losses import TRAINED_ON, REV_TRAINED_ON

GRAPH_DIGEST = "acddeb93d926efcb636c83d452fe360736a0e6e4ac68cab3f5fef210e7cef5db"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize(directions):
    counts = pd.Series(directions).value_counts().sort_index().to_dict()
    counts = {str(k): int(v) for k, v in counts.items()}
    n = len(directions)
    bad = counts.get("reward", 0) + counts.get("unknown", 0)
    return dict(total_forward_edges=n, counts=counts, reward_unknown_edges=bad,
                reward_unknown_percent=100 * bad / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root = args.data_root
    audit_path = REPO / "docs/1M/A0_runs/audit/A0_INPUT_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    query_path = REPO / "docs/1M/A0_runs/audit/A0_QUERY_IDENTITY.jsonl"
    query_rows = [json.loads(line) for line in query_path.read_text(encoding="utf-8").splitlines()]
    bindings = {}
    frames = {}
    for name in ("supervision_merged.parquet", "dataset_nodes_merged.parquet"):
        path = root / "rf/canon" / name
        entries = [x for x in audit["files"] if Path(x["path"]).name == name]
        require(len(entries) == 1, "Ambiguous A0 input binding: " + name)
        digest = sha(path)
        require(digest == entries[0]["sha256"], "Changed frozen input: " + name)
        bindings[name] = digest
        frames[name] = pd.read_parquet(path)
    edges = frames["supervision_merged.parquet"]
    nodes = frames["dataset_nodes_merged.parquet"]
    require(len(edges) == 247803, "Wrong final edge denominator")
    require(not edges.duplicated(["node", "model"]).any(), "Duplicate performance pairs")
    require(not edges.direction.isna().any(), "Missing direction")
    require(set(edges.direction) <= {"curated", "higher", "lower", "reward", "unknown"},
            "Unexpected direction class")

    graph = root / "a0_20260912/graph"
    meta = json.loads((graph / "meta.json").read_text())
    digest = hashlib.sha256(json.dumps(meta["files"], sort_keys=True,
                                      ensure_ascii=False).encode("utf-8")).hexdigest()
    require(digest == GRAPH_DIGEST, "Graph manifest differs from final A0 graph")
    for name in ("edges.npz", "unique_model_id.parquet", "unique_dataset_id.parquet"):
        bindings["graph/" + name] = sha(graph / name)
        require(bindings["graph/" + name] == meta["files"][name], "Graph input changed: " + name)
    udi = pd.read_parquet(graph / "unique_dataset_id.parquet")
    umi = pq.read_table(graph / "unique_model_id.parquet")
    nm, nd = len(umi), len(udi)
    require(np.array_equal(udi.mappedID, np.arange(nd)), "Dataset row order changed")
    require(np.array_equal(umi["mappedID"].to_numpy(), np.arange(nm)), "Model row order changed")
    with np.load(graph / "edges.npz") as f:
        ei = f["model__trained_on__dataset__edge_index"]
        weights = f["model__trained_on__dataset__edge_attr"]
        require(np.array_equal(f["dataset__rev_trained_on__model__edge_index"], ei[::-1]),
                "Reverse endpoints differ")
        require(np.array_equal(f["dataset__rev_trained_on__model__edge_attr"], weights),
                "Reverse weights differ")
    models = umi["model"].take(pa.array(ei[0])).to_pylist()
    graph_nodes = udi.dataset.to_numpy()[ei[1]]
    graph_pairs = pd.MultiIndex.from_arrays([graph_nodes, models], names=["node", "model"])
    aligned = edges.set_index(["node", "model"]).reindex(graph_pairs)
    require(len(aligned) == len(edges) and not aligned.direction.isna().any(),
            "Canonical/graph pair mismatch")
    require(np.array_equal(aligned.weight.to_numpy(np.float32), weights.reshape(-1)),
            "Canonical/graph weight mismatch")
    direction = aligned.direction.to_numpy()
    pair_keys = ei[0] * nd + ei[1]
    direction_lookup = pd.Series(direction, index=pair_keys)
    weight_lookup = pd.Series(weights.reshape(-1), index=pair_keys)
    require(direction_lookup.index.is_unique, "Graph contains duplicate pairs")

    def keys(index):
        v = index.numpy() if isinstance(index, torch.Tensor) else index
        return v[0] * nd + v[1]

    def summary(index):
        d = direction_lookup.reindex(keys(index))
        require(not d.isna().any(), "Split edge missing from frozen graph")
        return summarize(d.to_numpy())

    skeleton = HeteroData()
    skeleton["model"].num_nodes = nm
    skeleton["dataset"].num_nodes = nd
    skeleton[TRAINED_ON].edge_index = torch.from_numpy(ei.copy())
    skeleton[TRAINED_ON].edge_attr = torch.from_numpy(weights.copy())
    skeleton[REV_TRAINED_ON].edge_index = torch.from_numpy(ei[::-1].copy())
    skeleton[REV_TRAINED_ON].edge_attr = torch.from_numpy(weights.copy())
    bad = edges.direction.isin(["reward", "unknown"])
    mixed = edges[bad & edges.node.isin(nodes.loc[nodes.gold_eligible, "node"])]
    report = dict(
        status="PASS", input_audit_sha256=sha(audit_path), input_sha256=bindings,
        query_identity_sha256=sha(query_path),
        graph_digest=digest, script_sha256=sha(Path(__file__)),
        denominator="Forward performance pairs after source conflict resolution and per-node cap; reverse copies excluded",
        direction_rule="Use edge direction, not metric name; unknown_curated has direction curated",
        all_edges=summarize(direction),
        native_hf_edges=summarize(edges.loc[edges.source == "hf_model_index", "direction"].to_numpy()),
        reward_unknown_nodes=int(edges.loc[bad, "node"].nunique()),
        reward_unknown_edges_on_gold_eligible_nodes=int(len(mixed)),
        mixed_nodes=sorted(mixed.node.unique().tolist()),
        split_reconstruction="Production make_root_aware_splits with neg_ratio=0; negative generation occurs after root and disjoint partitions, so retained positive edges are unchanged",
        splits={},
    )
    for seed in range(3):
        tr, val, te = make_root_aware_splits(
            skeleton, udi.root.astype(str).tolist(), split_seed=seed, neg_ratio=0)
        train_all = val[TRAINED_ON].edge_index
        train_sup = tr[TRAINED_ON].edge_label_index
        train_msg = tr[TRAINED_ON].edge_index
        require(np.array_equal(np.sort(keys(train_all)),
                               np.sort(np.concatenate([keys(train_sup), keys(train_msg)]))),
                "Disjoint train partition mismatch")
        export = root / "a0_20260912/exports" / ("A0GD_full_s%d_e25" % seed)
        prior_path = export / ("prior_sidecar_s%d.npz" % seed)
        prior_meta = json.loads(prior_path.with_name(prior_path.stem + "_meta.json").read_text())
        require(sha(prior_path) == prior_meta["sidecar_sha256"], "Prior hash mismatch")
        require(prior_meta["graph_digest"] == GRAPH_DIGEST and prior_meta["split_seed"] == seed,
                "Prior graph/split binding mismatch")
        with np.load(prior_path) as f:
            prior_index = np.stack([f["edge_model"], f["edge_dataset"]])
            prior_weights = f["edge_acc"]
        require(np.array_equal(np.sort(keys(prior_index)), np.sort(keys(te[TRAINED_ON].edge_index))),
                "Actual prior does not equal reconstructed training+validation set")
        require(np.array_equal(weight_lookup.reindex(keys(prior_index)).to_numpy(), prior_weights.reshape(-1)),
                "Actual prior weights differ from frozen graph")
        with np.load(export / "gold_cands.npz") as f:
            cand_keys = np.concatenate([f[k][0].astype(np.int64) * nd + int(k) for k in f.files])
            gold_keys = np.array([int(f[k][0, np.argmax(f[k][1])]) * nd + int(k) for k in f.files])
            official = [r for r in query_rows if r["seed"] == seed]
            require(len(official) == (1476, 1101, 1545)[seed], "Wrong official query count")
            official_candidates, official_gold = [], []
            for row in official:
                q = row["query_mappedID"]
                vals = f[str(q)]
                mids, scores = vals[0].astype(np.int64), vals[1].astype(np.float64)
                require(hashlib.sha256(mids.tobytes()).hexdigest() == row["candidate_ids_sha256"],
                        "Official candidate identity changed")
                require(hashlib.sha256(scores.tobytes()).hexdigest() == row["oriented_values_float64_sha256"],
                        "Official held-out scores changed")
                require(udi.dataset.iloc[q] == row["node"], "Official query node mismatch")
                node = nodes.set_index("node").loc[row["node"]]
                require(bool(node.gold_eligible) and np.ptp(scores) > 0,
                        "Official query fails eligibility")
                official_candidates.extend((mids * nd + q).tolist())
                official_gold.append(int(mids[np.argmax(scores)]) * nd + q)
        cand_dirs = direction_lookup.reindex(cand_keys)
        gold_dirs = direction_lookup.reindex(gold_keys)
        require(not cand_dirs.isna().any() and not gold_dirs.isna().any(), "Unmatched evaluation pair")
        report["splits"][str(seed)] = dict(
            all_training_edges=summary(train_all),
            disjoint_supervision_edges=summary(train_sup),
            training_message_edges=summary(train_msg),
            actual_prior_edges=summary(prior_index),
            sidecar_sha256=sha(prior_path),
            exported_candidate_edges=summarize(cand_dirs.to_numpy()),
            exported_gold_directions=gold_dirs.value_counts().to_dict(),
            official_evaluation_queries=len(official),
            official_candidate_edges=summarize(direction_lookup.reindex(official_candidates).to_numpy()),
            official_gold_directions=direction_lookup.reindex(official_gold).value_counts().to_dict(),
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
