"""
build_ladder_rf.py -- F3: freeze the row order of the lake.

Runbook: docs/1M/1Mplan.md 5 (F3). Record: docs/1M/F3.md.

WHAT THE LADDER IS
    One row per candidate model, in the order their embeddings will sit in
    `z_m`. `mappedID == i` means row i, and every later stage -- features,
    graph, export, HNSW, evaluation -- indexes by it. A misordered ladder does
    not raise anywhere; it silently reassigns every embedding to the wrong
    model.

ROW ORDER
    0 .. 3,003,758          the 2026-08-18 snapshot, in crawl order
                            (createdAt desc), exactly as F2 wrote the canon
                            parts
    3,003,759 .. N-1        models that appear only in historical supervision
                            (D-63), sorted by normalised id

    Appending rather than interleaving keeps the snapshot an exact prefix, so
    "the first 3,003,759 rows are the lake as HF had it" stays true and the
    retrieval-side subsampling in 1Mplan 3.5 can use prefixes when it wants to.

ATTRIBUTES OF THE APPENDED ROWS
    They have no HF record, so `size_b` is unknown -- the same NaN every model
    without `safetensors.total` gets, and the same downstream consequence
    (bucket 0, no size clause in the descriptor). `family` is recovered from
    whichever historical graph carried the model, via that graph's own
    `family_vocab`; `family_source` records which graph it came from. Their
    `x` is NOT copied from those graphs: the D0 line used a different
    descriptor function, and constraint 2 requires one descriptor for the whole
    lake, so F4 recomputes it.

Run (from ModelLakeFishing/):
    python -m scale1m.build_ladder_rf --rf <rf dir> --out <ladder dir>
"""
import argparse
import collections
import glob
import json
import os

import pandas as pd
import torch

from scale1m.canonicalize_rf import NODE_SEP, sha256_of
from scale1m.hf_crawl import data_root, utcnow, write_json_atomic
from scale1m.merge_supervision import G, SOURCES
from scale1m.verify_raw import normalize

LAYER_NO_RECORD = "no_snapshot_record"


def load_canon(rf_dir):
    """The 61 canon parts, in shard order == crawl order."""
    parts = sorted(glob.glob(os.path.join(rf_dir, "canon", "part-*.parquet")))
    if not parts:
        raise SystemExit("no canon parts in %s" % rf_dir)
    df = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
    df["in_snapshot"] = True
    df["origin"] = "snapshot"
    return df, len(parts)


def family_from_history(models):
    """model -> (family string, which graph it came from).

    Historical graphs store `family_id` against their own `family_vocab`, so
    the id is only meaningful with the vocab that produced it. Sources are read
    in priority order and the first hit wins, matching D-64.
    """
    want, out = set(models), {}
    for key, fname, _prio in SOURCES:
        ck = torch.load(os.path.join(G, fname), map_location="cpu", weights_only=False)
        meta = ck.get("xm0_meta") or {}
        vocab = meta.get("family_vocab")
        if not vocab or "family_id" not in ck["data"]["model"]:
            continue
        inv = {v: k for k, v in vocab.items()}
        umi = ck["unique_model_id"].sort_values("mappedID")
        mcol = [c for c in umi.columns if c != "mappedID"][0]
        fam_id = ck["data"]["model"].family_id.tolist()
        for i, m in enumerate(umi[mcol]):
            n = normalize(m)
            if n in want and n not in out:
                fam = inv.get(int(fam_id[i]))
                if fam:
                    out[n] = (str(fam), "history:" + key)
        if len(out) == len(want):
            break
    return out


def build(rf_dir, out_dir):
    canon, n_parts = load_canon(rf_dir)
    extra = pd.read_parquet(os.path.join(rf_dir, "canon",
                                         "models_out_of_snapshot.parquet"))
    extra = extra.sort_values("model").reset_index(drop=True)

    fam = family_from_history(extra["model"].tolist())
    rows = pd.DataFrame({
        "model": extra["model"],
        "size_b": pd.NA,
        "family": [fam.get(m, ("other", "history:none"))[0] for m in extra["model"]],
        "family_source": [fam.get(m, ("other", "history:none"))[1] for m in extra["model"]],
        "lineage_base": pd.NA,
        "lineage_relation": pd.NA,
        "lineage_source": "none",
        "layer": LAYER_NO_RECORD,
        "in_snapshot": False,
        "origin": "historical_supervision",
    })
    ladder = pd.concat([canon[rows.columns], rows], ignore_index=True)
    ladder.insert(0, "mappedID", range(len(ladder)))

    n_snap = int(len(canon))
    checks = {
        "rows_equal_expected": len(ladder) == n_snap + len(rows),
        "mappedID_is_contiguous": ladder["mappedID"].tolist() == list(range(len(ladder))),
        "ids_unique_after_normalisation":
            ladder["model"].map(normalize).nunique() == len(ladder),
        "snapshot_is_an_exact_prefix":
            ladder.iloc[:n_snap]["model"].tolist() == canon["model"].tolist(),
        "appended_rows_are_not_in_the_snapshot":
            not (set(rows["model"]) & set(canon["model"])),
    }
    # every model carrying supervision must have a row, or its edges point nowhere
    sup = pd.read_parquet(os.path.join(rf_dir, "canon", "supervision_merged.parquet"))
    sup_models = set(sup["model"])
    checks["every_supervised_model_has_a_row"] = sup_models.issubset(set(ladder["model"]))

    os.makedirs(out_dir, exist_ok=True)
    lpath = os.path.join(out_dir, "full_model_ids.parquet")
    ladder.to_parquet(lpath, index=False)

    # dataset side: the same discipline, deterministic order by node string
    nodes = pd.read_parquet(os.path.join(rf_dir, "canon",
                                         "dataset_nodes_merged.parquet"))
    nodes = nodes.sort_values("node").reset_index(drop=True)
    nodes.insert(0, "mappedID", range(len(nodes)))
    dpath = os.path.join(out_dir, "full_dataset_ids.parquet")
    nodes.to_parquet(dpath, index=False)
    checks["dataset_mappedID_is_contiguous"] = \
        nodes["mappedID"].tolist() == list(range(len(nodes)))
    checks["every_supervised_node_has_a_row"] = \
        set(sup["node"]).issubset(set(nodes["node"]))

    report = {
        "artifact": "F3 ladder (RF)", "written_at": utcnow(),
        "models": {"total": int(len(ladder)), "snapshot": n_snap,
                   "historical_only": int(len(rows)),
                   "canon_parts_read": n_parts},
        "datasets": {"nodes": int(len(nodes)),
                     "gold_eligible": int(nodes["gold_eligible"].sum())},
        "appended_family_recovery": {
            k: int(v) for k, v in
            collections.Counter(rows["family_source"]).most_common()},
        "layer_distribution": {str(k): int(v) for k, v in
                               ladder["layer"].value_counts().items()},
        "size_b_known": int(ladder["size_b"].notna().sum()),
        "checks": {k: bool(v) for k, v in checks.items()},
        "sha256": {"full_model_ids.parquet": sha256_of(lpath),
                   "full_dataset_ids.parquet": sha256_of(dpath)},
    }
    write_json_atomic(os.path.join(out_dir, "LADDER_REPORT.json"), report)
    return ladder, nodes, report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="F3: build the RF ladder")
    p.add_argument("--rf", default=os.path.join(data_root(), "data1m", "rf"))
    p.add_argument("--out", default=os.path.join(data_root(), "data1m", "ladder_rf"))
    args = p.parse_args(argv)
    _l, _n, rep = build(args.rf, args.out)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    failed = [k for k, v in rep["checks"].items() if not v]
    if failed:
        print("[FAIL] " + ", ".join(failed))
        return 1
    print("[ok] ladder frozen: %d models, %d dataset nodes"
          % (rep["models"]["total"], rep["datasets"]["nodes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
