"""
phase0_audit.py -- Effective-Dataset guide, Phase 0.

Generates a machine-readable funnel for the CURRENT (hf1000d_2000m) pipeline and
reconciles every known count in the guide against the real artifacts. Writes:

    artifacts/effective_dataset_v2/current_funnel.json
    artifacts/effective_dataset_v2/current_funnel.md
    artifacts/effective_dataset_v2/current_dataset_failures.csv
    artifacts/effective_dataset_v2/frozen_models.json   (+ sha256)

Read-only w.r.t. hf1000d_2000m: it never mutates the source tables or graph.
"""

import csv
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.model import load_hgraph
from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import dedup_trained_on, TRAINED_ON
from ModelLakeFishing.stage2TrainGraphSAGE.cold_dataset_split import evaluable_datasets, build_split

SRC = os.path.join(_HERE, "hf1000d_2000m")
DATA = os.path.join(_HERE, "dataset_embed", "data_hf1000d_2000m")
GRAPH = os.path.join(_HERE, "hgraph_hf1000d_2000m_xm0_xd0.pt")
OUT = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage2TrainGraphSAGE",
                   "artifacts", "effective_dataset_v2")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---- frozen model universe -------------------------------------------------
    with open(os.path.join(SRC, "selected_2000_models.json"), encoding="utf-8") as f:
        sel = json.load(f)
    if isinstance(sel, dict):
        sel = sel.get("models", sel.get("data", list(sel.values())))
    def mid(r):
        return r if isinstance(r, str) else (r.get("model_id") or r.get("model") or r.get("id"))
    model_ids = sorted({mid(r) for r in sel if mid(r)})
    frozen = {"n_models": len(model_ids), "model_ids": model_ids}
    fp = os.path.join(OUT, "frozen_models.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(frozen, f, indent=2)
    frozen_sha = hashlib.sha256(
        ("\n".join(model_ids)).encode("utf-8")).hexdigest()
    with open(fp + ".sha256", "w", encoding="utf-8") as f:
        f.write(frozen_sha + "\n")

    # ---- source tables ---------------------------------------------------------
    selected = pd.read_csv(os.path.join(SRC, "selected_1000_datasets.csv"))
    edges = pd.read_csv(os.path.join(SRC, "performance_edges_single_metric.csv"))
    emb = pd.read_csv(os.path.join(SRC, "embedding_status.csv"))
    skipped = pd.read_csv(os.path.join(SRC, "dataset_skipped_report.md.csv"))
    records = pd.read_csv(os.path.join(DATA, "records.csv"))
    model_cfg = pd.read_csv(os.path.join(DATA, "model_config_dataset.csv"))

    frozen_set = set(model_ids)

    # distinct frozen-model counts per dataset from the real edge table
    edges_frozen = edges[edges["model_id"].isin(frozen_set)]
    per_ds = {}
    for key, grp in edges_frozen.groupby("canon_key"):
        models = grp["model_id"].nunique()
        vals = pd.to_numeric(grp["norm_value"], errors="coerce").dropna()
        nonconst = bool(vals.nunique() > 1)
        per_ds[key] = {"distinct_models": int(models), "nonconstant": nonconst}

    emb_by_key = {r["canon_key"]: r for _, r in emb.iterrows()}
    skip_by_key = {r["canon_key"]: r["reason"] for _, r in skipped.iterrows()}

    # ---- graph tail ------------------------------------------------------------
    data, _xm0, umi = load_hgraph(GRAPH)
    raw_rows = int(data[TRAINED_ON].edge_index.size(1))
    dd = dedup_trained_on(data)
    distinct_pairs = int(dd[TRAINED_ON].edge_index.size(1))
    ds_with_edge = sorted(set(dd[TRAINED_ON].edge_index[1].tolist()))
    ev = evaluable_datasets(dd)
    split = build_split()
    ud = torch.load(GRAPH, map_location="cpu", weights_only=False).get("unique_dataset_id")
    dname = {}
    if ud is not None and hasattr(ud, "sort_values"):
        for _, r in ud.iterrows():
            dname[int(r["mappedID"])] = r["dataset"]

    # ---- per-dataset failure funnel -------------------------------------------
    rows = []
    for _, r in selected.iterrows():
        key = r["canon_key"]
        d = per_ds.get(key, {"distinct_models": 0, "nonconstant": False})
        er = emb_by_key.get(key)
        has_edge = key in per_ds and d["distinct_models"] > 0
        embedded = bool(er is not None and str(er["status"]).startswith(("ok", "cached")))
        emb_status = str(er["status"]) if er is not None else ""
        if not has_edge:
            lost_at = "no_frozen_model_edges"
        elif er is None:
            lost_at = "not_attempted_embed"
        elif not embedded:
            lost_at = "embedding_failed"
        else:
            lost_at = "survived_to_graph"
        rows.append({
            "canon_key": key,
            "hf_id": r.get("hf_id", ""),
            "listed_by_selector": 1,
            "distinct_frozen_models": d["distinct_models"],
            "ge3_models": int(d["distinct_models"] >= 3),
            "ge10_models": int(d["distinct_models"] >= 10),
            "ge20_models": int(d["distinct_models"] >= 20),
            "nonconstant": int(d["nonconstant"]),
            "edge_bearing": int(has_edge),
            "embed_attempted": int(er is not None),
            "embed_status": emb_status,
            "embedded_ok": int(embedded),
            "embed_skip_reason": skip_by_key.get(key, ""),
            "lost_at": lost_at,
        })
    fail_df = pd.DataFrame(rows)
    fail_df.to_csv(os.path.join(OUT, "current_dataset_failures.csv"),
                   index=False, encoding="utf-8")

    # ---- reconciled funnel -----------------------------------------------------
    n_selected = len(selected)
    n_edge_bearing = int((fail_df["edge_bearing"] == 1).sum())
    n_embed_ok = int((fail_df["embedded_ok"] == 1).sum())
    dataset_nodes = int(data["dataset"].num_nodes)
    node_only = dataset_nodes - n_embed_ok
    n_records = len(records)

    dup_rows = raw_rows - distinct_pairs

    funnel = {
        "generated_from": {
            "graph": os.path.relpath(GRAPH, _REPO_ROOT),
            "graph_sha256": sha256_file(GRAPH),
            "src_dir": os.path.relpath(SRC, _REPO_ROOT),
            "data_dir": os.path.relpath(DATA, _REPO_ROOT),
        },
        "frozen_models": {"n": len(model_ids), "sha256_sorted_ids": frozen_sha},
        "funnel": [
            {"stage": "selected_dataset_candidates", "count": n_selected,
             "guide_expected": 1000, "source": "selected_1000_datasets.csv"},
            {"stage": "edge_bearing_selected_datasets", "count": n_edge_bearing,
             "guide_expected": 131,
             "source": "performance_edges_single_metric.csv distinct canon_key w/ frozen model, "
                       "cross-check embedding_status.csv (131 rows)"},
            {"stage": "edge_bearing_embedded_datasets", "count": n_embed_ok,
             "guide_expected": 102, "source": "embedding_status.csv status in {ok,cached}"},
            {"stage": "materialized_dataset_nodes", "count": dataset_nodes,
             "guide_expected": 362,
             "source": "graph data['dataset'].num_nodes = 102 edge-bearing + "
                       f"{node_only} node-only"},
            {"stage": "materialized_records", "count": n_records,
             "guide_expected": 9305, "source": "records.csv rows"},
            {"stage": "pre_fix_graph_trained_on_rows", "count": raw_rows,
             "guide_expected": 12205,
             "source": "graph trained_on edge_index cols; records-derived pairs CONCATENATED "
                       "with model_config_dataset rows (attributes.get_finetuned_records)"},
            {"stage": "distinct_graph_pairs_after_dedup", "count": distinct_pairs,
             "guide_expected": 7056,
             "source": f"dedup_trained_on(reduce=max); {dup_rows} duplicate-pair rows removed (LEAKAGE)"},
            {"stage": "datasets_with_any_positive_trained_on_edge", "count": len(ds_with_edge),
             "guide_expected": 72, "source": "distinct dataset id in dedup trained_on"},
            {"stage": "evaluable_datasets", "count": len(ev),
             "guide_expected": 61,
             "source": ">=3 distinct candidates, non-constant accuracy, finite dataset features"},
            {"stage": "cold_test_datasets", "count": split["n_cold_test_datasets"],
             "guide_expected": 9, "source": "20% stratified single cold split"},
        ],
        "leakage": {
            "provenance_bug": "attributes.py::get_finetuned_records() concatenates "
                              "model_config_dataset.csv rows into the performance-label table",
            "duplicate_pair_rows_in_graph": dup_rows,
            "pct_of_raw_rows": round(100.0 * dup_rows / raw_rows, 1),
            "note": "dedup_trained_on(reduce=max) is a stage-2 REPAIR, not a stage-1 fix. "
                    "Guide Phase 2 requires removing the concatenation at source in the v2 path.",
        },
        "cold_split": {
            "n_cold": split["n_cold_test_datasets"],
            "n_remaining": split["n_remaining_datasets"],
            "cold_dataset_ids": split["cold_test_dataset_ids"],
            "cold_dataset_names": [dname.get(d, str(d)) for d in split["cold_test_dataset_ids"]],
        },
        "concentration_warning": {
            "note": "text-classification dominates edges (see edge_coverage_report.md): "
                    "text-classification 8692 vs next bucket (token-classification) 253.",
        },
    }

    # reconciliation status
    ok = all(s["count"] == s["guide_expected"] for s in funnel["funnel"])
    funnel["reconciled"] = ok
    funnel["mismatches"] = [s for s in funnel["funnel"] if s["count"] != s["guide_expected"]]

    with open(os.path.join(OUT, "current_funnel.json"), "w", encoding="utf-8") as f:
        json.dump(funnel, f, indent=2)

    # ---- markdown --------------------------------------------------------------
    lines = []
    lines.append("# Current pipeline funnel (hf1000d_2000m) — Phase 0 audit\n")
    lines.append(f"- graph: `{funnel['generated_from']['graph']}`")
    lines.append(f"- graph sha256: `{funnel['generated_from']['graph_sha256'][:16]}…`")
    lines.append(f"- frozen models: **{len(model_ids)}** (sorted-id sha256 `{frozen_sha[:16]}…`)")
    lines.append(f"- reconciled with guide counts: **{'YES' if ok else 'NO'}**\n")
    lines.append("| stage | count | guide-expected | match | source |")
    lines.append("|---|---:|---:|:--:|---|")
    for s in funnel["funnel"]:
        m = "✓" if s["count"] == s["guide_expected"] else "✗"
        lines.append(f"| {s['stage']} | {s['count']} | {s['guide_expected']} | {m} | {s['source']} |")
    lines.append("")
    lines.append("## Leakage (the provenance failure the guide flags)\n")
    lg = funnel["leakage"]
    lines.append(f"- **{lg['duplicate_pair_rows_in_graph']} duplicate-pair rows** "
                 f"({lg['pct_of_raw_rows']}% of the {raw_rows} graph trained_on rows) come from "
                 f"concatenating `model_config_dataset.csv` into the label table.")
    lines.append(f"- {lg['note']}\n")
    lines.append("## Where datasets are lost\n")
    lost_counts = fail_df["lost_at"].value_counts().to_dict()
    for k, v in lost_counts.items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Cold split (single split, the guide warns not to over-conclude on this)\n")
    lines.append(f"- cold datasets ({split['n_cold_test_datasets']}): "
                 + ", ".join(funnel["cold_split"]["cold_dataset_names"]))
    lines.append(f"- remaining evaluable: {split['n_remaining_datasets']}")
    with open(os.path.join(OUT, "current_funnel.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("Phase 0 audit written to", os.path.relpath(OUT, _REPO_ROOT))
    print("reconciled:", ok)
    for s in funnel["funnel"]:
        m = "OK " if s["count"] == s["guide_expected"] else "MISMATCH"
        print(f"  [{m}] {s['stage']}: {s['count']} (expected {s['guide_expected']})")
    if not ok:
        print("MISMATCHES:", [s["stage"] for s in funnel["mismatches"]])


if __name__ == "__main__":
    main()
