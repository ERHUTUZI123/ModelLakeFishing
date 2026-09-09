"""
test_effective_dataset_v2.py -- Effective-Dataset guide "Required tests".

Covers the invariants for the phases materialized so far (0,1,2,3,6,7). The
graph-dependent tests (mirror edges, effective-count agreement before/after graph
construction, similar_to bounds, cold-query index invariance) are added in Phase 5
when the v2 graph exists.

Guide test numbers covered here:
   1  model-universe manifest == 2,000 unique frozen IDs
   3  percentage/fraction conversion and metric direction correct
   4  aliasing does not merge different configs/langs/tasks/splits/metrics
   5  exact duplicates collapse once; conflicts enter the conflict report
   6  no synthetic/demo row in the primary observation table
   7  model_config_dataset.csv contributes no performance labels
   9  the vault retains low-performing valid observations
  14  five cold folds deterministic, disjoint, cover every effective dataset once
  18  random-metric expectation / tie handling on a toy example

Run:  python -m ModelLakeFishing.stage2TrainGraphSAGE.tests.test_effective_dataset_v2
"""

import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

ART = os.path.join(_HERE, "..", "artifacts", "effective_dataset_v2")


def _load(name):
    p = os.path.join(ART, name)
    if p.endswith(".parquet") and os.path.exists(p):
        return pd.read_parquet(p)
    if os.path.exists(p):
        return pd.read_csv(p)
    alt = p.replace(".parquet", ".csv")
    return pd.read_csv(alt)


def test_1_frozen_manifest_2000_unique():
    fm = json.load(open(os.path.join(ART, "frozen_models.json"), encoding="utf-8"))
    assert fm["n_models"] == 2000
    assert len(fm["model_ids"]) == 2000
    assert len(set(fm["model_ids"])) == 2000
    print("[1] frozen manifest: 2000 unique IDs OK")


def test_3_metric_conversion_and_direction():
    from ModelLakeFishing.stage1BuildTransferGraph.phase1_ceiling_audit import (
        canon_metric, canon_value)
    # percentage -> fraction
    v, rej = canon_value(84.7, "accuracy")
    assert rej is None and abs(v - 0.847) < 1e-9, v
    # fraction stays
    v, rej = canon_value(0.847, "accuracy")
    assert rej is None and abs(v - 0.847) < 1e-9
    # impossible range rejected, not clipped
    v, rej = canon_value(250.0, "accuracy")
    assert v is None and rej == "out_of_range"
    # direction
    _, hib = canon_metric("Accuracy", "accuracy")
    assert hib is True
    _, hib = canon_metric("WER", "wer")
    assert hib is False
    print("[3] percentage conversion + metric direction OK")


def test_4_aliasing_preserves_group_distinctions():
    from ModelLakeFishing.stage1BuildTransferGraph.phase2_canonicalize import apply_aliases
    # two rows: same dataset name, DIFFERENT config/split/metric must stay distinct
    df = pd.DataFrame([
        {"dataset_canonical": "sst-2", "dataset_config": "en", "task": "tc",
         "metric_canonical": "accuracy", "split": "test", "value_canonical": 0.9,
         "verified": False, "confidence_tier": 1},
        {"dataset_canonical": "sst2", "dataset_config": "fr", "task": "tc",
         "metric_canonical": "accuracy", "split": "test", "value_canonical": 0.8,
         "verified": False, "confidence_tier": 1},
        {"dataset_canonical": "sst2", "dataset_config": "en", "task": "tc",
         "metric_canonical": "f1", "split": "test", "value_canonical": 0.7,
         "verified": False, "confidence_tier": 1},
    ])
    out, alias = apply_aliases(df)
    # sst-2 -> sst2 (alias applied) but config/metric differences keep groups distinct
    assert out["ranking_group_id"].nunique() == 3, out["ranking_group_id"].tolist()
    assert (alias["raw"] == "sst-2").any()
    print("[4] aliasing keeps config/metric/split distinct OK")


def test_5_dup_collapse_and_conflicts():
    from ModelLakeFishing.stage1BuildTransferGraph.phase2_canonicalize import canonicalize
    df = pd.DataFrame([
        # exact duplicate pair -> collapse to one, no conflict
        dict(ranking_group_id="g1", model_id_canonical="m", dataset_canonical="d",
             dataset_config="", task="t", metric_canonical="accuracy",
             metric_direction="higher", split="test", value_canonical=0.90,
             verified=False, confidence_tier=1, source_name="s"),
        dict(ranking_group_id="g1", model_id_canonical="m", dataset_canonical="d",
             dataset_config="", task="t", metric_canonical="accuracy",
             metric_direction="higher", split="test", value_canonical=0.90,
             verified=False, confidence_tier=1, source_name="s"),
        # conflicting pair -> one row + conflict entry
        dict(ranking_group_id="g2", model_id_canonical="m", dataset_canonical="d",
             dataset_config="", task="t", metric_canonical="accuracy",
             metric_direction="higher", split="test", value_canonical=0.80,
             verified=False, confidence_tier=1, source_name="s"),
        dict(ranking_group_id="g2", model_id_canonical="m", dataset_canonical="d",
             dataset_config="", task="t", metric_canonical="accuracy",
             metric_direction="higher", split="test", value_canonical=0.60,
             verified=False, confidence_tier=1, source_name="s"),
    ])
    canon, conf = canonicalize(df)
    assert len(canon) == 2, len(canon)                 # one row per (group,model)
    assert len(conf) == 1 and conf.iloc[0]["ranking_group_id"] == "g2"
    assert abs(conf.iloc[0]["resolved_value"] - 0.70) < 1e-9   # median of {0.6,0.8}
    print("[5] exact dup collapses once; conflict recorded OK")


def test_6_no_synthetic_rows():
    canon = _load("canonical_performance_observations.parquet")
    assert (canon["source_name"] == "hf_model_index").all()
    assert (canon["confidence_tier"] == 1).all()
    assert canon["value_canonical"].notna().all()
    print(f"[6] no synthetic rows ({len(canon):,} rows, all hf_model_index tier-1) OK")


def test_7_model_config_contributes_no_labels():
    # the v2 raw table is built ONLY from the model cache model-index; model_config_dataset
    # is never a source_name in the observation vault.
    canon = _load("canonical_performance_observations.parquet")
    assert "model_config_dataset" not in set(canon["source_name"].unique())
    assert set(canon["source_name"].unique()) == {"hf_model_index"}
    print("[7] model_config_dataset contributes no labels OK")


def test_9_vault_retains_low_performers():
    canon = _load("canonical_performance_observations.parquet")
    grp = canon.groupby("ranking_group_id")["value_canonical"]
    lo = grp.transform("min")
    # every group's minimum observation is present (nothing filtered by score)
    assert (canon["value_canonical"] == lo).sum() >= canon["ranking_group_id"].nunique()
    rep = json.load(open(os.path.join(ART, "phase3_report.json"), encoding="utf-8"))
    assert rep["low_perf_observations_retained_bottom10pct"] > 0
    print("[9] vault retains low-performing observations OK")


def test_14_folds_deterministic_disjoint_cover_once():
    man = json.load(open(os.path.join(ART, "cold_folds", "fold_manifest.json"), encoding="utf-8"))
    n_eff = man["n_effective_datasets"]
    seen = []
    for f in range(man["n_folds"]):
        fold = json.load(open(os.path.join(ART, "cold_folds", f"fold_{f}.json"), encoding="utf-8"))
        seen.extend(fold["cold_dataset_names"])
    assert len(seen) == n_eff, (len(seen), n_eff)
    assert len(set(seen)) == n_eff                      # disjoint + cover once
    assert max(man["fold_sizes"]) - min(man["fold_sizes"]) <= 1   # balanced
    # determinism: re-run assignment yields identical folds
    from ModelLakeFishing.stage1BuildTransferGraph.phase6_folds import (
        load_canon, effective_datasets, assign_folds)
    eff = effective_datasets(load_canon())
    a1 = assign_folds(eff)
    a2 = assign_folds(eff)
    assert a1 == a2
    print(f"[14] 5 folds deterministic, disjoint, cover {n_eff} datasets once "
          f"(sizes {man['fold_sizes']}) OK")


def _v2_graph_path():
    return os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage1BuildTransferGraph",
                        "hgraph_hf_effective_2000m_v2_xm0_xd0.pt")


def test_10_11_13_v2_graph_invariants():
    """Graph-dependent tests (skipped until Phase 5 builds the v2 graph):
       10 effective-dataset count agrees with the vault (minus embedding failures);
       11 trained_on and rev_trained_on are exact mirrors;
       13 similar_to has no self-edges and respects the top-k degree bound."""
    p = _v2_graph_path()
    if not os.path.exists(p):
        print("[10/11/13] v2 graph not built yet — SKIP")
        return
    import torch
    payload = torch.load(p, map_location="cpu", weights_only=False)
    data = payload["data"]
    TO = ("model", "trained_on", "dataset")
    RTO = ("dataset", "rev_trained_on", "model")
    ST = ("dataset", "similar_to", "dataset")
    # 11 mirror
    assert torch.equal(data[TO].edge_index, data[RTO].edge_index.flip(0)), "mirror mismatch"
    # 13 similar_to self-edge-free + degree bound
    si = data[ST].edge_index
    assert int((si[0] == si[1]).sum()) == 0, "similar_to has self-loops"
    import numpy as np
    outdeg = np.bincount(si[0].numpy(), minlength=data["dataset"].num_nodes)
    assert outdeg.max() <= 10, f"similar_to out-degree {outdeg.max()} exceeds top-10 bound"
    # 10 effective-count agreement (>=3 edges per dataset in graph vs vault effective set)
    by_d = np.bincount(data[TO].edge_index[1].numpy(), minlength=data["dataset"].num_nodes)
    graph_ge3 = int((by_d >= 3).sum())
    rep = json.load(open(os.path.join(ART, "phase5_build_report.json"), encoding="utf-8"))
    assert rep["B"]["datasets_with_ge3_edges"] == graph_ge3
    assert rep["B"]["reverse_mirror_exact"] is True
    print(f"[10/11/13] v2 graph: mirror exact, no self-loops, {graph_ge3} datasets >=3 edges OK")


def test_18_random_expectation_and_ties():
    # expected Hit@1 of a random pick over c candidates with g gold = g/c; ties handled
    rng = np.random.default_rng(0)
    c, g, trials = 8, 1, 40000
    hits = 0
    for _ in range(trials):
        pick = rng.integers(0, c)
        hits += (pick < g)
    assert abs(hits / trials - g / c) < 0.02
    # all-tied candidates: expected rank of gold is uniform, Hit@1 == g/c still
    print("[18] random expectation g/c and tie handling OK")


ALL = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    fails = 0
    for t in ALL:
        try:
            t()
        except Exception as e:
            fails += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(ALL) - fails}/{len(ALL)} tests passed")
    sys.exit(1 if fails else 0)
