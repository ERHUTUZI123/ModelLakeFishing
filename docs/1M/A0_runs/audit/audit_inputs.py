"""A0.1 read-only frozen-input audit. Writes only its audit report/identity lists.

No encoder, training loop, graph writer, index, or metric evaluator is invoked.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch_geometric.data import HeteroData

REPO = Path(__file__).resolve().parents[4]
DATA = REPO.parents[1] / "data/data1m"
OUT = REPO / "docs/1M/A0_runs/audit"
sys.path.insert(0, str(REPO.parent))
sys.path.insert(0, str(REPO))
from ModelLakeFishing.stage2TrainGraphSAGE.d0_splits import make_root_aware_splits
from ModelLakeFishing.stage2TrainGraphSAGE.losses import TRAINED_ON, REV_TRAINED_ON, accuracy_lookup
from ModelLakeFishing.stage2TrainGraphSAGE.top1_audit import candidates

REPORT = {"schema": "a0.input_audit.v1", "stage": "A0.1", "files": [], "checks": [],
          "measurements": {}, "splits": [], "errors": [],
          "authority": "docs/1M/EVIDENCE_SOURCE_LIBRARY_en.md",
          "scope": "Frozen inputs and split/query identity; no new retrieval measurements",
          "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
          "execution": {"python": sys.version, "executable": sys.executable,
                        "platform": platform.platform(), "argv": sys.argv}}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def stat(path):
    s = path.stat()
    return {"size_bytes": s.st_size, "mtime_ns": s.st_mtime_ns, "inode": s.st_ino}

def check(name, passed, observed=None, expected=None, source=None):
    REPORT["checks"].append({"name": name, "status": "PASS" if bool(passed) else "FAIL",
                             "observed": observed, "expected": expected, "source": source})

def bind(path, expected=None, source=None):
    before = stat(path)
    digest = sha(path)
    row = {"path": str(path), "before": before, "sha256": digest,
           "expected_sha256": expected, "expected_source": source}
    REPORT["files"].append(row)
    check("file_stable_while_hashing:" + str(path.relative_to(DATA)) if path.is_relative_to(DATA)
          else "source_stable", before == stat(path))
    if expected:
        check("hash:" + path.name, digest == expected, digest, expected, source)
    return digest

def digest_array(a):
    a = np.ascontiguousarray(a)
    return {"shape": list(a.shape), "dtype": str(a.dtype),
            "sha256_c_order_bytes": hashlib.sha256(a.tobytes()).hexdigest()}

def write_report():
    REPORT["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    REPORT["status"] = "PASS" if not REPORT["errors"] and all(
        c["status"] == "PASS" for c in REPORT["checks"]) else "FAIL"
    (OUT / "A0_INPUT_AUDIT.json").write_text(json.dumps(REPORT, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def main():
    torch.set_num_threads(4)
    graph = DATA / "graphs/hgraph_rf"
    source = REPO / REPORT["authority"]
    bind(source, "6c7d883734fb9a05ca99c4a8409e366955c86ba2e644bd5328c9ba696ff344cd", "A0 frozen source")
    meta = json.loads((graph / "meta.json").read_text(encoding="utf-8"))
    expected = {
        "ladder_rf/full_model_ids.parquet": "fee360d1fef9afa4c446b70af35bcbf4ead2c75387f47abc3b0d21c944ae3efa",
        "ladder_rf/full_dataset_ids.parquet": "31c027ff2eeb693aed6abaa5d235f4a6264bb4b3245bfd71cf1c8b85d32173cb",
        "feats_rf/family_vocab.csv": "00d304df9bd72acb2c2c1ca5b84ed0dc9d4adb0cbc9f098f32a9e62109d24005",
        "feats_rf/x_m.npy": "ba1020872ddb6e90726c755f9f08f236f4c73e506013c91d8a0075fc0c0361e6",
        "rf/rf_gold_rules.json": "be3fb05ecf92ea0112ba4de10b8191d71dfe096b8667a86911c2ebaeadb9cdb1"}
    paths = set((DATA / "rf").rglob("*.parquet")) | set((DATA / "rf").glob("*.json"))
    paths |= set((DATA / "feats_rf").glob("*")) | set(graph.glob("*"))
    paths |= {DATA / k for k in expected}
    paths |= {DATA / "datasets_full/dataset_cards_merged.parquet"}
    for path in [*(DATA / k for k in expected), *(graph / k for k in meta["files"])]:
        check("required_file_exists:" + str(path.relative_to(DATA)), path.is_file())
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in sorted(paths):
        if not path.is_file():
            continue
        rel = path.relative_to(DATA).as_posix()
        exp = expected.get(rel)
        es = "authoritative sections 3.2-3.4" if exp else None
        if path.parent == graph and path.name in meta["files"]:
            exp = meta["files"][path.name]
            es = "graph file table, whose digest is bound to authoritative section 3.6"
        bind(path, exp, es)
    actual_table = {p["path"].split("\\")[-1]: p["sha256"] for p in REPORT["files"]
                    if Path(p["path"]).parent == graph and Path(p["path"]).name in meta["files"]}
    gd = hashlib.sha256(json.dumps(actual_table, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    check("actual_file_graph_digest", gd == "0e80b8393846dcc4b9e354218fd5139a906d569125d2e8545e17ccd01612b76c",
          gd, "0e80b8393846dcc4b9e354218fd5139a906d569125d2e8545e17ccd01612b76c", "3.6")
    REPORT["measurements"]["graph_digest"] = gd
    print("[A0.1] frozen file hashes complete", flush=True)

    ml = pd.read_parquet(DATA / "ladder_rf/full_model_ids.parquet")
    dl = pd.read_parquet(DATA / "ladder_rf/full_dataset_ids.parquet")
    umi = pd.read_parquet(graph / "unique_model_id.parquet")
    udi = pd.read_parquet(graph / "unique_dataset_id.parquet")
    cn = pd.read_parquet(DATA / "rf/canon/dataset_nodes_merged.parquet")
    sup = pd.read_parquet(DATA / "rf/canon/supervision_merged.parquet")
    cards = pd.read_parquet(DATA / "datasets_full/dataset_cards_merged.parquet")
    N, Q = len(ml), len(dl)
    check("model_count", N == 3016439, N, 3016439, "3.3")
    check("dataset_task_count", Q == 18729, Q, 18729, "1")
    for name, table in [("model_ladder", ml), ("dataset_ladder", dl), ("model_graph_map", umi), ("dataset_graph_map", udi)]:
        check(name + "_contiguous_physical_order", np.array_equal(table.mappedID.to_numpy(), np.arange(len(table))))
    normalized = ml.model.str.strip().str.lower()
    check("model_keys_normalized_unique", normalized.equals(ml.model) and normalized.nunique() == N)
    check("graph_model_map_equal_ladder", umi.model.equals(ml.model))
    check("graph_dataset_map_equal_ladder", udi.dataset.equals(dl.node))
    check("dataset_nodes_sorted_unique", dl.node.is_monotonic_increasing and dl.node.nunique() == Q)
    check("canonical_dataset_nodes_same_set", set(cn.node) == set(dl.node) and cn.node.is_unique)
    left = dl.set_index("node")[cn.columns.drop("node")].sort_index()
    right = cn.set_index("node").sort_index()
    check("canonical_dataset_metadata_equal_ladder", left.equals(right))
    check("dataset_node_serialization", (dl.dataset + "\t" + dl.task.fillna("")).equals(dl.node))
    check("graph_roots_match_dataset_names", udi.root.equals(dl.dataset.map(lambda x: str(x).split("/")[0])))
    parts = sorted((DATA / "rf/canon").glob("part-*.parquet"))
    offset, identity = 0, hashlib.sha256()
    part_counts = []
    for p in parts:
        table = pd.read_parquet(p, columns=["model"])
        check("canonical_prefix:" + p.name, table.model.reset_index(drop=True).equals(ml.model.iloc[offset:offset+len(table)].reset_index(drop=True)))
        for mid in table.model:
            identity.update((str(mid) + "\n").encode())
        part_counts.append({"part": p.name, "rows": len(table)})
        offset += len(table)
    check("canonical_snapshot_count", offset == 3003759 and len(parts) == 61, [offset, len(parts)], [3003759, 61], "2.1,3.3")
    tail = ml.iloc[offset:]
    extra = pd.read_parquet(DATA / "rf/canon/models_out_of_snapshot.parquet")
    check("historical_suffix_count_order_identity", len(tail) == 12680 and tail.model.is_monotonic_increasing and set(tail.model) == set(extra.model))
    check("snapshot_origin_flags", ml.in_snapshot.iloc[:offset].all() and not ml.in_snapshot.iloc[offset:].any())
    check("historical_sizes_unknown", tail.size_b.isna().all())
    REPORT["measurements"].update({"model_rows": N, "dataset_task_rows": Q, "canonical_parts": part_counts,
          "canonical_snapshot_rows": offset, "canonical_model_identity_order_sha256_newline": identity.hexdigest(),
          "historical_appended_rows": len(tail), "dataset_root_count": int(udi.root.nunique())})
    REPORT["measurements"]["canonical_parquet_rows"] = {p.name: pq.ParquetFile(p).metadata.num_rows
                       for p in sorted((DATA / "rf/canon").glob("*.parquet")) if not p.name.startswith("part-")}
    check("native_capped_rows", REPORT["measurements"]["canonical_parquet_rows"]["supervision.parquet"] == 74346, source="3.1")
    check("native_uncapped_rows", REPORT["measurements"]["canonical_parquet_rows"]["supervision_uncapped.parquet"] == 143478, source="3.1")
    check("recorded_conflict_rows", REPORT["measurements"]["canonical_parquet_rows"]["supervision_conflicts.parquet"] == 1502, source="3.2")

    source_counts = {str(k): int(v) for k, v in sup.source.value_counts().items()}
    sc_expected = {"modellens_v2": 117898, "hf_model_index": 73670, "d0_v1_5": 45992,
                   "hf_effective": 5223, "a_ctrl_2000m": 3670, "diverse_zoo": 1350}
    check("retained_supervision_source_counts", source_counts == sc_expected, source_counts, sc_expected, "3.2")
    check("supervision_unique_pairs_cap", not sup.duplicated(["node", "model"]).any() and sup.groupby("node").size().max() <= 200)
    check("supervision_finite_oriented_range", np.isfinite(sup.weight).all() and sup.weight.between(0, 1).all())
    mi = pd.Series(np.arange(N), index=ml.model)
    di = pd.Series(np.arange(Q), index=dl.node)
    mids = sup.model.map(mi)
    dids = sup.node.map(di)
    check("supervision_all_endpoints_resolve", mids.notna().all() and dids.notna().all())
    e = np.load(graph / "edges.npz", allow_pickle=False)
    ei = e["model__trained_on__dataset__edge_index"]
    ea = e["model__trained_on__dataset__edge_attr"]
    check("graph_supervision_exact_table_order_weights", np.array_equal(ei, np.stack([mids.to_numpy(np.int64), dids.to_numpy(np.int64)])) and np.array_equal(ea, sup.weight.to_numpy(np.float32)))
    check("supervision_reverse_exact", np.array_equal(ei[::-1], e["dataset__rev_trained_on__model__edge_index"]) and np.array_equal(ea, e["dataset__rev_trained_on__model__edge_attr"]))
    counts = {k.removesuffix("__edge_index"): int(e[k].shape[1]) for k in e.files if k.endswith("__edge_index")}
    expected_relations = {"model__trained_on__dataset": 247803, "dataset__rev_trained_on__model": 247803,
                         "dataset__similar_to__dataset": 374580, "model__is_base_of__model": 859065,
                         "model__rev_is_base_of__model": 859065}
    check("graph_relation_counts", counts == expected_relations, counts, expected_relations, "3.6")
    sim = e["dataset__similar_to__dataset__edge_index"]
    lin = e["model__is_base_of__model__edge_index"]
    check("raw_similarity_degree20_no_self", np.all(np.bincount(sim[0], minlength=Q) == 20) and np.all(sim[0] != sim[1]))
    check("lineage_endpoints_no_self", lin.min() >= 0 and lin.max() < N and np.all(lin[0] != lin[1]))
    check("lineage_reverse_exact", np.array_equal(lin[::-1], e["model__rev_is_base_of__model__edge_index"]) and np.array_equal(e["model__is_base_of__model__relation_id"], e["model__rev_is_base_of__model__relation_id"]))
    REPORT["measurements"].update({"supervision_source_counts": source_counts, "relation_counts": counts,
                                    "total_directed_edges": sum(counts.values())})
    n = np.load(graph / "nodes.npz", allow_pickle=False)
    for nt, size in [("model", N), ("dataset", Q)]:
        check(nt + "_node_id_order", np.array_equal(n[nt+".node_id"], np.arange(size)))
    vocab = pd.read_csv(DATA / "feats_rf/family_vocab.csv", keep_default_na=False)
    fid, sid = n["model.family_id"], n["model.size_bucket_id"]
    check("family_vocab_41056", len(vocab) == 41056 and vocab.family.is_unique and vocab.family_id.is_unique, len(vocab), 41056, "3.4")
    check("family_vocab_matches_graph_meta", dict(zip(vocab.family, vocab.family_id)) == meta["xm0_meta"]["family_vocab"])
    check("family_and_size_ids_range", len(fid) == N and len(sid) == N and fid.min() >= 0 and fid.max() < len(vocab) and sid.min() == 0 and sid.max() == 14)
    check("separate_categorical_arrays_equal_graph", np.array_equal(fid, np.load(DATA / "feats_rf/family_id.npy")) and np.array_equal(sid, np.load(DATA / "feats_rf/size_bucket_id.npy")))
    unknown, other = 100 * float(np.mean(sid == 0)), 100 * float(np.mean(fid == 0))
    check("unknown_size_published_precision", f"{unknown:.3f}" == "71.929", unknown, "71.929%", "3.4")
    check("other_family_published_precision", f"{other:.3f}" == "13.561", other, "13.561%", "3.4")
    for key in ("dataset.task_type_id", "dataset.n_class_bucket_id", "dataset.arity_id"):
        check(key + "_single_zero_category", n[key].shape == (Q,) and np.all(n[key] == 0))
    card_count = int(cards.card_source.str.startswith("hf_card").sum())
    check("matched_cards", card_count == 3928 and len(cards) == Q and not cards.duplicated(["dataset", "task"]).any(), card_count, 3928, "2.3")
    check("card_keys_equal_nodes", set(zip(cards.dataset, cards.task.fillna(""))) == set(zip(dl.dataset, dl.task.fillna(""))))
    REPORT["measurements"].update({"family_vocab_rows": len(vocab), "unknown_size_percent": unknown,
          "other_family_percent": other, "matched_cards": card_count,
          "card_source_counts": {str(k): int(v) for k,v in cards.card_source.value_counts().items()},
          "gold_eligible_nodes": int(dl.gold_eligible.sum())})
    for name, shape in [("x_model.npy", (N,448)), ("x_dataset.npy", (Q,458))]:
        x = np.load(graph / name, mmap_mode="r", allow_pickle=False)
        check(name + "_shape_dtype", x.shape == shape and x.dtype == np.float32, [list(x.shape), str(x.dtype)])
        nonfinite = zero = 0
        for i in range(0,len(x),50000):
            block = x[i:i+50000]
            nonfinite += int(np.count_nonzero(~np.isfinite(block)))
            zero += int(np.count_nonzero(~np.any(block != 0,axis=1)))
        check(name + "_finite_nonzero", nonfinite == 0 and zero == 0, {"nonfinite": nonfinite,"all_zero_rows": zero})
    xd = np.load(graph / "x_dataset.npy", mmap_mode="r")
    a = sup.groupby("node")["weight"].agg(["size","mean","std","min","max"]).fillna(0).reindex(dl.node).fillna(0)
    root_sizes = dl.dataset.str.split("/").str[0].value_counts()
    stats = np.zeros((Q,10),dtype=np.float32)
    stats[:,0] = stats[:,1] = np.log1p(a["size"].to_numpy())
    stats[:,2:6] = a[["mean","std","min","max"]].to_numpy()
    stats[:,6] = np.log1p(dl.dataset.str.split("/").str[0].map(root_sizes).to_numpy())
    stats[:,7] = dl.gold_eligible.to_numpy().astype(np.float32)
    check("original_dataset_statistics_exact_reconstruction", np.array_equal(xd[:,448:],stats),
          float(np.max(np.abs(xd[:,448:]-stats))), 0.0, "3.5")
    REPORT["measurements"]["original_dataset_stats_nonzero_by_column"] = {str(i): int(np.count_nonzero(xd[:,i])) for i in range(448,458)}
    REPORT["feature_leakage_state"] = {"status": "EXPECTED_UNFIXED_ORIGINAL", "zero_in_A0_2": [448,449,450,451,452,453,455],
        "preserve": [454,456,457], "original_inputs_modified": False,
        "note": "A0.1 verifies the historical input; nonzero performance-derived columns here are the known reason for A0.2."}
    print("[A0.1] tables, maps, feature arrays and original statistics checked", flush=True)

    data = HeteroData()
    data["model"].num_nodes = N
    data["dataset"].num_nodes = Q
    data[TRAINED_ON].edge_index = torch.from_numpy(ei.copy())
    data[TRAINED_ON].edge_attr = torch.from_numpy(ea.copy())
    data[REV_TRAINED_ON].edge_index = data[TRAINED_ON].edge_index.flip(0)
    data[REV_TRAINED_ON].edge_attr = data[TRAINED_ON].edge_attr.clone()
    lookup = accuracy_lookup(data)
    roots = udi.root.astype(str).to_numpy()
    eligible = dl.gold_eligible.to_numpy().astype(bool)
    query_rows = []
    for seed, expect_q, expect_prior in [(0,1476,198216),(1,1101,196912),(2,1545,196124)]:
        split = make_root_aware_splits(data, roots.tolist(), split_seed=seed)
        tr, va, te = split
        raw = candidates(te, lookup)
        cands = {d: value for d,value in raw.items() if eligible[d]}
        qids = np.array(sorted(cands),dtype=np.int64)
        p = te[TRAINED_ON].edge_label_index[:,te[TRAINED_ON].edge_label == 1].numpy()
        # Independent qualification against canonical float32 weights.
        independent = []
        for d in np.unique(p[1]):
            ix = p[0,p[1]==d]
            weights = np.array([lookup[(int(m),int(d))] for m in ix],dtype=np.float32)
            if len(ix) >= 3 and np.std(weights) > 0 and eligible[d]: independent.append(int(d))
        check(f"seed{seed}_query_ids_independent_agreement", qids.tolist() == sorted(independent))
        check(f"seed{seed}_eligible_queries", len(qids) == expect_q,len(qids),expect_q,"5.4")
        prior = te[TRAINED_ON].edge_index.numpy()
        check(f"seed{seed}_prior_visible_edge_count", prior.shape[1] == expect_prior,prior.shape[1],expect_prior,"5.3")
        train_roots = set(roots[va[TRAINED_ON].edge_index[1].numpy()])
        val_pos = va[TRAINED_ON].edge_label_index[:,va[TRAINED_ON].edge_label == 1].numpy()
        val_roots, test_roots = set(roots[val_pos[1]]), set(roots[p[1]])
        check(f"seed{seed}_root_partition_disjoint", not(train_roots & val_roots or train_roots & test_roots or val_roots & test_roots))
        check(f"seed{seed}_query_root_excluded_from_prior", not(set(roots[qids]) & set(roots[prior[1]])))
        for name, s in zip(["train","validation","test"],split):
            check(f"seed{seed}_{name}_reverse_message_exact", torch.equal(s[TRAINED_ON].edge_index.flip(0),s[REV_TRAINED_ON].edge_index) and torch.equal(s[TRAINED_ON].edge_attr,s[REV_TRAINED_ON].edge_attr))
        details = {"seed":seed,"eligible_queries":len(qids),"candidate_qualified_queries_before_flag":len(raw),
            "query_roots":len(set(roots[qids])),"root_counts":{"train":len(train_roots),"validation":len(val_roots),"test":len(test_roots)},
            "query_ids":digest_array(qids), "test_positive_pairs":digest_array(p),
            "prior_visible_pairs_ordered":digest_array(prior),
            "message_edge_counts":{name:int(s[TRAINED_ON].edge_index.shape[1]) for name,s in zip(["train","validation","test"],split)},
            "positive_label_counts":{name:int((s[TRAINED_ON].edge_label==1).sum()) for name,s in zip(["train","validation","test"],split)},
            "negative_label_counts":{name:int((s[TRAINED_ON].edge_label==0).sum()) for name,s in zip(["train","validation","test"],split)}}
        REPORT["splits"].append(details)
        for d in qids:
            cm,cw=cands[int(d)]
            query_rows.append({"seed":seed,"query_mappedID":int(d),"node":dl.node.iloc[d],"root":roots[d],
                "historical_candidates":len(cm),"candidate_ids_sha256":digest_array(cm)["sha256_c_order_bytes"],
                "oriented_values_float64_sha256":digest_array(cw)["sha256_c_order_bytes"]})
        print(f"[A0.1] seed {seed}: eligible_queries={len(qids)}, prior_edges={prior.shape[1]}",flush=True)
    pd.DataFrame(query_rows).to_json(OUT / "A0_QUERY_IDENTITY.jsonl",orient="records",lines=True,force_ascii=False)
    REPORT["query_identity_artifact"]={"path":str(OUT/"A0_QUERY_IDENTITY.jsonl"),"sha256":sha(OUT/"A0_QUERY_IDENTITY.jsonl"),"rows":len(query_rows)}
    for row in REPORT["files"]:
        row["after"] = stat(Path(row["path"]))
        check("input_unchanged:" + Path(row["path"]).name, row["before"] == row["after"])
    REPORT["script_sha256"] = sha(Path(__file__))
    write_report()
    print(json.dumps({"status":REPORT["status"],"checks":len(REPORT["checks"]),
                      "failed":[x for x in REPORT["checks"] if x["status"] != "PASS"]}),flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        REPORT["errors"].append(traceback.format_exc())
        write_report()
        raise
    raise SystemExit(0 if REPORT["status"] == "PASS" else 1)
