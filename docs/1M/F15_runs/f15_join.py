"""F1.5 join: match the model-index dataset names against the HF dataset index.

The names that appear in `model-index` are whatever the model author typed:
sometimes a real repo id (`openai/gsm8k`), sometimes a bare legacy name
(`glue`), sometimes not a dataset at all (`lunarlander-v2`). The hit rate is a
measurement, not an assumption, and it decides how much of `e_card` is a real
card versus a fallback over the name alone.

Two matching passes, in order, each recorded separately:
  1. exact match on the normalized id
  2. basename match (the part after the slash) when it is unique in the index

Writes dataset_cards.parquet -- one row per (dataset, task) node, carrying the
text that dataset_descriptor() will consume in F2/F4.
"""
import collections
import json
import os
import sys

sys.path.insert(0, r"D:\research\model_lake\codes\ModelLakeFishing")

import pandas as pd

from scale1m.verify_raw import iter_records, normalize

DATA = os.environ.get("MLF_DATA_DIR", r"D:\research\model_lake\data")
MODELS = os.path.join(DATA, "data1m", "candidates_full")
DATASETS = os.path.join(DATA, "data1m", "datasets_full")
OUT_DIR = os.path.join(DATA, "data1m", "datasets_full")
SCRATCH = os.path.dirname(os.path.abspath(__file__))


def load_model_index_nodes():
    """-> {(dataset_norm, task): n_models}, plus per-dataset model counts."""
    prov = json.load(open(os.path.join(MODELS, "PROVENANCE.json"), encoding="utf-8"))
    nodes = collections.Counter()
    ds_models = collections.defaultdict(set)
    for _, r in iter_records(MODELS, prov["shards"]):
        mid = normalize(r.get("id"))
        mi = (r.get("cardData") or {}).get("model-index")
        if not mi:
            continue
        for entry in (mi if isinstance(mi, list) else [mi]):
            if not isinstance(entry, dict):
                continue
            ds = entry.get("dataset") or entry.get("dataset_name")
            if not ds or not (entry.get("metrics") or []):
                continue
            dsn = normalize(ds)
            task = str(entry.get("task") or "")
            nodes[(dsn, task)] += 1
            ds_models[dsn].add(mid)
    return nodes, {k: len(v) for k, v in ds_models.items()}


def load_dataset_index():
    prov = json.load(open(os.path.join(DATASETS, "PROVENANCE.json"), encoding="utf-8"))
    by_id, by_base = {}, collections.defaultdict(list)
    for _, r in iter_records(DATASETS, prov["shards"]):
        did = normalize(r.get("id"))
        by_id[did] = r
        by_base[did.split("/")[-1]].append(did)
    return by_id, by_base, prov


def main():
    nodes, ds_models = load_model_index_nodes()
    datasets = {d for d, _ in nodes}
    print("model-index: %d (dataset,task) nodes over %d datasets" % (len(nodes), len(datasets)))

    by_id, by_base, prov = load_dataset_index()
    print("dataset index: %d repos" % len(by_id))

    rows = []
    stat = collections.Counter()
    for (dsn, task), n_edges in sorted(nodes.items()):
        rec, how = by_id.get(dsn), "exact"
        if rec is None:
            cands = by_base.get(dsn.split("/")[-1], [])
            if len(cands) == 1:
                rec, how = by_id[cands[0]], "basename"
            elif len(cands) > 1:
                how = "basename_ambiguous"
                rec = None
            else:
                how = "miss"
        stat[how] += 1
        card = (rec or {}).get("cardData") or {}
        rows.append({
            "dataset": dsn, "task": task, "node": dsn + "\t" + task,
            "n_models": ds_models.get(dsn, 0), "n_records": n_edges,
            "matched_by": how,
            "matched_id": normalize(rec["id"]) if rec else None,
            "description": (rec or {}).get("description"),
            "tags": json.dumps((rec or {}).get("tags") or [], ensure_ascii=False),
            "task_categories": json.dumps(card.get("task_categories") or [], ensure_ascii=False),
            "language": json.dumps(card.get("language") or [], ensure_ascii=False),
            "downloads": (rec or {}).get("downloads"),
        })

    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "dataset_cards.parquet")
    df.to_parquet(out, index=False)

    matched = df[df.matched_by.isin(["exact", "basename"])]
    with_desc = matched[matched.description.notna() & (matched.description.str.len() > 0)]
    # weight by supervision: a miss on a 15k-model dataset costs more than on a 3-model one
    tot_e = df.n_records.sum()
    rep = {
        "nodes": len(df),
        "datasets": int(df.dataset.nunique()),
        "match": {k: int(v) for k, v in stat.items()},
        "match_pct": {k: round(100.0 * v / len(df), 2) for k, v in stat.items()},
        "matched_nodes": int(len(matched)),
        "matched_pct": round(100.0 * len(matched) / len(df), 2),
        "matched_records_pct": round(100.0 * matched.n_records.sum() / tot_e, 2),
        "with_nonempty_description": int(len(with_desc)),
        "with_description_pct": round(100.0 * len(with_desc) / len(df), 2),
        "description_chars_mean": round(float(with_desc.description.str.len().mean()), 1)
        if len(with_desc) else 0.0,
        "top_misses_by_models": df[df.matched_by == "miss"]
            .nlargest(20, "n_models")[["dataset", "task", "n_models"]]
            .to_dict("records"),
        "dataset_index_provenance": {
            k: prov.get(k) for k in ("total_records", "snapshot_window_utc",
                                     "exhausted_cursor", "wallclock_s", "pages")},
    }
    with open(os.path.join(SCRATCH, "f15_join.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in rep.items() if k != "top_misses_by_models"},
                     indent=2, ensure_ascii=False))
    print("top misses:", json.dumps(rep["top_misses_by_models"][:10], ensure_ascii=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
