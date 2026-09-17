"""
match_dataset_cards.py -- F1.5: dataset node -> HF dataset card text.

Runbook: docs/1M/1Mplan.md F1.5 and 3.7. Record: docs/1M/F1.5.md.

`e_card` (384 of the 458 dataset dims) is MiniLM over `dataset_descriptor()`,
whose input is the HF card: name + task_categories + tags + description. The
names that reach us are whatever an author typed in a model card or whatever a
historical graph builder recorded, so the match rate is a measurement.

MATCHING, TWO LEVELS, DELIBERATELY CONSERVATIVE (D-66)
    exact     the normalised id is a repo id in the index
    basename  the name carries no owner AND its basename is unique in the index
    rejected  a basename that would cross owners (`a/x` -> `b/x`), or a
              basename with several candidates. Not guessed, not resolved by
              downloads: picking the most-downloaded candidate lifts coverage
              from 33% to 53% while silently attributing another repo's card.

Everything unmatched falls back to the name text, which is the fallback
`dataset_descriptor()` already implements. `card_source` records which case
fired for every node.

Run (from ModelLakeFishing/):
    python -m scale1m.match_dataset_cards --nodes <parquet> --out <parquet>
"""
import argparse
import collections
import json
import os

import pandas as pd

from scale1m.hf_crawl import data_root, utcnow, write_json_atomic
from scale1m.verify_raw import iter_records, normalize

CARD_SOURCES = ("hf_card", "hf_card_via_parent", "name_only",
                "name_only_ambiguous", "name_only_cross_owner_rejected")


def load_index_for(wanted_ids, wanted_bases, ds_dir):
    """Two-pass so the 1M-row index never sits in memory in full.

    Only rows whose id or basename is asked for are retained; the rest are
    parsed and dropped.
    """
    prov = json.load(open(os.path.join(ds_dir, "PROVENANCE.json"), encoding="utf-8"))
    by_id, base_hits = {}, collections.defaultdict(list)
    for _, r in iter_records(ds_dir, prov["shards"]):
        did = normalize(r.get("id"))
        base = did.split("/")[-1]
        if did in wanted_ids:
            by_id[did] = r
        if base in wanted_bases:
            base_hits[base].append(did)
            if len(base_hits[base]) <= 2 and did not in by_id:
                by_id[did] = r          # keep enough to resolve or reject
    return by_id, base_hits, prov


def match(nodes, ds_dir=None):
    """nodes: DataFrame with columns dataset, task (task may be empty).

    Returns the same rows plus description / tags / task_categories /
    matched_id / card_source.
    """
    ds_dir = ds_dir or os.path.join(data_root(), "data1m", "datasets_full")
    wanted_ids = set(nodes["dataset"].map(normalize))
    # `a/b` is ambiguous between owner/name and dataset/config. The historical
    # D0 graphs use the second form for 94% of their unmatched nodes
    # (`ag_news/default`), so the parent is looked up too -- under its own
    # card_source, never merged into the exact-match count.
    parents = {d.split("/")[0] for d in wanted_ids if "/" in d}
    wanted_ids |= parents
    wanted_bases = {d.split("/")[-1] for d in wanted_ids}
    by_id, base_hits, prov = load_index_for(wanted_ids, wanted_bases, ds_dir)

    rows, stat = [], collections.Counter()
    for r in nodes.itertuples():
        dsn = normalize(r.dataset)
        rec, how = by_id.get(dsn), "hf_card"
        if rec is None:
            cands = base_hits.get(dsn.split("/")[-1], [])
            if "/" in dsn:
                how = "name_only_cross_owner_rejected" if cands else "name_only"
                rec = None
            elif len(cands) == 1:
                rec, how = by_id.get(cands[0]), "hf_card"
                if rec is None:
                    how = "name_only"
            elif len(cands) > 1:
                how, rec = "name_only_ambiguous", None
            else:
                how, rec = "name_only", None
        if rec is None and "/" in dsn:
            # try the parent: `ag_news/default` -> `ag_news`, whose card
            # describes every config of the dataset
            parent = dsn.split("/")[0]
            prec = by_id.get(parent)
            if prec is None:
                pc = base_hits.get(parent, [])
                prec = by_id.get(pc[0]) if len(pc) == 1 else None
            if prec is not None:
                rec, how = prec, "hf_card_via_parent"
        stat[how] += 1
        card = (rec or {}).get("cardData") or {}
        rows.append({
            "dataset": dsn, "task": getattr(r, "task", "") or "",
            "card_source": how,
            "matched_id": normalize(rec["id"]) if rec else None,
            "description": (rec or {}).get("description"),
            "tags": json.dumps((rec or {}).get("tags") or [], ensure_ascii=False),
            "task_categories": json.dumps(card.get("task_categories") or [],
                                          ensure_ascii=False),
            "downloads": (rec or {}).get("downloads"),
        })
    out = pd.DataFrame(rows)
    report = {
        "nodes": len(out),
        "distinct_datasets": int(out.dataset.nunique()),
        "card_source": {k: int(v) for k, v in stat.items()},
        "card_source_pct": {k: round(100.0 * v / max(len(out), 1), 2)
                            for k, v in stat.items()},
        "with_description": int((out.card_source.eq("hf_card")
                                 & out.description.notna()
                                 & out.description.str.len().gt(0)).sum()),
        "dataset_index": {k: prov.get(k) for k in
                          ("total_records", "snapshot_window_utc")},
    }
    return out, report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="F1.5: match dataset nodes to HF cards")
    p.add_argument("--nodes", required=True, help="parquet with dataset[,task]")
    p.add_argument("--datasets", default=None, help="datasets_full dir")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    nodes = pd.read_parquet(args.nodes)
    if "task" not in nodes.columns:
        nodes["task"] = ""
    cards, rep = match(nodes[["dataset", "task"]].drop_duplicates(), args.datasets)
    cards.to_parquet(args.out, index=False)
    write_json_atomic(os.path.splitext(args.out)[0] + "_REPORT.json",
                      dict(rep, written_at=utcnow(), out=os.path.basename(args.out)))
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
