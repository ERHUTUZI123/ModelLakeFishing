"""F1 -> F3 handoff: does the full crawl still contain R2's 100K prefix?

D-48 makes L_100K an exact prefix of L_full. That only holds if every id R2
selected is still enumerable today. Models get deleted, gated or made private
between snapshots, so the overlap is a measurement, not an assumption.

Reports, all against the normalized id:
  * how many of CORE's 30,183 appear in the crawl (they are in the lake either
    way -- their features come from the frozen graph -- but the number tells us
    how much of the supervised core still exists on the Hub)
  * how many of R2's 69,817 HALO appear (the ones missing are the "present in
    snapshot A, absent in snapshot B" set F3 has to count and disclose)
  * the resulting N_full and the headroom beyond the 100K prefix
"""
import json
import os
import sys

sys.path.insert(0, r"D:\research\model_lake\codes\ModelLakeFishing")

import pandas as pd
import torch

from scale1m.hf_crawl import sha256_of  # noqa: F401  (kept for parity of imports)
from scale1m.verify_raw import iter_records, normalize

DATA = os.environ.get("MLF_DATA_DIR", r"D:\research\model_lake\data")
CRAWL = os.path.join(DATA, "data1m", "candidates_full")
R2_HALO = os.path.join(DATA, "data1m", "candidates_v2", "selected", "selected_halo.parquet")
R2_LADDER = os.path.join(DATA, "data1m", "ladder", "100k_model_ids.csv")
CORE = r"D:\research\model_lake\codes\ModelLakeFishing\stage1BuildTransferGraph\hgraph_ml_v2.pt"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f1_nesting.json")


def main():
    with open(os.path.join(CRAWL, "PROVENANCE.json"), encoding="utf-8") as fh:
        prov = json.load(fh)

    ids = set()
    for _, rec in iter_records(CRAWL, prov["shards"]):
        ids.add(normalize(rec.get("id")))
    print("crawl unique ids: %d" % len(ids))

    core = torch.load(CORE, weights_only=False)
    core_ids = {normalize(m) for m in core["unique_model_id"]["model"]}
    halo_ids = {normalize(m) for m in pd.read_parquet(R2_HALO)["id_norm"]}
    ladder = pd.read_csv(R2_LADDER)
    ladder_ids = [normalize(m) for m in ladder["model"]]

    core_hit = len(core_ids & ids)
    halo_hit = len(halo_ids & ids)
    halo_missing = sorted(halo_ids - ids)
    new_beyond_prefix = len(ids - core_ids - halo_ids)

    res = {
        "snapshot_window_utc": prov.get("snapshot_window_utc"),
        "crawl_unique_ids": len(ids),
        "core_total": len(core_ids),
        "core_present_in_crawl": core_hit,
        "core_absent_from_crawl": len(core_ids) - core_hit,
        "r2_halo_total": len(halo_ids),
        "r2_halo_present_in_crawl": halo_hit,
        "r2_halo_absent_from_crawl": len(halo_missing),
        "r2_halo_absent_examples": halo_missing[:20],
        "models_beyond_the_100k_prefix": new_beyond_prefix,
        "n_full_if_prefix_kept": len(core_ids) + len(ids - core_ids),
        "r2_ladder_rows": len(ladder_ids),
        "r2_ladder_ids_unique": len(set(ladder_ids)),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    for k, v in res.items():
        if k.endswith("examples"):
            continue
        print("  %-32s %s" % (k, v))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
