"""
build_ladder.py -- T3 stage 2: assemble the rung's mappedID table.

Runbook: docs/1M/100kplan.md 6.2 / 6.3.   Record: docs/1M/T3.md.

THE ONE CONTRACT
    row i of the ladder is the model with mappedID == i, and rows 0..30182 are
    CORE **in CORE's own order, byte for byte**. Everything downstream -- the
    frozen feature matrix, the supervised edges, z_m, HNSW, the evaluator --
    is indexed by that integer. A misalignment does not raise anywhere; it
    silently rewires which model owns which embedding.

    So this file is mostly assertions, and that is deliberate.

HALO IS A SELECTED POPULATION, NOT A CRAWL PREFIX (D-30)
    HALO rows come from `selected_halo.parquet` -- the population that passed
    G1-G20 -- not from the first 69,817 rows of a downloads-ordered crawl.
    Their order inside the ladder is the selector's deterministic
    `selection_order`, so a rerun reproduces the same mappedIDs.

Run (from ModelLakeFishing/):
    .\\.venv\\Scripts\\python.exe -m scale1m.build_ladder --rung 100k --n 100000 `
        --core stage1BuildTransferGraph/hgraph_ml_v2.pt `
        --halo <dir>/selected/selected_halo.parquet `
        --canon <dir>/canon/hf_canon.parquet --out <dir>/ladder
"""

import argparse
import hashlib
import json
import os

import pandas as pd

from scale1m.hf_crawl import utcnow, write_json_atomic
from scale1m.hf_canonicalize import normalize

LADDER_COLUMNS = ["mappedID", "model", "layer", "size_b", "family", "lineage_base"]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(core_path, halo_path, canon_path, n, out_dir, rung):
    import torch
    core = torch.load(core_path, weights_only=False)
    umi = core["unique_model_id"].sort_values("mappedID").reset_index(drop=True)
    n_core = len(umi)
    core_ids = [normalize(m) for m in umi["model"]]
    core_id_set = set(core_ids)

    halo = pd.read_parquet(halo_path)
    canon = pd.read_parquet(canon_path).set_index("id_norm")
    need = n - n_core
    print("[in] CORE %d | HALO %d (need %d) | canon %d"
          % (n_core, len(halo), need, len(canon)))

    # --- the four checks that must happen BEFORE any concat ---------------
    assert len(halo) == need, \
        "HALO has %d rows, the rung needs %d" % (len(halo), need)
    overlap = core_id_set & set(halo["id_norm"])
    assert not overlap, (
        "%d HALO ids are already in CORE (e.g. %s). The rung would have %d "
        "distinct models, not %d. Re-run selection with --exclude-core (D-35)."
        % (len(overlap), sorted(overlap)[:3], n - len(overlap), n))
    assert halo["id_norm"].nunique() == len(halo), "duplicate ids inside HALO"
    missing = set(halo["id_norm"]) - set(canon.index)
    assert not missing, "%d HALO ids absent from the canon table" % len(missing)

    # --- CORE prefix: carried verbatim ------------------------------------
    # CORE's four graph quantities live in the frozen graph, not in canon --
    # iron rule 1 says they are copied, never recomputed. The ladder therefore
    # leaves them blank on the CORE side and marks the rows `core`; T4 reads
    # them straight out of hgraph_ml_v2.pt.
    core_block = pd.DataFrame({
        "mappedID": range(n_core),
        "model": umi["model"].tolist(),
        "layer": "core",
        "size_b": pd.NA,
        "family": pd.NA,
        "lineage_base": pd.NA,
    })

    # --- HALO block: selector order, canon values --------------------------
    halo = halo.sort_values("selection_order") if "selection_order" in halo else halo
    c = canon.loc[halo["id_norm"]]
    halo_block = pd.DataFrame({
        "mappedID": range(n_core, n),
        "model": c["model"].tolist(),
        "layer": c["layer"].tolist(),
        "size_b": c["size_b"].tolist(),
        "family": c["family"].tolist(),
        "lineage_base": c["lineage_base"].tolist(),
    })

    ladder = pd.concat([core_block, halo_block], ignore_index=True)

    # --- 6.3 exit gates ----------------------------------------------------
    assert len(ladder) == n
    assert ladder["mappedID"].tolist() == list(range(n))
    assert ladder.iloc[:n_core]["model"].tolist() == umi["model"].tolist(), \
        "CORE prefix is not byte-identical to CORE's own mappedID order"
    assert ladder["model"].map(normalize).nunique() == n, "duplicate ids in the ladder"

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "%s_model_ids.csv" % rung)
    ladder[LADDER_COLUMNS].to_csv(path, index=False)

    h = ladder.iloc[n_core:]
    report = {
        "written_at": utcnow(),
        "rung": rung, "n": n, "n_core": n_core, "n_halo": len(h),
        "core": os.path.abspath(core_path),
        "halo": os.path.abspath(halo_path),
        "canon": os.path.abspath(canon_path),
        "ladder_csv": os.path.abspath(path),
        "ladder_sha256": sha256_of(path),
        "core_halo_intersection": 0,
        "distinct_models": int(ladder["model"].map(normalize).nunique()),
        "halo_layer": h["layer"].value_counts().to_dict(),
        "halo_layer_share": {k: round(v, 5) for k, v in
                             h["layer"].value_counts(normalize=True).to_dict().items()},
        "halo_size_b_known": round(float(h["size_b"].notna().mean()), 5),
        "halo_size_b_missing": round(float(h["size_b"].isna().mean()), 5),
        "halo_family_distinct": int(h["family"].nunique()),
        "halo_family_other_share": round(float((h["family"] == "other").mean()), 5),
        "halo_lineage_declared": round(float(h["lineage_base"].notna().mean()), 5),
    }

    # family-vocab forecast: what FAMILY_MIN_COUNT would fold into Other at T4
    from dataset_embed.utils.fetch_metadata import FAMILY_OTHER
    import sys as _sys
    vc = h["family"].value_counts()
    kept = vc[vc >= 3]
    report["halo_family_ge3"] = int(len(kept))
    report["halo_family_id_other_share_forecast"] = round(
        float(1.0 - kept.sum() / max(len(h), 1) + (h["family"] == "other").mean()), 5)

    used, vocab = None, None
    try:
        from scale1m.hf_canonicalize import core_used_families
        used, vocab = core_used_families(core_path)
        report["halo_family_in_core_used_share"] = round(
            float(h["family"].isin(used).mean()), 5)
        report["core_used_families"] = len(used)
    except Exception as exc:                     # never let an audit stat fail the build
        report["halo_family_in_core_used_share"] = None
        report["family_stat_error"] = str(exc)

    write_json_atomic(os.path.join(out_dir, "LADDER_REPORT.json"), report)
    return ladder, report


def main(argv=None):
    p = argparse.ArgumentParser(description="T3: build the rung mappedID table")
    p.add_argument("--rung", default="100k")
    p.add_argument("--n", type=int, default=100_000)
    p.add_argument("--core", default="stage1BuildTransferGraph/hgraph_ml_v2.pt")
    p.add_argument("--halo", required=True)
    p.add_argument("--canon", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    ladder, rep = build(args.core, args.halo, args.canon, args.n, args.out, args.rung)
    print("\n[ok] ladder %d rows -> %s" % (len(ladder), rep["ladder_csv"]))
    print("  sha256                %s" % rep["ladder_sha256"][:16])
    print("  CORE prefix identical  yes (0..%d)" % (rep["n_core"] - 1))
    print("  CORE-HALO intersection %d | distinct models %d"
          % (rep["core_halo_intersection"], rep["distinct_models"]))
    print("  HALO layer             %s" % rep["halo_layer"])
    print("  HALO size_b known      %.4f (missing %.4f)"
          % (rep["halo_size_b_known"], rep["halo_size_b_missing"]))
    print("  HALO family distinct   %d | 'other' %.4f | in CORE-used %s"
          % (rep["halo_family_distinct"], rep["halo_family_other_share"],
             rep["halo_family_in_core_used_share"]))
    print("  HALO family_id=Other forecast after FAMILY_MIN_COUNT: %.4f"
          % rep["halo_family_id_other_share_forecast"])
    print("  HALO lineage declared  %.4f" % rep["halo_lineage_declared"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
