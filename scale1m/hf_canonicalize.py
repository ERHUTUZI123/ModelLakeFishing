"""
hf_canonicalize.py -- T3 stage 1: reduce a frozen candidate snapshot to the
FOUR quantities the graph builder is allowed to see, plus an audit-only layer.

Runbook: docs/1M/100kplan.md 6.1.   Record: docs/1M/T3.md.

THE FOUR QUANTITIES (iron rule 2 -- nothing else reaches the descriptor)
    unique_model_id   normalize(id) = strip().lower(); the dedupe key shared
                      with scale1m.verify_raw.normalize
    size_b            safetensors.total / 1e9 ONLY (D-26). No name regex.
    family            the string that will be looked up in CORE's family_vocab
    lineage_base      the declared parent, normalized

WHY `family` IS NOT JUST `_infer_one_family` (F-T3-1, measured)
    CORE's family strings came from ModelLens's `model_profile.family`, which
    is the HF `config.model_type` namespace: `bert`, `llama`, `vit`, `marian`.
    `_infer_one_family` returns the *rule table's* namespace: `BERT`, `LLaMA`,
    `ViT`. Those are DIFFERENT vocab entries -- and `family_vocab` already
    contains both, because KNOWN_FAMILIES is seeded and the ModelLens strings
    were admitted dynamically on top.

    Feeding HALO through `_infer_one_family` alone puts **0.64%** of it into a
    family CORE actually uses. Every llama in HALO would get row `LLaMA` while
    every llama in CORE sits in row `llama`: same architecture, two embedding
    rows, zero sharing, and nothing anywhere would raise.

    So resolution goes, in order:
      1. `config.model_type` if CORE already uses that family   (authoritative)
      2. lowercased `_infer_one_family` if CORE already uses it (rule fallback)
      3. `config.model_type` if present                         (new row, right namespace)
      4. lowercased `_infer_one_family`                         (last resort)
    Measured on the selected HALO: 29.63% -> 48.47% land in a CORE-used family.
    `family_source` records which rule fired for every row.

Run (from ModelLakeFishing/):
    .\\.venv\\Scripts\\python.exe -m scale1m.hf_canonicalize `
        --candidates <dir> --core stage1BuildTransferGraph/hgraph_ml_v2.pt --out <dir>
"""

import argparse
import gzip
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "stage1BuildTransferGraph"))

from scale1m.hf_crawl import default_candidates_out, utcnow, write_json_atomic
from dataset_embed.utils.fetch_metadata import _infer_one_family

LAYERS = ("labeled", "lineage", "plain", "dropped")


def normalize(mid) -> str:
    """The one id rule, shared with verify_raw / annotate / build_ladder."""
    return str(mid).strip().lower()


def load_shards(cdir):
    prov = json.load(open(os.path.join(cdir, "PROVENANCE.json"), encoding="utf-8"))
    recs = []
    for sh in prov["shards"]:
        with gzip.open(os.path.join(cdir, sh["file"]), "rt", encoding="utf-8") as fh:
            for line in fh:
                recs.append(json.loads(line))
    return recs, prov


def core_used_families(core_path):
    """The families CORE's embedding table actually has rows in USE for.

    Not `family_vocab.keys()`: the vocab is seeded with KNOWN_FAMILIES, most of
    which no CORE model ever hit. Sharing a row only matters if CORE put models
    in it, so the *used* set is what the resolution policy targets.
    """
    import torch
    core = torch.load(core_path, weights_only=False)
    vocab = core["xm0_meta"]["family_vocab"]
    inv = {v: k for k, v in vocab.items()}
    used = {inv[int(i)] for i in set(core["data"]["model"].family_id.tolist())}
    return used, vocab


def size_b_of(rec):
    """D-26: safetensors ONLY. A missing value stays missing."""
    total = (rec.get("safetensors") or {}).get("total")
    return (float(total) / 1e9, "safetensors") if total else (None, "none")


def lineage_base_of(rec):
    """-> (normalized parent id, relation, source).

    `baseModels` outranks `cardData.base_model`: the plan text says cardData,
    but T2 established that baseModels is HF's own structured relation while
    cardData.base_model is a free-text string an author typed. Both are read;
    which one fired is recorded.
    """
    bm = rec.get("baseModels")
    if isinstance(bm, dict) and bm.get("ids"):
        return normalize(bm["ids"][0]), bm.get("relation"), "baseModels"
    b = (rec.get("cardData") or {}).get("base_model")
    if isinstance(b, list):
        b = b[0] if b else None
    if isinstance(b, str) and b.strip():
        return normalize(b), (rec.get("cardData") or {}).get("base_model_relation"), "cardData.base_model"
    return None, None, "none"


def family_of(rec, used):
    """-> (family string, source). See the module docstring for the ordering."""
    mt = str((rec.get("config") or {}).get("model_type") or "").strip().lower()
    inferred = _infer_one_family(rec.get("id") or "")
    inf_low = str(inferred or "").strip().lower()
    if mt and mt in used:
        return mt, "config.model_type|core_used"
    if inf_low and inf_low in used:
        return inf_low, "name_rule|core_used"
    if mt:
        return mt, "config.model_type|new"
    return inf_low or "other", "name_rule|new"


def layer_of(rec, lineage_base):
    """Audit-only stratum (D-10: never enters training).

    `dropped` is kept verbatim from the plan even though D-27 recorded that it
    is a dead rule -- HF auto-tags every repo, so the conjunction is never
    true. Kept so the count is *reported* as zero rather than silently absent.
    """
    card = rec.get("cardData") or {}
    if not rec.get("pipeline_tag") and not rec.get("tags") \
            and not (rec.get("safetensors") or {}).get("total"):
        return "dropped"
    if card.get("model-index"):
        return "labeled"
    if lineage_base:
        return "lineage"
    return "plain"


def canonicalize(cdir, core_path, out_path, report_path):
    recs, prov = load_shards(cdir)
    used, vocab = core_used_families(core_path)
    print("[load] %d records from %s | CORE uses %d of %d vocab families"
          % (len(recs), cdir, len(used), len(vocab)))

    rows = []
    for rank, r in enumerate(recs):
        mid = r.get("id") or ""
        size_b, size_src = size_b_of(r)
        base, relation, base_src = lineage_base_of(r)
        family, fam_src = family_of(r, used)
        rows.append({
            "model": mid,
            "id_norm": normalize(mid),
            "size_b": size_b,
            "size_source": size_src,
            "family": family,
            "family_source": fam_src,
            "family_in_core_used": family in used,
            "lineage_base": base,
            "lineage_relation": relation,
            "lineage_source": base_src,
            "layer": layer_of(r, base),
            "downloads": r.get("downloads") or 0,
            "rank": rank,
        })
    df = pd.DataFrame(rows)

    dup = len(df) - df["id_norm"].nunique()
    if dup:
        raise SystemExit("[abort] %d duplicate normalized ids in the snapshot" % dup)
    df.to_parquet(out_path, index=False)

    report = {
        "written_at": utcnow(),
        "candidates_dir": os.path.abspath(cdir),
        "core": os.path.abspath(core_path),
        "snapshot_date_utc": prov.get("snapshot_date_utc"),
        "n": len(df),
        "size_b_known": float((df["size_source"] == "safetensors").mean()),
        "family_distinct": int(df["family"].nunique()),
        "family_in_core_used_share": float(df["family_in_core_used"].mean()),
        "family_source": df["family_source"].value_counts().to_dict(),
        "family_other_share": float((df["family"] == "other").mean()),
        "lineage_declared": float(df["lineage_base"].notna().mean()),
        "lineage_source": df["lineage_source"].value_counts().to_dict(),
        "lineage_relation": df["lineage_relation"].value_counts(dropna=False).head(12).to_dict(),
        "layer": df["layer"].value_counts().to_dict(),
        "core_used_families": len(used),
        "core_vocab_size": len(vocab),
    }
    write_json_atomic(report_path, report)
    return df, report


def main(argv=None):
    p = argparse.ArgumentParser(description="T3: canonicalize a frozen HF snapshot")
    p.add_argument("--candidates", default=default_candidates_out("v2"))
    p.add_argument("--core", default="stage1BuildTransferGraph/hgraph_ml_v2.pt")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    out_dir = args.out or os.path.join(args.candidates, "canon")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "hf_canon.parquet")
    df, rep = canonicalize(args.candidates, args.core, out,
                           os.path.join(out_dir, "CANON_REPORT.json"))
    print("\n[ok] %d canonicalized -> %s" % (len(df), out))
    print("  size_b known         %.4f  (D-26: safetensors only)" % rep["size_b_known"])
    print("  family distinct      %d" % rep["family_distinct"])
    print("  family in CORE-used  %.4f" % rep["family_in_core_used_share"])
    print("  family source        %s" % rep["family_source"])
    print("  lineage declared     %.4f  %s" % (rep["lineage_declared"], rep["lineage_source"]))
    print("  layer                %s" % rep["layer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
