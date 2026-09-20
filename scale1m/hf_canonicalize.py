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
    import torch
    core = torch.load(core_path, weights_only=False)
    vocab = core["xm0_meta"]["family_vocab"]
    inv = {v: k for k, v in vocab.items()}
    used = {inv[int(i)] for i in set(core["data"]["model"].family_id.tolist())}
    return used, vocab


def size_b_of(rec):
    total = (rec.get("safetensors") or {}).get("total")
    return (float(total) / 1e9, "safetensors") if total else (None, "none")


def lineage_base_of(rec):
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
