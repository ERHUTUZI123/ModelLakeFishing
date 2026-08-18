"""F1 forecast: the quantities F2/F4/F5/F8 can be bounded now.

Same purpose as R2's `verify_raw --forecast`: write down, before the work, the
numbers whose surprise value is highest, so that no later stage can be quietly
re-narrated. None of this is a gate.

Four questions:
  1. lineage -- how many `is_base_of` edges does a FULL lake resolve? R2 got
     16,385 because most parents were outside its 100K lake. This is D-8's
     real test.
  2. layers  -- labeled / lineage / plain / dropped over the whole lake.
  3. size    -- the descriptor's size-clause coverage against CORE's 47.91%.
  4. shape   -- publisher and family concentration, i.e. exactly how the real
     lake differs from R2's quota-balanced one (1Mplan 3.3).
"""
import collections
import json
import os
import sys

sys.path.insert(0, r"D:\research\model_lake\codes\ModelLakeFishing")

import torch

from scale1m.verify_raw import base_model_of, iter_records, normalize

DATA = os.environ.get("MLF_DATA_DIR", r"D:\research\model_lake\data")
CRAWL = os.path.join(DATA, "data1m", "candidates_full")
CORE = r"D:\research\model_lake\codes\ModelLakeFishing\stage1BuildTransferGraph\hgraph_ml_v2.pt"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f1_forecast.json")


def pct(a, b):
    return round(100.0 * a / max(b, 1), 3)


def main():
    with open(os.path.join(CRAWL, "PROVENANCE.json"), encoding="utf-8") as fh:
        prov = json.load(fh)

    core = torch.load(CORE, weights_only=False)
    core_ids = {normalize(m) for m in core["unique_model_id"]["model"]}
    sb = core["data"]["model"].size_bucket_id
    core_size_cov = pct(int((sb != 0).sum()), sb.numel())

    # pass 1: the id universe (needed before base_model can be resolved)
    ids = set()
    for _, rec in iter_records(CRAWL, prov["shards"]):
        ids.add(normalize(rec.get("id")))
    lake = core_ids | ids                      # N_full universe
    print("pass 1 done: %d crawl ids, lake %d" % (len(ids), len(lake)))

    # pass 2: everything else
    n = 0
    declared = resolved = self_ref = 0
    src_core = dst_core = 0
    parent_children = collections.Counter()
    layer = collections.Counter()
    size_known = 0
    authors = collections.Counter()
    gguf_like = quantized_like = 0
    relation = collections.Counter()

    for _, rec in iter_records(CRAWL, prov["shards"]):
        n += 1
        mid = normalize(rec.get("id"))
        card = rec.get("cardData") or {}
        has_size = bool((rec.get("safetensors") or {}).get("total"))
        size_known += has_size

        if not rec.get("pipeline_tag") and not rec.get("tags") and not has_size:
            layer["dropped"] += 1
        elif card.get("model-index"):
            layer["labeled"] += 1
        elif base_model_of(rec):
            layer["lineage"] += 1
        else:
            layer["plain"] += 1

        a = rec.get("author") or (rec.get("id") or "/").split("/")[0]
        authors[a] += 1

        tags = [str(t).lower() for t in (rec.get("tags") or [])]
        if "gguf" in tags:
            gguf_like += 1
        if any(t in tags for t in ("gguf", "awq", "gptq", "4-bit", "8-bit", "exl2", "mlx")):
            quantized_like += 1

        b = base_model_of(rec)
        if b:
            declared += 1
            if b == mid:
                self_ref += 1
            elif b in lake:
                resolved += 1
                parent_children[b] += 1
                if mid in core_ids:
                    dst_core += 1
                if b in core_ids:
                    src_core += 1
            r = card.get("base_model_relation")
            relation[str(r) if r else "unknown"] += 1

    top_authors = authors.most_common(10)
    hhi = sum((c / n) ** 2 for c in authors.values())
    top_parents = parent_children.most_common(10)
    child_counts = sorted(parent_children.values())

    res = {
        "n_records": n,
        "n_full_universe": len(lake),
        "lineage": {
            "declared_base_model": declared,
            "declared_pct": pct(declared, n),
            "self_referential_dropped": self_ref,
            "resolvable_inside_the_lake": resolved,
            "resolution_rate_pct": pct(resolved, declared),
            "edges_with_a_CORE_parent": src_core,
            "edges_with_a_CORE_child": dst_core,
            "distinct_parents": len(parent_children),
            "max_children_of_one_parent": child_counts[-1] if child_counts else 0,
            "children_p50": child_counts[len(child_counts) // 2] if child_counts else 0,
            "children_p99": (child_counts[int(0.99 * (len(child_counts) - 1))]
                             if child_counts else 0),
            "top10_parents": top_parents,
            "relation_field": relation.most_common(10),
            "R2_comparison": {"edges": 16385, "declared": 27672, "declared_pct": 39.64},
        },
        "layers": {k: {"n": v, "pct": pct(v, n)} for k, v in layer.items()},
        "size_clause": {
            "CORE_pct": core_size_cov,
            "full_lake_pct": pct(size_known, n),
            "gap_pp": round(pct(size_known, n) - core_size_cov, 2),
            "R2_halo_pct": 57.75,
        },
        "concentration": {
            "distinct_authors": len(authors),
            "effective_authors_1_over_hhi": round(1.0 / hhi, 1),
            "top1_author_pct": pct(top_authors[0][1], n),
            "top10_authors_pct": pct(sum(c for _, c in top_authors), n),
            "top10_authors": top_authors,
            "gguf_tagged_pct": pct(gguf_like, n),
            "quantization_tagged_pct": pct(quantized_like, n),
            "R2_balanced": {"top1_pct": 0.902, "top10_pct": 4.045,
                            "effective_authors": 2889.1,
                            "third_party_quant_pct": 14.999},
        },
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False)[:4000])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
