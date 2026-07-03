"""Run and consolidate the provenance-clean five-metric evaluation for all trials.

The original ablation JSON files predate ``dedup_trained_on`` and are not reused.
Every missing row is retrained on the 7,056-distinct-pair graph view with the same
split seeds and init seed used by ``top1_baselines.py``.  Existing clean T0/G rows
are then combined with these runs into one machine-readable artifact and table.

Run:
    python -m ModelLakeFishing.stage2TrainGraphSAGE.top1_full_versions
"""

from __future__ import annotations

import json
import os

from ModelLakeFishing.stage2TrainGraphSAGE.top1_audit import configs as clean_configs
from ModelLakeFishing.stage2TrainGraphSAGE.top1_baselines import OUT, run_configs


ORDER = [
    "B0", "B1", "B2", "B2e_ctrl", "B3_sim", "B3_simtr", "B3_all",
    "B4_grouped", "B6_heads", "BEST", "B5_ranknet", "R_tohet",
    "R_weights", "R_heads", "R_mg00", "R_mg02", "P6_dm05",
    "P6_dm10", "P7_es", "P7_lr3e3", "G1", "G2", "G1dm",
]


def _base():
    return dict(clean_configs()["B0"])


def _b5():
    return dict(clean_configs()["B5_ranknet"])


def configs_25():
    b0 = _base()
    ctrl = _base()
    ctrl.update(similar_to_mode="topk_unweighted", edge_aware=True,
                weighted_relations=[])
    b5 = _b5()

    def changed(base, **kwargs):
        out = dict(base)
        out.update(kwargs)
        return out

    return {
        "B1": changed(b0, similar_to_mode="none"),
        "B2": changed(b0, similar_to_mode="topk_unweighted"),
        "B2e_ctrl": dict(ctrl),
        "B3_sim": changed(ctrl, weighted_relations=["similar_to"]),
        "B3_simtr": changed(
            ctrl,
            weighted_relations=["similar_to", "trained_on", "rev_trained_on"],
        ),
        "B3_all": changed(ctrl, weighted_relations=None),
        "B4_grouped": changed(ctrl, grouped=True),
        "B6_heads": changed(ctrl, separate_heads=True),
        # Exact historical BEST command: edge-aware + all weights + RankNet +
        # separate heads + grouped full-batch training.
        "BEST": changed(
            b0, similar_to_mode="topk_unweighted", edge_aware=True,
            weighted_relations=None, rank_loss="ranknet", rank_min_gap=0.01,
            separate_heads=True, grouped=True,
        ),
        "R_tohet": changed(
            b0, similar_to_mode="topk_unweighted", rank_loss="ranknet",
            rank_min_gap=0.01,
        ),
        "R_weights": changed(b5, weighted_relations=None),
        "R_heads": changed(b5, separate_heads=True),
        "R_mg00": changed(b5, rank_min_gap=0.0),
        "P6_dm05": changed(b5, lambda_dm_contrast=0.5),
    }


def configs_40():
    b5 = _b5()
    es = dict(b5)
    es.update(early_stop=True, patience=10)
    lr = dict(b5)
    lr["lr"] = 0.003
    return {"P7_es": es, "P7_lr3e3": lr}


def _artifact_path(name):
    if name in {"B0", "B5_ranknet", "R_mg02", "P6_dm10"}:
        return os.path.join(OUT, "T0", f"{name}.json")
    if name in {"G1", "G2"}:
        return os.path.join(OUT, "G", f"{name}.json")
    if name == "G1dm":
        return os.path.join(OUT, "G1dm", "G1dm.json")
    tag = "FULL40" if name in {"P7_es", "P7_lr3e3"} else "FULL25"
    return os.path.join(OUT, tag, f"{name}.json")


def consolidate():
    rows = {}
    provenance = {}
    for name in ORDER:
        path = _artifact_path(name)
        with open(path, encoding="utf-8") as f:
            artifact = json.load(f)
        agg = artifact["aggregate_over_splits"]
        rows[name] = {
            "observed_hit@1": agg["observed_hit1"],
            "top3_hit@1": agg["top3_hit1"],
            "regret@1": agg["regret1"],
            "full2k_gold@1": agg["full2k_gold@1"],
            "full2k_gold@10": agg["full2k_gold@10"],
        }
        provenance[name] = os.path.relpath(path, OUT).replace("\\", "/")

    with open(os.path.join(OUT, "TOP1_BASELINES.json"), encoding="utf-8") as f:
        baseline = json.load(f)
    rnd = baseline["random_baselines"]
    random_row = {
        "observed_hit@1": [rnd["random_observed"]["observed_hit1"], 0.0],
        "top3_hit@1": [rnd["random_observed"]["top3_hit1"], 0.0],
        "regret@1": [rnd["random_observed"]["regret1"], 0.0],
        "full2k_gold@1": [rnd["random_fullpool"]["full2k_gold@1"], 0.0],
        "full2k_gold@10": [rnd["random_fullpool"]["full2k_gold@10"], 0.0],
    }

    output = {
        "note": (
            "All learned rows were retrained/evaluated after dedup_trained_on; "
            "pre-fix 12,205-row results are superseded. Values are [mean, std] "
            "over fixed split seeds 0,1,2 with init seed 0."
        ),
        "random_expected": random_row,
        "rows": rows,
        "artifact_provenance": provenance,
    }
    json_path = os.path.join(OUT, "FULL_VERSIONS_FIVE_METRICS.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    lines = [
        "# Full-version clean five-metric table",
        "",
        "All learned rows use the provenance-fixed graph (12,205 rows -> 7,056 "
        "distinct model-dataset pairs), split seeds `{0,1,2}`, and init seed `0`. "
        "Old pre-dedup measurements are superseded.",
        "",
        "| name | observed_hit@1 | top3_hit@1 | regret@1 | full2k_gold@1 | full2k_gold@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def fmt(name, row, show_std=True):
        hit = row["observed_hit@1"]
        h = f"{hit[0]:.3f} +/- {hit[1]:.3f}" if show_std else f"{hit[0]:.3f}"
        return (
            f"| {name} | {h} | {row['top3_hit@1'][0]:.3f} | "
            f"{row['regret@1'][0]:.4f} | {row['full2k_gold@1'][0]:.3f} | "
            f"{row['full2k_gold@10'][0]:.3f} |"
        )

    lines.append(fmt("Random (expected)", random_row, show_std=False))
    for name in ORDER:
        lines.append(fmt(name, rows[name]))
    lines.extend([
        "",
        "`full2k_gold@K` is gold survival among all 2,000 model embeddings, not "
        "full-pool precision; unobserved models are unknown rather than incorrect.",
    ])
    md_path = os.path.join(OUT, "FULL_VERSIONS_FIVE_METRICS.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return md_path, json_path


def main():
    _, pooled25, _ = run_configs(
        configs_25(), tag="FULL25", epochs=25,
        extra_note="Full-version clean five-metric sweep.",
    )
    with open(os.path.join(OUT, "FULL25", "pooled_per_dataset.json"), "w",
              encoding="utf-8") as f:
        json.dump(pooled25, f)

    _, pooled40, _ = run_configs(
        configs_40(), tag="FULL40", epochs=40,
        extra_note="Full-version clean five-metric sweep (40 epochs).",
    )
    with open(os.path.join(OUT, "FULL40", "pooled_per_dataset.json"), "w",
              encoding="utf-8") as f:
        json.dump(pooled40, f)

    md_path, json_path = consolidate()
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
