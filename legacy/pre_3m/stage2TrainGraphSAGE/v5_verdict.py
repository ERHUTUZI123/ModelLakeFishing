"""
v5_verdict.py -- the P1 (content) stratified verdict + §0.6 promise check.

Compares L1L3b on the D0.5+content graph (v5_content) vs the same-lake
content-zeroed control (v5_nocontent), 8 root-aware seeds each. Because the
control is the content graph with only the e_content block zeroed, node ids /
edges / splits are identical per seed -> per-dataset pairing is exact.

Reports, on the has_content stratum (where any content effect must show) and
the no-content stratum (the diff-in-diff control):
  * root_macro gold@10 / gold_gap@10 for content vs nocontent
  * paired bootstrap deltas
And the §0.6 promise check: content-graph root gold@10 vs the 0.22-0.29 target
(closing 25-40% of the L1L3b 0.100 -> O-sibling 0.563 gap) and gold_gap@10 >= 0.45.

Run (repo root):
  ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.stage2TrainGraphSAGE.v5_verdict
"""

import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.eval_harness import paired_bootstrap  # noqa: E402

OUT = os.path.join(_HERE, "artifacts", "ablation", "d0")
CONTENT_GRAPH = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage1BuildTransferGraph",
                             "hgraph_d0_v1_content.pt")
SEEDS = [str(s) for s in range(8)]
METRICS = ["full2k_gold@10", "gold_gap@10", "observed_hit1", "top3_hit1"]


def load_splits(tag):
    d = json.load(open(os.path.join(OUT, tag, "L1L3b.json"), encoding="utf-8"))
    return d["splits"]


def has_content_map():
    p = torch.load(CONTENT_GRAPH, map_location="cpu", weights_only=False)
    udi = p["unique_dataset_id"].sort_values("mappedID")
    hc = p["data"]["dataset"].has_content.numpy().astype(bool)
    gold = None
    # gold nodes: from pool
    pool_p = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage1BuildTransferGraph",
                          "artifacts", "d0_lake", "d0_dataset_pool.csv")
    import pandas as pd
    pool = pd.read_csv(pool_p)
    goldset = set(pool[pool["gold_evaluable"]]["dataset_node"])
    nodes = udi["dataset"].tolist()
    roots = udi["root"].tolist()
    return ({i: bool(hc[i]) for i in range(len(hc))},
            {i: roots[i] for i in range(len(roots))},
            {i: (nodes[i] in goldset) for i in range(len(nodes))})


def root_macro_stratum(splits, hc, root_of, keep):
    """root_macro over datasets where keep(mappedID) is True, per seed then mean."""
    per_seed = {m: [] for m in METRICS}
    n_ds = []
    for ss in SEEDS:
        by_root = {}
        for d, r in splits[ss]["per_dataset"].items():
            di = int(d)
            if not keep(di):
                continue
            by_root.setdefault(root_of[di], []).append(r)
        if not by_root:
            continue
        n_ds.append(sum(len(v) for v in by_root.values()))
        for m in METRICS:
            root_means = [np.mean([row[m] for row in rows]) for rows in by_root.values()]
            per_seed[m].append(float(np.mean(root_means)))
    return ({m: [float(np.mean(per_seed[m])), float(np.std(per_seed[m]))] for m in METRICS},
            int(np.mean(n_ds)) if n_ds else 0)


def pooled(splits, hc, keep, metric):
    out = {}
    for ss in SEEDS:
        for d, r in splits[ss]["per_dataset"].items():
            if keep(int(d)):
                out[f"s{ss}::{d}"] = {metric: r[metric]}
    return out


def main():
    content = load_splits("v5_content")
    nocontent = load_splits("v5_nocontent")
    hc, root_of, is_gold = has_content_map()

    strata = {
        "all": lambda i: True,
        "has_content": lambda i: hc.get(i, False),
        "no_content": lambda i: not hc.get(i, False),
        "gold_has_content": lambda i: hc.get(i, False) and is_gold.get(i, False),
        "gold_all": lambda i: is_gold.get(i, False),
    }
    report = {"graph": "hgraph_d0_v1_content.pt", "seeds": 8, "strata": {}}
    for name, keep in strata.items():
        c_rm, c_n = root_macro_stratum(content, hc, root_of, keep)
        n_rm, n_n = root_macro_stratum(nocontent, hc, root_of, keep)
        boot = {}
        for m in ("full2k_gold@10", "gold_gap@10"):
            bb = paired_bootstrap(pooled(nocontent, hc, keep, m),
                                  pooled(content, hc, keep, m), metric=m)
            boot[m] = [round(bb["ci_low"], 4), round(bb["ci_high"], 4)]
        report["strata"][name] = {
            "n_datasets_per_seed": c_n,
            "content": c_rm, "nocontent": n_rm,
            "delta_content_minus_nocontent_CI": boot,
        }

    # §0.6 promise check on the content graph, all-datasets root_macro
    c_all = report["strata"]["all"]["content"]
    g10 = c_all["full2k_gold@10"][0]
    gap = c_all["gold_gap@10"][0]
    report["promise_check"] = {
        "root_gold@10": round(g10, 4), "target_0.22_0.29": bool(0.22 <= g10 <= 0.29),
        "root_gold@10_ge_0.22": bool(g10 >= 0.22),
        "root_gold_gap@10": round(gap, 4), "gap_ge_0.45": bool(gap >= 0.45),
        "fraction_of_gap_closed": round((g10 - 0.100) / (0.563 - 0.100), 3),
    }
    json.dump(report, open(os.path.join(OUT, "v5_verdict.json"), "w"), indent=2)

    # markdown
    L = ["# v5 combined verdict -- P1 content, stratified (L1L3b, 8 seeds)", "",
         "root_macro gold@10 / gold_gap@10, content graph vs content-zeroed control:", "",
         "| stratum | n_ds | gold@10 (content) | gold@10 (ctrl) | Δgold@10 CI | gap@10 (content) | gap@10 (ctrl) | Δgap@10 CI |",
         "|---|---|---|---|---|---|---|---|"]
    for name, s in report["strata"].items():
        c, n = s["content"], s["nocontent"]
        L.append(f"| {name} | {s['n_datasets_per_seed']} | "
                 f"{c['full2k_gold@10'][0]:.3f}±{c['full2k_gold@10'][1]:.3f} | "
                 f"{n['full2k_gold@10'][0]:.3f}±{n['full2k_gold@10'][1]:.3f} | "
                 f"{s['delta_content_minus_nocontent_CI']['full2k_gold@10']} | "
                 f"{c['gold_gap@10'][0]:.3f} | {n['gold_gap@10'][0]:.3f} | "
                 f"{s['delta_content_minus_nocontent_CI']['gold_gap@10']} |")
    pc = report["promise_check"]
    L += ["", f"**Promise check (all datasets, content graph):** root gold@10 = "
          f"{pc['root_gold@10']} (target 0.22-0.29: {pc['target_0.22_0.29']}, "
          f"gap closed {pc['fraction_of_gap_closed']:.0%}); root gold-gap@10 = "
          f"{pc['root_gold_gap@10']} (>=0.45: {pc['gap_ge_0.45']})."]
    open(os.path.join(OUT, "v5_verdict.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
