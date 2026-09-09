"""Replace the stale Table B (old 9-cold historical block) in review_graph_training.md
with the new v2 five-fold cold Table B, pooled per-dataset across folds (each dataset
evaluated exactly once as cold). Requires all 5 cold folds present for every version.
Random (expected) is the first row (analytic, pooled over the same cold candidate sets).
"""
import json, os, re, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import numpy as np, torch
from ModelLakeFishing.stage2TrainGraphSAGE.model import load_hgraph
from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import dedup_trained_on, TRAINED_ON
from ModelLakeFishing.stage2TrainGraphSAGE.cold_graph import cold_labels_vault
from ModelLakeFishing.stage2TrainGraphSAGE.cold_metrics import dataset_random_expected, macro
from ModelLakeFishing.stage2TrainGraphSAGE.cold_config_manifest import VERSION_ORDER
from ModelLakeFishing.stage2TrainGraphSAGE.v2_full_version_rerun import (
    GRAPH, OUT, N_FOLDS, build_cold_folds)

TARGET = "D:/research/model_lake/weeks/week6_review_graph_training/review_graph_training.md"


def _row(name, a):
    return ("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
        name, a["tau_macro"], a["NDCG@1"], a["Hit@1"], a["top3_hit@1"], a["Rec@1"],
        a["NDCG@10"], a["Hit@10"], a["Rec@10"]))


def _complete_folds():
    """Folds for which EVERY version has a saved run (so the pooled table is
    comparable across versions). Returns a sorted list, possibly < N_FOLDS."""
    done = []
    for f in range(N_FOLDS):
        if all(os.path.exists(os.path.join(OUT, "runs", "cold", f"fold_{f}", f"{n}.json"))
               for n in VERSION_ORDER):
            done.append(f)
    return done


def build_table_b():
    data, _x, _u = load_hgraph(GRAPH)
    dedup = dedup_trained_on(data)
    folds, fold_man = build_cold_folds(dedup, GRAPH)

    use_folds = _complete_folds()
    if not use_folds:
        raise SystemExit("no fold is complete for all versions yet")
    partial = len(use_folds) < N_FOLDS

    cold_tbl = {}
    for n in VERSION_ORDER:
        pooled = {}
        for f in use_folds:
            p = os.path.join(OUT, "runs", "cold", f"fold_{f}", f"{n}.json")
            for d, v in json.load(open(p, encoding="utf-8"))["per_dataset"].items():
                pooled[str(d)] = v
        if pooled:
            cold_tbl[n] = macro({int(d): v for d, v in pooled.items()})
    if not cold_tbl:
        raise SystemExit("no cold results yet")

    # pooled analytic random over the used folds' cold vaults (each dataset once)
    rper = {}
    for f in use_folds:
        for d, v in cold_labels_vault(dedup, folds[f]).items():
            r = dataset_random_expected(np.array([a for _m, a in v], dtype=float))
            if r is not None:
                rper[int(d)] = r
    rand = macro(rper)
    nC = cold_tbl.get("B0", next(iter(cold_tbl.values())))["n_datasets"]
    sizes = [fold_man["fold_sizes"][f] for f in use_folds]

    hdr = "| name | tau_macro | NDCG@1 | Hit@1 | top3_hit@1 | Rec@1 | NDCG@10 | Hit@10 | Rec@10 |"
    sep = "|" + "---|" * 9
    scope = (f"five cold folds (sizes {fold_man['fold_sizes']})" if not partial
             else f"**{len(use_folds)} of 5 cold folds (folds {use_folds}, sizes {sizes}) — "
                  "PARTIAL, remaining folds still training**")
    L = ["## 5. Table B -- Completely unseen cold dataset IDs", "",
         f"Pooled macro mean over {nC} completely unseen datasets from {scope}, single init "
         "seed 0. Each dataset is evaluated exactly once as cold. Same frozen `z_m`; inductive "
         "`z_d`. Random (expected) is analytic. **v2 effective-dataset graph (2,000 models, "
         f"{data['dataset'].num_nodes} dataset nodes, "
         f"{int(dedup[TRAINED_ON].edge_index.size(1))} distinct trained_on pairs); supersedes "
         "the old 9-cold table.**", "", hdr, sep, _row("Random (expected)", rand)]
    for n in VERSION_ORDER:
        if n in cold_tbl:
            L.append(_row(n, cold_tbl[n]))
    return "\n".join(L), cold_tbl, rand, nC, fold_man


def replace_block(new_block):
    lines = open(TARGET, encoding="utf-8").read().split("\n")
    start = next(i for i, ln in enumerate(lines) if ln.strip().startswith("## 5. Table B"))
    # block ends at the last consecutive markdown table row after the header
    end = start
    for i in range(start, len(lines)):
        if lines[i].strip().startswith("|"):
            end = i
    new_lines = lines[:start] + new_block.split("\n") + lines[end + 1:]
    open(TARGET, "w", encoding="utf-8").write("\n".join(new_lines))
    return start, end


def main():
    block, cold_tbl, rand, nC, fold_man = build_table_b()
    start, end = replace_block(block)
    best = lambda k: max(cold_tbl, key=lambda n: cold_tbl[n][k])
    print(f"replaced review_graph_training.md lines {start+1}-{end+1} with v2 cold Table B "
          f"({nC} datasets, folds {fold_man['fold_sizes']})")
    print(f"best cold tau_macro `{best('tau_macro')}` ({cold_tbl[best('tau_macro')]['tau_macro']:.3f}); "
          f"Hit@1 `{best('Hit@1')}`; Hit@10 `{best('Hit@10')}` ({cold_tbl[best('Hit@10')]['Hit@10']:.3f}); "
          f"Rec@10 `{best('Rec@10')}`")
    print(f"best learned Hit@10 {max(cold_tbl[n]['Hit@10'] for n in cold_tbl):.3f} vs random {rand['Hit@10']:.3f}")


if __name__ == "__main__":
    main()
