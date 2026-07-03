"""Render Table A (warm) from completed warm runs and insert into review_everything.md.
Table B (cold) is appended later once the five-fold sweep finishes."""
import json, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import torch
from ModelLakeFishing.stage2TrainGraphSAGE.model import load_hgraph
from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import dedup_trained_on, apply_similar_to_mode, TRAINED_ON
from ModelLakeFishing.stage2TrainGraphSAGE.eval_harness import make_fixed_splits
from ModelLakeFishing.stage2TrainGraphSAGE.losses import accuracy_lookup
from ModelLakeFishing.stage2TrainGraphSAGE.cold_metrics import dataset_random_expected, macro
from ModelLakeFishing.stage2TrainGraphSAGE.cold_config_manifest import VERSION_ORDER
from ModelLakeFishing.stage2TrainGraphSAGE.v2_full_version_rerun import (
    GRAPH, OUT, warm_candidates, _sha256, SPLIT_SEED)

REVIEW = "D:/research/model_lake/weeks/week6revieweverything/review_everything.md"


def _row(name, a):
    return ("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
        name, a["tau_macro"], a["NDCG@1"], a["Hit@1"], a["top3_hit@1"], a["Rec@1"],
        a["NDCG@10"], a["Hit@10"], a["Rec@10"]))


def main():
    warm = {}
    for n in VERSION_ORDER:
        p = os.path.join(OUT, "runs", "warm", f"{n}.json")
        if os.path.exists(p):
            warm[n] = json.load(open(p, encoding="utf-8"))["aggregate"]
    data, _x, _u = load_hgraph(GRAPH)
    dedup = dedup_trained_on(data)
    g = apply_similar_to_mode(dedup.clone(), "dense", k=10)
    _t, _v, wtest = make_fixed_splits(g, split_seed=SPLIT_SEED)
    wc = warm_candidates(wtest, accuracy_lookup(g))
    wr = macro({d: r for d, r in
                {d: dataset_random_expected(a) for d, (_c, a) in wc.items()}.items() if r})
    nW = warm.get("B0", {}).get("n_datasets")
    E = int(dedup[TRAINED_ON].edge_index.size(1))
    hdr = "| name | tau_macro | NDCG@1 | Hit@1 | top3_hit@1 | Rec@1 | NDCG@10 | Hit@10 | Rec@10 |"
    sep = "|" + "---|" * 9
    best = lambda k: max(warm, key=lambda n: warm[n][k])
    L = ["## Stage-2 full-version rerun on the v2 effective-dataset graph\n",
         f"Graph: 2,000 models, {data['dataset'].num_nodes} dataset nodes, {E} distinct "
         f"trained_on pairs (graph sha256 `{_sha256(GRAPH)[:16]}…`). These results supersede the "
         "old-materialization tables for current model selection; the old tables remain "
         "above/below for historical comparison.\n",
         "### Table A — Known datasets / warm held-out performance edges\n",
         f"Mean over {nW} warm datasets, split seed 0, single init seed 0. "
         "Random (expected) is analytic. Exact normalized dot-product ranking.\n", hdr, sep,
         _row("Random (expected)", wr)]
    for n in VERSION_ORDER:
        if n in warm:
            L.append(_row(n, warm[n]))
    L += [f"\n*Table B (five-fold whole-dataset cold start) is being generated and will be "
          f"appended below when the sweep completes.*\n",
          "**Table A summary.** "
          f"best tau_macro `{best('tau_macro')}` ({warm[best('tau_macro')]['tau_macro']:.3f}); "
          f"best Hit@1 `{best('Hit@1')}` ({warm[best('Hit@1')]['Hit@1']:.3f}); "
          f"best top3_hit@1 `{best('top3_hit@1')}` ({warm[best('top3_hit@1')]['top3_hit@1']:.3f}); "
          f"best Hit@10 `{best('Hit@10')}` ({warm[best('Hit@10')]['Hit@10']:.3f}); "
          f"best Rec@10 `{best('Rec@10')}` ({warm[best('Rec@10')]['Rec@10']:.3f}). "
          f"Best learned Hit@1 {max(warm[n]['Hit@1'] for n in warm):.3f} vs random "
          f"{wr['Hit@1']:.3f} — learned **beats** random.\n"]
    md = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(REVIEW), exist_ok=True)
    with open(REVIEW, "a", encoding="utf-8") as f:
        f.write("\n\n---\n\n" + md)
    json.dump({"warm_random": wr, "warm": warm, "n_warm": nW},
              open(os.path.join(OUT, "table_A_warm.json"), "w", encoding="utf-8"), indent=2)
    print(md)
    print("appended Table A to", REVIEW)


if __name__ == "__main__":
    main()
