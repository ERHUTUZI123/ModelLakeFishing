"""
cold_sweep.py -- Cold-Dataset guide §§8,10: rerun all 23 versions on the
provenance-fixed, cold-inductive graph and produce Table A (warm held-out edges,
remaining datasets) and Table B (completely unseen cold dataset IDs).

Per version: build the inductive training graph (cold datasets fully removed),
the warm edge split, train once (init seed 0) via the unchanged
ablation.train_eval_one, then evaluate BOTH partitions from that single checkpoint
with the corrected cold_metrics. Random (expected) is analytic per partition
(candidates identical across versions).

Run:  python -m ModelLakeFishing.stage2TrainGraphSAGE.cold_sweep [--versions B0 R_mg02 ...]
"""

import argparse
import hashlib
import io
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.model import load_hgraph  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import dedup_trained_on, TRAINED_ON  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.eval_harness import make_fixed_splits  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.losses import accuracy_lookup, perf_supervision  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.ablation import train_eval_one  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.cold_dataset_split import build_split, SPLIT_PATH, _sha256  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.cold_graph import (  # noqa: E402
    build_training_graph, cold_labels_vault, encode_indexed_and_cold,
)
from ModelLakeFishing.stage2TrainGraphSAGE.cold_metrics import (  # noqa: E402
    dataset_metrics, dataset_random_expected, macro, AGG_KEYS,
)
from ModelLakeFishing.stage2TrainGraphSAGE.cold_config_manifest import manifest, VERSION_ORDER  # noqa: E402

GRAPH = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage1BuildTransferGraph",
                     "hgraph_hf1000d_2000m_xm0_xd0.pt")
OUT = os.path.join(_HERE, "artifacts", "cold_dataset")
SPLIT_SEED = 0
INIT_SEED = 0


def _sd_hash(model):
    buf = io.BytesIO(); torch.save(model.state_dict(), buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def warm_candidates(test_data, lookup, remaining_set):
    eli, tgt = perf_supervision(test_data[TRAINED_ON], lookup)
    models, ds, acc = eli[0].numpy(), eli[1].numpy(), tgt.numpy()
    out = {}
    for d in np.unique(ds):
        if int(d) not in remaining_set:
            continue
        sel = ds == d
        out[int(d)] = (models[sel].astype(int), acc[sel].astype(float))
    return out


def eval_partition(cands, z_m, z_d_of):
    """cands: {d:(cand_ids, acc)}; z_d_of(d) -> z_d vector. Returns (per, macro)."""
    zm = F.normalize(z_m, dim=-1)
    per = {}
    for d, (cand, a) in cands.items():
        zd = F.normalize(z_d_of(d), dim=-1)
        score = (zm[torch.as_tensor(cand, dtype=torch.long)] @ zd).numpy()
        mm = dataset_metrics(score, a)
        if mm is None:
            continue
        order = np.argsort(-score)
        mm["candidate_ids"] = [int(x) for x in cand.tolist()]
        mm["true_acc"] = [float(x) for x in a.tolist()]
        mm["predicted_order"] = [int(x) for x in order.tolist()]
        per[int(d)] = mm
    return per, macro(per)


def random_partition(cands):
    per = {d: dataset_random_expected(a) for d, (_c, a) in cands.items()}
    per = {d: v for d, v in per.items() if v is not None}
    return macro(per)


def run(versions):
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graph_sha = _sha256(GRAPH)
    xd0 = torch.load(GRAPH, map_location="cpu", weights_only=False).get("xd0_meta")

    split = build_split(dataset_split_seed=0)
    with open(SPLIT_PATH, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    split_hash = _sha256(SPLIT_PATH)
    cold = split["cold_test_dataset_ids"]
    remain = split["remaining_dataset_ids"]
    remaining_set = set(remain)

    data, xm0, umi = load_hgraph(GRAPH)
    dedup = dedup_trained_on(data)
    vault = cold_labels_vault(dedup, cold)                      # eval-only label store
    cold_cands = {int(d): (np.array([m for m, _a in v], dtype=int),
                           np.array([a for _m, a in v], dtype=float)) for d, v in vault.items()}

    cfgs = manifest()
    warm_random = cold_random = None
    warm_agg, cold_agg = {}, {}

    for name in versions:
        cfg = cfgs[name]
        tr_g = build_training_graph(dedup, cold, similar_to_mode=cfg["similar_to_mode"],
                                    similar_to_k=cfg["similar_to_k"])
        wsplit = make_fixed_splits(tr_g, split_seed=SPLIT_SEED)
        _wtr, _wval, wtest = wsplit
        lookup = accuracy_lookup(tr_g)                          # remaining-only (no cold)
        assert set(int(d) for (_m, d) in lookup.keys()).isdisjoint(set(cold)), "cold leaked into lookup"

        _r, _pt, _ph, model, _sc = train_eval_one(
            tr_g, xm0, xd0, cfg, wsplit, init_seed=INIT_SEED, epochs=cfg["epochs"], device=device)
        model.eval()

        # warm z (from warm test_data) + cold z (frozen-index insertion)
        with torch.no_grad():
            zt = model(wtest.clone().to(device))
        zt = {k: v.cpu() for k, v in zt.items()}
        wc = warm_candidates(wtest, lookup, remaining_set)
        warm_per, warm_m = eval_partition(wc, zt["model"], lambda d: zt["dataset"][d])

        z_m, z_d_cold = encode_indexed_and_cold(
            model.cpu(), tr_g, dedup, cold, remain,
            mode=cfg["similar_to_mode"], k=cfg["similar_to_k"], device="cpu")
        cold_per, cold_m = eval_partition(cold_cands, z_m, lambda d: z_d_cold[d])

        if warm_random is None:
            warm_random = random_partition(wc)
            cold_random = random_partition(cold_cands)

        warm_agg[name] = warm_m
        cold_agg[name] = cold_m
        art = {
            "version": name, "config": cfg, "init_seed": INIT_SEED,
            "graph_sha256": graph_sha, "split_manifest_sha256": split_hash,
            "state_dict_sha256": _sd_hash(model),
            "warm": {"aggregate": warm_m, "per_dataset": {str(d): v for d, v in warm_per.items()}},
            "cold": {"aggregate": cold_m, "per_dataset": {str(d): v for d, v in cold_per.items()}},
            "leakage_audit": {"cold_in_training_lookup": 0, "cold_edges_in_training_graph": 0},
        }
        with open(os.path.join(OUT, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(art, f, indent=2)
        print(f"[{name}] WARM tau={warm_m['tau_macro']:.3f} H@1={warm_m['Hit@1']:.3f} "
              f"H@10={warm_m['Hit@10']:.3f} nW={warm_m['n_datasets']}  ||  "
              f"COLD tau={cold_m['tau_macro']:.3f} H@1={cold_m['Hit@1']:.3f} "
              f"H@10={cold_m['Hit@10']:.3f} nC={cold_m['n_datasets']}")

    summary = {"graph_sha256": graph_sha, "split_manifest_sha256": split_hash,
               "split_seed": SPLIT_SEED, "init_seed": INIT_SEED,
               "n_cold": len(cold), "cold_ids": cold, "n_remaining": len(remain),
               "warm_random": warm_random, "cold_random": cold_random,
               "warm": warm_agg, "cold": cold_agg}
    with open(os.path.join(OUT, "cold_sweep_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_tables(summary, versions)
    return summary


def _row(name, agg):
    o = agg
    return "| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
        name, o["tau_macro"], o["NDCG@1"], o["Hit@1"], o["top3_hit@1"], o["Rec@1"],
        o["NDCG@10"], o["Hit@10"], o["Rec@10"])


def write_tables(summary, versions):
    hdr = "| name | tau_macro | NDCG@1 | Hit@1 | top3_hit@1 | Rec@1 | NDCG@10 | Hit@10 | Rec@10 |"
    sep = "|" + "---|" * 9
    L = ["# Cold-Dataset full-version tables\n",
         f"Graph sha256 `{summary['graph_sha256'][:16]}…`, split manifest "
         f"`{summary['split_manifest_sha256'][:16]}…`, split_seed {summary['split_seed']}, "
         f"init seed {summary['init_seed']} (single). Exact-dot ranking over observed "
         "candidates. Mean over datasets (std not shown: single init seed).\n",
         f"cold test datasets = {summary['n_cold']} (IDs {summary['cold_ids']}); "
         f"remaining = {summary['n_remaining']}.\n",
         "## Table A — Remaining datasets / warm held-out edges\n", hdr, sep,
         _row("Random (expected)", summary["warm_random"])]
    for n in versions:
        L.append(_row(n, summary["warm"][n]))
    L += ["\n## Table B — Completely unseen cold dataset IDs\n", hdr, sep,
          _row("Random (expected)", summary["cold_random"])]
    for n in versions:
        L.append(_row(n, summary["cold"][n]))
    with open(os.path.join(OUT, "COLD_TABLES.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\nwrote {os.path.join(OUT, 'COLD_TABLES.md')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="+", default=VERSION_ORDER)
    args = ap.parse_args()
    run(args.versions)


if __name__ == "__main__":
    main()
