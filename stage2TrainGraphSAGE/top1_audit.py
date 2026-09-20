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

from ModelLakeFishing.stage2TrainGraphSAGE.model import load_hgraph
from ModelLakeFishing.stage2TrainGraphSAGE.losses import (
    TRAINED_ON, accuracy_lookup, perf_supervision,
)
from ModelLakeFishing.stage2TrainGraphSAGE.eval_harness import make_fixed_splits, paired_bootstrap
from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import apply_similar_to_mode
from ModelLakeFishing.stage2TrainGraphSAGE.ablation import train_eval_one

GRAPH = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage1BuildTransferGraph",
                     "hgraph_hf1000d_2000m_xm0_xd0.pt")
OUT = os.path.join(_HERE, "artifacts", "ablation", "top1")
SPLIT_SEEDS = (0, 1, 2)
INIT_SEED = 0
EPOCHS = 25
STRATA = [(3, 10), (11, 20), (21, 50), (51, 100), (101, 10 ** 9)]
KS = (1, 10, 50, 100)


def _base_cfg():
    return dict(num_layers=1, top_frac=0.10, lr=1e-2,
                lambda_rank=1.0, lambda_contrast=1.0, lambda_mse=0.0, lambda_uniform=0.0,
                scorer="dot", similar_to_mode="dense", similar_to_k=10,
                edge_aware=False, weighted_relations=None,
                rank_loss="hinge", rank_temperature=0.1, rank_min_gap=0.0, rank_gap_weighted=False,
                separate_heads=False, grouped=False,
                lambda_dm_contrast=0.0, dm_temperature=0.1, dm_hard_neg_weight=1.0, dm_warmup=0,
                early_stop=False, patience=0)


def configs():
    b0 = _base_cfg()
    b5 = _base_cfg(); b5.update(similar_to_mode="topk_unweighted", edge_aware=True,
                                weighted_relations=[], rank_loss="ranknet", rank_min_gap=0.01)
    rmg02 = dict(b5); rmg02["rank_min_gap"] = 0.02
    p6 = dict(b5); p6["lambda_dm_contrast"] = 1.0
    return {"B0": b0, "B5_ranknet": b5, "R_mg02": rmg02, "P6_dm10": p6}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def state_dict_sha256(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def model_names(umi):
    if hasattr(umi, "sort_values"):
        return umi.sort_values("mappedID")["model"].astype(str).tolist()
    if isinstance(umi, dict):
        return [str(umi[i]) for i in range(len(umi))]
    return [str(x) for x in list(umi)]


def candidates(test_data, lookup):
    eli, target = perf_supervision(test_data[TRAINED_ON], lookup)
    models, ds, acc = eli[0].numpy(), eli[1].numpy(), target.numpy()
    out = {}
    for d in np.unique(ds):
        sel = ds == d
        cand, a = models[sel], acc[sel]
        if cand.size >= 3 and np.std(a) > 0:
            out[int(d)] = (cand.astype(int), a.astype(float))
    return out


def observed_top1(cands, z_m, z_d, names):
    zm = F.normalize(z_m, dim=-1)
    zd = F.normalize(z_d, dim=-1)
    per = {}
    for d, (cand, a) in cands.items():
        ct = torch.as_tensor(cand, dtype=torch.long)
        score = (zm[ct] @ zd[int(d)]).numpy()
        sel_local = int(np.argmax(score))
        best_local = int(np.argmax(a))
        top3_local = set(np.argsort(-a)[:3].tolist())
        max_acc = float(a.max())
        per[int(d)] = {
            "n_candidates": int(cand.size),
            "hit1": float(a[sel_local] == max_acc),
            "top3_hit1": float(sel_local in top3_local),
            "regret1": float(max_acc - a[sel_local]),
            "selected_model_id": int(cand[sel_local]),
            "selected_model_name": names[int(cand[sel_local])],
            "selected_acc": float(a[sel_local]),
            "true_best_model_id": int(cand[best_local]),
            "true_best_model_name": names[int(cand[best_local])],
            "true_best_acc": max_acc,
        }
    return per


def strata_agg(per):
    out = {}
    for lo, hi in STRATA:
        ds = [r for r in per.values() if lo <= r["n_candidates"] <= hi]
        key = f"{lo}-{hi if hi < 10**9 else 'inf'}"
        if ds:
            out[key] = {"n_datasets": len(ds),
                        "hit1": float(np.mean([r["hit1"] for r in ds])),
                        "top3_hit1": float(np.mean([r["top3_hit1"] for r in ds])),
                        "regret1": float(np.mean([r["regret1"] for r in ds]))}
        else:
            out[key] = {"n_datasets": 0}
    return out


def full2k_gold(cands, z_m, z_d):
    zm = F.normalize(z_m, dim=-1)
    zd = F.normalize(z_d, dim=-1)
    per = {}
    for d, (cand, a) in cands.items():
        gold = int(cand[int(np.argmax(a))])
        scores = (zm @ zd[int(d)]).numpy()
        order = np.argsort(-scores)
        rank = int(np.where(order == gold)[0][0]) + 1
        per[int(d)] = {
            "gold_model_id": gold,
            "gold_rank": rank,
            "reciprocal_rank": 1.0 / rank,
            **{f"gold_survival@{K}": float(rank <= K) for K in KS},
        }
    return per


def hnsw_fidelity(cands, z_m, z_d):
    import faiss
    zm = F.normalize(z_m, dim=-1).numpy().astype("float32")
    zd = F.normalize(z_d, dim=-1).numpy().astype("float32")
    dim = zm.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200
    index.add(zm)
    index.hnsw.efSearch = 256
    q_ds = sorted(cands.keys())
    q = zd[np.asarray(q_ds)]
    exact = q @ zm.T
    out = {}
    for K in KS:
        _, ann = index.search(q, K)
        recs = []
        for i in range(len(q_ds)):
            ex = set(np.argsort(-exact[i])[:K].tolist())
            recs.append(len(set(ann[i].tolist()) & ex) / K)
        out[f"recall@{K}"] = float(np.mean(recs))
    return out


def macro(per, key):
    vals = [r[key] for r in per.values()]
    return float(np.mean(vals)) if vals else float("nan")


def run():
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graph_sha = sha256_file(GRAPH)
    xd0_full = torch.load(GRAPH, map_location="cpu", weights_only=False).get("xd0_meta")

    cfgs = configs()
    cand_sig = {ss: {} for ss in SPLIT_SEEDS}
    results = {name: {"config_name": name, "config": cfg, "init_seed": INIT_SEED,
                      "graph": os.path.basename(GRAPH), "graph_sha256": graph_sha,
                      "ann_backend": "faiss-cpu IndexHNSWFlat(M=32,efC=200,efS=256); "
                                     "L2 on unit vectors == cosine top-k",
                      "note_retrained": "no valid per-split checkpoint set existed; "
                                        "all configs retrained exactly via ablation.train_eval_one",
                      "splits": {}}
               for name, cfg in cfgs.items()}
    pooled = {name: {} for name in cfgs}

    for name, cfg in cfgs.items():
        for ss in SPLIT_SEEDS:
            data, xm0, umi = load_hgraph(GRAPH)
            names = model_names(umi)
            data = apply_similar_to_mode(data, cfg["similar_to_mode"], k=cfg["similar_to_k"])
            split = make_fixed_splits(data, split_seed=ss)
            _tr, _val, test_data = split
            lookup = accuracy_lookup(data)

            _row, _pt, _ph, model, _scorer = train_eval_one(
                data, xm0, xd0_full, cfg, split, init_seed=INIT_SEED, epochs=EPOCHS, device=device)
            model.eval()
            with torch.no_grad():
                z = model(test_data.clone().to(device))
            z_m, z_d = z["model"].cpu(), z["dataset"].cpu()

            cands = candidates(test_data, lookup)
            cand_sig[ss][name] = {d: sorted(c.tolist()) for d, (c, _a) in cands.items()}

            obs = observed_top1(cands, z_m, z_d, names)
            gold = full2k_gold(cands, z_m, z_d)
            fid = hnsw_fidelity(cands, z_m, z_d)
            hit1_random = float(np.mean([1.0 / r["n_candidates"] for r in obs.values()]))

            ranks = [r["gold_rank"] for r in gold.values()]
            per_dataset = {str(d): {**obs[d], **gold[d]} for d in obs}
            results[name]["splits"][str(ss)] = {
                "split_seed": ss,
                "state_dict_sha256": state_dict_sha256(model),
                "n_datasets": len(obs),
                "observed_aggregate": {
                    "hit1": macro(obs, "hit1"), "top3_hit1": macro(obs, "top3_hit1"),
                    "regret1": macro(obs, "regret1"), "hit1_random": hit1_random},
                "observed_strata": strata_agg(obs),
                "full2k_aggregate": {
                    **{f"gold_survival@{K}": macro(gold, f"gold_survival@{K}") for K in KS},
                    "median_gold_rank": float(np.median(ranks)),
                    "mean_gold_rank": float(np.mean(ranks)),
                    "MRR": macro(gold, "reciprocal_rank")},
                "hnsw_fidelity": fid,
                "per_dataset": per_dataset,
            }
            for d, r in obs.items():
                pooled[name][f"s{ss}::{d}"] = {"hit1": r["hit1"], "regret1": r["regret1"]}
            print(f"[{name} s{ss}] datasets={len(obs)} obs_hit1={macro(obs,'hit1'):.3f} "
                  f"top3={macro(obs,'top3_hit1'):.3f} gold@1={macro(gold,'gold_survival@1'):.3f} "
                  f"gold@10={macro(gold,'gold_survival@10'):.3f} med_rank={np.median(ranks):.0f} "
                  f"hnsw_r@1={fid['recall@1']:.3f}")

    identity = {}
    ref_name = "B0"
    for ss in SPLIT_SEEDS:
        ref = cand_sig[ss][ref_name]
        ok = all(cand_sig[ss][n] == ref for n in cfgs)
        identity[str(ss)] = {"identical": bool(ok), "n_datasets": len(ref)}
    all_identical = all(v["identical"] for v in identity.values())
    for name in cfgs:
        results[name]["candidate_set_identity_across_configs"] = identity

    for name in cfgs:
        S = results[name]["splits"]
        def agg(path):
            vals = []
            for ss in S:
                cur = S[ss]
                for p in path.split("."):
                    cur = cur[p]
                vals.append(cur)
            return [float(np.mean(vals)), float(np.std(vals))]
        results[name]["aggregate_over_splits"] = {
            "observed_hit1": agg("observed_aggregate.hit1"),
            "top3_hit1": agg("observed_aggregate.top3_hit1"),
            "regret1": agg("observed_aggregate.regret1"),
            "hit1_random": agg("observed_aggregate.hit1_random"),
            **{f"gold_survival@{K}": agg(f"full2k_aggregate.gold_survival@{K}") for K in KS},
            "median_gold_rank": agg("full2k_aggregate.median_gold_rank"),
            "mean_gold_rank": agg("full2k_aggregate.mean_gold_rank"),
            "MRR": agg("full2k_aggregate.MRR"),
            "hnsw_recall@1": agg("hnsw_fidelity.recall@1"),
            "hnsw_recall@10": agg("hnsw_fidelity.recall@10"),
        }
        with open(os.path.join(OUT, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(results[name], f, indent=2)

    boots = {}
    for base in ("B0", "B5_ranknet", "R_mg02"):
        boots[f"P6_dm10_vs_{base}__hit1"] = paired_bootstrap(pooled[base], pooled["P6_dm10"], metric="hit1")
        boots[f"P6_dm10_vs_{base}__regret1"] = paired_bootstrap(pooled[base], pooled["P6_dm10"], metric="regret1")
    with open(os.path.join(OUT, "paired_bootstrap.json"), "w", encoding="utf-8") as f:
        json.dump(boots, f, indent=2)

    write_markdown(results, identity, all_identical, boots)
    print(f"\ncandidate sets identical across configs (all splits): {all_identical}")
    print(f"saved: {OUT}")
    return results, identity, boots


def write_markdown(results, identity, all_identical, boots):
    def a(name, k):
        return results[name]["aggregate_over_splits"][k]
    lines = []
    lines.append("# TOP1_RESULTS — evaluation-only Top-1 audit\n")
    lines.append(f"Graph `{results['B0']['graph']}` sha256 `{results['B0']['graph_sha256'][:16]}…`. "
                 f"Split seeds {list(SPLIT_SEEDS)}, init seed {INIT_SEED}, {EPOCHS} epochs. "
                 "Embeddings from test_data. ANN = faiss IndexHNSWFlat over the 2000 z_m.\n")
    ident_str = "; ".join("split {}: {} datasets".format(s, identity[s]["n_datasets"]) for s in identity)
    lines.append(f"**Candidate sets identical across configs (every split): {all_identical}** "
                 f"({ident_str}).\n")
    lines.append("All values mean over splits (±std where shown). full2k_gold@K = fraction of "
                 "datasets whose gold (highest-acc held-out) model ranks ≤K among ALL 2000 z_m by "
                 "exact z_d·z_m. NOT full-lake precision (non-gold models unlabeled). "
                 "exact-vs-HNSW recall = ANN fidelity, not a semantic metric.\n")
    hdr = ("| name | observed_hit@1 | top3_hit@1 | regret@1 | full2k_gold@1 | full2k_gold@10 "
           "| full2k_gold@50 | median_gold_rank | exact-vs-HNSW recall@1 |")
    sep = "|" + "---|" * 9
    lines.append(hdr); lines.append(sep)
    for name in ("B0", "B5_ranknet", "R_mg02", "P6_dm10"):
        lines.append("| {} | {:.3f}±{:.3f} | {:.3f} | {:.4f} | {:.3f} | {:.3f} | {:.3f} | {:.0f} | {:.3f} |".format(
            name, a(name, "observed_hit1")[0], a(name, "observed_hit1")[1], a(name, "top3_hit1")[0],
            a(name, "regret1")[0], a(name, "gold_survival@1")[0], a(name, "gold_survival@10")[0],
            a(name, "gold_survival@50")[0], a(name, "median_gold_rank")[0], a(name, "hnsw_recall@1")[0]))
    lines.append("\n## Random baseline (observed Top-1)\n")
    for name in ("B0", "B5_ranknet", "R_mg02", "P6_dm10"):
        lines.append(f"- {name}: hit@1 {a(name,'observed_hit1')[0]:.3f} vs random "
                     f"{a(name,'hit1_random')[0]:.3f}")
    lines.append("\n## Gold rank & MRR (full-2K, mean over splits)\n")
    lines.append("| name | gold@100 | mean_gold_rank | MRR | hnsw recall@10 |")
    lines.append("|---|---|---|---|---|")
    for name in ("B0", "B5_ranknet", "R_mg02", "P6_dm10"):
        lines.append(f"| {name} | {a(name,'gold_survival@100')[0]:.3f} | {a(name,'mean_gold_rank')[0]:.1f} "
                     f"| {a(name,'MRR')[0]:.3f} | {a(name,'hnsw_recall@10')[0]:.3f} |")
    lines.append("\n## Paired bootstrap — P6_dm10 vs others (observed hit@1, same datasets)\n")
    lines.append("| comparison | Δhit@1 | 95% CI | P(>0) | n |")
    lines.append("|---|---|---|---|---|")
    for base in ("B0", "B5_ranknet", "R_mg02"):
        r = boots[f"P6_dm10_vs_{base}__hit1"]
        lines.append(f"| P6_dm10 − {base} | {r['mean_delta']:+.3f} | "
                     f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {r['prob_positive']:.2f} | {r['n_paired']} |")
    lines.append("\nPer-config JSON (with every per-dataset record + strata + hashes): "
                 "`artifacts/ablation/top1/<name>.json`; bootstrap: `paired_bootstrap.json`.")
    with open(os.path.join(OUT, "TOP1_RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run()
