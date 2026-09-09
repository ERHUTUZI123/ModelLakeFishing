"""Run the minimal 3M-candidate baseline comparison defined in X6.

Training-free scorers are evaluated one query at a time and discarded, so no
``queries x 3M`` score matrix is materialized. Learned rows reuse the existing
RF embedding evaluator. Partial reports are atomic and survive interruption.
"""
import argparse
import gc
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_PARENT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_PARENT not in sys.path:
    sys.path.insert(0, _REPO_PARENT)

from ModelLakeFishing.scale import global_metrics as GM
from ModelLakeFishing.scale1m import baselines as BL
from ModelLakeFishing.scale1m.eval_rf import query_eligibility
from ModelLakeFishing.scale1m.hf_crawl import data_root, utcnow, write_json_atomic

SEEDS = (0, 1, 2)
N_TOTAL = 3_016_439
EXPECTED_QUERIES = {0: 1476, 1: 1101, 2: 1545}
EXPECTED_OURS = {0: 0.13550135501355012,
                 1: 0.14168937329700273,
                 2: 0.15080906148867315}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _same_model_mapping(left, right):
    if _sha256(left) == _sha256(right):
        return True
    a = pd.read_parquet(left, columns=["mappedID", "model"]).sort_values("mappedID")
    b = pd.read_parquet(right, columns=["mappedID", "model"]).sort_values("mappedID")
    return (np.array_equal(a["mappedID"].to_numpy(), b["mappedID"].to_numpy())
            and np.array_equal(a["model"].astype(str).to_numpy(),
                               b["model"].astype(str).to_numpy()))


def _load_candidates(run_dir):
    with np.load(os.path.join(run_dir, "gold_cands.npz")) as payload:
        return {int(key): (payload[key][0].astype(np.int64),
                           payload[key][1].astype(float))
                for key in payload.files}


def _bundle(exports, run_fmt, seed, dataset_nodes):
    run_dir = os.path.join(exports, run_fmt % seed)
    candidates = _load_candidates(run_dir)
    eligible, _ = query_eligibility(dataset_nodes, run_dir)
    candidates = {q: value for q, value in candidates.items() if eligible[q]}
    if len(candidates) != EXPECTED_QUERIES[seed]:
        raise AssertionError("seed %d has %d eligible queries, expected %d"
                             % (seed, len(candidates), EXPECTED_QUERIES[seed]))
    model_ids_path = os.path.join(run_dir, "model_ids.parquet")
    dataset_ids = pd.read_parquet(os.path.join(run_dir, "dataset_ids.parquet"))
    dataset_ids = dataset_ids.sort_values("mappedID")
    roots = dict(zip(dataset_ids["mappedID"].astype(int), dataset_ids["root"].astype(str)))
    return {"run_dir": run_dir, "candidates": candidates, "roots": roots,
            "model_ids_path": model_ids_path,
            "dataset_names": dataset_ids["dataset"].astype(str).to_numpy()}


def _load_bundles(args, exports, run_fmt):
    bundles = {seed: _bundle(exports, run_fmt, seed, args.dataset_nodes)
               for seed in args.seeds}
    first = bundles[args.seeds[0]]
    first_count = len(pd.read_parquet(first["model_ids_path"], columns=["mappedID"]))
    if first_count != N_TOTAL:
        raise AssertionError("candidate pool has %d rows, expected %d"
                             % (first_count, N_TOTAL))
    for seed, bundle in bundles.items():
        if not _same_model_mapping(bundle["model_ids_path"], first["model_ids_path"]):
            raise AssertionError("model mapping differs at seed %d" % seed)
        if not np.array_equal(bundle["dataset_names"], first["dataset_names"]):
            raise AssertionError("dataset mapping differs at seed %d" % seed)
    return bundles


def _row(per_query, roots, n_tied):
    aggregate = GM.aggregate(per_query, {q: roots[q] for q in per_query})
    aggregate.update({
        "N": N_TOTAL,
        "median_rank_over_N": aggregate["median_gold_rank"] / N_TOTAL,
        "vs_random": aggregate["gold@10"] / (10.0 / N_TOTAL),
        "median_tied_with_gold": float(np.median(n_tied)) if n_tied else 0.0,
    })
    return aggregate


def _score_training_free(name, scorer, ctx, bundles):
    per = {seed: {} for seed in bundles}
    tied = {seed: [] for seed in bundles}
    query_to_seeds = {}
    for seed, bundle in bundles.items():
        for query in bundle["candidates"]:
            query_to_seeds.setdefault(query, []).append(seed)
    start = time.time()
    for index, query in enumerate(sorted(query_to_seeds), 1):
        scores = np.asarray(scorer(ctx, query), dtype=np.float32)
        if scores.shape != (N_TOTAL,):
            raise AssertionError("%s returned shape %r" % (name, scores.shape))
        for seed in query_to_seeds[query]:
            cand, acc = bundles[seed]["candidates"][query]
            per[seed][query] = GM.query_ranks(
                scores, cand, acc, tie_break=ctx.tie_break)
            gold = int(cand[int(np.argmax(acc))])
            tied[seed].append(int((scores == scores[gold]).sum()))
        if index % 250 == 0:
            print("[%s] %d/%d queries %.0fs" %
                  (name, index, len(query_to_seeds), time.time() - start), flush=True)
    return {str(seed): _row(per[seed], bundles[seed]["roots"], tied[seed])
            for seed in bundles}


def _score_static(name, scores, ctx, bundles):
    """Evaluate one query-independent ranking after sorting it exactly once."""
    scores = np.asarray(scores, dtype=np.float32)
    if scores.shape != (N_TOTAL,):
        raise AssertionError("%s returned shape %r" % (name, scores.shape))
    order = np.lexsort((ctx.tie_break, -scores))
    ranks = np.empty(N_TOTAL, dtype=np.int64)
    ranks[order] = np.arange(1, N_TOTAL + 1, dtype=np.int64)
    unique_score, score_count = np.unique(scores, return_counts=True)
    counts = dict(zip(unique_score.tolist(), score_count.tolist()))
    out = {}
    for seed, bundle in bundles.items():
        per, tied = {}, []
        for query, (cand, acc) in bundle["candidates"].items():
            gold = int(cand[int(np.argmax(acc))])
            top3 = cand[np.argsort(-acc)[:min(3, len(acc))]].astype(int)
            near = cand[acc >= acc.max() - GM.GAP_DELTA].astype(int)
            per[query] = {"gold_rank": int(ranks[gold]),
                          "top3_rank": int(ranks[top3].min()),
                          "gap_rank": int(ranks[near].min()),
                          "n_candidates": int(len(cand))}
            tied.append(int(counts[float(scores[gold])]))
        out[str(seed)] = _row(per, bundle["roots"], tied)
    return out


def _embedding_rows(exports, run_fmt, bundles, device):
    rows = {}
    for seed, bundle in bundles.items():
        run_dir = os.path.join(exports, run_fmt % seed)
        model_ids_path = os.path.join(run_dir, "model_ids.parquet")
        if not _same_model_mapping(model_ids_path, bundle["model_ids_path"]):
            raise AssertionError("learned row model mapping differs at seed %d" % seed)
        z_model = np.load(os.path.join(run_dir, "z_m_eval.npy"), mmap_mode="r")
        z_query = np.load(os.path.join(run_dir, "z_d_eval.npy"), mmap_mode="r")
        if z_model.shape[0] != N_TOTAL:
            raise AssertionError("learned row pool is %d, expected %d"
                                 % (z_model.shape[0], N_TOTAL))
        aggregate, _ = GM.from_embeddings_streaming(
            z_model, z_query, bundle["candidates"],
            {q: bundle["roots"][q] for q in bundle["candidates"]}, device=device)
        aggregate.update({"N": N_TOTAL,
                          "median_rank_over_N": aggregate["median_gold_rank"] / N_TOTAL,
                          "vs_random": aggregate["gold@10"] / (10.0 / N_TOTAL)})
        rows[str(seed)] = aggregate
        print("[embedding] seed %d gold@10=%.6f" % (seed, aggregate["gold@10"]),
              flush=True)
    return rows


def _config_gate(reference_runs, nograph_runs, run_fmt, seeds):
    ignored = {"num_layers"}
    records = []
    for seed in seeds:
        ref_path = os.path.join(reference_runs, run_fmt % seed,
                                "metadata", "resolved_config.json")
        ng_path = os.path.join(nograph_runs, "X6NoGraph_full_s%d_e25" % seed,
                               "metadata", "resolved_config.json")
        with open(ref_path, encoding="utf-8") as handle:
            ref = json.load(handle)["resolved_config"]
        with open(ng_path, encoding="utf-8") as handle:
            ng = json.load(handle)["resolved_config"]
        differing = sorted(key for key in set(ref) | set(ng)
                           if ref.get(key) != ng.get(key))
        if differing != ["num_layers"] or ref["num_layers"] != 1 or ng["num_layers"] != 0:
            raise AssertionError("seed %d config differences are %r, expected only num_layers"
                                 % (seed, differing))
        records.append({"seed": seed, "differing_keys": differing,
                        "reference_num_layers": 1, "nograph_num_layers": 0})
    return records


def _summary(rows):
    out = {}
    for name, by_seed in rows.items():
        if name == "R_random":
            continue
        metrics = {}
        ordered_seeds = sorted(map(int, by_seed))
        first_row = by_seed[str(ordered_seeds[0])]
        for key in ("gold@1", "gold@10", "top3@10", "root_gold@10",
                    "median_gold_rank", "median_rank_over_N", "vs_random",
                    "median_tied_with_gold"):
            if key not in first_row:
                continue
            values = [float(by_seed[str(seed)][key]) for seed in ordered_seeds]
            metrics[key] = {"mean": float(np.mean(values)), "per_seed": values,
                            "min": min(values), "max": max(values)}
        out[name] = metrics
    return out


def _write(args, rows, gates, started, stage):
    report = {"written_at": utcnow(), "stage": stage,
              "protocol": {"N": N_TOTAL, "seeds": args.seeds,
                           "expected_queries": EXPECTED_QUERIES,
                           "tie_break": "uint64 affine permutation of mappedID; lower first"},
              "gates": gates, "per_baseline": rows,
              "summary": _summary(rows), "elapsed_s": time.time() - started}
    os.makedirs(args.out, exist_ok=True)
    write_json_atomic(os.path.join(args.out, "X6_BASELINES.json"), report)
    return report


def run_training_free(args, bundles, rows, gates, started):
    first = bundles[args.seeds[0]]["run_dir"]
    ctx = BL.Context(first, args.baseline_attrs, args.ladder, args.graph,
                     args.device)
    export_models = pd.read_parquet(
        bundles[args.seeds[0]]["model_ids_path"], columns=["mappedID", "model"])
    export_models = export_models.sort_values("mappedID")["model"].astype(str).to_numpy()
    if not np.array_equal(ctx.model_name, export_models):
        raise AssertionError("baseline context is not aligned to export model ids")
    gates["pool_and_mapping"] = True
    gates["fixed_unique_tie_break"] = bool(len(np.unique(ctx.tie_break)) == N_TOTAL)
    if not gates["fixed_unique_tie_break"]:
        raise AssertionError("tie-break is not unique")

    pop = BL.popularity(ctx)
    rows["P_popularity"] = _score_static("P_popularity", pop, ctx, bundles)
    _write(args, rows, gates, started, "training-free-partial")

    lexical = BL.BM25(ctx)
    rows["L_bm25"] = _score_training_free("L_bm25", lexical, ctx, bundles)
    _write(args, rows, gates, started, "training-free-partial")
    del lexical
    gc.collect()

    semantic = BL.FrozenMiniLM(ctx)
    rows["S_frozen_minilm"] = _score_training_free(
        "S_frozen_minilm", semantic, ctx, bundles)
    gates["semantic_slice_64_448_only"] = True
    del semantic
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    rows["G_graphsage"] = _embedding_rows(args.exports, args.run_fmt, bundles, args.device)
    for seed in args.seeds:
        got = rows["G_graphsage"][str(seed)]["gold@10"]
        if abs(got - EXPECTED_OURS[seed]) > 1e-12:
            raise AssertionError("seed %d target changed: %.15f != %.15f"
                                 % (seed, got, EXPECTED_OURS[seed]))
    gates["target_reproduced"] = True
    return _write(args, rows, gates, started, "training-free-complete")


def run_learned(args, bundles, rows, gates, started):
    gates["nograph_config_diff"] = _config_gate(
        args.reference_runs, args.nograph_runs, args.run_fmt, args.seeds)
    rows["T_nograph"] = _embedding_rows(
        args.nograph_exports, args.nograph_run_fmt, bundles, args.device)
    if "G_graphsage" not in rows:
        rows["G_graphsage"] = _embedding_rows(args.exports, args.run_fmt, bundles, args.device)
    return _write(args, rows, gates, started, "complete")


def main(argv=None):
    data = os.path.join(data_root(), "data1m")
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("training-free", "learned", "all"),
                        default="training-free")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--exports", default=os.path.join(data, "exports_x4"))
    parser.add_argument("--run-fmt", default="X4GD_full_s%d_e25")
    parser.add_argument("--nograph-exports", default=os.path.join(data, "exports_x6"))
    parser.add_argument("--nograph-run-fmt", default="X6NoGraph_full_s%d_e25")
    parser.add_argument("--reference-runs", default=os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "docs", "1M", "X4_runs"))
    parser.add_argument("--nograph-runs", default=os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "docs", "1M", "X6_runs"))
    parser.add_argument("--dataset-nodes", default=os.path.join(
        data, "rf", "canon", "dataset_nodes_merged.parquet"))
    parser.add_argument("--baseline-attrs", default=os.path.join(
        data, "baseline_rf", "baseline_attrs.parquet"))
    parser.add_argument("--ladder", default=os.path.join(
        data, "ladder_rf", "full_model_ids.parquet"))
    parser.add_argument("--graph", default=os.path.join(
        data, "graphs", "hgraph_rf"))
    parser.add_argument("--out", default=os.path.join(data, "metrics_x6"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    if any(seed not in SEEDS for seed in args.seeds):
        raise ValueError("X6 seeds must be drawn from 0,1,2")
    started = time.time()
    bundles = _load_bundles(args, args.exports, args.run_fmt)
    rows = {"R_random": {str(seed): {"N": N_TOTAL,
                                      "gold@1": 1.0 / N_TOTAL,
                                      "gold@10": 10.0 / N_TOTAL,
                                      "analytic": True,
                                      "n_queries": EXPECTED_QUERIES[seed]}
                         for seed in args.seeds}}
    gates = {"query_counts": True, "candidate_pool": True,
             "cross_seed_mapping": True}

    prior_path = os.path.join(args.out, "X6_BASELINES.json")
    if args.stage == "learned" and os.path.isfile(prior_path):
        with open(prior_path, encoding="utf-8") as handle:
            prior = json.load(handle)
        rows.update(prior.get("per_baseline", {}))
        gates.update(prior.get("gates", {}))

    if args.stage in ("training-free", "all"):
        report = run_training_free(args, bundles, rows, gates, started)
    if args.stage in ("learned", "all"):
        report = run_learned(args, bundles, rows, gates, started)
    print(json.dumps({name: values["gold@10"]["mean"]
                      for name, values in report["summary"].items()},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
