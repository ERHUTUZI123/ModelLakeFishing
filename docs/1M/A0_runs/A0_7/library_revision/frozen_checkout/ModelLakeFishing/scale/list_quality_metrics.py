r"""P5 list-quality evaluation for Model Lake and retrained ModelLens.

The existing global harness measures survival of one gold or near-gold model.
This module adds metrics for the complete top-K list, score ordering, an
evidence-based task-compatibility proxy, and ModelLens pointwise calibration.

The task proxy is deliberately conservative and auditable. A recommended
model is compatible when it has at least one non-held-out observation on the
query task. It is off-task when it has training observations, all on other
tasks. Models without training-task evidence remain unknown.

Run from the repository root:

  .\.venv\Scripts\python.exe -m scale.list_quality_metrics \
      --graph stage1BuildTransferGraph/hgraph_ml_v2_sub.pt
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, spearmanr

from scale import global_metrics as GM
from scale.retrain_modellens import build_modellens, EXPORT, LAKE, NODE_SEP, OUT


DEFAULT_OUT = os.path.join(OUT, "list_quality_metrics.json")
DEFAULT_SEED = 20260728


def _finite_or_none(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _safe_corr(fn, x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    result = fn(x, y)
    value = result.statistic if hasattr(result, "statistic") else result[0]
    return _finite_or_none(value)


def labeled_ndcg_at_k(scores: np.ndarray, accuracy: np.ndarray, k: int = 10) -> float:
    """NDCG after restricting the universe to models with observed accuracy.

    Accuracy is min-max normalized within each query before applying a linear
    gain. This prevents the absolute scale of one dataset from dominating the
    macro average. A constant-accuracy query has no ordering error and returns
    1.0.
    """
    scores = np.asarray(scores, dtype=float)
    accuracy = np.asarray(accuracy, dtype=float)
    if scores.size == 0:
        return float("nan")
    if np.ptp(accuracy) <= 1e-12:
        return 1.0
    relevance = (accuracy - accuracy.min()) / np.ptp(accuracy)
    depth = min(int(k), relevance.size)
    discount = 1.0 / np.log2(np.arange(depth, dtype=float) + 2.0)
    ranked = np.argsort(-scores, kind="stable")[:depth]
    ideal = np.argsort(-relevance, kind="stable")[:depth]
    dcg = float(np.sum(relevance[ranked] * discount))
    idcg = float(np.sum(relevance[ideal] * discount))
    return dcg / idcg if idcg > 0 else 1.0


def query_list_metrics(
    scores_all: np.ndarray,
    candidates: np.ndarray,
    accuracy: np.ndarray,
    query_task: int,
    model_tasks: list[set[int]],
    k: int = 10,
    near_delta: float = GM.GAP_DELTA,
) -> dict:
    """Compute full-list and labeled-ranking metrics for one query."""
    scores_all = np.asarray(scores_all, dtype=float)
    candidates = np.asarray(candidates, dtype=int)
    accuracy = np.asarray(accuracy, dtype=float)
    top = np.argsort(-scores_all, kind="stable")[:k]
    observed = {int(m): float(v) for m, v in zip(candidates, accuracy)}
    observed_top = [(int(m), observed[int(m)]) for m in top if int(m) in observed]
    observed_values = [v for _, v in observed_top]
    near_threshold = float(accuracy.max() - near_delta)

    compatible = 0
    off_task = 0
    unknown_task = 0
    for model_id in top:
        evidence = model_tasks[int(model_id)]
        if int(query_task) in evidence:
            compatible += 1
        elif evidence:
            off_task += 1
        else:
            unknown_task += 1

    labeled_scores = scores_all[candidates]
    ranks = GM.query_ranks(scores_all, candidates, accuracy, near_delta)
    return {
        **ranks,
        "verified_count@10": len(observed_top),
        "verified_coverage@10": len(observed_top) / float(k),
        "verified_any@10": float(bool(observed_top)),
        "best_observed_regret@10": (
            float(accuracy.max() - max(observed_values)) if observed_values else None
        ),
        "near_optimal_count@10": sum(v >= near_threshold for v in observed_values),
        "near_optimal_precision@10": (
            sum(v >= near_threshold for v in observed_values) / float(k)
        ),
        "labeled_ndcg@10": labeled_ndcg_at_k(labeled_scores, accuracy, k),
        "spearman": _safe_corr(spearmanr, labeled_scores, accuracy),
        "kendall": _safe_corr(kendalltau, labeled_scores, accuracy),
        "task_compatible_proxy@10": compatible / float(k),
        "off_task_proxy@10": off_task / float(k),
        "task_unknown@10": unknown_task / float(k),
    }


def regression_ece(prediction: np.ndarray, target: np.ndarray, bins: int = 10) -> dict:
    """Equal-width regression ECE over [0, 1]."""
    prediction = np.clip(np.asarray(prediction, dtype=float), 0.0, 1.0)
    target = np.asarray(target, dtype=float)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    ids = np.minimum(np.digitize(prediction, edges[1:-1], right=False), bins - 1)
    total = max(1, prediction.size)
    ece = 0.0
    rows = []
    for b in range(bins):
        mask = ids == b
        count = int(mask.sum())
        if count:
            mean_pred = float(prediction[mask].mean())
            mean_target = float(target[mask].mean())
            gap = abs(mean_pred - mean_target)
            ece += (count / total) * gap
        else:
            mean_pred = mean_target = gap = None
        rows.append({
            "bin": b,
            "lo": float(edges[b]),
            "hi": float(edges[b + 1]),
            "n": count,
            "mean_prediction": mean_pred,
            "mean_target": mean_target,
            "absolute_gap": gap,
        })
    return {"ece": float(ece), "bins": rows}


def aggregate_list_metrics(per_query: dict[int, dict], keys: Iterable[int]) -> dict:
    keys = list(keys)

    def mean(field: str) -> float | None:
        values = [per_query[d].get(field) for d in keys]
        values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
        return float(np.mean(values)) if values else None

    gold = GM.aggregate({d: per_query[d] for d in keys})
    return {
        "n_queries": len(keys),
        "gold@10": gold["gold@10"],
        "top3@10": gold["top3@10"],
        "gold-gap@10": gold["gold-gap@10"],
        "median_gold_rank": gold["median_gold_rank"],
        "verified_count@10": mean("verified_count@10"),
        "verified_coverage@10": mean("verified_coverage@10"),
        "verified_any@10": mean("verified_any@10"),
        "best_observed_regret@10": mean("best_observed_regret@10"),
        "regret_observed_query_fraction": mean("verified_any@10"),
        "near_optimal_count@10": mean("near_optimal_count@10"),
        "near_optimal_precision@10": mean("near_optimal_precision@10"),
        "labeled_ndcg@10": mean("labeled_ndcg@10"),
        "spearman": mean("spearman"),
        "kendall": mean("kendall"),
        "task_compatible_proxy@10": mean("task_compatible_proxy@10"),
        "off_task_proxy@10": mean("off_task_proxy@10"),
        "task_unknown@10": mean("task_unknown@10"),
    }


def _dataset_parts(node: str) -> tuple[str, str]:
    parts = str(node).split(NODE_SEP)
    return parts[0], parts[1] if len(parts) > 1 else ""


def _group_keys(cands: dict, nodes: list[str]) -> dict[str, list[int]]:
    groups = {"all": [], "mteb": [], "retrieval": [], "mteb_retrieval": [], "non_mteb": []}
    tasks = defaultdict(list)
    for d in cands:
        dataset, task = _dataset_parts(nodes[int(d)])
        groups["all"].append(d)
        (groups["mteb"] if dataset.startswith("MTEB ") else groups["non_mteb"]).append(d)
        if task == "Retrieval":
            groups["retrieval"].append(d)
            if dataset.startswith("MTEB "):
                groups["mteb_retrieval"].append(d)
        tasks[task].append(d)
    groups.update({f"task::{task}": keys for task, keys in tasks.items() if len(keys) >= 5})
    return groups


def _quadrants(per_lake: dict, per_ml: dict, nodes: list[str]) -> tuple[dict, dict]:
    groups = {"both_hit": [], "model_lake_only": [], "modellens_only": [], "both_miss": []}
    for d in per_lake:
        lake_hit = per_lake[d]["gold_rank"] <= 10
        ml_hit = per_ml[d]["gold_rank"] <= 10
        key = (
            "both_hit" if lake_hit and ml_hit else
            "model_lake_only" if lake_hit else
            "modellens_only" if ml_hit else
            "both_miss"
        )
        groups[key].append(d)

    rng = np.random.default_rng(DEFAULT_SEED)
    samples = {}
    for key, ids in groups.items():
        chosen = rng.choice(sorted(ids), size=min(3, len(ids)), replace=False).tolist()
        samples[key] = [
            {"dataset_id": int(d), "dataset": _dataset_parts(nodes[int(d)])[0],
             "task": _dataset_parts(nodes[int(d)])[1]}
            for d in chosen
        ]
    return {key: len(ids) for key, ids in groups.items()}, samples


def _selftest() -> int:
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    cand = np.array([0, 1, 4])
    acc = np.array([1.0, 0.99, 0.1])
    tasks = [{1}, {2}, set(), {1}, {3}]
    result = query_list_metrics(scores, cand, acc, 1, tasks, k=3)
    assert result["verified_count@10"] == 2
    assert abs(result["verified_coverage@10"] - 2 / 3) < 1e-12
    assert result["near_optimal_count@10"] == 2
    assert abs(result["task_compatible_proxy@10"] - 1 / 3) < 1e-12
    assert abs(result["off_task_proxy@10"] - 1 / 3) < 1e-12
    assert abs(result["task_unknown@10"] - 1 / 3) < 1e-12
    assert abs(labeled_ndcg_at_k(acc, acc, 3) - 1.0) < 1e-12
    cal = regression_ece(np.array([0.1, 0.9]), np.array([0.0, 1.0]), bins=2)
    assert abs(cal["ece"] - 0.1) < 1e-12
    print("list-quality metric self-test: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return _selftest()
    if not args.graph:
        parser.error("--graph is required unless --selftest is used")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(args.graph, map_location="cpu", weights_only=False)
    data = payload["data"]
    umi = payload["unique_model_id"].sort_values("mappedID").reset_index(drop=True)
    udi = payload["unique_dataset_id"].sort_values("mappedID").reset_index(drop=True)
    names = umi["model"].tolist()
    nodes = udi["dataset"].tolist()
    n_models, n_datasets = len(names), len(nodes)

    model_x = data["model"].x.numpy()
    dataset_x = data["dataset"].x.numpy()
    model_desc = torch.tensor(model_x[:, 64:448], dtype=torch.float32)
    dataset_desc = torch.tensor(dataset_x[:, 64:448], dtype=torch.float32).to(device)
    size_ids = data["model"].size_bucket_id.to(device)
    family_ids = data["model"].family_id.to(device)
    task_ids = data["dataset"].task_type_id

    pool = pd.read_csv(os.path.join(LAKE, "ml_dataset_pool.csv"))
    node_to_metric = dict(zip(pool["dataset_node"], pool["chosen_metric"]))
    metrics = sorted(set(str(node_to_metric.get(node, "unknown")) for node in nodes))
    metric_to_id = {"unknown_metric": 0, **{m: i for i, m in enumerate(metrics, 1)}}
    metric_ids = torch.tensor([
        metric_to_id.get(str(node_to_metric.get(node, "unknown")), 0) for node in nodes
    ], dtype=torch.long)

    model, _ = build_modellens(
        n_models, n_datasets,
        int(payload["xd0_meta"]["num_task_types"]), len(metric_to_id),
        int(payload["xm0_meta"]["num_size_buckets"]),
        int(payload["xm0_meta"]["num_families"]), device,
    )
    del model.model_desc_matrix
    model.register_buffer("model_desc_matrix", model_desc.clone().to(device))
    checkpoint = os.path.join(OUT, "retrained_modellens.pt")
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)
    model.eval()
    unknown_dataset = model.unk_dataset_id
    cache = model.build_model_cache(
        names, size_ids, all_model_family_ids=family_ids, device=device,
    )

    gold_file = np.load(os.path.join(EXPORT, "gold_cands.npz"))
    cands = {
        int(k): (gold_file[k][0].astype(int), gold_file[k][1].astype(float))
        for k in gold_file.files
    }
    held_out = set(cands)

    model_tasks: list[set[int]] = [set() for _ in range(n_models)]
    edge_index = data["model", "trained_on", "dataset"].edge_index.numpy()
    for column in range(edge_index.shape[1]):
        model_id, dataset_id = int(edge_index[0, column]), int(edge_index[1, column])
        if dataset_id not in held_out:
            model_tasks[model_id].add(int(task_ids[dataset_id]))

    z_model = np.load(os.path.join(EXPORT, "z_m_eval.npy")).astype(np.float64)
    z_dataset = np.load(os.path.join(EXPORT, "z_d_eval.npy")).astype(np.float64)
    z_model /= np.linalg.norm(z_model, axis=1, keepdims=True) + 1e-12
    z_dataset /= np.linalg.norm(z_dataset, axis=1, keepdims=True) + 1e-12

    per_lake = {}
    per_ml = {}
    calibration_prediction = []
    calibration_target = []
    calibration_mae_by_query = []
    score_matrix_forward_max_error = 0.0

    with torch.no_grad():
        for index, (dataset_id, (candidate, accuracy)) in enumerate(cands.items(), 1):
            lake_scores = z_model @ z_dataset[int(dataset_id)]

            desc_slot = min(unknown_dataset, model.dataset_desc_matrix.shape[0] - 1)
            model.dataset_desc_matrix[desc_slot] = dataset_desc[int(dataset_id)]
            desc_input = torch.tensor([[float(unknown_dataset)]], device=device)
            ml_scores = model.score_matrix(
                task_ids[int(dataset_id)].view(1).to(device), desc_input, cache,
                metric_ids=metric_ids[int(dataset_id)].view(1).to(device),
            ).squeeze(0).float().cpu().numpy()

            per_lake[int(dataset_id)] = query_list_metrics(
                lake_scores, candidate, accuracy, int(task_ids[int(dataset_id)]),
                model_tasks, args.k,
            )
            per_ml[int(dataset_id)] = query_list_metrics(
                ml_scores, candidate, accuracy, int(task_ids[int(dataset_id)]),
                model_tasks, args.k,
            )

            candidate_t = torch.tensor(candidate, dtype=torch.long, device=device)
            count = candidate_t.numel()
            desc_many = torch.full((count, 1), float(unknown_dataset), device=device)
            forward_scores, point_logits = model(
                task_ids[int(dataset_id)].repeat(count).to(device),
                desc_many,
                candidate_t,
                [names[int(m)] for m in candidate],
                size_ids[candidate_t], family_ids[candidate_t],
                metric_ids=metric_ids[int(dataset_id)].repeat(count).to(device),
            )
            forward_np = forward_scores.float().cpu().numpy()
            score_matrix_forward_max_error = max(
                score_matrix_forward_max_error,
                float(np.max(np.abs(forward_np - ml_scores[candidate]))),
            )
            prediction = torch.sigmoid(point_logits).float().cpu().numpy()
            calibration_prediction.append(prediction)
            calibration_target.append(accuracy)
            calibration_mae_by_query.append(float(np.mean(np.abs(prediction - accuracy))))

            if index % 100 == 0:
                print(f"evaluated {index}/{len(cands)} queries", flush=True)

    groups = _group_keys(cands, nodes)
    grouped = {
        group: {
            "model_lake": aggregate_list_metrics(per_lake, keys),
            "modellens_retrained": aggregate_list_metrics(per_ml, keys),
        }
        for group, keys in groups.items()
    }

    prediction = np.concatenate(calibration_prediction)
    target = np.concatenate(calibration_target)
    calibration = regression_ece(prediction, target, args.ece_bins)
    calibration.update({
        "n_observed_pairs": int(target.size),
        "mae_micro": float(np.mean(np.abs(prediction - target))),
        "mae_macro_query": float(np.mean(calibration_mae_by_query)),
        "rmse_micro": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "mean_prediction": float(prediction.mean()),
        "mean_target": float(target.mean()),
        "score_matrix_forward_max_abs_error": score_matrix_forward_max_error,
        "prediction_definition": "sigmoid(ModelLens pointwise_head logit)",
    })

    quadrant_counts, representative_samples = _quadrants(per_lake, per_ml, nodes)
    current_demo_ids = [1659, 1055, 2601, 256]
    current_demo_audit = []
    for dataset_id in current_demo_ids:
        if dataset_id not in per_lake:
            continue
        lake_hit = per_lake[dataset_id]["gold_rank"] <= 10
        ml_hit = per_ml[dataset_id]["gold_rank"] <= 10
        current_demo_audit.append({
            "dataset_id": dataset_id,
            "dataset": _dataset_parts(nodes[dataset_id])[0],
            "task": _dataset_parts(nodes[dataset_id])[1],
            "model_lake_gold_rank": per_lake[dataset_id]["gold_rank"],
            "modellens_gold_rank": per_ml[dataset_id]["gold_rank"],
            "quadrant": (
                "both_hit" if lake_hit and ml_hit else
                "model_lake_only" if lake_hit else
                "modellens_only" if ml_hit else "both_miss"
            ),
        })

    report = {
        "protocol": {
            "universe": n_models,
            "n_queries": len(cands),
            "k": args.k,
            "near_delta": GM.GAP_DELTA,
            "held_out": True,
            "task_proxy": (
                "compatible = at least one non-held-out observation on the exact query task; "
                "off-task = training observations exist only on other tasks; unknown = no "
                "training-task evidence"
            ),
            "missing_accuracy": "unknown; never treated as observed poor performance",
            "labeled_ndcg": (
                "rank only observed candidates; per-query min-max accuracy relevance; linear gain"
            ),
            "best_observed_regret": (
                "conditional on at least one observed top-k model; coverage reported separately"
            ),
        },
        "groups": grouped,
        "modellens_pointwise_calibration": calibration,
        "gold10_outcome_quadrants": quadrant_counts,
        "representative_sampling": {
            "method": "three fixed-seed random queries per gold@10 outcome quadrant",
            "seed": DEFAULT_SEED,
            "samples": representative_samples,
        },
        "previous_four_demo_audit": current_demo_audit,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print("\n=== LIST QUALITY, ALL 517 HELD-OUT QUERIES ===")
    for system in ("model_lake", "modellens_retrained"):
        row = grouped["all"][system]
        print(
            f"{system:22s} gold@10={row['gold@10']:.4f} "
            f"coverage@10={row['verified_coverage@10']:.4f} "
            f"near_precision@10={row['near_optimal_precision@10']:.4f} "
            f"ndcg@10={row['labeled_ndcg@10']:.4f} "
            f"spearman={row['spearman']:.4f} kendall={row['kendall']:.4f} "
            f"task_proxy={row['task_compatible_proxy@10']:.4f} "
            f"off_task={row['off_task_proxy@10']:.4f}"
        )
    print(f"ModelLens pointwise MAE={calibration['mae_micro']:.4f} ECE={calibration['ece']:.4f}")
    print(f"quadrants={quadrant_counts}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
