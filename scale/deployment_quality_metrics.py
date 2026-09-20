from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from scale import global_metrics as GM
from scale import modellens_adapter as MA
from scale.head_to_head import EXPORT, LAKE, cache_subset


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(
    ROOT, "docs", "scale", "P5", "artifacts", "deployment_quality_metrics.json"
)
K = 10


def _topk(scores: np.ndarray, k: int = K) -> np.ndarray:
    return np.argsort(-np.asarray(scores, dtype=float), kind="stable")[:k]


def query_deployment_metrics(
    scores: np.ndarray,
    candidates: np.ndarray,
    values: np.ndarray,
    query_task_id: int,
    model_tasks: list[set[int]],
    k: int = K,
    near_delta: float = GM.GAP_DELTA,
) -> tuple[dict, tuple[int, ...]]:
    candidates = np.asarray(candidates, dtype=int)
    values = np.asarray(values, dtype=float)
    top = _topk(scores, k)
    observed = {int(m): float(v) for m, v in zip(candidates, values)}
    observed_top = [observed[int(m)] for m in top if int(m) in observed]
    best = float(values.max())
    near_threshold = best - float(near_delta)
    near_count = sum(v >= near_threshold for v in observed_top)

    exact_task = 0
    known_off_task = 0
    unknown_task = 0
    for model_id in top:
        evidence = model_tasks[int(model_id)]
        if int(query_task_id) in evidence:
            exact_task += 1
        elif evidence:
            known_off_task += 1
        else:
            unknown_task += 1

    if best > 1e-12:
        utility_lb = sum(max(0.0, min(1.0, v / best)) for v in observed_top) / k
    else:
        utility_lb = 0.0

    ranks = GM.query_ranks(np.asarray(scores), candidates, values, near_delta)
    result = {
        **ranks,
        "verified_count@10": len(observed_top),
        "verified_precision@10": len(observed_top) / float(k),
        "all_unverified@10": float(not observed_top),
        "certified_near_optimal_count@10": near_count,
        "certified_near_optimal_precision@10": near_count / float(k),
        "no_certified_near_optimal@10": float(near_count == 0),
        "certified_utility_lower_bound@10": utility_lb,
        "exact_task_evidence@10": exact_task / float(k),
        "known_off_task@10": known_off_task / float(k),
        "unknown_task_evidence@10": unknown_task / float(k),
        "no_exact_task_evidence@10": 1.0 - exact_task / float(k),
        "task_pure_query@10": float(exact_task >= math.ceil(0.8 * k)),
        "majority_known_off_task_query@10": float(known_off_task >= math.ceil(0.5 * k)),
    }
    return result, tuple(int(x) for x in top)


def _mean(per_query: dict[int, dict], field: str) -> float | None:
    values = [row.get(field) for row in per_query.values()]
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(values)) if values else None


def aggregate_deployment_metrics(per_query: dict[int, dict]) -> dict:
    ranks = GM.aggregate(per_query)
    return {
        "n_queries": len(per_query),
        "verified_precision@10": _mean(per_query, "verified_precision@10"),
        "all_unverified_query_rate@10": _mean(per_query, "all_unverified@10"),
        "certified_near_optimal_precision@10": _mean(
            per_query, "certified_near_optimal_precision@10"
        ),
        "no_certified_near_optimal_query_rate@10": _mean(
            per_query, "no_certified_near_optimal@10"
        ),
        "certified_utility_lower_bound@10": _mean(
            per_query, "certified_utility_lower_bound@10"
        ),
        "exact_task_evidence@10": _mean(per_query, "exact_task_evidence@10"),
        "known_off_task@10": _mean(per_query, "known_off_task@10"),
        "no_exact_task_evidence@10": _mean(per_query, "no_exact_task_evidence@10"),
        "task_pure_query_rate@10": _mean(per_query, "task_pure_query@10"),
        "majority_known_off_task_query_rate@10": _mean(
            per_query, "majority_known_off_task_query@10"
        ),
        "gold_miss_rate@10": 1.0 - float(ranks["gold@10"]),
        "top3_miss_rate@10": 1.0 - float(ranks["top3@10"]),
        "median_gold_rank": float(ranks["median_gold_rank"]),
    }


def _jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    a, b = set(left), set(right)
    union = a | b
    return len(a & b) / float(len(union)) if union else 1.0


def recommendation_specificity(
    top_lists: dict[int, tuple[int, ...]],
    signatures: dict[int, tuple[str, str]],
    n_models: int,
) -> dict:
    lists = list(top_lists.values())
    slots = max(1, len(lists) * K)
    unique_models = set(m for row in lists for m in row)
    top1_counts = Counter(row[0] for row in lists if row)

    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for dataset_id, signature in signatures.items():
        if dataset_id in top_lists:
            grouped[signature].append(dataset_id)

    jaccards = []
    identical = 0
    pairs = 0
    for ids in grouped.values():
        if len(ids) < 2:
            continue
        for left_idx in range(len(ids)):
            for right_idx in range(left_idx + 1, len(ids)):
                left = top_lists[ids[left_idx]]
                right = top_lists[ids[right_idx]]
                jaccards.append(_jaccard(left, right))
                identical += int(left == right)
                pairs += 1

    mean_jaccard = float(np.mean(jaccards)) if jaccards else None
    return {
        "unique_recommended_models@10": len(unique_models),
        "catalog_coverage@10": len(unique_models) / float(n_models),
        "unique_models_per_1000_slots": len(unique_models) / slots * 1000.0,
        "unique_top10_list_fraction": len(set(lists)) / float(max(1, len(lists))),
        "top1_max_query_share": (
            max(top1_counts.values()) / float(max(1, len(lists))) if top1_counts else 0.0
        ),
        "same_task_metric_query_pairs": pairs,
        "same_task_metric_mean_jaccard@10": mean_jaccard,
        "same_task_metric_identical_list_pair_rate@10": (
            identical / float(pairs) if pairs else None
        ),
        "dataset_specificity@10": (1.0 - mean_jaccard if mean_jaccard is not None else None),
    }


def build_release_scores(
    names: list[str],
    nodes: list[str],
    query_ids: Iterable[int],
    node_to_task: dict[str, str],
    node_to_metric: dict[str, str],
) -> tuple[dict[int, np.ndarray], dict]:
    model, _, device, missing, unexpected = MA.load_modellens()
    model2id, task2id, metric2id, family2id, profile, size_bucket = MA.build_vocabs()
    family_allowed = {str(k).strip().lower(): int(v) for k, v in family2id.items()}
    unknown_values = {"unknown", "", "none", "null", "nan"}

    global_ids = []
    size_ids = []
    family_ids = []
    for name in names:
        global_ids.append(int(model2id.get(name, model.unk_model_id)))
        size_id = 0
        family_id = 0
        item = profile.get(name)
        if isinstance(item, dict):
            family = str(item.get("family", "unknown")).strip().lower()
            if family not in unknown_values:
                family_id = family_allowed.get(family, 0)
            size = str(item.get("size", "unknown")).strip().lower()
            if size not in unknown_values:
                try:
                    size_id = int(
                        min(np.searchsorted(size_bucket, float(size), side="right"), len(size_bucket))
                    )
                except ValueError:
                    pass
        size_ids.append(size_id)
        family_ids.append(family_id)

    cache = cache_subset(
        model,
        names,
        np.asarray(global_ids),
        torch.tensor(size_ids),
        torch.tensor(family_ids),
        device,
    )
    desc_input = torch.tensor([[float(model.unk_dataset_id)]], device=device)
    scores: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for dataset_id in query_ids:
            node = nodes[int(dataset_id)]
            task_id = int(task2id.get(str(node_to_task.get(node, "")), 0))
            metric_id = int(metric2id.get(str(node_to_metric.get(node, "")), 0))
            value = model.score_matrix(
                torch.tensor([task_id], device=device),
                desc_input,
                cache,
                metric_ids=torch.tensor([metric_id], device=device),
            )
            scores[int(dataset_id)] = value.squeeze(0).float().cpu().numpy()
    return scores, {
        "missing_checkpoint_keys": list(missing),
        "unexpected_checkpoint_keys": list(unexpected),
        "dataset_description_available": False,
        "dataset_id_mapping_available": False,
        "inference_signature": "task + metric + model features; dataset description/id blind",
    }


def _selftest() -> int:
    candidates = np.array([0, 1])
    values = np.array([1.0, 0.8])
    tasks = [{1}, {1}, {2}, {2}, set()]
    good, good_top = query_deployment_metrics(
        np.array([5.0, 4.0, 3.0, 2.0, 1.0]), candidates, values, 1, tasks, k=3
    )
    bad, bad_top = query_deployment_metrics(
        np.array([1.0, 2.0, 5.0, 4.0, 3.0]), candidates, values, 1, tasks, k=3
    )
    assert good["verified_precision@10"] > bad["verified_precision@10"]
    assert bad["all_unverified@10"] == 1.0
    assert bad["majority_known_off_task_query@10"] == 1.0
    spec = recommendation_specificity(
        {0: good_top, 1: good_top, 2: bad_top},
        {0: ("T", "m"), 1: ("T", "m"), 2: ("U", "m")},
        5,
    )
    assert spec["same_task_metric_identical_list_pair_rate@10"] == 1.0
    print("deployment_quality_metrics self-test: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return _selftest()
    if not args.graph:
        parser.error("--graph is required unless --selftest is used")

    payload = torch.load(args.graph, map_location="cpu", weights_only=False)
    data = payload["data"]
    model_ids = payload["unique_model_id"].sort_values("mappedID").reset_index(drop=True)
    dataset_ids = payload["unique_dataset_id"].sort_values("mappedID").reset_index(drop=True)
    names = model_ids["model"].tolist()
    nodes = dataset_ids["dataset"].tolist()
    n_models = len(names)

    export_models = pd.read_csv(os.path.join(EXPORT, "model_ids.csv")).sort_values("mappedID")
    export_datasets = pd.read_csv(os.path.join(EXPORT, "dataset_ids.csv")).sort_values("mappedID")
    assert export_models["model"].tolist() == names
    assert export_datasets["dataset"].tolist() == nodes

    gold_file = np.load(os.path.join(EXPORT, "gold_cands.npz"))
    candidates = {
        int(key): (gold_file[key][0].astype(int), gold_file[key][1].astype(float))
        for key in gold_file.files
    }
    held_out = set(candidates)

    pool = pd.read_csv(os.path.join(LAKE, "ml_dataset_pool.csv"))
    node_to_task = dict(zip(pool["dataset_node"], pool["task"]))
    node_to_metric = dict(zip(pool["dataset_node"], pool["chosen_metric"]))
    task_ids = data["dataset"].task_type_id.numpy()

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
    lake_scores = {dataset_id: z_model @ z_dataset[dataset_id] for dataset_id in candidates}

    release_scores, release_contract = build_release_scores(
        names, nodes, candidates, node_to_task, node_to_metric
    )

    per_system = {}
    top_system = {}
    for system, score_map in (
        ("model_lake", lake_scores),
        ("modellens_released", release_scores),
    ):
        per_query = {}
        top_lists = {}
        for dataset_id, (candidate, value) in candidates.items():
            row, top = query_deployment_metrics(
                score_map[dataset_id],
                candidate,
                value,
                int(task_ids[dataset_id]),
                model_tasks,
            )
            per_query[dataset_id] = row
            top_lists[dataset_id] = top
        per_system[system] = per_query
        top_system[system] = top_lists

    signatures = {
        dataset_id: (
            str(node_to_task.get(nodes[dataset_id], "")),
            str(node_to_metric.get(nodes[dataset_id], "")),
        )
        for dataset_id in candidates
    }
    systems = {
        system: {
            "top10_deployment_quality": aggregate_deployment_metrics(per_system[system]),
            "query_responsiveness": recommendation_specificity(
                top_system[system], signatures, n_models
            ),
        }
        for system in per_system
    }

    result = {
        "protocol": {
            "candidate_universe": n_models,
            "held_out_queries": len(candidates),
            "k": K,
            "near_optimal_delta": GM.GAP_DELTA,
            "evaluation_target": (
                "actual global top-10 from Model Lake and the scorer reproducible from "
                "the published ModelLens checkpoint"
            ),
            "missing_result_policy": (
                "unknown for semantic claims; zero only inside the explicitly named "
                "certified_utility_lower_bound@10"
            ),
            "excluded_as_primary_metrics": [
                "labeled-only NDCG",
                "labeled-only Spearman/Kendall",
                "conditional regret",
                "retrained-surrogate pointwise calibration",
            ],
        },
        "modellens_release_contract": release_contract,
        "systems": systems,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    print("\n=== DEPLOYMENT-CENTRED TOP-10 QUALITY ===")
    fields = (
        "verified_precision@10",
        "all_unverified_query_rate@10",
        "certified_near_optimal_precision@10",
        "no_certified_near_optimal_query_rate@10",
        "certified_utility_lower_bound@10",
        "known_off_task@10",
        "gold_miss_rate@10",
        "median_gold_rank",
    )
    for field in fields:
        left = systems["model_lake"]["top10_deployment_quality"][field]
        right = systems["modellens_released"]["top10_deployment_quality"][field]
        print(f"{field:48s} lake={left:.6f}  released_ml={right:.6f}")
    print("\n=== QUERY RESPONSIVENESS ===")
    for field in (
        "catalog_coverage@10",
        "unique_top10_list_fraction",
        "top1_max_query_share",
        "same_task_metric_mean_jaccard@10",
        "same_task_metric_identical_list_pair_rate@10",
        "dataset_specificity@10",
    ):
        left = systems["model_lake"]["query_responsiveness"][field]
        right = systems["modellens_released"]["query_responsiveness"][field]
        print(f"{field:48s} lake={left:.6f}  released_ml={right:.6f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
