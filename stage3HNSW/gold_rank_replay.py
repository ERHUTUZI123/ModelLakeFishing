"""Full-graph retrieval-quality replay for a Stage-3 embedding export.

This is a diagnostic, not the clean fixed-split evaluator.  It ranks the
highest-accuracy labelled model for every eligible dataset against the entire
exported model lake, reports gold survival and hub occupancy for raw cosine and
CSLS, and writes the same result as JSON and Markdown.

Run from the repository root::

    python -m ModelLakeFishing.stage3HNSW.gold_rank_replay \
        --export ModelLakeFishing/stage3HNSW/artifacts/exports/hf1000d_G2
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage3HNSW.export_embeddings import sha256_file  # noqa: E402


REPORT_SCHEMA_VERSION = "gold-rank-replay.v1"
GOLD_KS = (10, 50, 100, 200, 500)
HUB_TOP_K = 10
CSLS_K = 10
MIN_LABELLED_MODELS = 3
FULL_GRAPH_CAVEAT = (
    "Full-graph diagnostic only: this replay uses the serving export, whose "
    "message graph contains all trained_on edges (after the checkpoint-recorded "
    "surgery), including edges that are test edges under the clean fixed-split "
    "protocol. Absolute replay metrics are therefore optimistic. Use them only "
    "for same-protocol comparisons and candidate-K sizing; checkpoint promotion "
    "must continue to use top1_eval.py and the fixed-split five-metric protocol."
)


def _score_matrix(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores)
    if scores.ndim != 2 or not scores.shape[0] or not scores.shape[1]:
        raise ValueError(f"scores must be a non-empty [datasets, models] matrix; got {scores.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or infinity")
    return scores


def csls_scores(cosine: np.ndarray, *, k: int = CSLS_K) -> np.ndarray:
    """Return standard cross-domain similarity local scaling scores.

    For cosine matrix ``S[d, m]`` this computes
    ``2*S[d,m] - mean_topk_models(S[d,:]) - mean_topk_datasets(S[:,m])``.
    Each side clips k to the population available on that side, which keeps the
    function well-defined for small synthetic tests and small exports.
    """
    cosine = _score_matrix(cosine)
    if k <= 0:
        raise ValueError(f"CSLS k must be positive; got {k}")
    n_datasets, n_models = cosine.shape
    k_models = min(int(k), n_models)
    k_datasets = min(int(k), n_datasets)
    dataset_radius = np.partition(
        cosine, n_models - k_models, axis=1
    )[:, -k_models:].mean(axis=1)
    model_radius = np.partition(
        cosine, n_datasets - k_datasets, axis=0
    )[-k_datasets:, :].mean(axis=0)
    return 2.0 * cosine - dataset_radius[:, None] - model_radius[None, :]


def build_eligible_labels(
    edge_model_ids: Sequence[int],
    edge_dataset_ids: Sequence[int],
    edge_accuracies: Sequence[float],
    *,
    n_models: int,
    n_datasets: int,
    min_labelled_models: int = MIN_LABELLED_MODELS,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, int | str]]:
    """Deduplicate labels with max reduction and retain scorable datasets.

    This mirrors ``dedup_trained_on(reduce='max')`` without cloning the full
    PyG graph. Eligibility is the clean evaluator's dataset rule: at least
    three distinct labelled models and non-constant accuracy.
    """
    model_ids = np.asarray(edge_model_ids, dtype=np.int64).reshape(-1)
    dataset_ids = np.asarray(edge_dataset_ids, dtype=np.int64).reshape(-1)
    accuracies = np.asarray(edge_accuracies, dtype=np.float64).reshape(-1)
    if not (model_ids.size == dataset_ids.size == accuracies.size):
        raise ValueError("trained_on model, dataset, and accuracy arrays must have equal length")
    if min_labelled_models < 1:
        raise ValueError("min_labelled_models must be positive")
    if model_ids.size and (
        model_ids.min() < 0 or model_ids.max() >= n_models
        or dataset_ids.min() < 0 or dataset_ids.max() >= n_datasets
    ):
        raise ValueError("trained_on edge contains an out-of-range mappedID")
    if not np.isfinite(accuracies).all():
        raise ValueError("trained_on accuracy contains NaN or infinity")

    # One value per (model, dataset), keeping the best observed run.  Sorting
    # model IDs below also makes np.argmax's tie choice explicit and stable.
    by_dataset: dict[int, dict[int, float]] = defaultdict(dict)
    for model_id, dataset_id, accuracy in zip(model_ids, dataset_ids, accuracies):
        old = by_dataset[int(dataset_id)].get(int(model_id), float("-inf"))
        by_dataset[int(dataset_id)][int(model_id)] = max(old, float(accuracy))

    eligible: dict[int, dict[str, np.ndarray]] = {}
    too_small = 0
    constant = 0
    for dataset_id in sorted(by_dataset):
        pairs = by_dataset[dataset_id]
        mids = np.asarray(sorted(pairs), dtype=np.int64)
        acc = np.asarray([pairs[int(m)] for m in mids], dtype=np.float64)
        if mids.size < min_labelled_models:
            too_small += 1
            continue
        if float(np.ptp(acc)) == 0.0:
            constant += 1
            continue
        eligible[dataset_id] = {"model_ids": mids, "accuracies": acc}

    stats: dict[str, int | str] = {
        "raw_trained_on_rows": int(model_ids.size),
        "distinct_model_dataset_pairs": int(sum(len(v) for v in by_dataset.values())),
        "datasets_with_labels": int(len(by_dataset)),
        "eligible_datasets": int(len(eligible)),
        "excluded_too_few_labelled_models": int(too_small),
        "excluded_constant_accuracy": int(constant),
        "datasets_without_labels": int(n_datasets - len(by_dataset)),
        "duplicate_reduction": "max accuracy per (model, dataset)",
    }
    return eligible, stats


def _occupancy_bin(value: int) -> str:
    if value == 0:
        return "0"
    if value == 1:
        return "1"
    for lo, hi in ((2, 5), (6, 10), (11, 25), (26, 50), (51, 100), (101, 200)):
        if lo <= value <= hi:
            return f"{lo}-{hi}"
    return "201+"


_OCCUPANCY_BINS = ("0", "1", "2-5", "6-10", "11-25", "26-50", "51-100", "101-200", "201+")


def hub_occupancy(
    scores: np.ndarray,
    model_names: Sequence[str],
    *,
    top_k: int = HUB_TOP_K,
) -> dict:
    """Top-k occupancy over *all* exported dataset queries."""
    scores = _score_matrix(scores)
    n_datasets, n_models = scores.shape
    if len(model_names) != n_models:
        raise ValueError("model_names length does not match score matrix")
    if top_k <= 0:
        raise ValueError("hub top_k must be positive")
    effective_k = min(int(top_k), n_models)
    top = np.argsort(-scores, axis=1, kind="stable")[:, :effective_k]
    occupancy = np.bincount(top.reshape(-1), minlength=n_models)

    values, counts = np.unique(occupancy, return_counts=True)
    exact_histogram = [
        {"occupancy": int(value), "n_models": int(count)}
        for value, count in zip(values, counts)
    ]
    binned = {label: 0 for label in _OCCUPANCY_BINS}
    for value in occupancy:
        binned[_occupancy_bin(int(value))] += 1

    occupied = np.flatnonzero(occupancy)
    order = sorted(occupied.tolist(), key=lambda m: (-int(occupancy[m]), int(m)))
    models = [
        {
            "mappedID": int(model_id),
            "unique_model_id": str(model_names[model_id]),
            "occupancy": int(occupancy[model_id]),
            "query_fraction": float(occupancy[model_id] / n_datasets),
        }
        for model_id in order
    ]
    max_occupancy = int(occupancy.max(initial=0))
    return {
        "query_population": "all exported dataset rows",
        "n_queries": int(n_datasets),
        "requested_top_k": int(top_k),
        "effective_top_k": int(effective_k),
        "n_slots": int(n_datasets * effective_k),
        "distinct_models": int(occupied.size),
        "max_occupancy": max_occupancy,
        "max_occupancy_fraction": float(max_occupancy / n_datasets),
        "exact_histogram": exact_histogram,
        "binned_histogram": [
            {"occupancy_bin": label, "n_models": int(binned[label])}
            for label in _OCCUPANCY_BINS
        ],
        "occupied_models": models,
    }


def evaluate_score_matrix(
    scores: np.ndarray,
    labels: Mapping[int, Mapping[str, np.ndarray]],
    model_names: Sequence[str],
    dataset_names: Sequence[str],
    *,
    gold_ks: Iterable[int] = GOLD_KS,
    hub_top_k: int = HUB_TOP_K,
) -> dict:
    """Evaluate one ranking score matrix under the replay protocol."""
    scores = _score_matrix(scores)
    n_datasets, n_models = scores.shape
    if len(model_names) != n_models or len(dataset_names) != n_datasets:
        raise ValueError("ID snapshot lengths do not match score matrix")
    ks = tuple(int(k) for k in gold_ks)
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("gold_ks must contain positive integers")
    if not labels:
        raise ValueError("no eligible labelled datasets")

    per_dataset: dict[int, dict] = {}
    canonical_ranks: list[int] = []
    observed_hits: list[float] = []
    for dataset_id in sorted(labels):
        if dataset_id < 0 or dataset_id >= n_datasets:
            raise ValueError(f"eligible dataset mappedID {dataset_id} is out of range")
        mids = np.asarray(labels[dataset_id]["model_ids"], dtype=np.int64)
        acc = np.asarray(labels[dataset_id]["accuracies"], dtype=np.float64)
        if mids.size != acc.size or not mids.size:
            raise ValueError(f"dataset {dataset_id} has malformed labels")

        max_acc = float(acc.max())
        # mids is sorted by build_eligible_labels.  The first tied argmax is the
        # canonical gold used by top1_eval.py; every tied gold is still emitted.
        gold_ids = mids[acc == max_acc]
        canonical_gold = int(gold_ids[0])
        row = scores[dataset_id]

        gold_models = []
        for model_id in gold_ids:
            model_id = int(model_id)
            rank = int(np.count_nonzero(row > row[model_id]) + 1)
            gold_models.append({
                "mappedID": model_id,
                "unique_model_id": str(model_names[model_id]),
                "rank": rank,
                "score": float(row[model_id]),
            })
        canonical_rank = int(gold_models[0]["rank"])

        selected_pos = int(np.argmax(row[mids]))
        selected_model = int(mids[selected_pos])
        selected_acc = float(acc[selected_pos])
        observed_hit = float(selected_acc == max_acc)
        canonical_ranks.append(canonical_rank)
        observed_hits.append(observed_hit)
        per_dataset[int(dataset_id)] = {
            "dataset_mappedID": int(dataset_id),
            "unique_dataset_id": str(dataset_names[dataset_id]),
            "n_labelled_models": int(mids.size),
            "gold_accuracy": max_acc,
            "canonical_gold_model": gold_models[0],
            "all_argmax_gold_models": gold_models,
            "gold_rank": canonical_rank,
            "best_tied_gold_rank": int(min(g["rank"] for g in gold_models)),
            "observed_selected_model": {
                "mappedID": selected_model,
                "unique_model_id": str(model_names[selected_model]),
                "accuracy": selected_acc,
                "score": float(row[selected_model]),
            },
            "observed_hit_at_1": observed_hit,
        }

    rank_array = np.asarray(canonical_ranks)
    aggregate = {
        "n_eligible_datasets": int(rank_array.size),
        "gold_at_k": {
            str(k): float(np.mean(rank_array <= k)) for k in ks
        },
        "median_gold_rank": float(np.median(rank_array)),
        "mean_gold_rank": float(np.mean(rank_array)),
        "observed_hit_at_1": float(np.mean(observed_hits)),
    }
    return {
        "aggregate": aggregate,
        "hub_occupancy": hub_occupancy(scores, model_names, top_k=hub_top_k),
        "per_dataset": per_dataset,
    }


def replay_arrays(
    z_m: np.ndarray,
    z_d: np.ndarray,
    labels: Mapping[int, Mapping[str, np.ndarray]],
    model_names: Sequence[str],
    dataset_names: Sequence[str],
    *,
    gold_ks: Iterable[int] = GOLD_KS,
    csls_k: int = CSLS_K,
    hub_top_k: int = HUB_TOP_K,
) -> tuple[dict[str, dict], list[dict]]:
    """Run raw-cosine and CSLS variants and merge their dataset rows."""
    z_m = np.asarray(z_m)
    z_d = np.asarray(z_d)
    if z_m.ndim != 2 or z_d.ndim != 2 or z_m.shape[1] != z_d.shape[1]:
        raise ValueError(f"incompatible embedding shapes z_m={z_m.shape}, z_d={z_d.shape}")
    if not np.isfinite(z_m).all() or not np.isfinite(z_d).all():
        raise ValueError("embeddings contain NaN or infinity")
    cosine = z_d @ z_m.T
    variants = {
        "raw_cosine": evaluate_score_matrix(
            cosine, labels, model_names, dataset_names,
            gold_ks=gold_ks, hub_top_k=hub_top_k,
        ),
        f"csls_k{csls_k}": evaluate_score_matrix(
            csls_scores(cosine, k=csls_k), labels, model_names, dataset_names,
            gold_ks=gold_ks, hub_top_k=hub_top_k,
        ),
    }

    per_dataset = []
    for dataset_id in sorted(labels):
        raw = variants["raw_cosine"]["per_dataset"][dataset_id]
        common = {key: raw[key] for key in (
            "dataset_mappedID", "unique_dataset_id", "n_labelled_models", "gold_accuracy"
        )}
        common["all_argmax_gold_models"] = [
            {key: gold[key] for key in ("mappedID", "unique_model_id")}
            for gold in raw["all_argmax_gold_models"]
        ]
        common["variants"] = {
            name: variants[name]["per_dataset"][dataset_id]
            for name in variants
        }
        per_dataset.append(common)
    for variant in variants.values():
        del variant["per_dataset"]
    return variants, per_dataset


def _require_id_snapshot(df: pd.DataFrame, unique_column: str, expected_rows: int) -> list[str]:
    if list(df.columns) != ["mappedID", unique_column]:
        raise ValueError(
            f"ID snapshot must have columns mappedID,{unique_column}; got {list(df.columns)}"
        )
    if len(df) != expected_rows or not np.array_equal(
        df["mappedID"].to_numpy(), np.arange(expected_rows)
    ):
        raise ValueError(f"{unique_column} snapshot is not row-aligned to mappedID 0..N-1")
    return df[unique_column].astype(str).tolist()


def _resolve_export_dir(path: str) -> str:
    candidates = [path]
    if not os.path.isabs(path):
        candidates.extend((os.path.join(os.getcwd(), path), os.path.join(_HERE, path)))
    for candidate in candidates:
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(f"export directory not found: {path}")


def replay_export(export_dir: str) -> dict:
    """Load a manifest-bound export and return a JSON-serialisable report."""
    import torch

    from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import TRAINED_ON

    export_dir = _resolve_export_dir(export_dir)
    manifest_path = os.path.join(export_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"export manifest missing: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    for filename, expected_sha in manifest["files"].items():
        actual_sha = sha256_file(os.path.join(export_dir, filename))
        if actual_sha != expected_sha:
            raise ValueError(f"export file {filename} drifted from manifest")

    z_m = np.load(os.path.join(export_dir, "z_m.npy"), allow_pickle=False)
    z_d = np.load(os.path.join(export_dir, "z_d.npy"), allow_pickle=False)
    n_models = int(manifest["graph"]["num_models"])
    n_datasets = int(manifest["graph"]["num_datasets"])
    dim = int(manifest["embedding"]["dim"])
    if z_m.shape != (n_models, dim) or z_d.shape != (n_datasets, dim):
        raise ValueError("embedding shapes disagree with export manifest")
    if manifest["embedding"].get("normalized") is not True:
        raise ValueError("replay requires a normalized serving export")
    if not np.allclose(np.linalg.norm(z_m, axis=1), 1.0, atol=1e-5) or not np.allclose(
        np.linalg.norm(z_d, axis=1), 1.0, atol=1e-5
    ):
        raise ValueError("serving embeddings are not unit-normalized")

    model_df = pd.read_csv(os.path.join(export_dir, "model_ids.csv"))
    dataset_df = pd.read_csv(os.path.join(export_dir, "dataset_ids.csv"))
    model_names = _require_id_snapshot(model_df, "unique_model_id", n_models)
    dataset_names = _require_id_snapshot(dataset_df, "unique_dataset_id", n_datasets)

    graph_path = manifest["graph"]["path"]
    if not os.path.isabs(graph_path):
        graph_path = os.path.join(_REPO_ROOT, graph_path)
    graph_path = os.path.abspath(graph_path)
    if sha256_file(graph_path) != manifest["graph"]["sha256"]:
        raise ValueError("source graph drifted from export manifest")
    payload = torch.load(graph_path, map_location="cpu", weights_only=False)

    graph_models = payload["unique_model_id"].sort_values("mappedID")["model"].astype(str).tolist()
    graph_datasets = payload["unique_dataset_id"].sort_values("mappedID")["dataset"].astype(str).tolist()
    if graph_models != model_names or graph_datasets != dataset_names:
        raise ValueError("export ID snapshots do not align with the source graph")

    store = payload["data"][TRAINED_ON]
    labels, label_stats = build_eligible_labels(
        store.edge_index[0].cpu().numpy(),
        store.edge_index[1].cpu().numpy(),
        store.edge_attr.cpu().numpy(),
        n_models=n_models,
        n_datasets=n_datasets,
    )
    variants, per_dataset = replay_arrays(
        z_m, z_d, labels, model_names, dataset_names,
        gold_ks=GOLD_KS, csls_k=CSLS_K, hub_top_k=HUB_TOP_K,
    )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "caveat": FULL_GRAPH_CAVEAT,
        "export": {
            "name": os.path.basename(export_dir),
            "path": os.path.relpath(export_dir, _REPO_ROOT),
            "manifest_sha256": sha256_file(manifest_path),
            "graph_sha256": manifest["graph"]["sha256"],
            "checkpoint_sha256": manifest["checkpoint"]["sha256"],
            "checkpoint_config": manifest["checkpoint"].get("config_name"),
            "n_models": n_models,
            "n_datasets": n_datasets,
            "embedding_dim": dim,
            "embedding_surgery": manifest["surgery"],
        },
        "protocol": {
            "score": "exact dot product of stored unit vectors (cosine)",
            "variants": ["raw_cosine", f"csls_k{CSLS_K}"],
            "csls": {
                "requested_k": CSLS_K,
                "effective_dataset_to_model_k": min(CSLS_K, n_models),
                "effective_model_to_dataset_k": min(CSLS_K, n_datasets),
                "formula": "2*cosine - query_topk_mean - candidate_topk_mean",
                "status": "diagnostic ablation, not a production serving score",
            },
            "eligibility": (
                f">={MIN_LABELLED_MODELS} distinct labelled models after max reduction; "
                "non-constant accuracy"
            ),
            "gold_policy": (
                "emit every argmax-accuracy tie; aggregate the lowest mappedID tie "
                "to match stable np.argmax/top1_eval semantics"
            ),
            "rank_policy": "1 + count(scores strictly greater than gold score), tie-safe",
            "gold_ks": list(GOLD_KS),
            "hub_top_k": HUB_TOP_K,
            "hub_query_population": "all exported datasets, not only label-eligible datasets",
        },
        "label_population": label_stats,
        "variants": variants,
        "per_dataset": per_dataset,
    }


def _md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(report: Mapping) -> str:
    """Render the JSON report's principal curves, histograms, and audit rows."""
    export = report["export"]
    ks = [int(k) for k in report["protocol"]["gold_ks"]]
    variants = report["variants"]
    lines = [
        f"# Gold-rank replay — {_md(export['name'])}",
        "",
        f"> **Caveat:** {_md(report['caveat'])}",
        "",
        "## Provenance and population",
        "",
        "| field | value |",
        "|---|---|",
        f"| export | `{_md(export['path'])}` |",
        f"| export manifest sha256 | `{export['manifest_sha256']}` |",
        f"| graph / checkpoint sha256 | `{export['graph_sha256']}` / `{export['checkpoint_sha256']}` |",
        f"| model / dataset rows | {export['n_models']} / {export['n_datasets']} |",
        f"| eligible labelled datasets | {report['label_population']['eligible_datasets']} |",
        f"| eligibility | {_md(report['protocol']['eligibility'])} |",
        f"| gold policy | {_md(report['protocol']['gold_policy'])} |",
        "",
        "## Gold-survival curve and hub summary",
        "",
        "| variant | " + " | ".join(f"gold@{k}" for k in ks)
        + " | median rank | observed hit@1 | distinct top-10 models | max occupancy |",
        "|---|" + "---:|" * (len(ks) + 4),
    ]
    for name, variant in variants.items():
        aggregate = variant["aggregate"]
        hub = variant["hub_occupancy"]
        values = " | ".join(f"{aggregate['gold_at_k'][str(k)]:.3f}" for k in ks)
        lines.append(
            f"| `{name}` | {values} | {aggregate['median_gold_rank']:.1f} | "
            f"{aggregate['observed_hit_at_1']:.3f} | {hub['distinct_models']} | "
            f"{hub['max_occupancy']}/{hub['n_queries']} |"
        )

    lines.extend((
        "",
        "## Hub-occupancy histogram",
        "",
        "The histogram counts all indexed models, including zero-occupancy models.",
        "",
        "| variant | occupancy bin | models |",
        "|---|---:|---:|",
    ))
    for name, variant in variants.items():
        for row in variant["hub_occupancy"]["binned_histogram"]:
            lines.append(f"| `{name}` | {row['occupancy_bin']} | {row['n_models']} |")

    lines.extend((
        "",
        "## Highest-occupancy models",
        "",
        "| variant | mappedID | model | occupancy | query fraction |",
        "|---|---:|---|---:|---:|",
    ))
    for name, variant in variants.items():
        hub = variant["hub_occupancy"]
        for row in hub["occupied_models"][:10]:
            lines.append(
                f"| `{name}` | {row['mappedID']} | {_md(row['unique_model_id'])} | "
                f"{row['occupancy']}/{hub['n_queries']} | {row['query_fraction']:.3f} |"
            )

    lines.extend((
        "",
        "## Per-dataset canonical gold ranks",
        "",
        "All argmax ties are preserved in JSON; this table shows the canonical stable tie.",
        "",
        "| mappedID | dataset | labels | gold model | gold acc | raw rank | CSLS rank |",
        "|---:|---|---:|---|---:|---:|---:|",
    ))
    csls_name = next(name for name in variants if name.startswith("csls_k"))
    for row in report["per_dataset"]:
        raw = row["variants"]["raw_cosine"]
        csls = row["variants"][csls_name]
        gold = raw["canonical_gold_model"]
        lines.append(
            f"| {row['dataset_mappedID']} | {_md(row['unique_dataset_id'])} | "
            f"{row['n_labelled_models']} | {_md(gold['unique_model_id'])} | "
            f"{row['gold_accuracy']:.6g} | {raw['gold_rank']} | {csls['gold_rank']} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_report(report: Mapping, json_path: str, markdown_path: str) -> None:
    """Write one JSON report and its Markdown table rendering."""
    for path in (json_path, markdown_path):
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(report))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-3 full-graph gold-rank replay")
    parser.add_argument("--export", required=True, help="Phase-1 embedding export directory")
    parser.add_argument(
        "--out-stem", default=None,
        help="output path without extension (default: artifacts/reports/<export>/gold_rank_replay)",
    )
    args = parser.parse_args()

    export_dir = _resolve_export_dir(args.export)
    report = replay_export(export_dir)
    if args.out_stem:
        out_stem = os.path.abspath(args.out_stem)
    else:
        out_stem = os.path.join(
            _HERE, "artifacts", "reports", report["export"]["name"], "gold_rank_replay"
        )
    json_path, markdown_path = f"{out_stem}.json", f"{out_stem}.md"
    write_report(report, json_path, markdown_path)

    print(
        f"[{report['export']['name']}] {report['label_population']['eligible_datasets']} "
        f"eligible datasets; full-graph diagnostic"
    )
    for name, variant in report["variants"].items():
        aggregate = variant["aggregate"]
        hub = variant["hub_occupancy"]
        curve = " ".join(
            f"gold@{k}={aggregate['gold_at_k'][str(k)]:.3f}" for k in GOLD_KS
        )
        print(
            f"  {name:12} {curve} median_rank={aggregate['median_gold_rank']:.1f} "
            f"observed_hit@1={aggregate['observed_hit_at_1']:.3f} "
            f"hubs={hub['distinct_models']} max={hub['max_occupancy']}/{hub['n_queries']}"
        )
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {markdown_path}")
    print(f"  CAVEAT: {FULL_GRAPH_CAVEAT}")


if __name__ == "__main__":
    main()
