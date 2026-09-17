"""Independently recompute A0 metrics from hash-bound per-query records.

This module deliberately does not import eval_y2 or global_metrics. Full-lake
rank summaries use the saved integer all-model comparison counts; the top-ten
quality is independently checked against saved full-lake output IDs. Bounded
pool ordering is reconstructed from cosine, prior and the fixed tie bijection.
Missing resource/training/audit evidence never becomes a zero or an old value.

Optional ``--records A0_RECORD_SOURCES.json`` uses this descriptor shape::

    {"sources": [{"id": "pipeline_cost.training_seconds.0",
      "path": "/new/run/cost.json", "sha256": "<bound source SHA256>",
      "pointer": "/segments", "operation": "duration_ns"}]}

The source must already be a new manifest artifact and contain an ``a0``
envelope: ``{protocol: "a0", run_id, seed, graph_digest,
checkpoint_sha256}``. These identities must match the evaluator binding.
For duration_ns, /segments is [{start_ns,end_ns}, ...] across all resumptions;
max consumes raw resource samples, sum consumes numeric native measurements,
json_record preserves a native log record, and file_bytes stats the real file.
``--make-record-template PATH`` discovers available A0_RUN_RECORDS.json,
export and sidecar records and writes descriptors with actual hashes/pointers;
missing sources remain an explicit list. It does not run measurements.
Nested export envelopes live at /stages/embed/a0. The template also maps the
27 existing frozen A0.1 audit records; ``--source-counts A0_SOURCE_COUNTS.json``
consumes the separate newly executed read-only source recount for 15 further
input statistics. Original crawl-event counters remain missing without logs.
Frozen A0.1 input audits are the sole exception to the new-run envelope: only
the frozen_input_audit scope accepts the exact protocol-bound A0.1 audit files.
No descriptor accepts a caller-supplied result value or overrides retrieval.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

FINAL = "hnsw1000_task_prior"
EXACT = "exact1000_task_prior"
FULL = "exact_full_lake_task_prior"
QUALITY = {
    "gold_at_1": "gold@1", "gold_at_10": "gold@10",
    "top3_at_10": "top3@10", "gold_gap_at_1": "gold-gap@1",
    "gold_gap_at_10": "gold-gap@10",
    "root_macro_gold_at_1": "root_gold@1",
    "root_macro_gold_at_10": "root_gold@10",
    "root_macro_top3_at_10": "root_top3@10",
    "root_macro_gold_gap_at_10": "root_gold-gap@10",
}


class IntegrityError(ValueError):
    """The supplied artifacts cannot establish the frozen A0 computation."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrityError(message)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical(value: Any) -> str:
    # Wire-format contract of a0_evaluation._digest, independently implemented.
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _local(raw: Path, name: str) -> Path:
    path = (raw / name).resolve()
    _require(path.is_relative_to(raw.resolve()), f"Raw artifact escapes A0 directory: {name}")
    return path


def verify_manifest(raw: Path, protocol_path: Path, inventory: dict) -> tuple[dict, dict]:
    manifest = _json(raw / "A0_EVALUATION_MANIFEST.json")
    _require(manifest.get("schema_version") == "a0.raw.v1", "Unknown A0 raw schema")
    _require(manifest.get("protocol") == "a0", "Only A0 raw outputs are accepted")
    _require(bool(manifest.get("run_id")), "Missing A0 run identity")
    binding = manifest["binding"]
    _require(binding.get("run_id", manifest["run_id"]) == manifest["run_id"], "Run identity mismatch")
    _require(binding.get("protocol_sha256") == sha256(protocol_path), "Protocol SHA256 mismatch")
    _require(_canonical(binding) == manifest.get("binding_sha256"), "Binding SHA256 mismatch")
    protocol = _json(protocol_path)
    authority = protocol["authority"]
    _require(authority["sha256"] == inventory["authority"]["sha256"], "Inventory authority mismatch")
    _require(sha256(Path(authority["path"])) == authority["sha256"], "Authoritative evidence changed")
    _require(binding["N"] == protocol["frozen_data"]["candidate_models"], "Candidate universe changed")
    _require(binding["K"] == protocol["evaluation"]["hnsw"]["K"], "Candidate pool changed")
    _require(binding["return_k"] == protocol["evaluation"]["fusion"]["return_k"], "Return size changed")
    hashes = {str(protocol_path.resolve()): sha256(protocol_path)}

    def verify_file(item: dict, path: Path) -> None:
        _require(path.is_file(), f"Missing bound artifact: {path}")
        actual = sha256(path)
        _require(actual == item["sha256"], f"SHA256 mismatch: {path}")
        if "bytes" in item:
            _require(path.stat().st_size == item["bytes"], f"Byte count mismatch: {path}")
        hashes[str(path.resolve())] = actual

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                verify_file(value, Path(value["path"]))
            else:
                for child in value.values():
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(binding)
    for record in protocol.get("input_audit_binding", {}).values():
        if isinstance(record, dict) and "path" in record and "sha256" in record:
            verify_file(record, Path(record["path"]))
    for name, item in manifest["artifacts"].items():
        _require(item.get("new_artifact") is True, f"Old artifact presented as new: {name}")
        verify_file(item, _local(raw, name))
    for seed in manifest["seeds"]:
        seed_binding = binding["seed_inputs"][seed]
        _require(bool(seed_binding.get("graph_digest")), f"Missing graph digest for seed {seed}")
        _require(seed_binding["graph_digest"] != binding.get("old_graph_digest"),
                 f"Old graph used for seed {seed}")
    hashes[str((raw / "A0_EVALUATION_MANIFEST.json").resolve())] = sha256(raw / "A0_EVALUATION_MANIFEST.json")
    return manifest, hashes


def load_pack(raw: Path, manifest: dict, name: str, seed: int) -> dict[str, np.ndarray]:
    _require(name in manifest["artifacts"], f"Unbound raw package: {name}")
    _require(manifest["artifacts"][name].get("seed") in (seed, str(seed)), f"Artifact seed mismatch: {name}")
    with np.load(_local(raw, name), allow_pickle=False) as pack:
        result = {key: pack[key] for key in pack.files}
    if "seed" in result:
        _require(int(result["seed"]) == seed, f"Raw seed mismatch: {name}")
    return result


def _query_order(pack: dict, query: np.ndarray) -> np.ndarray:
    actual = np.asarray(pack["query"])
    _require(actual.ndim == 1 and np.issubdtype(actual.dtype, np.integer), "Query IDs must be integer vectors")
    _require(len(np.unique(actual)) == len(actual), "Duplicate query IDs")
    _require(set(actual.tolist()) == set(query.tolist()), "Query identity mismatch")
    lookup = {int(q): i for i, q in enumerate(actual)}
    return np.asarray([lookup[int(q)] for q in query], dtype=np.int64)


def labels_from_pack(pack: dict, n_models: int) -> dict:
    query = np.asarray(pack["query"])
    _query_order(pack, query)
    roots = np.asarray(pack["root"])
    tasks = np.asarray(pack["task_id"])
    offsets = np.asarray(pack["observed_offsets"])
    ids = np.asarray(pack["observed_ids"])
    values = np.asarray(pack["observed_values"], dtype=float)
    _require(len(query) > 0 and roots.shape == tasks.shape == query.shape, "Invalid query/root/task shapes")
    _require(offsets.shape == (len(query) + 1,) and np.issubdtype(offsets.dtype, np.integer), "Invalid observed offsets")
    _require(offsets[0] == 0 and offsets[-1] == len(ids) == len(values) and np.all(np.diff(offsets) >= 3),
             "Observed candidate slices must contain at least three models")
    _valid_ids(ids, n_models)
    _require(np.isfinite(values).all(), "Nonfinite held-out values")
    out = {"query": query, "root": roots, "task_id": tasks, "gold": [], "top3": [], "near": [], "observed": []}
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        models, perf = ids[lo:hi], values[lo:hi]
        _require(len(np.unique(models)) == len(models), "Repeated held-out model IDs")
        _require(np.std(perf.astype(np.float32)) > 0, "Held-out values fail original float32 standard-deviation eligibility")
        out["gold"].append(int(models[np.argmax(perf)]))
        out["top3"].append(models[np.argsort(-perf)[:3]])
        out["near"].append(models[perf >= perf.max() - 0.01])
        out["observed"].append((models, perf))
    out["gold"] = np.asarray(out["gold"], dtype=np.int64)
    return out


def _valid_ids(ids: np.ndarray, n_models: int) -> None:
    _require(np.issubdtype(ids.dtype, np.integer), "Model IDs must be integers")
    _require(bool(np.all((ids >= 0) & (ids < n_models))), "Model ID outside candidate universe")


def _unique_rows(ids: np.ndarray, width: int, n_models: int) -> None:
    _require(ids.ndim == 2 and ids.shape[1] == width, "Wrong candidate/return width")
    _valid_ids(ids, n_models)
    _require(all(len(set(row.tolist())) == width for row in ids), "Duplicate model IDs within query")


def _tie_key(ids: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return ids.astype(np.uint64) * np.uint64(11400714819323198485) + np.uint64(0xD1B54A32D192ED03)


def _quality(ranks: np.ndarray, labels: dict) -> tuple[dict, dict]:
    roots = labels["root"]
    _require(ranks.shape == (len(roots), 3), "Wrong probe rank shape")
    root_keys = np.unique(roots)
    flags = {
        "gold@1": ranks[:, 0] <= 1, "gold@10": ranks[:, 0] <= 10,
        "top3@10": ranks[:, 1] <= 10, "gold-gap@1": ranks[:, 2] <= 1,
        "gold-gap@10": ranks[:, 2] <= 10,
    }
    row = {name: float(value.mean()) for name, value in flags.items()}
    for name in ("gold@1", "gold@10", "top3@10", "gold-gap@10"):
        row["root_" + name] = float(np.mean([flags[name][roots == r].mean() for r in root_keys]))
    row.update(n_queries=len(roots), n_roots=len(root_keys))
    return row, {name: int(value.sum()) for name, value in flags.items()}


def pool_recompute(pack: dict, labels: dict, n_models: int, k: int, return_k: int) -> tuple[dict, dict, dict]:
    order = _query_order(pack, labels["query"])
    packed_labels = labels_from_pack(pack, n_models)
    for i, other_i in enumerate(order):
        _require(labels["root"][i] == packed_labels["root"][other_i] and
                 labels["task_id"][i] == packed_labels["task_id"][other_i], "Root/task identity mismatch")
        for a, b in zip(labels["observed"][i], packed_labels["observed"][other_i]):
            _require(np.array_equal(a, b), "Gold label records changed between exact and HNSW")
    model = np.asarray(pack["model"])[order]
    cosine, prior = np.asarray(pack["score"])[order], np.asarray(pack["prior"])[order]
    _unique_rows(model, k, n_models)
    _require(cosine.dtype == prior.dtype == np.float32, "Fusion inputs must retain float32 precision")
    _require(cosine.shape == prior.shape == model.shape, "Invalid pool feature shapes")
    _require(np.isfinite(cosine).all() and np.isfinite(prior).all(), "Nonfinite fusion inputs")
    _require(np.all(np.abs(cosine) <= 1.00001), "Cosine outside normalized geometry")
    _require(np.all((prior >= 0) & (prior <= 1)), "Task prior outside oriented range")
    fused = (cosine + np.float32(1)) * np.float32(0.5) + prior
    _require(np.array_equal(fused, np.asarray(pack["fused"])[order]), "Stored fusion disagrees with frozen float32 formula")
    ranking = np.stack([row[np.lexsort((_tie_key(row), -scores))] for row, scores in zip(model, fused)])
    top10 = ranking[:, :return_k]
    _require(np.array_equal(top10, np.asarray(pack["top10"])[order]), "Saved top-ten disagrees with independent tie ordering")
    ranks = np.full((len(order), 3), k + 1, dtype=np.int64)
    gold_pos = np.zeros(len(order), dtype=np.int64)
    for i, row in enumerate(ranking):
        position = {int(m): j + 1 for j, m in enumerate(row)}
        ranks[i, 0] = position.get(int(labels["gold"][i]), k + 1)
        ranks[i, 1] = min(position.get(int(m), k + 1) for m in labels["top3"][i])
        ranks[i, 2] = min(position.get(int(m), k + 1) for m in labels["near"][i])
        gold_pos[i] = position.get(int(labels["gold"][i]), 0)
    _require(np.array_equal(ranks - 1, np.asarray(pack["pool_counts"])[order]), "Stored pool probe counts disagree")
    _require(np.array_equal(gold_pos, np.asarray(pack["gold_position"])[order]), "Stored conditional gold positions disagree")
    row, successes = _quality(ranks, labels)
    found = gold_pos > 0
    row.update(N=k, median_gold_rank=None, median_rank_over_N=None, vs_random=None,
               **{"gold_in_first_stage@1000": float(found.mean())},
               median_gold_rank_if_retrieved=float(np.median(gold_pos[found])) if found.any() else None,
               gold_retrieved_query_count=int(found.sum()), candidate_universe_N=n_models,
               rerank_K=k, return_k=return_k)
    detail = {"model": model, "top10": top10, "ranks": ranks, "gold_position": gold_pos,
              "gold_retrieved_query_count": int(found.sum()), "quality_success_counts": successes}
    return row, successes, detail


def full_recompute(pack: dict, labels: dict, n_models: int, return_k: int) -> tuple[dict, dict]:
    order = _query_order(pack, labels["query"])
    counts = np.asarray(pack["full_counts"])[order]
    _require(np.issubdtype(counts.dtype, np.integer) and counts.shape == (len(order), 3), "Invalid full-lake comparison counts")
    _require(np.all((counts >= 0) & (counts < n_models)), "Full-lake comparison counts out of range")
    top10 = np.asarray(pack["full_top10"])[order]
    _unique_rows(top10, return_k, n_models)
    ranks = counts + 1
    for i, row in enumerate(top10):
        pos = {int(m): j + 1 for j, m in enumerate(row)}
        probes = [[labels["gold"][i]], labels["top3"][i], labels["near"][i]]
        for column, candidates in enumerate(probes):
            top_rank = min(pos.get(int(m), return_k + 1) for m in candidates)
            _require(min(int(ranks[i, column]), return_k + 1) == top_rank, "Full-lake top-ten disagrees with probe counts")
    row, successes = _quality(ranks, labels)
    row.update(N=n_models, median_gold_rank=float(np.median(ranks[:, 0])),
               median_rank_over_N=float(np.median(ranks[:, 0]) / n_models),
               vs_random=row["gold@10"] / (return_k / n_models))
    return row, {"quality_success_counts": successes, "top10": top10}


def _measurement(value: Any, *, status: str = "recomputed", reason: str | None = None,
                 numerator: Any = None, denominator: Any = None, unit: str | None = None) -> dict:
    record = {"value": value, "status": status, "reason": reason,
              "numerator": numerator, "denominator": denominator}
    if unit is not None:
        record["unit"] = unit
    return record


def _ratio(numerator: float, denominator: float) -> dict:
    if denominator == 0:
        return _measurement(None, status="undefined", reason="zero denominator",
                            numerator=numerator, denominator=denominator, unit="fraction")
    return _measurement(numerator / denominator, numerator=numerator, denominator=denominator, unit="fraction")


def _add(values: dict, scope: str, seed: int | str, name: str, value: Any, **kwargs: Any) -> None:
    values[f"{scope}.{name}.{seed}"] = _measurement(value, **kwargs)


def calibration_recompute(raw: Path, manifest: dict, seed: int, seed_meta: dict,
                          query: np.ndarray, exact_ids: np.ndarray, hnsw: dict,
                          protocol: dict) -> dict:
    config = protocol["evaluation"]["ef_calibration"]
    grid, threshold = config["grid"], config["threshold"]
    n_models, k = manifest["binding"]["N"], manifest["binding"]["K"]
    trace = []
    selected_ids = None
    for name in seed_meta["calibration"]:
        pack = load_pack(raw, manifest, name, seed)
        order = _query_order(pack, query)
        ids = pack["model"][order]
        _unique_rows(ids, k, n_models)
        ef = int(pack.get("ef_search", name.rsplit("_ef", 1)[1].split(".")[0]))
        _require(len(trace) < len(grid) and ef == grid[len(trace)], "Calibration grid order changed")
        _require(not trace or trace[-1]["recall"] < threshold, "Calibration continued after first passing ef")
        recalls = np.asarray([len(set(a.tolist()) & set(b.tolist())) / k for a, b in zip(ids, exact_ids)])
        if "recall_per_query" in pack:
            _require(np.allclose(recalls, pack["recall_per_query"][order], rtol=0, atol=1e-12), "Per-query recall mismatch")
        trace.append({"ef_search": ef, "recall": float(recalls.mean()), "query_count": len(query)})
        selected_ids = ids
    _require(bool(trace), "Missing calibration trace")
    selected, passed = trace[-1]["ef_search"], trace[-1]["recall"] >= threshold
    _require(passed or selected == grid[-1], "Failed calibration stopped before maximum preset ef")
    _require(selected == seed_meta["selected_ef"] == int(hnsw["selected_ef"]), "Selected ef mismatch")
    _require(passed == seed_meta["calibration_passed"], "Calibration status mismatch")
    hnsw_order = _query_order(hnsw, query)
    _require(np.array_equal(np.asarray(hnsw["calibration_ids"])[hnsw_order], selected_ids), "Selected calibration IDs disagree")
    return {"trace": trace, "selected": selected, "passed": passed}


def recompute_seed(raw: Path, manifest: dict, protocol: dict, seed: int) -> tuple[dict, dict]:
    meta = manifest["seeds"][str(seed)]
    n_models, k, return_k = (manifest["binding"][key] for key in ("N", "K", "return_k"))
    exact = load_pack(raw, manifest, meta["exact"], seed)
    hnsw = load_pack(raw, manifest, meta["hnsw"], seed)
    _require(int(exact["n_models"]) == int(hnsw["n_models"]) == n_models, "Raw candidate universe mismatch")
    labels = labels_from_pack(exact, n_models)
    _require(len(labels["query"]) == protocol["evaluation"]["expected_query_counts"][str(seed)], "Eligible query count changed")
    identity = protocol.get("input_audit_binding", {}).get("query_identity")
    if identity:
        frozen_rows = [json.loads(line) for line in Path(identity["path"]).read_text(encoding="utf-8").splitlines() if line.strip()]
        frozen = {int(row["query_mappedID"]): row for row in frozen_rows if int(row["seed"]) == seed}
        _require(set(frozen) == set(labels["query"].tolist()), "Query IDs differ from frozen A0.1 identity")
        for i, query_id in enumerate(labels["query"]):
            row = frozen[int(query_id)]
            models, perf = labels["observed"][i]
            _require(str(labels["root"][i]) == row["root"], "Query root differs from frozen A0.1 identity")
            _require(hashlib.sha256(np.ascontiguousarray(models).tobytes()).hexdigest() == row["candidate_ids_sha256"], "Held-out candidate identity changed")
            _require(hashlib.sha256(np.ascontiguousarray(perf).tobytes()).hexdigest() == row["oriented_values_float64_sha256"], "Held-out performance labels changed")
    exact_row, _, exact_detail = pool_recompute(exact, labels, n_models, k, return_k)
    hnsw_row, _, hnsw_detail = pool_recompute(hnsw, labels, n_models, k, return_k)
    full_row, full_detail = full_recompute(exact, labels, n_models, return_k)
    exact_row["full_fused_top10_in_dense_top1000"] = float(np.mean([
        len(set(top.tolist()) & set(pool.tolist())) / return_k
        for top, pool in zip(full_detail["top10"], exact_detail["model"])]))
    native = {EXACT: exact_row, FINAL: hnsw_row, FULL: full_row}
    details = {EXACT: exact_detail, FINAL: hnsw_detail, FULL: full_detail}
    values: dict[str, dict] = {}
    for scope, row in native.items():
        for name, key in QUALITY.items():
            _add(values, scope, seed, name, row[key])
        for name, value in {"queries": len(labels["query"]), "roots": row["n_roots"], "candidate_models": n_models,
                            "pool_size": n_models if scope == FULL else k, "return_size": return_k,
                            "quality_success_counts": details[scope]["quality_success_counts"],
                            "extra_original_scorer_fields": row}.items():
            _add(values, scope, seed, name, value)
        if scope != FULL:
            count = details[scope]["gold_retrieved_query_count"]
            _add(values, scope, seed, "gold_retrieved_query_count", count)
            _add(values, scope, seed, "median_gold_position_when_retrieved", row["median_gold_rank_if_retrieved"],
                 status="recomputed" if count else "undefined", reason=None if count else "no gold retrieved",
                 denominator=count)
            _add(values, scope, seed, "unavailable_full_lake_rank_fields", None, status="undefined",
                 reason="Bounded pool does not establish full-lake ranks")
        else:
            for name, key in {"median_full_fused_gold_rank": "median_gold_rank",
                              "median_full_fused_rank_over_N": "median_rank_over_N",
                              "vs_uniform_random_top10": "vs_random"}.items():
                _add(values, scope, seed, name, row[key])
    _add(values, FINAL, seed, "actual_hnsw_gold_coverage_at_1000", hnsw_row["gold_in_first_stage@1000"], unit="fraction")
    pairs = {}
    for name, numerator, denominator in (
        ("exact_pool_retention", exact_row["gold@10"], full_row["gold@10"]),
        ("ann_retention", hnsw_row["gold@10"], exact_row["gold@10"]),
        ("overall_retention", hnsw_row["gold@10"], full_row["gold@10"]),
    ):
        value = _ratio(numerator, denominator)
        values[f"retrieval_diagnostics.{name}.{seed}"] = value
        pairs[name] = {key: value[key] for key in ("numerator", "denominator", "reason")}
    overlap_count = sum(len(set(top.tolist()) & set(pool.tolist())) for top, pool in
                        zip(full_detail["top10"], exact_detail["model"]))
    for name, numerator, denominator in (
        ("exact_pool_gold_coverage_at_1000", exact_detail["gold_retrieved_query_count"], len(labels["query"])),
        ("full_fused_top10_in_exact_pool", overlap_count, len(labels["query"]) * return_k),
    ):
        values[f"retrieval_diagnostics.{name}.{seed}"] = _ratio(numerator, denominator)
        pairs[name] = {"numerator": numerator, "denominator": denominator}
    _add(values, "retrieval_diagnostics", seed, "retention_numerators_denominators", pairs)
    calibration = calibration_recompute(raw, manifest, seed, meta, labels["query"], exact_detail["model"], hnsw, protocol)
    for item in calibration["trace"]:
        _add(values, "ann_calibration", seed, f"recall_at_1000_ef_{item['ef_search']}", item["recall"])
    for ef in protocol["evaluation"]["ef_calibration"]["grid"][len(calibration["trace"]):]:
        _add(values, "ann_calibration", seed, f"recall_at_1000_ef_{ef}", None, status="not_applicable",
             reason="Unvisited ef after first documented recall threshold pass")
    for name, value in {"selected_ef_search": calibration["selected"], "selected_recall_at_1000": calibration["trace"][-1]["recall"],
                        "calibration_passed": calibration["passed"], "calibration_query_count": len(labels["query"]),
                        "calibration_trace": calibration["trace"]}.items():
        _add(values, "ann_calibration", seed, name, value)
    timing_order = _query_order(hnsw, labels["query"])
    timing = {}
    for key in ("hnsw", "rerank", "total"):
        array = np.asarray(hnsw[key + "_ns"])[timing_order]
        _require(array.shape == labels["query"].shape and np.issubdtype(array.dtype, np.integer) and np.all(array >= 0),
                 "Invalid raw nanosecond timing samples")
        timing[key] = array
        prefix = {"hnsw": "hnsw", "rerank": "prior_rerank", "total": "total"}[key]
        for quantile in (50, 95):
            _add(values, "retrieval_cost", seed, f"{prefix}_latency_p{quantile}_ms", float(np.percentile(array / 1e6, quantile)))
    _require(np.array_equal(timing["total"], timing["hnsw"] + timing["rerank"]), "Total timing is not paired component sum")
    index_path = _local(raw, meta["index"])
    _require(meta["index"] in manifest["artifacts"], "Unbound index")
    index_bytes = index_path.stat().st_size
    build = meta["index_build"]
    duration = (build["end_ns"] - build["start_ns"]) / 1e9
    _require(duration > 0 and abs(duration - build["build_seconds"]) <= 1e-6, "Index build timestamps disagree")
    for name, value in {"index_bytes": index_bytes, "index_gib": index_bytes / 2**30, "build_seconds": duration}.items():
        _add(values, "index_cost", seed, name, value)
    for name in ("final_output_candidate_membership", "fusion_and_tie_consistency"):
        _add(values, "reproducibility_checks", seed, name, True, status="verified")
    return values, native


def _compare_native(evaluator: dict, native: dict[int, dict], values: dict) -> None:
    """Compare every independent native field; missing fields do not pass."""
    paths = {FINAL: ("hnsw", "G_hnsw1000_task"), EXACT: ("per_seed", "G_exact1000_task"),
             FULL: ("per_seed", "G_full_task")}
    for seed, rows in native.items():
        for scope, row in rows.items():
            outer, key = paths[scope]
            reported = evaluator.get(outer, {}).get(str(seed), {}).get("rows", {}).get(key)
            _require(isinstance(reported, dict), f"Missing evaluator row {scope}/{seed}")
            for name, value in row.items():
                _require(name in reported, f"Evaluator omitted native field {scope}/{seed}/{name}")
                actual = reported[name]
                if value is None:
                    _require(actual is None, f"Undefined rank replaced with value: {scope}/{seed}/{name}")
                else:
                    _require(isinstance(actual, (int, float)) and np.isfinite(actual) and
                             np.isclose(actual, value, atol=1e-12, rtol=1e-12),
                             f"Independent metric mismatch: {scope}/{seed}/{name}")
            unhandled = set(reported) - set(row)
            # ANN recall may be attached to the native scorer row by the writer.
            if "recall@1000" in unhandled:
                expected = values[f"ann_calibration.selected_recall_at_1000.{seed}"]["value"]
                _require(np.isclose(reported["recall@1000"], expected, atol=1e-12, rtol=1e-12), "Native ANN recall mismatch")
                unhandled.remove("recall@1000")
            _require(not unhandled, f"Unhandled native scorer fields: {sorted(unhandled)}")
        _add(values, "reproducibility_checks", seed, "independent_metric_recomputation", True, status="verified")


def _pointer(document: Any, pointer: str) -> Any:
    if not pointer:
        return document
    _require(pointer.startswith("/"), "Records use RFC6901 JSON pointers")
    for part in pointer[1:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        document = document[int(key)] if isinstance(document, list) else document[key]
    return document


def supplemental_records(path: Path | None, hashes: dict, values: dict, inventory: dict,
                         manifest: dict | None = None, protocol: dict | None = None) -> None:
    """Derive remaining cost/log/audit items from explicitly bound raw records.

    Optional records JSON: {sources:[{id,path,sha256,pointer,operation,...}]}.
    Operations are file_bytes, json_record, sum, max, duration_ns, or disabled.
    A source file must already belong to the verified A0 manifest dependency
    chain. This adapter cannot override independently scored retrieval metrics.
    """
    if path is None:
        return
    specs = _json(path)
    allowed = {item["id"]: item for item in inventory["metrics"]}
    allowed_scopes = {"pipeline_cost", "training_records", "reproducibility", "frozen_input_audit",
                      "feature_repair_verification", "task_prior", "reproducibility_checks"}
    file_records = {}
    def collect_files(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                file_records[str(Path(value["path"]).resolve())] = value
            else:
                for child in value.values():
                    collect_files(child)
        elif isinstance(value, list):
            for child in value:
                collect_files(child)
    if manifest:
        collect_files(manifest["binding"])
        collect_files(manifest["artifacts"])
    for spec in specs["sources"]:
        metric_id = spec["id"]
        _require(metric_id in allowed and metric_id not in values, f"Invalid/duplicate supplemental metric: {metric_id}")
        item = allowed[metric_id]
        _require(item["scope"] in allowed_scopes, "Raw-record adapter cannot supply quality or retention")
        source = Path(spec["path"]).resolve()
        _require(str(source) in hashes and hashes[str(source)] == spec["sha256"], f"Supplemental record is not hash-bound: {source}")
        file_record = file_records.get(str(source), {})
        document = _json(source)
        if item["scope"] == "frozen_input_audit":
            permitted_audits = (protocol or {}).get("input_audit_binding", {})
            _require(any(isinstance(ref, dict) and ref.get("path") and
                         Path(ref["path"]).resolve() == source and ref.get("sha256") == spec["sha256"]
                         for ref in permitted_audits.values()), "Input audit is not the exact frozen A0.1 audit")
        else:
            _require(file_record.get("new_artifact") is True, "Old log cannot supply new measurements")
            try:
                envelope = _pointer(document, spec.get("envelope_pointer", "/a0")) if isinstance(document, dict) else {}
            except (KeyError, IndexError, TypeError):
                envelope = {}
            _require(manifest is not None and envelope.get("protocol") == "a0" and
                     envelope.get("run_id") == manifest["run_id"], "Raw log lacks matching A0 run envelope")
            metric_seed = item["seed"]
            _require(isinstance(metric_seed, int) and envelope.get("seed") == metric_seed,
                     "Raw log seed does not match metric seed")
            expected = manifest["binding"]["seed_inputs"][str(metric_seed)]
            _require(envelope.get("graph_digest") == expected["graph_digest"] and
                     envelope["graph_digest"] != manifest["binding"].get("old_graph_digest"),
                     "Raw log graph does not match new A0 graph")
            checkpoints = [rec["sha256"] for rec in expected.get("files", [])
                           if rec.get("role") == "new_epoch25_checkpoint" and rec.get("new_artifact") is True]
            _require(bool(checkpoints) and envelope.get("checkpoint_sha256") in checkpoints,
                     "Raw log is not bound to this seed's new final checkpoint")
        operation = spec["operation"]
        status, reason = "recomputed", None
        if operation == "file_bytes":
            value = source.stat().st_size
        else:
            value = _pointer(document, spec.get("pointer", ""))
            if operation in ("epoch_field", "epoch_final_field"):
                _require(isinstance(value, list) and len(value) == 25, "Loss extraction requires all 25 epoch records")
                field = spec["field"]
                series = [row[field] for row in value]
                _require(all(isinstance(v, (int, float)) and np.isfinite(v) for v in series), "Nonfinite or missing native loss field")
                value = series if operation == "epoch_field" else series[-1]
            elif operation in ("sum", "max"):
                data = np.asarray(value)
                _require(data.size > 0 and np.issubdtype(data.dtype, np.number) and np.isfinite(data).all(), "Invalid numeric record samples")
                value = float(data.sum() if operation == "sum" else data.max())
            elif operation in ("duration_ns", "evaluation_duration_ns"):
                if operation == "evaluation_duration_ns":
                    value = document.get("exact_segments", []) + document.get("hnsw_segments", [])
                _require(isinstance(value, list) and bool(value), "Timing segments must be a nonempty list")
                if any(seg.get("end_ns") is None for seg in value):
                    values[metric_id] = _measurement(None, status="missing", reason="Raw timing has an unobserved end after interruption; elapsed cost cannot be reconstructed")
                    continue
                if document.get("schema_version") == "a0.evaluation.resources.v1":
                    stages = ("exact", "hnsw") if operation == "evaluation_duration_ns" else ("exact",)
                    if not all(any(seg.get("status") == "completed" for seg in document.get(stage + "_segments", [])) for stage in stages):
                        values[metric_id] = _measurement(None, status="missing", reason="Evaluation stage has no completed timed attempt")
                        continue
                _require(all(seg["end_ns"] >= seg["start_ns"] for seg in value), "Invalid timing segment")
                value = sum(seg["end_ns"] - seg["start_ns"] for seg in value) / 1e9
            elif operation == "evaluation_peak":
                segments = document.get("exact_segments", []) + document.get("hnsw_segments", [])
                samples = [seg.get("resource_peaks", {}).get(spec["field"]) for seg in segments]
                if not segments or any(seg.get("end_ns") is None for seg in segments) or any(sample is None for sample in samples):
                    values[metric_id] = _measurement(None, status="missing", reason="Evaluation peak samples incomplete; observed partial maximum cannot substitute complete peak")
                    continue
                _require(all(isinstance(sample, (int, float)) and np.isfinite(sample) and sample >= 0 for sample in samples), "Invalid evaluation peak samples")
                value = max(samples)
            elif operation == "ratio":
                numerator = _pointer(document, spec["numerator_pointer"])
                denominator = _pointer(document, spec["denominator_pointer"])
                _require(denominator > 0, "Audit ratio denominator must be positive")
                value = numerator / denominator * spec.get("scale", 1)
            elif operation == "audit_files":
                value = {rec["path"]: rec["after"]["size_bytes"] for rec in value}
            elif operation == "audit_checks":
                value = {rec["name"]: rec for rec in value if any(token in rec["name"] for token in spec["name_contains"])}
                _require(value and all(rec["status"] == "PASS" for rec in value.values()), "Frozen input checks missing or failed")
            elif operation == "passed_gate":
                _require(isinstance(value, dict) and value.get("ok") is True and value.get("ran", True) is True,
                         "Expected executed producer gate did not pass")
                if value.get("gate") == "G-F7d":
                    deltas = value.get("max_abs_delta", {})
                    _require(set(deltas) == {"model", "dataset"} and
                             all(isinstance(delta, (int, float)) and np.isfinite(delta) and 0 <= delta < 1e-5 for delta in deltas.values()),
                             "Chunked inference raw deltas fail the frozen tolerance")
                value = {"passed": True, "raw_gate": value}
            elif operation == "disabled":
                _require(item["name"] == "disabled_diagnostics" and bool(spec.get("reason")), "Only frozen disabled diagnostics can use disabled status")
                _require(isinstance(value, dict) and value.get("skip_diagnostics") is True, "Disabled claim lacks bound config evidence")
                status, reason = "disabled", spec["reason"]
            else:
                _require(operation == "json_record", f"Unknown record operation: {operation}")
                status = "verified"
        if item["name"] == "all_existing_epoch_fields":
            _require(isinstance(value, list) and len(value) == 25, "All 25 native epoch records are required")
        if item["category"] == "correctness_gate":
            _require(value is True or (isinstance(value, dict) and value.get("passed") is True), f"Failed correctness evidence: {metric_id}")
        values[metric_id] = _measurement(value, status=status, reason=reason)
        if operation == "ratio":
            values[metric_id].update(numerator=numerator, denominator=denominator, scale=spec.get("scale", 1))
        values[metric_id]["source"] = f"{source}#{spec.get('pointer', '')} ({operation})"
    hashes[str(path.resolve())] = sha256(path)


def _escape_pointer(value):
    return str(value).replace("~", "~0").replace("/", "~1")


def frozen_audit_descriptors(protocol: dict) -> list[dict]:
    """Map actual A0.1 audit measurements; never read F/X/Y/Z result totals."""
    binding = protocol.get("input_audit_binding", {})
    sources = []
    def add(name, audit, pointer, operation="json_record", **extra):
        ref = binding.get(audit)
        if not isinstance(ref, dict) or not Path(ref["path"]).is_file():
            return
        document = _json(Path(ref["path"]))
        try:
            _pointer(document, pointer)
        except (KeyError, IndexError, TypeError):
            return
        sources.append({"id": f"frozen_input_audit.{name}.shared", "path": str(Path(ref["path"]).resolve()),
                        "sha256": ref["sha256"], "pointer": pointer, "operation": operation, **extra})
    simple = {"candidate_models": "model_rows", "dataset_task_nodes": "dataset_task_rows",
              "historical_only_models": "historical_appended_rows", "matched_card_nodes": "matched_cards",
              "stored_total_directed_edges": "total_directed_edges", "family_vocab_rows": "family_vocab_rows"}
    for name, key in simple.items():
        add(name, "input_audit", "/measurements/" + key)
    for name, key in {"trained_on_edges": "model__trained_on__dataset", "rev_trained_on_edges": "dataset__rev_trained_on__model",
                      "stored_similar_to_edges": "dataset__similar_to__dataset", "is_base_of_edges": "model__is_base_of__model",
                      "rev_is_base_of_edges": "model__rev_is_base_of__model"}.items():
        add(name, "input_audit", "/measurements/relation_counts/" + key)
    for key in ("modellens_v2", "hf_model_index", "d0_v1_5", "hf_effective", "a_ctrl_2000m", "diverse_zoo"):
        add("retained_" + key + "_edges", "input_audit", "/measurements/supervision_source_counts/" + key)
    add("supervision_edges", "input_audit", "/measurements/canonical_parquet_rows/supervision_merged.parquet")
    add("matched_card_percent", "input_audit", "/measurements/matched_cards", "ratio", scale=100,
        numerator_pointer="/measurements/matched_cards", denominator_pointer="/measurements/dataset_task_rows")
    add("resolved_lineage_percent", "input_review", "/lineage_reconstruction/resolved_nonself_edges", "ratio", scale=100,
        numerator_pointer="/lineage_reconstruction/resolved_nonself_edges", denominator_pointer="/lineage_reconstruction/declared")
    add("input_file_bytes", "input_audit", "/files", "audit_files")
    add("rowmap_identity_and_endpoint_checks", "input_audit", "/checks", "audit_checks",
        name_contains=["rowmap", "contiguous_physical_order", "equal_ladder", "endpoint", "node_id_order", "node_serialization", "roots_match", "historical_suffix"])
    add("input_hash_match", "input_audit", "/checks", "audit_checks", name_contains=["hash:", "input_unchanged:", "file_stable:", "source_stable"])
    if isinstance(binding.get("snapshot_audit"), dict):
        doc = _json(Path(binding["snapshot_audit"]["path"]))
        for index, snapshot in enumerate(doc.get("snapshots", [])):
            prefix = f"/snapshots/{index}/measured/"
            if snapshot["kind"] == "model":
                add("snapshot_unique_models", "snapshot_audit", prefix + "unique_normalized_repository_ids")
                add("model_snapshot_shards", "snapshot_audit", prefix + "shards")
            elif snapshot["kind"] == "dataset":
                # Raw repository identity is the frozen denominator, not lowercased collisions.
                add("snapshot_dataset_repositories", "snapshot_audit", prefix + "unique_raw_repository_ids")
                add("dataset_snapshot_shards", "snapshot_audit", prefix + "shards")
    return sources


def make_record_template(manifest: dict, protocol: dict | None = None) -> dict:
    """Discover descriptors from present native envelopes, without guessing data."""
    sources, missing = frozen_audit_descriptors(protocol or {}), []
    run_fields = {
        ("pipeline_cost", "training_seconds"): ("/train_segments", "duration_ns"),
        ("pipeline_cost", "training_peak_rss_bytes"): ("/peak_process_rss_bytes", "max"),
        ("pipeline_cost", "training_peak_vram_bytes"): ("/peak_gpu_allocated_bytes", "max"),
        ("training_records", "epochs_completed"): ("/epochs", "json_record"),
        ("training_records", "initialization_seed"): ("/initialization_seed", "json_record"),
        ("training_records", "split_seed"): ("/a0/seed", "json_record"),
        ("training_records", "effective_resolved_config"): ("/resolved_config", "json_record"),
        ("training_records", "mechanism_gate"): ("/mechanism_gate", "json_record"),
        ("training_records", "all_existing_epoch_fields"): ("/history", "json_record"),
        ("training_records", "disabled_diagnostics"): ("/resolved_config", "disabled"),
        ("reproducibility", "runtime_environment"): ("/runtime_environment", "json_record"),
    }
    for seed in (0, 1, 2):
        files = list(manifest["binding"]["seed_inputs"].get(str(seed), {}).get("files", []))
        files.extend(rec for rec in manifest.get("artifacts", {}).values() if rec.get("seed") in (seed, str(seed)) and rec.get("role") == "new_evaluation_resource_records")
        run_sources = [rec for rec in files if Path(rec.get("path", "")).name == "A0_RUN_RECORDS.json"]
        if not run_sources:
            missing.append(f"seed {seed}: no bound A0_RUN_RECORDS.json")
        for rec in files:
            path = Path(rec.get("path", ""))
            if not path.is_file() or path.suffix.lower() != ".json" or not rec.get("new_artifact"):
                continue
            document = _json(path)
            if not isinstance(document, dict):
                continue
            prefix = ""
            payload = document
            if "a0" not in payload and isinstance(document.get("stages", {}).get("embed"), dict):
                prefix, payload = "/stages/embed", document["stages"]["embed"]
            if "a0" not in payload:
                continue
            entries = dict(run_fields) if path.name == "A0_RUN_RECORDS.json" else {}
            if "export_segments" in payload:
                entries[("pipeline_cost", "export_seconds")] = (prefix + "/export_segments", "duration_ns")
                for index, gate in enumerate(payload.get("gates", [])):
                    if gate.get("gate") == "G-F7d":
                        entries[("reproducibility_checks", "chunked_inference_agreement")] = (prefix + f"/gates/{index}", "passed_gate")
            if "prior_build_segments" in payload:
                entries[("pipeline_cost", "prior_build_seconds")] = ("/prior_build_segments", "duration_ns")
            if rec.get("role") == "new_evaluation_resource_records":
                entries.update({("pipeline_cost", "exact_full_reference_seconds"): ("/exact_segments", "duration_ns"),
                    ("pipeline_cost", "exact1000_reference_seconds"): ("/exact_segments", "duration_ns"),
                    ("pipeline_cost", "complete_evaluation_seconds"): ("", "evaluation_duration_ns"),
                    ("pipeline_cost", "evaluation_peak_rss_bytes"): ("", "evaluation_peak"),
                    ("pipeline_cost", "evaluation_peak_vram_bytes"): ("", "evaluation_peak")})
            for (scope, name), (pointer, operation) in entries.items():
                metric_id = f"{scope}.{name}.{seed}"
                try:
                    value = _pointer(document, pointer)
                except (KeyError, IndexError, TypeError):
                    missing.append(metric_id + ": missing raw pointer " + pointer)
                    continue
                if value is None:
                    missing.append(metric_id + ": raw value missing")
                    continue
                spec = {"id": metric_id, "path": str(path.resolve()), "sha256": rec["sha256"],
                        "pointer": pointer, "operation": operation, "envelope_pointer": prefix + "/a0"}
                if operation == "disabled":
                    spec["reason"] = "Frozen original skip_diagnostics=True; no disabled measurement replaced by zero"
                if operation == "evaluation_peak":
                    spec["field"] = "rss_bytes" if name == "evaluation_peak_rss_bytes" else "vram_bytes"
                sources.append(spec)
            if path.name == "A0_RUN_RECORDS.json" and isinstance(document.get("history"), list):
                for field in ("total", "rank", "contrast", "global"):
                    for suffix, operation in (("by_epoch", "epoch_field"), ("final_epoch", "epoch_final_field")):
                        sources.append({"id": f"training_records.{field}_loss_{suffix}.{seed}", "path": str(path.resolve()),
                                        "sha256": rec["sha256"], "pointer": "/history", "operation": operation, "field": field})
    return {"schema_version": "a0.record-sources.v1", "run_id": manifest["run_id"], "sources": sources,
            "missing_sources": missing, "note": "Only descriptors; no measured result values are supplied by this file."}


def _recursive_summary(values: list[Any], operation: str) -> Any:
    if all(isinstance(value, dict) for value in values):
        _require(all(set(value) == set(values[0]) for value in values), "Native record summary key mismatch")
        return {key: _recursive_summary([value[key] for value in values], operation) for key in values[0]}
    if all(isinstance(value, list) for value in values):
        _require(all(len(value) == len(values[0]) for value in values), "Native record summary length mismatch")
        return [_recursive_summary([value[i] for value in values], operation) for i in range(len(values[0]))]
    if all(value is None for value in values):
        return None
    if all(isinstance(value, (int, float, bool)) for value in values):
        array = np.asarray(values, dtype=float)
        _require(np.isfinite(array).all(), "Nonfinite summary input")
        return float({"mean": np.mean, "min": np.min, "max": np.max, "sum": np.sum}[operation](array))
    _require(all(value == values[0] for value in values), "Non-numeric records cannot be averaged")
    return values[0]


SOURCE_COUNT_NAMES = {"native_raw_metric_rows", "native_finite_metric_rows", "native_median_deduplicated_rows",
    "native_primary_edges_before_cap", "native_primary_edges_after_cap", "merged_input_rows",
    "merged_within_source_duplicates_removed", "merged_cross_source_conflicts", "merged_pairs_before_cap",
    "training_similar_to_edges", "categorical_vocab_cardinalities", "node_feature_shapes_and_dtypes",
    "model_feature_file_gb", "unknown_model_size_percent", "other_model_family_percent"}


def consume_source_counts(path: Path | None, protocol_path: Path, manifest: dict, values: dict, hashes: dict):
    """Consume a newly executed, source/hash/code-bound preprocessing recount."""
    if path is None:
        return
    report = _json(path)
    protocol = _json(protocol_path)
    _require(report.get("schema_version") == "a0.source_counts.v1", "Unknown source-count schema")
    _require(report.get("protocol_sha256") == sha256(protocol_path) and report.get("authority") == protocol["authority"],
             "Source-count protocol/authority differs")
    envelope = report.get("a0", {})
    _require(envelope == {"protocol": "a0", "run_id": manifest["run_id"], "scope": "frozen_input_audit"},
             "Source-count run identity differs")
    expected_implementation = Path(__file__).with_name("a0_source_counts.py").resolve()
    implementations = {str(Path(name).resolve()): digest for name, digest in report.get("implementation_sha256", {}).items()}
    _require(implementations.get(str(expected_implementation)) == sha256(expected_implementation), "Source-count producer implementation is unbound or changed")
    _require(bool(report.get("inputs")), "Source-count report lacks raw inputs")
    for rec in report["inputs"]:
        source = Path(rec["path"]).resolve()
        _require(sha256(source) == rec["sha256"], f"Source-count raw input changed: {source}")
        hashes[str(source)] = rec["sha256"]
    for source, expected in implementations.items():
        _require(sha256(Path(source)) == expected, "Source-count implementation changed")
        hashes[source] = expected
    for name, value in report.get("measurements", {}).items():
        _require(name in SOURCE_COUNT_NAMES, "Source-count adapter cannot supply events or retrieval scores")
        if name.startswith("merged_"):
            identity = report.get("merge_output_identity", {})
            _require(identity.get("passed") is True and set(identity.get("tables", [])) ==
                     {"supervision_merged.parquet", "dataset_nodes_merged.parquet", "supervision_conflicts.parquet"},
                     "Merge source counts lack all three canonical output identity gates")
        metric_id = f"frozen_input_audit.{name}.shared"
        _require(metric_id not in values, "Source-count value duplicates another source")
        components = {}
        if isinstance(value, dict) and all(key in value for key in ("value", "numerator", "denominator", "scale")):
            components = {key: value[key] for key in ("numerator", "denominator", "scale")}
            _require(value["denominator"] > 0 and value["value"] == value["numerator"] / value["denominator"] * value["scale"], "Invalid source-count ratio")
            value = value["value"]
        values[metric_id] = _measurement(value, status="verified")
        values[metric_id].update(components)
        values[metric_id]["source"] = f"{path.resolve()}#/measurements/{name}; newly executed source recount with mandatory canonical edge identity gate"
    for name, reason in report.get("missing", {}).items():
        metric_id = f"frozen_input_audit.{name}.shared"
        _require(metric_id not in values, "A source statistic cannot be both measured and missing")
        values[metric_id] = _measurement(None, status="missing", reason=reason)
    hashes[str(path.resolve())] = sha256(path)


def metric_source_coverage(inventory_path: Path, protocol_path: Path) -> dict:
    """Audit metric-to-producer mappings only; never calculate metric values."""
    inventory, protocol = _json(inventory_path), _json(protocol_path)
    frozen = {spec["id"]: spec for spec in frozen_audit_descriptors(protocol)}
    repo = Path(__file__).resolve().parents[1]
    historical_identity = repo / "docs/1M/A0_runs/A0_2/A0_HISTORICAL_SOURCE_IDENTITY.json"
    events = {
        "model_snapshot_api_pages": "Original per-request event stream with request identity, response/page boundaries and retry counting policy. Retained shards omit page boundaries; PROVENANCE counters cannot replace fresh counts.",
        "model_snapshot_skipped_duplicates": "Original per-discard duplicate event stream with discarded occurrence identity. Retained shards have already removed those occurrences; zero retained duplicates cannot recover skipped-duplicate count."}
    providers = {
        "hnsw1000_task_prior": ("recompute_seed/pool_recompute", "a0_hnsw_s{seed}.npz: query/root/observed/model/score/prior/top10/pool_counts/gold_position"),
        "exact1000_task_prior": ("recompute_seed/pool_recompute", "a0_exact_s{seed}.npz: same-query dense1000 raw pool; diagnostic of the same GD embeddings"),
        "exact_full_lake_task_prior": ("recompute_seed/full_recompute", "a0_exact_s{seed}.npz: observed/full_counts/full_top10; integer all-model comparison counts, top10 cross-check"),
        "retrieval_diagnostics": ("recompute_seed", "Independently recomputed same-seed full/exact1000/HNSW raw outputs; retain numerator and denominator before three-seed averaging"),
        "ann_calibration": ("calibration_recompute", "All attempted a0_calibration_s{seed}_ef{ef}.npz plus exact raw pool; first passing ef rule"),
        "retrieval_cost": ("recompute_seed", "a0_hnsw_s{seed}.npz hnsw_ns/rerank_ns/total_ns paired per-query samples"),
        "index_cost": ("recompute_seed", "Actual new hnsw_a0_s{seed}.bin stat and manifest index_build.start_ns/end_ns"),
        "training_records": ("make_record_template/supplemental_records", "New manifest-bound A0_RUN_RECORDS.json: native history[25], resolved_config, initialization_seed, epochs, mechanism_gate"),
        "reproducibility": ("make_record_template/supplemental_records", "New A0_RUN_RECORDS.json/runtime_environment; shared result preserves each seed environment"),
        "feature_repair_verification": ("a0_recompute_checks.recompute_checks", "Independent new graph file hashes, actual seven-column zeros and all retained graph content versus A0.1 input"),
        "task_prior": ("a0_recompute_checks.recompute_checks", "New sidecar edge/root/task/rowmap raw arrays, bound new graph/checkpoint/export and frozen source pairs/weights; raw pool prior equality"),
        "reproducibility_checks": ("a0_recompute_checks.recompute_checks", "Actual new checkpoint/graph/embedding/rowmap arrays and manifest hashes"),
    }
    entries = []
    for item in inventory["metrics"]:
        name, scope, seed, metric_id = item["name"], item["scope"], item["seed"], item["id"]
        entry = {"metric_id": metric_id, "required": item["required"], "scope": scope, "seed": seed,
                 "classification": "runtime_raw_records", "provider_implemented": True,
                 "measurement_status": "not_computed_in_A0.2", "manual_descriptor_required": False,
                 "old_values_used_as_new": False}
        if scope == "frozen_input_audit":
            if metric_id in frozen:
                entry.update(classification="existing_A0.1_bound_audit_descriptor", entry_point="frozen_audit_descriptors/supplemental_records",
                             raw_source=frozen[metric_id])
            elif name in SOURCE_COUNT_NAMES:
                entry.update(classification="deferred_frozen_source_recount", entry_point="scale1m.a0_source_counts.audit -> consume_source_counts",
                             raw_source="Frozen model snapshot gzip shards, A0.1-bound graph categorical/feature/edge arrays; native parser and deterministic original supervision rules")
                if name.startswith("merged_"):
                    entry["raw_source"] = "Five A0.2 identity-bound stage1BuildTransferGraph/*.pt + A0.1 frozen supervision_uncapped.parquet; mandatory all-column equality of rebuilt edge/node/conflict tables"
                    entry["identity_binding"] = str(historical_identity)
            else:
                _require(name in events, f"Frozen input provider missing: {metric_id}")
                entry.update(classification="missing_original_event_evidence", provider_implemented=False,
                             measurement_status="missing", entry_point=None, raw_source=events[name],
                             blocking_reason="Original event evidence unavailable; no retained-snapshot-only recount can supply this metric")
        elif scope == "pipeline_cost":
            if name.startswith("training_"):
                source = "A0_RUN_RECORDS.json: train_segments and OS/CUDA peak samples; new run/seed/graph/checkpoint envelope"
            elif name == "export_seconds":
                source = "EXPORT_MANIFEST.json/stages/embed/{a0,export_segments}; nested envelope validated"
            elif name == "prior_build_seconds":
                source = "New prior/meta.json/{a0,prior_build_segments}"
            else:
                source = "A0_EVAL_RECORDS_s{seed}.json: exact_segments/hnsw_segments, per-segment resource_peaks; incomplete endpoints/samples remain missing"
            entry.update(entry_point="make_record_template/supplemental_records", raw_source=source)
        else:
            _require(scope in providers, f"Metric provider missing: {metric_id}")
            point, source = providers[scope]
            entry.update(entry_point=point, raw_source=source)
        if scope == "reproducibility_checks":
            if name in ("final_output_candidate_membership", "fusion_and_tie_consistency"):
                entry.update(entry_point="recompute_seed/pool_recompute", raw_source="Independent actual raw-pool validity, fusion/tie ordering and top10 membership checks")
            elif name == "independent_metric_recomputation":
                entry.update(entry_point="_compare_native", raw_source="Hash-bound A0_EVALUATION_REPORT.json compared field-by-field to independent per-query recomputation")
            elif name == "chunked_inference_agreement":
                entry.update(entry_point="make_record_template/supplemental_records(passed_gate)", raw_source="New EXPORT_MANIFEST.json/stages/embed/gates G-F7d ran plus model/dataset raw max_abs_delta < 1e-5")
        if scope == "training_records" and name == "disabled_diagnostics":
            entry.update(classification="conditionally_disabled", status_rule="disabled only with bound resolved_config.skip_diagnostics=true; never zero")
        if name == "unavailable_full_lake_rank_fields":
            entry.update(classification="mathematically_undefined", status_rule="Bounded pool cannot establish all-lake rank fields; preserve undefined")
        if scope == "ann_calibration" and name.startswith("recall_at_1000_ef_"):
            entry["status_rule"] = "Attempted ef recomputed; grid values after first passing ef are not_applicable, never missing or measured zero"
        if seed not in (0, 1, 2, "shared"):
            per_seed_name = "queries" if name == "query_observations" else name
            entry.update(aggregate_entry_point="fill_summaries", dependencies=[f"{scope}.{per_seed_name}.{s}" for s in (0, 1, 2)],
                         aggregate_rule="All three seeds required; ratios are computed per seed before mean; missing and undefined are preserved")
        entries.append(entry)
    files = [Path(__file__), Path(__file__).with_name("a0_source_counts.py"), Path(__file__).with_name("a0_recompute_checks.py")]
    return {"schema_version": "a0.metric-source-coverage.v1", "stage": "A0.2", "authority": protocol["authority"],
        "scope": "Source/entry-point coverage audit only; no real training, retrieval or source-count execution and no new metrics computed",
        "inventory_sha256": sha256(inventory_path), "protocol_sha256": sha256(protocol_path),
        "implementation_sha256": {str(path.resolve()): sha256(path) for path in files if path.is_file()},
        "registered_items": len(entries), "new_metric_values_computed": 0,
        "classification_counts": {kind: sum(entry["classification"] == kind for entry in entries) for kind in sorted({entry["classification"] for entry in entries})},
        "missing_original_evidence_ids": [entry["metric_id"] for entry in entries if entry["classification"] == "missing_original_event_evidence"],
        "historical_source_identity": {"path": str(historical_identity), "sha256": sha256(historical_identity) if historical_identity.is_file() else None,
            "files": [str((repo / "stage1BuildTransferGraph" / name).resolve()) for name in
                ("hgraph_ml_v2.pt", "hgraph_d05_nocontent.pt", "hgraph_A_ctrl_2000m_xm0_xd0.pt", "hgraph_hf_effective_2000m_v2_xm0_xd0.pt", "hgraph_diverse_xd0.pt")],
            "limit": "Five files are present and currently hash-bound. Only ModelLens has a matched prior full SHA256; other four require the recorded mandatory complete merge-output identity gate before accepting counts."},
        "commands": {"source_recount": "python -m scale1m.a0_source_counts --protocol docs/1M/A0_runs/A0_PROTOCOL.json --historical-bindings docs/1M/A0_runs/A0_2/A0_HISTORICAL_SOURCE_IDENTITY.json --out <new-audit>/A0_SOURCE_COUNTS.json --run-id A0_20260912",
            "make_descriptors": "python -m scale1m.recompute_a0 --raw <new-eval> --protocol docs/1M/A0_runs/A0_PROTOCOL.json --inventory docs/1M/A0_runs/A0_METRIC_INVENTORY.json --out <independent> --make-record-template <independent>/A0_RECORD_SOURCES.json",
            "recompute": "python -m scale1m.recompute_a0 --raw <new-eval> --protocol docs/1M/A0_runs/A0_PROTOCOL.json --inventory docs/1M/A0_runs/A0_METRIC_INVENTORY.json --out <independent> --records <independent>/A0_RECORD_SOURCES.json --source-counts <new-audit>/A0_SOURCE_COUNTS.json"},
        "completeness_rule": "A0.2 checker coverage is not A0.7 measured completeness. Missing original event evidence keeps both source recount and final all-required completeness nonzero; no old value is substituted.",
        "metrics": entries}


def fill_summaries(inventory: dict, values: dict, seeds: list[int]) -> None:
    for item in inventory["metrics"]:
        metric_id, aggregate = item["id"], item["seed"]
        if aggregate == "shared" and item["scope"] == "reproducibility" and item["name"] == "runtime_environment" and metric_id not in values:
            records = [values.get(f"reproducibility.runtime_environment.{seed}") for seed in seeds]
            if all(record and record["status"] in ("verified", "recomputed") for record in records):
                values[metric_id] = _measurement({str(seed): record["value"] for seed, record in zip(seeds, records)}, status="verified")
        if metric_id in values or isinstance(aggregate, int) or aggregate == "shared":
            continue
        name = "queries" if item["name"] == "query_observations" else item["name"]
        inputs = [values.get(f"{item['scope']}.{name}.{seed}") for seed in seeds]
        units = {value["unit"] for value in inputs if value and value.get("unit") is not None}
        _require(len(units) <= 1, f"Mixed raw measurement units for {metric_id}: {sorted(units)}")
        unit = next(iter(units), None)
        if any(value is None or value["status"] in ("missing", "pending") for value in inputs):
            values[metric_id] = _measurement(None, status="missing", reason="All three seed inputs are required for this summary", unit=unit)
            continue
        if any(value["status"] == "undefined" for value in inputs):
            values[metric_id] = _measurement(None, status="undefined", reason="At least one seed is mathematically undefined; no skip-missing aggregation",
                                             numerator=[value.get("numerator") for value in inputs],
                                             denominator=[value.get("denominator") for value in inputs], unit=unit)
            continue
        raw = [value["value"] for value in inputs]
        if item["name"] == "retention_numerators_denominators":
            result = {str(seed): value for seed, value in zip(seeds, raw)}
        elif aggregate == "vector":
            result = raw
        elif aggregate == "all":
            _require(all(isinstance(value, bool) for value in raw), "Boolean summary expected")
            result = all(raw)
        else:
            _require(aggregate in ("mean", "min", "max", "sum"), f"Unknown aggregation: {aggregate}")
            result = _recursive_summary(raw, aggregate)
        values[metric_id] = _measurement(result, unit=unit)


def expand_native_inventory(inventory: dict, native: dict[int, dict], values: dict) -> None:
    """Register each discovered scorer field explicitly, beyond object placeholders."""
    known = {item["id"] for item in inventory["metrics"]}
    for seed, scopes in native.items():
        for scope, row in scopes.items():
            parent = next(item for item in inventory["metrics"] if item["scope"] == scope and item["name"] == "extra_original_scorer_fields")
            for key, value in row.items():
                for target in (seed, "mean", "min", "max"):
                    name = "native_" + key
                    metric_id = f"{scope}.{name}.{target}"
                    if metric_id not in known:
                        item = copy.deepcopy(parent)
                        item.update(id=metric_id, name=name, seed=target,
                                    definition=f"Explicit independent original scorer field {key}",
                                    old_value=None, old_reference_id=None, old_source_section=None, old_display_precision=None,
                                    new_value=None, status="pending", note="Expanded from original scorer schema; not an additional historical fact.")
                        item["old_source"]["status"] = "not_reported"
                        inventory["metrics"].append(item)
                        known.add(metric_id)
                    if target == seed:
                        _add(values, scope, seed, name, value, status="undefined" if value is None else "recomputed",
                             reason="Native bounded-pool rank is undefined" if value is None else None)
    for seed in (0, 1, 2):
        record = values.get(f"training_records.all_existing_epoch_fields.{seed}")
        if record is None or record["status"] not in ("verified", "recomputed"):
            continue
        history = record["value"]
        _require(isinstance(history, list) and len(history) == 25, "Incomplete epoch record expansion")
        parent = next(item for item in inventory["metrics"] if item["id"] == f"training_records.all_existing_epoch_fields.{seed}")
        fields = set().union(*(row.keys() for row in history))
        for field in sorted(fields):
            _require(all(field in row for row in history), f"Native epoch field missing during training: {field}")
            series = [row[field] for row in history]
            name = "native_epoch_" + field
            for target in (seed, "mean"):
                metric_id = f"training_records.{name}.{target}"
                if metric_id not in known:
                    item = copy.deepcopy(parent)
                    item.update(id=metric_id, name=name, seed=target, definition=f"Every native epoch value of {field}",
                                old_value=None, old_reference_id=None, old_source_section=None, old_display_precision=None,
                                new_value=None, status="pending", note="Explicit native epoch-field expansion; all 25 records retained.")
                    item["old_source"]["status"] = "not_reported"
                    inventory["metrics"].append(item)
                    known.add(metric_id)
                if target == seed:
                    _add(values, "training_records", seed, name, series)


def _value_in_inventory_unit(item: dict, record: dict) -> tuple[Any, str | None, float]:
    """Convert reporting units once; raw measurements retain their native units."""
    declared_unit = item.get("unit")
    raw_unit = record.get("unit", declared_unit)
    value = record["value"]
    if raw_unit == declared_unit:
        return value, raw_unit, 1.0
    _require(raw_unit == "fraction" and declared_unit == "percent",
             f"Unsupported unit conversion for {item['id']}: {raw_unit} -> {declared_unit}")
    if value is None:
        return None, raw_unit, 100.0
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"Fraction-to-percent requires a scalar numeric value: {item['id']}")
    return value * 100.0, raw_unit, 100.0


def apply_inventory(inventory: dict, values: dict, hashes: dict) -> tuple[dict, dict]:
    output = copy.deepcopy(inventory)
    missing, invalid = [], []
    final_statuses = {"recomputed", "verified", "undefined", "disabled", "not_applicable"}
    seen = set()
    for item in output["metrics"]:
        metric_id = item["id"]
        _require(metric_id not in seen, f"Duplicate inventory ID: {metric_id}")
        seen.add(metric_id)
        record = values.get(metric_id, _measurement(None, status="missing", reason="No bound A0 raw measurement supplied"))
        new_value, raw_unit, scale = _value_in_inventory_unit(item, record)
        if metric_id in values and item.get("unit") == "percent":
            record.setdefault("unit", raw_unit)
        item.update(new_value=new_value, status=record["status"],
                    new_unit=item.get("unit"), raw_measurement_unit=raw_unit,
                    raw_to_inventory_scale=scale,
                    recomputed_from=[record.get("source", "Independent A0 raw-record recomputation")] if record["status"] not in ("missing", "pending") else [],
                    artifact_hashes=hashes if record["status"] not in ("missing", "pending") else {},
                    new_reason=record.get("reason"), new_numerator=record.get("numerator"), new_denominator=record.get("denominator"))
        if item.get("required") and item["status"] not in final_statuses:
            missing.append(metric_id)
        if item["status"] in ("undefined", "disabled", "not_applicable") and not item.get("new_reason"):
            invalid.append(metric_id)
        if item["status"] in ("recomputed", "verified") and item["new_value"] is None:
            invalid.append(metric_id)
        if item.get("category") == "correctness_gate" and item["status"] in final_statuses and item["new_value"] is not True:
            if not (isinstance(item["new_value"], dict) and item["new_value"].get("passed") is True):
                invalid.append(metric_id)
    completeness = {"complete": not missing and not invalid, "inventory_items": len(output["metrics"]),
                    "required_items": sum(bool(item.get("required")) for item in output["metrics"]),
                    "missing": missing, "invalid": invalid,
                    "status_counts": {status: sum(item["status"] == status for item in output["metrics"])
                                      for status in sorted({item["status"] for item in output["metrics"]})}}
    output.update(stage="A0.7", status="complete" if completeness["complete"] else "incomplete", completeness=completeness)
    return output, completeness


def _comparison(item: dict) -> dict:
    old, new = item.get("old_value"), item.get("new_value")
    numeric = isinstance(old, (int, float)) and isinstance(new, (int, float))
    return {"id": item["id"], "old": old, "new": new, "status": item["status"],
            "unit": item.get("unit"),
            "difference_unit": "percentage_points" if item.get("unit") == "percent" else item.get("unit"),
            "old_source_section": item.get("old_source_section"), "old_display_precision": item.get("old_display_precision"),
            "difference_from_published_value": new - old if numeric else None,
            "comparison_note": "Difference uses published old precision; no additional old digits inferred" if numeric else "Old value not reported or new value unavailable"}


def _display(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value).replace("|", "\\|")


def render_results(report: dict, inventory: dict) -> str:
    lookup = {item["id"]: item for item in inventory["metrics"]}
    lines = ["# A0 独立复算结果", "", "唯一最终系统：X4G+D → HNSW top-1000 → task prior → top-10。", "",
             "状态：" + ("完整性检查通过。" if report["completeness"]["complete"] else "未完成；缺测项保留为空，不能据此宣称 A0 全流程完成。"), "",
             "旧值只来自已冻结的英文事实文档。新值从 A0 原始记录复算；保留旧值公开精度。", ""]
    for title, scope, names in [
        ("最终系统", FINAL, list(QUALITY) + ["median_gold_position_when_retrieved", "actual_hnsw_gold_coverage_at_1000"]),
        ("Exact top-1000 参照", EXACT, ["gold_at_1", "gold_at_10", "top3_at_10"]),
        ("Exact full-lake 参照", FULL, ["gold_at_1", "gold_at_10", "top3_at_10"]),
        ("保留率与覆盖", "retrieval_diagnostics", ["exact_pool_retention", "ann_retention", "overall_retention", "exact_pool_gold_coverage_at_1000", "full_fused_top10_in_exact_pool"]),
        ("检索耗时（ms）", "retrieval_cost", ["hnsw_latency_p50_ms", "hnsw_latency_p95_ms", "prior_rerank_latency_p50_ms", "prior_rerank_latency_p95_ms", "total_latency_p50_ms", "total_latency_p95_ms"]),
    ]:
        lines.extend([f"## {title}", "", "| 指标 | seed 0 | seed 1 | seed 2 | 新均值 | 旧均值 |", "|---|---:|---:|---:|---:|---:|"])
        for name in names:
            items = [lookup.get(f"{scope}.{name}.{s}", {}) for s in (0, 1, 2, "mean")]
            unit = next((item.get("unit") for item in items if item.get("unit")), None)
            label = name + (" (%)" if unit == "percent" else "")
            lines.append("| " + " | ".join([label] + [_display(item.get("new_value")) for item in items] + [_display(items[-1].get("old_value"))]) + " |")
        lines.append("")
    lines.extend(["## 完整性与解释边界", "",
                  f"清单共 {report['completeness']['inventory_items']} 项；缺测 {len(report['completeness']['missing'])} 项，无效 {len(report['completeness']['invalid'])} 项。",
                  "三 seed 汇总只在三者齐全时生成；保留率先逐 seed 相除再平均。", "",
                  "标注 (%) 的行以百分数显示新旧值，差值单位为百分点；JSON new_measurements 中的原始比例显式标注 unit=fraction。", "",
                  "整湖排名统计使用原始全模型比较计数，质量另与 full-top10 核对；本复算不重新执行三百万模型打分。",
                  "在线延迟从预计算查询向量开始；合计分位数来自逐查询合计耗时。索引字节是磁盘大小。", "",
                  "本评估仍限于已存在节点、性能边留出的历史排名恢复。", ""])
    if report["completeness"]["missing"]:
        lines.extend(["缺测项目（完整原因见 inventory）：", ""] + ["- `" + metric_id + "`" for metric_id in report["completeness"]["missing"]])
    return "\n".join(lines) + "\n"


def recompute(raw: Path, protocol_path: Path, inventory_path: Path, out: Path,
              records_path: Path | None = None, source_counts_path: Path | None = None) -> dict:
    raw, out = raw.resolve(), out.resolve()
    _require(out != raw and not raw.is_relative_to(out), "Independent output directory must not replace raw directory")
    _require(out != inventory_path.resolve().parent, "Do not overwrite A0.1 frozen inventory directory")
    inventory, protocol = _json(inventory_path), _json(protocol_path)
    original_count = len(inventory["metrics"])
    manifest, hashes = verify_manifest(raw, protocol_path, inventory)
    if (out / "A0_REPORT.json").exists():
        previous = _json(out / "A0_REPORT.json")
        _require(previous.get("run_id") == manifest["run_id"] and previous.get("protocol_sha256") == sha256(protocol_path),
                 "Independent output directory belongs to a different A0 run/protocol")
    hashes[str(inventory_path.resolve())] = sha256(inventory_path)
    seeds = [int(item["split_seed"]) for item in protocol["official_runs"]]
    _require(seeds == [0, 1, 2], "A0 requires split seeds 0, 1 and 2")
    values, native, absent_seeds = {}, {}, []
    for seed in seeds:
        meta = manifest["seeds"].get(str(seed), {})
        if not all(key in meta for key in ("exact", "hnsw", "calibration", "index", "index_build", "selected_ef", "calibration_passed")):
            absent_seeds.append(seed)
            continue
        measured, rows = recompute_seed(raw, manifest, protocol, seed)
        values.update(measured)
        native[seed] = rows
    evaluator_path = raw / "A0_EVALUATION_REPORT.json"
    if evaluator_path.is_file():
        report_hash = sha256(evaluator_path)
        _require("A0_EVALUATION_REPORT.json" in manifest["artifacts"], "Evaluator report is not hash-bound")
        hashes[str(evaluator_path)] = report_hash
        _compare_native(_json(evaluator_path), native, values)
    from .a0_recompute_checks import recompute_checks
    checked_values = recompute_checks(manifest, protocol)
    _require(not set(checked_values).intersection(values), "Independent file checks duplicate retrieval measurements")
    values.update(checked_values)
    supplemental_records(records_path, hashes, values, inventory, manifest, protocol)
    consume_source_counts(source_counts_path, protocol_path, manifest, values, hashes)
    expand_native_inventory(inventory, native, values)
    fill_summaries(inventory, values, seeds)
    reader_files = [Path(__file__).resolve(), Path(__file__).with_name("a0_recompute_checks.py").resolve()]
    reader_code = {str(path): sha256(path) for path in reader_files}
    hashes.update(reader_code)
    updated, completeness = apply_inventory(inventory, values, hashes)
    completeness.update(original_registered_items=original_count, missing_seeds=absent_seeds)
    if absent_seeds:
        completeness["complete"] = False
    protocol_fidelity = {str(seed): values.get(f"ann_calibration.calibration_passed.{seed}", {}).get("value") for seed in seeds}
    report = {"schema_version": "a0.recomputed.v1", "stage": "A0.7", "run_id": manifest["run_id"],
              "sole_final_system": FINAL, "authority": copy.deepcopy(protocol["authority"]),
              "protocol_sha256": hashes[str(protocol_path.resolve())], "completeness": completeness,
              "independent_reader": {"entrypoint": "scale1m.recompute_a0", "code_sha256": reader_code,
                  "unit_contract": "new_measurements retain explicitly tagged native fractions; inventory new_value and comparisons use the declared inventory unit; percent differences are percentage points"},
              "ann_fidelity_per_seed": protocol_fidelity,
              "all_ann_fidelity_passed": all(value is True for value in protocol_fidelity.values()),
              "native_recomputed": {str(seed): rows for seed, rows in native.items()},
              "new_measurements": values, "old_reference": [{key: item.get(key) for key in
                  ("id", "old_value", "old_source_section", "old_display_precision", "old_source")} for item in updated["metrics"]],
              "comparisons": [_comparison(item) for item in updated["metrics"]], "artifact_hashes": hashes,
              "independence_boundary": "Pools independently fused and sorted; gold labels independently derived; full-lake rank summaries use saved raw integer comparison counts, checked against independently scored full top-ten IDs; timing from per-query ns."}
    out.mkdir(parents=True, exist_ok=True)
    for name, document in (("A0_REPORT.json", report), ("A0_METRIC_INVENTORY.json", updated)):
        (out / name).write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (out / "A0_RESULTS.md").write_text(render_results(report, updated), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="A0 evaluator output directory")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Separate independent recomputation directory")
    parser.add_argument("--records", type=Path, help="Optional hash-bound raw cost/log/audit extraction descriptors")
    parser.add_argument("--source-counts", type=Path, help="New hash-bound a0_source_counts report; missing event evidence remains missing")
    parser.add_argument("--make-record-template", type=Path, help="Write descriptors for available native record envelopes, then exit without recomputing")
    args = parser.parse_args(argv)
    try:
        if args.make_record_template:
            manifest, _ = verify_manifest(args.raw, args.protocol, _json(args.inventory))
            target = args.make_record_template.resolve()
            _require(target.parent != args.inventory.resolve().parent and
                     target not in (args.protocol.resolve(), args.inventory.resolve(), args.raw.resolve() / "A0_EVALUATION_MANIFEST.json"),
                     "Record template must not overwrite a frozen input or manifest")
            template = make_record_template(manifest, _json(args.protocol))
            args.make_record_template.parent.mkdir(parents=True, exist_ok=True)
            args.make_record_template.write_text(json.dumps(template, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
            print(f"A0 raw-record descriptors: {args.make_record_template}")
            return 0
        report = recompute(args.raw, args.protocol, args.inventory, args.out, args.records, args.source_counts)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"A0 integrity failure: {exc}")
        return 1
    complete = report["completeness"]["complete"]
    print(f"A0 recomputation {'complete' if complete else 'incomplete'}: {args.out / 'A0_REPORT.json'}")
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
