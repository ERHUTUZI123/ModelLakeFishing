import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO.parent) not in sys.path:
    sys.path.insert(0, str(REPO.parent))

from ModelLakeFishing.scale1m import recompute_a0 as R


def _pack(seed=0, reverse=False):
    q = np.array([101, 102, 103], dtype=np.int64)
    models = np.tile(np.arange(12, dtype=np.int64), (3, 1))
    score = np.tile(np.linspace(0.9, -0.9, 12, dtype=np.float32), (3, 1))
    prior = np.zeros_like(score)
    fused = (score + np.float32(1)) * np.float32(0.5) + prior
    observed = [(np.array([23, 20, 19]), np.array([1.0, 0.7, 0.4])),
                (np.array([0, 11, 10]), np.array([1.0, 0.995, 0.4])),
                (np.array([1, 11, 10]), np.array([1.0, 0.7, 0.4]))]
    gold = [23, 0, 1]
    ranked = np.array([sorted(row.tolist(), key=lambda m: (-float(fused[i, m]),
                       (m * 11400714819323198485 + 0xD1B54A32D192ED03) % 2**64))
                       for i, row in enumerate(models)], dtype=np.int64)
    counts, full_counts = [], []
    for i, ((ids, perf), row) in enumerate(zip(observed, ranked)):
        probes = [[gold[i]], ids[np.argsort(-perf)[:3]], ids[perf >= perf.max() - 0.01]]
        pos = {m: j for j, m in enumerate(row)}
        counts.append([min(pos.get(int(m), 12) for m in probe) for probe in probes])
        full_counts.append([min(int(m) for m in probe) for probe in probes])
    pack = dict(seed=np.array(seed), n_models=np.array(24), query=q,
                root=np.array(["same", "same", "other"]), task_id=np.array([0, 0, 0]),
                observed_offsets=np.array([0, 3, 6, 9]), observed_ids=np.concatenate([p[0] for p in observed]),
                observed_values=np.concatenate([p[1] for p in observed]), model=models, score=score,
                prior=prior, fused=fused, top10=ranked[:, :10], pool_counts=np.array(counts),
                gold_position=np.array([0, 1, 2]), full_top10=np.tile(np.arange(10), (3, 1)),
                full_counts=np.array(full_counts), dense_counts=np.array(full_counts), selected_ef=np.array(1000),
                calibration_ids=models.copy(), hnsw_ns=np.array([1, 100, 1], dtype=np.int64) * 1000000,
                rerank_ns=np.array([100, 1, 1], dtype=np.int64) * 1000000,
                total_ns=np.array([101, 101, 2], dtype=np.int64) * 1000000)
    if reverse:
        order = np.array([2, 1, 0])
        for name, value in list(pack.items()):
            if value.ndim and value.shape[0] == 3:
                pack[name] = value[order]
        pack["observed_ids"] = np.concatenate([observed[i][0] for i in order])
        pack["observed_values"] = np.concatenate([observed[i][1] for i in order])
    return pack


def _item(scope, name, seed, **kwargs):
    item = dict(id=f"{scope}.{name}.{seed}", scope=scope, name=name, seed=seed,
                definition="fixture measurement", unit="fraction", aggregation="fixture",
                old_value=None, old_source_section=None, old_display_precision=None,
                old_reference_id=None, old_source={"status": "not_reported"}, new_value=None,
                recompute_from=["raw fixture"], recomputed_from=[], artifact_hashes={}, status="pending",
                required=True, category="measurement", expected_final_status="recomputed")
    item.update(kwargs)
    return item


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    from ModelLakeFishing.scale1m import a0_recompute_checks as checks
    monkeypatch.setattr(checks, "recompute_checks", lambda manifest, protocol: {})
    raw = tmp_path / "raw"
    raw.mkdir()
    authority = tmp_path / "evidence.md"
    authority.write_text("Fixture authority, no historical measurements.\n", encoding="utf-8")
    protocol = {"authority": {"path": str(authority), "sha256": R.sha256(authority)},
                "official_runs": [{"split_seed": s} for s in range(3)],
                "frozen_data": {"candidate_models": 24},
                "evaluation": {"hnsw": {"K": 12}, "fusion": {"return_k": 10},
                               "expected_query_counts": {str(s): 3 for s in range(3)},
                               "ef_calibration": {"grid": [1000, 1500, 2000, 3000, 5000], "threshold": 0.99}}}
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    metrics = []
    for scope in (R.FINAL, R.EXACT, R.FULL):
        for name in ("gold_at_10", "extra_original_scorer_fields"):
            for seed in (0, 1, 2, "mean"):
                metrics.append(_item(scope, name, seed))
    metrics.extend([_item("retrieval_diagnostics", "overall_retention", s) for s in (0, 1, 2, "mean")])
    metrics.extend([_item("pipeline_cost", "training_peak_rss_bytes", s) for s in (0, 1, 2)])
    inventory = {"authority": protocol["authority"], "metrics": metrics}
    inventory_path = tmp_path / "frozen" / "inventory.json"
    inventory_path.parent.mkdir()
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    binding = {"protocol_sha256": R.sha256(protocol_path), "N": 24, "K": 12, "return_k": 10,
               "old_graph_digest": "old", "seed_inputs": {"0": {"graph_digest": "new"}}, "source_files": [], "code_files": []}
    manifest = {"schema_version": "a0.raw.v1", "protocol": "a0", "run_id": "fixture-A0",
                "binding": binding, "binding_sha256": R._canonical(binding), "artifacts": {}, "seeds": {}}
    def save(name, pack=None, role="fixture", seed=0):
        path = raw / name
        if pack is not None:
            np.savez(path, **pack)
        elif not path.exists():
            path.write_bytes(b"new index fixture")
        manifest["artifacts"][name] = {"sha256": R.sha256(path), "bytes": path.stat().st_size,
                                        "role": role, "seed": seed, "new_artifact": True}
    exact = _pack()
    hnsw = _pack(reverse=True)
    save("a0_exact_s0.npz", exact)
    save("a0_hnsw_s0.npz", hnsw)
    save("a0_calibration_s0_ef1000.npz", {"query": exact["query"], "model": exact["model"],
                                         "score": exact["score"], "recall_per_query": np.ones(3)})
    save("hnsw_a0_s0.bin")
    manifest["seeds"]["0"] = {"exact": "a0_exact_s0.npz", "hnsw": "a0_hnsw_s0.npz",
                                "calibration": ["a0_calibration_s0_ef1000.npz"], "index": "hnsw_a0_s0.bin",
                                "index_build": {"start_ns": 1000, "end_ns": 2000001000, "build_seconds": 2.0},
                                "selected_ef": 1000, "calibration_passed": True}
    def flush():
        manifest["binding_sha256"] = R._canonical(manifest["binding"])
        (raw / "A0_EVALUATION_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    flush()
    return dict(raw=raw, protocol=protocol, protocol_path=protocol_path, inventory=inventory,
                inventory_path=inventory_path, manifest=manifest, save=save, flush=flush, out=tmp_path / "independent")


def test_independent_pool_alignment_misses_and_root_macro():
    labels = R.labels_from_pack(_pack(), 24)
    row, successes, detail = R.pool_recompute(_pack(reverse=True), labels, 24, 12, 10)
    assert row["gold@10"] == 2 / 3
    assert row["gold@1"] == 1 / 3
    assert row["root_gold@10"] == 0.75
    assert row["root_gold@1"] == 0.25
    assert successes["gold@10"] == 2
    assert detail["gold_retrieved_query_count"] == 2
    assert row["median_gold_rank_if_retrieved"] == 1.5
    assert row["median_gold_rank"] is None


@pytest.mark.parametrize("field", ["fused", "top10", "pool_counts", "gold_position"])
def test_pool_raw_corruption_rejected(field):
    pack = _pack()
    pack[field].flat[0] += 1
    with pytest.raises(R.IntegrityError):
        R.pool_recompute(pack, R.labels_from_pack(_pack(), 24), 24, 12, 10)


def test_full_lake_quality_checks_saved_counts_against_topten():
    pack = _pack()
    row, _ = R.full_recompute(pack, R.labels_from_pack(pack, 24), 24, 10)
    assert row["gold@10"] == 2 / 3
    assert row["median_gold_rank"] == 2.0
    pack["full_counts"][0, 0] = 0
    with pytest.raises(R.IntegrityError, match="top-ten"):
        R.full_recompute(pack, R.labels_from_pack(pack, 24), 24, 10)


def test_file_and_binding_hash_failures_are_blocking(bundle):
    b = bundle
    R.verify_manifest(b["raw"], b["protocol_path"], b["inventory"])
    (b["raw"] / "hnsw_a0_s0.bin").write_bytes(b"tampered")
    with pytest.raises(R.IntegrityError, match="SHA256"):
        R.verify_manifest(b["raw"], b["protocol_path"], b["inventory"])


def test_old_graph_and_old_artifact_cannot_be_relabelled(bundle):
    b = bundle
    b["manifest"]["binding"]["seed_inputs"]["0"]["graph_digest"] = "old"
    b["flush"]()
    with pytest.raises(R.IntegrityError, match="Old graph"):
        R.verify_manifest(b["raw"], b["protocol_path"], b["inventory"])
    b["manifest"]["binding"]["seed_inputs"]["0"]["graph_digest"] = "new"
    b["manifest"]["artifacts"]["a0_hnsw_s0.npz"]["new_artifact"] = False
    b["flush"]()
    with pytest.raises(R.IntegrityError, match="Old artifact"):
        R.verify_manifest(b["raw"], b["protocol_path"], b["inventory"])


def test_seed_raw_metrics_and_paired_timing_percentiles(bundle):
    b = bundle
    values, _ = R.recompute_seed(b["raw"], b["manifest"], b["protocol"], 0)
    assert values["retrieval_cost.total_latency_p50_ms.0"]["value"] == 101
    assert values["retrieval_cost.hnsw_latency_p50_ms.0"]["value"] == 1
    assert values["retrieval_cost.prior_rerank_latency_p50_ms.0"]["value"] == 1
    assert values["ann_calibration.recall_at_1000_ef_1500.0"]["status"] == "not_applicable"
    assert values["index_cost.build_seconds.0"]["value"] == 2


def test_mean_of_seed_ratios_and_undefined_are_explicit():
    inventory = {"metrics": [_item("retrieval_diagnostics", "overall_retention", "mean")]}
    pairs = [(0.1, 0.2), (0.5, 1.0), (0.9, 1.0)]
    values = {f"retrieval_diagnostics.overall_retention.{s}": R._ratio(*pair) for s, pair in enumerate(pairs)}
    R.fill_summaries(inventory, values, [0, 1, 2])
    result = values["retrieval_diagnostics.overall_retention.mean"]["value"]
    assert result == pytest.approx((0.5 + 0.5 + 0.9) / 3)
    assert result != pytest.approx(sum(x for x, _ in pairs) / sum(y for _, y in pairs))
    values["retrieval_diagnostics.overall_retention.1"] = R._ratio(0.1, 0)
    del values["retrieval_diagnostics.overall_retention.mean"]
    R.fill_summaries(inventory, values, [0, 1, 2])
    row = values["retrieval_diagnostics.overall_retention.mean"]
    assert row["status"] == "undefined" and row["value"] is None
    assert row["denominator"] == [0.2, 0, 1.0]


def test_partial_run_writes_missing_inventory_without_fake_means(bundle):
    b = bundle
    original = b["inventory_path"].read_bytes()
    result = R.recompute(b["raw"], b["protocol_path"], b["inventory_path"], b["out"])
    assert result["completeness"]["missing_seeds"] == [1, 2]
    assert result["new_measurements"][f"{R.FINAL}.gold_at_10.mean"]["status"] == "missing"
    assert result["new_measurements"][f"{R.FINAL}.gold_at_10.mean"]["value"] is None
    assert not result["completeness"]["complete"]
    assert "pipeline_cost.training_peak_rss_bytes.0" in result["completeness"]["missing"]
    assert b["inventory_path"].read_bytes() == original
    assert (b["out"] / "A0_RESULTS.md").is_file()
    assert result["completeness"]["inventory_items"] > result["completeness"]["original_registered_items"]
    assert R.main(["--raw", str(b["raw"]), "--protocol", str(b["protocol_path"]),
                   "--inventory", str(b["inventory_path"]), "--out", str(b["out"])]) == 2


def test_inventory_preserves_old_precision_and_distinguishes_states():
    metric = _item(R.FINAL, "gold_at_10", 0, old_value=0.3279, old_display_precision="4 decimal places")
    inv = {"metrics": [metric, _item("pipeline_cost", "training_peak_rss_bytes", 0),
                       _item("training_records", "disabled_diagnostics", 0)]}
    values = {metric["id"]: R._measurement(0.123456789),
              "training_records.disabled_diagnostics.0": R._measurement({"skip_diagnostics": True}, status="disabled", reason="Frozen policy")}
    updated, complete = R.apply_inventory(inv, values, {"raw": "hash"})
    assert updated["metrics"][0]["old_value"] == 0.3279
    assert updated["metrics"][0]["new_value"] == 0.123456789
    assert updated["metrics"][1]["status"] == "missing"
    assert updated["metrics"][2]["status"] == "disabled"
    assert not complete["complete"]


def test_full_618_inventory_missing_never_becomes_recomputed():
    inventory = json.loads((REPO / "docs/1M/A0_runs/A0_METRIC_INVENTORY.json").read_text(encoding="utf-8"))
    updated, complete = R.apply_inventory(inventory, {}, {})
    assert complete["inventory_items"] == 618
    assert complete["status_counts"] == {"missing": 618}
    assert all(item["new_value"] is None and item["recomputed_from"] == [] for item in updated["metrics"])


def test_frozen_percent_retention_uses_mean_of_ratios_and_percentage_point_difference():
    frozen = json.loads((REPO / "docs/1M/A0_runs/A0_METRIC_INVENTORY.json").read_text(encoding="utf-8"))
    metric = copy.deepcopy(next(item for item in frozen["metrics"]
                                if item["id"] == "retrieval_diagnostics.exact_pool_retention.mean"))
    assert metric["unit"] == "percent" and metric["old_value"] == 94.32
    inventory = {"metrics": [metric]}
    original = copy.deepcopy(inventory)
    pairs = [(0.1, 0.2), (0.5, 1.0), (0.9, 1.0)]
    values = {f"retrieval_diagnostics.exact_pool_retention.{seed}": R._ratio(*pair)
              for seed, pair in enumerate(pairs)}
    R.fill_summaries(inventory, values, [0, 1, 2])
    raw = values[metric["id"]]
    expected_fraction = (0.5 + 0.5 + 0.9) / 3
    assert raw["value"] == pytest.approx(expected_fraction)
    assert raw["unit"] == "fraction"
    updated, completeness = R.apply_inventory(inventory, values, {})
    item = updated["metrics"][0]
    assert completeness["complete"]
    assert item["new_value"] == pytest.approx(100 * expected_fraction)
    assert item["new_value"] != pytest.approx(100 * sum(x for x, _ in pairs) / sum(y for _, y in pairs))
    assert item["new_unit"] == "percent" and item["raw_measurement_unit"] == "fraction"
    assert item["raw_to_inventory_scale"] == 100
    comparison = R._comparison(item)
    assert comparison["difference_from_published_value"] == pytest.approx(100 * expected_fraction - 94.32)
    assert comparison["unit"] == "percent" and comparison["difference_unit"] == "percentage_points"
    assert inventory == original and values[metric["id"]]["value"] == expected_fraction


@pytest.mark.parametrize("explicit_unit", [None, "percent"])
def test_source_percentage_already_scaled_is_not_converted_again(explicit_unit):
    metric = _item("frozen_input_audit", "unknown_model_size_percent", "shared", unit="percent", old_value=71.929)
    record = R._measurement(71.92938428391888, unit=explicit_unit)
    updated, _ = R.apply_inventory({"metrics": [metric]}, {metric["id"]: record}, {})
    item = updated["metrics"][0]
    assert item["new_value"] == 71.92938428391888
    assert item["raw_to_inventory_scale"] == 1
    assert record["unit"] == "percent"


@pytest.mark.parametrize("status,reason", [("missing", "No raw source"), ("undefined", "zero denominator")])
def test_percent_boundary_preserves_null_and_status(status, reason):
    metric = _item("retrieval_diagnostics", "overall_retention", 0, unit="percent", old_value=94.25)
    record = R._measurement(None, status=status, reason=reason, unit="fraction")
    updated, completeness = R.apply_inventory({"metrics": [metric]}, {metric["id"]: record}, {})
    item = updated["metrics"][0]
    assert item["new_value"] is None and item["status"] == status and item["new_reason"] == reason
    assert not completeness["invalid"]
    assert bool(completeness["missing"]) == (status == "missing")
    assert R._comparison(item)["difference_from_published_value"] is None


def test_raw_pool_fixture_to_percent_inventory_and_reader_provenance(bundle):
    b = bundle
    frozen = json.loads((REPO / "docs/1M/A0_runs/A0_METRIC_INVENTORY.json").read_text(encoding="utf-8"))
    names = {"actual_hnsw_gold_coverage_at_1000", "exact_pool_retention", "ann_retention",
             "overall_retention", "exact_pool_gold_coverage_at_1000", "full_fused_top10_in_exact_pool"}
    originals = [copy.deepcopy(item) for item in frozen["metrics"] if item["name"] in names and item["seed"] == 0]
    assert len(originals) == 6 and all(item["unit"] == "percent" for item in originals)
    replace_ids = {item["id"] for item in originals}
    b["inventory"]["metrics"] = [item for item in b["inventory"]["metrics"] if item["id"] not in replace_ids] + originals
    b["inventory_path"].write_text(json.dumps(b["inventory"]), encoding="utf-8")
    frozen_bytes = b["inventory_path"].read_bytes()
    result = R.recompute(b["raw"], b["protocol_path"], b["inventory_path"], b["out"])
    updated = json.loads((b["out"] / "A0_METRIC_INVENTORY.json").read_text(encoding="utf-8"))
    lookup = {item["id"]: item for item in updated["metrics"]}
    for metric_id in replace_ids:
        raw = result["new_measurements"][metric_id]
        assert raw["unit"] == "fraction"
        assert lookup[metric_id]["new_value"] == pytest.approx(100 * raw["value"])
    assert result["new_measurements"][f"{R.FINAL}.actual_hnsw_gold_coverage_at_1000.0"]["value"] == 2 / 3
    text = (b["out"] / "A0_RESULTS.md").read_text(encoding="utf-8")
    assert "actual_hnsw_gold_coverage_at_1000 (%) | 66.66666667" in text
    assert b["inventory_path"].read_bytes() == frozen_bytes
    code = result["independent_reader"]["code_sha256"]
    assert set(Path(path).name for path in code) == {"recompute_a0.py", "a0_recompute_checks.py"}
    assert all(R.sha256(Path(path)) == digest == result["artifact_hashes"][path] for path, digest in code.items())


def test_supplemental_cannot_supply_retrieval_scores_or_unbound_old_results(tmp_path):
    source = tmp_path / "old.json"
    source.write_text('{"score": 1}', encoding="utf-8")
    records = tmp_path / "sources.json"
    metric = _item(R.FINAL, "gold_at_10", 0)
    records.write_text(json.dumps({"sources": [{"id": metric["id"], "path": str(source), "sha256": R.sha256(source),
                                               "pointer": "/score", "operation": "json_record"}]}), encoding="utf-8")
    with pytest.raises(R.IntegrityError, match="cannot supply"):
        R.supplemental_records(records, {}, {}, {"metrics": [metric]})


def test_manifest_rejects_traversal_and_authority_change(bundle):
    b = bundle
    with pytest.raises(R.IntegrityError, match="escapes"):
        R._local(b["raw"], "../elsewhere.npz")
    Path(b["protocol"]["authority"]["path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(R.IntegrityError, match="evidence changed"):
        R.verify_manifest(b["raw"], b["protocol_path"], b["inventory"])


def test_actual_writer_manifest_hash_roundtrip_handles_unicode(bundle, tmp_path):
    from ModelLakeFishing.scale1m import a0_evaluation as writer
    b = bundle
    binding = copy.deepcopy(b["manifest"]["binding"])
    binding["description"] = "本轮 A0：新图"
    binding["run_id"] = "fixture-A0"
    out = tmp_path / "writer-output"
    writer.open_manifest(out, "fixture-A0", binding)
    manifest, _ = R.verify_manifest(out, b["protocol_path"], b["inventory"])
    assert manifest["binding_sha256"] == writer._digest(binding) == R._canonical(binding)


@pytest.mark.parametrize("bad", ["missing_envelope", "old_graph", "other_seed", "old_checkpoint", "not_new"])
def test_old_or_misbound_training_logs_cannot_supply_new_measurement(bundle, tmp_path, bad):
    b = bundle
    source = tmp_path / "training-records.json"
    envelope = {"protocol": "a0", "run_id": "fixture-A0", "seed": 0,
                "graph_digest": "new", "checkpoint_sha256": "new-checkpoint"}
    if bad == "old_graph":
        envelope["graph_digest"] = "old"
    elif bad == "other_seed":
        envelope["seed"] = 1
    elif bad == "old_checkpoint":
        envelope["checkpoint_sha256"] = "old-checkpoint"
    document = {"a0": envelope, "segments": [{"start_ns": 0, "end_ns": 1000}]}
    if bad == "missing_envelope":
        del document["a0"]
    source.write_text(json.dumps(document), encoding="utf-8")
    source_rec = {"path": str(source), "sha256": R.sha256(source), "new_artifact": bad != "not_new", "role": "new_run_records"}
    b["manifest"]["binding"]["seed_inputs"]["0"]["files"] = [source_rec,
        {"role": "new_epoch25_checkpoint", "new_artifact": True, "sha256": "new-checkpoint"}]
    metric = _item("pipeline_cost", "training_seconds", 0)
    descriptors = tmp_path / "records.json"
    descriptors.write_text(json.dumps({"sources": [{"id": metric["id"], "path": str(source), "sha256": R.sha256(source),
        "pointer": "/segments", "operation": "duration_ns"}]}), encoding="utf-8")
    with pytest.raises(R.IntegrityError):
        R.supplemental_records(descriptors, {str(source): R.sha256(source)}, {}, {"metrics": [metric]}, b["manifest"], b["protocol"])


def test_new_bound_segment_records_sum_all_resumes(bundle, tmp_path):
    b = bundle
    source = tmp_path / "records.json"
    source.write_text(json.dumps({"a0": {"protocol": "a0", "run_id": "fixture-A0", "seed": 0,
        "graph_digest": "new", "checkpoint_sha256": "new-checkpoint"},
        "segments": [{"start_ns": 100, "end_ns": 1000000100}, {"start_ns": 2000000000, "end_ns": 5000000000}]}), encoding="utf-8")
    record = {"path": str(source), "sha256": R.sha256(source), "new_artifact": True, "role": "new_run_records"}
    b["manifest"]["binding"]["seed_inputs"]["0"]["files"] = [record,
        {"role": "new_epoch25_checkpoint", "new_artifact": True, "sha256": "new-checkpoint"}]
    metric = _item("pipeline_cost", "training_seconds", 0)
    descriptors = tmp_path / "descriptors.json"
    descriptors.write_text(json.dumps({"sources": [{"id": metric["id"], "path": str(source), "sha256": R.sha256(source),
        "pointer": "/segments", "operation": "duration_ns"}]}), encoding="utf-8")
    values = {}
    R.supplemental_records(descriptors, {str(source): R.sha256(source)}, values, {"metrics": [metric]}, b["manifest"], b["protocol"])
    assert values[metric["id"]]["value"] == 4.0


def test_actual_writer_native_rows_are_independently_comparable(bundle, monkeypatch):
    from ModelLakeFishing.scale1m import a0_evaluation as writer
    from ModelLakeFishing.scale1m import eval_y2 as evaluator
    b = bundle
    pack = _pack()
    labels = R.labels_from_pack(pack, 24)
    candidates = {int(q): pair for q, pair in zip(pack["query"], labels["observed"])}
    roots = np.full(104, "", dtype="<U5")
    roots[pack["query"]] = pack["root"]
    class Prior:
        def values(self, query, ids):
            return np.zeros(len(ids), dtype=np.float32)
    original_pool = evaluator._pool_metrics
    monkeypatch.setattr(evaluator, "N_TOTAL", 24)
    monkeypatch.setattr(evaluator, "POOL_K", 12)
    monkeypatch.setattr(evaluator, "_pool_metrics", lambda *args: original_pool(*args, k=12))
    written, _ = writer.pool_arrays(pack["model"], pack["score"], pack["query"], candidates, roots, Prior(), evaluator._tie_ranks(24))
    measured, _counts, _detail = R.pool_recompute(pack, labels, 24, 12, 10)
    assert set(written) == set(measured)
    for key in written:
        assert written[key] == measured[key]


def test_all_three_seeds_enable_summaries_without_historical_fill(bundle):
    b = bundle
    for seed in (1, 2):
        pack = _pack(seed)
        exact, hnsw, cal, index = (f"a0_exact_s{seed}.npz", f"a0_hnsw_s{seed}.npz",
                                  f"a0_calibration_s{seed}_ef1000.npz", f"hnsw_a0_s{seed}.bin")
        b["save"](exact, pack, seed=seed)
        b["save"](hnsw, pack, seed=seed)
        b["save"](cal, {"query": pack["query"], "model": pack["model"], "recall_per_query": np.ones(3)}, seed=seed)
        b["save"](index, seed=seed)
        meta = copy.deepcopy(b["manifest"]["seeds"]["0"])
        meta.update(exact=exact, hnsw=hnsw, calibration=[cal], index=index)
        b["manifest"]["seeds"][str(seed)] = meta
        b["manifest"]["binding"]["seed_inputs"][str(seed)] = {"graph_digest": "new"}
    b["flush"]()
    result = R.recompute(b["raw"], b["protocol_path"], b["inventory_path"], b["out"])
    assert result["completeness"]["missing_seeds"] == []
    assert result["new_measurements"][f"{R.FINAL}.gold_at_10.mean"]["value"] == 2 / 3
    assert result["new_measurements"]["retrieval_diagnostics.overall_retention.mean"]["value"] == 1.0
    assert not result["completeness"]["complete"]


def test_record_template_discovers_actual_fields_and_preserves_missing(bundle, tmp_path):
    b = bundle
    source = tmp_path / "A0_RUN_RECORDS.json"
    source.write_text(json.dumps({"a0": {"protocol": "a0", "run_id": "fixture-A0", "seed": 0,
        "graph_digest": "new", "checkpoint_sha256": "new-checkpoint"},
        "history": [{"total": 10 - i / 10, "rank": 3, "contrast": 2, "global": 5} for i in range(25)],
        "epochs": 25, "resolved_config": {"skip_diagnostics": True},
        "train_segments": [{"start_ns": 0, "end_ns": 1000}], "peak_process_rss_bytes": None}), encoding="utf-8")
    b["manifest"]["binding"]["seed_inputs"]["0"]["files"] = [
        {"path": str(source), "sha256": R.sha256(source), "new_artifact": True}]
    template = R.make_record_template(b["manifest"])
    descriptors = {source["id"]: source for source in template["sources"]}
    assert descriptors["training_records.total_loss_by_epoch.0"]["operation"] == "epoch_field"
    assert "pipeline_cost.training_peak_rss_bytes.0" not in descriptors
    assert any("training_peak_rss_bytes" in message for message in template["missing_sources"])
    assert any("seed 1" in message for message in template["missing_sources"])
    assert all("new_value" not in source for source in template["sources"])


def test_loss_series_and_each_native_epoch_field_expand(bundle, tmp_path):
    b = bundle
    source = tmp_path / "records.json"
    history = [{"total": float(25 - i), "rank": float(i), "sampling_extra": float(i * 2)} for i in range(25)]
    source.write_text(json.dumps({"a0": {"protocol": "a0", "run_id": "fixture-A0", "seed": 0,
        "graph_digest": "new", "checkpoint_sha256": "checkpoint"}, "history": history}), encoding="utf-8")
    file = {"path": str(source), "sha256": R.sha256(source), "new_artifact": True}
    b["manifest"]["binding"]["seed_inputs"]["0"]["files"] = [file,
        {"role": "new_epoch25_checkpoint", "new_artifact": True, "sha256": "checkpoint"}]
    inv = {"metrics": [_item("training_records", "all_existing_epoch_fields", 0),
                       _item("training_records", "total_loss_by_epoch", 0)]}
    descriptors = tmp_path / "descriptors.json"
    specs = [{"id": inv["metrics"][0]["id"], "path": str(source), "sha256": R.sha256(source),
              "pointer": "/history", "operation": "json_record"},
             {"id": inv["metrics"][1]["id"], "path": str(source), "sha256": R.sha256(source),
              "pointer": "/history", "operation": "epoch_field", "field": "total"}]
    descriptors.write_text(json.dumps({"sources": specs}), encoding="utf-8")
    values = {}
    R.supplemental_records(descriptors, {str(source): R.sha256(source)}, values, inv, b["manifest"], b["protocol"])
    R.expand_native_inventory(inv, {}, values)
    assert values["training_records.total_loss_by_epoch.0"]["value"] == [float(25 - i) for i in range(25)]
    assert values["training_records.native_epoch_sampling_extra.0"]["value"] == [float(i * 2) for i in range(25)]


def test_actual_export_manifest_writer_nested_envelope_roundtrip(bundle, tmp_path):
    from scale1m import export_rf as exporter
    b = bundle
    envelope = {"protocol": "a0", "run_id": "fixture-A0", "seed": 0,
                "graph_digest": "new", "checkpoint_sha256": "checkpoint"}
    exporter.merge_stage(str(tmp_path), "embed", {"a0": envelope,
        "export_segments": [{"start_ns": 10, "end_ns": 2000000010}],
        "gates": [{"gate": "G-F7d", "ran": True, "ok": True, "max_abs_delta": {"model": 1e-7, "dataset": 0}}]})
    source = tmp_path / "EXPORT_MANIFEST.json"
    file = {"path": str(source), "sha256": R.sha256(source), "new_artifact": True}
    b["manifest"]["binding"]["seed_inputs"]["0"]["files"] = [file,
        {"role": "new_epoch25_checkpoint", "new_artifact": True, "sha256": "checkpoint"}]
    template = R.make_record_template(b["manifest"])
    assert template["sources"][0]["envelope_pointer"] == "/stages/embed/a0"
    inv = {"metrics": [_item("pipeline_cost", "export_seconds", 0),
                       _item("reproducibility_checks", "chunked_inference_agreement", 0)]}
    descriptors = tmp_path / "descriptors.json"
    descriptors.write_text(json.dumps(template), encoding="utf-8")
    values = {}
    R.supplemental_records(descriptors, {str(source): R.sha256(source)}, values, inv, b["manifest"], b["protocol"])
    assert values["pipeline_cost.export_seconds.0"]["value"] == 2.0
    assert values["reproducibility_checks.chunked_inference_agreement.0"]["value"]["passed"] is True


@pytest.mark.parametrize("interrupted", [False, True])
def test_actual_evaluation_resource_summary_independently_uses_segments(bundle, tmp_path, interrupted):
    from ModelLakeFishing.scale1m import a0_evaluation as writer
    b = bundle
    document = {"schema_version": "a0.evaluation.resources.v1", "a0": {"protocol": "a0", "run_id": "fixture-A0", "seed": 0,
        "graph_digest": "new", "checkpoint_sha256": "checkpoint"},
        "exact_segments": [{"start_ns": 10, "end_ns": 2000000010, "status": "completed", "resource_peaks": {"rss_bytes": 10, "vram_bytes": 20}}],
        "hnsw_segments": [{"start_ns": 3000000000, "end_ns": 6000000000, "status": "completed", "resource_peaks": {"rss_bytes": 30, "vram_bytes": 15}}]}
    if interrupted:
        document["exact_segments"].insert(0, {"start_ns": 1, "end_ns": None, "status": "interrupted_end_unobserved"})
    writer._summarize_evaluation_record(document)
    source = tmp_path / "A0_EVAL_RECORDS_s0.json"
    source.write_text(json.dumps(document), encoding="utf-8")
    rec = {"path": str(source), "sha256": R.sha256(source), "seed": 0,
           "new_artifact": True, "role": "new_evaluation_resource_records"}
    b["manifest"]["artifacts"][source.name] = rec
    b["manifest"]["binding"]["seed_inputs"]["0"]["files"] = [
        {"role": "new_epoch25_checkpoint", "new_artifact": True, "sha256": "checkpoint"}]
    names = ["exact_full_reference_seconds", "exact1000_reference_seconds", "complete_evaluation_seconds",
             "evaluation_peak_rss_bytes", "evaluation_peak_vram_bytes"]
    inv = {"metrics": [_item("pipeline_cost", name, 0) for name in names]}
    descriptors = tmp_path / "descriptors.json"
    descriptors.write_text(json.dumps(R.make_record_template(b["manifest"])), encoding="utf-8")
    values = {}
    R.supplemental_records(descriptors, {str(source): R.sha256(source)}, values, inv, b["manifest"], b["protocol"])
    for name in names:
        actual = values[f"pipeline_cost.{name}.0"]
        assert actual["value"] == document[name]
        if interrupted:
            assert actual["status"] == "missing"
    if not interrupted:
        assert values["pipeline_cost.complete_evaluation_seconds.0"]["value"] == 5


def test_frozen_input_mapping_covers_27_items_without_provenance_counters():
    protocol = R._json(REPO / "docs/1M/A0_runs/A0_PROTOCOL.json")
    descriptors = R.frozen_audit_descriptors(protocol)
    assert len(descriptors) == 27
    assert len({spec["id"] for spec in descriptors}) == 27
    assert all("api_pages" not in spec["id"] and "skipped_duplicates" not in spec["id"] for spec in descriptors)
    assert not any("PROVENANCE" in spec["pointer"] or "metadata" in spec["pointer"] for spec in descriptors)
    dataset = next(spec for spec in descriptors if "snapshot_dataset_repositories" in spec["id"])
    assert dataset["pointer"].endswith("unique_raw_repository_ids")


def test_source_count_adapter_rejects_old_run_and_keeps_missing_events(bundle, tmp_path):
    b = bundle
    source = tmp_path / "raw-input.txt"
    source.write_text("fixture", encoding="utf-8")
    producer = Path(R.__file__).with_name("a0_source_counts.py")
    document = {"schema_version": "a0.source_counts.v1", "protocol_sha256": R.sha256(b["protocol_path"]),
        "authority": b["protocol"]["authority"], "a0": {"protocol": "a0", "run_id": "fixture-A0", "scope": "frozen_input_audit"},
        "implementation_sha256": {str(producer): R.sha256(producer)},
        "inputs": [{"path": str(source), "sha256": R.sha256(source)}],
        "measurements": {"native_raw_metric_rows": 10,
            "unknown_model_size_percent": {"value": 50., "numerator": 1, "denominator": 2, "scale": 100}},
        "missing": {"model_snapshot_api_pages": "Original event stream unavailable"}}
    report = tmp_path / "source-counts.json"
    report.write_text(json.dumps(document), encoding="utf-8")
    values = {}
    R.consume_source_counts(report, b["protocol_path"], b["manifest"], values, {})
    assert values["frozen_input_audit.native_raw_metric_rows.shared"]["value"] == 10
    assert values["frozen_input_audit.unknown_model_size_percent.shared"]["denominator"] == 2
    assert values["frozen_input_audit.model_snapshot_api_pages.shared"]["status"] == "missing"
    document["a0"]["run_id"] = "old"
    report.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(R.IntegrityError, match="run identity"):
        R.consume_source_counts(report, b["protocol_path"], b["manifest"], {}, {})


def test_all_618_metrics_have_automatic_source_or_explicit_event_gap():
    root = REPO / "docs/1M/A0_runs"
    audit = R.metric_source_coverage(root / "A0_METRIC_INVENTORY.json", root / "A0_PROTOCOL.json")
    assert audit["registered_items"] == 618
    assert len({entry["metric_id"] for entry in audit["metrics"]}) == 618
    assert len(audit["missing_original_evidence_ids"]) == 2
    assert sum(audit["classification_counts"].values()) == 618
    assert audit["new_metric_values_computed"] == 0
    assert not any(entry["manual_descriptor_required"] for entry in audit["metrics"])


def test_cli_does_not_skip_independent_raw_file_correctness_failure(bundle, monkeypatch):
    from ModelLakeFishing.scale1m import a0_recompute_checks as checks
    b = bundle
    def fail(manifest, protocol):
        raise ValueError("new graph raw correctness failed")
    monkeypatch.setattr(checks, "recompute_checks", fail)
    assert R.main(["--raw", str(b["raw"]), "--protocol", str(b["protocol_path"]),
        "--inventory", str(b["inventory_path"]), "--out", str(b["out"])]) == 1
