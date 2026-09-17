"""A0 protocol/scoring fixtures; never runs training or a real HNSW index."""
import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ModelLakeFishing.scale1m import eval_y2 as E
from ModelLakeFishing.scale1m import a0_evaluation as A


def evaluation_record_fixture(tmp_path, monkeypatch):
    binding = {"seed_inputs": {"0": {"graph_digest": "new-graph",
               "files": [{"role": "new_epoch25_checkpoint", "sha256": "new-checkpoint"}]}}}
    manifest = A.open_manifest(tmp_path, "A0_20260912", binding)
    args = argparse.Namespace(out=str(tmp_path), device="cpu")
    monkeypatch.setattr(A, "evaluation_resource_peaks", lambda device: {
        "rss_bytes": 12345, "vram_bytes": 6789, "rss_unavailable_reason": None, "vram_unavailable_reason": None})
    return args, manifest


def test_evaluation_records_measure_shared_exact_once_and_bind_resource_bytes(tmp_path, monkeypatch):
    args, manifest = evaluation_record_fixture(tmp_path, monkeypatch)
    ticks = iter([1000, 2000, 3000, 5000])
    monkeypatch.setattr(A.time, "perf_counter_ns", lambda: next(ticks))
    exact = A.begin_evaluation_record(args, manifest, 0, "exact")
    assert exact["exact_segments"][0]["end_ns"] is None
    A.finish_evaluation_record(args, manifest, 0, "exact", exact)
    hnsw = A.begin_evaluation_record(args, manifest, 0, "hnsw")
    A.finish_evaluation_record(args, manifest, 0, "hnsw", hnsw)
    assert hnsw["exact_full_reference_seconds"] == hnsw["exact1000_reference_seconds"] == 1e-6
    assert hnsw["complete_evaluation_seconds"] == 3e-6
    assert hnsw["timing_complete"] and hnsw["status"] == "completed"
    assert hnsw["evaluation_peak_rss_bytes"] == 12345
    name = manifest["seeds"]["0"]["evaluation_records"]
    assert manifest["artifacts"][name]["new_artifact"]
    assert manifest["artifacts"][name]["role"] == "new_evaluation_resource_records"
    assert manifest["artifacts"][name]["sha256"] == A.E._sha256(tmp_path / name)
    A.verify_producer_envelope(hnsw, {"protocol": "a0", "run_id": "A0_20260912", "seed": 0,
        "graph_digest": "new-graph", "checkpoint_sha256": "new-checkpoint"}, "fixture")


def test_evaluation_records_keep_hard_interruption_missing_after_resume(tmp_path, monkeypatch):
    args, manifest = evaluation_record_fixture(tmp_path, monkeypatch)
    A.begin_evaluation_record(args, manifest, 0, "exact")
    resumed = A.begin_evaluation_record(args, manifest, 0, "exact")
    assert resumed["exact_segments"][0]["status"] == "interrupted_end_unobserved"
    assert resumed["exact_segments"][0]["end_ns"] is None
    A.finish_evaluation_record(args, manifest, 0, "exact", resumed)
    hnsw = A.begin_evaluation_record(args, manifest, 0, "hnsw")
    A.finish_evaluation_record(args, manifest, 0, "hnsw", hnsw)
    assert hnsw["status"] == "completed" and not hnsw["timing_complete"]
    assert hnsw["exact_full_reference_seconds"] is None
    assert hnsw["complete_evaluation_seconds"] is None
    assert hnsw["evaluation_peak_rss_bytes"] is None
    assert hnsw["evaluation_peak_rss_bytes_observed_max"] == 12345


def test_evaluation_records_caught_failure_records_actual_endpoint(tmp_path, monkeypatch):
    args, manifest = evaluation_record_fixture(tmp_path, monkeypatch)
    A.begin_evaluation_record(args, manifest, 0, "exact")
    A.fail_active_evaluation_record(args, manifest, "exact")
    record = A._read_json(tmp_path / "A0_EVAL_RECORDS_s0.json")
    assert record["exact_segments"][0]["status"] == "failed"
    assert record["exact_segments"][0]["end_ns"] is not None
    assert record["complete_evaluation_seconds"] is None


def test_evaluation_records_reject_tampered_resource_cache(tmp_path, monkeypatch):
    args, manifest = evaluation_record_fixture(tmp_path, monkeypatch)
    A.begin_evaluation_record(args, manifest, 0, "exact")
    (tmp_path / "A0_EVAL_RECORDS_s0.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        A.begin_evaluation_record(args, manifest, 0, "hnsw")


def test_chunked_export_requires_measured_original_scope_and_raw_delta_threshold():
    gate = {"gate": "G-F7d", "ran": True, "ok": True, "n_models": 100000,
            "max_abs_delta": {"model": 2e-7, "dataset": 1e-7}}
    meta = {"chunk": 50000, "gates": [gate]}
    assert A.verify_chunked_export(meta) == gate
    for bad in ({"gates": []}, {"chunk": 1000},
                {"gates": [{**gate, "ran": False}]},
                {"gates": [{**gate, "ok": False}]},
                {"gates": [{**gate, "n_models": 1000}]},
                {"gates": [{**gate, "max_abs_delta": {"model": 1e-5, "dataset": 0}}]},
                {"gates": [{**gate, "max_abs_delta": {"model": float("nan"), "dataset": 0}}]}):
        with pytest.raises(ValueError):
            A.verify_chunked_export({**meta, **bad})


class EmptyPrior:
    def __init__(self, q=3):
        self.task_id = np.zeros(q, dtype=np.int64)
        self.by_task = {}

    def values(self, query, ids):
        return np.zeros(np.shape(ids), dtype=np.float32)


def test_tied_full_top10_matches_fixed_permutation_across_chunks():
    rng = np.random.default_rng(9)
    values = rng.integers(0, 4, size=(35, 3)).astype(np.float32)
    ties = E._tie_ranks(35)
    buf = (None, None)
    for lo, hi in ((0, 12), (12, 29), (29, 35)):
        buf = E._topk_update(*buf, torch.tensor(values[lo:hi]), lo, 10,
                            tie_rank=torch.tensor(ties))
    got = buf[1].numpy()
    wanted = np.stack([np.lexsort((ties, -values[:, c]))[:10] for c in range(3)], axis=1)
    assert np.array_equal(got, wanted)


def test_actual_exact_duplicate_128d_vectors_have_consistent_rank_and_top10(tmp_path, monkeypatch):
    # Before the A0 GEMM-probe correction this exact fixture reported rank 1024
    # for the tie-winning gold while putting it first in fixed-tie full_top10.
    monkeypatch.setattr(E, "N_TOTAL", 1024)
    rng = np.random.default_rng(8)
    v = rng.normal(size=128).astype(np.float32)
    v /= np.linalg.norm(v)
    np.save(tmp_path / "z_m_eval.npy", np.tile(v, (1024, 1)))
    np.save(tmp_path / "z_d_eval.npy", np.tile(v, (2, 1)))
    tie = E._tie_ranks(1024)
    gold = int(np.argsort(tie)[0])
    candidates = {0: (np.array([gold, 17, 29]), np.array([1., .6, .2]))}
    result = E.evaluate_exact_seed(str(tmp_path), EmptyPrior(), candidates,
        np.array(["r", "other"]), tie, "cpu", model_chunk=1024, query_chunk=1, protocol="a0")
    assert int(result["full_top10"][0, 0]) == gold
    assert result["full_counts"][0, 0] == 0
    assert result["rows"]["G_full_task"]["gold@1"] == 1
    assert "P_task_only" not in result["rows"]
    gc.collect()


def test_actual_exact_random_and_duplicate_vectors_full_rank_agrees(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "N_TOTAL", 2048)
    rng = np.random.default_rng(13)
    base = rng.normal(size=(1024, 128)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    model = np.concatenate([base, base])
    query = base[[11, 81, 952]]
    np.save(tmp_path / "z_m_eval.npy", model)
    np.save(tmp_path / "z_d_eval.npy", query)
    candidates = {i: (np.array([int(m), int(m+1024), 0]), np.array([1., .8, .1]))
                  for i, m in enumerate([11, 81, 952])}
    ties = E._tie_ranks(2048)
    result = E.evaluate_exact_seed(str(tmp_path), EmptyPrior(), candidates,
        np.array(["r", "r", "s"]), ties, "cpu", model_chunk=1024, query_chunk=2, protocol="a0")
    for i, q in enumerate(result["queries"]):
        rank = int(result["full_counts"][i, 0]) + 1
        gold = int(candidates[q][0][0])
        if rank <= 10:
            assert int(result["full_top10"][i, rank-1]) == gold
    from_top, _ = E._top10_metrics(result["full_top10"], result["queries"], candidates,
                                  np.array(["r", "r", "s"]))
    for k in ("gold@1", "gold@10", "gold-gap@1", "gold-gap@10", "top3@10", "root_gold@10"):
        assert result["rows"]["G_full_task"][k] == from_top[k]
    gc.collect()


def test_pool_miss_stays_in_denominator_and_conditional_rank_has_own_count(monkeypatch):
    monkeypatch.setattr(E, "N_TOTAL", 2048)
    ids = np.tile(np.arange(1000), (2, 1))
    score = np.tile(np.linspace(1, -1, 1000, dtype=np.float32), (2, 1))
    cands = {0: (np.array([3, 5, 6]), np.array([1., .5, .3])),
             1: (np.array([1001, 4, 7]), np.array([1., .8, .7]))}
    row, raw = A.pool_arrays(ids, score, [0,1], cands, np.array(["r", "s"]), EmptyPrior(), E._tie_ranks(2048))
    assert row["gold@10"] == .5
    assert row["gold_retrieved_query_count"] == 1
    assert row["median_gold_rank_if_retrieved"] == 4
    assert raw["gold_position"].tolist() == [4, 0]
    assert row["median_gold_rank"] is None
    assert row["median_rank_over_N"] is None
    assert np.array_equal(raw["fused"], (score+np.float32(1))*np.float32(.5))
    assert set(raw["top10"][1]) <= set(ids[1])


class FakeIndex:
    def __init__(self, pass_at=None):
        self.pass_at = pass_at
        self.ef = None

    def set_ef(self, ef): self.ef = ef

    def knn_query(self, zq, k, num_threads):
        lo = 0 if self.pass_at is not None and self.ef >= self.pass_at else 1000
        return np.tile(np.arange(lo,lo+k), (len(zq),1)), np.zeros((len(zq),k), dtype=np.float32)


@pytest.mark.parametrize("pass_at,expected_ef,trace_len,passed", [(1500,1500,2,True),(None,5000,5,False)])
def test_ef_first_pass_or_measure_maximum_failure(pass_at, expected_ef, trace_len, passed):
    seen=[]
    result, trace, ok = A.calibrate(FakeIndex(pass_at), np.zeros((2,128),np.float32), [0,1],
        np.tile(np.arange(1000),(2,1)), 8,
        lambda ef,ids,scores,recalls: seen.append((ef,ids.copy(),recalls.copy())) or str(ef))
    assert result[0] == expected_ef
    assert len(trace)==trace_len and ok is passed
    assert len(seen)==trace_len
    assert np.all(seen[-1][2] == (1 if passed else 0))


def test_ratios_are_seed_ratios_and_zero_denominator_is_undefined():
    result=A.ratio_summary([1,2,3],[2,8,12])
    assert result["mean"]==pytest.approx((.5+.25+.25)/3)
    assert result["mean"] != sum([1,2,3])/sum([2,8,12])
    undefined=A.ratio_summary([1,0,3],[2,0,12])
    assert undefined["mean"] is None and undefined["status"]=="undefined"
    assert undefined["numerators"][1]==0 and undefined["denominators"][1]==0


def test_latency_total_percentiles_are_from_same_query_totals():
    h=np.array([1,1,100],np.int64)*1000000
    r=np.array([100,1,1],np.int64)*1000000
    out=A.latency_summary({"hnsw_ns":h,"rerank_ns":r,"total_ns":h+r})
    assert out["end_to_end_p50"]==101
    assert out["hnsw_p50"]+out["rerank_p50"]==2


def test_lower_effectiveness_does_not_become_true_to_complete_measurement():
    report={"per_seed":{str(s):{"rows":{"G_full_task":{"gold@10":.1},
        "G_dense":{"gold@10":.2},"G_exact1000_task":{"gold@10":.05}}} for s in E.SEEDS}}
    d=A.exact_decision(report)
    assert d["prior_improves_x4_mean"] is False
    assert d["build_hnsw"] is False and d["original_exact_effectiveness_gate"] is False
    assert d["a0_complete_measurement_despite_effectiveness_gate"] is True


def test_cache_refuses_different_binding_and_tampered_artifact(tmp_path):
    binding={"protocol_sha256":"abc","run_id":"test","path":"中文"}
    manifest=A.open_manifest(tmp_path,"test",binding)
    A.save_npz(tmp_path,"a0_exact_s0.npz",query=np.array([0]))
    A.register_artifact(tmp_path,manifest,"a0_exact_s0.npz","test_raw",0)
    A._save_manifest(tmp_path,manifest)
    assert A.open_manifest(tmp_path,"test",binding)["binding_sha256"]==A._digest(binding)
    with pytest.raises(ValueError,match="binding changed"):
        A.open_manifest(tmp_path,"test",dict(binding,protocol_sha256="changed"))
    (tmp_path/"a0_exact_s0.npz").write_bytes(b"wrong")
    with pytest.raises(ValueError,match="SHA256 mismatch"):
        A.open_manifest(tmp_path,"test",binding)


def test_nonempty_unbound_output_is_not_reused(tmp_path):
    (tmp_path/"old.json").write_text("{}")
    with pytest.raises(ValueError,match="nonempty"):
        A.open_manifest(tmp_path,"test",{})


def test_producer_hash_chain_rejects_stale_checkpoint_or_unmarked_old_output():
    expected={"protocol":"a0","run_id":"A0_20260912","seed":0,
              "graph_digest":"newgraph","checkpoint_sha256":"newcheckpoint"}
    A.verify_producer_envelope({"a0":expected.copy()}, expected, "export")
    with pytest.raises(ValueError,match="producer binding"):
        A.verify_producer_envelope({}, expected, "old export")
    stale={"a0":dict(expected,checkpoint_sha256="oldcheckpoint")}
    with pytest.raises(ValueError,match="checkpoint_sha256"):
        A.verify_producer_envelope(stale, expected, "export")


def test_legacy_cli_dispatch_unchanged_and_a0_replay_rejected(tmp_path, monkeypatch):
    seen=[]
    monkeypatch.setattr(E,"run_exact",lambda args:seen.append(args.protocol) or "legacy")
    assert E.main(["--stage","exact"])=="legacy" and seen==["legacy"]
    args=argparse.Namespace(stage="replay-exact",frozen_pools=None,hnsw_M=32,ef_construction=200,
        ef_search=A.GRID,hnsw_threads=8,model_chunk=50000,query_chunk=16,a0_run_id="a0",out=str(tmp_path))
    with pytest.raises(ValueError,match="replay"):
        A.check_fixed_options(args)


def test_all_native_quality_fields_and_nulls_survive_summary():
    row={"gold@1":.1,"gold@10":.2,"gold-gap@1":.1,"root_gold@1":.1,
         "root_top3@10":.2,"root_gold-gap@10":.3,"median_gold_rank":None}
    report={str(s):{"rows":{"G_hnsw1000_task":row.copy()}} for s in E.SEEDS}
    summary=A.summarize_rows(report)["G_hnsw1000_task"]
    assert set(summary)==set(row)
    assert summary["median_gold_rank"]["status"]=="undefined"
    with pytest.raises(ValueError,match="three"):
        A.summarize_rows({"0":report["0"]})
