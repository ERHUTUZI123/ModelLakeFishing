"""Recompute the X5 eligibility and dense top-10 claims from frozen pools.

The frozen Y2 exact pools contain the exact dense top 1,000 in score order.
Consequently their first ten rows are sufficient to replay every X5 @1/@10
claim.  Rank statistics below position ten are intentionally not reconstructed.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from .eval_rf import ELIGIBILITY_REASONS, query_eligibility
from .eval_y2 import (EXPECTED_DENSE, EXPECTED_EXACT_POOL_SHA256,
                      EXPECTED_QUERIES, N_TOTAL, _sha256, _top10_metrics)
from .hf_crawl import data_root, utcnow, write_json_atomic


SEEDS = (0, 1, 2)


def _candidates(run_dir: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    with np.load(os.path.join(run_dir, "gold_cands.npz"), allow_pickle=False) as z:
        return {
            int(key): (z[key][0].astype(np.int64), z[key][1].astype(float))
            for key in z.files
        }


def _seed(args, seed: int) -> dict:
    run_dir = os.path.join(args.exports, args.run_fmt % seed)
    pool_path = os.path.join(args.frozen_pools, "exact_pool_s%d.npz" % seed)
    digest = _sha256(pool_path)
    expected_digest = EXPECTED_EXACT_POOL_SHA256[seed]
    if digest != expected_digest:
        raise AssertionError("seed %d frozen pool sha256 %s != %s" %
                             (seed, digest, expected_digest))

    candidates = _candidates(run_dir)
    eligible, why = query_eligibility(args.dataset_nodes, run_dir)
    kept = sorted(query for query in candidates if eligible[query])
    dropped = sorted(set(candidates) - set(kept))
    if len(kept) != EXPECTED_QUERIES[seed]:
        raise AssertionError("seed %d has %d eligible queries, expected %d" %
                             (seed, len(kept), EXPECTED_QUERIES[seed]))

    with np.load(pool_path, allow_pickle=False) as pool:
        queries = pool["query"].astype(np.int64)
        model = pool["model"].astype(np.int64)
        score = pool["score"].astype(np.float32)
    if not np.array_equal(queries, np.asarray(kept, dtype=np.int64)):
        raise AssertionError("seed %d frozen-pool queries differ from X5 eligibility" % seed)
    if model.shape != (len(kept), 1000) or score.shape != model.shape:
        raise AssertionError("seed %d frozen pool has invalid shape" % seed)
    if model.min(initial=0) < 0 or model.max(initial=0) >= N_TOTAL:
        raise AssertionError("seed %d frozen pool contains an invalid model id" % seed)
    if not np.isfinite(score).all() or np.any(score[:, :-1] < score[:, 1:]):
        raise AssertionError("seed %d frozen dense pool is not finite and sorted" % seed)

    ids = pd.read_parquet(os.path.join(run_dir, "dataset_ids.parquet"))
    ids = ids.sort_values("mappedID")
    roots = dict(zip(ids["mappedID"].astype(int), ids["root"].astype(str)))
    eligible_candidates = {query: candidates[query] for query in kept}
    row, _ = _top10_metrics(model[:, :10], kept, eligible_candidates, roots)
    if abs(row["gold@10"] - EXPECTED_DENSE[seed]) > 1e-12:
        raise AssertionError("seed %d dense gold@10 changed: %.15f != %.15f" %
                             (seed, row["gold@10"], EXPECTED_DENSE[seed]))

    return {
        "n_queries_before": len(candidates),
        "n_queries_after": len(kept),
        "n_excluded": len(dropped),
        "share_excluded": len(dropped) / len(candidates),
        "excluded_by_reason": {
            reason: int(sum(bool(why[reason][query]) for query in dropped))
            for reason in ELIGIBILITY_REASONS
        },
        "rows": {"gold_eligible_only": row},
        "replay": {
            "scope": "exact dense @1/@10 metrics; ranks below ten are censored",
            "frozen_pool": os.path.abspath(pool_path),
            "frozen_pool_sha256": digest,
        },
    }


def main(argv=None) -> int:
    base = os.path.join(data_root(), "data1m")
    parser = argparse.ArgumentParser(
        description="Replay X5 eligibility and dense top-10 from hash-bound pools")
    parser.add_argument("--exports", default=os.path.join(base, "exports_x4"))
    parser.add_argument("--run-fmt", default="X4GD_full_s%d_e25")
    parser.add_argument("--dataset-nodes", default=os.path.join(
        base, "rf", "canon", "dataset_nodes_merged.parquet"))
    parser.add_argument("--frozen-pools", required=True)
    parser.add_argument("--out", default=os.path.join(base, "reproduced", "x5"))
    args = parser.parse_args(argv)

    started = time.time()
    per_seed = {str(seed): _seed(args, seed) for seed in SEEDS}
    payload = {
        "written_at": utcnow(),
        "axes": {"e": {
            "seconds": time.time() - started,
            "exports": os.path.abspath(args.exports),
            "run_fmt": args.run_fmt,
            "dataset_nodes": os.path.abspath(args.dataset_nodes),
            "mode": "hash-bound top-10 replay",
            "per_seed": per_seed,
        }},
        "gates": {
            "frozen_pool_sha256": True,
            "eligible_query_identity": True,
            "dense_top10_reproduced": True,
        },
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "X5_QUERY_ELIGIBILITY.json")
    write_json_atomic(path, payload)
    print(json.dumps({seed: row["rows"]["gold_eligible_only"]["gold@10"]
                      for seed, row in per_seed.items()}, indent=2))
    print("[ok] %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
