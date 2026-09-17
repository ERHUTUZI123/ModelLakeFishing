"""Y2: exact/HNSW top-1000 retrieval followed by the X2 task prior.

The experiment is specified in ``docs/1M/Y2.md``.  Stage ``exact`` first
separates pool truncation from ANN approximation.  Stage ``hnsw`` is allowed
only when the exact-pool retention gate passed.
"""
import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_PARENT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_PARENT not in sys.path:
    sys.path.insert(0, _REPO_PARENT)

from ModelLakeFishing.scale import global_metrics as GM
from ModelLakeFishing.scale1m.baselines import fixed_tie_break
from ModelLakeFishing.scale1m.eval_rf import _prior_tables, query_eligibility
from ModelLakeFishing.scale1m.hf_crawl import data_root, utcnow, write_json_atomic

SEEDS = (0, 1, 2)
N_TOTAL = 3_016_439
POOL_K = 1_000
EXPECTED_QUERIES = {0: 1476, 1: 1101, 2: 1545}
EXPECTED_DENSE = {0: 0.13550135501355012,
                  1: 0.14168937329700273,
                  2: 0.15080906148867315}
EXPECTED_SIDECAR_EDGES = {0: 198216, 1: 196912, 2: 196124}
EXPECTED_HELDOUT_DATASETS = {0: 4290, 1: 3112, 2: 3628}
EXPECTED_EXACT_POOL_SHA256 = {
    0: "b917db78a359ba1409ab482de9e17723fa4e9e87a2b8bcc816e72b10e7706375",
    1: "049b72ce1e9afd1ee4c0d7417b7069134ee3a6305b0783dd8ead3e083f2fd51c",
    2: "11953df97fbb2f8028bc5332bcba07c83fa97ca92c3ab67e3a2f0ee61a0faaf1",
}


def _ids(path, label):
    frame = pd.read_parquet(path).sort_values("mappedID")
    if not np.array_equal(frame["mappedID"].to_numpy(),
                          np.arange(len(frame), dtype=np.int64)):
        raise AssertionError("%s mappedID is not contiguous" % label)
    name = "model" if "model" in frame else "dataset"
    return frame[name].astype(str).to_numpy(), frame


def _same_ids(a, b, label):
    if not np.array_equal(a, b):
        raise AssertionError("%s mappings differ" % label)


def _sha256(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_candidates(export_dir, eligible_path, seed):
    with np.load(os.path.join(export_dir, "gold_cands.npz")) as payload:
        all_candidates = {
            int(key): (payload[key][0].astype(np.int64),
                       payload[key][1].astype(float))
            for key in payload.files
        }
    eligible, _ = query_eligibility(eligible_path, export_dir)
    candidates = {q: value for q, value in all_candidates.items() if eligible[q]}
    if len(candidates) != EXPECTED_QUERIES[seed]:
        raise AssertionError("seed %d has %d eligible queries, expected %d" %
                             (seed, len(candidates), EXPECTED_QUERIES[seed]))
    return candidates


def _tie_ranks(n):
    """Unique int64 rank of X6's uint64 permutation; smaller wins ties."""
    key = fixed_tie_break(n)
    order = np.argsort(key, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n, dtype=np.int64)
    if len(np.unique(rank)) != n:
        raise AssertionError("tie-break is not unique")
    return rank


class TaskPrior:
    def __init__(self, sidecar_path, meta_path, seed, candidates, roots):
        self.payload = np.load(sidecar_path)
        with open(meta_path, encoding="utf-8") as handle:
            self.meta = json.load(handle)
        for key, expected in (("split_seed", seed),
                              ("n_models", N_TOTAL),
                              ("n_edges", EXPECTED_SIDECAR_EDGES[seed]),
                              ("held_out_datasets", EXPECTED_HELDOUT_DATASETS[seed])):
            if int(self.meta[key]) != int(expected):
                raise AssertionError("sidecar %s=%r, expected %r" %
                                     (key, self.meta[key], expected))
        self.task_id = self.payload["task_id"].astype(np.int64)
        self.root_id = self.payload["root_id"].astype(np.int64)
        if len(self.task_id) != len(roots) or len(self.root_id) != len(roots):
            raise AssertionError("sidecar dataset mapping length differs")
        edge_model = self.payload["edge_model"].astype(np.int64)
        edge_dataset = self.payload["edge_dataset"].astype(np.int64)
        if (edge_model.min(initial=0) < 0 or edge_model.max(initial=0) >= N_TOTAL
                or edge_dataset.min(initial=0) < 0
                or edge_dataset.max(initial=0) >= len(roots)):
            raise AssertionError("sidecar contains an out-of-range mappedID")

        # Under the root-aware split, no visible edge may share the root of a
        # scored test query. This simultaneously checks self leakage and that
        # the sibling channel is empty under the X5/X6 protocol.
        visible_roots = np.unique(self.root_id[edge_dataset])
        query_roots = np.unique(self.root_id[np.fromiter(candidates, np.int64)])
        if np.intersect1d(visible_roots, query_roots).size:
            raise AssertionError("a test-query root appears in visible prior edges")
        if not np.array_equal(self.root_id.astype(str), roots.astype(str)):
            # The integer codes need not equal string roots, so compare their
            # induced equivalence relation on adjacent sorted groups below.
            frame = pd.DataFrame({"root": roots.astype(str), "code": self.root_id})
            if (frame.groupby("root")["code"].nunique().max() != 1
                    or frame.groupby("code")["root"].nunique().max() != 1):
                raise AssertionError("sidecar root ids disagree with dataset roots")
        self.by_task, _by_root, _null = _prior_tables(self.payload)

    def values(self, query, model_ids):
        model_ids = np.asarray(model_ids, dtype=np.int64)
        idx, val = self.by_task.get(
            int(self.task_id[int(query)]),
            (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)))
        out = np.zeros(model_ids.shape, dtype=np.float32)
        if idx.size:
            pos = np.searchsorted(idx, model_ids)
            clipped = np.minimum(pos, idx.size - 1)
            hit = (pos < idx.size) & (idx[clipped] == model_ids)
            out[hit] = np.asarray(val, dtype=np.float32)[clipped[hit]]
        return out


def _aggregate(counts, queries, candidates, roots, n_universe=N_TOTAL):
    per = {}
    for i, query in enumerate(queries):
        per[query] = {"gold_rank": int(counts[i, 0]) + 1,
                      "top3_rank": int(counts[i, 1]) + 1,
                      "gap_rank": int(counts[i, 2]) + 1,
                      "n_candidates": int(len(candidates[query][0]))}
    result = GM.aggregate(per, {q: roots[q] for q in queries})
    result.update({"N": int(n_universe),
                   "median_rank_over_N": result["median_gold_rank"] / n_universe,
                   "vs_random": result["gold@10"] / (10.0 / n_universe)})
    return result, per


def _pool_metrics(pool_ids, pool_dense, queries, candidates, roots, prior,
                  tie_rank, k=POOL_K):
    counts = np.zeros((len(queries), 3), dtype=np.int64)
    top_width = min(10, k)
    top10 = np.empty((len(queries), top_width), dtype=np.int64)
    gold_in_pool = np.zeros(len(queries), dtype=bool)
    retrieved_gold_ranks = []
    for i, query in enumerate(queries):
        ids = np.asarray(pool_ids[i], dtype=np.int64)
        if len(ids) != k or len(np.unique(ids)) != k:
            raise AssertionError("query %d pool is not %d unique ids" % (query, k))
        fused = ((np.asarray(pool_dense[i], dtype=np.float32) + 1.0) * 0.5
                 + prior.values(query, ids))
        order = np.lexsort((tie_rank[ids], -fused))
        ranked = ids[order]
        top10[i] = ranked[:top_width]
        cand, acc = candidates[query]
        gold, top3, near, _ = GM._probe_ids(cand, acc, GM.GAP_DELTA)
        pos = {int(model): rank for rank, model in enumerate(ranked, 1)}
        gold_in_pool[i] = gold in pos
        if gold in pos:
            retrieved_gold_ranks.append(pos[gold])
        # A missing probe cannot enter the returned top-10; K+1 is the censored
        # rank used only to compute @k metrics, never reported as a full rank.
        counts[i, 0] = pos.get(gold, k + 1) - 1
        counts[i, 1] = min(pos.get(int(model), k + 1) for model in top3) - 1
        counts[i, 2] = min(pos.get(int(model), k + 1) for model in near) - 1
    result, per = _aggregate(counts, queries, candidates, roots, n_universe=k)
    result["gold_in_first_stage@1000"] = float(gold_in_pool.mean())
    result["median_gold_rank_if_retrieved"] = (
        float(np.median(retrieved_gold_ranks)) if retrieved_gold_ranks else None)
    result["median_gold_rank"] = None
    result["median_rank_over_N"] = None
    result["vs_random"] = None
    return result, per, top10


def _top10_metrics(top10, queries, candidates, roots, n_universe=N_TOTAL):
    """Recompute metrics whose truth is completely determined by a top-10.

    A frozen top-10 proves every @1/@10 claim, including the root-macro
    variants, but cannot prove a rank below position ten.  Missing probes are
    therefore censored at 11 and the median rank is deliberately not reported.
    """
    top10 = np.asarray(top10, dtype=np.int64)
    if top10.shape != (len(queries), 10):
        raise AssertionError("top-10 shape %r != (%d, 10)" %
                             (top10.shape, len(queries)))
    if top10.min(initial=0) < 0 or top10.max(initial=0) >= n_universe:
        raise AssertionError("top-10 contains an out-of-range model id")
    per = {}
    for i, query in enumerate(queries):
        row = top10[i]
        if len(np.unique(row)) != 10:
            raise AssertionError("query %d top-10 contains duplicate ids" % query)
        rank = {int(model): pos for pos, model in enumerate(row, 1)}
        cand, acc = candidates[query]
        gold, top3, near, _ = GM._probe_ids(cand, acc, GM.GAP_DELTA)
        per[query] = {
            "gold_rank": rank.get(gold, 11),
            "top3_rank": min(rank.get(int(model), 11) for model in top3),
            "gap_rank": min(rank.get(int(model), 11) for model in near),
            "n_candidates": int(len(cand)),
        }
    result = GM.aggregate(per, {q: roots[q] for q in queries})
    result.update({"N": int(n_universe),
                   "median_gold_rank": None,
                   "median_rank_over_N": None,
                   "vs_random": result["gold@10"] / (10.0 / n_universe)})
    return result, per


def _task_only_top10(queries, prior, tie_rank):
    """Construct the exact full-lake task-prior top-10, cached per task.

    ``TaskPrior.by_task`` contains every model with a positive shrunken prior.
    Those models sort by the float32 serving value and then by X6's fixed
    tie-break.  If a task has fewer than ten observations, the remaining
    zero-prior models follow in the same fixed tie-break order.
    """
    result = np.empty((len(queries), 10), dtype=np.int64)
    cache = {}
    zero_order = None
    for i, query in enumerate(queries):
        task = int(prior.task_id[int(query)])
        if task not in cache:
            idx, value = prior.by_task.get(
                task, (np.zeros(0, dtype=np.int64),
                       np.zeros(0, dtype=np.float64)))
            idx = np.asarray(idx, dtype=np.int64)
            value = np.asarray(value, dtype=np.float32)
            if len(np.unique(idx)) != len(idx):
                raise AssertionError("task %d prior contains duplicate models" % task)
            if idx.size and (idx.min() < 0 or idx.max() >= N_TOTAL):
                raise AssertionError("task %d prior has out-of-range model ids" % task)
            if idx.size and not np.all(value > 0):
                raise AssertionError("task %d prior is not strictly positive" % task)
            order = np.lexsort((tie_rank[idx], -value)) if idx.size else []
            chosen = idx[order[:10]].tolist() if idx.size else []
            if len(chosen) < 10:
                if zero_order is None:
                    zero_order = np.argsort(tie_rank, kind="stable")
                observed = set(idx.tolist())
                for model in zero_order:
                    model = int(model)
                    if model not in observed:
                        chosen.append(model)
                        if len(chosen) == 10:
                            break
            cache[task] = np.asarray(chosen, dtype=np.int64)
        result[i] = cache[task]
    return result


def _copy_bound_pool(source, out_dir, seed):
    """Verify a frozen pool's identity, then place that exact file in ``out``."""
    expected = EXPECTED_EXACT_POOL_SHA256[seed]
    if not os.path.isfile(source):
        raise FileNotFoundError(source)
    actual = _sha256(source)
    if actual != expected:
        raise AssertionError("seed %d frozen pool sha256 %s != %s" %
                             (seed, actual, expected))
    target = os.path.join(out_dir, "exact_pool_s%d.npz" % seed)
    if os.path.abspath(source) != os.path.abspath(target):
        temporary = target + ".tmp"
        shutil.copyfile(source, temporary)
        if _sha256(temporary) != expected:
            os.remove(temporary)
            raise AssertionError("copied seed %d pool failed sha256" % seed)
        os.replace(temporary, target)
    elif _sha256(target) != expected:
        raise AssertionError("seed %d in-place pool changed during replay" % seed)
    return target, expected


def _topk_update(values, indices, score_block, offset, k, tie_rank=None):
    import torch
    if tie_rank is not None:
        # A0's full-fused top-10 must use the same label-free secondary key
        # as its exact rank counts. torch.topk alone picks arbitrary tied IDs.
        base = torch.arange(offset, offset + score_block.size(0),
                            device=score_block.device, dtype=torch.long)
        order = torch.argsort(tie_rank[base], stable=True)
        score = score_block[order]
        pos = torch.argsort(score, dim=0, descending=True, stable=True)
        pos = pos[:min(k, score.size(0))]
        v = torch.gather(score, 0, pos)
        i = base[order][pos]
        if values is None:
            return v, i
        cv, ci = torch.cat([values, v]), torch.cat([indices, i])
        tie_order = torch.argsort(tie_rank[ci], dim=0, stable=True)
        cv, ci = torch.gather(cv, 0, tie_order), torch.gather(ci, 0, tie_order)
        pos = torch.argsort(cv, dim=0, descending=True, stable=True)[:min(k, cv.size(0))]
        return torch.gather(cv, 0, pos), torch.gather(ci, 0, pos)
    v, i = torch.topk(score_block, min(k, score_block.size(0)), dim=0)
    i = i + offset
    if values is None:
        return v, i
    cv, ci = torch.cat([values, v]), torch.cat([indices, i])
    v2, pos = torch.topk(cv, k, dim=0)
    return v2, torch.gather(ci, 0, pos)


def _probe_metadata(queries, candidates):
    meta, flat_ids, flat_cols = [], [], []
    for col, query in enumerate(queries):
        cand, acc = candidates[query]
        gold, top3, near, ids = GM._probe_ids(cand, acc, GM.GAP_DELTA)
        meta.append((query, gold, top3, near, ids))
        flat_ids.append(ids)
        flat_cols.append(np.full(ids.size, col, dtype=np.int64))
    return meta, np.concatenate(flat_ids), np.concatenate(flat_cols)


def evaluate_exact_seed(export_dir, prior, candidates, roots, tie_rank, device,
                        model_chunk=50_000, query_chunk=16, protocol="legacy"):
    import torch

    started = time.time()
    zm_np = np.load(os.path.join(export_dir, "z_m_eval.npy"), mmap_mode="r")
    zd_np = np.load(os.path.join(export_dir, "z_d_eval.npy"), mmap_mode="r")
    if zm_np.shape[0] != N_TOTAL:
        raise AssertionError("candidate pool is %d, expected %d" %
                             (zm_np.shape[0], N_TOTAL))
    zm = torch.as_tensor(np.asarray(zm_np), dtype=torch.float32)
    zd = torch.as_tensor(np.asarray(zd_np), dtype=torch.float32)
    zm = (zm / zm.norm(dim=1, keepdim=True).clamp_min(1e-12)).to(device)
    zd = (zd / zd.norm(dim=1, keepdim=True).clamp_min(1e-12)).to(device)
    tie_t = torch.as_tensor(tie_rank, dtype=torch.long, device=device)
    queries = sorted(candidates)
    names = ("G_dense", "G_full_task") if protocol == "a0" else (
        "G_dense", "G_full_task", "P_task_only")
    all_counts = {name: np.zeros((len(queries), 3), dtype=np.int64)
                  for name in names}
    exact_ids = np.empty((len(queries), POOL_K), dtype=np.int32)
    exact_scores = np.empty((len(queries), POOL_K), dtype=np.float32)
    full_top10 = np.empty((len(queries), 10), dtype=np.int32)

    for qs in range(0, len(queries), query_chunk):
        block = queries[qs:qs + query_chunk]
        b = len(block)
        meta, flat, fcol = _probe_metadata(block, candidates)
        flat_t = torch.as_tensor(flat, dtype=torch.long, device=device)
        fcol_t = torch.as_tensor(fcol, dtype=torch.long, device=device)
        q_t = torch.as_tensor(block, dtype=torch.long, device=device)
        zq = zd[q_t]
        if protocol == "a0":
            # Use the exact same float32 GEMM shape/reduction as the full scan.
            # Elementwise dot reduction can differ by one ULP and make even
            # identical vectors rank ahead of their own probe. No tolerance,
            # score rounding or dtype change is introduced by this correction.
            raw_probe = torch.empty(flat_t.numel(), dtype=torch.float32, device=device)
            for ps in range(0, N_TOTAL, model_chunk):
                here = (flat_t >= ps) & (flat_t < min(ps + model_chunk, N_TOTAL))
                if bool(here.any()):
                    block_probe = zm[ps:ps + model_chunk] @ zq.t()
                    raw_probe[here] = block_probe[flat_t[here] - ps, fcol_t[here]]
                    del block_probe
        else:
            raw_probe = (zm[flat_t] * zq[fcol_t]).sum(-1)
        prior_probe_np = np.concatenate(
            [prior.values(q, ids) for q, _g, _t, _n, ids in meta])
        prior_probe = torch.as_tensor(prior_probe_np, dtype=torch.float32,
                                      device=device)
        fused_probe = (raw_probe + 1.0) * 0.5 + prior_probe
        tie_probe = tie_t[flat_t]
        greater = {name: torch.zeros(flat_t.numel(), dtype=torch.long, device=device)
                   for name in all_counts}
        dense_buf = (None, None)
        fused_buf = (None, None)

        # Sparse (model, query-column, prior) triples for this query block.
        bi, bc, bv = [], [], []
        for col, query in enumerate(block):
            idx, val = prior.by_task.get(
                int(prior.task_id[query]),
                (np.zeros(0, np.int64), np.zeros(0, np.float64)))
            bi.append(np.asarray(idx, dtype=np.int64))
            bc.append(np.full(len(idx), col, dtype=np.int64))
            bv.append(np.asarray(val, dtype=np.float32))
        BI = np.concatenate(bi) if bi else np.zeros(0, np.int64)
        BC = np.concatenate(bc) if bc else np.zeros(0, np.int64)
        BV = np.concatenate(bv) if bv else np.zeros(0, np.float32)
        order = np.argsort(BI, kind="stable")
        BI, BC, BV = BI[order], BC[order], BV[order]
        BI_t = torch.as_tensor(BI, dtype=torch.long, device=device)
        BC_t = torch.as_tensor(BC, dtype=torch.long, device=device)
        BV_t = torch.as_tensor(BV, dtype=torch.float32, device=device)

        probe_columns = torch.arange(flat_t.numel(), device=device)
        for ms in range(0, N_TOTAL, model_chunk):
            raw = zm[ms:ms + model_chunk] @ zq.t()
            c = raw.size(0)
            boost = torch.zeros((c, b), dtype=torch.float32, device=device)
            k0, k1 = np.searchsorted(BI, [ms, ms + c])
            if k1 > k0:
                boost[BI_t[k0:k1] - ms, BC_t[k0:k1]] = BV_t[k0:k1]
            fused = (raw + 1.0) * 0.5 + boost
            block_tie = tie_t[ms:ms + c].unsqueeze(1)

            dense_cmp = raw[:, fcol_t] > raw_probe.unsqueeze(0)
            fused_score = fused[:, fcol_t]
            fused_cmp = ((fused_score > fused_probe.unsqueeze(0)) |
                         ((fused_score == fused_probe.unsqueeze(0)) &
                          (block_tie < tie_probe.unsqueeze(0))))
            prior_score = boost[:, fcol_t]
            prior_cmp = ((prior_score > prior_probe.unsqueeze(0)) |
                         ((prior_score == prior_probe.unsqueeze(0)) &
                          (block_tie < tie_probe.unsqueeze(0))))
            here = (flat_t >= ms) & (flat_t < ms + c)
            if bool(here.any()):
                rows = flat_t[here] - ms
                cols = probe_columns[here]
                dense_cmp[rows, cols] = False
                fused_cmp[rows, cols] = False
                prior_cmp[rows, cols] = False
            greater["G_dense"] += dense_cmp.sum(0)
            greater["G_full_task"] += fused_cmp.sum(0)
            if "P_task_only" in greater:
                greater["P_task_only"] += prior_cmp.sum(0)
            dense_buf = _topk_update(*dense_buf, raw, ms, POOL_K)
            fused_buf = _topk_update(*fused_buf, fused, ms, 10,
                                     tie_rank=tie_t if protocol == "a0" else None)
            del raw, boost, fused, dense_cmp, fused_cmp, prior_cmp

        for name in all_counts:
            g = greater[name].cpu().numpy()
            off = 0
            for j, (_q, gold, top3, near, ids) in enumerate(meta):
                rank = {int(model): int(g[off + p])
                        for p, model in enumerate(ids)}
                off += len(ids)
                all_counts[name][qs + j] = [rank[gold],
                    min(rank[int(model)] for model in top3),
                    min(rank[int(model)] for model in near)]
        exact_ids[qs:qs + b] = dense_buf[1].t().cpu().numpy().astype(np.int32)
        exact_scores[qs:qs + b] = dense_buf[0].t().cpu().numpy()
        full_top10[qs:qs + b] = fused_buf[1].t().cpu().numpy().astype(np.int32)
        print("[Y2 exact] %d/%d queries %.1fs" %
              (min(qs + b, len(queries)), len(queries), time.time() - started),
              flush=True)

    rows = {}
    for name, counts in all_counts.items():
        rows[name], _ = _aggregate(counts, queries, candidates, roots)
    pool_row, _pool_per, pool_top10 = _pool_metrics(
        exact_ids, exact_scores, queries, candidates, roots, prior, tie_rank)
    full_in_pool = [len(set(full_top10[i].tolist()) & set(exact_ids[i].tolist())) / 10.0
                    for i in range(len(queries))]
    pool_row["full_fused_top10_in_dense_top1000"] = float(np.mean(full_in_pool))
    rows["G_exact1000_task"] = pool_row
    result = {"rows": rows, "queries": queries,
            "exact_pool_ids": exact_ids, "exact_pool_scores": exact_scores,
            "exact_top10": pool_top10, "full_top10": full_top10,
            "seconds": time.time() - started}
    if protocol == "a0":
        result["full_counts"] = all_counts["G_full_task"]
        result["dense_counts"] = all_counts["G_dense"]
    return result


def _summary(per_seed):
    names = list(next(iter(per_seed.values()))["rows"])
    out = {}
    for name in names:
        keys = set.intersection(*(set(per_seed[str(seed)]["rows"][name])
                                  for seed in SEEDS))
        metrics = {}
        for key in sorted(keys):
            vals = [per_seed[str(seed)]["rows"][name][key] for seed in SEEDS]
            if all(isinstance(v, (int, float)) and v is not None for v in vals):
                metrics[key] = {"per_seed": [float(v) for v in vals],
                                "mean": float(np.mean(vals)),
                                "min": float(np.min(vals)),
                                "max": float(np.max(vals))}
        out[name] = metrics
    return out


def _audit_seed(args, seed, first_model, first_dataset, tie_rank):
    x4 = os.path.join(args.exports, args.run_fmt % seed)
    f6 = os.path.join(args.sidecar_exports, args.sidecar_run_fmt % seed)
    x4_models, _ = _ids(os.path.join(x4, "model_ids.parquet"), "X4 model")
    x4_datasets, dataset_frame = _ids(
        os.path.join(x4, "dataset_ids.parquet"), "X4 dataset")
    f6_models, _ = _ids(os.path.join(f6, "model_ids.parquet"), "F6 model")
    f6_datasets, _ = _ids(os.path.join(f6, "dataset_ids.parquet"), "F6 dataset")
    if len(x4_models) != N_TOTAL:
        raise AssertionError("X4 model pool is %d" % len(x4_models))
    _same_ids(x4_models, f6_models, "X4/F6 model")
    _same_ids(x4_datasets, f6_datasets, "X4/F6 dataset")
    if first_model is not None:
        _same_ids(first_model, x4_models, "cross-seed model")
        _same_ids(first_dataset, x4_datasets, "cross-seed dataset")
    candidates = _load_candidates(x4, args.dataset_nodes, seed)
    roots = dataset_frame["root"].astype(str).to_numpy()
    prior = TaskPrior(
        os.path.join(f6, "prior_sidecar_s%d.npz" % seed),
        os.path.join(f6, "prior_sidecar_s%d_meta.json" % seed),
        seed, candidates, roots)
    if not np.array_equal(prior.task_id, np.load(
            os.path.join(f6, "prior_sidecar_s%d.npz" % seed))["task_id"]):
        raise AssertionError("task ids changed while loading")
    return {"x4": x4, "f6": f6, "models": x4_models,
            "datasets": x4_datasets, "candidates": candidates,
            "roots": roots, "prior": prior, "tie_rank": tie_rank}


def run_exact(args):
    os.makedirs(args.out, exist_ok=True)
    tie_rank = _tie_ranks(N_TOTAL)
    per_seed = {}
    first_model = first_dataset = None
    started = time.time()
    for seed in SEEDS:
        bundle = _audit_seed(args, seed, first_model, first_dataset, tie_rank)
        if first_model is None:
            first_model, first_dataset = bundle["models"], bundle["datasets"]
        result = evaluate_exact_seed(
            bundle["x4"], bundle["prior"], bundle["candidates"],
            bundle["roots"], tie_rank, args.device,
            model_chunk=args.model_chunk, query_chunk=args.query_chunk)
        got = result["rows"]["G_dense"]["gold@10"]
        if abs(got - EXPECTED_DENSE[seed]) > 1e-12:
            raise AssertionError("seed %d dense control %.15f != %.15f" %
                                 (seed, got, EXPECTED_DENSE[seed]))
        pool_path = os.path.join(args.out, "exact_pool_s%d.npz" % seed)
        np.savez(pool_path, query=np.asarray(result["queries"], dtype=np.int64),
                 model=result["exact_pool_ids"], score=result["exact_pool_scores"],
                 exact_top10=result["exact_top10"], full_top10=result["full_top10"])
        per_seed[str(seed)] = {"rows": result["rows"],
                               "seconds": result["seconds"],
                               "pool_artifact": pool_path}
        report = {"written_at": utcnow(), "stage": "exact-partial",
                  "protocol": {"N": N_TOTAL, "K": POOL_K,
                               "score": "(cos+1)/2 + task_prior",
                               "shrink_k": 5.0, "beta": 1.0,
                               "eligible_queries": EXPECTED_QUERIES},
                  "gates": {"mapping_and_leakage": True,
                            "dense_control_reproduced": True},
                  "per_seed": per_seed, "summary": _summary(per_seed)
                  if len(per_seed) == 3 else {},
                  "elapsed_s": time.time() - started}
        write_json_atomic(os.path.join(args.out, "Y2_REPORT.json"), report)
        del result, bundle
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    summary = _summary(per_seed)
    full = summary["G_full_task"]["gold@10"]["per_seed"]
    dense = summary["G_dense"]["gold@10"]["per_seed"]
    exact = summary["G_exact1000_task"]["gold@10"]["per_seed"]
    retention = [exact[i] / full[i] if full[i] else 0.0 for i in range(3)]
    prior_gain = float(np.mean(full)) > float(np.mean(dense))
    exact_gate = prior_gain and float(np.mean(retention)) >= 0.90
    report.update({"written_at": utcnow(), "stage": "exact-complete",
                   "summary": summary,
                   "decision": {"prior_improves_x4_mean": prior_gain,
                                "exact_pool_retention_per_seed": retention,
                                "exact_pool_retention_mean": float(np.mean(retention)),
                                "build_hnsw": exact_gate},
                   "elapsed_s": time.time() - started})
    write_json_atomic(os.path.join(args.out, "Y2_REPORT.json"), report)
    print(json.dumps(report["decision"], indent=2), flush=True)
    return report


def _replay_exact_seed(pool_path, bundle, tie_rank):
    """Recompute Y2's verifiable exact-stage results from one frozen pool."""
    started = time.time()
    required = {"query", "model", "score", "exact_top10", "full_top10"}
    with np.load(pool_path, allow_pickle=False) as payload:
        if set(payload.files) != required:
            raise AssertionError("frozen pool fields %r != %r" %
                                 (set(payload.files), required))
        queries = payload["query"].astype(np.int64)
        pool_ids = payload["model"].astype(np.int64)
        pool_scores = payload["score"].astype(np.float32)
        cached_exact_top10 = payload["exact_top10"].astype(np.int64)
        full_top10 = payload["full_top10"].astype(np.int64)

    expected_queries = np.asarray(sorted(bundle["candidates"]), dtype=np.int64)
    if not np.array_equal(queries, expected_queries):
        raise AssertionError("frozen-pool queries differ from eligible queries")
    expected_shape = (len(queries), POOL_K)
    if pool_ids.shape != expected_shape or pool_scores.shape != expected_shape:
        raise AssertionError("pool shapes %r/%r != %r" %
                             (pool_ids.shape, pool_scores.shape, expected_shape))
    if (cached_exact_top10.shape != (len(queries), 10)
            or full_top10.shape != (len(queries), 10)):
        raise AssertionError("frozen top-10 arrays have invalid shapes")
    if pool_ids.min(initial=0) < 0 or pool_ids.max(initial=0) >= N_TOTAL:
        raise AssertionError("frozen pool contains an out-of-range model id")
    if not np.isfinite(pool_scores).all():
        raise AssertionError("frozen pool has a non-finite dense score")
    if np.any(pool_scores[:, :-1] < pool_scores[:, 1:]):
        raise AssertionError("frozen dense pools are not score-sorted")

    dense_row, _ = _top10_metrics(
        pool_ids[:, :10], queries.tolist(), bundle["candidates"],
        bundle["roots"])
    full_row, _ = _top10_metrics(
        full_top10, queries.tolist(), bundle["candidates"], bundle["roots"])
    task_top10 = _task_only_top10(queries, bundle["prior"], tie_rank)
    task_row, _ = _top10_metrics(
        task_top10, queries.tolist(), bundle["candidates"], bundle["roots"])
    pool_row, _pool_per, recomputed_exact_top10 = _pool_metrics(
        pool_ids, pool_scores, queries.tolist(), bundle["candidates"],
        bundle["roots"], bundle["prior"], tie_rank)
    if not np.array_equal(recomputed_exact_top10, cached_exact_top10):
        mismatch = int(np.count_nonzero(
            np.any(recomputed_exact_top10 != cached_exact_top10, axis=1)))
        raise AssertionError("%d cached exact top-10 rows failed recomputation" %
                             mismatch)
    full_in_pool = [
        len(set(full_top10[i].tolist()) & set(pool_ids[i].tolist())) / 10.0
        for i in range(len(queries))]
    pool_row["full_fused_top10_in_dense_top1000"] = float(
        np.mean(full_in_pool))
    return {
        "rows": {"G_dense": dense_row,
                 "G_full_task": full_row,
                 "P_task_only": task_row,
                 "G_exact1000_task": pool_row},
        "queries": queries,
        "seconds": time.time() - started,
        "checks": {"query_order_matches_eligible_set": True,
                   "pool_scores_finite_and_sorted": True,
                   "exact_top10_recomputed_elementwise": True,
                   "task_only_top10_recomputed_from_sidecar": True},
    }


def run_replay_exact(args):
    """Replay the exact-stage metrics without rescoring all 3M embeddings.

    The `.npz` files are immutable sufficient statistics for top-10 claims and
    dense-pool reranking.  Their known SHA-256 values are checked before any
    result is computed.  The report is newly derived, never copied.
    """
    if not args.frozen_pools:
        raise SystemExit("--frozen-pools is required for --stage replay-exact")
    os.makedirs(args.out, exist_ok=True)
    tie_rank = _tie_ranks(N_TOTAL)
    per_seed = {}
    first_model = first_dataset = None
    started = time.time()
    pool_hashes = {}
    for seed in SEEDS:
        bundle = _audit_seed(args, seed, first_model, first_dataset, tie_rank)
        if first_model is None:
            first_model, first_dataset = bundle["models"], bundle["datasets"]
        source = os.path.join(args.frozen_pools, "exact_pool_s%d.npz" % seed)
        pool_path, digest = _copy_bound_pool(source, args.out, seed)
        result = _replay_exact_seed(pool_path, bundle, tie_rank)
        got = result["rows"]["G_dense"]["gold@10"]
        if abs(got - EXPECTED_DENSE[seed]) > 1e-12:
            raise AssertionError("seed %d dense control %.15f != %.15f" %
                                 (seed, got, EXPECTED_DENSE[seed]))
        pool_hashes[str(seed)] = digest
        per_seed[str(seed)] = {
            "rows": result["rows"], "seconds": result["seconds"],
            "pool_artifact": pool_path, "frozen_pool_source": source,
            "frozen_pool_sha256": digest, "replay_checks": result["checks"],
        }
        del result, bundle
        gc.collect()

    summary = _summary(per_seed)
    full = summary["G_full_task"]["gold@10"]["per_seed"]
    dense = summary["G_dense"]["gold@10"]["per_seed"]
    exact = summary["G_exact1000_task"]["gold@10"]["per_seed"]
    retention = [exact[i] / full[i] if full[i] else 0.0 for i in range(3)]
    prior_gain = float(np.mean(full)) > float(np.mean(dense))
    exact_gate = prior_gain and float(np.mean(retention)) >= 0.90
    report = {
        "written_at": utcnow(), "stage": "exact-complete",
        "protocol": {"N": N_TOTAL, "K": POOL_K,
                     "score": "(cos+1)/2 + task_prior",
                     "shrink_k": 5.0, "beta": 1.0,
                     "eligible_queries": EXPECTED_QUERIES,
                     "exact_stage_mode": "hash-bound frozen-pool replay"},
        "replay": {
            "frozen_pool_dir": os.path.abspath(args.frozen_pools),
            "sha256": pool_hashes,
            "scope": "top-10 metrics and complete dense-top-1000 reranking",
            "full_rank_statistics_available": False,
        },
        "gates": {"mapping_and_leakage": True,
                  "frozen_pool_sha256": True,
                  "dense_control_reproduced": True,
                  "cached_exact_top10_recomputed": True,
                  "task_only_ranking_recomputed": True},
        "per_seed": per_seed, "summary": summary,
        "decision": {"prior_improves_x4_mean": prior_gain,
                     "exact_pool_retention_per_seed": retention,
                     "exact_pool_retention_mean": float(np.mean(retention)),
                     "build_hnsw": exact_gate},
        "elapsed_s": time.time() - started,
    }
    write_json_atomic(os.path.join(args.out, "Y2_REPORT.json"), report)
    print(json.dumps({"gates": report["gates"],
                      "decision": report["decision"]}, indent=2), flush=True)
    return report


def run_hnsw(args):
    # Kept separate so exact-stage failure never creates multi-GB indexes.
    import hnswlib

    report_path = os.path.join(args.out, "Y2_REPORT.json")
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)
    if not report.get("decision", {}).get("build_hnsw"):
        raise SystemExit("Y2 exact-pool gate did not pass; refusing to build HNSW")
    tie_rank = _tie_ranks(N_TOTAL)
    first_model = first_dataset = None
    hrows = {}
    for seed in SEEDS:
        bundle = _audit_seed(args, seed, first_model, first_dataset, tie_rank)
        if first_model is None:
            first_model, first_dataset = bundle["models"], bundle["datasets"]
        with np.load(os.path.join(args.out, "exact_pool_s%d.npz" % seed)) as pool:
            queries = pool["query"].astype(np.int64)
            exact_ids = pool["model"].astype(np.int64)
        zm = np.load(os.path.join(bundle["x4"], "z_m_eval.npy"), mmap_mode="r")
        zd = np.load(os.path.join(bundle["x4"], "z_d_eval.npy"), mmap_mode="r")
        index_path = os.path.join(bundle["x4"], "hnsw_y2_eval.bin")
        build_seconds = 0.0
        if not os.path.isfile(index_path):
            t0 = time.time()
            index = hnswlib.Index(space="ip", dim=zm.shape[1])
            index.init_index(max_elements=N_TOTAL, ef_construction=args.ef_construction,
                             M=args.hnsw_M)
            index.add_items(zm, np.arange(N_TOTAL, dtype=np.int64),
                            num_threads=args.hnsw_threads)
            index.save_index(index_path)
            build_seconds = time.time() - t0
        else:
            index = hnswlib.Index(space="ip", dim=zm.shape[1])
            index.load_index(index_path, max_elements=N_TOTAL)
        index.set_num_threads(args.hnsw_threads)
        zq = np.asarray(zd[queries], dtype=np.float32)

        chosen = None
        trace = []
        for ef in args.ef_search:
            index.set_ef(max(int(ef), POOL_K))
            ids, dist = index.knn_query(zq, k=POOL_K,
                                        num_threads=args.hnsw_threads)
            recall = float(np.mean([
                len(set(ids[i].tolist()) & set(exact_ids[i].tolist())) / POOL_K
                for i in range(len(queries))]))
            trace.append({"ef_search": int(ef), "recall@1000": recall})
            print("[Y2 hnsw] seed %d ef=%d recall@1000=%.6f" %
                  (seed, ef, recall), flush=True)
            if recall >= 0.99:
                chosen = (int(ef), ids.astype(np.int64), (1.0 - dist).astype(np.float32), recall)
                break
        if chosen is None:
            hrows[str(seed)] = {"index": index_path,
                                "index_bytes": os.path.getsize(index_path),
                                "build_seconds": build_seconds, "ef_trace": trace,
                                "gate_recall@1000": False}
            report.setdefault("hnsw", {})[str(seed)] = hrows[str(seed)]
            write_json_atomic(report_path, report)
            continue
        ef, ids, scores, recall = chosen

        # Query and rerank latency are measured one query at a time. Index load
        # and construction are explicitly excluded.
        index.set_ef(ef)
        for i in range(min(50, len(queries))):
            index.knn_query(zq[i:i + 1], k=POOL_K, num_threads=1)
        query_ms, rerank_ms = [], []
        timed_ids, timed_scores = [], []
        for i, query in enumerate(queries):
            t0 = time.perf_counter_ns()
            one_id, one_dist = index.knn_query(zq[i:i + 1], k=POOL_K, num_threads=1)
            t1 = time.perf_counter_ns()
            one_score = 1.0 - one_dist[0]
            fused = ((one_score + 1.0) * 0.5
                     + bundle["prior"].values(int(query), one_id[0]))
            np.lexsort((tie_rank[one_id[0]], -fused))[:10]
            t2 = time.perf_counter_ns()
            query_ms.append((t1 - t0) / 1e6)
            rerank_ms.append((t2 - t1) / 1e6)
            timed_ids.append(one_id[0])
            timed_scores.append(one_score)
        timed_ids = np.asarray(timed_ids, dtype=np.int64)
        timed_scores = np.asarray(timed_scores, dtype=np.float32)
        row, _per, _top10 = _pool_metrics(
            timed_ids, timed_scores, queries.tolist(), bundle["candidates"],
            bundle["roots"], bundle["prior"], tie_rank)
        hrows[str(seed)] = {"rows": {"G_hnsw1000_task": row},
                            "index": index_path,
                            "index_bytes": os.path.getsize(index_path),
                            "build_seconds": build_seconds, "ef_trace": trace,
                            "ef_search": ef, "recall@1000": recall,
                            "gate_recall@1000": True,
                            "latency_ms": {
                                "hnsw_p50": float(np.percentile(query_ms, 50)),
                                "hnsw_p95": float(np.percentile(query_ms, 95)),
                                "rerank_p50": float(np.percentile(rerank_ms, 50)),
                                "rerank_p95": float(np.percentile(rerank_ms, 95)),
                                "end_to_end_p50": float(np.percentile(
                                    np.asarray(query_ms) + rerank_ms, 50)),
                                "end_to_end_p95": float(np.percentile(
                                    np.asarray(query_ms) + rerank_ms, 95))}}
        report.setdefault("hnsw", {})[str(seed)] = hrows[str(seed)]
        write_json_atomic(report_path, report)
        del index, bundle, zm, zd
        gc.collect()

    if all(hrows[str(seed)].get("gate_recall@1000") for seed in SEEDS):
        exact = report["summary"]["G_exact1000_task"]["gold@10"]["per_seed"]
        actual = [hrows[str(seed)]["rows"]["G_hnsw1000_task"]["gold@10"]
                  for seed in SEEDS]
        retention = [actual[i] / exact[i] if exact[i] else 0.0 for i in range(3)]
        report["decision"].update({
            "hnsw_pool_retention_per_seed": retention,
            "hnsw_pool_retention_mean": float(np.mean(retention)),
            "hnsw_deployment_gate": float(np.mean(retention)) >= 0.98})
        report["stage"] = "complete"
    else:
        report["stage"] = "hnsw-recall-failed"
    report["written_at"] = utcnow()
    write_json_atomic(report_path, report)
    print(json.dumps(report["decision"], indent=2), flush=True)
    return report


def finalize_report(args):
    """Add cross-seed HNSW summaries and reproducibility metadata."""
    report_path = os.path.join(args.out, "Y2_REPORT.json")
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("stage") != "complete":
        raise SystemExit("Y2 is not complete; refusing to finalize")
    hnsw = report["hnsw"]

    def values(path):
        out = []
        for seed in SEEDS:
            value = hnsw[str(seed)]
            for key in path:
                value = value[key]
            out.append(float(value))
        return out

    summary = {}
    for name, path in {
            "gold@1": ("rows", "G_hnsw1000_task", "gold@1"),
            "gold@10": ("rows", "G_hnsw1000_task", "gold@10"),
            "top3@10": ("rows", "G_hnsw1000_task", "top3@10"),
            "gold_in_first_stage@1000": (
                "rows", "G_hnsw1000_task", "gold_in_first_stage@1000"),
            "recall@1000": ("recall@1000",),
            "hnsw_p50_ms": ("latency_ms", "hnsw_p50"),
            "rerank_p50_ms": ("latency_ms", "rerank_p50"),
            "end_to_end_p50_ms": ("latency_ms", "end_to_end_p50"),
            "end_to_end_p95_ms": ("latency_ms", "end_to_end_p95"),
            "build_seconds": ("build_seconds",),
            "index_bytes": ("index_bytes",)}.items():
        vals = values(path)
        summary[name] = {"per_seed": vals, "mean": float(np.mean(vals)),
                         "min": min(vals), "max": max(vals)}
    report["hnsw_summary"] = summary
    report["protocol"].update({
        "hnsw_M": args.hnsw_M,
        "ef_construction": args.ef_construction,
        "hnsw_threads_build": args.hnsw_threads,
        "hnsw_threads_latency": 1,
        "ef_search_candidates": args.ef_search})
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        torch_version = torch.__version__
    except Exception:
        gpu, torch_version = None, None
    report["environment"] = {
        "hostname": platform.node(), "platform": platform.platform(),
        "processor": platform.processor(), "logical_cpus": os.cpu_count(),
        "python": platform.python_version(), "torch": torch_version,
        "gpu_for_exact_stage": gpu,
        "note": "HNSW construction/query and reranking are CPU operations; "
                "the GPU is used only by the exact 3M scoring stage."}
    report["gates"].update({
        "prior_improves_x4_mean": bool(report["decision"]["prior_improves_x4_mean"]),
        "exact_pool_retention_mean_ge_0.90":
            bool(report["decision"]["exact_pool_retention_mean"] >= 0.90),
        "all_hnsw_recall_at_1000_ge_0.99":
            all(hnsw[str(seed)]["recall@1000"] >= 0.99 for seed in SEEDS),
        "hnsw_retention_mean_ge_0.98":
            bool(report["decision"]["hnsw_pool_retention_mean"] >= 0.98),
        "all_artifacts_exist": all(
            os.path.isfile(hnsw[str(seed)]["index"])
            and os.path.isfile(os.path.join(args.out, "exact_pool_s%d.npz" % seed))
            for seed in SEEDS)})
    if not all(report["gates"].values()):
        raise AssertionError("a final Y2 gate failed: %r" % report["gates"])
    report["written_at"] = utcnow()
    write_json_atomic(report_path, report)
    print(json.dumps({"gates": report["gates"],
                      "hnsw_gold@10": summary["gold@10"],
                      "end_to_end_p50_ms": summary["end_to_end_p50_ms"]},
                     indent=2), flush=True)
    return report


def main(argv=None):
    data = os.path.join(data_root(), "data1m")
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=("legacy", "a0"), default="legacy",
                        help="legacy archived-value gates, or A0 fresh-artifact evaluation")
    parser.add_argument("--a0-protocol-file", default=os.path.join(
        _HERE, "..", "docs", "1M", "A0_runs", "A0_PROTOCOL.json"))
    parser.add_argument("--a0-query-identity", default=os.path.join(
        _HERE, "..", "docs", "1M", "A0_runs", "audit", "A0_QUERY_IDENTITY.jsonl"))
    parser.add_argument("--a0-run-id", default="A0_20260912")
    parser.add_argument("--stage", choices=("exact", "replay-exact", "hnsw",
                                             "finalize", "all"), default="exact")
    parser.add_argument("--exports", default=os.path.join(data, "exports_x4"))
    parser.add_argument("--run-fmt", default="X4GD_full_s%d_e25")
    parser.add_argument("--sidecar-exports", default=os.path.join(data, "exports_rf"))
    parser.add_argument("--sidecar-run-fmt", default="RF_full_s%d_e25")
    parser.add_argument("--dataset-nodes", default=os.path.join(
        data, "rf", "canon", "dataset_nodes_merged.parquet"))
    parser.add_argument("--out", default=os.path.join(data, "metrics_y2"))
    parser.add_argument("--frozen-pools", default=None,
                        help="directory containing hash-bound exact_pool_s{0,1,2}.npz")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-chunk", type=int, default=50_000)
    parser.add_argument("--query-chunk", type=int, default=16)
    parser.add_argument("--hnsw-M", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-threads", type=int, default=8)
    parser.add_argument("--ef-search", type=int, nargs="+",
                        default=[1000, 1500, 2000, 3000, 5000])
    args = parser.parse_args(argv)
    if args.protocol == "a0":
        from ModelLakeFishing.scale1m.a0_evaluation import run
        return run(args)
    report = None
    if args.stage in ("exact", "all"):
        report = run_exact(args)
    if args.stage == "replay-exact":
        report = run_replay_exact(args)
    if args.stage in ("hnsw", "all"):
        report = run_hnsw(args)
    if args.stage in ("finalize", "all"):
        report = finalize_report(args)
    return report


if __name__ == "__main__":
    main()
