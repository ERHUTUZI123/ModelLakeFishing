"""
export_rung.py -- T7: turn a T6 checkpoint into a servable index plus the
leakage-free embeddings the A axis is allowed to use.

Runbook: docs/1M/100kplan.md §10, docs/1M/T7.md.

WHAT THIS DOES NOT DO
    It does not train. T6 already produced the weights; re-training here would
    silently produce a *different* model from the one whose loss curve and
    mechanism gate were recorded, and every number downstream would describe a
    model nobody audited. The checkpoint is the input, not a suggestion.

THE TWO FORWARDS, AND WHY BOTH EXIST
    z_*_full  : whole graph. This is what gets indexed and served -- at serving
                time every edge legitimately exists.
    z_*_eval  : test-split forward. The held-out query dataset does NOT see its
                own trained_on edges.

    Only z_*_eval may produce a reported gold@K. P4 learned this the expensive
    way: scoring with the full-graph z_d let each held-out query see its own
    supervision and gold@10 read 0.61 instead of 0.42. That is why §10's G-D2
    is an *inequality* rather than a note -- it asserts the leak-free number is
    the smaller one, which fails loudly if the two are ever swapped.

THE BINDING CHECK IS THE POINT
    A checkpoint records the sha256 of the graph it was trained on. Exporting
    against a different graph is not a warning here, it is a hard stop. This
    matters concretely: the 100K graph was rebuilt on 2026-08-14 to carry
    lineage relation ids (T6more.md §1.8), so `hgraph_100k.pt` now hashes
    differently than it did on 2026-08-11. Checkpoints from before that rebuild
    must not be exported against the file sitting at that path today.

Run:
    python -m scale1m.export_rung --run $OUTPUT_ROOT/runs/R2rel_100k_s0_e25 \
        --chunk 50000 --hnsw-M 32 --ef-construction 200 --hnsw-threads 8
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(_HERE), ".."))
_GIT_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scale1m import checkpoint as CK                                # noqa: E402
from scale1m.hf_crawl import utcnow, write_json_atomic              # noqa: E402
from scale1m.train_rung import RUNGS                                # noqa: E402


# ── gates (§12 G-D1 / G-D2 / G-D3) ──────────────────────────────────────────

class GateFailure(RuntimeError):
    pass


def gate_pool_size(n_models, expect_n):
    """G-D1 / iron rule 3. The classic silent bug at 100K is scoring only the
    30,183 CORE models, seeing gold@10 barely move, and reporting that the
    system 'withstood' 83x the distractors."""
    ok = expect_n is None or n_models == expect_n
    return {"gate": "G-D1", "name": "candidate pool size", "ok": bool(ok),
            "n_models": int(n_models), "expect_n": expect_n}


def gate_leakage(gold10_eval, gold10_full):
    """G-D2. Held-out embeddings must score WORSE than full-graph ones; if they
    do not, the query saw its own supervision edges."""
    ok = gold10_eval < gold10_full
    return {"gate": "G-D2", "name": "held-out < full-graph gold@10", "ok": bool(ok),
            "gold10_eval": float(gold10_eval), "gold10_full": float(gold10_full),
            "margin": float(gold10_full - gold10_eval)}


def gate_row_order(model_ids, ladder_ids, n_probe=100, seed=0):
    """G-D3. Row order is risk #1 in CLAUDE.md and it never raises on its own:
    z_m row i simply stops meaning model i, and every downstream number is
    quietly about the wrong model. Probe random positions rather than the head,
    because an off-by-one prefix bug survives a head-only check."""
    if ladder_ids is None:
        return {"gate": "G-D3", "name": "row order vs ladder", "ok": None,
                "skipped": "no ladder for this rung (CORE graph is its own id source)"}
    if len(model_ids) != len(ladder_ids):
        return {"gate": "G-D3", "name": "row order vs ladder", "ok": False,
                "reason": f"length {len(model_ids)} != ladder {len(ladder_ids)}"}
    rng = np.random.default_rng(seed)
    probe = rng.choice(len(model_ids), size=min(n_probe, len(model_ids)), replace=False)
    bad = [int(i) for i in probe if str(model_ids[i]) != str(ladder_ids[i])]
    full = [i for i in range(len(model_ids)) if str(model_ids[i]) != str(ladder_ids[i])]
    return {"gate": "G-D3", "name": "row order vs ladder", "ok": not full,
            "n_probed": int(len(probe)), "probe_mismatches": bad[:10],
            "total_mismatches": len(full)}


# ── export ───────────────────────────────────────────────────────────────────

def resolve_ckpt(run_dir, which):
    if which and os.path.isfile(which):
        return which
    name = {"best": CK.BEST, "last": CK.LAST}.get(which or "best", which)
    p = os.path.join(run_dir, "ckpt", name)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"no checkpoint at {p}")
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a T6 run directory")
    ap.add_argument("--rung", default=None, help="default: read from MANIFEST")
    ap.add_argument("--graph", default=None, help="default: read from MANIFEST")
    ap.add_argument("--ckpt", default="best", help="best | last | explicit path")
    ap.add_argument("--out", default=None, help="default: <run>/exports")
    ap.add_argument("--ladder", default=None, help="ladder csv for G-D3")
    ap.add_argument("--chunk", type=int, default=50_000)
    ap.add_argument("--hnsw-M", type=int, default=32)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--hnsw-threads", type=int, default=None)
    ap.add_argument("--iso-recall", type=float, default=0.99)
    ap.add_argument("--device", default=None)
    ap.add_argument("--verify-chunked", dest="verify_chunked",
                    action="store_true", default=None,
                    help="also run the whole-graph forward and assert "
                         "max|delta| < 1e-5 (G-A2). Default: on below 50k nodes.")
    ap.add_argument("--no-verify-chunked", dest="verify_chunked",
                    action="store_false")
    args = ap.parse_args(argv)

    from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import apply_similar_to_mode
    from ModelLakeFishing.stage2TrainGraphSAGE.d0_splits import make_root_aware_splits
    from ModelLakeFishing.stage2TrainGraphSAGE.losses import accuracy_lookup
    from ModelLakeFishing.stage2TrainGraphSAGE.ablation import build_models
    from ModelLakeFishing.stage2TrainGraphSAGE.inference import chunked_forward
    from ModelLakeFishing.stage2TrainGraphSAGE.top1_audit import model_names, candidates
    from ModelLakeFishing.stage2TrainGraphSAGE.top1_eval import five_metric_eval, aggregate
    from ModelLakeFishing.scale import global_metrics as GM
    from ModelLakeFishing.scale.export_ours import build_hnsw

    run = os.path.abspath(args.run)
    with open(os.path.join(run, "MANIFEST.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    rung = args.rung or man["rung"]
    graph_path = args.graph or man["graph"]
    out = args.out or os.path.join(run, "exports")
    os.makedirs(out, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    expect_n = RUNGS.get(rung, {}).get("expect_n")

    # ── the binding check, before any compute is spent ──────────────────────
    ck_path = resolve_ckpt(run, args.ckpt)
    ck = CK.load(ck_path)
    binding = dict(ck["binding"])
    graph_sha = CK.sha256_of(graph_path)
    if graph_sha != binding.get("graph_sha256"):
        raise CK.IncompatibleCheckpoint(
            "checkpoint was trained on graph %s but %s hashes to %s.\n"
            "Exporting across a rebuild silently mislabels every embedding; "
            "point --graph at the graph this checkpoint was bound to."
            % (binding.get("graph_sha256"), graph_path, graph_sha))

    print(f"[export] rung={rung} run={os.path.basename(run)} ckpt={os.path.basename(ck_path)} "
          f"device={device}", flush=True)
    t0 = time.time()

    payload = torch.load(graph_path, map_location="cpu", weights_only=False)
    data, xm0, xd0 = payload["data"], payload["xm0_meta"], payload["xd0_meta"]
    udi = payload["unique_dataset_id"].sort_values("mappedID").reset_index(drop=True)
    umi = payload["unique_model_id"].sort_values("mappedID").reset_index(drop=True)
    root_of = udi["root"].astype(str).tolist()
    n_models = int(data["model"].num_nodes)

    cfg = dict(ck["cfg"])
    # split_seed lives in the binding, not in a flag: the checkpoint knows which
    # test set it was held out from, and re-deriving it from a CLI default is
    # how an "export" quietly evaluates on training edges.
    split_seed = int(binding["split_seed"])
    data = apply_similar_to_mode(data, cfg["similar_to_mode"], k=cfg["similar_to_k"])
    split = make_root_aware_splits(data, root_of, split_seed=split_seed)
    _tr, _val, test_data = split
    lookup = accuracy_lookup(data)
    names = model_names(payload["unique_model_id"])

    model, scorer = build_models(data, xm0, xd0, cfg, device=device)
    model.load_state_dict(ck["model"])          # strict: a partial load would
    scorer.load_state_dict(ck["scorer"])        # export a half-random model
    model.eval()

    # ── the two forwards ───────────────────────────────────────────────────
    with torch.no_grad():
        z_eval = chunked_forward(model, test_data, chunk_size=args.chunk, device=device)
        z_full = chunked_forward(model, data, chunk_size=args.chunk, device=device)
    verify = args.verify_chunked
    if verify is None:
        verify = n_models <= 50_000
    chunk_check = {"ran": bool(verify)}
    if verify:
        with torch.no_grad():
            whole = {k: v.cpu() for k, v in model(data.clone().to(device)).items()}
        deltas = {k: float((whole[k] - z_full[k]).abs().max()) for k in whole}
        chunk_check.update(max_abs_delta=deltas,
                           ok=bool(max(deltas.values()) < 1e-5))
        del whole

    z_m = z_full["model"].numpy().astype(np.float32)
    z_d = z_full["dataset"].numpy().astype(np.float32)
    z_m_eval = z_eval["model"].numpy().astype(np.float32)
    z_d_eval = z_eval["dataset"].numpy().astype(np.float32)

    # ── metrics on BOTH, so the leakage gate has something to compare ──────
    cands = candidates(test_data, lookup)
    per_eval = five_metric_eval(z_eval, cands, names=names, expect_n=expect_n)
    agg_eval = aggregate(per_eval)
    per_full = five_metric_eval(z_full, cands, names=names, expect_n=expect_n)
    agg_full = aggregate(per_full)

    gm_cands = {int(d): (np.asarray(c), np.asarray(a)) for d, (c, a) in cands.items()}
    roots_q = {int(d): root_of[int(d)] for d in cands}
    gm_agg, _ = GM.from_embeddings_streaming(z_m_eval, z_d_eval, gm_cands, roots_q)
    parity = {"five_metric_gold@10": agg_eval["full2k_gold@10"],
              "global_metrics_gold@10": gm_agg["gold@10"],
              "match": abs(agg_eval["full2k_gold@10"] - gm_agg["gold@10"]) < 1e-9}

    # §11.1's two derived columns. median_rank/N is the one the plan calls the
    # core column rather than gold@10: it is the gold's *relative* position in
    # the pool, which is the only thing that stays comparable as N grows.
    a_row = {
        "N": n_models,
        "gold@1": gm_agg["gold@1"], "gold@10": gm_agg["gold@10"],
        "top3@10": gm_agg["top3@10"], "gold-gap@10": gm_agg["gold-gap@10"],
        "root_gold@10": gm_agg["root_gold@10"],
        "median_gold_rank": gm_agg["median_gold_rank"],
        "median_rank_over_N": gm_agg["median_gold_rank"] / n_models,
        "vs_random": gm_agg["gold@10"] / (10.0 / n_models),
        "n_queries": gm_agg["n_queries"], "n_roots": gm_agg["n_roots"],
    }

    # ── gates ──────────────────────────────────────────────────────────────
    ladder_ids = None
    if args.ladder and os.path.isfile(args.ladder):
        import pandas as pd
        ladder_ids = (pd.read_csv(args.ladder).sort_values("mappedID")["model"]
                      .astype(str).tolist())
    gates = [gate_pool_size(n_models, expect_n),
             gate_leakage(agg_eval["full2k_gold@10"], agg_full["full2k_gold@10"]),
             gate_row_order(umi["model"].astype(str).tolist(), ladder_ids)]
    if chunk_check.get("ran"):
        gates.append({"gate": "G-A2", "name": "chunked == whole-graph forward",
                      "ok": chunk_check["ok"], **chunk_check})
    gates.append({"gate": "G-A3", "name": "five_metric == global_metrics",
                  "ok": bool(parity["match"]), **parity})

    # ── HNSW (index the SERVING embeddings, tune ef to iso-recall) ─────────
    idx, hnsw = build_hnsw(z_m, z_d, cands, ef=args.ef_construction, M=args.hnsw_M,
                           threads=args.hnsw_threads, iso_recall=args.iso_recall)
    idx.save_index(os.path.join(out, f"hnsw_{rung}.bin"))
    if hnsw.get("ef_trace"):
        write_json_atomic(os.path.join(out, "ef_tuning.json"),
                          {"target": hnsw["iso_recall"], "ef_search": hnsw["ef_search"],
                           "trace": hnsw["ef_trace"]})
    gates.append({"gate": "G-D4", "name": "HNSW recall@50 >= 0.99",
                  "ok": bool(hnsw["recall_at_50"] >= 0.99),
                  "recall_at_50": hnsw["recall_at_50"], "ef_search": hnsw["ef_search"]})

    # ── artifacts ──────────────────────────────────────────────────────────
    np.save(os.path.join(out, "z_m.npy"), z_m)
    np.save(os.path.join(out, "z_d.npy"), z_d)
    np.save(os.path.join(out, "z_m_eval.npy"), z_m_eval)
    np.save(os.path.join(out, "z_d_eval.npy"), z_d_eval)
    umi.to_csv(os.path.join(out, "model_ids.csv"), index=False)
    udi.to_csv(os.path.join(out, "dataset_ids.csv"), index=False)
    np.savez_compressed(
        os.path.join(out, "gold_cands.npz"),
        **{str(d): np.stack([np.asarray(c, float), np.asarray(a, float)])
           for d, (c, a) in cands.items()})

    failed = [g["gate"] for g in gates if g.get("ok") is False]
    manifest = {
        "written_at": utcnow(), "rung": rung, "run": run,
        "run_id": man.get("run_id"), "checkpoint": ck_path,
        "checkpoint_epoch": ck.get("epoch"),
        "graph": os.path.abspath(graph_path), "graph_sha256": graph_sha,
        "binding": binding, "split_seed": split_seed,
        "n_models": n_models, "n_datasets": int(data["dataset"].num_nodes),
        "n_test_datasets": len(cands), "expect_n": expect_n,
        "device": device, "export_sec": round(time.time() - t0, 1),
        "eval_metrics": agg_eval, "full_graph_metrics": agg_full,
        "global_metrics": gm_agg, "a_axis_row": a_row, "hnsw": hnsw,
        "gates": gates, "gates_passed": not failed, "gates_failed": failed,
        "artifacts": sorted(os.listdir(out) + ["EXPORT_MANIFEST.json"]),
    }
    write_json_atomic(os.path.join(out, "EXPORT_MANIFEST.json"), manifest)

    print(f"\n=== T7 EXPORT {rung} / {man.get('run_id')} ===")
    print(f"  gold@10  held-out {agg_eval['full2k_gold@10']:.4f}   "
          f"full-graph {agg_full['full2k_gold@10']:.4f}   "
          f"(A-axis uses the held-out one)")
    print(f"  gold@1={a_row['gold@1']:.4f}  top3@10={a_row['top3@10']:.4f}  "
          f"root_gold@10={a_row['root_gold@10']:.4f}  gold-gap@10={a_row['gold-gap@10']:.4f}")
    print(f"  median gold rank={a_row['median_gold_rank']:.0f}/{n_models} "
          f"(rank/N={a_row['median_rank_over_N']:.3e})  vs random={a_row['vs_random']:.1f}x  "
          f"queries={a_row['n_queries']}")
    print(f"  HNSW recall@50={hnsw['recall_at_50']:.4f} @ ef={hnsw['ef_search']} "
          f"p50={hnsw['hnsw_query_ms_p50']:.4f}ms")
    for g in gates:
        mark = {True: "PASS", False: "FAIL", None: "skip"}[g.get("ok")]
        print(f"  [{mark}] {g['gate']} {g['name']}")
    print(f"  -> {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
