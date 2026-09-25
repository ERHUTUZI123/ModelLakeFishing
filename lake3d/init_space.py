"""init_space.py -- the A0 seed-0 encoder before its first training step.

The viewer's "before" is not a random cloud drawn for effect.  It is the same
encoder A0 trained, holding the weights it started from, run over the same
held-out message graph through the same export path -- so the only thing that
differs between the two spaces is the 25 epochs of training.

This is the one file in lake3d that imports ModelLakeFishing: rebuilding the
start of training needs the training code itself.  `build.py` stays standalone
and reads what this writes, the same way it reads the A0 export.

    export  rebuild the initial encoder and check it against the trained
            checkpoint, check the forward path against the archived export,
            then embed all 3,016,439 models and 18,729 queries
    eval    score both spaces with the A0 exact evaluator: the trained export
            must reproduce the recorded seed-0 rows before the untrained
            export is scored by the same code

    python init_space.py --stage all

Why the start can be rebuilt exactly.  `train_eval_one` seeds torch with the
initialisation seed (0, recorded as `init_seed` in the run manifest) and then
builds the model on the CPU before moving it to the GPU, so the initial
weights come from the CPU generator alone.  The check does not rely on that
argument: Adam without weight decay leaves a parameter that never received a
gradient exactly where it started, and the saved optimizer state says which
ones those are.  Every such element in the trained checkpoint has to match the
rebuilt initialisation, and a rebuild from any other seed has to match none.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
# Register the repository's existing namespace alias, also in renamed checkouts.
import scale1m

DATA = os.path.join(REPO, os.environ.get("LAKE3D_DATA_ROOT", "data/data1m"))
A0 = os.path.join(REPO, os.environ.get("LAKE3D_A0_ROOT",
                                     os.path.join(DATA, "a0_20260912")))
GRAPH = os.path.join(A0, "graph")
EXPORT = os.path.join(A0, "exports", "A0GD_full_s0_e25")
REPORT = os.path.join(A0, "metrics", "A0_EVALUATION_REPORT.json")
RUN = os.path.join(REPO, os.environ.get("LAKE3D_RUN_DIR",
                                      "docs/1M/A0_runs/A0_4/delivery_s0"))
CKPT = os.path.join(RUN, "ckpt", "last.pt")
NODES = os.path.join(REPO, os.environ.get("LAKE3D_DATASET_NODES",
                                        os.path.join(DATA, "rf", "canon",
                                                     "dataset_nodes_merged.parquet")))
WORK = os.path.join(REPO, os.environ.get("LAKE3D_WORK",
                                       os.path.join(HERE, "payload")))
OUT = os.path.join(WORK, "before")

SPLIT_SEED = 0
CONTROL_SEED = 1          # a rebuild from this seed must match nothing
CHUNK = 50_000            # the export's own chunk, so every subgraph is its own
# chunks re-run with the trained weights and compared with the archived export
PARITY_STARTS = (0, 1_500_000, 3_000_000)
PARITY_TOL = 1e-4
# Absolute, not relative.  An initialiser computes each value from terms of
# its table's own scale -- the uniform bound for a linear layer, the
# Box-Muller radius for an embedding -- so where the training node's CPU and
# this one round differently in the last place, the error is about one ulp of
# that scale however small the value it lands on.  Two float32 epsilons covers
# the order-one embedding tables; a rebuild from the wrong seed misses by
# order one.
INIT_TOL = 2 * float(np.finfo(np.float32).eps)
# the graph, checkpoint and query table A0 recorded, so a stale copy stops here
GRAPH_DIGEST = "acddeb93d926efcb636c83d452fe360736a0e6e4ac68cab3f5fef210e7cef5db"
CKPT_SHA = "9ee3415caf361e0ed50126c0d7992eb0989284282ebee8ed097addc6375650fb"
NODES_SHA = "763201d8e6e103a664b658ec7148b657d28f1d0cdc6f70a54b06687122dbdc57"
CAST_QUERY = 15473
# the rows the evaluator reports; the trained export has to reproduce them
ROWS = ("G_dense", "G_full_task", "G_exact1000_task")
KEYS = ("gold@1", "gold@10", "top3@10", "gold-gap@10", "root_gold@10",
        "median_gold_rank", "gold_in_first_stage@1000",
        "median_gold_rank_if_retrieved", "n_queries")


def say(*a):
    print("[%7.1fs]" % (time.time() - say.t0), *a, flush=True)


say.t0 = time.time()


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ inputs --
def load_inputs():
    """The graph, the trained checkpoint and the held-out message graph."""
    from scale1m import checkpoint as CK
    from scale1m.graph_store import load_sharded
    from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import apply_similar_to_mode
    from ModelLakeFishing.stage2TrainGraphSAGE.d0_splits import make_root_aware_splits

    digest = CK.graph_digest(GRAPH)
    if digest != GRAPH_DIGEST:
        raise SystemExit("graph digests to %s, A0 recorded %s" % (digest, GRAPH_DIGEST))
    got = sha256(CKPT)
    if got != CKPT_SHA:
        raise SystemExit("checkpoint sha %s is not the one A0 exported (%s)" % (got, CKPT_SHA))
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    records = _read_json(os.path.join(RUN, "A0_RUN_RECORDS.json"))
    manifest = _read_json(os.path.join(RUN, "MANIFEST.json"))
    init_seed = int(manifest["init_seed"])
    assert records["initialization_seed"] == init_seed, "run records disagree on the init seed"
    assert records["a0"]["checkpoint_sha256"] == CKPT_SHA
    assert int(ck["binding"]["split_seed"]) == SPLIT_SEED
    assert ck["binding"]["graph_sha256"] == GRAPH_DIGEST
    assert int(ck["epoch"]) == 24 and len(ck["history"]) == 25
    say("graph %s…, checkpoint %s…, init seed %d, split seed %d"
        % (digest[:12], got[:12], init_seed, SPLIT_SEED))

    payload = load_sharded(GRAPH, mmap=True, verify_sha256=False)
    data, xm0, xd0 = payload["data"], payload["xm0_meta"], payload["xd0_meta"]
    udi = payload["unique_dataset_id"].sort_values("mappedID").reset_index(drop=True)
    root_of = udi["root"].astype(str).tolist()
    cfg = dict(ck["cfg"])

    # The surgery and the split each clone the whole graph, three times over in
    # the split, and the model features are 5.4 GB behind a memory map.  Both
    # read only edges and node counts, so the features are stood aside for a
    # zero-width tensor of the same length while they run and put back after.
    xm = data["model"].x
    n_models = int(xm.shape[0])
    data["model"].x = torch.empty((n_models, 0), dtype=xm.dtype)
    data = apply_similar_to_mode(data, cfg["similar_to_mode"], k=cfg["similar_to_k"])
    _tr, _val, test = make_root_aware_splits(data, root_of, split_seed=SPLIT_SEED)
    del _tr, _val
    data["model"].x = xm
    test["model"].x = xm
    assert data.metadata() == test.metadata()
    assert int(test["model"].num_nodes) == n_models
    say("held-out message graph: %s trained_on edges of %s"
        % (f"{test['model', 'trained_on', 'dataset'].edge_index.shape[1]:,}",
           f"{data['model', 'trained_on', 'dataset'].edge_index.shape[1]:,}"))
    return SimpleNamespace(ck=ck, cfg=cfg, data=data, test=test, xm0=xm0, xd0=xd0,
                           init_seed=init_seed, digest=digest)


def build(inp, seed):
    """Construct the encoder exactly as `train_eval_one` does."""
    from ModelLakeFishing.stage2TrainGraphSAGE.ablation import build_models
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, _scorer = build_models(inp.data, inp.xm0, inp.xd0, inp.cfg, device="cpu")
    return model


# ---------------------------------------------------------- init is the init --
def check_init(inp, model, label):
    """Compare a rebuild with every element training never moved.

    The optimizer was built over model.parameters() then scorer.parameters(),
    so its state index i is the i-th named parameter.  An element with both
    Adam moments exactly zero never received a gradient; under Adam without
    weight decay it is still its initial value.  (The first moment alone can
    underflow to zero after a single early gradient; the second decays at
    0.999 per step and does not.)
    """
    opt = inp.ck["opt"]
    assert all(g.get("weight_decay", 0) == 0 for g in opt["param_groups"]), "Adam had weight decay"
    trained = inp.ck["model"]
    out = {}
    n_never = n_bit = n_ulp = 0
    worst = 0.0
    for i, (name, p) in enumerate(model.named_parameters()):
        t = trained[name]
        assert tuple(p.shape) == tuple(t.shape), name
        st = opt["state"].get(i)
        if st is None:                   # Adam never stepped it at all
            never = torch.ones_like(t, dtype=torch.bool)
        else:
            assert tuple(st["exp_avg"].shape) == tuple(t.shape), name
            never = (st["exp_avg"] == 0) & (st["exp_avg_sq"] == 0)
        k = int(never.sum())
        if not k:
            continue
        a, b = p.detach()[never], t[never]
        bit = int((a == b).sum())
        near = int(((a - b).abs() <= INIT_TOL).sum())
        diff = float((a - b).abs().max())
        out[name] = {"never_updated": k, "bit_identical": bit,
                     "within_tolerance": near, "max_abs_diff": diff}
        n_never += k
        n_bit += bit
        n_ulp += near
        worst = max(worst, diff)
    say("%s: %s never-updated elements, %s bit-identical, %s within %.1e, "
        "max |diff| %.3g" % (label, f"{n_never:,}", f"{n_bit:,}", f"{n_ulp:,}",
                             INIT_TOL, worst))
    for name, r in out.items():
        say("    %-50s %6d never moved, %6d identical, max |diff| %.3g"
            % (name, r["never_updated"], r["bit_identical"], r["max_abs_diff"]))
    return {"by_parameter": out, "never_updated": n_never, "bit_identical": n_bit,
            "within_tolerance": n_ulp, "tolerance": INIT_TOL, "max_abs_diff": worst}


# --------------------------------------------------------- forward is export --
def forward_chunk(model, data, nt, start, csr, device):
    """One chunk of `chunked_forward`, verbatim, so a chunk can be re-run alone."""
    from ModelLakeFishing.stage2TrainGraphSAGE.inference import _closure
    n = int(data[nt].num_nodes)
    seed = torch.arange(start, min(start + CHUNK, n), dtype=torch.long)
    ids = _closure(data, {nt: seed}, model.num_layers, csr)
    for t in list(ids):
        if ids[t].numel() == 0:
            ids[t] = torch.zeros(1, dtype=torch.long)
    sub = data.subgraph(dict(ids))
    z = model(sub.clone().to(device))
    g2l = torch.full((n,), -1, dtype=torch.long)
    g2l[ids[nt]] = torch.arange(ids[nt].numel())
    return seed, z[nt][g2l[seed].to(z[nt].device)].cpu()


def check_path(inp, device):
    """The trained weights through this path must give back the archived rows."""
    from ModelLakeFishing.stage2TrainGraphSAGE.ablation import build_models
    from ModelLakeFishing.stage2TrainGraphSAGE.sampling import build_csr
    hashes = _read_json(os.path.join(EXPORT, "EXPORT_MANIFEST.json"))["stages"]["embed"]["artifact_hashes"]
    for name in ("z_m_eval.npy", "z_d_eval.npy"):
        got = sha256(os.path.join(EXPORT, name))
        if got != hashes[name]:
            raise SystemExit("%s sha %s differs from its export manifest" % (name, got))
    zm = np.load(os.path.join(EXPORT, "z_m_eval.npy"), mmap_mode="r")
    zd = np.load(os.path.join(EXPORT, "z_d_eval.npy"), mmap_mode="r")

    model, _ = build_models(inp.data, inp.xm0, inp.xd0, inp.cfg, device="cpu")
    model.load_state_dict(inp.ck["model"])
    model = model.to(device).eval()
    test = inp.test
    n_nodes = {t: test[t].num_nodes for t in test.node_types}
    csr = {et: (build_csr(test[et].edge_index, n_nodes[et[0]]),
                build_csr(test[et].edge_index.flip(0), n_nodes[et[2]]))
           for et in test.edge_types}
    rows = []
    with torch.no_grad():
        for nt, start, ref in ([("model", s, zm) for s in PARITY_STARTS]
                               + [("dataset", 0, zd)]):
            seed, z = forward_chunk(model, test, nt, start, csr, device)
            want = torch.as_tensor(np.asarray(ref[int(seed[0]):int(seed[-1]) + 1]))
            d = float((z - want).abs().max())
            rows.append({"node_type": nt, "start": int(seed[0]), "n": int(seed.numel()),
                         "max_abs_diff": d})
            say("  path check %s [%s, %s): max |diff| %.3g"
                % (nt, f"{int(seed[0]):,}", f"{int(seed[-1]) + 1:,}", d))
    worst = max(r["max_abs_diff"] for r in rows)
    if worst > PARITY_TOL:
        raise SystemExit("the forward path does not reproduce the archived export "
                         "(max |diff| %.3g > %.0e)" % (worst, PARITY_TOL))
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"chunks": rows, "max_abs_diff": worst, "tolerance": PARITY_TOL,
            "archived_hashes_verified": True}


# ------------------------------------------------------------------- stages --
def stage_export():
    from ModelLakeFishing.stage2TrainGraphSAGE.inference import chunked_forward
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inp = load_inputs()

    model = build(inp, inp.init_seed)
    init_check = check_init(inp, model, "init seed %d" % inp.init_seed)
    control = check_init(inp, build(inp, CONTROL_SEED), "control seed %d" % CONTROL_SEED)
    # the rebuild has to agree with every never-moved element to within float32
    # rounding -- the two CPUs round the initialisers' arithmetic differently
    # in the last place -- and the control has to fail
    if not (init_check["never_updated"] > 0
            and init_check["within_tolerance"] == init_check["never_updated"]
            and init_check["bit_identical"] >= 0.5 * init_check["never_updated"]):
        raise SystemExit("the rebuilt initialisation does not match the elements "
                         "training never moved")
    if control["bit_identical"] > 0 or control["max_abs_diff"] <= 1e-3:
        raise SystemExit("the control seed also matches; the check proves nothing")

    path = check_path(inp, device)

    say("embedding with the initial weights on %s" % device)
    model = model.to(device).eval()
    t0 = time.time()
    with torch.no_grad():
        z = chunked_forward(model, inp.test, chunk_size=CHUNK, device=device)
    say("forward done in %.1fs" % (time.time() - t0))
    arrays = {}
    for name, nt in (("z_m_eval", "model"), ("z_d_eval", "dataset")):
        a = z[nt].numpy().astype(np.float32)
        assert np.isfinite(a).all(), name
        np.save(os.path.join(OUT, name + ".npy"), a)
        norm = np.linalg.norm(a, axis=1)
        arrays[name + ".npy"] = {"shape": list(a.shape),
                                 "sha256": sha256(os.path.join(OUT, name + ".npy")),
                                 "max_norm_error": float(np.abs(norm - 1).max())}
        say("  %s %s" % (name, a.shape))
    del z

    import torch_geometric
    _json(os.path.join(OUT, "INIT_MANIFEST.json"), {
        "what": "the A0 seed-0 encoder at its initialisation, run over the "
                "seed-0 held-out message graph through the export path",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "graph": GRAPH, "graph_digest": inp.digest,
        "checkpoint": CKPT, "checkpoint_sha256": CKPT_SHA,
        "run_id": inp.ck["extra"]["run_id"],
        "init_seed": inp.init_seed, "split_seed": SPLIT_SEED,
        "control_seed": CONTROL_SEED, "chunk": CHUNK,
        "init_check": init_check, "control_check": control,
        "path_check": path, "arrays": arrays,
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "torch_geometric": torch_geometric.__version__,
                        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                        "platform": platform.platform()},
    })
    say("export: done ->", OUT)


def stage_eval():
    """The A0 exact evaluator over both spaces, trained first as the control."""
    from ModelLakeFishing.scale1m import eval_y2 as E
    got = sha256(NODES)
    if got != NODES_SHA:
        raise SystemExit("query table sha %s is not the one A0 bound (%s)" % (got, NODES_SHA))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = SimpleNamespace(exports=os.path.dirname(EXPORT), run_fmt="A0GD_full_s%d_e25",
                           sidecar_exports=os.path.dirname(EXPORT),
                           sidecar_run_fmt="A0GD_full_s%d_e25", dataset_nodes=NODES)
    tie_rank = E._tie_ranks(E.N_TOTAL)
    bundle = E._audit_seed(args, SPLIT_SEED, None, None, tie_rank)
    recorded = _read_json(REPORT)["per_seed"][str(SPLIT_SEED)]["rows"]

    def keep(row):
        return {k: row.get(k) for k in KEYS if k in row}

    out = {"protocol": "A0 exact evaluator (scale1m.eval_y2.evaluate_exact_seed, "
                       "protocol a0), seed 0, eligible held-out queries",
           "query_table_sha256": got, "spaces": {}}
    for label, export_dir in (("after", EXPORT), ("before", OUT)):
        say("eval: %s training" % label)
        res = E.evaluate_exact_seed(export_dir, bundle["prior"], bundle["candidates"],
                                    bundle["roots"], tie_rank, device,
                                    model_chunk=50_000, query_chunk=16, protocol="a0")
        queries = np.asarray(res["queries"], dtype=np.int64)
        dense_rank = res["dense_counts"][:, 0] + 1          # 1 = the top model
        rows = {name: keep(res["rows"][name]) for name in ROWS}
        at = int(np.flatnonzero(queries == CAST_QUERY)[0])
        out["spaces"][label] = {
            "rows": rows,
            "dense_gold_rank": {"median": float(np.median(dense_rank)),
                                "p25": float(np.percentile(dense_rank, 25)),
                                "p75": float(np.percentile(dense_rank, 75)),
                                "within_1000": int((dense_rank <= 1000).sum()),
                                "n_queries": int(len(queries))},
            "cast": {"query": CAST_QUERY, "dense_gold_rank": int(dense_rank[at]),
                     "gold_in_exact_pool": bool(dense_rank[at] <= E.POOL_K)},
            "seconds": round(float(res["seconds"]), 1),
        }
        np.savez(os.path.join(OUT, "eval_%s.npz" % label), query=queries,
                 dense_gold_rank=dense_rank.astype(np.int64),
                 exact_pool=res["exact_pool_ids"], full_top10=res["full_top10"])
        if label == "after":
            # the control: this machine and this call against the A0 record
            match = {}
            for name in ROWS:
                for k, v in rows[name].items():
                    r = recorded[name].get(k)
                    match["%s.%s" % (name, k)] = {"recorded": r, "recomputed": v,
                                                  "equal": r == v}
            out["after_reproduces_record"] = match
            bad = [k for k, m in match.items() if not m["equal"]]
            say("eval: trained export vs A0 record: %d of %d values equal%s"
                % (len(match) - len(bad), len(match),
                   "" if not bad else "; differ: " + ", ".join(bad)))
            if any(k.endswith("gold@10") for k in bad):
                raise SystemExit("the trained export does not reproduce A0's gold@10 "
                                 "here, so the untrained score would not be comparable")
        for name in ROWS:
            say("  %-17s %s" % (name, json.dumps(rows[name])))
        del res
        if device == "cuda":
            torch.cuda.empty_cache()
    bundle["prior"].payload.close()
    _json(os.path.join(OUT, "EVAL.json"), out)
    say("eval: done ->", os.path.join(OUT, "EVAL.json"))


STAGES = {"export": stage_export, "eval": stage_eval}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all"] + list(STAGES))
    a = ap.parse_args()
    for s in (list(STAGES) if a.stage == "all" else [a.stage]):
        say("=== stage", s, "===")
        STAGES[s]()
    say("done")


if __name__ == "__main__":
    main()
