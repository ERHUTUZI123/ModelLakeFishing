import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(_HERE), ".."))
_GIT_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scale1m import checkpoint as CK
from scale1m.hf_crawl import utcnow, write_json_atomic

_NO_GIT = "not-a-git-repo"
SMOKE_MARKER = "SMOKE_ONLY.json"
A0_RUN_RECORDS = "A0_RUN_RECORDS.json"

RUNGS = {
    "12k": {"expect_n": None, "anchor_gold10": 0.4159},
    "30k": {"expect_n": 30_183, "anchor_gold10": None},
    "100k": {"expect_n": 100_000, "anchor_gold10": None},
    "500k": {"expect_n": None, "anchor_gold10": None},
    "full": {"expect_n": 3_016_439, "anchor_gold10": None},
    "live": {"expect_n": None, "anchor_gold10": None},
}


def make_run_dir(out, *, smoke_only=False):
    prior = []
    for rel in ("MANIFEST.json", os.path.join("stdout", "train.log")):
        p = os.path.join(out, rel)
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            prior.append(rel.replace(os.sep, "/"))
    ckpt = os.path.join(out, "ckpt")
    if os.path.isdir(ckpt) and any(f.endswith(".pt") for f in os.listdir(ckpt)):
        prior.append("ckpt/*.pt")
    subdirs = (("smoke_ckpt", "metrics", "stdout", "metadata") if smoke_only
               else ("ckpt", "exports", "metrics", "stdout", "metadata"))
    for sub in subdirs:
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    return out, prior


def validate_run_mode(args):
    if args.smoke_only:
        if args.epochs != 1:
            raise ValueError("--smoke-only requires --epochs 1")
        if args.resume is not None:
            raise ValueError("--smoke-only starts fresh and cannot use --resume")
        if os.path.exists(args.out):
            raise ValueError("--smoke-only requires a new output directory")
    elif os.path.isfile(os.path.join(args.out, SMOKE_MARKER)):
        raise ValueError("a smoke-only directory cannot be used for formal training")


def reject_smoke_checkpoint(ck):
    extra = ck.get("extra", {})
    if (extra.get("smoke_only") or extra.get("run_purpose") == "smoke_only"
            or extra.get("formal_training_eligible") is False):
        raise CK.IncompatibleCheckpoint("a smoke-only checkpoint cannot initialize or resume formal training")


def capture_metadata(out, args, cfg, graph_path):
    md = os.path.join(out, "metadata")
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--short")
    tracked_dirty = _git("status", "--porcelain", "--untracked-files=no")
    meta = {
        "utc_time": utcnow(),
        "run_id": os.path.basename(os.path.abspath(out)),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory
                              / 2 ** 30, 2) if torch.cuda.is_available() else None,
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "account": os.environ.get("SLURM_JOB_ACCOUNT"),
        "command": " ".join(sys.argv),
        "git_head": head if head is not None else _NO_GIT,
        "git_status": status if status is not None else _NO_GIT,
        "git_tracked_dirty": bool(tracked_dirty) if tracked_dirty is not None else None,
        "graph": os.path.abspath(graph_path),
        "graph_sha256": CK.graph_digest(graph_path),
        "args": vars(args),
        "resolved_config": CK._jsonable(cfg),
    }
    write_json_atomic(os.path.join(md, "resolved_config.json"), meta)
    if tracked_dirty:
        patch = _git("diff", "HEAD")
        with open(os.path.join(md, "uncommitted.patch"), "w", encoding="utf-8") as fh:
            fh.write(patch or "")
    return meta


def _git(*a):
    try:
        r = subprocess.run(["git", *a], cwd=_GIT_ROOT, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def mechanism_gate(history, model):
    tot = [h["total"] for h in history]
    enc = getattr(model, "model_encoder", None)
    out = {
        "epochs": len(tot),
        "loss_first": float(tot[0]) if tot else None,
        "loss_last": float(tot[-1]) if tot else None,
        "loss_descended": bool(len(tot) > 1 and tot[-1] < tot[0]),
        "no_nan": bool(all(np.isfinite([v for v in h.values()
                                        if isinstance(v, (int, float))]).all()
                           for h in history)),
        "size_embedding_moved": None,
        "family_embedding_moved": None,
    }
    if enc is not None:
        for name, tag in (("size_embedding", "size_embedding_moved"),
                          ("family_embedding", "family_embedding_moved")):
            t = getattr(enc, name, None)
            out[tag] = bool(t is not None and float(t.weight.detach().abs().sum()) > 0)
    out["passed"] = bool(out["loss_descended"] and out["no_nan"])
    return out


def smoke_gate(history, model, scorer, opt):
    numeric = [v for h in history for v in h.values() if isinstance(v, (int, float))]
    params = [p for module in (model, scorer) for p in module.parameters() if p.requires_grad]
    grads = [p.grad for p in params if p.grad is not None]
    optimizer_values = [v for state in (opt.state.values() if opt is not None else [])
                        for v in state.values() if torch.is_tensor(v)]
    steps = [float(state["step"].item() if torch.is_tensor(state["step"]) else state["step"])
             for state in (opt.state.values() if opt is not None else []) if "step" in state]
    out = {
        "epochs": len(history), "one_epoch_completed": len(history) == 1,
        "loss_first": float(history[0]["total"]) if history and "total" in history[0] else None,
        "loss_last": float(history[-1]["total"]) if history and "total" in history[-1] else None,
        "loss_descended": None, "loss_descent_required": False,
        "no_nan": bool(numeric and np.isfinite(numeric).all()),
        "parameters_finite": bool(params and all(torch.isfinite(p).all().item() for p in params)),
        "gradient_tensors": len(grads),
        "gradients_finite": bool(grads and all(torch.isfinite(g).all().item() for g in grads)),
        "nonzero_gradient_tensors": sum(bool(torch.any(g != 0).item()) for g in grads),
        "optimizer_steps_max": max(steps) if steps else 0,
        "optimizer_state_finite": bool(optimizer_values and all(torch.isfinite(v).all().item() for v in optimizer_values)),
    }
    out["passed"] = bool(out["one_epoch_completed"] and out["loss_last"] is not None
                         and out["no_nan"] and out["parameters_finite"]
                         and out["gradients_finite"] and out["nonzero_gradient_tensors"] > 0
                         and out["optimizer_steps_max"] > 0 and out["optimizer_state_finite"])
    return out


def smoke_checkpoint_roundtrip(path, model, scorer, binding):
    ck = CK.load(path)
    return bool(ck.get("extra", {}).get("smoke_only") and not CK.validate(ck, binding)
                and all(set(ck[key]) == set(module.state_dict())
                        and all(torch.equal(value.detach().cpu(), ck[key][name].cpu())
                                for name, value in module.state_dict().items())
                        for key, module in (("model", model), ("scorer", scorer))))


def peak_process_rss():
    try:
        if sys.platform.startswith("win"):
            import psutil
            return int(psutil.Process().memory_info().peak_wset), None
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024), None
    except (ImportError, AttributeError, OSError, ValueError) as exc:
        return None, "%s: %s" % (type(exc).__name__, exc)


def capture_runtime_environment():
    cuda = torch.cuda.is_available()
    return {"captured_at": utcnow(), "hostname": platform.node(),
            "platform": platform.platform(), "python": sys.version.split()[0],
            "python_executable": sys.executable, "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda, "cuda_available": cuda,
            "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
            "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory) if cuda else None,
            "cpu_count": os.cpu_count(), "processor": platform.processor() or None}


def begin_a0_run_records(out, cfg, binding, run_name, seed, init_seed, resume_path):
    path = os.path.join(out, A0_RUN_RECORDS)
    identity = {"protocol": "a0", "run_id": "A0_20260912", "seed": seed,
                "graph_digest": binding["graph_sha256"]}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            records = json.load(handle)
        if (any(records.get("a0", {}).get(k) != v for k, v in identity.items())
                or records.get("formal_run_name") != run_name
                or records.get("resolved_config") != CK._jsonable(cfg)
                or records.get("epochs") != 25
                or records.get("initialization_seed") != init_seed):
            raise CK.IncompatibleCheckpoint("A0 run records belong to a different graph, seed, run or recipe")
        if resume_path is None and records.get("a0", {}).get("checkpoint_sha256"):
            raise CK.IncompatibleCheckpoint("A0 run records contain a checkpoint but this invocation starts fresh")
        if resume_path and records.get("a0", {}).get("checkpoint_sha256") != CK.sha256_of(resume_path):
            raise CK.IncompatibleCheckpoint("A0 resume checkpoint differs from the recorded same-run checkpoint")
        for segment in records.get("train_segments", []):
            if segment.get("end_ns") is None:
                segment["status"] = "interrupted_end_time_unavailable"
    else:
        if resume_path is not None:
            raise CK.IncompatibleCheckpoint("A0 resume requires the existing same-run A0_RUN_RECORDS.json")
        records = {"schema": "a0.run_records.v1", "a0": {**identity, "checkpoint_sha256": None},
                   "formal_run_name": run_name, "epochs": 25, "initialization_seed": init_seed,
                   "resolved_config": CK._jsonable(cfg), "history": [], "train_segments": [],
                   "created_at": utcnow()}
    records["runtime_environment"] = capture_runtime_environment()
    records["runtime_environment_scope"] = "current invocation; each train segment retains the environment observed for its own invocation"
    records["status"] = "running"
    records["timing_scope"] = "driver segment around native train_eval_one: setup, training with epoch checkpoints, existing post-training test/full evaluation, and the segment-start log write"
    records["peak_process_rss_scope"] = "OS process lifetime high-water mark; cumulative maximum across recorded process segments"
    return records


def build_config(args):
    from ModelLakeFishing.scale.export_ours import l1l3b_config
    cfg = l1l3b_config()
    cfg.update(fanout=args.fanout, sparse_M=args.sparse_M,
               contrast_n_neg=args.contrast_n_neg,
               contrast_max_pos_per_dataset=args.contrast_max_pos_per_dataset,
               infer_chunk=args.chunked_infer,
               skip_diagnostics=args.skip_diagnostics)
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.lake_gamma is not None:
        cfg["lake_gamma"] = args.lake_gamma
    if args.global_n_datasets is not None:
        cfg["global_n_datasets"] = args.global_n_datasets
    if getattr(args, "num_layers", None) is not None:
        cfg["num_layers"] = args.num_layers
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser(description="T6: train one rung")
    p.add_argument("--rung", default="100k", choices=sorted(RUNGS))
    p.add_argument("--graph", required=True)
    p.add_argument("--out", required=True, help="run directory")
    p.add_argument("--seed", type=int, default=0, help="split seed")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--ckpt-every", type=int, default=5)
    p.add_argument("--ckpt-keep", type=int, default=3)
    p.add_argument("--family-vocab", default=None,
                   help="family_vocab.csv this graph was built with (binding)")
    p.add_argument("--expect-n", type=int, default=None)
    p.add_argument("--fanout", action="store_true")
    p.add_argument("--sparse-M", action="store_true")
    p.add_argument("--contrast-n-neg", type=int, default=None)
    p.add_argument("--contrast-max-pos-per-dataset", type=int, default=None)
    p.add_argument("--chunked-infer", type=int, default=None)
    p.add_argument("--skip-diagnostics", action="store_true")
    p.add_argument("--smoke-only", action="store_true",
                   help="one fresh epoch with the same training recipe; skip all test/full evaluation, write isolated smoke artifacts")
    p.add_argument("--lake-gamma", type=float, default=None,
                   help="mixture weight of the uniform-over-labeled component in q")
    p.add_argument("--global-n-datasets", type=int, default=None,
                   help="datasets scored by the global term each step (default 16)")
    p.add_argument("--num-layers", type=int, choices=(0, 1, 2), default=None,
                   help="message-passing depth; 0 is the X6 no-graph control")
    args = p.parse_args(argv)
    try:
        validate_run_mode(args)
    except ValueError as exc:
        p.error(str(exc))

    from ModelLakeFishing.stage2TrainGraphSAGE.ablation import train_eval_one
    from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import apply_similar_to_mode
    from ModelLakeFishing.stage2TrainGraphSAGE.top1_audit import INIT_SEED
    from ModelLakeFishing.stage2TrainGraphSAGE.d0_splits import make_root_aware_splits

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out, prior_contents = make_run_dir(args.out, smoke_only=args.smoke_only)
    if args.smoke_only:
        write_json_atomic(os.path.join(out, SMOKE_MARKER), {
            "run_purpose": "smoke_only", "started_at": utcnow(),
            "formal_training_eligible": False, "evaluation_performed": False,
        })
    if prior_contents:
        print("[run-dir] reusing a non-empty run directory; prior contents: %s"
              % ", ".join(prior_contents), flush=True)
    ckpt_dir = os.path.join(out, "smoke_ckpt" if args.smoke_only else "ckpt")
    expect_n = args.expect_n if args.expect_n is not None else RUNGS[args.rung]["expect_n"]
    if expect_n is not None and expect_n <= 0:
        raise ValueError("--expect-n must be positive")

    a0_graph_verification = None
    if os.path.isdir(args.graph):
        from scale1m.graph_store import load_sharded
        with open(os.path.join(args.graph, "meta.json"), encoding="utf-8") as handle:
            graph_meta = json.load(handle)
        if ("a0" in graph_meta or "A0_FEATURE_REPAIR.json" in graph_meta.get("files", {})
                or os.path.isfile(os.path.join(args.graph, "A0_FEATURE_REPAIR.json"))):
            from scale1m.a0_graph_validation import verify_a0_graph
            a0_graph_verification = verify_a0_graph(args.graph)
        payload = load_sharded(args.graph, mmap=True, verify_sha256=False)
    else:
        payload = torch.load(args.graph, map_location="cpu", weights_only=False)
    data, xm0, xd0 = payload["data"], payload["xm0_meta"], payload["xd0_meta"]
    udi = payload["unique_dataset_id"].sort_values("mappedID").reset_index(drop=True)
    root_of = udi["root"].astype(str).tolist()

    n_models = int(data["model"].num_nodes)
    if args.rung == "live" and expect_n is None:
        expect_n = n_models
    if expect_n is not None:
        assert n_models == expect_n, \
            "graph has %d models, rung %s expects %d" % (n_models, args.rung, expect_n)

    cfg = build_config(args)
    meta = capture_metadata(out, args, cfg, args.graph)
    binding = CK.make_binding(xm0, graph_path=args.graph, split_seed=args.seed,
                              family_vocab_path=args.family_vocab,
                              encoder_name=xd0.get("encoder_name",
                                                   "all-MiniLM-L6-v2"))
    if a0_graph_verification is not None and binding["graph_sha256"] != a0_graph_verification["graph_sha256"]:
        raise ValueError("A0 graph file table changed after its actual bytes were verified")

    resume_path, how = ((None, "fresh_smoke") if args.smoke_only
                        else CK.resolve_resume(args.resume, ckpt_dir))
    resume_state, history0, start_epoch = None, None, 0
    if resume_path:
        ck = CK.load(resume_path)
        reject_smoke_checkpoint(ck)
        bad = CK.validate(ck, binding)
        if (a0_graph_verification is not None or args.rung == "live") and not args.smoke_only:
            bad.extend(CK.validate(ck, binding, cfg=cfg, strict_cfg=True))
            if ck.get("extra", {}).get("run_id") != meta["run_id"]:
                bad.append("Resume checkpoint belongs to a different formal run")
        if bad:
            raise CK.IncompatibleCheckpoint(
                "cannot resume from %s:\n  %s" % (resume_path, "\n  ".join(bad)))
        CK.set_rng_state(ck.get("rng"))
        resume_state = CK.to_resume_state(ck)
        history0 = ck.get("history", [])
        start_epoch = int(ck["epoch"]) + 1
        print("[resume] %s (%s) -> continuing at epoch %d"
              % (resume_path, how, start_epoch), flush=True)
        if start_epoch >= args.epochs:
            print("[resume] checkpoint is already at epoch %d of %d; nothing to do"
                  % (start_epoch, args.epochs), flush=True)
    else:
        print("[resume] none (%s); starting fresh" % how, flush=True)

    data = apply_similar_to_mode(data, cfg["similar_to_mode"], k=cfg["similar_to_k"])
    split = make_root_aware_splits(data, root_of, split_seed=args.seed)

    state = {"step": start_epoch, "saved": [], "history": list(history0 or []),
             "opt": None}
    a0_records = (begin_a0_run_records(out, cfg, binding, meta["run_id"], args.seed,
                                      INIT_SEED, resume_path)
                  if a0_graph_verification is not None and not args.smoke_only and args.epochs == 25
                  else None)

    def persist_a0_records(checkpoint_path=None):
        if a0_records is None:
            return
        if checkpoint_path is not None:
            a0_records["checkpoint_path"] = os.path.abspath(checkpoint_path)
            a0_records["a0"]["checkpoint_sha256"] = CK.sha256_of(checkpoint_path)
        a0_records["history"] = list(state["history"])
        rss, rss_reason = peak_process_rss()
        previous_rss = a0_records.get("peak_process_rss_observed_max_bytes")
        observed_rss = max(rss, previous_rss or 0) if rss is not None else previous_rss
        a0_records["peak_process_rss_observed_max_bytes"] = observed_rss
        complete_rss = a0_records.get("peak_process_rss_complete", True) and rss is not None
        a0_records["peak_process_rss_complete"] = complete_rss
        a0_records["peak_process_rss_bytes"] = observed_rss if complete_rss else None
        a0_records["peak_process_rss_unavailable_reason"] = rss_reason or (None if complete_rss else "peak RSS unavailable during an earlier recorded segment")
        gpu = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        previous_gpu = a0_records.get("peak_gpu_allocated_bytes")
        a0_records["peak_gpu_allocated_bytes"] = max(gpu, previous_gpu or 0) if gpu is not None else previous_gpu
        a0_records["peak_gpu_unavailable_reason"] = None if gpu is not None else "CUDA unavailable"
        a0_records["timing_complete"] = all(s.get("end_ns") is not None for s in a0_records["train_segments"])
        a0_records["updated_at"] = utcnow()
        write_json_atomic(os.path.join(out, A0_RUN_RECORDS), a0_records)
    checkpoint_extra = {"rung": args.rung, "run_id": meta["run_id"]}
    if args.smoke_only:
        checkpoint_extra.update(run_purpose="smoke_only", smoke_only=True,
                                formal_training_eligible=False)

    def on_epoch_end(epoch, metrics, ctx):
        state["history"].append(dict(metrics))
        state["step"] = epoch + 1
        state["opt"] = ctx["opt"]
        last = (epoch == args.epochs - 1)
        if not (last or (epoch + 1) % max(args.ckpt_every, 1) == 0):
            return
        path = CK.save(ckpt_dir, epoch=epoch, global_step=state["step"],
                       model=ctx["model"], scorer=ctx["scorer"], opt=ctx["opt"],
                       history=state["history"], cfg=cfg, binding=binding,
                       best={"metric": "none", "value": None},
                       extra=checkpoint_extra,
                       keep=args.ckpt_keep)
        state["saved"].append(os.path.basename(path))
        persist_a0_records(path)
        print("    [ckpt] epoch %d -> %s" % (epoch, os.path.basename(path)),
              flush=True)

    cfg["resume_state"] = resume_state
    cfg["history0"] = history0
    cfg["on_epoch_end"] = on_epoch_end

    print("[train] rung=%s N=%d seed=%d epochs=%d device=%s batch=%d"
          % (args.rung, n_models, args.seed, args.epochs, device,
             cfg["batch_size"]), flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    segment = None
    if a0_records is not None:
        segment = {"start_ns": time.perf_counter_ns(), "end_ns": None,
                   "status": "running", "start_epoch": start_epoch,
                   "resume_checkpoint": resume_path, "started_at": utcnow(),
                   "clock": "time.perf_counter_ns",
                   "runtime_environment": dict(a0_records["runtime_environment"])}
        a0_records["train_segments"].append(segment)
        persist_a0_records()
    try:
        row, _pt, _ph, model, scorer = train_eval_one(
            data, xm0, xd0, cfg, split, init_seed=INIT_SEED, epochs=args.epochs,
            device=device, **({"smoke_only": True} if args.smoke_only else {}))
    except BaseException:
        if segment is not None:
            segment.update(end_ns=time.perf_counter_ns(), status="failed", end_epoch=state["step"])
            a0_records["status"] = "failed"
            persist_a0_records()
        raise
    if segment is not None:
        segment.update(end_ns=time.perf_counter_ns(), status="returned", end_epoch=state["step"])
        persist_a0_records()
    wall = time.time() - t0

    history = state["history"]
    gate = (smoke_gate(history, model, scorer, state["opt"]) if args.smoke_only
            else mechanism_gate(history, model))
    peak_gb = (round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
               if torch.cuda.is_available() else None)

    final = os.path.join(ckpt_dir, CK.LAST)
    if state["opt"] is not None:
        final = CK.save(ckpt_dir, epoch=args.epochs - 1, global_step=args.epochs,
                        model=model, scorer=scorer, opt=state["opt"],
                        history=history, cfg=cfg, binding=binding,
                        extra={**checkpoint_extra, "final": True}, keep=args.ckpt_keep)
        if not args.smoke_only:
            CK.save_best(ckpt_dir, final)
    if args.smoke_only:
        gate["checkpoint_roundtrip"] = bool(os.path.isfile(final)
            and smoke_checkpoint_roundtrip(final, model, scorer, binding))
        gate["passed"] = bool(gate["passed"] and gate["checkpoint_roundtrip"])
    if args.family_vocab and os.path.exists(args.family_vocab):
        import shutil
        shutil.copy2(args.family_vocab, os.path.join(ckpt_dir, "family_vocab.csv"))

    manifest = {
        "written_at": utcnow(), "rung": args.rung, "run_id": meta["run_id"],
        "graph": os.path.abspath(args.graph),
        "graph_sha256": binding["graph_sha256"],
        "n_models": n_models, "expect_n": expect_n,
        "seed": args.seed, "init_seed": INIT_SEED, "epochs": args.epochs,
        "device": device, "wallclock_s": round(wall, 1),
        "peak_gpu_mem_gb": peak_gb,
        "resumed_from": resume_path, "resume_mode": how,
        "start_epoch": start_epoch,
        "checkpoints_written": sorted(set(state["saved"]
                                          + [os.path.basename(final)])),
        "checkpoints_retained": sorted(f for f in os.listdir(ckpt_dir)
                                       if f.endswith(".pt")),
        "run_dir_reused": bool(prior_contents),
        "run_dir_prior_contents": prior_contents,
        "binding": binding,
        "scale_switches": {k: getattr(args, k) for k in (
            "fanout", "sparse_M", "contrast_n_neg",
            "contrast_max_pos_per_dataset", "chunked_infer", "skip_diagnostics")},
        "mechanism_gate": gate,
        "train_row": CK._jsonable(row),
        "anchor_gold10": RUNGS[args.rung]["anchor_gold10"],
        "metadata": meta,
    }
    if args.rung == "live":
        manifest.update(evidence_mode="live_hf_only", run_purpose="live_hf_reproduction",
                        historical_a0_result=False)
    if args.smoke_only:
        manifest.update(run_purpose="smoke_only", smoke_only=True,
                        formal_training_eligible=False, evaluation_performed=False,
                        anchor_gold10=None, status="PASS" if gate["passed"] else "FAIL")
    if a0_graph_verification is not None:
        manifest["a0_graph_verification"] = a0_graph_verification
    if a0_records is not None:
        a0_records["status"] = "complete" if gate["passed"] and len(history) == 25 else "gate_failed"
        a0_records["mechanism_gate"] = gate
        persist_a0_records(final)
    write_json_atomic(os.path.join(out, "SMOKE_REPORT.json" if args.smoke_only else "MANIFEST.json"), manifest)
    write_json_atomic(os.path.join(out, "metrics", "train_history.json"), history)

    print("\n=== %s %s seed=%d ===" % ("SMOKE ONLY" if args.smoke_only else "T6", args.rung, args.seed))
    print("  wallclock %.1fs | peak GPU %s GB | epochs %d"
          % (wall, peak_gb, len(history)))
    if gate["loss_first"] is None:
        print("  no epochs ran (checkpoint was already at the target epoch)")
    else:
        print("  loss %.4f -> %.4f | descended %s | no NaN %s"
              % (gate["loss_first"], gate["loss_last"], gate["loss_descended"],
                 gate["no_nan"]))
    print("  mechanism gate: %s" % ("PASS" if gate["passed"] else "FAIL"))
    print("  ckpt %s" % os.path.basename(final))
    print("  -> %s" % out)
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
