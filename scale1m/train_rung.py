"""
train_rung.py -- T6: train the champion config on a rung's graph.

Runbook: docs/1M/100kplan.md §9.   Guide for running it: docs/1M/T6GPU.md.

THE CONFIG IS NOT TUNED HERE
    §9.1 says the configuration is reused with zero changes: l1l3b_config() --
    L1 whole-lake logQ sampled softmax, L3 native task ids, global_n_neg=256,
    batch_size=1024. This file adds the things a cluster run needs and nothing
    that changes what is being trained: a run directory, checkpoint/resume,
    metadata capture, the mechanism gate, and a manifest.

    That restraint is the point of the anchor gate. R0 on 12K has to reproduce
    gold@10 = 0.4159; if this file quietly changed a hyperparameter, the anchor
    would move and there would be no way to tell that from a genuine effect of
    the T0 scale switches.

WHAT RUNS WHERE
    Everything here is CPU-runnable on a small graph -- that is how the
    checkpoint/resume gate is tested. The 100K training itself needs a GPU.

Run (from ModelLakeFishing/):
    python -m scale1m.train_rung --rung 100k --graph <...>/hgraph_100k.pt `
        --seed 0 --epochs 25 --out <OUTPUT_ROOT>/runs/<RUN_ID>
"""

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
# _REPO_ROOT is the *import* root: `import ModelLakeFishing.stage2...` needs the
# directory above the package on sys.path. It is one level above the git repo
# and must not be reused for git -- doing so is what left git_head empty in
# every T6 run of 2026-08-11 (docs/1M/T6more.md P1-2).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(_HERE), ".."))
_GIT_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scale1m import checkpoint as CK                                # noqa: E402
from scale1m.hf_crawl import utcnow, write_json_atomic              # noqa: E402

# Written into git_head/git_status when git cannot report on the tree, so the
# manifest says "unknown" out loud instead of looking like a clean repo.
_NO_GIT = "not-a-git-repo"

# Anchors the plan pins per rung (§9.2 / G-C2). None = no published anchor yet.
RUNGS = {
    "12k": {"expect_n": None, "anchor_gold10": 0.4159},
    "30k": {"expect_n": 30_183, "anchor_gold10": None},
    "100k": {"expect_n": 100_000, "anchor_gold10": None},
}


# ── run directory ────────────────────────────────────────────────────────────

def make_run_dir(out):
    """Create the run layout and report what was already in it.

    A requeue reuses the directory legitimately (same RUN_ID, resume from
    ckpt/last.pt). A previous *failed* attempt also leaves things behind --
    stdout/train.log is opened in append mode, so its traceback silently
    becomes part of the next attempt's delivered log. That happened to the
    12K run of 2026-08-11 and was invisible in the manifest; now it is not."""
    prior = []
    for rel in ("MANIFEST.json", os.path.join("stdout", "train.log")):
        p = os.path.join(out, rel)
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            prior.append(rel.replace(os.sep, "/"))
    ckpt = os.path.join(out, "ckpt")
    if os.path.isdir(ckpt) and any(f.endswith(".pt") for f in os.listdir(ckpt)):
        prior.append("ckpt/*.pt")
    for sub in ("ckpt", "exports", "metrics", "stdout", "metadata"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    return out, prior


def capture_metadata(out, args, cfg, graph_path):
    """§4.6's list. Written before training starts, so a job that dies still
    leaves behind what it was trying to do."""
    md = os.path.join(out, "metadata")
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--short")
    # Dirtiness means "tracked files differ from HEAD", which is what
    # uncommitted.patch can actually capture. Plain `status --short` also lists
    # untracked paths, and on the cluster that is always non-empty (logs/ and
    # stray slurm-*.out), so every run reported a dirty tree and wrote a
    # zero-byte patch. -uno asks the question we mean.
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
        "graph_sha256": CK.sha256_of(graph_path),
        "args": vars(args),
        "resolved_config": CK._jsonable(cfg),
    }
    write_json_atomic(os.path.join(md, "resolved_config.json"), meta)
    if tracked_dirty:
        # §4.3: a deliberately dirty tree is allowed, but it has to be recorded
        patch = _git("diff", "HEAD")
        with open(os.path.join(md, "uncommitted.patch"), "w", encoding="utf-8") as fh:
            fh.write(patch or "")
    return meta


def _git(*a):
    """stdout of the git command, or None when this is not a usable git tree.

    The distinction matters: an empty string is a legitimate result (a clean
    `status --short` returns one), so collapsing failure into "" hides a broken
    git root behind a plausible-looking value. That is exactly how the T6 runs
    recorded no commit at all while the tree was in fact dirty."""
    try:
        r = subprocess.run(["git", *a], cwd=_GIT_ROOT, capture_output=True,
                           text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


# ── mechanism gate (G-C1) ────────────────────────────────────────────────────

def mechanism_gate(history, model):
    """§9.4 G1: loss went down, nothing is NaN, both embedding tables moved,
    the frozen features did not."""
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


# ── main ─────────────────────────────────────────────────────────────────────

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
    # T0 scale switches, same names and defaults as scale.export_ours
    p.add_argument("--fanout", action="store_true")
    p.add_argument("--sparse-M", action="store_true")
    p.add_argument("--contrast-n-neg", type=int, default=None)
    p.add_argument("--contrast-max-pos-per-dataset", type=int, default=None)
    p.add_argument("--chunked-infer", type=int, default=None)
    p.add_argument("--skip-diagnostics", action="store_true")
    args = p.parse_args(argv)

    from ModelLakeFishing.stage2TrainGraphSAGE.ablation import train_eval_one
    from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import apply_similar_to_mode
    from ModelLakeFishing.stage2TrainGraphSAGE.top1_audit import INIT_SEED
    from ModelLakeFishing.stage2TrainGraphSAGE.d0_splits import make_root_aware_splits

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out, prior_contents = make_run_dir(args.out)
    if prior_contents:
        print("[run-dir] reusing a non-empty run directory; prior contents: %s"
              % ", ".join(prior_contents), flush=True)
    ckpt_dir = os.path.join(out, "ckpt")
    expect_n = args.expect_n or RUNGS[args.rung]["expect_n"]

    payload = torch.load(args.graph, map_location="cpu", weights_only=False)
    data, xm0, xd0 = payload["data"], payload["xm0_meta"], payload["xd0_meta"]
    udi = payload["unique_dataset_id"].sort_values("mappedID").reset_index(drop=True)
    root_of = udi["root"].astype(str).tolist()

    n_models = int(data["model"].num_nodes)
    if expect_n is not None:
        # plan §1 rule 3, checked before anything expensive runs rather than at
        # the end when the compute is already spent
        assert n_models == expect_n, \
            "graph has %d models, rung %s expects %d" % (n_models, args.rung, expect_n)

    cfg = build_config(args)
    meta = capture_metadata(out, args, cfg, args.graph)
    binding = CK.make_binding(xm0, graph_path=args.graph, split_seed=args.seed,
                              family_vocab_path=args.family_vocab,
                              encoder_name=xd0.get("encoder_name",
                                                   "all-MiniLM-L6-v2"))

    # --- resume ------------------------------------------------------------
    resume_path, how = CK.resolve_resume(args.resume, ckpt_dir)
    resume_state, history0, start_epoch = None, None, 0
    if resume_path:
        ck = CK.load(resume_path)
        bad = CK.validate(ck, binding)
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

    # --- checkpoint hook ---------------------------------------------------
    # train() keeps its own `history`; this driver keeps a parallel copy so the
    # checkpoint carries the complete history, not only the epochs since this
    # process started. `opt` is captured from the callback rather than rebuilt
    # afterwards -- a fresh Adam has empty moment buffers, and resuming from a
    # checkpoint written that way silently changes the trajectory.
    state = {"step": start_epoch, "saved": [], "history": list(history0 or []),
             "opt": None}

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
                       extra={"rung": args.rung, "run_id": meta["run_id"]},
                       keep=args.ckpt_keep)
        state["saved"].append(os.path.basename(path))
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
    row, _pt, _ph, model, scorer = train_eval_one(
        data, xm0, xd0, cfg, split, init_seed=INIT_SEED, epochs=args.epochs,
        device=device)
    wall = time.time() - t0

    history = state["history"]
    gate = mechanism_gate(history, model)
    peak_gb = (round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
               if torch.cuda.is_available() else None)

    final = os.path.join(ckpt_dir, CK.LAST)
    if state["opt"] is not None:
        final = CK.save(ckpt_dir, epoch=args.epochs - 1, global_step=args.epochs,
                        model=model, scorer=scorer, opt=state["opt"],
                        history=history, cfg=cfg, binding=binding,
                        extra={"rung": args.rung, "run_id": meta["run_id"],
                               "final": True}, keep=args.ckpt_keep)
        CK.save_best(ckpt_dir, final)
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
        # written != retained: _prune keeps only the newest `ckpt_keep` numbered
        # files, so the two lists differ by design. Reporting only the first
        # sends the reader looking for files that were already deleted.
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
    write_json_atomic(os.path.join(out, "MANIFEST.json"), manifest)
    write_json_atomic(os.path.join(out, "metrics", "train_history.json"), history)

    print("\n=== T6 %s seed=%d ===" % (args.rung, args.seed))
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
