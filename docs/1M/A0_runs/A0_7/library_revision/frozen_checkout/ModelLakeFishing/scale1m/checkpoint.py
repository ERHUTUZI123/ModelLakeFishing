"""
checkpoint.py -- T6 checkpoint / resume (docs/1M/100kplan.md §9.3).

WHY THIS IS A SEPARATE MODULE WITH ITS OWN TESTS
    On watGPU a job can be preempted and requeued. A resume that quietly starts
    from epoch 0, or restores the weights but not the optimizer, produces a run
    that finishes, reports a number, and is wrong -- no exception anywhere. The
    plan's gate G-W4 ("save, reload, resume, steps stay continuous") is the only
    thing standing between that and a reported result, so the machinery it tests
    lives here rather than inline in the training driver.

WHAT A CHECKPOINT MUST CARRY
    §9.3 lists it: model, optimizer, scheduler, AMP scaler, epoch, global step,
    best-metric / early-stopping state, the four RNG streams, resolved config,
    and the data version. On top of that the checkpoint records its BINDING --
    family_vocab path and sha256, size-bucket constants, e_name seed, token_dim,
    encoder name. CLAUDE.md's rule is that the vocab is the only credential for
    embedding-row identity; a checkpoint that cannot prove which vocab it was
    trained against has orphaned family rows, and no error will say so.

ATOMICITY
    Write to a temp file, fsync, rename. `last.pt` is only repointed after the
    numbered checkpoint is safely on disk, so a job killed mid-write leaves the
    previous checkpoint intact rather than a truncated one.
"""

import glob
import hashlib
import json
import os
import random
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_S1 = os.path.join(os.path.dirname(_HERE), "stage1BuildTransferGraph")
if _S1 not in sys.path:
    sys.path.insert(0, _S1)

CKPT_VERSION = 1
LAST = "last.pt"
BEST = "best.pt"
STEM = "checkpoint_epoch_"

# Keys that must match for a resume to be legitimate. A mismatch here means the
# checkpoint was trained against different data or a different embedding-row
# identity, which makes the restored weights meaningless rather than merely
# stale -- so it is a hard failure, not a warning.
BINDING_KEYS = ("family_vocab_sha256", "num_families", "num_size_buckets",
                "size_bucket_version", "name_seed", "name_dim", "desc_dim",
                "encoder_name", "graph_sha256", "split_seed")


class IncompatibleCheckpoint(RuntimeError):
    pass


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def graph_digest(path):
    """Return a stable content identity for file and sharded-directory graphs."""
    if path and os.path.isdir(path):
        with open(os.path.join(path, "meta.json"), encoding="utf-8") as fh:
            files = json.load(fh)["files"]
        blob = json.dumps(files, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return sha256_of(path) if path and os.path.exists(path) else None


def size_bucket_version():
    """A string that changes whenever the bucket boundaries change.

    The size table needs no vocab file (the boundaries are constants in code),
    but CLAUDE.md notes that changing them requires retraining -- so the
    constants themselves are the version.
    """
    from dataset_embed.xm0_builder import (SIZE_LOG10_MIN, SIZE_LOG10_MAX,
                                           SIZE_BUCKET_WIDTH, NUM_SIZE_BUCKETS)
    return "min%.1f_max%.1f_w%.2f_n%d" % (SIZE_LOG10_MIN, SIZE_LOG10_MAX,
                                          SIZE_BUCKET_WIDTH, NUM_SIZE_BUCKETS)


def make_binding(xm0_meta, *, graph_path, split_seed, family_vocab_path=None,
                 name_seed=42, encoder_name="all-MiniLM-L6-v2"):
    return {
        "family_vocab_path": os.path.abspath(family_vocab_path)
        if family_vocab_path else None,
        "family_vocab_sha256": sha256_of(family_vocab_path)
        if family_vocab_path and os.path.exists(family_vocab_path) else None,
        "num_families": int(xm0_meta["num_families"]),
        "num_size_buckets": int(xm0_meta["num_size_buckets"]),
        "size_bucket_version": size_bucket_version(),
        "name_seed": int(name_seed),
        "name_dim": int(xm0_meta.get("name_dim", 64)),
        "desc_dim": int(xm0_meta.get("desc_dim", 384)),
        "encoder_name": encoder_name,
        "graph_sha256": graph_digest(graph_path),
        "split_seed": int(split_seed),
    }


# ── RNG ──────────────────────────────────────────────────────────────────────

def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available() else None,
    }


def set_rng_state(s):
    if not s:
        return
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"].cpu() if hasattr(s["torch"], "cpu") else s["torch"])
    if s.get("torch_cuda") and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(s["torch_cuda"])
        except (RuntimeError, ValueError):
            # different GPU count than the run that saved it; CPU streams are
            # restored either way and the mismatch is reported by the caller
            pass


# ── save / load ──────────────────────────────────────────────────────────────

def save(ckpt_dir, *, epoch, global_step, model, scorer, opt, history, cfg,
         binding, best=None, scheduler=None, scaler=None, extra=None, keep=3):
    """Write checkpoint_epoch_<N>.pt atomically, then repoint last.pt."""
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = {
        "ckpt_version": CKPT_VERSION,
        "epoch": int(epoch), "global_step": int(global_step),
        "model": model.state_dict(), "scorer": scorer.state_dict(),
        "opt": opt.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "history": list(history or []),
        "best": dict(best or {}),
        "cfg": _jsonable(cfg),
        "binding": dict(binding),
        "rng": rng_state(),
        "extra": dict(extra or {}),
    }
    path = os.path.join(ckpt_dir, "%s%d.pt" % (STEM, int(epoch)))
    _atomic_torch_save(payload, path)
    _atomic_torch_save(payload, os.path.join(ckpt_dir, LAST))
    _prune(ckpt_dir, keep)
    return path


def save_best(ckpt_dir, src_path):
    payload = torch.load(src_path, map_location="cpu", weights_only=False)
    _atomic_torch_save(payload, os.path.join(ckpt_dir, BEST))


def _atomic_torch_save(payload, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        torch.save(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _prune(ckpt_dir, keep):
    """Keep the newest `keep` numbered checkpoints. best.pt / last.pt are named
    files and are never in this list, so the run always retains best, latest and
    at least one historical copy."""
    if not keep or keep < 1:
        return
    nums = []
    for p in glob.glob(os.path.join(ckpt_dir, STEM + "*.pt")):
        try:
            nums.append((int(os.path.basename(p)[len(STEM):-3]), p))
        except ValueError:
            continue
    for _, p in sorted(nums, reverse=True)[keep:]:
        os.remove(p)


def load(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    if ck.get("ckpt_version") != CKPT_VERSION:
        raise IncompatibleCheckpoint(
            "checkpoint version %s, this code writes %s"
            % (ck.get("ckpt_version"), CKPT_VERSION))
    return ck


def resolve_resume(explicit, ckpt_dir):
    """§9.3's fixed order: explicit --resume, then <run>/ckpt/last.pt, then a
    new run. Returns (path_or_None, how) so the driver can log the choice --
    "resumed from nothing" and "started fresh" must not look the same in a log.
    """
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError("--resume %s does not exist" % explicit)
        return explicit, "explicit"
    last = os.path.join(ckpt_dir, LAST)
    if os.path.exists(last):
        return last, "last.pt"
    return None, "fresh"


def validate(ck, binding, cfg=None, strict_cfg=False):
    """Return the list of mismatches; empty means safe to resume."""
    bad = []
    ckb = ck.get("binding", {})
    for k in BINDING_KEYS:
        want, got = binding.get(k), ckb.get(k)
        if want is not None and got is not None and want != got:
            bad.append("%s: checkpoint has %r, this run has %r" % (k, got, want))
    if strict_cfg and cfg is not None:
        a, b = _jsonable(cfg), ck.get("cfg", {})
        for k in sorted(set(a) | set(b)):
            if k in ("resume_state", "history0", "on_epoch_end"):
                continue
            if a.get(k) != b.get(k):
                bad.append("cfg.%s: checkpoint has %r, this run has %r"
                           % (k, b.get(k), a.get(k)))
    return bad


def to_resume_state(ck):
    """The dict stage2TrainGraphSAGE.train() consumes.

    The RNG travels inside it rather than being restored by the caller: the
    model is built by train_eval_one, which reseeds, so anything restored
    before that call is thrown away.
    """
    return {"model": ck["model"], "scorer": ck["scorer"], "opt": ck["opt"],
            "epoch": int(ck["epoch"]), "rng": ck.get("rng")}


def _jsonable(obj):
    """A JSON-safe deep copy.

    Copying rather than returning `obj` when it already serialises matters:
    the training driver snapshots the config for metadata and only afterwards
    adds the callback and resume state to it. Returning the same dict would
    make that snapshot alias the live config, and writing the manifest at the
    end of the run would fail on a function object -- after the training had
    already been paid for.
    """
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return repr(obj)
