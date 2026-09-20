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
    if path and os.path.isdir(path):
        with open(os.path.join(path, "meta.json"), encoding="utf-8") as fh:
            files = json.load(fh)["files"]
        blob = json.dumps(files, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return sha256_of(path) if path and os.path.exists(path) else None


def size_bucket_version():
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
            pass


def save(ckpt_dir, *, epoch, global_step, model, scorer, opt, history, cfg,
         binding, best=None, scheduler=None, scaler=None, extra=None, keep=3):
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
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError("--resume %s does not exist" % explicit)
        return explicit, "explicit"
    last = os.path.join(ckpt_dir, LAST)
    if os.path.exists(last):
        return last, "last.pt"
    return None, "fresh"


def validate(ck, binding, cfg=None, strict_cfg=False):
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
    return {"model": ck["model"], "scorer": ck["scorer"], "opt": ck["opt"],
            "epoch": int(ck["epoch"]), "rng": ck.get("rng")}


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return repr(obj)
