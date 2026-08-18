"""
graph_store.py -- RF: sharded, memory-mappable storage for a rung graph.

WHY THIS EXISTS
    Up to 100K the whole graph is one `torch.save` file of ~223 MB and nothing
    about that is a problem. At full-lake scale the model feature matrix alone
    is N x 448 float32 -- 5.2 GB at N = 2.9M -- and `torch.load` of a single
    file materialises every tensor in RAM at once, on top of the pickle buffer
    it reads it from. That is the peak this module removes.

    Features are stored at full float32 precision. They are NOT downcast to
    fp16: `x` is the frozen half of the model input and the one input that the
    whole ladder holds byte-identical across rungs (constraint 1), so the
    storage format is not a place to introduce a numeric difference.
    The saving here comes from mapping the array instead of copying it, which
    costs nothing in precision.

LAYOUT
    <dir>/x_model.npy          [N, 448] float32   -- memory-mapped on load
    <dir>/x_dataset.npy        [D, 458] float32   -- memory-mapped on load
    <dir>/nodes.npz            the small integer node columns
    <dir>/edges.npz            edge_index / edge_attr / relation_id per relation
    <dir>/unique_model_id.parquet, unique_dataset_id.parquet
    <dir>/meta.json            xm0_meta, xd0_meta, provenance, dtypes, sha256s

    `meta.json` records a sha256 for every file, so a truncated or swapped
    shard is detected at load time instead of surfacing as a strange metric.

Run (from ModelLakeFishing/):
    python -m scale1m.graph_store --pt  <graph.pt>  --out <dir>       # convert
    python -m scale1m.graph_store --pt  <graph.pt>  --verify <dir>    # compare
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

X_FILES = {("model", "x"): "x_model.npy", ("dataset", "x"): "x_dataset.npy"}
EDGE_SEP = "__"


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _edge_key(et, field):
    return EDGE_SEP.join(et) + EDGE_SEP + field


def _parse_edge_key(key):
    *et, field = key.split(EDGE_SEP)
    return tuple(et), field


def save_sharded(ckpt, out_dir, mmap_fields=X_FILES):
    """Write a rung checkpoint as one directory of plain arrays."""
    os.makedirs(out_dir, exist_ok=True)
    data = ckpt["data"]
    meta = {
        "format": "scale1m.graph_store/1",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "node_types": list(data.node_types),
        "edge_types": [list(et) for et in data.edge_types],
        "num_nodes": {nt: int(data[nt].num_nodes) for nt in data.node_types},
        "xm0_meta": ckpt["xm0_meta"],
        "xd0_meta": ckpt["xd0_meta"],
        "provenance": ckpt.get("provenance"),
        "tensors": {},
        "files": {},
    }

    # 1. the two big float matrices, one .npy each so they can be mapped
    for (nt, field), fname in mmap_fields.items():
        t = data[nt][field]
        np.save(os.path.join(out_dir, fname), t.numpy())
        meta["tensors"][f"{nt}.{field}"] = {"file": fname, "shape": list(t.shape),
                                            "dtype": str(t.dtype).replace("torch.", "")}

    # 2. every remaining node column (all small integer vectors)
    node_arrays = {}
    for nt in data.node_types:
        for field, t in data[nt].items():
            if (nt, field) in mmap_fields:
                continue
            node_arrays[f"{nt}.{field}"] = t.numpy()
            meta["tensors"][f"{nt}.{field}"] = {"file": "nodes.npz",
                                                "shape": list(t.shape),
                                                "dtype": str(t.dtype).replace("torch.", "")}
    np.savez(os.path.join(out_dir, "nodes.npz"), **node_arrays)

    # 3. edges, keyed by "src__rel__dst__field"
    edge_arrays = {}
    for et in data.edge_types:
        for field, t in data[et].items():
            key = _edge_key(et, field)
            edge_arrays[key] = t.numpy()
            meta["tensors"][key] = {"file": "edges.npz", "shape": list(t.shape),
                                    "dtype": str(t.dtype).replace("torch.", "")}
    np.savez(os.path.join(out_dir, "edges.npz"), **edge_arrays)

    # 4. the id tables -- parquet keeps dtypes, csv does not
    for name in ("unique_model_id", "unique_dataset_id"):
        if name in ckpt:
            ckpt[name].to_parquet(os.path.join(out_dir, name + ".parquet"), index=False)
            meta["files"][name + ".parquet"] = None

    for fname in os.listdir(out_dir):
        if fname != "meta.json":
            meta["files"][fname] = _sha256(os.path.join(out_dir, fname))
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    return meta


def load_sharded(out_dir, mmap=True, verify_sha256=False):
    """Rebuild the checkpoint dict. With mmap=True the two `x` matrices stay
    on disk and are paged in by the sampler as it touches rows."""
    with open(os.path.join(out_dir, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)

    if verify_sha256:
        for fname, want in meta["files"].items():
            if want is None:
                continue
            got = _sha256(os.path.join(out_dir, fname))
            if got != want:
                raise ValueError("sha256 mismatch for %s: %s != %s" % (fname, got, want))

    data = HeteroData()
    for (nt, field), fname in X_FILES.items():
        path = os.path.join(out_dir, fname)
        if not os.path.exists(path):
            continue
        arr = np.load(path, mmap_mode="r" if mmap else None)
        # from_numpy on a read-only map shares the pages instead of copying
        data[nt][field] = torch.from_numpy(arr if mmap else np.ascontiguousarray(arr))

    with np.load(os.path.join(out_dir, "nodes.npz")) as z:
        for key in z.files:
            nt, field = key.split(".", 1)
            data[nt][field] = torch.from_numpy(z[key])
    with np.load(os.path.join(out_dir, "edges.npz")) as z:
        for key in z.files:
            et, field = _parse_edge_key(key)
            data[et][field] = torch.from_numpy(z[key])

    ckpt = {"data": data, "xm0_meta": meta["xm0_meta"], "xd0_meta": meta["xd0_meta"],
            "provenance": meta.get("provenance"), "_mmap": bool(mmap)}
    for name in ("unique_model_id", "unique_dataset_id"):
        path = os.path.join(out_dir, name + ".parquet")
        if os.path.exists(path):
            ckpt[name] = pd.read_parquet(path)
    return ckpt


def compare(a, b):
    """Every tensor and every metadata value, element by element.

    Returns a list of human-readable differences; empty means identical.
    """
    diffs = []
    da, db = a["data"], b["data"]
    if set(da.node_types) != set(db.node_types):
        diffs.append("node types differ: %s vs %s" % (da.node_types, db.node_types))
    if set(da.edge_types) != set(db.edge_types):
        diffs.append("edge types differ: %s vs %s" % (da.edge_types, db.edge_types))

    for nt in da.node_types:
        for field, t in da[nt].items():
            if field not in db[nt]:
                diffs.append("missing node field %s.%s" % (nt, field))
                continue
            o = db[nt][field]
            if t.shape != o.shape or t.dtype != o.dtype:
                diffs.append("%s.%s shape/dtype %s %s vs %s %s"
                             % (nt, field, tuple(t.shape), t.dtype,
                                tuple(o.shape), o.dtype))
            elif not torch.equal(t, torch.as_tensor(o)):
                diffs.append("%s.%s values differ" % (nt, field))
    for et in da.edge_types:
        for field, t in da[et].items():
            if field not in db[et]:
                diffs.append("missing edge field %s.%s" % (et, field))
                continue
            o = db[et][field]
            if t.shape != o.shape or not torch.equal(t, torch.as_tensor(o)):
                diffs.append("%s.%s differs" % (et, field))

    for key in ("xm0_meta", "xd0_meta"):
        if a[key] != b[key]:
            ka, kb = a[key], b[key]
            for k in set(ka) | set(kb):
                if ka.get(k) != kb.get(k):
                    diffs.append("%s[%s] differs" % (key, k))
    for name in ("unique_model_id", "unique_dataset_id"):
        if name in a or name in b:
            if name not in a or name not in b:
                diffs.append("missing %s" % name)
            elif not a[name].equals(b[name]):
                diffs.append("%s differs" % name)
    return diffs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="convert / verify a sharded rung graph")
    p.add_argument("--pt", required=True, help="the single-file .pt to convert")
    p.add_argument("--out", help="directory to write")
    p.add_argument("--verify", help="directory to compare against --pt")
    p.add_argument("--no-mmap", action="store_true")
    args = p.parse_args(argv)

    t0 = time.time()
    ckpt = torch.load(args.pt, weights_only=False)
    print("[pt] loaded in %.1fs" % (time.time() - t0))

    if args.out:
        t0 = time.time()
        meta = save_sharded(ckpt, args.out)
        total = sum(os.path.getsize(os.path.join(args.out, f)) for f in meta["files"])
        print("[shard] wrote %d files, %.1f MB, in %.1fs"
              % (len(meta["files"]), total / 1e6, time.time() - t0))

    target = args.verify or args.out
    if target:
        t0 = time.time()
        back = load_sharded(target, mmap=not args.no_mmap, verify_sha256=True)
        print("[shard] loaded in %.1fs (mmap=%s)" % (time.time() - t0, not args.no_mmap))
        diffs = compare(ckpt, back)
        if diffs:
            print("[FAIL] %d difference(s):" % len(diffs))
            for d in diffs[:20]:
                print("   ", d)
            return 1
        print("[ok] sharded copy is identical to the .pt, tensor by tensor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
