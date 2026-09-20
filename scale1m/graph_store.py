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

    for (nt, field), fname in mmap_fields.items():
        t = data[nt][field]
        np.save(os.path.join(out_dir, fname), t.numpy())
        meta["tensors"][f"{nt}.{field}"] = {"file": fname, "shape": list(t.shape),
                                            "dtype": str(t.dtype).replace("torch.", "")}

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

    edge_arrays = {}
    for et in data.edge_types:
        for field, t in data[et].items():
            key = _edge_key(et, field)
            edge_arrays[key] = t.numpy()
            meta["tensors"][key] = {"file": "edges.npz", "shape": list(t.shape),
                                    "dtype": str(t.dtype).replace("torch.", "")}
    np.savez(os.path.join(out_dir, "edges.npz"), **edge_arrays)

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
