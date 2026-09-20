import argparse
import gzip
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(_HERE), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.scale1m.hf_crawl import data_root, utcnow, write_json_atomic
from ModelLakeFishing.scale1m.hf_canonicalize import normalize

N_SNAPSHOT = 3_003_759
N_TOTAL = 3_016_439


def shard_paths(raw_dir):
    with open(os.path.join(raw_dir, "SHARDS.json"), encoding="utf-8") as fh:
        shards = json.load(fh)
    return [(os.path.join(raw_dir, s["file"]), int(s["n_records"])) for s in shards]


def harvest(raw_dir, ladder_model, verbose_every=250_000):
    n = N_SNAPSHOT
    downloads = np.zeros(n, dtype=np.int64)
    pipeline = [""] * n
    tags = [""] * n

    i = 0
    t0 = time.time()
    for path, n_rec in shard_paths(raw_dir):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                mid = normalize(rec.get("id"))
                if mid != ladder_model[i]:
                    raise AssertionError(
                        "row %d: shard id %r != ladder model %r (mappedID order "
                        "is not the crawl order; every popularity baseline would "
                        "be scored against the wrong model)"
                        % (i, mid, ladder_model[i]))
                downloads[i] = int(rec.get("downloads") or 0)
                pipeline[i] = str(rec.get("pipeline_tag") or "")
                t = rec.get("tags") or []
                tags[i] = " ".join(str(x) for x in t) if isinstance(t, list) else str(t)
                i += 1
                if i % verbose_every == 0:
                    print("[harvest] %d/%d  %.0fs" % (i, n, time.time() - t0), flush=True)
    if i != n:
        raise AssertionError("read %d records, expected %d" % (i, n))
    return downloads, pipeline, tags


def main(argv=None):
    root = data_root()
    parser = argparse.ArgumentParser(
        description="Build the mappedID-aligned X6 metadata sidecar")
    parser.add_argument("--candidates", default=os.path.join(
        root, "data1m", "candidates_full"))
    parser.add_argument("--ladder", default=os.path.join(
        root, "data1m", "ladder_rf", "full_model_ids.parquet"))
    parser.add_argument("--out", default=os.path.join(
        root, "data1m", "baseline_rf"))
    args = parser.parse_args(argv)
    raw_dir = args.candidates
    ladder_path = args.ladder
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    ladder = pd.read_parquet(ladder_path, columns=["mappedID", "model", "in_snapshot"])
    if len(ladder) != N_TOTAL:
        raise AssertionError("ladder has %d rows, expected %d" % (len(ladder), N_TOTAL))
    if ladder["mappedID"].tolist() != list(range(N_TOTAL)):
        raise AssertionError("ladder mappedID is not contiguous 0..N-1")
    lm = ladder["model"].astype(str).tolist()

    print("[start] %d snapshot rows + %d historical-only rows"
          % (N_SNAPSHOT, N_TOTAL - N_SNAPSHOT), flush=True)
    downloads, pipeline, tags = harvest(raw_dir, lm)

    pad_i = np.zeros(N_TOTAL - N_SNAPSHOT, dtype=np.int64)
    pad_s = [""] * (N_TOTAL - N_SNAPSHOT)
    df = pd.DataFrame({
        "mappedID": np.arange(N_TOTAL, dtype=np.int64),
        "downloads": np.concatenate([downloads, pad_i]),
        "pipeline_tag": pipeline + pad_s,
        "tags": tags + pad_s,
        "in_snapshot": ladder["in_snapshot"].to_numpy(),
    })
    out_path = os.path.join(out_dir, "baseline_attrs.parquet")
    df.to_parquet(out_path, index=False)

    gate = {
        "rows_is_N": len(df) == N_TOTAL,
        "mappedID_contiguous": df["mappedID"].tolist() == list(range(N_TOTAL)),
        "alignment_checked_every_row": True,
        "tail_rows_all_padded":
            bool((df["downloads"].to_numpy()[N_SNAPSHOT:] == 0).all()),
    }
    stats = {
        "n_rows": int(len(df)),
        "n_snapshot": N_SNAPSHOT,
        "downloads": {"nonzero": int((df["downloads"] > 0).sum()),
                      "max": int(df["downloads"].max()),
                      "median_nonzero": float(
                          df.loc[df["downloads"] > 0, "downloads"].median())},
        "pipeline_tag": {"nonempty": int((df["pipeline_tag"] != "").sum()),
                         "n_distinct": int(df["pipeline_tag"].nunique())},
        "tags": {"nonempty": int((df["tags"] != "").sum())},
    }
    write_json_atomic(os.path.join(out_dir, "SIDECAR_REPORT.json"),
                      {"written_at": utcnow(), "gate": gate, "stats": stats,
                       "out": out_path})
    print(json.dumps({"gate": gate, "stats": stats}, indent=2))
    if not all(v is True for k, v in gate.items() if isinstance(v, bool)):
        raise SystemExit("EXIT GATE FAILED")
    print("[ok] %s" % out_path)


if __name__ == "__main__":
    main()
