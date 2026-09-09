"""
p1_content_harvest.py -- v5 P1: dataset-side CONTENT feature harvest.

The v4 D0 graph gives each dataset node only a metadata descriptor string
(name + card task_categories + tags). Under root-aware splits the test root's
labels are invisible, so the query features are the ONLY cross-root channel --
and P0 proved that channel is starved (O-task 0.080 < model 0.100, O-sibling
upper bound 0.563). P1 supplies the missing signal: what the dataset's examples
actually look like.

For every dataset NODE in the graph (mappedID order), pull ~N example rows from
the HF datasets-server rows API and store the concatenated text. Bounded
network, resumable (one cache file per node), METADATA/TEXT ONLY -- never
downloads full datasets or weights. Encoding into e_content and graph rebuild
are a SEPARATE offline step (build_xd0 content view); this script only harvests.

datasets-server coverage is partial (~52%, P0 probe): nodes without a served
config get an empty sample and fall back to the metadata string at feature time
-- recorded per node so the P1 ablation can stratify by has_content.

Run (from stage1BuildTransferGraph/):
    ../.venv/Scripts/python.exe p1_content_harvest.py [--n-rows 100] [--limit N]
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time

import pandas as pd
import requests
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

DSS = "https://datasets-server.huggingface.co"
GRAPH = os.path.join(_HERE, "hgraph_d0_v1.pt")
POOL = os.path.join(_HERE, "artifacts", "d0_lake", "d0_dataset_pool.csv")
CACHE = os.path.join(_HERE, "d0_lake_cache", "content")
OUT = os.path.join(_HERE, "artifacts", "d0_lake", "p1_content")
SLEEP = 0.15
N_ROWS = 100
MAX_CHARS_PER_ROW = 400          # truncate long text fields
TEXT_HINTS = ("text", "sentence", "content", "question", "premise", "hypothesis",
              "document", "passage", "context", "review", "title", "abstract",
              "sentence1", "sentence2", "query", "comment", "body", "utterance")


def _get(url, params, timeout=25):
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (404, 501):     # no such dataset / not supported
                return None
        except Exception:
            pass
        time.sleep(0.5 + attempt)
    return None


def first_config_split(hf_id):
    j = _get(f"{DSS}/splits", {"dataset": hf_id})
    if not j or not j.get("splits"):
        return None
    sp = j["splits"]
    # prefer a train split, else the first
    for s in sp:
        if s.get("split") == "train":
            return s["config"], s["split"]
    return sp[0]["config"], sp[0]["split"]


def _row_to_text(row):
    parts = []
    # prefer known text-ish fields, else any string field
    keys = [k for k in row if k.lower() in TEXT_HINTS] or \
           [k for k, v in row.items() if isinstance(v, str)]
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip()[:MAX_CHARS_PER_ROW])
    return " ".join(parts)


def _safe_name(node):
    """Windows-safe cache filename. Some dataset nodes carry dict-like config
    names (`{'evaluation': ...}`) with chars illegal on NTFS (: { } ' , etc.);
    sanitize to `_` and append a short hash of the full node so distinct nodes
    that collapse to the same sanitized stem never collide."""
    stem = re.sub(r'[^A-Za-z0-9._-]', "_", node.replace("/", "__"))[:120]
    h = hashlib.md5(node.encode("utf-8")).hexdigest()[:8]
    return f"{stem}.{h}.json"


def harvest_one(node, hf_id, n_rows):
    """Return dict(node, hf_id, config, split, n_samples, has_content, text)."""
    p = os.path.join(CACHE, _safe_name(node))
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            pass
    cs = first_config_split(hf_id)
    time.sleep(SLEEP)
    rec = {"node": node, "hf_id": hf_id, "config": None, "split": None,
           "n_samples": 0, "has_content": False, "text": ""}
    if cs:
        cfg, split = cs
        j = _get(f"{DSS}/rows", {"dataset": hf_id, "config": cfg, "split": split,
                                 "offset": 0, "length": min(n_rows, 100)})
        time.sleep(SLEEP)
        if j and j.get("rows"):
            texts = [_row_to_text(r.get("row", {})) for r in j["rows"]]
            texts = [t for t in texts if t]
            rec.update(config=cfg, split=split, n_samples=len(texts),
                       has_content=bool(texts), text=" \n ".join(texts)[:60000])
    os.makedirs(CACHE, exist_ok=True)
    json.dump(rec, open(p, "w", encoding="utf-8"))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rows", type=int, default=N_ROWS)
    ap.add_argument("--limit", type=int, default=None, help="cap nodes (debug)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    payload = torch.load(GRAPH, map_location="cpu", weights_only=False)
    udi = payload["unique_dataset_id"].sort_values("mappedID").reset_index(drop=True)
    hf_of = {}
    if os.path.exists(POOL):
        pool = pd.read_csv(POOL)
        hf_of = dict(zip(pool["dataset_node"], pool["hf_id_candidate"]))

    nodes = udi["dataset"].tolist()
    if args.limit:
        nodes = nodes[: args.limit]
    rows, t0 = [], time.time()
    for i, node in enumerate(nodes):
        hf_id = hf_of.get(node)
        if not isinstance(hf_id, str) or not hf_id:
            hf_id = node.split("/")[0]
        rec = harvest_one(node, hf_id, args.n_rows)
        rows.append({k: rec[k] for k in
                     ("node", "hf_id", "config", "split", "n_samples", "has_content")})
        if (i + 1) % 100 == 0:
            cov = sum(r["has_content"] for r in rows) / len(rows)
            print(f"[p1] {i+1}/{len(nodes)} coverage={cov:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "p1_content_index.csv"), index=False)
    report = {"nodes": len(df), "has_content": int(df["has_content"].sum()),
              "coverage": float(df["has_content"].mean()),
              "mean_samples": float(df["n_samples"].mean()),
              "cache_dir": CACHE}
    json.dump(report, open(os.path.join(OUT, "p1_content_report.json"), "w"), indent=2)
    print(f"[p1] DONE: {report}", flush=True)


if __name__ == "__main__":
    main()
