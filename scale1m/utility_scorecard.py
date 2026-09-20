import argparse
import glob
import gzip
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(_HERE), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scale1m.hf_crawl import data_root, utcnow, write_json_atomic

K = 10
SEEDS = (0, 1, 2)
RUN_FMT = "RF_full_s%d_e25"
NAME_DIM, DESC_DIM = 64, 384

PIPELINE_TAGS = {
    "text-generation", "text-classification", "token-classification",
    "question-answering", "fill-mask", "summarization", "translation",
    "text2text-generation", "sentence-similarity", "feature-extraction",
    "zero-shot-classification", "automatic-speech-recognition",
    "audio-classification", "text-to-speech", "text-to-audio",
    "image-classification", "object-detection", "image-segmentation",
    "text-to-image", "image-to-image", "image-to-text", "image-text-to-text",
    "visual-question-answering", "video-classification", "depth-estimation",
    "reinforcement-learning", "tabular-classification", "tabular-regression",
    "robotics", "any-to-any", "sentence-transformers",
}
LIBRARY_TAGS = {"transformers", "diffusers", "peft", "sentence-transformers",
                "timm", "gguf", "mlx", "onnx", "keras", "flair", "espnet",
                "stable-baselines3", "ml-agents", "spacy", "fastai"}


def norm_task(t):
    return re.sub(r"[\s_]+", "-", str(t).strip().lower())


def stage_meta(args):
    t0 = time.time()
    lad = pd.read_parquet(args.ladder, columns=["model", "mappedID", "size_b",
                                                "family", "in_snapshot"])
    lad = lad.sort_values("mappedID")
    n = len(lad)
    row_of = pd.Series(lad["mappedID"].to_numpy(), index=lad["model"].astype(str))

    downloads = np.zeros(n, dtype=np.int64)
    flags = np.zeros(n, dtype=np.uint8)
    licensed = np.zeros(n, dtype=bool)
    endpoints = np.zeros(n, dtype=bool)
    safet = np.zeros(n, dtype=bool)
    has_lib = np.zeros(n, dtype=bool)
    task = np.full(n, "", dtype=object)

    for sp in sorted(glob.glob(os.path.join(args.candidates, "*.jsonl.gz"))):
        with gzip.open(sp, "rt", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                i = row_of.get(str(r.get("id")).strip().lower())
                if i is None:
                    continue
                i = int(i)
                downloads[i] = int(r.get("downloads") or 0)
                f = 0
                if str(r.get("gated")) not in ("False", "None"):
                    f |= 1
                if str(r.get("private")) not in ("False", "None"):
                    f |= 2
                if str(r.get("disabled")) not in ("False", "None"):
                    f |= 4
                flags[i] = f
                tags = r.get("tags") or []
                bare = {t for t in tags if ":" not in t}
                licensed[i] = any(t.startswith("license:") for t in tags)
                endpoints[i] = "endpoints_compatible" in bare
                safet[i] = "safetensors" in bare
                has_lib[i] = bool(bare & LIBRARY_TAGS)
                hit = bare & PIPELINE_TAGS
                task[i] = sorted(hit)[0] if hit else ""

    meta = pd.DataFrame({
        "mappedID": lad["mappedID"].to_numpy(), "model": lad["model"].to_numpy(),
        "in_snapshot": lad["in_snapshot"].to_numpy().astype(bool),
        "size_b": pd.to_numeric(lad["size_b"], errors="coerce").to_numpy(),
        "family": lad["family"].astype(str).to_numpy(),
        "downloads": downloads, "flags": flags, "licensed": licensed,
        "endpoints_compatible": endpoints, "safetensors": safet,
        "has_library_tag": has_lib, "pipeline_tag": task,
    })
    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, "model_meta.parquet")
    meta.to_parquet(p, index=False)
    print("[meta] %d rows in %.1fs -> %s" % (len(meta), time.time() - t0, p))
    print("  obtainable %.1f%% | licensed %.1f%% | endpoints_compatible %.1f%% | "
          "pipeline tag %.1f%% | library tag %.1f%%"
          % (100 * ((flags == 0) & meta["in_snapshot"]).mean(), 100 * licensed.mean(),
             100 * endpoints.mean(), 100 * (task != "").mean(), 100 * has_lib.mean()))
    return meta


def topk_from_scores(score, k):
    idx = np.argpartition(-score, k)[:k]
    return idx[np.argsort(-score[idx])]


def rank_of(score, target):
    return int((score > score[target]).sum()) + 1


def ours_and_text(args, cands, node_of, meta):
    import torch
    dev = args.device
    d0 = os.path.join(args.exports, RUN_FMT % 0)
    zm = torch.from_numpy(np.load(os.path.join(d0, "z_m_eval.npy")))
    zd = torch.from_numpy(np.load(os.path.join(d0, "z_d_eval.npy")))
    zm = torch.nn.functional.normalize(zm, dim=-1).to(dev)
    zd = torch.nn.functional.normalize(zd, dim=-1).to(dev)

    xm = np.load(os.path.join(args.feats, "x_m.npy"), mmap_mode="r")
    xd = np.load(os.path.join(args.graph, "x_dataset.npy"), mmap_mode="r")
    qd = np.array(xd[:, NAME_DIM:NAME_DIM + DESC_DIM], dtype=np.float32)
    qd /= (np.linalg.norm(qd, axis=1, keepdims=True) + 1e-12)
    qd_t = torch.from_numpy(qd).to(dev).half()

    qs = sorted(cands)
    out_ours, out_text = {}, {}
    for q in qs:
        s = (zm @ zd[q]).cpu().numpy()
        cand, acc = cands[q]
        gold = int(cand[int(np.argmax(acc))])
        out_ours[q] = (topk_from_scores(s, K), rank_of(s, gold))
    del zm, zd
    torch.cuda.empty_cache() if dev.startswith("cuda") else None

    N = xm.shape[0]
    gold_of = {q: int(cands[q][0][int(np.argmax(cands[q][1]))]) for q in qs}
    grows = np.array(sorted({gold_of[q] for q in qs}), dtype=np.int64)
    gvec = np.array(xm[grows, NAME_DIM:NAME_DIM + DESC_DIM], dtype=np.float32)
    gvec /= (np.linalg.norm(gvec, axis=1, keepdims=True) + 1e-12)
    gpos = {int(r): i for i, r in enumerate(grows)}
    qt = qd_t[[q for q in qs]]
    gs = (torch.from_numpy(gvec).to(dev).half() @ qt.T).float().cpu().numpy()
    gold_score = {q: float(gs[gpos[gold_of[q]], j]) for j, q in enumerate(qs)}

    best = {q: (np.full(0, 0, np.int64), np.full(0, 0.0, np.float32)) for q in qs}
    better = {q: 0 for q in qs}
    for lo in range(0, N, args.chunk):
        hi = min(lo + args.chunk, N)
        blk = np.array(xm[lo:hi, NAME_DIM:NAME_DIM + DESC_DIM], dtype=np.float32)
        blk /= (np.linalg.norm(blk, axis=1, keepdims=True) + 1e-12)
        bt = torch.from_numpy(blk).to(dev).half()
        sc = (bt @ qt.T).float().cpu().numpy()
        for j, q in enumerate(qs):
            col = sc[:, j]
            better[q] += int((col > gold_score[q]).sum())
            take = topk_from_scores(col, min(K, len(col) - 1))
            cid = np.concatenate([best[q][0], take + lo])
            cs = np.concatenate([best[q][1], col[take]])
            o = np.argsort(-cs)[:K]
            best[q] = (cid[o], cs[o])
        del bt, sc
    for q in qs:
        out_text[q] = (best[q][0], better[q] + 1)
    return out_ours, out_text


def cheap_sources(args, cands, node_of, meta, sup_task_pool, rng):
    dl = meta["downloads"].to_numpy()
    pop = topk_from_scores(dl.astype(np.float64), K)
    N = len(meta)
    out = {"popularity": {}, "random_task_pool": {}, "random_lake": {}}
    for q in sorted(cands):
        out["popularity"][q] = (pop, None)
        t = norm_task(node_of[q].split("\t", 1)[1])
        pool = sup_task_pool.get(t)
        out["random_task_pool"][q] = (
            (rng.choice(pool, min(K, len(pool)), replace=False), None)
            if pool is not None and len(pool) else (np.array([], np.int64), None))
        out["random_lake"][q] = (rng.integers(0, N, K), None)
    return out


def score_source(name, ranked, cands, node_of, meta, observed, task_models, N):
    dl = meta["downloads"].to_numpy()
    flags = meta["flags"].to_numpy()
    insnap = meta["in_snapshot"].to_numpy()
    lic = meta["licensed"].to_numpy()
    ep = meta["endpoints_compatible"].to_numpy()
    lib = meta["has_library_tag"].to_numpy()
    ptag = meta["pipeline_tag"].to_numpy()
    fam = meta["family"].to_numpy()
    size = meta["size_b"].to_numpy()
    obtainable = insnap & (flags == 0)

    acc = {k: [] for k in (
        "recorded_gold@10", "recorded_top3@10", "record_coverage@10",
        "unknown_on_query@10", "same_task_evidence@10", "available@10",
        "licensed@10", "endpoints_compatible@10", "has_library_tag@10",
        "task_tag_match@10", "family_diversity@10", "duplicate_family_rate@10",
        "at_least_one_feasible@10", "median_size_b@10")}
    ranks = []
    for q in sorted(cands):
        top, gr = ranked[q]
        top = np.asarray(top, dtype=np.int64)
        if top.size == 0:
            continue
        cand, a = cands[q]
        gold = int(cand[int(np.argmax(a))])
        top3 = set(cand[np.argsort(-a)[:3]].tolist())
        obs = observed.get(q, set())
        t = norm_task(node_of[q].split("\t", 1)[1])
        pool = task_models.get(t)
        k = len(top)

        acc["recorded_gold@10"].append(float(gold in set(top.tolist())))
        acc["recorded_top3@10"].append(float(bool(top3 & set(top.tolist()))))
        acc["record_coverage@10"].append(float(np.mean([int(m) in obs for m in top])))
        acc["unknown_on_query@10"].append(float(np.mean([int(m) not in obs for m in top])))
        acc["same_task_evidence@10"].append(
            float(np.isin(top, pool).mean()) if pool is not None and len(pool) else 0.0)
        acc["available@10"].append(float(obtainable[top].mean()))
        acc["licensed@10"].append(float(lic[top].mean()))
        acc["endpoints_compatible@10"].append(float(ep[top].mean()))
        acc["has_library_tag@10"].append(float(lib[top].mean()))
        acc["task_tag_match@10"].append(float(np.mean([norm_task(x) == t for x in ptag[top]])))
        acc["family_diversity@10"].append(len(set(fam[top].tolist())) / k)
        cnt = pd.Series(fam[top]).value_counts()
        acc["duplicate_family_rate@10"].append(float((cnt.max() - 1) / k) if len(cnt) else 0.0)
        feasible = obtainable[top] & lib[top]
        acc["at_least_one_feasible@10"].append(float(feasible.any()))
        s = size[top]
        acc["median_size_b@10"].append(float(np.nanmedian(s)) if np.isfinite(s).any()
                                       else float("nan"))
        if gr is not None:
            ranks.append(gr)

    row = {"source": name, "n_queries": len(acc["recorded_gold@10"])}
    for k, v in acc.items():
        row[k] = float(np.nanmean(v)) if v else float("nan")
    if ranks:
        row["median_recorded_gold_rank"] = float(np.median(ranks))
        row["median_rank_over_N"] = float(np.median(ranks)) / N
        row["vs_random"] = row["recorded_gold@10"] / (K / N)
    return row


def stage_score(args):
    import torch
    t0 = time.time()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    meta = pd.read_parquet(os.path.join(args.out, "model_meta.parquet"))
    N = len(meta)
    row_of = pd.Series(meta["mappedID"].to_numpy(), index=meta["model"].astype(str))

    sup = pd.read_parquet(args.supervision, columns=["node", "model"])
    sup["m_row"] = row_of.reindex(sup["model"].astype(str)).to_numpy()
    sup = sup.dropna(subset=["m_row"])
    sup["m_row"] = sup["m_row"].astype(int)
    sup["task"] = sup["node"].astype(str).str.split("\t").str[1].map(norm_task)
    task_models = {t: np.unique(g["m_row"].to_numpy()) for t, g in sup.groupby("task")}
    by_node = {k: set(v) for k, v in sup.groupby(sup["node"].astype(str))["m_row"].apply(set).items()}

    d0 = os.path.join(args.exports, RUN_FMT % 0)
    di = pd.read_parquet(os.path.join(d0, "dataset_ids.parquet")).sort_values("mappedID")
    nm = di.set_index("mappedID")["dataset"].astype(str)
    with np.load(os.path.join(d0, "gold_cands.npz")) as z:
        cands = {int(k): (z[k][0].astype(np.int64), z[k][1].astype(float)) for k in z.files}
    node_of = {q: nm[q] for q in cands}
    observed = {q: by_node.get(node_of[q], set()) for q in cands}

    ours, text = ours_and_text(args, cands, node_of, meta)
    rng = np.random.default_rng(args.seed)
    cheap = cheap_sources(args, cands, node_of, meta, task_models, rng)

    rows = [score_source("ours (graph + trained)", ours, cands, node_of, meta,
                         observed, task_models, N),
            score_source("text-only MiniLM", text, cands, node_of, meta,
                         observed, task_models, N)]
    for k in ("popularity", "random_task_pool", "random_lake"):
        rows.append(score_source(k, cheap[k], cands, node_of, meta,
                                 observed, task_models, N))

    frozen = []
    for q in sorted(cands):
        for r, m in enumerate(ours[q][0], 1):
            frozen.append({"node": node_of[q], "rank": r,
                           "model": meta["model"].iloc[int(m)],
                           "model_row": int(m),
                           "observed_on_node": int(m) in observed[q]})
    fz = pd.DataFrame(frozen)
    fz.to_parquet(os.path.join(args.out, "t0_recommendations.parquet"), index=False)

    rep = {"written_at": utcnow(), "seconds": round(time.time() - t0, 1),
           "N": N, "k": K, "n_queries": len(cands),
           "queries_are": "de-duplicated test queries of split seed 0",
           "rows": rows,
           "t0_frozen_rows": int(len(fz)),
           "note": "no model was downloaded or executed; layer 1 measures "
                   "recovery of historical records, layer 2 measures whether a "
                   "returned model can be obtained and run at all"}
    write_json_atomic(os.path.join(args.out, "F9_SCORECARD.json"), rep)
    df = pd.DataFrame(rows).set_index("source")
    pd.set_option("display.width", 200)
    print(df[["recorded_gold@10", "recorded_top3@10", "same_task_evidence@10",
              "record_coverage@10", "available@10", "task_tag_match@10",
              "family_diversity@10", "at_least_one_feasible@10"]].to_string())
    return rep


def main(argv=None):
    d = os.path.join(data_root(), "data1m")
    p = argparse.ArgumentParser(description="F9: recommendation-utility scorecard")
    p.add_argument("--stage", required=True, choices=["meta", "score"])
    p.add_argument("--candidates", default=os.path.join(d, "candidates_full"))
    p.add_argument("--ladder", default=os.path.join(d, "ladder_rf", "full_model_ids.parquet"))
    p.add_argument("--exports", default=os.path.join(d, "exports_rf"))
    p.add_argument("--feats", default=os.path.join(d, "feats_rf"))
    p.add_argument("--graph", default=os.path.join(d, "graphs", "hgraph_rf"))
    p.add_argument("--supervision",
                   default=os.path.join(d, "rf", "canon", "supervision_merged.parquet"))
    p.add_argument("--out", default=os.path.join(d, "utility_rf"))
    p.add_argument("--device", default=None)
    p.add_argument("--chunk", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    {"meta": stage_meta, "score": stage_score}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
