"""
d0_build_graph.py -- v4 W1 (C1/C2/D1): build the D0 lake graph
`hgraph_d0_v1.pt` from the Phase-A/B artifacts. Zero network.

Lake finalization (plan v4 §0.3 W1-C1, user decision):
  models   : STRICT intake -- has_model_index ∨ has_lineage (family-only pool
             dropped);
  datasets : every valid dataset node with >= 1 bounded chosen-metric
             observation against a strict model (gold 613 is a subset).

Supervision (D1): trained_on edges straight from d0_observations.parquet --
per (node, model) the node's chosen bounded metric, runs already median-folded
at source (the 42%-duplicate-edge lesson is closed in the parquet layer, not
re-fixed here). Values normalized to [0,1] (percent-scale auto-detected /100).

Features (C2, D0-v1 scope):
  xm0 frozen = [e_name 64 (MD5-hash mean, seed 42) || e_desc 384 (MiniLM over a
               metadata descriptor string)] -- same 448 layout as the 2K graph;
  learnable ids: size_bucket (safetensors.total, unknown->0),
               family (rule inference + FAMILY_MIN_COUNT fold, Other=0).
  xd0 frozen = [e_name 64 (seed 43) || e_card 384 (MiniLM over dataset
               descriptor) || e_stats 10] = 458;
  learnable ids: task_type (vocab built NATIVELY from observation dominant_task,
               min-count fold, Other=0 -- the L3 repair is built-in, not
               patched), n_class/arity = unknown(0) in v1.
  SCOPE NOTE: the 2K graph's 3840-d gpt-neo probe views are NOT rebuilt here --
  they need dataset sample downloads + GPU-days, and E13 says representation is
  not the lever. Recorded as the main D0-v1 feature deviation.

Edge schema replicates the 2K contract exactly (4 relation families):
  (model, trained_on, dataset)+attr / (dataset, rev_trained_on, model)+attr
  (dataset, similar_to, dataset)+attr   [KNN k=20 over e_card, cosine]
  (model, is_base_of, model)+attr / (model, rev_is_base_of, model)+attr

Run (from stage1BuildTransferGraph/):
    ../.venv/Scripts/python.exe d0_build_graph.py
"""

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from dataset_embed.xm0_builder import (  # noqa: E402
    build_name_embeddings, param_count_to_size_bucket,
    load_or_update_family_vocab, build_family_ids, NUM_SIZE_BUCKETS,
)
from dataset_embed.utils.fetch_metadata import _infer_one_family  # noqa: E402

ART = os.path.join(_HERE, "artifacts", "d0_lake")
CACHES_M = [os.path.join(_HERE, "hf1000d_2000m", "hf_cache", "models"),
            os.path.join(_HERE, "d0_lake_cache", "models")]
CACHES_D = [os.path.join(_HERE, "hf1000d_2000m", "hf_cache", "datasets"),
            os.path.join(_HERE, "d0_lake_cache", "datasets")]
OUT_GRAPH = os.path.join(_HERE, "hgraph_d0_v1.pt")
OUT_REPORT = os.path.join(ART, "d0_graph_report.json")

BOUNDED = {"accuracy", "f1", "matthews_correlation", "pearson", "spearman",
           "exact_match", "map", "mrr", "rougeL"}
TASK_MIN_COUNT = 5
SIM_K = 20
ENCODER = "all-MiniLM-L6-v2"


def _load_json(mid, caches):
    safe = mid.replace("/", "__") + ".json"
    for c in caches:
        p = os.path.join(c, safe)
        if os.path.exists(p):
            try:
                d = json.load(open(p, encoding="utf-8"))
                return d if isinstance(d, dict) else None
            except Exception:
                return None
    return None


def _norm_value(v):
    """Bounded metrics to [0,1]; percent scale (>1.5) auto-divided by 100."""
    v = float(v)
    if v > 1.5:
        v = v / 100.0
    return float(min(max(v, 0.0), 1.0))


def _minilm(texts, tag):
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER)
    emb = enc.encode(texts, batch_size=256, show_progress_bar=False,
                     convert_to_numpy=True, normalize_embeddings=False)
    print(f"[{tag}] MiniLM encoded {len(texts)} texts -> {emb.shape}", flush=True)
    return emb.astype(np.float32)


def model_descriptor(d, mid):
    if not d:
        return mid.replace("/", " ").replace("-", " ")
    tags = [t for t in (d.get("tags") or []) if ":" not in t][:12]
    card = d.get("cardData") or {}
    dsets = card.get("datasets") or []
    if isinstance(dsets, str):
        dsets = [dsets]
    parts = [mid.replace("/", " "), d.get("pipeline_tag") or "",
             d.get("library_name") or "", " ".join(tags),
             "datasets: " + " ".join(map(str, dsets[:8])) if dsets else ""]
    return " ".join(p for p in parts if p)


def load_content_texts(ds_nodes):
    """v5 P1: per-node sample text harvested by p1_content_harvest.py, keyed by
    the SAME _safe_name so nodes match by string (mappedID-independent). Returns
    (texts[list], has_content[bool array]) in ds_nodes order. Missing/empty ->
    "" + False (the node keeps its metadata views; only the content view is absent)."""
    from p1_content_harvest import _safe_name, CACHE as CONTENT_CACHE
    texts, has = [], []
    for node in ds_nodes:
        p = os.path.join(CONTENT_CACHE, _safe_name(node))
        t = ""
        if os.path.exists(p):
            try:
                rec = json.load(open(p, encoding="utf-8"))
                if rec.get("has_content"):
                    t = rec.get("text", "") or ""
            except Exception:
                t = ""
        texts.append(t[:20000])          # cap for MiniLM (512-token trunc anyway)
        has.append(bool(t))
    return texts, np.array(has, dtype=bool)


def dataset_descriptor(d, node):
    base = node.replace("/", " ").replace("_", " ").replace("-", " ")
    if not d:
        return base
    card = d.get("cardData") or {}
    tasks = card.get("task_categories") or []
    if isinstance(tasks, str):
        tasks = [tasks]
    desc = (d.get("description") or "")[:400]
    tags = [t for t in (d.get("tags") or []) if ":" not in t][:10]
    return " ".join(p for p in [base, " ".join(map(str, tasks)),
                                " ".join(tags), desc] if p)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", action="store_true",
                    help="v5 P1/P3: add the e_content view (MiniLM over P1 sample "
                         "text) and write the combined graph to a distinct path; "
                         "default OFF reproduces hgraph_d0_v1.pt exactly")
    args = ap.parse_args()
    out_graph = OUT_GRAPH.replace(".pt", "_content.pt") if args.content else OUT_GRAPH
    out_report = OUT_REPORT.replace(".json", "_content.json") if args.content else OUT_REPORT

    # ── C1: lake finalization ────────────────────────────────────────────────
    intake = pd.read_csv(os.path.join(ART, "d0_model_intake.csv"))
    strict = intake[(intake["has_model_index"] | intake["has_lineage"])
                    & ~intake["disabled"] & ~intake["private"]]
    model_ids = sorted(strict["model_id"])
    m_of = {m: i for i, m in enumerate(model_ids)}
    print(f"[lake] strict models: {len(model_ids)}", flush=True)

    pool = pd.read_csv(os.path.join(ART, "d0_dataset_pool.csv"))
    chosen = dict(zip(pool["dataset_node"], pool["chosen_metric"]))

    obs = pd.read_parquet(os.path.join(ART, "d0_observations.parquet"))
    obs = obs[obs["dataset_node"].notna()
              & ~obs["dataset_node"].isin(["", "-", "-/default"])
              & obs["model_id"].isin(m_of)
              & obs["metric_canonical"].isin(BOUNDED)]
    obs = obs[obs.apply(lambda r: chosen.get(r["dataset_node"]) == r["metric_canonical"],
                        axis=1)].copy()
    obs["value_norm"] = obs["value"].map(_norm_value)

    ds_nodes = sorted(obs["dataset_node"].unique())
    d_of = {n: i for i, n in enumerate(ds_nodes)}
    roots = [n.split("/")[0] for n in ds_nodes]
    print(f"[lake] dataset nodes: {len(ds_nodes)} | roots: {len(set(roots))} "
          f"| trained_on pairs: {len(obs)}", flush=True)

    # ── D1: supervision edges (provenance-clean at source) ───────────────────
    assert not obs.duplicated(["dataset_node", "model_id"]).any(), \
        "observations must be one row per (node, model) after chosen-metric filter"
    src = torch.tensor([m_of[m] for m in obs["model_id"]], dtype=torch.long)
    dst = torch.tensor([d_of[n] for n in obs["dataset_node"]], dtype=torch.long)
    attr = torch.tensor(obs["value_norm"].values, dtype=torch.float32)   # 1-D: 2K contract

    # lineage among lake models
    lin = strict[strict["lineage_base"].notna()]
    lb, ld = [], []
    for _, r in lin.iterrows():
        base = str(r["lineage_base"])
        if base in m_of and r["model_id"] in m_of and base != r["model_id"]:
            lb.append(m_of[base]); ld.append(m_of[r["model_id"]])
    print(f"[lake] lineage edges: {len(lb)}", flush=True)

    # ── C2: model features ───────────────────────────────────────────────────
    jsons = {m: _load_json(m, CACHES_M) for m in model_ids}
    e_name = build_name_embeddings(model_ids, token_dim=64, seed=42)
    e_desc = _minilm([model_descriptor(jsons[m], m) for m in model_ids], "xm0.desc")
    xm = np.concatenate([e_name, e_desc], axis=1)

    def _params(d):
        st = (d or {}).get("safetensors") or {}
        return st.get("total") if isinstance(st, dict) else None
    size_id = torch.tensor([param_count_to_size_bucket(_params(jsons[m]))
                            for m in model_ids], dtype=torch.long)
    fams = [_infer_one_family(m) for m in model_ids]
    family_vocab = load_or_update_family_vocab(fams, vocab_path=None)
    fam_id = torch.as_tensor(build_family_ids(fams, family_vocab), dtype=torch.long)
    print(f"[xm0] frozen {xm.shape} | size unknown share "
          f"{float((size_id == 0).float().mean()):.2f} | families {len(family_vocab)}",
          flush=True)

    # ── C2: dataset features + native task vocab ─────────────────────────────
    djson = {n: _load_json(n.split("/")[0], CACHES_D)
             or _load_json(str(pool.set_index('dataset_node')['hf_id_candidate']
                               .to_dict().get(n) or ""), CACHES_D)
             for n in ds_nodes}
    dn_emb = build_name_embeddings(ds_nodes, token_dim=64, seed=43)
    dc_emb = _minilm([dataset_descriptor(djson[n], n) for n in ds_nodes], "xd0.card")

    # v5 P1: e_content view -- MiniLM over harvested sample text. Missing-content
    # nodes get a ZERO content vector (explicit "no signal"); their metadata
    # views (dn/dc/stats) are untouched, so the fallback IS metadata-only. A
    # has_content flag rides in xd_stats so the GNN can gate the content block.
    has_content = None
    if args.content:
        ctexts, has_content = load_content_texts(ds_nodes)
        e_content = np.zeros((len(ds_nodes), 384), dtype=np.float32)
        idx = np.where(has_content)[0]
        if idx.size:
            enc = _minilm([ctexts[i] for i in idx], "xd0.content")
            e_content[idx] = enc
        print(f"[xd0] e_content: {int(has_content.sum())}/{len(ds_nodes)} nodes "
              f"({has_content.mean():.2f}) carry sample text", flush=True)

    g = obs.groupby("dataset_node")
    stats = {n: (np.log1p(len(gr)), np.log1p(gr["n_runs"].sum()),
                 gr["value_norm"].mean(), gr["value_norm"].std() or 0.0,
                 gr["value_norm"].min(), gr["value_norm"].max())
             for n, gr in g}
    root_sizes = pd.Series(roots).value_counts().to_dict()
    hc_col = has_content.astype(np.float32) if has_content is not None \
        else np.zeros(len(ds_nodes), dtype=np.float32)
    xd_stats = np.nan_to_num(np.array(
        [[*stats[n], np.log1p(root_sizes[n.split("/")[0]]),
          float(pool.set_index("dataset_node")["gold_evaluable"].get(n, False)),
          float(hc_col[i]), 0.0] for i, n in enumerate(ds_nodes)],
        dtype=np.float32))                                 # slot 8 = has_content flag
    views = [dn_emb, dc_emb]
    if args.content:
        views.append(e_content)                            # e_content before stats
    views.append(xd_stats)
    xd = np.concatenate(views, axis=1)

    task_counts = obs.groupby("dataset_node")["task"].agg(
        lambda s: s.str.lower().mode().iat[0] if s.notna().any() else "")
    counts = task_counts[task_counts != ""].value_counts()
    kept = sorted(counts[counts >= TASK_MIN_COUNT].index)
    task_vocab = {"Other": 0, **{t: i for i, t in enumerate(kept, 1)}}
    tt_id = torch.tensor([task_vocab.get(task_counts.get(n, ""), 0)
                          for n in ds_nodes], dtype=torch.long)
    print(f"[xd0] frozen {xd.shape} | task vocab {len(task_vocab)} | "
          f"Other share {float((tt_id == 0).float().mean()):.2f}", flush=True)

    # similar_to: KNN over e_card (cosine)
    c = dc_emb / (np.linalg.norm(dc_emb, axis=1, keepdims=True) + 1e-8)
    sims = c @ c.T
    np.fill_diagonal(sims, -1)
    k = min(SIM_K, len(ds_nodes) - 1)
    nbr = np.argsort(-sims, axis=1)[:, :k]
    s_src = np.repeat(np.arange(len(ds_nodes)), k)
    s_dst = nbr.flatten()
    s_attr = sims[s_src, s_dst].astype(np.float32)

    # ── assemble + contracts ─────────────────────────────────────────────────
    data = HeteroData()
    data["model"].node_id = torch.arange(len(model_ids))
    data["model"].x = torch.from_numpy(xm)
    data["model"].size_bucket_id = size_id
    data["model"].family_id = fam_id
    data["dataset"].node_id = torch.arange(len(ds_nodes))
    data["dataset"].x = torch.from_numpy(xd)
    data["dataset"].task_type_id = tt_id
    data["dataset"].n_class_bucket_id = torch.zeros(len(ds_nodes), dtype=torch.long)
    data["dataset"].arity_id = torch.zeros(len(ds_nodes), dtype=torch.long)
    if args.content:
        # rides on the node store so it auto-slices with any loader subgraph;
        # the P1 ablation stratifies eval on this column (has_content vs not)
        data["dataset"].has_content = torch.from_numpy(hc_col)

    data["model", "trained_on", "dataset"].edge_index = torch.stack([src, dst])
    data["model", "trained_on", "dataset"].edge_attr = attr
    data["dataset", "rev_trained_on", "model"].edge_index = torch.stack([dst, src])
    data["dataset", "rev_trained_on", "model"].edge_attr = attr.clone()
    data["dataset", "similar_to", "dataset"].edge_index = torch.tensor(
        np.stack([s_src, s_dst]), dtype=torch.long)
    data["dataset", "similar_to", "dataset"].edge_attr = torch.from_numpy(s_attr)
    li = torch.tensor([lb, ld], dtype=torch.long) if lb else torch.zeros(2, 0, dtype=torch.long)
    la = torch.ones(li.shape[1])
    data["model", "is_base_of", "model"].edge_index = li
    data["model", "is_base_of", "model"].edge_attr = la
    data["model", "rev_is_base_of", "model"].edge_index = li.flip(0)
    data["model", "rev_is_base_of", "model"].edge_attr = la.clone()

    # Stage-1 sanity + Stage-2 contract
    dataset_frozen = 458 + (384 if args.content else 0)   # +e_content view
    assert data["model"].x.shape == (len(model_ids), 448)
    assert data["dataset"].x.shape == (len(ds_nodes), dataset_frozen)
    assert not torch.isnan(data["model"].x).any() and not torch.isnan(data["dataset"].x).any()
    assert int(src.max()) < len(model_ids) and int(dst.max()) < len(ds_nodes)
    assert int(fam_id.max()) < len(family_vocab) and family_vocab["Other"] == 0
    assert int(tt_id.max()) < len(task_vocab) and task_vocab["Other"] == 0
    assert attr.dim() == 1 and (attr >= 0).all() and (attr <= 1).all()
    assert len(data.edge_types) == 5
    umi = pd.DataFrame({"model": model_ids, "mappedID": range(len(model_ids))})
    udi = pd.DataFrame({"dataset": ds_nodes, "mappedID": range(len(ds_nodes)),
                        "root": roots})

    payload = {
        "data": data,
        "xm0_meta": {"num_size_buckets": NUM_SIZE_BUCKETS,
                     "num_families": len(family_vocab),
                     "family_vocab": family_vocab, "name_dim": 64, "desc_dim": 384},
        "xd0_meta": {"num_task_types": len(task_vocab),
                     "task_type_vocab": task_vocab,
                     "n_class_buckets": 7, "num_arities": 5,
                     "arity_vocab": {"unknown": 0, "single": 1, "pair": 2,
                                     "multi": 3, "span_or_qa": 4},
                     "view_dims": ({"e_name": 64, "e_card": 384, "e_content": 384,
                                    "e_stats": 10} if args.content else
                                   {"e_name": 64, "e_card": 384, "e_stats": 10}),
                     "encoder_name": ENCODER,
                     "has_content_view": bool(args.content),
                     "probe_views": "DEFERRED (D0-v1 scope note)"},
        "unique_model_id": umi,
        "unique_dataset_id": udi,
    }
    torch.save(payload, out_graph)
    sha = hashlib.sha256(open(out_graph, "rb").read()).hexdigest()

    gold = pool[pool["gold_evaluable"]]["dataset_node"]
    report = {
        "graph": os.path.basename(out_graph), "sha256": sha,
        "content_view": bool(args.content),
        "content_coverage": (float(has_content.mean()) if has_content is not None else 0.0),
        "content_gold_coverage": (float(has_content[[d_of[n] for n in gold if n in d_of]].mean())
                                  if has_content is not None else 0.0),
        "dataset_frozen_dim": int(data["dataset"].x.shape[1]),
        "models": len(model_ids), "dataset_nodes": len(ds_nodes),
        "roots": len(set(roots)), "trained_on_edges": int(attr.numel()),
        "lineage_edges": len(lb), "similar_to_edges": int(len(s_src)),
        "gold_nodes_in_graph": int(gold.isin(d_of).sum()),
        "task_vocab": len(task_vocab),
        "task_other_share": float((tt_id == 0).float().mean()),
        "size_unknown_share": float((size_id == 0).float().mean()),
        "num_families": len(family_vocab),
        "feature_scope_note": "xd0 probe views deferred; metadata + content views",
    }
    json.dump(report, open(out_report, "w"), indent=2)
    for k_, v_ in report.items():
        print(f"{k_}: {v_}", flush=True)


if __name__ == "__main__":
    main()
