"""
hf1000d_phase4.py -- Phase 4/5 materialization for the hf1000d_2000m pilot.

Embeds selected datasets with the existing xd0-compatible gpt-neo-125m probe
pipeline and materializes a Stage-1 data dir, WITHOUT changing Stage-1/Stage-2
logic and WITHOUT fabricating edges.

Inputs (from Phases 1-3):
  hf1000d_2000m/selected_2000_models.csv
  hf1000d_2000m/selected_1000_datasets.csv
  hf1000d_2000m/performance_edges_single_metric.csv     (real model-index, Table A)

Outputs into dataset_embed/data_hf1000d_2000m/ (isolated; selected via MLF_DATA_DIR):
  embedded_dataset/domain_similarity/EleutherAI_gpt-neo-125m/<key>_feature.npy  (--phase embed)
  model_config_dataset.csv, records.csv, lineage_records.csv                    (--phase materialize)

Embed order prioritizes edge-bearing datasets (the trainable core) so an
interrupted run still covers the graph's supervised nodes. Resumable (skips
existing .npy). A lightweight loadability probe precedes each full embed; failures
record a precise reason.

Then build (Phase 5):
  MLF_DATA_DIR=.../data_hf1000d_2000m python build_graph.py \
    --contain_model_feature True --contain_rich_dataset_feature True \
    --gnn_method SAGEConv_without_transfer --out hgraph_hf1000d_2000m_xm0_xd0.pt
"""

import argparse
import ast
import os
import sys
from collections import Counter
from itertools import islice

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.join(HERE, "hf1000d_2000m")
DATA = os.path.join(HERE, "dataset_embed", "data_hf1000d_2000m")
EMB_DIR = os.path.join(DATA, "embedded_dataset", "domain_similarity", "EleutherAI_gpt-neo-125m")
PROBE = "EleutherAI/gpt-neo-125m"
N_SAMPLES = 256
MAX_LEN = 128

_TEXT_PRIORITY = ["text", "sentence", "sentence1", "premise", "question", "document",
                  "content", "tweet", "review", "context", "passage", "query", "title",
                  "comment_text", "sentence2", "hypothesis", "answer", "abstract", "body"]


def _sanitize(name):
    return str(name).replace("/", "_").replace(" ", "-")


# token-classification datasets carry text as a list of tokens, not a string
_TOKEN_FIELDS = ("tokens", "words", "ner_tags_str", "token")


def _example_text(ex):
    parts = []
    for k in _TEXT_PRIORITY:
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v)
    # token-list inputs (NER/POS): join the token sequence into a sentence
    if not parts:
        for k in _TOKEN_FIELDS:
            v = ex.get(k)
            if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                parts.append(" ".join(v))
                break
    if not parts:
        for k, v in ex.items():
            if isinstance(v, str) and v.strip():
                parts.append(v)
            elif isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                parts.append(" ".join(v))
            elif isinstance(v, dict):
                parts += [x for x in v.values() if isinstance(x, str)]
    return " ".join(parts)[:2000] if parts else ""


# canon_key -> working hf_id override: script-only/defunct repos that datasets-4.x
# cannot load, remapped to live parquet mirrors (verified via datasets-server)
_HFID_REMAP = {
    "amazon_reviews_multi": "mteb/amazon_reviews_multi",
    "banking77": "mteb/banking77",
    "conll2003": "tner/conll2003",
    "mtop_domain": "WillHeld/mtop",
    "mtop_intent": "WillHeld/mtop",
}


_DS_SERVER = "https://datasets-server.huggingface.co"
_SPLIT_PREF = ("train", "validation", "dev", "test")


def _api_pick(hf_id, prefer_config=None):
    """(config, split) via the datasets-server, preferring train + en/default."""
    import requests
    r = requests.get(f"{_DS_SERVER}/splits", params={"dataset": hf_id}, timeout=30)
    if r.status_code != 200:
        return None, None
    items = r.json().get("splits", [])
    if not items:
        return None, None

    def score(it):
        c, s = it.get("config", ""), it.get("split", "")
        sc = {"train": 10, "validation": 6, "dev": 6, "test": 4}.get(s, 1)
        if prefer_config and c == prefer_config:
            sc += 5
        if c in ("en", "english", "default", "plain_text"):
            sc += 2
        return -sc
    items.sort(key=score)
    return items[0].get("config"), items[0].get("split")


def _api_rows(hf_id, config, split, n):
    """Fetch up to n example dicts via the parquet REST API (bypasses dataset scripts)."""
    import requests
    rows = []
    off = 0
    while len(rows) < n:
        r = requests.get(f"{_DS_SERVER}/rows",
                         params={"dataset": hf_id, "config": config, "split": split,
                                 "offset": off, "length": min(100, n - len(rows))}, timeout=30)
        if r.status_code != 200:
            break
        data = r.json().get("rows", [])
        if not data:
            break
        rows += [d.get("row", {}) for d in data]
        off += len(data)
        if len(data) < 100:
            break
    return rows


def _resolve_load(load_dataset, hf_id, config, canon_key):
    """Return (examples_iterable, n_train_hint). Tries streaming load across splits
    and configs; on failure (incl. datasets-4.x 'scripts no longer supported') falls
    back to the datasets-server parquet REST API so script/parquet-only datasets still
    yield examples."""
    hf_id = _HFID_REMAP.get(canon_key, hf_id)
    if not config and "/" in canon_key and canon_key.split("/")[0] in ("glue", "tweet_eval", "super_glue"):
        config = canon_key.split("/", 1)[1]
    cfgs = [config, None] if config else [None]
    last = None
    for cfg in cfgs:
        for split in _SPLIT_PREF:
            try:
                ds = load_dataset(hf_id, cfg, split=split, streaming=True)
                n_train = None
                try:
                    sp = getattr(getattr(ds, "info", None), "splits", None)
                    if sp and "train" in sp and getattr(sp["train"], "num_examples", None):
                        n_train = float(sp["train"].num_examples)
                except Exception:
                    pass
                return ds, n_train
            except Exception as e:
                last = e
                msg = str(e).lower()
                if "bad split" not in msg and "unknown split" not in msg and "split" not in msg:
                    break   # config problem, not split -> try next config
    # REST API fallback (script-free)
    try:
        c, s = _api_pick(hf_id, prefer_config=config)
        if c is not None:
            rows = _api_rows(hf_id, c, s, N_SAMPLES)
            if rows:
                return rows, None
    except Exception as e:
        last = e
    raise last if last else RuntimeError("load failed")


_LABEL_KEYS = ["label", "labels", "class", "intent", "target", "category", "stars",
               "sentiment", "gold_label", "coarse_label", "fine_label"]
_TEXT_FIELD_KEYS = set(_TEXT_PRIORITY) | {"sentence2", "hypothesis", "answer"}


def _example_label(ex):
    """A single scalar label for one example (int/str/bool) or None. Token-level
    list labels (ner_tags) are intentionally skipped -- they are not a dataset-level
    class label and would distort H(Y)/probe stats."""
    for k in _LABEL_KEYS:
        if k in ex:
            v = ex[k]
            if isinstance(v, (int, bool)) or (isinstance(v, str) and v.strip()):
                return v
            if isinstance(v, float) and not (v != v):   # finite float -> regression target
                return float(v)
    return None


def _num_text_fields(ex):
    return sum(1 for k, v in ex.items()
              if k in _TEXT_FIELD_KEYS and isinstance(v, str) and v.strip())


LOAD_TIMEOUT = 90    # seconds; abandon a dataset whose streaming load/iter stalls


def _load_sample_with_timeout(load_dataset, hf_id, config, timeout, canon_key=""):
    """Load (streaming) + collect up to N_SAMPLES aligned (text,label,len,fields) in a
    worker thread; return the tuple or 'TIMEOUT'. A hung worker leaks (blocked on IO)
    but the main embed loop continues to the next dataset."""
    import concurrent.futures as cf

    def work():
        ds, n_train = _resolve_load(load_dataset, hf_id, config, canon_key)
        texts, raw_labels, word_lens, n_fields = [], [], [], []
        for ex in islice(ds, N_SAMPLES * 3):
            t = _example_text(ex)
            if t:
                texts.append(t)
                raw_labels.append(_example_label(ex))
                word_lens.append(len(t.split()))
                n_fields.append(_num_text_fields(ex))
            if len(texts) >= N_SAMPLES:
                break
        return texts, raw_labels, word_lens, n_fields, n_train

    ex = cf.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(work)
    try:
        out = fut.result(timeout=timeout)
        ex.shutdown(wait=False)
        return out
    except cf.TimeoutError:
        ex.shutdown(wait=False)
        return "TIMEOUT"
    except Exception:
        ex.shutdown(wait=False)
        raise


def _save_aux(key, hf_id, raw_labels, word_lens, n_fields, texts, n_train_hint):
    """Persist the xd0 aux sidecar `<key>_aux.npz`, ROW-ALIGNED to the cloud .npy.
    Holds per-example label ids + word lengths and dataset-level scalars needed for
    the full e_stats spec (H(Y), class ratio, length dist, vocab richness, num text
    fields, n_train, linear-probe acc, fisher ratio computed later in xd0)."""
    # label -> int id (regression floats kept separate). -1 == missing.
    is_float = [isinstance(v, float) for v in raw_labels]
    regression = sum(is_float) > 0.5 * len(raw_labels) and len(set(map(str, raw_labels))) > 6
    if regression:
        label_ids = np.array([float(v) if isinstance(v, float) else np.nan
                              for v in raw_labels], dtype=np.float32)
        label_kind = "regression"
    else:
        uniq, label_ids = {}, []
        for v in raw_labels:
            if v is None:
                label_ids.append(-1)
            else:
                label_ids.append(uniq.setdefault(str(v), len(uniq)))
        label_ids = np.array(label_ids, dtype=np.float32)
        label_kind = "class"
    # vocab richness: type/token ratio over the sample
    toks = " ".join(texts).lower().split()
    vocab_richness = float(len(set(toks)) / max(1, len(toks)))
    # best-effort true train size (from streaming ds.info if available, else sample count)
    n_train = float(n_train_hint) if n_train_hint else float(len(texts))
    np.savez(os.path.join(EMB_DIR, _sanitize(key) + "_aux.npz"),
             label_ids=label_ids, word_lens=np.array(word_lens, dtype=np.float32),
             n_fields=np.array(n_fields, dtype=np.float32),
             vocab_richness=np.float32(vocab_richness), n_train=np.float32(n_train),
             label_kind=label_kind)


def _edge_counts():
    p = os.path.join(PILOT, "performance_edges_single_metric.csv")
    if not os.path.exists(p):
        return {}
    e = pd.read_csv(p)
    if e.empty:
        return {}
    return e.drop_duplicates(["model_id", "canon_key"]).groupby("canon_key")["model_id"].nunique().to_dict()


def embed_datasets(limit=None, edges_only=False):
    os.makedirs(EMB_DIR, exist_ok=True)
    sel = pd.read_csv(os.path.join(PILOT, "selected_1000_datasets.csv"))
    ec = _edge_counts()
    sel["edgecnt"] = sel["canon_key"].map(lambda k: ec.get(k, 0))
    sel["trcnt"] = sel.get("trainability", 0)
    if edges_only:                         # recover the trainable core fast
        sel = sel[sel["edgecnt"] > 0]
    # edge-bearing first (desc), then predicted-trainable, then downloads
    sel = sel.sort_values(["edgecnt", "trcnt", "listed_downloads"], ascending=False)

    import torch
    from transformers import AutoTokenizer, AutoModel
    from datasets import load_dataset
    import datasets as _hfds
    # fail fast on flaky streaming hosts instead of 20x retry loops (the tail of the
    # 1000-dataset list is full of datasets that disconnect mid-stream)
    try:
        _hfds.config.STREAMING_READ_MAX_RETRIES = 2
        _hfds.config.STREAMING_READ_RETRY_INTERVAL = 1
    except Exception:
        pass
    print(f"loading probe {PROBE} ...")
    tok = AutoTokenizer.from_pretrained(PROBE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(PROBE)
    model.eval()
    dev = "cpu"

    status = []
    done = 0
    for r in sel.itertuples():
        key = r.canon_key
        hf_id = r.hf_id
        config = None if (pd.isna(r.config) if hasattr(r, "config") else True) else r.config
        out = os.path.join(EMB_DIR, _sanitize(key) + "_feature.npy")
        if os.path.exists(out):
            status.append((key, hf_id, int(r.edgecnt), "cached"))
            continue
        if limit is not None and done >= limit:
            break
        try:
            # streaming load + sampling under a wall-clock timeout: some datasets
            # stall indefinitely on network/builder. Run in a worker thread and
            # abandon it (leaked, blocked on IO) if it exceeds LOAD_TIMEOUT.
            res = _load_sample_with_timeout(load_dataset, hf_id, config, LOAD_TIMEOUT, canon_key=key)
            if res == "TIMEOUT":
                status.append((key, hf_id, int(r.edgecnt), "skip:timeout"))
                continue
            texts, raw_labels, word_lens, n_fields, n_train_hint = res
            if len(texts) < 8:
                status.append((key, hf_id, int(r.edgecnt), "skip:no-text"))
                continue
            feats = []
            with torch.no_grad():
                for i in range(0, len(texts), 16):
                    batch = tok(texts[i:i + 16], return_tensors="pt", truncation=True,
                                max_length=MAX_LEN, padding=True)
                    out_h = model(**{k: v.to(dev) for k, v in batch.items()},
                                  output_hidden_states=True)
                    feats.append(out_h.hidden_states[-1].mean(dim=1).cpu().numpy())
            arr = np.concatenate(feats, axis=0).astype(np.float32)
            np.save(out, arr)
            _save_aux(key, hf_id, raw_labels, word_lens, n_fields, texts, n_train_hint)
            status.append((key, hf_id, int(r.edgecnt), f"ok:{arr.shape[0]}x{arr.shape[1]}"))
            done += 1
            if done % 10 == 0 or int(r.edgecnt) > 0:
                print(f"  [{done}] {key} <- {hf_id} edges={int(r.edgecnt)} {arr.shape}")
        except Exception as e:
            status.append((key, hf_id, int(r.edgecnt), f"skip:{type(e).__name__}"))
        # periodic checkpoint of status
        if len(status) % 50 == 0:
            pd.DataFrame(status, columns=["canon_key", "hf_id", "edges", "status"]).to_csv(
                os.path.join(PILOT, "embedding_status.csv"), index=False)

    sdf = pd.DataFrame(status, columns=["canon_key", "hf_id", "edges", "status"])
    sdf.to_csv(os.path.join(PILOT, "embedding_status.csv"), index=False)
    _embed_report(sdf, sel)


def _embed_report(sdf, sel):
    ok = sdf["status"].str.startswith(("ok", "cached"))
    n_ok = int(ok.sum())
    edgeful_ok = int(((sdf["edges"] > 0) & ok).sum())
    reasons = dict(Counter(s.split(":")[0] for s in sdf["status"]))
    targets = {
        "embedded datasets >= 400": (n_ok, 400, n_ok >= 400),
        "edge-bearing embedded datasets >= 100": (edgeful_ok, 100, edgeful_ok >= 100),
    }
    lines = ["# Embedding survival report (Phase 4, hf1000d_2000m)", ""]
    lines.append(f"Selected datasets: {len(sel)}. Attempted: {len(sdf)}. Embedded ok/cached: **{n_ok}**.\n")
    lines.append("## Targets")
    for k, (got, tgt, okk) in targets.items():
        lines.append(f"- {k}: **{got}** ({'OK' if okk else 'SHORT'})")
    lines.append(f"\n## Status breakdown\n{reasons}")
    lines.append("\n## Skip reasons (top)")
    skip = sdf[~ok]
    for k, v in Counter(skip["status"]).most_common(20):
        lines.append(f"- {k}: {v}")
    with open(os.path.join(PILOT, "embedding_survival_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[embed] {n_ok}/{len(sel)} embedded; edge-bearing embedded {edgeful_ok}. reasons={reasons}")


def materialize(dataset_mode="embedded"):
    """dataset_mode: 'embedded' -> all embedded datasets are nodes (edgeless ones via
    model=NaN node rows + similarity edges); 'edged' -> only edge-bearing embedded."""
    sel_m = pd.read_csv(os.path.join(PILOT, "selected_2000_models.csv"))
    sel_d = pd.read_csv(os.path.join(PILOT, "selected_1000_datasets.csv"))
    edges = pd.read_csv(os.path.join(PILOT, "performance_edges_single_metric.csv"))
    embedded = {f[:-len("_feature.npy")] for f in os.listdir(EMB_DIR) if f.endswith("_feature.npy")}
    print(f"[materialize] {len(embedded)} embedded datasets available; mode={dataset_mode}")

    key2labels = dict(zip(sel_d["canon_key"], sel_d.get("label_names", pd.Series([None] * len(sel_d)))))

    # keep only edges whose dataset embedded; ONE metric per dataset (dominant)
    edges = edges[edges["canon_key"].map(lambda k: _sanitize(k) in embedded)].copy()
    keep_rows = []
    for key, sub in edges.groupby("canon_key"):
        mt = sub["metric_name"].value_counts().idxmax()
        s = sub[sub["metric_name"] == mt].drop_duplicates(["model_id"])
        for _, e in s.iterrows():
            keep_rows.append(dict(model=e["model_id"], dataset=key,
                                  accuracy=float(e["metric_value"]), metric=mt))
    edf = pd.DataFrame(keep_rows)
    edge_datasets = set(edf["dataset"]) if not edf.empty else set()
    print(f"[materialize] usable trained_on edges: {len(edf)} on {len(edge_datasets)} datasets")

    arch_map = dict(zip(sel_m["model_id"], sel_m.get("architecture", pd.Series(dtype=str))))
    type_map = dict(zip(sel_m["model_id"], sel_m.get("model_type", pd.Series(dtype=str))))
    param_map = dict(zip(sel_m["model_id"], sel_m.get("parameter_count", pd.Series(dtype=float))))
    i = 0

    def row(model, dataset, acc):
        nonlocal i
        a = arch_map.get(model)
        labels = None
        if dataset is not None:
            lab = key2labels.get(dataset)
            if isinstance(lab, str) and lab.startswith("["):
                labels = lab
            elif isinstance(lab, list) and lab:
                labels = str(lab)
        r = {"Unnamed: 0": i, "model": model,
             "architectures": f"['{a}']" if isinstance(a, str) and a else None,
             "number_of_labels": None, "labels": labels, "model_type": type_map.get(model),
             "number_of_parameters": param_map.get(model), "memory_consumption": None,
             "dataset": dataset, "accuracy": acc}
        i += 1
        return r

    mc_rows = []
    edge_models = set(edf["model"]) if not edf.empty else set()
    for _, e in edf.iterrows():
        mc_rows.append(row(e["model"], e["dataset"], e["accuracy"]))
    # node-only models (selected but no edge) -> keep as model nodes
    for m in sel_m["model_id"]:
        if m not in edge_models:
            mc_rows.append(row(m, None, None))
    # node-only datasets (embedded but edgeless) -> dataset nodes via model=NaN rows
    if dataset_mode == "embedded":
        embedded_keys = {k for k in sel_d["canon_key"] if _sanitize(k) in embedded}
        for k in embedded_keys - edge_datasets:
            mc_rows.append(row(np.nan, k, np.nan))
    mc = pd.DataFrame(mc_rows)

    # lineage (provenance; NOT installed under without_transfer build)
    lin_rows = []
    if "base_model" in sel_m.columns:
        node_models = set(sel_m["model_id"])
        for _, m in sel_m.iterrows():
            b = m["base_model"]
            if isinstance(b, str) and b and b in node_models and b != m["model_id"]:
                lin_rows.append(dict(model=m["model_id"], relation="finetune", base_model=b))
    lin = pd.DataFrame(lin_rows or [{"model": None, "relation": None, "base_model": None}])

    os.makedirs(DATA, exist_ok=True)
    mc.to_csv(os.path.join(DATA, "model_config_dataset.csv"), index=False)
    rec = edf.rename(columns={"dataset": "finetuned_dataset", "accuracy": "eval_accuracy"}) if not edf.empty \
        else pd.DataFrame(columns=["model", "finetuned_dataset", "eval_accuracy"])
    if not rec.empty:
        rec["model_name"] = rec["model"]
        rec["task_type"] = "text-classification"
    rec.to_csv(os.path.join(DATA, "records.csv"), index=False)
    lin.to_csv(os.path.join(DATA, "lineage_records.csv"), index=False)
    pd.DataFrame(columns=["model", "target_dataset", "score"]).to_csv(
        os.path.join(DATA, "transferability_score_records.csv"))

    n_ds_nodes = len(set(mc["dataset"].dropna()))
    lines = ["# Materialize report (Phase 4/5, hf1000d_2000m)", ""]
    lines.append(f"- data dir: `{DATA}`")
    lines.append(f"- dataset_mode: {dataset_mode}")
    lines.append(f"- model nodes: {mc['model'].nunique()} ({len(edge_models)} with >=1 edge, "
                 f"{len(sel_m) - len(edge_models)} node-only)")
    lines.append(f"- dataset nodes (embedded): {n_ds_nodes} ({len(edge_datasets)} edge-bearing, "
                 f"{n_ds_nodes - len(edge_datasets)} node-only)")
    lines.append(f"- trained_on edges: {len(edf)}")
    lines.append(f"- lineage rows (provenance): {len(lin_rows)} (NOT installed under without_transfer)")
    with open(os.path.join(PILOT, "materialize_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[materialize] model nodes={mc['model'].nunique()}, dataset nodes={n_ds_nodes}, "
          f"edges={len(edf)} -> {DATA}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["embed", "materialize"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--edges_only", action="store_true",
                    help="embed only edge-bearing datasets (recover the trainable core fast)")
    ap.add_argument("--dataset_mode", default="embedded", choices=["embedded", "edged"])
    args = ap.parse_args()
    if args.phase == "embed":
        embed_datasets(args.limit, edges_only=args.edges_only)
    else:
        materialize(args.dataset_mode)


if __name__ == "__main__":
    main()
