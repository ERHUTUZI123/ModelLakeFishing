"""
modellens_adapter.py -- P2: run the REAL ModelLens model (its own source +
published checkpoint) as a full-lake scorer, so it can be evaluated on OUR
global metrics (scale/global_metrics.py).

Faithfulness: we import ModelLens's own `ModelLens` class and load its published
`ModelLens.pt` with strict=False. Every trained weight (model_desc_matrix,
_id_emb, task/metric/size/family embeddings, backbone, heads, temperature)
loads from the checkpoint. The candidate universe is the FULL 47,242 models in
GLOBAL-ID ORDER, so build_model_cache's `arange(M)` indexes _id_emb and
model_desc_matrix correctly -- no id remapping needed.

D-5 (dataset-desc matrix, unpublished) -- recorded per the user ruling:
  (c) LOWER BOUND: the checkpoint has NO dataset_desc_matrix, so it stays zeros
      (their own code falls back to zeros). This is the honest floor.
  (a) MAIN: we rebuild the dataset-desc slot with the SAME MiniLM encoder as our
      e_card (all-MiniLM-L6-v2, 384-d), placed in the first 384 of the 1536-d
      slot -- their own `use_dim = min(...)` partial-fill convention. NOT the
      original encoder (unknown, likely OpenAI 1536-d); provenance stamped on
      every output.

Documented approximations (both minor priors; dominant signals are exact):
  * size bucket: `ds.modelid2bucket` is unpublished, so ModelLens's own default
    is all-zeros. We reconstruct via searchsorted on args.size_bucket (best
    effort); a --size-zeros flag reproduces their published-artifact default.
  * held-out datasets have no entry in a (also unpublished) dataset2id, so every
    query uses unk_dataset_id -- the genuine cold-start path for a new dataset.

Run (from ModelLakeFishing/):
    .\\.venv\\Scripts\\python.exe -m scale.modellens_adapter --wiring-check
    .\\.venv\\Scripts\\python.exe -m scale.modellens_adapter --eval --desc a
    .\\.venv\\Scripts\\python.exe -m scale.modellens_adapter --eval --desc c
"""

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from scale.pull_corpus import data_root
from scale import global_metrics as GM

ML_REPO = r"D:\research\model_lake\codes\ModelLens"
RAW = os.path.join(data_root(), "modellens_v2", "raw")
CKPT_DIR = os.path.join(data_root(), "modellens_ckpt", "raw")
LAKE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "stage1BuildTransferGraph", "artifacts", "modellens_v2_lake")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "scale", "P2", "artifacts")
NODE_SEP = "␟"
ENCODER = "all-MiniLM-L6-v2"


def load_modellens(size_zeros: bool = False):
    """Reconstruct ModelLens from its own source + published checkpoint."""
    if ML_REPO not in sys.path:
        sys.path.insert(0, ML_REPO)
    # its package imports are rooted at the repo dir
    os.chdir(ML_REPO)
    from module.model.registry import get_model_class
    import module.model.MLP  # noqa: F401  (registers "ModelLens")

    with open(os.path.join(CKPT_DIR, "args.json"), encoding="utf-8") as fh:
        args = SimpleNamespace(**json.load(fh))
    # eval-only: no dropout, no wandb, cpu/gpu
    args.is_train = False
    args.use_wandb = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = dev
    # Point the desc-matrix builders at the frozen model2id so the buffers size
    # to (47242, 1536) / (85938, 1536) and the checkpoint loads over them.
    # The .npz emb files stay "missing" -> built as zeros -> then the checkpoint
    # overwrites model_desc_matrix with the real trained values. dataset_desc
    # has no checkpoint key, so it stays zeros = D-5 lower bound (option c).
    args.model2id_path = os.path.join(RAW, "model2id.json")
    args.model_desp_emb_path = os.path.join(RAW, "__absent_model_desp__.npz")

    cls = get_model_class(args.model_name)          # MLPMetricFull -> ModelLens
    model = cls(args)                               # builds zeros for missing files
    sd = torch.load(os.path.join(CKPT_DIR, "ModelLens.pt"), map_location="cpu",
                    weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval().to(dev)
    return model, args, dev, list(missing), list(unexpected)


def build_vocabs():
    def j(name):
        with open(os.path.join(RAW, name), encoding="utf-8") as fh:
            return json.load(fh)
    model2id = j("model2id.json")
    task2id = j("task2id.json")
    metric2id = j("metric2id.json")
    family2id = j("family2id.json")
    profile = j("model_profile.json")
    size_bucket = json.load(open(os.path.join(CKPT_DIR, "args.json"),
                                 encoding="utf-8"))["size_bucket"]
    return model2id, task2id, metric2id, family2id, profile, np.array(size_bucket)


def candidate_tensors(model2id, family2id, profile, size_bucket, size_zeros):
    """Full 47,242 candidates in GLOBAL-ID ORDER (id == row index)."""
    n = max(model2id.values()) + 1
    names = [None] * n
    for name, i in model2id.items():
        names[int(i)] = name
    fam_allowed = {str(k).strip().lower(): int(v) for k, v in family2id.items()}
    size_ids = np.zeros(n, dtype=np.int64)
    fam_ids = np.zeros(n, dtype=np.int64)
    UNK = {"unknown", "", "none", "null", "nan"}
    for name, i in model2id.items():
        i = int(i)
        p = profile.get(name)
        if isinstance(p, dict):
            f = str(p.get("family", "unknown")).strip().lower()
            fam_ids[i] = fam_allowed.get(f, 0) if f not in UNK else 0
            if not size_zeros:
                s = str(p.get("size", "unknown")).strip().lower()
                if s not in UNK:
                    try:
                        sb = float(s)
                        # searchsorted on their own boundary list (best-effort)
                        size_ids[i] = int(min(np.searchsorted(size_bucket, sb,
                                          side="right"), len(size_bucket)))
                    except ValueError:
                        pass
    return names, size_ids, fam_ids


def minilm_encode(texts):
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer(ENCODER, device=dev)
    return enc.encode(texts, batch_size=256, convert_to_numpy=True,
                      show_progress_bar=False, normalize_embeddings=False).astype(np.float32)


def load_queries():
    """Gold query nodes (depth>=10) + labeled (model, acc) over the candidate
    universe, using GLOBAL model ids so they align with the candidate tensors."""
    obs = pd.read_parquet(os.path.join(LAKE, "ml_observations.parquet"))
    pool = pd.read_csv(os.path.join(LAKE, "ml_dataset_pool.csv"))
    gold = pool[pool["gold_evaluable"]].copy()
    return obs, gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiring-check", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--desc", choices=["a", "c"], default="c",
                    help="a=rebuilt MiniLM desc (main); c=zeros (lower bound)")
    ap.add_argument("--size-zeros", action="store_true",
                    help="reproduce ModelLens's published-artifact default (all size buckets 0)")
    ap.add_argument("--limit", type=int, default=0, help="cap #queries for a quick run")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    model, margs, dev, missing, unexpected = load_modellens(args.size_zeros)
    print(f"[load] device={dev}  missing_keys={len(missing)}  unexpected={len(unexpected)}")
    print(f"       dataset_desc_matrix present in ckpt: "
          f"{'dataset_desc_matrix' not in missing}  (D-5: zeros unless --desc a)")

    model2id, task2id, metric2id, family2id, profile, size_bucket = build_vocabs()
    names, size_ids_np, fam_ids_np = candidate_tensors(
        model2id, family2id, profile, size_bucket, args.size_zeros)
    M = len(names)
    size_ids = torch.as_tensor(size_ids_np, device=dev)
    fam_ids = torch.as_tensor(fam_ids_np, device=dev)
    print(f"[cand] {M} models | size!=0 {int((size_ids_np>0).sum())} | "
          f"family!=0 {int((fam_ids_np>0).sum())}")

    cache = model.build_model_cache(names, size_ids, all_model_family_ids=fam_ids,
                                    device=dev)

    obs, gold = load_queries()
    if args.limit:
        gold = gold.head(args.limit)
    # map node -> (global model ids, accs), task, metric
    obs = obs[obs["dataset_node"].isin(set(gold["dataset_node"]))]
    obs = obs.assign(gid=obs["model_id"].map(lambda m: model2id.get(m)))
    obs = obs[obs["gid"].notna()]
    grp = obs.groupby("dataset_node")

    unk_ds = model.unk_dataset_id
    desc_mat = model.dataset_desc_matrix          # buffer, zeros (D-5)
    # _encode_dataset CLAMPS ds_id to desc_mat.shape[0]-1 when reading, so inject
    # option-(a) vectors into that same clamped row.
    have_desc_buf = desc_mat.shape[0] > 0
    inj_row = desc_mat.shape[0] - 1

    # optionally rebuild desc (option a) with MiniLM over dataset_desp
    node_desc_vec = {}
    if args.desc == "a":
        nodes = list(gold["dataset_node"])
        texts = []
        pool_desc = dict(zip(gold["dataset_node"], gold["desc"].fillna("")))
        for nd in nodes:
            d = nd.split(NODE_SEP)[0] if NODE_SEP in nd else nd
            t = str(pool_desc.get(nd, "")) or d
            texts.append(t)
        vecs = minilm_encode(texts)               # [n, 384]
        for nd, v in zip(nodes, vecs):
            node_desc_vec[nd] = v

    # ---- score each query over the full universe ------------------------
    per_scores, cands, roots = {}, {}, {}
    wiring_corr = []
    n_done = 0
    for nd, sub in grp:
        row = gold[gold["dataset_node"] == nd].iloc[0]
        task = nd.split(NODE_SEP)[1] if NODE_SEP in nd else ""
        task_id = int(task2id.get(task, 0))
        metric_id = int(metric2id.get(str(row["chosen_metric"]), 0))

        # option a: inject this query's desc into the unk row (first 384 dims)
        if args.desc == "a" and have_desc_buf and nd in node_desc_vec:
            with torch.no_grad():
                desc_mat[inj_row].zero_()
                v = torch.from_numpy(node_desc_vec[nd]).to(dev)
                desc_mat[inj_row, :v.numel()] = v

        desc_in = torch.tensor([[float(unk_ds)]], device=dev)   # id in slot 0
        task_ids = torch.tensor([task_id], device=dev)
        metric_ids = torch.tensor([metric_id], device=dev)
        with torch.no_grad():
            s = model.score_matrix(task_ids, desc_in, cache,
                                   metric_ids=metric_ids).squeeze(0)  # [M]
        s = s.detach().float().cpu().numpy()

        gids = sub["gid"].astype(int).to_numpy()
        accs = sub["value_norm"].to_numpy()
        per_scores[nd] = s
        cands[nd] = (gids, accs)
        roots[nd] = str(row["root_a"])

        # wiring: correlation of ModelLens score vs observed acc over labeled set
        if len(gids) >= 5:
            sc = s[gids]
            if np.std(sc) > 0 and np.std(accs) > 0:
                wiring_corr.append(np.corrcoef(sc, accs)[0, 1])
        n_done += 1
        if n_done % 200 == 0:
            print(f"  scored {n_done}/{len(gold)} queries "
                  f"({time.time()-t0:.0f}s)", flush=True)

    result = {"desc_mode": args.desc, "size_zeros": bool(args.size_zeros),
              "n_queries": len(per_scores), "candidate_universe": M,
              "missing_keys": len(missing), "unexpected_keys": len(unexpected)}

    if args.wiring_check or wiring_corr:
        wc = float(np.mean(wiring_corr)) if wiring_corr else float("nan")
        result["wiring_mean_pearson_score_vs_acc"] = wc
        result["wiring_n_queries"] = len(wiring_corr)
        result["wiring_frac_positive"] = (float(np.mean(np.array(wiring_corr) > 0))
                                          if wiring_corr else float("nan"))
        print(f"[wiring] mean Pearson(score, observed acc) over "
              f"{len(wiring_corr)} queries = {wc:.4f} "
              f"(frac>0 = {result['wiring_frac_positive']:.2f})")

    if args.eval:
        agg, _ = GM.from_scores(per_scores, cands, roots)
        result["global_metrics"] = agg
        print("\n=== ModelLens under OUR global metrics "
              f"(desc={args.desc}, universe={M}) ===")
        for k in ("gold@1", "gold@10", "top3@10", "gold-gap@10",
                  "root_gold@1", "root_gold@10", "root_top3@10", "median_gold_rank"):
            if k in agg:
                print(f"  {k:20s} {agg[k]:.4f}")

    tag = f"desc{args.desc}{'_sz0' if args.size_zeros else ''}"
    with open(os.path.join(OUT, f"modellens_baseline_{tag}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nruntime {time.time()-t0:.0f}s -> {OUT}\\modellens_baseline_{tag}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
