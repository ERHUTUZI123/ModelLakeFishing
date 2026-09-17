"""
retrain_modellens.py -- P5: retrain ModelLens's METHOD under identical, fully
reproducible conditions, so the A-axis is a fair architecture-vs-architecture
comparison (not "our full system vs their blinded release").

Fairness (kills the "did you cripple it?" objection):
  * ModelLens's OWN architecture: its `ModelLens` class from ModelLens/module/
    model/MLP.py (cross-feature MLP, id-emb, name-enc, desc, task/metric/size/
    family priors, learnable temperature).
  * ModelLens's OWN loss family + hyperparameters (args.json): listwise +
    pairwise + pointwise ensemble, lambda_list=0.5, lambda_pair=1.0,
    point_loss_weight=0.1, id_dropout_rate=0.1.
  * The SAME dataset & model representations our GNN uses: pulled straight from
    the graph -- dataset_desc = e_card (MiniLM over dataset_desp, name+task
    fallback), model_desc = e_desc (MiniLM over name+family+size). So the ONLY
    difference is the architecture. NOT blind: it sees the dataset description.
  * The SAME 12K universe, SAME root-aware held-out 517 gold queries, SAME
    global-metric harness (scale/global_metrics).

Held-out (cold) query datasets get dataset_id = [UNK] (their id was never
supervised) but KEEP their known description -- the genuine new-dataset path.

Run (repo root):
  ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.scale.retrain_modellens \
     --graph ModelLakeFishing/stage1BuildTransferGraph/hgraph_ml_v2_sub.pt --epochs 20
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
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
ML_REPO = os.path.join(_ROOT, "ModelLens")
LAKE = os.path.join(_HERE, "..", "stage1BuildTransferGraph", "artifacts",
                    "modellens_v2_lake")
EXPORT = os.path.join(_HERE, "..", "docs", "scale", "P3", "exports", "ml_sub_L1L3b")
OUT = os.path.join(_HERE, "..", "docs", "scale", "P5", "artifacts")
NODE_SEP = "␟"

sys.path.insert(0, _HERE + "/..")  # for scale import when run as module it's fine
from scale import global_metrics as GM  # noqa: E402


def build_modellens(n_models, n_datasets, n_tasks, n_metrics, n_size, n_fam, dev):
    if ML_REPO not in sys.path:
        sys.path.insert(0, ML_REPO)
    from module.model.registry import get_model_class  # noqa
    import module.model.MLP  # noqa: registers ModelLens
    # their hyperparameters (args.json) where they define the METHOD; dims for
    # the desc slots set to OUR MiniLM width (384) so the shared embedding is
    # in-distribution for the retrained model (no OOD, no padding).
    a = SimpleNamespace(
        model_name="ModelLens", use_id_emb=True,
        num_models=n_models, num_tasks=n_tasks, num_metrics=n_metrics,
        num_datasets=n_datasets, num_size_buckets=n_size, num_families=n_fam,
        model_dim=256, token_dim=64, task_dim=128, metric_dim=128,
        size_dim=64, family_dim=64, hidden_dim=512,
        model_desp_emb_dim=384, dataset_desp_emb_dim=384, dataset_id_emb_dim=256,
        dataset_desp_dim=1, dropout_rate=0.1,
        id_dropout_rate=0.1, model_id_dropout_rate=0.1, dataset_id_dropout_rate=0.1,
        use_size_prior=True, use_size_feature=True, use_family_prior=True,
        use_metric_feature=True, use_dataset_id_as_desp=True,
        use_dataset_id_emb=True, use_dataset_desc_emb=True,
        use_model_id_emb=True, use_model_name_emb=True, use_model_desc_emb=True,
        unknown_metric_id=0, tau=10.0,
        model_desp_emb_path=os.path.join(LAKE, "__absent__.npz"),
        model2id_path=os.path.join(LAKE, "__absent__.json"), data_name="mlfair",
    )
    model = get_model_class("ModelLens")(a)
    return model.to(dev), a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64, help="datasets per step")
    ap.add_argument("--nneg", type=int, default=64, help="random negatives / dataset")
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    payload = torch.load(args.graph, map_location="cpu", weights_only=False)
    data = payload["data"]
    umi = payload["unique_model_id"].sort_values("mappedID").reset_index(drop=True)
    udi = payload["unique_dataset_id"].sort_values("mappedID").reset_index(drop=True)
    names = umi["model"].tolist()
    n_m = len(names); n_d = len(udi)

    # shared features straight from the graph (identical to our GNN inputs)
    xm = data["model"].x.numpy()          # [n_m, 448] = e_name64 || e_desc384
    xd = data["dataset"].x.numpy()        # [n_d, 458] = e_name64 || e_card384 || stats10
    model_desc = torch.tensor(xm[:, 64:448], dtype=torch.float32)   # e_desc
    dataset_desc = torch.tensor(xd[:, 64:448], dtype=torch.float32) # e_card
    size_ids = data["model"].size_bucket_id.clone()
    fam_ids = data["model"].family_id.clone()
    task_ids_d = data["dataset"].task_type_id.clone()   # per dataset node
    n_size = int(payload["xm0_meta"]["num_size_buckets"])
    n_fam = int(payload["xm0_meta"]["num_families"])
    n_task = int(payload["xd0_meta"]["num_task_types"])

    # metric vocab from chosen_metric per node
    pool = pd.read_csv(os.path.join(LAKE, "ml_dataset_pool.csv"))
    node2metric = dict(zip(pool["dataset_node"], pool["chosen_metric"]))
    metrics = sorted(set(str(node2metric.get(nd, "unknown")) for nd in udi["dataset"]))
    metric2id = {"unknown_metric": 0, **{m: i for i, m in enumerate(metrics, 1)}}
    metric_ids_d = torch.tensor(
        [metric2id.get(str(node2metric.get(nd, "unknown")), 0) for nd in udi["dataset"]],
        dtype=torch.long)
    n_metric = len(metric2id)

    # supervision tuples + held-out split = same 517 test datasets as our export
    ei = data["model", "trained_on", "dataset"].edge_index.numpy()
    ev = data["model", "trained_on", "dataset"].edge_attr.numpy()
    gc = np.load(os.path.join(EXPORT, "gold_cands.npz"))
    test_d = set(int(k) for k in gc.files)
    did_map = dict(zip(udi["mappedID"], udi["dataset"]))
    # per dataset -> (model ids, values)
    by_d = {}
    for k in range(ei.shape[1]):
        m, d, v = int(ei[0, k]), int(ei[1, k]), float(ev[k])
        by_d.setdefault(d, [[], []])
        by_d[d][0].append(m); by_d[d][1].append(v)
    train_ds = [d for d in by_d if d not in test_d and len(by_d[d][0]) >= 2]
    print(f"[data] models {n_m} datasets {n_d} | train datasets {len(train_ds)} "
          f"| test(gold) {len(test_d)} | tasks {n_task} metrics {n_metric}")

    model, a = build_modellens(n_m, n_d, n_task, n_metric, n_size, n_fam, dev)
    # inject the SHARED representations. model_desc_matrix inits to 0 rows (its
    # model2id path is absent), so re-register at the right shape; dataset_desc_
    # matrix is (n_d+1, 384) already, fill its real rows (unk row stays zeros).
    del model.model_desc_matrix
    model.register_buffer("model_desc_matrix", model_desc.clone().to(dev))
    with torch.no_grad():
        model.dataset_desc_matrix[:n_d].copy_(dataset_desc.to(dev))
    model_desc = model_desc.to(dev); dataset_desc = dataset_desc.to(dev)
    size_ids = size_ids.to(dev); fam_ids = fam_ids.to(dev)
    all_names = names
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def forward_scores(d_ids_list, m_ids):
        """score [len(d), len(m_ids)] via model.forward per (dataset,model) flat."""
        pass  # scoring done inline below

    # ---- train: listwise + pairwise + pointwise (their ensemble/weights) ----
    LAM_LIST, LAM_PAIR, W_POINT = 0.5, 1.0, 0.1
    steps = 0
    for ep in range(args.epochs):
        model.train()
        np.random.shuffle(train_ds)
        ep_loss = 0.0; nb = 0
        for bstart in range(0, len(train_ds), args.batch):
            batch_ds = train_ds[bstart:bstart + args.batch]
            loss = torch.zeros((), device=dev)
            for d in batch_ds:
                pos_m, pos_v = by_d[d]
                pos_m = torch.tensor(pos_m, device=dev)
                pos_v = torch.tensor(pos_v, device=dev, dtype=torch.float32)
                neg = torch.randint(n_m, (args.nneg,), device=dev)
                cand = torch.cat([pos_m, neg])
                B = cand.numel()
                desp = torch.full((B, 1), float(d), device=dev)       # ds id -> id emb + desc
                s, z = model(
                    task_ids_d[d].repeat(B).to(dev), desp, cand,
                    [all_names[int(i)] for i in cand.tolist()],
                    size_ids[cand], fam_ids[cand],
                    metric_ids=metric_ids_d[d].repeat(B).to(dev))
                np_ = pos_m.numel()
                # listwise (ListNet top-1): target = softmax(values) on positives,
                # 0 on negatives; push scores to match value ordering + beat negs
                tgt = torch.zeros(B, device=dev)
                tgt[:np_] = torch.softmax(pos_v / 0.1, dim=0)
                loss_list = -(tgt * F.log_softmax(s, dim=0)).sum()
                # pairwise BPR within positives
                if np_ >= 2:
                    i, j = torch.randint(np_, (min(64, np_ * np_),), device=dev), \
                           torch.randint(np_, (min(64, np_ * np_),), device=dev)
                    mask = pos_v[i] > pos_v[j]
                    if mask.any():
                        loss_pair = -F.logsigmoid((s[i] - s[j])[mask]).mean()
                    else:
                        loss_pair = torch.zeros((), device=dev)
                else:
                    loss_pair = torch.zeros((), device=dev)
                # pointwise regression on positives
                loss_pt = F.mse_loss(torch.sigmoid(z[:np_]), pos_v)
                loss = loss + LAM_LIST * loss_list + LAM_PAIR * loss_pair + W_POINT * loss_pt
            loss = loss / len(batch_ds)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += float(loss.detach()); nb += 1; steps += 1
        print(f"  epoch {ep+1}/{args.epochs} loss {ep_loss/max(nb,1):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---- eval: cold held-out datasets (dataset_id=[UNK], desc known) --------
    model.eval()
    unk_ds = model.unk_dataset_id
    cache = model.build_model_cache(
        all_names, size_ids, all_model_family_ids=fam_ids, device=dev)
    cands = {int(k): (gc[k][0].astype(int), gc[k][1].astype(float)) for k in gc.files}
    roots = {int(r.mappedID): str(r.root) for r in udi.itertuples()}
    scores, wiring = {}, []
    with torch.no_grad():
        for d in cands:
            desp = torch.tensor([[float(unk_ds)]], device=dev)
            # overwrite the unk desc row with THIS query's known description
            model.dataset_desc_matrix[min(unk_ds, model.dataset_desc_matrix.shape[0]-1)] \
                 = dataset_desc[d]
            s = model.score_matrix(task_ids_d[d].view(1).to(dev), desp, cache,
                                   metric_ids=metric_ids_d[d].view(1).to(dev))
            s = s.squeeze(0).float().cpu().numpy()
            scores[int(d)] = s
            c, v = cands[int(d)]
            if len(c) >= 5 and np.std(s[c]) > 0 and np.std(v) > 0:
                wiring.append(np.corrcoef(s[c], v)[0, 1])
    agg, _ = GM.from_scores(scores, cands, roots)

    report = dict(
        method="ModelLens (retrained, non-blind, same conditions)",
        epochs=args.epochs, train_datasets=len(train_ds), universe=n_m,
        n_queries=len(cands), train_sec=round(time.time() - t0, 1),
        wiring_mean_pearson=float(np.mean(wiring)) if wiring else None,
        wiring_frac_pos=float(np.mean(np.array(wiring) > 0)) if wiring else None,
        global_metrics=agg,
        note=("their ModelLens architecture + their loss family/weights "
              "(list0.5/pair1.0/point0.1, id_dropout0.1); shared MiniLM dataset "
              "& model embeddings (= our e_card/e_desc); NOT blind."))
    with open(os.path.join(OUT, "retrained_modellens.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    torch.save(model.state_dict(), os.path.join(OUT, "retrained_modellens.pt"))

    print("\n=== ModelLens RETRAINED (non-blind, same 12K/517/harness) ===")
    print(f"  wiring Pearson(score,acc) {report['wiring_mean_pearson']:.3f} "
          f"(frac>0 {report['wiring_frac_pos']:.2f})")
    for k in ("gold@1", "gold@10", "top3@10", "gold-gap@10", "root_gold@10",
              "median_gold_rank"):
        print(f"  {k:18s}{agg[k]:.4f}")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
