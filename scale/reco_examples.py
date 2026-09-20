import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

from scale import modellens_adapter as MA
from scale.head_to_head import EXPORT, LAKE, NODE_SEP, cache_subset

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "scale", "P4", "artifacts")


def md_table(rows, header):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="how many example datasets")
    ap.add_argument("--min-depth", type=int, default=40)
    args = ap.parse_args()

    zm = np.load(os.path.join(EXPORT, "z_m_eval.npy"))
    zd = np.load(os.path.join(EXPORT, "z_d_eval.npy"))
    gc = np.load(os.path.join(EXPORT, "gold_cands.npz"))
    mid = pd.read_csv(os.path.join(EXPORT, "model_ids.csv")).sort_values("mappedID")
    did = pd.read_csv(os.path.join(EXPORT, "dataset_ids.csv"))
    names = mid["model"].tolist()
    node_of = dict(zip(did["mappedID"], did["dataset"]))
    cands = {int(k): (gc[k][0].astype(int), gc[k][1].astype(float)) for k in gc.files}

    rows = []
    for d, (c, a) in cands.items():
        node = str(node_of[int(d)])
        task = node.split(NODE_SEP)[1] if NODE_SEP in node else ""
        rows.append((d, node, task, len(c), float(a.max())))
    rows.sort(key=lambda r: -r[3])
    picked, seen_task = [], set()
    for d, node, task, depth, best in rows:
        if depth < args.min_depth:
            continue
        if task in seen_task:
            continue
        seen_task.add(task)
        picked.append((d, node, task, depth, best))
        if len(picked) >= args.n:
            break

    model, margs, dev, missing, unexpected = MA.load_modellens()
    model2id, task2id, metric2id, family2id, profile, size_bucket = MA.build_vocabs()
    pool = pd.read_csv(os.path.join(LAKE, "ml_dataset_pool.csv"))
    node2metric = dict(zip(pool["dataset_node"], pool["chosen_metric"]))
    fam_allowed = {str(k).strip().lower(): int(v) for k, v in family2id.items()}
    UNK = {"unknown", "", "none", "null", "nan"}
    gids, size_ids, fam_ids = [], [], []
    for nm in names:
        gids.append(int(model2id.get(nm, model.unk_model_id)))
        p = profile.get(nm); s_id = f_id = 0
        if isinstance(p, dict):
            f = str(p.get("family", "unknown")).strip().lower()
            f_id = fam_allowed.get(f, 0) if f not in UNK else 0
            s = str(p.get("size", "unknown")).strip().lower()
            if s not in UNK:
                try:
                    s_id = int(min(np.searchsorted(size_bucket, float(s), side="right"),
                                   len(size_bucket)))
                except ValueError:
                    pass
        size_ids.append(s_id); fam_ids.append(f_id)
    cache = cache_subset(model, names, np.array(gids), torch.tensor(size_ids),
                         torch.tensor(fam_ids), dev)
    unk_ds = model.unk_dataset_id
    desc_in = torch.tensor([[float(unk_ds)]], device=dev)

    zmn = zm / (np.linalg.norm(zm, axis=1, keepdims=True) + 1e-12)
    zdn = zd / (np.linalg.norm(zd, axis=1, keepdims=True) + 1e-12)

    doc = ["### 附录 P4-R：同数据集推荐结果实例（我方 vs ModelLens，同 12K 宇宙）", "",
           "> **怎么读**：两系统对**同一个 held-out 查询**排同一批 12,000 个模型；"
           "`acc` = 该模型在这个数据集上的公开记录归一化准确率，`—` = 该模型没在这个数据集上被评测过"
           "（不代表它差）。我方 = held-out（无泄漏）嵌入的纯 MIPS `z_d·z_m`；"
           "ModelLens = 其自身打分。**ModelLens 是 blind release 版**（`dataset_desc` 与 "
           "`dataset2id` 均未发布，看不见「这是哪个数据集」，P2 §3.1）—— 故它倾向返回 "
           "(task,metric)-泛化强但非本数据集专属的模型；此对比是「公开件可复现」口径，非同台方法优劣。", ""]

    for d, node, task, depth, best in picked:
        c, a = cands[int(d)]
        acc_of = {int(m): float(v) for m, v in zip(c, a)}
        dsname = node.split(NODE_SEP)[0]

        s_ours = zmn @ zdn[int(d)]
        s_ml = model.score_matrix(torch.tensor([int(task2id.get(task, 0))], device=dev),
                                  desc_in, cache,
                                  metric_ids=torch.tensor([int(metric2id.get(str(node2metric.get(node, "")), 0))], device=dev)
                                  ).squeeze(0).float().cpu().numpy()
        top_o = np.argsort(-s_ours)[:10]
        top_m = np.argsort(-s_ml)[:10]

        def fmt(v):
            return "—" if v is None else f"**{v:.3f}**"

        o_rows = [(i + 1, f"`{names[int(mi)]}`", fmt(acc_of.get(int(mi))),
                   f"{s_ours[int(mi)]:.3f}") for i, mi in enumerate(top_o)]
        m_rows = [(i + 1, f"`{names[int(mi)]}`", fmt(acc_of.get(int(mi))),
                   f"{s_ml[int(mi)]:.2f}") for i, mi in enumerate(top_m)]
        n_lab_o = sum(1 for mi in top_o if int(mi) in acc_of)
        n_lab_m = sum(1 for mi in top_m if int(mi) in acc_of)

        doc += [f"#### {dsname} — {task}", "",
                f"该数据集最优 acc = **{best:.3f}**，{depth} 个带标签模型（候选宇宙 12,000）。",
                f"我方 top-10 带标签 {n_lab_o}/10；ModelLens top-10 带标签 {n_lab_m}/10。", "",
                "**我方 L1L3b（MIPS，held-out）：**", "",
                md_table(o_rows, ["rank", "model", "acc", "MIPS"]), "",
                "**ModelLens（blind release 版）：**", "",
                md_table(m_rows, ["rank", "model", "acc", "score"]), ""]

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "reco_examples.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(doc) + "\n")
    print(f"wrote {len(picked)} examples -> {path}")
    for d, node, task, depth, best in picked:
        print(f"  {node.split(NODE_SEP)[0][:40]:42s} task={task[:24]:26s} depth={depth} best={best:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
