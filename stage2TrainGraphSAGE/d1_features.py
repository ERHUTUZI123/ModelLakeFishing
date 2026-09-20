import os

import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_D1_DIR = os.path.join(_HERE, "..", "stage1BuildTransferGraph", "artifacts", "d1_features")
IDS_CSV = os.path.join(_D1_DIR, "model_task_ids.csv")
VOCAB_CSV = os.path.join(_D1_DIR, "task_vocab.csv")
_L3_DIR = os.path.join(_HERE, "..", "stage1BuildTransferGraph", "artifacts",
                       "task_type_enrichment")
L3_PATCH_CSV = os.path.join(_L3_DIR, "task_type_enrichment_patch.csv")
L3_VOCAB_CSV = os.path.join(_L3_DIR, "task_type_vocab_patch.csv")


def load_model_task_vocab(vocab_csv: str = VOCAB_CSV) -> dict:
    df = pd.read_csv(vocab_csv)
    vocab = dict(zip(df["task"], df["task_id"].astype(int)))
    assert vocab.get("Other") == 0, "task_vocab.csv must pin Other -> 0"
    return vocab


def attach_model_task_ids(data, umi, *, ids_csv: str = IDS_CSV,
                          vocab_csv: str = VOCAB_CSV) -> dict:
    ids = pd.read_csv(ids_csv)
    vocab = load_model_task_vocab(vocab_csv)

    merged = umi.merge(ids, left_on="model", right_on="unique_model_id",
                       how="left", validate="one_to_one")
    if merged["task_id"].isna().any():
        missing = merged.loc[merged["task_id"].isna(), "model"].head(5).tolist()
        raise ValueError(
            f"{int(merged['task_id'].isna().sum())} graph models missing from "
            f"{os.path.basename(ids_csv)} (e.g. {missing}) — rebuild the task ids "
            "for THIS graph with d1_model_task_vocab.py --graph <graph.pt>")
    if not (merged["mappedID_x"].values == merged["mappedID_y"].values).all():
        raise ValueError("mappedID mismatch between graph and model_task_ids.csv — "
                         "the CSV was built from a different graph")
    order = merged.sort_values("mappedID_x")
    tid = torch.tensor(order["task_id"].astype(int).values, dtype=torch.long)
    assert len(tid) == data["model"].num_nodes
    assert int(tid.max()) < len(vocab) and int(tid.min()) >= 0
    data["model"].task_id = tid
    return vocab


def apply_dataset_task_repair(data, xd0_meta, *, patch_csv: str = L3_PATCH_CSV,
                              vocab_csv: str = L3_VOCAB_CSV):
    patch = pd.read_csv(patch_csv)
    vpatch = pd.read_csv(vocab_csv)
    vocab = dict(zip(vpatch["task_type"], vpatch["task_type_id"].astype(int)))
    assert vocab.get("Other") == 0 and set(vocab.values()) == set(range(len(vocab))), \
        "task_type_vocab_patch.csv must be a contiguous bijection with Other -> 0"
    old_vocab = xd0_meta.get("task_type_vocab", {})
    for t, i in old_vocab.items():
        assert vocab.get(t) == i, f"vocab patch must preserve existing row {t}->{i}"

    apply_rows = patch[patch["action"] == "replace_Other_after_review"]
    tt = data["dataset"].task_type_id
    n_applied = 0
    for _, r in apply_rows.iterrows():
        node = int(r["mapped_id"])
        new_id = int(r["proposed_task_type_id"])
        assert 0 <= node < data["dataset"].num_nodes
        assert new_id in vocab.values()
        assert int(tt[node]) == 0, (
            f"patch expects node {node} ({r['dataset']}) to be Other, found "
            f"{int(tt[node])} — patch was built against a different graph")
        tt[node] = new_id
        n_applied += 1
    new_meta = dict(xd0_meta, num_task_types=len(vocab), task_type_vocab=vocab)
    stats = {"applied": n_applied, "vocab_rows": len(vocab),
             "still_other": int((tt == 0).sum())}
    return new_meta, stats
