"""
d0_splits.py -- v4 W1-D2: ROOT-AWARE fixed splits for the D0 graph.

Why node-level splitting is no longer acceptable (plan v4 §0.2-4): the D0 lake
carries per-language / per-config sibling nodes of one root dataset (tatoeba
112 nodes, amazon_massive 102, ...). Siblings are near-duplicates; putting one
in train and another in test leaks the answer through the message graph. Here
the SPLIT UNIT IS THE ROOT: every trained_on edge whose dataset belongs to a
root lands on exactly one side.

Semantics mirror split_trained_on / CustomRandomLinkSplit downstream contract:
  train_data : message = train edges minus the disjoint supervision part;
               edge_label_index/edge_label = disjoint part + sampled negatives
  val_data   : message = ALL train edges; labels = val edges + negatives
  test_data  : message = train + val edges; labels = test edges + negatives
  (binary edge_label: 1 = real edge, 0 = negative; accuracy comes from
  `accuracy_lookup` downstream, same as always. rev_trained_on mirrors the
  message edges of each split -- held-out edges have no reverse copy.)
"""

import torch

from ModelLakeFishing.stage2TrainGraphSAGE.losses import TRAINED_ON, REV_TRAINED_ON


def _neg_sample(pos_set, num_models, num_datasets, n, gen):
    """n random (model, dataset) pairs not in pos_set."""
    out_m, out_d = [], []
    while len(out_m) < n:
        m = torch.randint(num_models, (n,), generator=gen)
        d = torch.randint(num_datasets, (n,), generator=gen)
        for mi, di in zip(m.tolist(), d.tolist()):
            if (mi, di) not in pos_set and len(out_m) < n:
                out_m.append(mi); out_d.append(di)
    return torch.tensor([out_m, out_d], dtype=torch.long)


def make_root_aware_splits(data, dataset_roots, *, split_seed, num_val=0.1,
                           num_test=0.2, neg_ratio=1.0, disjoint_train_ratio=0.3):
    """
    dataset_roots : list[str] of length N_datasets (mappedID order), the root
                    of every dataset node (unique_dataset_id['root']).
    Returns (train_data, val_data, test_data).
    """
    gen = torch.Generator().manual_seed(split_seed)
    ei = data[TRAINED_ON].edge_index
    attr = data[TRAINED_ON].edge_attr
    E = ei.shape[1]

    # assign ROOTS (not edges) to sides, greedily filling test then val quota
    edge_root = [dataset_roots[int(d)] for d in ei[1]]
    uniq = sorted(set(edge_root))
    perm = torch.randperm(len(uniq), generator=gen).tolist()
    root_edges = {}
    for r in edge_root:
        root_edges[r] = root_edges.get(r, 0) + 1
    side = {}
    acc_test = acc_val = 0
    for i in perm:
        r = uniq[i]
        if acc_test < num_test * E:
            side[r] = "test"; acc_test += root_edges[r]
        elif acc_val < num_val * E:
            side[r] = "val"; acc_val += root_edges[r]
        else:
            side[r] = "train"
    mask = {s: torch.tensor([side[r] == s for r in edge_root]) for s in
            ("train", "val", "test")}

    # disjoint part of TRAIN: supervision-only edges removed from the message graph
    tr_idx = mask["train"].nonzero().flatten()
    tr_perm = tr_idx[torch.randperm(tr_idx.numel(), generator=gen)]
    n_disjoint = int(round(disjoint_train_ratio * tr_idx.numel()))
    sup_idx, msg_idx = tr_perm[:n_disjoint], tr_perm[n_disjoint:]

    pos_set = {(int(m), int(d)) for m, d in zip(ei[0].tolist(), ei[1].tolist())}
    nm, nd = data["model"].num_nodes, data["dataset"].num_nodes

    def _mk(message_idx, label_idx):
        out = data.clone()
        me = ei[:, message_idx]
        out[TRAINED_ON].edge_index = me
        out[TRAINED_ON].edge_attr = attr[message_idx]
        out[REV_TRAINED_ON].edge_index = me.flip(0)
        out[REV_TRAINED_ON].edge_attr = attr[message_idx].clone()
        pos = ei[:, label_idx]
        neg = _neg_sample(pos_set, nm, nd, int(round(neg_ratio * pos.shape[1])), gen)
        out[TRAINED_ON].edge_label_index = torch.cat([pos, neg], dim=1)
        out[TRAINED_ON].edge_label = torch.cat(
            [torch.ones(pos.shape[1]), torch.zeros(neg.shape[1])])
        return out

    val_idx = mask["val"].nonzero().flatten()
    test_idx = mask["test"].nonzero().flatten()
    train_data = _mk(msg_idx, sup_idx)
    val_data = _mk(tr_idx, val_idx)
    test_data = _mk(torch.cat([tr_idx, val_idx]), test_idx)

    # leakage guards: no test/val edge (or its reverse) in any message graph it
    # could leak through; every root on exactly one side
    for split, held in ((train_data, torch.cat([val_idx, test_idx])),
                        (val_data, test_idx)):
        held_set = {(int(m), int(d)) for m, d in zip(*ei[:, held].tolist())}
        msg = split[TRAINED_ON].edge_index
        assert not any((int(m), int(d)) in held_set for m, d in zip(*msg.tolist()))
    return train_data, val_data, test_data
