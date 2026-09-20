import os
import sys

import pandas as pd
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage1BuildTransferGraph.dataset_embed.model_node_encoder import (
    ModelNodeEncoder,
)
from ModelLakeFishing.stage2TrainGraphSAGE import learnable as L
from ModelLakeFishing.stage2TrainGraphSAGE.model import HeteroGraphSAGE, load_hgraph
from ModelLakeFishing.stage2TrainGraphSAGE.d1_features import (
    attach_model_task_ids, load_model_task_vocab,
)

GRAPH = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage1BuildTransferGraph",
                     "hgraph_hf1000d_2000m_xm0_xd0.pt")

B, FROZEN, NAME = 7, 448, 64


def _inputs(batch=B):
    x = torch.randn(batch, FROZEN)
    sid = torch.randint(0, 15, (batch,))
    fid = torch.randint(0, 185, (batch,))
    tid = torch.randint(0, 9, (batch,))
    return x, sid, fid, tid


def test_legacy_default_is_unchanged():
    torch.manual_seed(0)
    enc_new = ModelNodeEncoder(FROZEN, 15, 185)
    assert enc_new.out_dim == FROZEN + 16 + 16
    x, sid, fid, _ = _inputs()
    out = enc_new(x, sid, fid)
    assert torch.equal(out[:, :FROZEN], x)


def test_variant_out_dims():
    e = ModelNodeEncoder(FROZEN, 15, 185, name_dim=NAME, use_desc=False)
    assert e.out_dim == 64 + 16 + 16
    e = ModelNodeEncoder(FROZEN, 15, 185, name_dim=NAME, use_desc=False, use_family=False)
    assert e.out_dim == 64 + 16 and e.family_embedding is None
    e = ModelNodeEncoder(FROZEN, 15, 185, name_dim=NAME, use_desc=False,
                         use_family=False, num_model_tasks=9)
    assert e.out_dim == 64 + 16 + 16
    e = ModelNodeEncoder(FROZEN, 15, 185, name_dim=NAME, use_desc=False,
                         use_family=False, num_model_tasks=9, name_proj_dim=16)
    assert e.out_dim == 16 + 16 + 16
    x, sid, fid, tid = _inputs()
    assert e(x, sid, fid, tid).shape == (B, 48)


def test_slicing_uses_name_half_only():
    e = ModelNodeEncoder(FROZEN, 15, 185, name_dim=NAME, use_desc=False, use_family=False)
    x, sid, fid, _ = _inputs()
    out1 = e(x, sid, fid)
    x2 = x.clone()
    x2[:, NAME:] = 999.0
    assert torch.equal(out1, e(x2, sid, fid))


def test_name_projection_frozen_deterministic_and_scaled():
    e1 = ModelNodeEncoder(FROZEN, 15, 185, name_dim=NAME, use_desc=False,
                          use_family=False, name_proj_dim=16, name_proj_seed=42)
    e2 = ModelNodeEncoder(FROZEN, 15, 185, name_dim=NAME, use_desc=False,
                          use_family=False, name_proj_dim=16, name_proj_seed=42)
    assert torch.equal(e1.name_proj, e2.name_proj)
    e3 = ModelNodeEncoder(FROZEN, 15, 185, name_dim=NAME, use_desc=False,
                          use_family=False, name_proj_dim=16, name_proj_seed=7)
    assert not torch.equal(e1.name_proj, e3.name_proj)
    assert all("name_proj" not in n for n, _ in e1.named_parameters())
    assert "name_proj" in dict(e1.named_buffers())
    x = torch.randn(4096, NAME)
    r = (x @ e1.name_proj).pow(2).sum(1).mean() / x.pow(2).sum(1).mean()
    assert 0.5 < float(r) < 1.6


def test_gradient_boundary():
    e = ModelNodeEncoder(FROZEN, 15, 185, name_dim=NAME, use_desc=False,
                         num_model_tasks=9, name_proj_dim=16)
    x, sid, fid, tid = _inputs()
    x.requires_grad_(True)
    out = e(x, sid, fid, tid)
    out.sum().backward()
    assert e.size_embedding.weight.grad.abs().sum() > 0
    assert e.family_embedding.weight.grad.abs().sum() > 0
    assert e.task_embedding.weight.grad.abs().sum() > 0
    assert getattr(e.name_proj, "grad", None) is None
    assert x.grad[:, NAME:].abs().sum() == 0
    assert x.grad[:, :NAME].abs().sum() > 0


def test_task_id_required_when_table_exists():
    e = ModelNodeEncoder(FROZEN, 15, 185, name_dim=NAME, num_model_tasks=9)
    x, sid, fid, _ = _inputs()
    with pytest.raises(ValueError, match="task_id"):
        e(x, sid, fid)


def test_name_dim_required_for_variants():
    with pytest.raises(ValueError, match="name_dim"):
        ModelNodeEncoder(FROZEN, 15, 185, use_desc=False)


@pytest.mark.skipif(not os.path.exists(GRAPH), reason="shipped graph not present")
def test_attach_and_full_model_roundtrip(tmp_path):
    data, xm0, umi = load_hgraph(GRAPH)
    vocab = attach_model_task_ids(data, umi)
    assert data["model"].task_id.shape[0] == data["model"].num_nodes
    assert vocab == load_model_task_vocab()

    model = HeteroGraphSAGE(
        metadata=data.metadata(), frozen_dim=data["model"].x.shape[1],
        num_size_buckets=xm0["num_size_buckets"], num_families=xm0["num_families"],
        dataset_in_dim=data["dataset"].x.shape[1], num_layers=1,
        name_dim=xm0["name_dim"], use_desc=False, use_family=False,
        name_proj_dim=16, num_model_tasks=len(vocab))
    z = model(data)
    assert z["model"].shape == (data["model"].num_nodes, 128)

    p = os.path.join(tmp_path, "f4.pt")
    L.save_checkpoint(model, xm0["family_vocab"], p, model_task_vocab=vocab)
    m2, _fv, repro = L.load_checkpoint(p)
    assert repro["model_task_vocab"] == vocab
    z2 = m2(data)
    assert torch.allclose(z["model"], z2["model"], atol=1e-6)
    assert os.path.exists(os.path.join(tmp_path, "f4.model_task_vocab.csv"))
    bad = dict(vocab); bad.pop("Other"); bad["weird"] = 0
    with pytest.raises(ValueError):
        L.save_checkpoint(model, xm0["family_vocab"], p, model_task_vocab=bad)
    with pytest.raises(ValueError, match="model_task_vocab"):
        L.save_checkpoint(model, xm0["family_vocab"], p)


@pytest.mark.skipif(not os.path.exists(GRAPH), reason="shipped graph not present")
def test_attach_rejects_foreign_graph_csv(tmp_path):
    data, xm0, umi = load_hgraph(GRAPH)
    ids = pd.read_csv(os.path.join(
        _REPO_ROOT, "ModelLakeFishing", "stage1BuildTransferGraph",
        "artifacts", "d1_features", "model_task_ids.csv"))
    ids.loc[0, "mappedID"] = 1999 if ids.loc[0, "mappedID"] != 1999 else 1998
    bad_csv = os.path.join(tmp_path, "bad_ids.csv")
    ids.to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="mappedID mismatch"):
        attach_model_task_ids(data, umi, ids_csv=bad_csv)


OLD_CKPT = os.path.join(os.path.dirname(_HERE), "artifacts", "ablation", "top1",
                        "ckpt", "G2_s0_i0.pt")


@pytest.mark.skipif(not os.path.exists(OLD_CKPT), reason="G2 checkpoint not present")
def test_pre_d1_checkpoint_still_loads():
    model, vocab, _repro = L.load_checkpoint(OLD_CKPT)
    enc = model.model_encoder
    assert enc.use_desc and enc.use_family and enc.task_embedding is None
    assert enc.name_proj is None and enc.out_dim == 448 + 16 + 16
