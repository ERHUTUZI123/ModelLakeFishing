"""
Mechanism tests for the v3 Z-track (frozen-view projection, push-apart loss).

Run: ../.venv/Scripts/python.exe -m pytest tests/test_z_phase.py -q
"""

import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage1BuildTransferGraph.dataset_embed.model_node_encoder import (  # noqa: E402
    DatasetNodeEncoder,
)
from ModelLakeFishing.stage2TrainGraphSAGE.losses import dataset_push_apart_loss  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE import learnable as L  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.model import HeteroGraphSAGE, load_hgraph  # noqa: E402

GRAPH = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage1BuildTransferGraph",
                     "hgraph_hf1000d_2000m_xm0_xd0.pt")


def _ds_inputs(batch=6, frozen=100):
    x = torch.randn(batch, frozen)
    return x, torch.randint(0, 10, (batch,)), torch.randint(0, 5, (batch,)), \
        torch.randint(0, 3, (batch,))


def test_z1_projection_dims_and_learnable():
    enc = DatasetNodeEncoder(100, 10, 5, 3)                       # legacy
    assert enc.out_dim == 100 + 16 + 8 + 4 and enc.frozen_proj is None
    enc = DatasetNodeEncoder(100, 10, 5, 3, frozen_proj_dim=32)   # Z1
    assert enc.out_dim == 32 + 16 + 8 + 4
    # LEARNABLE (unlike F4's frozen buffer): projection sits in parameters
    assert any("frozen_proj" in n for n, _ in enc.named_parameters())
    x, t, c, a = _ds_inputs()
    out = enc(x, t, c, a)
    assert out.shape == (6, 60)
    out.sum().backward()
    assert enc.frozen_proj.weight.grad.abs().sum() > 0            # grad flows


def test_z2_push_apart_pair_selection():
    # 6 datasets: tasks [0,1,1,2,2,0] — Other (0) must never participate
    tt = torch.tensor([0, 1, 1, 2, 2, 0])
    z = torch.nn.functional.normalize(torch.ones(6, 4), dim=-1)   # all identical
    g = torch.Generator().manual_seed(0)
    loss, st = dataset_push_apart_loss(z, tt, margin=0.2, generator=g,
                                       return_stats=True)
    # identical vectors above margin -> positive loss; pairs only across 1<->2
    assert float(loss) > 0 and st["n_anchors"] == 4
    # orthogonal different-task vectors -> zero loss (below margin)
    z2 = torch.eye(6, 6)
    loss2 = dataset_push_apart_loss(z2, tt, margin=0.2,
                                    generator=torch.Generator().manual_seed(0))
    assert float(loss2) == 0.0


def test_z2_all_other_is_noop_and_grad_flows():
    z = torch.nn.functional.normalize(torch.randn(5, 4), dim=-1)
    loss = dataset_push_apart_loss(z, torch.zeros(5, dtype=torch.long))
    assert float(loss) == 0.0                                     # nothing known
    tt = torch.tensor([1, 2, 1, 2])
    z3 = torch.nn.functional.normalize(torch.ones(4, 4), dim=-1).requires_grad_(True)
    loss3 = dataset_push_apart_loss(z3, tt, margin=0.0)
    loss3.backward()
    assert z3.grad.abs().sum() > 0


def test_z2_same_task_never_repelled():
    # two same-task datasets, identical vectors: no different-task pair exists
    tt = torch.tensor([3, 3])
    z = torch.nn.functional.normalize(torch.ones(2, 4), dim=-1)
    loss = dataset_push_apart_loss(z, tt)
    assert float(loss) == 0.0


@pytest.mark.skipif(not os.path.exists(GRAPH), reason="shipped graph not present")
def test_z1_full_model_and_checkpoint_roundtrip(tmp_path):
    data, xm0, _umi = load_hgraph(GRAPH)
    payload = torch.load(GRAPH, map_location="cpu", weights_only=False)
    xd0 = payload["xd0_meta"]
    model = HeteroGraphSAGE(
        metadata=data.metadata(), frozen_dim=data["model"].x.shape[1],
        num_size_buckets=xm0["num_size_buckets"], num_families=xm0["num_families"],
        dataset_in_dim=data["dataset"].x.shape[1], num_layers=1,
        num_task_types=xd0["num_task_types"], n_class_buckets=xd0["n_class_buckets"],
        num_arities=xd0["num_arities"], dataset_frozen_proj_dim=128)
    assert model.dataset_encoder.out_dim == 128 + 16 + 8 + 4
    z = model(data)
    assert z["dataset"].shape == (362, 128)
    p = os.path.join(tmp_path, "z1.pt")
    L.save_checkpoint(model, xm0["family_vocab"], p)
    m2, _v, _r = L.load_checkpoint(p)
    assert m2.dataset_encoder.frozen_proj_dim == 128
    z2 = m2(data)
    assert torch.allclose(z["dataset"], z2["dataset"], atol=1e-6)
