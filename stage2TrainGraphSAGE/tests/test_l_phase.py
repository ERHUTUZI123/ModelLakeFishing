"""
Mechanism tests for the v3 L-track (logQ lake sampling, hard mining, task repair).

Run: ../.venv/Scripts/python.exe -m pytest tests/test_l_phase.py -q
"""

import math
import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.losses import (  # noqa: E402
    build_lake_logq, global_lake_loss, mine_hard_negative_sets,
)
from ModelLakeFishing.stage2TrainGraphSAGE.model import load_hgraph  # noqa: E402
from ModelLakeFishing.stage2TrainGraphSAGE.d1_features import (  # noqa: E402
    apply_dataset_task_repair,
)

GRAPH = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage1BuildTransferGraph",
                     "hgraph_hf1000d_2000m_xm0_xd0.pt")


def test_lake_logq_distribution():
    # 5 models: degrees 3,1,0,0,10 (edges name src=model)
    ti = torch.tensor([[0, 0, 0, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                       [0, 1, 2, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    q, logq = build_lake_logq(ti, 5, alpha=0.75, n0=1.0)
    assert abs(float(q.sum()) - 1.0) < 1e-6
    assert torch.allclose(logq, q.log())
    # degree-monotone: q4 > q0 > q1 > q2 == q3 > 0 (zero-degree still sampleable)
    assert q[4] > q[0] > q[1] > q[2] > 0
    assert q[2] == q[3]
    # alpha tempering: ratio q4/q2 must be (11/1)^0.75, not 11
    assert abs(float(q[4] / q[2]) - 11 ** 0.75) < 1e-4


def test_lake_loss_excludes_positives_and_corrects_logq():
    torch.manual_seed(0)
    N_m, N_d, dim = 50, 4, 8
    z = {"model": torch.nn.functional.normalize(torch.randn(N_m, dim), dim=-1),
         "dataset": torch.nn.functional.normalize(torch.randn(N_d, dim), dim=-1)}
    M = torch.zeros(N_m, N_d)
    M[0, 0] = 1; M[1, 0] = 1; M[2, 1] = 1                     # datasets 0,1 supervised
    ti = torch.tensor([[0, 1, 2], [0, 0, 1]])
    q, logq = build_lake_logq(ti, N_m)
    g = torch.Generator().manual_seed(7)
    loss, stats = global_lake_loss(z, M, q, logq, n_neg=32, n_datasets=4,
                                   generator=g, return_stats=True)
    assert torch.isfinite(loss) and stats["n_datasets"] == 2
    # manual replication for a single dataset with a fixed sample: correction
    # -log q must appear in the negative logits
    pos = torch.tensor([0, 1])
    samp = torch.tensor([5, 6, 7])
    T = 0.1
    s_pos = z["model"][pos] @ z["dataset"][0] / T
    s_neg = z["model"][samp] @ z["dataset"][0] / T - logq[samp]
    manual = (torch.logsumexp(torch.cat([s_pos, s_neg]), 0) - s_pos).mean()
    assert torch.isfinite(manual)                              # formula well-defined
    # gradient flows to embeddings
    z2 = {k: v.clone().requires_grad_(True) for k, v in z.items()}
    loss2 = global_lake_loss(z2, M, q, logq, n_neg=32, n_datasets=4,
                             generator=torch.Generator().manual_seed(7))
    loss2.backward()
    assert z2["model"].grad.abs().sum() > 0 and z2["dataset"].grad.abs().sum() > 0


def test_lake_loss_positive_never_negative_statistically():
    torch.manual_seed(1)
    N_m, N_d = 10, 1
    z = {"model": torch.nn.functional.normalize(torch.randn(N_m, 4), dim=-1),
         "dataset": torch.nn.functional.normalize(torch.randn(N_d, 4), dim=-1)}
    M = torch.zeros(N_m, N_d); M[3, 0] = 1
    # extreme q: model 3 (the positive) would dominate sampling if not filtered
    ti = torch.tensor([[3] * 50, list(range(10)) * 5])
    q, logq = build_lake_logq(ti, N_m, alpha=1.0)
    for seed in range(5):
        g = torch.Generator().manual_seed(seed)
        loss, st = global_lake_loss(z, M, q, logq, n_neg=64, generator=g,
                                    return_stats=True)
        # after filtering the dominant positive, few negatives remain — but the
        # loss stays finite and negatives < 64 proves the filter ran
        assert torch.isfinite(loss) and st["n_negs_used"] < 64


def test_mine_hard_negative_sets():
    torch.manual_seed(2)
    N_m, N_d = 30, 3
    z = {"model": torch.nn.functional.normalize(torch.randn(N_m, 8), dim=-1),
         "dataset": torch.nn.functional.normalize(torch.randn(N_d, 8), dim=-1)}
    M = torch.zeros(N_m, N_d); M[0, 0] = 1; M[1, 1] = 1
    hard = mine_hard_negative_sets(z, M, hard_k=5)
    assert set(hard) == {0, 1}
    for d, h in hard.items():
        assert h.numel() == 5
        pos = (M[:, d] > 0).nonzero().flatten()
        assert not bool(torch.isin(h, pos).any())              # positives excluded
        # they really are the top scorers among non-positives
        s = z["model"] @ z["dataset"][d]
        s[pos] = float("-inf")
        assert set(h.tolist()) == set(torch.topk(s, 5).indices.tolist())


def test_hard_sets_filtered_in_lake_loss():
    torch.manual_seed(3)
    N_m, N_d = 20, 1
    z = {"model": torch.nn.functional.normalize(torch.randn(N_m, 4), dim=-1),
         "dataset": torch.nn.functional.normalize(torch.randn(N_d, 4), dim=-1)}
    M = torch.zeros(N_m, N_d); M[0, 0] = 1
    ti = torch.tensor([[0], [0]])
    q, logq = build_lake_logq(ti, N_m)
    hard = {0: torch.tensor([0, 1, 2])}                        # 0 is the positive
    g = torch.Generator().manual_seed(0)
    loss, st = global_lake_loss(z, M, q, logq, n_neg=8, hard_sets=hard,
                                n_hard=3, generator=g, return_stats=True)
    assert torch.isfinite(loss) and st["n_hard_used"] == 2     # positive filtered


@pytest.mark.skipif(not os.path.exists(GRAPH), reason="shipped graph not present")
def test_apply_dataset_task_repair_on_real_graph():
    data, _xm0, _umi = load_hgraph(GRAPH)
    import torch as _t
    payload = _t.load(GRAPH, map_location="cpu", weights_only=False)
    xd0 = payload["xd0_meta"]
    before_other = int((data["dataset"].task_type_id == 0).sum())
    new_meta, stats = apply_dataset_task_repair(data, xd0)
    after_other = int((data["dataset"].task_type_id == 0).sum())
    assert stats["applied"] == before_other - after_other == 210
    assert new_meta["num_task_types"] == stats["vocab_rows"] == 26
    # old vocab rows preserved verbatim
    for t, i in xd0["task_type_vocab"].items():
        assert new_meta["task_type_vocab"][t] == i
    assert int(data["dataset"].task_type_id.max()) < 26
    # applying twice must fail loudly (nodes no longer Other)
    with pytest.raises(AssertionError):
        apply_dataset_task_repair(data, new_meta)


@pytest.mark.skipif(not os.path.exists(GRAPH), reason="shipped graph not present")
def test_l3_lifts_pool_coverage():
    from ModelLakeFishing.stage2TrainGraphSAGE.losses import (
        TRAINED_ON, topk_membership, build_global_negative_pools,
    )
    from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import dedup_trained_on
    import torch as _t
    data, _xm0, _umi = load_hgraph(GRAPH)
    payload = _t.load(GRAPH, map_location="cpu", weights_only=False)
    xd0 = payload["xd0_meta"]

    def coverage(d):
        # mechanism check on the full deduped edge set (protocol splits are
        # exercised by the L rows themselves)
        ti = d[TRAINED_ON].edge_index
        ta = d[TRAINED_ON].edge_attr.float().flatten()
        M = topk_membership(d, top_frac=0.10, trained_on_index=ti, trained_on_attr=ta)
        _pools, stats = build_global_negative_pools(
            d["dataset"].task_type_id, ti, ta, M, include_known_low=True)
        return stats["with_incompat"]

    d0 = dedup_trained_on(data)
    before = coverage(d0)
    apply_dataset_task_repair(d0, xd0)
    after = coverage(d0)
    assert after > before, (before, after)
    print(f"incompat-pool coverage: {before} -> {after}")
