"""
Mechanism tests for v4 W2 (IPW positives, alibi hard negatives).

Run: ../.venv/Scripts/python.exe -m pytest tests/test_w2_phase.py -q
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ModelLakeFishing.stage2TrainGraphSAGE.losses import (  # noqa: E402
    build_lake_logq, global_lake_loss, mine_alibi_hard_negative_sets,
    build_model_task_profiles,
)


def _z(nm, nd, dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"model": torch.nn.functional.normalize(torch.randn(nm, dim, generator=g), dim=-1),
            "dataset": torch.nn.functional.normalize(torch.randn(nd, dim, generator=g), dim=-1)}


def test_l4_ipw_none_is_bit_identical_and_weights_change_loss():
    z = _z(40, 3)
    M = torch.zeros(40, 3)
    M[0, 0] = 1; M[5, 0] = 1; M[2, 1] = 1; M[7, 1] = 1
    ti = torch.tensor([[0, 0, 0, 0, 5, 2, 7], [0, 1, 2, 0, 0, 1, 1]])  # deg0=4, others low
    q, logq = build_lake_logq(ti, 40)
    kw = dict(n_neg=16, n_datasets=3)
    l_none = global_lake_loss(z, M, q, logq, generator=torch.Generator().manual_seed(1), **kw)
    l_unif = global_lake_loss(z, M, q, logq, pos_ipw=torch.ones(40),
                              generator=torch.Generator().manual_seed(1), **kw)
    assert torch.allclose(l_none, l_unif, atol=1e-7)   # uniform IPW == no IPW
    w = (torch.bincount(ti[0], minlength=40).float() + 1) ** -0.5
    l_ipw = global_lake_loss(z, M, q, logq, pos_ipw=w,
                             generator=torch.Generator().manual_seed(1), **kw)
    assert not torch.allclose(l_none, l_ipw)           # non-trivial reweighting


def test_l4_ipw_downweights_hub_positive():
    # dataset 0 has two positives: hub (deg 10) scoring LOW, tail (deg 0) HIGH.
    # IPW shifts the loss toward the tail term -> loss decreases vs uniform.
    z = _z(30, 1, seed=3)
    M = torch.zeros(30, 1); M[0, 0] = 1; M[1, 0] = 1
    z["model"][1] = z["dataset"][0]                     # tail positive aligned
    z["model"][0] = -z["dataset"][0]                    # hub positive anti-aligned
    ti = torch.tensor([[0] * 10, list(range(10))])      # model 0 is the hub
    q, logq = build_lake_logq(ti, 30)
    w = (torch.bincount(ti[0], minlength=30).float() + 1) ** -0.5
    kw = dict(n_neg=8, n_datasets=1)
    l_unif = global_lake_loss(z, M, q, logq, generator=torch.Generator().manual_seed(2), **kw)
    l_ipw = global_lake_loss(z, M, q, logq, pos_ipw=w,
                             generator=torch.Generator().manual_seed(2), **kw)
    assert float(l_ipw) < float(l_unif)                 # hub's bad term counts less


def test_l2b_alibi_filter_logic():
    torch.manual_seed(4)
    nm, nd = 40, 2
    z = _z(nm, nd, seed=4)
    d0 = 0
    M = torch.zeros(nm, nd); M[30, d0] = 1
    # craft scores: models 0..3 top the ranking for dataset 0
    z["model"][0] = z["dataset"][d0]                    # hub (deg high)      -> kept
    z["model"][1] = z["dataset"][d0] * 0.99             # tail, task-mismatch -> kept
    z["model"][2] = z["dataset"][d0] * 0.98             # tail, task-match    -> DROPPED (hidden gold)
    z["model"][3] = z["dataset"][d0] * 0.97             # tail, no profile    -> DROPPED (unknown ≠ alibi)
    deg = torch.zeros(nm, dtype=torch.long)
    deg[0] = 50; deg[1] = 1; deg[2] = 1; deg[3] = 0
    deg[10:20] = torch.arange(1, 11)   # spread labeled mass: P90 ≈ 9.8 >> 1
    dataset_task_id = torch.tensor([5, 0])
    model_tasks = {0: {5}, 1: {7}, 2: {5}}              # 1 mismatches task 5; 2 matches
    hard = mine_alibi_hard_negative_sets(
        z, M, deg, dataset_task_id, model_tasks, hard_k=4, deg_quantile=0.9)
    kept = set(hard[d0].tolist())
    assert 0 in kept and 1 in kept
    assert 2 not in kept and 3 not in kept
    # positives never mined
    assert 30 not in kept


def test_l2b_unknown_dataset_task_uses_degree_only():
    z = _z(20, 2, seed=5)
    d1 = 1                                              # task unknown (0)
    M = torch.zeros(20, 2); M[19, d1] = 1
    z["model"][0] = z["dataset"][d1]
    z["model"][1] = z["dataset"][d1] * 0.99
    deg = torch.zeros(20, dtype=torch.long); deg[0] = 9; deg[5:15] = 1
    hard = mine_alibi_hard_negative_sets(
        z, M, deg, torch.tensor([3, 0]), {0: {3}, 1: {3}}, hard_k=3)
    kept = set(hard[d1].tolist())
    assert 0 in kept and 1 not in kept   # mismatch impossible when task unknown


def test_build_model_task_profiles():
    ti = torch.tensor([[0, 0, 1, 2], [0, 1, 1, 2]])
    tt = torch.tensor([4, 0, 7, 0])                     # datasets 1,3 unknown task... wait
    prof = build_model_task_profiles(ti, tt)
    # model 0: datasets 0(task 4), 1(task 0 -> skipped) => {4}
    # model 1: dataset 1(task 0) => absent; model 2: dataset 2(task 7)? tt[2]=7
    assert prof[0] == {4}
    assert 1 not in prof
    assert prof[2] == {7}
