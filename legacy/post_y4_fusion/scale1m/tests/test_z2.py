import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_PARENT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_PARENT not in sys.path:
    sys.path.insert(0, _REPO_PARENT)

from ModelLakeFishing.scale1m import eval_z2 as Z2


def test_g0_mix_is_rank_equivalent_to_dense_plus_eight_prior():
    dense_cos = np.asarray([[0.2, -0.4, 0.8]], dtype=np.float32)
    prior = np.asarray([[0.1, 0.9, 0.4]], dtype=np.float32)
    mixed = Z2._mix(dense_cos, prior, Z2.G0)[0]
    dense = (dense_cos[0].astype(float) + 1.0) * 0.5
    fixed = dense + 8.0 * prior[0]
    assert np.array_equal(np.argsort(-mixed), np.argsort(-fixed))


def test_history_counts_only_use_supplied_edges():
    history = Z2.HistoryStats(
        edge_model=np.asarray([1, 1, 3, 4]),
        edge_dataset=np.asarray([0, 1, 1, 2]),
        task_id=np.asarray([7, 7, 9, 7]),
        root_id=np.asarray([10, 11, 12, 13]),
    )
    task, counts = history.counts(3, np.asarray([0, 1, 3, 4]))
    assert task == 7
    assert counts.tolist() == [0, 2, 1, 0]
    assert history.n_models[7] == 2
    assert history.n_roots[7] == 2
    assert history.visible_root_ids == {10, 11, 12}


def test_gate_is_monotonic_in_oriented_features():
    median = np.zeros(len(Z2.FEATURE_NAMES))
    iqr = np.ones(len(Z2.FEATURE_NAMES))
    params = np.r_[0.0, np.ones(len(Z2.FEATURE_NAMES))]
    low = Z2._predict(np.zeros((1, len(Z2.FEATURE_NAMES))), median, iqr, params)
    high = Z2._predict(np.ones((1, len(Z2.FEATURE_NAMES))), median, iqr, params)
    assert high[0] > low[0]


def test_objective_gradient_matches_finite_difference():
    rng = np.random.default_rng(4)
    features = rng.normal(size=(5, len(Z2.FEATURE_NAMES)))
    a = rng.normal(scale=0.1, size=(5, Z2.N_HARD))
    b = rng.normal(scale=0.1, size=(5, Z2.N_HARD))
    usable = np.asarray([True, True, False, True, True])
    params = np.r_[0.03, np.full(len(Z2.FEATURE_NAMES), 0.05)]
    value, gradient = Z2._objective(params, features, a, b, usable)
    assert np.isfinite(value)
    eps = 1e-6
    numeric = np.empty_like(params)
    for i in range(len(params)):
        step = np.zeros_like(params)
        step[i] = eps
        hi = Z2._objective(params + step, features, a, b, usable)[0]
        lo = Z2._objective(params - step, features, a, b, usable)[0]
        numeric[i] = (hi - lo) / (2.0 * eps)
    assert np.allclose(gradient, numeric, rtol=2e-5, atol=2e-6)


def test_fit_gate_enforces_nonnegative_feature_weights():
    rng = np.random.default_rng(8)
    features = rng.normal(size=(12, len(Z2.FEATURE_NAMES)))
    a = rng.normal(scale=0.05, size=(12, Z2.N_HARD))
    b = rng.normal(scale=0.05, size=(12, Z2.N_HARD))
    params, status = Z2._fit_gate(features, a, b, np.ones(12, dtype=bool))
    assert status["success"]
    assert np.all(params[1:] >= 0.0)
