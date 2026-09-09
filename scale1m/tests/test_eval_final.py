import hashlib

import numpy as np

from scale1m.eval_final import _metrics, _recommendation_digest


def test_recommendation_digest_uses_logical_integer_content():
    matrix = np.array([[7, 3, 2], [8, 1, 0]], dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(np.asarray([7, 8], dtype="<i8").tobytes())
    digest.update(np.asarray([[3, 2], [1, 0]], dtype="<i8").tobytes())
    assert _recommendation_digest(matrix) == digest.hexdigest()


def test_final_metrics_measure_only_returned_models():
    recommendations = np.array([[4, 1, 3], [0, 2, 1]], dtype=np.int64)
    queries = [0, 1]
    candidates = {
        0: (np.array([4, 3, 2]), np.array([0.9, 0.89, 0.2])),
        1: (np.array([3, 2, 1]), np.array([0.9, 0.8, 0.7])),
    }
    roots = np.array(["a", "b"])
    result = _metrics(recommendations, queries, candidates, roots)
    assert result["gold@1"] == 0.5
    assert result["gold@10"] == 0.5
    assert result["top3@10"] == 1.0
    assert result["gold-gap@10"] == 0.5
