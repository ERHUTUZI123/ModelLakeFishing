"""Synthetic unit tests for the Stage-3 gold-rank replay harness.

Run:
  python -m unittest ModelLakeFishing.stage3HNSW.tests.test_gold_rank_replay
"""

import json
import os
import tempfile
import unittest

import numpy as np

from ModelLakeFishing.stage3HNSW.gold_rank_replay import (
    FULL_GRAPH_CAVEAT,
    build_eligible_labels,
    csls_scores,
    evaluate_score_matrix,
    replay_arrays,
    write_report,
)


class GoldRankReplayTests(unittest.TestCase):
    def test_label_dedup_eligibility_and_all_argmax_ties(self):
        labels, stats = build_eligible_labels(
            # d0 has a duplicate and two tied gold models; d1 is too small;
            # d2 has three labels but constant accuracy.
            [0, 0, 1, 2, 0, 1, 0, 1, 2],
            [0, 0, 0, 0, 1, 1, 2, 2, 2],
            [0.2, 0.8, 0.9, 0.9, 0.1, 0.2, 0.5, 0.5, 0.5],
            n_models=3,
            n_datasets=4,
        )
        self.assertEqual(list(labels), [0])
        np.testing.assert_array_equal(labels[0]["model_ids"], [0, 1, 2])
        np.testing.assert_allclose(labels[0]["accuracies"], [0.8, 0.9, 0.9])
        self.assertEqual(stats["raw_trained_on_rows"], 9)
        self.assertEqual(stats["distinct_model_dataset_pairs"], 8)
        self.assertEqual(stats["eligible_datasets"], 1)
        self.assertEqual(stats["excluded_too_few_labelled_models"], 1)
        self.assertEqual(stats["excluded_constant_accuracy"], 1)
        self.assertEqual(stats["datasets_without_labels"], 1)

        result = evaluate_score_matrix(
            np.asarray([[0.9, 0.8, 0.7]]),
            labels,
            ["m0", "m1", "m2"],
            ["d0"],
            gold_ks=(1, 2, 3),
            hub_top_k=2,
        )
        row = result["per_dataset"][0]
        self.assertEqual(row["canonical_gold_model"]["mappedID"], 1)
        self.assertEqual(row["gold_rank"], 2)
        self.assertEqual(
            [gold["mappedID"] for gold in row["all_argmax_gold_models"]], [1, 2]
        )
        self.assertEqual(
            [gold["rank"] for gold in row["all_argmax_gold_models"]], [2, 3]
        )

    def test_csls_matches_definition_and_clips_small_population_k(self):
        cosine = np.asarray([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]])
        # k=1: query radii [2,3], model radii [1,2,3].
        expected = 2.0 * cosine - np.asarray([2.0, 3.0])[:, None] \
            - np.asarray([1.0, 2.0, 3.0])[None, :]
        np.testing.assert_allclose(csls_scores(cosine, k=1), expected)

        # Requested k exceeds both sides: each radius is the population mean.
        expected_all = 2.0 * cosine - cosine.mean(axis=1)[:, None] \
            - cosine.mean(axis=0)[None, :]
        np.testing.assert_allclose(csls_scores(cosine, k=10), expected_all)

    def test_gold_curve_observed_hit_and_hub_histogram(self):
        scores = np.asarray([
            [0.90, 0.80, 0.70, 0.60, 0.50],
            [0.90, 0.10, 0.20, 0.30, 0.40],
            [0.95, 0.85, 0.75, 0.65, 0.55],
        ])
        labels = {
            0: {"model_ids": np.asarray([0, 1, 2]),
                "accuracies": np.asarray([0.1, 0.9, 0.9])},
            1: {"model_ids": np.asarray([0, 3, 4]),
                "accuracies": np.asarray([0.9, 0.2, 0.1])},
            2: {"model_ids": np.asarray([1, 2, 4]),
                "accuracies": np.asarray([0.2, 0.3, 0.9])},
        }
        result = evaluate_score_matrix(
            scores,
            labels,
            [f"m{i}" for i in range(5)],
            [f"d{i}" for i in range(3)],
            gold_ks=(1, 2, 5),
            hub_top_k=2,
        )
        aggregate = result["aggregate"]
        self.assertEqual([result["per_dataset"][i]["gold_rank"] for i in range(3)], [2, 1, 5])
        self.assertAlmostEqual(aggregate["gold_at_k"]["1"], 1 / 3)
        self.assertAlmostEqual(aggregate["gold_at_k"]["2"], 2 / 3)
        self.assertEqual(aggregate["gold_at_k"]["5"], 1.0)
        self.assertEqual(aggregate["median_gold_rank"], 2.0)
        self.assertAlmostEqual(aggregate["observed_hit_at_1"], 1 / 3)

        hub = result["hub_occupancy"]
        self.assertEqual(hub["n_queries"], 3)
        self.assertEqual(hub["n_slots"], 6)
        self.assertEqual(hub["distinct_models"], 3)
        self.assertEqual(hub["max_occupancy"], 3)
        self.assertEqual(hub["occupied_models"][0]["mappedID"], 0)
        self.assertEqual(hub["occupied_models"][0]["occupancy"], 3)
        exact = {row["occupancy"]: row["n_models"] for row in hub["exact_histogram"]}
        self.assertEqual(exact, {0: 2, 1: 1, 2: 1, 3: 1})

    def test_raw_and_csls_write_json_and_markdown(self):
        score_rows = np.asarray([
            [0.90, 0.80, 0.70, 0.60],
            [0.90, 0.20, 0.30, 0.40],
        ])
        # Identity model vectors make the normalized score rows directly
        # observable as cosine up to a positive row-wise scale.
        z_m = np.eye(4)
        z_d = score_rows / np.linalg.norm(score_rows, axis=1, keepdims=True)
        labels = {
            0: {"model_ids": np.asarray([0, 1, 2]),
                "accuracies": np.asarray([0.1, 0.9, 0.2])},
            1: {"model_ids": np.asarray([0, 2, 3]),
                "accuracies": np.asarray([0.9, 0.2, 0.1])},
        }
        variants, per_dataset = replay_arrays(
            z_m, z_d, labels,
            ["m0", "m1", "m2", "m3"], ["d0", "d1"],
            gold_ks=(1, 2, 4), csls_k=1, hub_top_k=2,
        )
        self.assertEqual(set(variants), {"raw_cosine", "csls_k1"})
        self.assertEqual(len(per_dataset), 2)

        report = {
            "schema_version": "test",
            "caveat": FULL_GRAPH_CAVEAT,
            "export": {
                "name": "synthetic", "path": "synthetic", "manifest_sha256": "a",
                "graph_sha256": "b", "checkpoint_sha256": "c",
                "n_models": 4, "n_datasets": 2,
            },
            "protocol": {
                "gold_ks": [1, 2, 4], "eligibility": ">=3, non-constant",
                "gold_policy": "stable canonical tie",
            },
            "label_population": {"eligible_datasets": 2},
            "variants": variants,
            "per_dataset": per_dataset,
        }
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "replay.json")
            md_path = os.path.join(tmp, "replay.md")
            write_report(report, json_path, md_path)
            with open(json_path, encoding="utf-8") as handle:
                loaded = json.load(handle)
            with open(md_path, encoding="utf-8") as handle:
                markdown = handle.read()
        self.assertEqual(loaded["caveat"], FULL_GRAPH_CAVEAT)
        self.assertIn("Full-graph diagnostic only", markdown)
        self.assertIn("Gold-survival curve and hub summary", markdown)
        self.assertIn("Hub-occupancy histogram", markdown)
        self.assertIn("`raw_cosine`", markdown)
        self.assertIn("`csls_k1`", markdown)


if __name__ == "__main__":
    unittest.main()
