"""Offline unit tests for task_type_enrichment.

These tests use only temporary JSON/CSV fixtures.  They make no network calls
and do not load or mutate the shipped graph.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from ModelLakeFishing.stage1BuildTransferGraph.task_type_enrichment import (
    Evidence,
    OTHER,
    append_only_vocab,
    build_alias_lookup,
    collect_card_evidence,
    collect_model_index_evidence,
    compute_incompatibility_coverage,
    decide_evidence,
    identity_evidence,
    write_outputs,
)


class EvidenceDecisionTests(unittest.TestCase):
    def test_specific_model_index_task_wins_over_lower_card_category(self):
        rows = [
            Evidence("model_index_task", "bitextmining", "bitext-mining", 1, count=12),
            Evidence("dataset_card_task_category", "translation", "translation", 3),
        ]
        decision = decide_evidence(rows)
        self.assertEqual(decision["decision"], "propose")
        self.assertEqual(decision["proposed_task_type"], "bitext-mining")
        self.assertEqual(decision["winning_tier"], 1)

    def test_same_tier_semantic_conflict_is_not_patched(self):
        rows = [
            Evidence("dataset_card_task_id", "intent-classification", "intent", 2),
            Evidence("dataset_card_task_id", "sentiment-classification", "sentiment", 2),
        ]
        decision = decide_evidence(rows)
        self.assertEqual(decision["decision"], "conflict")
        self.assertEqual(decision["conflicts"], ["intent", "sentiment"])
        self.assertEqual(decision["proposed_task_type"], "")

    def test_generic_classification_is_insufficient(self):
        rows = [Evidence("model_index_task", "classification", None, 1, count=100)]
        self.assertEqual(decide_evidence(rows)["decision"], "insufficient_evidence")

    def test_amazon_massive_identity_is_unambiguous_intent(self):
        rows = identity_evidence(
            "amazon_massive_intent",
            ["mteb/amazon_massive_intent"],
            "MassiveIntentClassification - an MTEB dataset",
            ["text-classification", "classification"],
            artifact="cached-card.json",
        )
        self.assertEqual({row.canonical for row in rows}, {"intent"})
        decision = decide_evidence(rows)
        self.assertEqual(decision["decision"], "propose")
        self.assertEqual(decision["proposed_task_type"], "intent")


class CacheExtractionTests(unittest.TestCase):
    def test_cached_card_and_model_index_are_joined_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_cache = root / "datasets"
            model_cache = root / "models"
            dataset_cache.mkdir()
            model_cache.mkdir()
            (dataset_cache / "mteb__amazon_massive_intent.json").write_text(
                json.dumps({
                    "id": "mteb/amazon_massive_intent",
                    "tags": ["task_categories:text-classification"],
                    "description": "MassiveIntentClassification benchmark",
                    "cardData": {
                        "task_categories": ["text-classification"],
                        "task_ids": [],
                    },
                }), encoding="utf-8")
            (model_cache / "model.json").write_text(json.dumps({
                "model-index": [{
                    "name": "fixture",
                    "results": [{
                        "task": {"type": "classification"},
                        "dataset": {"type": "mteb/amazon_massive_intent"},
                        "metrics": [{"type": "accuracy", "value": 0.9}],
                    }],
                }],
            }), encoding="utf-8")
            inventory = {
                "amazon_massive_intent": [{
                    "canon_key": "amazon_massive_intent",
                    "hf_id": "mteb/amazon_massive_intent",
                    "task_ids": "[]",
                    "task_categories": "['text-classification']",
                }]
            }
            aliases, ambiguous = build_alias_lookup(
                ["amazon_massive_intent"], inventory)
            self.assertFalse(ambiguous)
            model_ev, model_stats = collect_model_index_evidence(model_cache, aliases)
            card_ev, card_stats = collect_card_evidence(
                ["amazon_massive_intent"], inventory, dataset_cache)
            rows = model_ev["amazon_massive_intent"] + card_ev["amazon_massive_intent"]
            decision = decide_evidence(rows)
            self.assertEqual(decision["proposed_task_type"], "intent")
            self.assertEqual(model_stats["results_matched"], 1)
            self.assertEqual(card_stats["cards_found"], 1)


class PoolCoverageTests(unittest.TestCase):
    def test_enrichment_increases_only_reliable_nonempty_pools(self):
        # m0 is observed on d0, m1 on d1, m2 on unknown d2.  With d0/d1
        # unknown there are no incompatible pools.  Assigning distinct tasks
        # makes each known dataset see the other observed models as candidates;
        # d2 remains excluded exactly like Stage-2's production builder.
        edge_index = [[0, 1, 2], [0, 1, 2]]
        before = compute_incompatibility_coverage(
            [OTHER, OTHER, OTHER], edge_index)
        after = compute_incompatibility_coverage(
            ["intent", "sentiment", OTHER], edge_index)
        self.assertEqual(before["nonempty_incompatibility_pools"], 0)
        self.assertEqual(after["nonempty_incompatibility_pools"], 2)
        self.assertEqual(after["known_task_datasets"], 2)
        self.assertNotIn("2", after["pool_sizes_by_dataset_id"])

    def test_vocab_is_append_only_and_deterministic(self):
        current = {OTHER: 0, "sentiment": 1}
        updated = append_only_vocab(current, ["intent", "domain", "sentiment"])
        self.assertEqual(current, {OTHER: 0, "sentiment": 1})
        self.assertEqual(updated, {
            OTHER: 0, "sentiment": 1, "domain": 2, "intent": 3,
        })


class OutputContractTests(unittest.TestCase):
    def test_outputs_are_review_artifacts_not_source_mutations(self):
        review = [{
            "dataset": "amazon_massive_intent",
            "mapped_id": 6,
            "supervised_full_graph": 1,
            "current_task_type": OTHER,
            "decision": "propose",
            "proposed_task_type": "intent",
            "winning_tier": 2,
            "winning_sources": ["dataset_identity"],
            "conflicts": [],
            "evidence_json": "[]",
        }]
        report = {
            "inputs": {"graph": "fixture.pt"},
            "integrity": {"source_graph_unchanged": True},
            "summary": {
                "dataset_nodes": 1, "current_other": 1,
                "proposed_total": 1, "proposed_supervised": 1,
                "conflicts_total": 0, "insufficient_total": 0,
            },
            "pool_coverage": {
                "full_graph": {
                    "before": {"supervised_datasets": 1,
                               "nonempty_incompatibility_pools": 0},
                    "after": {"supervised_datasets": 1,
                              "nonempty_incompatibility_pools": 1},
                },
                "fixed_splits": {},
            },
            "target": {"nonempty_pools": 1},
            "amazon_massive_intent": {
                "decision": "propose", "proposed_task_type": "intent",
                "evidence_summary": "dataset_identity -> intent",
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            paths = write_outputs(out, review, {OTHER: 0}, report)
            self.assertEqual(set(paths), {
                "patch_csv", "vocab_patch_csv", "review_csv",
                "report_json", "report_md",
            })
            with (out / "task_type_enrichment_patch.csv").open(
                    encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["action"], "replace_Other_after_review")
            self.assertEqual(row["proposed_task_type"], "intent")
            self.assertEqual(row["proposed_task_type_id"], "1")


if __name__ == "__main__":
    unittest.main()
