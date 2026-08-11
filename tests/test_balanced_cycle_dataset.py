"""Unit tests for configurable CycleGRPO mixture sizes."""

import json
import tempfile
import unittest
from argparse import Namespace

from projects.rl.datasets.prepare_balanced_cyclegrpo_dataset import (
    assert_multi_rows,
    load_caption_qa_ids,
    mixture_counts,
    select_rows,
    select_rows_with_required_ids,
)


class BalancedCycleDatasetTest(unittest.TestCase):
    def test_accepts_a_25k_mixture(self):
        counts = mixture_counts(
            Namespace(
                single_count=8_750,
                multi_count=6_250,
                stuff_count=5_000,
                part_count=2_500,
                no_target_count=2_500,
            )
        )
        self.assertEqual(sum(counts.values()), 25_000)

    def test_rejects_an_empty_mixture(self):
        with self.assertRaisesRegex(ValueError, "positive total"):
            mixture_counts(
                Namespace(
                    single_count=0,
                    multi_count=0,
                    stuff_count=0,
                    part_count=0,
                    no_target_count=0,
                )
            )

    def test_qa_ids_are_required_before_random_fill(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl") as handle:
            handle.write(json.dumps({
                "dam_source_id": "stuff-a",
                "source": "cocostuff_cycle",
                "questions": [
                    {
                        "type": "positive",
                        "question": "What is the object?",
                        "choices": [["chair", 1], ["table", 0], ["lamp", 0], ["book", 0]],
                    },
                    {
                        "type": "positive",
                        "question": "What color is the object?",
                        "choices": [["blue", 1], ["red", 0], ["green", 0], ["white", 0]],
                    },
                ],
            }) + "\n")
            handle.flush()
            qa_ids = load_caption_qa_ids(__import__("pathlib").Path(handle.name))
        selected = select_rows_with_required_ids(
            [{"source": "cocostuff_cycle", "dam_source_id": "stuff-a", "images": ["a"]},
             {"source": "cocostuff_cycle", "dam_source_id": "stuff-b", "images": ["b"]}],
            source="cocostuff_cycle", count=2, required_ids=qa_ids["cocostuff_cycle"], seed=1, name="Stuff"
        )
        self.assertEqual(selected[0]["dam_source_id"], "stuff-a")

    def test_rejects_single_instance_in_multi_quota(self):
        with self.assertRaisesRegex(RuntimeError, "grounding_instance_count"):
            assert_multi_rows([{"grounding_instance_count": 1}])

    def test_multi_selection_filters_before_sampling(self):
        selected = select_rows(
            [
                {"source": "grefcoco_cycle", "grounding_instance_count": 1, "images": ["single"]},
                {"source": "grefcoco_cycle", "grounding_instance_count": 2, "images": ["multi"]},
            ],
            source="grefcoco_cycle",
            count=1,
            seed=1,
            name="multi",
            predicate=lambda row: row["grounding_instance_count"] >= 2,
        )
        self.assertEqual(selected[0]["grounding_instance_count"], 2)


if __name__ == "__main__":
    unittest.main()
