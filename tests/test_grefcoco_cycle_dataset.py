"""Regression tests for gRefCOCO direct-data selection."""

import unittest

from projects.rl.datasets.prepare_grefcoco_cycle_dataset import (
    image_id_from_path,
    normalized_expression,
    select_records,
)


def ref(ref_id: int, image_id: int, ann_id: list[int], phrase: str) -> dict:
    return {
        "ref_id": ref_id,
        "image_id": image_id,
        "ann_id": ann_id,
        "sentences": [{"raw": phrase}],
    }


class GrefCocoCycleDatasetTest(unittest.TestCase):
    def test_extracts_standard_coco_image_id(self):
        self.assertEqual(
            image_id_from_path("/data/train2014/COCO_train2014_000000123456.jpg"),
            123456,
        )

    def test_excludes_existing_image_expression_identity_before_sampling(self):
        refs = [
            ref(1, 10, [101, 102], "Two dogs"),
            ref(2, 11, [201, 202], "A red car"),
        ]
        _, multi, _ = select_records(
            refs,
            positive_samples=1,
            no_target_samples=0,
            single_fraction=0.0,
            seed=1,
            excluded_query_keys={(10, normalized_expression(" two   DOGS "))},
        )
        self.assertEqual(multi[0]["ref_id"], 2)

    def test_excludes_existing_target_mask_even_when_expression_differs(self):
        refs = [
            ref(1, 10, [101, 102], "The two dogs"),
            ref(2, 11, [201, 202], "A red car"),
        ]
        _, multi, _ = select_records(
            refs,
            positive_samples=1,
            no_target_samples=0,
            single_fraction=0.0,
            seed=1,
            excluded_mask_keys={(10, "same-union-mask")},
            positive_mask_keys={1: (10, "same-union-mask")},
        )
        self.assertEqual(multi[0]["ref_id"], 2)

    def test_deduplicates_same_target_inside_new_selection(self):
        refs = [
            ref(1, 10, [101, 102], "The two dogs"),
            ref(2, 10, [101, 102], "The dogs together"),
            ref(3, 11, [201, 202], "A red car"),
        ]
        _, multi, _ = select_records(
            refs,
            positive_samples=2,
            no_target_samples=0,
            single_fraction=0.0,
            seed=1,
            positive_mask_keys={1: (10, "same-union-mask"), 2: (10, "same-union-mask"), 3: (11, "car-mask")},
        )
        self.assertEqual({item["ref_id"] for item in multi}, {1, 3})


if __name__ == "__main__":
    unittest.main()
