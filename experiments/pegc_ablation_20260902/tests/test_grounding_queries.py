"""Pure-Python coverage for label-template direct-grounding queries."""

import unittest

from projects.rl.datasets.grounding_queries import (
    COCO_STUFF_CLASS_NAMES,
    cocostuff_grounding_query,
    paco_parent_parts_grounding_query,
)


class GroundingQueryTest(unittest.TestCase):
    def test_cocostuff_pixel_values_use_official_order(self):
        self.assertEqual(len(COCO_STUFF_CLASS_NAMES), 91)
        self.assertEqual(cocostuff_grounding_query(91), "the banner")
        self.assertEqual(cocostuff_grounding_query(181), "the wood")

    def test_paco_query_uses_only_observed_parent_label(self):
        self.assertEqual(
            paco_parent_parts_grounding_query("mountain_bike"),
            "the visible parts of the mountain bike",
        )

    def test_unknown_stuff_value_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            cocostuff_grounding_query(90)


if __name__ == "__main__":
    unittest.main()
