"""Pure-Python coverage for label-template direct-grounding queries."""

import unittest

from projects.rl.datasets.grounding_queries import (
    COCO_STUFF_CLASS_NAMES,
    cocostuff_grounding_query,
    paco_part_grounding_query,
)


class GroundingQueryTest(unittest.TestCase):
    def test_cocostuff_pixel_values_use_official_order(self):
        self.assertEqual(len(COCO_STUFF_CLASS_NAMES), 91)
        self.assertEqual(cocostuff_grounding_query(91), "the banner")
        self.assertEqual(cocostuff_grounding_query(181), "the wood")

    def test_paco_query_keeps_part_and_parent(self):
        self.assertEqual(
            paco_part_grounding_query("front-wheel", "mountain_bike"),
            "the front wheel of the mountain bike",
        )

    def test_unknown_stuff_value_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            cocostuff_grounding_query(90)


if __name__ == "__main__":
    unittest.main()
