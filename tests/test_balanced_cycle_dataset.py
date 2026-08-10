"""Unit tests for configurable CycleGRPO mixture sizes."""

import unittest
from argparse import Namespace

from projects.rl.datasets.prepare_balanced_cyclegrpo_dataset import mixture_counts


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


if __name__ == "__main__":
    unittest.main()
