import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - local docs-only environments
    np = None

if np is not None:
    from evaluation.gres.subset_metrics import (
        GresMetricAccumulator,
        TwoInstanceDiagnosticAccumulator,
        multi_instance_count_bucket,
        target_area_bucket,
        two_instance_area_balance_bucket,
        two_instance_center_distance_bucket,
        two_instance_diagnostic_sample,
    )


@unittest.skipIf(np is None, "NumPy is required by the GRES evaluator")
class TestGresSubsetMetrics(unittest.TestCase):
    def test_accumulator_preserves_empty_target_ciou_semantics(self):
        accumulator = GresMetricAccumulator()
        accumulator.add(np.array([[1, 0], [0, 0]]), np.array([[1, 1], [0, 0]]))
        accumulator.add(np.zeros((2, 2)), np.zeros((2, 2)))
        accumulator.add(np.array([[0, 0], [0, 1]]), np.zeros((2, 2)))

        metrics = accumulator.as_dict()
        self.assertEqual(metrics["samples"], 3)
        self.assertEqual(metrics["target_samples"], 1)
        self.assertEqual(metrics["no_target_samples"], 2)
        self.assertEqual(metrics["T_acc"], 100.0)
        self.assertEqual(metrics["N_acc"], 50.0)
        self.assertAlmostEqual(metrics["mIoU_target"], 50.0)
        self.assertAlmostEqual(metrics["gIoU"], 50.0)
        self.assertAlmostEqual(metrics["cIoU"], 100.0 / 3.0)

    def test_target_area_buckets(self):
        self.assertEqual(target_area_bucket(np.array([[1] * 20])), "large_ge_25pct")
        medium = np.zeros((10, 10), dtype=np.uint8)
        medium[:2, :5] = 1
        self.assertEqual(target_area_bucket(medium), "medium_5_to_25pct")
        small = np.zeros((10, 10), dtype=np.uint8)
        small[0, 0] = 1
        self.assertEqual(target_area_bucket(small), "small_lt_5pct")
        with self.assertRaises(ValueError):
            target_area_bucket(np.zeros((2, 2), dtype=np.uint8))

    def test_multi_instance_count_buckets(self):
        self.assertEqual(multi_instance_count_bucket(2), "2_instances")
        self.assertEqual(multi_instance_count_bucket(3), "3_instances")
        self.assertEqual(multi_instance_count_bucket(4), "4plus_instances")
        self.assertEqual(multi_instance_count_bucket(12), "4plus_instances")
        with self.assertRaises(ValueError):
            multi_instance_count_bucket(1)

    def test_two_instance_member_diagnostics(self):
        small_member = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        large_member = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]])
        prediction = small_member.copy()
        gt_mask = np.logical_or(small_member, large_member)
        sample = two_instance_diagnostic_sample(
            prediction,
            [small_member, large_member],
            member_recall_threshold=0.5,
        )
        self.assertEqual(sample.small_member_coverage, 1.0)
        self.assertEqual(sample.large_member_coverage, 0.0)
        self.assertTrue(sample.small_member_hit)
        self.assertFalse(sample.large_member_hit)
        self.assertEqual(two_instance_area_balance_bucket(sample.small_to_large_area_ratio), "moderate_0.2_to_0.5")
        self.assertEqual(two_instance_center_distance_bucket(sample.center_distance_diag), "far_ge_0.5diag")

        accumulator = TwoInstanceDiagnosticAccumulator(member_recall_threshold=0.5)
        accumulator.add(prediction, gt_mask, sample)
        metrics = accumulator.as_dict()
        self.assertEqual(metrics["one_member_only_recall_at_threshold"], 100.0)
        self.assertEqual(metrics["small_member_recall_at_threshold"], 100.0)
        self.assertEqual(metrics["large_member_recall_at_threshold"], 0.0)
