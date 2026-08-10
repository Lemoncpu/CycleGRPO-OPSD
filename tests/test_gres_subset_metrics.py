import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - local docs-only environments
    np = None

if np is not None:
    from evaluation.gres.subset_metrics import GresMetricAccumulator, target_area_bucket


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
