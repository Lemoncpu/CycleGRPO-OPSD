import unittest
import torch

from verl.workers.opsd.mask_iou import hierarchical_mask_token_weights
from verl.workers.opsd.routing import evidence_gate_weight
from verl.workers.opsd.cbba import cbba_stream_scales


class PEGCModuleTest(unittest.TestCase):
    def test_evidence_gate_penalizes_false_positive_evidence(self):
        clean = {"iou_mean": 0.6, "target_summary": {"area": 100}, "reconstruction_only_summary": {"area": 0}}
        noisy = {"iou_mean": 0.6, "target_summary": {"area": 100}, "reconstruction_only_summary": {"area": 50}}
        self.assertGreater(evidence_gate_weight(clean), evidence_gate_weight(noisy))

    def test_hierarchical_mask_token_weights(self):
        ids = torch.tensor([[151669, 151700, 151900, 152182, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 0]])
        weights = hierarchical_mask_token_weights(ids, mask, coarse_weight=1.5, fine_weight=1.0)
        self.assertEqual(weights.tolist(), [[1.0, 1.5, 1.0, 1.0, 1.0]])

    def test_cbba_shifts_weight_to_supervision_when_cycle_is_weak(self):
        cycle, supervised, confidence = cbba_stream_scales([0.4, 0.5])
        self.assertAlmostEqual(confidence, 0.45)
        self.assertLess(cycle, 1.0)
        self.assertGreater(supervised, 1.0)

    def test_cbba_is_neutral_at_target(self):
        self.assertEqual(cbba_stream_scales([0.7, 0.7]), (1.0, 1.0, 0.7))


if __name__ == "__main__":
    unittest.main()
