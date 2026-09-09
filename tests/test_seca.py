import unittest

import torch

from verl.workers.opsd.config import SECAConfig
from verl.workers.opsd.seca import hierarchical_mask_token_weights, spatial_evidence_weight


class SECATest(unittest.TestCase):
    def test_evidence_gate_penalizes_false_positive_reconstruction(self):
        clean = {
            "iou_mean": 0.6,
            "target_summary": {"area": 100},
            "reconstruction_only_summary": {"area": 0},
        }
        noisy = {
            "iou_mean": 0.6,
            "target_summary": {"area": 100},
            "reconstruction_only_summary": {"area": 50},
        }
        self.assertGreater(spatial_evidence_weight(clean), spatial_evidence_weight(noisy))

    def test_hierarchical_mask_token_weights(self):
        ids = torch.tensor([[151669, 151700, 151900, 152182, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 0]])
        weights = hierarchical_mask_token_weights(ids, mask, coarse_weight=1.5, fine_weight=1.0)
        self.assertEqual(weights.tolist(), [[1.0, 1.5, 1.0, 1.0, 1.0]])

    def test_config_defaults_to_disabled(self):
        config = SECAConfig()
        config.post_init()
        self.assertFalse(config.enabled)


if __name__ == "__main__":
    unittest.main()
