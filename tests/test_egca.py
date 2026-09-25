import unittest

import torch

from verl.workers.opsd.config import EGCAConfig
from verl.workers.opsd.egca import (
    combine_evidence_and_shapley,
    contrastive_token_credit,
    evidence_weight_from_masks,
    extract_code_positions,
    mask_group_text,
    nonnegative_mask_group_token_weights,
    shapley_depth_credit,
)


class EGCATest(unittest.TestCase):
    def test_shapley_efficiency(self):
        coarse, fine = shapley_depth_credit(0.1, 0.4, 0.2, 0.8, credit_clip=None)
        self.assertAlmostEqual(coarse + fine, 0.7)
        self.assertAlmostEqual(coarse, 0.45)
        self.assertAlmostEqual(fine, 0.25)

    def test_single_group_positions_are_separate(self):
        ids = torch.tensor([151669, 151700, 152056, 152182, 42, 151669, 151701, 152057, 152182])
        mask = torch.ones_like(ids)
        groups = extract_code_positions(ids, mask)
        self.assertEqual(groups, [([30, 130], [1, 2]), ([31, 131], [6, 7])])

    def test_group_text_uses_global_fine_code(self):
        self.assertIn("mt_0256", mask_group_text(3, 0))

    def test_evidence_gate_and_credit(self):
        target = torch.zeros(4, 4, dtype=torch.bool)
        target[:2, :2] = True
        prediction = target.clone()
        weight = evidence_weight_from_masks(target, prediction)
        self.assertAlmostEqual(weight, 1.0)
        coarse, fine = combine_evidence_and_shapley(0.4, -0.2, weight)
        self.assertAlmostEqual(coarse, 0.4)
        self.assertAlmostEqual(fine, -0.2)

    def test_config_defaults_to_disabled(self):
        config = EGCAConfig()
        config.post_init()
        self.assertFalse(config.enabled)
        self.assertTrue(config.opd_enabled)
        self.assertEqual(config.credit_mode, "contrastive")
        self.assertEqual(config.update_mode, "weighted_ce")
        self.assertEqual(config.reference_mode, "target")

    def test_contrastive_credit_preserves_only_relative_token_signal(self):
        credit = torch.tensor([[0.4, 0.2, 0.0, -0.1]])
        active = torch.tensor([[True, True, False, True]])
        response_mask = torch.ones_like(active)
        projected = contrastive_token_credit(credit, active, response_mask)
        self.assertAlmostEqual(float(projected[0, :].sum()), 0.0)
        self.assertEqual(float(projected[0, 2]), 0.0)
        self.assertAlmostEqual(float(projected[0, 0]), 0.23333333, places=5)
        self.assertAlmostEqual(float(projected[0, 1]), 0.03333333, places=5)
        self.assertAlmostEqual(float(projected[0, 3]), -0.26666667, places=5)

    def test_contrastive_credit_does_not_invent_signal_for_single_token(self):
        credit = torch.tensor([[0.7, 0.0]])
        active = torch.tensor([[True, False]])
        projected = contrastive_token_credit(credit, active)
        self.assertTrue(torch.equal(projected, torch.zeros_like(projected)))

    def test_config_rejects_unknown_credit_mode(self):
        config = EGCAConfig(credit_mode="unknown")
        with self.assertRaises(ValueError):
            config.post_init()

    def test_config_rejects_unknown_reference_mode(self):
        config = EGCAConfig(reference_mode="code_zero")
        with self.assertRaises(ValueError):
            config.post_init()

    def test_weighted_self_distillation_is_nonnegative_and_group_complete(self):
        # Start/code/fine/end, followed by a regular token. The helper should
        # weight the complete group, but never produce a negative CE weight.
        ids = torch.tensor([[151668, 151700, 152056, 151669, 42]])
        mask = torch.ones_like(ids)
        weights = nonnegative_mask_group_token_weights(
            ids,
            mask,
            sample_weights=[0.5],
            coarse_scores=[-0.8],
            fine_scores=[0.4],
        )
        self.assertTrue(torch.all(weights >= 0))
        self.assertAlmostEqual(float(weights[0, 0]), 0.5)
        self.assertAlmostEqual(float(weights[0, 1]), 0.5)
        self.assertGreater(float(weights[0, 2]), 0.5)
        self.assertAlmostEqual(float(weights[0, 3]), 0.5)
        self.assertAlmostEqual(float(weights[0, 4]), 1.0)


if __name__ == "__main__":
    unittest.main()
