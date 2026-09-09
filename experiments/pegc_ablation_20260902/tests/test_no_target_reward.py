"""Regression tests for GRES no-target reward semantics."""

import unittest

from projects.rl.reward_function.text2mask import no_target_check, no_target_check_bbox, no_target_reward_score


class NoTargetRewardTest(unittest.TestCase):
    def test_valid_refusal_receives_full_accuracy_reward(self):
        self.assertEqual(no_target_check("<answer>No target.</answer>", "No target."), 1.0)

    def test_missing_refusal_without_mask_receives_partial_reward(self):
        self.assertEqual(no_target_check("I cannot identify the object.", "No target."), 0.2)

    def test_any_mask_token_fragment_invalidates_refusal(self):
        responses = (
            "<answer>No target.</answer><|mt_start|><|mt_0003|><|mt_0331|><|mt_end|>",
            "<answer>No target.</answer><|mt_start|>",
            "<answer>No target.</answer><|mt_0003|>",
            "<answer>No target.</answer><|mt_end|>",
        )
        for response in responses:
            with self.subTest(response=response):
                self.assertEqual(no_target_check(response, "No target."), 0.0)

    def test_official_bbox_proxy_matches_public_cyclegrpo(self):
        self.assertEqual(no_target_check_bbox("<answer>No target.</answer>", "No target."), 1.0)
        self.assertEqual(no_target_check_bbox("I cannot identify the object.", "No target."), 0.2)
        self.assertEqual(
            no_target_check_bbox("<answer>No target.</answer> [1, 2, 3, 4]", "No target."),
            0.0,
        )

    def test_official_bbox_mode_is_selectable(self):
        score = no_target_reward_score(
            {
                "response": "<answer>No target.</answer><|mt_start|><|mt_0003|><|mt_0331|><|mt_end|>",
                "no_target_reward_mode": "official_bbox",
            }
        )
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
