import importlib.util
import unittest
from pathlib import Path

from projects.rl.reward_function.llm_judge_reward import (
    _parse_option,
    aggregate_caption_qa_outcomes,
)
from projects.rl.datasets.generate_dam_caption_qa import validate_caption_qa_questions
_CONFIG_PATH = Path(__file__).parents[1] / "verl/workers/supervised_anchors.py"
_CONFIG_SPEC = importlib.util.spec_from_file_location("supervised_anchors_test", _CONFIG_PATH)
_CONFIG_MODULE = importlib.util.module_from_spec(_CONFIG_SPEC)
assert _CONFIG_SPEC.loader is not None
_CONFIG_SPEC.loader.exec_module(_CONFIG_MODULE)
CaptionQAConfig = _CONFIG_MODULE.CaptionQAConfig
DirectGroundingConfig = _CONFIG_MODULE.DirectGroundingConfig
DirectMaskCEConfig = _CONFIG_MODULE.DirectMaskCEConfig
aligned_direct_prompt_count = _CONFIG_MODULE.aligned_direct_prompt_count
alternating_localization_prompt_variants = _CONFIG_MODULE.alternating_localization_prompt_variants
direct_grounding_loss_weight = _CONFIG_MODULE.direct_grounding_loss_weight
direct_mask_ce_response_fields = _CONFIG_MODULE.direct_mask_ce_response_fields
direct_mask_ce_source = _CONFIG_MODULE.direct_mask_ce_source
direct_grounding_source = _CONFIG_MODULE.direct_grounding_source
non_tensor_batch_row = _CONFIG_MODULE.non_tensor_batch_row


class SupervisedAnchorsTest(unittest.TestCase):
    def test_caption_qa_averages_positive_and_negative_scores(self):
        result = aggregate_caption_qa_outcomes(1, [0, 0, 0], [(1.0, False), (0.0, False), (-1.0, False)])
        self.assertEqual(result[0]["reward"], 0.0)
        self.assertEqual(result[0]["question_count"], 3.0)

    def test_caption_qa_parse_failure_is_zero_contribution(self):
        result = aggregate_caption_qa_outcomes(1, [0], [(0.0, True)])
        self.assertEqual(result[0]["reward"], 0.0)
        self.assertEqual(result[0]["failure_count"], 1.0)
        self.assertEqual(_parse_option("C", 4), 2)
        self.assertIsNone(_parse_option("choice C", 4))

    def test_configs_reject_invalid_enabled_values(self):
        with self.assertRaisesRegex(ValueError, "qa_jsonl"):
            CaptionQAConfig(enabled=True).post_init()
        with self.assertRaisesRegex(ValueError, "train_files"):
            CaptionQAConfig(
                enabled=True,
                qa_jsonl="qa.jsonl",
                judge_base_url="http://judge",
                judge_model="judge",
            ).post_init()
        with self.assertRaisesRegex(ValueError, "at least 2"):
            DirectGroundingConfig(enabled=True, rollouts=1).post_init()
        with self.assertRaisesRegex(ValueError, "train_files"):
            DirectGroundingConfig(enabled=True).post_init()
        with self.assertRaisesRegex(ValueError, "must be false"):
            DirectGroundingConfig(
                enabled=True,
                train_files=["refcoco_full.parquet"],
                batch_size=8,
                consume_no_target_caption=True,
            ).post_init()
        with self.assertRaisesRegex(ValueError, "warmup steps"):
            DirectGroundingConfig(warmup_start_step=31, warmup_end_step=30).post_init()
        with self.assertRaisesRegex(ValueError, "include_positive_sources or include_no_target"):
            DirectMaskCEConfig(
                enabled=True, include_positive_sources=False, include_no_target=False
            ).post_init()

    def test_direct_grounding_defaults_to_six_additive_rollouts(self):
        config = DirectGroundingConfig()
        self.assertEqual(config.rollouts, 6)
        self.assertFalse(config.consume_no_target_caption)

    def test_auxiliary_streams_have_independent_batch_sizes(self):
        direct = DirectGroundingConfig(
            enabled=True, train_files=["refcoco_full.parquet"], batch_size=128
        )
        qa = CaptionQAConfig(
            enabled=True,
            train_files=["dlc_qa.parquet"],
            batch_size=64,
            qa_jsonl="qa.jsonl",
            judge_base_url="http://judge",
            judge_model="judge",
        )
        direct.post_init()
        qa.post_init()
        self.assertEqual(direct.batch_size, 128)
        self.assertEqual(qa.batch_size, 64)

    def test_direct_grounding_weight_warmup_boundaries(self):
        self.assertEqual(direct_grounding_loss_weight(10, 0.15, 10, 30), 0.0)
        self.assertAlmostEqual(direct_grounding_loss_weight(20, 0.15, 10, 30), 0.075)
        self.assertEqual(direct_grounding_loss_weight(30, 0.15, 10, 30), 0.15)
        self.assertEqual(direct_grounding_loss_weight(29, 0.15, 30, 30), 0.0)
        self.assertEqual(direct_grounding_loss_weight(30, 0.15, 30, 30), 0.15)

    def test_direct_mask_ce_source_selection_supports_opt_in_no_target(self):
        self.assertTrue(direct_mask_ce_source("refcoco_cycle"))
        self.assertTrue(direct_mask_ce_source("grefcoco_cycle"))
        self.assertFalse(direct_mask_ce_source("gres_no_target"))
        self.assertTrue(direct_mask_ce_source("gres_no_target", include_no_target=True))
        self.assertFalse(direct_mask_ce_source("cocostuff_cycle"))
        self.assertFalse(direct_mask_ce_source("paco_part_cycle"))

    def test_direct_mask_ce_can_train_no_target_text_target(self):
        responses, loss_masks, attention_masks = direct_mask_ce_response_fields(
            [[71, 72, 73]], eos_token_id=2, pad_token_id=0
        )
        self.assertEqual(responses, [[71, 72, 73, 2]])
        self.assertEqual(loss_masks, [[1, 1, 1, 0]])
        self.assertEqual(attention_masks, [[1, 1, 1, 1]])

    def test_direct_mask_ce_masks_eos_and_padding(self):
        responses, loss_masks, attention_masks = direct_mask_ce_response_fields(
            [[11, 12], [21]], eos_token_id=2, pad_token_id=0
        )
        self.assertEqual(responses, [[11, 12, 2], [21, 2, 0]])
        self.assertEqual(loss_masks, [[1, 1, 0], [1, 0, 0]])
        self.assertEqual(attention_masks, [[1, 1, 1], [1, 1, 0]])

    def test_direct_mask_ce_keeps_non_tensor_batch_axis_for_media(self):
        media_rows = [{"images": ["first.jpg"]}, {"images": ["second.jpg"]}]
        self.assertEqual(non_tensor_batch_row(media_rows, 1), {"images": ["second.jpg"]})

    def test_direct_grounding_keeps_no_target_out_of_cycle_sources(self):
        self.assertEqual(direct_grounding_source("refcoco_cycle", True), "supervised_grounding")
        self.assertEqual(direct_grounding_source("grefcoco_cycle", True), "supervised_grounding")
        self.assertEqual(
            direct_grounding_source("gres_no_target", True), "supervised_grounding_no_target"
        )
        self.assertIsNone(direct_grounding_source("gres_no_target", False))
        self.assertIsNone(direct_grounding_source("cocostuff_cycle", True))
        self.assertIsNone(direct_grounding_source("paco_part_cycle", True))

    def test_direct_grounding_can_include_label_supervision_sources(self):
        self.assertEqual(
            direct_grounding_source("cocostuff_cycle", False, include_label_sources=True),
            "supervised_grounding",
        )
        self.assertEqual(
            direct_grounding_source("paco_part_cycle", False, include_label_sources=True),
            "supervised_grounding",
        )

    def test_direct_grounding_can_use_no_target_without_expression_positives(self):
        self.assertIsNone(
            direct_grounding_source("refcoco_cycle", True, include_positive_sources=False)
        )
        self.assertEqual(
            direct_grounding_source("gres_no_target", True, include_positive_sources=False),
            "supervised_grounding_no_target",
        )

    def test_direct_grounding_uses_both_localization_prompt_variants(self):
        self.assertEqual(
            alternating_localization_prompt_variants(6),
            ["refcoco", "groundingsuite", "refcoco", "groundingsuite", "refcoco", "groundingsuite"],
        )
        self.assertEqual(alternating_localization_prompt_variants(0), [])

    def test_direct_grounding_prompt_count_is_world_size_aligned(self):
        self.assertEqual(aligned_direct_prompt_count(12, 8), 8)
        self.assertEqual(aligned_direct_prompt_count(16, 8), 16)
        self.assertEqual(aligned_direct_prompt_count(7, 8), 0)
        with self.assertRaisesRegex(ValueError, "positive"):
            aligned_direct_prompt_count(8, 0)

    def test_caption_qa_rejects_out_of_contract_scores(self):
        with self.assertRaisesRegex(ValueError, "exactly one score 1"):
            validate_caption_qa_questions(
                [
                    {"type": "positive", "question": "one", "choices": [["a", 1], ["b", 1], ["c", 0], ["d", 0]]},
                    {"type": "positive", "question": "two", "choices": [["a", 1], ["b", 0], ["c", 0], ["d", 0]]},
                ]
            )


if __name__ == "__main__":
    unittest.main()
