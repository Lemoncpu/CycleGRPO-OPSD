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
alternating_localization_prompt_variants = _CONFIG_MODULE.alternating_localization_prompt_variants
direct_grounding_source = _CONFIG_MODULE.direct_grounding_source


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
        with self.assertRaisesRegex(ValueError, "at least 2"):
            DirectGroundingConfig(enabled=True, rollouts=1).post_init()

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
