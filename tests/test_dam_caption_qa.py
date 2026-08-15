import unittest

from projects.rl.datasets.generate_dam_caption_qa import (
    GENERATION_PROMPT,
    VALIDATION_PROMPT,
    extract_json_object,
    normalize_generated_payload,
)


class DamCaptionQaTest(unittest.TestCase):
    def test_extract_json_object_strips_markdown(self):
        payload = extract_json_object("```json\n{\"accepted\": true}\n```")
        self.assertEqual(payload, {"accepted": True})

    def test_normalize_valid_payload(self):
        payload = normalize_generated_payload(
            {
                "class_name": "wooden chair",
                "questions": [
                    {
                        "type": "positive",
                        "question": "What color is the chair?",
                        "choices": [["brown", 1], ["blue", 0], ["red", 0], ["green", 0]],
                    },
                    {
                        "type": "positive",
                        "question": "What material is the chair made of?",
                        "choices": [["wood", 1], ["glass", 0], ["steel", 0], ["plastic", 0]],
                    },
                    {
                        "type": "negative",
                        "question": "Does the description state that the chair is transparent?",
                        "choices": [["Yes", -1], ["No", 0]],
                    },
                ],
            }
        )
        self.assertEqual(payload["class_name"], "wooden chair")
        self.assertEqual(len(payload["questions"]), 3)

    def test_rejects_ambiguous_positive_scores(self):
        with self.assertRaisesRegex(ValueError, "exactly one score 1"):
            normalize_generated_payload(
                {
                    "class_name": "chair",
                    "questions": [
                        {
                            "type": "positive",
                            "question": "What is it?",
                            "choices": [["chair", 1], ["seat", 1], ["table", 0], ["lamp", 0]],
                        },
                        {
                            "type": "positive",
                            "question": "What color is it?",
                            "choices": [["blue", 1], ["red", 0], ["green", 0], ["yellow", 0]],
                        },
                    ],
                }
            )

    def test_prompt_templates_format_literal_json_schemas(self):
        caption = "A wooden chair with a curved back is painted blue."
        candidate_json = '{"class_name": "chair", "questions": []}'

        generation_prompt = GENERATION_PROMPT.format(caption=caption)
        validation_prompt = VALIDATION_PROMPT.format(
            caption=caption, candidate_json=candidate_json
        )

        self.assertIn(caption, generation_prompt)
        self.assertIn('"class_name": "short common-noun phrase"', generation_prompt)
        self.assertIn('"questions": [', generation_prompt)
        self.assertIn(caption, validation_prompt)
        self.assertIn(candidate_json, validation_prompt)
        self.assertIn('{"accepted": true_or_false, "reasons": ["short reason"]}', validation_prompt)


if __name__ == "__main__":
    unittest.main()
