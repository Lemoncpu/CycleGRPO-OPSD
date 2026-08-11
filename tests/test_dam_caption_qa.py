import unittest

from projects.rl.datasets.generate_dam_caption_qa import (
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


if __name__ == "__main__":
    unittest.main()
