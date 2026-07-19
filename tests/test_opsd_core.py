import unittest

import torch
from PIL import Image

from verl.workers.opsd.mask_iou import (
    coerce_raw_mask,
    compute_binary_iou,
    decode_mask_tokens,
    parse_mask_codes,
)
from verl.workers.opsd.routing import (
    GRPO_ROUTE,
    ON_POLICY_DISTILL_ROUTE,
    REGENERATE_ROUTE,
    aggregate_caption_rollouts,
    build_privileged_context,
    classify_route,
    distillation_weight,
    format_privileged_prompt,
    regenerate_weight,
    teacher_caption_is_safe,
)


class OPSDCoreTest(unittest.TestCase):
    def test_route_boundaries_are_strict(self):
        self.assertEqual(classify_route(0.4999), REGENERATE_ROUTE)
        self.assertEqual(classify_route(0.5), ON_POLICY_DISTILL_ROUTE)
        self.assertEqual(classify_route(0.85), ON_POLICY_DISTILL_ROUTE)
        self.assertEqual(classify_route(0.8501), GRPO_ROUTE)

    def test_six_caption_routes_are_exhaustive(self):
        routes = [classify_route(score) for score in (0.1, 0.4999, 0.5, 0.7, 0.85, 0.9)]
        self.assertEqual(len(routes), 6)
        self.assertEqual(routes.count(REGENERATE_ROUTE), 2)
        self.assertEqual(routes.count(ON_POLICY_DISTILL_ROUTE), 3)
        self.assertEqual(routes.count(GRPO_ROUTE), 1)

    def test_mask_token_offsets_are_validated(self):
        self.assertEqual(parse_mask_codes("<|mt_start|><|mt_0012|><|mt_0268|><|mt_end|>"), [12, 12])
        self.assertIsNone(parse_mask_codes("<|mt_start|><|mt_0256|><|mt_0268|><|mt_end|>"))
        self.assertIsNone(parse_mask_codes("not a mask"))

    def test_binary_iou(self):
        target = torch.tensor([[[1, 1], [0, 0]], [[0, 0], [0, 0]]])
        prediction = torch.tensor([[[1, 0], [1, 0]], [[0, 0], [0, 0]]])
        self.assertTrue(torch.allclose(compute_binary_iou(target, prediction), torch.tensor([1 / 3, 0.0])))
        resized = coerce_raw_mask([[1, 0], [0, 1]], (4, 4))
        self.assertEqual(tuple(resized.shape), (4, 4))

    def test_decoder_reuses_one_image_embedding_across_chunks(self):
        class FakeDecoder:
            device = torch.device("cpu")
            dtype = torch.float32

            def __init__(self):
                self.encode_calls = 0
                self.decode_calls = 0

            def encode_single_image(self, pixel_values):
                self.encode_calls += 1
                return pixel_values

            def decode_codes_from_single_image(self, image_state, codes):
                self.decode_calls += 1
                return torch.ones((len(codes), 1, 2, 2), dtype=torch.float32)

        decoder = FakeDecoder()
        image = Image.new("RGB", (3, 2), color="white")
        masks, size = decode_mask_tokens(
            vq_sam2=decoder,
            image=image,
            token_texts=[
                "<|mt_start|><|mt_0001|><|mt_0257|><|mt_end|>",
                "<|mt_start|><|mt_0002|><|mt_0258|><|mt_end|>",
                "invalid",
            ],
            decode_batch_size=1,
        )
        self.assertEqual(decoder.encode_calls, 1)
        self.assertEqual(decoder.decode_calls, 2)
        self.assertEqual(size, (2, 3))
        self.assertEqual(tuple(masks[0].shape), (2, 3))
        self.assertIsNone(masks[2])

    def test_aggregate_and_privileged_context(self):
        ious = [0.1, 0.4, 0.7, 0.8, 0.9, 0.6]
        aggregate = aggregate_caption_rollouts(ious)
        self.assertAlmostEqual(aggregate["R_Ci"], sum(ious) / len(ious), places=6)
        self.assertEqual(aggregate["route"], ON_POLICY_DISTILL_ROUTE)

        target = torch.tensor([[1, 1], [0, 0]], dtype=torch.bool)
        predictions = [target.clone() for _ in ious]
        context = build_privileged_context(
            student_caption="a red object",
            target_mask_token="<|mt_start|><|mt_0001|><|mt_0257|><|mt_end|>",
            predicted_mask_tokens=[None] * len(ious),
            pixel_ious=ious,
            target_mask=target,
            predicted_masks=predictions,
        )
        self.assertEqual(context["target_summary"]["area"], 2)
        self.assertEqual(len(context["pixel_ious"]), 6)
        self.assertEqual(context["representative_mask"]["shape"], [2, 2])
        self.assertEqual(context["relative_position"]["horizontal"], "aligned")
        self.assertEqual(context["valid_mask_token_count"], 0)
        diagnosis_prompt = format_privileged_prompt(context, mode="analysis")
        self.assertIn("failure_mode", diagnosis_prompt)
        self.assertIn("correction_focus", diagnosis_prompt)

    def test_route_weights(self):
        self.assertAlmostEqual(regenerate_weight(0.4, 0.7), 0.5)
        self.assertEqual(regenerate_weight(0.4, 0.3), 0.0)
        self.assertAlmostEqual(distillation_weight(0.5), 1.0)
        self.assertAlmostEqual(distillation_weight(0.85), 0.1)

    def test_teacher_caption_leakage_filter(self):
        self.assertTrue(teacher_caption_is_safe("A red ceramic cup beside a silver spoon."))
        self.assertFalse(teacher_caption_is_safe("The mask score has improved."))
        self.assertFalse(
            teacher_caption_is_safe("<|mt_start|><|mt_0001|><|mt_0257|><|mt_end|>")
        )
        self.assertFalse(teacher_caption_is_safe("A cup. <|im_start|>system"))


if __name__ == "__main__":
    unittest.main()
