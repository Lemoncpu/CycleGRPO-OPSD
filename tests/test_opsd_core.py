import unittest

import torch
from PIL import Image

from verl.workers.opsd.distillation import (
    caption_blocked_special_token_ids,
    chunked_weighted_jsd_loss,
)
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
    caption_safety_reason,
    classify_route,
    distillation_weight,
    format_privileged_prompt,
    regenerate_weight,
    teacher_caption_is_safe,
    uses_original_grpo,
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

    def test_caption_safety_rejects_special_tokens_json_and_overlength(self):
        self.assertIsNone(
            caption_safety_reason(
                "<think>brief reasoning</think> A red ceramic cup.",
                response_tokens=12,
                max_response_tokens=256,
            )
        )
        self.assertIsNone(caption_safety_reason("A red cup.<|im_end|>"))
        self.assertEqual(
            caption_safety_reason("<|mt_start|><|mt_0001|><|mt_end|>"),
            "special_token",
        )
        self.assertEqual(
            caption_safety_reason('{"mask_2d": "<|mt_start|>..."}'),
            "mask_json",
        )
        self.assertEqual(
            caption_safety_reason("A red cup.", response_tokens=257, max_response_tokens=256),
            "overlength",
        )

    def test_b_mode_preserves_original_grpo_only_for_safe_captions(self):
        self.assertTrue(
            uses_original_grpo(
                REGENERATE_ROUTE,
                caption_safe=True,
                preserve_original_grpo=True,
            )
        )
        self.assertTrue(
            uses_original_grpo(
                ON_POLICY_DISTILL_ROUTE,
                caption_safe=True,
                preserve_original_grpo=True,
            )
        )
        self.assertFalse(
            uses_original_grpo(
                REGENERATE_ROUTE,
                caption_safe=False,
                preserve_original_grpo=True,
            )
        )
        self.assertFalse(
            uses_original_grpo(
                ON_POLICY_DISTILL_ROUTE,
                caption_safe=True,
                preserve_original_grpo=False,
            )
        )
        self.assertTrue(
            uses_original_grpo(
                GRPO_ROUTE,
                caption_safe=True,
                preserve_original_grpo=False,
            )
        )

    def test_caption_distillation_blocks_mask_and_object_reference_vocabulary(self):
        vocab = {
            "ordinary": 0,
            "<|mt_start|>": 1,
            "<|mt_0007|>": 2,
            "<|mt_end|>": 3,
            "<|object_ref_start|>": 4,
            "<|object_ref_end|>": 5,
            "<|im_end|>": 6,
        }
        self.assertEqual(caption_blocked_special_token_ids(vocab), [1, 2, 3, 4, 5])

        torch.manual_seed(11)
        student_logits = torch.randn(1, 2, 7, requires_grad=True)
        teacher_logits = torch.randn(1, 2, 7)
        blocked_ids = torch.tensor([5, 6])
        common_kwargs = {
            "target_ids": torch.tensor([[0, 1]]),
            "response_mask": torch.ones(1, 2),
            "sample_weight": torch.ones(1),
            "beta": 0.5,
            "temperature": 1.0,
            "entropy_weight_beta": 1.0,
            "token_chunk_size": 1,
            "blocked_token_ids": blocked_ids,
        }
        baseline_loss, baseline_metrics = chunked_weighted_jsd_loss(
            student_logits,
            teacher_logits,
            **common_kwargs,
        )
        self.assertTrue(torch.isfinite(baseline_loss))
        self.assertTrue(torch.isfinite(baseline_metrics["teacher_entropy"]))
        self.assertTrue(torch.isfinite(baseline_metrics["local_loss"]))
        baseline_loss.backward()
        self.assertTrue(torch.isfinite(student_logits.grad).all())
        self.assertTrue(torch.equal(student_logits.grad[..., blocked_ids], torch.zeros_like(student_logits.grad[..., blocked_ids])))
        self.assertEqual(baseline_metrics["blocked_vocab_size"].item(), 2.0)

        changed_student = student_logits.detach().clone().requires_grad_(True)
        changed_teacher = teacher_logits.detach().clone()
        changed_student.data[..., blocked_ids] = 1e4
        changed_teacher[..., blocked_ids] = -1e4
        changed_loss, _ = chunked_weighted_jsd_loss(
            changed_student,
            changed_teacher,
            **common_kwargs,
        )
        self.assertTrue(torch.isfinite(changed_loss))
        changed_loss.backward()
        self.assertTrue(torch.isfinite(changed_student.grad).all())
        self.assertTrue(
            torch.equal(
                changed_student.grad[..., blocked_ids],
                torch.zeros_like(changed_student.grad[..., blocked_ids]),
            )
        )
        self.assertTrue(torch.allclose(changed_loss, baseline_loss.detach(), atol=1e-6, rtol=1e-6))

    def test_chunked_jsd_matches_dense_loss_and_gradient(self):
        torch.manual_seed(7)
        student_logits = torch.randn(2, 5, 11, requires_grad=True)
        teacher_logits = torch.randn(2, 5, 11)
        target_ids = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
        response_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0]])
        sample_weight = torch.tensor([0.75, 0.25])
        beta = 0.5
        temperature = 0.8
        entropy_weight_beta = 1.2

        student_log_probs = torch.log_softmax(student_logits.float() / temperature, dim=-1)
        teacher_log_probs = torch.log_softmax(teacher_logits.float() / temperature, dim=-1)
        mixture_log_probs = torch.logsumexp(
            torch.stack(
                [
                    student_log_probs + torch.log(torch.tensor(1.0 - beta)),
                    teacher_log_probs + torch.log(torch.tensor(beta)),
                ]
            ),
            dim=0,
        )
        kl_student = (
            student_log_probs.exp() * (student_log_probs - mixture_log_probs)
        ).sum(dim=-1)
        kl_teacher = (
            teacher_log_probs.exp() * (teacher_log_probs - mixture_log_probs)
        ).sum(dim=-1)
        jsd = beta * kl_teacher + (1.0 - beta) * kl_student
        entropy = -(teacher_log_probs.exp() * teacher_log_probs).sum(dim=-1)
        token_mask = response_mask.float()
        confidence = torch.exp(-entropy_weight_beta * entropy)
        confidence = confidence / (
            (confidence * token_mask).sum(dim=-1, keepdim=True)
            / token_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        ).clamp_min(1e-6)
        weights = token_mask * confidence * sample_weight.unsqueeze(-1)
        dense_loss = (jsd * weights).sum()
        dense_metrics = {
            "local_loss": dense_loss.detach() / token_mask.sum().clamp_min(1.0),
            "teacher_entropy": (entropy * token_mask).sum() / token_mask.sum().clamp_min(1.0),
            "teacher_sequence_score": (
                teacher_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1) * token_mask
            ).sum()
            / token_mask.sum().clamp_min(1.0),
        }
        dense_loss.backward()
        dense_gradient = student_logits.grad.detach().clone()

        chunked_student_logits = student_logits.detach().clone().requires_grad_(True)
        chunked_loss, chunked_metrics = chunked_weighted_jsd_loss(
            student_logits=chunked_student_logits,
            teacher_logits=teacher_logits,
            target_ids=target_ids,
            response_mask=response_mask,
            sample_weight=sample_weight,
            beta=beta,
            temperature=temperature,
            entropy_weight_beta=entropy_weight_beta,
            token_chunk_size=2,
        )
        chunked_loss.backward()

        self.assertTrue(torch.allclose(chunked_loss, dense_loss.detach(), atol=1e-6, rtol=1e-6))
        self.assertTrue(
            torch.allclose(chunked_student_logits.grad, dense_gradient, atol=1e-6, rtol=1e-6)
        )
        for key, value in dense_metrics.items():
            self.assertTrue(torch.allclose(chunked_metrics[key], value, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
