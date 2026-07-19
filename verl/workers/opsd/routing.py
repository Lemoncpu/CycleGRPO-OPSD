from typing import Optional, Sequence

import numpy as np
import torch

from .mask_iou import mask_summary


REGENERATE_ROUTE = "regenerate"
ON_POLICY_DISTILL_ROUTE = "on_policy_distill"
GRPO_ROUTE = "grpo"


PRIVILEGED_LEAKAGE_TERMS = (
    "iou",
    "gtmask",
    "refmask",
    "region1",
    "region2",
    "mask token",
    "mask score",
    "reconstruction",
    "reconstruction score",
    "localization score",
    "privileged",
    "target region",
    "region 1",
    "region 2",
    "<|",
)


def classify_route(score: float, low_threshold: float = 0.5, high_threshold: float = 0.85) -> str:
    score = float(score)
    if score < low_threshold:
        return REGENERATE_ROUTE
    if score <= high_threshold:
        return ON_POLICY_DISTILL_ROUTE
    return GRPO_ROUTE


def regenerate_weight(original_score: float, teacher_score: float, eps: float = 1e-6) -> float:
    improvement = float(teacher_score) - float(original_score)
    return float(np.clip(improvement / max(1.0 - float(original_score), eps), 0.0, 1.0))


def distillation_weight(
    score: float,
    low_threshold: float = 0.5,
    high_threshold: float = 0.85,
    min_weight: float = 0.1,
) -> float:
    denominator = max(float(high_threshold) - float(low_threshold), 1e-6)
    return float(np.clip((float(high_threshold) - float(score)) / denominator, min_weight, 1.0))


def aggregate_caption_rollouts(
    ious: Sequence[float], low_threshold: float = 0.5, high_threshold: float = 0.85
) -> dict[str, object]:
    values = np.asarray(ious, dtype=np.float32)
    if values.size == 0:
        raise ValueError("A caption must have at least one localization rollout.")
    mean = float(values.mean())
    representative_index = int(np.argmin(np.abs(values - mean)))
    best_index = int(np.argmax(values))
    return {
        "R_Ci": mean,
        "route": classify_route(mean, low_threshold, high_threshold),
        "iou_mean": mean,
        "iou_std": float(values.std()),
        "iou_min": float(values.min()),
        "iou_max": float(values.max()),
        "representative_index": representative_index,
        "best_index": best_index,
    }


def _difference_summary(left: Optional[torch.Tensor], right: Optional[torch.Tensor]) -> dict[str, object]:
    if left is None or right is None:
        return mask_summary(None)
    return mask_summary(torch.logical_and(left.to(dtype=torch.bool), torch.logical_not(right.to(dtype=torch.bool))))


def _pack_mask(mask: Optional[torch.Tensor]) -> Optional[dict[str, object]]:
    if mask is None:
        return None
    array = mask.detach().cpu().to(dtype=torch.uint8).numpy()
    return {
        "shape": list(array.shape),
        "packbits": np.packbits(array.reshape(-1)).tobytes(),
    }


def _relative_position(target: dict[str, object], reconstruction: dict[str, object]) -> dict[str, object]:
    target_center = target.get("center")
    reconstruction_center = reconstruction.get("center")
    if target_center is None or reconstruction_center is None:
        return {"horizontal": "unknown", "vertical": "unknown", "delta": None}
    dx = float(reconstruction_center[0]) - float(target_center[0])
    dy = float(reconstruction_center[1]) - float(target_center[1])
    tolerance = 2.0
    horizontal = "aligned" if abs(dx) <= tolerance else ("right" if dx > 0 else "left")
    vertical = "aligned" if abs(dy) <= tolerance else ("below" if dy > 0 else "above")
    return {"horizontal": horizontal, "vertical": vertical, "delta": [round(dx, 2), round(dy, 2)]}


def build_privileged_context(
    *,
    student_caption: str,
    target_mask_token: Optional[str],
    predicted_mask_tokens: Sequence[Optional[str]],
    pixel_ious: Sequence[float],
    target_mask: Optional[torch.Tensor],
    predicted_masks: Sequence[Optional[torch.Tensor]],
    low_threshold: float = 0.5,
    high_threshold: float = 0.85,
) -> dict[str, object]:
    aggregate = aggregate_caption_rollouts(pixel_ious, low_threshold, high_threshold)
    representative_index = int(aggregate["representative_index"])
    best_index = int(aggregate["best_index"])
    representative_mask = predicted_masks[representative_index]
    best_mask = predicted_masks[best_index]
    target_summary = mask_summary(target_mask)
    representative_summary = mask_summary(representative_mask)
    context = {
        **aggregate,
        "student_caption": student_caption,
        "target_mask_token": target_mask_token,
        "predicted_mask_tokens": list(predicted_mask_tokens),
        "pixel_ious": [float(value) for value in pixel_ious],
        "representative_mask_token": predicted_mask_tokens[representative_index],
        "best_mask_token": predicted_mask_tokens[best_index],
        "target_summary": target_summary,
        "representative_summary": representative_summary,
        "best_summary": mask_summary(best_mask),
        "target_only_summary": _difference_summary(target_mask, representative_mask),
        "reconstruction_only_summary": _difference_summary(representative_mask, target_mask),
        "relative_position": _relative_position(target_summary, representative_summary),
        "localization_status": [
            "valid_mask_token" if token is not None else "invalid_or_missing_mask_token"
            for token in predicted_mask_tokens
        ],
        "valid_mask_token_count": sum(token is not None for token in predicted_mask_tokens),
        "representative_mask": _pack_mask(representative_mask),
        "best_mask": _pack_mask(best_mask),
    }
    return context


def format_privileged_prompt(context: dict[str, object], *, mode: str) -> str:
    def metric(name: str) -> float:
        value = context.get(name, 0.0)
        return float(value) if value is not None else 0.0

    base = (
        "<image>\nYou are improving region captioning with privileged localization evidence.\n"
        f"Target region token: {context.get('target_mask_token') or 'unavailable'}\n"
        f"Typical reconstruction token: {context.get('representative_mask_token') or 'invalid'}\n"
        f"Best reconstruction token: {context.get('best_mask_token') or 'invalid'}\n"
        f"Student caption: {context.get('student_caption', '')}\n"
        f"Localization IoUs: {context.get('pixel_ious')}\n"
        f"Mean/std/min/max: {metric('iou_mean'):.4f} / {metric('iou_std'):.4f} / "
        f"{metric('iou_min'):.4f} / {metric('iou_max'):.4f}\n"
        f"Target summary: {context.get('target_summary')}\n"
        f"Typical reconstruction summary: {context.get('representative_summary')}\n"
        f"Missing target evidence: {context.get('target_only_summary')}\n"
        f"Distractor evidence: {context.get('reconstruction_only_summary')}\n"
        f"Typical reconstruction relative to target: {context.get('relative_position')}\n"
        f"Localization output status: {context.get('localization_status')}\n"
    )
    if mode == "regenerate":
        return base + (
            "Write one corrected, detailed and natural caption for the target region. "
            "Use visible attributes that distinguish it from the reconstructed distractor. "
            "Output only the caption. Never mention masks, tokens, scores, regions, or this analysis."
        )
    if mode == "distill":
        return base + (
            "Evaluate the exact student caption trajectory token by token. Shift probability toward visible "
            "details supported by the target and away from details supported only by the reconstruction."
        )
    raise ValueError(f"Unknown privileged prompt mode: {mode}")


def teacher_caption_is_safe(text: str) -> bool:
    lowered = (text or "").lower()
    return bool(lowered.strip()) and not any(term in lowered for term in PRIVILEGED_LEAKAGE_TERMS)
