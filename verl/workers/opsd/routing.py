import re
from typing import Any, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from .mask_iou import mask_summary


REGENERATE_ROUTE = "regenerate"
ON_POLICY_DISTILL_ROUTE = "on_policy_distill"
GRPO_ROUTE = "grpo"

_TERMINAL_SPECIAL_TOKENS = re.compile(r"<\|(?:im_end|endoftext|eos)\|>", re.IGNORECASE)
_SPECIAL_TOKEN = re.compile(r"<\|[^|]+\|>")
_MASK_JSON = re.compile(r"[\"']?mask_2d[\"']?\s*:", re.IGNORECASE)


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

_PRIVILEGED_CROP_MAX_PIXELS = 512 * 512


def classify_route(score: float, low_threshold: float = 0.5, high_threshold: float = 0.85) -> str:
    score = float(score)
    if score < low_threshold:
        return REGENERATE_ROUTE
    if score <= high_threshold:
        return ON_POLICY_DISTILL_ROUTE
    return GRPO_ROUTE


def caption_safety_reason(
    text: str,
    *,
    response_tokens: Optional[int] = None,
    max_response_tokens: Optional[int] = None,
) -> Optional[str]:
    """Return the first reason a caption cannot safely train caption PPO/JSD.

    Rollout decoding preserves special tokens so that accidental SAMTok mask output
    can be detected. Normal terminal markers are stripped because vLLM may append
    them after an otherwise valid caption.
    """
    if max_response_tokens is not None and response_tokens is not None:
        if response_tokens > max_response_tokens:
            return "overlength"
    content = _TERMINAL_SPECIAL_TOKENS.sub("", text or "")
    if _MASK_JSON.search(content):
        return "mask_json"
    if _SPECIAL_TOKEN.search(content):
        return "special_token"
    return None


def uses_original_grpo(
    route: str,
    *,
    caption_safe: bool,
    preserve_original_grpo: bool = False,
) -> bool:
    """Whether a caption rollout contributes its native CycleGRPO policy loss."""
    return bool(caption_safe) and (preserve_original_grpo or route == GRPO_ROUTE)


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


def _unpack_mask(packed_mask: Optional[dict[str, object]]) -> Optional[np.ndarray]:
    if not packed_mask:
        return None
    shape = tuple(int(value) for value in packed_mask.get("shape", []))
    if len(shape) != 2 or min(shape) <= 0:
        return None
    packed_bits = packed_mask.get("packbits")
    if not isinstance(packed_bits, bytes):
        return None
    size = int(np.prod(shape))
    values = np.unpackbits(np.frombuffer(packed_bits, dtype=np.uint8), count=size)
    if values.size != size:
        return None
    return values.reshape(shape).astype(bool)


def _align_mask_to_image(mask: Optional[np.ndarray], image: Image.Image) -> Optional[np.ndarray]:
    if mask is None:
        return None
    if mask.shape == (image.height, image.width):
        return mask
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
        image.size, resample=Image.Resampling.NEAREST
    )
    return np.asarray(resized, dtype=np.uint8).astype(bool)


def _union_crop_box(
    target_mask: Optional[np.ndarray],
    reconstruction_mask: Optional[np.ndarray],
    *,
    width: int,
    height: int,
    padding_fraction: float,
) -> tuple[int, int, int, int]:
    candidates = [
        mask for mask in (target_mask, reconstruction_mask) if mask is not None and mask.any()
    ]
    if not candidates:
        return (0, 0, width, height)
    union = np.logical_or.reduce(candidates)
    ys, xs = np.nonzero(union)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    padding = int(np.ceil(max(x1 - x0, y1 - y0) * padding_fraction))
    return (
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(width, x1 + padding),
        min(height, y1 + padding),
    )


def _masked_crop(
    image: Image.Image,
    mask: Optional[np.ndarray],
    box: tuple[int, int, int, int],
) -> Image.Image:
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    isolated = np.full_like(pixels, 127)
    if mask is not None:
        isolated[mask] = pixels[mask]
    return Image.fromarray(isolated, mode="RGB").crop(box)


def _limit_crop_pixels(image: Image.Image, max_pixels: int) -> Image.Image:
    if image.width * image.height <= max_pixels:
        return image
    scale = float(np.sqrt(max_pixels / (image.width * image.height)))
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, resample=Image.Resampling.LANCZOS)


def build_privileged_teacher_images(
    image: Image.Image,
    context: dict[str, Any],
    *,
    padding_fraction: float = 0.15,
    max_crop_pixels: int = _PRIVILEGED_CROP_MAX_PIXELS,
    include_reconstruction: bool = True,
) -> list[Image.Image]:
    """Build teacher-only scene/target evidence, optionally including reconstruction."""
    if not 0.0 <= padding_fraction <= 1.0:
        raise ValueError("padding_fraction must be in [0, 1].")
    if max_crop_pixels <= 0:
        raise ValueError("max_crop_pixels must be positive.")
    scene = image.convert("RGB")
    target_mask = _align_mask_to_image(_unpack_mask(context.get("target_mask")), scene)
    reconstruction_mask = _align_mask_to_image(
        _unpack_mask(context.get("representative_mask")), scene
    )
    box = _union_crop_box(
        target_mask,
        reconstruction_mask if include_reconstruction else None,
        width=scene.width,
        height=scene.height,
        padding_fraction=padding_fraction,
    )
    evidence = [
        scene,
        _limit_crop_pixels(_masked_crop(scene, target_mask, box), max_crop_pixels),
    ]
    if include_reconstruction:
        evidence.append(
            _limit_crop_pixels(_masked_crop(scene, reconstruction_mask, box), max_crop_pixels)
        )
    return evidence


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
        "target_mask": _pack_mask(target_mask),
        "representative_mask": _pack_mask(representative_mask),
        "best_mask": _pack_mask(best_mask),
    }
    return context


def format_privileged_prompt(context: dict[str, object], *, mode: str) -> str:
    caption = context.get("student_caption", "")
    base = "You are improving an object caption with privileged visual evidence.\n"
    if mode == "groundedness":
        return base + (
            "Image 1 is the full scene. Image 2 isolates the intended object.\n"
            f"Student caption: {caption}\n"
            "Extract at most 8 literal substrings from the student caption that make visual claims. "
            "For each claim, classify it as supported, contradicted, unsupported, or uncertain. "
            "Use supported only when Image 2 or directly visible scene evidence proves it. "
            "Use unsupported when the caption adds an unverified detail. Use uncertain for ambiguity. "
            "A relation such as right, behind, background, or nearby is supported only when both the "
            "target and a named reference are visibly verifiable. Do not require background details.\n"
            "Output only JSON: {\"claims\":[{\"text\":\"literal caption substring\",\"type\":"
            "appearance|part|count|relation|context|other\",\"verdict\":"
            "supported|contradicted|unsupported|uncertain\"}],\"overall\":"
            "supported|partially_supported|unsupported\"}."
        )
    base += (
        "Image 1 is the full scene. Image 2 isolates the intended object. "
        "Image 3 isolates the object or area recovered from the student caption.\n"
        f"Student caption: {caption}\n"
    )
    if mode == "regenerate":
        return base + (
            "Write one concise factual caption for the isolated target only. "
            "Include only attributes directly visible in Image 2. Do not infer hidden parts, brands, "
            "materials, background, or spatial relations unless directly visible and necessary to identify "
            "the target. Use Image 3 only to avoid confusing the target with the recovered area. "
            "Output only the caption. Never mention images, crops, scores, coordinates, or this analysis."
        )
    if mode == "distill":
        return base + (
            "Evaluate the exact student caption trajectory token by token. Shift probability toward visible "
            "details supported by Image 2 and away from details supported only by Image 3."
        )
    if mode == "analysis":
        return base + (
            "Produce a concise training-only diagnosis as JSON with the keys failure_mode, missing_evidence, "
            "distractor_evidence, and correction_focus. Describe visible evidence in natural language. "
            "Do not output scores, coordinates, or a replacement caption."
        )
    raise ValueError(f"Unknown privileged prompt mode: {mode}")


def teacher_caption_is_safe(text: str) -> bool:
    lowered = (text or "").lower()
    return bool(lowered.strip()) and not any(term in lowered for term in PRIVILEGED_LEAKAGE_TERMS)
