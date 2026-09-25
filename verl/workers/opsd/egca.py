"""Dynamic Evidence-Guided Credit Assignment for sampled mask groups.

The decoder exposes a legal SAMTok action only after both depth codes are
present.  EGCA therefore uses two-code Shapley counterfactuals: the coarse and
fine code receive separate, signed marginal credits while the complete group
credit remains conserved.  The functions in this module are deliberately
framework-light so they can be unit-tested without a model or a Ray worker.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch


def mask_group_text(
    coarse_code: int,
    fine_code: int,
    *,
    codebook_size: int = 256,
) -> str:
    """Format local coarse/fine codes as one legal depth-2 SAMTok group."""
    if not 0 <= int(coarse_code) < codebook_size:
        raise ValueError(f"coarse code must be in [0,{codebook_size}), got {coarse_code}")
    if not 0 <= int(fine_code) < codebook_size:
        raise ValueError(f"fine code must be in [0,{codebook_size}), got {fine_code}")
    return (
        f"<|mt_start|><|mt_{int(coarse_code):04d}|>"
        f"<|mt_{int(codebook_size + fine_code):04d}|><|mt_end|>"
    )


def extract_code_positions(
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    codebook_size: int = 256,
    codebook_depth: int = 2,
) -> list[tuple[list[int], list[int]]]:
    """Return ``[(local_codes, token_positions), ...]`` for one response.

    SAMTok code tokens are contiguous inside each ``mt_start``/``mt_end``
    group.  Marker IDs are intentionally excluded; this makes the returned
    positions point exactly at the coarse/fine logits that receive EGCA credit.
    Incomplete groups are ignored rather than silently assigned partial credit.
    """
    if response_ids.ndim != 1 or response_mask.ndim != 1:
        raise ValueError("response_ids and response_mask must be one-dimensional")
    if response_ids.shape != response_mask.shape:
        raise ValueError("response_ids and response_mask must have identical shapes")
    code_lo = int(151670)
    code_hi = code_lo + int(codebook_size * codebook_depth) - 1
    active = response_mask.to(dtype=torch.bool)
    groups: list[tuple[list[int], list[int]]] = []
    current: list[tuple[int, int]] = []
    for position, token in enumerate(response_ids.tolist()):
        if not bool(active[position]):
            break
        token = int(token)
        if code_lo <= token <= code_hi:
            current.append((token, position))
            continue
        if current:
            if len(current) == codebook_depth:
                local_codes = [
                    token_id - code_lo - depth * codebook_size
                    for depth, (token_id, _) in enumerate(current)
                ]
                if all(0 <= value < codebook_size for value in local_codes):
                    groups.append((local_codes, [pos for _, pos in current]))
            current = []
    if current and len(current) == codebook_depth:
        local_codes = [
            token_id - code_lo - depth * codebook_size
            for depth, (token_id, _) in enumerate(current)
        ]
        if all(0 <= value < codebook_size for value in local_codes):
            groups.append((local_codes, [pos for _, pos in current]))
    return groups


def shapley_depth_credit(
    value_empty: float,
    value_coarse: float,
    value_fine: float,
    value_both: float,
    *,
    credit_clip: Optional[float] = 1.0,
) -> tuple[float, float]:
    """Compute exact two-player Shapley coarse/fine credit.

    The efficiency identity ``coarse + fine = both - empty`` holds before the
    optional independent safety clipping.  Caller-side evidence projection is
    expected to happen before clipping when strict conservation is required.
    """
    coarse = 0.5 * ((float(value_coarse) - float(value_empty)) + (float(value_both) - float(value_fine)))
    fine = 0.5 * ((float(value_fine) - float(value_empty)) + (float(value_both) - float(value_coarse)))
    if credit_clip is not None:
        limit = abs(float(credit_clip))
        coarse = max(-limit, min(limit, coarse))
        fine = max(-limit, min(limit, fine))
    return coarse, fine


def evidence_weight_from_masks(
    target_mask: Optional[torch.Tensor],
    prediction: Optional[torch.Tensor],
    *,
    min_weight: float = 0.5,
    max_weight: float = 1.0,
    false_positive_penalty: float = 2.0,
) -> float:
    """Convert decoded target/reconstruction evidence into a bounded gate."""
    if target_mask is None or prediction is None:
        return float(min_weight)
    target = target_mask.to(dtype=torch.bool)
    predicted = prediction.to(dtype=torch.bool)
    if target.shape != predicted.shape:
        predicted = torch.nn.functional.interpolate(
            predicted[None, None].float(), size=target.shape, mode="nearest"
        )[0, 0].bool()
    intersection = torch.logical_and(target, predicted).sum().item()
    union = torch.logical_or(target, predicted).sum().item()
    iou = float(intersection / union) if union else 0.0
    false_positive = torch.logical_and(predicted, torch.logical_not(target)).sum().item()
    target_area = max(float(target.sum().item()), 1.0)
    fp_ratio = float(false_positive) / target_area
    quality = max(0.0, min(1.0, iou * (1.0 - float(false_positive_penalty) * fp_ratio)))
    return float(min_weight + (max_weight - min_weight) * quality)


def combine_evidence_and_shapley(
    coarse_credit: float,
    fine_credit: float,
    evidence_weight: float,
    *,
    credit_clip: Optional[float] = 1.0,
) -> tuple[float, float]:
    """Apply one evidence gate to both depth credits and optionally clip them."""
    coarse = float(coarse_credit) * float(evidence_weight)
    fine = float(fine_credit) * float(evidence_weight)
    if credit_clip is not None:
        limit = abs(float(credit_clip))
        coarse = max(-limit, min(limit, coarse))
        fine = max(-limit, min(limit, fine))
    return coarse, fine


def contrastive_token_credit(
    credit: torch.Tensor,
    active_tokens: torch.Tensor,
    response_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Remove the per-response common-mode component of EGCA credit.

    Shapley values describe the change in the completed mask trajectory.  If
    those signed values are added directly to PPO advantages, their common
    (all-mask-tokens) component becomes an extra trajectory reward.  A large
    negative common component can therefore make the actor prefer refusing to
    emit a mask, even though refusal rollouts never receive EGCA credit.  The
    GRPO trajectory advantage already carries the completed-mask reward, so
    actor-side EGCA should only express *relative* coarse/fine credit.

    The returned tensor is zero outside the legal code-token positions and has
    zero sum over those positions for every response.  This preserves the
    original trajectory-level objective while retaining token-level credit
    differences.  ``active_tokens`` is separate from the numeric credit so a
    legitimately zero Shapley value is still treated as a code position.
    """
    if credit.ndim != 2 or active_tokens.shape != credit.shape:
        raise ValueError("credit and active_tokens must be [batch, response] tensors of equal shape")
    active = active_tokens.to(dtype=torch.bool)
    if response_mask is not None:
        if response_mask.shape != credit.shape:
            raise ValueError("response_mask must have the same shape as credit")
        active = active & response_mask.to(dtype=torch.bool)
    active_float = active.to(dtype=credit.dtype)
    count = active_float.sum(dim=-1, keepdim=True)
    total = (credit * active_float).sum(dim=-1, keepdim=True)
    mean = torch.where(count > 0, total / count.clamp_min(1.0), torch.zeros_like(total))
    return torch.where(active, credit - mean, torch.zeros_like(credit))


def nonnegative_mask_group_token_weights(
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    sample_weights: Sequence[float],
    coarse_scores: Sequence[float],
    fine_scores: Sequence[float],
    codebook_size: int = 256,
    codebook_depth: int = 2,
    token_credit_scale: float = 0.5,
    token_weight_max: float = 2.0,
) -> torch.Tensor:
    """Create non-negative teacher-forcing weights for complete mask groups.

    The rollout-derived evidence is a *weight* for a validated target sequence,
    never a signed policy advantage.  Every complete group receives the same
    evidence base on its span; positive coarse/fine Shapley magnitudes may then
    increase the corresponding code-token weight.  Negative Shapley values are
    clipped to zero and therefore cannot create a refusal incentive.  Marker
    tokens are included whenever they are adjacent to the two code tokens.
    """
    if response_ids.ndim != 2 or response_mask.shape != response_ids.shape:
        raise ValueError("response_ids and response_mask must be [batch, tokens] tensors")
    batch_size = response_ids.shape[0]
    if not (
        len(sample_weights) == len(coarse_scores) == len(fine_scores) == batch_size
    ):
        raise ValueError("dynamic EGCA weight arrays must match response batch size")
    if token_credit_scale < 0.0 or token_weight_max <= 0.0:
        raise ValueError("token credit scale/max must be non-negative/positive")

    weights = torch.ones_like(response_ids, dtype=torch.float32)
    for row in range(batch_size):
        groups = extract_code_positions(
            response_ids[row].detach().cpu(),
            response_mask[row].detach().cpu(),
            codebook_size=codebook_size,
            codebook_depth=codebook_depth,
        )
        if not groups:
            continue
        base = max(0.0, min(float(sample_weights[row]), float(token_weight_max)))
        coarse = max(0.0, float(coarse_scores[row]))
        fine = max(0.0, float(fine_scores[row]))
        for group_index, (_, positions) in enumerate(groups):
            # The target is normally one group. If a multi-group target is
            # present, distribute the same validated evidence over all groups.
            start = max(0, positions[0] - 1)
            end = min(response_ids.shape[1], positions[-1] + 2)
            weights[row, start:end] = base
            coarse_weight = min(
                token_weight_max,
                base * (1.0 + token_credit_scale * coarse),
            )
            fine_weight = min(
                token_weight_max,
                base * (1.0 + token_credit_scale * fine),
            )
            weights[row, positions[0]] = coarse_weight
            if len(positions) > 1:
                weights[row, positions[1]] = fine_weight
    return weights


def egca_opd_weight(
    context: Optional[Mapping[str, object]],
    *,
    min_weight: float = 0.5,
    max_weight: float = 1.0,
    false_positive_penalty: float = 2.0,
) -> float:
    """Evidence gate for the OPD/JSD path.

    This mirrors the spatial evidence used by code credit, but is kept in the
    EGCA module so enabling EGCA consistently gates both mask credit and the
    privileged on-policy distillation sample weight.
    """
    context = context or {}
    iou = float(context.get("iou_mean", context.get("R_Ci", 0.0)) or 0.0)
    target = context.get("target_summary") or {}
    reconstruction_only = context.get("reconstruction_only_summary") or {}
    target_area = float(target.get("area", 0.0) or 0.0)
    false_positive_area = float(reconstruction_only.get("area", 0.0) or 0.0)
    fp_ratio = false_positive_area / max(target_area, 1.0)
    quality = max(0.0, min(1.0, iou * (1.0 - false_positive_penalty * fp_ratio)))
    return float(min_weight + (max_weight - min_weight) * quality)
