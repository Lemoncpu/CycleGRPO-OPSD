"""Spatial-Evidence Credit Assignment (SECA) for OPSD auxiliary updates.

SECA keeps the pixel-IoU reward unchanged while reusing spatial evidence at
the two places where auxiliary supervision is applied: privileged JSD samples
and direct teacher-forcing mask tokens.
"""

from __future__ import annotations

from typing import Mapping, Optional

import torch


def spatial_evidence_weight(
    context: Optional[Mapping[str, object]],
    *,
    min_weight: float = 0.5,
    max_weight: float = 1.0,
    false_positive_penalty: float = 2.0,
) -> float:
    """Convert IoU and reconstruction-only area into an auxiliary-loss weight."""
    context = context or {}
    iou = float(context.get("iou_mean", context.get("R_Ci", 0.0)) or 0.0)
    target = context.get("target_summary") or {}
    reconstruction_only = context.get("reconstruction_only_summary") or {}
    target_area = float(target.get("area", 0.0) or 0.0)
    false_positive_area = float(reconstruction_only.get("area", 0.0) or 0.0)
    fp_ratio = false_positive_area / max(target_area, 1.0)
    quality = max(0.0, min(1.0, iou * (1.0 - false_positive_penalty * fp_ratio)))
    return float(min_weight + (max_weight - min_weight) * quality)


def hierarchical_mask_token_weights(
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    mask_token_start: int = 151670,
    mask_token_end: int = 152181,
    coarse_weight: float = 1.5,
    fine_weight: float = 1.0,
) -> torch.Tensor:
    """Assign separate CE credit to the first two SAMTok code depths."""
    if response_ids.shape != response_mask.shape:
        raise ValueError(
            "response_ids and response_mask must have identical shapes, got "
            f"{response_ids.shape} and {response_mask.shape}."
        )
    weights = torch.ones_like(response_ids, dtype=torch.float32)
    active = response_mask.to(dtype=torch.bool)
    for row in range(response_ids.shape[0]):
        positions = torch.where(
            active[row]
            & (response_ids[row] >= mask_token_start)
            & (response_ids[row] <= mask_token_end)
        )[0]
        if positions.numel() >= 2:
            weights[row, positions[0]] = float(coarse_weight)
            weights[row, positions[1]] = float(fine_weight)
    return weights
