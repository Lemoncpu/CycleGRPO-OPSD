"""Shared mask metrics for offline gRefCOCO/GRES aggregate reports."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GresMetricAccumulator:
    """Accumulate the official GRES empty-target and mask-IoU semantics."""

    intersection: int = 0
    union: int = 0
    target_ious: list[float] = field(default_factory=list)
    all_ious: list[float] = field(default_factory=list)
    no_target_correct: int = 0
    no_target_total: int = 0
    target_correct: int = 0
    target_total: int = 0

    def add(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> None:
        pred_mask = np.asarray(pred_mask, dtype=bool)
        gt_mask = np.asarray(gt_mask, dtype=bool)
        if pred_mask.shape != gt_mask.shape:
            raise ValueError(f"Prediction/GT mask shape mismatch: {pred_mask.shape} vs {gt_mask.shape}")

        inter = int(np.logical_and(pred_mask, gt_mask).sum())
        current_union = int(np.logical_or(pred_mask, gt_mask).sum())
        is_target = bool(gt_mask.any())
        is_predicted = bool(pred_mask.any())
        if is_target:
            self.target_total += 1
            self.target_correct += int(is_predicted)
            self.intersection += inter
            self.union += current_union
            iou = inter / current_union if current_union else 0.0
            self.target_ious.append(iou)
        else:
            self.no_target_total += 1
            self.no_target_correct += int(not is_predicted)
            iou = 1.0 if not is_predicted else 0.0
            # Match the official protocol: a false-positive empty-target mask
            # increases cIoU union; a correct empty prediction adds no pixels.
            if is_predicted:
                self.union += current_union
        self.all_ious.append(iou)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "samples": len(self.all_ious),
            "no_target_samples": self.no_target_total,
            "target_samples": self.target_total,
            "N_acc": 100.0 * self.no_target_correct / self.no_target_total if self.no_target_total else 0.0,
            "T_acc": 100.0 * self.target_correct / self.target_total if self.target_total else 0.0,
            "gIoU": 100.0 * sum(self.all_ious) / len(self.all_ious) if self.all_ious else 0.0,
            "cIoU": 100.0 * self.intersection / self.union if self.union else 0.0,
            "mIoU_target": 100.0 * sum(self.target_ious) / len(self.target_ious) if self.target_ious else 0.0,
        }


def target_area_bucket(gt_mask: np.ndarray) -> str:
    """Classify a non-empty GT mask by its fraction of image pixels."""

    gt_mask = np.asarray(gt_mask, dtype=bool)
    if not gt_mask.any():
        raise ValueError("Target area buckets are defined only for non-empty GT masks.")
    area_ratio = float(gt_mask.mean())
    if area_ratio < 0.05:
        return "small_lt_5pct"
    if area_ratio < 0.25:
        return "medium_5_to_25pct"
    return "large_ge_25pct"
