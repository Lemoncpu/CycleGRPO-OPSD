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


def multi_instance_count_bucket(annotation_instance_count: int) -> str:
    """Bin exact positive annotation counts for multi-instance GRES references."""

    if annotation_instance_count < 2:
        raise ValueError(
            "Multi-instance count buckets require at least two positive annotation instances."
        )
    if annotation_instance_count == 2:
        return "2_instances"
    if annotation_instance_count == 3:
        return "3_instances"
    return "4plus_instances"


def two_instance_area_balance_bucket(small_to_large_area_ratio: float) -> str:
    """Bin the ratio of smaller to larger constituent instance area."""

    if not 0.0 < small_to_large_area_ratio <= 1.0:
        raise ValueError(f"Invalid two-instance area ratio: {small_to_large_area_ratio}")
    if small_to_large_area_ratio < 0.2:
        return "high_imbalance_lt_0.2"
    if small_to_large_area_ratio < 0.5:
        return "moderate_0.2_to_0.5"
    return "balanced_ge_0.5"


def two_instance_center_distance_bucket(center_distance_diag: float) -> str:
    """Bin centroid distance normalized by the image diagonal."""

    if not 0.0 <= center_distance_diag <= 1.0:
        raise ValueError(f"Invalid normalized two-instance center distance: {center_distance_diag}")
    if center_distance_diag < 0.25:
        return "near_lt_0.25diag"
    if center_distance_diag < 0.5:
        return "mid_0.25_to_0.5diag"
    return "far_ge_0.5diag"


@dataclass(frozen=True)
class TwoInstanceDiagnosticSample:
    """Per-case constituent coverage and geometry for a two-instance target."""

    small_member_coverage: float
    large_member_coverage: float
    small_member_hit: bool
    large_member_hit: bool
    small_to_large_area_ratio: float
    center_distance_diag: float


def two_instance_diagnostic_sample(
    pred_mask: np.ndarray,
    member_masks: list[np.ndarray],
    *,
    member_recall_threshold: float,
) -> TwoInstanceDiagnosticSample:
    """Measure whether a prediction covers each original annotation instance."""

    if not 0.0 < member_recall_threshold <= 1.0:
        raise ValueError(f"member_recall_threshold must be in (0, 1], got {member_recall_threshold}")
    if len(member_masks) != 2:
        raise ValueError(f"Expected exactly two member masks, got {len(member_masks)}")
    pred_mask = np.asarray(pred_mask, dtype=bool)
    members = [np.asarray(mask, dtype=bool) for mask in member_masks]
    if any(mask.shape != pred_mask.shape for mask in members):
        raise ValueError("Prediction and two-instance member masks must have identical shapes.")
    areas = [int(mask.sum()) for mask in members]
    if min(areas) <= 0:
        raise ValueError("Two-instance member masks must both be non-empty.")

    small_index, large_index = sorted(range(2), key=lambda index: areas[index])
    small_mask, large_mask = members[small_index], members[large_index]
    small_coverage = float(np.logical_and(pred_mask, small_mask).sum() / areas[small_index])
    large_coverage = float(np.logical_and(pred_mask, large_mask).sum() / areas[large_index])
    centers = [np.argwhere(mask).mean(axis=0) for mask in members]
    height, width = pred_mask.shape
    image_diagonal = float(np.hypot(height, width))
    center_distance = float(np.linalg.norm(centers[0] - centers[1]) / image_diagonal)
    return TwoInstanceDiagnosticSample(
        small_member_coverage=small_coverage,
        large_member_coverage=large_coverage,
        small_member_hit=small_coverage >= member_recall_threshold,
        large_member_hit=large_coverage >= member_recall_threshold,
        small_to_large_area_ratio=areas[small_index] / areas[large_index],
        center_distance_diag=center_distance,
    )


@dataclass
class TwoInstanceDiagnosticAccumulator:
    """Aggregate union-mask metrics with constituent recall and geometry statistics."""

    metrics: GresMetricAccumulator = field(default_factory=GresMetricAccumulator)
    member_recall_threshold: float = 0.5
    small_member_coverage_sum: float = 0.0
    large_member_coverage_sum: float = 0.0
    small_member_hit_count: int = 0
    large_member_hit_count: int = 0
    both_member_hit_count: int = 0
    one_member_hit_count: int = 0
    no_member_hit_count: int = 0
    area_ratio_sum: float = 0.0
    center_distance_sum: float = 0.0

    def add(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        sample: TwoInstanceDiagnosticSample,
    ) -> None:
        self.metrics.add(pred_mask, gt_mask)
        self.small_member_coverage_sum += sample.small_member_coverage
        self.large_member_coverage_sum += sample.large_member_coverage
        self.small_member_hit_count += int(sample.small_member_hit)
        self.large_member_hit_count += int(sample.large_member_hit)
        hit_count = int(sample.small_member_hit) + int(sample.large_member_hit)
        self.both_member_hit_count += int(hit_count == 2)
        self.one_member_hit_count += int(hit_count == 1)
        self.no_member_hit_count += int(hit_count == 0)
        self.area_ratio_sum += sample.small_to_large_area_ratio
        self.center_distance_sum += sample.center_distance_diag

    def as_dict(self) -> dict[str, float | int]:
        result = self.metrics.as_dict()
        samples = result["samples"]
        if not samples:
            return result
        result.update({
            "member_recall_threshold": self.member_recall_threshold,
            "mean_small_member_coverage_pct": 100.0 * self.small_member_coverage_sum / samples,
            "mean_large_member_coverage_pct": 100.0 * self.large_member_coverage_sum / samples,
            "small_member_recall_at_threshold": 100.0 * self.small_member_hit_count / samples,
            "large_member_recall_at_threshold": 100.0 * self.large_member_hit_count / samples,
            "both_members_recall_at_threshold": 100.0 * self.both_member_hit_count / samples,
            "one_member_only_recall_at_threshold": 100.0 * self.one_member_hit_count / samples,
            "no_members_recall_at_threshold": 100.0 * self.no_member_hit_count / samples,
            "mean_small_to_large_area_ratio": self.area_ratio_sum / samples,
            "mean_center_distance_diag": self.center_distance_sum / samples,
        })
        return result
