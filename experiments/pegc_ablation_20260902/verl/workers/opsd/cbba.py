"""Confidence-Balanced Branch Allocation (CBBA) helpers."""

from __future__ import annotations

from typing import Iterable


def cbba_stream_scales(
    scores: Iterable[float],
    *,
    target_confidence: float = 0.70,
    gain: float = 1.0,
    min_cycle_scale: float = 0.75,
    max_cycle_scale: float = 1.25,
    min_supervised_scale: float = 0.75,
    max_supervised_scale: float = 1.50,
) -> tuple[float, float, float]:
    """Return cycle scale, supervised scale, and mean cycle confidence."""
    if not 0.0 <= target_confidence <= 1.0 or gain < 0.0:
        raise ValueError("invalid CBBA confidence parameters")
    if not 0.0 < min_cycle_scale <= max_cycle_scale:
        raise ValueError("invalid CBBA cycle scale bounds")
    if not 0.0 < min_supervised_scale <= max_supervised_scale:
        raise ValueError("invalid CBBA supervised scale bounds")
    finite_scores = []
    for score in scores:
        value = float(score)
        if value == value and value not in (float("inf"), float("-inf")):
            finite_scores.append(value)
    if not finite_scores:
        return 1.0, 1.0, 0.0
    mean_confidence = max(0.0, min(1.0, sum(finite_scores) / len(finite_scores)))
    delta = gain * (mean_confidence - target_confidence)
    cycle_scale = max(min_cycle_scale, min(max_cycle_scale, 1.0 + delta))
    supervised_scale = max(min_supervised_scale, min(max_supervised_scale, 1.0 - delta))
    return cycle_scale, supervised_scale, mean_confidence
