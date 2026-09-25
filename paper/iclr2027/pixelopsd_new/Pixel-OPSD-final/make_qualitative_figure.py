#!/usr/bin/env python3
"""Render the matched qualitative comparison used by Figure 4.

The two prediction files must come from the same GroundingSuite sample manifest
and prompt protocol.  The defaults point to the audited local evaluation runs;
all paths can be overridden for a new checkpoint pair.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

NAVY = "#18324A"
BLUE = "#2F6F9F"
ORANGE = "#D9822B"
TEAL = "#2B9A93"
RED = "#C95B55"
GRID = "#D8E2E5"

DEFAULT_REPO = Path("/volume/ybo/xyc/CycleGRPO-OPSD")
DEFAULT_OURS = DEFAULT_REPO / (
    "logs/cyclegrpo20k_withnt_direct30k_notarget10k_dlcqa10k_"
    "bs112_directgrpo_ce005_pixel_empty_positivepenalty1/checkpoints/"
    "evaluation/step_178_response256_cpu_reassembled"
)
DEFAULT_BASE = DEFAULT_REPO / "logs/evaluation/Qwen3-VL-4B-SAMTok"
DEFAULT_CYCLE = DEFAULT_REPO / "logs/evaluation/CycleGRPO-4B"
DEFAULT_CASES = (1525, 1947, 2050)


def read_jsonl(path: Path) -> dict[int, dict]:
    return {int(row["idx"]): row for row in (json.loads(line) for line in path.open() if line.strip())}


def decode(entry: dict) -> np.ndarray:
    return mask_utils.decode(entry["predicted_segmentation"]).astype(bool)


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray((mask.astype(np.uint8) * 255))
    return np.asarray(image.resize(size, Image.Resampling.NEAREST)) > 127


def draw_prediction(
    ax,
    image: np.ndarray,
    mask: np.ndarray,
    color: str,
    iou: float | None = None,
    linestyle: str = "-",
) -> None:
    ax.imshow(image, interpolation="none")
    # A light fill carries the predicted area while the thin edge keeps object
    # boundaries legible even for masks with many small contours.
    if mask.any():
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[..., :3] = to_rgb(color)
        overlay[..., 3] = mask.astype(np.float32) * 0.22
        ax.imshow(overlay, interpolation="none")
        ax.contour(mask, levels=[0.5], colors=[color], linewidths=0.65, alpha=0.95, linestyles=linestyle)
    if iou is not None:
        ax.text(
            0.03,
            0.04,
            f"IoU {iou:.2f}",
            transform=ax.transAxes,
            fontsize=5.8,
            color="white",
            va="bottom",
            ha="left",
            bbox={"facecolor": color, "edgecolor": "none", "alpha": 0.78, "pad": 1.1},
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("auto")
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(0.6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours-dir", type=Path, default=DEFAULT_OURS)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--cycle-dir", type=Path, default=DEFAULT_CYCLE)
    parser.add_argument("--image-root", type=Path, default=Path("/volume/ybo/xyc/coco2017"))
    parser.add_argument("--output", type=Path, default=Path("figures/qualitative_comparison.pdf"))
    parser.add_argument("--cases", type=int, nargs="+", default=DEFAULT_CASES)
    args = parser.parse_args()

    ours_metrics = json.loads((args.ours_dir / "groundingsuite_official_long_prompt_metrics.json").read_text())
    base_metrics = json.loads((args.baseline_dir / "groundingsuite_official_training_long_prompt_metrics.json").read_text())
    cycle_metrics = json.loads((args.cycle_dir / "groundingsuite_official_training_long_prompt_metrics.json").read_text())
    ours = read_jsonl(args.ours_dir / "groundingsuite_official_long_prompt_pred.jsonl")
    base = read_jsonl(args.baseline_dir / "groundingsuite_official_training_long_prompt_pred.jsonl")
    cycle = read_jsonl(args.cycle_dir / "groundingsuite_official_training_long_prompt_pred.jsonl")
    ours_metrics = {int(row["idx"]): row for row in ours_metrics["results"]}
    base_metrics = {int(row["idx"]): row for row in base_metrics["results"]}
    cycle_metrics = {int(row["idx"]): row for row in cycle_metrics["results"]}

    nrows = len(args.cases) + 1
    figure_height = 0.95 + 0.70 * len(args.cases)
    fig = plt.figure(figsize=(7.1, figure_height), facecolor="white")
    grid = fig.add_gridspec(
        nrows,
        5,
        width_ratios=(2.35, 1, 1, 1, 1),
        height_ratios=(0.22, *([1] * len(args.cases))),
        wspace=0.06,
        hspace=0.14,
    )
    for col, title, color in [(1, "GT", RED), (2, "SAMTok", BLUE), (3, "CycleGRPO", ORANGE), (4, "Pixel-OPSD", TEAL)]:
        ax = fig.add_subplot(grid[0, col])
        ax.axis("off")
        ax.text(0.5, 0.2, title, ha="center", va="center", fontsize=8.2, color=color, weight="bold")

    for row, idx in enumerate(args.cases, start=1):
        ours_row = ours[idx]
        base_row = base[idx]
        cycle_row = cycle[idx]
        metric = ours_metrics[idx]
        base_metric = base_metrics[idx]
        cycle_metric = cycle_metrics[idx]
        image = np.asarray(Image.open(args.image_root / ours_row["image_path"]).convert("RGB"))
        height, width = image.shape[:2]
        target = resize_mask(mask_utils.decode(metric["gt_segmentation"]).astype(bool), (width, height))
        baseline = resize_mask(decode(base_row), (width, height))
        cycle_prediction = resize_mask(decode(cycle_row), (width, height))
        prediction = resize_mask(decode(ours_row), (width, height))

        caption_ax = fig.add_subplot(grid[row, 0])
        caption_ax.axis("off")
        caption_ax.text(
            0.5,
            0.5,
            textwrap.fill(metric["caption"], width=38),
            color=NAVY,
            fontsize=6.5,
            weight=500,
            va="center",
            ha="center",
            linespacing=1.05,
        )

        panels = [
            (1, target, RED, None, "-"),
            (2, baseline, BLUE, base_metric["iou"], "-"),
            (3, cycle_prediction, ORANGE, cycle_metric["iou"], "-"),
            (4, prediction, TEAL, metric["iou"], "-"),
        ]
        for col, mask, color, iou, linestyle in panels:
            ax = fig.add_subplot(grid[row, col])
            draw_prediction(ax, image, mask, color, iou, linestyle)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the source photographs above their native display resolution in the
    # PDF; ``none`` interpolation above preserves pixel detail when zoomed.
    fig.savefig(args.output, format="pdf", dpi=1200, bbox_inches="tight", pad_inches=0.01, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
