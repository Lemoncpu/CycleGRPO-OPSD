#!/usr/bin/env python3
"""Render the enhanced two-panel real-log supervision comparison.

Panel (a) keeps the GRPO row-wise scalar signal; panel (b) keeps the real
OPSD within-diagnosis IoU residuals.  The discrete multi-level diverging
colour map and short annotations make the supervision granularity visible
without adding case labels or decorative text.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap


BASE = Path(__file__).parent


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    a = np.asarray(data["grpo_heatmap"], dtype=np.float32)
    b = np.asarray(data["opsd_heatmap"], dtype=np.float32)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 6:
        raise ValueError(f"expected matching [stages, 6] heatmaps, got {a.shape}, {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("heatmaps contain non-finite values")
    return a, b


def render(grpo: np.ndarray, opsd: np.ndarray, output: Path, dpi: int) -> None:
    # Sample a continuous blue-white-apricot map into many explicit levels.
    base = LinearSegmentedColormap.from_list(
        "opsd_div", ["#75A8C8", "#B6D4E9", "#FBFAF7", "#F2C3A7", "#D8895D"]
    )
    n_levels = 17
    cmap = ListedColormap(base(np.linspace(0.0, 1.0, n_levels - 1)))
    vmax = float(np.percentile(np.abs(np.concatenate([grpo.ravel(), opsd.ravel()])), 99))
    vmax = max(vmax, 1e-6)
    boundaries = np.linspace(-vmax, vmax, n_levels)
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    fig = plt.figure(figsize=(10.8, 4.25), facecolor="white")
    gs = fig.add_gridspec(1, 2, left=0.075, right=0.985, bottom=0.26, top=0.83,
                          wspace=0.24, width_ratios=[1.0, 1.0])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    panels = [
        (axes[0], grpo, r"(a) GRPO  |  shared credit", r"$k_1 = k_2 = \cdots = k_6$",
         "1 outcome → 6 rollouts"),
        (axes[1], opsd, r"(b) OPSD  |  evidence-resolved", r"$k_1 \ne k_2 \ne \cdots \ne k_6$",
         "6 IoUs → 6 signals"),
    ]
    for idx, (ax, values, title, symbol, note) in enumerate(panels):
        ax.imshow(values, cmap=cmap, norm=norm, aspect="auto", interpolation="none")
        ax.set_title(title, loc="left", fontsize=10.5, color="#44545F", pad=4, fontweight="semibold")
        ax.text(0.5, 1.045, symbol, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9.0, color="#5E7582")
        ax.text(0.5, -0.205, note, transform=ax.transAxes, ha="center", va="top",
                fontsize=8.1, color="#5E7582")
        ax.set_xlabel("localization rollout  $k$", fontsize=8.5, color="#44545F", labelpad=2)
        ax.set_xticks(range(6), ["1", "2", "3", "4", "5", "6"])
        ax.set_ylabel("training stage" if idx == 0 else "", fontsize=8.5, color="#44545F", labelpad=2)
        ax.set_yticks(range(values.shape[0]), ["1", "", "", "", "", "6", "", "", "", "", "", "12"])
        ax.tick_params(labelsize=7.5, colors="#44545F", length=2, pad=2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.65)
            spine.set_color("#44545F")

    cax = fig.add_axes([0.235, 0.105, 0.53, 0.026])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", boundaries=boundaries)
    cb.set_ticks([-vmax, -vmax / 2, 0, vmax / 2, vmax])
    cb.set_ticklabels(["−", "", "0", "", "+"])
    cb.set_label("signed supervision signal (centered)", fontsize=8, color="#44545F", labelpad=2)
    cb.ax.tick_params(labelsize=7, colors="#44545F", length=2, pad=1)
    cb.outline.set_linewidth(0.45)
    cb.outline.set_edgecolor("#AAB6BC")

    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=BASE / "signals.npz")
    ap.add_argument("--out-dir", type=Path, default=BASE)
    args = ap.parse_args()
    grpo, opsd = load_data(args.data)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in [("png", 400), ("pdf", 400), ("svg", 400)]:
        render(grpo, opsd, args.out_dir / f"figure_opsd_vs_grpo_ab.{suffix}", dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
