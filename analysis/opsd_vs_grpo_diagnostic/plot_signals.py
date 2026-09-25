#!/usr/bin/env python3
"""Plot the validated signals extracted by ``extract_signals.py``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path(__file__).parent)
    args = p.parse_args()
    metadata_path = args.data_dir / "metadata.json"
    signals_path = args.data_dir / "signals.npz"
    if not metadata_path.exists() or not signals_path.exists():
        raise SystemExit("No validated signals.npz/metadata.json; refusing to draw a synthetic figure.")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("status") != "complete_local_logit_gradients":
        raise SystemExit(f"metadata status={metadata.get('status')!r}; figure is intentionally not produced.")
    d = np.load(signals_path, allow_pickle=False)
    for key in ("bin_grpo", "bin_opsd", "bin_mask", "local_x_grpo", "local_y_grpo",
                "local_x_opsd", "local_y_opsd"):
        if key not in d:
            raise SystemExit(f"signals.npz lacks {key}; refusing to guess.")

    g = np.asarray(d["bin_grpo"], dtype=float)
    o = np.asarray(d["bin_opsd"], dtype=float)
    mask = np.asarray(d["bin_mask"], dtype=bool)
    if g.shape != o.shape or mask.shape != g.shape or g.ndim != 2:
        raise SystemExit("Heatmap matrices have inconsistent shapes.")
    g[~mask] = np.nan
    o[~mask] = np.nan
    max_g = float(np.nanmax(np.abs(g))) if np.isfinite(g).any() else 0.0
    max_o = float(np.nanmax(np.abs(o))) if np.isfinite(o).any() else 0.0
    if max_g <= 0 or max_o <= 0:
        raise SystemExit("One heatmap has no non-zero finite signal; refusing to amplify it artificially.")
    gn, on = g / max_g, o / max_o

    cmap = LinearSegmentedColormap.from_list("opsd_diverging", ["#B6D4E9", "#FBFAF7", "#F2C3A7"])
    fig = plt.figure(figsize=(12.8, 4.2), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.82], wspace=0.24)
    ax0, ax1, ax2 = [fig.add_subplot(gs[0, i]) for i in range(3)]
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    for ax, arr, title in [(ax0, gn, "(a) GRPO"), (ax1, on, "(b) OPSD")]:
        im = ax.imshow(arr, aspect="auto", interpolation="none", cmap=cmap, norm=norm)
        ax.set_title(title, loc="left", fontsize=10, color="#44545F", pad=5)
        ax.set_xlabel("relative token position", fontsize=8, color="#44545F")
        ax.set_xticks([0, 16, 32, 48, 63], ["0", ".25", ".5", ".75", "1"])
        ax.set_ylabel("diagnostic trajectory" if ax is ax0 else "")
        ax.tick_params(labelsize=7, colors="#44545F", length=2)
        for spine in ax.spines.values(): spine.set_linewidth(0.6); spine.set_color("#44545F")
    cbar = fig.colorbar(im, ax=[ax0, ax1], fraction=0.025, pad=0.02, shrink=0.82)
    cbar.set_ticks([-1, 0, 1]); cbar.set_ticklabels(["−1", "0", "+1"])
    cbar.ax.tick_params(labelsize=7, colors="#44545F", length=2)
    cbar.set_label("signed logit gradient (each method / global max |s|)", fontsize=7, color="#44545F")

    xg, yg = np.asarray(d["local_x_grpo"], float), np.asarray(d["local_y_grpo"], float)
    xo, yo = np.asarray(d["local_x_opsd"], float), np.asarray(d["local_y_opsd"], float)
    if xg.size != yg.size or xo.size != yo.size or xg.size != xo.size:
        raise SystemExit("2D group arrays have inconsistent lengths.")
    ax2.axhline(0, color="#D7DDE0", lw=0.6); ax2.axvline(0, color="#D7DDE0", lw=0.6)
    ax2.scatter(xg, yg, s=11, color="#87ADC7", alpha=0.35, linewidths=0, label="GRPO")
    ax2.scatter(xo, yo, s=11, color="#B9DDD0", alpha=0.45, linewidths=0, label="OPSD")
    if xg.size:
        scale = max(np.ptp(np.r_[xg, xo]), np.ptp(np.r_[yg, yo]), 1e-6) * 0.035
    else:
        scale = 1e-3
    if xg.size:
        ax2.arrow(0, 0, float(np.mean(xg)), float(np.mean(yg)), color="#87ADC7", lw=1.2,
                  head_width=scale, length_includes_head=True)
        ax2.arrow(0, 0, float(np.mean(xo)), float(np.mean(yo)), color="#85B4A3", lw=1.2,
                  head_width=scale, length_includes_head=True)
    ax2.set_title("(c) measured local logit gradients", loc="left", fontsize=10, color="#44545F", pad=5)
    ax2.set_xlabel("A: teacher higher, mean s", fontsize=8, color="#44545F")
    ax2.set_ylabel("B: teacher lower, mean s", fontsize=8, color="#44545F")
    ax2.tick_params(labelsize=7, colors="#44545F", length=2)
    ax2.legend(frameon=False, fontsize=7, loc="best", handlelength=1.0)
    for spine in ax2.spines.values(): spine.set_linewidth(0.6); spine.set_color("#44545F")
    fig.savefig(args.data_dir / "figure_opsd_vs_grpo.png", dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(args.data_dir / "figure_opsd_vs_grpo.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(args.data_dir / "figure_opsd_vs_grpo.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
