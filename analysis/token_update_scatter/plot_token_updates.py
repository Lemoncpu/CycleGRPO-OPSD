#!/usr/bin/env python3
"""Render the registered real checkpoint token-update diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


COLORS = {"grpo": "#A9CDE4", "opsd": "#EDBC9E"}
EDGES = {"grpo": "#709EBB", "opsd": "#C18E6D"}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def finite(data: np.ndarray) -> np.ndarray:
    return np.asarray(data, dtype=np.float64).reshape(-1)


def save(fig: plt.Figure, path: Path) -> None:
    for ext in ("pdf", "svg", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    a = args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(a.input)
    valid = np.asarray(d["valid"], dtype=bool)
    x_all = np.asarray(d["contribution"], dtype=float)
    ys = {"grpo": np.asarray(d["grpo"], dtype=float), "opsd": np.asarray(d["opsd"], dtype=float)}
    rel_all = np.asarray(d["reliability"], dtype=float)
    finite_mask = valid & np.isfinite(x_all) & np.isfinite(rel_all)
    for k in ys:
        finite_mask &= np.isfinite(ys[k])
    x = x_all[finite_mask]
    rel = rel_all[finite_mask]
    yvals = {k: ys[k][finite_mask] for k in ys}
    if x.size == 0:
        raise RuntimeError("no finite token rows")
    xlim = (float(np.min(x)), float(np.max(x)))
    ylim = (float(min(np.min(y) for y in yvals.values())), float(max(np.max(y) for y in yvals.values())))
    dx = max((xlim[1] - xlim[0]) * 0.06, 1e-3)
    dy = max((ylim[1] - ylim[0]) * 0.08, 1e-3)
    xlim = (xlim[0] - dx, xlim[1] + dx)
    ylim = (ylim[0] - dy, ylim[1] + dy)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.labelcolor": "#46515A",
        "text.color": "#34404A", "xtick.color": "#65717A", "ytick.color": "#65717A",
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axarr = plt.subplots(1, 2, figsize=(10.4, 4.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.095, right=.98, bottom=.22, top=.86, wspace=.12)
    for ax, key, title in zip(axarr, ("grpo", "opsd"), ("GRPO", "Pixel-OPSD")):
        ax.add_patch(Rectangle((0, 0), max(xlim[1], 0) - 0, max(ylim[1], 0) - 0,
                               facecolor="#B9DDD0", alpha=.15, lw=0, zorder=0))
        ax.add_patch(Rectangle((min(xlim[0], 0), min(ylim[0], 0)), 0 - min(xlim[0], 0),
                               0 - min(ylim[0], 0), facecolor="#B9DDD0", alpha=.15, lw=0, zorder=0))
        ax.axhline(0, color="#9EABB2", lw=.8, zorder=1)
        ax.axvline(0, color="#9EABB2", lw=.8, zorder=1)
        ax.scatter(x, yvals[key], s=25, c=COLORS[key], edgecolors="none", alpha=.64, zorder=2)
        ax.set_title(title, fontsize=13, pad=12, fontweight="medium")
        for spine in ax.spines.values():
            spine.set_color("#B3BDC3"); spine.set_linewidth(.75)
        ax.tick_params(length=3, width=.6)
    axarr[0].set_ylabel(r"Token update  $\Delta\log p$", labelpad=10)
    fig.supxlabel("Teacher alignment  $\log p_{teacher}-\log p_{base}$", y=.115, fontsize=11)
    fig.text(.5, .038, "SAMTok mask-code tokens · held-out direct RefCOCO diagnostic · checkpoint deltas",
             ha="center", fontsize=8, color="#65717A")
    save(fig, out / "token_update_2d")

    zlim = (float(min(np.min(y) for y in yvals.values())), float(max(np.max(y) for y in yvals.values())))
    fig = plt.figure(figsize=(11.5, 5.15))
    fig.subplots_adjust(left=.015, right=.98, bottom=.15, top=.90, wspace=.02)
    for i, (key, title) in enumerate(zip(("grpo", "opsd"), ("GRPO", "Pixel-OPSD"))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.scatter(x, rel, yvals[key], s=19, c=COLORS[key], alpha=.60, edgecolors="none", depthshade=False)
        ax.set(xlim=xlim, ylim=(0, 1), zlim=zlim)
        ax.set_xlabel("Teacher alignment", labelpad=6, fontsize=9)
        ax.set_ylabel("Target-sequence reliability", labelpad=7, fontsize=9)
        ax.set_zlabel(r"$\Delta\log p$", labelpad=5, fontsize=10)
        ax.set_title(title, pad=4, fontsize=13)
        ax.view_init(elev=23, azim=-58)
        ax.tick_params(labelsize=8, pad=0)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((.97, .98, .98, 1)); axis.pane.set_edgecolor("#DEE4E6")
            axis._axinfo["grid"].update(color="#DFE5E8", linewidth=.45)
    fig.text(.5, .035, "Reliability = exp(mean base log p over target mask-code tokens); same points and limits",
             ha="center", fontsize=8, color="#65717A")
    save(fig, out / "token_update_3d")
    print(f"rendered {x.size} finite tokens")


if __name__ == "__main__":
    main()
