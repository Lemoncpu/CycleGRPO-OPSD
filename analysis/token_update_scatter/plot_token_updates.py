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


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average-rank values without adding a scipy dependency."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    xr, yr = rankdata(x), rankdata(y)
    if np.std(xr) == 0 or np.std(yr) == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def paired_delta_strip(ax: plt.Axes, x: np.ndarray, delta: np.ndarray) -> None:
    """Show the paired OPSD-minus-GRPO change without inventing observations."""
    order = np.argsort(x, kind="mergesort")
    xs = np.arange(x.size)
    values = delta[order]
    colors = np.where(values >= 0, "#5A9E88", "#B77A6D")
    ax.axhline(0, color="#7F8B91", lw=.8)
    ax.vlines(xs, 0, values, color=colors, alpha=.58, lw=.8)
    ax.scatter(xs, values, c=colors, s=18, edgecolors="white", linewidths=.35, zorder=3)
    ax.set_xlim(-1, max(x.size, 1))
    ax.set_ylabel("$\\Delta\\log p$\nOPSD $-$ GRPO", fontsize=8, labelpad=5)
    ax.set_xlabel("same token pairs sorted by teacher alignment", fontsize=8, labelpad=3)
    ax.tick_params(axis="both", labelsize=7, length=2)
    ax.grid(axis="y", color="#E4E8EA", lw=.5)
    for spine in ax.spines.values():
        spine.set_color("#B3BDC3"); spine.set_linewidth(.7)


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
    rho = {key: spearman(x, yvals[key]) for key in yvals}
    delta = yvals["opsd"] - yvals["grpo"]
    fig = plt.figure(figsize=(10.4, 5.25), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[4.0, 1.18], hspace=.42, wspace=.12,
                          left=.095, right=.98, bottom=.16, top=.88)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharex=ax0, sharey=ax0)
    axarr = [ax0, ax1]
    for ax, key, title in zip(axarr, ("grpo", "opsd"), ("GRPO", "Pixel-OPSD")):
        ax.add_patch(Rectangle((0, 0), max(xlim[1], 0) - 0, max(ylim[1], 0) - 0,
                               facecolor="#B9DDD0", alpha=.15, lw=0, zorder=0))
        ax.add_patch(Rectangle((min(xlim[0], 0), min(ylim[0], 0)), 0 - min(xlim[0], 0),
                               0 - min(ylim[0], 0), facecolor="#B9DDD0", alpha=.15, lw=0, zorder=0))
        ax.axhline(0, color="#9EABB2", lw=.8, zorder=1)
        ax.axvline(0, color="#9EABB2", lw=.8, zorder=1)
        ax.scatter(x, yvals[key], s=42, c=COLORS[key], edgecolors=EDGES[key],
                   linewidths=.35, alpha=.82, zorder=2)
        ax.set_title(f"{title}   $n={x.size}$, $\u03c1_s={rho[key]:+.2f}$",
                     fontsize=12.5, pad=12, fontweight="medium")
        for spine in ax.spines.values():
            spine.set_color("#B3BDC3"); spine.set_linewidth(.75)
        ax.tick_params(length=3, width=.6)
    axarr[0].set_ylabel(r"Token update  $\Delta\log p$", labelpad=10)
    axarr[0].set_xlabel("Teacher alignment", fontsize=9, labelpad=3)
    axarr[1].set_xlabel("Teacher alignment", fontsize=9, labelpad=3)
    delta_ax = fig.add_subplot(gs[1, :])
    paired_delta_strip(delta_ax, x, delta)
    fig.text(.5, .015,
             "SAMTok mask-code tokens · same token pairs · checkpoint deltas; $\u03c1_s$ is descriptive, not causal",
             ha="center", fontsize=7.5, color="#65717A")
    save(fig, out / "token_update_2d")

    zlim = (float(min(np.min(y) for y in yvals.values())), float(max(np.max(y) for y in yvals.values())))
    fig = plt.figure(figsize=(11.5, 6.0), facecolor="white")
    gs3 = fig.add_gridspec(2, 2, height_ratios=[4.0, 1.0], hspace=.35, wspace=.02,
                           left=.015, right=.98, bottom=.13, top=.90)
    for i, (key, title) in enumerate(zip(("grpo", "opsd"), ("GRPO", "Pixel-OPSD"))):
        ax = fig.add_subplot(gs3[0, i], projection="3d")
        ax.scatter(x, rel, yvals[key], s=31, c=COLORS[key], alpha=.78,
                   edgecolors=EDGES[key], linewidths=.25, depthshade=False)
        ax.set(xlim=xlim, ylim=(0, 1), zlim=zlim)
        ax.set_xlabel("Teacher alignment", labelpad=6, fontsize=9)
        ax.set_ylabel("Target-sequence reliability", labelpad=7, fontsize=9)
        ax.set_zlabel(r"$\Delta\log p$", labelpad=5, fontsize=10)
        ax.set_title(f"{title}   $n={x.size}$, $\u03c1_s={rho[key]:+.2f}$", pad=4, fontsize=12.5)
        ax.view_init(elev=23, azim=-58)
        ax.tick_params(labelsize=8, pad=0)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((.97, .98, .98, 1)); axis.pane.set_edgecolor("#DEE4E6")
            axis._axinfo["grid"].update(color="#DFE5E8", linewidth=.45)
    delta3 = fig.add_subplot(gs3[1, :])
    paired_delta_strip(delta3, x, delta)
    delta3.set_xlabel("same token pairs sorted by teacher alignment", fontsize=8, labelpad=3)
    fig.text(.5, .018,
             "Reliability = exp(mean base log p over target mask-code tokens); same points and limits",
             ha="center", fontsize=7.5, color="#65717A")
    save(fig, out / "token_update_3d")
    print(f"rendered {x.size} finite tokens")


if __name__ == "__main__":
    main()
