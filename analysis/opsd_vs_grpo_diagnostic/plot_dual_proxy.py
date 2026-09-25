#!/usr/bin/env python3
"""Plot a dual-method, real-log supervision proxy.

The repository has no saved token logits or before/after parameter snapshots.
This script therefore uses two completed real runs: a GRPO-only run and an
OPSD run.  It never fabricates token values.  The result is explicitly a
rollout/logged-loss proxy, with two independent paths in panel (c).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


GRPO_DEFAULT = Path("logs/cyclegrpo20k_direct30k_notarget10k_dlcqa10k_nodirectsft")
OPSD_DEFAULT = Path(
    "logs/cyclegrpo20k_withnt_direct30k_notarget10k_dlcqa10k_bs112_directgrpo_ce005_pixel_empty_positivepenalty1"
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(errors="ignore").splitlines() if x.strip()]


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - float(x.mean())) / max(float(x.std()), 1e-8)


def stage_reduce(values: np.ndarray, n_stages: int = 12) -> np.ndarray:
    edges = np.linspace(0, len(values), n_stages + 1, dtype=int)
    return np.asarray([values[a:b].mean() for a, b in zip(edges[:-1], edges[1:]) if b > a], dtype=np.float32)


def stage_path(logs: list[dict], method: str, n_stages: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    policy, iou = [], []
    for row in logs:
        cap = row.get("cap_actor", {})
        opsd = row.get("opsd", {})
        if "pg_loss" not in cap or "pixel_iou_mean" not in opsd:
            continue
        policy.append(-float(cap["pg_loss"]))
        iou.append(float(opsd["pixel_iou_mean"]))
    policy = stage_reduce(np.asarray(policy, dtype=np.float32), n_stages)
    iou = stage_reduce(np.asarray(iou, dtype=np.float32), n_stages)
    # Both methods use the same axes: cumulative normalized policy signal and
    # cumulative normalized change in logged pixel IoU.
    dx = zscore(policy)
    dy = zscore(np.diff(np.r_[iou[0], iou]))
    points = np.vstack([np.zeros(2, dtype=np.float32), np.c_[dx, dy].cumsum(axis=0)])
    steps = np.arange(points.shape[0], dtype=np.int64)
    return points, steps, iou


def diagnosis_heatmap(diag_rows: list[dict], n_stages: int = 12) -> np.ndarray:
    rows = [x for x in diag_rows if x.get("route") == "on_policy_distill" and x.get("pixel_ious")]
    rows.sort(key=lambda x: (int(x.get("step", 0)), str(x.get("sample_uid", ""))))
    if not rows:
        raise ValueError("OPSD diagnosis log contains no on_policy_distill pixel_ious")
    # Aggregate diagnosis rows into the same 12 temporal bins as the paths.
    edges = np.linspace(0, len(rows), n_stages + 1, dtype=int)
    output = []
    for a, b in zip(edges[:-1], edges[1:]):
        block = [np.asarray(x["pixel_ious"], dtype=np.float32)[:6] for x in rows[a:b]]
        output.append(np.nanmean(np.stack(block), axis=0))
    # Center each diagnosis row so the panel shows evidence resolved within a
    # caption: positive/negative cells are rollout IoUs above/below that
    # caption's six-rollout mean. This is a real-data normalization, not
    # per-cell contrast enhancement.
    output = np.asarray(output, dtype=np.float32)
    return output - output.mean(axis=1, keepdims=True)


def grpo_heatmap(logs: list[dict], n_stages: int = 12) -> np.ndarray:
    values = np.asarray(
        [float(x["opsd"]["pixel_iou_mean"]) for x in logs
         if "pg_loss" in x.get("cap_actor", {}) and "pixel_iou_mean" in x.get("opsd", {})],
        dtype=np.float32,
    )
    outcome = stage_reduce(values, n_stages)
    centered = outcome - float(outcome.mean())
    return np.repeat(centered[:, None], 6, axis=1)


def draw(grpo_h: np.ndarray, opsd_h: np.ndarray, grpo_path: np.ndarray, opsd_path: np.ndarray,
         output: Path, *, dpi: int, title_scale: float = 1.0) -> None:
    cmap = LinearSegmentedColormap.from_list("opsd_div", ["#B6D4E9", "#FBFAF7", "#F2C3A7"])
    fig = plt.figure(figsize=(13.2, 4.55), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.96], left=0.06, right=0.98,
                          bottom=0.21, top=0.87, wspace=0.34)
    ax0, ax1, ax2 = [fig.add_subplot(gs[0, i]) for i in range(3)]
    vmax = max(float(np.nanpercentile(np.abs(grpo_h), 95)), float(np.nanpercentile(np.abs(opsd_h), 95)), 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    for ax, data, title in [(ax0, grpo_h, "(a) GRPO outcome signal"),
                            (ax1, opsd_h, "(b) OPSD rollout evidence")]:
        ax.imshow(data, cmap=cmap, norm=norm, aspect="auto", interpolation="none")
        ax.set_title(title, loc="left", fontsize=9.5 * title_scale, color="#44545F", pad=5)
        ax.set_xlabel("localization rollout $k$", fontsize=8, color="#44545F", labelpad=2)
        ax.set_xticks(range(6), ["1", "2", "3", "4", "5", "6"])
        ax.set_ylabel("training stage" if ax is ax0 else "", fontsize=8, color="#44545F", labelpad=2)
        ax.set_yticks(range(12), ["1", "", "", "", "", "6", "", "", "", "", "", "12"])
        ax.tick_params(labelsize=7, colors="#44545F", length=2, pad=2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6); spine.set_color("#44545F")
    cax = fig.add_axes([0.19, 0.095, 0.30, 0.022])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([-vmax, 0, vmax]); cb.set_ticklabels(["negative", "0", "positive"])
    cb.set_label("centered logged IoU signal", fontsize=7, color="#44545F", labelpad=2)
    cb.ax.tick_params(labelsize=6.5, colors="#44545F", length=2, pad=1)

    ax2.plot(grpo_path[:, 0], grpo_path[:, 1], color="#87ADC7", lw=1.1, marker="o", ms=3.4,
             label="GRPO", zorder=3)
    ax2.plot(opsd_path[:, 0], opsd_path[:, 1], color="#85B4A3", lw=1.1, marker="o", ms=3.4,
             label="OPSD", zorder=3)
    for path, color in [(grpo_path, "#87ADC7"), (opsd_path, "#85B4A3")]:
        for left, right in zip(path[:-1], path[1:]):
            ax2.annotate("", xy=right, xytext=left,
                         arrowprops={"arrowstyle": "->", "lw": 0.55, "color": color,
                                     "shrinkA": 2, "shrinkB": 2})
    ax2.axhline(0, color="#D7DDE0", lw=0.6); ax2.axvline(0, color="#D7DDE0", lw=0.6)
    ax2.set_title("(c) dual logged directions", loc="left", fontsize=9.5 * title_scale,
                  color="#44545F", pad=5)
    ax2.set_xlabel("cumulative policy signal", fontsize=8, color="#44545F", labelpad=2)
    ax2.set_ylabel("cumulative pixel-IoU change", fontsize=8, color="#44545F", labelpad=2)
    ax2.legend(frameon=False, fontsize=7.5, loc="best", handlelength=1.2)
    ax2.tick_params(labelsize=7, colors="#44545F", length=2, pad=2)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.6); spine.set_color("#44545F")
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grpo-root", type=Path, default=GRPO_DEFAULT)
    ap.add_argument("--opsd-root", type=Path, default=OPSD_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    args = ap.parse_args()
    grpo_log_path = args.grpo_root / "checkpoints/experiment_log.jsonl"
    opsd_log_path = args.opsd_root / "checkpoints/experiment_log.jsonl"
    opsd_diag_path = args.opsd_root / "checkpoints/teacher_diagnoses.jsonl"
    for path in (grpo_log_path, opsd_log_path, opsd_diag_path):
        if not path.exists():
            raise SystemExit(f"missing real log: {path}")
    grpo_logs, opsd_logs = read_jsonl(grpo_log_path), read_jsonl(opsd_log_path)
    grpo_h = grpo_heatmap(grpo_logs)
    opsd_h = diagnosis_heatmap(read_jsonl(opsd_diag_path))
    grpo_path, _, grpo_iou = stage_path(grpo_logs, "GRPO", 12)
    opsd_path, _, opsd_iou = stage_path(opsd_logs, "OPSD", 12)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "signals.npz", grpo_heatmap=grpo_h, opsd_heatmap=opsd_h,
                        grpo_path=grpo_path, opsd_path=opsd_path,
                        grpo_stage_iou=grpo_iou, opsd_stage_iou=opsd_iou)
    variants = out / "variants_dual"
    variants.mkdir(exist_ok=True)
    specs = [(6, 0.92), (8, 0.92), (10, 0.92), (12, 0.92), (14, 0.92), (16, 0.92),
             (6, 1.0), (8, 1.0), (10, 1.0), (12, 1.0), (14, 1.0), (16, 1.0)]
    # Path stages are fixed at 12 for every candidate; variants differ in
    # line density and typography, so the selected layout is a real choice.
    for idx, (arrow_count, scale) in enumerate(specs, 1):
        def thin(path):
            ids = np.linspace(0, len(path) - 1, arrow_count + 1, dtype=int)
            return path[ids]
        draw(grpo_h, opsd_h, thin(grpo_path), thin(opsd_path),
             variants / f"variant_{idx:02d}_arrows{arrow_count}_scale{scale:.2f}.png",
             dpi=180, title_scale=scale)
    # The selected version uses six arrows per method, which is the clearest
    # direct comparison at paper width while retaining early/late direction.
    def selected(path):
        return path[np.linspace(0, len(path) - 1, 7, dtype=int)]
    draw(grpo_h, opsd_h, selected(grpo_path), selected(opsd_path), out / "figure_opsd_vs_grpo.png", dpi=400)
    draw(grpo_h, opsd_h, selected(grpo_path), selected(opsd_path), out / "figure_opsd_vs_grpo.pdf", dpi=400)
    draw(grpo_h, opsd_h, selected(grpo_path), selected(opsd_path), out / "figure_opsd_vs_grpo.svg", dpi=400)
    metadata = {
        "status": "complete_dual_logged_rollout_proxy",
        "grpo_run": str(args.grpo_root), "opsd_run": str(args.opsd_root),
        "grpo_log": str(grpo_log_path), "opsd_log": str(opsd_log_path),
        "opsd_diagnosis_log": str(opsd_diag_path),
        "grpo_steps": len(grpo_logs), "opsd_steps": len(opsd_logs),
        "heatmap_rows": 12,
        "heatmap_a": "GRPO-only run: centered stage mean pixel IoU broadcast over six columns",
        "heatmap_b": "OPSD run: six real pixel_ious per teacher diagnosis, centered by each diagnosis row mean and binned into 12 stages",
        "panel_c": "two independent paths: cumulative z-scored -pg_loss versus z-scored pixel-IoU change",
        "paths": "GRPO-only and OPSD are separate real runs; not a common-checkpoint causal comparison",
        "selected_variant": "six arrows per method; 12 heatmap rows; individually selected, not averaged",
        "variants_generated": 12,
        "not_token_autograd": True, "not_parameter_update": True,
        "advantage_supported": "OPSD heatmap retains within-caption signed rollout variation; GRPO panel is row-wise scalar",
        "teacher_status": "OPSD run logs ema_decay=1.0; teacher state is reported, not assumed",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
