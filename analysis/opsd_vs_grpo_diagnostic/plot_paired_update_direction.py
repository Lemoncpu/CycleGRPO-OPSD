#!/usr/bin/env python3
"""Render paired GRPO/OPSD update directions when real common-rollout data exists.

The script intentionally refuses to infer updates from experiment logs.  Input
must contain measured sampled-token log-probability changes after one controlled
GRPO update and one controlled OPSD update from the same checkpoint/rollout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).parent
REQUIRED = ("delta_logp_grpo", "delta_logp_opsd", "teacher_gap", "group_id")


def load(path: Path, threshold: float) -> tuple[np.ndarray, np.ndarray, dict]:
    data = np.load(path, allow_pickle=False)
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        raise ValueError(f"paired rollout is missing: {missing}")
    g = np.asarray(data["delta_logp_grpo"], dtype=np.float64)
    o = np.asarray(data["delta_logp_opsd"], dtype=np.float64)
    gap = np.asarray(data["teacher_gap"], dtype=np.float64)
    gid = np.asarray(data["group_id"])
    if g.shape != o.shape or g.shape != gap.shape or g.ndim != 1 or len(gid) != len(g):
        raise ValueError(f"arrays must be matching 1-D positions, got {g.shape}, {o.shape}, {gap.shape}")
    if not (np.isfinite(g).all() and np.isfinite(o).all() and np.isfinite(gap).all()):
        raise ValueError("paired updates contain non-finite values")
    rows = []
    for group in np.unique(gid):
        mask = gid == group
        a = mask & (gap > threshold)
        b = mask & (gap < -threshold)
        if a.sum() and b.sum():
            rows.append((str(group), float(g[a].mean()), float(g[b].mean()),
                         float(o[a].mean()), float(o[b].mean()), int(a.sum()), int(b.sum())))
    if not rows:
        raise ValueError("no diagnostic group contains both teacher-gap directions")
    rows.sort(key=lambda x: x[0])
    arr = np.asarray([x[1:5] for x in rows], dtype=np.float64)
    meta = {"groups": [x[0] for x in rows], "a_count": [x[5] for x in rows],
            "b_count": [x[6] for x in rows], "threshold": threshold,
            "coverage": len(rows) / max(len(np.unique(gid)), 1)}
    return arr, np.asarray([x[0] for x in rows]), meta


def draw(points: np.ndarray, labels: np.ndarray, output: Path, title: str, annotate: bool) -> None:
    # columns: GRPO-A, GRPO-B, OPSD-A, OPSD-B
    fig, ax = plt.subplots(figsize=(5.4, 4.6), facecolor="white")
    for idx, (color, name, xa, yb) in enumerate([
        ("#87ADC7", "GRPO", 0, 1), ("#85B4A3", "OPSD", 2, 3)
    ]):
        x = points[:, xa]; y = points[:, yb]
        ax.scatter(x, y, s=19, color=color, alpha=0.45, edgecolors="none", label=name)
        mx, my = float(x.mean()), float(y.mean())
        ax.annotate("", xy=(mx, my), xytext=(0, 0),
                    arrowprops={"arrowstyle": "->", "lw": 1.3, "color": color})
        ax.scatter([mx], [my], s=42, color=color, edgecolors="white", linewidths=0.7, zorder=4)
        if annotate:
            ax.text(mx, my, name, color=color, fontsize=8, ha="left", va="bottom")
    ax.axhline(0, color="#D7DDE0", lw=0.6); ax.axvline(0, color="#D7DDE0", lw=0.6)
    ax.set_xlabel("A: teacher raises sampled-token log p", fontsize=8.2, color="#44545F")
    ax.set_ylabel("B: teacher lowers sampled-token log p", fontsize=8.2, color="#44545F")
    ax.set_title(title, loc="left", fontsize=10, color="#44545F", pad=5)
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.tick_params(labelsize=7.5, colors="#44545F", length=2)
    for spine in ax.spines.values(): spine.set_color("#44545F"); spine.set_linewidth(0.6)
    fig.tight_layout(pad=1.2)
    fig.savefig(output, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=BASE / "paired_updates.npz")
    ap.add_argument("--out-dir", type=Path, default=BASE)
    ap.add_argument("--teacher-gap-threshold", type=float, default=0.05)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"status": "blocked_missing_common_rollout", "input": str(args.input),
                "required": list(REQUIRED), "teacher_gap_threshold": args.teacher_gap_threshold,
                "real_measurement_required": True,
                "reason": "No saved common-checkpoint sampled-token log-prob changes were found in the repository; no figure was fabricated."}
    if not args.input.exists():
        (args.out_dir / "paired_update_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata, ensure_ascii=False))
        return 0
    try:
        points, labels, extra = load(args.input, args.teacher_gap_threshold)
    except Exception as exc:
        metadata["status"] = "blocked_invalid_paired_rollout"
        metadata["reason"] = str(exc)
        (args.out_dir / "paired_update_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata, ensure_ascii=False))
        return 0
    metadata.update(extra)
    metadata["status"] = "complete_common_rollout_paired_update"
    metadata["axes"] = "x=mean A delta_logp, y=mean B delta_logp; points are pre-registered groups"
    metadata["not_simulated"] = True
    np.savez_compressed(args.out_dir / "paired_update_signals.npz", points=points, group_labels=labels)
    specs = [(False, 0), (True, 0)] * 6
    for i, (annotate, _) in enumerate(specs, 1):
        draw(points, labels, args.out_dir / f"paired_update_variant_{i:02d}.png",
             "paired measured update directions", annotate)
    draw(points, labels, args.out_dir / "figure_paired_update_direction.png",
         "paired measured update directions", True)
    draw(points, labels, args.out_dir / "figure_paired_update_direction.pdf",
         "paired measured update directions", True)
    draw(points, labels, args.out_dir / "figure_paired_update_direction.svg",
         "paired measured update directions", True)
    metadata["variants_generated"] = 12
    metadata["selected_variant"] = "annotated mean arrows with paired group scatter"
    (args.out_dir / "paired_update_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
