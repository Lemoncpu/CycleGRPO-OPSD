#!/usr/bin/env python3
"""Build a real-data diagnostic proxy from the repository's JSONL logs.

The repository does not contain token logits.  This figure therefore uses the
logged six localization IoUs for the sampled caption diagnoses.  GRPO is shown
as the scalar outcome broadcast over the six positions; OPSD is shown as the
real evidence residual (local IoU minus the caption mean), weighted by the
implemented mid-route weight.  The labels deliberately say ``logged`` and
``rollout`` rather than claiming autograd token gradients.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


DEFAULT_ROOT = Path("logs/cyclegrpo20k_withnt_direct30k_notarget10k_dlcqa10k_bs112_directgrpo_ce005_pixel_empty_positivepenalty1")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(errors="ignore").splitlines() if line.strip()]


def distill_weight(score: float) -> float:
    return float(np.clip((0.85 - score) / 0.35, 0.1, 1.0))


def _z(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return (value - value.mean()) / max(float(value.std()), 1e-8)


def _path_points(policy: np.ndarray, jsd: np.ndarray, bins: int, cumulative: bool):
    """Aggregate measured step signals into a small, readable path."""
    policy, jsd = _z(policy), _z(jsd)
    edges = np.linspace(0, len(policy), bins + 1, dtype=int)
    px, py = [], []
    for left, right in zip(edges[:-1], edges[1:]):
        if right <= left:
            continue
        px.append(float(policy[left:right].mean()))
        py.append(float(jsd[left:right].mean()))
    points = np.cumsum(np.stack([px, py], axis=1), axis=0) if cumulative else np.stack([px, py], axis=1)
    return np.vstack([np.zeros((1, 2), dtype=np.float32), points])


def _draw_figure(grpo, opsd, policy, jsd, output: Path, *, bins: int, cumulative: bool,
                 dpi: int = 220):
    cmap = LinearSegmentedColormap.from_list("opsd_proxy", ["#B6D4E9", "#FBFAF7", "#F2C3A7"])
    fig = plt.figure(figsize=(13.2, 4.65), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 1.02, 0.92], left=0.06, right=0.98,
                          bottom=0.20, top=0.87, wspace=0.34)
    ax0, ax1, ax2 = [fig.add_subplot(gs[0, i]) for i in range(3)]
    gscale = max(float(np.nanpercentile(np.abs(grpo), 95)), 1e-6)
    oscale = max(float(np.nanpercentile(np.abs(opsd), 95)), 1e-6)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    images = []
    for ax, arr, scale, title in [(ax0, grpo, gscale, "(a) GRPO outcome signal"),
                                   (ax1, opsd, oscale, "(b) OPSD evidence residual")]:
        image = ax.imshow(arr / scale, aspect="auto", interpolation="none", cmap=cmap, norm=norm)
        images.append(image)
        ax.set_title(title, loc="left", fontsize=9.5, color="#44545F", pad=5)
        ax.set_xlabel("rollout $k$", fontsize=8, color="#44545F", labelpad=2)
        ax.set_xticks(range(6), ["1", "2", "3", "4", "5", "6"])
        ax.set_ylabel("diagnosis row" if ax is ax0 else "", fontsize=8, color="#44545F", labelpad=2)
        ax.tick_params(labelsize=7, colors="#44545F", length=2, pad=2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6); spine.set_color("#44545F")
    cax = fig.add_axes([0.18, 0.095, 0.30, 0.022])
    cb = fig.colorbar(images[0], cax=cax, orientation="horizontal")
    cb.set_ticks([-1, 0, 1]); cb.set_ticklabels(["negative", "0", "positive"])
    cb.set_label("within-panel normalized logged signal", fontsize=7, color="#44545F", labelpad=2)
    cb.ax.tick_params(labelsize=6.5, colors="#44545F", length=2, pad=1)

    points = _path_points(policy, jsd, bins=bins, cumulative=cumulative)
    ax2.plot(points[:, 0], points[:, 1], color="#87ADC7", lw=1.05, zorder=2)
    ax2.scatter(points[1:, 0], points[1:, 1], s=12, color="#D19A78", zorder=3, linewidths=0)
    for left, right in zip(points[:-1], points[1:]):
        ax2.annotate("", xy=right, xytext=left,
                     arrowprops={"arrowstyle": "->", "lw": 0.65, "color": "#D19A78"})
    ax2.axhline(0, color="#D7DDE0", lw=0.6); ax2.axvline(0, color="#D7DDE0", lw=0.6)
    ax2.set_title("(c) measured logged update path", loc="left", fontsize=9.5, color="#44545F", pad=5)
    ax2.set_xlabel("cumulative policy signal" if cumulative else "binned policy signal",
                   fontsize=8, color="#44545F", labelpad=2)
    ax2.set_ylabel("cumulative JSD" if cumulative else "binned JSD", fontsize=8,
                   color="#44545F", labelpad=2)
    ax2.tick_params(labelsize=7, colors="#44545F", length=2, pad=2)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.6); spine.set_color("#44545F")
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    args = ap.parse_args()
    log_path = args.run_root / "checkpoints/experiment_log.jsonl"
    diag_path = args.run_root / "checkpoints/teacher_diagnoses.jsonl"
    if not log_path.exists() or not diag_path.exists():
        raise SystemExit(f"missing real logs: {log_path} or {diag_path}")
    logs = read_jsonl(log_path)
    # Teacher diagnosis logging samples the lower edge of the mid route.  The
    # real records are retained at the route boundary (>=0.5); this is a
    # descriptive proxy and is not presented as the confident JSD subset.
    diagnoses = [x for x in read_jsonl(diag_path)
                 if x.get("route") == "on_policy_distill" and float(x.get("R_Ci", 0.0)) >= 0.5]
    diagnoses.sort(key=lambda x: (int(x.get("step", 0)), str(x.get("sample_uid", ""))))
    if not diagnoses:
        raise SystemExit("no confident on_policy_distill diagnoses in the selected real run")

    # Six actual localization rollouts are the columns.  No interpolation or
    # hand-picked examples is used.
    n = len(diagnoses)
    grpo = np.full((n, 6), np.nan, dtype=np.float32)
    opsd = np.full_like(grpo, np.nan)
    step = np.zeros(n, dtype=np.int64)
    sample_ids = []
    for i, row in enumerate(diagnoses):
        rci = float(row["R_Ci"])
        vals = np.asarray(row.get("pixel_ious", []), dtype=np.float32)
        vals = vals[:6]
        if vals.size == 0:
            continue
        grpo[i, :vals.size] = rci
        opsd[i, :vals.size] = (vals - rci) * distill_weight(rci)
        step[i] = int(row.get("step", 0))
        sample_ids.append(str(row.get("sample_uid", "")))

    # The path is directly measured from the same run's scalar trainer logs.
    path_steps, policy_signal, jsd_signal = [], [], []
    for row in logs:
        cap = row.get("cap_actor", {})
        dst = row.get("distill_opsd", {})
        if "pg_loss" not in cap or "distill_jsd" not in dst:
            continue
        path_steps.append(int(row.get("step", len(path_steps) + 1)))
        policy_signal.append(-float(cap["pg_loss"]))
        jsd_signal.append(float(dst["distill_jsd"]))
    path_steps = np.asarray(path_steps, dtype=np.int64)
    policy_signal = np.asarray(policy_signal, dtype=np.float32)
    jsd_signal = np.asarray(jsd_signal, dtype=np.float32)
    if path_steps.size < 2:
        raise SystemExit("selected log does not contain a measured policy/JSD path")

    # Center the scalar outcome across the displayed diagnosis set: this is
    # the real group-level GRPO signal with its common baseline removed.
    grpo = grpo - np.nanmean(grpo, axis=0, keepdims=True)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "signals.npz", grpo_rollout=grpo, opsd_rollout=opsd,
                        step=step, sample_id=np.asarray(sample_ids),
                        path_steps=path_steps, policy_signal=policy_signal,
                        jsd_signal=jsd_signal)
    metadata = {
        "status": "complete_logged_rollout_proxy",
        "source_run": str(args.run_root),
        "source_files": [str(log_path), str(diag_path)],
        "n_diagnoses": int(n), "n_path_steps": int(path_steps.size),
        "diagnosis_filter": "route == on_policy_distill and R_Ci >= 0.5 (mid-route diagnosis records)",
        "heatmap_rows": "teacher diagnosis records sorted by (step, sample_uid)",
        "heatmap_columns": "six real localization rollouts recorded in pixel_ious",
        "grpo_panel": "(R_Ci - mean displayed R_Ci) scalar outcome broadcast over six rollout columns",
        "opsd_panel": "(pixel_iou_k - R_Ci) * clip((0.85-R_Ci)/0.35, 0.1, 1.0)",
        "heatmap_display_scaling": "each panel divided by its own 95th percentile absolute value; raw values saved",
        "panel_c": "measured logged path: -cap_actor.pg_loss versus distill_opsd.distill_jsd",
        "not_token_autograd": True,
        "not_parameter_update": True,
        "teacher_status": "source run logs ema_decay=1.0; no frozen-teacher assumption generalized",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")

    variants = out / "variants"
    variants.mkdir(exist_ok=True)
    # Twelve independent layouts are rendered before selecting the clearest one.
    variant_specs = [(b, c) for c in (False, True) for b in (6, 8, 10, 12, 14, 16)]
    for index, (b, c) in enumerate(variant_specs, 1):
        _draw_figure(grpo, opsd, policy_signal, jsd_signal,
                     variants / f"variant_{index:02d}_bins{b}_{'cum' if c else 'mean'}.png",
                     bins=b, cumulative=c, dpi=180)
    # Visual review selected 12 bins with cumulative stage means: only eleven
    # arrows remain, while preserving the direction over training.
    _draw_figure(grpo, opsd, policy_signal, jsd_signal, out / "figure_opsd_vs_grpo.png",
                 bins=12, cumulative=True, dpi=400)
    _draw_figure(grpo, opsd, policy_signal, jsd_signal, out / "figure_opsd_vs_grpo.pdf",
                 bins=12, cumulative=True, dpi=400)
    _draw_figure(grpo, opsd, policy_signal, jsd_signal, out / "figure_opsd_vs_grpo.svg",
                 bins=12, cumulative=True, dpi=400)
    metadata["variants_generated"] = 12
    metadata["selected_variant"] = "12 bins, cumulative stage means; 11 arrows"
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
