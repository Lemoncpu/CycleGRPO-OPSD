#!/usr/bin/env python3
"""Extract GRPO/OPSD token gradients from a fixed diagnostic rollout.

The input is deliberately an explicit ``.npz`` contract.  This script never
creates trajectories, logits, rewards, or teacher scores.  A training rollout
must be exported with the student and teacher logits before this script can
produce ``signals.npz``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch


REQUIRED = {
    "student_logits",
    "teacher_logits",
    "target_ids",
    "response_mask",
    "old_log_probs",
    "advantages",
    "group_id",
    "sample_id",
    "route",
    "R_Ci",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def _write_missing(out_dir: Path, input_path: Path | None, missing: list[str]) -> None:
    metadata = {
        "status": "blocked_missing_diagnostic_data",
        "created": date.today().isoformat(),
        "input": None if input_path is None else str(input_path),
        "missing": missing,
        "message": (
            "No figure is produced. A real fixed rollout must provide the listed "
            "arrays; random logits or hand-written heatmaps are intentionally rejected."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.float().sum().clamp_min(1.0)
    return (values * mask.float()).sum() / denom


def _grpo_loss(logits: torch.Tensor, target_ids: torch.Tensor, old_log_probs: torch.Tensor,
               advantages: torch.Tensor, response_mask: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    from verl.trainer.core_algos import compute_policy_loss

    log_probs = torch.log_softmax(logits.float(), dim=-1).gather(
        -1, target_ids.long().unsqueeze(-1)
    ).squeeze(-1)
    loss, _ = compute_policy_loss(
        old_log_probs=old_log_probs.float(),
        log_probs=log_probs,
        advantages=advantages.float(),
        response_mask=response_mask.float(),
        clip_ratio_low=args.clip_ratio_low,
        clip_ratio_high=args.clip_ratio_high,
        clip_ratio_dual=args.clip_ratio_dual,
        loss_type=args.loss_type,
        loss_avg_mode=args.loss_avg_mode,
    )
    return loss


def _finite_difference(logits: torch.Tensor, target_ids: torch.Tensor, old: torch.Tensor,
                       adv: torch.Tensor, mask: torch.Tensor, args: argparse.Namespace) -> float:
    """Check one selected logit against central finite differences."""
    row = logits.detach().clone().float().requires_grad_(True)
    loss = _grpo_loss(row.unsqueeze(0), target_ids.unsqueeze(0), old.unsqueeze(0),
                      adv.unsqueeze(0), mask.unsqueeze(0), args)
    grad = torch.autograd.grad(loss, row)[0]
    valid = torch.nonzero(mask > 0, as_tuple=False)
    if valid.numel() == 0:
        return float("nan")
    t = int(valid[0].item())
    token = int(target_ids[t].item())
    eps = 1e-3
    with torch.no_grad():
        plus = row.detach().clone()
        minus = row.detach().clone()
        plus[t, token] += eps
        minus[t, token] -= eps
        lp = _grpo_loss(plus.unsqueeze(0), target_ids.unsqueeze(0), old.unsqueeze(0),
                        adv.unsqueeze(0), mask.unsqueeze(0), args)
        lm = _grpo_loss(minus.unsqueeze(0), target_ids.unsqueeze(0), old.unsqueeze(0),
                        adv.unsqueeze(0), mask.unsqueeze(0), args)
    return float((grad[t, token] - (lp - lm) / (2 * eps)).abs().item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="NPZ exported from a fixed rollout")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--clip-ratio-low", type=float, default=0.2)
    parser.add_argument("--clip-ratio-high", type=float, default=0.3)
    parser.add_argument("--clip-ratio-dual", type=float, default=3.0)
    parser.add_argument("--loss-type", choices=["default", "gspo", "gspo_token", "cispo"], default="default")
    parser.add_argument("--loss-avg-mode", choices=["token", "seq"], default="token")
    parser.add_argument("--opsd-beta", type=float, default=0.5)
    parser.add_argument("--opsd-temperature", type=float, default=1.0)
    parser.add_argument("--opsd-entropy-weight-beta", type=float, default=1.0)
    parser.add_argument("--opsd-min-sample-weight", type=float, default=0.1)
    parser.add_argument("--opsd-token-chunk-size", type=int, default=256)
    parser.add_argument("--teacher-diff-threshold", type=float, default=0.05)
    args = parser.parse_args()
    out_dir = args.out_dir

    if args.input is None:
        _write_missing(out_dir, None, ["--input fixed_rollout.npz"])
        return 2
    if not args.input.exists():
        _write_missing(out_dir, args.input, [str(args.input)])
        return 2

    data = np.load(args.input, allow_pickle=False)
    missing = sorted(REQUIRED - set(data.files))
    if missing:
        _write_missing(out_dir, args.input, missing)
        return 2

    # All rows are loaded before route filtering: GRPO advantages must already
    # have been computed from the complete sampled group.
    student = torch.from_numpy(np.asarray(data["student_logits"])).float()
    teacher = torch.from_numpy(np.asarray(data["teacher_logits"])).float()
    target = torch.from_numpy(np.asarray(data["target_ids"])).long()
    response_mask = torch.from_numpy(np.asarray(data["response_mask"])).bool()
    old = torch.from_numpy(np.asarray(data["old_log_probs"])).float()
    advantages = torch.from_numpy(np.asarray(data["advantages"])).float()
    group_id = np.asarray(data["group_id"])
    sample_id = np.asarray(data["sample_id"]).astype(str)
    route = np.asarray(data["route"]).astype(str)
    rci = np.asarray(data["R_Ci"], dtype=np.float32)
    if student.ndim != 3 or teacher.shape != student.shape:
        raise ValueError("student_logits and teacher_logits must both be [N,T,V] with identical shape")
    n, t, _ = student.shape
    for name, value in [("target_ids", target), ("response_mask", response_mask),
                        ("old_log_probs", old), ("advantages", advantages)]:
        if tuple(value.shape) != (n, t):
            raise ValueError(f"{name} must have shape {(n, t)}, got {tuple(value.shape)}")
    if len(group_id) != n or len(sample_id) != n or len(route) != n or len(rci) != n:
        raise ValueError("sample metadata length does not match logits")
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise ValueError("student/teacher logits contain non-finite values")
    if np.any(response_mask.numpy().sum(axis=1) == 0):
        raise ValueError("every diagnostic trajectory needs at least one response token")
    group_sizes = {str(g): int(np.sum(group_id == g)) for g in np.unique(group_id)}
    if any(size < 2 for size in group_sizes.values()):
        raise ValueError("GRPO group advantage check failed: every complete group needs >=2 samples")

    # OPSD's actual mid-route gate.  This is the only distribution-teaching
    # branch aligned with the original student response; regenerate is excluded.
    opsd_route_mask = (route == "on_policy_distill") & (rci >= 0.65)
    keep = np.flatnonzero(opsd_route_mask)
    if keep.size == 0:
        _write_missing(out_dir, args.input, ["on_policy_distill rows with R_Ci >= 0.65"])
        return 2
    order = np.array(sorted(keep.tolist(), key=lambda i: (float(rci[i]), str(sample_id[i]))), dtype=np.int64)
    student = student[order]
    teacher = teacher[order]
    target = target[order]
    response_mask = response_mask[order]
    old = old[order]
    advantages = advantages[order]
    sample_ids = sample_id[order]
    routes = route[order]
    rci_kept = rci[order]
    group_kept = group_id[order]

    policy_masks = torch.ones_like(response_mask, dtype=torch.float32)
    if "policy_loss_mask" in data.files:
        pm = torch.from_numpy(np.asarray(data["policy_loss_mask"])).float()[order]
        if pm.ndim == 1:
            pm = pm[:, None].expand(-1, t)
        policy_masks = policy_masks * pm
    grpo_grad = torch.zeros((len(order), t), dtype=torch.float32)
    opsd_grad = torch.zeros_like(grpo_grad)
    fd_error = float("nan")
    from verl.workers.opsd.distillation import chunked_weighted_jsd_loss
    from verl.workers.opsd.routing import distillation_weight

    for i in range(len(order)):
        mask = response_mask[i].float() * policy_masks[i]
        x = student[i].detach().clone().requires_grad_(True)
        gl = _grpo_loss(x.unsqueeze(0), target[i:i+1], old[i:i+1], advantages[i:i+1], mask.unsqueeze(0), args)
        grad = torch.autograd.grad(gl, x)[0]
        grpo_grad[i] = -grad.gather(-1, target[i].unsqueeze(-1)).squeeze(-1).detach()
        if i == 0:
            fd_error = _finite_difference(student[i], target[i], old[i], advantages[i], mask, args)

        x = student[i].detach().clone().requires_grad_(True)
        w = torch.tensor([distillation_weight(float(rci_kept[i]), 0.5, 0.85, args.opsd_min_sample_weight)])
        ol, _ = chunked_weighted_jsd_loss(
            student_logits=x.unsqueeze(0), teacher_logits=teacher[i:i+1].detach(),
            target_ids=target[i:i+1], response_mask=response_mask[i:i+1].float(),
            sample_weight=w, beta=args.opsd_beta, temperature=args.opsd_temperature,
            entropy_weight_beta=args.opsd_entropy_weight_beta,
            token_chunk_size=args.opsd_token_chunk_size,
        )
        grad = torch.autograd.grad(ol, x)[0]
        opsd_grad[i] = -grad.gather(-1, target[i].unsqueeze(-1)).squeeze(-1).detach()

    lengths = response_mask.sum(dim=1).numpy().astype(np.int64)
    bins = np.full((len(order), 64), np.nan, dtype=np.float32)
    bin_opsd = np.full_like(bins, np.nan)
    bin_mask = np.zeros_like(bins, dtype=bool)
    for i, length in enumerate(lengths):
        for b in range(64):
            lo, hi = b / 64.0, (b + 1) / 64.0
            pos = np.arange(length) / max(length - 1, 1)
            m = (pos >= lo) & ((pos < hi) if b < 63 else (pos <= hi))
            if np.any(m):
                bins[i, b] = float(grpo_grad[i, :length][m].mean())
                bin_opsd[i, b] = float(opsd_grad[i, :length][m].mean())
                bin_mask[i, b] = True

    # Teacher/student sampled-token difference defines A/B before any update.
    diff = (torch.log_softmax(teacher.float(), -1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
            - torch.log_softmax(student.float(), -1).gather(-1, target.unsqueeze(-1)).squeeze(-1)).numpy()
    threshold = float(args.teacher_diff_threshold)
    group_names = np.unique(group_kept.astype(str))
    local_x_grpo, local_y_grpo = [], []
    local_x_opsd, local_y_opsd = [], []
    group_out = []
    for g in group_names:
        rows = np.flatnonzero(group_kept.astype(str) == g)
        a = (diff[rows] > threshold) & response_mask[rows].numpy()
        b = (diff[rows] < -threshold) & response_mask[rows].numpy()
        if a.any() and b.any():
            local_x_grpo.append(float(grpo_grad[rows].numpy()[a].mean()))
            local_y_grpo.append(float(grpo_grad[rows].numpy()[b].mean()))
            local_x_opsd.append(float(opsd_grad[rows].numpy()[a].mean()))
            local_y_opsd.append(float(opsd_grad[rows].numpy()[b].mean()))
            group_out.append(str(g))

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "signals.npz", sample_id=sample_ids, group_id=group_kept,
        route=routes, R_Ci=rci_kept, response_mask=response_mask.numpy(),
        grad_grpo=grpo_grad.numpy(), grad_opsd=opsd_grad.numpy(),
        bin_grpo=bins, bin_opsd=bin_opsd, bin_mask=bin_mask,
        teacher_sampled_diff=diff,
        local_x_grpo=np.asarray(local_x_grpo, dtype=np.float32),
        local_y_grpo=np.asarray(local_y_grpo, dtype=np.float32),
        local_x_opsd=np.asarray(local_x_opsd, dtype=np.float32),
        local_y_opsd=np.asarray(local_y_opsd, dtype=np.float32),
        local_group=np.asarray(group_out),
    )
    root = Path(__file__).resolve().parents[2]
    metadata = {
        "status": "complete_local_logit_gradients",
        "created": date.today().isoformat(), "input": str(args.input),
        "checkpoint": args.checkpoint, "teacher_checkpoint": args.teacher_checkpoint,
        "dataset": args.dataset, "config": args.config, "seed": args.seed,
        "code_revision": _git_revision(root), "n_input_rows": int(n),
        "n_rows": int(len(order)), "opsd_route_coverage": float(len(order) / max(n, 1)),
        "route_filter": "route == on_policy_distill and R_Ci >= 0.65",
        "sample_order": "ascending (R_Ci, sample_id), fixed before inspecting signals",
        "losses": {
            "GRPO": "verl.trainer.core_algos.compute_policy_loss; default clipped surrogate",
            "OPSD": "verl.workers.opsd.distillation.chunked_weighted_jsd_loss; distribution branch only",
            "grpo_clip_ratio_low": args.clip_ratio_low, "grpo_clip_ratio_high": args.clip_ratio_high,
            "grpo_clip_ratio_dual": args.clip_ratio_dual, "grpo_loss_avg_mode": args.loss_avg_mode,
            "opsd_beta": args.opsd_beta, "opsd_temperature": args.opsd_temperature,
            "opsd_entropy_weight_beta": args.opsd_entropy_weight_beta,
            "opsd_min_sample_weight": args.opsd_min_sample_weight,
        },
        "teacher_student_group_threshold": threshold,
        "bins": 64, "normalization": "plot-only per-method global max abs; raw gradients saved",
        "causal_alignment": "input logits must be next-token logits for target_ids; no extra shift applied",
        "finite_difference_max_abs_error_grpo_first_row": fd_error,
        "2d_panel": "local logit gradients; actual parameter update deltas were not supplied",
        "n_2d_groups": len(local_x_grpo),
        "teacher_source": "provided teacher_logits; no frozen-teacher assumption made",
    }
    (out_dir / "metadata.json").write_text(json.dumps(_jsonable(metadata), ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
