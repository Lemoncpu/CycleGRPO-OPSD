"""Memory-bounded OPSD generalized-JSD distillation helpers."""

from __future__ import annotations

import math
import re
from typing import Dict, Mapping, Optional, Tuple

import torch
from torch.utils.checkpoint import checkpoint


_CAPTION_BLOCKED_SPECIAL_TOKEN = re.compile(
    r"^<\|(?:mt_(?:start|end|\d+)|object_ref(?:_[^|]+)?)\|>$"
)


def caption_blocked_special_token_ids(vocab: Mapping[str, int]) -> list[int]:
    """Return SAMTok mask and object-reference token ids that captions cannot use."""
    return sorted(
        {
            int(token_id)
            for token, token_id in vocab.items()
            if _CAPTION_BLOCKED_SPECIAL_TOKEN.fullmatch(str(token))
        }
    )


def _masked_log_softmax(
    logits: torch.Tensor,
    temperature: float,
    blocked_token_ids: Optional[torch.Tensor],
) -> torch.Tensor:
    scaled_logits = logits.float() / temperature
    if blocked_token_ids is not None and blocked_token_ids.numel() > 0:
        valid_ids = blocked_token_ids[
            (blocked_token_ids >= 0) & (blocked_token_ids < scaled_logits.size(-1))
        ]
        if valid_ids.numel() > 0:
            scaled_logits = scaled_logits.index_fill(-1, valid_ids, -torch.inf)
    return torch.log_softmax(scaled_logits, dim=-1)


def _validate_inputs(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target_ids: torch.Tensor,
    response_mask: torch.Tensor,
    sample_weight: torch.Tensor,
    token_chunk_size: int,
) -> None:
    if student_logits.ndim != 3 or teacher_logits.ndim != 3:
        raise ValueError("student_logits and teacher_logits must have shape [batch, tokens, vocab].")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student_logits and teacher_logits must have identical shapes.")
    batch_size, response_length, _ = student_logits.shape
    if target_ids.shape != (batch_size, response_length):
        raise ValueError("target_ids must have shape [batch, tokens].")
    if response_mask.shape != (batch_size, response_length):
        raise ValueError("response_mask must have shape [batch, tokens].")
    if sample_weight.shape != (batch_size,):
        raise ValueError("sample_weight must have shape [batch].")
    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be positive.")


def chunked_weighted_jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target_ids: torch.Tensor,
    response_mask: torch.Tensor,
    sample_weight: torch.Tensor,
    *,
    beta: float,
    temperature: float,
    entropy_weight_beta: float,
    token_chunk_size: int,
    blocked_token_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Return the original weighted generalized-JSD without full-response fp32 tensors.

    The teacher is fixed, so its entropy and target-token score are calculated one
    response-token chunk at a time.  Student/teacher JSD chunks are wrapped in
    non-reentrant activation checkpointing: their fp32 softmax/probability
    intermediates are recomputed in backward instead of being retained for all
    response tokens.
    """
    _validate_inputs(
        student_logits,
        teacher_logits,
        target_ids,
        response_mask,
        sample_weight,
        token_chunk_size,
    )
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be in (0, 1).")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if entropy_weight_beta < 0.0:
        raise ValueError("entropy_weight_beta must be non-negative.")

    batch_size, response_length, _ = student_logits.shape
    token_mask = response_mask.float()
    teacher_entropy = torch.empty(
        (batch_size, response_length), device=teacher_logits.device, dtype=torch.float32
    )
    teacher_target_log_probs = torch.empty_like(teacher_entropy)

    # These per-token statistics are small.  Do not retain any full-response
    # vocabulary-sized float32 tensor while constructing confidence weights.
    with torch.no_grad():
        for start in range(0, response_length, token_chunk_size):
            end = min(start + token_chunk_size, response_length)
            teacher_log_probs = _masked_log_softmax(
                teacher_logits[:, start:end, :], temperature, blocked_token_ids
            )
            teacher_entropy[:, start:end] = -(
                teacher_log_probs.exp() * teacher_log_probs
            ).sum(dim=-1)
            teacher_target_log_probs[:, start:end] = teacher_log_probs.gather(
                -1, target_ids[:, start:end].unsqueeze(-1)
            ).squeeze(-1)

    confidence = torch.exp(-entropy_weight_beta * teacher_entropy)
    confidence_mean = (confidence * token_mask).sum(dim=-1, keepdim=True) / token_mask.sum(
        dim=-1, keepdim=True
    ).clamp_min(1.0)
    weights = token_mask * confidence / confidence_mean.clamp_min(1e-6)
    weights = weights * sample_weight.float().unsqueeze(-1)

    log_student_weight = math.log(1.0 - beta)
    log_teacher_weight = math.log(beta)
    loss_numerator = torch.zeros((), device=student_logits.device, dtype=torch.float32)
    for start in range(0, response_length, token_chunk_size):
        end = min(start + token_chunk_size, response_length)
        teacher_chunk = teacher_logits[:, start:end, :]
        weight_chunk = weights[:, start:end]

        # Default arguments freeze each loop's tensor views for checkpoint's
        # backward recomputation instead of capturing mutable loop variables.
        def chunk_loss(
            student_chunk: torch.Tensor,
            teacher_chunk: torch.Tensor = teacher_chunk,
            weight_chunk: torch.Tensor = weight_chunk,
        ) -> torch.Tensor:
            student_log_probs = _masked_log_softmax(student_chunk, temperature, blocked_token_ids)
            teacher_log_probs = _masked_log_softmax(teacher_chunk, temperature, blocked_token_ids)
            mixture_log_probs = torch.logaddexp(
                student_log_probs + log_student_weight,
                teacher_log_probs + log_teacher_weight,
            )
            kl_student = (
                student_log_probs.exp() * (student_log_probs - mixture_log_probs)
            ).sum(dim=-1)
            kl_teacher = (
                teacher_log_probs.exp() * (teacher_log_probs - mixture_log_probs)
            ).sum(dim=-1)
            jsd = beta * kl_teacher + (1.0 - beta) * kl_student
            return (jsd * weight_chunk).sum()

        loss_numerator = loss_numerator + checkpoint(
            chunk_loss,
            student_logits[:, start:end, :],
            use_reentrant=False,
        )

    valid_tokens = token_mask.sum().clamp_min(1.0)
    metrics = {
        "local_loss": loss_numerator.detach() / valid_tokens,
        "teacher_entropy": (teacher_entropy * token_mask).sum() / valid_tokens,
        "teacher_sequence_score": (teacher_target_log_probs * token_mask).sum() / valid_tokens,
        "blocked_vocab_size": torch.tensor(
            0 if blocked_token_ids is None else blocked_token_ids.numel(),
            device=student_logits.device,
            dtype=torch.float32,
        ),
    }
    return loss_numerator, metrics
