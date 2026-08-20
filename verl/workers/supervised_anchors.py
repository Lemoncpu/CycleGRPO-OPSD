"""Configuration for external rewards that do not alter CycleGRPO routing."""

from dataclasses import dataclass, field
from typing import Any, Optional


def direct_grounding_source(
    source: object,
    include_no_target: bool,
    include_positive_sources: bool = True,
    include_label_sources: bool = False,
) -> Optional[str]:
    """Map a selected supervised source to an isolated grounding reward source.

    The normal CycleGRPO source split deliberately puts ``gres_no_target`` in
    the non-cycle batch.  Keep that routing intact while allowing the direct
    anchor to explicitly opt it into the existing refusal reward.
    """
    if include_positive_sources and source in {"refcoco_cycle", "grefcoco_cycle"}:
        return "supervised_grounding"
    if include_label_sources and source in {"cocostuff_cycle", "paco_part_cycle"}:
        return "supervised_grounding"
    if source == "gres_no_target" and include_no_target:
        return "supervised_grounding_no_target"
    return None


def alternating_localization_prompt_variants(size: int) -> list[str]:
    """Return a balanced sequence of the two image-localization prompts."""
    return ["refcoco" if index % 2 == 0 else "groundingsuite" for index in range(size)]


def aligned_direct_prompt_count(prompt_count: int, world_size: int) -> int:
    """Keep a prefix that can be dispatched evenly to every rollout rank."""
    if world_size <= 0:
        raise ValueError("world_size must be positive.")
    return (prompt_count // world_size) * world_size


def direct_grounding_loss_weight(
    step: int,
    target_weight: float,
    warmup_start_step: int,
    warmup_end_step: int,
) -> float:
    """Return the direct-GRPO weight without changing GRPO group statistics."""
    if target_weight < 0:
        raise ValueError("target_weight must be non-negative.")
    if warmup_start_step < 0 or warmup_end_step < warmup_start_step:
        raise ValueError("warmup steps must satisfy 0 <= start <= end.")
    if warmup_start_step == warmup_end_step:
        return target_weight if step >= warmup_start_step else 0.0
    if step <= warmup_start_step:
        return 0.0
    if step >= warmup_end_step:
        return target_weight
    return target_weight * (step - warmup_start_step) / (warmup_end_step - warmup_start_step)


def direct_mask_ce_source(source: object, include_positive_sources: bool = True) -> bool:
    """GT mask CE is deliberately restricted to human positive expressions."""
    return bool(include_positive_sources and source in {"refcoco_cycle", "grefcoco_cycle"})


def non_tensor_batch_row(values: Any, index: int) -> Any:
    """Select a row before DataProto indexing can collapse an object array."""
    return values[index]


def direct_mask_ce_response_fields(
    target_token_ids: list[list[int]], eos_token_id: int, pad_token_id: int
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Pad GT targets while excluding EOS and padding from the CE loss mask."""
    if not target_token_ids or any(not tokens for tokens in target_token_ids):
        raise ValueError("direct_mask_ce requires at least one non-empty GT mask-token target.")
    response_length = max(len(tokens) for tokens in target_token_ids) + 1
    responses, response_masks, attention_masks = [], [], []
    for tokens in target_token_ids:
        with_eos = tokens + [eos_token_id]
        padding = response_length - len(with_eos)
        responses.append(with_eos + [pad_token_id] * padding)
        response_masks.append([1] * len(tokens) + [0] * (1 + padding))
        attention_masks.append([1] * len(with_eos) + [0] * padding)
    return responses, response_masks, attention_masks


@dataclass
class CaptionQAConfig:
    enabled: bool = False
    train_files: list[str] = field(default_factory=list)
    batch_size: int = 0
    qa_jsonl: str = ""
    judge_base_url: str = ""
    judge_model: str = ""
    judge_api_key: str = "EMPTY"
    max_concurrency: int = 16
    timeout_seconds: float = 60.0
    reward_weight: float = 1.0
    loss_weight: float = 1.0

    def post_init(self):
        if not self.enabled:
            return
        if not self.qa_jsonl or not self.judge_base_url or not self.judge_model:
            raise ValueError("caption_qa requires qa_jsonl, judge_base_url, and judge_model when enabled.")
        if not self.train_files or self.batch_size <= 0:
            raise ValueError("caption_qa requires train_files and a positive batch_size when enabled.")
        if self.max_concurrency <= 0 or self.timeout_seconds <= 0:
            raise ValueError("caption_qa concurrency and timeout must be positive.")
        if self.reward_weight < 0:
            raise ValueError("caption_qa.reward_weight must be non-negative.")
        if self.loss_weight < 0:
            raise ValueError("caption_qa.loss_weight must be non-negative.")


@dataclass
class DirectGroundingConfig:
    enabled: bool = False
    train_files: list[str] = field(default_factory=list)
    batch_size: int = 0
    rollouts: int = 6
    loss_weight: float = 0.25
    warmup_start_step: int = 10
    warmup_end_step: int = 30
    include_no_target: bool = True
    include_positive_sources: bool = True
    include_label_sources: bool = False
    consume_no_target_caption: bool = False

    def post_init(self):
        if self.enabled and self.rollouts < 2:
            raise ValueError("direct_grounding.rollouts must be at least 2 for GRPO.")
        if self.enabled and (not self.train_files or self.batch_size <= 0):
            raise ValueError("direct_grounding requires train_files and a positive batch_size when enabled.")
        if self.loss_weight < 0:
            raise ValueError("direct_grounding.loss_weight must be non-negative.")
        if self.warmup_start_step < 0 or self.warmup_end_step < self.warmup_start_step:
            raise ValueError(
                "direct_grounding warmup steps must satisfy 0 <= warmup_start_step <= warmup_end_step."
            )
        if self.consume_no_target_caption:
            raise ValueError(
                "direct_grounding.consume_no_target_caption must be false: "
                "direct grounding is additive to main no-target caption GRPO."
            )


@dataclass
class DirectMaskCEConfig:
    """Small GT SAMTok teacher-forcing anchor for human referring expressions."""

    enabled: bool = False
    loss_weight: float = 0.02
    include_positive_sources: bool = True

    def post_init(self):
        if self.loss_weight < 0:
            raise ValueError("direct_mask_ce.loss_weight must be non-negative.")
        if self.enabled and not self.include_positive_sources:
            raise ValueError(
                "direct_mask_ce requires include_positive_sources=true because no-target has no GT mask CE target."
            )


@dataclass
class SupervisedAnchorsConfig:
    caption_qa: CaptionQAConfig = field(default_factory=CaptionQAConfig)
    direct_grounding: DirectGroundingConfig = field(default_factory=DirectGroundingConfig)
    direct_mask_ce: DirectMaskCEConfig = field(default_factory=DirectMaskCEConfig)

    def post_init(self):
        self.caption_qa.post_init()
        self.direct_grounding.post_init()
        self.direct_mask_ce.post_init()
