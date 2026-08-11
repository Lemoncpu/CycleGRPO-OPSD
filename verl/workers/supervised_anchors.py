"""Configuration for external rewards that do not alter CycleGRPO routing."""

from dataclasses import dataclass, field
from typing import Optional


def direct_grounding_source(
    source: object, include_no_target: bool, include_positive_sources: bool = True
) -> Optional[str]:
    """Map a dataset source to an isolated direct-grounding reward source.

    The normal CycleGRPO source split deliberately puts ``gres_no_target`` in
    the non-cycle batch.  Keep that routing intact while allowing the direct
    anchor to explicitly opt it into the existing refusal reward.
    """
    if include_positive_sources and source in {"refcoco_cycle", "grefcoco_cycle"}:
        return "supervised_grounding"
    if source == "gres_no_target" and include_no_target:
        return "supervised_grounding_no_target"
    return None


def alternating_localization_prompt_variants(size: int) -> list[str]:
    """Return a balanced sequence of the two image-localization prompts."""
    return ["refcoco" if index % 2 == 0 else "groundingsuite" for index in range(size)]


@dataclass
class CaptionQAConfig:
    enabled: bool = False
    qa_jsonl: str = ""
    judge_base_url: str = ""
    judge_model: str = ""
    judge_api_key: str = "EMPTY"
    max_concurrency: int = 16
    timeout_seconds: float = 60.0
    reward_weight: float = 1.0

    def post_init(self):
        if not self.enabled:
            return
        if not self.qa_jsonl or not self.judge_base_url or not self.judge_model:
            raise ValueError("caption_qa requires qa_jsonl, judge_base_url, and judge_model when enabled.")
        if self.max_concurrency <= 0 or self.timeout_seconds <= 0:
            raise ValueError("caption_qa concurrency and timeout must be positive.")
        if self.reward_weight < 0:
            raise ValueError("caption_qa.reward_weight must be non-negative.")


@dataclass
class DirectGroundingConfig:
    enabled: bool = False
    rollouts: int = 2
    loss_weight: float = 0.25
    include_no_target: bool = True
    include_positive_sources: bool = True
    consume_no_target_caption: bool = False

    def post_init(self):
        if self.enabled and self.rollouts < 2:
            raise ValueError("direct_grounding.rollouts must be at least 2 for GRPO.")
        if self.loss_weight < 0:
            raise ValueError("direct_grounding.loss_weight must be non-negative.")


@dataclass
class SupervisedAnchorsConfig:
    caption_qa: CaptionQAConfig = field(default_factory=CaptionQAConfig)
    direct_grounding: DirectGroundingConfig = field(default_factory=DirectGroundingConfig)

    def post_init(self):
        self.caption_qa.post_init()
        self.direct_grounding.post_init()
