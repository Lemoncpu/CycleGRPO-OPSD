from dataclasses import dataclass, field

from ..actor.config import FSDPConfig, OffloadConfig


@dataclass
class PixelIoUConfig:
    enabled: bool = True
    decode_batch_size: int = 32
    mask_threshold: float = 0.5
    prefer_raw_gt: bool = True
    invalid_iou: float = 0.0
    require_exactly_one_mask: bool = True
    extra_mask_penalty: float = 1.0
    segmentation_max_response_tokens: int = 32

    def post_init(self):
        if self.decode_batch_size <= 0:
            raise ValueError("pixel_iou.decode_batch_size must be positive.")
        if not 0.0 <= self.mask_threshold <= 1.0:
            raise ValueError("pixel_iou.mask_threshold must be in [0, 1].")
        if not 0.0 <= self.invalid_iou <= 1.0:
            raise ValueError("pixel_iou.invalid_iou must be in [0, 1].")
        if self.extra_mask_penalty < 0.0:
            raise ValueError("pixel_iou.extra_mask_penalty must be non-negative.")
        if self.segmentation_max_response_tokens <= 0:
            raise ValueError("pixel_iou.segmentation_max_response_tokens must be positive.")


@dataclass
class RoutingConfig:
    enabled: bool = True
    low_threshold: float = 0.5
    high_threshold: float = 0.85
    preserve_original_grpo: bool = False

    def post_init(self):
        if not 0.0 <= self.low_threshold <= self.high_threshold <= 1.0:
            raise ValueError("OPSD thresholds must satisfy 0 <= low <= high <= 1.")


@dataclass
class CaptionSafetyConfig:
    """Keep malformed caption rollouts out of caption PPO and JSD updates."""

    enabled: bool = True
    max_response_tokens: int = 256
    force_regenerate: bool = True
    block_special_token_vocab: bool = True

    def post_init(self):
        if self.max_response_tokens <= 0:
            raise ValueError("caption_safety.max_response_tokens must be positive.")


@dataclass
class EMATeacherConfig:
    enabled: bool = True
    decay: float = 0.999
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    offload: OffloadConfig = field(default_factory=lambda: OffloadConfig(offload_params=True))

    def post_init(self):
        if not 0.0 < self.decay <= 1.0:
            raise ValueError("EMA decay must be in (0, 1].")


@dataclass
class RegenerateConfig:
    n: int = 6
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 192
    validation_rollouts: int = 1
    min_improvement: float = 0.05
    max_targets_per_prompt: int = 2

    def post_init(self):
        if self.n <= 0 or self.validation_rollouts <= 0 or self.max_new_tokens <= 0:
            raise ValueError("Regenerate rollout counts and max_new_tokens must be positive.")
        if self.temperature < 0.0 or not 0.0 < self.top_p <= 1.0:
            raise ValueError("Regenerate sampling requires temperature >= 0 and top_p in (0, 1].")
        if self.max_targets_per_prompt <= 0:
            raise ValueError("regenerate.max_targets_per_prompt must be positive.")
        if not 0.0 <= self.min_improvement <= 1.0:
            raise ValueError("regenerate.min_improvement must be in [0, 1].")


@dataclass
class DistillationConfig:
    beta: float = 0.5
    temperature: float = 1.0
    entropy_weight_beta: float = 1.0
    min_sample_weight: float = 0.1
    token_chunk_size: int = 256
    block_caption_special_token_vocab: bool = True

    def post_init(self):
        if not 0.0 < self.beta < 1.0:
            raise ValueError("distillation.beta must be in (0, 1).")
        if self.temperature <= 0.0:
            raise ValueError("distillation.temperature must be positive.")
        if self.entropy_weight_beta < 0.0:
            raise ValueError("distillation.entropy_weight_beta must be non-negative.")
        if not 0.0 <= self.min_sample_weight <= 1.0:
            raise ValueError("distillation.min_sample_weight must be in [0, 1].")
        if self.token_chunk_size <= 0:
            raise ValueError("distillation.token_chunk_size must be positive.")


@dataclass
class TeacherConfidenceConfig:
    """Gate privileged auxiliary losses to teacher signals with cycle evidence."""

    enabled: bool = False
    regenerate_min_teacher_score: float = 0.65
    regenerate_min_normalized_improvement: float = 0.30
    distill_min_caption_score: float = 0.65

    def post_init(self):
        for name, value in (
            ("regenerate_min_teacher_score", self.regenerate_min_teacher_score),
            (
                "regenerate_min_normalized_improvement",
                self.regenerate_min_normalized_improvement,
            ),
            ("distill_min_caption_score", self.distill_min_caption_score),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"teacher_confidence.{name} must be in [0, 1].")


@dataclass
class TeacherAnalysisConfig:
    enabled: bool = True
    max_samples_per_step: int = 2
    max_new_tokens: int = 96
    temperature: float = 0.0

    def post_init(self):
        if self.max_samples_per_step < 0 or self.max_new_tokens <= 0:
            raise ValueError("Teacher analysis sample count and max_new_tokens must be valid.")
        if self.temperature < 0.0:
            raise ValueError("teacher_analysis.temperature must be non-negative.")


@dataclass
class GroundednessConfig:
    """Conservative visual fact checks for target-bearing cycle captions."""

    enabled: bool = False
    teacher_must_be_frozen: bool = True
    max_claims: int = 8
    max_claim_chars: int = 160
    max_new_tokens: int = 96
    temperature: float = 0.0
    unsupported_penalty: float = 0.25
    contradicted_penalty: float = 0.75
    min_checked_claims: int = 1
    min_groundedness_score: float = 0.85
    min_distill_caption_score: float = 0.65
    no_target_enabled: bool = False
    token_jsd_enabled: bool = False
    token_jsd_multiplier: float = 1.0

    def post_init(self):
        if self.max_claims <= 0 or self.max_claim_chars <= 0 or self.max_new_tokens <= 0:
            raise ValueError("groundedness claim and response limits must be positive.")
        if self.temperature < 0.0:
            raise ValueError("groundedness.temperature must be non-negative.")
        if self.unsupported_penalty < 0.0 or self.contradicted_penalty < 0.0:
            raise ValueError("groundedness penalties must be non-negative.")
        if self.min_checked_claims <= 0:
            raise ValueError("groundedness.min_checked_claims must be positive.")
        if not 0.0 <= self.min_groundedness_score <= 1.0:
            raise ValueError("groundedness.min_groundedness_score must be in [0, 1].")
        if not 0.0 <= self.min_distill_caption_score <= 1.0:
            raise ValueError("groundedness.min_distill_caption_score must be in [0, 1].")
        if self.token_jsd_multiplier < 0.0:
            raise ValueError("groundedness.token_jsd_multiplier must be non-negative.")


@dataclass
class OPSDConfig:
    enabled: bool = False
    localization_rollouts: int = 6
    caption_loss_weight: float = 0.5
    localization_loss_weight: float = 0.5
    caption_anchor_kl_coef: float = 0.0
    caption_anchor_kl_all_safe_routes: bool = False
    segmentation_anchor_kl_coef: float = 0.0
    asymmetric_gradient_projection: bool = False
    pixel_iou: PixelIoUConfig = field(default_factory=PixelIoUConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    caption_safety: CaptionSafetyConfig = field(default_factory=CaptionSafetyConfig)
    ema_teacher: EMATeacherConfig = field(default_factory=EMATeacherConfig)
    regenerate: RegenerateConfig = field(default_factory=RegenerateConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    teacher_confidence: TeacherConfidenceConfig = field(default_factory=TeacherConfidenceConfig)
    teacher_analysis: TeacherAnalysisConfig = field(default_factory=TeacherAnalysisConfig)
    groundedness: GroundednessConfig = field(default_factory=GroundednessConfig)

    def post_init(self):
        if self.localization_rollouts <= 0:
            raise ValueError("localization_rollouts must be positive.")
        if self.caption_loss_weight < 0 or self.localization_loss_weight < 0:
            raise ValueError("OPSD task loss weights must be non-negative.")
        if self.caption_loss_weight + self.localization_loss_weight <= 0:
            raise ValueError("At least one OPSD task loss weight must be positive.")
        if self.caption_anchor_kl_coef < 0:
            raise ValueError("caption_anchor_kl_coef must be non-negative.")
        if self.segmentation_anchor_kl_coef < 0:
            raise ValueError("segmentation_anchor_kl_coef must be non-negative.")
        self.teacher_confidence.post_init()
        self.groundedness.post_init()
        if self.enabled and self.routing.enabled and not self.pixel_iou.enabled:
            raise ValueError("OPSD three-route training requires pixel_iou.enabled=true.")
        if self.enabled and self.routing.enabled and not self.ema_teacher.enabled:
            raise ValueError("OPSD three-route training requires ema_teacher.enabled=true.")
        if self.enabled and self.groundedness.enabled and not self.routing.enabled:
            raise ValueError("groundedness requires routing.enabled=true.")
        if self.enabled and self.groundedness.enabled and not self.ema_teacher.enabled:
            raise ValueError("groundedness requires ema_teacher.enabled=true.")
        if (
            self.enabled
            and self.groundedness.enabled
            and self.groundedness.teacher_must_be_frozen
            and self.ema_teacher.enabled
            and self.ema_teacher.decay != 1.0
        ):
            raise ValueError(
                "groundedness requires a frozen initial teacher; set ema_teacher.decay=1.0."
            )
