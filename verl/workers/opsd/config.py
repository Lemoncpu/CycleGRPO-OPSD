from dataclasses import dataclass, field

from ..actor.config import FSDPConfig, OffloadConfig


@dataclass
class PixelIoUConfig:
    enabled: bool = True
    decode_batch_size: int = 32
    mask_threshold: float = 0.5
    prefer_raw_gt: bool = True
    invalid_iou: float = 0.0

    def post_init(self):
        if self.decode_batch_size <= 0:
            raise ValueError("pixel_iou.decode_batch_size must be positive.")
        if not 0.0 <= self.mask_threshold <= 1.0:
            raise ValueError("pixel_iou.mask_threshold must be in [0, 1].")
        if not 0.0 <= self.invalid_iou <= 1.0:
            raise ValueError("pixel_iou.invalid_iou must be in [0, 1].")


@dataclass
class RoutingConfig:
    enabled: bool = True
    low_threshold: float = 0.5
    high_threshold: float = 0.85

    def post_init(self):
        if not 0.0 <= self.low_threshold <= self.high_threshold <= 1.0:
            raise ValueError("OPSD thresholds must satisfy 0 <= low <= high <= 1.")


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

    def post_init(self):
        if not 0.0 < self.beta < 1.0:
            raise ValueError("distillation.beta must be in (0, 1).")
        if self.temperature <= 0.0:
            raise ValueError("distillation.temperature must be positive.")
        if self.entropy_weight_beta < 0.0:
            raise ValueError("distillation.entropy_weight_beta must be non-negative.")
        if not 0.0 <= self.min_sample_weight <= 1.0:
            raise ValueError("distillation.min_sample_weight must be in [0, 1].")


@dataclass
class OPSDConfig:
    enabled: bool = False
    localization_rollouts: int = 6
    caption_loss_weight: float = 0.5
    localization_loss_weight: float = 0.5
    pixel_iou: PixelIoUConfig = field(default_factory=PixelIoUConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    ema_teacher: EMATeacherConfig = field(default_factory=EMATeacherConfig)
    regenerate: RegenerateConfig = field(default_factory=RegenerateConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)

    def post_init(self):
        if self.localization_rollouts <= 0:
            raise ValueError("localization_rollouts must be positive.")
        if self.caption_loss_weight < 0 or self.localization_loss_weight < 0:
            raise ValueError("OPSD task loss weights must be non-negative.")
        if self.caption_loss_weight + self.localization_loss_weight <= 0:
            raise ValueError("At least one OPSD task loss weight must be positive.")
        if self.enabled and self.routing.enabled and not self.pixel_iou.enabled:
            raise ValueError("OPSD three-route training requires pixel_iou.enabled=true.")
        if self.enabled and self.routing.enabled and not self.ema_teacher.enabled:
            raise ValueError("OPSD three-route training requires ema_teacher.enabled=true.")
