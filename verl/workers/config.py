# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
ActorRolloutRef config
"""

from dataclasses import dataclass, field

from .actor import ActorConfig, FSDPConfig, ModelConfig, OptimConfig, RefConfig
from .critic import CriticConfig
from .reward import RewardConfig
from .rollout import RolloutConfig
from .opsd import OPSDConfig
from .supervised_anchors import SupervisedAnchorsConfig


__all__ = [
    "ActorConfig",
    "CriticConfig",
    "FSDPConfig",
    "ModelConfig",
    "OptimConfig",
    "OPSDConfig",
    "RefConfig",
    "RewardConfig",
    "RolloutConfig",
    "WorkerConfig",
]


@dataclass
class WorkerConfig:
    hybrid_engine: bool = True
    export_mode: bool = False
    """Build only the actor FSDP model for checkpoint-to-HF export."""
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    ref: RefConfig = field(default_factory=RefConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    opsd: OPSDConfig = field(default_factory=OPSDConfig)
    supervised_anchors: SupervisedAnchorsConfig = field(default_factory=SupervisedAnchorsConfig)

    def post_init(self):
        self.ref.micro_batch_size_per_device_for_experience = self.actor.micro_batch_size_per_device_for_experience
        self.ref.padding_free = self.actor.padding_free
        self.ref.dynamic_batching = self.actor.dynamic_batching
        self.ref.ulysses_size = self.actor.ulysses_size
        self.ref.use_torch_compile = self.actor.use_torch_compile
        if self.opsd.enabled and self.actor.ulysses_size != 1:
            raise ValueError("OPSD privileged distillation currently requires actor.ulysses_size=1.")
        self.supervised_anchors.post_init()
        if self.supervised_anchors.direct_grounding.enabled and (
            not self.opsd.enabled or not self.opsd.pixel_iou.enabled
        ):
            raise ValueError("direct_grounding requires OPSD pixel_iou.enabled=true.")
        if self.supervised_anchors.direct_grounding.enabled and not self.actor.optimize_segmenter:
            raise ValueError("direct_grounding requires actor.optimize_segmenter=true.")
