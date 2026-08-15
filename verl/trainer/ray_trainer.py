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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface.
"""

import json
import math
import os
import re
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin
from torch.utils.data._utils.collate import default_collate

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from ..utils.dataset import process_image
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import FunctionRewardManager
from ..workers.opsd import (
    caption_safety_reason,
    build_privileged_teacher_images,
    disabled_groundedness,
    distillation_weight,
    format_privileged_prompt,
    groundedness_token_mask,
    parse_groundedness_verdict,
    regenerate_weight,
    teacher_caption_is_safe,
    uses_original_grpo,
)
from ..workers.supervised_anchors import (
    aligned_direct_prompt_count,
    alternating_localization_prompt_variants,
    direct_grounding_loss_weight,
    direct_mask_ce_response_fields,
    direct_mask_ce_source,
    direct_grounding_source,
)
from .config import PPOConfig
from .core_algos import (
    AdvantageEstimator,
    FixedKLController,
    KLController,
    compute_advantage_return,
    compute_kl,
    get_kl_controller,
)
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from pycocotools import mask as mask_utils

class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create ray resource pools for distributed training."""
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards."""
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = torch.mean(VF.masked_mean(kld, mask=response_mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def strip_temporal_priors(response_text: str, replacement: str = "in this segment") -> str:
    """Remove explicit temporal expressions from a free-form response.

    This is a post-processing guard to prevent leaking timestamp priors into
    temporal grounding prompts.
    """
    if not isinstance(response_text, str):
        return response_text

    text = response_text
    # Normalize dash variants to simplify matching.
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    interval_patterns = [
        # e.g. "100.3 - 104.2 seconds", "100.3 to 104.2 sec"
        r"\b\d+(?:\.\d+)?\s*(?:-|to|~)\s*\d+(?:\.\d+)?\s*(?:s|sec(?:ond)?s?|seconds?|mins?|minutes?|分钟|秒)\b",
        # e.g. "00:01:40 - 00:01:44", "1:40 to 1:44"
        r"\b\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*(?:-|to|~)\s*\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\b",
        # e.g. "during 100.3 to 104.2 seconds", "from 3 to 7 s"
        r"\b(?:during|from|between|at|around|approximately|about|在|约|大约)\s+\d+(?:\.\d+)?\s*(?:-|to|~|至|到)\s*\d+(?:\.\d+)?\s*(?:s|sec(?:ond)?s?|seconds?|mins?|minutes?|分钟|秒)\b",
    ]

    point_patterns = [
        # e.g. "at 103.2 seconds", "around 1.5 min"
        r"\b(?:at|around|approximately|about|timestamp|time|在|约|大约)\s+\d+(?:\.\d+)?\s*(?:s|sec(?:ond)?s?|seconds?|mins?|minutes?|分钟|秒)\b",
        # e.g. "at 01:42", "timestamp 00:01:42"
        r"\b(?:at|around|timestamp|time)\s+\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\b",
    ]

    for pattern in interval_patterns:
        text = re.sub(pattern, " __TIME__ ", text, flags=re.IGNORECASE)
    for pattern in point_patterns:
        text = re.sub(pattern, " __TIME__ ", text, flags=re.IGNORECASE)

    # Replace context phrases around removed timestamps with neutral wording.
    text = re.sub(
        r"\b(?:during|from|between|at|around|approximately|about|within)\s+__TIME__\b",
        f" {replacement} ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:the\s+)?(?:time\s*frame|time\s*window|timestamp(?:\s*interval)?)\s*(?:of)?\s*__TIME__\b",
        f" {replacement} ",
        text,
        flags=re.IGNORECASE,
    )
    text = text.replace("__TIME__", replacement)

    # Cleanup spacing and punctuation after substitution.
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,", ", ", text)

    return text.strip()


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    """Compute advantage estimates for policy optimization."""
    adv_inputs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,
    }
    if "values" in data.batch:
        adv_inputs["values"] = data.batch["values"]

    if "reward_baselines" in data.batch:
        adv_inputs["reward_baselines"] = data.batch["reward_baselines"]

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[FunctionRewardManager] = None,
        val_reward_fn: Optional[FunctionRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor, rollout and ref
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self, model_only: bool = False) -> None:
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path, tracker_info = find_latest_ckpt(self.config.trainer.save_checkpoint_path)
            if tracker_info is not None:
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0)
                self.best_global_step = tracker_info.get("best_global_step", 0)
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            return

        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path, load_optimizer=not model_only)
        if self.use_critic and not model_only:
            critic_path = os.path.join(load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        if not model_only:
            dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
            if os.path.exists(dataloader_path):
                dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
                self.train_dataloader.load_state_dict(dataloader_state_dict)
            else:
                print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def export_huggingface_checkpoint(self, output_path: str) -> None:
        """Load an existing FSDP checkpoint and export its actor for offline evaluation."""
        self.global_step = 0
        self._load_checkpoint(model_only=True)
        if self.global_step <= 0:
            raise RuntimeError(
                "No FSDP checkpoint was loaded. Set trainer.load_checkpoint_path to global_step_<N>."
            )
        self.actor_rollout_ref_wg.export_huggingface_model(output_path)

    def _maybe_log_val_generations(
        self, inputs: list[str], outputs: list[str], labels: list[str], scores: list[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            # store generations
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
        print("Finish validation.")
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics, **val_length_metrics}

    def _balance_batch(self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _make_batch_data(self, metrics: dict[str, Any]) -> tuple[Optional[DataProto], Optional[DataProto]]:
        cycle_batch = None
        non_cycle_batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0

        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }

            DW_SOURCES = [
                'denseworld_single', 'denseworld_multiple', 'refcoco_cycle', 'grefcoco_cycle',
                'cocostuff_cycle', 'paco_part_cycle',
                'tg_multi_merged', 'dam_cyclegrpo', None,
            ]
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )

            # prepare gen_batch for generation of captioning / segmentation training
            task_dict = {
                k.replace('cap_', ''): new_batch.batch.pop(k)
                for k in ['cap_input_ids', 'cap_attention_mask', 'cap_position_ids']
            }
            task_dict.update(
                {
                    k.replace('cap_', ''): new_batch.non_tensor_batch.pop(k)
                    for k in ['cap_raw_prompt_ids', 'cap_multi_modal_data']
                }
            )
            gen_batch = DataProto.from_single_dict(task_dict, meta_info=meta_info)
            gen_batch.meta_info.update({'task': 'caption'})
            # gen_batch.meta_info["mm_processor_kwargs"] = {"fps": 0.5, "do_sample_frames": True,}

            # generate on the whole batch directly
            # gen_batch.batch.keys(): ['input_ids', 'attention_mask', 'position_ids']
            # gen_batch.non_tensor_batch.keys(): ['raw_prompt_ids', 'multi_modal_data']
            # gen_batch.meta_info: {'min_pixels': 3136, 'max_pixels': 1605632, 'video_fps': 0.5, 'task': 'caption'}
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            # repeat to align with repeated responses in rollout
            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            # filter group
            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if len(kept_sample_idxs) == 0:
                    raise RuntimeError("No sample is kept after filtering. Please check your data.")

                new_batch = new_batch[kept_sample_idxs]

            # split after rollout
            rollout_sources = new_batch.non_tensor_batch["source"]
            cycle_indices = [i for i, src in enumerate(rollout_sources) if src in DW_SOURCES]
            non_cycle_indices = [i for i, src in enumerate(rollout_sources) if src not in DW_SOURCES]

            if cycle_indices:
                cycle_part = new_batch[cycle_indices]
                cycle_batch = DataProto.concat([cycle_batch, cycle_part]) if cycle_batch is not None else cycle_part
            if non_cycle_indices:
                non_cycle_part = new_batch[non_cycle_indices]
                non_cycle_batch = (
                    DataProto.concat([non_cycle_batch, non_cycle_part]) if non_cycle_batch is not None else non_cycle_part
                )

            # 检查累积的数据是否足够
            cycle_batch_size = len(cycle_batch) // self.config.worker.rollout.n if cycle_batch is not None else 0
            non_cycle_batch_size = len(non_cycle_batch) // self.config.worker.rollout.n if non_cycle_batch is not None else 0
            total_batch_size = cycle_batch_size + non_cycle_batch_size
            rollout_batch_size = self.config.data.rollout_batch_size
            if total_batch_size < rollout_batch_size:
                print(f"{total_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise RuntimeError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{total_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

                # cycle_batch / non_cycle_batch are balanced & dispatched SEPARATELY, so each
                # must have a sample count divisible by world_size. Their sizes are data-dependent
                # (how many DW vs no-target samples landed this rollout), so trim each down to a
                # whole number of groups such that groups*n % world_size == 0. Avoids the
                # "len(seqlen_list) % k_partitions != 0" assertion in _balance_batch for any n /
                # node count. Drops at most (group_align-1) groups per sub-batch (typically <1%).
                n = self.config.worker.rollout.n
                world_size = self.actor_rollout_ref_wg.world_size
                group_align = world_size // math.gcd(n, world_size)  # group count must be a multiple of this

                def _aligned_groups(num_groups: int) -> int:
                    return (num_groups // group_align) * group_align

                cycle_groups = _aligned_groups(cycle_batch_size) if cycle_batch is not None else 0
                non_cycle_groups = _aligned_groups(non_cycle_batch_size) if non_cycle_batch is not None else 0
                if cycle_batch is not None and cycle_groups != cycle_batch_size:
                    print(f"[make_batch] trim cycle groups {cycle_batch_size}->{cycle_groups} (divisible by world_size={world_size})")
                if non_cycle_batch is not None and non_cycle_groups != non_cycle_batch_size:
                    print(f"[make_batch] trim non_cycle groups {non_cycle_batch_size}->{non_cycle_groups} (divisible by world_size={world_size})")

                cycle_batch_ret = cycle_batch[: cycle_groups * n] if cycle_groups > 0 else None
                non_cycle_batch_ret = non_cycle_batch[: non_cycle_groups * n] if non_cycle_groups > 0 else None
                return cycle_batch_ret, non_cycle_batch_ret

    def _make_seg_batch_data_for_caption(
        self,
        batch: DataProto,
        rollout_count: Optional[int] = None,
        rollout_overrides: Optional[dict[str, Any]] = None,
        seg_problem_overrides: Optional[list[str]] = None,
        source_overrides: Optional[list[str]] = None,
        localization_prompt_variant_overrides: Optional[list[str]] = None,
    ) -> DataProto:

        all_seg_problems = []
        gen_seg_batch_list = []
        for i in range(len(batch.non_tensor_batch['multi_modal_data'])):
            seg_problem = (
                seg_problem_overrides[i]
                if seg_problem_overrides is not None
                else self.tokenizer.decode(batch.batch['responses'][i], skip_special_tokens=True)
            )
            seg_mm_data = batch.non_tensor_batch['seg_multi_modal_data'][i]
            # Remove empty thinking tags if present
            seg_problem = re.sub(r'<think>\s*</think>\s*', '', seg_problem)
            # Strip vision-related markers the captioner may have echoed back; otherwise
            # they get parsed as extra <image>/<video> references and the processor
            # IndexErrors on video_metadata.
            seg_problem = re.sub(
                r'<(?:image|video)>|<\|(?:image|video)_pad\|>|<\|vision_(?:start|end)\|>',
                '',
                seg_problem,
            )
            if 'videos' in seg_mm_data:
                seg_problem = strip_temporal_priors(seg_problem)
            all_seg_problems.append(seg_problem)
            cap_mm_data = batch.non_tensor_batch['multi_modal_data'][i]
            # Use both downstream image-localization phrasings for cycle captions.
            # Direct no-target grounding passes an explicit RefCOCO/GRES variant so
            # its prompt has positive cycle counterparts instead of standing alone.
            prompt_variant = (
                localization_prompt_variant_overrides[i]
                if localization_prompt_variant_overrides is not None
                else ("refcoco" if i % 2 == 0 else "groundingsuite")
            )
            example = {'seg_problem': seg_problem,
                        'localization_prompt_variant': prompt_variant,
                        'seg_ground_truth': batch.non_tensor_batch['seg_ground_truth'][i],
                        'source': source_overrides[i] if source_overrides is not None else batch.non_tensor_batch['source'][i],
                        'masks': batch.non_tensor_batch['masks'][i],
                        'cap_ground_truth': batch.non_tensor_batch['cap_ground_truth'][i]}
            if 'images' in seg_mm_data:
                example['images'] = seg_mm_data['images']
                example['cap_images'] = cap_mm_data['images']
            elif 'videos' in seg_mm_data:
                example['videos'] = seg_mm_data['videos']
                example['nframes'] = seg_mm_data.get('nframes')
                example['cap_videos'] = cap_mm_data['videos']

            gen_seg_batch_list.append(self.train_dataloader.dataset._gen_seg_preprocess(example))

        gen_seg_batch_dict = {}
        for k in gen_seg_batch_list[0].keys():
            values = [d[k] for d in gen_seg_batch_list]
            if isinstance(values[0], torch.Tensor):
                gen_seg_batch_dict[k] = torch.stack(values, dim=0)
            else:
                gen_seg_batch_dict[k] = np.array(values, dtype=object)

        gen_seg_batch: DataProto = DataProto.from_single_dict(gen_seg_batch_dict, meta_info=batch.meta_info)
        caption_uids = np.array(
            [str(uuid.uuid4()) for _ in range(len(gen_seg_batch.batch))], dtype=object
        )
        gen_seg_batch.non_tensor_batch["uid"] = caption_uids
        batch.non_tensor_batch["caption_uid"] = caption_uids.copy()

        caption_counters = defaultdict(int)
        caption_indices = []
        for sample_uid in batch.non_tensor_batch["uid"]:
            key = str(sample_uid)
            caption_indices.append(caption_counters[key])
            caption_counters[key] += 1
        batch.non_tensor_batch["caption_index"] = np.array(caption_indices, dtype=object)

        # prepare gen_batch for generation of segmentation training
        task_dict = {k.replace(f'seg_', ''): gen_seg_batch.batch.pop(k) for k in [f'seg_input_ids', f'seg_attention_mask', f'seg_position_ids']}
        task_dict.update({k.replace(f'seg_', ''): gen_seg_batch.non_tensor_batch.pop(k) for k in [f'seg_raw_prompt_ids', f'seg_multi_modal_data']})
        gen_batch = DataProto.from_single_dict(task_dict, meta_info=batch.meta_info)
        gen_batch.meta_info.update({'task': 'segmentation'})
        # Keep malformed early rollouts bounded. The strict reward below is the
        # actual one-mask constraint; this only prevents long repeat loops from
        # wasting generation time before the policy has adapted.
        gen_batch.meta_info["max_tokens"] = self.config.worker.opsd.pixel_iou.segmentation_max_response_tokens
        gen_batch.non_tensor_batch.update({'seg_ground_truth': gen_seg_batch.non_tensor_batch['seg_ground_truth']})
        gen_batch.non_tensor_batch.update({'seg_problems': np.array(all_seg_problems, dtype=object)})
        # Store media under a unified 'media' key for downstream compatibility (images or videos)
        first_mm = gen_batch.non_tensor_batch['multi_modal_data'][0]
        media_key = 'images' if 'images' in first_mm else 'videos'
        gen_batch.non_tensor_batch.update({media_key: gen_batch.non_tensor_batch['multi_modal_data']})
        
        # generate a batch using ref (pretrained) model
        # generate_sequences_with_ref automatically swaps weights to ref model and back to actor
        ori_rollout_n = self.config.worker.rollout.n
        self.config.worker.rollout.n = rollout_count or self.config.worker.opsd.localization_rollouts
        gen_batch.meta_info["n"] = self.config.worker.rollout.n
        if rollout_overrides:
            gen_batch.meta_info.update(rollout_overrides)
        
        # gen_batch_output = self.actor_rollout_ref_wg.generate_sequences_with_ref(gen_batch)
        gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)
        # gen_batch_output.batch.keys(): ['prompts', 'responses', 'input_ids', 'attention_mask', 'response_mask', 'position_ids']
        # gen_batch_output.non_tensor_batch.keys(): ['multi_modal_data', 'mask_token_accuracy', 'format_correct']

        # Aggregate mask_token_accuracy and format_correct from gen_batch_output to batch
        # gen_batch_output has n_prompts * n_samples results, aggregate to n_prompts
        n_samples = self.config.worker.rollout.n  # 32
        n_prompts = len(gen_batch)
        
        # Add uid to gen_batch_output: repeat each uid n_samples times to match the expanded batch size
        gen_batch_output.non_tensor_batch['uid'] = np.repeat(gen_seg_batch.non_tensor_batch["uid"], n_samples)
        gen_batch_output.non_tensor_batch['caption_uid'] = np.repeat(caption_uids, n_samples)
        gen_batch_output.non_tensor_batch['sample_uid'] = np.repeat(batch.non_tensor_batch["uid"], n_samples)
        gen_batch_output.non_tensor_batch['caption_index'] = np.repeat(
            batch.non_tensor_batch["caption_index"], n_samples
        )
        gen_batch_output.non_tensor_batch['localization_index'] = np.tile(
            np.arange(n_samples), n_prompts
        ).astype(object)
        gen_batch_output.non_tensor_batch['caption_text'] = np.repeat(
            np.array(all_seg_problems, dtype=object), n_samples
        )
        gen_batch_output.non_tensor_batch['source'] = np.repeat(gen_seg_batch.non_tensor_batch["source"], n_samples)
        if "masks" in batch.non_tensor_batch:
            gen_batch_output.non_tensor_batch['raw_gt_mask'] = np.repeat(
                batch.non_tensor_batch["masks"], n_samples
            )
        
        if 'mask_token_accuracy' in gen_batch_output.non_tensor_batch:
            flat_accuracy = np.array(gen_batch_output.non_tensor_batch['mask_token_accuracy']).astype(float)
            # Compute mean accuracy for each prompt
            mean_accuracy = [
                np.mean(flat_accuracy[b * n_samples : (b + 1) * n_samples])
                for b in range(n_prompts)
            ]
            batch.non_tensor_batch['iou_scores'] = np.array(mean_accuracy, dtype=object)
            # Also add iou_scores to gen_batch_output: repeat each mean_accuracy n_samples times
            gen_batch_output.non_tensor_batch['iou_scores'] = np.repeat(mean_accuracy, n_samples).astype(object)
        
        if 'format_correct' in gen_batch_output.non_tensor_batch:
            flat_format_correct = np.array(gen_batch_output.non_tensor_batch['format_correct']).astype(int)
            # Count how many samples have correct format for each prompt
            correct_counts = [
                np.sum(flat_format_correct[b * n_samples : (b + 1) * n_samples])
                for b in range(n_prompts)
            ]
            batch.non_tensor_batch['correct_mask'] = np.array(correct_counts, dtype=object)

        self.config.worker.rollout.n = ori_rollout_n
        # batch.batch: ['prompts', 'responses', 'input_ids', 'attention_mask', 'response_mask', 'position_ids']
        # batch.non_tensor_batch: ['masks', 'source', 'seg_multi_modal_data', 'cap_ground_truth', 'seg_ground_truth', 'uid', 'multi_modal_data', 'iou_scores', 'correct_mask']
        # gen_batch_output.batch.keys(): ['prompts', 'responses', 'input_ids', 'attention_mask', 'response_mask', 'position_ids']
        # gen_batch_output.non_tensor_batch.keys(): ['multi_modal_data', 'mask_token_accuracy', 'format_correct', 'iou_scores']
        return batch, gen_batch_output

    def _make_direct_grounding_batch(self, cycle_batch: DataProto) -> Optional[DataProto]:
        """Generate a separate query-to-mask grounding GRPO batch.

        The source batch contains G caption rollouts per original image. Keep one
        row per original uid and assign fresh localization UIDs in the shared
        helper, so these samples can never affect caption-cycle R_Ci grouping.
        """
        config = self.config.worker.supervised_anchors.direct_grounding
        if not config.enabled or "grounding_query" not in cycle_batch.non_tensor_batch:
            return None
        seen_uids: set[str] = set()
        indices, queries, sources = [], [], []
        for index, (uid, source, query) in enumerate(
            zip(
                cycle_batch.non_tensor_batch["uid"],
                cycle_batch.non_tensor_batch["source"],
                cycle_batch.non_tensor_batch["grounding_query"],
            )
        ):
            uid_text = str(uid)
            if uid_text in seen_uids or not isinstance(query, str) or not query.strip():
                continue
            direct_source = direct_grounding_source(
                source,
                config.include_no_target,
                config.include_positive_sources,
                config.include_label_sources,
            )
            if direct_source is None:
                continue
            seen_uids.add(uid_text)
            indices.append(index)
            queries.append(query.strip())
            sources.append(direct_source)
        if not indices:
            return None
        aligned_count = aligned_direct_prompt_count(
            len(indices), self.actor_rollout_ref_wg.world_size
        )
        if aligned_count == 0:
            print(
                "[direct_grounding] skip "
                f"{len(indices)} prompts; need at least world_size="
                f"{self.actor_rollout_ref_wg.world_size} for equal dispatch."
            )
            return None
        if aligned_count != len(indices):
            print(
                "[direct_grounding] trim prompts "
                f"{len(indices)}->{aligned_count} "
                f"(divisible by world_size={self.actor_rollout_ref_wg.world_size})"
            )
            indices = indices[:aligned_count]
            queries = queries[:aligned_count]
            sources = sources[:aligned_count]
        direct_parent = cycle_batch[indices]
        _, direct_batch = self._make_seg_batch_data_for_caption(
            direct_parent,
            rollout_count=config.rollouts,
            seg_problem_overrides=queries,
            source_overrides=sources,
            localization_prompt_variant_overrides=alternating_localization_prompt_variants(
                len(queries)
            ),
        )
        direct_batch.non_tensor_batch["direct_grounding"] = np.ones(len(direct_batch), dtype=object)
        return direct_batch

    def _make_direct_mask_ce_batch(self, cycle_batch: DataProto) -> Optional[DataProto]:
        """Build one human-expression GT-mask CE target per original sample UID.

        This intentionally does not reuse direct GRPO rollouts: sampled direct
        responses retain their own independent GRPO advantages, whereas this
        batch teacher-forces the stored SAMTok target exactly once.
        """
        config = self.config.worker.supervised_anchors.direct_mask_ce
        if not config.enabled or "grounding_query" not in cycle_batch.non_tensor_batch:
            return None

        seen_uids: set[str] = set()
        indices: list[int] = []
        queries: list[str] = []
        for index, (uid, source, query, target) in enumerate(
            zip(
                cycle_batch.non_tensor_batch["uid"],
                cycle_batch.non_tensor_batch["source"],
                cycle_batch.non_tensor_batch["grounding_query"],
                cycle_batch.non_tensor_batch["seg_ground_truth"],
            )
        ):
            uid_text = str(uid)
            if uid_text in seen_uids:
                continue
            seen_uids.add(uid_text)
            if (
                not direct_mask_ce_source(source, config.include_positive_sources)
                or not isinstance(query, str)
                or not query.strip()
                or not isinstance(target, str)
                or not target.strip()
            ):
                continue
            indices.append(index)
            queries.append(query.strip())
        if not indices:
            return None

        dataset = self.train_dataloader.dataset
        prompt_records = []
        target_ids = []
        selected_uids = []
        for output_index, parent_index in enumerate(indices):
            parent = cycle_batch[parent_index]
            seg_media = parent.non_tensor_batch["seg_multi_modal_data"][0]
            cap_media = parent.non_tensor_batch["multi_modal_data"][0]
            example = {
                "seg_problem": queries[output_index],
                "localization_prompt_variant": alternating_localization_prompt_variants(len(indices))[output_index],
                "seg_ground_truth": parent.non_tensor_batch["seg_ground_truth"][0],
                "source": "supervised_grounding",
                "masks": parent.non_tensor_batch["masks"][0],
                "cap_ground_truth": parent.non_tensor_batch["cap_ground_truth"][0],
            }
            if "images" in seg_media:
                example["images"] = seg_media["images"]
                example["cap_images"] = cap_media["images"]
            elif "videos" in seg_media:
                # Direct mask CE is image-only in the current supervised data contract.
                continue
            else:
                continue
            record = dataset._gen_seg_preprocess(example)
            prompt_records.append(record)
            target_ids.append(
                self.tokenizer.encode(record["seg_ground_truth"], add_special_tokens=False)
            )
            selected_uids.append(str(cycle_batch.non_tensor_batch["uid"][parent_index]))
        if not prompt_records:
            return None

        eos_token_id = self.tokenizer.eos_token_id
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0]
        if eos_token_id is None:
            raise ValueError("direct_mask_ce requires tokenizer.eos_token_id.")
        response_rows, response_mask_rows, response_attention_rows = direct_mask_ce_response_fields(
            target_ids,
            int(eos_token_id),
            int(self.tokenizer.pad_token_id),
        )
        target_length = len(response_rows[0])
        responses = torch.tensor(response_rows, dtype=torch.long)
        response_masks = torch.tensor(response_mask_rows, dtype=torch.long)
        response_attention_mask = torch.tensor(response_attention_rows, dtype=torch.long)
        prompts = torch.stack([record["seg_input_ids"] for record in prompt_records])
        prompt_attention_mask = torch.stack(
            [record["seg_attention_mask"] for record in prompt_records]
        )
        prompt_position_ids = torch.stack(
            [record["seg_position_ids"] for record in prompt_records]
        )
        input_ids = torch.cat([prompts, responses], dim=-1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=-1)
        delta = torch.arange(1, target_length + 1).view(1, -1).expand(len(prompt_records), -1)
        if prompt_position_ids.ndim == 3:
            delta = delta.view(len(prompt_records), 1, -1).expand(
                len(prompt_records), prompt_position_ids.size(1), -1
            )
        position_ids = torch.cat(
            [prompt_position_ids, prompt_position_ids[..., -1:] + delta], dim=-1
        )
        return DataProto.from_dict(
            tensors={
                "prompts": prompts,
                "responses": responses,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "response_mask": response_masks,
                "position_ids": position_ids,
                "sample_weight": torch.ones(len(prompt_records), dtype=torch.float32),
            },
            non_tensors={
                "multi_modal_data": np.array(
                    [record["seg_multi_modal_data"] for record in prompt_records], dtype=object
                ),
                "uid": np.array(
                    selected_uids,
                    dtype=object,
                ),
            },
            meta_info=dict(cycle_batch.meta_info),
        )

    def _merge_pixel_iou_metadata(self, caption_batch: DataProto, segmentation_batch: DataProto) -> None:
        caption_uids = segmentation_batch.non_tensor_batch["caption_uid"]
        contexts = segmentation_batch.non_tensor_batch["privileged_context"]
        context_by_uid = {}
        for uid, context in zip(caption_uids, contexts):
            if context is not None:
                context_by_uid.setdefault(str(uid), context)

        caption_contexts = [context_by_uid[str(uid)] for uid in caption_batch.non_tensor_batch["caption_uid"]]
        safety_config = self.config.worker.opsd.caption_safety
        response_lengths = torch.sum(caption_batch.batch["response_mask"], dim=-1)
        caption_safe = []
        caption_safety_reasons = []
        caption_response_tokens = []
        caption_route_before_safety = []
        caption_forced_regenerate = []

        # Pixel IoU has already assigned the normal low/mid/high route. Inspect
        # the exact caption rollout before privileged JSD or caption PPO can use it.
        for index, context in enumerate(caption_contexts):
            response_tokens = int(response_lengths[index].item())
            response_ids = caption_batch.batch["responses"][index][:response_tokens]
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
            reason = (
                caption_safety_reason(
                    response_text,
                    response_tokens=response_tokens,
                    max_response_tokens=safety_config.max_response_tokens,
                )
                if safety_config.enabled
                else None
            )
            original_route = str(context["route"])
            forced_regenerate = bool(reason is not None and safety_config.force_regenerate)
            if forced_regenerate:
                context["route"] = "regenerate"
            context["caption_safe"] = reason is None
            context["caption_safety_reason"] = reason
            context["caption_response_tokens"] = response_tokens
            context["route_before_caption_safety"] = original_route
            caption_safe.append(reason is None)
            caption_safety_reasons.append(reason)
            caption_response_tokens.append(response_tokens)
            caption_route_before_safety.append(original_route)
            caption_forced_regenerate.append(forced_regenerate)

        caption_batch.non_tensor_batch["privileged_context"] = np.array(caption_contexts, dtype=object)
        caption_batch.non_tensor_batch["R_Ci"] = np.array(
            [context["R_Ci"] for context in caption_contexts], dtype=object
        )
        caption_batch.non_tensor_batch["route"] = np.array(
            [context["route"] for context in caption_contexts], dtype=object
        )
        caption_batch.non_tensor_batch["caption_safe"] = np.array(caption_safe, dtype=object)
        caption_batch.non_tensor_batch["caption_safety_reason"] = np.array(caption_safety_reasons, dtype=object)
        caption_batch.non_tensor_batch["caption_response_tokens"] = np.array(caption_response_tokens, dtype=object)
        caption_batch.non_tensor_batch["route_before_caption_safety"] = np.array(
            caption_route_before_safety, dtype=object
        )
        caption_batch.non_tensor_batch["caption_forced_regenerate"] = np.array(
            caption_forced_regenerate, dtype=object
        )
        for key in (
            "representative_mask",
            "best_mask",
            "iou_mean",
            "iou_std",
            "iou_min",
            "iou_max",
        ):
            caption_batch.non_tensor_batch[key] = np.array(
                [context.get(key) for context in caption_contexts], dtype=object
            )
        caption_batch.non_tensor_batch["iou_scores"] = caption_batch.non_tensor_batch["R_Ci"].copy()
        segmentation_batch.non_tensor_batch["iou_scores"] = segmentation_batch.non_tensor_batch["R_Ci"].copy()
        final_route_by_uid = {
            str(uid): route for uid, route in zip(caption_batch.non_tensor_batch["caption_uid"], caption_batch.non_tensor_batch["route"])
        }
        segmentation_batch.non_tensor_batch["route"] = np.array(
            [final_route_by_uid.get(str(uid), route) for uid, route in zip(caption_uids, segmentation_batch.non_tensor_batch["route"])],
            dtype=object,
        )

    def _prepare_direct_grounding_advantage(
        self, batch: DataProto, metrics: dict[str, Any], timing_raw: dict[str, Any]
    ) -> DataProto:
        """Score direct query-to-mask grounding without entering OPSD routing."""
        self._balance_batch(batch, metrics=metrics, logging_prefix="direct_grounding_seqlen")
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
        with timer("direct_grounding_reward", timing_raw):
            reward_ref = self.reward_fn.compute_reward.remote(self.global_step, batch, task="segmentation")
        with timer("direct_grounding_old", timing_raw):
            batch = batch.union(self.actor_rollout_ref_wg.compute_log_probs(batch))
        if self.use_reference_policy:
            with timer("direct_grounding_ref", timing_raw):
                batch = batch.union(self.actor_rollout_ref_wg.compute_ref_log_probs(batch))
        if self.use_critic:
            with timer("direct_grounding_values", timing_raw):
                batch = batch.union(self.critic_wg.compute_values(batch))
        with timer("direct_grounding_adv", timing_raw):
            reward_tensor, reward_metrics = ray.get(reward_ref)
            batch.batch["token_level_scores"] = reward_tensor
            metrics.update(
                {
                    f"reward/direct_grounding_{key}": value
                    for key, value in reduce_metrics(reward_metrics).items()
                }
            )
            if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                batch, kl_metrics = apply_kl_penalty(
                    batch, self.kl_ctrl, self.config.algorithm.kl_penalty
                )
                metrics.update({f"direct_grounding_{key}": value for key, value in kl_metrics.items()})
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
            )
        return batch

    def _build_opsd_prompt_batch(
        self,
        caption_batch: DataProto,
        indices: list[int],
        mode: str,
        caption_overrides: Optional[list[str]] = None,
    ) -> DataProto:
        if caption_overrides is not None and len(caption_overrides) != len(indices):
            raise ValueError("caption_overrides must align with privileged prompt indices.")
        records = []
        dataset = self.train_dataloader.dataset
        for position, index in enumerate(indices):
            context = dict(caption_batch.non_tensor_batch["privileged_context"][index])
            if caption_overrides is not None:
                context["student_caption"] = caption_overrides[position]
            prompt_text = format_privileged_prompt(context, mode=mode)
            original_media = caption_batch.non_tensor_batch["multi_modal_data"][index]
            original_images = original_media.get("images", [])
            if not original_images:
                raise ValueError("OPSD privileged teacher prompts require an image.")
            scene = process_image(original_images[0], None, None)
            teacher_media = {
                "images": build_privileged_teacher_images(
                    scene,
                    context,
                    include_reconstruction=(mode != "groundedness"),
                ),
            }
            records.append(
                dataset.preprocess_opsd_prompt(
                    prompt_text,
                    teacher_media,
                )
            )
        tensors = {
            key: torch.stack([record[key] for record in records])
            for key in ("input_ids", "attention_mask", "position_ids")
        }
        non_tensors = {
            "raw_prompt_ids": np.array([record["raw_prompt_ids"] for record in records], dtype=object),
            "multi_modal_data": np.array([record["multi_modal_data"] for record in records], dtype=object),
            "parent_index": np.array(indices, dtype=object),
        }
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=dict(caption_batch.meta_info))

    def _generate_groundedness_verification(
        self, caption_batch: DataProto
    ) -> dict[str, float]:
        """Verify target-bearing captions with frozen privileged visual evidence."""
        config = self.config.worker.opsd.groundedness
        rows = [disabled_groundedness() for _ in range(len(caption_batch))]
        token_masks = torch.zeros_like(caption_batch.batch["response_mask"], dtype=torch.bool)
        eligible = [
            index
            for index, source in enumerate(caption_batch.non_tensor_batch["source"])
            if str(source) != "gres_no_target" or config.no_target_enabled
        ]
        metrics = {
            "opsd/groundedness_coverage": 0.0,
            "opsd/groundedness_parse_failure_rate": 0.0,
            "opsd/unsupported_claim_rate": 0.0,
            "opsd/contradicted_claim_rate": 0.0,
            "opsd/groundedness_penalty": 0.0,
        }
        if not config.enabled or not eligible:
            caption_batch.non_tensor_batch["groundedness"] = np.array(rows, dtype=object)
            caption_batch.batch["groundedness_token_mask"] = token_masks
            return metrics

        prompts = self._build_opsd_prompt_batch(caption_batch, eligible, mode="groundedness")
        prompts.meta_info.update(
            {
                "n": 1,
                "temperature": config.temperature,
                "top_p": 1.0,
                "max_tokens": config.max_new_tokens,
                "task": "opsd_groundedness",
            }
        )
        prompts, pad_size = pad_dataproto_to_divisor(prompts, self.actor_rollout_ref_wg.world_size)
        self.actor_rollout_ref_wg.prepare_teacher_rollout_engine()
        output = self.actor_rollout_ref_wg.generate_sequences_with_teacher(prompts)
        self.actor_rollout_ref_wg.release_teacher_rollout_engine()
        output = unpad_dataproto(output, pad_size)

        response_lengths = torch.sum(output.batch["response_mask"], dim=-1)
        valid_parse = []
        unsupported_rates = []
        contradicted_rates = []
        penalties = []
        for output_index, caption_index in enumerate(eligible):
            response_ids = output.batch["responses"][output_index][: int(response_lengths[output_index].item())]
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
            caption_ids = caption_batch.batch["responses"][caption_index]
            caption_length = int(torch.sum(caption_batch.batch["response_mask"][caption_index]).item())
            caption_text = self.tokenizer.decode(
                caption_ids[:caption_length], skip_special_tokens=False
            )
            verdict = parse_groundedness_verdict(
                response_text,
                caption_text,
                max_claims=config.max_claims,
                max_claim_chars=config.max_claim_chars,
                unsupported_penalty=config.unsupported_penalty,
                contradicted_penalty=config.contradicted_penalty,
                min_checked_claims=config.min_checked_claims,
            )
            # Preserve malformed output on the driver for a bounded debug log.
            if not verdict["parse_ok"]:
                verdict = dict(verdict)
                verdict["raw_response"] = response_text
            rows[caption_index] = verdict
            aligned_mask = groundedness_token_mask(
                caption_ids[:caption_length].tolist(), self.tokenizer, verdict["claims"]
            )
            if aligned_mask:
                token_masks[caption_index, : len(aligned_mask)] = torch.tensor(
                    aligned_mask, dtype=torch.bool
                )
            if verdict["parse_ok"]:
                valid_parse.append(verdict)
                checked = max(int(verdict["checked_claim_count"]), 1)
                unsupported_rates.append(verdict["unsupported_count"] / checked)
                contradicted_rates.append(verdict["contradicted_count"] / checked)
                penalties.append(float(verdict["penalty"]))

        parsed_count = len(valid_parse)
        metrics["opsd/groundedness_coverage"] = parsed_count / max(len(eligible), 1)
        metrics["opsd/groundedness_parse_failure_rate"] = 1.0 - metrics["opsd/groundedness_coverage"]
        if parsed_count:
            metrics["opsd/unsupported_claim_rate"] = float(np.mean(unsupported_rates))
            metrics["opsd/contradicted_claim_rate"] = float(np.mean(contradicted_rates))
            metrics["opsd/groundedness_penalty"] = float(np.mean(penalties))
        caption_batch.non_tensor_batch["groundedness"] = np.array(rows, dtype=object)
        caption_batch.batch["groundedness_token_mask"] = token_masks
        self._write_groundedness_records(caption_batch)
        return metrics

    def _write_groundedness_records(self, caption_batch: DataProto) -> None:
        """Persist parsed verdicts and a bounded sample of parser failures."""
        path = os.path.join(
            self.config.trainer.save_checkpoint_path, "caption_groundedness.jsonl"
        )
        os.makedirs(self.config.trainer.save_checkpoint_path, exist_ok=True)
        failure_records_written = 0
        max_failure_records_per_step = 8
        with open(path, "a", encoding="utf-8") as file:
            for index, verdict in enumerate(caption_batch.non_tensor_batch["groundedness"]):
                parsed = bool(verdict.get("parse_ok", False))
                parse_failure = not parsed and verdict.get("parse_failure_reason") not in {
                    None,
                    "disabled",
                }
                if not parsed and not parse_failure:
                    continue
                if parse_failure and failure_records_written >= max_failure_records_per_step:
                    continue
                response_length = int(torch.sum(caption_batch.batch["response_mask"][index]).item())
                caption = self.tokenizer.decode(
                    caption_batch.batch["responses"][index][:response_length],
                    skip_special_tokens=False,
                )
                logged_verdict = dict(verdict)
                raw_response = str(logged_verdict.pop("raw_response", ""))
                record = {
                    "step": self.global_step,
                    "sample_uid": str(caption_batch.non_tensor_batch["uid"][index]),
                    "caption_index": int(caption_batch.non_tensor_batch["caption_index"][index]),
                    "route": str(caption_batch.non_tensor_batch["route"][index]),
                    "R_Ci": float(caption_batch.non_tensor_batch["R_Ci"][index]),
                    "student_caption": caption,
                    "groundedness": logged_verdict,
                }
                if parse_failure:
                    record["parse_ok"] = False
                    record["parse_failure_reason"] = verdict.get("parse_failure_reason")
                    record["discarded_claim_reasons"] = verdict.get("discarded_claim_reasons", {})
                    record["verifier_response"] = raw_response[:2048]
                    failure_records_written += 1
                file.write(
                    json.dumps(record, ensure_ascii=False)
                    + "\n"
                )

    def _verify_caption_texts(
        self,
        caption_batch: DataProto,
        parent_indices: list[int],
        captions: list[str],
    ) -> list[dict[str, Any]]:
        """Run the target-only verifier on teacher regenerate candidates."""
        config = self.config.worker.opsd.groundedness
        if not config.enabled or not captions:
            return [disabled_groundedness() for _ in captions]
        prompts = self._build_opsd_prompt_batch(
            caption_batch,
            parent_indices,
            mode="groundedness",
            caption_overrides=captions,
        )
        prompts.meta_info.update(
            {
                "n": 1,
                "temperature": config.temperature,
                "top_p": 1.0,
                "max_tokens": config.max_new_tokens,
                "task": "opsd_groundedness",
            }
        )
        prompts, pad_size = pad_dataproto_to_divisor(prompts, self.actor_rollout_ref_wg.world_size)
        self.actor_rollout_ref_wg.prepare_teacher_rollout_engine()
        output = self.actor_rollout_ref_wg.generate_sequences_with_teacher(prompts)
        self.actor_rollout_ref_wg.release_teacher_rollout_engine()
        output = unpad_dataproto(output, pad_size)
        response_lengths = torch.sum(output.batch["response_mask"], dim=-1)
        verdicts = []
        for output_index, caption in enumerate(captions):
            response_ids = output.batch["responses"][output_index][: int(response_lengths[output_index].item())]
            verdicts.append(
                parse_groundedness_verdict(
                    self.tokenizer.decode(response_ids, skip_special_tokens=False),
                    caption,
                    max_claims=config.max_claims,
                    max_claim_chars=config.max_claim_chars,
                    unsupported_penalty=config.unsupported_penalty,
                    contradicted_penalty=config.contradicted_penalty,
                    min_checked_claims=config.min_checked_claims,
                )
            )
        return verdicts

    def _build_supervised_batch(
        self,
        caption_batch: DataProto,
        selected: list[tuple[int, torch.Tensor, torch.Tensor, float]],
    ) -> Optional[DataProto]:
        if not selected:
            return None
        prompt_rows = [item[0] for item in selected]
        prompts = caption_batch[prompt_rows]
        responses = torch.stack([item[1] for item in selected])
        response_masks = torch.stack([item[2] for item in selected])
        input_ids = torch.cat([prompts.batch["prompts"], responses], dim=-1)
        attention_mask = torch.cat(
            [
                prompts.batch["attention_mask"][..., : prompts.batch["prompts"].shape[-1]],
                response_masks,
            ],
            dim=-1,
        )
        prompt_position_ids = prompts.batch["position_ids"][..., : prompts.batch["prompts"].shape[-1]]
        response_length = responses.shape[-1]
        delta = torch.arange(1, response_length + 1).view(1, -1).expand(len(selected), -1)
        if prompt_position_ids.ndim == 3:
            delta = delta.view(len(selected), 1, -1).expand(len(selected), prompt_position_ids.size(1), -1)
        response_position_ids = prompt_position_ids[..., -1:] + delta
        position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=-1)
        tensors = {
            "prompts": prompts.batch["prompts"],
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_masks,
            "position_ids": position_ids,
            "sample_weight": torch.tensor([item[3] for item in selected], dtype=torch.float32),
        }
        non_tensors = {
            "multi_modal_data": prompts.non_tensor_batch["multi_modal_data"],
            "uid": prompts.non_tensor_batch["uid"],
        }
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=dict(caption_batch.meta_info))

    def _generate_regenerate_supervision(
        self, caption_batch: DataProto
    ) -> tuple[Optional[DataProto], dict[str, float]]:
        low_indices = [
            index
            for index, route in enumerate(caption_batch.non_tensor_batch["route"])
            if route == "regenerate"
        ]
        regen_metrics = {
            "opsd/regenerate_safe_rate": 0.0,
            "opsd/regenerate_improvement_rate": 0.0,
            "opsd/teacher_target_acceptance_rate": 0.0,
            "opsd/regenerate_mean_improvement": 0.0,
            "opsd/regenerate_validated_candidate_count": 0,
            "opsd/regenerate_confident_candidate_count": 0,
            "opsd/regenerate_confident_candidate_rate": 0.0,
            "opsd/regenerate_confident_target_acceptance_rate": 0.0,
            "opsd/regenerate_grounded_candidate_count": 0,
            "opsd/regenerate_grounded_candidate_rate": 0.0,
        }
        if not low_indices:
            return None, regen_metrics
        config = self.config.worker.opsd.regenerate
        teacher_prompts = self._build_opsd_prompt_batch(caption_batch, low_indices, mode="regenerate")
        teacher_prompts.meta_info.update(
            {
                "n": config.n,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_tokens": config.max_new_tokens,
                "task": "opsd_regenerate",
            }
        )
        teacher_prompts, teacher_prompt_pad = pad_dataproto_to_divisor(
            teacher_prompts, self.actor_rollout_ref_wg.world_size
        )
        self.actor_rollout_ref_wg.prepare_teacher_rollout_engine()
        teacher_output = self.actor_rollout_ref_wg.generate_sequences_with_teacher(teacher_prompts)
        self.actor_rollout_ref_wg.release_teacher_rollout_engine()
        teacher_output = unpad_dataproto(teacher_output, teacher_prompt_pad * config.n)

        parent_indices = np.repeat(np.array(low_indices, dtype=object), config.n)
        decoded = []
        contains_sanitized_token = []
        lengths = torch.sum(teacher_output.batch["response_mask"], dim=-1)
        eos_token_ids = self.tokenizer.eos_token_id
        if not isinstance(eos_token_ids, (list, tuple, set)):
            eos_token_ids = [eos_token_ids]
        for index in range(len(teacher_output)):
            valid_ids = teacher_output.batch["responses"][index][: int(lengths[index].item())]
            body_ids = valid_ids
            if len(body_ids) > 0 and int(body_ids[-1]) in eos_token_ids:
                body_ids = body_ids[:-1]
            contains_sanitized_token.append(
                bool(torch.any(body_ids == self.tokenizer.pad_token_id).item())
            )
            decoded.append(
                re.sub(
                    r"<\|(?:im_end|endoftext)\|>",
                    "",
                    self.tokenizer.decode(
                        valid_ids,
                        skip_special_tokens=False,
                    ),
                ).strip()
            )
        safe_indices = [
            index
            for index, text in enumerate(decoded)
            if teacher_caption_is_safe(text) and not contains_sanitized_token[index]
        ]
        regen_metrics["opsd/regenerate_safe_rate"] = len(safe_indices) / max(len(decoded), 1)
        if not safe_indices:
            return None, regen_metrics
        teacher_output = teacher_output[safe_indices]
        parent_indices = parent_indices[safe_indices]
        decoded = [decoded[index] for index in safe_indices]

        groundedness_config = self.config.worker.opsd.groundedness
        candidate_verdicts = self._verify_caption_texts(
            caption_batch,
            [int(index) for index in parent_indices],
            decoded,
        )
        grounded_keep = []
        for candidate_index, (parent_index, verdict) in enumerate(
            zip(parent_indices, candidate_verdicts)
        ):
            source = str(caption_batch.non_tensor_batch["source"][int(parent_index)])
            is_no_target = source == "gres_no_target"
            if (
                not groundedness_config.enabled
                or is_no_target
                or (
                    verdict.get("parse_ok", False)
                    and float(verdict.get("groundedness_score", 0.0))
                    >= groundedness_config.min_groundedness_score
                )
            ):
                grounded_keep.append(candidate_index)
        regen_metrics["opsd/regenerate_grounded_candidate_count"] = len(grounded_keep)
        regen_metrics["opsd/regenerate_grounded_candidate_rate"] = len(grounded_keep) / max(
            len(decoded), 1
        )
        if not grounded_keep:
            return None, regen_metrics
        teacher_output = teacher_output[grounded_keep]
        parent_indices = parent_indices[grounded_keep]
        decoded = [decoded[index] for index in grounded_keep]

        def repeat_parent_field(name, default=None):
            source = caption_batch.non_tensor_batch.get(name)
            if source is None:
                return np.array([default for _ in parent_indices], dtype=object)
            return np.array([source[int(index)] for index in parent_indices], dtype=object)

        teacher_output.non_tensor_batch["uid"] = repeat_parent_field("uid")
        teacher_output.non_tensor_batch["source"] = repeat_parent_field("source")
        teacher_output.non_tensor_batch["seg_ground_truth"] = repeat_parent_field("seg_ground_truth")
        teacher_output.non_tensor_batch["cap_ground_truth"] = repeat_parent_field("cap_ground_truth")
        teacher_output.non_tensor_batch["masks"] = repeat_parent_field("masks")
        teacher_output.non_tensor_batch["multi_modal_data"] = repeat_parent_field("multi_modal_data")
        teacher_output.non_tensor_batch["seg_multi_modal_data"] = repeat_parent_field("seg_multi_modal_data")

        validation_input, validation_pad = pad_dataproto_to_divisor(
            teacher_output, self.actor_rollout_ref_wg.world_size
        )
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        validation_cap, validation_seg = self._make_seg_batch_data_for_caption(
            validation_input,
            rollout_count=config.validation_rollouts,
            rollout_overrides={"temperature": 0.0, "top_p": 1.0},
        )
        self.actor_rollout_ref_wg.release_rollout_engine()
        self.actor_rollout_ref_wg.prepare_mask_decoder()
        validation_seg = self.actor_rollout_ref_wg.compute_pixel_mask_ious(validation_seg)
        self.actor_rollout_ref_wg.release_mask_decoder()
        validation_cap = unpad_dataproto(validation_cap, validation_pad)
        validation_seg = unpad_dataproto(
            validation_seg, validation_pad * config.validation_rollouts
        )
        self._merge_pixel_iou_metadata(validation_cap, validation_seg)

        candidates_by_parent = defaultdict(list)
        confidence_config = self.config.worker.opsd.teacher_confidence
        validated_candidates = 0
        confident_candidates = 0
        for candidate_index, parent_index in enumerate(parent_indices):
            parent_index = int(parent_index)
            original_score = float(caption_batch.non_tensor_batch["R_Ci"][parent_index])
            teacher_score = float(validation_cap.non_tensor_batch["R_Ci"][candidate_index])
            improvement = teacher_score - original_score
            if improvement < config.min_improvement:
                continue
            weight = regenerate_weight(original_score, teacher_score)
            validated_candidates += 1
            if confidence_config.enabled and (
                teacher_score < confidence_config.regenerate_min_teacher_score
                or weight < confidence_config.regenerate_min_normalized_improvement
            ):
                continue
            confident_candidates += 1
            response = validation_cap.batch["responses"][candidate_index][: config.max_new_tokens]
            response_mask = validation_cap.batch["response_mask"][candidate_index][: config.max_new_tokens]
            candidates_by_parent[parent_index].append(
                (teacher_score, decoded[candidate_index], response, response_mask, weight)
            )

        per_sample = defaultdict(list)
        for parent_index, candidates in candidates_by_parent.items():
            best = max(candidates, key=lambda item: item[0])
            sample_uid = str(caption_batch.non_tensor_batch["uid"][parent_index])
            per_sample[sample_uid].append((parent_index, *best))

        selected = []
        selected_improvements = []
        for candidates in per_sample.values():
            seen = set()
            for parent_index, score, text, response, response_mask, weight in sorted(
                candidates, key=lambda item: item[1], reverse=True
            ):
                normalized = " ".join(text.lower().split())
                if normalized in seen:
                    continue
                seen.add(normalized)
                selected.append((parent_index, response, response_mask, weight))
                selected_improvements.append(score - float(caption_batch.non_tensor_batch["R_Ci"][parent_index]))
                if len(seen) >= config.max_targets_per_prompt:
                    break
        regen_metrics["opsd/regenerate_improvement_rate"] = validated_candidates / max(
            len(decoded), 1
        )
        regen_metrics["opsd/teacher_target_acceptance_rate"] = len(selected) / max(len(low_indices), 1)
        regen_metrics["opsd/regenerate_validated_candidate_count"] = validated_candidates
        regen_metrics["opsd/regenerate_confident_candidate_count"] = confident_candidates
        regen_metrics["opsd/regenerate_confident_candidate_rate"] = confident_candidates / max(
            validated_candidates, 1
        )
        regen_metrics["opsd/regenerate_confident_target_acceptance_rate"] = len(selected) / max(
            len(low_indices), 1
        )
        if selected_improvements:
            regen_metrics["opsd/regenerate_mean_improvement"] = float(np.mean(selected_improvements))
        return self._build_supervised_batch(caption_batch, selected), regen_metrics

    def _build_distillation_batch(
        self, caption_batch: DataProto
    ) -> tuple[Optional[DataProto], dict[str, float]]:
        mid_indices = [
            index
            for index, route in enumerate(caption_batch.non_tensor_batch["route"])
            if route == "on_policy_distill"
        ]
        distill_metrics = {
            "opsd/distillation_route_count": len(mid_indices),
            "opsd/distillation_confident_count": 0,
            "opsd/distillation_confident_rate": 0.0,
            "opsd/distillation_confident_R_Ci_mean": 0.0,
            "opsd/distill_grounded_rate": 0.0,
        }
        if not mid_indices:
            return None, distill_metrics
        confidence_config = self.config.worker.opsd.teacher_confidence
        if confidence_config.enabled:
            mid_indices = [
                index
                for index in mid_indices
                if float(caption_batch.non_tensor_batch["R_Ci"][index])
                >= confidence_config.distill_min_caption_score
            ]
        groundedness_config = self.config.worker.opsd.groundedness
        if groundedness_config.enabled:
            # A GT-crop-conditioned distribution is only a useful auxiliary
            # target once the actor caption has minimum cycle evidence. This
            # gate is intentionally independent from the optional historical
            # teacher-confidence ablation.
            mid_indices = [
                index
                for index in mid_indices
                if float(caption_batch.non_tensor_batch["R_Ci"][index])
                >= groundedness_config.min_distill_caption_score
            ]
        groundedness_route_count = len(mid_indices)
        groundedness_rows = caption_batch.non_tensor_batch.get("groundedness", [])
        if groundedness_config.enabled:
            mid_indices = [
                index
                for index in mid_indices
                if bool(groundedness_rows[index].get("parse_ok", False))
                and float(groundedness_rows[index].get("groundedness_score", 0.0))
                >= groundedness_config.min_groundedness_score
            ]
            distill_metrics["opsd/distill_grounded_rate"] = len(mid_indices) / max(
                groundedness_route_count, 1
            )
        distill_metrics["opsd/distillation_confident_count"] = len(mid_indices)
        distill_metrics["opsd/distillation_confident_rate"] = len(mid_indices) / max(
            distill_metrics["opsd/distillation_route_count"], 1
        )
        if mid_indices:
            distill_metrics["opsd/distillation_confident_R_Ci_mean"] = float(
                np.mean([float(caption_batch.non_tensor_batch["R_Ci"][index]) for index in mid_indices])
            )
        if not mid_indices:
            return None, distill_metrics
        student = caption_batch[mid_indices]
        teacher_prompts = self._build_opsd_prompt_batch(caption_batch, mid_indices, mode="distill")
        responses = student.batch["responses"]
        response_mask = student.batch["response_mask"]
        teacher_input_ids = torch.cat([teacher_prompts.batch["input_ids"], responses], dim=-1)
        teacher_attention_mask = torch.cat([teacher_prompts.batch["attention_mask"], response_mask], dim=-1)
        prompt_position_ids = teacher_prompts.batch["position_ids"]
        delta = torch.arange(1, responses.shape[-1] + 1).view(1, -1).expand(len(student), -1)
        if prompt_position_ids.ndim == 3:
            delta = delta.view(len(student), 1, -1).expand(len(student), prompt_position_ids.size(1), -1)
        teacher_position_ids = torch.cat(
            [prompt_position_ids, prompt_position_ids[..., -1:] + delta], dim=-1
        )
        routing = self.config.worker.opsd.routing
        distill_config = self.config.worker.opsd.distillation
        scores = torch.tensor(
            [float(caption_batch.non_tensor_batch["R_Ci"][index]) for index in mid_indices],
            dtype=torch.float32,
        )
        weights = torch.tensor(
            [
                distillation_weight(
                    score,
                    routing.low_threshold,
                    routing.high_threshold,
                    distill_config.min_sample_weight,
                )
                for score in scores.tolist()
            ],
            dtype=torch.float32,
        )
        student.batch["teacher_input_ids"] = teacher_input_ids
        student.batch["teacher_attention_mask"] = teacher_attention_mask
        student.batch["teacher_position_ids"] = teacher_position_ids
        student.batch["distill_weight"] = weights
        if groundedness_config.enabled and groundedness_config.token_jsd_enabled:
            base_mask = student.batch["groundedness_token_mask"].to(dtype=torch.float32)
            student.batch["groundedness_token_weight"] = 1.0 + (
                groundedness_config.token_jsd_multiplier * base_mask
            )
        student.non_tensor_batch["teacher_multi_modal_data"] = teacher_prompts.non_tensor_batch[
            "multi_modal_data"
        ]
        return student, distill_metrics

    def _select_teacher_analysis_indices(self, caption_batch: DataProto) -> list[int]:
        config = self.config.worker.opsd.teacher_analysis
        if (
            not config.enabled
            or not self.config.worker.opsd.routing.enabled
            or not self.config.worker.opsd.ema_teacher.enabled
            or config.max_samples_per_step == 0
        ):
            return []
        routes = caption_batch.non_tensor_batch["route"]
        scores = caption_batch.non_tensor_batch["R_Ci"]
        selected = []
        for route in ("regenerate", "on_policy_distill"):
            candidates = [index for index, value in enumerate(routes) if value == route]
            if candidates:
                selected.append(min(candidates, key=lambda index: float(scores[index])))
        remaining = sorted(
            [
                index
                for index, value in enumerate(routes)
                if value in {"regenerate", "on_policy_distill"} and index not in selected
            ],
            key=lambda index: float(scores[index]),
        )
        return (selected + remaining)[: config.max_samples_per_step]

    def _write_teacher_diagnoses(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        os.makedirs(self.config.trainer.save_checkpoint_path, exist_ok=True)
        path = os.path.join(self.config.trainer.save_checkpoint_path, "teacher_diagnoses.jsonl")
        with open(path, "a", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _generate_teacher_diagnoses(self, caption_batch: DataProto) -> list[dict[str, Any]]:
        indices = self._select_teacher_analysis_indices(caption_batch)
        if not indices:
            return []
        config = self.config.worker.opsd.teacher_analysis
        prompts = self._build_opsd_prompt_batch(caption_batch, indices, mode="analysis")
        prompts.meta_info.update(
            {
                "n": 1,
                "temperature": config.temperature,
                "top_p": 1.0,
                "max_tokens": config.max_new_tokens,
                "task": "opsd_teacher_analysis",
            }
        )
        prompts, pad_size = pad_dataproto_to_divisor(prompts, self.actor_rollout_ref_wg.world_size)
        self.actor_rollout_ref_wg.prepare_teacher_rollout_engine()
        output = self.actor_rollout_ref_wg.generate_sequences_with_teacher(prompts)
        self.actor_rollout_ref_wg.release_teacher_rollout_engine()
        output = unpad_dataproto(output, pad_size)

        records = []
        response_lengths = torch.sum(output.batch["response_mask"], dim=-1)
        for output_index, caption_index in enumerate(indices):
            response_ids = output.batch["responses"][output_index][: int(response_lengths[output_index].item())]
            analysis = self.tokenizer.decode(response_ids, skip_special_tokens=False)
            analysis = re.sub(r"<\|[^|]+\|>", "", analysis).strip()
            context = caption_batch.non_tensor_batch["privileged_context"][caption_index]
            records.append(
                {
                    "step": self.global_step,
                    "route": str(caption_batch.non_tensor_batch["route"][caption_index]),
                    "sample_uid": str(caption_batch.non_tensor_batch["uid"][caption_index]),
                    "caption_index": int(caption_batch.non_tensor_batch["caption_index"][caption_index]),
                    "R_Ci": float(context["R_Ci"]),
                    "pixel_ious": [float(value) for value in context["pixel_ious"]],
                    "student_caption": str(context["student_caption"]),
                    "analysis": analysis,
                }
            )
        return records

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # make a batch of data
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    cycle_batch, non_cycle_batch = self._make_batch_data(metrics=metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()

                # balance the number of valid tokens on each dp rank.
                # NOTE: this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                # 对分离后的每个子集分别进行 balance
                if non_cycle_batch is not None:
                    self._balance_batch(non_cycle_batch, metrics=metrics)
                    # compute global valid tokens
                    non_cycle_batch.meta_info["global_token_num"] = torch.sum(non_cycle_batch.batch["attention_mask"], dim=-1).tolist()

                if cycle_batch is not None:
                    self._balance_batch(cycle_batch, metrics=metrics)
                    # compute global valid tokens
                    cycle_batch.meta_info["global_token_num"] = torch.sum(cycle_batch.batch["attention_mask"], dim=-1).tolist()

                if non_cycle_batch is not None:

                    if "token_level_scores" not in non_cycle_batch.batch:
                        with timer("reward", timing_raw):
                            reward_ref = self.reward_fn.compute_reward.remote(self.global_step, non_cycle_batch, task='caption')

                    # recompute old_log_probs
                    with timer("old", timing_raw):
                        # batch.batch.keys(): ['prompts', 'responses', 'input_ids', 'attention_mask', 'response_mask', 'position_ids']
                        # batch.non_tensor_batch.keys(): ['masks', 'source', 'cap_ground_truth', 'seg_ground_truth', 'uid', 'multi_modal_data', 'iou_scores', 'correct_mask']
                        old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(non_cycle_batch) 
                        non_cycle_batch = non_cycle_batch.union(old_log_probs)

                    # compute ref_log_probs
                    if self.use_reference_policy:
                        with timer("ref", timing_raw):
                            ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(non_cycle_batch)
                            non_cycle_batch = non_cycle_batch.union(ref_log_probs)

                    # compute values
                    if self.use_critic:
                        with timer("values", timing_raw):
                            values = self.critic_wg.compute_values(non_cycle_batch)
                            non_cycle_batch = non_cycle_batch.union(values)

                    with timer("adv", timing_raw):
                        if "token_level_scores" not in non_cycle_batch.batch:
                            # get token level scores asynchronously
                            reward_tensor, reward_metrics = ray.get(reward_ref)
                            non_cycle_batch.batch["token_level_scores"] = reward_tensor  # [8, 8192]
                            reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                            metrics.update(reward_metrics)

                        # apply kl penalty if available
                        if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                            # apply kl penalty to reward
                            non_cycle_batch, kl_metrics = apply_kl_penalty(non_cycle_batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            non_cycle_batch.batch["token_level_rewards"] = non_cycle_batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        non_cycle_batch = compute_advantage(
                            non_cycle_batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                        )

                    # update critic
                    if self.use_critic:
                        with timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(non_cycle_batch)

                        critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                        metrics.update(critic_metrics)
                    
                    # Clear multi-modal cache before switching to segmentation task
                    self.actor_rollout_ref_wg.clear_multi_modal_cache()

                cycle_cap_batch = None
                cycle_seg_batch = None
                regenerate_batch = None
                distillation_batch = None
                seg_batch = None
                direct_grounding_batch = None
                direct_mask_ce_batch = None
                if cycle_batch is not None and (
                    self.config.worker.actor.optimize_captioner or self.config.worker.actor.optimize_segmenter
                ):
                    # build both cycle caption and cycle segmentation batches once
                    with timer("gen", timing_raw):
                        cycle_batch.meta_info["n"] = self.config.worker.opsd.localization_rollouts
                        self.actor_rollout_ref_wg.prepare_rollout_engine()
                        cycle_cap_batch, cycle_seg_batch = self._make_seg_batch_data_for_caption(cycle_batch)
                        direct_batches = [
                            direct_batch
                            for parent_batch in (cycle_batch, non_cycle_batch)
                            if parent_batch is not None
                            for direct_batch in [self._make_direct_grounding_batch(parent_batch)]
                            if direct_batch is not None
                        ]
                        if direct_batches:
                            direct_grounding_batch = (
                                direct_batches[0]
                                if len(direct_batches) == 1
                                else DataProto.concat(direct_batches)
                            )
                        direct_mask_ce_batch = self._make_direct_mask_ce_batch(cycle_batch)
                        cycle_cap_batch.meta_info.pop("n", None)
                        self.actor_rollout_ref_wg.release_rollout_engine()
                        if self.config.worker.opsd.enabled and self.config.worker.opsd.pixel_iou.enabled:
                            self.actor_rollout_ref_wg.prepare_mask_decoder()
                            cycle_seg_batch = self.actor_rollout_ref_wg.compute_pixel_mask_ious(cycle_seg_batch)
                            if direct_grounding_batch is not None:
                                # The decoder explicitly skips direct no-target groups; their
                                # reward remains the established ``No target.`` refusal reward.
                                direct_grounding_batch = self.actor_rollout_ref_wg.compute_pixel_mask_ious(
                                    direct_grounding_batch
                                )
                                direct_grounding_batch.non_tensor_batch["iou_scores"] = (
                                    direct_grounding_batch.non_tensor_batch["pixel_iou"].copy()
                                )
                            self.actor_rollout_ref_wg.release_mask_decoder()
                            self._merge_pixel_iou_metadata(cycle_cap_batch, cycle_seg_batch)
                            route_values = list(cycle_cap_batch.non_tensor_batch["route"])
                            for route_name in ("regenerate", "on_policy_distill", "grpo"):
                                metrics[f"opsd/route_{route_name}_count"] = route_values.count(route_name)
                                metrics[f"opsd/route_{route_name}_rate"] = route_values.count(route_name) / max(
                                    len(route_values), 1
                                )
                            caption_safety_reasons = list(
                                cycle_cap_batch.non_tensor_batch["caption_safety_reason"]
                            )
                            caption_safe_values = np.asarray(
                                cycle_cap_batch.non_tensor_batch["caption_safe"], dtype=bool
                            )
                            forced_regenerate = np.asarray(
                                cycle_cap_batch.non_tensor_batch["caption_forced_regenerate"], dtype=bool
                            )
                            base_grpo_active = np.asarray(
                                [
                                    uses_original_grpo(
                                        route,
                                        caption_safe=caption_safe,
                                        preserve_original_grpo=self.config.worker.opsd.routing.preserve_original_grpo,
                                    )
                                    for route, caption_safe in zip(route_values, caption_safe_values)
                                ],
                                dtype=bool,
                            )
                            metrics.update(
                                {
                                    "opsd/caption_safe_rate": float(np.mean(caption_safe_values)),
                                    "opsd/caption_unsafe_rate": float(np.mean(~caption_safe_values)),
                                    "opsd/caption_special_token_rate": caption_safety_reasons.count("special_token")
                                    / max(len(caption_safety_reasons), 1),
                                    "opsd/caption_mask_json_rate": caption_safety_reasons.count("mask_json")
                                    / max(len(caption_safety_reasons), 1),
                                    "opsd/caption_overlength_rate": caption_safety_reasons.count("overlength")
                                    / max(len(caption_safety_reasons), 1),
                                    "opsd/caption_forced_regenerate_count": int(np.sum(forced_regenerate)),
                                    "opsd/caption_original_grpo_active_count": int(np.sum(base_grpo_active)),
                                    "opsd/caption_original_grpo_active_rate": float(np.mean(base_grpo_active)),
                                }
                            )
                            pixel_values = cycle_seg_batch.non_tensor_batch["pixel_iou"].astype(float)
                            caption_scores = cycle_cap_batch.non_tensor_batch["R_Ci"].astype(float)
                            seg_group_counts = np.asarray(
                                cycle_seg_batch.non_tensor_batch["mask_group_count"], dtype=float
                            )
                            seg_exactly_one = np.asarray(
                                cycle_seg_batch.non_tensor_batch["exactly_one_mask_group"], dtype=bool
                            )
                            seg_extra_groups = np.asarray(
                                cycle_seg_batch.non_tensor_batch["extra_mask_group_count"], dtype=float
                            )
                            metrics.update(
                                {
                                    "opsd/pixel_iou_mean": float(np.mean(pixel_values)),
                                    "opsd/pixel_iou_std": float(np.std(pixel_values)),
                                    "opsd/pixel_iou_min": float(np.min(pixel_values)),
                                    "opsd/pixel_iou_max": float(np.max(pixel_values)),
                                    "opsd/R_Ci_mean": float(np.mean(caption_scores)),
                                    "opsd/R_Ci_std": float(np.std(caption_scores)),
                                    "opsd/seg_exactly_one_mask_rate": float(np.mean(seg_exactly_one)),
                                    "opsd/seg_multi_mask_rate": float(np.mean(seg_group_counts > 1)),
                                    "opsd/seg_mean_mask_group_count": float(np.mean(seg_group_counts)),
                                    "opsd/seg_mean_extra_mask_group_count": float(np.mean(seg_extra_groups)),
                                }
                            )
                            if direct_grounding_batch is not None:
                                direct_sources = np.asarray(
                                    direct_grounding_batch.non_tensor_batch["source"], dtype=object
                                )
                                positive_direct = direct_sources == "supervised_grounding"
                                if np.any(positive_direct):
                                    direct_group_counts = np.asarray(
                                        direct_grounding_batch.non_tensor_batch["mask_group_count"], dtype=float
                                    )[positive_direct]
                                    direct_exactly_one = np.asarray(
                                        direct_grounding_batch.non_tensor_batch["exactly_one_mask_group"], dtype=bool
                                    )[positive_direct]
                                    metrics.update(
                                        {
                                            "supervised_anchors/direct_exactly_one_mask_rate": float(np.mean(direct_exactly_one)),
                                            "supervised_anchors/direct_multi_mask_rate": float(np.mean(direct_group_counts > 1)),
                                            "supervised_anchors/direct_mean_mask_group_count": float(np.mean(direct_group_counts)),
                                        }
                                    )
                            references = list(cycle_seg_batch.non_tensor_batch["iou_reference"])
                            metrics["opsd/raw_gt_reference_rate"] = references.count("raw_gt") / max(
                                len(references), 1
                            )
                            groundedness_metrics = self._generate_groundedness_verification(
                                cycle_cap_batch
                            )
                            metrics.update(groundedness_metrics)
                            teacher_diagnoses = self._generate_teacher_diagnoses(cycle_cap_batch)
                            self._write_teacher_diagnoses(teacher_diagnoses)
                            metrics["opsd/teacher_analysis_count"] = len(teacher_diagnoses)
                            distillation_batch, distillation_metrics = self._build_distillation_batch(
                                cycle_cap_batch
                            )
                            metrics.update(distillation_metrics)
                            regenerate_batch, regenerate_metrics = self._generate_regenerate_supervision(
                                cycle_cap_batch
                            )
                            metrics.update(regenerate_metrics)

                if cycle_cap_batch is not None and self.config.worker.actor.optimize_captioner:

                    if "token_level_scores" not in cycle_cap_batch.batch:
                        with timer("reward", timing_raw):
                            reward_ref = self.reward_fn.compute_reward.remote(self.global_step, cycle_cap_batch, task='caption')

                    # recompute old_log_probs
                    with timer("old", timing_raw):
                        # cycle_cap_batch.batch.keys(): ['prompts', 'responses', 'input_ids', 'attention_mask', 'response_mask', 'position_ids']
                        # cycle_cap_batch.non_tensor_batch.keys(): ['masks', 'source', 'cap_ground_truth', 'seg_ground_truth', 'uid', 'multi_modal_data', 'iou_scores', 'correct_mask']
                        old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(cycle_cap_batch)
                        cycle_cap_batch = cycle_cap_batch.union(old_log_probs)

                    # compute ref_log_probs
                    if self.use_reference_policy:
                        with timer("ref", timing_raw):
                            ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(cycle_cap_batch)
                            cycle_cap_batch = cycle_cap_batch.union(ref_log_probs)

                    # compute values
                    if self.use_critic:
                        with timer("values", timing_raw):
                            values = self.critic_wg.compute_values(cycle_cap_batch)
                            cycle_cap_batch = cycle_cap_batch.union(values)

                    with timer("adv", timing_raw):
                        if "token_level_scores" not in cycle_cap_batch.batch:
                            # get token level scores asynchronously
                            reward_tensor, reward_metrics = ray.get(reward_ref)
                            cycle_cap_batch.batch["token_level_scores"] = reward_tensor  # [8, 8192]
                            reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                            metrics.update(reward_metrics)

                        # apply kl penalty if available
                        if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                            # apply kl penalty to reward
                            cycle_cap_batch, kl_metrics = apply_kl_penalty(cycle_cap_batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            cycle_cap_batch.batch["token_level_rewards"] = cycle_cap_batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        cycle_cap_batch = compute_advantage(
                            cycle_cap_batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                        )
                        if "route" in cycle_cap_batch.non_tensor_batch:
                            cycle_cap_batch.batch["policy_loss_mask"] = torch.tensor(
                                [
                                    uses_original_grpo(
                                        route,
                                        caption_safe=caption_safe,
                                        preserve_original_grpo=self.config.worker.opsd.routing.preserve_original_grpo,
                                    )
                                    for route, caption_safe in zip(
                                        cycle_cap_batch.non_tensor_batch["route"],
                                        cycle_cap_batch.non_tensor_batch["caption_safe"],
                                    )
                                ],
                                dtype=cycle_cap_batch.batch["response_mask"].dtype,
                            )
                        if (
                            self.config.worker.opsd.caption_anchor_kl_coef > 0
                            and self.config.worker.opsd.caption_anchor_kl_all_safe_routes
                            and self.use_reference_policy
                        ):
                            anchor_kl_values = np.asarray(
                                cycle_cap_batch.non_tensor_batch["caption_safe"], dtype=bool
                            )
                            cycle_cap_batch.batch["caption_anchor_kl_mask"] = torch.tensor(
                                anchor_kl_values,
                                dtype=cycle_cap_batch.batch["response_mask"].dtype,
                            )
                            metrics["opsd/caption_anchor_kl_active_count"] = int(
                                np.sum(anchor_kl_values)
                            )
                            metrics["opsd/caption_anchor_kl_active_rate"] = float(
                                np.mean(anchor_kl_values)
                            )

                    # update critic
                    if self.use_critic:
                        with timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(cycle_cap_batch)

                        critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                        metrics.update(critic_metrics)
                    
                    # Clear multi-modal cache before switching to segmentation task
                    self.actor_rollout_ref_wg.clear_multi_modal_cache()

                if cycle_seg_batch is not None and self.config.worker.actor.optimize_segmenter:
                    ## Start Segmentation Session
                    # seg_batch.batch.keys(): ['prompts', 'responses', 'input_ids', 'attention_mask', 'response_mask', 'position_ids']
                    # seg_batch.non_tensor_batch.keys(): ['multi_modal_data', 'mask_token_accuracy', 'format_correct', 'iou_scores', 'uid']
                    self._balance_batch(cycle_seg_batch, metrics=metrics)
                    cycle_seg_batch.meta_info["global_token_num"] = torch.sum(cycle_seg_batch.batch["attention_mask"], dim=-1).tolist()

                    # compute reward
                    if cycle_seg_batch is not None and "token_level_scores" not in cycle_seg_batch.batch:
                        with timer("reward", timing_raw):
                            seg_reward_ref = self.reward_fn.compute_reward.remote(self.global_step, cycle_seg_batch, task='segmentation')
                    
                    # recompute old_log_probs
                    if cycle_seg_batch is not None:
                        with timer("old", timing_raw):
                            old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(cycle_seg_batch) 
                            cycle_seg_batch = cycle_seg_batch.union(old_log_probs)

                    # compute ref_log_probs
                    if cycle_seg_batch is not None and self.use_reference_policy:
                        with timer("ref", timing_raw):
                            ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(cycle_seg_batch)
                            cycle_seg_batch = cycle_seg_batch.union(ref_log_probs)

                    # compute values
                    if cycle_seg_batch is not None and self.use_critic:
                        with timer("values", timing_raw):
                            values = self.critic_wg.compute_values(cycle_seg_batch)
                            cycle_seg_batch = cycle_seg_batch.union(values)

                    if cycle_seg_batch is not None:
                        with timer("adv", timing_raw):
                            if "token_level_scores" not in cycle_seg_batch.batch:
                                # get token level scores asynchronously
                                reward_tensor, reward_metrics = ray.get(seg_reward_ref)
                                cycle_seg_batch.batch["token_level_scores"] = reward_tensor
                                reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                                metrics.update(reward_metrics)

                            # apply kl penalty if available
                            if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                                # apply kl penalty to reward
                                cycle_seg_batch, kl_metrics = apply_kl_penalty(cycle_seg_batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                                metrics.update(kl_metrics)
                            else:
                                cycle_seg_batch.batch["token_level_rewards"] = cycle_seg_batch.batch["token_level_scores"]

                            # compute advantages, executed on the driver process
                            cycle_seg_batch = compute_advantage(
                                cycle_seg_batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                            )
                            if (
                                self.config.worker.opsd.segmentation_anchor_kl_coef > 0
                                and self.use_reference_policy
                            ):
                                segmentation_anchor_values = np.ones(len(cycle_seg_batch), dtype=bool)
                                cycle_seg_batch.batch["segmentation_anchor_kl_mask"] = torch.tensor(
                                    segmentation_anchor_values,
                                    dtype=cycle_seg_batch.batch["response_mask"].dtype,
                                )
                                metrics["opsd/segmentation_anchor_kl_active_count"] = int(
                                    np.sum(segmentation_anchor_values)
                                )
                                metrics["opsd/segmentation_anchor_kl_active_rate"] = float(
                                    np.mean(segmentation_anchor_values)
                                )

                    # update critic
                    if cycle_seg_batch is not None and self.use_critic:
                        with timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(cycle_seg_batch)

                        critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                        metrics.update(critic_metrics)
                    
                    # Clear multi-modal cache before switching to segmentation task
                    self.actor_rollout_ref_wg.clear_multi_modal_cache()

                    seg_batch = cycle_seg_batch

                if direct_grounding_batch is not None and self.config.worker.actor.optimize_segmenter:
                    direct_grounding_batch = self._prepare_direct_grounding_advantage(
                        direct_grounding_batch, metrics, timing_raw
                    )
                    metrics["supervised_anchors/direct_grounding_rollouts"] = len(direct_grounding_batch)
                    metrics["supervised_anchors/direct_grounding_prompt_count"] = (
                        len(direct_grounding_batch)
                        // self.config.worker.supervised_anchors.direct_grounding.rollouts
                    )

                # Groundedness is fully consumed by the reward and optional
                # auxiliary JSD batch. Keep it off every main PPO actor batch.
                if cycle_cap_batch is not None:
                    for key in ("groundedness_token_mask", "groundedness_token_weight"):
                        if key in cycle_cap_batch.batch:
                            cycle_cap_batch.batch.pop(key)
                    cycle_cap_batch.non_tensor_batch.pop("groundedness", None)

                # Concatenate non_cycle_batch and cycle_cap_batch into cap_batch.
                if non_cycle_batch is not None and cycle_cap_batch is not None:
                    if "policy_loss_mask" in cycle_cap_batch.batch:
                        non_cycle_batch.batch["policy_loss_mask"] = torch.ones(
                            len(non_cycle_batch), dtype=non_cycle_batch.batch["response_mask"].dtype
                        )
                    if "caption_anchor_kl_mask" in cycle_cap_batch.batch:
                        # The OPSD anchor is caption-cycle specific. Non-cycle tasks
                        # retain their normal algorithm KL but receive no extra anchor.
                        non_cycle_batch.batch["caption_anchor_kl_mask"] = torch.zeros(
                            len(non_cycle_batch), dtype=non_cycle_batch.batch["response_mask"].dtype
                        )
                    # Remove iou_scores and correct_mask from batch before concat
                    for key in (
                        "iou_scores",
                        "correct_mask",
                        "caption_uid",
                        "caption_index",
                        "privileged_context",
                        "R_Ci",
                        "route",
                        "caption_safe",
                        "caption_safety_reason",
                        "caption_response_tokens",
                        "route_before_caption_safety",
                        "caption_forced_regenerate",
                        "groundedness",
                        "representative_mask",
                        "best_mask",
                        "iou_mean",
                        "iou_std",
                        "iou_min",
                        "iou_max",
                    ):
                        cycle_cap_batch.non_tensor_batch.pop(key, None)
                    cap_batch = DataProto.concat([non_cycle_batch, cycle_cap_batch])
                elif non_cycle_batch is not None:
                    cap_batch = non_cycle_batch
                elif cycle_cap_batch is not None:
                    cap_batch = cycle_cap_batch
                else:
                    cap_batch = None

                if self.config.worker.actor.optimize_segmenter and self.config.worker.actor.optimize_captioner:
                    # Case 1: Both tasks - Use gradient accumulation for cap_batch and seg_batch
                    cap_batch_size = len(cap_batch) if cap_batch is not None else 0
                    seg_batch_size = len(seg_batch) if seg_batch is not None else 0
                    direct_grounding_size = len(direct_grounding_batch) if direct_grounding_batch is not None else 0
                    direct_mask_ce_size = len(direct_mask_ce_batch) if direct_mask_ce_batch is not None else 0
                    total_size = cap_batch_size + seg_batch_size + direct_grounding_size + direct_mask_ce_size
                    
                    cap_grad_weight = self.config.worker.opsd.caption_loss_weight
                    seg_grad_weight = self.config.worker.opsd.localization_loss_weight
                    direct_config = self.config.worker.supervised_anchors.direct_grounding
                    direct_target_weight = direct_config.loss_weight if direct_config.enabled else 0.0
                    direct_grad_weight = direct_grounding_loss_weight(
                        self.global_step,
                        direct_target_weight,
                        direct_config.warmup_start_step,
                        direct_config.warmup_end_step,
                    )
                    direct_mask_ce_config = self.config.worker.supervised_anchors.direct_mask_ce
                    metrics.update(
                        {
                            "supervised_anchors/direct_loss_weight_effective": direct_grad_weight,
                            "supervised_anchors/direct_loss_weight_target": direct_target_weight,
                            "supervised_anchors/direct_mask_ce_weight": (
                                direct_mask_ce_config.loss_weight
                                if direct_mask_ce_config.enabled
                                else 0.0
                            ),
                            "supervised_anchors/direct_mask_ce_samples": direct_mask_ce_size,
                        }
                    )

                    if self.config.trainer.critic_warmup <= self.global_step:
                        self.actor_rollout_ref_wg.clear_multi_modal_cache()
                        
                        with timer("update_actor", timing_raw):
                            actor_metrics = {}
                            
                            # Step 1: Accumulate gradients from cap_batch (combined non_single + single)
                            if cap_batch is not None and cap_batch_size > 0:
                                cap_batch.meta_info['grad_weight'] = cap_grad_weight
                                cap_batch.meta_info['global_batch_size_per_device'] = len(cap_batch) // self.actor_rollout_ref_wg.world_size
                                cap_output = self.actor_rollout_ref_wg.accumulate_actor_gradients(cap_batch)
                                actor_metrics.update({f"cap_{k}": v for k, v in reduce_metrics(cap_output.non_tensor_batch).items()})
                            
                            def accumulate_caption_auxiliary_gradients() -> None:
                                if regenerate_batch is not None and len(regenerate_batch) > 0:
                                    regen_count = len(regenerate_batch)
                                    regen_batch, regen_pad = pad_dataproto_to_divisor(
                                        regenerate_batch,
                                        self.actor_rollout_ref_wg.world_size
                                        * self.config.worker.actor.micro_batch_size_per_device_for_update,
                                    )
                                    if regen_pad:
                                        regen_batch.batch["sample_weight"][-regen_pad:] = 0.0
                                    self._balance_batch(regen_batch, metrics=metrics, logging_prefix="regen_seqlen")
                                    regen_batch.meta_info["global_token_num"] = torch.sum(
                                        regen_batch.batch["attention_mask"], dim=-1
                                    ).tolist()
                                    regen_batch.meta_info["grad_weight"] = (
                                        self.config.worker.opsd.caption_loss_weight
                                        * regen_count
                                        / max(len(cycle_batch), 1)
                                    )
                                    regen_output = self.actor_rollout_ref_wg.accumulate_supervised_gradients(
                                        regen_batch
                                    )
                                    actor_metrics.update(
                                        {
                                            f"regen_{key}": value
                                            for key, value in reduce_metrics(
                                                regen_output.non_tensor_batch
                                            ).items()
                                        }
                                    )

                                if distillation_batch is not None and len(distillation_batch) > 0:
                                    distill_count = len(distillation_batch)
                                    distill_batch, distill_pad = pad_dataproto_to_divisor(
                                        distillation_batch,
                                        self.actor_rollout_ref_wg.world_size
                                        * self.config.worker.actor.micro_batch_size_per_device_for_update,
                                    )
                                    if distill_pad:
                                        distill_batch.batch["distill_weight"][-distill_pad:] = 0.0
                                    self._balance_batch(
                                        distill_batch, metrics=metrics, logging_prefix="distill_seqlen"
                                    )
                                    distill_batch.meta_info["global_token_num"] = torch.sum(
                                        distill_batch.batch["attention_mask"], dim=-1
                                    ).tolist()
                                    distill_batch.meta_info["grad_weight"] = (
                                        self.config.worker.opsd.caption_loss_weight
                                        * distill_count
                                        / max(len(cycle_batch), 1)
                                    )
                                    distill_output = self.actor_rollout_ref_wg.accumulate_distillation_gradients(
                                        distill_batch
                                    )
                                    actor_metrics.update(
                                        {
                                            f"distill_{key}": value
                                            for key, value in reduce_metrics(
                                                distill_output.non_tensor_batch
                                            ).items()
                                        }
                                    )

                            projection_enabled = bool(
                                self.config.worker.opsd.asymmetric_gradient_projection
                                and cap_batch_size > 0
                                and seg_batch_size > 0
                            )
                            if projection_enabled:
                                # Keep every caption-side loss separate before taking segmentation gradients.
                                accumulate_caption_auxiliary_gradients()
                                stash_output = self.actor_rollout_ref_wg.stash_actor_caption_gradients()
                                if stash_output and hasattr(stash_output[0], "non_tensor_batch"):
                                    actor_metrics.update(reduce_metrics(stash_output[0].non_tensor_batch))

                            # Step 2: Accumulate gradients from the localization batch.
                            if seg_batch is not None and seg_batch_size > 0:
                                seg_batch.meta_info['grad_weight'] = seg_grad_weight
                                seg_batch.meta_info['global_batch_size_per_device'] = len(seg_batch) // self.actor_rollout_ref_wg.world_size
                                seg_output = self.actor_rollout_ref_wg.accumulate_actor_gradients(seg_batch)
                                actor_metrics.update({f"seg_{k}": v for k, v in reduce_metrics(seg_output.non_tensor_batch).items()})

                            if direct_grounding_batch is not None and direct_grounding_size > 0:
                                direct_grounding_batch.meta_info["grad_weight"] = direct_grad_weight
                                direct_grounding_batch.meta_info["global_batch_size_per_device"] = (
                                    len(direct_grounding_batch) // self.actor_rollout_ref_wg.world_size
                                )
                                direct_output = self.actor_rollout_ref_wg.accumulate_actor_gradients(
                                    direct_grounding_batch
                                )
                                actor_metrics.update(
                                    {
                                        f"direct_grounding_{key}": value
                                        for key, value in reduce_metrics(direct_output.non_tensor_batch).items()
                                    }
                                )

                            if direct_mask_ce_batch is not None and direct_mask_ce_size > 0:
                                ce_batch, ce_pad = pad_dataproto_to_divisor(
                                    direct_mask_ce_batch,
                                    self.actor_rollout_ref_wg.world_size
                                    * self.config.worker.actor.micro_batch_size_per_device_for_update,
                                )
                                if ce_pad:
                                    ce_batch.batch["sample_weight"][-ce_pad:] = 0.0
                                self._balance_batch(
                                    ce_batch,
                                    metrics=metrics,
                                    logging_prefix="direct_mask_ce_seqlen",
                                )
                                ce_batch.meta_info["global_token_num"] = torch.sum(
                                    ce_batch.batch["attention_mask"], dim=-1
                                ).tolist()
                                ce_batch.meta_info["grad_weight"] = direct_mask_ce_config.loss_weight
                                ce_batch.meta_info["global_batch_size_per_device"] = (
                                    len(ce_batch) // self.actor_rollout_ref_wg.world_size
                                )
                                ce_output = self.actor_rollout_ref_wg.accumulate_direct_mask_ce_gradients(
                                    ce_batch
                                )
                                actor_metrics.update(reduce_metrics(ce_output.non_tensor_batch))

                            if not projection_enabled:
                                # Preserve the historical accumulation order when projection is disabled.
                                accumulate_caption_auxiliary_gradients()
                            else:
                                projection_output = self.actor_rollout_ref_wg.merge_asymmetric_actor_gradients()
                                if projection_output and hasattr(projection_output[0], "non_tensor_batch"):
                                    actor_metrics.update(reduce_metrics(projection_output[0].non_tensor_batch))
                            
                            # Step 3: Perform optimizer step with accumulated gradients
                            opt_output = self.actor_rollout_ref_wg.step_actor_optimizer()

                        # opt_output is a list from ONE_TO_ALL dispatch, take first element's metrics
                        if opt_output and len(opt_output) > 0 and hasattr(opt_output[0], 'non_tensor_batch'):
                            actor_metrics.update(reduce_metrics(opt_output[0].non_tensor_batch))
                        metrics.update(actor_metrics)

                elif self.config.worker.actor.optimize_captioner and not self.config.worker.actor.optimize_segmenter:
                    # Case 2: Only captioner - Use cap_batch to update
                    if self.config.trainer.critic_warmup <= self.global_step:
                        self.actor_rollout_ref_wg.clear_multi_modal_cache()
                        
                        with timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_ref_wg.update_actor(cap_batch)

                        actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                        metrics.update(actor_metrics)

                elif self.config.worker.actor.optimize_segmenter and not self.config.worker.actor.optimize_captioner:
                    # Case 3: Only segmenter - Use seg_batch to update
                    if self.config.trainer.critic_warmup <= self.global_step:
                        self.actor_rollout_ref_wg.clear_multi_modal_cache()
                        
                        with timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_ref_wg.update_actor(seg_batch)

                        actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                        metrics.update(actor_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=cap_batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=cap_batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=cap_batch, timing_raw=timing_raw, num_gpus=num_gpus))
            if seg_batch is not None:           
                metrics.update(compute_data_metrics(batch=seg_batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=seg_batch, timing_raw=timing_raw))
                metrics.update(compute_throughout_metrics(batch=seg_batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # A non-positive frequency explicitly disables validation, including the
        # final pass.  Always continue to the final checkpoint below.
        if self.val_reward_fn is not None and self.config.trainer.val_freq > 0:
            if (
                val_metrics is None
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
