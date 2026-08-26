#!/usr/bin/env bash

BASE_DIR="${BASE_DIR:-/volume/ybo/xyc}"
OFFICIAL_REPO="${OFFICIAL_REPO:-$BASE_DIR/CycleGRPO}"
ENV_DIR="${ENV_DIR:-$BASE_DIR/envs/cyclegrpo}"
MODEL_PATH="${MODEL_PATH:-$BASE_DIR/Qwen3-VL-4B-SAMTok}"
TRAIN_DATA="${TRAIN_DATA:-$BASE_DIR/datasets/cyclegrpo_20k_official_htg_seed20260825/cyclegrpo_20k_official_htg_15k_single_4k_multi_1k_gres_seed20260825.parquet}"
RUN_NAME="${RUN_NAME:-cyclegrpo_official_20k_htg_8gpu}"
RUN_ROOT="${RUN_ROOT:-$OFFICIAL_REPO/logs/$RUN_NAME}"
KEEPALIVE_SCRIPT="${KEEPALIVE_SCRIPT:-$BASE_DIR/CycleGRPO-OPSD/tools/cuda_keepalive.py}"
KEEPALIVE_MEMORY_MB="${KEEPALIVE_MEMORY_MB:-40000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TRAINER_MAX_STEPS_ARG=()
if [ -n "${MAX_STEPS:-}" ]; then
  TRAINER_MAX_STEPS_ARG=("trainer.max_steps=$MAX_STEPS")
fi

test -x "$ENV_DIR/bin/python3" || {
  echo "Python not found: $ENV_DIR/bin/python3" >&2
  exit 2
}
test -f "$TRAIN_DATA" || {
  echo "Training data not found: $TRAIN_DATA" >&2
  exit 2
}
test -f "$MODEL_PATH/mask_tokenizer_256x2.pth" || {
  echo "Mask tokenizer checkpoint not found" >&2
  exit 2
}
test -f "$MODEL_PATH/sam2.1_hiera_large.pt" || {
  echo "SAM2 checkpoint not found" >&2
  exit 2
}
test -f "$KEEPALIVE_SCRIPT" || {
  echo "Keepalive script not found: $KEEPALIVE_SCRIPT" >&2
  exit 2
}

mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/wandb"
cd "$OFFICIAL_REPO" || exit 2

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$OFFICIAL_REPO${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="$RUN_ROOT/wandb"

"$ENV_DIR/bin/python3" -m verl.trainer.main \
  config="$OFFICIAL_REPO/projects/rl/config.yaml" \
  data.train_files="['$TRAIN_DATA']" \
  data.val_files="['$TRAIN_DATA']" \
  data.format_prompt="$OFFICIAL_REPO/projects/rl/format_prompt/non_thinking.jinja" \
  data.region_format=mask_token \
  data.rollout_batch_size=128 \
  worker.actor.global_batch_size=128 \
  worker.actor.micro_batch_size_per_device_for_experience=2 \
  worker.actor.micro_batch_size_per_device_for_update=1 \
  worker.actor.model.model_path="$MODEL_PATH" \
  worker.actor.model.freeze_vision_tower=true \
  worker.actor.optimize_captioner=true \
  worker.actor.optimize_segmenter=true \
  worker.reward.reward_function="$OFFICIAL_REPO/projects/rl/reward_function/text2mask.py:compute_score" \
  worker.reward.mask_tokenizer_path="$MODEL_PATH/mask_tokenizer_256x2.pth" \
  worker.reward.sam2_pretrained_weight="$MODEL_PATH/sam2.1_hiera_large.pt" \
  worker.reward.sam2_config_dir_path="$OFFICIAL_REPO/projects/transformers/vq_sam2/sam2/sam2_configs" \
  worker.rollout.n=8 \
  trainer.total_epochs=1 \
  "${TRAINER_MAX_STEPS_ARG[@]}" \
  trainer.val_freq=-1 \
  trainer.val_before_train=false \
  trainer.save_freq=5 \
  trainer.save_limit=2 \
  trainer.save_checkpoint_path="$RUN_ROOT/checkpoints" \
  trainer.find_last_checkpoint=false \
  trainer.project_name=cyclegrpo_official \
  trainer.experiment_name="$RUN_NAME" \
  trainer.logger="['file']" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=8

TRAIN_STATUS=$?
echo "Official CycleGRPO training exited with status $TRAIN_STATUS" >&2
if [ "$TRAIN_STATUS" -ne 0 ]; then
  exit "$TRAIN_STATUS"
fi

echo "Training completed; starting CUDA keepalive." >&2
exec "$ENV_DIR/bin/python3" "$KEEPALIVE_SCRIPT" --memory-mb "$KEEPALIVE_MEMORY_MB"
