#!/usr/bin/env bash
# Temporary Qwen3-VL-8B-SAMTok 100k-scaling smoke wrapper:
# CUDA 2 trains and CUDA 3 runs the local Llama judge. It runs one optimizer
# step and does not start the post-run GPU hold.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-/volume/ybo/xyc}"

export MODEL_PATH="${MODEL_PATH:-${BASE_DIR}/Qwen3-VL-8B-SAMTok}"
export DATA_ROOT="${DATA_ROOT:-${BASE_DIR}/datasets/cyclegrpo100k_scaling_20260911}"
export RUN_NAME="${RUN_NAME:-cyclegrpo100k_seca_8b_2gpu_smoke}"
export RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/../logs/${RUN_NAME}}"
export RAY_PORT="${RAY_PORT:-29705}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-29706}"
export JUDGE_PORT="${JUDGE_PORT:-18013}"
export RAY_SHORT_ROOT="${RAY_SHORT_ROOT:-/dev/shm/cgrpo100k-2b}"

# One Ray/FSDP rank on CUDA 2; CUDA 3 is reserved for the judge.
export TRAIN_NUM_GPUS=1
export TRAIN_CUDA_VISIBLE_DEVICES=2
export ALL_CUDA_VISIBLE_DEVICES=2,3
export JUDGE_CUDA_DEVICE=3

# Keep the 2:4:1 stream ratio while making the one-rank smoke executable.
export ROLLOUT_BATCH_SIZE=112
export ACTOR_GLOBAL_BATCH_SIZE=112
export DIRECT_BATCH_SIZE=224
export CAPTION_QA_BATCH_SIZE=56
export MAX_STEPS=1 SAVE_FREQ=1 SAVE_LIMIT=1 HOLD_AFTER_EXIT=false
export SECA_ENABLED=true SECA_SELF_SUPERVISED_ENABLED=true

exec "${SCRIPT_DIR}/train_supervised_100k_seca_8b_8gpu.sh" "$@"
