#!/usr/bin/env bash
# Temporary Qwen3-VL-8B-SAMTok smoke wrapper: 3 Ray/FSDP GPUs + GPU 3 judge.
# Runs one optimizer step and does not start the post-run GPU hold.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-/volume/ybo/xyc}"
export MODEL_PATH="${MODEL_PATH:-${BASE_DIR}/Qwen3-VL-8B-SAMTok}"
export RUN_NAME="${RUN_NAME:-cyclegrpo70k_historical_seca_8b_4gpu_smoke}"
export RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/../logs/${RUN_NAME}}"
export RAY_PORT="${RAY_PORT:-29695}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-29696}"
export JUDGE_PORT="${JUDGE_PORT:-18012}"
export RAY_SHORT_ROOT="${RAY_SHORT_ROOT:-/dev/shm/cgrpo70k-seca8b4}"
export TRAIN_NUM_GPUS=3
export TRAIN_CUDA_VISIBLE_DEVICES=0,1,2
export ALL_CUDA_VISIBLE_DEVICES=0,1,2,3
export JUDGE_CUDA_DEVICE=3
export ROLLOUT_BATCH_SIZE=114 ACTOR_GLOBAL_BATCH_SIZE=114 DIRECT_BATCH_SIZE=228 CAPTION_QA_BATCH_SIZE=57
export MAX_STEPS=1 SAVE_FREQ=1 SAVE_LIMIT=1 HOLD_AFTER_EXIT=false
export SECA_ENABLED=true SECA_SELF_SUPERVISED_ENABLED=true
exec "${SCRIPT_DIR}/train_supervised_70k_common_8gpu.sh" "$@"
