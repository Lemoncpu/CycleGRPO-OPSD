#!/usr/bin/env bash
# Temporary local smoke wrapper: 3 Ray/FSDP training GPUs + GPU 3 Llama judge.
# It runs one optimizer step and does not start the post-run GPU hold, so the
# two smoke wrappers can be executed sequentially after the full 4-GPU job is
# stopped or completed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_NAME="${RUN_NAME:-cyclegrpo70k_historical_seca_4gpu_smoke}"
export RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/../logs/${RUN_NAME}}"
export RAY_PORT="${RAY_PORT:-29693}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-29694}"
export JUDGE_PORT="${JUDGE_PORT:-18011}"
export RAY_SHORT_ROOT="${RAY_SHORT_ROOT:-/dev/shm/cgrpo70k-seca4}"
export TRAIN_NUM_GPUS=3
export TRAIN_CUDA_VISIBLE_DEVICES=0,1,2
export ALL_CUDA_VISIBLE_DEVICES=0,1,2,3
export JUDGE_CUDA_DEVICE=3
export ROLLOUT_BATCH_SIZE=114 ACTOR_GLOBAL_BATCH_SIZE=114 DIRECT_BATCH_SIZE=228 CAPTION_QA_BATCH_SIZE=57
export MAX_STEPS=1 SAVE_FREQ=1 SAVE_LIMIT=1 HOLD_AFTER_EXIT=false
export SECA_ENABLED=true SECA_SELF_SUPERVISED_ENABLED=true
exec "${SCRIPT_DIR}/train_supervised_70k_common_8gpu.sh" "$@"
