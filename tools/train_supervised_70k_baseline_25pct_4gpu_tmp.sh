#!/usr/bin/env bash
# Full 25% baseline on four GPUs: GPU 0-2 train, GPU 3 serves the judge.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_NAME="${RUN_NAME:-cyclegrpo70k_historical_baseline_25pct_4gpu}"
export RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/../logs/${RUN_NAME}}"
export RAY_PORT="${RAY_PORT:-29695}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-29696}"
export JUDGE_PORT="${JUDGE_PORT:-18012}"
export RAY_SHORT_ROOT="${RAY_SHORT_ROOT:-/dev/shm/cgrpo25-4gpu}"
export DATA_FRACTION=25
export TRAIN_NUM_GPUS=3
export TRAIN_CUDA_VISIBLE_DEVICES=0,1,2
export ALL_CUDA_VISIBLE_DEVICES=0,1,2,3
export JUDGE_CUDA_DEVICE=3
export ROLLOUT_BATCH_SIZE=114 ACTOR_GLOBAL_BATCH_SIZE=114 DIRECT_BATCH_SIZE=228 CAPTION_QA_BATCH_SIZE=57
export MAX_STEPS=45 SAVE_FREQ=5 SAVE_LIMIT=2 HOLD_AFTER_EXIT=false
export SECA_ENABLED=false SECA_SELF_SUPERVISED_ENABLED=false
exec "${SCRIPT_DIR}/train_supervised_70k_common_8gpu.sh" "$@"
