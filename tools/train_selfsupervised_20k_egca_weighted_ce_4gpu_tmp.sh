#!/usr/bin/env bash
# Temporary local smoke wrapper. It overrides only topology and max steps;
# the production implementation remains the 8-GPU script beside this file.

BASE_DIR="${BASE_DIR:-/volume/ybo/xyc}"
REPO_DIR="${REPO_DIR:-${BASE_DIR}/CycleGRPO-OPSD}"
export NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RAY_PORT="${RAY_PORT:-29682}"
export RUN_NAME="${RUN_NAME:-cyclegrpo20k_egca_weighted_ce_4gpu_smoke}"
export RUN_ROOT="${RUN_ROOT:-${REPO_DIR}/logs/${RUN_NAME}}"
export MAX_STEPS="${MAX_STEPS:-1}"
export KEEPALIVE_AFTER_TRAINING=false
exec bash "${REPO_DIR}/tools/train_selfsupervised_20k_egca_weighted_ce_8gpu.sh"
