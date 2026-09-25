#!/usr/bin/env bash
# Local four-GPU smoke wrapper for train_selfsupervised_20k_seca_8gpu.sh.

BASE_DIR="${BASE_DIR:-/volume/ybo/xyc}"
REPO_DIR="${REPO_DIR:-${BASE_DIR}/CycleGRPO-OPSD}"
ENV_DIR="${ENV_DIR:-${BASE_DIR}/envs/cyclegrpo}"
export NUM_GPUS=4
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NNODES=1
export MULTINODE_ENABLED=false
export CONDA_PREFIX="${ENV_DIR}"
export RAY_PORT="${RAY_PORT:-29684}"
export RAY_SHORT_ROOT="${RAY_SHORT_ROOT:-/dev/shm/cgrpo-seca-ss4}"
export RUN_NAME="${RUN_NAME:-cyclegrpo20k_seca_selfsupervised_pixel_empty_4gpu_smoke}"
export MAX_STEPS="${MAX_STEPS:-1}"
export KEEPALIVE_AFTER_TRAINING=false
exec bash "${REPO_DIR}/tools/train_selfsupervised_20k_seca_8gpu.sh"
