#!/usr/bin/env bash
# Historical 70k mixed OPSD + supervised run with SECA, using Qwen3-VL-8B-SAMTok.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-/volume/ybo/xyc}"
export MODEL_PATH="${MODEL_PATH:-${BASE_DIR}/Qwen3-VL-8B-SAMTok}"
export RUN_NAME="${RUN_NAME:-cyclegrpo70k_historical_seca_8b_8gpu}"
export RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/../logs/${RUN_NAME}}"
export RAY_PORT="${RAY_PORT:-29691}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-29692}"
export JUDGE_PORT="${JUDGE_PORT:-18010}"
export SECA_ENABLED=true
export SECA_SELF_SUPERVISED_ENABLED=true
exec "${SCRIPT_DIR}/train_supervised_70k_common_8gpu.sh" "$@"
