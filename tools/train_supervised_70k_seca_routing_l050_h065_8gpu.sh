#!/usr/bin/env bash
# 70k historical SECA run with routing thresholds low=0.50/high=0.65.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_NAME="${RUN_NAME:-cyclegrpo70k_historical_seca_routing_l050_h065_8gpu}"
export RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/../logs/${RUN_NAME}}"
export RAY_PORT="${RAY_PORT:-29705}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-29706}"
export JUDGE_PORT="${JUDGE_PORT:-18015}"
export RAY_SHORT_ROOT="${RAY_SHORT_ROOT:-/dev/shm/cgrpo70k-seca-l05h065}"
export ROUTING_LOW_THRESHOLD=0.50
export ROUTING_HIGH_THRESHOLD=0.65
exec "${SCRIPT_DIR}/train_supervised_70k_seca_8gpu.sh" "$@"
