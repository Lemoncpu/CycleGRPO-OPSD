#!/usr/bin/env bash
# 70k historical SECA run with routing thresholds low=0.30/high=0.85.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_NAME="${RUN_NAME:-cyclegrpo70k_historical_seca_routing_l030_h085_8gpu}"
export RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/../logs/${RUN_NAME}}"
export RAY_PORT="${RAY_PORT:-29701}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-29702}"
export JUDGE_PORT="${JUDGE_PORT:-18013}"
export RAY_SHORT_ROOT="${RAY_SHORT_ROOT:-/dev/shm/cgrpo70k-seca-l03h085}"
export ROUTING_LOW_THRESHOLD=0.30
export ROUTING_HIGH_THRESHOLD=0.85
exec "${SCRIPT_DIR}/train_supervised_70k_seca_8gpu.sh" "$@"
