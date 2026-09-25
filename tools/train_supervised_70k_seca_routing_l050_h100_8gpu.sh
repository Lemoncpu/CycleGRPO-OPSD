#!/usr/bin/env bash
# 70k historical SECA run with routing thresholds low=0.50/high=1.00.
# The requested +0.20 high-threshold variant is capped at 1.00 by the
# routing configuration's [0, 1] validity constraint.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_NAME="${RUN_NAME:-cyclegrpo70k_historical_seca_routing_l050_h100_8gpu}"
export RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/../logs/${RUN_NAME}}"
export RAY_PORT="${RAY_PORT:-29707}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-29708}"
export JUDGE_PORT="${JUDGE_PORT:-18016}"
export RAY_SHORT_ROOT="${RAY_SHORT_ROOT:-/dev/shm/cgrpo70k-seca-l05h100}"
export ROUTING_LOW_THRESHOLD=0.50
export ROUTING_HIGH_THRESHOLD=1.00
exec "${SCRIPT_DIR}/train_supervised_70k_seca_8gpu.sh" "$@"
