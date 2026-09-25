#!/usr/bin/env bash
# Historical 70k mixed OPSD + supervised baseline (SECA disabled).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_NAME="${RUN_NAME:-cyclegrpo70k_historical_baseline_8gpu}"
export RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/../logs/${RUN_NAME}}"
export RAY_PORT="${RAY_PORT:-29687}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-29688}"
export JUDGE_PORT="${JUDGE_PORT:-18008}"
export SECA_ENABLED=false
export SECA_SELF_SUPERVISED_ENABLED=false
exec "${SCRIPT_DIR}/train_supervised_70k_common_8gpu.sh" "$@"
