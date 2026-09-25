#!/usr/bin/env bash
# Historical supervised baseline with every training stream reduced to 50%.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_NAME="${RUN_NAME:-cyclegrpo70k_historical_baseline_50pct_8gpu}"
export RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/../logs/${RUN_NAME}}"
export RAY_PORT="${RAY_PORT:-29691}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-29692}"
export JUDGE_PORT="${JUDGE_PORT:-18010}"
export RAY_SHORT_ROOT="${RAY_SHORT_ROOT:-/dev/shm/cgrpo70k-50}"
export DATA_FRACTION=50
export MAX_STEPS="${MAX_STEPS:-90}"
export SECA_ENABLED=false
export SECA_SELF_SUPERVISED_ENABLED=false
# Check the exact Qwen3-VL AutoModel mapping used by the shared FSDP worker
# before reserving the seven training GPUs and starting the local judge.
PYTHON_BIN="${PYTHON_BIN:-${BASE_DIR:-/volume/ybo/xyc}/envs/cyclegrpo/bin/python3}"
"${PYTHON_BIN}" - <<'PY'
from transformers import AutoModelForImageTextToText, Qwen3VLConfig

if AutoModelForImageTextToText._model_mapping.get(Qwen3VLConfig, None) is None:
    raise RuntimeError("Qwen3-VL image-text AutoModel mapping is unavailable")
PY
preflight_status=$?
if [[ "${preflight_status}" != 0 ]]; then
    exit "${preflight_status}"
fi
exec "${SCRIPT_DIR}/train_supervised_70k_common_8gpu.sh" "$@"
