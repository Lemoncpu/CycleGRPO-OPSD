#!/usr/bin/env bash
BASE_DIR=${BASE_DIR:-/volume/ybo/xyc}
REPO_DIR=${REPO_DIR:-${BASE_DIR}/CycleGRPO-OPSD}
ENV_DIR=${ENV_DIR:-${BASE_DIR}/envs/cyclegrpo}
PYTHON_BIN=${PYTHON_BIN:-${ENV_DIR}/bin/python3}
HOLD_SCRIPT=${HOLD_SCRIPT:-${REPO_DIR}/tools/gpu_power_hold.sh}
PY_SCRIPT=${REPO_DIR}/evaluation/refcoco/demo_three_capabilities.py
PREDICTIONS_DIR=${PREDICTIONS_DIR:-${REPO_DIR}/logs/demo_refcoco_1000/Ours/predictions}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_DIR}/logs/demo_three_capabilities}
if [ ! -x "$PYTHON_BIN" ]; then
echo "Python not found: $PYTHON_BIN" >&2
exit 2
fi
if [ ! -f "$PY_SCRIPT" ] || [ ! -f "$HOLD_SCRIPT" ]; then
echo "Demo script or GPU hold helper missing." >&2
exit 2
fi
if [ ! -d "$PREDICTIONS_DIR" ]; then
echo "Run demo_refcoco_compare_1000.sh first, or set PREDICTIONS_DIR." >&2
exit 2
fi
# Release only this project's hold workers so the VLM can use GPU 0.
GPU_LIST=0,1,2,3 PYTHON_BIN="$PYTHON_BIN" bash "$HOLD_SCRIPT" stop >/dev/null 2>&1
sleep 3
busy_gpus=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '$2 + 0 > 2048 {gsub(/[[:space:]]/, "", $1); print $1}')
if [ -n "$busy_gpus" ]; then
echo "GPU workload detected; demo not started." >&2
exit 3
fi
restore_gpu_hold() {
result=$?
trap - EXIT INT TERM
GPU_LIST=1,2,3 PYTHON_BIN="$PYTHON_BIN" bash "$HOLD_SCRIPT" stop >/dev/null 2>&1
GPU_LIST=0,1,2,3 PYTHON_BIN="$PYTHON_BIN" bash "$HOLD_SCRIPT" start >/dev/null 2>&1
exit "$result"
}
trap restore_gpu_hold EXIT INT TERM
# Keep the other three cards active while inference uses GPU 0.
GPU_LIST=1,2,3 PYTHON_BIN="$PYTHON_BIN" bash "$HOLD_SCRIPT" start >/dev/null 2>&1
cd "$REPO_DIR" || exit 2
PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" "$PY_SCRIPT" \
--ours-model "${OURS_MODEL:-${REPO_DIR}/logs/cyclegrpo70k_historical_seca_8gpu/evaluation/step_179_cpu_export/hf_global_step_179}" \
--predictions-dir "$PREDICTIONS_DIR" \
--output-dir "$OUTPUT_DIR" \
--device cuda:0 2>&1 | grep -v "select best iou"
result=${PIPESTATUS[0]}
exit "$result"
