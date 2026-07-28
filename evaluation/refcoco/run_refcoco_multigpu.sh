#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS=${1:-8}
MODEL_PATH=${2:?"model path is required"}
SAVE_DIR=${3:?"save directory is required"}
REFCOCO_ROOT=${4:?"RefCOCO root is required"}
SPLIT=${5:-val}
SPLIT_BY=${SPLIT_BY:-unc}
VQ_SAM2_PATH=${VQ_SAM2_PATH:?"VQ_SAM2_PATH is required"}
SAM2_PATH=${SAM2_PATH:?"SAM2_PATH is required"}
PYTHON_BIN=${PYTHON_BIN:-python}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$(dirname "$SAVE_DIR")/logs"
mkdir -p "$SAVE_DIR" "$LOG_DIR"
pids=()
for ((task=0; task<NUM_GPUS; task++)); do
    CUDA_VISIBLE_DEVICES=$task "$PYTHON_BIN" "$SCRIPT_DIR/qwen3vl_refcoco_eval.py" \
        --model_path "$MODEL_PATH" --vq_sam2_path "$VQ_SAM2_PATH" --sam2_path "$SAM2_PATH" \
        --refcoco_root "$REFCOCO_ROOT" --split_by "$SPLIT_BY" --split "$SPLIT" \
        --save_dir "$SAVE_DIR" --task_id "$task" --num_tasks "$NUM_GPUS" --gpu_id 0 \
        > "$LOG_DIR/refcoco_${SPLIT}_shard${task}.log" 2>&1 &
    pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
(( rc == 0 )) || exit "$rc"
"$PYTHON_BIN" "$SCRIPT_DIR/qwen3vl_refcoco_eval.py" --save_dir "$SAVE_DIR" --metric_only
