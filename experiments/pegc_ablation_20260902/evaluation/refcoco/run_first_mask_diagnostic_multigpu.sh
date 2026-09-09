#!/usr/bin/env bash
# Re-decode existing RefCOCO responses using only their first complete mask.

NUM_GPUS=${1:-8}
INPUT_DIR=${2:?input response directory is required}
OUTPUT_DIR=${3:?output directory is required}
REFCOCO_ROOT=${4:?RefCOCO root is required}
VQ_SAM2_PATH=${VQ_SAM2_PATH:?VQ_SAM2_PATH is required}
SAM2_PATH=${SAM2_PATH:?SAM2_PATH is required}
PYTHON_BIN=${PYTHON_BIN:-python}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUTPUT_DIR"
pids=()
for ((task=0; task<NUM_GPUS; task++)); do
    CUDA_VISIBLE_DEVICES=$task "$PYTHON_BIN" "$SCRIPT_DIR/first_mask_diagnostic.py" \
        --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR" --refcoco-root "$REFCOCO_ROOT" \
        --vq-sam2-path "$VQ_SAM2_PATH" --sam2-path "$SAM2_PATH" \
        --task-id "$task" --num-tasks "$NUM_GPUS" --gpu-id 0 \
        > "${OUTPUT_DIR}/shard${task}.log" 2>&1 &
    pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
(( rc == 0 )) || exit "$rc"
"$PYTHON_BIN" "$SCRIPT_DIR/first_mask_diagnostic.py" \
    --output-dir "$OUTPUT_DIR" --metric-only
