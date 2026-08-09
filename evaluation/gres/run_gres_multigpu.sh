#!/usr/bin/env bash
# Multi-GPU gRefCOCO/GRES localization evaluation. Predictions are resumable.
set -euo pipefail

NUM_GPUS=${1:-8}
MODEL_PATH=${2:?"model path is required"}
SAVE_DIR=${3:?"save directory is required"}
DATASET=${4:?"prepared GRES dataset is required"}
VQ_SAM2_PATH=${VQ_SAM2_PATH:?"VQ_SAM2_PATH is required"}
SAM2_PATH=${SAM2_PATH:?"SAM2_PATH is required"}
PYTHON_BIN=${PYTHON_BIN:-python}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL="$SCRIPT_DIR/qwen3vl_gres_eval.py"
LOG_DIR="$(dirname "$SAVE_DIR")/logs"

[[ -f "$DATASET" ]] || { echo "GRES dataset not found: $DATASET" >&2; exit 1; }
mkdir -p "$SAVE_DIR" "$LOG_DIR"
cd "$REPO_ROOT"

TOTAL=$("$PYTHON_BIN" -c "import json; print(len(json.load(open('$DATASET'))))")
echo "repo root: $REPO_ROOT"
echo "Launching $NUM_GPUS shards | model=$MODEL_PATH | save=$SAVE_DIR | samples=$TOTAL"

pids=()
for ((task=0; task<NUM_GPUS; task++)); do
    CUDA_VISIBLE_DEVICES=$task "$PYTHON_BIN" "$EVAL" \
        --model_path "$MODEL_PATH" \
        --vq_sam2_path "$VQ_SAM2_PATH" \
        --sam2_path "$SAM2_PATH" \
        --dataset "$DATASET" \
        --save_dir "$SAVE_DIR" \
        --task_id "$task" --num_tasks "$NUM_GPUS" --gpu_id 0 \
        > "$LOG_DIR/gres_shard${task}.log" 2>&1 &
    pids+=("$!")
    echo "  shard $task -> GPU $task (pid ${pids[-1]}, log $LOG_DIR/gres_shard${task}.log)"
done

start=$(date +%s)
echo "monitoring progress: $TOTAL samples total (status every 20s)"
while true; do
    running=0
    for pid in "${pids[@]}"; do
        kill -0 "$pid" 2>/dev/null && running=$((running + 1))
    done
    done_count=$(find "$SAVE_DIR" -maxdepth 1 -name 'case_*.json' -type f 2>/dev/null | wc -l)
    elapsed=$(( $(date +%s) - start ))
    if [[ "$done_count" -gt 0 && "$elapsed" -gt 0 ]]; then
        awk -v done="$done_count" -v total="$TOTAL" -v elapsed="$elapsed" -v running="$running" 'BEGIN {
            rate=done/elapsed; eta=(rate>0)?(total-done)/rate:0;
            printf "[%4dm%02ds] %d/%d (%.0f%%) | %.2f/s | %d shards alive | ETA ~%dm%02ds\\n", elapsed/60, elapsed%60, done, total, 100*done/total, rate, running, eta/60, eta%60
        }'
    else
        printf "[%4dm%02ds] %d/%d | %d shards alive (loading models...)\n" $((elapsed / 60)) $((elapsed % 60)) "$done_count" "$TOTAL" "$running"
    fi
    [[ "$running" -eq 0 ]] && break
    sleep 20
done

rc=0
for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
done
done_count=$(find "$SAVE_DIR" -maxdepth 1 -name 'case_*.json' -type f 2>/dev/null | wc -l)
echo "All shards finished (rc=$rc). $done_count/$TOTAL outputs in $SAVE_DIR"
(( rc == 0 )) || exit "$rc"
(( done_count == TOTAL )) || { echo "GRES outputs incomplete; do not score partial predictions." >&2; exit 1; }

"$PYTHON_BIN" "$EVAL" --save_dir "$SAVE_DIR" --metric_only \
    --metrics-file "$(dirname "$SAVE_DIR")/gres_metrics.json"
