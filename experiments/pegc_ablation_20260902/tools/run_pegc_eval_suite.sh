#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EVAL="$ROOT/projects/eval/qwen3vl_4b_volcengine.sh"
NUM_GPUS="${NUM_GPUS:-3}"
EVAL_GPU_LIST="${EVAL_GPU_LIST:-0,1,2}"
RUN_TAG="${RUN_TAG:-4k}"
EVAL_STEP="${EVAL_STEP:-${MAX_STEPS:-36}}"
for name in baseline evidence_gate mask_credit adaptive_balance cbba; do
  CKPT="$ROOT/logs/pegc_${name}_${RUN_TAG}/checkpoints/global_step_${EVAL_STEP}"
  OUT="$ROOT/logs/pegc_${name}_${RUN_TAG}/evaluation/step_${EVAL_STEP}"
  echo "=== evaluating $name ==="
  CUDA_VISIBLE_DEVICES="$EVAL_GPU_LIST" NUM_GPUS="$NUM_GPUS" TRAIN_MODEL_PATH=/volume/ybo/xyc/Qwen3-VL-4B-SAMTok \
    CHECKPOINT_PATH="$CKPT" HF_MODEL_PATH="$OUT/hf_global_step_${EVAL_STEP}" EVAL_ROOT="$OUT" \
    bash "$EVAL" all
done
