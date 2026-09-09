#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LAUNCHER="$ROOT/projects/rl/qwen3vl_4b_pegc_ablation.sh"
COMMON=(CUDA_VISIBLE_DEVICES=0,1,2 NUM_GPUS=3 RAY_CLUSTER_EXPECTED_GPUS=3 TRAINER_LOGGERS='["console"]' \
  ROLLOUT_BATCH_SIZE=108 ACTOR_GLOBAL_BATCH_SIZE=108 DIRECT_BATCH_SIZE=216 CAPTION_QA_BATCH_SIZE=54 \
  MAX_STEPS="${MAX_STEPS:-1}" SAVE_FREQ="${SAVE_FREQ:-1}" SUPERVISED_CAPTION_QA_ENABLED=false CAPTION_QA_JUDGE_MODEL="${CAPTION_QA_JUDGE_MODEL:-llama3.1-8b}" \
  CAPTION_QA_JUDGE_BASE_URL="${CAPTION_QA_JUDGE_BASE_URL:-http://127.0.0.1:8007/v1}" CAPTION_QA_JUDGE_API_KEY="${CAPTION_QA_JUDGE_API_KEY:-EMPTY}" \
  DIRECT_GROUNDING_ENABLED=true DIRECT_MASK_CE_ENABLED=true DIRECT_GROUNDING_INCLUDE_POSITIVE_SOURCES=true \
  DIRECT_GROUNDING_INCLUDE_NO_TARGET=true DIRECT_MASK_CE_INCLUDE_NO_TARGET=true DIRECT_GROUNDING_LOSS_WEIGHT=0.15 \
  DIRECT_MASK_CE_LOSS_WEIGHT=0.005 CAPTION_QA_LOSS_WEIGHT=1.0 CAPTION_QA_REWARD_WEIGHT=1.0 \
  CAPTION_ANCHOR_KL_COEF=0 SEGMENTATION_ANCHOR_KL_COEF=0 RAY_SHORT_ROOT=/dev/shm/pegc-ray-gpu012 RAY_LOCAL_NUM_CPUS=32 RUN_ROOT="$ROOT/logs")
for name in baseline evidence_gate mask_credit adaptive_balance cbba; do
  echo "=== PEGC $name ==="
  gate=false; credit=false; balance=false
  case "$name" in
    evidence_gate) gate=true ;;
    mask_credit) credit=true ;;
    adaptive_balance) balance=true ;;
    cbba) cbba=true ;;
  esac
  env "${COMMON[@]}" RUN_NAME="pegc_${name}_4k" EVIDENCE_GATE_ENABLED="$gate" MASK_CREDIT_ENABLED="$credit" ADAPTIVE_BALANCE_ENABLED="$balance" CBBA_ENABLED="${cbba:-false}" \
    RUN_ROOT="$ROOT/logs/pegc_${name}_4k" bash "$LAUNCHER"
done
