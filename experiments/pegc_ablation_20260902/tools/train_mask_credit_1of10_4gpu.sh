#!/usr/bin/env bash

# 1/10, one-epoch direct GRPO + hierarchical Mask Credit (Refusal Credit off).
# Positive supervision: direct GRPO plus GT-mask CE with hierarchical token credit.
# No-target supervision: direct GRPO only; no-target refusal teacher-forcing is off.
# GPU0-3 train; no Llama service is needed because DLC-QA is disabled. This is
# a clean comparison with the baseline and Refusal Credit-only direct-GRPO runs.

BASE_DIR="${BASE_DIR:-/volume/ybo/xyc}"
REPO_DIR="${REPO_DIR:-${BASE_DIR}/CycleGRPO-OPSD/experiments/pegc_ablation_20260902}"
ENV_DIR="${ENV_DIR:-${BASE_DIR}/envs/cyclegrpo}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_DIR}/bin/python3}"
LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH:-${BASE_DIR}/models/Meta-Llama-3.1-8B-Instruct-hf-v2}"
LLAMA_MODEL_NAME="${LLAMA_MODEL_NAME:-llama3.1-8b}"
LLAMA_API_KEY="${LLAMA_API_KEY:-sk-cyclegrpo-local}"
LLAMA_PORT="${LLAMA_PORT:-8007}"
LLAMA_TEMPLATE="${LLAMA_TEMPLATE:-${REPO_DIR}/tools/multinode/llama3_chat_template.jinja}"
TRAIN_GPU_LIST="${TRAIN_GPU_LIST:-0,1,2,3}"
TRAIN_NUM_GPUS="${TRAIN_NUM_GPUS:-4}"
RAY_SHORT_ROOT="${RAY_SHORT_ROOT:-/dev/shm/pegc-e-g0123456}"
RUN_ROOT="${REPO_DIR}/logs/pegc_direct_grpo_mask_credit_1of10_epoch1"
export CONDA_PREFIX="${ENV_DIR}"

if ! cd "${REPO_DIR}"; then
  echo "Cannot enter exploration branch directory: ${REPO_DIR}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${TRAIN_GPU_LIST}" \
NUM_GPUS="${TRAIN_NUM_GPUS}" \
ROLLOUT_BATCH_SIZE=108 \
ACTOR_GLOBAL_BATCH_SIZE=108 \
DIRECT_BATCH_SIZE=216 \
CAPTION_QA_BATCH_SIZE=54 \
LOCAL_JUDGE_ENABLED=false \
RAY_CLUSTER_EXPECTED_GPUS="${TRAIN_NUM_GPUS}" \
RAY_LOCAL_NUM_CPUS=32 \
RAY_SHORT_ROOT="${RAY_SHORT_ROOT}" \
STREAM_LOG=true \
TRAINER_LOGGERS='["console"]' \
REPO_DIR="${REPO_DIR}" \
MODEL_PATH="${BASE_DIR}/Qwen3-VL-4B-SAMTok" \
TRAIN_DATA="${REPO_DIR}/data_1of10/cycle_2k.parquet" \
VAL_DATA="${REPO_DIR}/data_1of10/cycle_2k.parquet" \
DIRECT_TRAIN_DATA="${REPO_DIR}/data_1of10/direct_4k.parquet" \
DIRECT_NO_TARGET_TRAIN_DATA="${REPO_DIR}/data_1of10/no_target_1k.parquet" \
CAPTION_QA_TRAIN_DATA="${REPO_DIR}/data_1of10/qa_1k.parquet" \
CAPTION_QA_JSONL="${REPO_DIR}/data_1of10/qa_1k.jsonl" \
SUPERVISED_CAPTION_QA_ENABLED=false \
CAPTION_QA_JUDGE_BASE_URL=http://127.0.0.1:8007/v1 \
CAPTION_QA_JUDGE_MODEL="${LLAMA_MODEL_NAME}" \
CAPTION_QA_JUDGE_API_KEY="${LLAMA_API_KEY}" \
CAPTION_QA_MAX_CONCURRENCY=16 \
DIRECT_GROUNDING_ENABLED=true \
DIRECT_MASK_CE_ENABLED=true \
DIRECT_GROUNDING_INCLUDE_POSITIVE_SOURCES=true \
DIRECT_GROUNDING_INCLUDE_NO_TARGET=true \
DIRECT_MASK_CE_INCLUDE_NO_TARGET=false \
DIRECT_GROUNDING_LOSS_WEIGHT=0.15 \
DIRECT_GROUNDING_WARMUP_START_STEP=0 \
DIRECT_GROUNDING_WARMUP_END_STEP=0 \
DIRECT_MASK_CE_LOSS_WEIGHT=0.005 \
CAPTION_QA_REWARD_WEIGHT=0.0 \
CAPTION_QA_LOSS_WEIGHT=0.0 \
PRESERVE_ORIGINAL_GRPO=true \
TEACHER_EMA_DECAY=1.0 \
CAPTION_ANCHOR_KL_COEF=0 \
SEGMENTATION_ANCHOR_KL_COEF=0 \
EVIDENCE_GATE_ENABLED=false \
MASK_CREDIT_ENABLED=true \
ADAPTIVE_BALANCE_ENABLED=false \
CBBA_ENABLED=false \
REFUSAL_CREDIT_ENABLED=false \
REFUSAL_CREDIT_LOSS_WEIGHT=0.05 \
REFUSAL_CREDIT_WARMUP_START_STEP=0 \
REFUSAL_CREDIT_WARMUP_END_STEP=0 \
MAX_STEPS=18 \
TOTAL_EPOCHS=1 \
SAVE_FREQ=18 \
SAVE_LIMIT=3 \
RUN_NAME=pegc_direct_grpo_mask_credit_1of10_epoch1 \
RUN_ROOT="${RUN_ROOT}" \
bash "${REPO_DIR}/projects/rl/qwen3vl_4b_pegc_ablation.sh"
