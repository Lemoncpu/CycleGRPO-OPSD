#!/usr/bin/env bash

# 1/10, one-epoch supervised direct GRPO + Mask Credit + Refusal Credit.
# Self-contained 8-GPU entry point: GPU0-6 train and GPU3 runs the DLC judge.

BASE_DIR="${BASE_DIR:-/volume/ybo/xyc}"
REPO_DIR="${REPO_DIR:-${BASE_DIR}/CycleGRPO-OPSD/experiments/pegc_ablation_20260902}"
ENV_DIR="${ENV_DIR:-${BASE_DIR}/envs/cyclegrpo}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_DIR}/bin/python3}"
LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH:-${BASE_DIR}/models/Meta-Llama-3.1-8B-Instruct-hf-v2}"
LLAMA_MODEL_NAME="${LLAMA_MODEL_NAME:-llama3.1-8b}"
LLAMA_API_KEY="${LLAMA_API_KEY:-sk-cyclegrpo-local}"
LLAMA_PORT="${LLAMA_PORT:-8007}"
LLAMA_TEMPLATE="${LLAMA_TEMPLATE:-${REPO_DIR}/tools/multinode/llama3_chat_template.jinja}"
TRAIN_GPU_LIST="${TRAIN_GPU_LIST:-0,1,2}"
TRAIN_NUM_GPUS="${TRAIN_NUM_GPUS:-3}"
LLAMA_GPU="${LLAMA_GPU:-3}"
RAY_SHORT_ROOT="${RAY_SHORT_ROOT:-/dev/shm/pegc-e-g0123456}"
RUN_ROOT="${REPO_DIR}/logs/pegc_mask_refusal_credit_1of10_epoch1"
LLAMA_LOG="${LLAMA_LOG:-${RUN_ROOT}/llama_judge.log}"
LLAMA_PID=""
LLAMA_STARTED=false

cleanup_llama() {
  exit_code=$?
  if [[ "${LLAMA_STARTED}" == "true" && -n "${LLAMA_PID}" ]] && kill -0 "${LLAMA_PID}" 2>/dev/null; then
    kill -TERM -- "-${LLAMA_PID}" 2>/dev/null || kill "${LLAMA_PID}" 2>/dev/null || true
    wait "${LLAMA_PID}" 2>/dev/null || true
  fi
  exit "${exit_code}"
}
trap cleanup_llama EXIT INT TERM

if [[ ! -x "${PYTHON_BIN}" || ! -d "${LLAMA_MODEL_PATH}" || ! -f "${LLAMA_TEMPLATE}" ]]; then
  echo "Missing Python, Llama model, or chat template." >&2
  exit 1
fi
mkdir -p "$(dirname "${LLAMA_LOG}")"
if ! curl --silent --max-time 3 -H "Authorization: Bearer ${LLAMA_API_KEY}" "http://127.0.0.1:${LLAMA_PORT}/v1/models" | grep -q "\"${LLAMA_MODEL_NAME}\""; then
  echo "Starting local Llama judge on physical GPU3..."
  setsid env CUDA_VISIBLE_DEVICES="${LLAMA_GPU}" TOKENIZERS_PARALLELISM=true "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
    --model "${LLAMA_MODEL_PATH}" --served-model-name "${LLAMA_MODEL_NAME}" \
    --host 127.0.0.1 --port "${LLAMA_PORT}" --api-key "${LLAMA_API_KEY}" \
    --tensor-parallel-size 1 --pipeline-parallel-size 1 --trust-remote-code \
    --dtype bfloat16 --gpu-memory-utilization 0.85 --chat-template "${LLAMA_TEMPLATE}" \
    >"${LLAMA_LOG}" 2>&1 &
  LLAMA_PID=$!
  LLAMA_STARTED=true
  deadline=$((SECONDS + ${LLAMA_STARTUP_TIMEOUT_SECONDS:-240}))
  while (( SECONDS < deadline )); do
    if ! kill -0 "${LLAMA_PID}" 2>/dev/null; then
      echo "Llama judge exited; inspect ${LLAMA_LOG}" >&2
      tail -80 "${LLAMA_LOG}" >&2
      exit 1
    fi
    if curl --silent --max-time 3 -H "Authorization: Bearer ${LLAMA_API_KEY}" "http://127.0.0.1:${LLAMA_PORT}/v1/models" | grep -q "${LLAMA_MODEL_NAME}"; then break; fi
    sleep 2
  done
  if ! curl --silent --max-time 3 -H "Authorization: Bearer ${LLAMA_API_KEY}" "http://127.0.0.1:${LLAMA_PORT}/v1/models" | grep -q "${LLAMA_MODEL_NAME}"; then
    echo "Llama judge did not become healthy; inspect ${LLAMA_LOG}" >&2
    exit 1
  fi
else
  echo "Reusing healthy Llama judge at http://127.0.0.1:${LLAMA_PORT}/v1"
fi

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
LOCAL_JUDGE_ENABLED=true \
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
SUPERVISED_CAPTION_QA_ENABLED=true \
CAPTION_QA_JUDGE_BASE_URL=http://127.0.0.1:8007/v1 \
CAPTION_QA_JUDGE_MODEL="${LLAMA_MODEL_NAME}" \
CAPTION_QA_JUDGE_API_KEY="${LLAMA_API_KEY}" \
CAPTION_QA_MAX_CONCURRENCY=16 \
DIRECT_GROUNDING_ENABLED=true \
DIRECT_MASK_CE_ENABLED=false \
DIRECT_GROUNDING_INCLUDE_POSITIVE_SOURCES=true \
DIRECT_GROUNDING_INCLUDE_NO_TARGET=true \
DIRECT_MASK_CE_INCLUDE_NO_TARGET=true \
DIRECT_GROUNDING_LOSS_WEIGHT=0.15 \
DIRECT_MASK_CE_LOSS_WEIGHT=0.005 \
CAPTION_QA_REWARD_WEIGHT=1.0 \
CAPTION_QA_LOSS_WEIGHT=1.0 \
PRESERVE_ORIGINAL_GRPO=true \
TEACHER_EMA_DECAY=1.0 \
CAPTION_ANCHOR_KL_COEF=0 \
SEGMENTATION_ANCHOR_KL_COEF=0 \
EVIDENCE_GATE_ENABLED=false \
MASK_CREDIT_ENABLED=true \
ADAPTIVE_BALANCE_ENABLED=false \
CBBA_ENABLED=false \
REFUSAL_CREDIT_ENABLED=true \
REFUSAL_CREDIT_LOSS_WEIGHT=0.05 \
REFUSAL_CREDIT_WARMUP_START_STEP=0 \
REFUSAL_CREDIT_WARMUP_END_STEP=0 \
MAX_STEPS=18 \
TOTAL_EPOCHS=1 \
SAVE_FREQ=18 \
SAVE_LIMIT=3 \
RUN_NAME=pegc_mask_refusal_credit_1of10_epoch1 \
RUN_ROOT="${REPO_DIR}/logs/pegc_mask_refusal_credit_1of10_epoch1" \
bash "${REPO_DIR}/projects/rl/qwen3vl_4b_pegc_ablation.sh"
