#!/usr/bin/env bash
# Single-node 8-GPU RefCOCO OPSD training for the current Volcengine workspace.

set -euo pipefail

BASE_DIR="${BASE_DIR:-/mnt/cxzx/workspace/data_transfer/houzhiyan}"
REPO_DIR="${REPO_DIR:-${BASE_DIR}/CycleGRPO-OPSD}"
ENV_DIR="${ENV_DIR:-${BASE_DIR}/envs/cyclegrpo}"
MODEL_PATH="${MODEL_PATH:-${BASE_DIR}/Qwen3-VL-4B-SAMTok}"
TRAIN_DATA="${TRAIN_DATA:-${BASE_DIR}/refcoco-train2014-assets/refcoco_train_10k_seed20260722_workspace_paths.parquet}"
VAL_DATA="${VAL_DATA:-${TRAIN_DATA}}"

NUM_GPUS="${NUM_GPUS:-8}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
ACTOR_GLOBAL_BATCH_SIZE="${ACTOR_GLOBAL_BATCH_SIZE:-128}"
CAPTION_ROLLOUTS="${CAPTION_ROLLOUTS:-6}"
LOCALIZATION_ROLLOUTS="${LOCALIZATION_ROLLOUTS:-6}"
OPSD_ENABLED="${OPSD_ENABLED:-true}"
PIXEL_IOU_ENABLED="${PIXEL_IOU_ENABLED:-${OPSD_ENABLED}}"
SEGMENTATION_MAX_RESPONSE_TOKENS="${SEGMENTATION_MAX_RESPONSE_TOKENS:-32}"
ROUTING_ENABLED="${ROUTING_ENABLED:-${OPSD_ENABLED}}"
CAPTION_SAFETY_ENABLED="${CAPTION_SAFETY_ENABLED:-true}"
CAPTION_SAFETY_FORCE_REGENERATE="${CAPTION_SAFETY_FORCE_REGENERATE:-true}"
CAPTION_BLOCK_SPECIAL_TOKEN_VOCAB="${CAPTION_BLOCK_SPECIAL_TOKEN_VOCAB:-true}"
EMA_TEACHER_ENABLED="${EMA_TEACHER_ENABLED:-true}"
TEACHER_ANALYSIS_ENABLED="${TEACHER_ANALYSIS_ENABLED:-true}"
# Caption rollouts should be short natural descriptions. This cap also bounds
# the OPSD safety gate before any caption PPO or JSD update.
CAPTION_MAX_RESPONSE_LENGTH="${CAPTION_MAX_RESPONSE_LENGTH:-256}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
# Leave empty for the complete epoch.  Set 5, 10, ... to create a checkpoint
# boundary for an offline RefCOCO validation before continuing with RESUME=true.
MAX_STEPS="${MAX_STEPS:-}"
# decay=1.0 makes the existing EMA update an identity, so teacher parameters
# remain the SAMTok actor weights copied during worker initialization.
TEACHER_EMA_DECAY="${TEACHER_EMA_DECAY:-1.0}"
# B experiment: keep native CycleGRPO caption GRPO for every safe rollout;
# regenerate CE and privileged JSD remain additive auxiliary gradients.
PRESERVE_ORIGINAL_GRPO="${PRESERVE_ORIGINAL_GRPO:-true}"
# C experiment: keep segmentation-only vocabulary out of privileged caption
# JSD and anchor every safe caption to the frozen SAMTok reference.
CAPTION_ANCHOR_KL_COEF="${CAPTION_ANCHOR_KL_COEF:-0.05}"
CAPTION_ANCHOR_KL_ALL_SAFE_ROUTES="${CAPTION_ANCHOR_KL_ALL_SAFE_ROUTES:-true}"
SEGMENTATION_ANCHOR_KL_COEF="${SEGMENTATION_ANCHOR_KL_COEF:-0.05}"
# The measured caption/segmentation cosine is nearly zero, so projection does
# not materially alter updates and is disabled for the high-confidence teacher run.
ASYMMETRIC_GRADIENT_PROJECTION="${ASYMMETRIC_GRADIENT_PROJECTION:-false}"
JSD_BLOCK_CAPTION_SPECIAL_TOKEN_VOCAB="${JSD_BLOCK_CAPTION_SPECIAL_TOKEN_VOCAB:-true}"
GROUNDEDNESS_ENABLED="${GROUNDEDNESS_ENABLED:-true}"
GROUNDEDNESS_MAX_CLAIMS="${GROUNDEDNESS_MAX_CLAIMS:-8}"
GROUNDEDNESS_MAX_NEW_TOKENS="${GROUNDEDNESS_MAX_NEW_TOKENS:-96}"
GROUNDEDNESS_UNSUPPORTED_PENALTY="${GROUNDEDNESS_UNSUPPORTED_PENALTY:-0.25}"
GROUNDEDNESS_CONTRADICTED_PENALTY="${GROUNDEDNESS_CONTRADICTED_PENALTY:-0.75}"
GROUNDEDNESS_MIN_SCORE="${GROUNDEDNESS_MIN_SCORE:-0.85}"
GROUNDEDNESS_MIN_DISTILL_CAPTION_SCORE="${GROUNDEDNESS_MIN_DISTILL_CAPTION_SCORE:-0.65}"
TEACHER_CONFIDENCE_ENABLED="${TEACHER_CONFIDENCE_ENABLED:-true}"
REGENERATE_MIN_TEACHER_SCORE="${REGENERATE_MIN_TEACHER_SCORE:-0.65}"
REGENERATE_MIN_NORMALIZED_IMPROVEMENT="${REGENERATE_MIN_NORMALIZED_IMPROVEMENT:-0.30}"
DISTILL_MIN_CAPTION_SCORE="${DISTILL_MIN_CAPTION_SCORE:-0.65}"
SUPERVISED_CAPTION_QA_ENABLED="${SUPERVISED_CAPTION_QA_ENABLED:-false}"
CAPTION_QA_JSONL="${CAPTION_QA_JSONL:-}"
CAPTION_QA_JUDGE_BASE_URL="${CAPTION_QA_JUDGE_BASE_URL:-}"
CAPTION_QA_JUDGE_MODEL="${CAPTION_QA_JUDGE_MODEL:-}"
CAPTION_QA_JUDGE_API_KEY="${CAPTION_QA_JUDGE_API_KEY:-EMPTY}"
CAPTION_QA_MAX_CONCURRENCY="${CAPTION_QA_MAX_CONCURRENCY:-16}"
CAPTION_QA_TIMEOUT_SECONDS="${CAPTION_QA_TIMEOUT_SECONDS:-60}"
CAPTION_QA_REWARD_WEIGHT="${CAPTION_QA_REWARD_WEIGHT:-1.0}"
# Direct grounding is an external supervised-anchor ablation, not part of the
# original CycleGRPO path.  Keep it opt-in so gRefCOCO no-target rows retain
# their outer-caption GRPO update by default.
DIRECT_GROUNDING_ENABLED="${DIRECT_GROUNDING_ENABLED:-false}"
DIRECT_GROUNDING_ROLLOUTS="${DIRECT_GROUNDING_ROLLOUTS:-6}"
DIRECT_GROUNDING_LOSS_WEIGHT="${DIRECT_GROUNDING_LOSS_WEIGHT:-0.5}"
DIRECT_GROUNDING_WARMUP_START_STEP="${DIRECT_GROUNDING_WARMUP_START_STEP:-10}"
DIRECT_GROUNDING_WARMUP_END_STEP="${DIRECT_GROUNDING_WARMUP_END_STEP:-30}"
DIRECT_GROUNDING_INCLUDE_NO_TARGET="${DIRECT_GROUNDING_INCLUDE_NO_TARGET:-false}"
DIRECT_GROUNDING_INCLUDE_POSITIVE_SOURCES="${DIRECT_GROUNDING_INCLUDE_POSITIVE_SOURCES:-false}"
DIRECT_GROUNDING_INCLUDE_LABEL_SOURCES="${DIRECT_GROUNDING_INCLUDE_LABEL_SOURCES:-false}"
DIRECT_GROUNDING_CONSUME_NO_TARGET_CAPTION="${DIRECT_GROUNDING_CONSUME_NO_TARGET_CAPTION:-false}"
DIRECT_MASK_CE_ENABLED="${DIRECT_MASK_CE_ENABLED:-false}"
DIRECT_MASK_CE_LOSS_WEIGHT="${DIRECT_MASK_CE_LOSS_WEIGHT:-0.02}"
SAVE_FREQ="${SAVE_FREQ:-5}"
SAVE_LIMIT="${SAVE_LIMIT:-20}"
# A frozen-teacher run must start from MODEL_PATH. Set RESUME=true only when
# continuing a checkpoint produced by this same frozen-teacher experiment.
RESUME="${RESUME:-false}"

RUN_NAME="${RUN_NAME:-refcoco10k_opsd_frozen_teacher}"
RUN_ROOT="${RUN_ROOT:-${REPO_DIR}/logs/refcoco10k_opsd_frozen_teacher}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_ROOT}/checkpoints}"
CACHE_DIR="${CACHE_DIR:-${BASE_DIR}/cache}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_LOG="${RUN_LOG:-${RUN_ROOT}/train_${RUN_STAMP}.log}"
# Keep Ray's object store and spill files on the local tmpfs.  RUN_ROOT is on
# the persistent workspace mount, which can be nearly full independently.
RAY_SHORT_ROOT="${RAY_SHORT_ROOT:-/dev/shm/cgrpo-ray-${UID:-$(id -u)}}"

if [[ ! -d "${REPO_DIR}" ]]; then
    echo "Repository directory not found: ${REPO_DIR}" >&2
    exit 1
fi

if [[ ! "${TEACHER_EMA_DECAY}" =~ ^(0|1)(\.[0-9]+)?$ ]] \
    || ! awk -v value="${TEACHER_EMA_DECAY}" 'BEGIN { exit !(value > 0 && value <= 1) }'; then
    echo "TEACHER_EMA_DECAY must be in (0, 1]: ${TEACHER_EMA_DECAY}" >&2
    exit 1
fi

if [[ "${RESUME}" != "true" && "${RESUME}" != "false" ]]; then
    echo "RESUME must be true or false: ${RESUME}" >&2
    exit 1
fi

if [[ "${PRESERVE_ORIGINAL_GRPO}" != "true" && "${PRESERVE_ORIGINAL_GRPO}" != "false" ]]; then
    echo "PRESERVE_ORIGINAL_GRPO must be true or false: ${PRESERVE_ORIGINAL_GRPO}" >&2
    exit 1
fi

if [[ "${OPSD_ENABLED}" != "true" && "${OPSD_ENABLED}" != "false" ]]; then
    echo "OPSD_ENABLED must be true or false: ${OPSD_ENABLED}" >&2
    exit 1
fi

for bool_name in \
    PIXEL_IOU_ENABLED \
    ROUTING_ENABLED \
    CAPTION_SAFETY_ENABLED \
    CAPTION_SAFETY_FORCE_REGENERATE \
    CAPTION_BLOCK_SPECIAL_TOKEN_VOCAB \
    EMA_TEACHER_ENABLED \
    TEACHER_ANALYSIS_ENABLED \
    GROUNDEDNESS_ENABLED \
    SUPERVISED_CAPTION_QA_ENABLED \
    DIRECT_GROUNDING_ENABLED \
    DIRECT_GROUNDING_INCLUDE_NO_TARGET \
    DIRECT_GROUNDING_INCLUDE_POSITIVE_SOURCES \
    DIRECT_GROUNDING_INCLUDE_LABEL_SOURCES \
    DIRECT_GROUNDING_CONSUME_NO_TARGET_CAPTION \
    DIRECT_MASK_CE_ENABLED; do
    bool_value="${!bool_name}"
    if [[ "${bool_value}" != "true" && "${bool_value}" != "false" ]]; then
        echo "${bool_name} must be true or false: ${bool_value}" >&2
        exit 1
    fi
done

if [[ "${SUPERVISED_CAPTION_QA_ENABLED}" == "true" ]]; then
    for caption_qa_required in CAPTION_QA_JSONL CAPTION_QA_JUDGE_BASE_URL CAPTION_QA_JUDGE_MODEL; do
        if [[ -z "${!caption_qa_required}" ]]; then
            echo "${caption_qa_required} is required when SUPERVISED_CAPTION_QA_ENABLED=true." >&2
            exit 1
        fi
    done
    if [[ ! -f "${CAPTION_QA_JSONL}" ]]; then
        echo "CAPTION_QA_JSONL not found: ${CAPTION_QA_JSONL}" >&2
        exit 1
    fi
fi

if [[ ! "${DIRECT_GROUNDING_ROLLOUTS}" =~ ^[2-9][0-9]*$ ]] \
    || [[ ! "${CAPTION_QA_MAX_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
    echo "DIRECT_GROUNDING_ROLLOUTS must be >=2 and CAPTION_QA_MAX_CONCURRENCY must be positive." >&2
    exit 1
fi

if [[ ! "${DIRECT_GROUNDING_WARMUP_START_STEP}" =~ ^[0-9]+$ ]] \
    || [[ ! "${DIRECT_GROUNDING_WARMUP_END_STEP}" =~ ^[0-9]+$ ]] \
    || (( DIRECT_GROUNDING_WARMUP_END_STEP < DIRECT_GROUNDING_WARMUP_START_STEP )); then
    echo "Direct grounding warmup must satisfy 0 <= start <= end: ${DIRECT_GROUNDING_WARMUP_START_STEP}, ${DIRECT_GROUNDING_WARMUP_END_STEP}" >&2
    exit 1
fi

for direct_weight_name in DIRECT_GROUNDING_LOSS_WEIGHT DIRECT_MASK_CE_LOSS_WEIGHT; do
    if ! awk -v value="${!direct_weight_name}" 'BEGIN { exit !(value >= 0) }'; then
        echo "${direct_weight_name} must be non-negative: ${!direct_weight_name}" >&2
        exit 1
    fi
done

if [[ "${DIRECT_GROUNDING_CONSUME_NO_TARGET_CAPTION}" != "false" ]]; then
    echo "DIRECT_GROUNDING_CONSUME_NO_TARGET_CAPTION must be false: direct grounding is additive to main no-target caption GRPO." >&2
    exit 1
fi

if [[ "${ROUTING_ENABLED}" == "true" && "${OPSD_ENABLED}" != "true" ]]; then
    echo "ROUTING_ENABLED=true requires OPSD_ENABLED=true." >&2
    exit 1
fi

if [[ "${ROUTING_ENABLED}" == "true" && "${PIXEL_IOU_ENABLED}" != "true" ]]; then
    echo "ROUTING_ENABLED=true requires PIXEL_IOU_ENABLED=true." >&2
    exit 1
fi

if [[ "${ROUTING_ENABLED}" == "true" && "${EMA_TEACHER_ENABLED}" != "true" ]]; then
    echo "ROUTING_ENABLED=true requires EMA_TEACHER_ENABLED=true." >&2
    exit 1
fi

if [[ "${GROUNDEDNESS_ENABLED}" == "true" && "${TEACHER_EMA_DECAY}" != "1.0" ]]; then
    echo "GROUNDEDNESS_ENABLED=true requires TEACHER_EMA_DECAY=1.0." >&2
    exit 1
fi

for groundedness_count_name in GROUNDEDNESS_MAX_CLAIMS GROUNDEDNESS_MAX_NEW_TOKENS; do
    groundedness_count_value="${!groundedness_count_name}"
    if [[ ! "${groundedness_count_value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${groundedness_count_name} must be a positive integer: ${groundedness_count_value}" >&2
        exit 1
    fi
done

for groundedness_penalty_name in GROUNDEDNESS_UNSUPPORTED_PENALTY GROUNDEDNESS_CONTRADICTED_PENALTY; do
    groundedness_penalty_value="${!groundedness_penalty_name}"
    if [[ ! "${groundedness_penalty_value}" =~ ^(0|[1-9][0-9]*)(\.[0-9]+)?$ ]] \
        || ! awk -v value="${groundedness_penalty_value}" 'BEGIN { exit !(value >= 0) }'; then
        echo "${groundedness_penalty_name} must be non-negative: ${groundedness_penalty_value}" >&2
        exit 1
    fi
done

for groundedness_score_name in GROUNDEDNESS_MIN_SCORE GROUNDEDNESS_MIN_DISTILL_CAPTION_SCORE; do
    groundedness_score_value="${!groundedness_score_name}"
    if [[ ! "${groundedness_score_value}" =~ ^(0|1)(\.[0-9]+)?$ ]] \
        || ! awk -v value="${groundedness_score_value}" 'BEGIN { exit !(value >= 0 && value <= 1) }'; then
        echo "${groundedness_score_name} must be in [0, 1]: ${groundedness_score_value}" >&2
        exit 1
    fi
done

if [[ ! "${CAPTION_ANCHOR_KL_COEF}" =~ ^(0|[1-9][0-9]*)(\.[0-9]+)?$ ]] \
    || ! awk -v value="${CAPTION_ANCHOR_KL_COEF}" 'BEGIN { exit !(value >= 0) }'; then
    echo "CAPTION_ANCHOR_KL_COEF must be a non-negative number: ${CAPTION_ANCHOR_KL_COEF}" >&2
    exit 1
fi

if [[ "${CAPTION_ANCHOR_KL_ALL_SAFE_ROUTES}" != "true" \
    && "${CAPTION_ANCHOR_KL_ALL_SAFE_ROUTES}" != "false" ]]; then
    echo "CAPTION_ANCHOR_KL_ALL_SAFE_ROUTES must be true or false: ${CAPTION_ANCHOR_KL_ALL_SAFE_ROUTES}" >&2
    exit 1
fi

if [[ ! "${SEGMENTATION_ANCHOR_KL_COEF}" =~ ^(0|[1-9][0-9]*)(\.[0-9]+)?$ ]] \
    || ! awk -v value="${SEGMENTATION_ANCHOR_KL_COEF}" 'BEGIN { exit !(value >= 0) }'; then
    echo "SEGMENTATION_ANCHOR_KL_COEF must be a non-negative number: ${SEGMENTATION_ANCHOR_KL_COEF}" >&2
    exit 1
fi

if [[ "${ASYMMETRIC_GRADIENT_PROJECTION}" != "true" \
    && "${ASYMMETRIC_GRADIENT_PROJECTION}" != "false" ]]; then
    echo "ASYMMETRIC_GRADIENT_PROJECTION must be true or false: ${ASYMMETRIC_GRADIENT_PROJECTION}" >&2
    exit 1
fi

if [[ "${JSD_BLOCK_CAPTION_SPECIAL_TOKEN_VOCAB}" != "true" \
    && "${JSD_BLOCK_CAPTION_SPECIAL_TOKEN_VOCAB}" != "false" ]]; then
    echo "JSD_BLOCK_CAPTION_SPECIAL_TOKEN_VOCAB must be true or false: ${JSD_BLOCK_CAPTION_SPECIAL_TOKEN_VOCAB}" >&2
    exit 1
fi

if [[ "${TEACHER_CONFIDENCE_ENABLED}" != "true" \
    && "${TEACHER_CONFIDENCE_ENABLED}" != "false" ]]; then
    echo "TEACHER_CONFIDENCE_ENABLED must be true or false: ${TEACHER_CONFIDENCE_ENABLED}" >&2
    exit 1
fi

for threshold_name in REGENERATE_MIN_TEACHER_SCORE REGENERATE_MIN_NORMALIZED_IMPROVEMENT DISTILL_MIN_CAPTION_SCORE; do
    threshold_value="${!threshold_name}"
    if [[ ! "${threshold_value}" =~ ^(0|1)(\.[0-9]+)?$ ]] \
        || ! awk -v value="${threshold_value}" 'BEGIN { exit !(value >= 0 && value <= 1) }'; then
        echo "${threshold_name} must be a number in [0, 1]: ${threshold_value}" >&2
        exit 1
    fi
done

if [[ -n "${MAX_STEPS}" && ! "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_STEPS must be empty or a positive integer: ${MAX_STEPS}" >&2
    exit 1
fi

if [[ ! "${SAVE_LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "SAVE_LIMIT must be a positive integer: ${SAVE_LIMIT}" >&2
    exit 1
fi

if [[ ! "${CAPTION_MAX_RESPONSE_LENGTH}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CAPTION_MAX_RESPONSE_LENGTH must be a positive integer: ${CAPTION_MAX_RESPONSE_LENGTH}" >&2
    exit 1
fi

if [[ ! "${SEGMENTATION_MAX_RESPONSE_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "SEGMENTATION_MAX_RESPONSE_TOKENS must be a positive integer: ${SEGMENTATION_MAX_RESPONSE_TOKENS}" >&2
    exit 1
fi

TRAINER_MAX_STEPS_ARG=()
if [[ -n "${MAX_STEPS}" ]]; then
    TRAINER_MAX_STEPS_ARG=("trainer.max_steps=${MAX_STEPS}")
fi

# OmegaConf parses an empty command-line string as None. Keep disabled optional
# caption-QA fields in their string defaults instead of overriding them with None.
CAPTION_QA_OVERRIDES=(
    "worker.supervised_anchors.caption_qa.enabled=${SUPERVISED_CAPTION_QA_ENABLED}"
)
if [[ "${SUPERVISED_CAPTION_QA_ENABLED}" == "true" ]]; then
    CAPTION_QA_OVERRIDES+=(
        "worker.supervised_anchors.caption_qa.qa_jsonl=${CAPTION_QA_JSONL}"
        "worker.supervised_anchors.caption_qa.judge_base_url=${CAPTION_QA_JUDGE_BASE_URL}"
        "worker.supervised_anchors.caption_qa.judge_model=${CAPTION_QA_JUDGE_MODEL}"
        "worker.supervised_anchors.caption_qa.judge_api_key=${CAPTION_QA_JUDGE_API_KEY}"
        "worker.supervised_anchors.caption_qa.max_concurrency=${CAPTION_QA_MAX_CONCURRENCY}"
        "worker.supervised_anchors.caption_qa.timeout_seconds=${CAPTION_QA_TIMEOUT_SECONDS}"
        "worker.supervised_anchors.caption_qa.reward_weight=${CAPTION_QA_REWARD_WEIGHT}"
    )
fi

if [[ "${CONDA_PREFIX:-}" != "${ENV_DIR}" ]] && command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${ENV_DIR}"
fi

export PATH="${ENV_DIR}/bin:${PATH}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_DIR}/bin/python3}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

required_paths=(
    "${TRAIN_DATA}"
    "${VAL_DATA}"
    "${MODEL_PATH}/config.json"
    "${MODEL_PATH}/model.safetensors.index.json"
    "${MODEL_PATH}/mask_tokenizer_256x2.pth"
    "${MODEL_PATH}/sam2.1_hiera_large.pt"
    "${REPO_DIR}/projects/rl/config.yaml"
    "${REPO_DIR}/projects/rl/format_prompt/non_thinking.jinja"
)
for path in "${required_paths[@]}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Required path not found: ${path}" >&2
        exit 1
    fi
done

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
    if (( GPU_COUNT < NUM_GPUS )); then
        echo "Expected at least ${NUM_GPUS} GPUs, but nvidia-smi found ${GPU_COUNT}." >&2
        exit 1
    fi
fi

mkdir -p \
    "${RUN_ROOT}" \
    "${CHECKPOINT_DIR}" \
    "${RUN_ROOT}/wandb" \
    "${RUN_ROOT}/ray" \
    "${CACHE_DIR}/huggingface" \
    "${CACHE_DIR}/hf_datasets" \
    "${CACHE_DIR}/modelscope"

if [[ "${RAY_SHORT_ROOT}" != /* ]] || (( ${#RAY_SHORT_ROOT} > 32 )); then
    echo "RAY_SHORT_ROOT must be an absolute path no longer than 32 characters: ${RAY_SHORT_ROOT}" >&2
    exit 1
fi

if [[ -L "${RAY_SHORT_ROOT}" ]]; then
    echo "RAY_SHORT_ROOT must be a real local directory, not a symlink: ${RAY_SHORT_ROOT}" >&2
    echo "Use a new short path on a local filesystem; old launchers linked this path to RUN_ROOT." >&2
    exit 1
fi
mkdir -p "${RAY_SHORT_ROOT}"

if [[ ! -d "${RAY_SHORT_ROOT}" ]] || (( ${#RAY_SHORT_ROOT} > 32 )); then
    echo "RAY_SHORT_ROOT must be an absolute path no longer than 32 characters: ${RAY_SHORT_ROOT}" >&2
    exit 1
fi

RAY_TMP_USE_PERCENT="$(df -P "${RAY_SHORT_ROOT}" | awk 'NR == 2 {gsub(/%/, "", $5); print $5}')"
if [[ ! "${RAY_TMP_USE_PERCENT}" =~ ^[0-9]+$ ]] || (( RAY_TMP_USE_PERCENT >= 95 )); then
    echo "Ray temporary filesystem must be below 95% utilization: ${RAY_SHORT_ROOT}" >&2
    df -h "${RAY_SHORT_ROOT}" >&2
    exit 1
fi

validate_dataset_images() {
    local dataset_path="$1"
    "${PYTHON_BIN}" - "${dataset_path}" <<'PY'
import os
import sys

import pyarrow.parquet as pq

dataset_path = sys.argv[1]
parquet = pq.ParquetFile(dataset_path)
if "images" not in parquet.schema_arrow.names:
    raise RuntimeError(f"{dataset_path} has no 'images' column")

checked = 0
for batch in parquet.iter_batches(columns=["images"], batch_size=2048):
    for paths in batch.column(0).to_pylist():
        if not isinstance(paths, list) or not paths:
            raise RuntimeError(f"Invalid images entry: {paths!r}")
        for image_path in paths:
            checked += 1
            if not os.path.isfile(image_path):
                raise FileNotFoundError(
                    f"Dataset image does not exist: {image_path}\n"
                    "Re-export the parquet for this server or repair its images paths."
                )

print(f"Verified {checked} image paths in {dataset_path}")
PY
}

validate_dataset_images "${TRAIN_DATA}"
if [[ "${VAL_DATA}" != "${TRAIN_DATA}" ]]; then
    validate_dataset_images "${VAL_DATA}"
fi

echo "CycleGRPO training output: ${RUN_LOG}"
exec >>"${RUN_LOG}" 2>&1

INHERITED_RAY_ADDRESS="${RAY_ADDRESS:-}"
# Volcengine injects a Python 3.12 / Ray 2.53 cluster address. This job uses the
# repository Python 3.10 environment, so let ray.init() create a matching local cluster.
unset RAY_ADDRESS
unset RAY_NAMESPACE

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export HF_HOME="${CACHE_DIR}/huggingface"
export HF_DATASETS_CACHE="${CACHE_DIR}/hf_datasets"
export MODELSCOPE_CACHE="${CACHE_DIR}/modelscope"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_DIR="${RUN_ROOT}/wandb"
TRAINER_LOGGERS="${TRAINER_LOGGERS:-[\"file\",\"wandb\"]}"
export RAY_TMPDIR="${RAY_SHORT_ROOT}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

cd "${REPO_DIR}"

echo "Start time: $(date --iso-8601=seconds)"
echo "Repository: ${REPO_DIR}"
echo "Training data: ${TRAIN_DATA}"
echo "Model: ${MODEL_PATH}"
echo "Teacher EMA decay: ${TEACHER_EMA_DECAY} (1.0 freezes the initial SAMTok teacher)"
echo "OPSD enabled: ${OPSD_ENABLED} (false uses original HTG token grading)"
echo "Pixel-IoU reward: ${PIXEL_IOU_ENABLED}; OPSD routing: ${ROUTING_ENABLED}"
echo "Positive segmentation mask protocol: union of all complete legal SAMTok groups"
echo "Segmentation response limit: ${SEGMENTATION_MAX_RESPONSE_TOKENS} tokens"
echo "Caption safety: ${CAPTION_SAFETY_ENABLED} (force regenerate: ${CAPTION_SAFETY_FORCE_REGENERATE})"
echo "Caption special-token generation block: ${CAPTION_BLOCK_SPECIAL_TOKEN_VOCAB}"
echo "EMA teacher: ${EMA_TEACHER_ENABLED}; teacher analysis: ${TEACHER_ANALYSIS_ENABLED}"
echo "Preserve original caption GRPO: ${PRESERVE_ORIGINAL_GRPO}"
echo "Caption anchor KL: ${CAPTION_ANCHOR_KL_COEF} (all safe routes: ${CAPTION_ANCHOR_KL_ALL_SAFE_ROUTES})"
echo "Segmentation anchor KL: ${SEGMENTATION_ANCHOR_KL_COEF} (all cycle localization responses)"
echo "Asymmetric caption-to-segmentation gradient projection: ${ASYMMETRIC_GRADIENT_PROJECTION}"
echo "JSD blocks caption special-token vocabulary: ${JSD_BLOCK_CAPTION_SPECIAL_TOKEN_VOCAB}"
echo "Caption groundedness: ${GROUNDEDNESS_ENABLED} (unsupported=${GROUNDEDNESS_UNSUPPORTED_PENALTY}, contradicted=${GROUNDEDNESS_CONTRADICTED_PENALTY}, min score=${GROUNDEDNESS_MIN_SCORE}, min distill R_Ci=${GROUNDEDNESS_MIN_DISTILL_CAPTION_SCORE})"
echo "High-confidence teacher gate: ${TEACHER_CONFIDENCE_ENABLED} (regenerate score >= ${REGENERATE_MIN_TEACHER_SCORE}, normalized gain >= ${REGENERATE_MIN_NORMALIZED_IMPROVEMENT}, distill R_Ci >= ${DISTILL_MIN_CAPTION_SCORE})"
echo "DLC-QA caption anchor: ${SUPERVISED_CAPTION_QA_ENABLED} (weight=${CAPTION_QA_REWARD_WEIGHT}, all questions per eligible rollout)"
echo "Direct grounding anchor: ${DIRECT_GROUNDING_ENABLED} (K=${DIRECT_GROUNDING_ROLLOUTS}, target weight=${DIRECT_GROUNDING_LOSS_WEIGHT}, warmup=${DIRECT_GROUNDING_WARMUP_START_STEP}-${DIRECT_GROUNDING_WARMUP_END_STEP}, human-positive=${DIRECT_GROUNDING_INCLUDE_POSITIVE_SOURCES}, label-positive=${DIRECT_GROUNDING_INCLUDE_LABEL_SOURCES}, no-target=${DIRECT_GROUNDING_INCLUDE_NO_TARGET}, consume no-target caption=${DIRECT_GROUNDING_CONSUME_NO_TARGET_CAPTION})"
echo "Direct GT-mask CE anchor: ${DIRECT_MASK_CE_ENABLED} (weight=${DIRECT_MASK_CE_LOSS_WEIGHT}, human positive expressions only)"
echo "Resume: ${RESUME}"
echo "Maximum global step: ${MAX_STEPS:-<full epoch>}"
echo "Caption response limit: ${CAPTION_MAX_RESPONSE_LENGTH} tokens"
echo "Checkpoint directory: ${CHECKPOINT_DIR}"
echo "Checkpoint retention limit: ${SAVE_LIMIT}"
echo "Trainer loggers: ${TRAINER_LOGGERS}"
echo "Ray temp root: ${RAY_SHORT_ROOT} (local filesystem ${RAY_TMP_USE_PERCENT}% used)"
echo "Ray session logs: ${RAY_SHORT_ROOT}/ray"
echo "Ignored inherited RAY_ADDRESS: ${INHERITED_RAY_ADDRESS:-<unset>}"
"${PYTHON_BIN}" --version
"${PYTHON_BIN}" -c 'import ray, torch, vllm; print(f"Ray: {ray.__version__}"); print(f"PyTorch: {torch.__version__}"); print(f"vLLM: {vllm.__version__}"); print(f"CUDA devices: {torch.cuda.device_count()}")'

# Generic trainer validation does not run the RefCOCO mask/cIoU path. Evaluate
# the checkpoints saved below with the offline RefCOCO evaluator instead.
exec "${PYTHON_BIN}" -m verl.trainer.main \
    config=projects/rl/config.yaml \
    "data.train_files=['${TRAIN_DATA}']" \
    "data.val_files=['${VAL_DATA}']" \
    data.format_prompt="${REPO_DIR}/projects/rl/format_prompt/non_thinking.jinja" \
    data.region_format=mask_token \
    data.shuffle=true \
    data.seed=1 \
    data.rollout_batch_size="${ROLLOUT_BATCH_SIZE}" \
    data.max_prompt_length=8192 \
    data.max_response_length="${CAPTION_MAX_RESPONSE_LENGTH}" \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.model.freeze_vision_tower=true \
    worker.actor.model.enable_gradient_checkpointing=true \
    worker.actor.optimize_captioner=true \
    worker.actor.optimize_segmenter=true \
    worker.actor.global_batch_size="${ACTOR_GLOBAL_BATCH_SIZE}" \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=2 \
    worker.actor.dynamic_batching=true \
    worker.actor.padding_free=true \
    worker.rollout.n="${CAPTION_ROLLOUTS}" \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.gpu_memory_utilization=0.6 \
    worker.rollout.max_num_batched_tokens=16384 \
    worker.rollout.disable_tqdm=true \
    worker.opsd.enabled="${OPSD_ENABLED}" \
    worker.opsd.localization_rollouts="${LOCALIZATION_ROLLOUTS}" \
    worker.opsd.caption_loss_weight=0.5 \
    worker.opsd.localization_loss_weight=0.5 \
    worker.opsd.caption_anchor_kl_coef="${CAPTION_ANCHOR_KL_COEF}" \
    worker.opsd.caption_anchor_kl_all_safe_routes="${CAPTION_ANCHOR_KL_ALL_SAFE_ROUTES}" \
    worker.opsd.segmentation_anchor_kl_coef="${SEGMENTATION_ANCHOR_KL_COEF}" \
    worker.opsd.asymmetric_gradient_projection="${ASYMMETRIC_GRADIENT_PROJECTION}" \
    worker.opsd.teacher_confidence.enabled="${TEACHER_CONFIDENCE_ENABLED}" \
    worker.opsd.teacher_confidence.regenerate_min_teacher_score="${REGENERATE_MIN_TEACHER_SCORE}" \
    worker.opsd.teacher_confidence.regenerate_min_normalized_improvement="${REGENERATE_MIN_NORMALIZED_IMPROVEMENT}" \
    worker.opsd.teacher_confidence.distill_min_caption_score="${DISTILL_MIN_CAPTION_SCORE}" \
    worker.opsd.pixel_iou.enabled="${PIXEL_IOU_ENABLED}" \
    worker.opsd.pixel_iou.segmentation_max_response_tokens="${SEGMENTATION_MAX_RESPONSE_TOKENS}" \
    worker.opsd.routing.enabled="${ROUTING_ENABLED}" \
    worker.opsd.routing.low_threshold=0.5 \
    worker.opsd.routing.high_threshold=0.85 \
    worker.opsd.routing.preserve_original_grpo="${PRESERVE_ORIGINAL_GRPO}" \
    worker.opsd.caption_safety.enabled="${CAPTION_SAFETY_ENABLED}" \
    worker.opsd.caption_safety.max_response_tokens="${CAPTION_MAX_RESPONSE_LENGTH}" \
    worker.opsd.caption_safety.force_regenerate="${CAPTION_SAFETY_FORCE_REGENERATE}" \
    worker.opsd.caption_safety.block_special_token_vocab="${CAPTION_BLOCK_SPECIAL_TOKEN_VOCAB}" \
    worker.opsd.distillation.block_caption_special_token_vocab="${JSD_BLOCK_CAPTION_SPECIAL_TOKEN_VOCAB}" \
    worker.opsd.ema_teacher.enabled="${EMA_TEACHER_ENABLED}" \
    worker.opsd.ema_teacher.decay="${TEACHER_EMA_DECAY}" \
    worker.opsd.teacher_analysis.enabled="${TEACHER_ANALYSIS_ENABLED}" \
    worker.opsd.groundedness.enabled="${GROUNDEDNESS_ENABLED}" \
    worker.opsd.groundedness.teacher_must_be_frozen=true \
    worker.opsd.groundedness.max_claims="${GROUNDEDNESS_MAX_CLAIMS}" \
    worker.opsd.groundedness.max_new_tokens="${GROUNDEDNESS_MAX_NEW_TOKENS}" \
    worker.opsd.groundedness.temperature=0.0 \
    worker.opsd.groundedness.unsupported_penalty="${GROUNDEDNESS_UNSUPPORTED_PENALTY}" \
    worker.opsd.groundedness.contradicted_penalty="${GROUNDEDNESS_CONTRADICTED_PENALTY}" \
    worker.opsd.groundedness.min_checked_claims=1 \
    worker.opsd.groundedness.min_groundedness_score="${GROUNDEDNESS_MIN_SCORE}" \
    worker.opsd.groundedness.min_distill_caption_score="${GROUNDEDNESS_MIN_DISTILL_CAPTION_SCORE}" \
    worker.opsd.groundedness.no_target_enabled=false \
    worker.opsd.groundedness.token_jsd_enabled=false \
    worker.opsd.groundedness.token_jsd_multiplier=1.0 \
    "${CAPTION_QA_OVERRIDES[@]}" \
    worker.supervised_anchors.direct_grounding.enabled="${DIRECT_GROUNDING_ENABLED}" \
    worker.supervised_anchors.direct_grounding.rollouts="${DIRECT_GROUNDING_ROLLOUTS}" \
    worker.supervised_anchors.direct_grounding.loss_weight="${DIRECT_GROUNDING_LOSS_WEIGHT}" \
    worker.supervised_anchors.direct_grounding.warmup_start_step="${DIRECT_GROUNDING_WARMUP_START_STEP}" \
    worker.supervised_anchors.direct_grounding.warmup_end_step="${DIRECT_GROUNDING_WARMUP_END_STEP}" \
    worker.supervised_anchors.direct_grounding.include_no_target="${DIRECT_GROUNDING_INCLUDE_NO_TARGET}" \
    worker.supervised_anchors.direct_grounding.include_positive_sources="${DIRECT_GROUNDING_INCLUDE_POSITIVE_SOURCES}" \
    worker.supervised_anchors.direct_grounding.include_label_sources="${DIRECT_GROUNDING_INCLUDE_LABEL_SOURCES}" \
    worker.supervised_anchors.direct_grounding.consume_no_target_caption="${DIRECT_GROUNDING_CONSUME_NO_TARGET_CAPTION}" \
    worker.supervised_anchors.direct_mask_ce.enabled="${DIRECT_MASK_CE_ENABLED}" \
    worker.supervised_anchors.direct_mask_ce.loss_weight="${DIRECT_MASK_CE_LOSS_WEIGHT}" \
    worker.supervised_anchors.direct_mask_ce.include_positive_sources=true \
    worker.reward.mask_tokenizer_path="${MODEL_PATH}/mask_tokenizer_256x2.pth" \
    worker.reward.sam2_pretrained_weight="${MODEL_PATH}/sam2.1_hiera_large.pt" \
    trainer.project_name=cyclegrpo \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    "${TRAINER_MAX_STEPS_ARG[@]+"${TRAINER_MAX_STEPS_ARG[@]}"}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.val_freq=-1 \
    trainer.val_before_train=false \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.save_limit="${SAVE_LIMIT}" \
    trainer.save_checkpoint_path="${CHECKPOINT_DIR}" \
    trainer.find_last_checkpoint="${RESUME}" \
    "trainer.logger=${TRAINER_LOGGERS}"
