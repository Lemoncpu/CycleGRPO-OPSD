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
ASYMMETRIC_GRADIENT_PROJECTION="${ASYMMETRIC_GRADIENT_PROJECTION:-true}"
JSD_BLOCK_CAPTION_SPECIAL_TOKEN_VOCAB="${JSD_BLOCK_CAPTION_SPECIAL_TOKEN_VOCAB:-true}"
SAVE_FREQ="${SAVE_FREQ:-5}"
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
RAY_SHORT_ROOT="${RAY_SHORT_ROOT:-/tmp/cgrpo-ray-${UID:-$(id -u)}}"

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

if [[ -n "${MAX_STEPS}" && ! "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_STEPS must be empty or a positive integer: ${MAX_STEPS}" >&2
    exit 1
fi

if [[ ! "${CAPTION_MAX_RESPONSE_LENGTH}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CAPTION_MAX_RESPONSE_LENGTH must be a positive integer: ${CAPTION_MAX_RESPONSE_LENGTH}" >&2
    exit 1
fi

TRAINER_MAX_STEPS_ARG=()
if [[ -n "${MAX_STEPS}" ]]; then
    TRAINER_MAX_STEPS_ARG=("trainer.max_steps=${MAX_STEPS}")
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

if [[ "${RAY_SHORT_ROOT}" != /tmp/* ]] || (( ${#RAY_SHORT_ROOT} > 32 )); then
    echo "RAY_SHORT_ROOT must be a /tmp path no longer than 32 characters: ${RAY_SHORT_ROOT}" >&2
    exit 1
fi

if [[ -L "${RAY_SHORT_ROOT}" ]]; then
    echo "RAY_SHORT_ROOT must be a real local directory, not a symlink: ${RAY_SHORT_ROOT}" >&2
    echo "Use a new short /tmp path; old launchers linked this path to RUN_ROOT." >&2
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
export RAY_TMPDIR="${RAY_SHORT_ROOT}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

cd "${REPO_DIR}"

echo "Start time: $(date --iso-8601=seconds)"
echo "Repository: ${REPO_DIR}"
echo "Training data: ${TRAIN_DATA}"
echo "Model: ${MODEL_PATH}"
echo "Teacher EMA decay: ${TEACHER_EMA_DECAY} (1.0 freezes the initial SAMTok teacher)"
echo "Preserve original caption GRPO: ${PRESERVE_ORIGINAL_GRPO}"
echo "Caption anchor KL: ${CAPTION_ANCHOR_KL_COEF} (all safe routes: ${CAPTION_ANCHOR_KL_ALL_SAFE_ROUTES})"
echo "Segmentation anchor KL: ${SEGMENTATION_ANCHOR_KL_COEF} (all cycle localization responses)"
echo "Asymmetric caption-to-segmentation gradient projection: ${ASYMMETRIC_GRADIENT_PROJECTION}"
echo "JSD blocks caption special-token vocabulary: ${JSD_BLOCK_CAPTION_SPECIAL_TOKEN_VOCAB}"
echo "Resume: ${RESUME}"
echo "Maximum global step: ${MAX_STEPS:-<full epoch>}"
echo "Caption response limit: ${CAPTION_MAX_RESPONSE_LENGTH} tokens"
echo "Checkpoint directory: ${CHECKPOINT_DIR}"
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
    worker.opsd.enabled=true \
    worker.opsd.localization_rollouts="${LOCALIZATION_ROLLOUTS}" \
    worker.opsd.caption_loss_weight=0.5 \
    worker.opsd.localization_loss_weight=0.5 \
    worker.opsd.caption_anchor_kl_coef="${CAPTION_ANCHOR_KL_COEF}" \
    worker.opsd.caption_anchor_kl_all_safe_routes="${CAPTION_ANCHOR_KL_ALL_SAFE_ROUTES}" \
    worker.opsd.segmentation_anchor_kl_coef="${SEGMENTATION_ANCHOR_KL_COEF}" \
    worker.opsd.asymmetric_gradient_projection="${ASYMMETRIC_GRADIENT_PROJECTION}" \
    worker.opsd.pixel_iou.enabled=true \
    worker.opsd.routing.enabled=true \
    worker.opsd.routing.low_threshold=0.5 \
    worker.opsd.routing.high_threshold=0.85 \
    worker.opsd.routing.preserve_original_grpo="${PRESERVE_ORIGINAL_GRPO}" \
    worker.opsd.caption_safety.enabled=true \
    worker.opsd.caption_safety.max_response_tokens="${CAPTION_MAX_RESPONSE_LENGTH}" \
    worker.opsd.caption_safety.force_regenerate=true \
    worker.opsd.distillation.block_caption_special_token_vocab="${JSD_BLOCK_CAPTION_SPECIAL_TOKEN_VOCAB}" \
    worker.opsd.ema_teacher.enabled=true \
    worker.opsd.ema_teacher.decay="${TEACHER_EMA_DECAY}" \
    worker.opsd.teacher_analysis.enabled=true \
    worker.reward.mask_tokenizer_path="${MODEL_PATH}/mask_tokenizer_256x2.pth" \
    worker.reward.sam2_pretrained_weight="${MODEL_PATH}/sam2.1_hiera_large.pt" \
    trainer.project_name=cyclegrpo \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    "${TRAINER_MAX_STEPS_ARG[@]}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.val_freq=-1 \
    trainer.val_before_train=false \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.save_limit=20 \
    trainer.save_checkpoint_path="${CHECKPOINT_DIR}" \
    trainer.find_last_checkpoint="${RESUME}" \
    'trainer.logger=["file","wandb"]'
