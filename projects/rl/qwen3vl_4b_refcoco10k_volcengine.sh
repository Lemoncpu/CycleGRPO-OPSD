#!/usr/bin/env bash
# Single-node 8-GPU RefCOCO OPSD training for the current Volcengine workspace.

set -euo pipefail

BASE_DIR="${BASE_DIR:-/mnt/cxzx/workspace/data_transfer/houzhiyan}"
REPO_DIR="${REPO_DIR:-${BASE_DIR}/CycleGRPO-OPSD}"
ENV_DIR="${ENV_DIR:-${BASE_DIR}/envs/cyclegrpo}"
MODEL_PATH="${MODEL_PATH:-${BASE_DIR}/Qwen3-VL-4B-SAMTok}"
TRAIN_DATA="${TRAIN_DATA:-${BASE_DIR}/refcoco-train2014-assets/refcoco_train_10k_seed20260722.parquet}"
VAL_DATA="${VAL_DATA:-${TRAIN_DATA}}"

NUM_GPUS="${NUM_GPUS:-8}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
ACTOR_GLOBAL_BATCH_SIZE="${ACTOR_GLOBAL_BATCH_SIZE:-128}"
CAPTION_ROLLOUTS="${CAPTION_ROLLOUTS:-6}"
LOCALIZATION_ROLLOUTS="${LOCALIZATION_ROLLOUTS:-6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"

RUN_NAME="${RUN_NAME:-refcoco10k_opsd_qwen3vl4b}"
RUN_ROOT="${RUN_ROOT:-${REPO_DIR}/logs/refcoco10k_opsd}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_ROOT}/checkpoints}"
CACHE_DIR="${CACHE_DIR:-${BASE_DIR}/cache}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_LOG="${RUN_LOG:-${RUN_ROOT}/train_${RUN_STAMP}.log}"
RAY_SHORT_ROOT="${RAY_SHORT_ROOT:-/tmp/cgrpo-${UID:-$(id -u)}}"

if [[ ! -d "${REPO_DIR}" ]]; then
    echo "Repository directory not found: ${REPO_DIR}" >&2
    exit 1
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

# Ray appends session and socket names to RAY_TMPDIR. Use a short pathname to
# stay below Linux's 107-byte AF_UNIX limit while keeping the files in RUN_ROOT.
if [[ -L "${RAY_SHORT_ROOT}" ]]; then
    RAY_LINK_TARGET="$(readlink -f "${RAY_SHORT_ROOT}")"
    EXPECTED_RAY_TARGET="$(readlink -f "${RUN_ROOT}")"
    if [[ "${RAY_LINK_TARGET}" != "${EXPECTED_RAY_TARGET}" ]]; then
        echo "Ray temp link points to an unexpected directory: ${RAY_SHORT_ROOT} -> ${RAY_LINK_TARGET}" >&2
        echo "Expected target: ${EXPECTED_RAY_TARGET}" >&2
        echo "Set RAY_SHORT_ROOT to another short, unused path." >&2
        exit 1
    fi
elif [[ -e "${RAY_SHORT_ROOT}" ]]; then
    echo "Ray temp path exists and is not a symlink: ${RAY_SHORT_ROOT}" >&2
    echo "Set RAY_SHORT_ROOT to another short, unused path." >&2
    exit 1
else
    ln -s "${RUN_ROOT}" "${RAY_SHORT_ROOT}"
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
echo "Checkpoint directory: ${CHECKPOINT_DIR}"
echo "Ray temp root: ${RAY_TMPDIR} -> ${RUN_ROOT}"
echo "Ray session logs: ${RUN_ROOT}/ray"
echo "Ignored inherited RAY_ADDRESS: ${INHERITED_RAY_ADDRESS:-<unset>}"
"${PYTHON_BIN}" --version
"${PYTHON_BIN}" -c 'import ray, torch, vllm; print(f"Ray: {ray.__version__}"); print(f"PyTorch: {torch.__version__}"); print(f"vLLM: {vllm.__version__}"); print(f"CUDA devices: {torch.cuda.device_count()}")'

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
    data.max_response_length=8192 \
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
    worker.opsd.pixel_iou.enabled=true \
    worker.opsd.routing.enabled=true \
    worker.opsd.routing.low_threshold=0.5 \
    worker.opsd.routing.high_threshold=0.85 \
    worker.opsd.ema_teacher.enabled=true \
    worker.opsd.teacher_analysis.enabled=true \
    worker.reward.mask_tokenizer_path="${MODEL_PATH}/mask_tokenizer_256x2.pth" \
    worker.reward.sam2_pretrained_weight="${MODEL_PATH}/sam2.1_hiera_large.pt" \
    trainer.project_name=cyclegrpo \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.val_freq=-1 \
    trainer.val_before_train=false \
    trainer.save_freq=5 \
    trainer.save_limit=20 \
    trainer.save_checkpoint_path="${CHECKPOINT_DIR}" \
    trainer.find_last_checkpoint=true \
    'trainer.logger=["file","wandb"]'
