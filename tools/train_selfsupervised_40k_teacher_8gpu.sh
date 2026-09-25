#!/usr/bin/env bash
# Pure self-supervised OPSD/teacher training on the 40k cycle dataset.
# One self-contained invocation starts the repository's local single-node Ray
# trainer on all eight GPUs.  No external Llama judge or auxiliary loader is
# used in this experiment.

set -euo pipefail

BASE_DIR="${BASE_DIR:-/volume/ybo/xyc}"
REPO_DIR="${REPO_DIR:-${BASE_DIR}/CycleGRPO-OPSD}"
ENV_DIR="${ENV_DIR:-${BASE_DIR}/envs/cyclegrpo}"
DATA_ROOT="${DATA_ROOT:-${BASE_DIR}/datasets/cyclegrpo100k_scaling_20260911}"
MODEL_PATH="${MODEL_PATH:-${BASE_DIR}/Qwen3-VL-4B-SAMTok}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_ROOT}/cyclegrpo_selfsupervised_40k.parquet}"
RUN_NAME="${RUN_NAME:-cyclegrpo_selfsupervised_40k_teacher_bs128_8gpu}"
RUN_ROOT="${RUN_ROOT:-${REPO_DIR}/logs/${RUN_NAME}}"
RAY_BIN="${RAY_BIN:-${ENV_DIR}/bin/ray}"
RAY_PORT="${RAY_PORT:-29679}"
RAY_ADDRESS="127.0.0.1:${RAY_PORT}"
RAY_NAMESPACE="${RAY_NAMESPACE:-${RUN_NAME}}"

if [[ ! -d "${REPO_DIR}" ]]; then
    echo "Repository directory not found: ${REPO_DIR}" >&2
    exit 1
fi
if [[ ! -x "${ENV_DIR}/bin/python3" ]]; then
    echo "Python executable not found: ${ENV_DIR}/bin/python3" >&2
    exit 1
fi
if [[ ! -x "${RAY_BIN}" ]]; then
    echo "Ray executable not found: ${RAY_BIN}" >&2
    exit 1
fi
if [[ ! -f "${TRAIN_DATA}" ]]; then
    echo "Training parquet not found: ${TRAIN_DATA}" >&2
    exit 1
fi

cd "${REPO_DIR}"

export BASE_DIR REPO_DIR ENV_DIR DATA_ROOT MODEL_PATH TRAIN_DATA RUN_NAME RUN_ROOT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export NNODES="${NNODES:-1}"
export MULTINODE_ENABLED=true
export RAY_ADDRESS RAY_NAMESPACE RAY_CLUSTER_EXPECTED_NODES=1 RAY_CLUSTER_EXPECTED_GPUS=8
export LOCAL_JUDGE_ENABLED=false
export RAY_SHORT_ROOT="${RAY_SHORT_ROOT:-/dev/shm/cgrpo-self40k8}"
export RAY_LOCAL_NUM_CPUS="${RAY_LOCAL_NUM_CPUS:-64}"

# Full one-epoch runs: the launcher treats an explicitly empty MAX_STEPS as
# "no max_steps override" and derives the number of steps from the dataset.
export ROLLOUT_BATCH_SIZE=128
export ACTOR_GLOBAL_BATCH_SIZE=128
export TOTAL_EPOCHS=1
export MAX_STEPS=""
export SAVE_FREQ="${SAVE_FREQ:-5}"
export SAVE_LIMIT="${SAVE_LIMIT:-1}"
export RESUME=false
export TRAINER_LOGGERS='["file"]'

# Pixel-OPSD and all teacher-side gates are explicit in this experiment.
export OPSD_ENABLED=true
export PIXEL_IOU_ENABLED=true
export ROUTING_ENABLED=true
export CAPTION_SAFETY_ENABLED=true
export CAPTION_SAFETY_FORCE_REGENERATE=true
export CAPTION_BLOCK_SPECIAL_TOKEN_VOCAB=true
export EMA_TEACHER_ENABLED=true
export TEACHER_ANALYSIS_ENABLED=true
export TEACHER_CONFIDENCE_ENABLED=true
export TEACHER_EMA_DECAY=1.0
export PRESERVE_ORIGINAL_GRPO=true
export CAPTION_ANCHOR_KL_COEF=0.05
export CAPTION_ANCHOR_KL_ALL_SAFE_ROUTES=true
export SEGMENTATION_ANCHOR_KL_COEF=0.05
export JSD_BLOCK_CAPTION_SPECIAL_TOKEN_VOCAB=true
export ASYMMETRIC_GRADIENT_PROJECTION=false

# Positive examples that refuse or decode to an empty mask receive the
# explicit penalty.  Keep the separate no-target nonempty penalty disabled so
# this run isolates the requested positive-example penalty.
export NO_TARGET_REWARD_MODE=pixel_empty
export POSITIVE_EMPTY_MASK_PENALTY=1.0
export NO_TARGET_NONEMPTY_MASK_PENALTY=0.0
export NO_TARGET_EMPTY_AREA_TAU=0.0
export CAPTION_MAX_RESPONSE_LENGTH=256
export SEGMENTATION_MAX_RESPONSE_TOKENS=256
export MASK_DECODE_MODE=union

# Pure self-supervised path: no direct RefCOCO GRPO/CE and no DLC-QA loader.
export DIRECT_GROUNDING_ENABLED=false
export DIRECT_MASK_CE_ENABLED=false
export SUPERVISED_CAPTION_QA_ENABLED=false

mkdir -p "${RUN_ROOT}"
RAY_STARTED=false
ray_cleanup() {
    local status=$?
    if [[ "${RAY_STARTED}" == "true" ]]; then
        echo "Stopping local Ray head: ${RAY_ADDRESS}"
        RAY_STARTED=false
        "${RAY_BIN}" stop --force >/dev/null 2>&1 || true
    fi
    exit "${status}"
}
trap ray_cleanup EXIT INT TERM

echo "Starting local Ray head: address=${RAY_ADDRESS} namespace=${RAY_NAMESPACE} gpus=8"
"${RAY_BIN}" stop --force >/dev/null 2>&1 || true
RAY_STARTED=true
"${RAY_BIN}" start --head \
    --node-ip-address=127.0.0.1 \
    --port="${RAY_PORT}" \
    --num-gpus=8 \
    --dashboard-host=127.0.0.1 \
    --temp-dir="${RAY_SHORT_ROOT}" \
    --disable-usage-stats \
    >"${RUN_ROOT}/ray_start.log" 2>&1

bash "${REPO_DIR}/projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh"
