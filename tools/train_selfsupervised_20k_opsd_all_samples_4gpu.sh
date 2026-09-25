#!/usr/bin/env bash
# Four-GPU self-supervised routing ablation on the historical 20k recipe.
# R_Ci low/mid/high route classification is bypassed: every eligible image
# cycle sample receives the OPSD privileged correction.  Native CycleGRPO is
# retained through PRESERVE_ORIGINAL_GRPO=true.

set -euo pipefail

BASE_DIR="${BASE_DIR:-/volume/ybo/xyc}"
REPO_DIR="${REPO_DIR:-${BASE_DIR}/CycleGRPO-OPSD}"
ENV_DIR="${ENV_DIR:-${BASE_DIR}/envs/cyclegrpo}"
MODEL_PATH="${MODEL_PATH:-${BASE_DIR}/Qwen3-VL-4B-SAMTok}"
TRAIN_DATA="${TRAIN_DATA:-${BASE_DIR}/datasets/cyclegrpo_20k_raw_seed20260820/cyclegrpo_20k_40_20_25_10_5_seed20260820.parquet}"
RUN_NAME="${RUN_NAME:-cyclegrpo20k_opsd_all_samples_routing_4gpu}"
RUN_ROOT="${RUN_ROOT:-${REPO_DIR}/logs/${RUN_NAME}}"
RAY_BIN="${RAY_BIN:-${ENV_DIR}/bin/ray}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_DIR}/bin/python3}"
RAY_PORT="${RAY_PORT:-29685}"
RAY_BIND_IP="${RAY_BIND_IP:-$(hostname -I | awk '{print $1}')}"
RAY_BIND_IP="${RAY_BIND_IP//[[:space:]]/}"
RAY_NAMESPACE="${RAY_NAMESPACE:-${RUN_NAME}}"
RAY_SHORT_ROOT="${RAY_SHORT_ROOT:-/dev/shm/cgrpo-opsd-all4}"
KEEPALIVE_SCRIPT="${KEEPALIVE_SCRIPT:-${REPO_DIR}/tools/cuda_keepalive.py}"
KEEPALIVE_MEMORY_MB="${KEEPALIVE_MEMORY_MB:-30000}"
KEEPALIVE_AFTER_TRAINING="${KEEPALIVE_AFTER_TRAINING:-true}"

for required in "${REPO_DIR}" "${PYTHON_BIN}" "${RAY_BIN}" \
    "${MODEL_PATH}/config.json" "${MODEL_PATH}/model.safetensors.index.json" \
    "${MODEL_PATH}/mask_tokenizer_256x2.pth" "${MODEL_PATH}/sam2.1_hiera_large.pt" \
    "${TRAIN_DATA}" "${KEEPALIVE_SCRIPT}"; do
    if [[ ! -e "${required}" ]]; then
        echo "Required path not found: ${required}" >&2
        exit 1
    fi
done
if [[ ! -x "${PYTHON_BIN}" || ! -x "${RAY_BIN}" ]]; then
    echo "Python/Ray executable is not executable: ${PYTHON_BIN} ${RAY_BIN}" >&2
    exit 1
fi
if [[ -z "${RAY_BIND_IP}" ]]; then
    echo "Unable to determine a local Ray bind IP." >&2
    exit 1
fi

ROWS="$(${PYTHON_BIN} - "${TRAIN_DATA}" <<'PY'
import sys
import pyarrow.parquet as pq

path = sys.argv[1]
rows = pq.ParquetFile(path).metadata.num_rows
if rows != 20000:
    raise SystemExit(f"Expected exactly 20000 training rows, got {rows}: {path}")
print(rows)
PY
)"
echo "Verified self-supervised training rows: ${ROWS}"

cd "${REPO_DIR}"
export BASE_DIR REPO_DIR ENV_DIR MODEL_PATH TRAIN_DATA RUN_NAME RUN_ROOT PYTHON_BIN RAY_BIN
export CONDA_PREFIX="${ENV_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NUM_GPUS=4 NNODES=1 MULTINODE_ENABLED=false
export RAY_ADDRESS="127.0.0.1:${RAY_PORT}" RAY_NAMESPACE
export RAY_CLUSTER_EXPECTED_NODES=1 RAY_CLUSTER_EXPECTED_GPUS=4
export LOCAL_JUDGE_ENABLED=false RAY_LOCAL_NUM_CPUS="${RAY_LOCAL_NUM_CPUS:-32}"
export RAY_SHORT_ROOT

# Historical experiment_config.json values.
export ROLLOUT_BATCH_SIZE=128 ACTOR_GLOBAL_BATCH_SIZE=128
export CAPTION_ROLLOUTS=6 LOCALIZATION_ROLLOUTS=6 TOTAL_EPOCHS=1
export MAX_STEPS="${MAX_STEPS-}"
export SAVE_FREQ="${SAVE_FREQ:-5}" SAVE_LIMIT="${SAVE_LIMIT:-2}" RESUME=false TRAINER_LOGGERS='["file"]'
export OPSD_ENABLED=true PIXEL_IOU_ENABLED=true
export ROUTING_ENABLED=true OPSD_ALL_SAMPLES_OPD=true PRESERVE_ORIGINAL_GRPO=true
export CAPTION_SAFETY_ENABLED=true CAPTION_SAFETY_FORCE_REGENERATE=true
export CAPTION_BLOCK_SPECIAL_TOKEN_VOCAB=true EMA_TEACHER_ENABLED=true
export TEACHER_ANALYSIS_ENABLED=true TEACHER_CONFIDENCE_ENABLED=true TEACHER_EMA_DECAY=1.0
export CAPTION_ANCHOR_KL_COEF=0.05 CAPTION_ANCHOR_KL_ALL_SAFE_ROUTES=true
export SEGMENTATION_ANCHOR_KL_COEF=0.05 JSD_BLOCK_CAPTION_SPECIAL_TOKEN_VOCAB=true
export ASYMMETRIC_GRADIENT_PROJECTION=false
export SECA_ENABLED=false SECA_SELF_SUPERVISED_ENABLED=false EGCA_ENABLED=false EGCA_OPD_ENABLED=false
export NO_TARGET_REWARD_MODE=pixel_empty NO_TARGET_SEGMENTATION_LOSS_WEIGHT=1.0
export POSITIVE_EMPTY_MASK_PENALTY=1.0 NO_TARGET_NONEMPTY_MASK_PENALTY=0.0 NO_TARGET_EMPTY_AREA_TAU=0.0
export CAPTION_MAX_RESPONSE_LENGTH=256 SEGMENTATION_MAX_RESPONSE_TOKENS=256
export MASK_DECODE_MODE=union LOCALIZATION_PROMPT_MODE=mixed CYCLE_PROMPT_MODE=current
export DIRECT_GROUNDING_ENABLED=false DIRECT_MASK_CE_ENABLED=false SUPERVISED_CAPTION_QA_ENABLED=false
export THREE_STREAM_2_4_1_ENABLED=false MULTITASK_GRADIENT_DIAGNOSTICS_ENABLED=false
export RUN_ROOT CHECKPOINT_DIR="${RUN_ROOT}/checkpoints" CACHE_DIR="${CACHE_DIR:-${BASE_DIR}/cache}"
export RAY_TMPDIR="${RAY_SHORT_ROOT}"

mkdir -p "${RUN_ROOT}" "${CHECKPOINT_DIR}" "${RAY_SHORT_ROOT}"
RAY_STARTED=false
cleanup() {
    status=$?
    if [[ "${RAY_STARTED}" == true ]]; then
        RAY_STARTED=false
        "${RAY_BIN}" stop --force >/dev/null 2>&1 || true
    fi
    if [[ "${status}" -eq 0 && "${KEEPALIVE_AFTER_TRAINING}" == true ]]; then
        echo "Training completed; keeping four GPUs occupied."
        exec "${PYTHON_BIN}" "${KEEPALIVE_SCRIPT}" --memory-mb "${KEEPALIVE_MEMORY_MB}"
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

echo "Starting four-GPU OPSD all-sample routing ablation: ${RUN_NAME}"
echo "R_Ci route classification: disabled; all eligible samples: on_policy_distill"
echo "Native CycleGRPO retained: ${PRESERVE_ORIGINAL_GRPO}"
echo "Run log: ${RUN_ROOT}"

if [[ "${MULTINODE_ENABLED}" == true ]]; then
    "${RAY_BIN}" stop --force >/dev/null 2>&1 || true
    RAY_STARTED=true
    "${RAY_BIN}" start --head --node-ip-address="${RAY_BIND_IP}" --port="${RAY_PORT}" \
        --num-gpus=4 --dashboard-host=127.0.0.1 --temp-dir="${RAY_SHORT_ROOT}" \
        --disable-usage-stats >"${RUN_ROOT}/ray_start.log" 2>&1
fi

bash "${REPO_DIR}/projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh"
