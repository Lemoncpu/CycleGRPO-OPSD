#!/usr/bin/env bash
# Eight-GPU 20k self-supervised EGCA experiment.
# EGCA uses rollout evidence only as non-negative weights for an independent
# GT SAMTok teacher-forcing update; it does not inject signed credit into GRPO
# advantages.  The local Ray head and the trainer are started by this file.

BASE_DIR="${BASE_DIR:-/volume/ybo/xyc}"
REPO_DIR="${REPO_DIR:-${BASE_DIR}/CycleGRPO-OPSD}"
ENV_DIR="${ENV_DIR:-${BASE_DIR}/envs/cyclegrpo}"
MODEL_PATH="${MODEL_PATH:-${BASE_DIR}/Qwen3-VL-4B-SAMTok}"
TRAIN_DATA="${TRAIN_DATA:-${BASE_DIR}/datasets/cyclegrpo_20k_raw_seed20260820/cyclegrpo_20k_40_20_25_10_5_seed20260820.parquet}"
RUN_NAME="${RUN_NAME:-cyclegrpo20k_egca_weighted_ce_pixel_empty_8gpu}"
RUN_ROOT="${RUN_ROOT:-${REPO_DIR}/logs/${RUN_NAME}}"
RAY_BIN="${RAY_BIN:-${ENV_DIR}/bin/ray}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_DIR}/bin/python3}"
RAY_PORT="${RAY_PORT:-29681}"
RAY_BIND_IP="${RAY_BIND_IP:-$(hostname -I | awk '{print $1}')}"
RAY_BIND_IP="${RAY_BIND_IP//[[:space:]]/}"
RAY_ADDRESS="${RAY_BIND_IP}:${RAY_PORT}"
RAY_NAMESPACE="${RAY_NAMESPACE:-${RUN_NAME}}"
RAY_SHORT_ROOT="${RAY_SHORT_ROOT:-/dev/shm/cgrpo-egca-weighted8}"
KEEPALIVE_SCRIPT="${KEEPALIVE_SCRIPT:-${REPO_DIR}/tools/cuda_keepalive.py}"
KEEPALIVE_MEMORY_MB="${KEEPALIVE_MEMORY_MB:-40000}"

for required in "${REPO_DIR}" "${PYTHON_BIN}" "${RAY_BIN}" \
    "${MODEL_PATH}/config.json" "${MODEL_PATH}/model.safetensors.index.json" \
    "${MODEL_PATH}/mask_tokenizer_256x2.pth" "${MODEL_PATH}/sam2.1_hiera_large.pt" \
    "${TRAIN_DATA}" "${KEEPALIVE_SCRIPT}"; do
    if [ ! -e "${required}" ]; then
        echo "Required path not found: ${required}" >&2
        exit 1
    fi
done
if [ ! -x "${PYTHON_BIN}" ] || [ ! -x "${RAY_BIN}" ]; then
    echo "Python/Ray executable is not executable: ${PYTHON_BIN} ${RAY_BIN}" >&2
    exit 1
fi
if [ -z "${RAY_BIND_IP}" ]; then
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

cd "${REPO_DIR}" || exit 1
export BASE_DIR REPO_DIR ENV_DIR MODEL_PATH TRAIN_DATA RUN_NAME RUN_ROOT PYTHON_BIN RAY_BIN
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export NNODES=1
export MULTINODE_ENABLED=true
export RAY_ADDRESS RAY_NAMESPACE
export RAY_CLUSTER_EXPECTED_NODES=1
export RAY_CLUSTER_EXPECTED_GPUS="${NUM_GPUS}"
export LOCAL_JUDGE_ENABLED=false
export RAY_LOCAL_NUM_CPUS="${RAY_LOCAL_NUM_CPUS:-64}"
export RAY_SHORT_ROOT

export ROLLOUT_BATCH_SIZE=128
export ACTOR_GLOBAL_BATCH_SIZE=128
export TOTAL_EPOCHS=1
export MAX_STEPS="${MAX_STEPS:-}"
export SAVE_FREQ="${SAVE_FREQ:-5}"
export SAVE_LIMIT="${SAVE_LIMIT:-2}"
export RESUME=false
export TRAINER_LOGGERS='["file"]'

export OPSD_ENABLED=true
export PIXEL_IOU_ENABLED=true
export EGCA_ENABLED=true
export EGCA_OPD_ENABLED=true
export EGCA_UPDATE_MODE=weighted_ce
export EGCA_REFERENCE_MODE=target
export EGCA_ACTOR_COEF=0.0
export EGCA_CREDIT_MODE=contrastive
export EGCA_CE_LOSS_WEIGHT="${EGCA_CE_LOSS_WEIGHT:-0.05}"
export EGCA_CE_MIN_WEIGHT="${EGCA_CE_MIN_WEIGHT:-0.25}"
export EGCA_CE_MAX_WEIGHT="${EGCA_CE_MAX_WEIGHT:-1.0}"
export EGCA_CE_TOKEN_CREDIT_SCALE="${EGCA_CE_TOKEN_CREDIT_SCALE:-0.5}"
export EGCA_CE_TOKEN_WEIGHT_MAX="${EGCA_CE_TOKEN_WEIGHT_MAX:-2.0}"
export EGCA_EVIDENCE_MIN_WEIGHT="${EGCA_EVIDENCE_MIN_WEIGHT:-0.5}"
export EGCA_EVIDENCE_MAX_WEIGHT="${EGCA_EVIDENCE_MAX_WEIGHT:-1.0}"
export EGCA_FALSE_POSITIVE_PENALTY="${EGCA_FALSE_POSITIVE_PENALTY:-2.0}"
export EGCA_REFERENCE_COARSE_CODE=0
export EGCA_REFERENCE_FINE_CODE=0
export EGCA_CREDIT_CLIP=1.0
export EGCA_WARMUP_STEPS=0
export EGCA_RAMP_STEPS=0
export EGCA_MAX_GROUPS=8

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

export NO_TARGET_REWARD_MODE=pixel_empty
export POSITIVE_EMPTY_MASK_PENALTY=1.0
export NO_TARGET_NONEMPTY_MASK_PENALTY=0.0
export NO_TARGET_EMPTY_AREA_TAU=0.0
export CAPTION_MAX_RESPONSE_LENGTH=256
export SEGMENTATION_MAX_RESPONSE_TOKENS=256
export MASK_DECODE_MODE=union
export LOCALIZATION_PROMPT_MODE=mixed
export CYCLE_PROMPT_MODE=current

export DIRECT_GROUNDING_ENABLED=false
export DIRECT_MASK_CE_ENABLED=false
export SUPERVISED_CAPTION_QA_ENABLED=false
export RUN_ROOT CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
export KEEPALIVE_AFTER_TRAINING="${KEEPALIVE_AFTER_TRAINING:-true}"
export CACHE_DIR="${CACHE_DIR:-${BASE_DIR}/cache}"
export RAY_TMPDIR="${RAY_SHORT_ROOT}"

mkdir -p "${RUN_ROOT}" "${CHECKPOINT_DIR}" "${RAY_SHORT_ROOT}"
RAY_STARTED=false
cleanup() {
    status=$?
    if [ "${RAY_STARTED}" = true ]; then
        RAY_STARTED=false
        "${RAY_BIN}" stop --force >/dev/null 2>&1 || true
    fi
    if [ "${status}" -eq 0 ] && [ "${KEEPALIVE_AFTER_TRAINING}" = true ]; then
        echo "Training completed; keeping eight GPUs occupied."
        exec "${PYTHON_BIN}" "${KEEPALIVE_SCRIPT}" --memory-mb "${KEEPALIVE_MEMORY_MB}"
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

"${RAY_BIN}" stop --force >/dev/null 2>&1 || true
RAY_STARTED=true
"${RAY_BIN}" start --head \
    --node-ip-address="${RAY_BIND_IP}" \
    --port="${RAY_PORT}" \
    --num-gpus="${NUM_GPUS}" \
    --dashboard-host=127.0.0.1 \
    --temp-dir="${RAY_SHORT_ROOT}" \
    --disable-usage-stats \
    >"${RUN_ROOT}/ray_start.log" 2>&1
ray_status=$?
if [ "${ray_status}" -ne 0 ]; then
    echo "Ray head failed to start; see ${RUN_ROOT}/ray_start.log" >&2
    exit "${ray_status}"
fi

echo "Starting eight-GPU weighted-EGCA training: ${RUN_NAME}"
bash "${REPO_DIR}/projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh"
