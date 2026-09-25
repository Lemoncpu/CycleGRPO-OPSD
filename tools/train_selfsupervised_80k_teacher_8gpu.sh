#!/usr/bin/env bash
# Pure self-supervised OPSD/teacher training on an 80k merged cycle dataset:
# 40k cycle self-supervision plus 40k segmentation-supervision records.  The
# merge is created once, atomically, under the same scaling-data directory so
# this script remains a single-command training entry point.

set -euo pipefail

BASE_DIR="${BASE_DIR:-/volume/ybo/xyc}"
REPO_DIR="${REPO_DIR:-${BASE_DIR}/CycleGRPO-OPSD}"
ENV_DIR="${ENV_DIR:-${BASE_DIR}/envs/cyclegrpo}"
DATA_ROOT="${DATA_ROOT:-${BASE_DIR}/datasets/cyclegrpo100k_scaling_20260911}"
MODEL_PATH="${MODEL_PATH:-${BASE_DIR}/Qwen3-VL-4B-SAMTok}"
SELF_DATA="${SELF_DATA:-${DATA_ROOT}/cyclegrpo_selfsupervised_40k.parquet}"
SEG_DATA="${SEG_DATA:-${DATA_ROOT}/direct_supervised_40k.parquet}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_ROOT}/cyclegrpo_selfsupervised_plus_direct_supervised_80k.parquet}"
RUN_NAME="${RUN_NAME:-cyclegrpo_selfsupervised_80k_teacher_bs128_8gpu}"
RUN_ROOT="${RUN_ROOT:-${REPO_DIR}/logs/${RUN_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_DIR}/bin/python3}"
RAY_BIN="${RAY_BIN:-${ENV_DIR}/bin/ray}"
RAY_PORT="${RAY_PORT:-29680}"
RAY_ADDRESS="127.0.0.1:${RAY_PORT}"
RAY_NAMESPACE="${RAY_NAMESPACE:-${RUN_NAME}}"

for required_dir in "${REPO_DIR}" "${DATA_ROOT}"; do
    if [[ ! -d "${required_dir}" ]]; then
        echo "Required directory not found: ${required_dir}" >&2
        exit 1
    fi
done
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -x "${RAY_BIN}" ]]; then
    echo "Ray executable not found: ${RAY_BIN}" >&2
    exit 1
fi
for required_file in "${SELF_DATA}" "${SEG_DATA}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required source parquet not found: ${required_file}" >&2
        exit 1
    fi
done

if [[ ! -f "${TRAIN_DATA}" ]] || [[ "${REBUILD_MERGED_DATA:-false}" == "true" ]]; then
    temporary_data="${TRAIN_DATA}.tmp.$$"
    trap 'rm -f "${temporary_data:-}"' EXIT
    "${PYTHON_BIN}" - "${SELF_DATA}" "${SEG_DATA}" "${temporary_data}" <<'PY'
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

self_path, seg_path, output_path = map(Path, sys.argv[1:])
self_table = pq.read_table(self_path)
seg_table = pq.read_table(seg_path)
merged = pa.concat_tables([self_table, seg_table], promote_options="default")
if merged.num_rows != 80000:
    raise RuntimeError(f"Expected 80000 merged rows, got {merged.num_rows}")
pq.write_table(merged, output_path, compression="zstd")
print(f"Created merged parquet: rows={merged.num_rows} path={output_path}")
PY
    mv -f "${temporary_data}" "${TRAIN_DATA}"
    trap - EXIT
fi

"${PYTHON_BIN}" - "${TRAIN_DATA}" <<'PY'
import sys
import pyarrow.parquet as pq

path = sys.argv[1]
rows = pq.ParquetFile(path).metadata.num_rows
if rows != 80000:
    raise RuntimeError(f"Merged training parquet must contain 80000 rows, got {rows}")
print(f"Verified merged parquet: rows={rows} path={path}")
PY

cd "${REPO_DIR}"

export BASE_DIR REPO_DIR ENV_DIR DATA_ROOT MODEL_PATH TRAIN_DATA RUN_NAME RUN_ROOT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export NNODES="${NNODES:-1}"
export MULTINODE_ENABLED=true
export RAY_ADDRESS RAY_NAMESPACE RAY_CLUSTER_EXPECTED_NODES=1 RAY_CLUSTER_EXPECTED_GPUS=8
export LOCAL_JUDGE_ENABLED=false
export RAY_SHORT_ROOT="${RAY_SHORT_ROOT:-/dev/shm/cgrpo-self80k8}"
export RAY_LOCAL_NUM_CPUS="${RAY_LOCAL_NUM_CPUS:-64}"

export ROLLOUT_BATCH_SIZE=128
export ACTOR_GLOBAL_BATCH_SIZE=128
export TOTAL_EPOCHS=1
export MAX_STEPS=""
export SAVE_FREQ="${SAVE_FREQ:-5}"
export SAVE_LIMIT="${SAVE_LIMIT:-1}"
export RESUME=false
export TRAINER_LOGGERS='["file"]'

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

export NO_TARGET_REWARD_MODE=pixel_empty
export POSITIVE_EMPTY_MASK_PENALTY=1.0
export NO_TARGET_NONEMPTY_MASK_PENALTY=0.0
export NO_TARGET_EMPTY_AREA_TAU=0.0
export CAPTION_MAX_RESPONSE_LENGTH=256
export SEGMENTATION_MAX_RESPONSE_TOKENS=256
export MASK_DECODE_MODE=union

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
