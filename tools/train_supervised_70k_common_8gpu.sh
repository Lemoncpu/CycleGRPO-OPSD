#!/usr/bin/env bash
# Shared launcher for the historical 70k mixed self-supervised/supervised run.
# Public wrappers set the run identity, SECA switches, and optional data fraction.
# This script starts its own 7-GPU Ray head and a local Llama-3.1-8B judge on
# physical GPU 7, then delegates the actual Hydra configuration to the main
# RefCOCO launcher.

BASE_DIR="${BASE_DIR:-/volume/ybo/xyc}"
REPO_DIR="${REPO_DIR:-${BASE_DIR}/CycleGRPO-OPSD}"
ENV_DIR="${ENV_DIR:-${BASE_DIR}/envs/cyclegrpo}"
MODEL_PATH="${MODEL_PATH:-${BASE_DIR}/Qwen3-VL-4B-SAMTok}"

TRAIN_DATA="${TRAIN_DATA:-${BASE_DIR}/datasets/cyclegrpo_20k_raw_seed20260820/cyclegrpo_20k_40_20_25_10_5_seed20260820.parquet}"
VAL_DATA="${VAL_DATA:-${TRAIN_DATA}}"
DIRECT_TRAIN_DATA="${DIRECT_TRAIN_DATA:-${BASE_DIR}/datasets/direct_refcoco30k_disjoint_cycle20k/refcoco_train_30k_disjoint_cycle20k_seed20260823.parquet}"
DIRECT_NO_TARGET_TRAIN_DATA="${DIRECT_NO_TARGET_TRAIN_DATA:-${BASE_DIR}/datasets/direct_refcoco30k_notarget10k_disjoint_cycle20k/grefcoco_train_0pos_10000notarget_disjoint_cycle20k_and_refcoco_seed20260823.parquet}"
CAPTION_QA_TRAIN_DATA="${CAPTION_QA_TRAIN_DATA:-${BASE_DIR}/datasets/dlc_qa/dlc_qa_10000.parquet}"
CAPTION_QA_JSONL="${CAPTION_QA_JSONL:-${BASE_DIR}/datasets/dlc_qa/dam_caption_qa_10000.jsonl}"

RUN_NAME="${RUN_NAME:-cyclegrpo70k_historical_baseline_8gpu}"
RUN_ROOT="${RUN_ROOT:-${REPO_DIR}/logs/${RUN_NAME}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_ROOT}/checkpoints}"
RUN_LOG="${RUN_LOG:-${RUN_ROOT}/run.log}"
TRAIN_LOG="${TRAIN_LOG:-${RUN_ROOT}/training.log}"

PYTHON_BIN="${PYTHON_BIN:-${ENV_DIR}/bin/python3}"
RAY_BIN="${RAY_BIN:-${ENV_DIR}/bin/ray}"
TRAIN_ENTRY="${TRAIN_ENTRY:-${REPO_DIR}/projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh}"
RAY_BIND_IP="${RAY_BIND_IP:-127.0.0.1}"
RAY_PORT="${RAY_PORT:-29687}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-29688}"
RAY_ADDRESS="${RAY_ADDRESS:-${RAY_BIND_IP}:${RAY_PORT}}"
RAY_NAMESPACE="${RAY_NAMESPACE:-${RUN_NAME}}"
RAY_SHORT_ROOT="${RAY_SHORT_ROOT:-/dev/shm/cgrpo70k-base8}"
JUDGE_PORT="${JUDGE_PORT:-18008}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-${BASE_DIR}/models/Meta-Llama-3.1-8B-Instruct-hf-v2}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-llama3.1-8b}"
JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
JUDGE_TIMEOUT_SECONDS="${JUDGE_TIMEOUT_SECONDS:-600}"
JUDGE_LOG_DIR="${JUDGE_LOG_DIR:-${RUN_ROOT}/llama_judge}"
KEEPALIVE_SCRIPT="${KEEPALIVE_SCRIPT:-${REPO_DIR}/tools/cuda_keepalive.py}"
DRY_RUN="${DRY_RUN:-false}"
TRAIN_NUM_GPUS="${TRAIN_NUM_GPUS:-7}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
ALL_CUDA_VISIBLE_DEVICES="${ALL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
JUDGE_CUDA_DEVICE="${JUDGE_CUDA_DEVICE:-7}"
# The formal topology reserves physical GPU 7 for the Llama judge, so the
# Ray/FSDP world size is 7.  Every parent batch must therefore be divisible
# by 7.  Keep the historical 2:4:1 stream ratio with 112/224/56; 179 steps
# consume one approximately complete pass of 20k/40k/10k rows.
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-112}"
ACTOR_GLOBAL_BATCH_SIZE="${ACTOR_GLOBAL_BATCH_SIZE:-112}"
DIRECT_BATCH_SIZE="${DIRECT_BATCH_SIZE:-224}"
CAPTION_QA_BATCH_SIZE="${CAPTION_QA_BATCH_SIZE:-56}"
MAX_STEPS="${MAX_STEPS:-179}"
SAVE_FREQ="${SAVE_FREQ:-5}"
SAVE_LIMIT="${SAVE_LIMIT:-2}"
HOLD_AFTER_EXIT="${HOLD_AFTER_EXIT:-true}"
DATA_FRACTION="${DATA_FRACTION:-100}"
case "${DATA_FRACTION}" in
    25) EXPECTED_MAIN_ROWS=5000; EXPECTED_DIRECT_POSITIVE_ROWS=7500; EXPECTED_DIRECT_NO_TARGET_ROWS=2500; EXPECTED_QA_ROWS=2500 ;;
    50) EXPECTED_MAIN_ROWS=10000; EXPECTED_DIRECT_POSITIVE_ROWS=15000; EXPECTED_DIRECT_NO_TARGET_ROWS=5000; EXPECTED_QA_ROWS=5000 ;;
    100) EXPECTED_MAIN_ROWS=20000; EXPECTED_DIRECT_POSITIVE_ROWS=30000; EXPECTED_DIRECT_NO_TARGET_ROWS=10000; EXPECTED_QA_ROWS=10000 ;;
    *) printf '[error] DATA_FRACTION must be 25, 50, or 100: %s\n' "${DATA_FRACTION}" >&2; exit 1 ;;
esac

# The wrappers explicitly set these two values.  Keeping defaults here makes
# the common file safe to inspect or invoke directly as the baseline.
SECA_ENABLED="${SECA_ENABLED:-false}"
SECA_SELF_SUPERVISED_ENABLED="${SECA_SELF_SUPERVISED_ENABLED:-false}"

CURRENT_STAGE="bootstrap"
RAY_SESSION_DIR=""
JUDGE_STARTED=false
RAY_STARTED=false
TRAIN_PID=""

log_line() {
    printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"
}

verify_parquet_rows() {
    local path="$1" expected="$2" label="$3"
    "${PYTHON_BIN}" - "${path}" "${expected}" "${label}" <<'PY'
import sys
import pyarrow.parquet as pq

path, expected, label = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rows = pq.ParquetFile(path).metadata.num_rows
if rows != expected:
    raise SystemExit(f"{label}: expected {expected} rows, got {rows}: {path}")
print(f"[progress:data-count] {label} rows={rows} path={path}")
PY
}

stop_owned_ray() {
    [[ "${RAY_STARTED}" == true ]] || return 0
    if [[ -z "${RAY_SESSION_DIR}" || ! -d "${RAY_SESSION_DIR}" ]]; then
        log_line "[cleanup] owned Ray session directory was not discovered; leaving other Ray clusters untouched"
        return 0
    fi
    local pids pid
    pids="$(ps -eo pid=,args= | awk -v needle="${RAY_SESSION_DIR}" 'index($0, needle) {print $1}')"
    for pid in ${pids}; do
        [[ "${pid}" == "$$" ]] || kill "${pid}" 2>/dev/null || true
    done
    sleep 2
    pids="$(ps -eo pid=,args= | awk -v needle="${RAY_SESSION_DIR}" 'index($0, needle) {print $1}')"
    for pid in ${pids}; do
        [[ "${pid}" == "$$" ]] || kill -KILL "${pid}" 2>/dev/null || true
    done
    RAY_STARTED=false
    log_line "[cleanup] stopped owned Ray session ${RAY_SESSION_DIR}"
}

start_hold_after_exit() {
    local status="$1"
    # Always leave the eight physical GPUs occupied after this standalone
    # experiment, including a failed run, so the server does not reclaim them.
    [[ "${HOLD_AFTER_EXIT}" == true ]] || return 0
    CUDA_VISIBLE_DEVICES="${ALL_CUDA_VISIBLE_DEVICES}" \
        GPU_LIST="${ALL_CUDA_VISIBLE_DEVICES}" MEMORY_MIB=20000 MATMUL_DIM=4096 \
        PYTHON_BIN="${PYTHON_BIN}" bash "${REPO_DIR}/tools/gpu_power_hold.sh" start || true
    log_line "[cleanup] eight-GPU hold requested; preserving exit status ${status}"
}

cleanup() {
    local status=$?
    CURRENT_STAGE="cleanup"
    if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
        log_line "[cleanup] terminating interrupted trainer pid=${TRAIN_PID}"
        kill "${TRAIN_PID}" 2>/dev/null || true
        wait "${TRAIN_PID}" 2>/dev/null || true
    fi
    if [[ "${JUDGE_STARTED}" == true ]]; then
        "${PYTHON_BIN}" "${REPO_DIR}/tools/multinode/local_llama_judge.py" stop \
            --ray-address "${RAY_ADDRESS}" --namespace "${RAY_NAMESPACE}" --expected-nodes 1 \
            --env-dir "${ENV_DIR}" --model-path "${JUDGE_MODEL_PATH}" \
            --served-model-name "${JUDGE_MODEL_NAME}" --port "${JUDGE_PORT}" --gpu-device "${JUDGE_CUDA_DEVICE}" \
            --gpu-memory-utilization 0.85 \
            --chat-template "${REPO_DIR}/tools/multinode/llama3_chat_template.jinja" \
            --log-dir "${JUDGE_LOG_DIR}" --name-prefix "${RUN_NAME}-llama" \
            --timeout-seconds "${JUDGE_TIMEOUT_SECONDS}" || true
        JUDGE_STARTED=false
    fi
    stop_owned_ray
    start_hold_after_exit "${status}"
    log_line "[stage:exit] status=${status} run_log=${RUN_LOG}"
    return "${status}"
}
trap cleanup EXIT INT TERM

mkdir -p "${RUN_ROOT}" "${CHECKPOINT_DIR}"
exec > >(tee -a "${RUN_LOG}") 2>&1
log_line "[stage:bootstrap] historical 70k supervised run started: ${RUN_NAME}"
log_line "[config] seca=${SECA_ENABLED} self_supervised_seca=${SECA_SELF_SUPERVISED_ENABLED}"
if [[ "${DATA_FRACTION}" != 100 ]]; then
    log_line "[stage:data] preparing deterministic ${DATA_FRACTION}% subsets under ${RUN_ROOT}/data"
    SUBSET_ROOT="${RUN_ROOT}/data"
    "${PYTHON_BIN}" - "${DATA_FRACTION}" "${SUBSET_ROOT}" "${TRAIN_DATA}" \
        "${DIRECT_TRAIN_DATA}" "${DIRECT_NO_TARGET_TRAIN_DATA}" \
        "${CAPTION_QA_TRAIN_DATA}" "${CAPTION_QA_JSONL}" <<'PY'
import json
import os
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

fraction = int(sys.argv[1])
out = Path(sys.argv[2])
sources = list(map(Path, sys.argv[3:]))
out.mkdir(parents=True, exist_ok=True)

def select_indices(size, label):
    order = list(range(size))
    random.Random(f"cyclegrpo70k-subset-20260925:{label}").shuffle(order)
    return sorted(order[:size * fraction // 100])

def write_parquet(table, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temp)
    os.replace(temp, path)

cycle = pq.read_table(sources[0])
if len(cycle) != 20000:
    raise ValueError(f"cycle source expected 20000 rows, got {len(cycle)}: {sources[0]}")
by_source = {}
for index, source in enumerate(cycle.column("source").to_pylist()):
    by_source.setdefault(source, []).append(index)
cycle_indices = sorted(index for source, indices in by_source.items()
                       for index in (indices[i] for i in select_indices(len(indices), f"cycle:{source}")))
write_parquet(cycle.take(pa.array(cycle_indices)), out / "cycle.parquet")

for label, path, expected in zip(("direct_positive", "direct_no_target", "dlc_qa"), sources[1:4],
                                 (30000, 10000, 10000)):
    table = pq.read_table(path)
    if len(table) != expected:
        raise ValueError(f"{label} source expected {expected} rows, got {len(table)}: {path}")
    selected = table.take(pa.array(select_indices(expected, label)))
    write_parquet(selected, out / f"{label}.parquet")
    if label == "dlc_qa":
        selected_ids = selected.column("dam_source_id").to_pylist()

with sources[4].open(encoding="utf-8") as stream:
    qa = [json.loads(line) for line in stream if line.strip()]
qa_by_id = {row["dam_source_id"]: row for row in qa}
if len(qa) != 10000 or len(qa_by_id) != len(qa) or len(selected_ids) != len(set(selected_ids)):
    raise ValueError("DLC-QA JSONL/parquet must contain 10000 unique dam_source_id values")
if set(qa_by_id) != set(pq.read_table(sources[3], columns=["dam_source_id"]).column(0).to_pylist()):
    raise ValueError("DLC-QA JSONL and parquet dam_source_id sets differ")
qa_path = out / "dlc_qa.jsonl"
temp = qa_path.with_suffix(".jsonl.tmp")
with temp.open("w", encoding="utf-8") as stream:
    for source_id in selected_ids:
        stream.write(json.dumps(qa_by_id[source_id], ensure_ascii=False) + "\n")
os.replace(temp, qa_path)
print(f"[progress:data-subset] fraction={fraction}% cycle={len(cycle_indices)} "
      f"direct_positive={30000 * fraction // 100} direct_no_target={10000 * fraction // 100} "
      f"dlc_qa={len(selected_ids)}")
PY
    subset_status=$?
    if [[ "${subset_status}" != 0 ]]; then
        log_line "[error] data subset preparation failed with status ${subset_status}" >&2
        exit "${subset_status}"
    fi
    TRAIN_DATA="${SUBSET_ROOT}/cycle.parquet"
    VAL_DATA="${TRAIN_DATA}"
    DIRECT_TRAIN_DATA="${SUBSET_ROOT}/direct_positive.parquet"
    DIRECT_NO_TARGET_TRAIN_DATA="${SUBSET_ROOT}/direct_no_target.parquet"
    CAPTION_QA_TRAIN_DATA="${SUBSET_ROOT}/dlc_qa.parquet"
    CAPTION_QA_JSONL="${SUBSET_ROOT}/dlc_qa.jsonl"
fi
log_line "[config] main=${TRAIN_DATA} direct=${DIRECT_TRAIN_DATA}+${DIRECT_NO_TARGET_TRAIN_DATA} qa=${CAPTION_QA_TRAIN_DATA}"

for required in "${PYTHON_BIN}" "${RAY_BIN}" "${TRAIN_ENTRY}" \
    "${MODEL_PATH}/config.json" "${MODEL_PATH}/model.safetensors.index.json" \
    "${MODEL_PATH}/mask_tokenizer_256x2.pth" "${MODEL_PATH}/sam2.1_hiera_large.pt" \
    "${JUDGE_MODEL_PATH}/config.json" "${CAPTION_QA_JSONL}"; do
    if [[ ! -e "${required}" ]]; then
        log_line "[error] missing required path: ${required}" >&2
        exit 1
    fi
done
if [[ ! -x "${PYTHON_BIN}" || ! -x "${RAY_BIN}" ]]; then
    log_line "[error] Python/Ray executable is not executable: ${PYTHON_BIN} ${RAY_BIN}" >&2
    exit 1
fi
if [[ ! -d "${RAY_SHORT_ROOT}" ]]; then
    mkdir -p "${RAY_SHORT_ROOT}"
fi
if [[ "${RAY_SHORT_ROOT}" != /* || ${#RAY_SHORT_ROOT} -gt 32 || -L "${RAY_SHORT_ROOT}" ]]; then
    log_line "[error] RAY_SHORT_ROOT must be a real absolute path of at most 32 characters: ${RAY_SHORT_ROOT}" >&2
    exit 1
fi

verify_parquet_rows "${TRAIN_DATA}" "${EXPECTED_MAIN_ROWS}" main_cycle
verify_parquet_rows "${DIRECT_TRAIN_DATA}" "${EXPECTED_DIRECT_POSITIVE_ROWS}" direct_refcoco_positive
verify_parquet_rows "${DIRECT_NO_TARGET_TRAIN_DATA}" "${EXPECTED_DIRECT_NO_TARGET_ROWS}" direct_no_target
verify_parquet_rows "${CAPTION_QA_TRAIN_DATA}" "${EXPECTED_QA_ROWS}" dlc_qa
QA_LINES="$(awk 'NF {n++} END {print n+0}' "${CAPTION_QA_JSONL}")"
if [[ "${QA_LINES}" != "${EXPECTED_QA_ROWS}" ]]; then
    log_line "[error] DLC-QA JSONL expected ${EXPECTED_QA_ROWS} non-empty lines, got ${QA_LINES}: ${CAPTION_QA_JSONL}" >&2
    exit 1
fi
log_line "[progress:data-count] dlc_qa_jsonl rows=${QA_LINES} path=${CAPTION_QA_JSONL}"

cd "${REPO_DIR}" || exit 1
export BASE_DIR REPO_DIR ENV_DIR MODEL_PATH TRAIN_DATA VAL_DATA DIRECT_TRAIN_DATA DIRECT_NO_TARGET_TRAIN_DATA
export CAPTION_QA_TRAIN_DATA CAPTION_QA_JSONL PYTHON_BIN RAY_BIN CONDA_PREFIX="${ENV_DIR}"
export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}"
export RAY_ADDRESS RAY_NAMESPACE RAY_SHORT_ROOT
export RAY_TMPDIR="${RAY_SHORT_ROOT}"
export RAY_CLUSTER_EXPECTED_NODES=1 RAY_CLUSTER_EXPECTED_GPUS="${TRAIN_NUM_GPUS}"
export MULTINODE_ENABLED=true LOCAL_JUDGE_ENABLED=true NNODES=1 NUM_GPUS="${TRAIN_NUM_GPUS}"
export LOCAL_JUDGE_TRAIN_GPUS="${TRAIN_NUM_GPUS}"
export LOCAL_JUDGE_CUDA_DEVICE="${JUDGE_CUDA_DEVICE}" LOCAL_JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH}"
export LOCAL_JUDGE_SERVED_MODEL_NAME="${JUDGE_MODEL_NAME}" LOCAL_JUDGE_API_KEY="${JUDGE_API_KEY}"
export LOCAL_JUDGE_PORT="${JUDGE_PORT}" LOCAL_JUDGE_GPU_MEMORY_UTILIZATION=0.85
export CAPTION_QA_JUDGE_BASE_URL="http://127.0.0.1:${JUDGE_PORT}/v1"
export CAPTION_QA_JUDGE_MODEL="${JUDGE_MODEL_NAME}" CAPTION_QA_JUDGE_API_KEY="${JUDGE_API_KEY}"
export TRAINER_LOGGERS='["file"]'

# Historical 70k setup: 20k cycle + 30k positive direct grounding + 10k
# gRefCOCO no-target + 10k DLC-QA.  The base 20k experiment's
# 112/112/224/56, 179-step settings are used so every parent batch is
# divisible by the seven training ranks while preserving the historical
# 2:4:1 stream ratio and approximately one pass over each loader; only the
# additional supervised streams and their corresponding loaders are enabled.
export ROLLOUT_BATCH_SIZE ACTOR_GLOBAL_BATCH_SIZE DIRECT_BATCH_SIZE CAPTION_QA_BATCH_SIZE
export CAPTION_ROLLOUTS=6 LOCALIZATION_ROLLOUTS=6 TOTAL_EPOCHS=1 MAX_STEPS SAVE_FREQ SAVE_LIMIT RESUME=false
export OPSD_ENABLED=true PIXEL_IOU_ENABLED=true EGCA_ENABLED=false EGCA_OPD_ENABLED=false
export SECA_ENABLED SECA_SELF_SUPERVISED_ENABLED
export SECA_MIN_WEIGHT=0.5 SECA_MAX_WEIGHT=1.0 SECA_FALSE_POSITIVE_PENALTY=2.0
export SECA_COARSE_TOKEN_WEIGHT=1.5 SECA_FINE_TOKEN_WEIGHT=1.0
export ROUTING_ENABLED=true PRESERVE_ORIGINAL_GRPO=true EMA_TEACHER_ENABLED=true TEACHER_EMA_DECAY=1.0
export TEACHER_ANALYSIS_ENABLED=true TEACHER_CONFIDENCE_ENABLED=true CAPTION_SAFETY_ENABLED=true CAPTION_SAFETY_FORCE_REGENERATE=true
export CAPTION_BLOCK_SPECIAL_TOKEN_VOCAB=true JSD_BLOCK_CAPTION_SPECIAL_TOKEN_VOCAB=true
export CAPTION_ANCHOR_KL_COEF=0.05 CAPTION_ANCHOR_KL_ALL_SAFE_ROUTES=true SEGMENTATION_ANCHOR_KL_COEF=0.05 ASYMMETRIC_GRADIENT_PROJECTION=false
export NO_TARGET_REWARD_MODE=pixel_empty NO_TARGET_SEGMENTATION_LOSS_WEIGHT=1.0
export CAPTION_MAX_RESPONSE_LENGTH=256 SEGMENTATION_MAX_RESPONSE_TOKENS=256 MASK_DECODE_MODE=union
export LOCALIZATION_PROMPT_MODE=mixed CYCLE_PROMPT_MODE=current
export DIRECT_GROUNDING_ENABLED=true DIRECT_GROUNDING_ROLLOUTS=6 DIRECT_GROUNDING_LOSS_WEIGHT=0.15
export DIRECT_GROUNDING_WARMUP_START_STEP=10 DIRECT_GROUNDING_WARMUP_END_STEP=30
export DIRECT_GROUNDING_INCLUDE_POSITIVE_SOURCES=true DIRECT_GROUNDING_INCLUDE_NO_TARGET=true DIRECT_GROUNDING_INCLUDE_LABEL_SOURCES=false
export DIRECT_GROUNDING_CONSUME_NO_TARGET_CAPTION=false
export DIRECT_MASK_CE_ENABLED=true DIRECT_MASK_CE_LOSS_WEIGHT=0.005 DIRECT_MASK_CE_INCLUDE_NO_TARGET=true
export DIRECT_MASK_CE_WARMUP_START_STEP=10 DIRECT_MASK_CE_WARMUP_END_STEP=30 DIRECT_MASK_CE_RECORD_BASE_GRADIENT_COSINE=false
export SUPERVISED_CAPTION_QA_ENABLED=true CAPTION_QA_MAX_CONCURRENCY=8 CAPTION_QA_TIMEOUT_SECONDS=90
export CAPTION_QA_REWARD_WEIGHT=1.0 CAPTION_QA_LOSS_WEIGHT=1.0 THREE_STREAM_2_4_1_ENABLED=false
export MULTITASK_GRADIENT_DIAGNOSTICS_ENABLED=false
export RUN_NAME RUN_ROOT CHECKPOINT_DIR RUN_LOG

if [[ "${DRY_RUN}" == true ]]; then
    log_line "[stage:dry-run] preflight and configuration checks passed; Ray/judge/trainer not started"
    trap - EXIT INT TERM
    exit 0
fi

CURRENT_STAGE="ray"
log_line "[stage:ray] starting isolated local Ray head at ${RAY_ADDRESS} with ${TRAIN_NUM_GPUS} training GPUs"
"${RAY_BIN}" start --head --node-ip-address="${RAY_BIND_IP}" --port="${RAY_PORT}" \
    --num-gpus="${TRAIN_NUM_GPUS}" --dashboard-host=127.0.0.1 --dashboard-port="${RAY_DASHBOARD_PORT}" \
    --temp-dir="${RAY_SHORT_ROOT}" --disable-usage-stats >"${RUN_ROOT}/ray_start.log" 2>&1
RAY_STARTED=true
for _ in $(seq 1 30); do
    RAY_SESSION_DIR="$(find "${RAY_SHORT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'session_*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR == 1 {print $2}')"
    [[ -n "${RAY_SESSION_DIR}" ]] && break
    sleep 1
done
if [[ -z "${RAY_SESSION_DIR}" ]]; then
    log_line "[error] could not identify the Ray session directory under ${RAY_SHORT_ROOT}" >&2
    exit 1
fi
log_line "[stage:ray] owned session=${RAY_SESSION_DIR}"

CURRENT_STAGE="judge"
log_line "[stage:judge] starting local Llama-3.1-8B judge on physical GPU ${JUDGE_CUDA_DEVICE}"
JUDGE_STARTED=true
"${PYTHON_BIN}" "${REPO_DIR}/tools/multinode/local_llama_judge.py" start \
    --ray-address "${RAY_ADDRESS}" --namespace "${RAY_NAMESPACE}" --expected-nodes 1 \
    --env-dir "${ENV_DIR}" --model-path "${JUDGE_MODEL_PATH}" \
    --served-model-name "${JUDGE_MODEL_NAME}" --port "${JUDGE_PORT}" --gpu-device "${JUDGE_CUDA_DEVICE}" \
    --gpu-memory-utilization 0.85 --chat-template "${REPO_DIR}/tools/multinode/llama3_chat_template.jinja" \
    --log-dir "${JUDGE_LOG_DIR}" --name-prefix "${RUN_NAME}-llama" --timeout-seconds "${JUDGE_TIMEOUT_SECONDS}"
log_line "[stage:judge] local judge is healthy at ${CAPTION_QA_JUDGE_BASE_URL}"

CURRENT_STAGE="train"
log_line "[stage:train] launching 70k historical supervised training; train log=${TRAIN_LOG}"
RUN_LOG=/dev/stdout bash "${TRAIN_ENTRY}" > >(tee -a "${TRAIN_LOG}") 2>&1 &
TRAIN_PID=$!
if wait "${TRAIN_PID}"; then
    TRAIN_STATUS=0
else
    TRAIN_STATUS=$?
fi
TRAIN_PID=""
if [[ "${TRAIN_STATUS}" != 0 ]]; then
    log_line "[error] training exited with status ${TRAIN_STATUS}" >&2
    exit "${TRAIN_STATUS}"
fi
CURRENT_STAGE="done"
log_line "[stage:done] 70k ${RUN_NAME} completed successfully"
