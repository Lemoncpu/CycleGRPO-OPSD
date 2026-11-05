#!/usr/bin/env bash
# Qwen3-VL-8B-SAMTok SECA training on the pre-built 100k scaling dataset.
# The optimizer, SECA, routing, judge, and 7+1 topology come from the 70k 8B
# wrapper; only the data paths, expected stream sizes, and one-pass length are
# changed to match run_100k_scaling_opsd_8gpu.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-/volume/ybo/xyc}"
DATA_ROOT="${DATA_ROOT:-${BASE_DIR}/datasets/cyclegrpo100k_scaling_20260911}"

export MODEL_PATH="${MODEL_PATH:-${BASE_DIR}/Qwen3-VL-8B-SAMTok}"
export TRAIN_DATA="${TRAIN_DATA:-${DATA_ROOT}/cyclegrpo_selfsupervised_40k.parquet}"
export VAL_DATA="${VAL_DATA:-${TRAIN_DATA}}"
export DIRECT_TRAIN_DATA="${DIRECT_TRAIN_DATA:-${DATA_ROOT}/direct_supervised_positive_32k.parquet}"
export DIRECT_NO_TARGET_TRAIN_DATA="${DIRECT_NO_TARGET_TRAIN_DATA:-${DATA_ROOT}/direct_supervised_no_target_8k.parquet}"
export CAPTION_QA_TRAIN_DATA="${CAPTION_QA_TRAIN_DATA:-${DATA_ROOT}/dlc_qa_20k.parquet}"
export CAPTION_QA_JSONL="${CAPTION_QA_JSONL:-${DATA_ROOT}/dam_caption_qa_20k.jsonl}"

export DATA_FRACTION=100
export EXPECTED_MAIN_ROWS_OVERRIDE=40000
export EXPECTED_DIRECT_POSITIVE_ROWS_OVERRIDE=32000
export EXPECTED_DIRECT_NO_TARGET_ROWS_OVERRIDE=8000
export EXPECTED_QA_ROWS_OVERRIDE=20000
export MAX_STEPS="${MAX_STEPS:-357}"
export SAVE_FREQ="${SAVE_FREQ:-25}"
export SAVE_LIMIT="${SAVE_LIMIT:-1}"

export RUN_NAME="${RUN_NAME:-cyclegrpo100k_seca_8b_8gpu}"
export RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/../logs/${RUN_NAME}}"
export RAY_PORT="${RAY_PORT:-29701}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-29702}"
export JUDGE_PORT="${JUDGE_PORT:-18011}"
export RAY_NAMESPACE="${RAY_NAMESPACE:-${RUN_NAME}}"
export RAY_SHORT_ROOT="${RAY_SHORT_ROOT:-/dev/shm/cgrpo100k-8b}"

export SECA_ENABLED=true
export SECA_SELF_SUPERVISED_ENABLED=true
# The 100k direct stream intentionally includes Stuff/PACO label sources,
# matching run_100k_scaling_opsd_8gpu.sh rather than the historical 70k filter.
export DIRECT_GROUNDING_INCLUDE_LABEL_SOURCES=true

exec "${SCRIPT_DIR}/train_supervised_70k_common_8gpu.sh" "$@"
