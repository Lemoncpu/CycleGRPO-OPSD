#!/usr/bin/env bash

BASE_DIR=${BASE_DIR:-/volume/ybo/xyc}
REPO_DIR=${REPO_DIR:-${BASE_DIR}/CycleGRPO-OPSD}
ENV_DIR=${ENV_DIR:-${BASE_DIR}/envs/cyclegrpo}
PYTHON_BIN=${PYTHON_BIN:-${ENV_DIR}/bin/python3}
MODEL_PATH=${MODEL_PATH:-${REPO_DIR}/logs/cyclegrpo70k_historical_seca_8gpu/evaluation/step_179_cpu_export/hf_global_step_179}
VQ_SAM2_PATH=${VQ_SAM2_PATH:-${BASE_DIR}/Qwen3-VL-4B-SAMTok/mask_tokenizer_256x2.pth}
SAM2_PATH=${SAM2_PATH:-${BASE_DIR}/Qwen3-VL-4B-SAMTok/sam2.1_hiera_large.pt}
INPUT_DIR=${INPUT_DIR:-${REPO_DIR}/logs/pixel_opsd_recording_demo/inputs}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_DIR}/logs/pixel_opsd_recording_demo/outputs}
DEVICE=${DEVICE:-cuda:0}

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python not found: $PYTHON_BIN" >&2
    exit 2
fi
if [ ! -d "$INPUT_DIR" ]; then
    echo "Input directory not found: $INPUT_DIR" >&2
    exit 2
fi
if [ ! -d "$MODEL_PATH" ]; then
    echo "Model directory not found: $MODEL_PATH" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR" || exit 2
LOG_FILE=${LOG_FILE:-${OUTPUT_DIR}/../evaluation.log}
cd "$REPO_DIR" || exit 2
PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -u \
    "$REPO_DIR/evaluation/refcoco/record_three_tasks.py" \
    --model-path "$MODEL_PATH" \
    --vq-sam2-path "$VQ_SAM2_PATH" \
    --sam2-path "$SAM2_PATH" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" 2>&1 | tee "$LOG_FILE"
run_status=${PIPESTATUS[0]}
if [ "$run_status" -ne 0 ]; then
    echo "Evaluation failed (exit $run_status). See $LOG_FILE; existing outputs may be from an earlier run." >&2
fi
exit "$run_status"
