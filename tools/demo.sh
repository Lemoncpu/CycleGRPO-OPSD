#!/usr/bin/env bash
set -euo pipefail

BASE_DIR=${BASE_DIR:-/volume/ybo/xyc}
REPO_DIR=${REPO_DIR:-${BASE_DIR}/CycleGRPO-OPSD}
PYTHON_BIN=${PYTHON_BIN:-${BASE_DIR}/envs/cyclegrpo/bin/python}
DEFAULT_MODEL=${DEFAULT_MODEL:-${REPO_DIR}/logs/cyclegrpo70k_historical_seca_8gpu/evaluation/step_179_cpu_export/hf_global_step_179}

MODEL_PATH=${MODEL_PATH:-$DEFAULT_MODEL}
if [[ $# -gt 0 && "$1" != --* ]]; then
    MODEL_PATH=$1
    shift
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python not found: $PYTHON_BIN" >&2
    exit 2
fi
if [[ ! -f "$MODEL_PATH/config.json" || ! -f "$MODEL_PATH/model.safetensors.index.json" ]]; then
    echo "Invalid Hugging Face model directory: $MODEL_PATH" >&2
    exit 2
fi

# Base and co checkpoints carry their matching mask decoder files. Exported
# training checkpoints fall back to the base SAMTok decoder unless overridden.
VQ_SAM2_PATH=${VQ_SAM2_PATH:-$MODEL_PATH/mask_tokenizer_256x2.pth}
SAM2_PATH=${SAM2_PATH:-$MODEL_PATH/sam2.1_hiera_large.pt}
[[ -f "$VQ_SAM2_PATH" ]] || VQ_SAM2_PATH=${BASE_DIR}/Qwen3-VL-4B-SAMTok/mask_tokenizer_256x2.pth
[[ -f "$SAM2_PATH" ]] || SAM2_PATH=${BASE_DIR}/Qwen3-VL-4B-SAMTok/sam2.1_hiera_large.pt

if [[ ! -f "$VQ_SAM2_PATH" || ! -f "$SAM2_PATH" ]]; then
    echo "VQ-SAM2 or SAM2 checkpoint not found." >&2
    exit 2
fi

MODEL_NAME=$(basename "$MODEL_PATH")
MODEL_NAME=${MODEL_NAME//[^[:alnum:]._-]/_}
has_output_root=false
for argument in "$@"; do
    [[ "$argument" == "--output-root" || "$argument" == --output-root=* ]] && has_output_root=true
done
OUTPUT_ARGS=()
if [[ "$has_output_root" == false ]]; then
    OUTPUT_ARGS=(--output-root "$REPO_DIR/tools/demo/output/$MODEL_NAME")
fi

cd "$REPO_DIR"
exec "$PYTHON_BIN" -u tools/demo_segmentation.py \
    --model-path "$MODEL_PATH" \
    --vq-sam2-path "$VQ_SAM2_PATH" \
    --sam2-path "$SAM2_PATH" \
    "${OUTPUT_ARGS[@]}" \
    "$@"
