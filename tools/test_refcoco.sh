#!/usr/bin/env bash
set -euo pipefail

BASE_DIR=${BASE_DIR:-/volume/ybo/xyc}
REPO_DIR=${REPO_DIR:-${BASE_DIR}/CycleGRPO-OPSD}
PYTHON_BIN=${PYTHON_BIN:-${BASE_DIR}/envs/cyclegrpo/bin/python}
DEFAULT_MODEL=${DEFAULT_MODEL:-${REPO_DIR}/logs/cyclegrpo70k_historical_seca_8gpu/evaluation/step_179_cpu_export/hf_global_step_179}

# Match the repository's standard RefCOCO evaluator defaults. Explicit CLI
# arguments passed by the caller appear after these defaults and take priority.
REFCOCO_SPLIT=${REFCOCO_SPLIT:-val}
REFCOCO_SPLIT_BY=${REFCOCO_SPLIT_BY:-unc}
REFCOCO_GPUS=${REFCOCO_GPUS:-0,1,2,3}
REFCOCO_BATCH_SIZE=${REFCOCO_BATCH_SIZE:-16}
REFCOCO_MAX_NEW_TOKENS=${REFCOCO_MAX_NEW_TOKENS:-256}
REFCOCO_MASK_PROTOCOL=${REFCOCO_MASK_PROTOCOL:-legacy_union}
REFCOCO_PROMPT_TEMPLATE=${REFCOCO_PROMPT_TEMPLATE:-'Please segment {phrase} in this image.'}
REFCOCO_ROOT=${REFCOCO_ROOT:-$BASE_DIR/refcoco-train2014-assets}

MODEL_PATH=${MODEL_PATH:-$DEFAULT_MODEL}
if [[ $# -gt 0 && "$1" != --* ]]; then
    MODEL_PATH=$1
    shift
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python not found: $PYTHON_BIN" >&2
    exit 2
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "Invalid Hugging Face model directory: $MODEL_PATH" >&2
    exit 2
fi
MODEL_TYPE=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("model_type", ""))' "$MODEL_PATH/config.json")

# Keep predictions from different models isolated. The underlying evaluator's
# resume key is prompt/protocol, so sharing one directory could reuse another
# model's completed JSON files.
MODEL_NAME=$(basename "$MODEL_PATH")
MODEL_NAME=${MODEL_NAME//[^[:alnum:]._-]/_}
has_output_dir=false
for argument in "$@"; do
    [[ "$argument" == "--output-dir" || "$argument" == --output-dir=* ]] && has_output_dir=true
done
OUTPUT_ARGS=()
if [[ "$has_output_dir" == false ]]; then
    OUTPUT_ARGS=(--output-dir "$REPO_DIR/tools/demo/output/performance/$MODEL_NAME")
fi

cd "$REPO_DIR"
if [[ "$MODEL_TYPE" == "sa2va_chat" ]]; then
    if [[ ! -f "$MODEL_PATH/model.safetensors.index.json" && ! -f "$MODEL_PATH/pytorch_model.bin.index.json" ]]; then
        echo "Sa2VA checkpoint index not found: $MODEL_PATH" >&2
        exit 2
    fi
    for argument in "$@"; do
        case "$argument" in
            --vq-sam2-path|--sam2-path|--batch-size|--max-new-tokens|--mask-protocol)
                echo "${argument} is only supported by the SAMTok evaluator, not Sa2VA." >&2
                exit 2
                ;;
        esac
    done
    exec "$PYTHON_BIN" -u tools/eval_refcoco_sa2va.py \
        --model-path "$MODEL_PATH" \
        --python-bin "$PYTHON_BIN" \
        --refcoco-root "$REFCOCO_ROOT" \
        --split "$REFCOCO_SPLIT" \
        --split-by "$REFCOCO_SPLIT_BY" \
        --gpus "$REFCOCO_GPUS" \
        --prompt-template "$REFCOCO_PROMPT_TEMPLATE" \
        "${OUTPUT_ARGS[@]}" \
        "$@"
fi

if [[ ! -f "$MODEL_PATH/model.safetensors.index.json" && ! -f "$MODEL_PATH/pytorch_model.bin.index.json" ]]; then
    echo "Invalid Hugging Face model directory (checkpoint index not found): $MODEL_PATH" >&2
    exit 2
fi

VQ_SAM2_PATH=${VQ_SAM2_PATH:-$MODEL_PATH/mask_tokenizer_256x2.pth}
SAM2_PATH=${SAM2_PATH:-$MODEL_PATH/sam2.1_hiera_large.pt}
[[ -f "$VQ_SAM2_PATH" ]] || VQ_SAM2_PATH=${BASE_DIR}/Qwen3-VL-4B-SAMTok/mask_tokenizer_256x2.pth
[[ -f "$SAM2_PATH" ]] || SAM2_PATH=${BASE_DIR}/Qwen3-VL-4B-SAMTok/sam2.1_hiera_large.pt

if [[ ! -f "$VQ_SAM2_PATH" || ! -f "$SAM2_PATH" ]]; then
    echo "VQ-SAM2 or SAM2 checkpoint not found." >&2
    exit 2
fi

exec "$PYTHON_BIN" -u tools/eval_refcoco.py \
    --model-path "$MODEL_PATH" \
    --vq-sam2-path "$VQ_SAM2_PATH" \
    --sam2-path "$SAM2_PATH" \
    --split "$REFCOCO_SPLIT" \
    --split-by "$REFCOCO_SPLIT_BY" \
    --gpus "$REFCOCO_GPUS" \
    --batch-size "$REFCOCO_BATCH_SIZE" \
    --max-new-tokens "$REFCOCO_MAX_NEW_TOKENS" \
    --mask-protocol "$REFCOCO_MASK_PROTOCOL" \
    --prompt-template "$REFCOCO_PROMPT_TEMPLATE" \
    "${OUTPUT_ARGS[@]}" \
    "$@"
