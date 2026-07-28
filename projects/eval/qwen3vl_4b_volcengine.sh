#!/usr/bin/env bash
# Export a Volcengine FSDP checkpoint and run the supported offline evaluations.
set -euo pipefail

ACTION=${1:-help}
BASE_DIR=${BASE_DIR:-/mnt/cxzx/workspace/data_transfer/houzhiyan}
REPO_DIR=${REPO_DIR:-${BASE_DIR}/CycleGRPO-OPSD}
ENV_DIR=${ENV_DIR:-${BASE_DIR}/envs/cyclegrpo}
TRAIN_MODEL_PATH=${TRAIN_MODEL_PATH:-${BASE_DIR}/Qwen3-VL-4B-SAMTok}
TRAIN_DATA=${TRAIN_DATA:-${BASE_DIR}/refcoco-train2014-assets/refcoco_train_10k_seed20260722_workspace_paths.parquet}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-${REPO_DIR}/logs/refcoco10k_opsd/checkpoints/global_step_78}
HF_MODEL_PATH=${HF_MODEL_PATH:-${REPO_DIR}/logs/refcoco10k_opsd/evaluation/hf_global_step_78}
EVAL_ROOT=${EVAL_ROOT:-${REPO_DIR}/logs/refcoco10k_opsd/evaluation}
REFCOCO_ROOT=${REFCOCO_ROOT:-${BASE_DIR}/refcoco-train2014-assets}
GROUNDINGSUITE_ROOT=${GROUNDINGSUITE_ROOT:-${BASE_DIR}/GSEval}
DLC_ROOT=${DLC_ROOT:-${BASE_DIR}/describe-anything/evaluation/DLC-Bench}
NUM_GPUS=${NUM_GPUS:-8}
RAY_SHORT_ROOT=${RAY_SHORT_ROOT:-/tmp/cgrpo-export-${UID:-$(id -u)}}

case "$ACTION" in
    export|refcoco|groundingsuite|dlc|all) ;;
    *) echo "Usage: $0 {export|refcoco|groundingsuite|dlc|all}" >&2; exit 2 ;;
esac

if [[ "${CONDA_PREFIX:-}" != "$ENV_DIR" ]] && command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_DIR"
fi
export PATH="$ENV_DIR/bin:$PATH"
PYTHON_BIN=${PYTHON_BIN:-$ENV_DIR/bin/python3}
[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }
[[ -d "$REPO_DIR" ]] || { echo "Repository not found: $REPO_DIR" >&2; exit 1; }
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
unset RAY_ADDRESS RAY_NAMESPACE
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=true WANDB_MODE=offline
export HF_HOME=${HF_HOME:-$BASE_DIR/cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$BASE_DIR/cache/hf_datasets}
cd "$REPO_DIR"
mkdir -p "$EVAL_ROOT"
"$PYTHON_BIN" -c 'import projects; print("Project import root:", projects.__path__[0])'

require_hf_model() {
    [[ -f "$HF_MODEL_PATH/config.json" ]] || { echo "HF export missing: run '$0 export' first." >&2; exit 1; }
    [[ -f "$HF_MODEL_PATH/model.safetensors" || -f "$HF_MODEL_PATH/model.safetensors.index.json" ]] || {
        echo "HF export has no safetensors weights: $HF_MODEL_PATH" >&2; exit 1;
    }
}

run_export() {
    [[ -d "$CHECKPOINT_PATH/actor" ]] || { echo "Checkpoint not found: $CHECKPOINT_PATH" >&2; exit 1; }
    [[ -f "$TRAIN_DATA" ]] || { echo "Training parquet not found: $TRAIN_DATA" >&2; exit 1; }
    [[ "$RAY_SHORT_ROOT" == /tmp/* ]] || { echo "RAY_SHORT_ROOT must be under /tmp." >&2; exit 1; }
    mkdir -p "$RAY_SHORT_ROOT"
    export RAY_TMPDIR="$RAY_SHORT_ROOT"
    "$PYTHON_BIN" -m verl.trainer.main config=projects/rl/config.yaml \
        "data.train_files=['$TRAIN_DATA']" "data.val_files=['$TRAIN_DATA']" \
        data.format_prompt="$REPO_DIR/projects/rl/format_prompt/non_thinking.jinja" \
        worker.actor.model.model_path="$TRAIN_MODEL_PATH" worker.export_mode=true \
        worker.opsd.enabled=false trainer.nnodes=1 trainer.n_gpus_per_node="$NUM_GPUS" \
        trainer.load_checkpoint_path="$CHECKPOINT_PATH" trainer.find_last_checkpoint=false \
        trainer.export_hf_model_path="$HF_MODEL_PATH" 'trainer.logger=["file"]'
}

run_refcoco() {
    require_hf_model
    VQ_SAM2_PATH="$TRAIN_MODEL_PATH/mask_tokenizer_256x2.pth" \
    SAM2_PATH="$TRAIN_MODEL_PATH/sam2.1_hiera_large.pt" PYTHON_BIN="$PYTHON_BIN" \
    bash evaluation/refcoco/run_refcoco_multigpu.sh "$NUM_GPUS" "$HF_MODEL_PATH" \
        "$EVAL_ROOT/refcoco_${REFCOCO_SPLIT:-val}" "$REFCOCO_ROOT" "${REFCOCO_SPLIT:-val}"
}

run_groundingsuite() {
    require_hf_model
    local dataset=${GROUNDINGSUITE_DATASET:-$GROUNDINGSUITE_ROOT/GroundingSuite-Eval.jsonl}
    [[ -f "$dataset" ]] || { echo "GroundingSuite JSONL not found: $dataset" >&2; exit 1; }
    VQ_SAM2_PATH="$TRAIN_MODEL_PATH/mask_tokenizer_256x2.pth" \
    SAM2_PATH="$TRAIN_MODEL_PATH/sam2.1_hiera_large.pt" DATASET="$dataset" \
    DATA_ROOT="$GROUNDINGSUITE_ROOT" COCO_ROOT="$REFCOCO_ROOT" PYTHON_BIN="$PYTHON_BIN" \
    bash evaluation/groundingsuite/run_groundingsuite_multigpu.sh "$NUM_GPUS" "$HF_MODEL_PATH" "$EVAL_ROOT/groundingsuite"
    "$PYTHON_BIN" projects/vlm/tokenmask/evaluation/groundingsuite_metric.py \
        --image_dir "$GROUNDINGSUITE_ROOT" --gt_file "$dataset" \
        --pred_folder "$EVAL_ROOT/groundingsuite_pred.jsonl" --mode mask --vis_samples 0 \
        --output_file "$EVAL_ROOT/groundingsuite_metrics.json"
}

run_dlc() {
    require_hf_model
    local annotations=${DLC_ANNOTATIONS:-$DLC_ROOT/annotations.json}
    [[ -f "$annotations" ]] || { echo "DLC annotations not found: $annotations" >&2; exit 1; }
    "$PYTHON_BIN" evaluation/dlc_bench/inference.py --model_path "$HF_MODEL_PATH" \
        --vq_sam2_path "$TRAIN_MODEL_PATH/mask_tokenizer_256x2.pth" \
        --sam2_path "$TRAIN_MODEL_PATH/sam2.1_hiera_large.pt" --data_type bf16 --seed 42 \
        --anno_file "$annotations" --image_folder "$DLC_ROOT" --cache_name refcoco10k_opsd \
        --output "$EVAL_ROOT/dlc_bench_predictions.json"
}

case "$ACTION" in
    export) run_export ;;
    refcoco) run_refcoco ;;
    groundingsuite) run_groundingsuite ;;
    dlc) run_dlc ;;
    all) run_export; run_refcoco; run_groundingsuite; run_dlc ;;
esac
