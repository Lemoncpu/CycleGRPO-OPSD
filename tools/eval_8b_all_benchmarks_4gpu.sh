#!/usr/bin/env bash
# Run RefCOCO/+/g, GroundingSuite, DLC-Bench, MR-RefCOCO/+/g and MR-PACO
# for both local Qwen3-VL-8B weights.
# Every multi-GPU evaluator is explicitly launched with four visible GPUs.
# The DLC evaluator is currently single-process in the upstream implementation;
# it is recorded separately instead of pretending that four workers were used.

BASE_DIR=${BASE_DIR:-/volume/ybo/xyc}
REPO_DIR=${REPO_DIR:-$BASE_DIR/CycleGRPO-OPSD}
PYTHON_BIN=${PYTHON_BIN:-$BASE_DIR/envs/cyclegrpo/bin/python3}
TRAIN_MODEL_PATH=${TRAIN_MODEL_PATH:-$BASE_DIR/Qwen3-VL-8B-SAMTok}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-$REPO_DIR/logs/cyclegrpo70k_historical_seca_8b_8gpu/checkpoints/global_step_179}
HF_MODEL_PATH=${HF_MODEL_PATH:-$REPO_DIR/logs/cyclegrpo70k_historical_seca_8b_8gpu/evaluation/step_179_cpu_export/hf_global_step_179}
EVAL_ROOT=${EVAL_ROOT:-$REPO_DIR/logs/cyclegrpo70k_historical_seca_8b_8gpu/evaluation/all_benchmarks_4gpu}
REFCOCO_ROOT=${REFCOCO_ROOT:-$BASE_DIR/refcoco-train2014-assets}
REFCOCO_PLUS_ROOT=${REFCOCO_PLUS_ROOT:-$BASE_DIR/refcoco-plus-train2014-assets}
REFCOCOG_ROOT=${REFCOCOG_ROOT:-$BASE_DIR/refcocog-train2014-assets}
REFCOCO_SPLIT_BY=${REFCOCO_SPLIT_BY:-unc}
REFCOCOG_SPLIT_BY=${REFCOCOG_SPLIT_BY:-umd}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-'Please segment {phrase} in this image.'}
GROUNDINGSUITE_ROOT=${GROUNDINGSUITE_ROOT:-$BASE_DIR/third_party/GroundingSuite}
DLC_ROOT=${DLC_ROOT:-$BASE_DIR/describe-anything/evaluation/DLC-Bench}
NUM_GPUS=${NUM_GPUS:-4}
BENCHMARKS=${BENCHMARKS:-refcoco,refcoco_plus,refcocog,groundingsuite,dlc,mr_refcoco,mr_refcoco_plus,mr_refcocog,mr_paco}
WEIGHTS=${WEIGHTS:-base,step179}
MR_ROUNDS=${MR_ROUNDS:-1,2,3,4,5,6,7,8,9,10}
MR_DATA_ROOT=${MR_DATA_ROOT:-$BASE_DIR/CycleGRPO/data/segllm_data/conversation_folder/all_data_mix_val}
MR_IMAGE_ROOT=${MR_IMAGE_ROOT:-$BASE_DIR/refcoco-train2014-assets/train2014}
MR_REFCOCO_DATASET=${MR_REFCOCO_DATASET:-$MR_DATA_ROOT/mr_refcoco_val_sampled.json}
MR_REFCOCO_PLUS_DATASET=${MR_REFCOCO_PLUS_DATASET:-$MR_DATA_ROOT/mr_refcoco+_val_sampled.json}
MR_REFCOCOG_DATASET=${MR_REFCOCOG_DATASET:-$MR_DATA_ROOT/mr_refcocog_val_sampled.json}
MR_PACO_DATASET=${MR_PACO_DATASET:-$MR_DATA_ROOT/mr_paco_val_sampled.json}

log() { printf '[8b-eval] %s\n' "$*"; }
fail() { log "ERROR: $*" >&2; return 2; }

if [ "$NUM_GPUS" -ne 4 ]; then
    fail "NUM_GPUS must remain 4 for this runner (got $NUM_GPUS)" || exit $?
fi
if [ ! -f "$HF_MODEL_PATH/config.json" ]; then
    fail "missing HF export: $HF_MODEL_PATH/config.json" || exit $?
fi
if [ ! -f "$HF_MODEL_PATH/model.safetensors.index.json" ]; then
    fail "missing HF weights: $HF_MODEL_PATH/model.safetensors.index.json" || exit $?
fi
if [ ! -f "$TRAIN_MODEL_PATH/mask_tokenizer_256x2.pth" ] || [ ! -f "$TRAIN_MODEL_PATH/sam2.1_hiera_large.pt" ]; then
    fail "missing SAMTok auxiliary weights under $TRAIN_MODEL_PATH" || exit $?
fi

case "$WEIGHTS" in
    *base*) BASE_MODEL="$TRAIN_MODEL_PATH" ;;
esac
case "$WEIGHTS" in
    *step179*) STEP179_MODEL="$HF_MODEL_PATH" ;;
esac
case "$WEIGHTS" in
    *step200*) STEP200_MODEL="$HF_MODEL_PATH" ;;
esac
case "$WEIGHTS" in
    *step357*) STEP357_MODEL="$HF_MODEL_PATH" ;;
esac

run_action() {
    local model_name=$1 model_path=$2 action=$3 root=$4
    local requested_action="$action" split_by="$REFCOCO_SPLIT_BY"
    local ref_root="$REFCOCO_ROOT"
    case "$action" in
        refcoco_plus) ref_root="$REFCOCO_PLUS_ROOT"; action=refcoco ;;
        refcocog) ref_root="$REFCOCOG_ROOT"; action=refcoco; split_by="$REFCOCOG_SPLIT_BY" ;;
    esac
    log "$model_name: $action using $model_path"
    BASE_DIR="$BASE_DIR" REPO_DIR="$REPO_DIR" ENV_DIR="$BASE_DIR/envs/cyclegrpo" \
    PYTHON_BIN="$PYTHON_BIN" TRAIN_MODEL_PATH="$TRAIN_MODEL_PATH" \
    HF_MODEL_PATH="$model_path" EVAL_ROOT="$root/$requested_action" \
    CHECKPOINT_PATH="$CHECKPOINT_PATH" REFCOCO_ROOT="$REFCOCO_ROOT" \
    GROUNDINGSUITE_ROOT="$GROUNDINGSUITE_ROOT" DLC_ROOT="$DLC_ROOT" \
    REFCOCO_SPLIT=val SPLIT_BY="$split_by" \
    REFCOCO_ROOT="$ref_root" \
    PROMPT_TEMPLATE="$PROMPT_TEMPLATE" \
    NUM_GPUS=4 bash "$REPO_DIR/projects/eval/qwen3vl_4b_volcengine.sh" "$action"
    local status=$?
    log "$model_name: $action exited status=$status"
    return "$status"
}

run_mr_benchmark() {
    local model_name=$1 model_path=$2 bench=$3 root=$4 dataset script pred_dir
    case "$bench" in
        mr_refcoco) dataset="$MR_REFCOCO_DATASET"; script=qwen3vl_mrrefcoco_eval.py; pred_dir=mr_refcoco ;;
        mr_refcoco_plus) dataset="$MR_REFCOCO_PLUS_DATASET"; script=qwen3vl_mrrefcoco+_eval.py; pred_dir=mr_refcoco_plus ;;
        mr_refcocog) dataset="$MR_REFCOCOG_DATASET"; script=qwen3vl_mrrefcocog_eval.py; pred_dir=mr_refcocog ;;
        mr_paco) dataset="$MR_PACO_DATASET"; script=qwen3vl_mrpaco_eval.py; pred_dir=mr_paco ;;
        *) log "$model_name: unknown MR benchmark '$bench'"; return 2 ;;
    esac
    if [ ! -f "$dataset" ]; then
        log "$model_name: $bench dataset missing: $dataset"
        return 2
    fi
    local mr_root="$root/$bench"
    case "$mr_root" in
        /*) ;;
        *) mr_root="$REPO_DIR/$mr_root" ;;
    esac
    mkdir -p "$mr_root"
    local normalized_dataset="$mr_root/$(basename "$dataset" .json).absolute.json"
    if [ ! -f "$normalized_dataset" ]; then
        "$PYTHON_BIN" - "$dataset" "$normalized_dataset" "$MR_IMAGE_ROOT" <<'PY'
import json, os, sys
source, target, image_root = sys.argv[1:]
data = json.load(open(source))
for rows in data.values():
    for row in rows:
        image = row.get("image", "")
        if image.startswith("./data/coco/train2014/"):
            row["image"] = os.path.join(image_root, os.path.basename(image))
json.dump(data, open(target, "w"), ensure_ascii=False)
PY
    fi
    dataset="$normalized_dataset"
    IFS=',' read -ra rounds <<< "$MR_ROUNDS"
    local gpu idx round work pid status=0
    pids=()
    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
        (
            for ((idx=gpu; idx<${#rounds[@]}; idx+=NUM_GPUS)); do
                round=${rounds[$idx]}
                work="$mr_root/gpu${gpu}/round${round}"
                mkdir -p "$work/temp_save/$pred_dir"
                cd "$work" || exit 2
                CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" \
                "$PYTHON_BIN" -u "$REPO_DIR/projects/vlm/tokenmask/evaluation/qwen3vl/$script" \
                    "$model_path" --vq_sam2_path "$TRAIN_MODEL_PATH/mask_tokenizer_256x2.pth" \
                    --dataset "$dataset" --round_id "$round" \
                    > inference.log 2>&1 || exit $?
            done
        ) &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    log "$model_name: $bench exited status=$status"
    return "$status"
}

run_model() {
    local model_name=$1 model_path=$2
    local model_root="$EVAL_ROOT/$model_name"
    mkdir -p "$model_root"
    log "starting $model_name (hidden_size is checked by the evaluator; expected 4096)"
    IFS=',' read -ra actions <<< "$BENCHMARKS"
    local action status=0
    for action in "${actions[@]}"; do
        case "$action" in
            refcoco|refcoco_plus|refcocog|groundingsuite|gres|dlc)
                run_action "$model_name" "$model_path" "$action" "$model_root" || status=1 ;;
            mr_refcoco|mr_refcoco_plus|mr_refcocog|mr_paco)
                run_mr_benchmark "$model_name" "$model_path" "$action" "$model_root" || status=1 ;;
            *)
                log "$model_name: unknown benchmark '$action' (skipped)"
                status=1 ;;
        esac
    done
    return "$status"
}

mkdir -p "$EVAL_ROOT"
overall=0
case "$WEIGHTS" in
    *base*) run_model base "$BASE_MODEL" || overall=1 ;;
esac
case "$WEIGHTS" in
    *step179*) run_model step179 "$STEP179_MODEL" || overall=1 ;;
esac
case "$WEIGHTS" in
    *step200*) run_model step200 "$STEP200_MODEL" || overall=1 ;;
esac
case "$WEIGHTS" in
    *step357*) run_model step357 "$STEP357_MODEL" || overall=1 ;;
esac
log "finished all requested 8B weights; overall_status=$overall"
exit "$overall"
