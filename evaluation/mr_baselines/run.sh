#!/usr/bin/env bash
# One model per GPU; retain the existing hold worker's allocation throughout.
REPO_DIR=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO_DIR" || exit 2
model=$1
gpu=$2
mode=${3:-full}
start_id=${4:-0}
ROOT=${BASELINE_SOURCE_ROOT:-/tmp/cyclegrpo-baseline-eval.DIimEr}
PYTHON_BIN=${PYTHON_BIN:-/volume/ybo/xyc/envs/cyclegrpo/bin/python3}
SITE_PACKAGES=$("$PYTHON_BIN" -c 'import site; print(site.getsitepackages()[0])') || exit 2
RUN_ROOT=${RUN_ROOT:-$REPO_DIR/logs/mr_baselines_20260924}
DATA_ROOT=${DATA_ROOT:-/volume/ybo/xyc/CycleGRPO/data/segllm_data/conversation_folder/all_data_mix_val}
if [ "$mode" = smoke ]; then DATA_ROOT=$RUN_ROOT/smoke_data; fi
OUTPUT_DIR=${OUTPUT_DIR:-$RUN_ROOT/${model}_${mode}}
mkdir -p "$OUTPUT_DIR" || exit 2
exec > >(tee "$OUTPUT_DIR/inference.log") 2>&1
case "$model" in
  unipixel) checkpoint=UniPixel-3B; export PYTHONPATH=$SITE_PACKAGES:$ROOT/UniPixel/vendor:$ROOT/UniPixel ;;
  evfsam) checkpoint=EVF-SAM; export PYTHONPATH=$ROOT/EVF-SAM ;;
  padt) checkpoint=PaDT_Pro_3B; export PYTHONPATH=$ROOT/padt-transformers-compat:$ROOT/PaDT/src:$ROOT/PaDT/eval/evaluation_scripts ;;
  instructseg) checkpoint=InstructSeg-3B; export PYTHONPATH=$SITE_PACKAGES:$ROOT/InstructSeg/vendor:$ROOT/detectron2:$ROOT/InstructSeg:$ROOT/InstructSeg/instructseg/model/mask_decoder/Mask2Former_Simplify/modeling/pixel_decoder/ops/build/lib.linux-x86_64-cpython-310:$ROOT/InstructSeg/instructseg/model/mask_decoder/Mask2Former_Simplify/modeling/pixel_decoder/ops ;;
  *) echo "Unsupported model: $model"; exit 2 ;;
esac
if [ -e "$OUTPUT_DIR/rank0.jsonl" ] || [ -e "$OUTPUT_DIR/sample_map.json" ]; then
  echo "Refusing to overwrite existing predictions: $OUTPUT_DIR"; exit 2
fi
hold_pid=$(ps -eo pid,args | awk -v gpu="$gpu" '$0 ~ /--worker-tag cyclegrpo_gpu_power_hold_worker/ && $NF == gpu {print $1}')
if [ -z "$hold_pid" ] || [ "$(echo "$hold_pid" | wc -w)" -ne 1 ]; then
  echo "Expected one existing hold worker for GPU $gpu; refusing to create an occupancy gap."; exit 2
fi
# STOP pauses compute but preserves CUDA context and allocated VRAM.
trap 'kill -CONT "$hold_pid"' EXIT
trap 'exit 130' INT TERM
kill -STOP "$hold_pid" || exit 2
export CUDA_VISIBLE_DEVICES=$gpu WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29710 + gpu))
extra=()
if [ "$start_id" -gt 0 ]; then extra+=(--start-id "$start_id"); fi
if [ "$model" = padt ]; then extra+=(--batch-size "${PADT_BATCH_SIZE:-8}"); fi
if [ "$model" = instructseg ]; then
  extra+=(--repo "$ROOT/InstructSeg" --vision-tower "$RUN_ROOT/assets/siglip-so400m-patch14-384" --workers 2)
fi
"$PYTHON_BIN" -u "evaluation/mr_baselines/${model}_infer.py" \
 --model "/volume/ybo/xyc/models/benchmark_baselines/$checkpoint" \
 --data-root "$DATA_ROOT" --images /volume/ybo/xyc/refcoco-train2014-assets/train2014 \
 --output "$OUTPUT_DIR" --benchmarks mr_refcoco mr_refcoco+ mr_refcocog mr_paco "${extra[@]}"
status=$?
if [ "$status" -eq 0 ] && [ "$model" = padt ]; then
 "$PYTHON_BIN" evaluation/mr_baselines/convert_padt.py --pred-dir "$OUTPUT_DIR"
 status=$?
fi
if [ "$status" -eq 0 ] && [ "$start_id" -eq 0 ]; then
 "$PYTHON_BIN" evaluation/mr_baselines/score.py --data-root "$DATA_ROOT" --pred-dir "$OUTPUT_DIR" --output "$OUTPUT_DIR/metrics.json"
 status=$?
fi
printf '%s\n' "$status" > "$OUTPUT_DIR/exit_status"
exit "$status"
