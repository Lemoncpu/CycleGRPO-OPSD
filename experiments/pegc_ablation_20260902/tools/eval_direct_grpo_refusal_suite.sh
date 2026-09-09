#!/usr/bin/env bash
set -euo pipefail
BASE=/volume/ybo/xyc
REPO=$BASE/CycleGRPO-OPSD
BR=$REPO/experiments/pegc_ablation_20260902
export PYTHONPATH="$BR${PYTHONPATH:+:$PYTHONPATH}"
cd "$BR"
PY=$BASE/envs/cyclegrpo/bin/python3
EVAL=$BR/projects/eval/qwen3vl_4b_volcengine.sh
MODEL=$BASE/Qwen3-VL-4B-SAMTok
KEY=/tmp/llama_eval_key
while [[ ! -d "$BR/logs/pegc_direct_grpo_only_baseline_1of10_epoch1/checkpoints/global_step_18" || ! -d "$BR/logs/pegc_direct_grpo_only_refusal_credit_1of10_epoch1/checkpoints/global_step_18" ]]; do sleep 60; done
echo "Both clean checkpoints found at $(date -u)"
LLAMA_LOG=$BR/logs/pegc_clean_eval_llama.log
setsid env CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=true "$PY" -m vllm.entrypoints.openai.api_server --model "$BASE/models/Meta-Llama-3.1-8B-Instruct-hf-v2" --served-model-name llama3.1-8b --host 127.0.0.1 --port 8007 --api-key sk-cyclegrpo-local --tensor-parallel-size 1 --pipeline-parallel-size 1 --trust-remote-code --dtype bfloat16 --gpu-memory-utilization 0.85 --chat-template "$BR/tools/multinode/llama3_chat_template.jinja" >"$LLAMA_LOG" 2>&1 &
LPID=$!
for i in $(seq 1 240); do if ! kill -0 "$LPID" 2>/dev/null; then echo "Llama judge exited" >&2; tail -80 "$LLAMA_LOG" >&2; exit 1; fi; if curl --silent --max-time 3 -H "Authorization: Bearer sk-cyclegrpo-local" http://127.0.0.1:8007/v1/models | grep -q llama3.1-8b; then break; fi; sleep 2; done
curl --silent --max-time 3 -H "Authorization: Bearer sk-cyclegrpo-local" http://127.0.0.1:8007/v1/models | grep -q llama3.1-8b || { echo "Llama judge unhealthy" >&2; exit 1; }
for NAME in baseline refusal_credit; do
  if [[ "$NAME" == baseline ]]; then RUN=$BR/logs/pegc_direct_grpo_only_baseline_1of10_epoch1; else RUN=$BR/logs/pegc_direct_grpo_only_refusal_credit_1of10_epoch1; fi
  OUT=$RUN/evaluation/step_18; mkdir -p "$OUT"
  COMMON="BASE_DIR=$BASE REPO_DIR=$BR PYTHON_BIN=$PY TRAIN_MODEL_PATH=$MODEL TRAIN_DATA=$BR/data_1of10/cycle_2k.parquet CHECKPOINT_PATH=$RUN/checkpoints/global_step_18 HF_MODEL_PATH=$OUT/hf_global_step_18 EVAL_ROOT=$OUT NUM_GPUS=3 CUDA_VISIBLE_DEVICES=0,1,2 RAY_SHORT_ROOT=/tmp/pegc-export-${NAME}"
  eval "$COMMON" bash "$EVAL" export >"$OUT/export.log" 2>&1
  eval "$COMMON" bash "$EVAL" refcoco >"$OUT/refcoco_infer.log" 2>&1
  "$PY" "$BR/evaluation/refcoco/qwen3vl_refcoco_eval.py" --save_dir "$OUT/refcoco_val" --metric_only >"$OUT/refcoco_metrics.json"
  eval "$COMMON" bash "$EVAL" groundingsuite >"$OUT/groundingsuite.log" 2>&1
  "$PY" "$BR/projects/vlm/tokenmask/evaluation/groundingsuite_metric.py" --image_dir "$BASE/third_party/GroundingSuite" --gt_file "$BASE/third_party/GroundingSuite/GroundingSuite-Eval.jsonl" --pred_folder "$OUT/groundingsuite" --mode mask --vis_samples 0 --output_file "$OUT/groundingsuite_metrics.json" >"$OUT/groundingsuite_metrics.log" 2>&1
  eval "$COMMON" bash "$EVAL" gres >"$OUT/gres.log" 2>&1
  eval "$COMMON" bash "$EVAL" dlc >"$OUT/dlc.log" 2>&1
  "$PY" "$BR/evaluation/dlc_bench/eval_llama_without_image.py" --pred "$OUT/dlc_bench_predictions.json" --qa "$BASE/describe-anything/evaluation/DLC-Bench/qa.json" --class-names "$BASE/describe-anything/evaluation/DLC-Bench/class_names.json" --model llama3.1-8b --base-url http://127.0.0.1:8007/v1 --api-key "$KEY" --quiet >"$OUT/dlc_metrics.log" 2>&1
done
printf "%s\n" "clean direct-GRPO comparison completed $(date -u)" > /tmp/pegc_clean_eval_summary.txt
for NAME in baseline refusal_credit; do
  if [[ "$NAME" == baseline ]]; then RUN=$BR/logs/pegc_direct_grpo_only_baseline_1of10_epoch1; else RUN=$BR/logs/pegc_direct_grpo_only_refusal_credit_1of10_epoch1; fi
  OUT=$RUN/evaluation/step_18
  "$PY" -c "import json, pathlib; o=pathlib.Path(\"$OUT\"); r=json.load(open(o/\"refcoco_metrics.json\")); g=json.load(open(o/\"groundingsuite_metrics.json\")); s=json.load(open(o/\"gres_metrics.json\")); d=json.load(open(o/\"dlc_bench_predictions_eval.json\")); print(\"$NAME\\tRefCOCO cIoU=%.4f mIoU=%.4f\\tGroundingSuite overall_gIoU=%.4f\\tGRES T_acc=%.4f N_acc=%.4f gIoU=%.4f cIoU=%.4f\\tDLC Pos=%.4f Neg=%.4f Avg=%.4f\" % (r.get(\"cIoU\",0),r.get(\"mIoU\",0),g.get(\"overall_giou\",g.get(\"overall_gIoU\",0)),s.get(\"T_acc\",0),s.get(\"N_acc\",0),s.get(\"gIoU\",0),s.get(\"cIoU\",0),d.get(\"avg_pos\",0),d.get(\"avg_neg\",0),(d.get(\"avg_pos\",0)+d.get(\"avg_neg\",0))/2)" | tee -a /tmp/pegc_clean_eval_summary.txt
done
kill -TERM -- -"$LPID" 2>/dev/null || kill "$LPID" 2>/dev/null || true
GPU_LIST=0,1,2,3 MEMORY_MIB=20000 PYTHON_BIN="$PY" bash "$REPO/tools/gpu_power_hold.sh" start
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> /tmp/pegc_clean_eval_summary.txt

while true; do sleep 60; done
