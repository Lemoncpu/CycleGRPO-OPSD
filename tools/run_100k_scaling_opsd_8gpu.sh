#!/usr/bin/env bash
# Build disjoint 40k/40k/20k data, generate/verify DLC-QA with a runner-owned
# local Llama-3.1-8B OpenAI-compatible service, then run the original 70k
# OPSD+SECA 7-train+1-judge topology at 100k scale.

set -euo pipefail

BASE_DIR="${BASE_DIR:-/volume/ybo/xyc}"
REPO_DIR="${REPO_DIR:-${BASE_DIR}/CycleGRPO-OPSD}"
ENV_DIR="${ENV_DIR:-${BASE_DIR}/envs/cyclegrpo}"
MODEL_PATH="${MODEL_PATH:-${BASE_DIR}/Qwen3-VL-4B-SAMTok}"
DATA_ROOT="${DATA_ROOT:-${BASE_DIR}/datasets/cyclegrpo100k_scaling_20260911}"
QA_PORT="${QA_PORT:-8007}"
QA_BASE_URL="${QA_BASE_URL:-http://127.0.0.1:${QA_PORT}/v1}"
QA_MODEL_NAME="${QA_MODEL_NAME:-llama3.1-8b}"
QA_API_KEY="${QA_API_KEY:-sk-cyclegrpo-local}"
QA_LLAMA_MODEL_PATH="${QA_LLAMA_MODEL_PATH:-${BASE_DIR}/models/Meta-Llama-3.1-8B-Instruct-hf-v2}"
QA_LLAMA_GPU="${QA_LLAMA_GPU:-7}"
QA_LLAMA_TEMPLATE="${QA_LLAMA_TEMPLATE:-${REPO_DIR}/tools/multinode/llama3_chat_template.jinja}"
QA_LLAMA_LOG="${QA_LLAMA_LOG:-${DATA_ROOT}/llama3.1-8b_qa_vllm.log}"
JUDGE_TIMEOUT_SECONDS="${JUDGE_TIMEOUT_SECONDS:-600}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
TRAIN_GPU_LIST="${TRAIN_GPU_LIST:-0,1,2,3,4,5,6}"
RUN_NAME="${RUN_NAME:-cyclegrpo100k_opsd_seca_positive_only_8gpu}"
RUN_ROOT="${RUN_ROOT:-${REPO_DIR}/logs/${RUN_NAME}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUN_ROOT/checkpoints}"

PY="${ENV_DIR}/bin/python3"
VLLM="${ENV_DIR}/bin/vllm"
DATA_HELPER="${REPO_DIR}/tools/prepare_100k_scaling_data.py"
TRAIN_ENTRY="${REPO_DIR}/projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh"
RAY_ADDRESS="${RAY_ADDRESS:-127.0.0.1:6379}"
RAY_NAMESPACE="${RAY_NAMESPACE:-cyclegrpo-100k-scaling-8gpu}"
RAY_SHORT_ROOT="${RAY_SHORT_ROOT:-/dev/shm/cgrpo-ray100k-${UID:-$(id -u)}}"
QA_PID=""
QA_SERVICE_OWNED=false
TRAIN_PID=""
TRAIN_HEARTBEAT_PID=""
RUN_LOG="${RUN_LOG:-$RUN_ROOT/run.log}"
TRAIN_LOG="${TRAIN_LOG:-$RUN_ROOT/training.log}"
QA_GENERATION_LOG="${QA_GENERATION_LOG:-$DATA_ROOT/llama_caption_generation.log}"

# Initialize the persistent log stream before traps and preflight so bootstrap
# failures are recorded just like data-export and training failures.
mkdir -p "$DATA_ROOT" "$RUN_ROOT"
touch "$RUN_LOG"
exec > >(tee -a "$RUN_LOG") 2>&1

log_line() {
    printf '[%s] %s
' "$(date -u +%FT%TZ)" "$*"
}

CURRENT_STAGE="bootstrap"
trap 'status=$?; if (( status != 0 )); then log_line "[error] stage=${CURRENT_STAGE} status=${status} line=${BASH_LINENO[0]:-0} command=${BASH_COMMAND}"; fi' ERR

start_final_hold() {
    STATE_DIR="${FINAL_HOLD_STATE_DIR:-/tmp/cyclegrpo_gpu_power_hold_${RUN_NAME}}" \
        GPU_LIST=0,1,2,3,4,5,6,7 MEMORY_MIB=20000 MATMUL_DIM=4096 PYTHON_BIN="$PY" \
        bash "$REPO_DIR/tools/gpu_power_hold.sh" start || true
    log_line "[stage:hold] eight-GPU power hold requested"
}
trap start_final_hold EXIT INT TERM

CURRENT_STAGE="preflight"
log_line "100k scaling runner started: RUN_NAME=$RUN_NAME"
log_line "[stage:bootstrap] log initialized; pid=$$ shell=$BASH_VERSION cwd=$(pwd)"
log_line "[config] repo=$REPO_DIR env=$ENV_DIR model=$MODEL_PATH qa_model=$QA_MODEL_NAME qa_base_url=$QA_BASE_URL"
log_line "[config] data_root=$DATA_ROOT run_root=$RUN_ROOT run_log=$RUN_LOG train_log=$TRAIN_LOG"
log_line "[config] generation_gpus=$GPU_LIST training_gpus=$TRAIN_GPU_LIST qa_port=$QA_PORT ray=$RAY_ADDRESS"
cd "$REPO_DIR"
log_line "[stage:preflight] start; run_log=$RUN_LOG data_root=$DATA_ROOT run_root=$RUN_ROOT"

for required in "$PY" "$VLLM" "$DATA_HELPER" "$TRAIN_ENTRY" "$MODEL_PATH/config.json" \
    "$MODEL_PATH/mask_tokenizer_256x2.pth" "$MODEL_PATH/sam2.1_hiera_large.pt" "$QA_LLAMA_MODEL_PATH/config.json" "$QA_LLAMA_TEMPLATE"; do
    if [[ ! -e "$required" ]]; then
        log_line "[error] missing required path: $required" >&2
        exit 1
    fi
    log_line "[progress:preflight] verified $required"
done

if [[ "$GPU_LIST" != "0,1,2,3,4,5,6,7" ]]; then
    log_line "[error] GPU_LIST must expose all eight GPUs for the data-generation stage: $GPU_LIST" >&2
    exit 1
fi
log_line "[stage:preflight] complete; generation_gpus=$GPU_LIST training_gpus=$TRAIN_GPU_LIST"

# The source pools are intentionally over-generated on eight independent
# devices. Cross-shard duplicates are removed by prepare_100k_scaling_data.py.
POOL_ROOT="$DATA_ROOT/pools"
mkdir -p "$POOL_ROOT"
POOL_PIDS=()
QA_POOL_PIDS=()
wait_for_jobs() {
    local label="$1" total="$2"
    shift 2
    local pids=("$@")
    local started=$SECONDS active completed failed=0 pid status
    while true; do
        active=0
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                active=$((active + 1))
            fi
        done
        completed=$((total - active))
        log_line "[progress:$label] completed=$completed/$total active=$active elapsed=$((SECONDS - started))s"
        log_export_progress "$label" "$total"
        if (( active == 0 )); then
            break
        fi
        sleep 30
    done
    for pid in "${pids[@]}"; do
        if wait "$pid"; then
            status=0
        else
            status=$?
        fi
        log_line "[progress:$label] pid=$pid exit_status=$status"
        if (( status != 0 )); then
            failed=1
        fi
    done
    if (( failed != 0 )); then
        log_line "[stage:$label] one or more workers failed; see per-shard logs"
        return 1
    fi
}

log_parquet_rows() {
    local label="$1" path="$2"
    if [[ ! -f "$path" ]]; then
        log_line "[progress:data-count] $label missing: $path"
        return 1
    fi
    "$PY" - "$label" "$path" <<'PY'
import sys
import pyarrow.parquet as pq

label, path = sys.argv[1:]
rows = pq.ParquetFile(path).metadata.num_rows
print(f"[progress:data-count] {label} rows={rows} path={path}")
PY
}

log_jsonl_rows() {
    local label="$1" path="$2" count=0
    if [[ -f "$path" ]]; then
        count=$(awk 'NF {n++} END {print n+0}' "$path")
        log_line "[progress:data-count] $label rows=$count path=$path"
    else
        log_line "[progress:data-count] $label missing: $path"
        return 1
    fi
}

log_json_object_keys() {
    local label="$1" path="$2"
    if [[ ! -f "$path" ]]; then
        log_line "[progress:data-count] $label missing: $path"
        return 1
    fi
    "$PY" - "$label" "$path" <<'PY'
import json
import sys
label, path = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)
if not isinstance(payload, dict):
    raise SystemExit(f"{path} is not a JSON object")
print(f"[progress:data-count] {label} keys={len(payload)} path={path}")
PY
}
log_export_progress() {
    local label="$1" expected="$2" summary
    local -a paths=()
    shopt -s nullglob
    case "$label" in
        data-pools) paths=("$POOL_ROOT"/stuff_shard_*.parquet "$POOL_ROOT"/part_shard_*.parquet) ;;
        qa-pools) paths=("$QA_STUFF_SHARD_DIR"/shard_*.parquet "$QA_STUFF_SHARD_DIR"/shard_*.jsonl "$QA_PART_SHARD_DIR"/shard_*.parquet "$QA_PART_SHARD_DIR"/shard_*.jsonl) ;;
        *) shopt -u nullglob; log_line "[progress:export] stage=$label artifacts=untracked"; return 0 ;;
    esac
    shopt -u nullglob
    # Count shards, rows, artifacts and bytes in one helper invocation. QA has
    # two files per shard, so files must not be reported as shard progress.
    if summary=$("$PY" - "$label" "$expected" "${paths[@]}" <<'PY'
import re
import sys
from pathlib import Path

label, expected = sys.argv[1], int(sys.argv[2])
paths = [Path(value) for value in sys.argv[3:]]
completed = set()
artifacts = 0
bytes_total = 0
parquet_rows = 0
jsonl_rows = 0
try:
    import pyarrow.parquet as pq
except Exception:
    pq = None

for path in paths:
    if not path.is_file() or path.stat().st_size == 0:
        continue
    artifacts += 1
    bytes_total += path.stat().st_size
    if label == "data-pools":
        match = re.search(r"(stuff|part)_shard_(\d+)\.parquet$", path.name)
        if match:
            completed.add(f"{match.group(1)}-{match.group(2)}")
    elif label == "qa-pools":
        match = re.search(r"shard_(\d+)\.(parquet|jsonl)$", path.name)
        if match:
            completed.add(f"{path.parent.name}-{match.group(1)}")
    if path.suffix == ".parquet" and pq is not None:
        try:
            parquet_rows += int(pq.ParquetFile(path).metadata.num_rows)
        except Exception:
            pass
    elif path.suffix == ".jsonl":
        try:
            with path.open(encoding="utf-8") as handle:
                jsonl_rows += sum(1 for line in handle if line.strip())
        except OSError:
            pass

# A DAM QA shard is complete only after both its parquet and JSONL manifest
# are non-empty.
if label == "qa-pools":
    complete_pairs = set()
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            continue
        match = re.search(r"shard_(\d+)\.(parquet|jsonl)$", path.name)
        if match:
            complete_pairs.add((path.parent.name, match.group(1), match.group(2)))
    completed = {
        f"{parent}-{shard}"
        for parent, shard, _ in complete_pairs
        if (parent, shard, "parquet") in complete_pairs
        and (parent, shard, "jsonl") in complete_pairs
    }

print(
    f"stage={label} completed_shards={len(completed)}/{expected} "
    f"artifacts={artifacts} parquet_rows={parquet_rows} "
    f"jsonl_rows={jsonl_rows} bytes={bytes_total}"
)
PY
); then
        log_line "[progress:export] $summary"
    else
        log_line "[progress:export] stage=$label completed_shards=unknown/$expected summary_failed"
    fi
}

stop_training_watchers() {
    local pid
    for pid in "$TRAIN_HEARTBEAT_PID"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    TRAIN_HEARTBEAT_PID=""
}

start_training_watchers() {
    touch "$TRAIN_LOG"
    local training_pid="$1"
    (
        local started=$SECONDS bytes latest checkpoints
        while kill -0 "$training_pid" 2>/dev/null; do
            bytes=$(wc -c < "$TRAIN_LOG" 2>/dev/null || printf 0)
            latest=$(tail -n 1 "$TRAIN_LOG" 2>/dev/null | tr '
' ' ' | cut -c1-240)
            checkpoints=$(find "$CHECKPOINT_DIR" -maxdepth 2 -type d -name 'global_step_*' 2>/dev/null | sort | tr '
' ',' | cut -c1-240)
            log_line "[progress:train] pid=$training_pid elapsed=$((SECONDS - started))s log_bytes=$bytes checkpoints=${checkpoints:-<none>} last=${latest:-<no-output>}"
            sleep 60
        done
        log_line "[progress:train] pid=$training_pid exited; final_log_bytes=$(wc -c < "$TRAIN_LOG" 2>/dev/null || printf 0)"
    ) &
    TRAIN_HEARTBEAT_PID=$!
}

run_pool() {
    local gpu="$1" kind="$2" shard="$3" count="$4"
    local output="$POOL_ROOT/${kind}_shard_${shard}.parquet"
    if [[ -s "$output" ]]; then
        log_line "[stage:data-pools] reusing $output"
        log_parquet_rows "${kind}_shard_${shard}" "$output"
        return
    fi
    if [[ "$kind" == "stuff" ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO_DIR" "$PY" \
            projects/rl/datasets/prepare_cocostuff_cycle_dataset.py \
            --masks-dir "$BASE_DIR/COCO-Stuff/train2017" --images-dir "$BASE_DIR/coco2017/train2017" \
            --output "$output" --mask-tokenizer-path "$MODEL_PATH/mask_tokenizer_256x2.pth" \
            --sam2-checkpoint "$MODEL_PATH/sam2.1_hiera_large.pt" \
            --sam2-config-dir "$REPO_DIR/projects/transformers/vq_sam2/sam2/sam2_configs" \
            --max-samples "$count" --seed "$((20260911 + shard * 17))" --device cuda \
            > >(tee -a "$RUN_LOG" "${output%.parquet}.log" >/dev/null) 2>&1 &
    elif [[ "$kind" == "part" ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO_DIR" "$PY" \
            projects/rl/datasets/prepare_paco_lvis_part_cycle_dataset.py \
            --annotations "$BASE_DIR/PACO-LVIS/annotations/paco_lvis_v1_train.json" \
            --images-dir "$BASE_DIR/coco2017" --output "$output" \
            --mask-tokenizer-path "$MODEL_PATH/mask_tokenizer_256x2.pth" \
            --sam2-checkpoint "$MODEL_PATH/sam2.1_hiera_large.pt" \
            --sam2-config-dir "$REPO_DIR/projects/transformers/vq_sam2/sam2/sam2_configs" \
            --max-samples "$count" --seed "$((20260911 + shard * 19))" --device cuda \
            > >(tee -a "$RUN_LOG" "${output%.parquet}.log" >/dev/null) 2>&1 &
    else
        log_line "[error] unknown pool kind: $kind" >&2
        return 2
    fi
    POOL_PIDS+=("$!")
    log_line "[stage:data-pools] launched kind=$kind shard=$shard gpu=$gpu pid=$! output=$output"
}

CURRENT_STAGE="data-pools"
log_line "[stage:data-pools] start 4 Stuff + 4 PACO shards; each target=10000"
run_pool 0 stuff 0 10000
run_pool 1 stuff 1 10000
run_pool 2 stuff 2 10000
run_pool 3 stuff 3 10000
run_pool 4 part 0 10000
run_pool 5 part 1 10000
run_pool 6 part 2 10000
run_pool 7 part 3 10000
wait_for_jobs data-pools 8 "${POOL_PIDS[@]}"
log_line "[stage:data-pools] complete"
for shard in 0 1 2 3; do
    log_parquet_rows "stuff_shard_$shard" "$POOL_ROOT/stuff_shard_$shard.parquet"
done
for shard in 0 1 2 3; do
    log_parquet_rows "part_shard_$shard" "$POOL_ROOT/part_shard_$shard.parquet"
done

REFCOCO_CANDIDATE="$BASE_DIR/datasets/refcoco_full_train/refcoco_train_full_42404_seed20260820.parquet"
GREFCOCO_CANDIDATE="$BASE_DIR/datasets/direct_refcoco30k_grefcoco_multi10k_disjoint_cycle20k_seed20260905/grefcoco_train_10000pos_0notarget_seed20260905_positive.parquet"
GREFCOCO_CYCLE="$BASE_DIR/datasets/cyclegrpo_20k_raw_seed20260820/cyclegrpo_20k_40_20_25_10_5_seed20260820.parquet"
NO_TARGET_CANDIDATE="$BASE_DIR/datasets/grefcoco_no_target_all_train/grefcoco_train_0pos_11466notarget_seed20260823_no_target.parquet"
QA_STUFF_MANIFEST="$DATA_ROOT/qa_stuff_manifest.jsonl"
QA_PART_MANIFEST="$DATA_ROOT/qa_part_manifest.jsonl"
QA_STUFF_POOL="$POOL_ROOT/qa_stuff.parquet"
QA_PART_POOL="$POOL_ROOT/qa_part.parquet"

CURRENT_STAGE="candidate-validation"
log_line "[stage:candidate-validation] start"
for candidate in "$REFCOCO_CANDIDATE" "$GREFCOCO_CANDIDATE" "$GREFCOCO_CYCLE" "$NO_TARGET_CANDIDATE"; do
    if [[ ! -f "$candidate" ]]; then
        log_line "[error] missing candidate parquet: $candidate" >&2
        exit 1
    fi
done
log_parquet_rows "candidate_refcoco" "$REFCOCO_CANDIDATE"
log_parquet_rows "candidate_grefcoco_positive" "$GREFCOCO_CANDIDATE"
log_parquet_rows "candidate_cycle" "$GREFCOCO_CYCLE"
log_parquet_rows "candidate_no_target" "$NO_TARGET_CANDIDATE"
log_line "[stage:candidate-validation] complete"

# Build two DAM caption pools for the QA stream. Each pool is split into
# four 2.5k shards so every physical GPU remains busy during this stage.
# They are separate from the label-query Stuff/Part pools used by direct
# supervision; the merge is deterministic after all workers succeed.
QA_STUFF_SHARD_DIR="$POOL_ROOT/qa_stuff_shards"
QA_PART_SHARD_DIR="$POOL_ROOT/qa_part_shards"
mkdir -p "$QA_STUFF_SHARD_DIR" "$QA_PART_SHARD_DIR"
run_qa_pool() {
    local gpu="$1" kind="$2" shard="$3"
    local output manifest annotations paco_annotations images source seed log
    if [[ "$kind" == "stuff" ]]; then
        output="$QA_STUFF_SHARD_DIR/shard_${shard}.parquet"
        manifest="$QA_STUFF_SHARD_DIR/shard_${shard}.jsonl"
        annotations="$BASE_DIR/datasets/dam_data/COCOStuff/annotations.json"
        paco_annotations=""
        images="$BASE_DIR/coco2017/train2017"
        source="cocostuff_cycle"
        seed=$((20260911 + shard * 23))
    elif [[ "$kind" == "part" ]]; then
        output="$QA_PART_SHARD_DIR/shard_${shard}.parquet"
        manifest="$QA_PART_SHARD_DIR/shard_${shard}.jsonl"
        annotations="$BASE_DIR/datasets/dam_data/PACO/annotations.json"
        paco_annotations="$BASE_DIR/PACO-LVIS/annotations/paco_lvis_v1_train.json"
        images="$BASE_DIR/coco2017/train2017"
        source="paco_part_cycle"
        seed=$((20260929 + shard * 29))
    else
        log_line "[error] unknown QA pool kind: $kind" >&2
        return 2
    fi
    if [[ -s "$output" && -s "$manifest" ]]; then
        log_line "[stage:qa-pools] reusing QA $kind shard $shard: $output"
        log_parquet_rows "qa_${kind}_shard_${shard}" "$output"
        log_jsonl_rows "qa_${kind}_manifest_shard_${shard}" "$manifest"
        return
    fi
    log="${output%.parquet}.log"
    if [[ "$kind" == "stuff" ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO_DIR" "$PY" \
            projects/rl/datasets/prepare_dam_cycle_dataset.py \
            --dam-annotations "$annotations" --images-dir "$images" --output "$output" \
            --caption-manifest "$manifest" --source "$source" \
            --mask-tokenizer-path "$MODEL_PATH/mask_tokenizer_256x2.pth" \
            --sam2-checkpoint "$MODEL_PATH/sam2.1_hiera_large.pt" \
            --sam2-config-dir "$REPO_DIR/projects/transformers/vq_sam2/sam2/sam2_configs" \
            --max-samples 2500 --seed "$seed" --device cuda > >(tee -a "$RUN_LOG" "$log" >/dev/null) 2>&1 &
    else
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO_DIR" "$PY" \
            projects/rl/datasets/prepare_dam_cycle_dataset.py \
            --dam-annotations "$annotations" --paco-annotations "$paco_annotations" \
            --images-dir "$images" --output "$output" --caption-manifest "$manifest" \
            --source "$source" --mask-tokenizer-path "$MODEL_PATH/mask_tokenizer_256x2.pth" \
            --sam2-checkpoint "$MODEL_PATH/sam2.1_hiera_large.pt" \
            --sam2-config-dir "$REPO_DIR/projects/transformers/vq_sam2/sam2/sam2_configs" \
            --max-samples 2500 --seed "$seed" --device cuda > >(tee -a "$RUN_LOG" "$log" >/dev/null) 2>&1 &
    fi
    QA_POOL_PIDS+=("$!")
    log_line "[stage:qa-pools] launched kind=$kind shard=$shard gpu=$gpu pid=$! output=$output manifest=$manifest"
}
CURRENT_STAGE="qa-pools"
log_line "[stage:qa-pools] start 4 Stuff + 4 PACO DAM shards; each target=2500"
run_qa_pool 0 stuff 0
run_qa_pool 1 stuff 1
run_qa_pool 2 stuff 2
run_qa_pool 3 stuff 3
run_qa_pool 4 part 0
run_qa_pool 5 part 1
run_qa_pool 6 part 2
run_qa_pool 7 part 3
wait_for_jobs qa-pools 8 "${QA_POOL_PIDS[@]}"
log_line "[stage:qa-pools] complete; merging eight QA shards"

merge_qa_pool() {
    local output="$1" manifest="$2" shard_dir="$3"
    if [[ -s "$output" && -s "$manifest" ]]; then
        return
    fi
    "$PY" - "$output" "$manifest" "$shard_dir" <<PY
import sys
from datasets import Dataset, concatenate_datasets
out, manifest, shard_dir = sys.argv[1:]
parquet_paths = [f"{shard_dir}/shard_{i}.parquet" for i in range(4)]
manifest_paths = [f"{shard_dir}/shard_{i}.jsonl" for i in range(4)]
concatenate_datasets([Dataset.from_parquet(path) for path in parquet_paths]).to_parquet(out)
with open(manifest, "w", encoding="utf-8") as target:
    for path in manifest_paths:
        with open(path, encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    target.write(line)
PY
}
CURRENT_STAGE="qa-merge"
log_line "[stage:qa-merge] start"
merge_qa_pool "$QA_STUFF_POOL" "$QA_STUFF_MANIFEST" "$QA_STUFF_SHARD_DIR"
merge_qa_pool "$QA_PART_POOL" "$QA_PART_MANIFEST" "$QA_PART_SHARD_DIR"
log_line "[stage:qa-merge] complete"
log_parquet_rows "qa_stuff_pool" "$QA_STUFF_POOL"
log_parquet_rows "qa_part_pool" "$QA_PART_POOL"
log_jsonl_rows "qa_stuff_manifest" "$QA_STUFF_MANIFEST"
log_jsonl_rows "qa_part_manifest" "$QA_PART_MANIFEST"

CURRENT_STAGE="selection"
log_line "[stage:selection] start global 100k identity selection"
log_line "[stage:selection] QA pool=combined Stuff+Part total=20000; identical DAM records are retained with deterministic IDs, conflicting records fail"
"$PY" "$DATA_HELPER" \
    --refcoco "$REFCOCO_CANDIDATE" --grefcoco-multi "$GREFCOCO_CANDIDATE" --grefcoco-multi "$GREFCOCO_CYCLE" \
    --no-target "$NO_TARGET_CANDIDATE" \
    --stuff "$POOL_ROOT/stuff_shard_0.parquet" --stuff "$POOL_ROOT/stuff_shard_1.parquet" --stuff "$POOL_ROOT/stuff_shard_2.parquet" --stuff "$POOL_ROOT/stuff_shard_3.parquet" \
    --part "$POOL_ROOT/part_shard_0.parquet" --part "$POOL_ROOT/part_shard_1.parquet" --part "$POOL_ROOT/part_shard_2.parquet" --part "$POOL_ROOT/part_shard_3.parquet" \
    --qa-stuff "$QA_STUFF_POOL" --qa-part "$QA_PART_POOL" \
    --qa-stuff-manifest "$QA_STUFF_MANIFEST" --qa-part-manifest "$QA_PART_MANIFEST" \
    --qa-duplicate-id-policy uniquify \
    --output-dir "$DATA_ROOT" > >(tee -a "$RUN_LOG" "$DATA_ROOT/selection.log" >/dev/null) 2>&1
log_line "[stage:selection] complete; manifest=$DATA_ROOT/scaling_100k_manifest.json"
log_parquet_rows "cyclegrpo_selfsupervised_40k" "$DATA_ROOT/cyclegrpo_selfsupervised_40k.parquet"
log_parquet_rows "direct_supervised_40k" "$DATA_ROOT/direct_supervised_40k.parquet"
log_parquet_rows "direct_supervised_positive_32k" "$DATA_ROOT/direct_supervised_positive_32k.parquet"
log_parquet_rows "direct_supervised_no_target_8k" "$DATA_ROOT/direct_supervised_no_target_8k.parquet"
log_parquet_rows "dlc_qa_20k" "$DATA_ROOT/dlc_qa_20k.parquet"
log_jsonl_rows "dam_caption_manifest_20k" "$DATA_ROOT/dam_caption_manifest_20k.jsonl"

# Llama-3.1-8B generates and independently validates every accepted QA row.
# The runner owns the service on QA_LLAMA_GPU for this stage and stops it
# before Ray training. A complete output is reused; otherwise accepted rows
# are kept and only missing IDs are retried with the generator --resume mode.
stop_qa_service() {
    if [[ -n "${QA_PID:-}" ]] && kill -0 "$QA_PID" 2>/dev/null; then
        kill -- -"$QA_PID" 2>/dev/null || kill "$QA_PID" 2>/dev/null || true
        wait "$QA_PID" 2>/dev/null || true
    fi
    QA_PID=""
    QA_SERVICE_OWNED=false
}
stop_qa_service_and_hold() {
    stop_qa_service
    start_final_hold
}
CURRENT_STAGE="qa-generation"
log_line "[stage:qa-generation] checking reusable output"
if [[ -s "$DATA_ROOT/dam_caption_qa_20k.jsonl" && -f "$DATA_ROOT/dam_caption_qa_20k.rejected.jsonl" && -s "$DATA_ROOT/dlc_qa_20k.json" ]] && [[ "$(awk "NF{n++} END{print n+0}" "$DATA_ROOT/dam_caption_qa_20k.jsonl")" -eq 20000 ]] && [[ "$(awk "NF{n++} END{print n+0}" "$DATA_ROOT/dam_caption_qa_20k.rejected.jsonl")" -eq 0 ]]; then
    log_line "[stage:qa-generation] reusing validated 20k QA output"
    log_jsonl_rows "dam_caption_qa_20k.reused" "$DATA_ROOT/dam_caption_qa_20k.jsonl"
else
    log_line "[stage:qa-generation] starting runner-owned Llama-3.1-8B vLLM on GPU $QA_LLAMA_GPU"
    setsid env CUDA_VISIBLE_DEVICES="$QA_LLAMA_GPU" TOKENIZERS_PARALLELISM=true "$PY" -m vllm.entrypoints.openai.api_server \
        --model "$QA_LLAMA_MODEL_PATH" --served-model-name "$QA_MODEL_NAME" --host 127.0.0.1 --port "$QA_PORT" \
        --api-key "$QA_API_KEY" --tensor-parallel-size 1 --pipeline-parallel-size 1 --gpu-memory-utilization 0.85 --dtype bfloat16 \
        --max-model-len 4096 --trust-remote-code --chat-template "$QA_LLAMA_TEMPLATE" \
        > >(tee -a "$RUN_LOG" "$QA_LLAMA_LOG" >/dev/null) 2>&1 &
    QA_PID=$!
    QA_SERVICE_OWNED=true
    trap stop_qa_service_and_hold EXIT INT TERM
    deadline=$((SECONDS + 600))
    until curl -fsS -H "Authorization: Bearer $QA_API_KEY" "$QA_BASE_URL/models" >/dev/null 2>&1; do
        (( SECONDS < deadline )) || { echo "runner-owned Llama-3.1-8B did not become healthy; inspect $QA_LLAMA_LOG" >&2; exit 1; }
        sleep 5
    done
    log_line "[stage:qa-generation]  Llama service healthy; entering incremental export loop"
    qa_attempt=0
    while [[ -e "$DATA_ROOT/dam_caption_qa_20k.rejected.attempt_$((qa_attempt + 1)).jsonl" ]]; do
        qa_attempt=$((qa_attempt + 1))
    done
    while true; do
        qa_attempt=$((qa_attempt + 1))
        qa_rejected="$DATA_ROOT/dam_caption_qa_20k.rejected.attempt_${qa_attempt}.jsonl"
        if [[ -s "$DATA_ROOT/dam_caption_qa_20k.jsonl" ]]; then
            qa_resume_args=(--resume --overwrite)
        else
            qa_resume_args=(--overwrite)
        fi
        # The generator exits non-zero when an attempt leaves rejected rows.
        # Keep accepted rows and continue with --resume for missing IDs.
        (
            "$PY" projects/rl/datasets/generate_dam_caption_qa.py \
            --input-manifest "$DATA_ROOT/dam_caption_manifest_20k.jsonl" \
            --output "$DATA_ROOT/dam_caption_qa_20k.jsonl" \
            --rejected-output "$qa_rejected" \
            --dlc-qa-output "$DATA_ROOT/dlc_qa_20k.json" \
            --class-names-output "$DATA_ROOT/dlc_class_names_20k.json" \
            --base-url "$QA_BASE_URL" --api-key "$QA_API_KEY" --model "$QA_MODEL_NAME" \
            --validator-model "$QA_MODEL_NAME" --max-samples 20000 --max-concurrency 32 \
            --generation-attempts 3 --request-retries 3 --timeout 180 --temperature 0.2 --max-tokens 900 \
            "${qa_resume_args[@]}" > >(tee -a "$RUN_LOG" "$QA_GENERATION_LOG" >/dev/null) 2>&1
        ) &
        qa_export_pid=$!
        qa_watch_started=$SECONDS
        while kill -0 "$qa_export_pid" 2>/dev/null; do
            accepted_progress=0
            rejected_progress=0
            if [[ -f "$DATA_ROOT/dam_caption_qa_20k.jsonl" ]]; then
                accepted_progress=$(awk 'NF {n++} END {print n+0}' "$DATA_ROOT/dam_caption_qa_20k.jsonl")
            fi
            if [[ -f "$qa_rejected" ]]; then
                rejected_progress=$(awk 'NF {n++} END {print n+0}' "$qa_rejected")
            fi
            log_line "[progress:qa-export] attempt=$qa_attempt accepted=$accepted_progress/20000 rejected=$rejected_progress elapsed=$((SECONDS-qa_watch_started))s"
            sleep 30
        done
        if wait "$qa_export_pid"; then
            qa_status=0
        else
            qa_status=$?
            log_line "[progress:qa-export] attempt=$qa_attempt exit_status=$qa_status; retaining accepted rows and retrying missing IDs."
        fi
        accepted_lines=0
        if [[ -f "$DATA_ROOT/dam_caption_qa_20k.jsonl" ]]; then
            accepted_lines=$(awk "NF{n++} END{print n+0}" "$DATA_ROOT/dam_caption_qa_20k.jsonl")
        fi
        if [[ "$accepted_lines" -eq 20000 ]]; then
            : > "$DATA_ROOT/dam_caption_qa_20k.rejected.jsonl"
            log_line "[progress:qa-export] accepted=20000/20000 after $qa_attempt attempt(s)."
            break
        fi
        missing=$((20000 - accepted_lines))
        log_line "[progress:qa-export] attempt=$qa_attempt retained=$accepted_lines/20000 retrying_missing=$missing"
    done
    kill -- -"$QA_PID" 2>/dev/null || kill "$QA_PID" 2>/dev/null || true
    wait "$QA_PID" 2>/dev/null || true
    QA_PID=""
    log_line "[stage:qa-generation] Llama service stopped; moving to Ray training setup"
    trap start_final_hold EXIT INT TERM
fi
log_line "[stage:qa-generation] complete; accepted QA output ready"
log_jsonl_rows "dam_caption_qa_20k" "$DATA_ROOT/dam_caption_qa_20k.jsonl"
CURRENT_STAGE="qa-join-validation"
log_line "[stage:qa-join-validation] start"
log_json_object_keys "dlc_qa_20k_json" "$DATA_ROOT/dlc_qa_20k.json"
log_jsonl_rows "dam_caption_qa_20k.rejected" "$DATA_ROOT/dam_caption_qa_20k.rejected.jsonl"


# The generated QA sidecar must exactly cover the 20k parquet IDs.
"$PY" - "$DATA_ROOT/dlc_qa_20k.parquet" "$DATA_ROOT/dam_caption_qa_20k.jsonl" <<'PY'
import json, sys
from datasets import Dataset
parquet, qa_jsonl = sys.argv[1:]
ids = [row.get("dam_source_id") for row in Dataset.from_parquet(parquet)]
qa = [json.loads(line)["dam_source_id"] for line in open(qa_jsonl, encoding="utf-8") if line.strip()]
if len(ids) != 20000 or len(qa) != 20000 or set(ids) != set(qa) or len(set(ids)) != 20000:
    raise SystemExit(f"DLC-QA join mismatch: parquet={len(ids)} unique={len(set(ids))}, qa={len(qa)} unique={len(set(qa))}")
print("DLC-QA join verified: 20000 unique IDs")
PY
log_line "[stage:qa-join-validation] complete; one-to-one dam_source_id join verified"

CURRENT_STAGE="training-configuration"
log_line "[stage:training-configuration] start"
# Use the original 70k settings, changing only the three data paths and the
# main/QA one-pass length (40k / 112 ~= 357 steps). GPU 7 is the local Llama
# judge; GPUs 0-6 are registered with Ray.
export BASE_DIR REPO_DIR ENV_DIR MODEL_PATH
export TRAIN_DATA="$DATA_ROOT/cyclegrpo_selfsupervised_40k.parquet"
export VAL_DATA="$TRAIN_DATA"
export DIRECT_TRAIN_DATA="$DATA_ROOT/direct_supervised_positive_32k.parquet"
export DIRECT_NO_TARGET_TRAIN_DATA="$DATA_ROOT/direct_supervised_no_target_8k.parquet"
export CAPTION_QA_TRAIN_DATA="$DATA_ROOT/dlc_qa_20k.parquet"
export CAPTION_QA_JSONL="$DATA_ROOT/dam_caption_qa_20k.jsonl"
export CUDA_VISIBLE_DEVICES="$TRAIN_GPU_LIST"
export RAY_ADDRESS RAY_NAMESPACE RAY_SHORT_ROOT
export MULTINODE_ENABLED=true LOCAL_JUDGE_ENABLED=true NNODES=1 NUM_GPUS=7
export RAY_CLUSTER_EXPECTED_NODES=1 RAY_CLUSTER_EXPECTED_GPUS=7
export OPSD_ENABLED=true PIXEL_IOU_ENABLED=true SECA_ENABLED=true
export NO_TARGET_REWARD_MODE=pixel_empty POSITIVE_EMPTY_MASK_PENALTY=1.0 NO_TARGET_NONEMPTY_MASK_PENALTY=0.0
export ROUTING_ENABLED=true PRESERVE_ORIGINAL_GRPO=true EMA_TEACHER_ENABLED=true TEACHER_EMA_DECAY=1.0
export TEACHER_ANALYSIS_ENABLED=true TEACHER_CONFIDENCE_ENABLED=true CAPTION_SAFETY_ENABLED=true CAPTION_SAFETY_FORCE_REGENERATE=true
export CAPTION_BLOCK_SPECIAL_TOKEN_VOCAB=true CAPTION_ANCHOR_KL_COEF=0 SEGMENTATION_ANCHOR_KL_COEF=0 ASYMMETRIC_GRADIENT_PROJECTION=false
export DIRECT_GROUNDING_ENABLED=true DIRECT_GROUNDING_ROLLOUTS=6 DIRECT_GROUNDING_LOSS_WEIGHT=0.15
export DIRECT_GROUNDING_WARMUP_START_STEP=10 DIRECT_GROUNDING_WARMUP_END_STEP=30
export DIRECT_GROUNDING_INCLUDE_POSITIVE_SOURCES=true DIRECT_GROUNDING_INCLUDE_NO_TARGET=true DIRECT_GROUNDING_INCLUDE_LABEL_SOURCES=true
export DIRECT_GROUNDING_CONSUME_NO_TARGET_CAPTION=false
export DIRECT_MASK_CE_ENABLED=true DIRECT_MASK_CE_LOSS_WEIGHT=0.005 DIRECT_MASK_CE_INCLUDE_NO_TARGET=true
export DIRECT_MASK_CE_WARMUP_START_STEP=10 DIRECT_MASK_CE_WARMUP_END_STEP=30
export SUPERVISED_CAPTION_QA_ENABLED=true CAPTION_QA_BATCH_SIZE=56 CAPTION_QA_MAX_CONCURRENCY=16 CAPTION_QA_TIMEOUT_SECONDS=90
export CAPTION_QA_REWARD_WEIGHT=1.0 CAPTION_QA_LOSS_WEIGHT=1.0 CAPTION_QA_JUDGE_BASE_URL=http://127.0.0.1:18007/v1
export CAPTION_QA_JUDGE_MODEL=llama3.1-8b CAPTION_QA_JUDGE_API_KEY=EMPTY
export ROLLOUT_BATCH_SIZE=112 ACTOR_GLOBAL_BATCH_SIZE=112 DIRECT_BATCH_SIZE=224 CAPTION_ROLLOUTS=6 LOCALIZATION_ROLLOUTS=6
export TOTAL_EPOCHS=1 MAX_STEPS=357 SAVE_FREQ=25 SAVE_LIMIT=1 RESUME=true TRAINER_LOGGERS='["file"]'
export RUN_NAME RUN_ROOT CHECKPOINT_DIR="$RUN_ROOT/checkpoints"
log_line "[stage:training-configuration] complete; train_data=$TRAIN_DATA direct_positive=$DIRECT_TRAIN_DATA direct_no_target=$DIRECT_NO_TARGET_TRAIN_DATA qa_data=$CAPTION_QA_TRAIN_DATA batches=112/224/56 max_steps=$MAX_STEPS ray_gpus=$NUM_GPUS judge_gpu=7"
log_line "[config:storage] checkpoint_save_freq=$SAVE_FREQ checkpoint_save_limit=$SAVE_LIMIT; one FSDP checkpoint is approximately 69 GiB"

CURRENT_STAGE="ray"
log_line "[stage:ray] stopping stale local Ray head"
"$ENV_DIR/bin/ray" stop --force > >(tee -a "$RUN_LOG" "$RUN_ROOT/ray_stop.log" >/dev/null) 2>&1 || true
log_line "[stage:ray] starting local Ray head with 7 training GPUs"
"$ENV_DIR/bin/ray" start --head --node-ip-address=127.0.0.1 --port=6379 --num-gpus=7 \
    --dashboard-host=127.0.0.1 --temp-dir="$RAY_SHORT_ROOT" --disable-usage-stats
log_line "[stage:ray] local Ray head ready"

cleanup() {
    local exit_status=$?
    CURRENT_STAGE="cleanup"
    log_line "[stage:cleanup] stopping judge/Ray and starting eight-GPU hold; prior_exit_status=$exit_status"
    stop_training_watchers
    if [[ -n "${TRAIN_PID:-}" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        log_line "[stage:cleanup] terminating interrupted training pid=$TRAIN_PID"
        kill "$TRAIN_PID" 2>/dev/null || true
        wait "$TRAIN_PID" 2>/dev/null || true
    fi
    "$PY" "$REPO_DIR/tools/multinode/local_llama_judge.py" stop \
        --ray-address "$RAY_ADDRESS" --namespace "$RAY_NAMESPACE" --expected-nodes 1 --env-dir "$ENV_DIR" \
        --model-path "$BASE_DIR/models/Meta-Llama-3.1-8B-Instruct-hf-v2" --served-model-name llama3.1-8b \
        --port 18007 --gpu-device 7 --gpu-memory-utilization 0.85 \
        --chat-template "$REPO_DIR/tools/multinode/llama3_chat_template.jinja" \
        --log-dir "$REPO_DIR/logs/llama_judge_${RUN_NAME}" --name-prefix "${RUN_NAME}-llama" --timeout-seconds "$JUDGE_TIMEOUT_SECONDS" || true
    "$ENV_DIR/bin/ray" stop --force || true
    STATE_DIR="${FINAL_HOLD_STATE_DIR:-/tmp/cyclegrpo_gpu_power_hold_${RUN_NAME}}" \
        GPU_LIST=0,1,2,3,4,5,6,7 MEMORY_MIB=20000 MATMUL_DIM=4096 PYTHON_BIN="$PY" \
        bash "$REPO_DIR/tools/gpu_power_hold.sh" start || true
    log_line "[stage:cleanup] judge/Ray stopped; eight-GPU hold is active; preserving_exit_status=$exit_status"
    log_line "[stage:exit] status=$exit_status elapsed=${SECONDS}s run_log=$RUN_LOG"
    return "$exit_status"
}
trap cleanup EXIT INT TERM

CURRENT_STAGE="judge"
log_line "[stage:judge] starting local Llama-3.1-8B judge on GPU 7"
"$PY" "$REPO_DIR/tools/multinode/local_llama_judge.py" start \
    --ray-address "$RAY_ADDRESS" --namespace "$RAY_NAMESPACE" --expected-nodes 1 --env-dir "$ENV_DIR" \
    --model-path "$BASE_DIR/models/Meta-Llama-3.1-8B-Instruct-hf-v2" --served-model-name llama3.1-8b \
    --port 18007 --gpu-device 7 --gpu-memory-utilization 0.85 \
    --chat-template "$REPO_DIR/tools/multinode/llama3_chat_template.jinja" \
    --log-dir "$REPO_DIR/logs/llama_judge_${RUN_NAME}" --name-prefix "${RUN_NAME}-llama" --timeout-seconds "$JUDGE_TIMEOUT_SECONDS"
log_line "[stage:judge] local Llama-3.1-8B judge ready"

CURRENT_STAGE="train"
log_line "[stage:train] starting 1-epoch 100k Ray training; progress is in $TRAIN_LOG and $RUN_LOG"
(
    # The training entry redirects its stdout to RUN_LOG. Point that variable
    # at the inherited pipe so this parent tee remains the single log writer.
    RUN_LOG=/dev/stdout bash "$TRAIN_ENTRY" > >(tee -a "$RUN_LOG" "$TRAIN_LOG" >/dev/null) 2>&1
) &
TRAIN_PID=$!
start_training_watchers "$TRAIN_PID"
if wait "$TRAIN_PID"; then
    train_status=0
else
    train_status=$?
fi
stop_training_watchers
if (( train_status != 0 )); then
    log_line "[stage:train] training failed with exit_status=$train_status; inspect $TRAIN_LOG and $RUN_LOG" >&2
    exit "$train_status"
fi
TRAIN_PID=""
log_line "[stage:train] training command exited successfully; final_log_bytes=$(wc -c < "$TRAIN_LOG" 2>/dev/null || printf 0)"
CURRENT_STAGE="done"
log_line "[stage:done] all requested stages completed; cleanup will retain GPU hold"
