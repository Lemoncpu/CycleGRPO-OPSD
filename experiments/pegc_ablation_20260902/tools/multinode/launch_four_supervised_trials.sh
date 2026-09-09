#!/usr/bin/env bash
# Launch four isolated mixed-task experiments on platform-provided Ray clusters.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
INVENTORY="${INVENTORY:-${SCRIPT_DIR}/supervised_clusters.tsv.example}"
RAY_READY_TIMEOUT_SECONDS="${RAY_READY_TIMEOUT_SECONDS:-120}"
DRY_RUN=false

usage() {
    cat <<'EOF'
Usage:
  tools/multinode/launch_four_supervised_trials.sh launch [--dry-run]
  tools/multinode/launch_four_supervised_trials.sh status [--dry-run]
  tools/multinode/launch_four_supervised_trials.sh stop [--dry-run]

Inventory rows are five tab-separated fields:
  trial_id  ray_address  ray_namespace  head_judge_base_url  experiment_env

Use '-' for head_judge_base_url when LOCAL_JUDGE_ENABLED=false. The platform
must provide an isolated Ray cluster for each trial; this script does not use
SSH or start/stop Ray.
EOF
}

fail() { echo "ERROR: $*" >&2; exit 1; }

require_integer() {
    local name="$1" value="$2"
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || fail "${name} must be a positive integer: ${value}"
}

resolve_env_file() {
    local env_file="$1"
    [[ "${env_file}" == /* ]] || env_file="${REPO_DIR}/${env_file}"
    printf '%s\n' "${env_file}"
}

load_run_root() {
    local env_file="$1"
    (
        source "${env_file}"
        [[ -n "${RUN_NAME:-}" && -n "${RUN_ROOT:-}" ]] || exit 2
        printf '%s\t%s\n' "${RUN_NAME}" "${RUN_ROOT}"
    )
}

load_contract() {
    local env_file="$1"
    source "${env_file}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${NUM_GPUS:-}" "${NNODES:-1}" "${LOCAL_JUDGE_ENABLED:-false}" \
        "${ROLLOUT_BATCH_SIZE:-}" "${ACTOR_GLOBAL_BATCH_SIZE:-}" "${DIRECT_BATCH_SIZE:-}" \
        "${CAPTION_QA_BATCH_SIZE:-}" "${MAX_STEPS:-}" "${TRAIN_DATA:-}" \
        "${DIRECT_GROUNDING_ENABLED:-false}" "${DIRECT_MASK_CE_ENABLED:-false}" \
        "${SUPERVISED_CAPTION_QA_ENABLED:-false}" "${NO_TARGET_REWARD_MODE:-}"
}

validate_trial_contract() {
    local env_file="$1" contract
    contract="$(load_contract "${env_file}")" || fail "Unable to read experiment env: ${env_file}"
    local gpus nodes judge rollout actor direct qa max_steps train_data direct_enabled ce_enabled qa_enabled reward_mode
    IFS=$'\t' read -r gpus nodes judge rollout actor direct qa max_steps train_data direct_enabled ce_enabled qa_enabled reward_mode <<< "${contract}"
    [[ "${gpus}" == "8" || "${gpus}" == "7" ]] || fail "NUM_GPUS must be 7 or 8: ${env_file}"
    [[ "${nodes}" == "1" || "${nodes}" == "2" ]] || fail "NNODES must be 1 or 2: ${env_file}"
    if [[ "${judge}" == "true" ]]; then
        [[ "${gpus}" == "7" && "${nodes}" == "1" ]] || fail "Judge trial must use NNODES=1 and NUM_GPUS=7: ${env_file}"
        [[ "${qa_enabled}" == "true" ]] || fail "Judge trial must enable DLC-QA: ${env_file}"
    else
        [[ "${gpus}" == "8" ]] || fail "Non-judge trial must use NUM_GPUS=8: ${env_file}"
        [[ "${qa_enabled}" == "false" ]] || fail "DLC-QA requires a judge: ${env_file}"
    fi
    require_integer "ROLLOUT_BATCH_SIZE in ${env_file}" "${rollout}"
    require_integer "ACTOR_GLOBAL_BATCH_SIZE in ${env_file}" "${actor}"
    (( actor > 0 && rollout % actor == 0 )) || fail "ACTOR_GLOBAL_BATCH_SIZE must divide ROLLOUT_BATCH_SIZE: ${env_file}"
    if [[ "${direct_enabled}" == "true" || "${ce_enabled}" == "true" ]]; then
        require_integer "DIRECT_BATCH_SIZE in ${env_file}" "${direct}"
    fi
    if [[ "${qa_enabled}" == "true" ]]; then
        require_integer "CAPTION_QA_BATCH_SIZE in ${env_file}" "${qa}"
        (( direct == 2 * rollout && rollout == 2 * qa )) || fail "Supervised parent batches must be 2:4:1: ${env_file}"
    fi
    require_integer "MAX_STEPS in ${env_file}" "${max_steps}"
    [[ -n "${train_data}" ]] || fail "TRAIN_DATA is missing: ${env_file}"
    [[ "${judge}" == "true" || "${judge}" == "false" ]] || fail "LOCAL_JUDGE_ENABLED must be true/false: ${env_file}"
    [[ "${reward_mode}" == "pixel_empty" || "${reward_mode}" == "official_bbox" ]] || fail "Unsupported no-target reward mode: ${env_file}"
}

verify_ray_cluster() {
    local ray_address="$1" env_file="$2" expected_nodes="$3" expected_gpus="$4" python_bin
    source "${env_file}"
    python_bin="${ENV_DIR}/bin/python3"
    if [[ "${DRY_RUN}" == "true" ]]; then
        printf '%s\n' "${python_bin} -c 'verify Ray ${ray_address}: ${expected_nodes} node(s) / ${expected_gpus} GPUs'"
        return 0
    fi
    [[ -x "${python_bin}" ]] || fail "Python executable not found: ${python_bin}"
    "${python_bin}" - "${ray_address}" "${expected_nodes}" "${expected_gpus}" "${RAY_READY_TIMEOUT_SECONDS}" <<'PY'
import sys
import time
import ray

address, expected_nodes, expected_gpus, timeout_seconds = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
deadline = time.monotonic() + timeout_seconds
last_state = "not connected"
while time.monotonic() < deadline:
    try:
        ray.init(address=address, logging_level="ERROR")
        nodes = [node for node in ray.nodes() if node.get("Alive")]
        gpus = sum(float(node.get("Resources", {}).get("GPU", 0)) for node in nodes)
        last_state = f"alive_nodes={len(nodes)} gpus={gpus:g}"
        ray.shutdown()
        if len(nodes) == expected_nodes and gpus >= expected_gpus:
            print(f"ray_address={address} {last_state}")
            break
    except Exception as exc:
        last_state = f"{type(exc).__name__}: {exc}"
        try:
            ray.shutdown()
        except Exception:
            pass
    time.sleep(2)
else:
    raise SystemExit(f"Ray cluster {address} did not reach {expected_nodes} node(s) / {expected_gpus:g} GPUs within {timeout_seconds}s; last state: {last_state}")
PY
}

judge_command() {
    local action="$1" ray_address="$2" ray_namespace="$3" env_file="$4" log_dir="$5"
    source "${env_file}"
    printf '%q ' "${ENV_DIR}/bin/python3" "${REPO_DIR}/tools/multinode/local_llama_judge.py" "${action}" \
        "--ray-address" "${ray_address}" "--namespace" "${ray_namespace}" "--expected-nodes" "1" \
        "--env-dir" "${ENV_DIR}" "--model-path" "${LOCAL_JUDGE_MODEL_PATH}" \
        "--served-model-name" "${LOCAL_JUDGE_SERVED_MODEL_NAME}" "--port" "${LOCAL_JUDGE_PORT}" \
        "--gpu-device" "${LOCAL_JUDGE_CUDA_DEVICE}" "--gpu-memory-utilization" "${LOCAL_JUDGE_GPU_MEMORY_UTILIZATION}" \
        "--chat-template" "${REPO_DIR}/tools/multinode/llama3_chat_template.jinja" \
        "--log-dir" "${log_dir}" "--name-prefix" "cyclegrpo-local-llama-${RUN_NAME}" \
        "--timeout-seconds" "${RAY_READY_TIMEOUT_SECONDS}"
    printf '\n'
}

run_judges() {
    local action="$1" ray_address="$2" ray_namespace="$3" env_file="$4" log_dir="$5" judge_enabled
    judge_enabled="$(source "${env_file}"; printf '%s' "${LOCAL_JUDGE_ENABLED:-false}")"
    [[ "${judge_enabled}" == "true" ]] || return 0
    local command
    command="$(judge_command "${action}" "${ray_address}" "${ray_namespace}" "${env_file}" "${log_dir}")"
    if [[ "${DRY_RUN}" == "true" ]]; then printf '%s\n' "${command}"; else bash -lc "${command}"; fi
}

verify_head_judge() {
    local judge_url="$1" env_file="$2" judge_enabled
    judge_enabled="$(source "${env_file}"; printf '%s' "${LOCAL_JUDGE_ENABLED:-false}")"
    [[ "${judge_enabled}" == "true" ]] || return 0
    local python_bin served_model_name
    source "${env_file}"
    python_bin="${ENV_DIR}/bin/python3"
    served_model_name="${LOCAL_JUDGE_SERVED_MODEL_NAME}"
    if [[ "${DRY_RUN}" == "true" ]]; then
        printf '%s\n' "${python_bin} -c 'verify ${judge_url}/models includes ${served_model_name}'"
        return 0
    fi
    "${python_bin}" - "${judge_url%/}/models" "${served_model_name}" <<'PY'
import json
import sys
import urllib.request
url, expected_model = sys.argv[1:]
with urllib.request.urlopen(url, timeout=15) as response:
    if not 200 <= response.status < 300:
        raise RuntimeError(f"judge returned HTTP {response.status}: {url}")
    payload = json.load(response)
ids = {str(item.get("id")) for item in payload.get("data", []) if isinstance(item, dict) and item.get("id") is not None}
if expected_model not in ids:
    raise RuntimeError(f"judge {url} does not serve {expected_model!r}; available={sorted(ids)}")
print(f"judge_url={url} served_model={expected_model}")
PY
}

launch_training() {
    local trial_id="$1" ray_address="$2" ray_namespace="$3" judge_url="$4" env_file="$5" run_root="$6"
    local log_dir="${run_root}/multinode"
    local pid_file="${log_dir}/trainer_launcher.pid" log_file="${log_dir}/trainer_launcher.log"
    local contract gpus nodes judge
    contract="$(load_contract "${env_file}")"
    IFS=$'\t' read -r gpus nodes judge _ <<< "${contract}"
    local visible
    visible=""
    for ((gpu=0; gpu<gpus; gpu++)); do
        [[ -z "${visible}" ]] || visible+=","
        visible+="${gpu}"
    done
    if [[ "${DRY_RUN}" == "true" ]]; then
        printf 'set -a; source %q; set +a\n' "${env_file}"
        printf 'export RAY_ADDRESS=%q RAY_NAMESPACE=%q MULTINODE_ENABLED=true NNODES=%q NUM_GPUS=%q RAY_CLUSTER_EXPECTED_NODES=%q RAY_CLUSTER_EXPECTED_GPUS=%q CUDA_VISIBLE_DEVICES=%q\n' \
            "${ray_address}" "${ray_namespace}" "${nodes}" "${gpus}" "${nodes}" "$((gpus * nodes))" "${visible}"
        [[ "${judge}" == "true" ]] && printf 'export CAPTION_QA_JUDGE_BASE_URL=%q\n' "${judge_url}"
        printf 'nohup bash "$REPO_DIR/projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh" > %q 2>&1 < /dev/null &\n' "${log_file}"
        printf 'echo $! > %q\n' "${pid_file}"
        return 0
    fi
    mkdir -p "${log_dir}"
    (
        set -a
        source "${env_file}"
        set +a
        export RAY_ADDRESS="${ray_address}" RAY_NAMESPACE="${ray_namespace}" MULTINODE_ENABLED=true NNODES="${nodes}" NUM_GPUS="${gpus}"
        export RAY_CLUSTER_EXPECTED_NODES="${nodes}" RAY_CLUSTER_EXPECTED_GPUS="$((gpus * nodes))" CUDA_VISIBLE_DEVICES="${visible}"
        [[ "${judge}" == "true" ]] && export CAPTION_QA_JUDGE_BASE_URL="${judge_url}"
        nohup bash "${REPO_DIR}/projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh" > "${log_file}" 2>&1 < /dev/null &
        printf '%s\n' "$!" > "${pid_file}"
    )
    echo "Started ${trial_id}; pid=$(cat "${pid_file}"); log=${log_file}"
}

trial_status() {
    local index="$1" trial_id="${TRIAL_IDS[index]}" ray_address="${RAY_ADDRESSES[index]}" env_file="${ENV_FILES[index]}" run_root="${RUN_ROOTS[index]}"
    local pid_file="${run_root}/multinode/trainer_launcher.pid"
    local contract gpus nodes
    contract="$(load_contract "${env_file}")"; IFS=$'\t' read -r gpus nodes _ <<< "${contract}"
    printf '== %s ==\n' "${trial_id}"
    if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then echo "trainer=running pid=$(cat "${pid_file}")"; else echo "trainer=stopped-or-pid-file-missing"; fi
    verify_ray_cluster "${ray_address}" "${env_file}" "${nodes}" "$((gpus * nodes))" || echo "ray=unavailable"
    run_judges status "${ray_address}" "${RAY_NAMESPACES[index]}" "${env_file}" "${run_root}/multinode"
}

stop_trial() {
    local index="$1" trial_id="${TRIAL_IDS[index]}" env_file="${ENV_FILES[index]}" run_root="${RUN_ROOTS[index]}"
    local pid_file="${run_root}/multinode/trainer_launcher.pid"
    if [[ "${DRY_RUN}" == "true" ]]; then echo "kill -TERM <${trial_id} trainer PID from ${pid_file}>"; else
        if [[ -f "${pid_file}" ]]; then kill -TERM "$(cat "${pid_file}")" 2>/dev/null || true; echo "Stopped ${trial_id} trainer pid=$(cat "${pid_file}")"; else echo "No trainer PID file for ${trial_id}: ${pid_file}"; fi
    fi
    run_judges stop "${RAY_ADDRESSES[index]}" "${RAY_NAMESPACES[index]}" "${env_file}" "${run_root}/multinode"
}

[[ $# -ge 1 && $# -le 2 ]] || { usage; exit 2; }
ACTION="$1"
case "${ACTION}" in launch|status|stop) ;; *) usage; exit 2 ;; esac
if [[ $# -eq 2 ]]; then [[ "$2" == "--dry-run" ]] || fail "Unknown argument: $2"; DRY_RUN=true; fi
require_integer "RAY_READY_TIMEOUT_SECONDS" "${RAY_READY_TIMEOUT_SECONDS}"
[[ -f "${INVENTORY}" ]] || fail "Inventory not found: ${INVENTORY}"

TRIAL_IDS=() RAY_ADDRESSES=() RAY_NAMESPACES=() JUDGE_URLS=() ENV_FILES=() RUN_ROOTS=()
while IFS=$'\t' read -r trial_id ray_address ray_namespace judge_url env_file extra; do
    trial_id="${trial_id%$'\r'}"
    [[ -z "${trial_id}" || "${trial_id}" == \#* ]] && continue
    [[ -z "${extra:-}" ]] || fail "Inventory row for ${trial_id} has more than five columns."
    [[ -n "${ray_address}" && -n "${ray_namespace}" && -n "${judge_url}" && -n "${env_file}" ]] || fail "Incomplete row for ${trial_id}."
    [[ "${trial_id}" =~ ^[A-Za-z0-9._-]+$ ]] || fail "Invalid trial_id: ${trial_id}"
    [[ "${judge_url}" == "-" || "${judge_url}" =~ ^http://[^/]+/v1$ ]] || fail "head_judge_base_url must be '-' or http://host:port/v1: ${judge_url}"
    env_file="$(resolve_env_file "${env_file}")"; [[ -f "${env_file}" ]] || fail "Experiment env file not found: ${env_file}"
    validate_trial_contract "${env_file}"
    judge_enabled="$(source "${env_file}"; printf '%s' "${LOCAL_JUDGE_ENABLED:-false}")"
    if [[ "${judge_enabled}" == "true" && "${judge_url}" == "-" ]]; then fail "Judge-enabled trial ${trial_id} requires a head_judge_base_url."; fi
    if [[ "${judge_enabled}" == "false" && "${judge_url}" != "-" ]]; then fail "Non-judge trial ${trial_id} must use '-' for head_judge_base_url."; fi
    run_info="$(load_run_root "${env_file}")" || fail "Env must define RUN_NAME and RUN_ROOT: ${env_file}"
    run_root="${run_info#*$'\t'}"
    for existing in "${TRIAL_IDS[@]}"; do [[ "${trial_id}" != "${existing}" ]] || fail "Duplicate trial_id: ${trial_id}"; done
    for existing in "${RAY_ADDRESSES[@]}"; do [[ "${ray_address}" != "${existing}" ]] || fail "Ray address is reused: ${ray_address}"; done
    TRIAL_IDS+=("${trial_id}"); RAY_ADDRESSES+=("${ray_address}"); RAY_NAMESPACES+=("${ray_namespace}"); JUDGE_URLS+=("${judge_url}"); ENV_FILES+=("${env_file}"); RUN_ROOTS+=("${run_root}")
done < "${INVENTORY}"
(( ${#TRIAL_IDS[@]} == 4 )) || fail "Inventory must declare exactly four trials; found ${#TRIAL_IDS[@]}."

case "${ACTION}" in
    launch)
        for ((index=0; index<${#TRIAL_IDS[@]}; index++)); do
            contract="$(load_contract "${ENV_FILES[index]}")"; IFS=$'\t' read -r gpus nodes judge _ <<< "${contract}"
            echo "Checking ${TRIAL_IDS[index]} (${RAY_ADDRESSES[index]})"
            verify_ray_cluster "${RAY_ADDRESSES[index]}" "${ENV_FILES[index]}" "${nodes}" "$((gpus * nodes))" || fail "Ray cluster verification failed for ${TRIAL_IDS[index]}"
        done
        for ((index=0; index<${#TRIAL_IDS[@]}; index++)); do
            run_judges start "${RAY_ADDRESSES[index]}" "${RAY_NAMESPACES[index]}" "${ENV_FILES[index]}" "${RUN_ROOTS[index]}/multinode"
            verify_head_judge "${JUDGE_URLS[index]}" "${ENV_FILES[index]}" || fail "Head judge verification failed for ${TRIAL_IDS[index]}"
            launch_training "${TRIAL_IDS[index]}" "${RAY_ADDRESSES[index]}" "${RAY_NAMESPACES[index]}" "${JUDGE_URLS[index]}" "${ENV_FILES[index]}" "${RUN_ROOTS[index]}"
        done
        ;;
    status) for ((index=0; index<${#TRIAL_IDS[@]}; index++)); do trial_status "${index}"; done ;;
    stop) for ((index=0; index<${#TRIAL_IDS[@]}; index++)); do stop_trial "${index}"; done ;;
esac
