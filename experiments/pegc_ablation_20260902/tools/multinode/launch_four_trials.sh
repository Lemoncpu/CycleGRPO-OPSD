#!/usr/bin/env bash
# Submit four isolated trials to externally managed Ray clusters.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
INVENTORY="${INVENTORY:-${SCRIPT_DIR}/clusters.tsv.example}"
RAY_READY_TIMEOUT_SECONDS="${RAY_READY_TIMEOUT_SECONDS:-120}"
DRY_RUN=false

usage() {
    cat <<'EOF'
Usage:
  tools/multinode/launch_four_trials.sh launch [--dry-run]
  tools/multinode/launch_four_trials.sh status [--dry-run]
  tools/multinode/launch_four_trials.sh stop [--dry-run]

Required environment variables / files:
  INVENTORY=/absolute/path/to/ray_trials.tsv

Each non-comment inventory row must contain exactly four tab-separated columns:
  trial_id  ray_address  ray_namespace  experiment_env

Ray head/worker processes and NCCL networking are provisioned by the platform;
this controller only verifies the cluster and launches the trainer locally.
EOF
}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

quote() {
    printf '%q' "$1"
}

require_integer() {
    local name="$1"
    local value="$2"
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || fail "${name} must be a positive integer: ${value}"
}

resolve_env_file() {
    local env_file="$1"
    if [[ "${env_file}" != /* ]]; then
        env_file="${REPO_DIR}/${env_file}"
    fi
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

validate_trial_contract() {
    local env_file="$1"
    local contract
    contract="$(
        source "${env_file}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${NUM_GPUS:-}" "${LOCAL_JUDGE_ENABLED:-}" "${ROLLOUT_BATCH_SIZE:-}" \
            "${ACTOR_GLOBAL_BATCH_SIZE:-}" "${MAX_STEPS:-}" "${TRAIN_DATA:-}"
    )" || fail "Unable to read experiment env: ${env_file}"
    local trial_gpus judge_enabled rollout_batch actor_batch max_steps train_data
    IFS=$'\t' read -r trial_gpus judge_enabled rollout_batch actor_batch max_steps train_data <<< "${contract}"
    [[ "${trial_gpus}" == "8" ]] || fail "Pure-cycle multi-node env must set NUM_GPUS=8: ${env_file}"
    [[ "${judge_enabled}" == "false" ]] || fail "Pure-cycle multi-node env must set LOCAL_JUDGE_ENABLED=false: ${env_file}"
    require_integer "ROLLOUT_BATCH_SIZE in ${env_file}" "${rollout_batch}"
    require_integer "ACTOR_GLOBAL_BATCH_SIZE in ${env_file}" "${actor_batch}"
    (( rollout_batch % 16 == 0 )) || fail "ROLLOUT_BATCH_SIZE must be divisible by the 16-GPU Ray world size: ${env_file}"
    (( actor_batch > 0 )) || fail "ACTOR_GLOBAL_BATCH_SIZE must be positive: ${env_file}"
    [[ -n "${max_steps}" ]] || fail "MAX_STEPS must be set for the four-trial comparison: ${env_file}"
    [[ -n "${train_data}" ]] || fail "TRAIN_DATA is missing: ${env_file}"
}

verify_ray_cluster() {
    local ray_address="$1" env_file="$2" expected_gpus="$3"
    local python_bin
    # shellcheck disable=SC1090
    source "${env_file}"
    python_bin="${ENV_DIR}/bin/python3"
    if [[ "${DRY_RUN}" == "true" ]]; then
        printf '%s\n' "${python_bin} -c 'verify Ray ${ray_address}: 2 nodes / ${expected_gpus} GPUs'"
        return 0
    fi
    [[ -x "${python_bin}" ]] || fail "Python executable not found: ${python_bin}"
    "${python_bin}" - "${ray_address}" "${expected_gpus}" "${RAY_READY_TIMEOUT_SECONDS}" <<'PY'
import sys
import time

import ray

address, expected_gpus, timeout_seconds = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
deadline = time.monotonic() + timeout_seconds
last_state = "not connected"

while time.monotonic() < deadline:
    try:
        ray.init(address=address, logging_level="ERROR")
        alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
        gpu_count = sum(float(node.get("Resources", {}).get("GPU", 0)) for node in alive_nodes)
        last_state = f"alive_nodes={len(alive_nodes)} gpus={gpu_count:g}"
        if len(alive_nodes) == 2 and gpu_count >= expected_gpus:
            print(f"ray_address={address} {last_state}")
            ray.shutdown()
            break
        ray.shutdown()
    except Exception as exc:
        last_state = f"{type(exc).__name__}: {exc}"
    time.sleep(2)
else:
    raise SystemExit(
        f"Ray cluster {address} did not reach 2 nodes / {expected_gpus:g} GPUs "
        f"within {timeout_seconds}s; last state: {last_state}"
    )
PY
}

launch_training() {
    local trial_id="$1" ray_address="$2" ray_namespace="$3" env_file="$4" run_root="$5"
    local log_dir="${run_root}/multinode"
    local pid_file="${log_dir}/trainer_launcher.pid"
    local log_file="${log_dir}/trainer_launcher.log"
    if [[ "${DRY_RUN}" == "true" ]]; then
        printf 'set -a; source %q; set +a\n' "${env_file}"
        printf 'export RAY_ADDRESS=%q RAY_NAMESPACE=%q MULTINODE_ENABLED=true NNODES=2 NUM_GPUS=8 RAY_CLUSTER_EXPECTED_NODES=2 RAY_CLUSTER_EXPECTED_GPUS=16\n' \
            "${ray_address}" "${ray_namespace}"
        printf 'nohup bash "$REPO_DIR/projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh" > %q 2>&1 < /dev/null &\n' "${log_file}"
        printf 'echo $! > %q\n' "${pid_file}"
    else
        mkdir -p "${log_dir}"
        (
            # shellcheck disable=SC1090
            set -a
            source "${env_file}"
            set +a
            export RAY_ADDRESS="${ray_address}"
            export RAY_NAMESPACE="${ray_namespace}"
            export MULTINODE_ENABLED=true
            export NNODES=2
            export NUM_GPUS=8
            export RAY_CLUSTER_EXPECTED_NODES=2
            export RAY_CLUSTER_EXPECTED_GPUS=16
            nohup bash "${REPO_DIR}/projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh" \
                > "${log_file}" 2>&1 < /dev/null &
            printf '%s\n' "$!" > "${pid_file}"
        )
        echo "Started ${trial_id}; pid=$(cat "${pid_file}"); log=${log_file}"
    fi
}

trial_status() {
    local index="$1"
    local trial_id="${TRIAL_IDS[index]}" ray_address="${RAY_ADDRESSES[index]}" env_file="${ENV_FILES[index]}" run_root="${RUN_ROOTS[index]}"
    local pid_file="${run_root}/multinode/trainer_launcher.pid"
    printf '== %s ==\n' "${trial_id}"
    if [[ -f "${pid_file}" ]]; then
        local pid
        pid="$(cat "${pid_file}")"
        if kill -0 "${pid}" 2>/dev/null; then
            echo "trainer=running pid=${pid}"
        else
            echo "trainer=stopped pid=${pid}"
        fi
    else
        echo "trainer=pid-file-missing"
    fi
    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "ray=${ray_address} (status query skipped in dry-run)"
    else
        verify_ray_cluster "${ray_address}" "${env_file}" 16 || echo "ray=unavailable"
    fi
}

stop_trial() {
    local index="$1"
    local trial_id="${TRIAL_IDS[index]}" run_root="${RUN_ROOTS[index]}"
    local pid_file="${run_root}/multinode/trainer_launcher.pid"
    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "kill -TERM <${trial_id} trainer pid from ${pid_file}>"
        return 0
    fi
    if [[ -f "${pid_file}" ]]; then
        local pid
        pid="$(cat "${pid_file}")"
        kill -TERM "${pid}" 2>/dev/null || true
        echo "Stopped ${trial_id} trainer pid=${pid}"
    else
        echo "No trainer pid file for ${trial_id}: ${pid_file}"
    fi
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 2
fi

ACTION="$1"
case "${ACTION}" in
    launch|status|stop) ;;
    *) usage; exit 2 ;;
esac

if [[ $# -eq 2 ]]; then
    [[ "$2" == "--dry-run" ]] || fail "Unknown argument: $2"
    DRY_RUN=true
fi

require_integer "RAY_READY_TIMEOUT_SECONDS" "${RAY_READY_TIMEOUT_SECONDS}"
[[ -f "${INVENTORY}" ]] || fail "Inventory not found: ${INVENTORY}"

TRIAL_IDS=()
RAY_ADDRESSES=()
RAY_NAMESPACES=()
ENV_FILES=()
RUN_ROOTS=()

while IFS=$'\t' read -r trial_id ray_address ray_namespace env_file extra; do
    trial_id="${trial_id%$'\r'}"
    [[ -z "${trial_id}" || "${trial_id}" == \#* ]] && continue
    [[ -z "${extra:-}" ]] || fail "Inventory row for ${trial_id} has more than four columns."
    [[ -n "${ray_address}" && -n "${ray_namespace}" && -n "${env_file}" ]] \
        || fail "Inventory row for ${trial_id} is missing Ray address, namespace, or env file."
    [[ "${trial_id}" =~ ^[A-Za-z0-9._-]+$ ]] || fail "Invalid trial_id: ${trial_id}"
    env_file="$(resolve_env_file "${env_file}")"
    [[ -f "${env_file}" ]] || fail "Experiment env file not found: ${env_file}"
    validate_trial_contract "${env_file}"
    run_info="$(load_run_root "${env_file}")" || fail "Env must define RUN_NAME and RUN_ROOT: ${env_file}"
    run_root="${run_info#*$'\t'}"
    for existing_trial in "${TRIAL_IDS[@]}"; do
        [[ "${trial_id}" != "${existing_trial}" ]] || fail "Duplicate trial_id: ${trial_id}"
    done
    for existing_address in "${RAY_ADDRESSES[@]}"; do
        [[ "${ray_address}" != "${existing_address}" ]] || fail "Ray address is reused: ${ray_address}"
    done
    TRIAL_IDS+=("${trial_id}")
    RAY_ADDRESSES+=("${ray_address}")
    RAY_NAMESPACES+=("${ray_namespace}")
    ENV_FILES+=("${env_file}")
    RUN_ROOTS+=("${run_root}")
done < "${INVENTORY}"

(( ${#TRIAL_IDS[@]} == 4 )) || fail "Inventory must declare exactly four trials; found ${#TRIAL_IDS[@]}."

case "${ACTION}" in
    launch)
        for ((index = 0; index < ${#TRIAL_IDS[@]}; index++)); do
            echo "Checking ${TRIAL_IDS[index]} (${RAY_ADDRESSES[index]})"
            verify_ray_cluster "${RAY_ADDRESSES[index]}" "${ENV_FILES[index]}" 16 \
                || fail "Ray cluster verification failed for ${TRIAL_IDS[index]}"
        done
        for ((index = 0; index < ${#TRIAL_IDS[@]}; index++)); do
            launch_training "${TRIAL_IDS[index]}" "${RAY_ADDRESSES[index]}" "${RAY_NAMESPACES[index]}" "${ENV_FILES[index]}" "${RUN_ROOTS[index]}"
        done
        ;;
    status)
        for ((index = 0; index < ${#TRIAL_IDS[@]}; index++)); do trial_status "${index}"; done
        ;;
    stop)
        for ((index = 0; index < ${#TRIAL_IDS[@]}; index++)); do stop_trial "${index}"; done
        ;;
esac
