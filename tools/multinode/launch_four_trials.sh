#!/usr/bin/env bash
# Launch four isolated two-node CycleGRPO Ray clusters over passwordless SSH.
# Each TSV row owns both listed nodes and therefore exactly 16 H20 GPUs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
INVENTORY="${INVENTORY:-${SCRIPT_DIR}/clusters.tsv.example}"
SSH_CONNECT_TIMEOUT_SECONDS="${SSH_CONNECT_TIMEOUT_SECONDS:-10}"
RAY_READY_TIMEOUT_SECONDS="${RAY_READY_TIMEOUT_SECONDS:-120}"
CLEAN_RAY="${CLEAN_RAY:-false}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
DRY_RUN=false

usage() {
    cat <<'EOF'
Usage:
  tools/multinode/launch_four_trials.sh launch [--dry-run]
  tools/multinode/launch_four_trials.sh status [--dry-run]
  tools/multinode/launch_four_trials.sh stop [--dry-run]

Required environment variables / files:
  INVENTORY=/absolute/path/to/clusters.tsv
  CLEAN_RAY=true                 Allow `ray stop --force` on the eight listed nodes before launch.

The inventory must contain exactly four tab-separated rows. Copy
clusters.tsv.example, replace every placeholder, and keep its header/comment lines.
EOF
}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

quote() {
    printf '%q' "$1"
}

remote() {
    local host="$1"
    local command="$2"
    local encoded
    encoded="$(quote "${command}")"
    if [[ "${DRY_RUN}" == "true" ]]; then
        printf 'ssh -o BatchMode=yes -o ConnectTimeout=%q %q bash -lc %s\n' \
            "${SSH_CONNECT_TIMEOUT_SECONDS}" "${host}" "${encoded}"
        return 0
    fi
    ssh -o BatchMode=yes -o "ConnectTimeout=${SSH_CONNECT_TIMEOUT_SECONDS}" \
        "${host}" "bash -lc ${encoded}"
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
        # shellcheck disable=SC1090
        source "${env_file}"
        if [[ -z "${RUN_NAME:-}" || -z "${RUN_ROOT:-}" ]]; then
            exit 2
        fi
        printf '%s\t%s\n' "${RUN_NAME}" "${RUN_ROOT}"
    )
}

require_integer() {
    local name="$1"
    local value="$2"
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || fail "${name} must be a positive integer: ${value}"
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 2
fi

ACTION="$1"
case "${ACTION}" in
    launch|status|stop) ;;
    *)
        usage
        exit 2
        ;;
esac

if [[ $# -eq 2 ]]; then
    [[ "$2" == "--dry-run" ]] || fail "Unknown argument: $2"
    DRY_RUN=true
fi

[[ "${CLEAN_RAY}" == "true" || "${CLEAN_RAY}" == "false" ]] \
    || fail "CLEAN_RAY must be true or false: ${CLEAN_RAY}"
require_integer "SSH_CONNECT_TIMEOUT_SECONDS" "${SSH_CONNECT_TIMEOUT_SECONDS}"
require_integer "RAY_READY_TIMEOUT_SECONDS" "${RAY_READY_TIMEOUT_SECONDS}"
[[ -f "${INVENTORY}" ]] || fail "Inventory not found: ${INVENTORY}"

TRIAL_IDS=()
HEAD_HOSTS=()
HEAD_IPS=()
WORKER_HOSTS=()
WORKER_IPS=()
RAY_PORTS=()
DASHBOARD_PORTS=()
NCCL_INTERFACES=()
ENV_FILES=()
RUN_NAMES=()
RUN_ROOTS=()
ALL_HOSTS=()

while IFS=$'\t' read -r trial_id head_host head_ip worker_host worker_ip ray_port dashboard_port nccl_interface env_file extra; do
    trial_id="${trial_id%$'\r'}"
    [[ -z "${trial_id}" || "${trial_id}" == \#* ]] && continue
    [[ -z "${extra:-}" ]] || fail "Inventory row for ${trial_id} has more than nine columns."
    [[ -n "${head_host}" && -n "${head_ip}" && -n "${worker_host}" && -n "${worker_ip}" ]] \
        || fail "Inventory row for ${trial_id} is missing a host or IP."
    [[ -n "${nccl_interface}" && -n "${env_file}" ]] \
        || fail "Inventory row for ${trial_id} is missing NCCL interface or env file."
    [[ "${trial_id}" =~ ^[A-Za-z0-9._-]+$ ]] \
        || fail "trial_id must only contain letters, digits, dot, underscore, or dash: ${trial_id}"
    require_integer "ray_port for ${trial_id}" "${ray_port}"
    require_integer "dashboard_port for ${trial_id}" "${dashboard_port}"
    (( ray_port <= 65535 && dashboard_port <= 65535 )) \
        || fail "Ports for ${trial_id} must be <= 65535."

    env_file="$(resolve_env_file "${env_file}")"
    [[ -f "${env_file}" ]] || fail "Experiment env file not found for ${trial_id}: ${env_file}"
    run_info="$(load_run_root "${env_file}")" \
        || fail "Experiment env file must define RUN_NAME and RUN_ROOT: ${env_file}"
    run_name="${run_info%%$'\t'*}"
    run_root="${run_info#*$'\t'}"

    for existing_trial in "${TRIAL_IDS[@]}"; do
        [[ "${trial_id}" != "${existing_trial}" ]] || fail "Duplicate trial_id: ${trial_id}"
    done
    for existing_host in "${ALL_HOSTS[@]}"; do
        [[ "${head_host}" != "${existing_host}" ]] || fail "Host ${head_host} is assigned to more than one trial."
        [[ "${worker_host}" != "${existing_host}" ]] || fail "Host ${worker_host} is assigned to more than one trial."
    done

    head_tmp="/dev/shm/cgrpo-${trial_id}-h"
    worker_tmp="/dev/shm/cgrpo-${trial_id}-w"
    (( ${#head_tmp} <= 32 && ${#worker_tmp} <= 32 )) \
        || fail "trial_id is too long for the required short Ray temp path: ${trial_id}"

    TRIAL_IDS+=("${trial_id}")
    HEAD_HOSTS+=("${head_host}")
    HEAD_IPS+=("${head_ip}")
    WORKER_HOSTS+=("${worker_host}")
    WORKER_IPS+=("${worker_ip}")
    RAY_PORTS+=("${ray_port}")
    DASHBOARD_PORTS+=("${dashboard_port}")
    NCCL_INTERFACES+=("${nccl_interface}")
    ENV_FILES+=("${env_file}")
    RUN_NAMES+=("${run_name}")
    RUN_ROOTS+=("${run_root}")
    ALL_HOSTS+=("${head_host}" "${worker_host}")
done < "${INVENTORY}"

(( ${#TRIAL_IDS[@]} == 4 )) \
    || fail "Inventory must declare exactly four trials; found ${#TRIAL_IDS[@]}."

remote_preflight() {
    local host="$1"
    local env_file="$2"
    local role="$3"
    local remote_command
    printf -v remote_command 'source %q && test -d "$REPO_DIR" && test -x "$ENV_DIR/bin/python3" && test -x "$ENV_DIR/bin/ray" && test -f "$TRAIN_DATA" && test -f "$MODEL_PATH/config.json" && test -f "$MODEL_PATH/mask_tokenizer_256x2.pth" && test -f "$MODEL_PATH/sam2.1_hiera_large.pt" && if [ "${DIRECT_GROUNDING_ENABLED}" = true ] || [ "${DIRECT_MASK_CE_ENABLED}" = true ]; then test -f "$DIRECT_TRAIN_DATA"; fi && if [ "${DIRECT_GROUNDING_INCLUDE_NO_TARGET}" = true ] || [ "${DIRECT_MASK_CE_INCLUDE_NO_TARGET}" = true ]; then test -f "$DIRECT_NO_TARGET_TRAIN_DATA"; fi && if [ "${SUPERVISED_CAPTION_QA_ENABLED}" = true ]; then test -f "$CAPTION_QA_TRAIN_DATA" && test -f "$CAPTION_QA_JSONL" && command -v curl >/dev/null && curl --fail --silent --show-error --max-time 15 "${CAPTION_QA_JUDGE_BASE_URL%%/}/models" >/dev/null; fi && GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d " ")" && [ "$GPU_COUNT" -eq 8 ] && "$ENV_DIR/bin/python3" -c %q && printf "role=%s ray=" && "$ENV_DIR/bin/python3" -c %q' \
        "${env_file}" \
        'import ray, torch, vllm; print(f"python={__import__("sys").version_info[:3]} ray={ray.__version__} torch={torch.__version__} vllm={vllm.__version__}")' \
        "${role}" \
        'import ray, torch, vllm; print(f"{ray.__version__}|{torch.__version__}|{vllm.__version__}")'
    remote "${host}" "${remote_command}"
}

assert_no_existing_ray() {
    local host="$1"
    local env_file="$2"
    local remote_command
    if [[ "${CLEAN_RAY}" == "true" ]]; then
        printf -v remote_command 'source %q && { "$ENV_DIR/bin/ray" stop --force >/dev/null 2>&1 || true; } && sleep 2 && ! pgrep -af "raylet|gcs_server|dashboard.py" >/dev/null' "${env_file}"
    else
        printf -v remote_command '! pgrep -af "raylet|gcs_server|dashboard.py" >/dev/null'
    fi
    remote "${host}" "${remote_command}" \
        || fail "Existing Ray processes found on ${host}. Re-run with CLEAN_RAY=true only if the node is dedicated to this trial."
}

start_ray_head() {
    local trial_id="$1" host="$2" head_ip="$3" ray_port="$4" dashboard_port="$5" nccl_interface="$6" env_file="$7" run_root="$8"
    local ray_tmp="/dev/shm/cgrpo-${trial_id}-h"
    local remote_command
    printf -v remote_command 'source %q && mkdir -p %q %q && nohup env CUDA_VISIBLE_DEVICES=%q NCCL_SOCKET_IFNAME=%q RAY_TMPDIR=%q "$ENV_DIR/bin/ray" start --head --node-ip-address=%q --port=%q --dashboard-host=0.0.0.0 --dashboard-port=%q --num-gpus=8 --temp-dir=%q --block > %q 2>&1 < /dev/null & echo $! > %q' \
        "${env_file}" "${ray_tmp}" "${run_root}/multinode" "${CUDA_DEVICES}" "${nccl_interface}" "${ray_tmp}" "${head_ip}" "${ray_port}" "${dashboard_port}" "${ray_tmp}" "${run_root}/multinode/ray_head.log" "${run_root}/multinode/ray_head.pid"
    remote "${host}" "${remote_command}"
}

start_ray_worker() {
    local trial_id="$1" host="$2" worker_ip="$3" head_ip="$4" ray_port="$5" nccl_interface="$6" env_file="$7" run_root="$8"
    local ray_tmp="/dev/shm/cgrpo-${trial_id}-w"
    local remote_command
    printf -v remote_command 'source %q && mkdir -p %q %q && nohup env CUDA_VISIBLE_DEVICES=%q NCCL_SOCKET_IFNAME=%q RAY_TMPDIR=%q "$ENV_DIR/bin/ray" start --address=%q --node-ip-address=%q --num-gpus=8 --temp-dir=%q --block > %q 2>&1 < /dev/null & echo $! > %q' \
        "${env_file}" "${ray_tmp}" "${run_root}/multinode" "${CUDA_DEVICES}" "${nccl_interface}" "${ray_tmp}" "${head_ip}:${ray_port}" "${worker_ip}" "${ray_tmp}" "${run_root}/multinode/ray_worker.log" "${run_root}/multinode/ray_worker.pid"
    remote "${host}" "${remote_command}"
}

wait_for_ray_cluster() {
    local host="$1" head_ip="$2" ray_port="$3" env_file="$4"
    local remote_command attempt
    printf -v remote_command 'source %q && "$ENV_DIR/bin/python3" -c %q %q' \
        "${env_file}" \
        'import sys, ray; ray.init(address=sys.argv[1], logging_level="ERROR"); nodes=[n for n in ray.nodes() if n.get("Alive")]; gpus=sum(float(n.get("Resources", {}).get("GPU", 0)) for n in nodes); print(f"alive_nodes={len(nodes)} gpus={gpus:g}"); ray.shutdown(); raise SystemExit(0 if len(nodes) == 2 and gpus >= 16 else 1)' \
        "${head_ip}:${ray_port}"
    if [[ "${DRY_RUN}" == "true" ]]; then
        remote "${host}" "${remote_command}"
        return 0
    fi
    for ((attempt = 1; attempt <= RAY_READY_TIMEOUT_SECONDS / 2; attempt++)); do
        if remote "${host}" "${remote_command}"; then
            return 0
        fi
        sleep 2
    done
    fail "Ray cluster ${head_ip}:${ray_port} did not become a two-node 16-GPU cluster within ${RAY_READY_TIMEOUT_SECONDS}s."
}

launch_training() {
    local trial_id="$1" head_host="$2" head_ip="$3" ray_port="$4" nccl_interface="$5" env_file="$6" run_root="$7"
    local head_tmp="/dev/shm/cgrpo-${trial_id}-h"
    local remote_command
    printf -v remote_command 'source %q && mkdir -p %q && export CUDA_VISIBLE_DEVICES=%q NCCL_SOCKET_IFNAME=%q RAY_TMPDIR=%q RAY_ADDRESS=%q RAY_NAMESPACE=%q MULTINODE_ENABLED=true NNODES=2 RAY_CLUSTER_EXPECTED_NODES=2 RAY_CLUSTER_EXPECTED_GPUS=16 && nohup bash "$REPO_DIR/projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh" > %q 2>&1 < /dev/null & echo $! > %q' \
        "${env_file}" "${run_root}/multinode" "${CUDA_DEVICES}" "${nccl_interface}" "${head_tmp}" "${head_ip}:${ray_port}" "cyclegrpo-${trial_id}" "${run_root}/multinode/trainer_launcher.log" "${run_root}/multinode/trainer_launcher.pid"
    remote "${head_host}" "${remote_command}"
}

trial_status() {
    local index="$1"
    local trial_id="${TRIAL_IDS[index]}" head_host="${HEAD_HOSTS[index]}" worker_host="${WORKER_HOSTS[index]}"
    local head_ip="${HEAD_IPS[index]}" ray_port="${RAY_PORTS[index]}" env_file="${ENV_FILES[index]}" run_root="${RUN_ROOTS[index]}"
    local remote_command
    printf '== %s ==\n' "${trial_id}"
    printf -v remote_command 'source %q && if [ -f %q ]; then TRAIN_PID="$(cat %q)"; if kill -0 "$TRAIN_PID" 2>/dev/null; then echo "trainer=running pid=$TRAIN_PID"; else echo "trainer=stopped pid=$TRAIN_PID"; fi; else echo "trainer=pid-file-missing"; fi; "$ENV_DIR/bin/python3" -c %q %q' \
        "${env_file}" "${run_root}/multinode/trainer_launcher.pid" "${run_root}/multinode/trainer_launcher.pid" \
        'import sys, ray; ray.init(address=sys.argv[1], logging_level="ERROR"); nodes=[n for n in ray.nodes() if n.get("Alive")]; gpus=sum(float(n.get("Resources", {}).get("GPU", 0)) for n in nodes); print(f"ray_alive_nodes={len(nodes)} ray_gpus={gpus:g}"); ray.shutdown()' \
        "${head_ip}:${ray_port}"
    remote "${head_host}" "${remote_command}" || true
    printf -v remote_command 'if [ -f %q ]; then RAY_PID="$(cat %q)"; if kill -0 "$RAY_PID" 2>/dev/null; then echo "ray_worker=running pid=$RAY_PID"; else echo "ray_worker=stopped pid=$RAY_PID"; fi; else echo "ray_worker=pid-file-missing"; fi' \
        "${run_root}/multinode/ray_worker.pid" "${run_root}/multinode/ray_worker.pid"
    remote "${worker_host}" "${remote_command}" || true
}

stop_trial() {
    local index="$1"
    local head_host="${HEAD_HOSTS[index]}" worker_host="${WORKER_HOSTS[index]}" env_file="${ENV_FILES[index]}" run_root="${RUN_ROOTS[index]}"
    local remote_command
    printf -v remote_command 'source %q && if [ -f %q ]; then TRAIN_PID="$(cat %q)"; kill -TERM "$TRAIN_PID" 2>/dev/null || true; fi; "$ENV_DIR/bin/ray" stop --force || true' \
        "${env_file}" "${run_root}/multinode/trainer_launcher.pid" "${run_root}/multinode/trainer_launcher.pid"
    remote "${head_host}" "${remote_command}"
    printf -v remote_command 'source %q && "$ENV_DIR/bin/ray" stop --force || true' "${env_file}"
    remote "${worker_host}" "${remote_command}"
}

case "${ACTION}" in
    launch)
        VERSION_SIGNATURE=""
        for ((index = 0; index < ${#TRIAL_IDS[@]}; index++)); do
            printf 'Preflight %s (%s, %s)\n' "${TRIAL_IDS[index]}" "${HEAD_HOSTS[index]}" "${WORKER_HOSTS[index]}"
            assert_no_existing_ray "${HEAD_HOSTS[index]}" "${ENV_FILES[index]}"
            assert_no_existing_ray "${WORKER_HOSTS[index]}" "${ENV_FILES[index]}"
            if [[ "${DRY_RUN}" == "true" ]]; then
                remote_preflight "${HEAD_HOSTS[index]}" "${ENV_FILES[index]}" head
                remote_preflight "${WORKER_HOSTS[index]}" "${ENV_FILES[index]}" worker
            else
                head_signature="$(remote_preflight "${HEAD_HOSTS[index]}" "${ENV_FILES[index]}" head)" || fail "Preflight failed on ${HEAD_HOSTS[index]}"
                worker_signature="$(remote_preflight "${WORKER_HOSTS[index]}" "${ENV_FILES[index]}" worker)" || fail "Preflight failed on ${WORKER_HOSTS[index]}"
                head_version="${head_signature##*ray=}"
                worker_version="${worker_signature##*ray=}"
                [[ "${head_version}" == "${worker_version}" ]] \
                    || fail "Ray/Torch/vLLM versions differ inside ${TRIAL_IDS[index]}: ${head_version} vs ${worker_version}"
                if [[ -z "${VERSION_SIGNATURE}" ]]; then
                    VERSION_SIGNATURE="${head_version}"
                else
                    [[ "${VERSION_SIGNATURE}" == "${head_version}" ]] \
                        || fail "Ray/Torch/vLLM versions differ between trials: ${VERSION_SIGNATURE} vs ${head_version}"
                fi
            fi
        done

        for ((index = 0; index < ${#TRIAL_IDS[@]}; index++)); do
            printf 'Starting Ray cluster for %s\n' "${TRIAL_IDS[index]}"
            start_ray_head "${TRIAL_IDS[index]}" "${HEAD_HOSTS[index]}" "${HEAD_IPS[index]}" "${RAY_PORTS[index]}" "${DASHBOARD_PORTS[index]}" "${NCCL_INTERFACES[index]}" "${ENV_FILES[index]}" "${RUN_ROOTS[index]}" \
                || fail "Failed to start Ray head for ${TRIAL_IDS[index]}"
            start_ray_worker "${TRIAL_IDS[index]}" "${WORKER_HOSTS[index]}" "${WORKER_IPS[index]}" "${HEAD_IPS[index]}" "${RAY_PORTS[index]}" "${NCCL_INTERFACES[index]}" "${ENV_FILES[index]}" "${RUN_ROOTS[index]}" \
                || fail "Failed to start Ray worker for ${TRIAL_IDS[index]}"
            wait_for_ray_cluster "${HEAD_HOSTS[index]}" "${HEAD_IPS[index]}" "${RAY_PORTS[index]}" "${ENV_FILES[index]}"
            launch_training "${TRIAL_IDS[index]}" "${HEAD_HOSTS[index]}" "${HEAD_IPS[index]}" "${RAY_PORTS[index]}" "${NCCL_INTERFACES[index]}" "${ENV_FILES[index]}" "${RUN_ROOTS[index]}" \
                || fail "Failed to launch training for ${TRIAL_IDS[index]}"
            printf 'Started %s; head log: %s/multinode/trainer_launcher.log\n' "${TRIAL_IDS[index]}" "${RUN_ROOTS[index]}"
        done
        ;;
    status)
        for ((index = 0; index < ${#TRIAL_IDS[@]}; index++)); do
            trial_status "${index}"
        done
        ;;
    stop)
        for ((index = 0; index < ${#TRIAL_IDS[@]}; index++)); do
            printf 'Stopping %s\n' "${TRIAL_IDS[index]}"
            stop_trial "${index}"
        done
        ;;
esac
