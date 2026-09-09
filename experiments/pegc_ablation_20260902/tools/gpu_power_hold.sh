#!/usr/bin/env bash
# Reserve GPU memory and continuously run Tensor Core work on selected GPUs.
#
# Usage:
#   bash tools/gpu_power_hold.sh start
#   bash tools/gpu_power_hold.sh status
#   bash tools/gpu_power_hold.sh stop
#
# Environment overrides:
#   GPU_LIST=0,1,2,3 MEMORY_MIB=40000 MATMUL_DIM=12288
#   PYTHON_BIN=/path/to/python STATE_DIR=/tmp/cyclegrpo_gpu_power_hold

ACTION=${1:-status}
GPU_LIST=${GPU_LIST:-0,1,2,3}
MEMORY_MIB=${MEMORY_MIB:-40000}
MATMUL_DIM=${MATMUL_DIM:-12288}
PYTHON_BIN=${PYTHON_BIN:-python3}
STATE_DIR=${STATE_DIR:-/tmp/cyclegrpo_gpu_power_hold}
WORKER_TAG=cyclegrpo_gpu_power_hold_worker

if [ "$ACTION" != start ] && [ "$ACTION" != status ] && [ "$ACTION" != stop ]; then
  echo "Usage: $0 {start|status|stop}" >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required." >&2
  exit 2
fi

if ! "$PYTHON_BIN" -c 'import torch' >/dev/null 2>&1; then
  echo "PYTHON_BIN must provide PyTorch with CUDA support: $PYTHON_BIN" >&2
  exit 2
fi

case "$GPU_LIST" in
  ''|*[!0-9,]*)
    echo "GPU_LIST must be numeric GPU indices separated by commas: $GPU_LIST" >&2
    exit 2
    ;;
esac

case "$MEMORY_MIB" in
  ''|*[!0-9]*)
    echo "MEMORY_MIB must be a positive integer: $MEMORY_MIB" >&2
    exit 2
    ;;
esac

case "$MATMUL_DIM" in
  ''|*[!0-9]*)
    echo "MATMUL_DIM must be a positive integer: $MATMUL_DIM" >&2
    exit 2
    ;;
esac

if [ "$MEMORY_MIB" -le 0 ] || [ "$MATMUL_DIM" -le 0 ]; then
  echo "MEMORY_MIB and MATMUL_DIM must be positive." >&2
  exit 2
fi

gpu_exists() {
  nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ' | grep -qx "$1"
}

worker_is_ours() {
  pid=$1
  ps -p "$pid" -o args= 2>/dev/null | grep -q "$WORKER_TAG"
}

print_status() {
  echo "State directory: $STATE_DIR"
  for gpu in $(printf '%s' "$GPU_LIST" | tr ',' ' '); do
    pid_file="$STATE_DIR/gpu${gpu}.pid"
    if [ -f "$pid_file" ]; then
      pid=$(cat "$pid_file")
      if worker_is_ours "$pid"; then
        echo "GPU $gpu: worker PID $pid is running"
      else
        echo "GPU $gpu: stale or non-matching PID file ($pid)"
      fi
    else
      echo "GPU $gpu: no worker PID file"
    fi
  done
  nvidia-smi --query-gpu=index,memory.used,memory.total,power.draw,power.limit,utilization.gpu \
    --format=csv
}

if [ "$ACTION" = status ]; then
  print_status
  exit 0
fi

if [ "$ACTION" = stop ]; then
  for gpu in $(printf '%s' "$GPU_LIST" | tr ',' ' '); do
    pid_file="$STATE_DIR/gpu${gpu}.pid"
    [ -f "$pid_file" ] || continue
    pid=$(cat "$pid_file")
    if worker_is_ours "$pid"; then
      kill "$pid"
      echo "Stopped GPU $gpu worker PID $pid."
    else
      echo "Refusing to kill PID $pid for GPU $gpu: it is not this script's worker." >&2
    fi
    rm -f "$pid_file"
  done
  print_status
  exit 0
fi

mkdir -p "$STATE_DIR"

for gpu in $(printf '%s' "$GPU_LIST" | tr ',' ' '); do
  if ! gpu_exists "$gpu"; then
    echo "GPU $gpu is not reported by nvidia-smi." >&2
    exit 2
  fi

  pid_file="$STATE_DIR/gpu${gpu}.pid"
  if [ -f "$pid_file" ]; then
    existing_pid=$(cat "$pid_file")
    if worker_is_ours "$existing_pid"; then
      echo "GPU $gpu worker is already running as PID $existing_pid."
      continue
    fi
    echo "Removing stale PID record for GPU $gpu: $existing_pid."
    rm -f "$pid_file"
  fi

  CUDA_VISIBLE_DEVICES="$gpu" \
  GPU_HOLD_MEMORY_MIB="$MEMORY_MIB" \
  GPU_HOLD_MATMUL_DIM="$MATMUL_DIM" \
  nohup "$PYTHON_BIN" - --worker-tag "$WORKER_TAG" "$gpu" \
    > "$STATE_DIR/gpu${gpu}.log" 2>&1 <<'PY' &
import os
import signal
import sys
import time

import torch

physical_gpu = sys.argv[-1]
memory_mib = int(os.environ["GPU_HOLD_MEMORY_MIB"])
matrix_dim = int(os.environ["GPU_HOLD_MATMUL_DIM"])
stopping = False


def request_stop(signum, _frame):
    global stopping
    stopping = True
    print(f"GPU {physical_gpu}: received signal {signum}; releasing allocations.", flush=True)


signal.signal(signal.SIGINT, request_stop)
signal.signal(signal.SIGTERM, request_stop)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")

torch.cuda.set_device(0)
reserve = torch.empty(memory_mib * 1024 * 1024, dtype=torch.uint8, device="cuda")
reserve.fill_(1)

# These three BF16 tensors add about 0.84 GiB at the default dimension. They
# intentionally remain live while matmul keeps Tensor Cores and board power active.
a = torch.randn((matrix_dim, matrix_dim), dtype=torch.bfloat16, device="cuda")
b = torch.randn((matrix_dim, matrix_dim), dtype=torch.bfloat16, device="cuda")
c = torch.empty((matrix_dim, matrix_dim), dtype=torch.bfloat16, device="cuda")
torch.cuda.synchronize()
print(
    f"GPU {physical_gpu}: reserved {memory_mib} MiB plus BF16 matmul workspace "
    f"(dim={matrix_dim}).",
    flush=True,
)

while not stopping:
    torch.mm(a, b, out=c)
    torch.cuda.synchronize()

del c, b, a, reserve
torch.cuda.empty_cache()
PY
  worker_pid=$!
  echo "$worker_pid" > "$pid_file"
  echo "Started GPU $gpu worker PID $worker_pid; log: $STATE_DIR/gpu${gpu}.log"
done

print_status
