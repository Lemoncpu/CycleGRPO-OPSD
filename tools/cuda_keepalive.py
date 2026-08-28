"""Reserve visible idle GPUs, but never compete with an active CUDA workload."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time


def visible_gpu_indices(available_indices: list[int]) -> list[int]:
    """Return physical GPU indexes for the numeric CUDA_VISIBLE_DEVICES form."""
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not configured:
        return available_indices
    values = [value.strip() for value in configured.split(",") if value.strip()]
    if not values or not all(value.isdigit() for value in values):
        raise RuntimeError(
            "cuda_keepalive requires numeric CUDA_VISIBLE_DEVICES, for example "
            "CUDA_VISIBLE_DEVICES=0,1,2,3."
        )
    indices = [int(value) for value in values]
    if len(set(indices)) != len(indices) or any(index not in available_indices for index in indices):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must name distinct physical GPUs reported by nvidia-smi."
        )
    return indices


def gpu_memory_used_mib() -> dict[int, int]:
    """Read total device memory use, including processes outside this Python process."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    usage: dict[int, int] = {}
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise RuntimeError(f"Unexpected nvidia-smi output: {line!r}")
        usage[int(fields[0])] = int(fields[1])
    return usage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memory-mb",
        type=int,
        default=40000,
        help="Approximate memory to reserve after an idle GPU is detected (default: 40000).",
    )
    parser.add_argument(
        "--idle-threshold-mb",
        type=int,
        default=1,
        help="Reserve only when total nvidia-smi use is below this many MiB (default: 1).",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=15.0,
        help="Seconds between idle-GPU checks (default: 15).",
    )
    args = parser.parse_args()
    if args.memory_mb <= 0:
        parser.error("--memory-mb must be positive")
    if args.idle_threshold_mb < 0:
        parser.error("--idle-threshold-mb must be non-negative")
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")

    import torch

    # Do not call any torch.cuda API before observing idleness: an initialized
    # CUDA context could itself make an otherwise idle card look non-idle.
    physical_indices = visible_gpu_indices(sorted(gpu_memory_used_mib()))
    bytes_per_device = args.memory_mb * 1024 * 1024
    reservations: dict[int, torch.Tensor] = {}
    stopping = False

    def stop_handler(signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True
        print(f"Received signal {signum}; releasing CUDA reservations.", flush=True)

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    print(
        "Monitoring "
        f"{len(physical_indices)} visible CUDA device(s); reserving {args.memory_mb} MiB only "
        f"when total GPU use is below {args.idle_threshold_mb} MiB.",
        flush=True,
    )

    while not stopping:
        try:
            used_mib = gpu_memory_used_mib()
            for local_index, physical_index in enumerate(physical_indices):
                if local_index in reservations:
                    continue
                current_use = used_mib.get(physical_index)
                if current_use is None:
                    raise RuntimeError(f"GPU {physical_index} is missing from nvidia-smi output")
                if current_use >= args.idle_threshold_mb:
                    print(
                        f"CUDA device {local_index} (physical {physical_index}) remains in use: "
                        f"{current_use} MiB.",
                        flush=True,
                    )
                    continue
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available")
                torch.cuda.set_device(local_index)
                reservation = torch.zeros(
                    (bytes_per_device + 3) // 4,
                    dtype=torch.float32,
                    device=local_index,
                )
                torch.cuda.synchronize(local_index)
                reservations[local_index] = reservation
                print(
                    f"CUDA device {local_index} (physical {physical_index}) was idle at "
                    f"{current_use} MiB; reserved approximately {args.memory_mb} MiB.",
                    flush=True,
                )
        except Exception as error:
            print(f"Idle-GPU check failed: {type(error).__name__}: {error}", flush=True)

        if not stopping:
            time.sleep(args.interval_seconds)

    reservations.clear()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
