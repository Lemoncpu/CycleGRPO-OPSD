"""Keep the CUDA devices listed by CUDA_VISIBLE_DEVICES occupied until stopped."""

from __future__ import annotations

import argparse
import signal
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memory-mb",
        type=int,
        default=40000,
        help="Approximate memory to reserve on each visible CUDA device (default: 40000).",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=60.0,
        help="Status interval while keeping devices occupied (default: 60).",
    )
    args = parser.parse_args()
    if args.memory_mb <= 0:
        parser.error("--memory-mb must be positive")
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    bytes_per_device = args.memory_mb * 1024 * 1024
    reservations = []
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.set_device(device_index)
        reservation = torch.zeros(
            (bytes_per_device + 3) // 4,
            dtype=torch.float32,
            device=device_index,
        )
        reservations.append(reservation)
        torch.cuda.synchronize(device_index)
        print(
            f"CUDA device {device_index} reserved approximately {args.memory_mb} MiB",
            flush=True,
        )

    stopping = False

    def stop_handler(signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True
        print(f"Received signal {signum}; releasing CUDA reservations.", flush=True)

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    print(
        f"Holding {len(reservations)} CUDA device(s). Send SIGTERM or press Ctrl-C to stop.",
        flush=True,
    )
    while not stopping:
        time.sleep(args.interval_seconds)

    del reservations
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
