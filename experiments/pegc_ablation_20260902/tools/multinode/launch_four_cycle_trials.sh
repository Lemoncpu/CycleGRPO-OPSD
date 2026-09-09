#!/usr/bin/env bash
# Explicit entrypoint for the four pure 20k CycleGRPO two-node trials.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/launch_four_trials.sh" "$@"
