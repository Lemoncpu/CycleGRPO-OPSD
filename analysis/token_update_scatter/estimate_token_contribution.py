#!/usr/bin/env python3
"""Compute the registered teacher-alignment x coordinate from log-probs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="token_data.npz containing base_logp and teacher_logp")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    data = np.load(args.input)
    contribution = data["teacher_logp"] - data["base_logp"]
    valid = np.isfinite(contribution)
    np.savez_compressed(args.output, contribution=contribution, valid=valid)
    print(f"wrote {Path(args.output)}: {int(valid.sum())} finite token contributions")


if __name__ == "__main__":
    main()
