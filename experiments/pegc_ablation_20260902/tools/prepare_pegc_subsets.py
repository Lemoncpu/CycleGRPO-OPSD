#!/usr/bin/env python3
"""Create deterministic, source-stratified one-fifth PEGC ablation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def stratified_take(table: pa.Table, fraction: float, seed: int) -> pa.Table:
    source = np.asarray(table["source"].to_pylist(), dtype=object)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for value in sorted(set(str(item) for item in source)):
        indices = np.flatnonzero(source == value)
        count = max(1, int(round(len(indices) * fraction)))
        selected.extend(int(index) for index in rng.choice(indices, size=count, replace=False))
    selected.sort()
    return table.take(pa.array(selected, type=pa.int64()))


def write_subset(source: Path, destination: Path, fraction: float, seed: int) -> set[str]:
    table = stratified_take(pq.read_table(source), fraction, seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination, compression="zstd")
    ids = set()
    if "dam_source_id" in table.column_names:
        ids = {str(value) for value in table["dam_source_id"].to_pylist()}
    print(f"{source.name}: {table.num_rows} rows -> {destination} sources={sorted(set(table['source'].to_pylist()))}")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", type=Path, required=True)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--no-target", type=Path, required=True)
    parser.add_argument("--qa-parquet", type=Path, required=True)
    parser.add_argument("--qa-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        raise SystemExit("fraction must be in (0, 1]")

    output = args.output_dir
    write_subset(args.cycle, output / "cycle_4k.parquet", args.fraction, args.seed)
    write_subset(args.direct, output / "direct_8k.parquet", args.fraction, args.seed + 1)
    write_subset(args.no_target, output / "no_target_2k.parquet", args.fraction, args.seed + 2)
    qa_ids = write_subset(args.qa_parquet, output / "qa_2k.parquet", args.fraction, args.seed + 3)

    qa_output = output / "qa_2k.jsonl"
    count = 0
    with args.qa_jsonl.open(encoding="utf-8") as source, qa_output.open("w", encoding="utf-8") as destination:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if str(record.get("dam_source_id")) in qa_ids:
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    if count != len(qa_ids):
        raise RuntimeError(f"QA parquet/JSONL join mismatch: parquet={len(qa_ids)} jsonl={count}")
    print(f"qa jsonl: {count} rows -> {qa_output}")


if __name__ == "__main__":
    main()
