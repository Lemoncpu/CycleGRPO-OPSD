#!/usr/bin/env python3
"""Build the fixed 20k Single/Multi/Stuff/Part/no-target CycleGRPO mixture."""

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset


COUNTS = {
    "single": 7_000,
    "multi": 5_000,
    "stuff": 4_000,
    "part": 2_000,
    "no_target": 2_000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refcoco", type=Path, required=True, help="RefCOCO cycle parquet with at least 7k rows")
    parser.add_argument("--grefcoco", type=Path, required=True, help="5k multi + 2k no-target gRefCOCO parquet")
    parser.add_argument("--cocostuff", type=Path, required=True, help="4k COCO-Stuff cycle parquet")
    parser.add_argument("--paco-parts", type=Path, required=True, help="2k PACO part cycle parquet")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [dict(record) for record in Dataset.from_parquet(str(path))]


def select_rows(
    rows: list[dict[str, Any]],
    *,
    source: str,
    count: int,
    seed: int,
    name: str,
) -> list[dict[str, Any]]:
    candidates = [row for row in rows if row.get("source") == source]
    candidates.sort(key=lambda row: str(row.get("images", [""])[0]))
    random.Random(seed).shuffle(candidates)
    if len(candidates) < count:
        raise RuntimeError(f"{name} has {len(candidates)} rows for source {source!r}; need {count}.")
    return candidates[:count]


def strip_positive_caption_answers(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove inherited referring expressions from positive source parquets."""
    sanitized = []
    for row in rows:
        cleaned = dict(row)
        cleaned["cap_answer"] = None
        sanitized.append(cleaned)
    return sanitized


def main() -> None:
    args = parse_args()
    ref_rows = load_rows(args.refcoco)
    gref_rows = load_rows(args.grefcoco)
    stuff_rows = load_rows(args.cocostuff)
    part_rows = load_rows(args.paco_parts)

    selected = {
        "single": select_rows(
            ref_rows, source="refcoco_cycle", count=COUNTS["single"], seed=args.seed + 1, name="RefCOCO"
        ),
        "multi": select_rows(
            gref_rows, source="grefcoco_cycle", count=COUNTS["multi"], seed=args.seed + 2, name="gRefCOCO"
        ),
        "stuff": select_rows(
            stuff_rows, source="cocostuff_cycle", count=COUNTS["stuff"], seed=args.seed + 3, name="COCO-Stuff"
        ),
        "part": select_rows(
            part_rows, source="paco_part_cycle", count=COUNTS["part"], seed=args.seed + 4, name="PACO-LVIS"
        ),
        "no_target": select_rows(
            gref_rows, source="gres_no_target", count=COUNTS["no_target"], seed=args.seed + 5, name="gRefCOCO"
        ),
    }
    for role in ("single", "multi", "stuff", "part"):
        selected[role] = strip_positive_caption_answers(selected[role])
    combined = [row for role in COUNTS for row in selected[role]]
    random.Random(args.seed).shuffle(combined)
    if len(combined) != sum(COUNTS.values()):
        raise AssertionError(f"Expected 20k rows, got {len(combined)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(combined).to_parquet(str(args.output))
    manifest = {
        "seed": args.seed,
        "total": len(combined),
        "mixture": COUNTS,
        "inputs": {
            "refcoco": str(args.refcoco.resolve()),
            "grefcoco": str(args.grefcoco.resolve()),
            "cocostuff": str(args.cocostuff.resolve()),
            "paco_parts": str(args.paco_parts.resolve()),
        },
        "source_counts": {
            source: sum(row["source"] == source for row in combined)
            for source in sorted({str(row["source"]) for row in combined})
        },
        "positive_cap_answer": None,
        "no_target_expression_location": "cap_problem only",
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(json.dumps(manifest, indent=2))
    print(f"Wrote {len(combined)} samples to {args.output}")


if __name__ == "__main__":
    main()
