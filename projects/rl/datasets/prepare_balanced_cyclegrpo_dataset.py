#!/usr/bin/env python3
"""Build a configurable 20k Single/Multi/Stuff/Part/no-target CycleGRPO mixture."""

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset


DEFAULT_COUNTS = {
    "single": 7_000,
    "multi": 5_000,
    "stuff": 4_000,
    "part": 2_000,
    "no_target": 2_000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refcoco", type=Path, required=True, help="RefCOCO cycle parquet")
    parser.add_argument("--grefcoco", type=Path, required=True, help="gRefCOCO multi/no-target parquet")
    parser.add_argument("--cocostuff", type=Path, required=True, help="COCO-Stuff cycle parquet")
    parser.add_argument("--paco-parts", type=Path, required=True, help="PACO-LVIS part cycle parquet")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--single-count", type=int, default=DEFAULT_COUNTS["single"])
    parser.add_argument("--multi-count", type=int, default=DEFAULT_COUNTS["multi"])
    parser.add_argument("--stuff-count", type=int, default=DEFAULT_COUNTS["stuff"])
    parser.add_argument("--part-count", type=int, default=DEFAULT_COUNTS["part"])
    parser.add_argument("--no-target-count", type=int, default=DEFAULT_COUNTS["no_target"])
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


def mixture_counts(args: argparse.Namespace) -> dict[str, int]:
    counts = {
        "single": args.single_count,
        "multi": args.multi_count,
        "stuff": args.stuff_count,
        "part": args.part_count,
        "no_target": args.no_target_count,
    }
    if any(count < 0 for count in counts.values()):
        raise ValueError(f"Mixture counts must be non-negative: {counts}")
    if sum(counts.values()) != 20_000:
        raise ValueError(f"Mixture counts must sum to 20000, got {sum(counts.values())}: {counts}")
    return counts


def main() -> None:
    args = parse_args()
    counts = mixture_counts(args)
    ref_rows = load_rows(args.refcoco)
    gref_rows = load_rows(args.grefcoco)
    stuff_rows = load_rows(args.cocostuff)
    part_rows = load_rows(args.paco_parts)

    selected = {
        "single": select_rows(
            ref_rows, source="refcoco_cycle", count=counts["single"], seed=args.seed + 1, name="RefCOCO"
        ),
        "multi": select_rows(
            gref_rows, source="grefcoco_cycle", count=counts["multi"], seed=args.seed + 2, name="gRefCOCO"
        ),
        "stuff": select_rows(
            stuff_rows, source="cocostuff_cycle", count=counts["stuff"], seed=args.seed + 3, name="COCO-Stuff"
        ),
        "part": select_rows(
            part_rows, source="paco_part_cycle", count=counts["part"], seed=args.seed + 4, name="PACO-LVIS"
        ),
        "no_target": select_rows(
            gref_rows, source="gres_no_target", count=counts["no_target"], seed=args.seed + 5, name="gRefCOCO"
        ),
    }
    for role in ("single", "multi", "stuff", "part"):
        selected[role] = strip_positive_caption_answers(selected[role])
    combined = [row for role in counts for row in selected[role]]
    random.Random(args.seed).shuffle(combined)
    if len(combined) != sum(counts.values()):
        raise AssertionError(f"Expected 20k rows, got {len(combined)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(combined).to_parquet(str(args.output))
    manifest = {
        "seed": args.seed,
        "total": len(combined),
        "mixture": counts,
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
