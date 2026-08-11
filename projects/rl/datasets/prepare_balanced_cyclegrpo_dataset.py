#!/usr/bin/env python3
"""Build a configurable Single/Multi/Stuff/Part/no-target CycleGRPO mixture."""

import argparse
import json
import random
from pathlib import Path
from typing import Any

from projects.rl.datasets.generate_dam_caption_qa import validate_caption_qa_questions

DEFAULT_COUNTS = {
    "single": 8_000,
    "multi": 4_000,
    "stuff": 5_000,
    "part": 2_000,
    "no_target": 1_000,
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
    parser.add_argument(
        "--require-grounding-query",
        action="store_true",
        help="Fail unless every selected sample has a non-empty direct-grounding query.",
    )
    parser.add_argument(
        "--caption-qa-manifest",
        type=Path,
        default=None,
        help="Accepted DAM QA JSONL. Its Stuff/PACO IDs are included before random fill.",
    )
    parser.add_argument("--qa-stuff-count", type=int, default=3_000)
    parser.add_argument("--qa-part-count", type=int, default=2_000)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    from datasets import Dataset

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
    predicate: Any = None,
) -> list[dict[str, Any]]:
    candidates = [row for row in rows if row.get("source") == source]
    if predicate is not None:
        candidates = [row for row in candidates if predicate(row)]
    candidates.sort(key=lambda row: str(row.get("images", [""])[0]))
    random.Random(seed).shuffle(candidates)
    if len(candidates) < count:
        raise RuntimeError(f"{name} has {len(candidates)} rows for source {source!r}; need {count}.")
    return candidates[:count]


def load_caption_qa_ids(path: Path | None) -> dict[str, set[str]]:
    """Read accepted QA IDs and keep source membership explicit."""
    result = {"cocostuff_cycle": set(), "paco_part_cycle": set()}
    if path is None:
        return result
    if not path.is_file():
        raise FileNotFoundError(path)
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            dam_id, source, questions = row.get("dam_source_id"), row.get("source"), row.get("questions")
            if not isinstance(dam_id, str) or not dam_id:
                raise ValueError(f"{path}:{line_number} is missing dam_source_id")
            if dam_id in seen:
                raise ValueError(f"{path}:{line_number} duplicates dam_source_id {dam_id!r}")
            if source not in result:
                raise ValueError(f"{path}:{line_number} has unsupported QA source {source!r}")
            try:
                validate_caption_qa_questions(questions)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number} has invalid QA questions: {error}") from error
            seen.add(dam_id)
            result[source].add(dam_id)
    return result


def select_rows_with_required_ids(
    rows: list[dict[str, Any]], *, source: str, count: int, required_ids: set[str], seed: int, name: str
) -> list[dict[str, Any]]:
    candidates = [row for row in rows if row.get("source") == source]
    by_id = {row.get("dam_source_id"): row for row in candidates}
    missing = sorted(required_ids - set(by_id))
    if missing:
        raise RuntimeError(f"{name} QA manifest ID is absent from source parquet: {missing[0]!r}")
    if len(required_ids) > count:
        raise RuntimeError(f"{name} has {len(required_ids)} required QA IDs; quota is only {count}.")
    required = [by_id[dam_id] for dam_id in sorted(required_ids)]
    remainder = [row for row in candidates if row.get("dam_source_id") not in required_ids]
    remainder.sort(key=lambda row: str(row.get("images", [""])[0]))
    random.Random(seed).shuffle(remainder)
    if len(required) + len(remainder) < count:
        raise RuntimeError(f"{name} has insufficient rows for quota {count}.")
    return required + remainder[: count - len(required)]


def assert_multi_rows(rows: list[dict[str, Any]]) -> None:
    if any(int(row.get("grounding_instance_count", 0)) < 2 for row in rows):
        raise RuntimeError(
            "gRefCOCO multi quota requires grounding_instance_count >= 2; "
            "re-export gRefCOCO with this converter before mixing."
        )


def strip_positive_caption_answers(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove inherited referring expressions from positive source parquets."""
    sanitized = []
    for row in rows:
        cleaned = dict(row)
        cleaned["cap_answer"] = None
        sanitized.append(cleaned)
    return sanitized


def assert_grounding_queries(rows: list[dict[str, Any]]) -> None:
    missing = [
        row.get("source")
        for row in rows
        if not isinstance(row.get("grounding_query"), str) or not row["grounding_query"].strip()
    ]
    if missing:
        raise RuntimeError(
            "All selected samples must have non-empty grounding_query when "
            f"--require-grounding-query is set; first missing source={missing[0]!r}."
        )


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
    if sum(counts.values()) <= 0:
        raise ValueError(f"Mixture counts must sum to a positive total, got {counts}")
    return counts


def main() -> None:
    from datasets import Dataset

    args = parse_args()
    counts = mixture_counts(args)
    ref_rows = load_rows(args.refcoco)
    gref_rows = load_rows(args.grefcoco)
    stuff_rows = load_rows(args.cocostuff)
    part_rows = load_rows(args.paco_parts)
    qa_ids = load_caption_qa_ids(args.caption_qa_manifest)
    if args.caption_qa_manifest is not None:
        expected_qa = {
            "cocostuff_cycle": args.qa_stuff_count,
            "paco_part_cycle": args.qa_part_count,
        }
        actual_qa = {source: len(ids) for source, ids in qa_ids.items()}
        if actual_qa != expected_qa:
            raise RuntimeError(
                "Caption QA manifest source counts must match the requested 3k Stuff / 2k PACO split: "
                f"expected={expected_qa}, actual={actual_qa}."
            )

    selected = {
        "single": select_rows(
            ref_rows, source="refcoco_cycle", count=counts["single"], seed=args.seed + 1, name="RefCOCO"
        ),
        "multi": select_rows(
            gref_rows,
            source="grefcoco_cycle",
            count=counts["multi"],
            seed=args.seed + 2,
            name="gRefCOCO multi",
            predicate=lambda row: int(row.get("grounding_instance_count", 0)) >= 2,
        ),
        "stuff": select_rows_with_required_ids(
            stuff_rows, source="cocostuff_cycle", count=counts["stuff"],
            required_ids=qa_ids["cocostuff_cycle"], seed=args.seed + 3, name="COCO-Stuff"
        ),
        "part": select_rows_with_required_ids(
            part_rows, source="paco_part_cycle", count=counts["part"],
            required_ids=qa_ids["paco_part_cycle"], seed=args.seed + 4, name="PACO-LVIS"
        ),
        "no_target": select_rows(
            gref_rows, source="gres_no_target", count=counts["no_target"], seed=args.seed + 5, name="gRefCOCO"
        ),
    }
    assert_multi_rows(selected["multi"])
    for role in ("single", "multi", "stuff", "part"):
        selected[role] = strip_positive_caption_answers(selected[role])
    combined = [row for role in counts for row in selected[role]]
    if args.require_grounding_query:
        assert_grounding_queries(combined)
    random.Random(args.seed).shuffle(combined)
    if len(combined) != sum(counts.values()):
        raise AssertionError(f"Expected {sum(counts.values())} rows, got {len(combined)}")

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
        "grounding_query": "preserved separately; never used as cap_answer",
        "require_grounding_query": args.require_grounding_query,
        "grounding_query_counts": {
            source: sum(
                row["source"] == source
                and isinstance(row.get("grounding_query"), str)
                and bool(row["grounding_query"].strip())
                for row in combined
            )
            for source in sorted({str(row["source"]) for row in combined})
        },
        "qa_manifest": str(args.caption_qa_manifest.resolve()) if args.caption_qa_manifest else None,
        "qa_selected": {source: len(ids) for source, ids in qa_ids.items()},
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(json.dumps(manifest, indent=2))
    print(f"Wrote {len(combined)} samples to {args.output}")


if __name__ == "__main__":
    main()
