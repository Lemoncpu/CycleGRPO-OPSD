#!/usr/bin/env python3
"""Build the 40k/40k/20k disjoint scaling data used by the 8-GPU runner.

The three output streams are deliberately kept separate because the trainer
has three independent loaders.  ``main_40k`` follows the original
40/20/25/10/5 CycleGRPO mixture.  ``direct_40k`` contains the requested
single/multi/no-target/stuff/part supervised mixture and is also split into
positive and no-target files for the existing direct-loader contract.

"Disjoint" is enforced at the strongest identity that the source datasets
can support: no repeated (image, mask) pair and no repeated non-empty mask
RLE across the 100k rows.  COCO referring datasets intentionally contain
multiple expressions and regions per image, so requiring every row to have a
different image would make the requested 32k RefCOCO rows mathematically
impossible (the train split has only about 20k images).  No-target rows have
no physical mask; their identity is (image, normalized query).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


MAIN_COUNTS = {
    "refcoco_single": 16_000,
    "grefcoco_multi": 8_000,
    "stuff": 10_000,
    "part": 4_000,
    "no_target": 2_000,
}
DIRECT_COUNTS = {
    "refcoco_single": 16_000,
    "grefcoco_multi": 4_000,
    "stuff": 8_000,
    "part": 4_000,
    "no_target": 8_000,
}
QA_COUNTS = {"stuff": 10_000, "part": 10_000}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refcoco", type=Path, action="append", required=True)
    parser.add_argument("--grefcoco-multi", type=Path, action="append", required=True)
    parser.add_argument("--no-target", type=Path, action="append", required=True)
    parser.add_argument("--stuff", type=Path, action="append", required=True)
    parser.add_argument("--part", type=Path, action="append", required=True)
    parser.add_argument("--qa-stuff", type=Path, action="append", required=True)
    parser.add_argument("--qa-part", type=Path, action="append", required=True)
    parser.add_argument("--qa-stuff-manifest", type=Path, action="append", required=True)
    parser.add_argument("--qa-part-manifest", type=Path, action="append", required=True)
    parser.add_argument(
        "--qa-duplicate-id-policy",
        choices=("error", "uniquify"),
        default="error",
        help=(
            "How to handle repeated DAM IDs across over-generated manifests. "
            "'uniquify' accepts only byte-equivalent records and deterministically "
            "suffixes selected repeated rows so the final QA join remains one-to-one."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--main-single-count", type=int, default=MAIN_COUNTS["refcoco_single"])
    parser.add_argument("--main-multi-count", type=int, default=MAIN_COUNTS["grefcoco_multi"])
    parser.add_argument("--main-stuff-count", type=int, default=MAIN_COUNTS["stuff"])
    parser.add_argument("--main-part-count", type=int, default=MAIN_COUNTS["part"])
    parser.add_argument("--main-no-target-count", type=int, default=MAIN_COUNTS["no_target"])
    parser.add_argument("--direct-single-count", type=int, default=DIRECT_COUNTS["refcoco_single"])
    parser.add_argument("--direct-multi-count", type=int, default=DIRECT_COUNTS["grefcoco_multi"])
    parser.add_argument("--direct-stuff-count", type=int, default=DIRECT_COUNTS["stuff"])
    parser.add_argument("--direct-part-count", type=int, default=DIRECT_COUNTS["part"])
    parser.add_argument("--direct-no-target-count", type=int, default=DIRECT_COUNTS["no_target"])
    parser.add_argument(
        "--qa-total-count",
        type=int,
        default=sum(QA_COUNTS.values()),
        help="Total DLC-QA rows selected from the combined Stuff+Part pool (default: 20000).",
    )
    parser.add_argument(
        "--qa-stuff-count",
        type=int,
        default=None,
        help="Deprecated optional per-source quota; omit to select from the combined QA pool.",
    )
    parser.add_argument(
        "--qa-part-count",
        type=int,
        default=None,
        help="Deprecated optional per-source quota; omit to select from the combined QA pool.",
    )
    return parser.parse_args()


def load_parquet(paths: Iterable[Path]) -> list[dict[str, Any]]:
    from datasets import Dataset

    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.extend(dict(row) for row in Dataset.from_parquet(str(path)))
    if not rows:
        raise RuntimeError("No rows loaded from parquet inputs.")
    return rows


def normalize_query(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def image_key(row: dict[str, Any]) -> str:
    images = row.get("images")
    if isinstance(images, (list, tuple)):
        value = images[0] if images else ""
    else:
        value = images or ""
    if not value:
        raise ValueError("row has no images field")
    return str(Path(str(value)).resolve())


def mask_key(row: dict[str, Any]) -> str | None:
    value = row.get("masks")
    if value is None:
        return None
    if not isinstance(value, dict) or "size" not in value or "counts" not in value:
        raise ValueError(f"row has malformed masks field: {value!r}")
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sample_key(row: dict[str, Any]) -> tuple[str, str, str]:
    image = image_key(row)
    mask = mask_key(row)
    if mask is not None:
        return ("mask", image, mask)
    query = normalize_query(row.get("grounding_query"))
    if not query:
        raise ValueError(f"no-target row has no grounding_query: {image}")
    return ("no_target", image, query)


def source_rows(rows: Iterable[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("source") == source]


def shuffled(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    result = list(rows)
    result.sort(key=lambda row: (image_key(row), normalize_query(row.get("grounding_query"))))
    random.Random(seed).shuffle(result)
    return result


class GlobalSelector:
    """Reserve unique sample/mask identities across all three streams."""

    def __init__(self) -> None:
        self.sample_ids: set[tuple[str, str, str]] = set()
        self.mask_ids: set[str] = set()

    def take(
        self,
        rows: list[dict[str, Any]],
        count: int,
        *,
        name: str,
        seed: int,
        allow_duplicate_identities: bool = False,
    ) -> list[dict[str, Any]]:
        if count < 0:
            raise ValueError(f"{name} count must be non-negative")
        selected: list[dict[str, Any]] = []
        for row in shuffled(rows, seed):
            identity = sample_key(row)
            if not allow_duplicate_identities and identity in self.sample_ids:
                continue
            mask = identity[2] if identity[0] == "mask" else None
            if not allow_duplicate_identities and mask is not None and mask in self.mask_ids:
                continue
            self.sample_ids.add(identity)
            if mask is not None:
                self.mask_ids.add(mask)
            selected.append(row)
            if len(selected) == count:
                return selected
        raise RuntimeError(
            f"{name} has only {len(selected)} globally disjoint candidates; need {count}. "
            "Increase the raw candidate pool or lower a quota."
        )


def require_multi(rows: list[dict[str, Any]], name: str) -> None:
    bad = [row for row in rows if int(row.get("grounding_instance_count", 0)) < 2]
    if bad:
        raise RuntimeError(f"{name} contains non-multi rows; first source={bad[0].get('source')!r}")


def require_query(rows: list[dict[str, Any]], name: str) -> None:
    for row in rows:
        if not isinstance(row.get("grounding_query"), str) or not row["grounding_query"].strip():
            raise RuntimeError(f"{name} contains a row without grounding_query: {image_key(row)}")


def sanitize_main(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        value = dict(row)
        if value.get("source") != "gres_no_target":
            value["cap_answer"] = None
        result.append(value)
    return result


def load_qa_captions(
    paths: Iterable[Path], *, duplicate_id_policy: str = "error"
) -> dict[str, dict[str, Any]]:
    if duplicate_id_policy not in {"error", "uniquify"}:
        raise ValueError(f"unsupported QA duplicate ID policy: {duplicate_id_policy}")
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                identifier = row.get("dam_source_id")
                caption = row.get("caption")
                if not isinstance(identifier, str) or not identifier:
                    raise ValueError(f"{path}:{line_number} missing dam_source_id")
                if not isinstance(caption, str) or not caption.strip():
                    raise ValueError(f"{path}:{line_number} missing caption")
                if identifier in result:
                    if duplicate_id_policy == "error":
                        raise ValueError(f"duplicate QA manifest ID: {identifier}")
                    previous = json.dumps(result[identifier], ensure_ascii=False, sort_keys=True)
                    current = json.dumps(row, ensure_ascii=False, sort_keys=True)
                    if previous != current:
                        raise ValueError(f"conflicting QA manifest records for ID: {identifier}")
                    continue
                result[identifier] = row
    return result


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    from datasets import Dataset

    if not rows:
        raise ValueError(f"Refusing to write empty parquet: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(path))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_sources(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("source")) for row in rows).items()))


def qa_manifest_rows(rows: list[dict[str, Any]], captions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        identifier = row.get("dam_source_id")
        if not isinstance(identifier, str) or identifier not in captions:
            raise RuntimeError(f"QA parquet row is missing a matching DAM caption: {identifier!r}")
        manifest = dict(captions[identifier])
        manifest["image_path"] = image_key(row)
        manifest["source"] = row.get("source")
        result.append(manifest)
    return result



def uniquify_qa_ids(
    rows: list[dict[str, Any]], manifests: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Give repeated QA copies unique IDs while preserving their source linkage.

    Over-generated DAM shards can contain the same caption record more than once.
    The selected parquet row and its caption manifest must receive the same ID,
    otherwise the downstream generator's one-to-one join is ambiguous.
    """
    if len(rows) != len(manifests):
        raise AssertionError("QA rows and caption manifests have different lengths")
    occurrences: Counter[str] = Counter()
    used: set[str] = set()
    unique_rows: list[dict[str, Any]] = []
    unique_manifests: list[dict[str, Any]] = []
    duplicate_count = 0
    for row, manifest in zip(rows, manifests):
        original_id = row.get("dam_source_id")
        if not isinstance(original_id, str) or not original_id:
            raise ValueError(f"QA parquet row is missing dam_source_id: {original_id!r}")
        if manifest.get("dam_source_id") != original_id:
            raise AssertionError("QA row and manifest IDs are not aligned before uniquification")
        ordinal = occurrences[original_id]
        candidate = original_id if ordinal == 0 else f"{original_id}#duplicate-{ordinal}"
        while candidate in used:
            ordinal += 1
            candidate = f"{original_id}#duplicate-{ordinal}"
        occurrences[original_id] = ordinal + 1
        used.add(candidate)
        if ordinal > 0:
            duplicate_count += 1
        row_copy = dict(row)
        manifest_copy = dict(manifest)
        row_copy["dam_source_id"] = candidate
        manifest_copy["dam_source_id"] = candidate
        unique_rows.append(row_copy)
        unique_manifests.append(manifest_copy)
    return unique_rows, unique_manifests, duplicate_count


def main() -> None:
    args = parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    main_counts = {
        "refcoco_single": args.main_single_count,
        "grefcoco_multi": args.main_multi_count,
        "stuff": args.main_stuff_count,
        "part": args.main_part_count,
        "no_target": args.main_no_target_count,
    }
    direct_counts = {
        "refcoco_single": args.direct_single_count,
        "grefcoco_multi": args.direct_multi_count,
        "stuff": args.direct_stuff_count,
        "part": args.direct_part_count,
        "no_target": args.direct_no_target_count,
    }
    qa_total_count = args.qa_total_count
    if qa_total_count < 0 or qa_total_count != 20_000:
        raise ValueError(f"Expected total DLC-QA count 20000, got {qa_total_count}")
    legacy_qa_counts = None
    if args.qa_stuff_count is not None or args.qa_part_count is not None:
        if args.qa_stuff_count is None or args.qa_part_count is None:
            raise ValueError("--qa-stuff-count and --qa-part-count must be provided together")
        legacy_qa_counts = {"stuff": args.qa_stuff_count, "part": args.qa_part_count}
        if sum(legacy_qa_counts.values()) != qa_total_count:
            raise ValueError(f"Per-source DLC-QA counts must sum to {qa_total_count}, got {legacy_qa_counts}")
    if sum(main_counts.values()) != 40_000 or sum(direct_counts.values()) != 40_000:
        raise ValueError(f"Expected 40k/40k counts, got {main_counts}, {direct_counts}")

    ref_rows = load_parquet(args.refcoco)
    multi_rows = load_parquet(args.grefcoco_multi)
    no_target_rows = load_parquet(args.no_target)
    stuff_rows = load_parquet(args.stuff)
    part_rows = load_parquet(args.part)
    qa_stuff_rows = load_parquet(args.qa_stuff)
    qa_part_rows = load_parquet(args.qa_part)
    qa_captions = load_qa_captions(
        [*args.qa_stuff_manifest, *args.qa_part_manifest],
        duplicate_id_policy=args.qa_duplicate_id_policy,
    )

    by_role = {
        "refcoco_single": source_rows(ref_rows, "refcoco_cycle"),
        "grefcoco_multi": source_rows(multi_rows, "grefcoco_cycle"),
        "no_target": source_rows(no_target_rows, "gres_no_target"),
        "stuff": source_rows(stuff_rows, "cocostuff_cycle"),
        "part": source_rows(part_rows, "paco_part_cycle"),
    }
    qa_by_role = {
        "stuff": source_rows(qa_stuff_rows, "cocostuff_cycle"),
        "part": source_rows(qa_part_rows, "paco_part_cycle"),
    }
    require_multi(by_role["grefcoco_multi"], "gRefCOCO candidates")
    require_query(by_role["refcoco_single"] + by_role["grefcoco_multi"] + by_role["stuff"] + by_role["part"], "positive candidates")
    for rows in qa_by_role.values():
        for row in rows:
            if not row.get("dam_source_id"):
                raise RuntimeError("DLC-QA candidate is missing dam_source_id")

    selector = GlobalSelector()
    main_rows: list[dict[str, Any]] = []
    direct_rows: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []
    # Reserve the QA pool before larger training streams. By default Stuff and
    # Part are one combined 20k quota; optional per-source quotas remain for
    # reproducing older runs. Exact duplicate identities are allowed here,
    # then paired IDs are uniquified before writing the QA artifacts.
    if legacy_qa_counts is None:
        qa_candidates = qa_by_role["stuff"] + qa_by_role["part"]
        qa_rows.extend(
            selector.take(
                qa_candidates,
                qa_total_count,
                name="qa/combined",
                seed=args.seed + 200,
                allow_duplicate_identities=True,
            )
        )
    else:
        for index, role in enumerate(("stuff", "part")):
            qa_rows.extend(
                selector.take(
                    qa_by_role[role],
                    legacy_qa_counts[role],
                    name=f"qa/{role}",
                    seed=args.seed + 200 + index,
                    allow_duplicate_identities=True,
                )
            )
    main_roles = ["refcoco_single", "grefcoco_multi", "stuff", "part", "no_target"]
    direct_roles = ["refcoco_single", "grefcoco_multi", "stuff", "part", "no_target"]
    for index, role in enumerate(main_roles):
        selected = selector.take(by_role[role], main_counts[role], name=f"main/{role}", seed=args.seed + index)
        main_rows.extend(selected)
    for index, role in enumerate(direct_roles):
        selected = selector.take(by_role[role], direct_counts[role], name=f"direct/{role}", seed=args.seed + 100 + index)
        direct_rows.extend(selected)
    random.Random(args.seed).shuffle(main_rows)
    random.Random(args.seed + 1).shuffle(direct_rows)
    random.Random(args.seed + 2).shuffle(qa_rows)
    if len(main_rows) != 40_000 or len(direct_rows) != 40_000 or len(qa_rows) != 20_000:
        raise AssertionError("stream size changed during selection")
    require_multi([row for row in main_rows + direct_rows if row.get("source") == "grefcoco_cycle"], "selected multi rows")
    require_query([row for row in direct_rows if row.get("source") != "gres_no_target"], "selected direct rows")

    direct_positive = [row for row in direct_rows if row.get("source") != "gres_no_target"]
    direct_negative = [row for row in direct_rows if row.get("source") == "gres_no_target"]
    if len(direct_positive) + len(direct_negative) != 40_000:
        raise AssertionError("direct positive/no-target split is incomplete")
    captions_for_qa = qa_manifest_rows(qa_rows, qa_captions)
    qa_duplicate_count = 0
    if args.qa_duplicate_id_policy == "uniquify":
        qa_rows, captions_for_qa, qa_duplicate_count = uniquify_qa_ids(qa_rows, captions_for_qa)
    if len({row["dam_source_id"] for row in qa_rows}) != len(qa_rows):
        raise AssertionError("selected DLC-QA rows still contain duplicate dam_source_id values")
    if len({row["dam_source_id"] for row in captions_for_qa}) != len(captions_for_qa):
        raise AssertionError("selected DLC-QA manifest still contains duplicate dam_source_id values")

    main_path = output / "cyclegrpo_selfsupervised_40k.parquet"
    direct_path = output / "direct_supervised_40k.parquet"
    direct_positive_path = output / "direct_supervised_positive_32k.parquet"
    direct_negative_path = output / "direct_supervised_no_target_8k.parquet"
    qa_path = output / "dlc_qa_20k.parquet"
    qa_manifest_path = output / "dam_caption_manifest_20k.jsonl"
    write_parquet(main_path, sanitize_main(main_rows))
    write_parquet(direct_path, direct_rows)
    write_parquet(direct_positive_path, direct_positive)
    write_parquet(direct_negative_path, direct_negative)
    write_parquet(qa_path, qa_rows)
    write_jsonl(qa_manifest_path, captions_for_qa)

    manifest = {
        "seed": args.seed,
        "qa_duplicate_id_policy": args.qa_duplicate_id_policy,
        "qa_selection": "combined" if legacy_qa_counts is None else "per_source",
        "identity_contract": "unique (resolved image, mask RLE) pairs and unique non-empty mask RLE globally; no-target uses (image, normalized query)",
        "main": {"path": str(main_path.resolve()), "rows": len(main_rows), "source_counts": count_sources(main_rows), "requested": main_counts},
        "direct": {
            "path": str(direct_path.resolve()),
            "positive_path": str(direct_positive_path.resolve()),
            "no_target_path": str(direct_negative_path.resolve()),
            "rows": len(direct_rows),
            "source_counts": count_sources(direct_rows),
            "requested": direct_counts,
        },
        "dlc_qa": {
            "path": str(qa_path.resolve()),
            "manifest_path": str(qa_manifest_path.resolve()),
            "rows": len(qa_rows),
            "duplicate_ids_uniquified": qa_duplicate_count,
            "source_counts": count_sources(qa_rows),
            "requested": (
                {"total": qa_total_count}
                if legacy_qa_counts is None
                else legacy_qa_counts
            ),
        },
        "candidate_inputs": {key: [str(path.resolve()) for path in value] for key, value in {
            "refcoco": args.refcoco,
            "grefcoco_multi": args.grefcoco_multi,
            "no_target": args.no_target,
            "stuff": args.stuff,
            "part": args.part,
            "qa_stuff": args.qa_stuff,
            "qa_part": args.qa_part,
        }.items()},
        "global_unique_masks": len(selector.mask_ids),
        "global_unique_sample_ids": len(selector.sample_ids),
    }
    (output / "scaling_100k_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
