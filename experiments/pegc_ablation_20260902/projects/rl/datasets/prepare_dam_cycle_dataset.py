#!/usr/bin/env python3
"""Convert DAM region annotations into CycleGRPO image/mask Parquet records."""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset
from PIL import Image

from projects.rl.datasets.prepare_refcoco_rl_dataset import (
    CAPTION_PROMPT,
    build_mask_tokenizer,
    decode_segmentation,
    encode_rle,
    encode_mask_token,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dam-annotations", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--caption-manifest", type=Path, required=True)
    parser.add_argument(
        "--source",
        choices=("cocostuff_cycle", "paco_part_cycle"),
        required=True,
    )
    parser.add_argument(
        "--paco-annotations",
        type=Path,
        help="Official PACO-LVIS train annotation used to retain part annotations only.",
    )
    parser.add_argument("--mask-tokenizer-path", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--sam2-config-dir", type=Path, required=True)
    parser.add_argument("--sam2-config", default="sam2.1_hiera_l.yaml")
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-area-ratio", type=float, default=0.01)
    parser.add_argument("--max-area-ratio", type=float, default=0.90)
    return parser.parse_args()


def iter_annotation_items(payload: Any):
    """Flatten the DAM split's dict-of-lists or list annotation layout."""
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield None, item
        return
    if isinstance(payload, dict):
        for group_key, value in payload.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield str(group_key), item
            elif isinstance(value, dict):
                yield str(group_key), value


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_image(images_dir: Path, image_value: str) -> Path | None:
    raw = Path(image_value)
    basename = raw.name
    candidates = [
        images_dir / raw,
        images_dir / basename,
        images_dir / "train2017" / basename,
        images_dir / "val2017" / basename,
    ]
    return next((path for path in candidates if path.is_file()), None)


def load_paco_part_ids(path: Path) -> set[int]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    part_ids = set()
    for annotation in payload.get("annotations", []):
        ann_id = as_int(annotation.get("id"))
        obj_ann_id = as_int(annotation.get("obj_ann_id", annotation.get("id")))
        if ann_id is not None and obj_ann_id is not None and ann_id != obj_ann_id:
            part_ids.add(ann_id)
    return part_ids


def build_candidates(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Counter[str]]:
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if not 0 < args.min_area_ratio <= args.max_area_ratio <= 1:
        raise ValueError("area ratios must satisfy 0 < min <= max <= 1")
    for path in (args.dam_annotations, args.images_dir, args.mask_tokenizer_path, args.sam2_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.source == "paco_part_cycle" and args.paco_annotations is None:
        raise ValueError("--paco-annotations is required for paco_part_cycle")
    if args.paco_annotations is not None and not args.paco_annotations.is_file():
        raise FileNotFoundError(args.paco_annotations)

    with args.dam_annotations.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    part_ids = load_paco_part_ids(args.paco_annotations) if args.paco_annotations else set()

    candidates: list[dict[str, Any]] = []
    skipped = Counter()
    for group_key, item in iter_annotation_items(payload):
        image_value = item.get("image")
        caption = item.get("caption")
        mask_value = item.get("mask_rle")
        ann_id = as_int(item.get("ann_id"))
        if not image_value:
            skipped["missing_image"] += 1
            continue
        if not isinstance(caption, str) or not caption.strip():
            skipped["missing_caption"] += 1
            continue
        if not isinstance(mask_value, dict):
            skipped["missing_mask_rle"] += 1
            continue
        if args.source == "paco_part_cycle":
            item_obj_id = as_int(item.get("obj_ann_id"))
            is_part = (
                ann_id is not None
                and ann_id in part_ids
                and (item_obj_id is None or item_obj_id != ann_id)
            )
            if not is_part:
                skipped["not_verified_part"] += 1
                continue

        image_path = resolve_image(args.images_dir, str(image_value))
        if image_path is None:
            skipped["missing_image_file"] += 1
            continue
        with Image.open(image_path) as file:
            image = file.convert("RGB")
        mask = decode_segmentation(mask_value, image.height, image.width)
        if mask is None:
            skipped["invalid_mask"] += 1
            continue
        area_ratio = float(mask.mean())
        if not args.min_area_ratio <= area_ratio <= args.max_area_ratio:
            skipped["area_out_of_range"] += 1
            continue

        candidates.append(
            {
                "group_key": group_key,
                "dam_ann_id": ann_id,
                "dam_image": str(image_value),
                "image_path": image_path,
                "image_basename": image_path.name,
                "caption": caption.strip(),
                "mask": mask,
                "area_ratio": area_ratio,
            }
        )
    return candidates, skipped


def main() -> None:
    args = parse_args()
    candidates, skipped = build_candidates(args)
    random.Random(args.seed).shuffle(candidates)

    records: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    used_images: set[str] = set()
    category_counts: Counter[str] = Counter()
    model = build_mask_tokenizer(args)

    for candidate in candidates:
        if candidate["image_basename"] in used_images:
            continue
        image_path = candidate["image_path"]
        with Image.open(image_path) as file:
            image = file.convert("RGB")
        mask_token = encode_mask_token(model, image, candidate["mask"])
        dam_id = (
            f"{args.source}:{candidate['image_basename']}:{candidate['dam_ann_id']}"
        )
        records.append(
            {
                "images": [str(image_path.resolve())],
                "cap_problem": CAPTION_PROMPT.format(mask_token=mask_token),
                "cap_answer": None,
                # DAM captions stay in the sidecar QA manifest, never in the actor prompt.
                "grounding_query": None,
                "seg_problem": None,
                "seg_answer": f"<answer>{mask_token}</answer>",
                "masks": encode_rle(candidate["mask"]),
                "source": args.source,
                "dam_source_id": dam_id,
            }
        )
        manifest_rows.append(
            {
                "dam_source_id": dam_id,
                "source": args.source,
                "dam_ann_id": candidate["dam_ann_id"],
                "dam_image": candidate["dam_image"],
                "image_path": str(image_path.resolve()),
                "caption": candidate["caption"],
                "area_ratio": candidate["area_ratio"],
            }
        )
        used_images.add(candidate["image_basename"])
        category_counts[str(candidate["group_key"])] += 1
        if len(records) == args.max_samples:
            break

    if len(records) != args.max_samples:
        raise RuntimeError(
            f"Only built {len(records)} {args.source} records; expected {args.max_samples}. "
            f"Candidates after filtering: {len(candidates)}; skipped: {dict(skipped)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.caption_manifest.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(str(args.output))
    with args.caption_manifest.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "source": args.source,
        "samples": len(records),
        "unique_images": len(used_images),
        "seed": args.seed,
        "dam_annotations": str(args.dam_annotations.resolve()),
        "output": str(args.output.resolve()),
        "caption_manifest": str(args.caption_manifest.resolve()),
        "area_ratio_range": [args.min_area_ratio, args.max_area_ratio],
        "skipped": dict(skipped),
        "group_counts": dict(category_counts),
    }
    summary_path = args.output.with_suffix(".manifest.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
