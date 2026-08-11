#!/usr/bin/env python3
"""Convert PACO-LVIS object-part masks into CycleGRPO training records.

PACO annotations identify an object annotation by ``id == obj_ann_id``.  This
tool only accepts the complementary annotations, so the output contains true
part masks rather than parent-object masks.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset
from PIL import Image

from projects.rl.datasets.prepare_refcoco_rl_dataset import (
    CAPTION_PROMPT,
    build_mask_tokenizer,
    decode_segmentation,
    encode_mask_token,
    encode_rle,
)
from projects.rl.datasets.grounding_queries import paco_part_grounding_query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="paco_lvis_v1_train.json")
    parser.add_argument("--images-dir", type=Path, required=True, help="COCO 2017 image root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-tokenizer-path", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--sam2-config-dir",
        type=Path,
        default=Path("projects/transformers/vq_sam2/sam2/sam2_configs"),
    )
    parser.add_argument("--sam2-config", default="sam2.1_hiera_l.yaml")
    parser.add_argument("--max-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resolve_image(images_dir: Path, file_name: str) -> Path | None:
    direct = images_dir / file_name
    candidates = [direct]
    if not direct.is_file():
        basename = Path(file_name).name
        candidates.extend([images_dir / "train2017" / basename, images_dir / "val2017" / basename])
    return next((path for path in candidates if path.is_file()), None)


def part_group_key(
    annotation: dict[str, Any],
    annotations_by_id: dict[int, dict[str, Any]],
    part_names: dict[int, str],
    object_names: dict[int, str],
) -> tuple[int, str, str] | None:
    """Return the semantic query key whose masks must share one target."""
    part_name = part_names.get(int(annotation["category_id"]))
    parent = annotations_by_id.get(int(annotation.get("obj_ann_id", -1)))
    parent_name = (
        object_names.get(int(parent["category_id"]))
        if parent is not None and parent.get("category_id") is not None
        else None
    )
    if not part_name or not parent_name:
        return None
    return int(annotation["image_id"]), part_name, parent_name


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    for path in (args.annotations, args.images_dir, args.mask_tokenizer_path, args.sam2_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)

    with args.annotations.open(encoding="utf-8") as file:
        data: dict[str, Any] = json.load(file)
    images = {int(record["id"]): record for record in data["images"]}
    part_names = {int(record["id"]): str(record["name"]) for record in data.get("part_categories", [])}
    object_names = {
        int(record["id"]): str(record["name"])
        for record in (data.get("categories") or data.get("object_categories") or [])
    }
    annotations_by_id = {int(record["id"]): record for record in data["annotations"]}
    if not part_names or not object_names:
        raise ValueError("PACO-LVIS annotation must provide non-empty part_categories and object categories.")
    candidates: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for record in data["annotations"]:
        if int(record["id"]) == int(record.get("obj_ann_id", record["id"])):
            continue
        key = part_group_key(record, annotations_by_id, part_names, object_names)
        if key is not None:
            candidates.setdefault(key, []).append(record)
    candidate_groups = sorted(candidates.items(), key=lambda item: min(int(row["id"]) for row in item[1]))
    random.Random(args.seed).shuffle(candidate_groups)

    model = build_mask_tokenizer(args)
    records: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    used_images: set[int] = set()
    skipped = 0
    for (image_id, part_name, parent_name), annotations in candidate_groups:
        # One part per image avoids a few densely annotated objects dominating the 2k budget.
        if image_id in used_images:
            continue
        image_info = images.get(image_id)
        if image_info is None:
            skipped += 1
            continue
        image_path = resolve_image(args.images_dir, str(image_info["file_name"]))
        if image_path is None:
            skipped += 1
            continue
        with Image.open(image_path) as file:
            image = file.convert("RGB")
        part_masks = [
            decode_segmentation(annotation.get("segmentation"), image.height, image.width)
            for annotation in annotations
        ]
        if any(mask is None for mask in part_masks):
            skipped += 1
            continue
        mask = np.logical_or.reduce([np.asarray(mask, dtype=bool) for mask in part_masks])
        mask_token = encode_mask_token(model, image, mask)
        records.append(
            {
                "images": [str(image_path.resolve())],
                "cap_problem": CAPTION_PROMPT.format(mask_token=mask_token),
                "cap_answer": None,
                "grounding_query": paco_part_grounding_query(part_name, parent_name),
                "grounding_query_kind": "part_label",
                "seg_problem": None,
                "seg_answer": f"<answer>{mask_token}</answer>",
                "masks": encode_rle(mask),
                "source": "paco_part_cycle",
            }
        )
        used_images.add(image_id)
        category_counts[part_name] += 1
        if len(records) == args.max_samples:
            break

    if len(records) != args.max_samples:
        raise RuntimeError(f"Only created {len(records)} part samples; expected {args.max_samples}. Skipped {skipped}.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(str(args.output))
    manifest = {
        "source": "PACO-LVIS train",
        "sample_type": "part",
        "samples": len(records),
        "unique_images": len(used_images),
        "seed": args.seed,
        "part_category_counts": dict(sorted(category_counts.items())),
        "grounding_query_kind": "part_label",
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(json.dumps(manifest, indent=2))
    print(f"Wrote {len(records)} samples to {args.output}")


if __name__ == "__main__":
    main()
