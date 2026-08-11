#!/usr/bin/env python3
"""Convert COCO-Stuff semantic regions into image-and-mask CycleGRPO records."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import Dataset
from PIL import Image

from projects.rl.datasets.prepare_refcoco_rl_dataset import (
    CAPTION_PROMPT,
    build_mask_tokenizer,
    encode_mask_token,
    encode_rle,
)
from projects.rl.datasets.grounding_queries import cocostuff_grounding_query

# Official stuffthingmaps store label id - 1 in PNG pixels.  Official label ids
# 92..182 are stuff, so valid PNG values are 91..181; 255 is void.
STUFF_PIXEL_VALUES = tuple(range(91, 182))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masks-dir", type=Path, required=True, help="COCO-Stuff train2017 PNG directory")
    parser.add_argument("--images-dir", type=Path, required=True, help="COCO 2017 train2017 JPEG directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-tokenizer-path", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--sam2-config-dir",
        type=Path,
        default=Path("projects/transformers/vq_sam2/sam2/sam2_configs"),
    )
    parser.add_argument("--sam2-config", default="sam2.1_hiera_l.yaml")
    parser.add_argument("--max-samples", type=int, default=4_000)
    parser.add_argument("--min-area-ratio", type=float, default=0.01)
    parser.add_argument("--max-area-ratio", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def choose_stuff_mask(
    label_map: np.ndarray,
    class_counts: Counter[int],
    rng: random.Random,
    min_area_ratio: float,
    max_area_ratio: float,
) -> tuple[int, np.ndarray] | None:
    candidates = [int(value) for value in np.unique(label_map) if int(value) in STUFF_PIXEL_VALUES]
    rng.shuffle(candidates)
    # Favor underrepresented semantic classes while retaining randomized ties.
    candidates.sort(key=lambda value: class_counts[value])
    pixel_count = label_map.size
    for value in candidates:
        mask = label_map == value
        ratio = float(mask.sum()) / pixel_count
        if min_area_ratio <= ratio <= max_area_ratio:
            return value, mask
    return None


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if not 0 < args.min_area_ratio <= args.max_area_ratio <= 1:
        raise ValueError("Area ratios must satisfy 0 < min <= max <= 1.")
    for path in (args.masks_dir, args.images_dir, args.mask_tokenizer_path, args.sam2_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)

    mask_paths = sorted(args.masks_dir.glob("*.png"))
    if not mask_paths:
        raise RuntimeError(f"No PNG label maps found in {args.masks_dir}")
    rng = random.Random(args.seed)
    rng.shuffle(mask_paths)

    model = build_mask_tokenizer(args)
    records: list[dict[str, object]] = []
    class_counts: Counter[int] = Counter()
    skipped = 0
    for mask_path in mask_paths:
        image_path = args.images_dir / f"{mask_path.stem}.jpg"
        if not image_path.is_file():
            skipped += 1
            continue
        with Image.open(mask_path) as file:
            label_map = np.asarray(file)
        selected = choose_stuff_mask(
            label_map, class_counts, rng, args.min_area_ratio, args.max_area_ratio
        )
        if selected is None:
            skipped += 1
            continue
        stuff_pixel_value, mask = selected
        with Image.open(image_path) as file:
            image = file.convert("RGB")
        if mask.shape != (image.height, image.width):
            skipped += 1
            continue
        mask_token = encode_mask_token(model, image, mask)
        records.append(
            {
                "images": [str(image_path.resolve())],
                "cap_problem": CAPTION_PROMPT.format(mask_token=mask_token),
                "cap_answer": None,
                "grounding_query": cocostuff_grounding_query(stuff_pixel_value),
                "grounding_query_kind": "semantic_label",
                "seg_problem": None,
                "seg_answer": f"<answer>{mask_token}</answer>",
                "masks": encode_rle(mask),
                "source": "cocostuff_cycle",
            }
        )
        class_counts[stuff_pixel_value] += 1
        if len(records) == args.max_samples:
            break

    if len(records) != args.max_samples:
        raise RuntimeError(
            f"Only created {len(records)} semantic-stuff samples; expected {args.max_samples}. Skipped {skipped}."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(str(args.output))
    manifest = {
        "source": "COCO-Stuff train2017",
        "sample_type": "stuff",
        "samples": len(records),
        "seed": args.seed,
        "stuff_png_value_counts": {str(key): value for key, value in sorted(class_counts.items())},
        "stuff_label_id_range": [92, 182],
        "stuff_png_value_range": [91, 181],
        "grounding_query_kind": "semantic_label",
        "area_ratio_range": [args.min_area_ratio, args.max_area_ratio],
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(json.dumps(manifest, indent=2))
    print(f"Wrote {len(records)} samples to {args.output}")


if __name__ == "__main__":
    main()
