#!/usr/bin/env python3
"""Create a reproducible gRefCOCO CycleGRPO mix with positive and no-target samples.

gRefCOCO is a COCO-instance GRES dataset: it supports single-target,
multi-target, and no-target expressions, but it does not provide object-part
masks. Positive samples become ``grefcoco_cycle`` records; no-target samples
remain ``gres_no_target`` records and use the existing rejection reward.
"""

import argparse
import json
import random
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


# Match the standard RefCOCO/GRES evaluation instruction. The no-target reward
# still requires the response to reject the absent expression as "No target.".
NO_TARGET_PROMPT = "<image>\nPlease segment {expression} in this image."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True, help="COCO instances.json")
    parser.add_argument("--grefs", type=Path, required=True, help="grefs(unc).json")
    parser.add_argument("--images-dir", type=Path, required=True, help="COCO train2014 image directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask-tokenizer-path", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--sam2-config-dir",
        type=Path,
        default=Path("projects/transformers/vq_sam2/sam2/sam2_configs"),
    )
    parser.add_argument("--sam2-config", default="sam2.1_hiera_l.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--positive-samples", type=int, default=9000)
    parser.add_argument("--no-target-samples", type=int, default=1000)
    parser.add_argument(
        "--single-fraction",
        type=float,
        default=0.5,
        help="Fraction of positive samples whose target has exactly one instance.",
    )
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def annotation_ids(ref: dict[str, Any]) -> list[int]:
    values = ref.get("ann_id", [])
    if isinstance(values, int):
        values = [values]
    return [int(value) for value in values]


def expression(ref: dict[str, Any]) -> str:
    sentences = ref.get("sentences") or []
    if not sentences:
        raise ValueError(f"gRefCOCO reference {ref.get('ref_id')} has no sentence.")
    value = sentences[0].get("raw") if isinstance(sentences[0], dict) else str(sentences[0])
    value = value.strip()
    if not value:
        raise ValueError(f"gRefCOCO reference {ref.get('ref_id')} has an empty sentence.")
    return value


def union_mask(
    annotation_ids_for_ref: list[int], annotations: dict[int, dict[str, Any]], image: Image.Image
) -> np.ndarray:
    masks = []
    for annotation_id in annotation_ids_for_ref:
        annotation = annotations.get(annotation_id)
        if annotation is None:
            raise KeyError(f"Missing COCO annotation {annotation_id}.")
        decoded = decode_segmentation(annotation.get("segmentation"), image.height, image.width)
        if decoded is None:
            raise ValueError(f"Annotation {annotation_id} has an empty or invalid mask.")
        masks.append(decoded.astype(bool))
    if not masks:
        raise ValueError("A positive gRefCOCO reference must have at least one target annotation.")
    return np.logical_or.reduce(masks)


def image_path_for_ref(
    ref: dict[str, Any], images: dict[int, dict[str, Any]], images_dir: Path
) -> Path:
    image_info = images.get(int(ref["image_id"]))
    if image_info is None:
        raise KeyError(f"Missing COCO image {ref['image_id']}.")
    image_path = images_dir / image_info["file_name"]
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    return image_path


def select_records(
    refs: list[dict[str, Any]],
    *,
    positive_samples: int,
    no_target_samples: int,
    single_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if positive_samples <= 0 or no_target_samples <= 0:
        raise ValueError("Positive and no-target sample counts must be positive.")
    if not 0.0 <= single_fraction <= 1.0:
        raise ValueError("--single-fraction must be in [0, 1].")
    single = [ref for ref in refs if len(annotation_ids(ref)) == 1 and annotation_ids(ref)[0] >= 0]
    multiple = [ref for ref in refs if len(annotation_ids(ref)) > 1 and all(x >= 0 for x in annotation_ids(ref))]
    no_target = [ref for ref in refs if annotation_ids(ref) == [-1]]
    rng = random.Random(seed)
    for group in (single, multiple, no_target):
        group.sort(key=lambda ref: int(ref["ref_id"]))
        rng.shuffle(group)
    single_count = round(positive_samples * single_fraction)
    multiple_count = positive_samples - single_count
    if len(single) < single_count or len(multiple) < multiple_count or len(no_target) < no_target_samples:
        raise RuntimeError(
            "gRefCOCO train split does not contain enough requested records: "
            f"single={len(single)}/{single_count}, multiple={len(multiple)}/{multiple_count}, "
            f"no_target={len(no_target)}/{no_target_samples}."
        )
    return single[:single_count], multiple[:multiple_count], no_target[:no_target_samples]


def main() -> None:
    args = parse_args()
    for path in (
        args.instances,
        args.grefs,
        args.images_dir,
        args.mask_tokenizer_path,
        args.sam2_checkpoint,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    with args.instances.open(encoding="utf-8") as file:
        instance_data = json.load(file)
    with args.grefs.open(encoding="utf-8") as file:
        refs = [record for record in json.load(file) if record.get("split") == args.split]
    annotations = {int(record["id"]): record for record in instance_data["annotations"]}
    images = {int(record["id"]): record for record in instance_data["images"]}
    single_refs, multiple_refs, no_target_refs = select_records(
        refs,
        positive_samples=args.positive_samples,
        no_target_samples=args.no_target_samples,
        single_fraction=args.single_fraction,
        seed=args.seed,
    )

    model = build_mask_tokenizer(args)
    positive_records: list[dict[str, Any]] = []
    for ref in single_refs + multiple_refs:
        path = image_path_for_ref(ref, images, args.images_dir)
        with Image.open(path) as image_file:
            image = image_file.convert("RGB")
        binary_mask = union_mask(annotation_ids(ref), annotations, image)
        mask_token = encode_mask_token(model, image, binary_mask)
        positive_records.append(
            {
                "images": [str(path.resolve())],
                "cap_problem": CAPTION_PROMPT.format(mask_token=mask_token),
                # Positive cycle records are image-and-mask only. Keeping the
                # source expression would retain text supervision in the data.
                "cap_answer": None,
                "grounding_query": expression(ref),
                "grounding_instance_count": len(annotation_ids(ref)),
                "seg_problem": None,
                "seg_answer": f"<answer>{mask_token}</answer>",
                "masks": encode_rle(binary_mask),
                "source": "grefcoco_cycle",
            }
        )

    no_target_records: list[dict[str, Any]] = []
    for ref in no_target_refs:
        path = image_path_for_ref(ref, images, args.images_dir)
        no_target_records.append(
            {
                "images": [str(path.resolve())],
                "cap_problem": NO_TARGET_PROMPT.format(expression=expression(ref)),
                "cap_answer": None,
                "grounding_query": expression(ref),
                "grounding_instance_count": 0,
                "seg_problem": None,
                "seg_answer": "<answer>No target.</answer>",
                "masks": None,
                "source": "gres_no_target",
            }
        )

    combined_records = positive_records + no_target_records
    random.Random(args.seed).shuffle(combined_records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"grefcoco_{args.split}_{len(positive_records)}pos_{len(no_target_records)}notarget_seed{args.seed}"
    Dataset.from_list(positive_records).to_parquet(str(args.output_dir / f"{prefix}_positive.parquet"))
    Dataset.from_list(no_target_records).to_parquet(str(args.output_dir / f"{prefix}_no_target.parquet"))
    Dataset.from_list(combined_records).to_parquet(str(args.output_dir / f"{prefix}_combined.parquet"))
    manifest = {
        "split": args.split,
        "seed": args.seed,
        "positive": len(positive_records),
        "single_instance": len(single_refs),
        "multi_instance": len(multiple_refs),
        "no_target": len(no_target_records),
        "part_instance": 0,
        "part_note": "gRefCOCO uses COCO instance masks and has no part-mask labels.",
    }
    with (args.output_dir / f"{prefix}_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(json.dumps(manifest, indent=2))
    print(f"Combined training parquet: {args.output_dir / f'{prefix}_combined.parquet'}")


if __name__ == "__main__":
    main()
