#!/usr/bin/env python3
"""Build a CycleGRPO Parquet dataset from a standard RefCOCO training split."""

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torchvision
from datasets import Dataset
from hydra import initialize_config_dir
from PIL import Image
from pycocotools import mask as mask_utils

from projects.transformers.vq_sam2 import SAM2Config, VQ_SAM2, VQ_SAM2Config


CAPTION_PROMPT = "<image>\nProvide a detailed description of this region {mask_token}."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True, help="COCO instances.json")
    parser.add_argument("--refs", type=Path, required=True, help="RefCOCO refs(unc).p file")
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory containing COCO images")
    parser.add_argument("--output", type=Path, required=True, help="Output RL Parquet path")
    parser.add_argument("--mask-tokenizer-path", type=Path, required=True, help="SAMTok VQ-SAM2 state dict")
    parser.add_argument("--sam2-checkpoint", type=Path, required=True, help="SAM2 base checkpoint")
    parser.add_argument(
        "--sam2-config-dir",
        type=Path,
        default=Path("projects/transformers/vq_sam2/sam2/sam2_configs"),
        help="Directory containing sam2.1_hiera_l.yaml",
    )
    parser.add_argument("--sam2-config", default="sam2.1_hiera_l.yaml")
    parser.add_argument("--split", default="train", help="RefCOCO split to convert")
    parser.add_argument("--max-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    state: Any = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported mask-tokenizer checkpoint at {path}")
    return {key.removeprefix("hf_model."): value for key, value in state.items()}


def build_mask_tokenizer(args: argparse.Namespace) -> VQ_SAM2:
    config_dir = args.sam2_config_dir.resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        sam2_config = SAM2Config(cfg_path=args.sam2_config, ckpt_path=str(args.sam2_checkpoint))
    config = VQ_SAM2Config(
        sam2_config=sam2_config,
        codebook_size=256,
        codebook_depth=2,
        shared_codebook=False,
        latent_dim=256,
    )
    model = VQ_SAM2(config)
    model.load_state_dict(load_state_dict(args.mask_tokenizer_path))
    return model.to(args.device).eval()


def decode_segmentation(segmentation: Any, height: int, width: int) -> Optional[np.ndarray]:
    if isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
        decoded = mask_utils.decode(rle)
    elif isinstance(segmentation, list) and segmentation:
        rles = mask_utils.frPyObjects(segmentation, height, width)
        decoded = mask_utils.decode(mask_utils.merge(rles))
    else:
        return None
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=-1)
    decoded = np.asarray(decoded, dtype=np.uint8)
    return decoded if decoded.any() else None


def encode_rle(binary_mask: np.ndarray) -> dict[str, Any]:
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {"size": [int(value) for value in rle["size"]], "counts": counts}


@torch.inference_mode()
def encode_mask_token(model: VQ_SAM2, image: Image.Image, binary_mask: np.ndarray) -> str:
    width, height = image.size
    resized = np.asarray(image.resize((1024, 1024)), dtype=np.uint8).copy()
    pixel_values = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).to(model.device)
    mask = torch.from_numpy(np.ascontiguousarray(binary_mask)).unsqueeze(0).to(model.device)
    boxes = torchvision.ops.masks_to_boxes(mask).to(dtype=torch.float32)
    boxes = boxes / torch.tensor([[width, height, width, height]], device=model.device)
    output = model(
        pixel_values=pixel_values,
        gt_masks=[mask],
        gt_boxes=boxes,
        reconstruct_mask=False,
        freeze_codebook=True,
    )
    quant_codes = output.quant_codes.detach()
    if quant_codes.numel() != 2:
        raise ValueError(
            "Expected exactly two VQ codes for one RefCOCO mask, "
            f"got shape {tuple(quant_codes.shape)}: {quant_codes.cpu().tolist()}"
        )
    # VQ_SAM2 returns codes as (batch, mask_tokens, codebook_depth).
    local_codes = quant_codes.reshape(-1).cpu().tolist()
    global_codes = [depth * 256 + int(code) for depth, code in enumerate(local_codes)]
    return "<|mt_start|>" + "".join(f"<|mt_{code:04d}|>" for code in global_codes) + "<|mt_end|>"


def sentence_text(ref: dict[str, Any]) -> Optional[str]:
    sentences = ref.get("sentences") or []
    if not sentences:
        return None
    text = sentences[0].get("raw") if isinstance(sentences[0], dict) else str(sentences[0])
    return text.strip() or None


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    for path in (args.instances, args.refs, args.images_dir, args.mask_tokenizer_path, args.sam2_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)

    with args.instances.open(encoding="utf-8") as file:
        instances = json.load(file)
    with args.refs.open("rb") as file:
        refs = pickle.load(file, encoding="latin1")

    annotations = {annotation["id"]: annotation for annotation in instances["annotations"]}
    images = {image["id"]: image for image in instances["images"]}
    candidates = [ref for ref in refs if ref.get("split") == args.split]
    candidates.sort(key=lambda ref: int(ref["ref_id"]))
    random.Random(args.seed).shuffle(candidates)

    model = build_mask_tokenizer(args)
    records: list[dict[str, Any]] = []
    skipped = 0
    for ref in candidates:
        annotation = annotations.get(ref.get("ann_id"))
        if annotation is None:
            skipped += 1
            continue
        image_info = images.get(annotation.get("image_id"))
        if image_info is None:
            skipped += 1
            continue
        image_path = args.images_dir / image_info["file_name"]
        if not image_path.is_file():
            skipped += 1
            continue
        with Image.open(image_path) as loaded_image:
            image = loaded_image.convert("RGB")
        binary_mask = decode_segmentation(annotation.get("segmentation"), image.height, image.width)
        if binary_mask is None:
            skipped += 1
            continue
        mask_token = encode_mask_token(model, image, binary_mask)
        records.append(
            {
                "images": [str(image_path.resolve())],
                "cap_problem": CAPTION_PROMPT.format(mask_token=mask_token),
                "cap_answer": sentence_text(ref),
                "seg_problem": None,
                "seg_answer": f"<answer>{mask_token}</answer>",
                "masks": encode_rle(binary_mask),
                "source": "refcoco_cycle",
            }
        )
        if len(records) == args.max_samples:
            break

    if len(records) != args.max_samples:
        raise RuntimeError(
            f"Only created {len(records)} valid samples from split {args.split!r}; "
            f"expected {args.max_samples}. Skipped {skipped} references."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(str(args.output))
    print(f"Wrote {len(records)} samples to {args.output}")
    print(f"Skipped {skipped} invalid or missing-image references")


if __name__ == "__main__":
    main()
