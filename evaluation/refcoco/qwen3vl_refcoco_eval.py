"""Shardable standard RefCOCO evaluation for Qwen3-VL SAMTok checkpoints."""

import argparse
import json
import math
import os
import pickle
import re

import hydra
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torchvision.transforms.functional import to_pil_image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from evaluation.mask_protocol import (
    MASK_PROTOCOLS,
    complete_mask_group_count,
    generation_eos_token_id,
    parse_mask_groups,
)
from projects.transformers.vq_sam2 import SAM2Config, VQ_SAM2, VQ_SAM2Config


class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        return np.array(to_pil_image(image, mode="RGB").resize((self.target_length, self.target_length)))


def parse_args():
    parser = argparse.ArgumentParser(description="Standard RefCOCO cIoU/mIoU evaluation.")
    parser.add_argument("--model_path")
    parser.add_argument("--vq_sam2_path")
    parser.add_argument("--sam2_path")
    parser.add_argument("--refcoco_root")
    parser.add_argument("--split_by", default="unc")
    parser.add_argument("--split", default="val", choices=("val", "testA", "testB"))
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--num_tasks", type=int, default=1)
    parser.add_argument("--gpu_id", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mask_protocol", choices=MASK_PROTOCOLS, default="legacy_union")
    parser.add_argument("--metric_only", action="store_true")
    return parser.parse_args()


def encode_rle(mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def decode_rle(rle: dict) -> np.ndarray:
    return mask_utils.decode(rle).astype(bool)


def decode_annotation(annotation: dict, height: int, width: int) -> np.ndarray:
    segmentation = annotation.get("segmentation", [])
    if isinstance(segmentation, dict):
        rle = mask_utils.frPyObjects(segmentation, height, width) if isinstance(segmentation.get("counts"), list) else segmentation
        decoded = mask_utils.decode(rle)
    elif segmentation:
        decoded = mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width)))
    else:
        return np.zeros((height, width), dtype=bool)
    return decoded.any(axis=-1) if decoded.ndim == 3 else decoded.astype(bool)


def split_matches(ref_split: str, requested_split: str) -> bool:
    if requested_split in ("testA", "testB"):
        return requested_split[-1] in ref_split
    return ref_split == requested_split


def load_samples(root: str, split_by: str, split: str) -> list[dict]:
    with open(os.path.join(root, f"refs({split_by}).p"), "rb") as file:
        refs = pickle.load(file)
    with open(os.path.join(root, "instances.json"), "r") as file:
        instances = json.load(file)
    images = {image["id"]: image for image in instances["images"]}
    annotations = {annotation["id"]: annotation for annotation in instances["annotations"]}
    samples = []
    for ref in refs:
        if not split_matches(ref.get("split", ""), split):
            continue
        image = images.get(ref["image_id"])
        annotation = annotations.get(ref["ann_id"])
        if image is None or annotation is None:
            continue
        target = encode_rle(decode_annotation(annotation, image["height"], image["width"]))
        for sentence in ref["sentences"]:
            samples.append(
                {
                    "case_id": str(sentence["sent_id"]),
                    "image_path": os.path.join(root, "train2014", image["file_name"]),
                    "phrase": sentence["sent"].strip(),
                    "target": target,
                }
            )
    return samples


def parse_mask_codes(text: str, mask_protocol: str) -> list[list[int]]:
    return parse_mask_groups(text, codebook_size=256, protocol=mask_protocol)


def has_matching_protocol(path: str, mask_protocol: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r") as file:
            return json.load(file).get("mask_protocol") == mask_protocol
    except (OSError, json.JSONDecodeError):
        return False


def compute_metrics(save_dir: str) -> dict:
    intersection = 0
    union = 0
    ious = []
    for name in sorted(os.listdir(save_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(save_dir, name), "r") as file:
            record = json.load(file)
        prediction = decode_rle(record["prediction"])
        target = decode_rle(record["target"])
        current_intersection = np.logical_and(prediction, target).sum()
        current_union = np.logical_or(prediction, target).sum()
        intersection += int(current_intersection)
        union += int(current_union)
        ious.append(1.0 if current_union == 0 else float(current_intersection / current_union))
    metrics = {
        "samples": len(ious),
        "cIoU": 0.0 if union == 0 else 100.0 * intersection / union,
        "mIoU": 0.0 if not ious else 100.0 * sum(ious) / len(ious),
    }
    print(json.dumps(metrics, indent=2))
    return metrics


def batched(items: list[dict], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def main():
    args = parse_args()
    if args.metric_only:
        compute_metrics(args.save_dir)
        return
    required = ("model_path", "vq_sam2_path", "sam2_path", "refcoco_root")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError(f"Missing required inference arguments: {', '.join(missing)}.")
    if args.num_tasks <= 0 or not 0 <= args.task_id < args.num_tasks:
        raise ValueError("task_id must be in [0, num_tasks).")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    gpu_id = args.task_id if args.gpu_id < 0 else args.gpu_id
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    os.makedirs(args.save_dir, exist_ok=True)
    all_samples = load_samples(args.refcoco_root, args.split_by, args.split)
    per_task = math.ceil(len(all_samples) / args.num_tasks)
    samples = all_samples[args.task_id * per_task : min((args.task_id + 1) * per_task, len(all_samples))]
    print(
        f"[task {args.task_id}/{args.num_tasks}] evaluating {len(samples)} RefCOCO samples "
        f"on {device} with batch_size={args.batch_size}, mask_protocol={args.mask_protocol}."
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype="auto").to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../projects/transformers/vq_sam2/sam2/sam2_configs"))
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        sam2_config = SAM2Config(cfg_path="sam2.1_hiera_l.yaml", ckpt_path=args.sam2_path)
        vq_config = VQ_SAM2Config(sam2_config=sam2_config, codebook_size=256, codebook_depth=2, shared_codebook=False, latent_dim=256)
    vq_sam2 = VQ_SAM2(vq_config).to(device).eval()
    state = torch.load(args.vq_sam2_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    vq_sam2.load_state_dict({key.removeprefix("hf_model."): value for key, value in state.items()})
    resize = DirectResize(1024)

    pending = [
        sample
        for sample in samples
        if not has_matching_protocol(
            os.path.join(args.save_dir, f"{sample['case_id']}.json"), args.mask_protocol
        )
    ]
    eos_token_id = generation_eos_token_id(model, processor, args.mask_protocol)

    for sample_batch in batched(pending, args.batch_size):
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": sample["image_path"]},
                        {"type": "text", "text": f"Please segment {sample['phrase']} in this image."},
                    ],
                }
            ]
            for sample in sample_batch
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(device)
        with torch.no_grad():
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )
            if eos_token_id is not None:
                generation_kwargs["eos_token_id"] = eos_token_id
            generated = model.generate(**generation_kwargs)
        responses = processor.batch_decode(
            generated[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        for sample, response in zip(sample_batch, responses):
            image = Image.open(sample["image_path"]).convert("RGB")
            width, height = image.size
            codes = parse_mask_codes(response, args.mask_protocol)
            if codes:
                pixels = torch.from_numpy(resize.apply_image(np.array(image))).permute(2, 0, 1).unsqueeze(0).to(vq_sam2.dtype).to(device)
                pixels = pixels.repeat(len(codes), 1, 1, 1)
                with torch.no_grad():
                    masks = vq_sam2.forward_with_codes(pixels, torch.tensor(codes, device=device))
                prediction = torch.nn.functional.interpolate(masks, size=(height, width), mode="bilinear")[:, 0].gt(0.5).any(dim=0).cpu().numpy()
            else:
                prediction = np.zeros((height, width), dtype=bool)
            output_path = os.path.join(args.save_dir, f"{sample['case_id']}.json")
            with open(output_path, "w") as file:
                json.dump(
                    {
                        "target": sample["target"],
                        "prediction": encode_rle(prediction),
                        "response": response,
                        "mask_protocol": args.mask_protocol,
                        "mask_group_count": complete_mask_group_count(response),
                    },
                    file,
                )


if __name__ == "__main__":
    main()
