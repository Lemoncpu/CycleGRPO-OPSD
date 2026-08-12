#!/usr/bin/env python3
"""Re-score RefCOCO responses using only their first complete SAMTok mask.

The standard evaluator unions every valid mask group emitted in a response.
This diagnostic never changes those saved predictions: it reuses their GT RLE
and response text, decodes only the first valid two-code group, and writes a
separate prediction directory plus aggregate metrics.
"""

import argparse
import json
import os
import re
from pathlib import Path

MASK_PATTERN = re.compile(
    r"<\|mt_start\|><\|mt_(\d{4})\|><\|mt_(\d{4})\|><\|mt_end\|>"
)


class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image):
        import numpy as np
        from torchvision.transforms.functional import to_pil_image

        return np.array(to_pil_image(image, mode="RGB").resize((self.target_length, self.target_length)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Existing RefCOCO per-case JSON directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Separate first-mask prediction directory.")
    parser.add_argument("--refcoco-root", type=Path, required=True)
    parser.add_argument("--vq-sam2-path", type=Path, required=True)
    parser.add_argument("--sam2-path", type=Path, required=True)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=1)
    parser.add_argument("--gpu-id", type=int, default=-1)
    parser.add_argument("--metric-only", action="store_true")
    return parser.parse_args()


def encode_rle(mask) -> dict:
    import numpy as np
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def decode_rle(rle: dict):
    from pycocotools import mask as mask_utils

    return mask_utils.decode(rle).astype(bool)


def first_mask_codes(response: str) -> list[int] | None:
    """Return the first complete valid depth-2 SAMTok group, if present."""
    for first_token, second_token in MASK_PATTERN.findall(response):
        first, second = int(first_token), int(second_token) - 256
        if 0 <= first < 256 and 0 <= second < 256:
            return [first, second]
    return None


def compute_metrics(output_dir: Path) -> dict:
    import numpy as np

    intersection = 0
    union = 0
    ious = []
    first_group_count = 0
    for path in sorted(output_dir.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            row = json.load(handle)
        prediction = decode_rle(row["prediction"])
        target = decode_rle(row["target"])
        current_intersection = np.logical_and(prediction, target).sum()
        current_union = np.logical_or(prediction, target).sum()
        intersection += int(current_intersection)
        union += int(current_union)
        ious.append(1.0 if current_union == 0 else float(current_intersection / current_union))
        first_group_count += bool(row.get("first_mask_found"))
    metrics = {
        "samples": len(ious),
        "first_mask_found": first_group_count,
        "first_mask_missing": len(ious) - first_group_count,
        "cIoU": 0.0 if union == 0 else 100.0 * intersection / union,
        "mIoU": 0.0 if not ious else 100.0 * sum(ious) / len(ious),
    }
    print(json.dumps(metrics, indent=2))
    return metrics


def load_image_path_by_case_id(refcoco_root: Path) -> dict[str, str]:
    import pickle

    with (refcoco_root / "refs(unc).p").open("rb") as handle:
        refs = pickle.load(handle)
    with (refcoco_root / "instances.json").open(encoding="utf-8") as handle:
        instances = json.load(handle)
    images = {image["id"]: image for image in instances["images"]}
    paths = {}
    for ref in refs:
        image = images.get(ref.get("image_id"))
        if image is None:
            continue
        path = str(refcoco_root / "train2014" / image["file_name"])
        for sentence in ref.get("sentences", []):
            paths[str(sentence["sent_id"])] = path
    return paths


def build_vq_sam2(args: argparse.Namespace, device):
    import hydra
    import torch

    from projects.transformers.vq_sam2 import SAM2Config, VQ_SAM2, VQ_SAM2Config

    config_dir = Path(__file__).resolve().parents[2] / "projects/transformers/vq_sam2/sam2/sam2_configs"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        sam2_config = SAM2Config(cfg_path="sam2.1_hiera_l.yaml", ckpt_path=str(args.sam2_path))
        config = VQ_SAM2Config(
            sam2_config=sam2_config,
            codebook_size=256,
            codebook_depth=2,
            shared_codebook=False,
            latent_dim=256,
        )
    model = VQ_SAM2(config).to(device).eval()
    state = torch.load(args.vq_sam2_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict({key.removeprefix("hf_model."): value for key, value in state.items()})
    return model


def main() -> None:
    import numpy as np
    import torch
    from PIL import Image

    args = parse_args()
    if args.metric_only:
        metrics = compute_metrics(args.output_dir)
        metrics_path = args.output_dir.parent / f"{args.output_dir.name}_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return
    if args.num_tasks <= 0 or not 0 <= args.task_id < args.num_tasks:
        raise ValueError("task_id must be in [0, num_tasks).")
    for path in (args.input_dir, args.refcoco_root, args.vq_sam2_path, args.sam2_path):
        if not path.exists():
            raise FileNotFoundError(path)

    gpu_id = args.task_id if args.gpu_id < 0 else args.gpu_id
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    input_paths = sorted(args.input_dir.glob("*.json"))
    chunk_size = (len(input_paths) + args.num_tasks - 1) // args.num_tasks
    paths = input_paths[args.task_id * chunk_size : (args.task_id + 1) * chunk_size]
    print(f"[task {args.task_id}/{args.num_tasks}] rescoring {len(paths)} of {len(input_paths)} responses.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = load_image_path_by_case_id(args.refcoco_root)
    model = build_vq_sam2(args, device)
    resize = DirectResize(1024)

    for input_path in paths:
        output_path = args.output_dir / input_path.name
        if output_path.exists():
            continue
        with input_path.open(encoding="utf-8") as handle:
            row = json.load(handle)
        target = row["target"]
        codes = first_mask_codes(str(row.get("response", "")))
        height, width = target["size"]
        prediction = np.zeros((height, width), dtype=bool)
        if codes is not None:
            image_path = image_paths.get(input_path.stem)
            if image_path is None:
                raise KeyError(f"No RefCOCO image for case ID {input_path.stem!r}.")
            with Image.open(image_path) as file:
                image = file.convert("RGB")
            pixels = torch.from_numpy(resize.apply_image(np.array(image))).permute(2, 0, 1)
            pixels = pixels.unsqueeze(0).to(model.dtype).to(device)
            with torch.no_grad():
                masks = model.forward_with_codes(pixels, torch.tensor([codes], device=device))
            prediction = torch.nn.functional.interpolate(
                masks, size=(height, width), mode="bilinear"
            )[0, 0].gt(0.5).cpu().numpy()
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "target": target,
                    "prediction": encode_rle(prediction),
                    "first_mask_found": codes is not None,
                },
                handle,
            )


if __name__ == "__main__":
    main()
