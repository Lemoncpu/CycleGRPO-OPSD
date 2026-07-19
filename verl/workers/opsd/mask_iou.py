import re
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


MASK_TOKEN_PATTERN = re.compile(
    r"<\|mt_start\|><\|mt_(\d{4})\|><\|mt_(\d{4})\|><\|mt_end\|>"
)


def extract_mask_token(text: Optional[str]) -> Optional[str]:
    if not isinstance(text, str) or not text:
        return None
    match = MASK_TOKEN_PATTERN.search(text)
    return match.group(0) if match else None


def parse_mask_codes(text: Optional[str], codebook_size: int = 256, codebook_depth: int = 2) -> Optional[list[int]]:
    token = extract_mask_token(text)
    if token is None:
        return None
    match = MASK_TOKEN_PATTERN.fullmatch(token)
    if match is None:
        return None
    global_codes = [int(value) for value in match.groups()]
    if len(global_codes) != codebook_depth:
        return None
    local_codes = [value - depth * codebook_size for depth, value in enumerate(global_codes)]
    if any(value < 0 or value >= codebook_size for value in local_codes):
        return None
    return local_codes


def compute_binary_iou(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    target = target.to(dtype=torch.bool)
    prediction = prediction.to(dtype=torch.bool)
    if target.shape != prediction.shape:
        raise ValueError(f"IoU masks must have identical shapes, got {target.shape} and {prediction.shape}.")
    reduce_dims = tuple(range(1, target.ndim))
    intersection = torch.logical_and(target, prediction).sum(dim=reduce_dims, dtype=torch.float32)
    union = torch.logical_or(target, prediction).sum(dim=reduce_dims, dtype=torch.float32)
    return torch.where(union > 0, intersection / union, torch.zeros_like(union))


def coerce_raw_mask(value, image_size: tuple[int, int]) -> Optional[torch.Tensor]:
    """Convert common dense, PIL, COCO RLE, or polygon annotations to one 2D mask."""
    if value is None:
        return None
    height, width = image_size
    if isinstance(value, Image.Image):
        value = np.asarray(value)
    if isinstance(value, dict) and "counts" in value:
        from pycocotools import mask as mask_utils

        rle = value
        if isinstance(rle["counts"], list):
            rle = mask_utils.frPyObjects(rle, height, width)
        value = mask_utils.decode(rle)
    elif isinstance(value, (list, tuple)):
        try:
            dense = np.asarray(value)
        except ValueError:
            dense = np.asarray([], dtype=np.uint8)
        is_dense_binary = dense.ndim == 2 and dense.size > 0 and np.isin(dense, [0, 1]).all()
        if is_dense_binary:
            value = dense
        else:
            from pycocotools import mask as mask_utils

            rles = mask_utils.frPyObjects(value, height, width)
            value = mask_utils.decode(mask_utils.merge(rles))
    if isinstance(value, np.ndarray) and value.ndim == 3:
        value = np.any(value, axis=-1)
    try:
        mask = torch.as_tensor(value).squeeze().to(dtype=torch.bool)
    except (TypeError, ValueError, RuntimeError):
        return None
    if mask.ndim != 2:
        return None
    if mask.shape != (height, width):
        mask = F.interpolate(mask[None, None].float(), size=(height, width), mode="nearest")[0, 0].bool()
    return mask


def _load_rgb_image(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, str):
        return Image.open(value).convert("RGB")
    if isinstance(value, dict) and "bytes" in value:
        from io import BytesIO

        return Image.open(BytesIO(value["bytes"])).convert("RGB")
    if isinstance(value, bytes):
        from io import BytesIO

        return Image.open(BytesIO(value)).convert("RGB")
    raise TypeError(f"Unsupported image value: {type(value)!r}")


@torch.inference_mode()
def decode_mask_tokens(
    *,
    vq_sam2,
    image,
    token_texts: Sequence[Optional[str]],
    codebook_size: int = 256,
    codebook_depth: int = 2,
    threshold: float = 0.5,
    decode_batch_size: int = 32,
) -> tuple[list[Optional[torch.Tensor]], tuple[int, int]]:
    if decode_batch_size <= 0:
        raise ValueError("decode_batch_size must be positive.")
    pil_image = _load_rgb_image(image)
    width, height = pil_image.size
    resized = pil_image.resize((1024, 1024))
    pixel_values = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).contiguous()
    pixel_values = pixel_values.unsqueeze(0).to(device=vq_sam2.device, dtype=vq_sam2.dtype)

    parsed = [parse_mask_codes(text, codebook_size, codebook_depth) for text in token_texts]
    valid_indices = [index for index, codes in enumerate(parsed) if codes is not None]
    output: list[Optional[torch.Tensor]] = [None] * len(token_texts)
    if not valid_indices:
        return output, (height, width)

    image_state = None
    if hasattr(vq_sam2, "encode_single_image") and hasattr(vq_sam2, "decode_codes_from_single_image"):
        image_state = vq_sam2.encode_single_image(pixel_values)

    # Decode in bounded chunks while retaining one shared image embedding.
    for start in range(0, len(valid_indices), decode_batch_size):
        batch_indices = valid_indices[start : start + decode_batch_size]
        codes = torch.tensor([parsed[index] for index in batch_indices], device=vq_sam2.device, dtype=torch.long)
        if image_state is not None:
            logits = vq_sam2.decode_codes_from_single_image(image_state, codes)
        elif hasattr(vq_sam2, "forward_codes_single_image"):
            logits = vq_sam2.forward_codes_single_image(pixel_values, codes)
        else:
            repeated_pixels = pixel_values.expand(len(batch_indices), -1, -1, -1)
            logits = vq_sam2.forward_with_codes(repeated_pixels, codes)
        logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
        masks = logits[:, 0] > threshold
        for index, mask in zip(batch_indices, masks):
            output[index] = mask.detach().cpu()
    return output, (height, width)


def mask_summary(mask: Optional[torch.Tensor]) -> dict[str, object]:
    if mask is None:
        return {"empty": True, "area": 0, "area_ratio": 0.0, "bbox": None, "center": None}
    array = mask.detach().cpu().to(dtype=torch.bool)
    height, width = array.shape[-2:]
    ys, xs = torch.where(array)
    area = int(xs.numel())
    if area == 0:
        return {"empty": True, "area": 0, "area_ratio": 0.0, "bbox": None, "center": None}
    return {
        "empty": False,
        "area": area,
        "area_ratio": area / float(max(height * width, 1)),
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "center": [round(float(xs.float().mean()), 2), round(float(ys.float().mean()), 2)],
    }
