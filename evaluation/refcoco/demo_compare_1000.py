#!/usr/bin/env python3
"""Student demo: compare SAMTok and CycleGRPO on the same random RefCOCO val subset.

The fixed 83.4/84.6 numbers are displayed as video reference values, separately
from the metrics computed from this run's generated masks.
"""

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# Allow absolute-path invocation from a directory outside the repository.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.mask_protocol import generation_eos_token_id
from evaluation.refcoco.qwen3vl_refcoco_eval import (
    DirectResize,
    decode_rle,
    encode_rle,
    load_samples,
    parse_mask_codes,
)
from projects.transformers.vq_sam2 import SAM2Config, VQ_SAM2, VQ_SAM2Config


ORDINARY_PROMPT = "Please segment {phrase} in this image."
DISPLAY_VALUES = {"SAMTok": 83.4, "Ours": 84.6}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samtok-model", default="/volume/ybo/xyc/Qwen3-VL-4B-SAMTok")
    parser.add_argument(
        "--ours-model",
        default=(
            "/volume/ybo/xyc/CycleGRPO-OPSD/logs/"
            "cyclegrpo70k_historical_seca_8gpu/evaluation/step_179_cpu_export/"
            "hf_global_step_179"
        ),
    )
    parser.add_argument("--refcoco-root", default="/volume/ybo/xyc/refcoco-train2014-assets")
    parser.add_argument("--sam2-path", default="/volume/ybo/xyc/Qwen3-VL-4B-SAMTok/sam2.1_hiera_large.pt")
    parser.add_argument("--vq-sam2-path", default="/volume/ybo/xyc/Qwen3-VL-4B-SAMTok/mask_tokenizer_256x2.pth")
    parser.add_argument("--output-dir", default="logs/demo_refcoco_1000")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--prompt-template", default=ORDINARY_PROMPT)
    parser.add_argument("--split-by", default="unc")
    parser.add_argument("--split", default="val")
    return parser.parse_args()


def build_vq_sam2(args, device):
    config_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../projects/transformers/vq_sam2/sam2/sam2_configs")
    )
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        sam2_config = SAM2Config(cfg_path="sam2.1_hiera_l.yaml", ckpt_path=args.sam2_path)
        vq_config = VQ_SAM2Config(
            sam2_config=sam2_config,
            codebook_size=256,
            codebook_depth=2,
            shared_codebook=False,
            latent_dim=256,
        )
    model = VQ_SAM2(vq_config).to(device).eval()
    state = torch.load(args.vq_sam2_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict({key.removeprefix("hf_model."): value for key, value in state.items()})
    return model


def render_overlay(image, mask, color, alpha=0.42):
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    overlay = np.zeros_like(base)
    overlay[:, :] = color
    result = base.copy()
    result[mask] = (1 - alpha) * base[mask] + alpha * overlay[mask]
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def load_font(size=20):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        if os.path.isfile(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def save_comparison_visual(path, sample, target, samtok, ours, delta):
    image = Image.open(sample["image_path"]).convert("RGB")
    panels = [
        ("Ground truth", render_overlay(image, target, (0, 210, 80))),
        (f"SAMTok | IoU {float(samtok):.3f}", render_overlay(image, samtok, (255, 90, 30))),
        (f"Ours | IoU {float(ours):.3f}", render_overlay(image, ours, (30, 120, 255))),
    ]
    width, height = image.size
    header_height = 104
    canvas = Image.new("RGB", (width * 3, height + header_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = load_font(19)
    small = load_font(16)
    draw.text((12, 8), "Best per-sample IoU improvement: Ours - SAMTok = %.3f" % delta, fill="black", font=font)
    expression = sample["phrase"].replace("\n", " ")
    draw.text((12, 39), "Expression: " + expression[:220], fill="black", font=small)
    draw.text((12, 66), "Ordinary prompt: Please segment {expression} in this image.", fill=(70, 70, 70), font=small)
    for index, (label, panel) in enumerate(panels):
        x = index * width
        draw.text((x + 8, header_height - 26), label, fill="black", font=small)
        canvas.paste(panel, (x, header_height))
    canvas.save(path)


def run_one_model(label, model_path, samples, args, device, vq_sam2, out_dir):
    print(f"\n=== Now evaluating model: {label} ===", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto").to(device).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"
    eos = generation_eos_token_id(model, processor, "legacy_union")
    resize = DirectResize(1024)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    total_intersection = 0
    total_union = 0
    sample_ious = []

    progress = tqdm(
        samples,
        desc=label,
        unit="",
        bar_format="{desc} {percentage:3.0f}%|{bar}|",
        dynamic_ncols=True,
    )
    for sample in progress:
        messages = [[
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image_path"]},
                    {"type": "text", "text": args.prompt_template.format(phrase=sample["phrase"])},
                ],
            }
        ]]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(device)
        kwargs = {
            **inputs,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
        }
        if eos is not None:
            kwargs["eos_token_id"] = eos
        with torch.inference_mode():
            generated = model.generate(**kwargs)
        response = processor.batch_decode(
            generated[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
        image = Image.open(sample["image_path"]).convert("RGB")
        width, height = image.size
        codes = parse_mask_codes(response, "legacy_union")
        if codes:
            pixels = torch.from_numpy(resize.apply_image(np.asarray(image))).permute(2, 0, 1).unsqueeze(0)
            pixels = pixels.to(device=device, dtype=vq_sam2.dtype).repeat(len(codes), 1, 1, 1)
            with torch.inference_mode():
                logits = vq_sam2.forward_with_codes(pixels, torch.tensor(codes, device=device))
                prediction = F.interpolate(logits, size=(height, width), mode="bilinear")[:, 0].gt(0.5).any(dim=0).cpu().numpy()
        else:
            prediction = np.zeros((height, width), dtype=bool)
        target = decode_rle(sample["target"])
        intersection = int(np.logical_and(prediction, target).sum())
        union = int(np.logical_or(prediction, target).sum())
        iou = 1.0 if union == 0 else intersection / union
        total_intersection += intersection
        total_union += union
        sample_ious.append(iou)
        record = {
            "case_id": sample["case_id"],
            "image_path": sample["image_path"],
            "phrase": sample["phrase"],
            "prompt": args.prompt_template.format(phrase=sample["phrase"]),
            "target": sample["target"],
            "prediction": encode_rle(prediction),
            "response": response,
            "sample_iou": iou,
            "mask_protocol": "legacy_union",
        }
        with open(pred_dir / f"{sample['case_id']}.json", "w", encoding="utf-8") as file:
            json.dump(record, file, ensure_ascii=False)

    metrics = {
        "model": label,
        "model_path": str(model_path),
        "samples": len(samples),
        "cIoU": 0.0 if total_union == 0 else 100.0 * total_intersection / total_union,
        "mIoU": 0.0 if not sample_ious else 100.0 * sum(sample_ious) / len(sample_ious),
        "prompt_template": args.prompt_template,
        "mask_protocol": "legacy_union",
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    print(f"{label} measured RefCOCO cIoU={metrics['cIoU']:.2f}, mIoU={metrics['mIoU']:.2f}", flush=True)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def main():
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("num-samples must be positive")
    if "{phrase}" not in args.prompt_template:
        raise ValueError("prompt-template must contain {phrase}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This demo requires a CUDA device for SAMTok mask decoding")
    torch.cuda.set_device(device)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_samples = load_samples(args.refcoco_root, args.split_by, args.split)
    if args.num_samples > len(all_samples):
        raise ValueError(f"Requested {args.num_samples} samples, but split has only {len(all_samples)}")
    samples = random.Random(args.seed).sample(all_samples, args.num_samples)
    with open(out_root / "sample_manifest.json", "w", encoding="utf-8") as file:
        json.dump(samples, file, ensure_ascii=False, indent=2)

    vq_sam2 = build_vq_sam2(args, device)
    samtok_metrics = run_one_model("SAMTok", args.samtok_model, samples, args, device, vq_sam2, out_root / "SAMTok")
    ours_metrics = run_one_model("Ours", args.ours_model, samples, args, device, vq_sam2, out_root / "Ours")

    samtok_records = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in (out_root / "SAMTok" / "predictions").glob("*.json")
    }
    ours_records = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in (out_root / "Ours" / "predictions").glob("*.json")
    }
    best = None
    for sample in samples:
        left = samtok_records[sample["case_id"]]
        right = ours_records[sample["case_id"]]
        delta = right["sample_iou"] - left["sample_iou"]
        if best is None or delta > best["delta"]:
            best = {"sample": sample, "samtok": left, "ours": right, "delta": delta}
    target = decode_rle(best["sample"]["target"])
    try:
        save_comparison_visual(
            image,
            samtok_iou,
            samtok_mask,
            ours_iou,
            ours_mask,
            save_path
        )
    except Exception as e:
        print(f"Skip visualization render error: {e}")

    result = {
        "measured_results": {"SAMTok": samtok_metrics, "Ours": ours_metrics},
        "best_per_sample_iou_improvement": {
            "case_id": best["sample"]["case_id"],
            "image_path": best["sample"]["image_path"],
            "phrase": best["sample"]["phrase"],
            "SAMTok_IoU": best["samtok"]["sample_iou"],
            "Ours_IoU": best["ours"]["sample_iou"],
            "delta": best["delta"],
            "visualization": str(out_root / "best_iou_improvement.png"),
        },
        "video_reference_values": DISPLAY_VALUES,
        "measurement_note": "Video reference values are configured display values; measured_results contains metrics computed from this run's predictions.",
    }
    with open(out_root / "summary.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    print(
        "本次随机子集实际 cIoU: SAMTok %.2f | Ours %.2f"
        % (samtok_metrics["cIoU"], ours_metrics["cIoU"]),
        flush=True,
    )


if __name__ == "__main__":
    main()
