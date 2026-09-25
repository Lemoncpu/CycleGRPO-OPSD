#!/usr/bin/env python3
"""Run one fixed-image Pixel-OPSD recording demo.

The input directory contains three human-readable JSON task descriptions that
all reference the same image.  The script runs referring segmentation,
region captioning, and image-only VQA in one model process and writes one JSON
and one large visualization per task.
"""

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.mask_protocol import parse_mask_groups
from evaluation.refcoco.qwen3vl_refcoco_eval import DirectResize, decode_rle, encode_rle
from projects.transformers.vq_sam2 import SAM2Config, VQ_SAM2, VQ_SAM2Config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=(
            "/volume/ybo/xyc/CycleGRPO-OPSD/logs/"
            "cyclegrpo70k_historical_seca_8gpu/evaluation/step_179_cpu_export/"
            "hf_global_step_179"
        ),
    )
    parser.add_argument("--vq-sam2-path", default="/volume/ybo/xyc/Qwen3-VL-4B-SAMTok/mask_tokenizer_256x2.pth")
    parser.add_argument("--sam2-path", default="/volume/ybo/xyc/Qwen3-VL-4B-SAMTok/sam2.1_hiera_large.pt")
    parser.add_argument("--input-dir", default="logs/pixel_opsd_recording_demo/inputs")
    parser.add_argument("--output-dir", default="logs/pixel_opsd_recording_demo/outputs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--segmentation-max-new-tokens", type=int, default=256)
    parser.add_argument("--caption-max-new-tokens", type=int, default=96)
    parser.add_argument("--vqa-max-new-tokens", type=int, default=96)
    return parser.parse_args()


def font(size):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if os.path.isfile(candidate):
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def wrap_text(draw, text, max_width, text_font):
    lines = []
    current = ""
    for word in str(text).replace("\n", " ").split():
        candidate = word if not current else current + " " + word
        if draw.textbbox((0, 0), candidate, font=text_font)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def draw_text_block(draw, x, y, text, width, text_font, fill=(20, 20, 20), gap=12):
    for line in wrap_text(draw, text, width, text_font):
        draw.text((x, y), line, fill=fill, font=text_font)
        y += draw.textbbox((0, 0), line or "Ag", font=text_font)[3] + gap
    return y


def overlay(image, mask, color=(30, 120, 255), alpha=0.45):
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    tint = np.zeros_like(base)
    tint[:] = color
    base[mask] = (1.0 - alpha) * base[mask] + alpha * tint[mask]
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def resized(image, width):
    return image.resize((width, round(image.height * width / image.width)), Image.Resampling.LANCZOS)


def generate(model, processor, image_path, prompt, device, max_new_tokens, keep_special_tokens):
    messages = [[
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
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
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(
        generated[:, inputs.input_ids.shape[1] :],
        skip_special_tokens=not keep_special_tokens,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def load_vq(args, device):
    config_dir = str(REPO_ROOT / "projects/transformers/vq_sam2/sam2/sam2_configs")
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        sam2_config = SAM2Config(cfg_path="sam2.1_hiera_l.yaml", ckpt_path=args.sam2_path)
        vq_config = VQ_SAM2Config(
            sam2_config=sam2_config,
            codebook_size=256,
            codebook_depth=2,
            shared_codebook=False,
            latent_dim=256,
        )
    vq = VQ_SAM2(vq_config).to(device).eval()
    state = torch.load(args.vq_sam2_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    vq.load_state_dict({key.removeprefix("hf_model."): value for key, value in state.items()})
    return vq


def decode_prediction(vq, image, groups, device):
    if not groups:
        return np.zeros((image.height, image.width), dtype=bool)
    resize = DirectResize(1024)
    pixels = torch.from_numpy(resize.apply_image(np.asarray(image))).permute(2, 0, 1).unsqueeze(0)
    pixels = pixels.to(device=device, dtype=vq.dtype).repeat(len(groups), 1, 1, 1)
    with torch.inference_mode():
        logits = vq.forward_with_codes(pixels, torch.tensor(groups, device=device))
        masks = F.interpolate(logits, size=(image.height, image.width), mode="bilinear")[:, 0].gt(0.5)
    return masks.any(dim=0).cpu().numpy()


def save_board(path, image, title, blocks):
    """Landscape recording board; measure text before allocating the canvas."""
    display = image.copy()
    display.thumbnail((620, 820), Image.Resampling.LANCZOS)
    probe = ImageDraw.Draw(Image.new("RGB", (1600, 1000)))
    layouts = []
    y = 160
    for label, text in blocks:
        lines = wrap_text(probe, text, 800, font(30))
        layouts.append((y, label, lines))
        y += 54 + len(lines) * 44 + 35
    canvas = Image.new("RGB", (1600, max(1000, y + 40)), "#f5f7fb")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 1600, 108), fill="#102847")
    draw.text((42, 30), title, font=font(38), fill="white")
    canvas.paste(display, (40 + (620 - display.width) // 2, 150))
    for y, label, lines in layouts:
        draw.text((720, y), label, font=font(27), fill="#2563a6")
        for line in lines:
            y += 44
            draw.text((720, y), line, font=font(30), fill="#172538")
    canvas.save(path)


def clean_model_text(text):
    text = re.sub(r"<think\b[^>]*>.*?</think\s*>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<think\b[^>]*>.*$", " ", text, flags=re.I | re.S)
    text = re.sub(r"</?think\b[^>]*>|<\|(?:im_end|endoftext)\|>", " ", text, flags=re.I)
    return " ".join(text.split()).strip()


def save_segmentation_visual(path, image, prediction, expression, prompt, response, iou=None):
    metric = "No GT supplied" if iou is None else f"IoU = {100 * iou:.2f}% (one selected case)"
    save_board(path, overlay(image, prediction), "Pixel-OPSD | 1. REFERRING SEGMENTATION", [
        ("Prompt", prompt), ("Predicted mask", "Blue overlay: model prediction."),
        ("Measured result", metric),
        ("Generated mask tokens", " ".join(re.findall(r"<\|mt_[^|]+\|>", response))),
    ])


def save_caption_visual(path, image, mask, prompt, caption):
    save_board(path, overlay(image, mask), "Pixel-OPSD | 2. REGION CAPTIONING", [
        ("Input prompt", prompt), ("Model caption", caption),
        ("Input region", "Blue overlay: the supplied SAMTok region."),
    ])


def save_vqa_visual(path, image, prompt, answer):
    save_board(path, image, "Pixel-OPSD | 3. GENERAL VQA", [
        ("Question", prompt), ("Model answer", answer),
        ("Input", "Original image + question; no region mask."),
    ])


def save_input_boards(input_dir, tasks, image):
    for index, key, title in ((1, "segmentation", "REFERRING SEGMENTATION"),
                              (2, "region_captioning", "REGION CAPTIONING"),
                              (3, "general_vqa", "GENERAL VQA")):
        task = tasks[key]
        shown = image
        blocks = [("Test case", "RefCOCO selected case 115981 | same image for all 3 tasks"),
                  ("Prompt", task["prompt"])]
        if key == "region_captioning":
            shown = overlay(image, decode_rle(task["region_mask_rle"]))
            blocks.append(("Region input", "Blue overlay shows the supplied mask; its SAMTok tokens are included in the prompt."))
        name = "referring_segmentation" if key == "segmentation" else key
        save_board(input_dir / f"{index:02d}_{name}_input.png", shown,
                   f"Pixel-OPSD | INPUT {index}: {title}", blocks)


def load_task(path, input_dir):
    with path.open(encoding="utf-8") as file:
        task = json.load(file)
    task["_path"] = str(path)
    task["image_path"] = str((input_dir / task["image"]).resolve())
    return task


def main():
    args = parse_args()
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This recording demo requires a CUDA device")
    torch.cuda.set_device(device)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = {
        "segmentation": load_task(input_dir / "01_referring_segmentation.json", input_dir),
        "region_captioning": load_task(input_dir / "02_region_captioning.json", input_dir),
        "general_vqa": load_task(input_dir / "03_general_vqa.json", input_dir),
    }
    image_paths = {task["image_path"] for task in tasks.values()}
    if len(image_paths) != 1:
        raise ValueError("The three task files must reference one shared image")
    image_path = Path(next(iter(image_paths)))
    image = Image.open(image_path).convert("RGB")
    save_input_boards(input_dir, tasks, image)
    print("Evaluation set: 1 selected image, 3 task inputs (qualitative demo).", flush=True)
    print(f"Loading model: {args.model_path}", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype="auto").to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "left"
    vq = load_vq(args, device)

    print("[1/3] Running referring segmentation...", flush=True)
    seg = tasks["segmentation"]
    seg_response = generate(model, processor, image_path, seg["prompt"], device, args.segmentation_max_new_tokens, True)
    groups = parse_mask_groups(seg_response, codebook_size=256, protocol=seg.get("mask_protocol", "legacy_union"))
    prediction = decode_prediction(vq, image, groups, device)
    if not groups or not prediction.any():
        raise RuntimeError("Segmentation produced no nonempty legal mask")
    Image.fromarray(prediction.astype(np.uint8) * 255).save(output_dir / "01_predicted_mask.png")
    iou = None
    if seg.get("ground_truth_mask"):
        ground_truth = np.asarray(Image.open(input_dir / seg["ground_truth_mask"]).convert("L")) > 127
        if ground_truth.shape != prediction.shape:
            raise ValueError("ground_truth_mask dimensions do not match the model prediction")
        intersection = np.logical_and(prediction, ground_truth).sum()
        union = np.logical_or(prediction, ground_truth).sum()
        iou = 1.0 if union == 0 else float(intersection / union)
        Image.fromarray(ground_truth.astype(np.uint8) * 255).save(output_dir / "01_ground_truth_mask.png")
    seg_result = {
        "task": "referring_segmentation",
        "image": str(image_path),
        "prompt": seg["prompt"],
        "response": seg_response,
        "mask_groups": groups,
        "mask_protocol": seg.get("mask_protocol", "legacy_union"),
        "prediction_rle": encode_rle(prediction),
        "iou": iou,
    }
    (output_dir / "01_referring_segmentation.json").write_text(json.dumps(seg_result, ensure_ascii=False, indent=2), encoding="utf-8")
    save_segmentation_visual(
        output_dir / "01_referring_segmentation.png",
        image,
        prediction,
        seg["expression"],
        seg["prompt"],
        seg_response,
        iou,
    )

    print("[2/3] Running region captioning...", flush=True)
    cap = tasks["region_captioning"]
    caption = generate(model, processor, image_path, cap["prompt"], device, args.caption_max_new_tokens, False)
    raw_caption = caption
    caption = clean_model_text(caption)
    if not caption:
        raise RuntimeError("Region captioning returned no final answer")
    region_mask = decode_rle(cap["region_mask_rle"]) if "region_mask_rle" in cap else prediction
    cap_result = {
        "task": "region_captioning",
        "image": str(image_path),
        "region_mask": cap.get("region_mask", "01_region_mask.png"),
        "prompt": cap["prompt"],
        "answer": caption,
        "raw_response": raw_caption,
    }
    (output_dir / "02_region_captioning.json").write_text(json.dumps(cap_result, ensure_ascii=False, indent=2), encoding="utf-8")
    save_caption_visual(output_dir / "02_region_captioning.png", image, region_mask, cap["prompt"], caption)

    print("[3/3] Running general VQA...", flush=True)
    vqa = tasks["general_vqa"]
    answer = generate(model, processor, image_path, vqa["prompt"], device, args.vqa_max_new_tokens, False)
    raw_answer = answer
    answer = clean_model_text(answer)
    if not answer:
        raise RuntimeError("General VQA returned no final answer")
    vqa_result = {
        "task": "general_vqa",
        "image": str(image_path),
        "prompt": vqa["prompt"],
        "answer": answer,
        "raw_response": raw_answer,
    }
    (output_dir / "03_general_vqa.json").write_text(json.dumps(vqa_result, ensure_ascii=False, indent=2), encoding="utf-8")
    save_vqa_visual(output_dir / "03_general_vqa.png", image, vqa["prompt"], answer)

    elapsed = time.perf_counter() - started
    summary = {
        "model_checkpoint": str(Path(args.model_path).resolve()),
        "started_at_utc": started_at,
        "elapsed_seconds": elapsed,
        "evaluation_scope": "One selected case, three tasks; text nonempty checks are not accuracy metrics.",
        "image": str(image_path),
        "metrics": {
            "segmentation_iou": iou,
            "segmentation_mask_groups": len(groups),
            "region_captioning_nonempty": bool(caption),
            "general_vqa_nonempty": bool(answer),
        },
        "tasks": [seg_result, cap_result, vqa_result],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    iou_text = "n/a" if iou is None else f"{iou:.4f}"
    print("=== Pixel-OPSD three-task evaluation complete ===", flush=True)
    print(f"SCOPE selected_images=1 task_inputs=3 completed_tasks=3 elapsed_seconds={elapsed:.1f}", flush=True)
    print("REPORT: 1 quality metric (mask IoU); 2 text-output checks (not accuracy).", flush=True)
    print(f"METRIC referring_segmentation: IoU={iou_text} mask_groups={len(groups)} foreground_pixels={int(prediction.sum())}", flush=True)
    print(f"CHECK region_captioning: nonempty={bool(caption)}", flush=True)
    print(f"CHECK general_vqa: nonempty={bool(answer)}", flush=True)
    print(f"REGION_CAPTIONING answer={caption}", flush=True)
    print(f"GENERAL_VQA answer={answer}", flush=True)
    report = (
        f"Pixel-OPSD | LIVE INFERENCE RESULTS\n"
        f"Started (UTC): {started_at}\nCheckpoint: {args.model_path}\n"
        f"Scope: 1 selected image, 3 tasks; not a full benchmark.\n"
        f"Completed tasks: 3/3 | Elapsed: {elapsed:.1f}s\n\n"
        f"[QUALITY METRIC] Segmentation IoU: {iou_text}\n"
        f"[OUTPUT CHECK] Region captioning nonempty: {bool(caption)}\n"
        f"[OUTPUT CHECK] General VQA nonempty: {bool(answer)}\n\n"
        f"Region caption: {caption}\n\nGeneral VQA: {answer}\n"
    )
    (output_dir / "results.txt").write_text(report, encoding="utf-8")
    for name in ("01_referring_segmentation.png", "02_region_captioning.png", "03_general_vqa.png"):
        print(f"OUTPUT {output_dir / name}", flush=True)


if __name__ == "__main__":
    main()
