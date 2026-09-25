#!/usr/bin/env python3
"""Student demo of expression segmentation, mask captioning, and ordinary VQA.

Selects the highest-IoU Ours prediction from demo_compare_1000.py output, then
uses that image and its predicted SAMTok mask tokens for reverse captioning and
an image-only VQA question. The generated text and all visualization assets are
saved under the requested output directory.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# Allow absolute-path invocation from a directory outside the repository.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.mask_protocol import parse_mask_groups
from evaluation.refcoco.qwen3vl_refcoco_eval import decode_rle, encode_rle

THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", flags=re.IGNORECASE | re.DOTALL)
OPEN_THINK_TAIL = re.compile(r"<think\b[^>]*>.*$", flags=re.IGNORECASE | re.DOTALL)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours-model", default=(
        "/volume/ybo/xyc/CycleGRPO-OPSD/logs/"
        "cyclegrpo70k_historical_seca_8gpu/evaluation/step_179_cpu_export/"
        "hf_global_step_179"
    ))
    parser.add_argument("--predictions-dir", default="logs/demo_refcoco_1000/Ours/predictions")
    parser.add_argument("--output-dir", default="logs/demo_three_capabilities")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--caption-max-new-tokens", type=int, default=96)
    parser.add_argument("--vqa-max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--vqa-question",
        default="What is the main activity shown in this image? Answer in one concise sentence.",
    )
    return parser.parse_args()


def font(size=20):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if os.path.isfile(candidate):
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def overlay(image, mask, color=(30, 120, 255), alpha=0.42):
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    rgb = np.zeros_like(base)
    rgb[:] = color
    base[mask] = (1 - alpha) * base[mask] + alpha * rgb[mask]
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def wrap_text(draw, text, max_width, text_font):
    words = text.replace("\n", " ").split()
    lines = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if draw.textbbox((0, 0), candidate, font=text_font)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_wrapped_block(draw, x, y, text, max_width, text_font, fill, line_gap=12):
    for paragraph in text.splitlines() or [""]:
        for line in wrap_text(draw, paragraph, max_width, text_font) or [""]:
            draw.text((x, y), line, fill=fill, font=text_font)
            y += draw.textbbox((x, y), line or "Ag", font=text_font)[3] + line_gap
    return y


def resized_width(image, width):
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def create_segmentation_visual(path, image, target, prediction, expression, iou):
    panel_width = 480
    panels = [
        ("INPUT IMAGE", image.convert("RGB")),
        ("GROUND TRUTH", overlay(image, target, color=(30, 200, 80))),
        (f"OURS PREDICTION  |  IoU {iou:.3f}", overlay(image, prediction)),
    ]
    panels = [(label, resized_width(panel, panel_width)) for label, panel in panels]
    panel_height = max(panel.height for _, panel in panels)
    gap = 24
    margin = 36
    width = margin * 2 + panel_width * 3 + gap * 2
    header_height = 174
    footer_height = 230
    canvas = Image.new("RGB", (width, header_height + panel_height + footer_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 20), "1. EXPRESSION SEGMENTATION", fill=(12, 44, 90), font=font(42))
    draw.text((margin, 76), f"Measured mask IoU: {iou:.3f}", fill=(20, 20, 20), font=font(30))
    for index, (label, panel) in enumerate(panels):
        x = margin + index * (panel_width + gap)
        draw.text((x, header_height - 36), label, fill=(15, 45, 90), font=font(23))
        canvas.paste(panel, (x, header_height))
    y = header_height + panel_height + 20
    draw.text((margin, y), "Referring expression", fill=(12, 44, 90), font=font(28))
    draw_wrapped_block(draw, margin, y + 42, expression, width - margin * 2, font(30), (20, 20, 20), line_gap=12)
    canvas.save(path)


def create_caption_visual(path, image, prediction, caption):
    display = resized_width(overlay(image, prediction), 1280)
    margin = 48
    header = 100
    text_font = font(38)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines = wrap_text(probe, caption, 1500 - margin * 2, text_font) or [""]
    line_height = probe.textbbox((0, 0), "Ag", font=text_font)[3] + 16
    text_height = 44 + len(lines) * line_height + 36
    canvas = Image.new("RGB", (1500, header + display.height + text_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 22), "2. MASK-CONDITIONED CAPTIONING", fill=(12, 44, 90), font=font(42))
    canvas.paste(display, ((canvas.width - display.width) // 2, header))
    y = header + display.height + 22
    draw.text((margin, y), "Caption from the predicted mask", fill=(12, 44, 90), font=font(30))
    draw_wrapped_block(draw, margin, y + 48, caption, canvas.width - margin * 2, text_font, (20, 20, 20), line_gap=16)
    canvas.save(path)


def create_vqa_visual(path, image, question, answer):
    display = resized_width(image.convert("RGB"), 1280)
    margin = 48
    header = 100
    question_font = font(32)
    answer_font = font(38)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    question_lines = wrap_text(probe, question, 1500 - margin * 2, question_font) or [""]
    answer_lines = wrap_text(probe, answer, 1500 - margin * 2, answer_font) or [""]
    line_height = probe.textbbox((0, 0), "Ag", font=answer_font)[3] + 16
    text_height = 40 + len(question_lines) * line_height + 28 + len(answer_lines) * line_height + 36
    canvas = Image.new("RGB", (1500, header + display.height + text_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 22), "3. TRADITIONAL VISUAL QUESTION ANSWERING", fill=(12, 44, 90), font=font(40))
    canvas.paste(display, ((canvas.width - display.width) // 2, header))
    y = header + display.height + 22
    draw.text((margin, y), "QUESTION", fill=(12, 44, 90), font=font(28))
    y = draw_wrapped_block(draw, margin, y + 42, question, canvas.width - margin * 2, question_font, (20, 20, 20), line_gap=14)
    y += 14
    draw.text((margin, y), "ANSWER", fill=(12, 44, 90), font=font(28))
    draw_wrapped_block(draw, margin, y + 42, answer, canvas.width - margin * 2, answer_font, (20, 20, 20), line_gap=16)
    canvas.save(path)


def generate_text(model, processor, image_path, prompt, device, max_new_tokens):
    messages = [[
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
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
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    text = processor.batch_decode(
        generated[:, inputs.input_ids.shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    text = clean_model_text(text)
    if not text:
        raise RuntimeError("The model returned no final answer outside its <think> block.")
    return text


def clean_model_text(text):
    """Keep the user-facing answer and discard reasoning tags and their contents."""
    text = THINK_BLOCK.sub(" ", text)
    text = OPEN_THINK_TAIL.sub(" ", text)
    text = re.sub(r"</?think\b[^>]*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|(?:im_end|endoftext)\|>", " ", text)
    return " ".join(text.split()).strip()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This demo requires CUDA")
    torch.cuda.set_device(device)
    pred_dir = Path(args.predictions_dir)
    prediction_files = sorted(pred_dir.glob("*.json"))
    if not prediction_files:
        raise FileNotFoundError(
            f"No prediction JSONs in {pred_dir}. Run demo_compare_1000.py first, or pass a RefCOCO results directory."
        )

    best = None
    for path in prediction_files:
        record = json.loads(path.read_text(encoding="utf-8"))
        if "target" not in record or "prediction" not in record:
            continue
        target = decode_rle(record["target"])
        prediction = decode_rle(record["prediction"])
        intersection = int(np.logical_and(target, prediction).sum())
        union = int(np.logical_or(target, prediction).sum())
        iou = 1.0 if union == 0 else intersection / union
        candidate = {"record": record, "target": target, "prediction": prediction, "iou": iou}
        if best is None or iou > best["iou"]:
            best = candidate
    if best is None:
        raise ValueError(f"No valid target/prediction pairs found under {pred_dir}")

    record = best["record"]
    if not record.get("image_path"):
        raise ValueError(
            "The selected records must include image_path. Use predictions from demo_compare_1000.py, "
            "which stores the expression and image path for reproducibility."
        )
    image_path = record["image_path"]
    expression = record.get("phrase", "")
    response = record.get("response", "")
    groups = parse_mask_groups(response, codebook_size=256, protocol="legacy_union")
    mask_tokens = "".join(
        f"<|mt_start|><|mt_{first:04d}|><|mt_{second + 256:04d}|><|mt_end|>"
        for first, second in groups
    )
    if not mask_tokens:
        raise ValueError("Best-IoU prediction has no valid SAMTok mask group to condition captioning on")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.ours_model, torch_dtype="auto").to(device).eval()
    processor = AutoProcessor.from_pretrained(args.ours_model)
    processor.tokenizer.padding_side = "left"

    caption_prompt = (
        "Describe the object or region marked by the following SAMTok segmentation mask in this image. "
        "Give one concise, visually grounded sentence. Mask: " + mask_tokens
    )
    caption = generate_text(
        model,
        processor,
        image_path,
        caption_prompt,
        device,
        args.caption_max_new_tokens,
    )
    answer = generate_text(
        model,
        processor,
        image_path,
        args.vqa_question,
        device,
        args.vqa_max_new_tokens,
    )

    image = Image.open(image_path).convert("RGB")
    if image.size != best["prediction"].shape[::-1]:
        raise ValueError("Saved prediction dimensions do not match the selected image")
    Image.fromarray((best["prediction"].astype(np.uint8) * 255)).save(output_dir / "predicted_mask.png")
    Image.fromarray((best["target"].astype(np.uint8) * 255)).save(output_dir / "ground_truth_mask.png")
    visualizations = {
        "expression_segmentation": output_dir / "01_expression_segmentation.png",
        "mask_captioning": output_dir / "02_mask_captioning.png",
        "traditional_vqa": output_dir / "03_traditional_vqa.png",
    }
    create_segmentation_visual(
        visualizations["expression_segmentation"], image, best["target"], best["prediction"], expression, best["iou"]
    )
    create_caption_visual(visualizations["mask_captioning"], image, best["prediction"], caption)
    create_vqa_visual(visualizations["traditional_vqa"], image, args.vqa_question, answer)
    result = {
        "case_id": record.get("case_id"),
        "image_path": image_path,
        "expression": expression,
        "segmentation": {
            "prompt": record.get("prompt", "Please segment {expression} in this image."),
            "response": response,
            "IoU": best["iou"],
            "prediction_rle": encode_rle(best["prediction"]),
        },
        "mask_captioning": {"prompt": caption_prompt, "answer": caption},
        "traditional_vqa": {"question": args.vqa_question, "answer": answer},
        "visualizations": {name: str(path) for name, path in visualizations.items()},
    }
    with open(output_dir / "result.json", "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(f"Mask caption: {caption}", flush=True)
    print(f"VQA answer: {answer}", flush=True)
    for capability, path in visualizations.items():
        print(f"{capability}: {path}", flush=True)


if __name__ == "__main__":
    main()
