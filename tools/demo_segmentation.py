#!/usr/bin/env python3
"""Visual functional test for RefCOCO and ReasonSeg with our SAMTok model.

The console and artifact layout intentionally follow ``tools/demo.py``. Input
is discovered from task-named subdirectories:

  tools/demo/input/<task>/<sample>.<image extension>
  tools/demo/input/<task>/<sample>.json
  tools/demo/output/result/<task>/<sample>.json
  tools/demo/output/visualization/<task>/<sample>.jpg

Use ``--task_name refcoco``, ``--task_name reasonseg``, or the default ``all``.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.mask_protocol import generation_eos_token_id, parse_mask_groups
from evaluation.refcoco.qwen3vl_refcoco_eval import (
    DirectResize,
    decode_rle,
    encode_rle,
)
from projects.transformers.vq_sam2 import SAM2Config, VQ_SAM2, VQ_SAM2Config

DEFAULT_MODEL = (
    "/volume/ybo/xyc/CycleGRPO-OPSD/logs/"
    "cyclegrpo70k_historical_seca_8gpu/evaluation/step_179_cpu_export/"
    "hf_global_step_179"
)
DEFAULT_VQ = "/volume/ybo/xyc/Qwen3-VL-4B-SAMTok/mask_tokenizer_256x2.pth"
DEFAULT_SAM2 = "/volume/ybo/xyc/Qwen3-VL-4B-SAMTok/sam2.1_hiera_large.pt"
DEFAULT_INPUT_ROOT = REPO_ROOT / "tools/demo/input"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tools/demo/output"
TASK_NAMES = ("refcoco", "reasonseg")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
REFCOCO_PROMPT = "Please segment {phrase} in this image."


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用我们的权重运行 RefCOCO 指代分割与 ReasonSeg 推理分割功能可视化。"
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--vq-sam2-path", default=DEFAULT_VQ)
    parser.add_argument("--sam2-path", default=DEFAULT_SAM2)
    parser.add_argument(
        "--input-root",
        default=str(DEFAULT_INPUT_ROOT),
        help="输入根目录；其下必须存在以任务名命名的子目录及图像/同名 JSON。",
    )
    parser.add_argument("--task_name", default="all", help="refcoco、reasonseg 或 all；可用逗号/空格分隔。")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mask-protocol", default="legacy_union", choices=("legacy_union", "first_mask"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def expand_tasks(raw):
    tokens = [item for item in raw.replace(",", " ").split() if item]
    if not tokens or tokens == ["all"]:
        return list(TASK_NAMES)
    selected = []
    for token in tokens:
        if token == "all":
            selected.extend(name for name in TASK_NAMES if name not in selected)
        elif token in TASK_NAMES and token not in selected:
            selected.append(token)
        elif token not in TASK_NAMES:
            raise SystemExit(f"未知 task_name: {token}；可选值：all, refcoco, reasonseg")
    return selected


def font(size):
    for candidate in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if os.path.isfile(candidate):
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def wrap_text(draw, text, text_font, width):
    lines = []
    for paragraph in str(text).splitlines() or [""]:
        current = ""
        for character in paragraph:
            trial = current + character
            if not current or draw.textbbox((0, 0), trial, font=text_font)[2] <= width:
                current = trial
            else:
                lines.append(current)
                current = character
        lines.append(current)
    return lines or [""]


def overlay(image, mask, color, alpha=0.48):
    base = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    tint = np.zeros_like(base)
    tint[:] = color
    base[mask] = (1.0 - alpha) * base[mask] + alpha * tint[mask]
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def visualization(task, image, prediction, target, output_path):
    pred = overlay(image, prediction, (32, 112, 235))
    gt = overlay(image, target, (30, 180, 85))
    display_width = 620
    pred.thumbnail((display_width, 720), Image.Resampling.LANCZOS)
    gt.thumbnail((display_width, 720), Image.Resampling.LANCZOS)
    panel_width = 720
    probe = ImageDraw.Draw(Image.new("RGB", (panel_width, 10)))
    body_font, heading_font, title_font = font(24), font(27), font(32)
    blocks = [
        ("Instruction", task["instruction"]),
        ("Model output", task["response"] or "(empty)"),
        ("Measured result", f"IoU={100 * task['iou']:.2f}% | mask groups={len(task['mask_groups'])}"),
        ("Legend", "Left: blue predicted mask. Middle: green ground-truth mask."),
    ]
    layouts, y = [], 72
    for label, text in blocks:
        lines = wrap_text(probe, text, body_font, panel_width - 40)
        layouts.append((label, lines))
        y += 42 + len(lines) * 34 + 18
    height = max(pred.height, gt.height, y + 20)
    canvas = Image.new("RGB", (pred.width + gt.width + panel_width + 32, height), "white")
    canvas.paste(pred, (0, 0))
    canvas.paste(gt, (pred.width + 16, 0))
    panel_x = pred.width + gt.width + 32
    ImageDraw.Draw(canvas).rectangle((panel_x, 0, canvas.width, height), fill=(248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    draw.text((panel_x + 20, 18), f"{task['task_name']} | {task['title']}", font=title_font, fill=(20, 20, 20))
    y = 72
    for label, lines in layouts:
        draw.text((panel_x + 20, y), label, font=heading_font, fill=(25, 90, 160))
        y += 42
        for line in lines:
            draw.text((panel_x + 20, y), line, font=body_font, fill=(35, 35, 35))
            y += 34
        y += 18
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def load_vq(args, device):
    config_dir = str(REPO_ROOT / "projects/transformers/vq_sam2/sam2/sam2_configs")
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        sam2_config = SAM2Config(cfg_path="sam2.1_hiera_l.yaml", ckpt_path=args.sam2_path)
        config = VQ_SAM2Config(
            sam2_config=sam2_config,
            codebook_size=256,
            codebook_depth=2,
            shared_codebook=False,
            latent_dim=256,
        )
    decoder = VQ_SAM2(config).to(device).eval()
    state = torch.load(args.vq_sam2_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    decoder.load_state_dict({key.removeprefix("hf_model."): value for key, value in state.items()})
    return decoder


def decode_prediction(decoder, image, groups, device):
    if not groups:
        return np.zeros((image.height, image.width), dtype=bool)
    values = DirectResize(1024).apply_image(np.asarray(image))
    values = torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0)
    values = values.to(device=device, dtype=decoder.dtype).repeat(len(groups), 1, 1, 1)
    with torch.inference_mode():
        logits = decoder.forward_with_codes(values, torch.tensor(groups, device=device))
        masks = F.interpolate(logits, (image.height, image.width), mode="bilinear")[:, 0] > 0.5
    return masks.any(dim=0).cpu().numpy()


def generate(model, processor, image_path, prompt, device, max_new_tokens, mask_protocol):
    messages = [[{"role": "user", "content": [
        {"type": "image", "image": str(image_path)},
        {"type": "text", "text": prompt},
    ]}]]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", padding=True,
    ).to(device)
    generation_kwargs = dict(inputs, max_new_tokens=max_new_tokens, do_sample=False)
    eos_token_id = generation_eos_token_id(model, processor, mask_protocol)
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id
    with torch.inference_mode():
        generated = model.generate(**generation_kwargs)
    return processor.batch_decode(
        generated[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def reasonseg_annotation(path, height, width):
    annotation = json.loads(path.read_text(encoding="utf-8"))
    regions = []
    for shape in annotation["shapes"]:
        if shape["label"].lower() == "flag":
            continue
        region = np.zeros((height, width), dtype=np.uint8)
        polygon = np.asarray(shape["points"], dtype=np.int32)
        cv2.fillPoly(region, [polygon], 1)
        regions.append((int(region.sum()), shape))
    mask = np.zeros((height, width), dtype=np.uint8)
    for _, shape in sorted(regions, key=lambda item: item[0], reverse=True):
        polygon = np.asarray(shape["points"], dtype=np.int32)
        value = 255 if "ignore" in shape["label"].lower() else 1
        cv2.polylines(mask, [polygon], True, value, 1)
        cv2.fillPoly(mask, [polygon], value)
    return annotation, mask


def load_refcoco_task(image_path, annotation_path, task_root):
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    query = str(annotation.get("query", annotation.get("phrase", ""))).strip()
    if not query:
        raise ValueError(f"RefCOCO annotation lacks query/phrase: {annotation_path}")
    target_rle = annotation.get("target_rle", annotation.get("target"))
    if not target_rle:
        raise ValueError(f"RefCOCO annotation lacks target_rle/target: {annotation_path}")
    image = Image.open(image_path).convert("RGB")
    target = decode_rle(target_rle)
    if target.shape != (image.height, image.width):
        raise ValueError(f"RefCOCO mask/image size mismatch: {annotation_path}")
    return {
        "task_name": "refcoco",
        "title": "指代分割",
        "case_id": str(annotation.get("case_id", image_path.stem)),
        "relative_stem": image_path.relative_to(task_root).with_suffix(""),
        "image_path": image_path,
        "image": image,
        "query": query,
        "instruction": annotation.get("instruction", REFCOCO_PROMPT.format(phrase=query)),
        "visual_prompt": annotation.get("visual_prompt", "无"),
        "target": target,
        "valid": np.ones((image.height, image.width), dtype=bool),
        "dataset": annotation.get("dataset", "RefCOCO"),
    }


def load_reasonseg_task(image_path, annotation_path, task_root):
    image = Image.open(image_path).convert("RGB")
    annotation, mask = reasonseg_annotation(annotation_path, image.height, image.width)
    query = annotation["text"][0].strip()
    prompt = annotation.get(
        "instruction",
        f"{query} Please output segmentation mask."
        if annotation["is_sentence"]
        else f"Can you segment the {query.lower()} in this image?",
    )
    return {
        "task_name": "reasonseg",
        "title": "推理分割",
        "case_id": str(annotation.get("case_id", image_path.stem)),
        "relative_stem": image_path.relative_to(task_root).with_suffix(""),
        "image_path": image_path,
        "image": image,
        "query": query,
        "instruction": prompt,
        "visual_prompt": annotation.get("visual_prompt", "无"),
        "target": mask == 1,
        "valid": mask != 255,
        "dataset": annotation.get("dataset", "ReasonSeg"),
    }


def discover_tasks(input_root, selected):
    loaders = {"refcoco": load_refcoco_task, "reasonseg": load_reasonseg_task}
    tasks = []
    for task_name in selected:
        task_root = input_root / task_name
        if not task_root.is_dir():
            raise FileNotFoundError(f"任务目录不存在: {task_root}")
        images = sorted(
            path for path in task_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise FileNotFoundError(f"任务目录中没有图像: {task_root}")
        for image_path in images:
            annotation_path = image_path.with_suffix(".json")
            if not annotation_path.is_file():
                raise FileNotFoundError(f"图像缺少同名 JSON 标注: {annotation_path}")
            tasks.append(loaders[task_name](image_path, annotation_path, task_root))
    return tasks


def show_path(path):
    return str(Path(path).resolve())


def print_header(task, args):
    print("\n==========", flush=True)
    print(f"task_name: {task['task_name']}", flush=True)
    print(f"任务: {task['title']}", flush=True)
    print(f"数据集: {task['dataset']}", flush=True)
    print(f"样本ID: {task['case_id']}", flush=True)
    print(f"待处理图像: {show_path(task['image_path'])}", flush=True)
    print(f"模型路径: {show_path(args.model_path)}", flush=True)
    print(f"文本指令: {task['instruction']}", flush=True)
    print(f"视觉提示: {task['visual_prompt']}", flush=True)
    print("==========", flush=True)


def main():
    args = parse_args()
    selected = expand_tasks(args.task_name)
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The SAMTok segmentation demo requires CUDA")
    torch.cuda.set_device(device)
    input_root = Path(args.input_root).resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"待处理图像根目录不存在: {input_root}")
    tasks = discover_tasks(input_root, selected)
    result_dir = Path(args.output_root) / "result"
    visual_dir = Path(args.output_root) / "visualization"

    print(f"模型路径: {show_path(args.model_path)}", flush=True)
    print(f"待处理图像根目录: {input_root}", flush=True)
    print("task_name: " + ", ".join(selected), flush=True)
    print(f"样本数: {len(tasks)}", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype="auto").to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "left"
    decoder = load_vq(args, device)

    for task in tasks:
        print_header(task, args)
        response = generate(
            model,
            processor,
            task["image_path"],
            task["instruction"],
            device,
            args.max_new_tokens,
            args.mask_protocol,
        )
        groups = parse_mask_groups(response, codebook_size=256, protocol=args.mask_protocol)
        prediction = decode_prediction(decoder, task["image"], groups, device)
        valid, target = task["valid"], task["target"]
        intersection = int(np.logical_and(prediction, target)[valid].sum())
        union = int(np.logical_or(prediction, target)[valid].sum())
        iou = 1.0 if union == 0 else intersection / union
        task.update({"response": response, "mask_groups": groups, "iou": iou})
        vis_path = visual_dir / task["task_name"] / task["relative_stem"].with_suffix(".jpg")
        visualization(task, task["image"], prediction, target, vis_path)
        record = {
            "task_name": task["task_name"],
            "title": task["title"],
            "dataset": task["dataset"],
            "case_id": task["case_id"],
            "image": show_path(task["image_path"]),
            "instruction": task["instruction"],
            "visual_prompt": task["visual_prompt"],
            "query": task["query"],
            "output": response,
            "mask_groups": groups,
            "mask_protocol": args.mask_protocol,
            "prediction_rle": encode_rle(prediction),
            "target_rle": encode_rle(target),
            "intersection": intersection,
            "union": union,
            "iou": iou,
            "visualization": show_path(vis_path),
            "model_path": show_path(args.model_path),
        }
        json_path = result_dir / task["task_name"] / task["relative_stem"].with_suffix(".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print("输出文本:", flush=True)
        print(response if response else "（空）", flush=True)
        print(f"解析掩码数量: {len(groups)}", flush=True)
        print(f"单样本IoU: {100 * iou:.2f}%", flush=True)
        print(f"可视化结果: {show_path(vis_path)}", flush=True)
        print(f"结果文件: {show_path(json_path)}", flush=True)


if __name__ == "__main__":
    main()
