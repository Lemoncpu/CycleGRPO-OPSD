#!/usr/bin/env python3
"""Run Griffon-G visual tasks and save demo outputs.

Pass --task_name to run one sample, for example --task_name ocr.
Several names can be separated by commas or spaces. The default is all.

Layout:
  <image-dir>/<sample>/image.jpg     input image
  <image-dir>/<sample>/prompt.jpg    visual prompt, when the sample needs one
  demo/output/result/                one JSON file per sample
  demo/output/visualization/         one composed result image per sample
"""

import argparse
import json
import os
import re
import shutil

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from transformers import CLIPImageProcessor

from griffon.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from griffon.conversation import SeparatorStyle, conv_templates
from griffon.eval.run_griffon import extract, load_image
from griffon.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, tokenizer_image_token
from griffon.model.builder import load_pretrained_model
from griffon.utils import disable_torch_init

ROOT = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(ROOT, "demo")
TASK_ROOT = os.path.join(DEMO, "task")
RESULT_DIR = os.path.join(DEMO, "output", "result")
VIS_DIR = os.path.join(DEMO, "output", "visualization")
COCO = os.path.join(ROOT, "data", "coco", "images", "val2017")
FONT_PATH = os.path.join(ROOT, "Times New Roman.ttf")
MODEL_PATH = os.path.join(ROOT, "Griffon-Unimodel")

DET_PART = "Griffon-G-gemma2-9b"
BOX_PAT = re.compile(r"(\[\d{1,}\.\d{3},\d{1,}\.\d{3},\d{1,}\.\d{3},\d{1,}\.\d{3}\])")

TASK_NAMES = [
    "image_caption",
    "general_vqa",
    "ocr",
    "scene_text_vqa",
    "document_vqa",
    "infographic",
    "region_caption",
    "region_dialogue",
    "object_detection",
    "visual_grounding",
    "rec",
    "reasoning_grounding",
    "visual_referring",
    "co_referring",
    "object_counting",
]

DET_PROMPT = (
    "Examine the image for any objects from the category set. "
    "Report the coordinates of each detected object. "
    "The category set includes {categories}. "
    "Answer in the format: class name-[x1, y1, x2, y2], and concatenate them with &."
)


def ensure_layout():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)


def task_catalog():
    lines = ["可选 task_name（多个用逗号或空格分隔，默认 all）:", "  all"]
    lines.extend(f"  {name}" for name in TASK_NAMES)
    return "\n".join(lines)


def expand_task_names(raw):
    tokens = [item for item in raw.replace(",", " ").split() if item]
    known = set(TASK_NAMES)
    if not tokens or tokens == ["all"]:
        return list(TASK_NAMES)
    selected = []
    unknown = []
    for token in tokens:
        if token == "all":
            for name in TASK_NAMES:
                if name not in selected:
                    selected.append(name)
        elif token in known:
            if token not in selected:
                selected.append(token)
        else:
            unknown.append(token)
    if unknown:
        raise SystemExit("未知 task_name: " + ", ".join(unknown) + "\n" + task_catalog())
    return selected


def resolve_dir(path):
    if os.path.isabs(path):
        return os.path.abspath(path)
    cwd_path = os.path.abspath(path)
    root_path = os.path.abspath(os.path.join(ROOT, path))
    if os.path.isdir(cwd_path):
        return cwd_path
    return root_path


def ensure_file(path, allow_fallback, producer):
    if os.path.exists(path):
        return path
    if not allow_fallback:
        raise SystemExit(f"缺少待处理图像: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    producer(path)
    return path


def copy_coco(image_root, name, filename, allow_fallback):
    dst = os.path.join(image_root, name, "image.jpg")
    return ensure_file(dst, allow_fallback, lambda path: shutil.copy(os.path.join(COCO, filename), path))


def copy_asset(image_root, name, filename, src, allow_fallback):
    dst = os.path.join(image_root, name, filename)
    return ensure_file(dst, allow_fallback, lambda path: shutil.copy(src, path))


def save_rgb(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path, quality=95)
    return path


def make_text_image(lines, size, bg, fg, title=None):
    image = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, 42)
    title_font = ImageFont.truetype(FONT_PATH, 36)
    y = 40
    if title:
        draw.text((40, y), title, fill=(30, 30, 30), font=title_font)
        y += 70
    for line in lines:
        draw.text((40, y), line, fill=fg, font=font)
        y += 64
    return image


def add_task(tasks, selected, task):
    if task["task_name"] in selected:
        tasks.append(task)


def prepare_inputs(image_root, allow_fallback, selected):
    """Build only the samples named by --task_name."""
    tasks = []

    if "image_caption" in selected:
        add_task(tasks, selected, {
            "task_name": "image_caption",
            "title": "全图描述",
            "image": copy_coco(image_root, "image_caption", "000000363840.jpg", allow_fallback),
            "instruction": "Detail what is happening in this image.",
            "draw": None,
        })
    if "general_vqa" in selected:
        add_task(tasks, selected, {
            "task_name": "general_vqa",
            "title": "通用问答",
            "image": copy_coco(image_root, "general_vqa", "000000142472.jpg", allow_fallback),
            "instruction": "What vehicles are visible in this street scene? Answer briefly.",
            "draw": None,
        })

    if "ocr" in selected:
        ocr_path = os.path.join(image_root, "ocr", "image.jpg")
        ensure_file(ocr_path, allow_fallback, lambda path: save_rgb(make_text_image(
            ["The editorial approach of the", "magazine has remained broadly", "unchanged since its founding."],
            (960, 360), (248, 244, 230), (20, 20, 20),
        ), path))
        add_task(tasks, selected, {
            "task_name": "ocr",
            "title": "文档文字",
            "image": ocr_path,
            "instruction": "Please read out the text in the image and return the text.",
            "draw": None,
        })

    if "scene_text_vqa" in selected:
        scene_path = os.path.join(image_root, "scene_text_vqa", "image.jpg")
        ensure_file(scene_path, allow_fallback, lambda path: save_rgb(make_text_image(
            ["COFFEE SHOP", "Open 9 AM - 6 PM", "Latte  $4", "Tea    $3"],
            (900, 420), (32, 74, 120), (255, 255, 255),
        ), path))
        add_task(tasks, selected, {
            "task_name": "scene_text_vqa",
            "title": "场景文字",
            "image": scene_path,
            "instruction": "What is the price of a latte? Answer with the price only.",
            "draw": None,
        })

    if "document_vqa" in selected:
        chart_path = os.path.join(image_root, "document_vqa", "image.jpg")

        def write_chart(path):
            chart = Image.new("RGB", (980, 560), (255, 255, 255))
            draw = ImageDraw.Draw(chart)
            font = ImageFont.truetype(FONT_PATH, 28)
            draw.text((40, 24), "Active users (million)", fill=(20, 20, 20), font=font)
            bars = [("Pinterest", 70, (220, 80, 80)), ("X", 120, (80, 120, 200)), ("TikTok", 180, (80, 160, 90))]
            for i, (label, value, color) in enumerate(bars):
                x = 80 + i * 280
                h = int(value * 2.0)
                draw.rectangle([x, 460 - h, x + 120, 460], fill=color)
                draw.text((x, 470), f"{label}\n{value}", fill=(20, 20, 20), font=font)
            save_rgb(chart, path)

        ensure_file(chart_path, allow_fallback, write_chart)
        add_task(tasks, selected, {
            "task_name": "document_vqa",
            "title": "图表理解",
            "image": chart_path,
            "instruction": "Which platform has the least number of active users?",
            "draw": None,
        })

    if "infographic" in selected:
        info_path = os.path.join(image_root, "infographic", "image.jpg")
        ensure_file(info_path, allow_fallback, lambda path: save_rgb(make_text_image(
            ["CITY FACTS", "Population   2.1 million", "Parks        140", "Founded      1850"],
            (900, 420), (245, 248, 240), (20, 50, 40),
        ), path))
        add_task(tasks, selected, {
            "task_name": "infographic",
            "title": "信息图理解",
            "image": info_path,
            "instruction": "In which year was the city founded? Answer with the year only.",
            "draw": None,
        })

    if "region_caption" in selected or "region_dialogue" in selected:
        reg_image = copy_coco(image_root, "region_caption", "000000363840.jpg", allow_fallback)
        reg_box = [0.429, 0.177, 0.811, 0.738]
        reg_boxes = [{"category_name": "query region", "bbox": reg_box}]
        add_task(tasks, selected, {
            "task_name": "region_caption",
            "title": "区域描述",
            "image": reg_image,
            "instruction": (
                "Please generate a distinguishing description for the region "
                "[0.429, 0.177, 0.811, 0.738] in the image."
            ),
            "draw": "input",
            "input_boxes": reg_boxes,
        })
        add_task(tasks, selected, {
            "task_name": "region_dialogue",
            "title": "局部对话",
            "image": reg_image,
            "instruction": (
                "What is inside the region [0.429, 0.177, 0.811, 0.738]? Answer briefly."
            ),
            "draw": "input",
            "input_boxes": reg_boxes,
        })

    if "object_detection" in selected:
        add_task(tasks, selected, {
            "task_name": "object_detection",
            "title": "类别限定",
            "image": copy_coco(image_root, "object_detection", "000000007574.jpg", allow_fallback),
            "instruction": DET_PROMPT.format(
                categories="bottle, bowl, microwave, oven, refrigerator, sink, vase, wine glass, person, chair"
            ),
            "draw": "output",
        })
    if "visual_grounding" in selected:
        add_task(tasks, selected, {
            "task_name": "visual_grounding",
            "title": "开放类别",
            "image": copy_coco(image_root, "visual_grounding", "000000193926.jpg", allow_fallback),
            "instruction": (
                "Examine the image for any object from the person, bicycle, and banana. "
                "Answer in the format: class name-[x1, y1, x2, y2], and concatenate them with &."
            ),
            "draw": "output",
        })
    if "rec" in selected:
        add_task(tasks, selected, {
            "task_name": "rec",
            "title": "属性限定",
            "image": copy_coco(image_root, "rec", "000000363840.jpg", allow_fallback),
            "instruction": (
                "Where is the laptop on the right side of the desk? "
                "Answer in [x1, y1, x2, y2] format."
            ),
            "draw": "output",
        })
    if "reasoning_grounding" in selected:
        add_task(tasks, selected, {
            "task_name": "reasoning_grounding",
            "title": "推理条件",
            "image": copy_coco(image_root, "reasoning_grounding", "000000007574.jpg", allow_fallback),
            "instruction": (
                "Which appliance in this kitchen is used to heat food? "
                "Answer in the format: class name-[x1, y1, x2, y2], and concatenate them with &."
            ),
            "draw": "output",
        })

    if "visual_referring" in selected:
        refer_image = copy_asset(image_root, "visual_referring", "image.jpg", os.path.join(DEMO, "woods.jpg"), allow_fallback)
        refer_prompt = copy_asset(image_root, "visual_referring", "prompt.jpg", os.path.join(DEMO, "woods_crop.jpg"), allow_fallback)
        add_task(tasks, selected, {
            "task_name": "visual_referring",
            "title": "图指代定位",
            "image": refer_image,
            "prompt_image": refer_prompt,
            "instruction": (
                "In this image, find all objects similar to <region> and output their coordinates "
                "in the format [x1, y1, x2, y2], concatenated with &."
            ),
            "draw": "output",
        })

    if "co_referring" in selected:
        coref_image = copy_asset(image_root, "co_referring", "image.jpg", os.path.join(DEMO, "woods.jpg"), allow_fallback)
        coref_prompt = copy_asset(image_root, "co_referring", "prompt.jpg", os.path.join(DEMO, "woods_crop.jpg"), allow_fallback)
        add_task(tasks, selected, {
            "task_name": "co_referring",
            "title": "图文双指代",
            "image": coref_image,
            "prompt_image": coref_prompt,
            "instruction": (
                "Find objects similar to <region> that stand in the lower half of the image. "
                "Answer in the format: class name-[x1, y1, x2, y2], and concatenate them with &."
            ),
            "draw": "output",
        })

    if "object_counting" in selected:
        count_image = copy_asset(image_root, "object_counting", "image.jpg", os.path.join(DEMO, "woods.jpg"), allow_fallback)
        count_prompt = copy_asset(image_root, "object_counting", "prompt.jpg", os.path.join(DEMO, "woods_crop.jpg"), allow_fallback)
        add_task(tasks, selected, {
            "task_name": "object_counting",
            "title": "密集目标计数",
            "image": count_image,
            "prompt_image": count_prompt,
            "instruction": (
                "Count every object similar to <region>. Reply with the total number first, "
                "then list each instance as [x1, y1, x2, y2], concatenated with &."
            ),
            "draw": "output",
        })
    tasks.sort(key=lambda task: selected.index(task["task_name"]))
    return tasks


def image_token(model):
    if model.config.mm_use_im_start_end:
        return DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    return DEFAULT_IMAGE_TOKEN


def process_images(images, image_processor, model_cfg, prompt_flags):
    resize = transforms.Resize((image_processor.size["shortest_edge"], image_processor.size["shortest_edge"]))
    if getattr(model_cfg, "unshared_visual_encoder", False):
        prompt_processor = CLIPImageProcessor.from_pretrained(os.path.join(ROOT, "checkpoints", "clip-vit-large-patch14"))
        prompt_resize = transforms.Resize((prompt_processor.size["shortest_edge"], prompt_processor.size["shortest_edge"]))
    else:
        prompt_processor = None
        prompt_resize = None
    new_images, new_prompts = [], []
    for i, image in enumerate(images):
        use_prompt = prompt_flags[i] and prompt_resize is not None
        if not use_prompt:
            image = image_processor.preprocess(resize(image), return_tensors="pt")["pixel_values"][0]
            new_images.append(image)
        else:
            image = prompt_processor.preprocess(prompt_resize(image), return_tensors="pt")["pixel_values"][0]
            new_prompts.append(image)
    images_tensor = torch.stack(new_images, dim=0)
    prompts_tensor = torch.stack(new_prompts, dim=0) if new_prompts else None
    return images_tensor, prompts_tensor


def run_one(task, tokenizer, model, image_processor, torch_dtype):
    token = image_token(model)
    instruction = task["instruction"]
    image_files = [task["image"]]
    prompt_flags = [False]
    if task.get("prompt_image"):
        image_files.append(task["prompt_image"])
        prompt_flags.append(True)
        instruction = instruction.replace("<region>", "<image>")
        query = token + "\n" + instruction
    else:
        query = token + "\n" + instruction

    conv = conv_templates["gemma_instruct"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    images = [load_image(path) for path in image_files]
    images_tensor, prompt_tensors = process_images(images, image_processor, model.config, prompt_flags)
    device = next(model.parameters()).device
    images_tensor = images_tensor.to(device, dtype=torch_dtype)
    generate_kwargs = {}
    if prompt_tensors is not None:
        generate_kwargs["regions"] = prompt_tensors.to(device, dtype=torch_dtype)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    max_new_tokens = 1536 if task["draw"] == "output" else 512
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            **generate_kwargs,
        )
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if stop_str and output.endswith(stop_str):
        output = output[: -len(stop_str)].strip()
    boxes = []
    if task["draw"] == "output":
        try:
            boxes = extract(output, BOX_PAT)
        except Exception:
            boxes = []
    elif task["draw"] == "input":
        boxes = task.get("input_boxes", [])
    return output, boxes


def wrap_text(draw, text, font, width):
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for word in paragraph.split(" "):
            trial = word if not current else current + " " + word
            if draw.textlength(trial, font=font) <= width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines


def draw_boxes(image, boxes):
    canvas = image.convert("RGB").copy()
    if not boxes:
        return canvas
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    font = ImageFont.truetype(FONT_PATH, max(16, int(height * 0.03)))
    line = max(2, int(width * 0.004))
    colors = [(220, 40, 40), (30, 120, 220), (20, 150, 70), (180, 90, 20), (120, 40, 180)]
    for i, item in enumerate(boxes):
        x1, y1, x2, y2 = item["bbox"]
        box = [x1 * width, y1 * height, x2 * width, y2 * height]
        color = colors[i % len(colors)]
        draw.rectangle(box, outline=color, width=line)
        label = item.get("category_name") or "target"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        tw, th = right - left, bottom - top
        draw.rectangle([box[0], max(0, box[1] - th - 4), box[0] + tw + 8, max(0, box[1])], fill=color)
        draw.text((box[0] + 4, max(0, box[1] - th - 2)), label, fill=(255, 255, 255), font=font)
    return canvas


def save_visualization(task, output, boxes):
    image = draw_boxes(Image.open(task["image"]).convert("RGB"), boxes if task["draw"] else [])
    side = []
    if task.get("prompt_image"):
        prompt = Image.open(task["prompt_image"]).convert("RGB")
        prompt.thumbnail((360, 360))
        side.append(("Visual prompt", prompt))

    font = ImageFont.truetype(FONT_PATH, 22)
    title_font = ImageFont.truetype(FONT_PATH, 28)
    panel_w = 720
    dummy = Image.new("RGB", (panel_w, 10))
    draw = ImageDraw.Draw(dummy)
    instruction_lines = wrap_text(draw, task["instruction"].replace("<region>", "<visual prompt>"), font, panel_w - 40)
    output_lines = wrap_text(draw, output if output else "(empty)", font, panel_w - 40)
    text_h = 80 + (len(instruction_lines) + len(output_lines) + 4) * 30
    thumb_h = sum(item[1].height + 50 for item in side)
    panel_h = max(image.height, text_h + thumb_h + 20)
    panel = Image.new("RGB", (panel_w, panel_h), (248, 248, 248))
    pdraw = ImageDraw.Draw(panel)
    y = 16
    pdraw.text((20, y), f"{task['task_name']} {task['title']}", fill=(20, 20, 20), font=title_font)
    y += 46
    pdraw.text((20, y), "Instruction", fill=(20, 70, 140), font=title_font)
    y += 36
    for line in instruction_lines:
        pdraw.text((20, y), line, fill=(30, 30, 30), font=font)
        y += 30
    y += 16
    for label, thumb in side:
        pdraw.text((20, y), label, fill=(20, 70, 140), font=title_font)
        y += 36
        panel.paste(thumb, (20, y))
        y += thumb.height + 16
    pdraw.text((20, y), "Model output", fill=(140, 40, 40), font=title_font)
    y += 36
    for line in output_lines:
        if y + 30 > panel_h:
            break
        pdraw.text((20, y), line, fill=(30, 30, 30), font=font)
        y += 30

    gap = 16
    canvas = Image.new("RGB", (image.width + gap + panel_w, max(image.height, panel_h)), (255, 255, 255))
    canvas.paste(image, (0, 0))
    canvas.paste(panel, (image.width + gap, 0))
    out_path = os.path.join(VIS_DIR, f"{task['task_name']}.jpg")
    canvas.save(out_path, quality=90)
    return out_path


def save_json(task, output, boxes, vis_path):
    record = {
        "task_name": task["task_name"],
        "title": task["title"],
        "image": os.path.relpath(task["image"], DEMO),
        "instruction": task["instruction"],
        "visual_prompt": os.path.relpath(task["prompt_image"], DEMO) if task.get("prompt_image") else None,
        "input_boxes": task.get("input_boxes", []),
        "output": output,
        "boxes": boxes,
        "visualization": os.path.relpath(vis_path, DEMO),
    }
    path = os.path.join(RESULT_DIR, f"{task['task_name']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return path


def align_prompt_modules(model, device, dtype):
    """Keep the counting prompt encoder on the same device as the language model.

    It is created with its own device_map=\"auto\", so on a multi-GPU machine it
    can land on a different card than the model selected above.
    """
    inner = model.get_model()
    modules = [
        getattr(inner, "prompt_visual_encoder", None),
        getattr(inner, "prompt_projector", None),
    ]
    try:
        from accelerate.hooks import remove_hook_from_module
    except ImportError:
        remove_hook_from_module = None
    for module in modules:
        if module is None:
            continue
        if remove_hook_from_module is not None:
            remove_hook_from_module(module, recurse=True)
        module.to(device=device, dtype=dtype)


def select_device():
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return "cpu"
    best_idx, best_free = 0, -1
    for idx in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(idx)
        if free > best_free:
            best_idx, best_free = idx, free
    if best_free < 22 * (1024 ** 3):
        print(f"No GPU has 22 GiB free (max free {best_free / 1024 ** 3:.1f} GiB). Loading on CPU.")
        return "cpu"
    print(f"Loading on cuda:{best_idx} ({best_free / 1024 ** 3:.1f} GiB free).")
    return f"cuda:{best_idx}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="按任务类型运行 Griffon 视觉示例。检测相关任务固定读取 DET_PART，无需在参数里指定。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        default=MODEL_PATH,
        help="模型目录，指向 Griffon-Unimodel，不要写到具体 PART。",
    )
    parser.add_argument(
        "--task_name",
        default="all",
        help="要运行的任务名，例如 ocr。多个用逗号或空格分隔。默认 all。\n" + task_catalog(),
    )
    parser.add_argument(
        "--image-dir",
        default=TASK_ROOT,
        help=(
            "待处理图像根目录。每个样本一个子目录，内含 image.jpg；\n"
            "图指代样本另需 prompt.jpg。\n"
            "默认 demo/task；使用该默认目录时，缺图会自动准备示例。"
        ),
    )
    return parser.parse_args()


def show_path(path):
    if not path:
        return "无"
    return os.path.abspath(path)


def print_task_header(task, model_dir):
    print("\n==========", flush=True)
    print(f"task_name: {task['task_name']}", flush=True)
    print(f"任务: {task['title']}", flush=True)
    print(f"待处理图像: {show_path(task['image'])}", flush=True)
    print(f"模型路径: {show_path(model_dir)}", flush=True)
    print(f"文本指令: {task['instruction']}", flush=True)
    prompt = task.get("prompt_image")
    print(f"视觉提示: {show_path(prompt) if prompt else '无'}", flush=True)
    print("==========", flush=True)


def print_task_result(output, boxes, vis_path, json_path):
    print("输出文本:", flush=True)
    print(output if output else "（空）", flush=True)
    print(f"解析框数量: {len(boxes)}", flush=True)
    print(f"可视化结果: {show_path(vis_path)}", flush=True)
    print(f"结果文件: {show_path(json_path)}", flush=True)


def main():
    args = parse_args()
    model_dir = resolve_dir(args.model_path)
    image_dir = resolve_dir(args.image_dir)
    selected = expand_task_names(args.task_name)
    allow_fallback = os.path.realpath(image_dir) == os.path.realpath(TASK_ROOT)
    ensure_layout()
    tasks = prepare_inputs(image_dir, allow_fallback, selected)
    if not tasks:
        raise SystemExit("所选任务没有可运行的样本。\n" + task_catalog())

    checkpoint = os.path.join(model_dir, DET_PART)
    if not os.path.isdir(checkpoint):
        raise SystemExit(f"模型目录中缺少默认检测权重 {DET_PART}: {checkpoint}")

    print(f"模型路径: {show_path(model_dir)}", flush=True)
    print("task_name: " + ", ".join(selected), flush=True)
    print(f"待处理图像目录: {show_path(image_dir)}", flush=True)
    print(f"样本数: {len(tasks)}", flush=True)

    disable_torch_init()
    device = select_device()
    dtype = torch.float16 if device == "cpu" else torch.bfloat16
    model_name = get_model_name_from_path(checkpoint)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        checkpoint, None, model_name, device=device, torch_dtype=dtype
    )
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.disable_compile = True
    align_prompt_modules(model, device, dtype)
    model.eval()

    for task in tasks:
        print_task_header(task, model_dir)
        output, boxes = run_one(task, tokenizer, model, image_processor, dtype)
        vis_path = save_visualization(task, output, boxes)
        json_path = save_json(task, output, boxes, vis_path)
        print_task_result(output, boxes, vis_path, json_path)


if __name__ == "__main__":
    main()
