#!/usr/bin/env python3
"""Run standard RefCOCO evaluation with a native Sa2VA checkpoint."""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.refcoco.qwen3vl_refcoco_eval import encode_rle, load_samples  # noqa: E402


DEFAULT_REFCOCO_ROOT = "/volume/ybo/xyc/refcoco-train2014-assets"
DEFAULT_PROMPT = "Please segment {phrase} in this image."


def parse_args():
    parser = argparse.ArgumentParser(description="Native Sa2VA RefCOCO cIoU/mIoU evaluation.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--refcoco-root", default=DEFAULT_REFCOCO_ROOT)
    parser.add_argument("--split", default="val", choices=("val", "testA", "testB"))
    parser.add_argument("--split-by", default="unc")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--metric-only", action="store_true")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--num-tasks", type=int)
    return parser.parse_args()


def parse_gpu_ids(raw):
    ids = [item.strip() for item in raw.split(",") if item.strip()]
    if not ids or any(not item.isdigit() for item in ids):
        raise ValueError("--gpus must be a comma-separated list of physical GPU ids")
    return ids


def decode_rle(rle):
    return mask_utils.decode(rle).astype(bool)


def calculate_metrics(prediction_dir, samples):
    intersection = 0
    union = 0
    iou_sum = 0.0
    expected_names = {f"{sample['case_id']}.json" for sample in samples}
    actual_names = {path.name for path in prediction_dir.glob("*.json")}
    if actual_names != expected_names:
        raise RuntimeError(
            f"Sa2VA RefCOCO output is incomplete or contains extra files: "
            f"expected={len(expected_names)} actual={len(actual_names)} "
            f"missing={len(expected_names - actual_names)} extra={len(actual_names - expected_names)}"
        )

    for sample in samples:
        record = json.loads((prediction_dir / f"{sample['case_id']}.json").read_text())
        prediction = decode_rle(record["prediction"])
        target = decode_rle(record["target"])
        if prediction.shape != target.shape:
            raise ValueError(f"Mask shape mismatch for RefCOCO case {sample['case_id']}")
        current_intersection = int(np.logical_and(prediction, target).sum())
        current_union = int(np.logical_or(prediction, target).sum())
        intersection += current_intersection
        union += current_union
        iou_sum += 1.0 if current_union == 0 else current_intersection / current_union

    return {
        "samples": len(samples),
        "cIoU": 0.0 if union == 0 else 100.0 * intersection / union,
        "mIoU": 0.0 if not samples else 100.0 * iou_sum / len(samples),
    }


def matching_record(path, model_path, prompt_template):
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        record.get("backend") == "sa2va"
        and record.get("model_path") == str(Path(model_path).resolve())
        and record.get("prompt_template") == prompt_template
    )


def run_worker(args, samples):
    import torch
    from transformers import AutoModel, AutoTokenizer

    task_id = args.task_id
    num_tasks = args.num_tasks
    if task_id is None or num_tasks is None or num_tasks <= 0 or not 0 <= task_id < num_tasks:
        raise ValueError("worker requires task-id in [0, num-tasks)")
    torch.cuda.set_device(0)

    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    per_task = math.ceil(len(samples) / num_tasks)
    worker_samples = samples[task_id * per_task : min((task_id + 1) * per_task, len(samples))]
    prediction_dir = Path(args.output_dir) / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(Path(args.model_path).resolve())
    pending = [
        sample for sample in worker_samples
        if not matching_record(
            prediction_dir / f"{sample['case_id']}.json", model_path, args.prompt_template
        )
    ]
    print(
        f"[Sa2VA shard {task_id}/{num_tasks}] pending={len(pending)}/{len(worker_samples)}",
        flush=True,
    )

    with torch.inference_mode():
        for index, sample in enumerate(pending, 1):
            with Image.open(sample["image_path"]) as source:
                image = source.convert("RGB")
            height, width = image.height, image.width
            prompt = args.prompt_template.format(phrase=sample["phrase"])
            text = "<image>\n " + prompt
            result = model.predict_forward(image=image, text=text, tokenizer=tokenizer)
            response = str(result.get("prediction", ""))
            masks = result.get("prediction_masks", [])
            marker_count = response.count("[SEG]")
            selected_masks = masks[:marker_count] if marker_count else []
            prediction = np.zeros((height, width), dtype=np.uint8)
            for item in selected_masks:
                if hasattr(item, "detach"):
                    item = item.detach().cpu().numpy()
                mask = np.asarray(item).squeeze().astype(np.uint8)
                if mask.shape != (height, width):
                    raise ValueError(
                        f"Sa2VA mask shape {mask.shape} != image shape {(height, width)} "
                        f"for RefCOCO case {sample['case_id']}"
                    )
                prediction |= mask

            output = {
                "backend": "sa2va",
                "model_path": model_path,
                "prompt_template": args.prompt_template,
                "prediction": encode_rle(prediction),
                "target": sample["target"],
                "response": response,
                "mask_count": len(selected_masks),
            }
            output_path = prediction_dir / f"{sample['case_id']}.json"
            temporary_path = output_path.with_suffix(".json.tmp")
            temporary_path.write_text(json.dumps(output, ensure_ascii=False))
            temporary_path.replace(output_path)
            if index % 100 == 0 or index == len(pending):
                print(
                    f"[Sa2VA shard {task_id}/{num_tasks}] "
                    f"progress={index}/{len(pending)}",
                    flush=True,
                )


def run_parent(args, samples):
    output_dir = Path(args.output_dir).resolve()
    prediction_dir = output_dir / "predictions"
    log_dir = output_dir / "logs"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    gpus = parse_gpu_ids(args.gpus)

    if not args.metric_only:
        processes = []
        handles = []
        try:
            for task_id, gpu in enumerate(gpus):
                log_path = log_dir / f"refcoco_{args.split}_sa2va_shard{task_id}.log"
                handle = log_path.open("w", encoding="utf-8")
                handles.append(handle)
                command = [
                    args.python_bin,
                    str(Path(__file__).resolve()),
                    "--model-path", args.model_path,
                    "--refcoco-root", args.refcoco_root,
                    "--split", args.split,
                    "--split-by", args.split_by,
                    "--prompt-template", args.prompt_template,
                    "--output-dir", str(output_dir),
                    "--max-samples", str(args.max_samples),
                    "--task-id", str(task_id),
                    "--num-tasks", str(len(gpus)),
                ]
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = gpu
                environment["OMP_NUM_THREADS"] = "1"
                environment["PYTHONUNBUFFERED"] = "1"
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                processes.append((task_id, gpu, process, log_path))
                print(
                    f"启动 Sa2VA 分片 {task_id + 1}/{len(gpus)}: GPU {gpu}; 日志={log_path}",
                    flush=True,
                )
            failures = []
            for task_id, gpu, process, log_path in processes:
                status = process.wait()
                print(f"完成 Sa2VA 分片 {task_id + 1}/{len(gpus)}: exit={status}", flush=True)
                if status:
                    failures.append((task_id, status, log_path))
            if failures:
                detail = ", ".join(
                    f"shard={task} exit={status} log={path}" for task, status, path in failures
                )
                raise RuntimeError("Sa2VA RefCOCO worker failed: " + detail)
        finally:
            for handle in handles:
                handle.close()

    metrics = calculate_metrics(prediction_dir, samples)
    report = {
        "task_name": "refcoco_performance",
        "backend": "sa2va",
        "model_path": str(Path(args.model_path).resolve()),
        "dataset": f"RefCOCO {args.split} ({args.split_by})",
        "samples_expected": len(samples),
        "samples_completed": metrics["samples"],
        "prompt_template": args.prompt_template,
        "metrics": metrics,
        "gpus": gpus,
        "predictions": str(prediction_dir),
        "logs": str(log_dir),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    results_path = output_dir / "results.txt"
    results_path.write_text(
        "RefCOCO PERFORMANCE TEST RESULTS\n"
        f"Backend: Sa2VA native segmentation\n"
        f"Model: {report['model_path']}\n"
        f"Dataset: {report['dataset']}\n"
        f"Samples: {metrics['samples']}/{len(samples)}\n"
        f"cIoU: {metrics['cIoU']:.4f}\n"
        f"mIoU: {metrics['mIoU']:.4f}\n"
        f"Prompt: {args.prompt_template}\n"
    )
    print("后端: Sa2VA 原生分割", flush=True)
    print(f"cIoU: {metrics['cIoU']:.4f}", flush=True)
    print(f"mIoU: {metrics['mIoU']:.4f}", flush=True)
    print(f"完成样本: {metrics['samples']}/{len(samples)}", flush=True)
    print(f"结果文件: {metrics_path}", flush=True)
    print(f"文本报告: {results_path}", flush=True)


def main():
    args = parse_args()
    if "{phrase}" not in args.prompt_template:
        raise ValueError("prompt-template must contain {phrase}")
    samples = load_samples(args.refcoco_root, args.split_by, args.split)
    if args.max_samples:
        if args.max_samples < 0:
            raise ValueError("max-samples must be non-negative")
        samples = samples[: args.max_samples]
    if args.task_id is not None:
        run_worker(args, samples)
    else:
        run_parent(args, samples)


if __name__ == "__main__":
    main()
