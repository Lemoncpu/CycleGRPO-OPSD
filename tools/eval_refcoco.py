#!/usr/bin/env python3
"""Run the full RefCOCO performance test with our exported SAMTok weights."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.refcoco.qwen3vl_refcoco_eval import compute_metrics, load_samples

DEFAULT_MODEL = (
    "/volume/ybo/xyc/CycleGRPO-OPSD/logs/"
    "cyclegrpo70k_historical_seca_8gpu/evaluation/step_179_cpu_export/"
    "hf_global_step_179"
)
DEFAULT_VQ = "/volume/ybo/xyc/Qwen3-VL-4B-SAMTok/mask_tokenizer_256x2.pth"
DEFAULT_SAM2 = "/volume/ybo/xyc/Qwen3-VL-4B-SAMTok/sam2.1_hiera_large.pt"
DEFAULT_REFCOCO_ROOT = "/volume/ybo/xyc/refcoco-train2014-assets"
DEFAULT_OUTPUT = REPO_ROOT / "tools/demo/output/performance/refcoco"
ORDINARY_PROMPT = "Please segment {phrase} in this image."


def parse_args():
    parser = argparse.ArgumentParser(description="使用我们的权重执行 RefCOCO 全量性能测试。")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--vq-sam2-path", default=DEFAULT_VQ)
    parser.add_argument("--sam2-path", default=DEFAULT_SAM2)
    parser.add_argument("--refcoco-root", default=DEFAULT_REFCOCO_ROOT)
    parser.add_argument("--split", default="val", choices=("val", "testA", "testB"))
    parser.add_argument("--split-by", default="unc")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--mask-protocol", default="legacy_union", choices=("legacy_union", "first_mask"))
    parser.add_argument("--prompt-template", default=ORDINARY_PROMPT)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--ciou-threshold", type=float, default=70.0)
    parser.add_argument("--metric-only", action="store_true")
    return parser.parse_args()


def gpu_ids(raw):
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values or any(not value.isdigit() for value in values):
        raise ValueError("--gpus must be a comma-separated list of physical GPU ids")
    return values


def audit_predictions(prediction_dir, samples, mask_protocol, prompt_template):
    expected = {f"{sample['case_id']}.json" for sample in samples}
    actual = {path.name for path in prediction_dir.glob("*.json")}
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    if missing or extra:
        raise RuntimeError(
            f"RefCOCO prediction audit failed: expected={len(expected)} actual={len(actual)} "
            f"missing={len(missing)} extra={len(extra)}"
        )
    mismatched = []
    for name in sorted(expected):
        try:
            record = json.loads((prediction_dir / name).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            mismatched.append(name)
            continue
        if (
            record.get("mask_protocol") != mask_protocol
            or record.get("prompt_template") != prompt_template
        ):
            mismatched.append(name)
    if mismatched:
        raise RuntimeError(
            "RefCOCO prediction metadata audit failed: "
            f"mismatched={len(mismatched)} first={mismatched[0]}"
        )


def main():
    args = parse_args()
    if "{phrase}" not in args.prompt_template:
        raise ValueError("prompt-template must contain {phrase}")
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("batch-size and max-new-tokens must be positive")
    devices = gpu_ids(args.gpus)
    output_dir = Path(args.output_dir).resolve()
    prediction_dir, log_dir = output_dir / "predictions", output_dir / "logs"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.refcoco_root, args.split_by, args.split)

    print("模型路径: " + str(Path(args.model_path).resolve()), flush=True)
    print("task_name: refcoco_performance", flush=True)
    print("数据集: RefCOCO " + args.split, flush=True)
    print(f"样本数: {len(samples)}", flush=True)
    print(f"GPU: {', '.join(devices)}", flush=True)
    print(f"文本指令模板: {args.prompt_template}", flush=True)
    print(f"输出目录: {output_dir}", flush=True)

    if not args.metric_only:
        evaluator = REPO_ROOT / "evaluation/refcoco/qwen3vl_refcoco_eval.py"
        processes = []
        handles = []
        try:
            for task_id, physical_gpu in enumerate(devices):
                log_path = log_dir / f"refcoco_{args.split}_shard{task_id}.log"
                handle = log_path.open("w", encoding="utf-8")
                handles.append(handle)
                command = [
                    args.python_bin, str(evaluator),
                    "--model_path", args.model_path,
                    "--vq_sam2_path", args.vq_sam2_path,
                    "--sam2_path", args.sam2_path,
                    "--refcoco_root", args.refcoco_root,
                    "--split_by", args.split_by,
                    "--split", args.split,
                    "--save_dir", str(prediction_dir),
                    "--task_id", str(task_id),
                    "--num_tasks", str(len(devices)),
                    "--gpu_id", "0",
                    "--batch_size", str(args.batch_size),
                    "--mask_protocol", args.mask_protocol,
                    "--max_new_tokens", str(args.max_new_tokens),
                    "--prompt_template", args.prompt_template,
                ]
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = physical_gpu
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                processes.append((task_id, physical_gpu, process, log_path))
                print(f"启动分片 {task_id + 1}/{len(devices)}: GPU {physical_gpu} 日志={log_path}", flush=True)
            failures = []
            for task_id, physical_gpu, process, log_path in processes:
                return_code = process.wait()
                print(f"完成分片 {task_id + 1}/{len(devices)}: GPU {physical_gpu} exit={return_code}", flush=True)
                if return_code:
                    failures.append((task_id, return_code, log_path))
            if failures:
                details = ", ".join(f"shard={item[0]} exit={item[1]} log={item[2]}" for item in failures)
                raise RuntimeError("RefCOCO worker failed: " + details)
        finally:
            for handle in handles:
                handle.close()

    audit_predictions(prediction_dir, samples, args.mask_protocol, args.prompt_template)
    metrics = compute_metrics(str(prediction_dir))
    passed = metrics["cIoU"] >= args.ciou_threshold
    report = {
        "task_name": "refcoco_performance",
        "model_path": str(Path(args.model_path).resolve()),
        "dataset": f"RefCOCO {args.split} ({args.split_by})",
        "samples_expected": len(samples),
        "samples_completed": metrics["samples"],
        "prompt_template": args.prompt_template,
        "mask_protocol": args.mask_protocol,
        "max_new_tokens": args.max_new_tokens,
        "batch_size_per_gpu": args.batch_size,
        "gpus": devices,
        "metrics": metrics,
        "ciou_threshold": args.ciou_threshold,
        "passed": passed,
        "predictions": str(prediction_dir),
        "logs": str(log_dir),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    result_text = (
        "RefCOCO PERFORMANCE TEST RESULTS\n"
        f"Model: {report['model_path']}\n"
        f"Dataset: {report['dataset']}\n"
        f"Samples: {metrics['samples']}/{len(samples)}\n"
        f"cIoU: {metrics['cIoU']:.4f}\n"
        f"mIoU: {metrics['mIoU']:.4f}\n"
        f"cIoU threshold: {args.ciou_threshold:.4f}\n"
        f"Status: {'PASS' if passed else 'FAIL'}\n"
        f"Prompt: {args.prompt_template}\n"
        f"Mask protocol: {args.mask_protocol}\n"
    )
    results_path = output_dir / "results.txt"
    results_path.write_text(result_text, encoding="utf-8")
    print("输出指标:", flush=True)
    print(f"cIoU: {metrics['cIoU']:.4f}", flush=True)
    print(f"mIoU: {metrics['mIoU']:.4f}", flush=True)
    print(f"完成样本: {metrics['samples']}/{len(samples)}", flush=True)
    print(f"结果文件: {metrics_path}", flush=True)
    print(f"文本报告: {results_path}", flush=True)


if __name__ == "__main__":
    main()
