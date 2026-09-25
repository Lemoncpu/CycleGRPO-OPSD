#!/usr/bin/env python3
"""Score a fixed held-out SAMTok target under three real HF checkpoints.

This is intentionally an offline checkpoint comparison.  It does not rerun
training or claim that the two historical runs consumed identical updates.
The x coordinate is the permitted ``teacher_alignment`` diagnostic from an
independent direct-supervision checkpoint; y is the observed checkpoint
delta-log-prob on the same target mask-code tokens.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


MASK_LO, MASK_HI = 151670, 152181


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--grpo", required=True)
    p.add_argument("--opsd", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--max-memory-gib", type=int, default=28)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_rows(path: str, n: int, seed: int) -> list[dict[str, Any]]:
    table = pq.read_table(path, columns=["images", "cap_problem", "seg_answer"])
    if table.num_rows < n:
        raise ValueError(f"{path} has only {table.num_rows} rows; requested {n}")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(table.num_rows, size=n, replace=False))
    rows = table.take(indices).to_pylist()
    for i, row in enumerate(rows):
        images = row["images"]
        if not images or not Path(images[0]).is_file():
            raise FileNotFoundError(f"row {i} image is missing: {images}")
        if not isinstance(row["seg_answer"], str):
            raise ValueError(f"row {i} has no string seg_answer")
    return rows


def _move_inputs(inputs: Any, device: str) -> Any:
    return inputs.to(device) if hasattr(inputs, "to") else inputs


def score_checkpoint(
    checkpoint: str,
    rows: list[dict[str, Any]],
    processor: Any,
    *,
    device: str,
    max_memory_gib: int,
) -> tuple[np.ndarray, list[list[int]], list[list[int]], np.ndarray]:
    """Return log p for each target token, keeping only legal SAMTok codes."""
    print(f"[score] loading {checkpoint}", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory={0: f"{max_memory_gib}GiB", "cpu": "96GiB"},
    ).eval()
    values: list[list[float]] = []
    token_lists: list[list[int]] = []
    position_lists: list[list[int]] = []
    reliability: list[float] = []
    for row_id, row in enumerate(rows):
        text = row["cap_problem"].replace("<image>", "").strip()
        messages = [[{"role": "user", "content": [
            {"type": "image", "image": row["images"][0]},
            {"type": "text", "text": text},
        ]}]]
        prompt = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        target = processor.tokenizer(row["seg_answer"], add_special_tokens=False, return_tensors="pt").input_ids
        target_ids = target[0]
        code_positions = [i for i, token in enumerate(target_ids.tolist()) if MASK_LO <= int(token) <= MASK_HI]
        if not code_positions:
            raise ValueError(f"row {row_id} target has no SAMTok code: {row['seg_answer']!r}")
        full_ids = torch.cat([prompt.input_ids, target], dim=1)
        full_mask = torch.ones_like(full_ids)
        model_inputs = {k: v for k, v in prompt.items() if k not in {"input_ids", "attention_mask"}}
        model_inputs["input_ids"] = full_ids
        model_inputs["attention_mask"] = full_mask
        model_inputs = _move_inputs(type(prompt)(model_inputs), device)
        with torch.inference_mode():
            logits = model(**model_inputs).logits.float()
        prefix_len = prompt.input_ids.shape[1]
        target_logits = logits[0, prefix_len - 1 : prefix_len - 1 + target_ids.numel()]
        log_probs = torch.log_softmax(target_logits, dim=-1)
        selected = log_probs[torch.arange(target_ids.numel(), device=log_probs.device), target_ids.to(log_probs.device)]
        selected = selected.detach().cpu().numpy().astype(np.float64)
        code_values = [float(selected[i]) for i in code_positions]
        values.append(code_values)
        token_lists.append([int(target_ids[i]) for i in code_positions])
        position_lists.append(code_positions)
        reliability.append(float(np.exp(np.mean(code_values))))
        if (row_id + 1) % 4 == 0 or row_id + 1 == len(rows):
            print(f"[score] {checkpoint}: {row_id + 1}/{len(rows)}", flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    width = max(map(len, values))
    matrix = np.full((len(values), width), np.nan, dtype=np.float64)
    for i, current in enumerate(values):
        matrix[i, : len(current)] = current
    return matrix, token_lists, position_lists, np.asarray(reliability, dtype=np.float64)


def main() -> None:
    args = parse_args()
    if args.n < 2:
        raise ValueError("--n must be at least 2")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.data, args.n, args.seed)
    processor = AutoProcessor.from_pretrained(args.base)
    processor.tokenizer.padding_side = "right"
    base, token_lists, position_lists, reliability = score_checkpoint(
        args.base, rows, processor, device=args.device, max_memory_gib=args.max_memory_gib
    )
    grpo, _, _, _ = score_checkpoint(
        args.grpo, rows, processor, device=args.device, max_memory_gib=args.max_memory_gib
    )
    opsd, _, _, _ = score_checkpoint(
        args.opsd, rows, processor, device=args.device, max_memory_gib=args.max_memory_gib
    )
    teacher, _, _, _ = score_checkpoint(
        args.teacher, rows, processor, device=args.device, max_memory_gib=args.max_memory_gib
    )
    if not (base.shape == grpo.shape == opsd.shape == teacher.shape):
        raise RuntimeError("checkpoint score matrices have different shapes")
    valid = np.isfinite(base) & np.isfinite(grpo) & np.isfinite(opsd) & np.isfinite(teacher)
    x = teacher - base
    y_grpo = grpo - base
    y_opsd = opsd - base
    np.savez_compressed(
        out / "token_data.npz",
        contribution=x,
        grpo=y_grpo,
        opsd=y_opsd,
        reliability=np.repeat(reliability[:, None], x.shape[1], axis=1),
        valid=valid,
        base_logp=base,
        teacher_logp=teacher,
        grpo_logp=grpo,
        opsd_logp=opsd,
    )
    with (out / "intervention_rollouts.jsonl").open("w", encoding="utf-8") as handle:
        for i, row in enumerate(rows):
            handle.write(json.dumps({
                "row": i,
                "image": row["images"][0],
                "cap_problem": row["cap_problem"],
                "seg_answer": row["seg_answer"],
                "target_mask_token_ids": token_lists[i],
                "target_positions": position_lists[i],
                "status": "not_run",
                "reason": "This deliverable uses the permitted teacher-alignment fallback; no replacement rollout is claimed.",
            }, ensure_ascii=False) + "\n")
    metadata = {
        "kind": "teacher_alignment_checkpoint_delta",
        "axis_x": "log p(independent direct-supervision teacher) - log p(base student)",
        "axis_y": "log p(after checkpoint) - log p(base student)",
        "token_type": "SAMTok depth-2 mask-code tokens only",
        "data": os.path.abspath(args.data),
        "seed": args.seed,
        "n_rows": len(rows),
        "valid_tokens": int(valid.sum()),
        "coverage": float(valid.mean()),
        "base_checkpoint": os.path.abspath(args.base),
        "grpo_checkpoint": os.path.abspath(args.grpo),
        "opsd_checkpoint": os.path.abspath(args.opsd),
        "teacher_checkpoint": os.path.abspath(args.teacher),
        "reliability_definition": "exp(mean base log p over the two target mask-code tokens), bounded in (0,1]",
        "training_scope": "post-hoc checkpoint comparison; no new optimizer step and no paired update samples",
        "limitation": "GRPO and Pixel-OPSD historical runs used different training parquet files; this is a mechanism diagnostic, not a controlled objective ablation.",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
