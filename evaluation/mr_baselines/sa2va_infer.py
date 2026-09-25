import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import AutoModel, AutoTokenizer


def encode(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--benchmarks", nargs="+", required=True)
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl")

    from projects.sa2va.models.utils import find_seg_indices

    model = AutoModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True,
    ).eval().cuda(local_rank)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    records = []
    for bench in args.benchmarks:
        path = Path(args.data_root) / f"{bench}_val_sampled.json"
        rounds = json.loads(path.read_text())
        for round_key in sorted(rounds, key=lambda x: int(x)):
            for i, row in enumerate(rounds[round_key]):
                records.append((bench, int(round_key), i, row))

    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)
    pred_path = out_path / f"rank{rank}.jsonl"
    with pred_path.open("w") as fout, torch.inference_mode():
        for n in range(rank, len(records), world):
            bench, round_id, row_id, row = records[n]
            rel = row["image"].replace("./data/coco/train2014/", "")
            image_path = Path(args.images) / rel
            with Image.open(image_path) as im:
                image = im.convert("RGB")
            h, w = image.height, image.width
            messages = row["history"]
            text = "\n".join(
                ("User: " if m.get("from") == "human" else "Assistant: ") + m["value"].replace("<image>", "").strip()
                for m in messages
            )
            text = "<image>\n" + text + "\nPlease provide the segmentation mask for the final request."
            result = model.predict_forward(image=image, text=text, tokenizer=tokenizer)
            prediction = result.get("prediction", "")
            masks = result.get("prediction_masks", [])
            clean = prediction.replace("<|im_end|>", "").replace("<|end|>", "").strip()
            _, seg_indices = find_seg_indices(clean)
            selected = [masks[j] for j in seg_indices if j < len(masks)] if seg_indices else []
            pred = np.zeros((h, w), dtype=np.uint8)
            for item in selected:
                arr = np.asarray(item).squeeze().astype(np.uint8)
                if arr.shape == pred.shape:
                    pred |= arr
            out = {
                "benchmark": bench,
                "round": round_id,
                "row": row_id,
                "prediction": encode(pred),
                "target": row["target_masks"],
            }
            fout.write(json.dumps(out) + "\n")
            if args.max_samples and n + 1 >= args.max_samples:
                break
            if n % 100 == rank:
                fout.flush()
                print(f"rank={rank} progress={n+1}/{len(records)}", flush=True)

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
