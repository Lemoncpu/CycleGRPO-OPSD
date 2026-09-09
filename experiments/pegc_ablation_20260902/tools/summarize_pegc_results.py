#!/usr/bin/env python3
"""Extract comparable route/IoU/loss metrics from four PEGC run logs."""
import argparse, json, re
from pathlib import Path

KEYS=("pixel_iou_mean", "R_Ci_mean", "distill_jsd", "evidence_gate_mean", "adaptive_balance_caption_scale", "adaptive_balance_segmentation_scale")
parser=argparse.ArgumentParser(); parser.add_argument("root", type=Path); parser.add_argument("--output", type=Path, default=None); args=parser.parse_args()
rows=[]
for name in ("baseline", "evidence_gate", "mask_credit", "adaptive_balance"):
    values={"variant":name}
    for log in sorted((args.root/f"pegc_{name}_4k").glob("train_*.log")):
        for line in log.read_text(errors="ignore").splitlines():
            for key in KEYS:
                m=re.search(r"(?:opsd/)?"+re.escape(key)+r"[^0-9.-]*([-+]?[0-9]*\.?[0-9]+)", line)
                if m: values[key]=float(m.group(1))
    rows.append(values)
text=json.dumps(rows, indent=2, ensure_ascii=False)
print(text)
if args.output: args.output.write_text(text+"\n")
