# MR benchmark evaluation comparison

Evaluation uses the official sampled multi-round validation files. CycleGRPO uses the standard expression-segmentation conversation history (no force-mask/no-target rewrite), retaining the original `history` field and its prior turns. Baseline-specific adapters are documented below. Scores below are percentages; CIoU is summed foreground intersection/union and GIoU is mean per-sample IoU. Counts must match the full official sampled split before a result is considered complete.

For CycleGRPO checkpoints, inference uses the Sa2VA repository's Qwen3-VL MR evaluator with the repository's compatible VQ-SAM2 implementation; the shim is runtime-only and does not modify evaluator or project source files. Raw per-sample outputs and per-round logs are retained under the paths listed below.

## CycleGRPO checkpoints

| Checkpoint | MR benchmark | Samples | CIoU | GIoU | Status |
|---|---|---:|---:|---:|---|
| `cyclegrpo20k_opsd_all_samples_routing_4gpu/evaluation/step_156/hf_global_step_156` | MR-RefCOCO | 6,678/6,678 | 74.69 | 75.61 | Complete |
| `cyclegrpo20k_opsd_all_samples_routing_4gpu/evaluation/step_156/hf_global_step_156` | MR-RefCOCO+ | 6,655/6,655 | 69.80 | 71.52 | Complete |
| `cyclegrpo20k_opsd_all_samples_routing_4gpu/evaluation/step_156/hf_global_step_156` | MR-RefCOCOg | 3,746/3,746 | 72.93 | 73.31 | Complete |
| `cyclegrpo20k_opsd_all_samples_routing_4gpu/evaluation/step_156/hf_global_step_156` | MR-PACO | 9,281/9,281 | 40.42 | 31.03 | Complete |
| `cyclegrpo70k_historical_baseline_8gpu/evaluation/step_179_cpu_export/hf_global_step_179` | MR-RefCOCO | 6,678/6,678 | 81.64 | 82.53 | Complete |
| `cyclegrpo70k_historical_baseline_8gpu/evaluation/step_179_cpu_export/hf_global_step_179` | MR-RefCOCO+ | 6,655/6,655 | 78.55 | 80.49 | Complete |
| `cyclegrpo70k_historical_baseline_8gpu/evaluation/step_179_cpu_export/hf_global_step_179` | MR-RefCOCOg | 3,746/3,746 | 81.05 | 80.78 | Complete |
| `cyclegrpo70k_historical_baseline_8gpu/evaluation/step_179_cpu_export/hf_global_step_179` | MR-PACO | 9,281/9,281 | 51.09 | 38.40 | Complete |
| `cyclegrpo70k_historical_seca_8gpu/evaluation/step_179_cpu_export/hf_global_step_179` | MR-RefCOCO | 6,678/6,678 | 81.70 | 82.61 | Complete |
| `cyclegrpo70k_historical_seca_8gpu/evaluation/step_179_cpu_export/hf_global_step_179` | MR-RefCOCO+ | 6,655/6,655 | 78.64 | 80.64 | Complete |
| `cyclegrpo70k_historical_seca_8gpu/evaluation/step_179_cpu_export/hf_global_step_179` | MR-RefCOCOg | 3,746/3,746 | 80.29 | 80.58 | Complete |
| `cyclegrpo70k_historical_seca_8gpu/evaluation/step_179_cpu_export/hf_global_step_179` | MR-PACO | 9,281/9,281 | 50.69 | 38.49 | Complete |

## CycleGRPO standard RefCOCO family (ordinary prompt)

Uses the normal expression-segmentation prompt `Please segment {phrase} in this image.` with the `legacy_union` decoding protocol and validation split. Scores are percentages; mIoU is the evaluator's per-sample mean IoU.

| Checkpoint | RefCOCO+ samples | RefCOCO+ cIoU / mIoU | RefCOCOg samples | RefCOCOg cIoU / mIoU | Status |
|---|---:|---:|---:|---:|---|
| `cyclegrpo20k_opsd_all_samples_routing_4gpu/evaluation/step_156/hf_global_step_156` | 10,758/10,758 | 24.41 / 17.25 | 4,896/4,896 | 54.35 / 46.17 | Complete |
| `cyclegrpo70k_historical_baseline_8gpu/evaluation/step_179_cpu_export/hf_global_step_179` | 10,758/10,758 | 75.16 / 77.08 | 4,896/4,896 | 77.69 / 77.99 | Complete |
| `cyclegrpo70k_historical_seca_8gpu/evaluation/step_179_cpu_export/hf_global_step_179` | 10,758/10,758 | 75.44 / 77.31 | 4,896/4,896 | 77.65 / 78.06 | Complete |

Raw standard RefCOCO outputs are retained under `/tmp/cyclegrpo_refcoco_normal_prompt_20260924/`; per-shard logs and metrics are saved inside each dataset's `logs/` directory.

## Baseline models

Cells are **CIoU / GIoU (%)**. Sa2VA counts are 6,678 / 6,655 / 3,746 / 9,281 respectively (26,360 total); strict re-scoring rejects duplicate/unexpected keys, inference errors and incorrect mask dimensions. Predictions and per-round metrics are archived at `logs/mr_baselines_20260924/sa2va/`.

**Adapter boundary:** baseline single-query interfaces receive the complete history serialized as `User:` / `Assistant:` text, with original SAMTok strings retained as text. Sa2VA and PaDT additionally request the mask for the final question; UniPixel uses its native sentence template with a segmentation request. This is not the same native mask-conditioned conversation interface as CycleGRPO. Native baseline dense masks are scored directly; `evaluation/mask_protocol.py` parses SAMTok history only and does not decode these native outputs. EVF-SAM's earlier last-question-only adapter is excluded. These are locally adapted MR results, not published native MR benchmark scores.

| Model | MR-RefCOCO | MR-RefCOCO+ | MR-RefCOCOg | MR-PACO | Status |
|---|---:|---:|---:|---:|---|
| Sa2VA | 76.43 / 78.13 | 71.96 / 74.56 | 77.03 / 77.60 | 26.41 / 17.03 | Complete (serialized-history adapter) |
| PaDT Pro | 72.85 / 75.51 | 67.73 / 71.94 | 74.68 / 75.04 | 16.38 / 14.02 | Complete (serialized-history adapter) |
| UniPixel | 73.80 / 76.69 | 68.49 / 72.47 | 75.42 / 76.16 | 37.61 / 26.35 | Complete (serialized-history adapter) |
| InstructSeg | 53.19 / 62.52 | 49.52 / 59.52 | 60.26 / 67.83 | 24.08 / 18.61 | Complete (serialized-history adapter) |
| EVF-SAM | — | — | — | — | Running full split; 40/40 multi-round smoke passed |

Raw predictions and per-round logs for the running first checkpoint are under `/tmp/mr_bench_eval_20260923_retry1/all_samples_routing/` and `logs/mr_bench_eval_20260923_retry1/all_samples_routing/` respectively. GPU hold is restored on all four cards after each evaluation job.

Baseline run artifacts: `logs/mr_baselines_20260924/{padt,unipixel,instructseg,evfsam}_full/`. Each full run covers all 26,360 samples, with per-round metrics written only after inference and strict scoring. `*_rounds_smoke` outputs are interface checks (one sample per round per dataset), never benchmark results. Source commits, local compatibility patches and model config hashes are in `logs/mr_baselines_20260924/provenance/`; canonical data SHA256 and SAMTok history audit are in `data_audit.json`. During each run the existing GPU hold worker retains its allocation, pauses compute, and resumes compute when the launcher exits.
