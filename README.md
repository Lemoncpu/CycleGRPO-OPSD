# CycleGRPO-OPSD

> **Work in progress.** This repository contains the CycleGRPO implementation
> and the current controlled OPSD/direct-supervision extensions. It is not a
> bit-for-bit reproduction of every result in the paper.

CycleGRPO jointly trains a Qwen3-VL + SAMTok policy for two inverse tasks:

```text
image + target mask --caption rollout--> description
description + image --localization rollout--> SAMTok mask tokens --VQ-SAM2--> mask IoU
```

The caption and localization policy gradients are computed from cycle
consistency. The current implementation additionally supports real pixel-IoU
scoring, optional teacher routing/anchors, direct referring-expression GRPO,
GT-mask CE, and DAM/DLC-QA caption reward. These additions are controlled
ablations outside the original image-mask-only CycleGRPO objective.

The detailed implementation contract and change history live in [code.md](code.md).

## Contents

- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Models and checkpoints](#models-and-checkpoints)
- [Datasets and path layout](#datasets-and-path-layout)
- [Preparing CycleGRPO Parquet data](#preparing-cyclegrpo-parquet-data)
- [Training](#training)
- [Optional DAM/DLC-QA supervision](#optional-damdlc-qa-supervision)
- [Exporting and evaluating checkpoints](#exporting-and-evaluating-checkpoints)
- [Troubleshooting](#troubleshooting)

## Repository layout

```text
verl/                                # Ray/FSDP/vLLM RL engine
  trainer/ray_trainer.py             # cycle, direct GRPO and GT-mask CE orchestration
  workers/opsd/                      # pixel IoU, routing, teacher, token parsing
  workers/supervised_anchors.py      # DLC-QA/direct/CE configuration helpers
projects/
  rl/
    config.yaml                      # base RL configuration
    qwen3vl_4b_mt.sh                 # generic paper-style entry with placeholders
    qwen3vl_4b_refcoco10k_volcengine.sh  # maintained 8-GPU server entry
    datasets/                        # RefCOCO/gRefCOCO/Stuff/PACO/DAM Parquet converters
  eval/qwen3vl_4b_volcengine.sh      # FSDP-to-HF export and supported evaluations
  transformers/vq_sam2/              # VQ-SAM2 tokenizer and SAM2 code
evaluation/
  refcoco/                           # RefCOCO cIoU/mIoU
  gres/                              # gRefCOCO/GRES gIoU, cIoU, T-acc, N-acc
  groundingsuite/                    # GroundingSuite mask gIoU
  dlc_bench/                         # DLC prediction and Llama judge utilities
tests/                               # CPU unit tests for core data/reward helpers
```

## Requirements

### Hardware and runtime

The maintained training recipe assumes one Linux node with:

- 8 NVIDIA GPUs with CUDA/NCCL peer communication;
- enough CPU RAM for FSDP optimizer offload and the Ray object store;
- a short local filesystem path for Ray (`/tmp` or `/dev/shm`), below 95% usage;
- enough persistent disk for FSDP checkpoints. A full optimizer checkpoint can
  be several GB per rank; use a small `SAVE_LIMIT` when space is constrained.

The tested server profile uses Python 3.10, PyTorch/CUDA compatible with
`vllm==0.11.0`, Ray, FSDP and FlashAttention. The package metadata permits
Python >=3.9, but that does **not** guarantee that arbitrary Torch/CUDA/vLLM
combinations can run Qwen3-VL rollout.

### Installation

Create and activate a CUDA-enabled environment first, then install the
repository and the profiles used by your workflow:

```bash
conda create -n cyclegrpo python=3.10 -y
conda activate cyclegrpo

# Install the Torch/TorchVision build matching the server CUDA driver first.
# Follow the official PyTorch selector; do not mix an arbitrary CUDA wheel with vLLM.

python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .

# Required for RefCOCO/gRefCOCO/Stuff/PACO Parquet conversion.
python -m pip install -r requirements/refcoco-data.txt

# Required for Ray/FSDP CycleGRPO training and online pixel-IoU reward.
python -m pip install -r requirements/rl-train.txt
python -m pip install --no-build-isolation -r requirements/cuda-kernels.txt
python -m pip install -r requirements/rollout-qwen3vl.txt

# Required for RefCOCO/GRES/GroundingSuite/DLC evaluation.
python -m pip install -r requirements/eval.txt

# Optional: online W&B logging. File-only logs work without it.
python -m pip install -r requirements/tracking.txt
```

Verify the critical imports in the intended environment:

```bash
python -c 'import torch, ray, vllm; print(torch.__version__, torch.cuda.is_available(), ray.__version__, vllm.__version__)'
python -m unittest tests.test_supervised_anchors tests.test_dam_caption_qa
```

## Models and checkpoints

### Base model required for training and evaluation

`MODEL_PATH` (or `TRAIN_MODEL_PATH` during evaluation) must point to a
Qwen3-VL-4B SAMTok directory that contains at least:

```text
<MODEL_PATH>/
  config.json
  model.safetensors.index.json        # or a single model.safetensors
  tokenizer.json / tokenizer_config.json / processor files
  mask_tokenizer_256x2.pth            # VQ-SAM2 discrete-mask tokenizer
  sam2.1_hiera_large.pt               # SAM2 backbone checkpoint
```

The released cold-start model can be downloaded from
[Qwen3-VL-4B-SAMTok](https://huggingface.co/zhouyik/Qwen3-VL-4B-SAMTok).
The released CycleGRPO checkpoint is
[XinNUS/CycleGRPO-4B](https://huggingface.co/XinNUS/CycleGRPO-4B).
Download it through the active environment rather than relying on a globally
installed `hf`/`huggingface-cli` executable:

```bash
python -c '
from huggingface_hub import snapshot_download
snapshot_download(repo_id="XinNUS/CycleGRPO-4B", local_dir="/path/to/workspace/CycleGRPO-4B")
'
```

### FSDP checkpoints versus Hugging Face checkpoints

Training writes FSDP shards under:

```text
<RUN_ROOT>/checkpoints/global_step_<N>/actor/model_world_size_8_rank_*.pt
```

These files cannot be passed directly to `from_pretrained` or the evaluation
scripts. First run the `export` action described below. The exported HF
directory contains the trained language-model safetensors and processor files;
the VQ-SAM2/SAM2 files remain at `TRAIN_MODEL_PATH` and are supplied separately
by the evaluation entrypoint.

To initialize a **new** training experiment from an exported HF checkpoint,
either copy or symlink the two mask files into that directory:

```bash
ln -s /path/to/base/mask_tokenizer_256x2.pth /path/to/exported_hf/mask_tokenizer_256x2.pth
ln -s /path/to/base/sam2.1_hiera_large.pt /path/to/exported_hf/sam2.1_hiera_large.pt
```

Use `RESUME=true` only for continuing the same FSDP run (same run root,
optimizer, dataloader and frozen-teacher setup). Use `RESUME=false` with an
exported HF model to start a new specialization stage.

## Datasets and path layout

The commands below use a workspace root. Set it once and adapt the paths to
your server; do not assume the example directories exist locally.

```bash
BASE_DIR=/mnt/cxzx/workspace/data_transfer/houzhiyan
REPO_DIR=$BASE_DIR/CycleGRPO-OPSD
ENV_DIR=$BASE_DIR/envs/cyclegrpo
MODEL_PATH=$BASE_DIR/Qwen3-VL-4B-SAMTok
```

### Raw datasets

| Dataset | Required files/directories | Used for |
|---|---|---|
| RefCOCO | `instances.json`, `refs(unc).p`, COCO `train2014/` | single-instance cycle data; RefCOCO evaluation |
| gRefCOCO | `instances.json`, `grefs(unc).json`, COCO `train2014/` | multi-instance union masks, no-target training, GRES evaluation |
| COCO-Stuff | `train2017/`, COCO `train2017/` | semantic Stuff masks |
| PACO-LVIS | `paco_lvis_v1_train.json`, COCO/PACO `train2017/` | parent-conditioned visible-part union masks |
| Describe Anything (optional) | DAM `COCOStuff` and `PACO` annotation JSON plus images | DAM captions and DLC-QA sidecar |
| GroundingSuite | `GroundingSuite-Eval.jsonl` and released assets | GroundingSuite evaluation only |
| DLC-Bench | `annotations.json` and images | caption prediction and external Llama judge |

Keep training and evaluation data separate. In particular, GRES evaluation is
rebuilt from official gRefCOCO annotations, not from the training parquet.

### Download public training assets

Run these commands on the training server to download the public COCO,
COCO-Stuff, RefCOCO, and PACO-LVIS assets into the layout used below. They use
`wget -c` and `unzip -n` so an interrupted download can be rerun without
overwriting extracted files. Read and accept each upstream dataset license
before downloading or redistributing it.

```bash
BASE_DIR=/mnt/cxzx/workspace/data_transfer/houzhiyan
DOWNLOAD_DIR=$BASE_DIR/downloads
mkdir -p "$DOWNLOAD_DIR"

# COCO 2014 images and instances: required by RefCOCO and gRefCOCO.
mkdir -p "$BASE_DIR/refcoco-train2014-assets"
wget -c -P "$DOWNLOAD_DIR" http://images.cocodataset.org/zips/train2014.zip
wget -c -P "$DOWNLOAD_DIR" http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip -n "$DOWNLOAD_DIR/train2014.zip" -d "$BASE_DIR/refcoco-train2014-assets"
unzip -n "$DOWNLOAD_DIR/annotations_trainval2014.zip" -d "$BASE_DIR/coco2014"
cp "$BASE_DIR/coco2014/annotations/instances_train2014.json" "$BASE_DIR/refcoco-train2014-assets/instances.json"

# COCO 2017 train images: required by COCO-Stuff and PACO-LVIS.
mkdir -p "$BASE_DIR/coco2017"
wget -c -P "$DOWNLOAD_DIR" http://images.cocodataset.org/zips/train2017.zip
unzip -n "$DOWNLOAD_DIR/train2017.zip" -d "$BASE_DIR/coco2017"

# COCO-Stuff semantic masks. The official archive expands directly to
# COCO-Stuff/train2017/ and COCO-Stuff/val2017/.
mkdir -p "$BASE_DIR/COCO-Stuff"
wget -c -P "$DOWNLOAD_DIR" http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip
unzip -n "$DOWNLOAD_DIR/stuffthingmaps_trainval2017.zip" -d "$BASE_DIR/COCO-Stuff"

# PACO-LVIS v1 annotation. PACO reuses COCO 2017 images, so expose the
# existing train split at the path used by the converter.
mkdir -p "$BASE_DIR/PACO-LVIS/annotations" "$BASE_DIR/PACO-LVIS/images"
wget -c -P "$DOWNLOAD_DIR" https://dl.fbaipublicfiles.com/paco/paco_lvis_v1.zip
unzip -n "$DOWNLOAD_DIR/paco_lvis_v1.zip" -d "$BASE_DIR/PACO-LVIS/annotations"
ln -s "$BASE_DIR/coco2017/train2017" "$BASE_DIR/PACO-LVIS/images/train2017"
```

The final `ln -s` is optional when the PACO image directory already contains a
copy of COCO `train2017`. If the target link already exists, inspect it with
`ls -ld "$BASE_DIR/PACO-LVIS/images/train2017"`; do not replace a valid
dataset directory.

RefCOCO expressions are distributed by the [Refer project](https://github.com/lichengunc/refer).
After accepting its terms, download the `refcoco.zip` package from the
[official Refer data page](https://bvisionweb1.cs.unc.edu/licheng/referit/data/)
and place `refs(unc).p` beside the COCO 2014 images:

```bash
wget -c -P "$DOWNLOAD_DIR" https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip
unzip -n "$DOWNLOAD_DIR/refcoco.zip" -d "$BASE_DIR/refer-data"
REFCOCO_REFS=$(find "$BASE_DIR/refer-data" -type f -name 'refs(unc).p' -print -quit)
cp "$REFCOCO_REFS" "$BASE_DIR/refcoco-train2014-assets/refs(unc).p"
```

The `find` step tolerates Refer releases with different top-level extraction
directories. If it finds no file, `cp` fails with its original terminal error;
inspect the extracted package before continuing.

gRefCOCO, DAM, GroundingSuite, and DLC-Bench are not mirrored by this
repository. Obtain them from their official project releases under the
respective terms, then use the placement commands below. This is intentional:
these releases may require access approval, have changing URLs, or include
evaluation-only assets that should not be mixed into training data.

| Dataset | Official release | Required local placement |
|---|---|---|
| gRefCOCO | [gRefCOCO](https://github.com/heng-hw/GRIT) release/instructions | `$BASE_DIR/gRefCOCO/grefs(unc).json` and `$BASE_DIR/gRefCOCO/instances.json` |
| Describe Anything (DAM) | [Describe Anything](https://github.com/ttxskk/Describe-Anything) release/instructions | `$BASE_DIR/datasets/dam_data/COCOStuff` and `$BASE_DIR/datasets/dam_data/PACO` |
| GroundingSuite | [GroundingSuite](https://github.com/hustvl/GroundingSuite) release/instructions | `$BASE_DIR/GSEval/GroundingSuite-Eval.jsonl` and its released assets |
| DLC-Bench | the official Describe Anything release | `$BASE_DIR/describe-anything/evaluation/DLC-Bench/annotations.json` and images |

For a supplied gRefCOCO package that contains an instance JSON under a
different name, use the same COCO 2014 `instances_train2014.json` copied above
only when the official release specifies that it shares that annotation file.

### Training parquet contract

The RL loader requires image paths and fields including `cap_problem`,
`seg_answer`, `masks`, and `source`. Current converters additionally save
`grounding_query` for optional direct supervision. Positive `cap_answer` is
cleared by the mixer so the main CycleGRPO caption rollout does not consume
human referring expressions. `grounding_query` is used only by explicitly
enabled direct GRPO/GT-mask CE; it does not alter the image-mask-only cycle
objective by default.

The maintained sources are:

| Source | Meaning |
|---|---|
| `refcoco_cycle` | single-instance RefCOCO region |
| `grefcoco_cycle` | gRefCOCO positive union target; multi quota requires >=2 annotations |
| `gres_no_target` | gRefCOCO null target with the existing refusal reward |
| `cocostuff_cycle` | complete semantic Stuff-class union |
| `paco_part_cycle` | visible parts of one parent category; PACO v1 has no reliable per-mask part label |

## Preparing CycleGRPO Parquet data

All converters encode every target mask through VQ-SAM2 and therefore need one
GPU and the two mask checkpoint files. They write absolute image paths; rerun
the converter or repair the parquet when moving to a different filesystem.

### RefCOCO

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO_DIR" "$ENV_DIR/bin/python3" \
  projects/rl/datasets/prepare_refcoco_rl_dataset.py \
  --instances "$BASE_DIR/refcoco-train2014-assets/instances.json" \
  --refs "$BASE_DIR/refcoco-train2014-assets/refs(unc).p" \
  --images-dir "$BASE_DIR/refcoco-train2014-assets/train2014" \
  --output "$BASE_DIR/datasets/refcoco/refcoco_train_10k.parquet" \
  --mask-tokenizer-path "$MODEL_PATH/mask_tokenizer_256x2.pth" \
  --sam2-checkpoint "$MODEL_PATH/sam2.1_hiera_large.pt" \
  --sam2-config-dir "$REPO_DIR/projects/transformers/vq_sam2/sam2/sam2_configs" \
  --split train --max-samples 10000 --seed 20260815 --device cuda
```

Use `--max-samples 42404` to export the complete RefCOCO train split when all
references are valid.

### gRefCOCO, COCO-Stuff and PACO-LVIS

```bash
# gRefCOCO: 6,250 true multi-instance positives + 2,500 no-target expressions.
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO_DIR" "$ENV_DIR/bin/python3" projects/rl/datasets/prepare_grefcoco_cycle_dataset.py --instances "$BASE_DIR/gRefCOCO/instances.json" --grefs "$BASE_DIR/gRefCOCO/grefs(unc).json" --images-dir "$BASE_DIR/refcoco-train2014-assets/train2014" --output-dir "$BASE_DIR/datasets/grefcoco_8750" --mask-tokenizer-path "$MODEL_PATH/mask_tokenizer_256x2.pth" --sam2-checkpoint "$MODEL_PATH/sam2.1_hiera_large.pt" --sam2-config-dir "$REPO_DIR/projects/transformers/vq_sam2/sam2/sam2_configs" --positive-samples 6250 --no-target-samples 2500 --single-fraction 0.0 --seed 20260815 --device cuda

# COCO-Stuff: official semantic PNG masks are in COCO-Stuff/train2017.
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO_DIR" "$ENV_DIR/bin/python3" projects/rl/datasets/prepare_cocostuff_cycle_dataset.py --masks-dir "$BASE_DIR/COCO-Stuff/train2017" --images-dir "$BASE_DIR/coco2017/train2017" --output "$BASE_DIR/datasets/cocostuff_5k.parquet" --mask-tokenizer-path "$MODEL_PATH/mask_tokenizer_256x2.pth" --sam2-checkpoint "$MODEL_PATH/sam2.1_hiera_large.pt" --sam2-config-dir "$REPO_DIR/projects/transformers/vq_sam2/sam2/sam2_configs" --max-samples 5000 --seed 20260815 --device cuda

# PACO-LVIS: use actual PACO annotation and image paths.
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO_DIR" "$ENV_DIR/bin/python3" projects/rl/datasets/prepare_paco_lvis_part_cycle_dataset.py --annotations "$BASE_DIR/PACO-LVIS/annotations/paco_lvis_v1_train.json" --images-dir "$BASE_DIR/PACO-LVIS/images/train2017" --output "$BASE_DIR/datasets/paco_part_2500.parquet" --mask-tokenizer-path "$MODEL_PATH/mask_tokenizer_256x2.pth" --sam2-checkpoint "$MODEL_PATH/sam2.1_hiera_large.pt" --sam2-config-dir "$REPO_DIR/projects/transformers/vq_sam2/sam2/sam2_configs" --max-samples 2500 --seed 20260815 --device cuda
```

### Build a balanced 25k mixture

The following recipe is 35% RefCOCO single, 25% gRefCOCO true multi, 20%
Stuff, 10% PACO part and 10% no-target. It preserves per-source grounding
queries for optional supervised anchors.

```bash
PYTHONPATH="$REPO_DIR" "$ENV_DIR/bin/python3" \
  projects/rl/datasets/prepare_balanced_cyclegrpo_dataset.py \
  --refcoco "$BASE_DIR/datasets/refcoco/refcoco_train_10k.parquet" \
  --grefcoco "$BASE_DIR/datasets/grefcoco_8750/grefcoco_train_6250pos_2500notarget_seed20260815_combined.parquet" \
  --cocostuff "$BASE_DIR/datasets/cocostuff_5k.parquet" \
  --paco-parts "$BASE_DIR/datasets/paco_part_2500.parquet" \
  --output "$BASE_DIR/datasets/groundingsuite_25k.parquet" \
  --single-count 8750 --multi-count 6250 --stuff-count 5000 --part-count 2500 --no-target-count 2500 \
  --require-grounding-query --seed 20260815
```

Inspect the companion `.manifest.json` before training. The mixer fails instead
of silently replacing a multi-instance quota with single-instance data.

## Training

### Recommended eight-GPU server entry

`projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` is the maintained entry for
the current architecture. It validates model files and every parquet image
path, clears an incompatible inherited `RAY_ADDRESS`, creates a short local Ray
directory, and writes stdout to `<RUN_ROOT>/train_<timestamp>.log`.

The default controlled setup uses `G=6` caption rollouts and `K=6`
localization rollouts. With OPSD enabled, localization uses decoded pixel IoU;
all complete, codebook-valid SAMTok groups in one response are decoded and
unioned before scoring. The default teacher/routing settings are documented in
`code.md`.

```bash
RUN_NAME=gs25k_cycle
RUN_ROOT=$REPO_DIR/logs/$RUN_NAME
TRAIN_DATA=$BASE_DIR/datasets/groundingsuite_25k.parquet

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH="$MODEL_PATH" RUN_NAME="$RUN_NAME" RUN_ROOT="$RUN_ROOT" \
TRAIN_DATA="$TRAIN_DATA" VAL_DATA="$TRAIN_DATA" \
TOTAL_EPOCHS=1 RESUME=false SAVE_FREQ=25 SAVE_LIMIT=2 \
OPSD_ENABLED=true PIXEL_IOU_ENABLED=true ROUTING_ENABLED=true \
GROUNDEDNESS_ENABLED=false TRAINER_LOGGERS='["file"]' \
bash "$REPO_DIR/projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh"
```

`SAVE_LIMIT=2` is important on limited storage. A failed checkpoint write
usually indicates a full or quota-limited filesystem; check `df -h` and remove
only checkpoints you no longer need before restarting.

### Optional direct referring supervision

Direct supervision is disabled by default. When enabled it is **additive** to
the cycle update:

```text
L_total = 0.5 L_cycle_caption + 0.5 L_cycle_localization
        + lambda_direct(step) L_direct_GRPO + 0.02 L_direct_mask_CE
        + existing regenerate / JSD / KL auxiliary losses
```

- Direct GRPO samples `K=6` masks from stored human referring expressions and
  uses independent GRPO UID groups.
- The default schedule is zero through step 10, linearly grows until step 30,
  then reaches `DIRECT_GROUNDING_LOSS_WEIGHT`.
- GT-mask CE teacher-forces one RefCOCO/gRefCOCO positive expression per
  original UID. It excludes no-target and Stuff/PACO label templates.
- gRefCOCO no-target remains in the original outer caption GRPO path even if
  it is also selected for direct supervision.

#### Export the full RefCOCO train Parquet

For a direct/GT-mask CE specialization after a mixed-data run, export the full
RefCOCO train split rather than reusing the 10k subset. The official
`refs(unc).p` train split contains 42,404 references; this converter validates
each image and target mask before writing the Parquet.

```bash
REF_FULL_DIR=$BASE_DIR/datasets/refcoco_full_train
REF_FULL_PA=$REF_FULL_DIR/refcoco_train_full_42404_seed20260815.parquet
mkdir -p "$REF_FULL_DIR"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO_DIR" "$ENV_DIR/bin/python3" \
  projects/rl/datasets/prepare_refcoco_rl_dataset.py \
  --instances "$BASE_DIR/refcoco-train2014-assets/instances.json" \
  --refs "$BASE_DIR/refcoco-train2014-assets/refs(unc).p" \
  --images-dir "$BASE_DIR/refcoco-train2014-assets/train2014" \
  --output "$REF_FULL_PA" \
  --mask-tokenizer-path "$MODEL_PATH/mask_tokenizer_256x2.pth" \
  --sam2-checkpoint "$MODEL_PATH/sam2.1_hiera_large.pt" \
  --sam2-config-dir "$REPO_DIR/projects/transformers/vq_sam2/sam2/sam2_configs" \
  --split train --max-samples 42404 --seed 20260815 --device cuda
```

Use `TRAIN_DATA="$REF_FULL_PA"` in the direct training command below when
running this RefCOCO-only specialization. The converter writes absolute image
paths, so rerun it after moving the COCO 2014 image directory.

Enable the two direct terms for a controlled experiment:

```bash
RUN_NAME=gs25k_direct_ce
RUN_ROOT=$REPO_DIR/logs/$RUN_NAME
TRAIN_DATA=$BASE_DIR/datasets/groundingsuite_25k.parquet

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH="$MODEL_PATH" RUN_NAME="$RUN_NAME" RUN_ROOT="$RUN_ROOT" \
TRAIN_DATA="$TRAIN_DATA" VAL_DATA="$TRAIN_DATA" \
TOTAL_EPOCHS=1 RESUME=false SAVE_FREQ=25 SAVE_LIMIT=2 \
OPSD_ENABLED=true PIXEL_IOU_ENABLED=true ROUTING_ENABLED=true \
GROUNDEDNESS_ENABLED=false TRAINER_LOGGERS='["file"]' \
DIRECT_GROUNDING_ENABLED=true \
DIRECT_GROUNDING_ROLLOUTS=6 \
DIRECT_GROUNDING_LOSS_WEIGHT=0.15 \
DIRECT_GROUNDING_WARMUP_START_STEP=10 \
DIRECT_GROUNDING_WARMUP_END_STEP=30 \
DIRECT_GROUNDING_INCLUDE_POSITIVE_SOURCES=true \
DIRECT_GROUNDING_INCLUDE_NO_TARGET=true \
DIRECT_GROUNDING_INCLUDE_LABEL_SOURCES=false \
DIRECT_MASK_CE_ENABLED=true \
DIRECT_MASK_CE_LOSS_WEIGHT=0.02 \
bash "$REPO_DIR/projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh"
```

This is a new run; use a distinct `RUN_NAME`/`RUN_ROOT` and change
`TRAIN_DATA` only after the parquet is prepared. Do not set
`DIRECT_GROUNDING_CONSUME_NO_TARGET_CAPTION=true`; the launcher rejects it.

### Resume versus specialization

Continue an interrupted run only when its checkpoint was saved successfully:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
RESUME=true RUN_ROOT="$REPO_DIR/logs/gs25k_cycle" \
MODEL_PATH="$MODEL_PATH" TRAIN_DATA="$TRAIN_DATA" VAL_DATA="$TRAIN_DATA" \
bash "$REPO_DIR/projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh"
```

To specialize an exported step-195 model on full RefCOCO, set `MODEL_PATH` to
the exported HF directory (after adding the two VQ-SAM2/SAM2 links), choose a
new `RUN_ROOT`, and use `RESUME=false`. This intentionally resets optimizer and
global-step state.

## Optional DAM/DLC-QA supervision

DAM captions are never put into the actor caption prompt. They are used only
to create a text-only QA sidecar for the selected DAM-backed Stuff/PACO rows.

1. Convert DAM regions with `prepare_dam_cycle_dataset.py`, once per source.
   Its `--caption-manifest` output records `dam_source_id`, caption and source.
   For PACO, provide `--paco-annotations` so only verified non-parent part
   masks are retained.
2. Generate and validate QA using a running OpenAI-compatible LLM endpoint:

```bash
PYTHONPATH="$REPO_DIR" "$ENV_DIR/bin/python3" \
  projects/rl/datasets/generate_dam_caption_qa.py \
  --input-manifest "$BASE_DIR/datasets/dam/dam_cocostuff_manifest.jsonl" \
  --input-manifest "$BASE_DIR/datasets/dam/dam_paco_manifest.jsonl" \
  --output "$BASE_DIR/datasets/dam/dam_caption_qa_5k.jsonl" \
  --rejected-output "$BASE_DIR/datasets/dam/dam_caption_qa_5k.rejected.jsonl" \
  --base-url http://127.0.0.1:8007/v1 --api-key sk-abc123 \
  --model llama3.1-8b --validator-model llama3.1-8b \
  --max-concurrency 8 --generation-attempts 5 --request-retries 3 --seed 20260815
```

3. Mix the **accepted** QA sidecar into a DAM-backed parquet. The mixer requires
exactly 3,000 Stuff and 2,000 PACO accepted IDs unless its QA count arguments
are explicitly changed:

```bash
PYTHONPATH="$REPO_DIR" "$ENV_DIR/bin/python3" projects/rl/datasets/prepare_balanced_cyclegrpo_dataset.py \
  --refcoco /path/to/refcoco.parquet --grefcoco /path/to/grefcoco.parquet \
  --cocostuff /path/to/dam_stuff.parquet --paco-parts /path/to/dam_paco.parquet \
  --caption-qa-manifest "$BASE_DIR/datasets/dam/dam_caption_qa_5k.jsonl" \
  --output /path/to/dam_balanced_25k.parquet \
  --single-count 8750 --multi-count 6250 --stuff-count 5000 --part-count 2500 --no-target-count 2500 \
  --seed 20260815
```

DAM rows intentionally have `grounding_query=null`, so do **not** pass
`--require-grounding-query` to this DAM-backed mixture. That flag is for a
five-source direct-grounding experiment in which every selected row has a
localization query.

4. Add these variables to a complete training invocation (including its
   `CUDA_VISIBLE_DEVICES`, `MODEL_PATH`, `RUN_ROOT`, and `TRAIN_DATA`):

```bash
SUPERVISED_CAPTION_QA_ENABLED=true \
CAPTION_QA_JSONL="$BASE_DIR/datasets/dam/dam_caption_qa_5k.jsonl" \
CAPTION_QA_JUDGE_BASE_URL=http://127.0.0.1:8007/v1 \
CAPTION_QA_JUDGE_MODEL=llama3.1-8b \
CAPTION_QA_JUDGE_API_KEY=sk-abc123 \
CAPTION_QA_REWARD_WEIGHT=1.0 \
bash "$REPO_DIR/projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh"
```

The QA reward is an external caption reward added to eligible DAM caption
rollouts. It does not change cycle `R_Ci`, pixel-IoU calculation, or teacher
routing. Inspect rejected QA JSONL and resume incomplete generation with
`--resume`; do not add rejected records to the mixer.

## Exporting and evaluating checkpoints

The unified evaluation entry supports `export`, `refcoco`, `groundingsuite`,
`gres`, `dlc`, and `all`. All supported segmentation evaluators decode the
first complete valid SAMTok group, so regenerate old checkpoints under the same
protocol before comparing values.

```bash
CKPT=$REPO_DIR/logs/gs25k_cycle/checkpoints/global_step_195
OUT=$REPO_DIR/logs/gs25k_cycle/evaluation/step_195
HF_MODEL=$OUT/hf_global_step_195
EVAL_SCRIPT=$REPO_DIR/projects/eval/qwen3vl_4b_volcengine.sh

# Convert world-size-8 FSDP actor shards to standard HF safetensors.
CHECKPOINT_PATH="$CKPT" HF_MODEL_PATH="$HF_MODEL" EVAL_ROOT="$OUT" \
TRAIN_MODEL_PATH="$MODEL_PATH" TRAIN_DATA="$TRAIN_DATA" NUM_GPUS=8 \
bash "$EVAL_SCRIPT" export

# RefCOCO val cIoU/mIoU. The H20 recipe starts at 16 images per GPU; raise to
# 24 or 32 only after one successful run confirms the available headroom.
HF_MODEL_PATH="$HF_MODEL" EVAL_ROOT="$OUT" TRAIN_MODEL_PATH="$MODEL_PATH" \
REFCOCO_ROOT="$BASE_DIR/refcoco-train2014-assets" REFCOCO_SPLIT=val NUM_GPUS=8 EVAL_BATCH_SIZE=16 \
bash "$EVAL_SCRIPT" refcoco

# GroundingSuite mask GIoU. legacy_union is the historical SAMTok-compatible protocol.
HF_MODEL_PATH="$HF_MODEL" EVAL_ROOT="$OUT" TRAIN_MODEL_PATH="$MODEL_PATH" MASK_PROTOCOL=legacy_union \
GROUNDINGSUITE_ROOT="$BASE_DIR/GSEval" REFCOCO_ROOT="$BASE_DIR/refcoco-train2014-assets" NUM_GPUS=8 \
bash "$EVAL_SCRIPT" groundingsuite

# gRefCOCO/GRES target and no-target metrics.
HF_MODEL_PATH="$HF_MODEL" EVAL_ROOT="$OUT" TRAIN_MODEL_PATH="$MODEL_PATH" MASK_PROTOCOL=legacy_union \
GRES_ROOT="$BASE_DIR/gRefCOCO" GRES_SPLIT=val NUM_GPUS=8 \
bash "$EVAL_SCRIPT" gres

# DLC-Bench prediction JSON only; score it with a judge separately.
HF_MODEL_PATH="$HF_MODEL" EVAL_ROOT="$OUT" TRAIN_MODEL_PATH="$MODEL_PATH" \
DLC_ROOT="$BASE_DIR/describe-anything/evaluation/DLC-Bench" \
bash "$EVAL_SCRIPT" dlc
```

Expected outputs include:

```text
<OUT>/hf_global_step_195/             # HF safetensors export
<OUT>/refcoco_val/                    # per-sample predictions + aggregate metrics
<OUT>/groundingsuite/                 # per-sample predictions
<OUT>/groundingsuite_metrics.json
<OUT>/gres/case_*.json
<OUT>/gres_metrics.json
<OUT>/dlc_bench_predictions.json
```

For offline GRES subsets such as target area, cardinality, and two-instance
geometry, reuse complete predictions rather than running inference again:

```bash
PYTHONPATH="$REPO_DIR" "$ENV_DIR/bin/python3" evaluation/gres/qwen3vl_gres_eval.py \
  --metric-only --grefs-file "$BASE_DIR/gRefCOCO/grefs(unc).json" \
  --instances-file "$BASE_DIR/gRefCOCO/instances.json" \
  --split val --dataset "$OUT/gres_val_samples.json" --save-dir "$OUT/gres" \
  --subset-report-file "$OUT/gres_subset_metrics.jsonl"
```

### DLC judge

Run the Llama judge in a separate terminal, substituting an actual local
Llama-3.1-8B-Instruct HF directory. Invoke `vllm serve` directly: the legacy
`serve_judge.sh` hard-codes its own model path and does not honor a caller's
`MODEL_PATH`.

```bash
JUDGE_MODEL_PATH=/path/to/Meta-Llama-3.1-8B-Instruct
CUDA_VISIBLE_DEVICES=0 "$ENV_DIR/bin/vllm" serve "$JUDGE_MODEL_PATH" \
  --served-model-name llama3.1-8b --api-key sk-abc123 \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 --trust-remote-code \
  --dtype bfloat16 --gpu-memory-utilization 0.85 --port 8007 --host localhost
```

Then score the exported prediction file:

```bash
PYTHONPATH="$REPO_DIR" "$ENV_DIR/bin/python3" \
  evaluation/dlc_bench/eval_llama_without_image.py \
  --pred "$OUT/dlc_bench_predictions.json" \
  --base-url http://127.0.0.1:8007/v1 \
  --api-key sk-abc123 \
  --model llama3.1-8b
```

## Troubleshooting

| Symptom | Cause and action |
|---|---|
| `ModuleNotFoundError: imageio` during legacy evaluation | Install `imageio` in the active environment. The maintained server evaluation path uses the profiles above. |
| `HF export missing` | FSDP shards are not an HF model. Run `bash projects/eval/qwen3vl_4b_volcengine.sh export` with the correct `CHECKPOINT_PATH`. |
| Missing `mask_tokenizer_256x2.pth` after using exported HF model for training | Link/copy the two VQ-SAM2/SAM2 files from the base SAMTok directory into the export directory. |
| All RefCOCO predictions are `No target.` | Inspect the per-sample response JSON and training prompt/data configuration. This is model behavior, not an evaluator issue. |
| Checkpoint `PytorchStreamWriter failed writing file` | Persistent disk/quota is insufficient. Reduce `SAVE_LIMIT`, free old checkpoints, and restart from the last valid checkpoint. |
| `KeyError: 0` in `_make_direct_mask_ce_batch` | Sync the current `ray_trainer.py` and `supervised_anchors.py`; old code incorrectly indexed an unwrapped media dictionary. |
| `gRefCOCO refs file not found` | The directory is case-sensitive: the maintained default is `$BASE_DIR/gRefCOCO`, not `grefcoco`. |
| Training shell exits while pasting a multi-line command | Paste variables and commands separately, or use a single-line command. A trailing `\` joins the next line into the same command. |

## Results and citation

The released CycleGRPO results and paper are available on the
[project page](https://devinxzhang.github.io/CycleGRPO-Page/) and
[arXiv](https://arxiv.org/abs/2607.11581). Record the exact data mixture,
prompt protocol, direct/QA flags, checkpoint step and evaluation version with
every comparison: the controlled extensions in this repository are not the
paper's original image-mask-only setup.

```bibtex
@inproceedings{cyclegrpo2026,
  title     = {Actor as Its Own Critic: Unifying Region Understanding and Localization via CycleGRPO},
  author    = {Zhang, Xin and Wang, Haochen and Zhou, Yikang and Wang, Zhuochen and Li, Jason and Tan, Robby T.},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## Acknowledgements

Built on [EasyR1](https://github.com/hiyouga/EasyR1) and
[veRL](https://github.com/volcengine/verl), with SAMTok and
[SAM2](https://github.com/facebookresearch/sam2).
