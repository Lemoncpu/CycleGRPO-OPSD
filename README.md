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
- [70k end-to-end recipe](#70k-end-to-end-recipe)
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

## 70k end-to-end recipe

This is the reproducible recipe for the current 70k experiment. It uses one
node with seven Ray training GPUs (`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6`) and
GPU 7 reserved for the local Llama-3.1-8B judge. It runs 20k CycleGRPO data,
30k direct RefCOCO data, 10k gRefCOCO no-target data and 10k DLC-QA data for
one epoch. SECA, direct GRPO, direct mask CE, pixel-empty no-target reward and
the positive no-target empty-mask penalty are enabled; the no-target nonempty
penalty is deliberately disabled.

The commands below assume a new server and use `/mnt/opsd` as its workspace.
Replace `REPO_URL` with the repository mirror available on that server. All
generated data currently used by this project is mirrored at
[`Untitled111/train-opsd`](https://huggingface.co/datasets/Untitled111/train-opsd).

### 1. Clone and install

```bash
export BASE_DIR=/mnt/opsd
export REPO_DIR="$BASE_DIR/CycleGRPO-OPSD"
export ENV_DIR="$BASE_DIR/envs/cyclegrpo"
export DOWNLOAD_DIR="$BASE_DIR/downloads"
mkdir -p "$BASE_DIR" "$DOWNLOAD_DIR"
export REPO_URL=https://github.com/Lemoncpu/CycleGRPO-OPSD.git
git clone "$REPO_URL" "$REPO_DIR"
conda create -p "$ENV_DIR" python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
cd "$REPO_DIR"
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install -r requirements/refcoco-data.txt -r requirements/rl-train.txt \
  -r requirements/cuda-kernels.txt -r requirements/rollout-qwen3vl.txt \
  -r requirements/eval.txt
```

The 70k path is complete with these profiles: `base` (Hydra/Transformers),
`refcoco-data` (Parquet/COCO utilities), `rl-train` (Ray/FSDP/PEFT),
`cuda-kernels` (FlashAttention), `rollout-qwen3vl` (vLLM 0.11.0), and `eval`
(COCO and evaluation utilities). `huggingface_hub` is installed explicitly in
step 2 for model/data snapshots. `tracking.txt` is optional because the canonical
70k command uses file-only logging and does not require W&B. The host must also
provide `git`, `conda`, `wget`, `unzip`, and `curl`; these are system tools, not
Python dependencies.

Run this import smoke before downloading the large checkpoints:

```bash
"$ENV_DIR/bin/python3" - <<'PY'
mods = (
    "torch", "torchvision", "transformers", "qwen_vl_utils", "ray", "vllm",
    "accelerate", "codetiming", "datasets", "iopath", "openai", "peft",
    "pyarrow", "pycocotools", "pycocoevalcap", "tensordict", "torchdata",
    "flash_attn", "einops", "scipy", "requests", "tqdm",
)
for name in mods:
    __import__(name)
    print(f"import ok: {name}")
print("70k runtime dependency smoke passed")
PY
```

### 2. Download models and raw assets

Use the exact Hugging Face repositories below. The RefCOCO image/annotation bundle is
kept in its own same-name dataset; all other assets that were uploaded from the
current server are downloaded from `Untitled111/train-opsd`.

```bash
export MODEL_PATH="$BASE_DIR/Qwen3-VL-4B-SAMTok"
export LLAMA_PATH="$BASE_DIR/Meta-Llama-3.1-8B-Instruct"
export REFCOCO_ASSET_DIR="$BASE_DIR/refcoco-train2014-assets"
export TRAIN_OPSD_DIR="$BASE_DIR/train-opsd"
python -m pip install -U huggingface_hub
huggingface-cli login
huggingface-cli download zhouyik/Qwen3-VL-4B-SAMTok --local-dir "$MODEL_PATH"
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir "$LLAMA_PATH"

# RefCOCO/gRefCOCO COCO-2014 images, instances and refs. This archive is 13.5 GB.
huggingface-cli download Untitled111/refcoco-train2014-assets --repo-type dataset \
  --local-dir "$REFCOCO_ASSET_DIR"
unzip -n "$REFCOCO_ASSET_DIR/train2014.zip" -d "$REFCOCO_ASSET_DIR"

# COCO-Stuff masks, COCO2014 annotations, COCO2017 image archive and gRefCOCO.
huggingface-cli download Untitled111/train-opsd --repo-type dataset --local-dir "$TRAIN_OPSD_DIR"
mkdir -p "$BASE_DIR/COCO-Stuff" "$BASE_DIR/coco2014" "$BASE_DIR/coco2017" "$BASE_DIR/gRefCOCO"
cp -a "$TRAIN_OPSD_DIR/COCO-Stuff/." "$BASE_DIR/COCO-Stuff/"
cp -a "$TRAIN_OPSD_DIR/coco2014/." "$BASE_DIR/coco2014/"
unzip -n "$TRAIN_OPSD_DIR/coco2017/train2017.zip" -d "$BASE_DIR/coco2017"
unzip -n "$TRAIN_OPSD_DIR/coco2017/unlabeled2017.zip" -d "$BASE_DIR/coco2017"
cp -a "$TRAIN_OPSD_DIR/gRefCOCO/." "$BASE_DIR/gRefCOCO/"

# Generated DAM annotations and all generated training parquet/JSONL files.
mkdir -p "$BASE_DIR/datasets"
cp -a "$TRAIN_OPSD_DIR/datasets/." "$BASE_DIR/datasets/"

# PACO raw annotations are only needed when regenerating the uploaded PACO parquet.
# The 70k run itself consumes encoded masks and does not need this download.
mkdir -p "$BASE_DIR/PACO-LVIS/annotations" "$BASE_DIR/PACO-LVIS/images"
wget -c -P "$DOWNLOAD_DIR" https://dl.fbaipublicfiles.com/paco/paco_lvis_v1.zip
unzip -n "$DOWNLOAD_DIR/paco_lvis_v1.zip" -d "$BASE_DIR/PACO-LVIS/annotations"
ln -sfn "$BASE_DIR/coco2017/train2017" "$BASE_DIR/PACO-LVIS/images/train2017"

# DAM raw image shards are needed only to regenerate DAM captions; annotations and
# the 10k DLC-QA sidecar are already in train-opsd.
huggingface-cli download nvidia/describe-anything-dataset --repo-type dataset \
  --local-dir "$BASE_DIR/dam_raw"

# Evaluation-only assets used by GroundingSuite and DLC-Bench.
huggingface-cli download hustvl/GSEval --repo-type dataset --local-dir "$BASE_DIR/third_party/GroundingSuite"
huggingface-cli download nvidia/DLC-Bench --repo-type dataset --local-dir "$BASE_DIR/third_party/DLC-Bench"
```

The gated Llama download requires accepting the model license first. SAMTok is
[zhouyik/Qwen3-VL-4B-SAMTok](https://huggingface.co/zhouyik/Qwen3-VL-4B-SAMTok),
Llama is [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct),
the RefCOCO bundle is
[Untitled111/refcoco-train2014-assets](https://huggingface.co/datasets/Untitled111/refcoco-train2014-assets),
and the uploaded assets/parquet are in
[Untitled111/train-opsd](https://huggingface.co/datasets/Untitled111/train-opsd).
The raw DAM collection is
[nvidia/describe-anything-dataset](https://huggingface.co/datasets/nvidia/describe-anything-dataset),
GroundingSuite is [hustvl/GSEval](https://huggingface.co/datasets/hustvl/GSEval),
and DLC-Bench is [nvidia/DLC-Bench](https://huggingface.co/datasets/nvidia/DLC-Bench).


### 3. Download the complete generated-data collection

`train-opsd` contains the exact generated files used by the 70k command. Copy
only the repository subtrees; do not rename files because the launcher and the
path-rewrite preflight use these canonical paths.

```bash
mkdir -p "$BASE_DIR/datasets"
cp -a "$TRAIN_OPSD_DIR/datasets/." "$BASE_DIR/datasets/"

# Exact 70k files (20k cycle + 30k direct + 10k no-target + 10k DLC-QA).
export TRAIN_DATA="$BASE_DIR/datasets/cyclegrpo_20k_raw_seed20260820/cyclegrpo_20k_40_20_25_10_5_seed20260820.parquet"
export DIRECT_TRAIN_DATA="$BASE_DIR/datasets/direct_refcoco30k_disjoint_cycle20k/refcoco_train_30k_disjoint_cycle20k_seed20260823.parquet"
export DIRECT_NO_TARGET_TRAIN_DATA="$BASE_DIR/datasets/grefcoco_no_target_direct_10k/grefcoco_train_0pos_10000notarget_seed20260821_no_target.parquet"
export CAPTION_QA_TRAIN_DATA="$BASE_DIR/datasets/dlc_qa/dlc_qa_10000.parquet"
export CAPTION_QA_JSONL="$BASE_DIR/datasets/dlc_qa/dam_caption_qa_10000.jsonl"
```

The supervised segmentation stream is **40,000 samples total**, composed as follows:

| Role | Canonical file | Rows | `source` |
|---|---|---:|---|
| RefCOCO positive direct supervision | `datasets/direct_refcoco30k_disjoint_cycle20k/refcoco_train_30k_disjoint_cycle20k_seed20260823.parquet` | 30,000 | `refcoco_cycle` |
| gRefCOCO no-target direct supervision | `datasets/grefcoco_no_target_direct_10k/grefcoco_train_0pos_10000notarget_seed20260821_no_target.parquet` | 10,000 | `gres_no_target` |
| **Total** | `DIRECT_TRAIN_DATA` + `DIRECT_NO_TARGET_TRAIN_DATA` | **40,000** | — |

`train-opsd` also contains the convenience file
`datasets/direct_refcoco30k_notarget10k_disjoint_cycle20k/direct_train_40k_refcoco30k_notarget10k_disjoint_cycle20k_seed20260823.parquet`, whose verified composition is 30,000 `refcoco_cycle` + 10,000 `gres_no_target`. The 70k launcher intentionally uses the split pair above because it appends `DIRECT_NO_TARGET_TRAIN_DATA` when no-target GRPO/CE is enabled; do not pass the merged 40k file together with the separate 10k file.

The same files are also visible at the dataset root as convenience copies, but
the paths above are the canonical layout consumed by the training launcher.
The uploaded `datasets/dam_data/` contains DAM annotation metadata; use the
`dam_raw` download in step 2 when the raw DAM image shards are needed.


### 4. Rewrite image paths (mandatory)

Parquet records contain absolute paths from the source server. Rewrite and
validate every image before training; do not rely on row counts alone.

```bash
export PYTHON_BIN="$ENV_DIR/bin/python3"
export REF_IMAGE_ROOT="$BASE_DIR/refcoco-train2014-assets"
export COCO17_IMAGE_ROOT="$BASE_DIR/coco2017"
"$PYTHON_BIN" - "$BASE_DIR" <<'PY'
import os, sys
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

base = Path(sys.argv[1])
files = {
    base / "datasets/cyclegrpo_20k_raw_seed20260820/cyclegrpo_20k_40_20_25_10_5_seed20260820.parquet": 20000,
    base / "datasets/direct_refcoco30k_disjoint_cycle20k/refcoco_train_30k_disjoint_cycle20k_seed20260823.parquet": 30000,
    base / "datasets/grefcoco_no_target_direct_10k/grefcoco_train_0pos_10000notarget_seed20260821_no_target.parquet": 10000,
    base / "datasets/dlc_qa/dlc_qa_10000.parquet": 10000,
}
old_ref = "/volume/ybo/xyc/refcoco-train2014-assets"
old_coco = "/volume/ybo/xyc/coco2017"
new_ref = os.environ["REF_IMAGE_ROOT"]
new_coco = os.environ["COCO17_IMAGE_ROOT"]
for path, expected in files.items():
    table = pq.read_table(path)
    if table.num_rows != expected:
        raise RuntimeError(f"{path}: expected {expected} rows, got {table.num_rows}")
    values = table["images"].to_pylist()
    rewritten = []
    for value in values:
        one = value if isinstance(value, list) else [value]
        out = []
        for item in one:
            item = str(item).replace(old_ref, new_ref).replace(old_coco, new_coco)
            if not Path(item).is_file():
                raise FileNotFoundError(f"{path}: missing image {item}")
            out.append(item)
        rewritten.append(out if isinstance(value, list) else out[0])
    table = table.set_column(table.schema.get_field_index("images"), "images", pa.array(rewritten, type=table["images"].type))
    tmp = path.with_suffix(path.suffix + ".rewrite.tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)
    print(f"rewrote {path} rows={table.num_rows}")
PY

"$PYTHON_BIN" - "$BASE_DIR" <<'PY'
import json, sys
from pathlib import Path
import pyarrow.parquet as pq
base = Path(sys.argv[1])
ids = {str(r["dam_source_id"]) for r in pq.read_table(base / "datasets/dlc_qa/dlc_qa_10000.parquet").to_pylist()}
qa = {str(json.loads(x)["dam_source_id"]) for x in (base / "datasets/dlc_qa/dam_caption_qa_10000.jsonl").read_text().splitlines()}
if len(ids) != 10000 or len(qa) != 10000 or ids != qa:
    raise RuntimeError(f"DLC-QA join failed: parquet={len(ids)} jsonl={len(qa)}")
print("validated DLC-QA dam_source_id join: 10000 rows")
PY
```

### 5. Start 70k training

[`opsd_70k_positive_empty_penalty_only_8gpu.txt`](opsd_70k_positive_empty_penalty_only_8gpu.txt)
starts Ray and the local Llama judge itself. Copy it to the new workspace,
rewrite its source-server prefixes, check syntax, then run it from any
directory:

```bash
cp "$REPO_DIR/opsd_70k_positive_empty_penalty_only_8gpu.txt" "$BASE_DIR/run_opsd_70k.sh"
sed -i "s#/volume/ybo/xyc#$BASE_DIR#g" "$BASE_DIR/run_opsd_70k.sh"
sed -i "s#$BASE_DIR/models/Meta-Llama-3.1-8B-Instruct-hf-v2#$LLAMA_PATH#g" "$BASE_DIR/run_opsd_70k.sh"
bash -n "$BASE_DIR/run_opsd_70k.sh"
bash "$BASE_DIR/run_opsd_70k.sh"
```

The launcher must report `NUM_GPUS=7`, `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6`,
`SECA_ENABLED=true`, direct GRPO/CE and DLC-QA enabled,
`NO_TARGET_REWARD_MODE=pixel_empty`, `POSITIVE_EMPTY_MASK_PENALTY=1.0`, and
`NO_TARGET_NONEMPTY_MASK_PENALTY=0.0`. One epoch ends at `MAX_STEPS=178`.
Logs and FSDP checkpoints are written under
`$REPO_DIR/logs/cyclegrpo70k_opsd_seca_positive_only_8gpu/`.

Before consuming GPUs, run the following preflight. It validates the exact model/data
artifacts, the four expected row counts, the launcher syntax, and the required
training switches. The launcher itself starts Ray and the local Llama judge, so no
separate `ray start` or Llama command is needed.

```bash
for required in \
  "$MODEL_PATH/config.json" \
  "$MODEL_PATH/mask_tokenizer_256x2.pth" \
  "$MODEL_PATH/sam2.1_hiera_large.pt" \
  "$TRAIN_DATA" \
  "$DIRECT_TRAIN_DATA" \
  "$DIRECT_NO_TARGET_TRAIN_DATA" \
  "$CAPTION_QA_TRAIN_DATA" \
  "$CAPTION_QA_JSONL"; do
  test -f "$required" || { echo "missing: $required" >&2; exit 1; }
done
bash -n "$BASE_DIR/run_opsd_70k.sh"
bash -n "$REPO_DIR/projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh"
for required_flag in \
  OPSD_ENABLED=true PIXEL_IOU_ENABLED=true SECA_ENABLED=true \
  ROUTING_ENABLED=true EMA_TEACHER_ENABLED=true TEACHER_ANALYSIS_ENABLED=true \
  TEACHER_CONFIDENCE_ENABLED=true CAPTION_SAFETY_ENABLED=true \
  NO_TARGET_REWARD_MODE=pixel_empty \
  POSITIVE_EMPTY_MASK_PENALTY=1.0 NO_TARGET_NONEMPTY_MASK_PENALTY=0.0 \
  DIRECT_GROUNDING_ENABLED=true DIRECT_MASK_CE_ENABLED=true \
  SUPERVISED_CAPTION_QA_ENABLED=true NUM_GPUS=7 \
  TRAINER_LOGGERS='["file"]'; do
  grep -Fq "$required_flag" "$BASE_DIR/run_opsd_70k.sh" || { echo "missing flag: $required_flag" >&2; exit 1; }
done
$PYTHON_BIN - "$BASE_DIR" <<'PY'
import sys
from pathlib import Path
import pyarrow.parquet as pq
base = Path(sys.argv[1])
expected = {
  "datasets/cyclegrpo_20k_raw_seed20260820/cyclegrpo_20k_40_20_25_10_5_seed20260820.parquet": 20000,
  "datasets/direct_refcoco30k_disjoint_cycle20k/refcoco_train_30k_disjoint_cycle20k_seed20260823.parquet": 30000,
  "datasets/grefcoco_no_target_direct_10k/grefcoco_train_0pos_10000notarget_seed20260821_no_target.parquet": 10000,
  "datasets/dlc_qa/dlc_qa_10000.parquet": 10000,
}
for rel, rows in expected.items():
  got = pq.read_metadata(base / rel).num_rows
  if got != rows:
    raise SystemExit(f"{rel}: expected {rows}, got {got}")
print("preflight passed: model files, 70k parquet row counts, syntax and switches")
PY
```

### 6. Export and evaluate all four benches

After `global_step_178` is present, export the FSDP actor shards and run the
standard RefCOCO, GroundingSuite, GRES and DLC-Bench evaluations. GRES writes
all four required metrics (`T_acc`, `N_acc`, `gIoU`, `cIoU`). DLC-Bench must be
scored with the local Llama-3.1-8B service, not an external GPT API.

```bash
export RUN_ROOT="$REPO_DIR/logs/cyclegrpo70k_opsd_seca_positive_only_8gpu"
export CKPT="$RUN_ROOT/checkpoints/global_step_178"
export EVAL_ROOT="$RUN_ROOT/evaluation/step_178"
export HF_MODEL_PATH="$EVAL_ROOT/hf_global_step_178"
export DLC_ROOT="$BASE_DIR/third_party/DLC-Bench"
export GSE_ROOT="$BASE_DIR/third_party/GroundingSuite"

CHECKPOINT_PATH="$CKPT" HF_MODEL_PATH="$HF_MODEL_PATH" EVAL_ROOT="$EVAL_ROOT" \
TRAIN_MODEL_PATH="$MODEL_PATH" TRAIN_DATA="$TRAIN_DATA" NUM_GPUS=7 \
bash "$REPO_DIR/projects/eval/qwen3vl_4b_volcengine.sh" export

HF_MODEL_PATH="$HF_MODEL_PATH" EVAL_ROOT="$EVAL_ROOT" TRAIN_MODEL_PATH="$MODEL_PATH" \
REFCOCO_ROOT="$REFCOCO_ASSET_DIR" REFCOCO_SPLIT=val NUM_GPUS=7 \
bash "$REPO_DIR/projects/eval/qwen3vl_4b_volcengine.sh" refcoco

HF_MODEL_PATH="$HF_MODEL_PATH" EVAL_ROOT="$EVAL_ROOT" TRAIN_MODEL_PATH="$MODEL_PATH" \
GROUNDINGSUITE_ROOT="$GSE_ROOT" GROUNDINGSUITE_DATASET="$GSE_ROOT/GroundingSuite-Eval.jsonl" \
REFCOCO_ROOT="$REFCOCO_ASSET_DIR" NUM_GPUS=7 \
bash "$REPO_DIR/projects/eval/qwen3vl_4b_volcengine.sh" groundingsuite

HF_MODEL_PATH="$HF_MODEL_PATH" EVAL_ROOT="$EVAL_ROOT" TRAIN_MODEL_PATH="$MODEL_PATH" \
GRES_ROOT="$BASE_DIR/gRefCOCO" GRES_SPLIT=val GRES_IMAGE_ROOT="$REFCOCO_ASSET_DIR/train2014" NUM_GPUS=7 \
bash "$REPO_DIR/projects/eval/qwen3vl_4b_volcengine.sh" gres

HF_MODEL_PATH="$HF_MODEL_PATH" EVAL_ROOT="$EVAL_ROOT" TRAIN_MODEL_PATH="$MODEL_PATH" \
DLC_ROOT="$DLC_ROOT" NUM_GPUS=7 \
bash "$REPO_DIR/projects/eval/qwen3vl_4b_volcengine.sh" dlc

# Start the local language judge on the reserved eighth GPU.
export LLAMA_PORT=18007
export LLAMA_KEY=sk-cyclegrpo-local
CUDA_VISIBLE_DEVICES=7 "$ENV_DIR/bin/python3" -m vllm.entrypoints.openai.api_server \
  --model "$LLAMA_PATH" --served-model-name llama3.1-8b \
  --host 127.0.0.1 --port "$LLAMA_PORT" --api-key "$LLAMA_KEY" \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 --trust-remote-code \
  --dtype bfloat16 --gpu-memory-utilization 0.85 \
  --chat-template "$REPO_DIR/tools/multinode/llama3_chat_template.jinja" \
  >"$RUN_ROOT/llama_dlc_eval.log" 2>&1 &
LLAMA_PID=$!
trap 'kill "$LLAMA_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do
  curl --silent --fail --max-time 3 -H "Authorization: Bearer $LLAMA_KEY" \
    "http://127.0.0.1:$LLAMA_PORT/v1/models" | grep -q llama3.1-8b && break
  sleep 2
done
curl --silent --fail --max-time 3 -H "Authorization: Bearer $LLAMA_KEY" \
  "http://127.0.0.1:$LLAMA_PORT/v1/models" | grep -q llama3.1-8b
"$ENV_DIR/bin/python3" "$REPO_DIR/evaluation/dlc_bench/eval_llama_without_image.py" \
  --pred "$EVAL_ROOT/dlc_bench_predictions.json" \
  --qa "$DLC_ROOT/qa.json" --class-names "$DLC_ROOT/class_names.json" \
  --model llama3.1-8b --base-url "http://127.0.0.1:$LLAMA_PORT/v1" \
  --api-key "$LLAMA_KEY" --quiet \
  > "$EVAL_ROOT/dlc_metrics.log"
kill "$LLAMA_PID" 2>/dev/null || true
trap - EXIT

# Print the scalar summaries; GRES intentionally shows every metric.
"$ENV_DIR/bin/python3" - <<PY
import json
from pathlib import Path
out = Path("$EVAL_ROOT")
print("RefCOCO", json.load(open(out / "refcoco_metrics.json")))
print("GroundingSuite gIoU", json.load(open(out / "groundingsuite_metrics.json"))["overall_giou"])
print("GRES", {k: json.load(open(out / "gres_metrics.json"))[k] for k in ("T_acc", "N_acc", "gIoU", "cIoU")})
PY
```

The eval action uses seven GPUs (`0--6`); only the temporary Llama judge uses
GPU 7. If evaluation is interrupted, rerun the same action: all per-sample
RefCOCO/GroundingSuite/GRES outputs are resumable. Do not compare GRES without
checking that its case count equals the prepared dataset count.

### 7. Upload weights



After `global_step_178` is present, use the `HF_MODEL_PATH` exported by step 6,
copy the SAMTok files, and upload the resulting Hugging Face model. If step 6 was
run in a different shell, re-export the three paths first:

```bash
test -d "$HF_MODEL_PATH"
test -f "$HF_MODEL_PATH/config.json"
cp "$MODEL_PATH/mask_tokenizer_256x2.pth" "$HF_MODEL_PATH/"
cp "$MODEL_PATH/sam2.1_hiera_large.pt" "$HF_MODEL_PATH/"

huggingface-cli login
export WEIGHT_REPO=Untitled111/opsd-70k-seca
huggingface-cli upload "$WEIGHT_REPO" "$HF_MODEL_PATH" . --repo-type model \
  --commit-message "OPSD 70k SECA positive-only pixel-empty model"
```

The uploaded model is available at `https://huggingface.co/$WEIGHT_REPO`.
Keep the launcher copy, data manifest, training log and evaluation outputs
beside the checkpoint for exact reproducibility.

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

### Optional regeneration-only downloads


The authoritative 70k workflow is steps 1--7 above: it downloads RefCOCO from
`Untitled111/refcoco-train2014-assets` and all uploaded generated assets from
`Untitled111/train-opsd`. The commands in this subsection are only for rebuilding
parquet files from raw annotations, not for the supplied 70k run.

Run these commands on a training server to download the public COCO,
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

For the supplied 70k run, gRefCOCO files, COCO-Stuff files, COCO2014/2017
archives, generated DAM metadata and all generated parquet/JSONL files are already
mirrored in `Untitled111/train-opsd`; the RefCOCO image/annotation bundle is in
`Untitled111/refcoco-train2014-assets`. The raw PACO archive is not part of
`train-opsd` and is downloaded only when regenerating PACO data. Raw DAM shards
and the GroundingSuite/DLC-Bench evaluation assets are also downloaded separately
by step 2. Do not repeat these downloads unless you are rebuilding or evaluating
with those raw assets.

| Dataset | Official release | Required local placement |
|---|---|---|
| gRefCOCO | [gRefCOCO HF dataset](https://huggingface.co/datasets/FudanCVL/gRefCOCO) and [official code](https://github.com/henghuiding/gRefCOCO) | `$BASE_DIR/gRefCOCO/grefs(unc).json` and `$BASE_DIR/gRefCOCO/instances.json` |
| Describe Anything (DAM) | [DAM HF dataset](https://huggingface.co/datasets/nvidia/describe-anything-dataset) and [official code](https://github.com/NVlabs/describe-anything) | `$BASE_DIR/datasets/dam_data/COCOStuff` and `$BASE_DIR/datasets/dam_data/PACO` |
| GroundingSuite | [GSEval HF dataset](https://huggingface.co/datasets/hustvl/GSEval) and [official code](https://github.com/hustvl/GroundingSuite) | `$BASE_DIR/third_party/GroundingSuite/GroundingSuite-Eval.jsonl` and its released assets |
| DLC-Bench | [DLC-Bench HF dataset](https://huggingface.co/datasets/nvidia/DLC-Bench) | `$BASE_DIR/third_party/DLC-Bench/annotations.json` and images |

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

For the seven-training-GPU plus one-DLC-judge-GPU setup, the canonical 70k
recipe above uses `THREE_STREAM_2_4_1_ENABLED=false`, `NUM_GPUS=7`, and parent
prompt batches `112/224/56` for main/direct/DLC-QA. The older strict 2:4:1
profile (`28/56/14`) is a separate legacy ablation and is not the current 70k
run. The launcher starts the judge on the reserved eighth GPU; do not include
it in the training process's `CUDA_VISIBLE_DEVICES`.

```bash
RUN_NAME=gs25k_cycle
RUN_ROOT=$REPO_DIR/logs/$RUN_NAME
TRAIN_DATA=$BASE_DIR/datasets/groundingsuite_25k.parquet

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH="$MODEL_PATH" RUN_NAME="$RUN_NAME" RUN_ROOT="$RUN_ROOT" \
TRAIN_DATA="$TRAIN_DATA" VAL_DATA="$TRAIN_DATA" \
TOTAL_EPOCHS=1 RESUME=false SAVE_FREQ=25 SAVE_LIMIT=2 \
OPSD_ENABLED=true PIXEL_IOU_ENABLED=true ROUTING_ENABLED=true \
TRAINER_LOGGERS='["file"]' \
bash "$REPO_DIR/projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh"
```

`SAVE_LIMIT=2` is important on limited storage. A failed checkpoint write
usually indicates a full or quota-limited filesystem; check `df -h` and remove
only checkpoints you no longer need before restarting.

### Isolated direct referring supervision

Direct supervision is disabled by default. The production three-stream setup
keeps its data separate: the main 20k image-mask mix only runs CycleGRPO/OPSD;
The direct stream can pair 20k RefCOCO positive expressions with 20k gRefCOCO
no-target expressions. Both receive direct segmentation GRPO; SFT teacher-forces
GT SAMTok groups for positives and `<answer>No target.</answer>` for negatives.
DLC-QA remains caption GRPO with the language-judge reward. The two auxiliary
loaders each consume one batch per main step and never draw from, or change the
epoch length of, the main loader.

The RefCOCO term is additive to the cycle update:

```text
L_total = 0.5 L_cycle_caption + 0.5 L_cycle_localization
        + lambda_direct(step) L_direct_GRPO + 0.02 L_direct_mask_CE
        + existing regenerate / JSD / KL auxiliary losses
```

- Direct GRPO samples `K=6` responses from stored human referring expressions in
  `DIRECT_TRAIN_DATA` and `DIRECT_NO_TARGET_TRAIN_DATA`, with independent GRPO
  UID groups.
- The default schedule is zero through step 10, linearly grows until step 30,
  then reaches `DIRECT_GROUNDING_LOSS_WEIGHT`.
- Direct SFT teacher-forces one positive mask-token or no-target refusal target
  per direct parent UID. EOS and padding remain outside the CE loss mask.
- The launcher accepts only `refcoco_cycle` positives and enabled
  `gres_no_target` rows, rejecting label/template sources.

#### Export the 20k + 20k direct data

For direct GRPO/SFT, export a 20k RefCOCO positive file and a separate 20k
gRefCOCO no-target file. The RefCOCO converter validates each image and target
mask; pure no-target export only validates the gRefCOCO records and images.

```bash
REF_DIRECT_DIR=$BASE_DIR/datasets/refcoco_direct_20k
REF_DIRECT_PA=$REF_DIRECT_DIR/refcoco_train_20k_seed20260821.parquet
GREF_NO_TARGET_DIR=$BASE_DIR/datasets/grefcoco_no_target_direct_20k
GREF_NO_TARGET_PA=$GREF_NO_TARGET_DIR/grefcoco_train_0pos_20000notarget_seed20260821_no_target.parquet
mkdir -p "$REF_DIRECT_DIR" "$GREF_NO_TARGET_DIR"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO_DIR" "$ENV_DIR/bin/python3" \
  projects/rl/datasets/prepare_refcoco_rl_dataset.py \
  --instances "$BASE_DIR/refcoco-train2014-assets/instances.json" \
  --refs "$BASE_DIR/refcoco-train2014-assets/refs(unc).p" \
  --images-dir "$BASE_DIR/refcoco-train2014-assets/train2014" \
  --output "$REF_DIRECT_PA" \
  --mask-tokenizer-path "$MODEL_PATH/mask_tokenizer_256x2.pth" \
  --sam2-checkpoint "$MODEL_PATH/sam2.1_hiera_large.pt" \
  --sam2-config-dir "$REPO_DIR/projects/transformers/vq_sam2/sam2/sam2_configs" \
  --split train --max-samples 20000 --seed 20260821 --device cuda

PYTHONPATH="$REPO_DIR" "$ENV_DIR/bin/python3" \
  projects/rl/datasets/prepare_grefcoco_cycle_dataset.py \
  --instances "$BASE_DIR/gRefCOCO/instances.json" \
  --grefs "$BASE_DIR/gRefCOCO/grefs(unc).json" \
  --images-dir "$BASE_DIR/refcoco-train2014-assets/train2014" \
  --output-dir "$GREF_NO_TARGET_DIR" \
  --split train --positive-samples 0 --no-target-samples 20000 \
  --seed 20260821
```

The converter writes absolute image paths, so rerun it after moving the COCO
2014 image directory. Keep `TRAIN_DATA` below on the 20k cycle mix; pass the
two direct files only through the direct variables below.

Enable the two direct terms for a controlled experiment:

```bash
RUN_NAME=gs20k_ref20k_notarget_direct_grpo_sft
RUN_ROOT=$REPO_DIR/logs/$RUN_NAME
TRAIN_DATA=$BASE_DIR/datasets/cyclegrpo_20k.parquet

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH="$MODEL_PATH" RUN_NAME="$RUN_NAME" RUN_ROOT="$RUN_ROOT" \
TRAIN_DATA="$TRAIN_DATA" VAL_DATA="$TRAIN_DATA" \
TOTAL_EPOCHS=1 RESUME=false SAVE_FREQ=25 SAVE_LIMIT=2 \
OPSD_ENABLED=true PIXEL_IOU_ENABLED=true ROUTING_ENABLED=true \
TRAINER_LOGGERS='["file"]' \
DIRECT_GROUNDING_ENABLED=true \
DIRECT_TRAIN_DATA="$REF_DIRECT_PA" \
DIRECT_NO_TARGET_TRAIN_DATA="$GREF_NO_TARGET_PA" \
DIRECT_BATCH_SIZE=128 \
DIRECT_GROUNDING_ROLLOUTS=6 \
DIRECT_GROUNDING_LOSS_WEIGHT=0.15 \
DIRECT_GROUNDING_WARMUP_START_STEP=10 \
DIRECT_GROUNDING_WARMUP_END_STEP=30 \
DIRECT_GROUNDING_INCLUDE_POSITIVE_SOURCES=true \
DIRECT_GROUNDING_INCLUDE_NO_TARGET=true \
DIRECT_GROUNDING_INCLUDE_LABEL_SOURCES=false \
DIRECT_MASK_CE_ENABLED=true \
DIRECT_MASK_CE_LOSS_WEIGHT=0.02 \
DIRECT_MASK_CE_INCLUDE_NO_TARGET=true \
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

## Isolated DLC-QA description supervision

DLC-QA is a third independent training stream. Its 10k Parquet is used only to
form caption prompts, and every row must expose a `dam_source_id` with a
matching accepted entry in the QA JSONL. The reward is only the Llama judge's
mean QA score: it has no cycle IoU, mask format, or caption-safety,
teacher regenerate, or JSD term. `CAPTION_QA_LOSS_WEIGHT` controls the actual
actor-gradient contribution after GRPO normalization.

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

3. Prepare a dedicated QA Parquet containing the selected 10k rows. Do not mix
it into the main 20k CycleGRPO Parquet. Every `dam_source_id` in this file must
occur exactly once in the accepted QA sidecar.

```bash
QA_TRAIN_DATA=$BASE_DIR/datasets/dlc_qa/dlc_qa_10000.parquet
CAPTION_QA_JSONL=$BASE_DIR/datasets/dlc_qa/dlc_qa_10000.jsonl
```

4. Add these variables to the full three-stream training invocation:

```bash
SUPERVISED_CAPTION_QA_ENABLED=true \
CAPTION_QA_TRAIN_DATA="$QA_TRAIN_DATA" CAPTION_QA_BATCH_SIZE=128 \
CAPTION_QA_JSONL="$CAPTION_QA_JSONL" \
CAPTION_QA_JUDGE_BASE_URL=http://127.0.0.1:8007/v1 \
CAPTION_QA_JUDGE_MODEL=llama3.1-8b \
CAPTION_QA_JUDGE_API_KEY=sk-abc123 \
CAPTION_QA_REWARD_WEIGHT=1.0 \
CAPTION_QA_LOSS_WEIGHT=1.0 \
bash "$REPO_DIR/projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh"
```

The runner fails before reward scoring if an auxiliary row lacks a matching QA
entry. Inspect rejected QA JSONL and resume incomplete generation with
`--resume`; do not add rejected records to the dedicated QA data.

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
GROUNDINGSUITE_ROOT="$BASE_DIR/third_party/GroundingSuite" \
GROUNDINGSUITE_DATASET="$BASE_DIR/third_party/GroundingSuite/GroundingSuite-Eval.jsonl" \
REFCOCO_ROOT="$BASE_DIR/refcoco-train2014-assets" NUM_GPUS=8 \
bash "$EVAL_SCRIPT" groundingsuite

# gRefCOCO/GRES target and no-target metrics.
HF_MODEL_PATH="$HF_MODEL" EVAL_ROOT="$OUT" TRAIN_MODEL_PATH="$MODEL_PATH" MASK_PROTOCOL=legacy_union \
GRES_ROOT="$BASE_DIR/gRefCOCO" GRES_SPLIT=val NUM_GPUS=8 \
bash "$EVAL_SCRIPT" gres

# DLC-Bench prediction JSON only; score it with a judge separately.
HF_MODEL_PATH="$HF_MODEL" EVAL_ROOT="$OUT" TRAIN_MODEL_PATH="$MODEL_PATH" \
DLC_ROOT="$BASE_DIR/third_party/DLC-Bench" \
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
