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
- [DLC-QA data contract](#dlc-qa-data-contract)
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
# Use the disjoint 10k no-target file paired with the 30k RefCOCO direct file.
export DIRECT_NO_TARGET_TRAIN_DATA="$BASE_DIR/datasets/direct_refcoco30k_notarget10k_disjoint_cycle20k/grefcoco_train_0pos_10000notarget_disjoint_cycle20k_and_refcoco_seed20260823.parquet"
export CAPTION_QA_TRAIN_DATA="$BASE_DIR/datasets/dlc_qa/dlc_qa_10000.parquet"
export CAPTION_QA_JSONL="$BASE_DIR/datasets/dlc_qa/dam_caption_qa_10000.jsonl"
```

The supervised segmentation stream is **40,000 samples total**, composed as follows:

| Role | Canonical file | Rows | `source` |
|---|---|---:|---|
| RefCOCO positive direct supervision | `datasets/direct_refcoco30k_disjoint_cycle20k/refcoco_train_30k_disjoint_cycle20k_seed20260823.parquet` | 30,000 | `refcoco_cycle` |
| gRefCOCO no-target direct supervision | `datasets/direct_refcoco30k_notarget10k_disjoint_cycle20k/grefcoco_train_0pos_10000notarget_disjoint_cycle20k_and_refcoco_seed20260823.parquet` | 10,000 | `gres_no_target` |
| **Total** | `DIRECT_TRAIN_DATA` + `DIRECT_NO_TARGET_TRAIN_DATA` | **40,000** | — |

`train-opsd` also contains the convenience file
`datasets/direct_refcoco30k_notarget10k_disjoint_cycle20k/direct_train_40k_refcoco30k_notarget10k_disjoint_cycle20k_seed20260823.parquet`, whose verified composition is 30,000 `refcoco_cycle` + 10,000 `gres_no_target`. The 70k launcher intentionally uses the split pair above because it appends `DIRECT_NO_TARGET_TRAIN_DATA` when no-target GRPO/CE is enabled; do not pass the merged 40k file together with the separate 10k file.

The same files are also visible at the dataset root as convenience copies, but
the paths above are the canonical layout consumed by the training launcher.
The uploaded `datasets/dam_data/` contains DAM annotation metadata; use the
`dam_raw` download in step 2 when the raw DAM image shards are needed.


### 4. Rewrite image paths (mandatory)

Parquet records contain absolute paths from the source server. Rewrite and
validate every image before training; do not rely on row counts alone. The
snippet assumes the uploaded Parquet still contains the source-server prefix
`/volume/ybo/xyc`; if the files were rebuilt elsewhere, change `old_ref` and
`old_coco` to the prefixes actually present in that Parquet.

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
    base / "datasets/direct_refcoco30k_notarget10k_disjoint_cycle20k/grefcoco_train_0pos_10000notarget_disjoint_cycle20k_and_refcoco_seed20260823.parquet": 10000,
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

### 5. Run the four routing-threshold ablations

The environment and data preparation above are shared by all four experiments. The
only maintained training commands in this README are the four routing wrappers
listed in [Training](#training). They all run the same full 70k SECA recipe:
20k CycleGRPO + 30k RefCOCO direct positives + 10k gRefCOCO no-target direct
rows + 10k DLC-QA rows, with seven Ray/FSDP training GPUs and the eighth GPU
reserved for the local Llama judge. Do not start Ray or the judge manually; each
wrapper starts and cleans up its own local services.

Complete the path-rewrite step first, then follow [Training](#training) to
preflight and run the four wrappers. Each wrapper has its own run name, Ray
ports, judge port and short Ray directory, so the four runs can be kept as
independent directories. Run them sequentially on one eight-GPU node unless
the server has four completely isolated eight-GPU allocations.

After each completed run, record the wrapper name, routing pair, git revision,
data manifest, `run.log`, `training.log` and checkpoint directory together.
The evaluation section below is generic; substitute the corresponding
experiment's `RUN_ROOT` rather than reusing one shared output directory.

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
<RUN_ROOT>/checkpoints/global_step_<N>/actor/model_world_size_7_rank_*.pt
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


The authoritative 70k workflow is steps 1--5 above: it downloads RefCOCO from
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

This README intentionally exposes only the four formal routing-threshold ablations.
Do not use the generic `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` command
directly for these experiments: the wrapper is responsible for the 70k data streams,
seven-GPU Ray topology, local judge, fixed SECA configuration and run isolation.

### Fixed experiment matrix

All four wrappers use exactly the same model, data, optimizer, rollout, auxiliary
loss, checkpoint and 7+1 GPU settings. Only the routing thresholds differ; the
run names, ports and Ray temporary directories are also different so logs cannot
collide.

| Wrapper | `low_threshold` | `high_threshold` | Ray ports | Judge port | Default run root |
|---|---:|---:|---|---:|---|
| `train_supervised_70k_seca_routing_l030_h085_8gpu.sh` | 0.30 | 0.85 | 29701 / 29702 | 18013 | `logs/cyclegrpo70k_historical_seca_routing_l030_h085_8gpu` |
| `train_supervised_70k_seca_routing_l070_h085_8gpu.sh` | 0.70 | 0.85 | 29703 / 29704 | 18014 | `logs/cyclegrpo70k_historical_seca_routing_l070_h085_8gpu` |
| `train_supervised_70k_seca_routing_l050_h065_8gpu.sh` | 0.50 | 0.65 | 29705 / 29706 | 18015 | `logs/cyclegrpo70k_historical_seca_routing_l050_h065_8gpu` |
| `train_supervised_70k_seca_routing_l050_h100_8gpu.sh` | 0.50 | 1.00 | 29707 / 29708 | 18016 | `logs/cyclegrpo70k_historical_seca_routing_l050_h100_8gpu` |

The last row is the requested `high=0.85+0.20` direction clipped to `1.00`,
because the launcher rejects thresholds outside `[0,1]`. The four values are
fixed in the wrappers and are not inherited from the shell environment.

### Required server layout

Use one node with eight visible GPUs. The wrappers reserve physical GPUs 0--6
for Ray/FSDP training and GPU 7 for the local Llama-3.1-8B judge. Stop any
old Ray head, judge service or `gpu_power_hold.sh` worker before starting a
run. Do not manually start Ray or vLLM: the shared launcher starts an isolated
Ray head and a Ray-managed local judge, then removes only the processes it owns.

Set these variables after completing the environment installation and data
path-rewrite steps above. The paths are examples; replace `BASE_DIR` and keep
the five data variables pointed at the files below.

```bash
export BASE_DIR=/mnt/opsd
export REPO_DIR="$BASE_DIR/CycleGRPO-OPSD"
export ENV_DIR="$BASE_DIR/envs/cyclegrpo"
export MODEL_PATH="$BASE_DIR/Qwen3-VL-4B-SAMTok"
export JUDGE_MODEL_PATH="$BASE_DIR/Meta-Llama-3.1-8B-Instruct"

export TRAIN_DATA="$BASE_DIR/datasets/cyclegrpo_20k_raw_seed20260820/cyclegrpo_20k_40_20_25_10_5_seed20260820.parquet"
export VAL_DATA="$TRAIN_DATA"
export DIRECT_TRAIN_DATA="$BASE_DIR/datasets/direct_refcoco30k_disjoint_cycle20k/refcoco_train_30k_disjoint_cycle20k_seed20260823.parquet"
export DIRECT_NO_TARGET_TRAIN_DATA="$BASE_DIR/datasets/direct_refcoco30k_notarget10k_disjoint_cycle20k/grefcoco_train_0pos_10000notarget_disjoint_cycle20k_and_refcoco_seed20260823.parquet"
export CAPTION_QA_TRAIN_DATA="$BASE_DIR/datasets/dlc_qa/dlc_qa_10000.parquet"
export CAPTION_QA_JSONL="$BASE_DIR/datasets/dlc_qa/dam_caption_qa_10000.jsonl"
export PYTHON_BIN="$ENV_DIR/bin/python3"
# Do not let stale values from an earlier run override each wrapper's fixed
# run name, ports or Ray temporary directory.
unset RUN_NAME RUN_ROOT RAY_PORT RAY_DASHBOARD_PORT JUDGE_PORT RAY_SHORT_ROOT
cd "$REPO_DIR"
```

The exact training inputs must contain 20,000 main-cycle rows, 30,000 direct
RefCOCO-positive rows, 10,000 direct gRefCOCO no-target rows, 10,000 DLC-QA
Parquet rows and 10,000 non-empty DLC-QA JSONL records. The direct files are
intentionally split: do not replace them with the convenience merged 40k file.
All Parquet `images` paths must already point to this server's local image
roots; the launcher validates existence before it starts Ray.

The model directory must contain at least `config.json`,
`model.safetensors.index.json`, `mask_tokenizer_256x2.pth` and
`sam2.1_hiera_large.pt`. The judge directory must contain `config.json` and
the Llama tokenizer/model shards. If the model or data lives elsewhere, change
only the exported path variables; do not edit the four wrapper files.

### Preflight all four experiments

Run this before consuming GPUs. `DRY_RUN=true` performs the same path, row-count,
JSONL-count and shell configuration checks as a real run, but starts no Ray, judge
or trainer.

```bash
set -euo pipefail
test -x "$PYTHON_BIN"
test -f "$MODEL_PATH/config.json"
test -f "$MODEL_PATH/model.safetensors.index.json"
test -f "$MODEL_PATH/mask_tokenizer_256x2.pth"
test -f "$MODEL_PATH/sam2.1_hiera_large.pt"
test -f "$JUDGE_MODEL_PATH/config.json"
for path in "$TRAIN_DATA" "$DIRECT_TRAIN_DATA" "$DIRECT_NO_TARGET_TRAIN_DATA" "$CAPTION_QA_TRAIN_DATA" "$CAPTION_QA_JSONL"; do
  test -f "$path" || { echo "missing: $path" >&2; exit 1; }
done

scripts=(
  "$REPO_DIR/tools/train_supervised_70k_seca_routing_l030_h085_8gpu.sh"
  "$REPO_DIR/tools/train_supervised_70k_seca_routing_l070_h085_8gpu.sh"
  "$REPO_DIR/tools/train_supervised_70k_seca_routing_l050_h065_8gpu.sh"
  "$REPO_DIR/tools/train_supervised_70k_seca_routing_l050_h100_8gpu.sh"
)
for script in "${scripts[@]}"; do
  bash -n "$script"
  DRY_RUN=true HOLD_AFTER_EXIT=false bash "$script"
done
echo "routing-ablation preflight passed"
```

The preflight must print the four row counts (`20000/30000/10000/10000`), the
matching 10,000 JSONL lines, and a successful dry-run line for every wrapper.
The dry-run does not invoke the main trainer's full image scan; that scan runs
after Ray/judge startup, so the mandatory path-rewrite step above must still
finish successfully.
Any missing image, model file, judge file, row-count mismatch or dry-run error
must be fixed before starting training. Before a real run, also verify that the
wrapper's listed Ray and judge ports are free.

### Run the experiments

Run the four experiments sequentially. The formal wrappers default to
`MAX_STEPS=179`, parent batches `112/224/56` (main/direct/DLC-QA), six caption
and six localization rollouts, `SAVE_FREQ=5`, `SAVE_LIMIT=2`, file-only logging,
and `RESUME=false`. They enable OPSD, pixel-IoU, routing, EMA teacher, teacher
analysis/confidence, caption safety, SECA, direct GRPO, direct mask CE and
DLC-QA; no other training flags should be changed for this ablation.

For one experiment, stop any hold workers and launch its wrapper:

```bash
GPU_LIST=0,1,2,3,4,5,6,7 PYTHON_BIN="$PYTHON_BIN" \
  bash "$REPO_DIR/tools/gpu_power_hold.sh" stop || true
bash "$REPO_DIR/tools/train_supervised_70k_seca_routing_l030_h085_8gpu.sh"
```

To run all four in sequence without occupying the GPUs between runs:

```bash
set -euo pipefail
GPU_LIST=0,1,2,3,4,5,6,7 PYTHON_BIN="$PYTHON_BIN" \
  bash "$REPO_DIR/tools/gpu_power_hold.sh" stop || true
scripts=(
  "$REPO_DIR/tools/train_supervised_70k_seca_routing_l030_h085_8gpu.sh"
  "$REPO_DIR/tools/train_supervised_70k_seca_routing_l070_h085_8gpu.sh"
  "$REPO_DIR/tools/train_supervised_70k_seca_routing_l050_h065_8gpu.sh"
  "$REPO_DIR/tools/train_supervised_70k_seca_routing_l050_h100_8gpu.sh"
)
for script in "${scripts[@]}"; do
  echo "===== starting $script"
  HOLD_AFTER_EXIT=false bash "$script"
done
GPU_LIST=0,1,2,3,4,5,6,7 PYTHON_BIN="$PYTHON_BIN" MEMORY_MIB=20000 MATMUL_DIM=4096 \
  bash "$REPO_DIR/tools/gpu_power_hold.sh" start
```

If a run is interrupted or fails, inspect its `run.log` and `training.log`,
clean up only its own Ray/judge processes, fix the reported issue, and rerun
that wrapper with a fresh `RUN_NAME`/`RUN_ROOT` if the checkpoint is incomplete.
Do not launch a second wrapper while the previous run's Ray/judge or GPU-hold
processes are still alive.

Each run writes `run.log`, `training.log`, `ray_start.log`, judge logs and
`checkpoints/` below its own run root. A successful run should contain the
final `global_step_*` FSDP checkpoint plus `experiment_config.json` and
trainer logs. The launcher prints the active routing pair; verify that it
matches the matrix before comparing results.

### What is and is not changed

The four scripts are a controlled routing ablation, not four different data
or model recipes. The data files, image-path rewrite, model checkpoint,
rollout count, parent-batch ratio, optimizer, SECA weights, direct/DLC-QA
anchors, checkpoint cadence and GPU topology are shared. Only the routing
`low_threshold`/`high_threshold` pair changes, with run identity and service
ports changed solely to isolate outputs.

## DLC-QA data contract

The four routing wrappers consume the already prepared 10k DLC-QA Parquet and
its accepted 10k-record JSONL sidecar. This section is intentionally a data
contract, not a fifth training recipe: do not add another launcher or override
the wrapper's fixed `SUPERVISED_CAPTION_QA_ENABLED`, batch, judge or loss
settings for the routing comparison.

Every Parquet row must expose a unique `dam_source_id`; the JSONL sidecar must
contain exactly one accepted QA record for every ID. The shared launcher checks
the Parquet row count and JSONL line count before starting Ray, and the trainer
checks the join before scoring. If the sidecar is regenerated on another
server, keep the same 10k-row selection and rewrite only local image paths in
the Parquet; do not mix rejected QA records into the accepted JSONL.

The DAM converters and QA-generation commands remain in the data-preparation
sections above. After this contract passes, use only the four wrappers in
[Training](#training).

## Exporting and evaluating checkpoints

The unified evaluation entry supports `export`, `refcoco`, `groundingsuite`,
`gres`, `dlc`, and `all`. All supported segmentation evaluators decode the
first complete valid SAMTok group, so regenerate old checkpoints under the same
protocol before comparing values.

```bash
# Select one completed routing run; repeat the block with another RUN_ROOT for
# each of the four ablations.
export RUN_ROOT="$REPO_DIR/logs/cyclegrpo70k_historical_seca_routing_l030_h085_8gpu"
export CKPT="$RUN_ROOT/checkpoints/global_step_179"
export OUT="$RUN_ROOT/evaluation/step_179"
export HF_MODEL="$OUT/hf_global_step_179"
EVAL_SCRIPT=$REPO_DIR/projects/eval/qwen3vl_4b_volcengine.sh

# Convert world-size-7 FSDP actor shards to standard HF safetensors.
CHECKPOINT_PATH="$CKPT" HF_MODEL_PATH="$HF_MODEL" EVAL_ROOT="$OUT" \
TRAIN_MODEL_PATH="$MODEL_PATH" TRAIN_DATA="$TRAIN_DATA" NUM_GPUS=7 \
bash "$EVAL_SCRIPT" export

# RefCOCO val cIoU/mIoU. The H20 recipe starts at 16 images per GPU; raise to
# 24 or 32 only after one successful run confirms the available headroom.
HF_MODEL_PATH="$HF_MODEL" EVAL_ROOT="$OUT" TRAIN_MODEL_PATH="$MODEL_PATH" \
REFCOCO_ROOT="$BASE_DIR/refcoco-train2014-assets" REFCOCO_SPLIT=val NUM_GPUS=7 EVAL_BATCH_SIZE=16 \
bash "$EVAL_SCRIPT" refcoco

# GroundingSuite mask GIoU. legacy_union is the historical SAMTok-compatible protocol.
HF_MODEL_PATH="$HF_MODEL" EVAL_ROOT="$OUT" TRAIN_MODEL_PATH="$MODEL_PATH" MASK_PROTOCOL=legacy_union \
GROUNDINGSUITE_ROOT="$BASE_DIR/third_party/GroundingSuite" \
GROUNDINGSUITE_DATASET="$BASE_DIR/third_party/GroundingSuite/GroundingSuite-Eval.jsonl" \
REFCOCO_ROOT="$BASE_DIR/refcoco-train2014-assets" NUM_GPUS=7 \
bash "$EVAL_SCRIPT" groundingsuite

# gRefCOCO/GRES target and no-target metrics.
HF_MODEL_PATH="$HF_MODEL" EVAL_ROOT="$OUT" TRAIN_MODEL_PATH="$MODEL_PATH" MASK_PROTOCOL=legacy_union \
GRES_ROOT="$BASE_DIR/gRefCOCO" GRES_SPLIT=val NUM_GPUS=7 \
bash "$EVAL_SCRIPT" gres

# DLC-Bench prediction JSON only; score it with a judge separately.
HF_MODEL_PATH="$HF_MODEL" EVAL_ROOT="$OUT" TRAIN_MODEL_PATH="$MODEL_PATH" \
DLC_ROOT="$BASE_DIR/third_party/DLC-Bench" \
bash "$EVAL_SCRIPT" dlc
```

Expected outputs include:

```text
<OUT>/hf_global_step_179/             # HF safetensors export
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
