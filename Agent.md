# CycleGRPO Agent 维护规则

本仓库的代码知识库是根目录 `code.md`。任何 Agent 或开发者修改代码时，以下流程是强制要求：

1. 修改任何 `.py`、`.sh`、`.yaml`、`.jinja`、模型配置、数据转换或评测逻辑前，先完整阅读 `code.md`，并确认将要修改的代码属于当前有效路径还是历史/备用路径。
2. 实现修改时保持 `code.md` 中记录的数据契约、训练调用链和论文/代码边界一致；如果实现发生变化，必须同步修正文档正文。
3. 每次代码修改完成后，必须在 `code.md` 的“变更日志”追加记录，至少写明日期、修改文件、行为变化和实际验证方式。
4. 新增、移动、重命名或删除代码文件时，必须同步更新 `code.md` 的“目录与代码职责”。
5. 若修改导致实现偏离论文公式、默认实验参数或主训练入口，必须在 `code.md` 的“论文结论与实现边界”或“关键注意事项”中明确说明。
6. 未同步更新 `code.md` 的代码修改不视为完成。

文档本身的纯排版修订可以只写变更日志；涉及事实、调用关系或维护规则的文档修改仍需核对相应代码。

## 终端命令交付

给用户提供手动执行的服务器训练、数据导出、评测和诊断命令时，默认不得使用 `set -e`、`set -u` 或 `set -o pipefail`（包括 `set -euo pipefail`）。命令失败时必须保留 Python/bash 的原始错误输出和 traceback，便于用户在同一终端直接定位并继续执行后续诊断。只有用户明确要求 fail-fast 行为时，才可以加入这些 shell 选项。

## 当前服务器训练数据路径

以下路径来自实际执行的 disjoint 诊断训练命令，后续命令默认以此为基准：

```text
BASE_DIR=/volume/ybo/xyc
REPO_DIR=/volume/ybo/xyc/CycleGRPO-OPSD
ENV_DIR=/volume/ybo/xyc/envs/cyclegrpo
MODEL_PATH=/volume/ybo/xyc/Qwen3-VL-4B-SAMTok

TRAIN_DATA=/volume/ybo/xyc/datasets/cyclegrpo_20k_raw_seed20260820/cyclegrpo_20k_40_20_25_10_5_seed20260820.parquet
NO_TARGET_FREE_TRAIN_DATA=/volume/ybo/xyc/datasets/cyclegrpo_20k_refcoco_replacement_seed20260901/cyclegrpo_20k_45_20_25_10_refcoco_replacement_seed20260901.parquet
NO_TARGET_FREE_TRAIN_MANIFEST=/volume/ybo/xyc/datasets/cyclegrpo_20k_refcoco_replacement_seed20260901/cyclegrpo_20k_45_20_25_10_refcoco_replacement_seed20260901.manifest.json
OFFICIAL_HTG_TRAIN_DATA=/volume/ybo/xyc/datasets/cyclegrpo_20k_official_htg_seed20260825/cyclegrpo_20k_official_htg_15k_single_4k_multi_1k_gres_seed20260825.parquet
OFFICIAL_HTG_TRAIN_MANIFEST=/volume/ybo/xyc/datasets/cyclegrpo_20k_official_htg_seed20260825/cyclegrpo_20k_official_htg_15k_single_4k_multi_1k_gres_seed20260825.manifest.json
DIRECT_DATA=/volume/ybo/xyc/datasets/direct_refcoco30k_disjoint_cycle20k/refcoco_train_30k_disjoint_cycle20k_seed20260823.parquet
NO_TARGET_DATA=/volume/ybo/xyc/datasets/direct_refcoco30k_notarget10k_disjoint_cycle20k/grefcoco_train_0pos_10000notarget_disjoint_cycle20k_and_refcoco_seed20260823.parquet
DLC_DATA=/volume/ybo/xyc/datasets/dlc_qa/dlc_qa_10000.parquet
DLC_QA=/volume/ybo/xyc/datasets/dlc_qa/dam_caption_qa_10000.jsonl

RUN_NAME=cyclegrpo20k_direct30k_notarget10k_dlcqa10k_ce002_disjoint
RUN_ROOT=/volume/ybo/xyc/CycleGRPO-OPSD/logs/cyclegrpo20k_direct30k_notarget10k_dlcqa10k_ce002_disjoint
```

数据流对应关系：`TRAIN_DATA` 是 20k CycleGRPO 主数据，`DIRECT_DATA` 是 disjoint RefCOCO
正例，`NO_TARGET_DATA` 是 disjoint gRefCOCO no-target，`DLC_DATA` 是 DLC-QA parquet，
`DLC_QA` 是对应的 QA JSONL。启用 no-target direct GRPO/SFT 前仍需在服务器执行
`test -f "$NO_TARGET_DATA"`，确认该实际文件存在。

`NO_TARGET_FREE_TRAIN_DATA` 是当前用于不含主数据 no-target caption GRPO 的纯正样本 CycleGRPO 自监督的 20k 派生数据：从
`TRAIN_DATA` 删除全部 1,000 条 `source=gres_no_target`，补入 1,000 条 `source=refcoco_cycle`
正例。因此其 source 组成固定为 RefCOCO 9,000、gRefCOCO positive 4,000、COCO-Stuff 5,000、
PACO part 2,000，且不含任何 no-target 行。补入的 RefCOCO 行按 `COCO image filename + 原始
mask RLE` 排除了保留的主 19k 与 `DIRECT_DATA` 的 30k RefCOCO 正例；实际导出 seed 为
`20260901`，候选池经排除后为 1,685 条，最终选择前 1,000 条。后续需要禁用主 CycleGRPO
no-target caption GRPO 的实验应显式将 `TRAIN_DATA` 和 `VAL_DATA` 同时设为该路径；
独立 direct no-target 监督仍只使用 `NO_TARGET_DATA`，不要用此主数据替代。

`OFFICIAL_HTG_TRAIN_DATA` 是由 `TRAIN_DATA` 导出的官方 CycleGRPO 兼容副本，仅把
`source` 映射为官方代码已有标签：RefCOCO/COCO-Stuff/PACO 的 15k 行为
`denseworld_single`，gRefCOCO positive 的 4k 行为 `denseworld_multiple`，1k
`gres_no_target` 保持不变；所有其他 parquet 列保持原样。它只能配合
`/volume/ybo/xyc/CycleGRPO` 的未修改官方训练代码做 HTG 对照，不应用于 OPSD 主训练，
也不能表述为使用论文原始 DenseWorld 数据的复现。
