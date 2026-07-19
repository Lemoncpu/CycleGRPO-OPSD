# CycleGRPO 代码文档

> 文档基线：论文 `2607.11581v1`（29 页，2026-07-17）与仓库提交 `586e970`。
> 本文是仓库代码知识库，也是强制维护的变更日志。修改任何 `.py`、`.sh`、`.yaml`、`.jinja`、模型配置或评测逻辑前，必须先读本文；修改完成后，必须同步更新相关章节和末尾的“变更日志”。

## 1. 项目定位

论文标题是 **Actor as Its Own Critic: Unifying Region Understanding and Localization via CycleGRPO**。核心目标不是单独优化“区域描述”或“文本定位”，而是把二者视为互逆映射：

```text
图像 + 目标区域 M
        |
        | Phase 1: captioning rollout，采样 G 个候选描述 C_i
        v
候选描述 C_1 ... C_G
        |
        | Phase 2: localization rollout，每个 C_i 采样 K 次重建 M_hat_i,k
        v
SAMTok 完整解码后的像素 IoU / 空间一致性分数 s_i,k
        |
        +--> caption reward:  R_cap_i = mean_k(s_i,k)
        +--> location reward: R_loc_i,k = R_cap_i * s_i,k
```

同一个多模态大模型先作为 Actor 生成描述，再作为自己的 Critic 根据该描述重建区域。高质量描述必须包含足够独特、可验证的细节，才能让模型找回原区域。

论文正文用 IoU 解释空间一致性；原始公开代码为降低高分辨率 mask 解码开销，采用 **Hierarchical Token Grading**。当前 OPSD 扩展已把图像 cycle source 改为训练时完整解码 SAMTok token 并计算真实像素 IoU；`worker.opsd.enabled=false` 时仍可回到原始 token-domain CycleGRPO。

当前扩展在每条 caption 的 `K` 次真实 IoU 均值 `R_Ci` 上执行候选级三路由：`R_Ci<0.5` 进入 EMA teacher regenerate，`0.5<=R_Ci<=0.85` 进入 privileged on-policy distillation，`R_Ci>0.85` 保留 CycleGRPO caption GRPO。三路由只替换 caption 侧更新，所有 localization rollout 始终参与 CycleGRPO 更新。

## 2. 论文结论与实现边界

### 2.1 论文方法

- 基座：Qwen3-VL-4B 上的 SAMTok，mask 被离散为 `<|mt_start|><|mt_xxxx|><|mt_xxxx|><|mt_end|>`。
- 训练数据：论文报告约 20k DenseWorld 图像/区域，加约 1k GRES no-target 表达。
- 论文默认：caption group size `G=6`，每个描述的 localization rollout `K=6`，总 batch size 128，AdamW，学习率 `1e-6`，weight decay `1e-2`，1 epoch。
- 训练时冻结 vision encoder，优化 projection/LLM 参数。
- 主要评测：DLC-Bench、GAR-Bench-VQA、GCG、GRES、GroundingSuite；不在这些下游集上继续微调。

### 2.2 当前公开代码的有效配置

主入口是 `projects/rl/qwen3vl_4b_mt.sh`，它覆盖 `projects/rl/config.yaml` 的若干默认值：

| 项 | 当前主入口 | 说明 |
|---|---:|---|
| 模型 | `<PATH_TO_COLD_START_CKPT>` | 必须替换为 co-SFT/SAMTok checkpoint |
| 外层 rollout `G` | `worker.rollout.n=6` | 与论文及 OPSD 默认一致 |
| 内层 rollout `K` | `worker.opsd.localization_rollouts=6` | 已从 trainer 硬编码迁入配置 |
| 路由阈值 | `0.5 / 0.85` | 边界分别为 low: `<0.5`、mid: `[0.5,0.85]`、high: `>0.85` |
| EMA teacher | `decay=0.999`、CPU offload | 与 frozen reference policy 完全独立 |
| regenerate | `T=6`、`temperature=0.8`、`top_p=0.95` | 每候选一次 greedy localization 验证，提升至少 `0.05` 才接收 |
| rollout/global batch | `128` | 与论文一致 |
| epoch | `1` | 与论文一致 |
| GPU | 1 node x 8 GPU | Ray + FSDP + vLLM SPMD |
| vision tower | frozen | shell 覆盖为 `true` |
| caption/segmenter | 都优化 | 最终按 `0.5/0.5` 梯度权重累积 |
| 验证 | 关闭 | `val_freq=-1`、`val_before_train=false` |
| 日志 | file + wandb | shell 强制 `WANDB_MODE=offline` |

仓库 README 明确标记为 WIP，不应假设它是论文所有实验的逐字复现版本。

## 3. 主训练调用链

### 3.1 启动与配置合并

1. `projects/rl/qwen3vl_4b_mt.sh` 调用 `python3 -m verl.trainer.main`。
2. `verl/trainer/main.py::main` 按“dataclass 默认值 -> YAML -> CLI 覆盖”合并配置，并初始化 Ray。
3. `Runner.run` 加载 tokenizer/processor，创建共享 GPU resource pool、`FSDPWorker`、batch reward manager 和 dataloader。
4. `RayPPOTrainer.init_workers` 建立 actor、reference policy、可选 critic、vLLM rollout engine、FSDP/vLLM 权重同步器。
5. `RayPPOTrainer.fit` 反复生成经验、算奖励/优势、更新策略、记录日志和保存 checkpoint。

配置分层：

- `verl/trainer/config.py`：data、algorithm、trainer 总配置。
- `verl/workers/actor/config.py`：模型、优化器、FSDP、offload、PPO loss。
- `verl/workers/rollout/config.py`：vLLM 采样参数。
- `verl/workers/reward/config.py`：自定义奖励入口及历史 VQ-SAM2 参数。
- `projects/rl/config.yaml`：本项目运行值。
- shell 中的 `key=value`：优先级最高。

### 3.2 数据装载

`verl/trainer/data_loader.py` 创建 `RLHFDataset`。主训练 parquet 至少需要以下字段：

| 字段 | 含义 |
|---|---|
| `cap_problem` | 区域描述 prompt，通常含图像和目标 mask token |
| `cap_answer` | 可选 caption ground truth；CycleGRPO 主分支不依赖它 |
| `seg_problem` | 预置定位 prompt/描述字段；循环训练时会被 actor 新生成的 caption 替换 |
| `seg_answer` | 原目标 mask token，作为闭环重建目标 |
| `images` / `videos` | 多模态输入路径列表 |
| `source` | 决定 cycle/non-cycle 分流和奖励分支 |
| `masks`、`extra_info` | 部分数据/评测分支的附加信息 |

`verl/utils/dataset.py` 的关键行为：

- 载入一个或多个本地/Hugging Face 数据集并拼接。
- `_filter_overlong_prompts` 在完整展开视觉 token 后过滤过长样本，避免图像特征数与 image token 数不一致。
- `sample_single_target_from_multi_target(..., max_targets=1)` 从多目标样本中随机选一个训练目标。
- `_build_messages` 同时构造 caption 和原始 segmentation prompt。
- `_build_gen_seg_messages` 把 actor caption 放进论文补充材料给出的定位模板；支持 mask token、bbox 和视频时间区间三种格式。
- 图像 caption 可使用多图，segmentation 只保留第一张图；视频保留帧率和帧数元数据。
- 返回 `cap_*` 和 `seg_*` 两套 input ids、attention mask、position ids、raw prompt ids 与多模态数据。

### 3.3 Phase 1：caption rollout

`RayPPOTrainer._make_batch_data`：

1. 从 dataloader 取 batch，为原始 prompt 分配 `uid`，用 `cap_*` 字段构造 `task=caption` 的 `DataProto`。
2. `FSDPWorker.generate_sequences` 通过 `FSDPVLLMShardingManager` 把当前 actor 权重同步到 vLLM，再采样配置的 `G=6` 个回答。
3. 原样本按 `n` 重复并与 rollout 输出合并。
4. 按 `source` 分流：`denseworld_single`、`denseworld_multiple`、`tg_multi_merged`、`dam_cyclegrpo` 和 `None` 进入 cycle batch；其他 source 进入 non-cycle batch。
5. cycle/non-cycle 分别裁成能被 world size 整除的完整 GRPO groups，并按 token 数重排，降低各 rank 负载不均。

`vllm_rollout_spmd.py` 负责：

- 把 raw prompt 和图像/视频整理成 vLLM 输入。
- 采样并 pad response，构造完整 `input_ids`、`attention_mask`、`response_mask` 和扩展后的 position ids。
- 删除模型误生成的 vision 特殊 token，防止后续前向出现视觉 token/feature 数量不匹配。

### 3.4 Phase 2：localization rollout

`RayPPOTrainer._make_seg_batch_data_for_caption` 是 CycleGRPO 的核心桥梁：

1. 解码每个 caption response，删除空 thinking tag 和误回显的视觉标记。
2. 对视频描述去掉显式时间先验，避免模型直接复述时间戳。
3. 调用 dataset 的 `_gen_seg_preprocess`，把 caption 注入 localization prompt。
4. 从 `worker.opsd.localization_rollouts` 读取 `K`，用当前 actor 为每条 caption 采样定位结果。
5. vLLM offload 后再把 VQ-SAM2 移入 GPU；按原图分组，仅计算一次 SAM2 image embedding，并分 chunk 解码目标 token 与 `G*K` 个预测 token。
6. 非法、缺失或空 mask 记为 IoU `0`。优先使用可转换的 dense/PIL/COCO RLE/polygon 原始 GT；缺失时解码 `seg_answer` 的目标 token，并记录 `raw_gt` 或 `decoded_target` reference 来源。
7. mask logits 双线性恢复原图尺寸并以 `0.5` 二值化；每条 caption 的 `K` 个 IoU 求均值得 `R_Ci`，再严格按 `0.5/0.85` 分路由。
8. 视频 cycle 保留原 tIoU 与 GRPO 路径，不进入 image-only OPSD teacher 路由。
7. 恢复外层 rollout `n`，返回 `cycle_cap_batch` 和 `cycle_seg_batch`。

代码中存在 `generate_sequences_with_ref`，可临时把 vLLM 换成 reference policy 权重，但当前调用已注释，实际调用 `generate_sequences`。因此当前有效实现确实是“actor 作为自己的 critic”，而不是冻结的外部 critic。

### 3.5 奖励

`verl/workers/reward/function.py::BatchFunctionRewardManager` 动态调用 `projects/rl/reward_function/text2mask.py:compute_score`，并只把标量奖励写到 response 最后一个有效 token；之后优势会扩展到整个 response mask。

核心图像 cycle source 的有效奖励仍保留 CycleGRPO 的倍率、格式和非重复项，但 `s_i,k` 与 `m_i` 已替换成真实像素值：

```text
s_i,k = graded_match(pred_mask_token_i,k, target_mask_token)
m_i   = mean_k(s_i,k)

caption:
  R_cap_i = (non_repeat_i + 10*m_i) * valid_i + valid_i
  valid_i 检查 caption 中没有 bbox、没有中文；违规时正奖励被门控清零。

localization:
  R_loc_i,k = 10 * (s_i,k * m_i) + non_repeat_i,k + mask_format_i,k
```

这与论文的 `R_cap_i=mean(s_i,k)`、`R_loc_i,k=R_cap_i*s_i,k` 对应，但代码额外乘 `10` 并加入格式/重复约束。

`text2mask.py` 还保留多任务分支：

| `source` | 奖励行为 |
|---|---|
| `groundingme` / `denseworld_*` / `dam_cyclegrpo` / `None` | 图像 CycleGRPO 主分支 |
| `gres_no_target` | no-target/null 正确性 + 非重复奖励 |
| `tg_multi_merged` | 视频循环：tIoU、时间格式、段数门控、禁止 caption 泄漏时间 |
| `dam_captioning` / `tg_captioning` | 外部 OpenAI-compatible vLLM judge 的布尔 caption reward；不是主 CycleGRPO 路径 |
| `dam_grounding` / `tg_grounding` | 独立 grounding 任务，分别做 mask-token 或时间区间奖励 |
| `gcg`、`psg` 等 | grounded caption/scene graph 的 token、短语、格式奖励或保留分支 |

`tg_reward.py` 是可配置的 temporal grounding 奖励库，支持 tIoU、format、precision/recall/F1、C-Acc、caption judge 和长度惩罚；当前 `text2mask.py` 的主要视频路径只直接复用其中少量逻辑或保留了注释调用。

### 3.6 GRPO 与策略更新

`verl/trainer/core_algos.py::compute_grpo_outcome_advantage`：

1. 对每个 response 求 token reward 总和。
2. 按 `uid` 聚合同一 prompt 的 `G` 个 rollout。
3. 计算组内均值和标准差，优势为 `(r_i - mean_group) / (std_group + eps)`。
4. 将该标量乘 response mask，作为每个生成 token 的 advantage/return。

`DataParallelPPOActor.update_policy` 重新计算 log probability，使用 clipped PPO/GRPO surrogate loss。caption 优势仍用同一 prompt 的全部 `G=6` 候选标准化，再由 `policy_loss_mask` 只对 high route 启用 caption PPO/KL；因此不会因 high 子集只有一条而失去组内基线。

low route 用 EMA teacher 在 privileged prompt 下采样 6 条自然 caption，过滤所有特殊 token/诊断泄漏，以当前 actor 做一次 greedy 重建，选每个低分轨迹的最佳改进 caption；相对原 `R_Ci` 提升至少 `0.05` 才采用，同 prompt 去重后最多两个 target。student 始终在原始 prompt 上做加权 CE，权重为 `(R_teacher-R_Ci)/(1-R_Ci+eps)`。

mid route 不重采样 caption。EMA teacher 在包含原图、目标/典型/最佳 mask token、IoU 向量及空间差异摘要的 privileged prompt 上 teacher-force 同一 student 轨迹；student 仍使用原 prompt。两者在完整词表上计算 `beta=0.5` generalized JSD，以归一化的 `exp(-H_teacher)` 强调 teacher 有把握的 token，样本权重为 `clamp((0.85-R_Ci)/0.35,0.1,1)`。

当 captioner 和 segmenter 都启用时，trainer 不分别 optimizer step，而是：

```text
high GRPO + low CE + mid JSD，按候选比例归一化，再乘 caption_loss_weight=0.5
全部 route 的 localization GRPO，再乘 localization_loss_weight=0.5
clip grad norm -> one optimizer.step()
optimizer.step 后原地执行 EMA shard 更新
```

这保证单一模型被两个方向联合优化。

## 4. SAMTok / VQ-SAM2 实现

### 4.1 离散 mask 表示

`projects/transformers/vq_sam2/modeling_vq_sam2.py`：

- `VQEmebedding`：EMA 更新的向量量化 codebook，支持重启未使用 code。
- `ResidualQuantizer`：逐层量化残差；当前训练配置是 depth 2、每层 size 256，得到两个 mask token id。
- `VQ_SAM2.forward`：SAM2 从图像、GT mask 和 bbox prompt 提取 mask embedding，残差量化后可重建 mask；训练损失含 commitment、sigmoid CE 和 Dice。
- `forward_with_codes`：把离散 code 还原为 embedding，再注入 SAM2 decoder 生成像素 mask，主要用于离线可视化和评测。
- `encode_single_image` / `decode_codes_from_single_image`：当前 OPSD 在线奖励路径复用单张图的 SAM2 backbone embedding，在受控 batch 中解码多组 code，避免 `G*K` 次重复图像编码。

`projects/transformers/vq_sam2/modeling_sam2.py` 和 `sam2/` 是 Hugging Face 化及 vendored 的 SAM2 图像编码器、prompt/mask decoder、memory attention/encoder、Hiera backbone 与 CUDA connected-components 代码。

### 4.2 MLLM 与 mask token

`projects/vlm/tokenmask/models/qwen3vl.py::QWEN3VL_VQSAM2Model` 是 cold-start/SFT 侧的 Qwen3-VL 包装：

- 加载 Qwen3-VL、tokenizer 和 processor。
- 冻结或解冻 vision encoder，支持 LoRA、activation checkpointing 和 checkpoint 权重导入。
- `forward` 当前只调用 Qwen3-VL 的 language-model loss；mask 已作为普通扩展词表 token 学习。
- `state_dict` 只保留 language model、lm head、投影层以及可选视觉参数。

RL 阶段直接通过 Hugging Face checkpoint 加载模型，不实例化上述 xtuner wrapper。`verl/models/monkey_patch.py` 根据模型类型替换 attention/forward；`verl/models/transformers/qwen3_vl.py` 实现多模态 RoPE、视觉 embedding 注入、文本/图像/视频混合 batch 和无视觉样本的 dummy graph 保活。

## 5. 目录与代码职责

### 5.1 根目录

| 文件 | 职责 |
|---|---|
| `README.md` | CycleGRPO 项目入口、训练/评测命令、公开结果和路径占位符 |
| `README_EasyR1.md` | 上游 EasyR1/veRL 框架说明 |
| `TRAIN.md` | 旧的单/多节点 cold-start SFT 环境备忘，路径具有内部环境痕迹 |
| `setup.py` / `pyproject.toml` | 将仓库安装为 `verl`；ruff 规则和 Python `>=3.9` |
| `requirements.txt` | CUDA/PyTorch 之外的核心依赖；Transformers 锁定 `4.54-4.57`，vLLM `>=0.8` |
| `Makefile` | 上游开发命令 |

### 5.2 `verl/`：RL 引擎

| 模块 | 实现职责 |
|---|---|
| `protocol.py` | `DataProto`：tensor/non-tensor/meta 三类数据的 select、union、repeat、concat、chunk、Ray 序列化 |
| `trainer/main.py` | CLI/YAML 配置合并、Ray runner、worker/reward/dataloader/trainer 组装 |
| `trainer/ray_trainer.py` | Cycle/non-cycle 分流、双阶段 rollout、reward/advantage、caption/seg 联合更新、验证与 checkpoint |
| `trainer/ray_trainer_old.py` | 上游/旧训练循环，仅供对照，不是主入口 |
| `trainer/core_algos.py` | GAE、GRPO、RLOO、ReMax、REINFORCE++，PPO clip loss、KL/value loss |
| `trainer/data_loader.py` | train/val `RLHFDataset` 和 sampler/DataLoader |
| `trainer/metrics.py` | reward、length、timing、throughput 指标汇总 |
| `workers/fsdp_workers.py` | actor/ref/critic 构建，FSDP-vLLM 权重切换，多模态前处理，rollout 后 token/tIoU 评分 |
| `workers/actor/dp_actor.py` | log-prob 前向、动态 micro-batch、PPO loss、梯度累积和 optimizer step |
| `workers/critic/dp_critic.py` | GAE/PPO 可选 value model；GRPO 主配置通常不启用 critic |
| `workers/rollout/vllm_rollout_spmd.py` | SPMD vLLM engine、采样参数、视觉输入和 response tensor 构造 |
| `workers/sharding_manager/fsdp_vllm.py` | FSDP 参数与 vLLM engine 同步/offload |
| `workers/sharding_manager/fsdp_ulysses.py` | sequence parallel 数据切分/还原 |
| `workers/reward/function.py` | 动态加载 sequential/batch 自定义 reward 并写 token-level score |
| `workers/opsd/config.py` | pixel IoU、路由、EMA teacher、regenerate 与 distillation 配置及边界校验 |
| `workers/opsd/mask_iou.py` | 严格 token 解析、原始 GT 转换、批量 mask 解码、尺寸恢复和像素 IoU |
| `workers/opsd/routing.py` | `R_Ci` 聚合、三路由边界、privileged context、route 权重与泄漏过滤 |
| `models/monkey_patch.py` | 为多种 HF MLLM 注册 flash attention 和混合多模态 forward |
| `models/transformers/*.py` | Qwen2/3-VL、Qwen3.5、Gemma4 的 RoPE、embedding 与 forward 适配 |
| `single_controller/` | Ray worker、worker group、注册装饰器、资源/dispatch 管理 |
| `utils/dataset.py` | 本项目数据 schema、图像/视频处理、双 prompt 构建和过滤 |
| `utils/dataset_old.py` | 上游/旧 dataset，仅供回溯 |
| `utils/checkpoint/` | FSDP 模型、优化器、scheduler、processor 的保存/恢复 |
| `utils/logger/` | file/wandb 等 experiment logger 和 generation logger |
| `utils/fsdp_utils.py` | FSDP wrap、state/offload、模型初始化工具 |
| `utils/seqlen_balancing.py` | 按 token 数均衡数据并记录不均衡指标 |
| `utils/ulysses.py` | Ulysses sequence parallel pad/slice/gather |
| 其余 `utils/*.py` | tokenizer、dtype、FLOPs、tensor/通用函数 |

### 5.3 `projects/rl/`：论文训练实现

| 文件/组 | 职责 |
|---|---|
| `qwen3vl_4b_mt.sh` | 当前论文主训练入口 |
| `config.yaml` | CycleGRPO 的 data/algorithm/worker/reward/trainer 配置 |
| `format_prompt/non_thinking.jinja` | 原样输出 prompt；主入口使用 |
| `format_prompt/r1v.jinja` | 旧的 think/answer 包装模板 |
| `reward_function/text2mask.py` | 图像 mask、bbox、视频时间段、GCG/PSG/no-target 的总奖励路由 |
| `reward_function/tg_reward.py` | temporal grounding 可组合奖励库 |
| `reward_function/llm_judge_reward.py` | 可选外部 vLLM caption judge 客户端，不属于无外部 judge 的主闭环 |

`projects/rl/datasets/` 全部是离线数据工具，不在 trainer 内自动运行：

- `prepare_dw_rl_dataset.py` / `prepare_dw_single_rl_dataset.py`：DenseWorld 多目标/单目标转 RL parquet，构造区域叠加图、caption/seg prompt 和 mask token。
- `prepare_gres_no_target_rl_dataset.py`：构造 no-target/null 拒识样本，是主 shell 的第二个数据源。
- `prepare_gres_rl_dataset.py`、`prepare_more_gres_rl_dataset.py`、`prepare_res_rl_dataset.py`、`prepare_reasonseg_rl_dataset.py`：不同 referring segmentation 数据转统一 schema。
- `prepare_gm_rl_dataset.py`：GroundingME；`prepare_padt_ric_rl_dataset.py`：PADT region-in-context。
- `prepare_gcg_rl_dataset.py`、`prepare_other_gcg_rl_dataset.py`、`prepare_grandf_rl_dataset.py`、`prepare_detail_gcg_cold_start_and_rl_data.py`：grounded caption 数据。
- `prepare_psg_rl_dataset.py`：panoptic scene graph；`prepare_ver*_data.py`：VER 数据。
- `prepare_coconut*_dataset.py`：COCONut/COCONut-DW 数据。
- `*_cold_start_*`：生成 co-SFT 数据，不直接进入 CycleGRPO rollout。
- `convert_mask_token_to_bbox.py` / `convert_json_mask_tokens_to_bbox.py`：用 VQ-SAM2 解码 token 并取 bbox，服务论文 bbox 泛化实验。
- `convert_gar_multi_regions_to_sam2tokens_with_zoom_in.py`：GAR 多区域及 zoom-in 预处理。
- `visualize_*.py` / `vis_mask_overlay.py`：解码、叠加和检查 parquet/mask token。

这些脚本普遍含本地数据路径，运行前必须逐个替换；生成后应先用可视化脚本抽样检查 schema、图像路径和 token 对齐。

### 5.4 `projects/transformers/`：模型定义

- `vq_sam2/configuration_vq_sam2.py`：SAM2/VQ-SAM2 HF config。
- `vq_sam2/modeling_vq_sam2.py`：离散 mask tokenizer。
- `vq_sam2/modeling_sam2.py`：较轻的 HF SAM2 wrapper。
- `vq_sam2/losses/`：CE、Dice、point sampling、accuracy。
- `vq_sam2/sam2/`：完整 SAM2 配置、图像/视频 predictor、automatic mask generator、Hiera、memory 模块与 CUDA 扩展。
- `qwen2_5_vl_vq_sam2/`：旧 Qwen2.5-VL + VQ-SAM2 HF 联合模型，主要服务历史 cold-start/SFT，不是 Qwen3-VL RL 主入口。

### 5.5 `projects/vlm/`：SFT、数据与历史实验

该目录有大量数据集转换脚本，按三条实现线组织：

1. `tokenmask/`：当前 SAMTok/Qwen3-VL cold-start 与评测栈。
   - `models/qwen3vl.py`、`qwen25vl.py`、`perceptionlm.py`：不同 MLLM wrapper。
   - `datasets/tokenmask_dataset.py`、`qwen3vl_dataset.py`、`qwen25vl_dataset.py`：conversation/多模态预处理；`collect_fns.py`：padding/collate。
   - `configs/`：Qwen3-VL/Qwen2.5-VL/PerceptionLM 的 SFT、微调、消融配置；`cycleGRPO_dam_ft.py` 是特定 DAM 微调配置，不是 RL 入口。
   - `utils/add_special_tokens.py`：扩展 mask token 词表；`merge_weight_*.py`：导出/合并权重。
   - `evaluation/`：RefCOCO/+/g、GRES、GCG、DAM、GAR、GroundingSuite、MR/PSG/PerceptionLM 及消融/可视化脚本。文件名前缀决定模型后端，后缀决定数据集与指标。
2. `vq_sam2/`：mask tokenizer 本身的预训练和数据工程。
   - `models/vq_sam2.py` / `sam2.py`：xtuner 风格 VQ-SAM2 与完整内联 SAM2。
   - `datasets/`：SA-1B、COCONut、ADE20K、Cityscapes、OpenPSG、Flickr、RefCOCO/GRES/ReVOS 等 source 的 dataset、collector 和统一格式转换。`collect_*_dataset_info.py` 生成索引，`convert_*_to_uniformat.py` 统一样本格式，`visualize_*` 负责 QA。
   - `configs/`：A100/H20/Ascend、多 codebook depth/size、共享与否、warmup/continue/ablation 配置；配置文件只是实验参数，不会被 RL shell 引用。
3. `qwen2_5_vl_vq_sam2/`：旧的 Qwen2.5-VL 联合训练栈。
   - `models/`、`configs/`：联合模型和 official trainer。
   - `datasets/`：大量 `convert_<source>_to_sam2tokens.py`，其共同职责是读取各 source annotation/mask，调用 tokenizer，输出统一 conversation/mask-token 格式；`collect_*` 汇总训练项；`refer.py`/`grefer.py` 是数据 API。
   - `evaluation/`：RefCOCO 和 GCG 的旧评测实现。

因此 `projects/vlm/` 中以 `convert_`、`collect_`、`prepare_` 开头的文件不是运行时模块，而是按文件名指定 source 的一次性 ETL；修改它们时仍须在本文变更日志记录输入 schema、输出 schema 和验证样本。

### 5.6 `evaluation/`：论文评测入口

| 目录 | 文件职责 |
|---|---|
| `gres/` | `qwen3vl_gres_eval.py` 解码 mask token、保存 shard、计算 gIoU/cIoU/N-acc；shell 多 GPU 分片 |
| `groundingsuite/` | Qwen3-VL 推理、按 task 分片和自动合并；数据路径需替换 `<PATH_TO_COCO2014>` |
| `gcg/` | 生成 interleaved text-mask，解码 mask 并保存 RLE/文本供官方 GCG 指标；数据根需替换 |
| `gar/` | VQA 和 detailed caption 两个推理入口；`gar_vqa_metrics.py` 汇总总体与属性类别准确率 |
| `dlc_bench/` | 多后端 caption inference、裁剪/区域输入、judge server、GPT-with-image/Llama-without-image 评测和绘图 |
| `bbox/` | Qwen2.5/3/3.5、InternVL、Gemma、Llama 的 bbox 输出泛化；解析 `[x1,y1,x2,y2]` 并按 0-1000 坐标还原 |

评测脚本通常直接加载 Hugging Face checkpoint 和 mask tokenizer 权重，不经过 `verl` trainer。它们含数据路径占位符、benchmark 特定依赖和输出约定，不能只凭主 README 直接运行。

## 6. 当前实现中的关键注意事项

1. **当前主配置是 `G=6,K=6`。** `G` 来自 `worker.rollout.n`，`K` 来自 `worker.opsd.localization_rollouts`。
2. **图像 cycle 训练已使用真实像素 IoU。** 必须提供有效的 `mask_tokenizer_path`、SAM2 权重和足够显存；vLLM 与 VQ-SAM2 严格分时驻留 GPU。
3. **内层使用当前 actor。** `generate_sequences_with_ref` 已实现但未启用；不要把它误写成冻结 critic。
4. **cycle source 是硬编码列表。** 新增数据源若未同步 `_make_batch_data`、reward manager 和 `text2mask.compute_score`，会落入错误分支或抛 `NotImplementedError`。
5. **外层/内层 batch 必须可按 world size 分发。** trainer 会丢弃少量不完整 group；混合 source 或修改 `n` 后要检查有效样本数。
6. **奖励量纲并非论文原始公式。** 主分支含 `10x`、format、non-repeat、语言/bbox gate；对实验解释必须写明。
7. **调试文件会被覆盖。** 当前 inner rollout 每次写 `debug_response_cap_debug0223.txt`；多 rank/并发环境可能相互覆盖。
8. **存在大量历史代码。** `*_old.py`、Qwen2.5-VL 联合栈、未调用 reward 分支和注释块不应被当作当前执行路径。
9. **路径尚未参数化完整。** 主训练、评测和 ETL 都有 `<PATH_TO_*>` 或本地路径，生产运行前必须审计。
10. **测试覆盖有限。** 多数验证依赖 GPU、checkpoint 和数据集；小改动至少运行语法检查/导入检查，训练路径改动还应做最小单 batch smoke test。
11. **OPSD dataclass 默认关闭，项目 YAML 显式开启。** 原始 SAMTok 消融设 `worker.opsd.enabled=false`；仅真实 IoU 的 CycleGRPO 设 `opsd.enabled=true`、`routing.enabled=false`、`ema_teacher.enabled=false`；完整版本保持主 YAML 默认。
12. **privileged distillation 第一版要求 `actor.ulysses_size=1`。** response 会裁到当前 micro-batch 的最大有效长度再计算完整词表 JSD；其他 sequence-parallel 配置会在启动时显式报错。
13. **EMA checkpoint 位于 `actor/ema_teacher/`。** resume 优先恢复完整 EMA shard；旧 checkpoint 缺失 teacher 时从已恢复 actor 初始化，frozen reference policy 始终保持 cold-start anchor。

## 7. 修改代码时的文档维护规则

每次代码修改都必须执行：

1. 修改前阅读本文件，确认当前调用链、有效分支和历史分支。
2. 在对应章节更新新的行为、配置、schema、调用关系或风险；不能只在变更日志写一句话。
3. 在下方变更日志新增一条，包含日期、修改文件、行为变化和验证方式。
4. 若新增模块/脚本，把它加入“目录与代码职责”；若删除或弃用模块，明确迁移路径。
5. 若实现与论文公式产生偏差，在“论文结论与实现边界”或“关键注意事项”中写明。

推荐日志格式：

```markdown
### YYYY-MM-DD - 简短标题

- 代码：`path/to/file.py`
- 文档：更新了第 X 节
- 行为：说明修改前后差异、配置或数据契约变化
- 验证：列出实际执行的命令/测试；未执行时说明原因
```

## 8. 变更日志

### 2026-07-19 - 建立论文与代码知识库

- 代码：未修改训练或评测代码。
- 文档：新增 `code.md`、`Agent.md` 和标准 Agent 入口 `AGENTS.md`。
- 行为：记录论文 2607.11581v1、CycleGRPO 双阶段调用链、有效奖励、SAMTok/VQ-SAM2、数据与评测模块；建立“改代码前阅读、改代码后同步文档和日志”的强制规则。
- 验证：渲染并检查论文 29 页，提取正文/补充材料；逐段核对主 shell、YAML、trainer、dataset、FSDP/vLLM worker、reward、GRPO、SAMTok/VQ-SAM2 和各评测入口；执行 Markdown/仓库状态检查。

### 2026-07-19 - 实现 OPSD 真实 IoU 三路由训练

- 代码：新增 `verl/workers/opsd/` 与 `tests/test_opsd_core.py`；修改 `projects/rl/config.yaml`、`qwen3vl_4b_mt.sh`、VQ-SAM2 decoder、Qwen3-VL forward、dataset、trainer、actor 和 FSDP worker。
- 文档：更新第 1-6 节的真实像素 IoU、G/K 配置、候选级路由、EMA teacher、regenerate/JSD/high-GRPO、checkpoint、消融开关及限制。
- 行为：训练时完整解码 mask token；按 `R_Ci<0.5`、`0.5<=R_Ci<=0.85`、`R_Ci>0.85` 分别执行 teacher regenerate、on-policy generalized JSD 和 CycleGRPO。localization 对全部 route 保持 `R_Ci*s_i,k`；optimizer 后更新独立 EMA teacher，checkpoint 保存到 `actor/ema_teacher/`。
- 工程：vLLM 与 VQ-SAM2 分时驻留；每图复用一次 SAM2 embedding；route 子批次补齐到 world size；teacher/student 多图输入分离；JSD 只保留有效 response logits；privileged mask 压缩且每 caption 只传一份；特殊 token 与诊断文本不会进入 teacher SFT target。
- 验证：所有修改 Python 文件通过 `py_compile`，`git diff --check` 通过；8 个纯 OPSD 单元测试通过，覆盖 token offset、mask IoU、单图 embedding 复用、路由边界/穷尽、权重、privileged context 和泄漏过滤。当前机器缺少完整 veRL 依赖、8 卡 GPU 与模型 checkpoint，未执行 FSDP/vLLM 单 batch smoke training。
