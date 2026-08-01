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

当前扩展在每条 caption 的 `K` 次真实 IoU 均值 `R_Ci` 上执行候选级三路由：`R_Ci<0.5` 进入 EMA teacher regenerate，`0.5<=R_Ci<=0.85` 进入 privileged on-policy distillation，`R_Ci>0.85` 保留 CycleGRPO caption GRPO。通用配置默认维持三路由替换 caption 更新；火山引擎 B 实验显式启用 `routing.preserve_original_grpo=true`，使所有安全 caption 都保留原始 CycleGRPO GRPO，再把 regenerate CE 或 privileged JSD 作为附加梯度。高置信 teacher 消融进一步只让满足 cycle 证据阈值的 regenerate/JSD 样本提供辅助梯度，不会关闭任一安全 caption 的原始 GRPO。所有 localization rollout 始终参与 CycleGRPO 更新。

## 2. 论文结论与实现边界

### 2.1 论文方法

- 基座：Qwen3-VL-4B 上的 SAMTok，mask 被离散为 `<|mt_start|><|mt_xxxx|><|mt_xxxx|><|mt_end|>`。
- 训练数据：论文报告约 20k DenseWorld 图像/区域，加约 1k GRES no-target 表达。
- 论文默认：caption group size `G=6`，每个描述的 localization rollout `K=6`，总 batch size 128，AdamW，学习率 `1e-6`，weight decay `1e-2`，1 epoch。
- 训练时冻结 vision encoder，优化 projection/LLM 参数。
- 主要评测：DLC-Bench、GAR-Bench-VQA、GCG、GRES、GroundingSuite；不在这些下游集上继续微调。

### 2.2 当前公开代码的有效配置

通用主入口是 `projects/rl/qwen3vl_4b_mt.sh`；火山引擎 RefCOCO 10k 单节点部署入口是
`projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`。二者都覆盖 `projects/rl/config.yaml`
的若干默认值：

| 项 | 当前主入口 | 说明 |
|---|---:|---|
| 模型 | `<PATH_TO_COLD_START_CKPT>` | 必须替换为 co-SFT/SAMTok checkpoint |
| 外层 rollout `G` | `worker.rollout.n=6` | 与论文及 OPSD 默认一致 |
| caption response 上限 | `256` token | 火山引擎入口的稳定化消融值；同时是 caption 安全门控的超长阈值 |
| 内层 rollout `K` | `worker.opsd.localization_rollouts=6` | 已从 trainer 硬编码迁入配置 |
| 路由阈值 | `0.5 / 0.85` | 边界分别为 low: `<0.5`、mid: `[0.5,0.85]`、high: `>0.85` |
| caption 原始 GRPO | B 入口默认保留 | 所有安全 rollout 都计算原始 CycleGRPO policy loss；low/mid 的 teacher 更新改为附加梯度 |
| C: caption anchor KL | `0.05`、全部安全 route | 火山引擎入口的稳定化值；独立于 `algorithm.kl_coef=0.01`，只锚定 cycle caption |
| C2: segmentation anchor KL | `0.05`、全部 cycle localization response | 与通用 KL 分开记录；以 frozen reference 约束 mask-token policy，避免共享 actor 的 caption/teacher 更新快速破坏 text-to-mask |
| C2: 非对称梯度投影 | 关闭 | 仍可显式启用；实测 caption/seg cosine 接近 0，投影对更新方向影响很小 |
| 高置信 teacher gate | 开启 | regenerate 要求 `teacher R_Ci>=0.65` 且归一化改善 `>=0.30`；JSD 仅接收 `R_Ci>=0.65` 的 mid route |
| C: JSD 特殊词表屏蔽 | 开启 | teacher/student JSD 同时禁止 `mt_*` 和 `object_ref_*` token |
| teacher 消融入口 | 默认 `decay=1.0`、CPU offload | `qwen3vl_4b_refcoco10k_volcengine.sh` 默认冻结启动时复制的 SAMTok teacher；主 YAML 仍为 EMA `0.999`，与 frozen reference policy 独立 |
| regenerate | `T=6`、`temperature=0.8`、`top_p=0.95` | 每候选一次 greedy localization 验证，提升至少 `0.05` 才接收 |
| teacher diagnosis | 每 step 最多 2 条、96 tokens、temperature 0 | 仅写入本地 privileged diagnostics 日志，不参与 student 更新 |
| rollout/global batch | `128` | 与论文一致 |
| epoch | `1` | 与论文一致 |
| GPU | 1 node x 8 GPU | Ray + FSDP + vLLM SPMD |
| vision tower | frozen | shell 覆盖为 `true` |
| caption/segmenter | 都优化 | 最终按 `0.5/0.5` 梯度权重累积 |
| 验证 | checkpoint 后离线 RefCOCO | 入口每 5 step 保存 checkpoint，`val_freq=-1`、`val_before_train=false`；通用 trainer validation 不执行 mask reconstruction，不能代替标准 RefCOCO cIoU/mIoU |
| 日志 | file + wandb | shell 强制 `WANDB_MODE=offline` |

火山引擎入口默认使用 `/mnt/cxzx/workspace/data_transfer/houzhiyan` 下的仓库、Conda
环境、已修复绝对图像路径的 RefCOCO 10k parquet 和 SAMTok checkpoint，可用同名环境变量覆盖。当前默认输出根为
`logs/refcoco10k_opsd_frozen_teacher`，并设置 `worker.opsd.ema_teacher.decay=1.0`，所以 FSDP
worker 初始化时从 actor 复制的 SAMTok 参数之后不会更新；需要继续同一冻结实验时才显式设 `RESUME=true`。
该入口还默认设置 `CAPTION_MAX_RESPONSE_LENGTH=256`，并把同一值传给
`worker.opsd.caption_safety.max_response_tokens`。这是针对 OPSD caption 任务漂移的稳定化消融，
与论文历史的长 response 配置不同；可显式覆盖环境变量，但 rollout 上限和安全阈值必须保持一致。
入口的 `PRESERVE_ORIGINAL_GRPO=true` 是 B 实验默认值：所有安全 caption 通过原始 CycleGRPO
reward/advantage 计算 PPO/GRPO，low route 的 regenerate CE 和 mid route 的 JSD 不再替代它，而是
在同一 optimizer step 前额外累积。设为 `false` 可复现之前的 route-replacement 消融。
入口当前还默认启用 C：`CAPTION_ANCHOR_KL_COEF=0.05` 会以 frozen reference
对全部安全 cycle caption 增加独立 KL；`JSD_BLOCK_CAPTION_SPECIAL_TOKEN_VOCAB=true` 会在
privileged JSD 的 softmax 前同步屏蔽 SAMTok mask 与 object-reference token。C2 保留原 OPSD 的
GT mask、代表性 reconstruction mask 和 student caption 的 teacher diagnosis 语义，但把前两者从 raw
mask token、IoU、坐标和差异摘要文本改为 teacher-only 的三图证据：全图、GT 目标 crop 和代表性重建 crop。
teacher 仍分别执行 regenerate、同轨迹 JSD 和训练诊断；student prompt、原始 GRPO target 与 localization
rollout 保持不变。入口默认关闭已验证效果很小的 `ASYMMETRIC_GRADIENT_PROJECTION`，并启用
`TEACHER_CONFIDENCE_ENABLED`：low route 的 teacher caption 必须同时达到最终 `R_Ci>=0.65`、填补原
caption 剩余 IoU 差距的比例 `>=0.30` 与原有绝对改善 `>=0.05`，才加入 regenerate CE；mid route 仅当原
caption 的 `R_Ci>=0.65` 时才加入 privileged JSD。低置信辅助样本仍保留其原始 GRPO，全部 localization
rollout 仍保留 CycleGRPO，因此这是 teacher auxiliary-loss 的样本选择消融，不是直接 RefCOCO CE 或
移除闭环训练。
入口的 `OPSD_ENABLED=false` 是原始 HTG 纯 CycleGRPO 对照：不会构建/调用 pixel mask decoder 或任何
teacher routing、anchor KL、CE/JSD 辅助项；localization rollout 保留 FSDP worker 中的二级 mask-token
grading（完整匹配 `1.0`、两个 code 匹配 `0.8`、首 code 匹配 `0.4`、否则 `0.0`）。这与
`OPSD_ENABLED=true` 的真实 pixel-IoU 路径是互斥的，不能将两者的 `R_Ci` 数值或 `0.5/0.85` 阈值直接比较。
为隔离真实 pixel-IoU 与 teacher 辅助项，入口也接受 `OPSD_ENABLED=true`、`PIXEL_IOU_ENABLED=true`、
`ROUTING_ENABLED=false`。该组合仍在每个 cycle 生成后完整解码预测/GT mask 并把真实 pixel IoU 写回
caption 与 localization CycleGRPO reward，但所有 caption route 固定为 `grpo`，不创建 EMA teacher，
不执行 regenerate CE、privileged JSD 或 teacher diagnosis。用于该对照时还应显式将 caption/segmentation
anchor KL 设为 `0`、关闭 caption safety，以免保留 C/C2 的额外策略约束；这是一项当前扩展的受控消融，
不是论文公开 HTG 实现。`CAPTION_SAFETY_ENABLED`、`CAPTION_SAFETY_FORCE_REGENERATE`、
`EMA_TEACHER_ENABLED` 与 `TEACHER_ANALYSIS_ENABLED` 默认均为 `true`；`PIXEL_IOU_ENABLED` 与
`ROUTING_ENABLED` 默认跟随 `OPSD_ENABLED`，所以主 C2 与原始 HTG
启动行为均不变。入口会拒绝没有 pixel IoU 或 EMA teacher 的三路由配置。
`trainer.val_freq` 保持关闭，因为其仅生成 caption 并调用通用 reward，既不运行 CycleGRPO 的 localization
rollout，也不能计算标准 RefCOCO cIoU/mIoU。每 5 step 保存的 checkpoint 应在训练进程退出、释放 8 卡后通过
离线评测入口执行 RefCOCO val。设置入口的可选 `MAX_STEPS=5,10,...` 可将训练分段停在这些 checkpoint，
再以 `RESUME=true` 继续同一固定-teacher 实验。平台会注入
指向 Python 3.12 / Ray 2.53 集群的 `RAY_ADDRESS`，但项目环境是 Python 3.10 / Ray
2.56；该入口会清除继承的 Ray 地址，让 `verl.trainer.main` 创建版本一致的本地单节点
Ray。训练 stdout、W&B、teacher diagnosis 和 checkpoint 写到仓库内
`logs/refcoco10k_opsd/`；Ray session、object store 与 spill 文件写到本地短路径
`/tmp/cgrpo-ray-<uid>`。这同时保持 Ray socket 路径不超过 Linux `AF_UNIX` 的 107
字节限制，并避免持久化 workspace 挂载接近满盘时使 Ray 停止创建/溢写对象。入口拒绝
符号链接的 Ray 临时目录及使用率不低于 95% 的临时文件系统，并在创建 GPU/Ray worker
前扫描 parquet 的 `images` 列，验证所有图像路径均存在。它不修改论文算法或训练超参数，
只固定当前服务器的数据与运行环境。

火山引擎离线评测入口是 `projects/eval/qwen3vl_4b_volcengine.sh`。训练 checkpoint 中的
`actor/model_world_size_8_rank_*.pt` 是 FSDP shard，`actor/huggingface/` 只包含配置和
processor；因此必须先执行 `export` action，以相同 8-rank FSDP 拓扑只加载 actor model shard 并导出
标准 safetensors HF 目录。之后 `refcoco`、`groundingsuite` 和 `dlc` action 使用独立 CUDA
进程，不连接训练 Ray cluster。标准 RefCOCO 读取服务器的 `instances.json`、`refs(unc).p`
及 `train2014`，输出 cIoU/mIoU；它不能由 GRES/gRefCOCO 脚本替代。GroundingSuite 接收其
数据根和可选 COCO 图像根，并在推理后保留逐样本 JSON 与合并 JSONL；仓库 metric 使用逐样本 JSON
目录计算 mask GIoU。GroundingSuite 的 JSONL 若只保存
12 位 COCO image ID（如 `000000123456.jpg`），推理器会在 `data_root`、其 `assets/`、
`unlabeled2017/`、可用的 `train2014/` 子目录及 `coco_root/train2014` 中同时尝试该名称及官方的
`COCO_train2014_000000123456.jpg` 名称；无法
解析或读取的图像会立即令对应 shard 失败，不会经过多次退避后静默跳过并产生不完整结果。DLC-Bench action 只产出 prediction JSON，最终语言 judge 需要
单独配置可用凭据。DLC caption inference 对全局图和可选 zoom-in 图使用与训练相同的正向 caption 指令
`Provide a detailed factual description of this region {SEG}.`；评测 prompt 不出现 `mask`、`token`、`JSON` 或
`reasoning` 等 segmentation 格式词，以免 SAMTok 将描述请求误解为定位请求。生成上限固定为 192 token。此协议是评测条件的一部分，比较任何 checkpoint 前都必须以同一版本重新推理。

该入口以项目 Conda 的明确解释器运行，并将仓库根目录加入 `PYTHONPATH`。顶层
`evaluation/*/*.py` 是按文件路径执行的脚本，Python 默认只会把其子目录加入 `sys.path`；若
遗漏该设置，`from projects...` 会因找不到仓库顶层包而失败。

仓库 README 明确标记为 WIP，不应假设它是论文所有实验的逐字复现版本。

### 2.3 RefCOCO 20k 受控训练数据

`projects/rl/datasets/prepare_refcoco_rl_dataset.py` 可把标准 RefCOCO 的
`instances.json`、`refs(unc).p` 和 COCO 图像目录转换成当前 RL loader 所需的
Parquet。它按 seed 固定打乱 train refs，逐个用当前 SAMTok VQ-SAM2 权重编码目标
mask，并在获得 `max_samples` 条有效样本后停止；默认正好产出 20,000 条。VQ-SAM2
返回的 code 张量形状为 `(batch, mask_tokens, codebook_depth)`；转换器仅接受一个
mask、两个 code，并在校验元素数量后展平为 SAMTok token。

生成数据的 source 是 `refcoco_cycle`，会进入图像 CycleGRPO 分支。每条数据同时
保存 mask token 和压缩 COCO RLE，因此 OPSD 训练奖励优先使用原始 RLE 计算像素 IoU。
该数据替代论文报告的 DenseWorld 约 20k 区域数据，属于受控数据替换实验，不能将结果
直接表述为论文原始数据设置的复现。

### 2.4 GroundingSuite 类型均衡 20k 受控训练数据

`projects/rl/datasets/prepare_balanced_cyclegrpo_dataset.py` 将五份已转换的 parquet 固定混合为
20,000 条图像/mask CycleGRPO 数据：7,000 条 RefCOCO 单实例 (`refcoco_cycle`)、5,000 条
gRefCOCO 多实例 union mask (`grefcoco_cycle`)、4,000 条 COCO-Stuff 语义区域
(`cocostuff_cycle`)、2,000 条 PACO-LVIS 真 part mask (`paco_part_cycle`) 和 2,000 条
gRefCOCO no-target (`gres_no_target`)。这些比例分别为 Single 35%、Multi 25%、Stuff 20%、
Part 10%、no-target 10%，用于针对 GroundingSuite 的四类目标类型做数据分布受控消融。

`prepare_paco_lvis_part_cycle_dataset.py` 只接受 `id != obj_ann_id` 的 PACO annotation，并且每张图
最多选一个 part，因而不会把 parent object mask 混入 Part 配额或让少数密集标注图主导。PACO-LVIS
需要其对应的 COCO 2017 图像。`prepare_cocostuff_cycle_dataset.py` 只从官方
`stuffthingmaps_trainval2017` PNG 的像素值 91..181 选取区域；这些值对应 COCO-Stuff 官方 label id
92..182，0..90 是 COCO thing、255 是 void，均不进入 Stuff 配额。Stuff 转换按类别频率优先选取，
仅保留面积占图像 1% 到 90% 的语义区域。

所有正样本只写入图像、目标 mask、由当前 SAMTok VQ-SAM2 编码出的 `seg_answer` 与从该 mask 构造的
caption prompt；RefCOCO、gRefCOCO、PACO 与 COCO-Stuff 正样本均不注入任何 referring expression
或人工 `cap_answer`。gRefCOCO no-target 继续沿用已有 `gres_no_target` schema 与 rejection reward，
其表达式是必须被拒识的 query。混合器写入 manifest，固定记录
输入 parquet、seed、五类数量与最终 source counts；任一来源不足请求数量时 fail-fast。该数据配方是当前
OPSD 扩展的 GroundingSuite 覆盖实验，并非论文原始 DenseWorld 数据或方法公式的一部分。

## 3. 主训练调用链

### 3.1 启动与配置合并

1. `projects/rl/qwen3vl_4b_mt.sh` 或服务器入口 `qwen3vl_4b_refcoco10k_volcengine.sh` 调用 `python3 -m verl.trainer.main`；服务器入口会先清除不兼容的外部 `RAY_ADDRESS`。
2. `verl/trainer/main.py::main` 按“dataclass 默认值 -> YAML -> CLI 覆盖”合并配置，并初始化 Ray。
3. `Runner.run` 加载 tokenizer/processor，创建共享 GPU resource pool、`FSDPWorker`、batch reward manager 和 dataloader。若 Qwen3-VL checkpoint 的 processor 元数据不完整，`get_processor` 会在 `AutoProcessor` 返回 tokenizer/image processor 等非复合对象时，根据 `config.json` 的 `model_type=qwen3_vl` 显式回退到 `Qwen3VLProcessor`；其他模型仍保持原有的可选 processor 行为。
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
| `source` | 决定 cycle/non-cycle 分流和奖励分支；RefCOCO 转换工具写入 `refcoco_cycle` |
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
4. 对 image OPSD，像素 IoU 回写后 driver 用未跳过 special token 的实际 caption rollout 检查：非终止的 `<|...|>` special token、`mask_2d` JSON 和超过 `caption_safety.max_response_tokens` 的输出都标为不安全。默认强制将其 route 改为 `regenerate`；不安全 caption 不进入原始 caption GRPO 或 mid-route JSD，但 localization rollout/奖励仍保留。
5. 按 `source` 分流：`denseworld_single`、`denseworld_multiple`、`refcoco_cycle`、`grefcoco_cycle`、`cocostuff_cycle`、`paco_part_cycle`、`tg_multi_merged`、`dam_cyclegrpo` 和 `None` 进入 cycle batch；其他 source 进入 non-cycle batch。`grefcoco_cycle` 将 gRefCOCO 正样本的一个或多个 COCO instance mask 合并为 cycle target；`cocostuff_cycle` 是单类语义 Stuff 区域，`paco_part_cycle` 是真 object-part 区域；三者均走相同的 caption-to-localization rollout、真实 pixel IoU 与 CycleGRPO reward。其 `ann_id=[-1]` no-target 表达保留为 `gres_no_target` non-cycle 样本。
6. cycle/non-cycle 分别裁成能被 world size 整除的完整 GRPO groups，并按 token 数重排，降低各 rank 负载不均。

`vllm_rollout_spmd.py` 负责：

- 把 raw prompt 和图像/视频整理成 vLLM 输入。
- 为未原生识别 Qwen3-VL 的旧版 vLLM generic Transformers backend 暴露
  `Qwen3VLConfig.text_config` 中的语言模型字段，避免要求修改 checkpoint 的
  `config.json`；新版 vLLM 有原生实现时该兼容层无副作用。
- 采样并 pad response，构造完整 `input_ids`、`attention_mask`、`response_mask` 和扩展后的 position ids。
- 删除模型误生成的 vision 特殊 token，防止后续前向出现视觉 token/feature 数量不匹配。

`BatchFunctionRewardManager` 只会把 cycle 元数据中的 `iou_scores` 传给已注册的
image/video cycle source。`refcoco_cycle` 与 DenseWorld 一样需要此字段，供 caption
reward 和 segmentation reward 使用；遗漏该 source 会使 `text2mask.compute_score` 在
计算 `10 * iou_scores` 时收到 `None` 并在首个 batch 退出。

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
  valid_i 同时检查没有 bbox/中文，且没有非终止 special token 或 mask_2d JSON；违规时正奖励被门控清零。

localization:
  R_loc_i,k = 10 * (s_i,k * m_i) + non_repeat_i,k + mask_format_i,k
```

这与论文的 `R_cap_i=mean(s_i,k)`、`R_loc_i,k=R_cap_i*s_i,k` 对应，但代码额外乘 `10` 并加入格式/重复约束。

`text2mask.py` 还保留多任务分支：

| `source` | 奖励行为 |
|---|---|
| `groundingme` / `denseworld_*` / `refcoco_cycle` / `grefcoco_cycle` / `cocostuff_cycle` / `paco_part_cycle` / `dam_cyclegrpo` / `None` | 图像 CycleGRPO 主分支 |
| `gres_no_target` | no-target/null 正确性 + 非重复奖励 |
| `tg_multi_merged` | 视频循环：tIoU、时间格式、段数门控、禁止 caption 泄漏时间 |
| `dam_captioning` / `tg_captioning` | 外部 OpenAI-compatible vLLM judge 的布尔 caption reward；仅当 batch 实际包含这些 source 时才初始化 judge client，不是主 CycleGRPO 路径 |
| `dam_grounding` / `tg_grounding` | 独立 grounding 任务，分别做 mask-token 或时间区间奖励 |
| `gcg`、`psg` 等 | grounded caption/scene graph 的 token、短语、格式奖励或保留分支 |

`tg_reward.py` 是可配置的 temporal grounding 奖励库，支持 tIoU、format、precision/recall/F1、C-Acc、caption judge 和长度惩罚；当前 `text2mask.py` 的主要视频路径只直接复用其中少量逻辑或保留了注释调用。

### 3.6 GRPO 与策略更新

`verl/trainer/core_algos.py::compute_grpo_outcome_advantage`：

1. 对每个 response 求 token reward 总和。
2. 按 `uid` 聚合同一 prompt 的 `G` 个 rollout。
3. 计算组内均值和标准差，优势为 `(r_i - mean_group) / (std_group + eps)`。
4. 将该标量乘 response mask，作为每个生成 token 的 advantage/return。

`DataParallelPPOActor.update_policy` 重新计算 log probability，使用 clipped PPO/GRPO surrogate loss。caption 优势仍用同一 prompt 的全部 `G=6` 候选标准化。默认 route-replacement 消融由 `policy_loss_mask` 只对 high route 启用 caption PPO/KL；因此不会因 high 子集只有一条而失去组内基线。B 实验设置 `routing.preserve_original_grpo=true` 后，`policy_loss_mask` 改为所有 `caption_safe` rollout：safe high 只保留原始 GRPO，safe low 在它之上增加可接受的 regenerate CE，safe mid 在它之上增加 JSD。像素 IoU 的原始三路由之后，caption safety 会把 special-token、mask JSON 或超长 rollout 强制改为 low regenerate；它们不作为原始 GRPO/JSD 的 student trajectory，但若 teacher 生成安全且经 greedy reconstruction 验证的候选，仍可提供 regenerate CE。该门控不改变 segmentation batch，全部 localization rollout 继续参与其 GRPO 更新。主日志记录 `opsd/caption_safe_rate`、三种 unsafe rate、`opsd/caption_forced_regenerate_count` 以及 B 的 `opsd/caption_original_grpo_active_{count,rate}`；reward 指标也拆分为 `cap_no_bbox_no_chinese_score` 与 `cap_no_special_token_or_json_score`，不再用错误的 `cap_no_mask_token_check_score` 名称代表 bbox/CJK gate。

low route 用 EMA teacher 在 privileged prompt 下采样 6 条自然 caption，过滤所有特殊 token/诊断泄漏，以当前 actor 做一次 greedy 重建，选每个低分轨迹的最佳改进 caption；相对原 `R_Ci` 提升至少 `0.05` 才采用，同 prompt 去重后最多两个 target。启用 `teacher_confidence` 时还必须满足 `R_teacher>=0.65` 及归一化改善 `(R_teacher-R_Ci)/(1-R_Ci+eps)>=0.30`，防止只在低 IoU 区间内相对更好、但仍没有可靠定位证据的 teacher 文本改写共享 actor。student 始终在原始 prompt 上做加权 CE，权重为同一归一化改善值。

mid route 不重采样 caption。EMA teacher 使用三张 teacher-only 图像：原图全景、由 GT mask 隔离出的目标 crop、以及由代表性 localization reconstruction 隔离出的 crop；并根据 student caption 进行同轨迹 teacher forcing。两个 crop 使用同一 GT/reconstruction union box、外扩 15%、mask 外中性灰填充，并在送入 processor 前各自限制为最多 `512x512` 等效像素，避免三图使 teacher FSDP 的视觉 token 峰值失控。GT/reconstruction mask 仍是 privileged evidence，但 teacher prompt 不再写 raw mask token、IoU 向量、面积/中心、相对位置或差异摘要，避免这些几何文本诱导全图定位语言。启用 `teacher_confidence` 时，仅 `R_Ci>=0.65` 的 mid route 进入 JSD；低于该值表示 student caption 尚缺少稳定的 cycle grounding，teacher 的 GT-conditioned token distribution 不作为共享 actor 的直接锚点。其余 JSD 细节保持不变：`beta=0.5` generalized JSD、归一化的 `exp(-H_teacher)` teacher 置信度和 `clamp((0.85-R_Ci)/0.35,0.1,1)` 样本权重。C 的第一部分在每个 JSD chunk 的 teacher/student softmax 前将 tokenizer 词表中所有 `<|mt_start|>`、`<|mt_####|>`、`<|mt_end|>` 和 `<|object_ref_*|>` logit 置为不可选，因此这些分割结构没有概率质量、JSD 梯度也不会把它们泄漏到 caption。student 原始 GRPO target 与 localization rollout 不变。为控制 Qwen3-VL 大词表的峰值显存，`workers/opsd/distillation.py` 继续按 response token 块计算 teacher 熵、token score 和 JSD；每块的 student JSD softmax/probability 中间量使用 activation checkpoint 在反向时重算。

C 还新增独立 caption anchor KL：PPO 继续使用 `policy_loss_mask`，但当 `caption_anchor_kl_all_safe_routes=true` 时，cycle caption 的 KL 使用原始 response mask 与全部 `caption_safe` route，不复用 PPO route mask。它以 `caption_anchor_kl_coef=0.05` 加入自己的 token-weighted loss numerator；non-cycle caption 和 segmentation batch 不接收该额外项，原有 `algorithm.kl_coef` 保持不变。C2 同时增加独立 segmentation anchor KL：所有 cycle localization response 都以完整 response mask 对 frozen reference 计算 `segmentation_anchor_kl_coef=0.05` 的附加 KL；它与通用 `algorithm.kl_coef=0.01` 相加，但不会施加到 caption 或 non-cycle batch。非对称梯度投影仍保留为可选诊断：`asymmetric_gradient_projection=true` 时每个 FSDP rank 先暂存 caption GRPO、regenerate CE、JSD 和 caption-anchor 的梯度，再计算 localization GRPO/segmentation-anchor 梯度；若全局内积为负，仅从 caption gradient 中减去其沿 localization gradient 的反向分量，最后仍执行原有的单次 optimizer step。当前服务器日志的 cosine 仅约 `-0.004` 到 `-0.018`，故入口默认关闭它。高置信 gate 新增 `opsd/regenerate_validated_candidate_count`、`opsd/regenerate_confident_candidate_{count,rate}`、`opsd/regenerate_confident_target_acceptance_rate`、`opsd/distillation_route_count`、`opsd/distillation_confident_{count,rate}` 与 `opsd/distillation_confident_R_Ci_mean`，必须同时检查这些项，避免阈值过严而使辅助 loss 静默为空。原有 anchor、projection、JSD finite 检查行为不变。

为可观测性，`teacher_analysis` 可在每一步从 regenerate 和 mid route 各抽取一条最低 `R_Ci` 候选。EMA teacher 在独立 privileged prompt 中输出 JSON diagnosis：`failure_mode`、`missing_evidence`、`distractor_evidence`、`correction_focus`。driver 将其写入 checkpoint 根目录的 `teacher_diagnoses.jsonl`，记录 route、`R_Ci`、IoU 向量、student caption 和诊断文本；主标量日志只记录 `opsd/teacher_analysis_count`。诊断严格不进入 student prompt、teacher caption target、模型 checkpoint 或推理输出。该 pass 会增加一次小型 teacher rollout，设置 `worker.opsd.teacher_analysis.enabled=false` 可关闭。

当 captioner 和 segmenter 都启用时，trainer 不分别 optimizer step，而是：

```text
route-replacement: high GRPO + low CE + mid JSD
B: safe 全部原始 GRPO + low CE + mid JSD
辅助 CE/JSD 均按候选比例归一化，caption 侧再乘 caption_loss_weight=0.5
全部 route 的 localization GRPO，再乘 localization_loss_weight=0.5
clip grad norm -> one optimizer.step()
optimizer.step 后原地执行 EMA shard 更新；当 `ema_teacher.decay=1.0` 时，更新为恒等映射，teacher
保持 worker 初始化时从初始 SAMTok actor 复制的参数。
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
| `requirements.txt` | CUDA/PyTorch 之外的核心依赖；包括 VQ-SAM2/RefCOCO 转换所需的 Hydra、iopath、COCO RLE、COCO caption 评价和 torchvision；NumPy 限制在 2 以下以兼容当前 W&B，Transformers 锁定 `4.54-4.57`，vLLM `>=0.8` |
| `Makefile` | 上游开发命令 |
| `tests/test_opsd_core.py` / `tests/test_tokenizer.py` | 无 GPU 单元测试；分别覆盖 OPSD 核心契约，以及 Qwen3-VL 复合 processor 的自动加载回退 |

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
| `workers/actor/dp_actor.py` | log-prob 前向、动态 micro-batch、PPO loss、独立 caption anchor KL、梯度累积和 optimizer step |
| `workers/critic/dp_critic.py` | GAE/PPO 可选 value model；GRPO 主配置通常不启用 critic |
| `workers/rollout/vllm_rollout_spmd.py` | SPMD vLLM engine、采样参数、视觉输入和 response tensor 构造 |
| `workers/sharding_manager/fsdp_vllm.py` | FSDP 参数与 vLLM engine 同步/offload |
| `workers/sharding_manager/fsdp_ulysses.py` | sequence parallel 数据切分/还原 |
| `workers/reward/function.py` | 动态加载 sequential/batch 自定义 reward 并写 token-level score |
| `workers/opsd/config.py` | pixel IoU、路由、caption safety、EMA teacher、regenerate 与 distillation 配置及边界校验 |
| `workers/opsd/distillation.py` | response-token 分块的 checkpointed generalized-JSD、teacher 置信度权重、caption 分割 special-token vocab 屏蔽和 distillation metrics |
| `workers/opsd/mask_iou.py` | 严格 token 解析、原始 GT 转换、批量 mask 解码、尺寸恢复和像素 IoU |
| `workers/opsd/routing.py` | `R_Ci` 聚合、三路由边界、caption 特殊 token/JSON/长度安全检查、原始 GRPO 启用判定、packed mask context、GT/reconstruction teacher crop 构造、route 权重与泄漏过滤 |
| `models/monkey_patch.py` | 为多种 HF MLLM 注册 flash attention 和混合多模态 forward |
| `models/transformers/*.py` | Qwen2/3-VL、Qwen3.5、Gemma4 的 RoPE、embedding 与 forward 适配 |
| `single_controller/` | Ray worker、worker group、注册装饰器、资源/dispatch 管理 |
| `utils/dataset.py` | 本项目数据 schema、图像/视频处理、双 prompt 构建和过滤 |
| `utils/dataset_old.py` | 上游/旧 dataset，仅供回溯 |
| `utils/tokenizer.py` | 加载 tokenizer 与复合多模态 processor；Qwen3-VL 自动加载退化时按模型配置显式回退到 `Qwen3VLProcessor` |
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
| `qwen3vl_4b_refcoco10k_volcengine.sh` | 火山引擎单节点 8 卡 RefCOCO 10k OPSD 入口；校验服务器路径/GPU 与 parquet 图像路径，隔离外部 Ray 集群，使用本地短 `/tmp` 目录承载 Ray session/object spill，并将训练日志与 checkpoint 持久化 |
| `config.yaml` | CycleGRPO 的 data/algorithm/worker/reward/trainer 配置 |
| `format_prompt/non_thinking.jinja` | 原样输出 prompt；主入口使用 |
| `format_prompt/r1v.jinja` | 旧的 think/answer 包装模板 |
| `reward_function/text2mask.py` | 图像 mask、bbox、视频时间段、GCG/PSG/no-target 的总奖励路由 |
| `reward_function/tg_reward.py` | temporal grounding 可组合奖励库 |
| `reward_function/llm_judge_reward.py` | 可选外部 vLLM caption judge 客户端，不属于无外部 judge 的主闭环 |

`projects/eval/qwen3vl_4b_volcengine.sh` 是评测编排入口，支持 FSDP actor 导出及 RefCOCO、GroundingSuite、DLC-Bench 的服务器路径、Conda/Ray 环境隔离和输出目录约定。

`projects/rl/datasets/` 全部是离线数据工具，不在 trainer 内自动运行：

- `prepare_dw_rl_dataset.py` / `prepare_dw_single_rl_dataset.py`：DenseWorld 多目标/单目标转 RL parquet，构造区域叠加图、caption/seg prompt 和 mask token。
- `prepare_grefcoco_cycle_dataset.py`：从 gRefCOCO `train` 按 seed 分层抽取 single/multi positive 与 `ann_id=[-1]` no-target 表达；正样本合并多个 COCO instance mask 并编码为 `grefcoco_cycle`，no-target 保留为 `gres_no_target`，写出正样本、no-target、合并训练 parquet 与类别清单。gRefCOCO 不包含 part mask，清单明确记录 `part_instance=0`。
- `prepare_paco_lvis_part_cycle_dataset.py`：从 PACO-LVIS train 的 `id != obj_ann_id` annotation 确定性抽取真 part mask，限制每图一个 part，编码为 `paco_part_cycle` parquet 与 part-category manifest。
- `prepare_cocostuff_cycle_dataset.py`：从 COCO-Stuff 官方 stuffthingmaps 的真 Stuff 类别区域构造 `cocostuff_cycle` parquet；仅接受 PNG 值 91..181，并记录区域面积范围和类别直方图。
- `prepare_balanced_cyclegrpo_dataset.py`：验证各 source 数量后固定抽取 Single/Multi/Stuff/Part/no-target 的 `7k/5k/4k/2k/2k`，随机打散为一个 20k parquet 和可复现实验 manifest。
- `prepare_refcoco_rl_dataset.py`：标准 RefCOCO train split 转固定数量的单目标 CycleGRPO parquet；以 VQ-SAM2 编码 mask token，并保留原始 COCO RLE 供训练时真实 IoU 使用。
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

### 5.3.1 `projects/eval/`：火山引擎离线评测编排

| 文件 | 职责 |
|---|---|
| `qwen3vl_4b_volcengine.sh` | export-only FSDP actor 转 HF safetensors，并顺序启动标准 RefCOCO、GroundingSuite mask GIoU 与 DLC-Bench prediction 生成；设置服务器路径、Conda、缓存和 Ray 地址隔离 |

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
| `refcoco/` | 标准 RefCOCO 的 `instances.json`/`refs(unc).p` 多 GPU 分片推理和 cIoU/mIoU 汇总；不依赖 Detectron2 或内部数据目录 |
| `groundingsuite/` | Qwen3-VL 推理、按 task 分片和自动合并；支持显式 data root 与可选 COCO 图像根 |
| `gcg/` | 生成 interleaved text-mask，解码 mask 并保存 RLE/文本供官方 GCG 指标；数据根需替换 |
| `gar/` | VQA 和 detailed caption 两个推理入口；`gar_vqa_metrics.py` 汇总总体与属性类别准确率 |
| `dlc_bench/` | 多后端 caption inference、裁剪/区域输入、judge server、GPT-with-image/Llama-without-image 评测和绘图；Qwen3-VL 推理使用训练同构的正向 caption prompt 与 192-token 上限 |
| `bbox/` | Qwen2.5/3/3.5、InternVL、Gemma、Llama 的 bbox 输出泛化；解析 `[x1,y1,x2,y2]` 并按 0-1000 坐标还原 |

评测脚本通常直接加载 Hugging Face checkpoint 和 mask tokenizer 权重，不经过 `verl` trainer。训练的 `global_step_*/actor` 是 world-size 相关 FSDP shard，不能直接传给 `from_pretrained`；先通过火山引擎评测入口的 export-only worker 导出 safetensors。评测推理不需要、也不应连接训练 Ray cluster。DLC-Bench 的模型推理与外部语言 judge 分离，前者可离线运行，后者需要单独配置凭据。

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
11. **OPSD dataclass 默认关闭，项目 YAML 显式开启。** 原始 SAMTok 消融设 `worker.opsd.enabled=false`；仅真实 IoU 的 CycleGRPO 设 `opsd.enabled=true`、`pixel_iou.enabled=true`、`routing.enabled=false`、`ema_teacher.enabled=false`，并将 C/C2 anchor KL 与 caption safety 关闭；完整版本保持主 YAML 默认。
12. **privileged distillation 第一版要求 `actor.ulysses_size=1`。** response 会裁到当前 micro-batch 的最大有效长度；完整词表 JSD 以 response-token chunk 加 checkpoint 计算，`distillation.token_chunk_size` 只在 CUDA 峰值显存与 softmax 重算时间间取舍。其他 sequence-parallel 配置会在启动时显式报错。
13. **EMA checkpoint 位于 `actor/ema_teacher/`。** resume 优先恢复完整 teacher shard；旧 checkpoint 缺失 teacher 时从已恢复 actor 初始化，frozen reference policy 始终保持 cold-start anchor。`decay=1.0` 时这个 shard 是启动时的 SAMTok teacher；不能用旧 EMA 实验的 checkpoint 启动新的固定-teacher 消融。
14. **teacher diagnosis 文件含特权信息。** `teacher_diagnoses.jsonl` 仅用于受控训练调试；公开日志、共享实验产物或发布 checkpoint 前应删除该文件，或关闭 `teacher_analysis`。
15. **火山引擎注入的 Ray 集群与项目环境不兼容。** 当前平台集群使用 Python 3.12 / Ray 2.53，而项目 Conda 环境使用 Python 3.10 / Ray 2.56；服务器入口必须清除继承的 `RAY_ADDRESS`，由当前解释器启动本地单节点 Ray。不要仅降级 Ray 而保留不同 Python 版本。Ray 的 `RAY_TMPDIR` 不能直接使用仓库长路径，否则 `session_*/sockets/plasma_store` 会超过 Linux `AF_UNIX` 的 107 字节限制；它也不能链接到使用率不低于 95% 的 workspace。入口固定使用短的真实本地 `/tmp/cgrpo-ray-<uid>` 目录，并在启动时检查临时盘利用率。
16. **RefCOCO parquet 的图像路径必须与当前服务器一致。** `images` 保存的是绝对路径；跨服务器复制 parquet 后必须重新导出或修复该列。火山引擎入口在初始化 Ray/FSDP/vLLM 前逐条验证 `images`，避免模型全部加载后才由 DataLoader 抛出 `FileNotFoundError`。
17. **Qwen3-VL checkpoint 必须使用复合 processor。** 自定义导出的 checkpoint 可能缺少让 `AutoProcessor` 识别 `Qwen3VLProcessor` 的元数据；loader 会根据 `config.json` 的 `model_type=qwen3_vl` 显式回退。若 `config.json` 也缺失或模型类型错误，必须先修正 checkpoint 元数据，不能用 tokenizer 或 image processor 代替，否则多模态 prompt 无法展开。
18. **FSDP checkpoint 不是可直接评测的 HF 模型。** `actor/huggingface/` 仅保存 config/generation config/processor；必须使用与保存 world size 相同的 export-only FSDP worker 恢复 shard 后导出。不要把原 cold-start `MODEL_PATH` 当作训练后模型传给评测脚本。
19. **caption safety 是当前 OPSD 的稳定化消融。** 它在 IoU 路由之后排除特殊 token、`mask_2d` JSON 和超长 caption 对原始 GRPO/mid JSD 的影响，并把它们导向 regenerate；这不改变论文的单 actor 双任务设计、privileged prompt 或 JSD 公式。比较该消融与历史实验时，必须同时报告 `CAPTION_MAX_RESPONSE_LENGTH` 和安全指标，不能仅比较最终 benchmark 分数。
20. **B 保留原始 GRPO 是另一项受控消融。** `PRESERVE_ORIGINAL_GRPO=true` 使低/中路由的 teacher CE/JSD 成为额外梯度，而非替代原 CycleGRPO caption 梯度；这会改变 caption 梯度总量和与 teacher 的相对权重，不能与 route-replacement 结果直接混合。必须检查 `caption_original_grpo_active_rate` 是否接近 `caption_safe_rate`，否则说明安全门控或 batch 组合没有按预期生效。
21. **C 当前同时处理 special-token 支持集、reference anchor、teacher 特权信息形态和共享梯度冲突。** JSD 屏蔽和 caption anchor KL 能阻止特权 token 分布写入 caption、并将安全 caption 拉回 frozen SAMTok。C2 保留 GT/reconstruction 的诊断信息，但仅以全图、GT crop 和 reconstruction crop 传给 teacher，不把 IoU、几何或 raw mask 文本写进 teacher prompt；student 不会看到这些图。为控制已观察到的纯 CycleGRPO text-to-mask 遗忘，C2 以 segmentation anchor KL 约束全部 cycle localization response，并可用非对称梯度投影移除 caption-side gradient 中与 localization gradient 冲突的分量；两者都不把 `seg_answer` 作为 student CE target，保持单 actor 的 cycle-only 训练信号。投影会额外保留一份本 rank 的 caption gradient，因此增加约一个 FSDP gradient shard 的显存；系数和冲突率必须通过 10-step RefCOCO/GroundingSuite 消融验证。屏蔽词表的实现不得把 logits 设为 `-inf` 后直接参与 entropy/JSD；必须保留有限 log-probability，且任何非有限 JSD 或 actor gradient 都必须 fail-fast，不能静默跳过 optimizer step。

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

### 2026-08-01 - 增加 GroundingSuite 类型均衡 20k 图像/mask 训练数据

- 代码：新增 `projects/rl/datasets/prepare_paco_lvis_part_cycle_dataset.py`、`projects/rl/datasets/prepare_cocostuff_cycle_dataset.py` 和 `projects/rl/datasets/prepare_balanced_cyclegrpo_dataset.py`；修改 `verl/trainer/ray_trainer.py`、`verl/workers/reward/function.py` 与 `projects/rl/reward_function/text2mask.py`。
- 文档：新增第 2.4 节，并更新第 3.3、3.5、5.3 节的 source/转换器契约。
- 行为：新增 PACO-LVIS 真 Part、COCO-Stuff 真 Stuff 以及确定性五路混合器。20k 默认比例为 RefCOCO Single 7k、gRefCOCO Multi 5k、COCO-Stuff 4k、PACO Part 2k、gRefCOCO no-target 2k；新增 `cocostuff_cycle` 与 `paco_part_cycle` 被明确接入 caption/localization cycle batch、pixel-IoU metadata 和 text2mask reward。PACO parent object 与 COCO-Stuff thing/void 像素均被排除。该数据消融只使用图像、目标 mask 和其 SAMTok code，不把 PACO/COCO-Stuff 类别名称、referring expression 或人工 caption 写入正样本 prompt。
- 验证：执行新增 Python 的无缓存语法编译、`bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`、source 白名单静态检查与 `git diff --check`；本机没有服务器 PACO/COCO-Stuff/gRefCOCO/RefCOCO 数据、CUDA 或 VQ-SAM2 权重，未执行 20k token 编码和 8-GPU smoke training。

### 2026-08-01 - 移除 gRefCOCO 正样本残留的 referring expression

- 代码：修改 `projects/rl/datasets/prepare_grefcoco_cycle_dataset.py`。
- 文档：修正第 2.4 节正样本数据契约。
- 行为：gRefCOCO single/multi 正样本现在将 `cap_answer=None`，只保留图像、union target mask、SAMTok mask code 与由 mask 构造的 prompt；`gres_no_target` 保留表达式，因为 null-grounding 的拒识奖励必须有待判断的 query。
- 验证：执行该转换器的 Python 语法编译与 `git diff --check`；本机未运行服务器侧 VQ-SAM2 编码。

### 2026-08-01 - 参数化真实 pixel-IoU 纯 CycleGRPO 对照入口

- 代码：修改 `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`。
- 文档：更新第 2.2 与第 6 节的 HTG/真实 pixel-IoU 消融边界。
- 行为：入口新增 `PIXEL_IOU_ENABLED`、`ROUTING_ENABLED`、caption safety、EMA teacher 与 teacher analysis 的环境变量，默认均保持现有 C2 主路径。设置 `OPSD_ENABLED=true`、`PIXEL_IOU_ENABLED=true`、`ROUTING_ENABLED=false` 可保留训练时 SAMTok 解码像素 IoU，同时使全部 cycle caption 回到原始 GRPO，跳过 teacher 创建、regenerate CE、privileged JSD 和诊断；入口拒绝无 pixel-IoU/EMA teacher 的三路由组合。
- 验证：执行 `bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 与 `git diff --check`；本机没有服务器 Ray/FSDP/vLLM、CUDA、模型或数据，未执行 8-GPU smoke training。

### 2026-07-31 - 支持 gRefCOCO single/multi/no-target CycleGRPO 训练集

- 代码：新增 `projects/rl/datasets/prepare_grefcoco_cycle_dataset.py`；修改 `verl/trainer/ray_trainer.py`、`verl/workers/reward/function.py` 与 `projects/rl/reward_function/text2mask.py`。
- 文档：更新第 3.3、3.5、5.2 节的 source 和数据转换契约。
- 行为：新增 `grefcoco_cycle` 主 cycle source，进入原有 caption-to-localization rollout、真实 pixel-IoU、OPSD route 与 CycleGRPO reward；转换器从 gRefCOCO train 的单实例和多实例正样本构造 union mask、编码 SAMTok token 与原始 RLE。`ann_id=[-1]` 记录不伪造空 mask，而是写为 `gres_no_target`，继续使用现有 no-target reward。转换器按默认 `4.5k single + 4.5k multi + 1k no-target` 输出三个 parquet 和 manifest。gRefCOCO 基于 COCO instance masks，不能提供 GroundingSuite Part 类的真实 part masks，manifest 因而固定记录 `part_instance=0`；需要 Part 覆盖时须额外混入 part-aware 数据。
- 验证：对新增/修改 Python 源执行无缓存语法编译并运行 `git diff --check`；本机没有 PyTorch、datasets、Hydra、SAMTok 权重、COCO/gRefCOCO 数据或 CUDA，未运行 token 编码与 8-GPU train smoke test。

### 2026-07-31 - 高置信 cycle-evidence teacher 辅助更新

- 代码：修改 `projects/rl/config.yaml`、`projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`、`verl/workers/opsd/config.py`、`verl/trainer/ray_trainer.py` 与 `tests/test_opsd_core.py`。
- 文档：更新第 1、2.2、3.6 节。
- 行为：新增 `worker.opsd.teacher_confidence`。通用 YAML 默认关闭以保持历史 all-low/mid 辅助更新，火山引擎入口默认开启。开启时，regenerate 候选在原有安全和绝对 IoU 改善 `>=0.05` 之后，还要求 greedy 验证的 teacher `R_Ci>=0.65` 且归一化改善 `>=0.30`，才形成 CE target；mid route 的 privileged JSD 仅保留原 caption `R_Ci>=0.65` 的候选。全部安全 caption 原始 GRPO 与所有 localization GRPO 保持不变，因此未增加 GT `seg_answer` CE、没有脱离单 actor 的循环训练范式。新增 regenerate 验证/高置信候选数、接受率，以及 distillation 路由/高置信数和 score 均值日志。实测 gradient cosine 接近零后，入口默认关闭但保留可选非对称梯度投影。
- 验证：执行 `python3 -m py_compile verl/workers/opsd/config.py verl/trainer/ray_trainer.py tests/test_opsd_core.py`、`bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 与 `git diff --check`，均通过；本机无 `PyYAML`，未执行 YAML 运行时加载，随后 OPSD 单测须使用服务器的训练环境运行。

### 2026-07-31 - 使火山引擎入口可运行原始 HTG 纯 CycleGRPO 对照

- 代码：修改 `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`。
- 文档：更新第 2.2 节。
- 行为：新增 `OPSD_ENABLED`，默认 `true` 保持真实 pixel-IoU OPSD 路径。显式设为 `false` 时，入口将 `worker.opsd.enabled=false` 传给 trainer；不创建 pixel decoder、EMA teacher 或 teacher auxiliary update，训练使用历史 FSDP `mask_token_accuracy` 的 HTG score 和纯 CycleGRPO。其余数据、模型、batch、rollout、学习率、冻结 vision tower 与 checkpoint 周期不变，适合作为“仅改变 reward representation”的 10-step 对照。
- 验证：执行 shell 语法检查与 `git diff --check`；本机没有 Ray/FSDP/vLLM、CUDA 或服务器数据，未执行 8-GPU training smoke test。

### 2026-07-30 - C2 非对称投影保护 localization 梯度

- 代码：修改 `projects/rl/config.yaml`、`projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`、`verl/workers/opsd/config.py`、`verl/trainer/ray_trainer.py`、`verl/workers/fsdp_workers.py` 与 `tests/test_opsd_core.py`。
- 文档：更新第 2.2、3.6、6 节。
- 行为：新增 `worker.opsd.asymmetric_gradient_projection`，通用 YAML 默认关闭、火山引擎入口默认开启。启用后，FSDP worker 在单次 optimizer step 前保存 caption GRPO/regenerate/JSD/anchor 梯度并清零，再计算 localization 梯度；全 rank all-reduce 得到 gradient dot product。仅当内积为负时，从 caption gradient 中投影掉其反向 localization 分量，segmentation gradient 保持原样，随后相加并沿用原 optimizer/clip/EMA 路径。没有 GT `seg_answer` CE，单 actor 闭环训练与原 reward 保持不变。新增 cosine、冲突率及两个未投影梯度范数日志。
- 验证：配置开关单测、修改模块语法检查、shell 语法检查和 `git diff --check`；本机没有 PyTorch，OPSD 单测和 8-GPU FSDP smoke training 须在服务器执行，并确认 projection 指标有限且没有梯度 NaN。

### 2026-07-30 - 为 C2 增加独立 segmentation anchor KL

- 代码：修改 `projects/rl/config.yaml`、`projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`、`verl/trainer/config.py`、`verl/workers/opsd/config.py`、`verl/workers/actor/{config,dp_actor}.py`、`verl/trainer/ray_trainer.py` 与 `tests/test_opsd_core.py`。
- 文档：更新第 2.2、3.6、6 节。
- 行为：新增 `worker.opsd.segmentation_anchor_kl_coef`。火山引擎 C2 入口默认 `0.05`，在全部 cycle localization response 的原始 response mask 上对 frozen reference 施加独立 token-weighted KL；它不作用于 caption/non-cycle batch，且在 actor 内与通用 `algorithm.kl_coef=0.01` 相加。新增 segmentation anchor 激活数/比例、loss 和系数日志。该约束是针对纯 CycleGRPO 10-step 已观测到的 RefCOCO/Single grounding 遗忘，不是论文原始公式。
- 验证：新增配置系数非负单测；本机执行语法、shell 和差异检查，服务器仍需运行 OPSD 单测与 10-step FSDP/vLLM smoke training，确认 segmentation anchor 指标和 actor grad norm 均有限。

### 2026-07-30 - C2 将 OPSD teacher 特权 mask 改为目标/重建视觉证据

- 代码：修改 `verl/workers/opsd/{__init__,routing}.py`、`verl/utils/dataset.py`、`verl/trainer/ray_trainer.py` 与 `tests/test_opsd_core.py`。
- 文档：更新第 2.2、3.6、5.2、6 节。
- 行为：privileged context 现在保存 packed GT mask；driver 以原图、GT-isolated crop、representative-reconstruction-isolated crop 组成 teacher-only 三图输入。两个 crop 在同一 union box 内以灰色隔离背景显示，并各自限制到最多 `512x512` 等效像素；teacher 仍能根据 student caption 比较 intended/recovered 对象并执行 regenerate、JSD 或诊断，但 prompt 已移除 raw mask token、IoU、位置和面积差异文字。`preprocess_opsd_prompt` 泛化为任意数量图片，student 数据与 teacher FSDP 多图前向保持原有接口。此变更与论文原方法的 token/text 特权输入不同，是为抑制已观察到的全图几何描述漂移而引入的 C2 受控消融。
- 验证：新增 CPU unit test 覆盖 packed GT/reconstruction mask 生成三图证据、灰色隔离背景及 prompt 不含 mask token/IoU；本机运行语法和差异检查，服务器仍需运行 OPSD 单测与 10-step FSDP/vLLM smoke training。

### 2026-07-30 - 修复 C 屏蔽 special-token JSD 的 NaN 与静默跳步

- 代码：修改 `verl/workers/opsd/distillation.py`、`verl/workers/actor/dp_actor.py` 与 `tests/test_opsd_core.py`。
- 文档：更新第 3.6、6 节。
- 行为：C 的 vocab mask 改为写入 float32 有限最小值而非 `-inf`，因此被屏蔽 token 在 softmax 中仍为零概率和零 student gradient，但不会在 entropy/JSD 中触发 `0 * -inf`。JSD loss/metric 非有限时立即抛 `FloatingPointError`；actor 的全局 gradient norm 非有限时清理梯度后立即失败，不再只打印并跳过 optimizer step。此前 C 实验中每 step 的 `actor.grad_norm=NaN` 表示 actor 没有发生有效更新，相关 checkpoint 不可用于算法效果归因。
- 验证：扩展 OPSD unit test，覆盖 blocked logits 下 loss、metrics 和全部 student gradient 均为 finite，以及 blocked gradient 为零；本机执行该单测和语法/差异检查，服务器仍需完成 10-step FSDP/vLLM smoke training，确认 `actor.grad_norm` 为有限值。

### 2026-07-29 - C 第一部分阻断 caption 分割词表泄漏并新增安全 route anchor KL

- 代码：修改 `projects/rl/config.yaml`、`projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`、`verl/trainer/config.py`、`verl/workers/actor/{config,dp_actor}.py`、`verl/workers/fsdp_workers.py`、`verl/workers/opsd/{__init__,config,distillation}.py` 与 `tests/test_opsd_core.py`。
- 文档：更新第 2.2、3.6、5.2、6 节。
- 行为：C 默认在 privileged JSD 的 teacher/student softmax 前屏蔽 tokenizer 实际词表中的 `mt_*` 和 `object_ref_*` special token；其概率及 student JSD 梯度为零。新增 `worker.opsd.caption_anchor_kl_coef` 和 `caption_anchor_kl_all_safe_routes`，火山引擎入口默认 `0.05/true`，使用全部安全 cycle caption 的原始 response mask 计算独立 frozen-reference KL。PPO 的 B route mask、标准 algorithm KL、teacher raw mask prompt 和 GT crop 计划保持原样。
- 验证：执行修改文件的语法检查、shell 语法检查与 `git diff --check`；新增单元测试覆盖 special-token vocab 发现、被屏蔽 logit 不改变 JSD、且其 student gradient 为零。本机缺少 `torch`，单测与 GPU/Ray/vLLM 10-step smoke training 须在服务器运行。

### 2026-07-29 - B 实验保留安全 caption 的原始 CycleGRPO

- 代码：修改 `projects/rl/config.yaml`、`projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`、`verl/trainer/ray_trainer.py`、`verl/workers/opsd/{__init__,config,routing}.py` 与 `tests/test_opsd_core.py`。
- 文档：更新第 1、2.2、3.3、3.6、5.2、6 节。
- 行为：新增 `routing.preserve_original_grpo`。通用 YAML 保持 `false` 以兼容 route-replacement；火山引擎入口默认以 `PRESERVE_ORIGINAL_GRPO=true` 开启 B 模式，使全部安全 caption 保留原始 CycleGRPO GRPO，low regenerate CE 与 mid JSD 在同一次 actor optimizer step 中附加累积。不安全 caption 继续被 A 层门控排除。新增原始 GRPO 激活数/比例指标。
- 验证：执行 `python3 -m py_compile verl/workers/opsd/config.py verl/workers/opsd/routing.py verl/workers/opsd/__init__.py verl/trainer/ray_trainer.py tests/test_opsd_core.py`、`bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 与 `git diff --check`。本机缺少 `torch`，`python3 -m unittest tests.test_opsd_core` 不能导入；服务器须运行该单测和 10-step smoke training。

### 2026-07-29 - 增加 OPSD caption 安全路由第一层稳定化

- 代码：修改 `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`、`projects/rl/config.yaml`、`projects/rl/reward_function/text2mask.py`、`verl/trainer/ray_trainer.py`、`verl/workers/opsd/{__init__,config,routing}.py` 与 `tests/test_opsd_core.py`。
- 文档：更新第 2.2、3.3、3.5、3.6、5.2、6 节。
- 行为：服务器训练入口默认将 caption rollout 限制为 256 token，并使 OPSD safety 使用相同阈值。driver 在像素 IoU 路由后检查真实 response：非终止 special token、`mask_2d` JSON 或超长输出会记录原因并强制进入 regenerate，不参与 caption GRPO/JSD；localization 更新保持不变。caption reward 的 bbox/CJK gate 之外新增 special-token/JSON gate，指标改为语义准确的两个独立名称。
- 验证：执行 `python3 -m py_compile verl/workers/opsd/config.py verl/workers/opsd/routing.py verl/workers/opsd/__init__.py verl/trainer/ray_trainer.py projects/rl/reward_function/text2mask.py tests/test_opsd_core.py`、`bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 与 `git diff --check`。`python3 -m unittest tests.test_opsd_core` 在本机因缺少 `torch` 未能导入；GPU/Ray/vLLM 端到端训练和该单测须在火山引擎服务器环境运行。

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

### 2026-07-19 - 增加 teacher 特权诊断日志

- 代码：修改 `verl/workers/opsd/config.py`、`routing.py`、`verl/trainer/ray_trainer.py`、`projects/rl/config.yaml` 和 `tests/test_opsd_core.py`。
- 文档：更新第 2、3、6 节的诊断配置、输出位置、隔离边界和特权信息处理要求。
- 行为：每步最多对两条低/中路由候选运行独立 EMA teacher diagnosis，并将安全清洗后的文字结论写入 `teacher_diagnoses.jsonl`；不改变 regenerate target、JSD、GRPO 或 student 输入。仅真实 IoU 消融自动跳过该 pass。
- 验证：修改文件通过 `py_compile`；8 个纯 OPSD 测试通过，包含 analysis prompt 契约；FSDP/vLLM runtime 未在本机执行。

### 2026-07-19 - 增加 RefCOCO 20k CycleGRPO 数据转换

- 代码：新增 `projects/rl/datasets/prepare_refcoco_rl_dataset.py`；修改 `verl/trainer/ray_trainer.py`、`projects/rl/reward_function/text2mask.py` 和 `requirements.txt`。
- 文档：更新第 2.3、3.2、3.3、3.5、5.3 节的 RefCOCO 数据契约、`refcoco_cycle` 路由和目录职责。
- 行为：转换工具从标准 RefCOCO train refs 固定采样指定数量，使用当前 VQ-SAM2 生成两个 mask token，写入 RL schema 和原始 COCO RLE；`refcoco_cycle` 与 DenseWorld 一样执行图像 CycleGRPO caption/localization 奖励与真实像素 IoU。依赖清单显式包含转换所需的 Hydra、COCO RLE 和 torchvision。该数据源是对论文 DenseWorld 的受控替代，不是论文原始数据复现。
- 验证：新增脚本、trainer 和 reward 文件通过 `py_compile`；`git diff --check` 通过。当前机器没有服务器侧 RefCOCO、SAMTok/SAM2 权重和 CUDA，未执行 20k 转换或 FSDP/vLLM smoke training。

### 2026-07-20 - 修正 RefCOCO VQ code 形状和 SAM2 依赖

- 代码：修改 `projects/rl/datasets/prepare_refcoco_rl_dataset.py` 和 `requirements.txt`。
- 文档：更新第 2.3 节和根目录依赖职责说明。
- 行为：转换器现在将单目标 VQ-SAM2 返回的 `(1, 1, 2)` code 张量校验后展平为两个 SAMTok code；元素数不是两个时仍明确报错。依赖清单加入 SAM2 Hiera backbone 所需的 `iopath>=0.1.10`。
- 验证：`python3 -m py_compile projects/rl/datasets/prepare_refcoco_rl_dataset.py` 和 `git diff --check` 通过；服务器实际输出确认此前 code 张量为 `[[73, 5]]`，即单目标的有效两层 code。本机缺少 PyTorch 和服务器侧 GPU/权重，未运行转换。

### 2026-07-20 - 补全奖励模块运行时依赖

- 代码：修改 `requirements.txt`。
- 文档：更新根目录依赖职责说明。
- 行为：显式安装 `text2mask.py` 导入 CIDER 所需的 `pycocoevalcap`；将 NumPy 约束为 `<2`，避免当前 W&B 版本在导入时访问已删除的 `np.float_`。
- 验证：服务器训练初始化已确认缺失 `pycocoevalcap`，并以 NumPy 2.1.3 复现 W&B 导入错误；未在本机安装完整 CUDA/Ray 依赖。

### 2026-07-20 - 延迟初始化外部 caption judge

- 代码：修改 `projects/rl/reward_function/text2mask.py` 和 `projects/rl/reward_function/llm_judge_reward.py`。
- 文档：更新奖励 source 表。
- 行为：仅当 reward batch 包含 `dam_captioning` 或 `tg_captioning` 时才创建 OpenAI-compatible judge client，且同一 reward actor 只创建一次。纯 `refcoco_cycle` 训练不再依赖 HTTP/SOCKS proxy、judge endpoint 或 `socksio`。
- 验证：`python3 -m py_compile` 和 `git diff --check`；服务器在纯 RefCOCO 数据加载期间复现了导入阶段创建 judge client 后缺失 `socksio` 的失败。未在本机执行 GPU/Ray 训练。

### 2026-07-20 - 在 rollout 侧兼容旧版 vLLM 的 Qwen3-VL 配置访问

- 代码：修改 `verl/workers/rollout/vllm_rollout_spmd.py`。
- 文档：更新第 3.3 节 rollout 职责。
- 行为：在构建 vLLM LLM 前，为 `Qwen3VLConfig` 缺失的顶层语言模型字段提供从 `text_config` 读取/写入的 property。旧版 vLLM generic Transformers fallback 可访问 `vocab_size`、层数和 attention 配置；即使 checkpoint `config.json` 曾手工写入这些顶层字段，Transformers 构造配置时也会回写到 `text_config` 而不会报只读属性错误。原生支持 Qwen3-VL 的新版 vLLM 不受影响。
- 验证：`python3 -m py_compile` 和 `git diff --check`；服务器使用 vLLM 0.8.3 时依次复现缺失顶层 `vocab_size` 和 `num_hidden_layers`，以及手工写入 `vocab_size` 后的只读属性错误，字段均存在于 Qwen3-VL `text_config`。未在本机运行 vLLM/GPU smoke test。

### 2026-07-22 - 明确 AGENTS.md 的变更日志维护要求

- 代码：未修改训练、评测或数据处理代码；修改根目录 `AGENTS.md`。
- 文档：明确每次代码相关修改前必须阅读完整 `code.md` 及其最新变更日志，修改完成后必须基于既有历史追加新的日志条目；保留现有模块清单和代码架构说明。
- 行为：后续代码、配置、数据转换、训练和评测逻辑的变更流程统一受 `code.md` 记录约束，避免遗漏行为影响和验证结果。
- 验证：已检查 `code.md` 第 5 节模块清单、第 8 节变更日志、`Agent.md` 维护规则及更新后的 `AGENTS.md`；本次未运行代码测试，因为没有运行时代码变更。

### 2026-07-22 - 增加火山引擎 RefCOCO 10k 八卡训练入口

- 代码：新增 `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`。
- 文档：更新第 2.2、3.1、5.3、6 节的服务器入口、绝对路径、日志位置和 Ray 版本隔离要求。
- 行为：新增当前服务器可直接运行的单节点 8 卡入口，默认读取 RefCOCO 10k parquet 和 Qwen3-VL-4B-SAMTok 权重；使用项目 Python 3.10 环境，清除平台 Python 3.12 / Ray 2.53 的注入地址，让 trainer 创建 Ray 2.56 本地集群；全部运行日志和 checkpoint 写入仓库 `logs/refcoco10k_opsd/`。训练仍保持 `G=6`、`K=6`、batch 128、1 epoch 和 OPSD 三路由，不改变论文/当前算法路径。
- 验证：执行 shell 语法检查、路径与参数静态核对和 `git diff --check`；本机没有服务器挂载路径、CUDA、Ray/vLLM 环境及 8 张 GPU，未执行服务器训练 smoke test。

### 2026-07-23 - 修复火山引擎 Ray Unix socket 路径过长

- 代码：修改 `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`。
- 文档：更新第 2.2、5.3、6 节的 Ray 临时目录、日志落盘位置和 Unix socket 路径限制。
- 行为：不再把长仓库路径直接设为 `RAY_TMPDIR`；入口默认创建并复用 `/tmp/cgrpo-<uid>` 到仓库 `logs/refcoco10k_opsd/` 的符号链接，使 Ray 的 `plasma_store` socket 路径保持在 107 字节以内，同时实际 Ray session 日志仍保存在仓库。入口会拒绝过长、非绝对、非符号链接或指向其他目录的短路径，避免日志误写。
- 验证：执行 `bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`、短路径符号链接静态核对和 `git diff --check`；修复针对服务器已复现的 `validate_socket_filename failed`，本机没有服务器挂载路径、Ray/CUDA 环境及 8 张 GPU，未执行训练 smoke test。

### 2026-07-23 - 修复 Qwen3-VL 复合 processor 加载

- 代码：修改 `verl/utils/tokenizer.py`；新增 `tests/test_tokenizer.py`。
- 文档：更新第 3.1、5.1、5.2、6 节的 processor 加载调用链、模块清单和 checkpoint 元数据约束。
- 行为：当 `AutoProcessor` 对 `model_type=qwen3_vl` checkpoint 只返回 tokenizer、image processor 等非 `ProcessorMixin` 对象时，显式加载 `Qwen3VLProcessor`，避免 RefCOCO 多模态数据过滤阶段因 `processor=None` 调用 `apply_chat_template` 失败；非 Qwen3-VL 模型和已正确加载的复合 processor 保持原行为。自定义 chat template 现在在最终 processor 确定后应用。
- 验证：针对 Qwen3-VL 回退、非 Qwen 保持 `None`、正常复合 processor 不回退三个分支新增无下载 mock 单测；`python -m py_compile verl/utils/tokenizer.py tests/test_tokenizer.py` 和 `git diff --check` 通过。本机运行时缺少 Transformers/pytest，因此未实际执行单测；本机也无服务器 checkpoint、Ray/CUDA 和 8 卡环境，未执行完整训练 smoke test。

### 2026-07-27 - 修复火山引擎 RefCOCO 路径与 Ray 满盘启动失败

- 代码：修改 `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`。
- 文档：更新第 2.2、5.3、6 节的默认 parquet、Ray 临时目录与图像路径数据契约。
- 行为：入口默认使用 `*_workspace_paths.parquet`；在初始化 GPU worker 前扫描 train/val parquet 的 `images` 列并验证每个文件存在。Ray session、object store 和 spill 文件改为真实本地 `/tmp/cgrpo-ray-<uid>` 目录，不再链接到 `RUN_ROOT`；启动前拒绝符号链接或利用率不低于 95% 的 Ray 临时盘。训练日志、W&B、teacher diagnosis 与 checkpoint 继续写入 `RUN_ROOT`。
- 验证：服务器实际修复后的 parquet 已写入并验证 10,000 个图像路径；`df -hT` 确认 workspace 挂载使用率 98%，`/tmp` 使用率 1%。本机执行 shell 语法检查和 `git diff --check`；本机没有火山引擎挂载、Ray/vLLM 与 8 张 GPU，未运行训练 smoke test。

### 2026-07-27 - 修复 RefCOCO cycle 奖励的 IoU 传递

- 代码：修改 `verl/workers/reward/function.py`。
- 文档：更新第 3.3 节的 batch reward 元数据契约。
- 行为：将 `refcoco_cycle` 加入 batch reward manager 的 IoU source 白名单，因此 caption 和 segmentation reward 都会接收 trainer 计算的 `R_Ci`；caption metrics 也记录 `correct_mask`。此前 RefCOCO 在模型完成首个 rollout 后将 `iou_scores=None` 传给 `text2mask.compute_score`，触发 `int * NoneType`。
- 验证：服务器实际训练日志在首个 batch 复现该异常；本机执行该模块语法检查、奖励 source 白名单静态断言和 `git diff --check`。本机没有 Ray/vLLM、模型、数据或 8 张 GPU，未执行端到端训练。

### 2026-07-27 - 修复 OPSD distillation 的缺失导入

- 代码：修改 `verl/workers/fsdp_workers.py`。
- 文档：更新第 3.6 节的 OPSD 策略更新调用链。
- 行为：显式导入 `collections.defaultdict`，使 `accumulate_distillation_gradients` 能创建跨 micro-batch 的 metric 容器。此前 RefCOCO 首步已完成 rollout、IoU reward 和 policy update，但在中路由 distillation 梯度累积时因 `NameError` 退出。
- 验证：服务器日志定位到 `fsdp_workers.py:1696` 的未定义 `defaultdict`；本机执行模块语法检查、导入静态断言和 `git diff --check`。本机没有 Ray/vLLM、模型、数据或 8 张 GPU，未执行端到端训练。

### 2026-07-27 - 修复 EMA teacher FSDP checkpoint 保存

- 代码：修改 `verl/workers/fsdp_workers.py`。
- 文档：更新第 3.6 节的 EMA checkpoint 调用边界。
- 行为：保存 EMA teacher shard 前，若 teacher 参数已 offload 到 CPU，则先将 FSDP 模型移回对应 CUDA compute device；`get_model_state_dict(..., cpu_offload=True)` 和落盘完成后始终重新 offload。此前第 5 个 global step 保存 `global_step_5` 时，FSDP 在 CPU 参数上 unshard，触发 `Expects tensor to be on the compute device cuda:0, was on cpu` 并使全部 Ray worker 退出。
- 验证：服务器 traceback 定位到 `save_checkpoint()` 的 `get_model_state_dict(self.teacher_fsdp_module)`；本机执行模块语法检查、保存路径静态断言和 `git diff --check`。本机没有 Ray/vLLM、FSDP、模型、数据或 8 张 GPU，未执行 checkpoint round-trip。

### 2026-07-28 - 分块计算 OPSD distillation JSD 以降低峰值显存

- 代码：新增 `verl/workers/opsd/distillation.py`；修改 `verl/workers/opsd/__init__.py`、`config.py`、`verl/workers/fsdp_workers.py`、`projects/rl/config.yaml` 和 `tests/test_opsd_core.py`。
- 文档：更新第 3.6 节的 mid-route JSD 计算/显存边界、第 5.2 节 OPSD 模块清单和第 6 节关键注意事项。
- 行为：mid-route 保持原 generalized-JSD、teacher entropy confidence、mask 与样本权重；teacher 统计与 JSD 改为 response-token chunk，JSD 块用 non-reentrant activation checkpoint，避免整段 response 同时持有多份 float32 student/teacher softmax、probability 和 mixture logits。默认 `token_chunk_size=256`，可按显存调小；更小值仅增加 softmax 重算，不改变算法。
- 验证：新增 CPU 单元测试，逐项比较 dense 与 chunked loss、student gradient 和三个 metrics；修改文件执行 `py_compile` 与 `git diff --check`。本机缺少 PyTorch/FSDP、CUDA、模型与 8 卡环境，未执行 FSDP/vLLM smoke training；需在服务器环境运行 `python -m unittest tests/test_opsd_core.py` 后从 checkpoint 恢复训练。

### 2026-07-28 - 关闭验证时直接保存最终 checkpoint

- 代码：修改 `verl/trainer/ray_trainer.py`。
- 文档：更新第 2.2 节火山引擎入口的验证语义。
- 行为：`trainer.val_freq<=0` 现在同时跳过训练结束后的 final validation；训练循环完成后仍会立即执行既有的最终 checkpoint 保存。此前 `val_freq=-1` 虽关闭周期验证，仍在 step 结束后对完整 val dataloader 运行 validation，使 `global_step_78` 在验证完成前无法落盘。
- 验证：执行 `py_compile`、收尾分支静态检查和 `git diff --check`。本机没有 Ray/vLLM、模型、数据或 8 张 GPU，未执行恢复训练；服务器应从 `global_step_75` 恢复，确认生成 `global_step_78` 与 tracker 更新。

### 2026-07-28 - 增加火山引擎 FSDP 导出与离线评测入口

- 代码：新增 `projects/eval/qwen3vl_4b_volcengine.sh`、`evaluation/refcoco/`；修改 trainer/worker/FSDP checkpoint 管理器、GroundingSuite launcher/inference 及 DLC inference。
- 文档：更新第 2.2、5.3、5.6、6 节的 checkpoint 导出、RefCOCO/GroundingSuite/DLC 评测路径和环境边界。
- 行为：`trainer.export_hf_model_path` 触发 export-only worker，跳过 rollout vLLM、reference policy、EMA teacher、SAMTok、optimizer/scheduler/RNG/dataloader state，使用完整 FSDP world-size 恢复 actor model shard 并在 rank 0 导出 safetensors/processor。服务器入口为 RefCOCO、GroundingSuite 和 DLC 传递已配置路径并清除平台 Ray 地址；RefCOCO 以逐句 shard 可恢复输出汇总 cIoU/mIoU；GroundingSuite 推理后合并 JSONL 并运行 mask GIoU metric；DLC action 仅生成 prediction JSON，不调用外部 judge。
- 验证：执行新增/修改 Python 的 `py_compile`、三个 shell 的 `bash -n`、配置/路径静态检查和 `git diff --check`。本机缺少 PyTorch/Ray/FSDP、CUDA、服务器数据和 8 张 GPU，未实际导出或运行 benchmark；服务器先运行 export，再分别运行三个 evaluation action。

### 2026-07-28 - 修复文件路径评测脚本的项目导入根目录

- 代码：修改 `projects/eval/qwen3vl_4b_volcengine.sh`。
- 文档：更新第 2.2 节评测入口的 Python 模块路径约定。
- 行为：入口显式导出 `PYTHONPATH=$REPO_DIR`，并在启动任一 action 前验证 `import projects`；DLC、RefCOCO 和 GroundingSuite 以文件路径运行时都可导入 `projects.transformers.vq_sam2`。此前环境解释器即使正确，`sys.path[0]` 仍是 `evaluation/<benchmark>` 子目录，导致 `ModuleNotFoundError: projects`。
- 验证：执行 shell 语法检查、`PYTHONPATH` 导入静态检查和 `git diff --check`；本机没有完整 PyTorch/Ray/CUDA 环境，未运行模型推理。

### 2026-07-28 - 修复 GroundingSuite COCO 文件名解析与缺图静默跳过

- 代码：修改 `evaluation/groundingsuite/qwen3vl_groundingsuite_infer.py`。
- 文档：更新第 2.2 节 GroundingSuite 的服务器图像路径契约。
- 行为：当 JSONL 使用无前缀的 12 位 COCO image ID 时，评测同时尝试标准 `COCO_train2014_<id>.jpg` 文件名；仅对已存在的图片执行 NAS I/O 重试。真实缺图或无法读取的样本现在立即使 shard 失败，不再每条退避 31.5 秒后跳过且不写预测，从而避免进度停滞、未完成的 JSONL 合并和无效指标。
- 验证：服务器日志复现无前缀 GroundingSuite 路径在已有 RefCOCO `train2014` 目录中找不到、每条等待约 31.5 秒的问题；本机执行 Python 语法检查和 `git diff --check`。本机没有服务器数据、CUDA、Qwen3-VL/SAMTok 权重，未运行 8 卡 GroundingSuite 推理。

### 2026-07-28 - 补全 GroundingSuite 发布数据的资产根目录

- 代码：修改 `evaluation/groundingsuite/qwen3vl_groundingsuite_infer.py`。
- 文档：更新第 2.2 节 GroundingSuite 图像解析根目录说明。
- 行为：除 JSONL 相对路径和外部 COCO `train2014` 外，解析器也检查 GroundingSuite 发布包的 `assets/`、`unlabeled2017/` 及其可能的 `train2014/` 子目录。此前服务器的 `GSEval` 根目录包含这两个资产目录，但 bare image ID 无法被解析器发现，fail-fast 修复因而使全部 shard 立即退出。
- 验证：服务器 8 shard 日志确认修复无前缀 COCO 名称后仍在 `GSEval/assets`/`unlabeled2017` 之外查找，60 秒内全部退出且仅保留上一轮 9 条输出；本机执行 Python 语法检查和 `git diff --check`。本机没有发布数据、CUDA、Qwen3-VL/SAMTok 权重，未运行 8 卡推理。

### 2026-07-28 - 修复 GroundingSuite metric 的预测输入类型

- 代码：修改 `projects/eval/qwen3vl_4b_volcengine.sh`。
- 文档：更新第 2.2 节 GroundingSuite 推理/metric 输出约定。
- 行为：统一入口继续保留合并后的 `groundingsuite_pred.jsonl`，但向 `groundingsuite_metric.py --pred_folder` 传入逐样本 JSON 的 `groundingsuite/` 目录。该 metric 使用 `os.listdir()` 加载目录中的 JSON，不能直接读取合并 JSONL；此前推理完成后必然因 `NotADirectoryError` 退出。
- 验证：服务器已完成 3715/3715 shard 输出并以合并 JSONL 作为 `--pred_folder` 复现 `NotADirectoryError`；本机执行 shell 语法检查和 `git diff --check`。本机没有 GroundingSuite 数据、CUDA 和模型，未运行 metric。

### 2026-07-28 - 冻结初始 SAMTok teacher 的 RefCOCO 10k 受控消融

- 代码：修改 `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`。
- 文档：更新第 2.2、3.6、6 节的 teacher 更新语义、checkpoint 与验证边界。
- 行为：入口默认 `TEACHER_EMA_DECAY=1.0`，EMA 更新系数因而为 0，teacher 保持 FSDP worker 启动时从 `MODEL_PATH` 复制的初始 SAMTok 权重。实验输出改为新的 frozen-teacher 目录且默认不恢复任何 checkpoint；仅显式 `RESUME=true` 才恢复同一冻结实验。仍每 5 step 保存 checkpoint，供释放 GPU 后用标准 RefCOCO 离线评测。可选 `MAX_STEPS=5,10,...` 让训练在每个 checkpoint 停止，以便评测完成后恢复；未启用通用 `trainer.val_freq`，因为它不计算 RefCOCO mask/cIoU。
- 验证：执行 shell 语法检查、对 `decay=1.0` 的 FSDP EMA 公式进行静态核对、执行 `git diff --check`。本机没有服务器的 Ray、FSDP、CUDA、模型和 8 卡，未执行训练或 RefCOCO 评测。

### 2026-07-28 - 修正 DLC 区域描述评测协议并限制生成长度

- 代码：修改 `evaluation/dlc_bench/inference.py`。
- 文档：更新第 2.2、5.6 节的 DLC 推理协议。
- 行为：将原来语义错误的 `Given a detailed description ...` 改为与训练 caption 动词一致的 `Provide a detailed factual description ...`，并显式限制模型只描述 mask 指定区域的可见对象、属性和空间关系，禁止 reasoning、mask token、JSON 及区域外内容。zoom-in 分支明确第二张图是同一目标的放大视图。最大新 token 从 1024 降至 192，以阻断实测中接近长度上限的重复和虚构扩写。此修改只改变 DLC 推理协议，不改变训练；原始 SAMTok 与所有训练 checkpoint 必须在该协议下重新生成 prediction JSON 后才可横向比较。
- 验证：执行该文件的 Python 语法检查和 `git diff --check`。本机没有 DLC 数据、Qwen3-VL/SAMTok 权重或 CUDA，未实际运行生成或 judge。

### 2026-07-29 - 移除 DLC prompt 中的 segmentation 格式触发词

- 代码：修改 `evaluation/dlc_bench/inference.py`。
- 文档：更新第 2.2、5.6 节的 DLC 推理协议。
- 行为：DLC 的正向 caption 指令保留 `Provide a detailed factual description of this region {SEG}.`，移除此前加入的 `mask tokens`、`JSON` 和 `reasoning` 等负向格式词。实际推理中，96/100 个样本将这些词解释为输出格式提示，主动生成 `mask_2d` JSON 而非自然语言描述。zoom-in 图仍被声明为同一 region 的放大视图，192-token 上限保持不变。此前产生的 DLC prediction JSON 无效，必须重新生成。
- 验证：执行该文件的 Python 语法检查和 `git diff --check`。本机没有 DLC 数据、Qwen3-VL/SAMTok 权重或 CUDA，未实际运行生成或 judge。
