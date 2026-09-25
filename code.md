# CycleGRPO 代码文档

> 文档基线：论文 `2607.11581v1`（29 页，2026-07-17）与仓库提交 `586e970`。
> 本文是仓库代码知识库，也是强制维护的变更日志。修改任何 `.py`、`.sh`、`.yaml`、`.jinja`、模型配置或评测逻辑前，必须先读本文；修改完成后，必须同步更新相关章节和末尾的“变更日志”。

## 1. 项目定位

### 隔离 PEGC Refusal Credit 对照（2026-09-07）

`experiments/pegc_ablation_20260902/` 新增了只作用于该探索副本的 `worker.supervised_anchors.refusal_credit`：它对 gRefCOCO no-target 文本执行独立 `No target.` teacher-forcing，并统计每个 prompt 的 6-rollout 全 mask/全空/同质率；根目录主训练入口和主代码实现不受影响。

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

当前扩展在每条 caption 的 `K` 次真实 IoU 均值 `R_Ci` 上执行候选级三路由：`R_Ci<0.5` 进入 EMA teacher regenerate，`0.5<=R_Ci<=0.85` 进入 privileged on-policy distillation，`R_Ci>0.85` 保留 CycleGRPO caption GRPO。通用配置默认维持三路由替换 caption 更新；火山引擎 B 实验显式启用 `routing.preserve_original_grpo=true`，使所有安全 caption 都保留原始 CycleGRPO GRPO，再把 regenerate CE 或 privileged JSD 作为附加梯度。新增 `routing.all_samples_opsd=true` 的 routing 消融会跳过 `R_Ci` 分类，将所有可用 image-cycle caption 统一送入 privileged on-policy distillation，同时仍按 `preserve_original_grpo` 独立保留原始 GRPO；高置信 gate 在该模式下不再按 `R_Ci` 丢弃样本。所有 localization rollout 始终参与 CycleGRPO 更新。

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
| segmentation response 上限 | `256` token | 与历史 20k OPSD 运行相同；localization rollout 默认继承等长上限，而非后加的 32-token 截断 |
| mask 解码模式 | `union` | 默认解码每个 response 的全部合法 group 并取像素 union；设 `MASK_DECODE_MODE=first_mask` 可复现原始单 group 训练解码 |
| localization prompt 模式 | `mixed` | 默认 1:1 交替 RefCOCO/GroundingSuite 模板；设 `LOCALIZATION_PROMPT_MODE=refcoco` 可让所有图像 localization 使用 RefCOCO prompt |
| 两阶段 prompt 模式 | `current` | 设 `CYCLE_PROMPT_MODE=official_source_aware` 可按 RefCOCO/gRefCOCO/PACO/Stuff source 重建官方 caption prompt，并让图像 localization 使用官方长模板；默认不改变当前行为 |
| 内层 rollout `K` | `worker.opsd.localization_rollouts=6` | 已从 trainer 硬编码迁入配置 |
| 路由阈值 | baseline `0.5 / 0.85` | 边界分别为 low: `<low_threshold`、mid: `[low_threshold,high_threshold]`、high: `>high_threshold`；主入口通过 `ROUTING_LOW_THRESHOLD`/`ROUTING_HIGH_THRESHOLD` 覆盖，默认保持 baseline |
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
| 三流 parent batch 默认值 | 正式 7+1 70k 入口为 `main=112, direct=224, DLC-QA=56` | 三条 batch 均可被 7 个训练 rank 等分，保持 `2:4:1` 配额；179 step 近似完整消费 20k/40k/10k 三条流 |
| 火山引擎默认最大步数 | `156`（20k 主入口）；`179`（正式 70k 三流入口） | 分别与 `20000/128` 和 `20000/112` 对齐；可用 `MAX_STEPS` 覆盖，显式设空可恢复完整 epoch |
| epoch | `1` | 与论文一致 |
| GPU | 默认 1 node x 8 GPU；显式 Ray attach 按试验族区分 | 单机仍由 Ray + FSDP + vLLM SPMD 运行；正式 70k 有监督 trial 使用物理 GPU 0--6 的 7 张 Ray 训练卡与 GPU 7 本机 Llama judge。attach 模式也允许 1--7 张 Ray 训练卡作开发 smoke，但必须在 `CUDA_VISIBLE_DEVICES` 外保留一张卡给外部 judge，且所有实际 parent-prompt batch 必须能被训练卡数整除。 |
| vision tower | frozen | shell 覆盖为 `true` |
| caption/segmenter | 都优化 | 最终按 `0.5/0.5` 梯度权重累积 |
| 验证 | checkpoint 后离线 RefCOCO | 入口默认每 5 step 保存 checkpoint，`SAVE_LIMIT` 可限制保留数量；`val_freq=-1`、`val_before_train=false`；通用 trainer validation 不执行 mask reconstruction，不能代替标准 RefCOCO cIoU/mIoU |
| 日志 | file + wandb | shell 强制 `WANDB_MODE=offline`；未安装可选 `wandb` 依赖的服务器应设 `TRAINER_LOGGERS='["file"]'`，避免条件导入后的 `NameError` 并将指标完整写入本地 JSONL |

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
离线评测入口执行 RefCOCO val。入口默认 `MAX_STEPS=156`，用于使 20k/40k/10k 三条流在同一轮内对齐；设置 `MAX_STEPS=5,10,...` 可将训练分段停在这些 checkpoint，
再以 `RESUME=true` 继续同一固定-teacher 实验。100k scaling runner `tools/run_100k_scaling_opsd_8gpu.sh` 使用约 69 GiB 的单个 FSDP actor checkpoint；由于 trainer 在写入新 checkpoint 前清理旧目录，runner 固定 `SAVE_LIMIT=1` 并将周期保存设为 `SAVE_FREQ=25`，避免双 checkpoint 峰值触发容器的 `Errno 28 No space left on device`，同时保留断点续训能力。该 runner 的本地 Llama judge 首次 vLLM 编译可能超过 180 秒，`JUDGE_TIMEOUT_SECONDS` 默认设为 600 秒并传入启动与停止命令。平台会注入
指向 Python 3.12 / Ray 2.53 集群的 `RAY_ADDRESS`，但项目环境是 Python 3.10 / Ray
2.56；独立 70k 命令文件必须用 `$ENV_DIR/bin/ray` 启动和停止本地 head，并先清理旧 head，确保 ray CLI 与 trainer 使用同一解释器环境；默认单机入口会清除继承的 Ray 地址，让 `verl.trainer.main` 创建版本一致的本地单节点
Ray。显式连接平台 Ray 时设置 `MULTINODE_ENABLED=true`、`NNODES=1|2` 和由项目 `$ENV_DIR/bin/ray`
创建的私有 `RAY_ADDRESS`；入口会保留该地址，并在 trainer 启动前验证对应数量的节点和 Ray GPU。纯 20k
controller 使用 `NNODES=2`、`NUM_GPUS=8`、GPU 0--7 全训练的拓扑（16 Ray GPU）；有监督 controller
使用一台 32-GPU 节点、`NNODES=1`、`NUM_GPUS=7`，仅向 Ray 登记物理 GPU 0--6，预留物理 GPU 7
运行本机 Llama，并要求恰有 1 个 Ray 节点和恰好 7 张 Ray GPU。该实验不调度物理 GPU 8--31。为在开发机 smoke 验证外部 judge、Ray attach 和三流数据路径，入口也接受 `LOCAL_JUDGE_ENABLED=true` 的 `NUM_GPUS=1..7`；这不是正式实验拓扑，调用方必须显式设置与该 world size 整除的 main/direct/DLC-QA batch，并在 Ray 中只登记相同数量的 GPU。训练 stdout、W&B、teacher diagnosis 和 checkpoint 写到仓库内
`logs/refcoco10k_opsd/`；Ray session、object store 与 spill 文件写到本地短路径
`/dev/shm/cgrpo-ray-<uid>` 或其他本地数据盘上的短绝对路径（例如 `/data5/ray-<uid>`）。这同时保持 Ray socket 路径不超过 Linux `AF_UNIX` 的 107
字节限制，并避免持久化 workspace 挂载接近满盘时使 Ray 停止创建/溢写对象。入口拒绝
符号链接的 Ray 临时目录及使用率不低于 95% 的临时文件系统，并在创建 GPU/Ray worker
前扫描 parquet 的 `images` 列，验证所有图像路径均存在。除本节记录的 70k 三流 batch/step
对齐默认值外，它不修改论文算法；其他训练超参数和数据路径仍可由环境变量覆盖。

新增的 scaling 纯自监督入口 `tools/train_selfsupervised_40k_teacher_8gpu.sh` 和
`tools/train_selfsupervised_80k_teacher_8gpu.sh` 使用当前服务器
`datasets/cyclegrpo100k_scaling_20260911` 下的 parquet，均为单节点 8 GPU、主 rollout/global
batch `128`、1 个完整 epoch、无 direct/DLC-QA 辅助 loader。两者显式开启 OPSD pixel-IoU、三路由、
caption safety、EMA teacher、teacher confidence 和 teacher analysis，并固定
`TEACHER_EMA_DECAY=1.0` 复现当前 frozen-teacher 配置。no-target 采用严格的
`NO_TARGET_REWARD_MODE=pixel_empty` 二值判定：解码 mask union 为空才记为正确，非空即记为错误，
不额外叠加 `NO_TARGET_NONEMPTY_MASK_PENALTY`（设为 `0.0`）。该模式不使用面积比例，
`NO_TARGET_EMPTY_AREA_TAU=0.0` 明确关闭连续分数；只有显式选择 `pixel_empty_iou` 时才要求正的 tau。
正例拒识或空 mask 则通过
`POSITIVE_EMPTY_MASK_PENALTY=1.0` 单独扣分。40k 入口直接读取
`cyclegrpo_selfsupervised_40k.parquet`；80k 入口首次运行时将该文件与
`direct_supervised_40k.parquet` 按统一 Arrow schema 合并为 80,000 行
`cyclegrpo_selfsupervised_plus_direct_supervised_80k.parquet`，再交给同一主入口训练。合并写入临时
文件后原子改名，重复运行直接复用并校验行数。

这两个入口现在显式使用项目环境的 `${ENV_DIR}/bin/ray start --head`：40k 默认绑定
`127.0.0.1:29679`，80k 默认绑定 `127.0.0.1:29680`，均注册 8 张 GPU，并以
`MULTINODE_ENABLED=true`、`NNODES=1` 和 `RAY_CLUSTER_EXPECTED_GPUS=8` attach 到该本地 head。
训练退出、失败或收到中断信号时，入口执行同一环境的 `ray stop --force` 清理 head；这两个脚本不再依赖
`verl.trainer.main` 隐式创建 Ray，也不连接平台注入的外部 Ray 地址。

四卡 EGCA 实验入口 `tools/train_selfsupervised_20k_egca_4gpu.sh` 使用当前服务器的
`cyclegrpo_20k_raw_seed20260820/cyclegrpo_20k_40_20_25_10_5_seed20260820.parquet`，该文件固定为
20,000 行（19,000 个正样本与 1,000 个 `gres_no_target`）。它默认由主 launcher 创建项目环境的本地 Ray 单节点，显式设置 `MULTINODE_ENABLED=true` 时才由入口预先启动 Ray head；使用
CUDA 0--3、`ROLLOUT_BATCH_SIZE=128`/`ACTOR_GLOBAL_BATCH_SIZE=128` 完整训练 1 epoch，并开启
`OPSD_ENABLED`、`PIXEL_IOU_ENABLED`、动态 `EGCA`、routing、EMA teacher、teacher confidence 与
teacher analysis；direct GRPO、direct mask CE、DLC-QA 均关闭。no-target 使用严格的
`NO_TARGET_REWARD_MODE=pixel_empty` 二值 decoded-union 判定，正样本空 mask/拒识的
`POSITIVE_EMPTY_MASK_PENALTY=1.0` 开启，而独立 no-target 非空 mask 负项保持 `0.0`。该入口
固定 `SAVE_FREQ=5`、`SAVE_LIMIT=2`，训练正常退出后停止 Ray head 并启动 `cuda_keepalive.py`；训练
失败则保留原退出码，不会把失败伪装成完成。主火山入口的 attach 校验允许无 judge 的 1--8 卡本地
Ray 开发/消融运行，正式 8 卡与 7+1 judge 拓扑约束保持不变。
当调用方已经传入可执行的 `PYTHON_BIN` 时，主入口跳过 Conda re-activate，避免某些
外层 `base` shell 的 Conda PATH 栈触发激活器异常；Python/Ray 仍通过显式环境路径校验。

新增的独立 70k 混合训练入口为
`tools/train_supervised_70k_baseline_8gpu.sh` 和
`tools/train_supervised_70k_seca_8gpu.sh`。两者均使用历史 70k 配方：20k raw CycleGRPO
主流、30k RefCOCO direct positive、10k gRefCOCO no-target direct 数据和 10k DLC-QA，
正式 7+1 拓扑下分别以 `112/224/56` 的 main/direct/QA parent batch、179 个 optimizer step 运行：
三条 batch 均可被 7 个 Ray/FSDP 训练 rank 等分，同时保持历史 `2:4:1` 配额比例并近似完整消费
20k/40k/10k 三条 loader；开启
pixel-IoU、pixel-empty 二值 no-target reward、direct GRPO、direct mask CE 和 DLC-QA，
并使用物理 GPU 0--6 作为 Ray/FSDP 训练卡、GPU 7 运行本地 Llama-3.1-8B judge。baseline
wrapper 将 `SECA_ENABLED` 与 `SECA_SELF_SUPERVISED_ENABLED` 都关闭；SECA wrapper 将两者都
开启，使 20k cycle 的 mid-route JSD 和 direct mask CE 同时获得 SECA evidence/token credit，
但不改变 reward 或原始 GRPO。每个 wrapper 都通过
`tools/train_supervised_70k_common_8gpu.sh` 启动独立端口的项目 Ray head、健康检查本地 judge，
并在退出时只清理自己发现的 Ray session；数据路径、模型路径、端口和输出目录可用环境变量覆盖。
基于 SECA wrapper 的四个 routing 阈值入口分别为
`tools/train_supervised_70k_seca_routing_l030_h085_8gpu.sh`、
`tools/train_supervised_70k_seca_routing_l070_h085_8gpu.sh`、
`tools/train_supervised_70k_seca_routing_l050_h065_8gpu.sh` 和
`tools/train_supervised_70k_seca_routing_l050_h100_8gpu.sh`；它们只额外覆盖
`ROUTING_LOW_THRESHOLD`/`ROUTING_HIGH_THRESHOLD`，其余 SECA、70k 数据流、112/224/56 batch、
179 step、7+1 Ray/judge 拓扑和 checkpoint 配置均继承参考 wrapper。四组值依次为
`(0.30,0.85)`、`(0.70,0.85)`、`(0.50,0.65)`、`(0.50,1.00)`；最后一组是 high 基线
`0.85` 增加 `0.20` 后因阈值合法范围 `[0,1]` 截断到 `1.00`，因此实际增量为 `0.15`。
主 RefCOCO launcher 对这两个变量执行 `[0,1]` 与 `low<=high` 校验，再传给
`worker.opsd.routing.low_threshold/high_threshold`，不会再把 routing 阈值硬编码为 `0.5/0.85`。
跨服务器复现实验时，仓库根目录 `README.md` 的 70k 训练部分保留上述四个
routing wrapper，并单独列出 50% baseline 数据量控制实验；README 明确了另一台服务器需要替换的环境、
模型、judge、四路数据和图像路径变量，并要求先执行 routing wrapper 与 baseline wrapper 的
`DRY_RUN` 预检，再按 `7+1` 拓扑顺序运行，避免误用旧的通用训练示例。50% baseline 不属于 routing
阈值矩阵，保持 SECA 关闭，仅将四个训练流固定抽样到 `10000/15000/5000/5000`，默认 `MAX_STEPS=90`。
新增 `tools/train_supervised_70k_baseline_25pct_8gpu.sh` 与
`tools/train_supervised_70k_baseline_50pct_8gpu.sh` 沿用 baseline 的全部模型、算法、batch、
7+1 GPU、judge 与 checkpoint 配置，仅在启动前对四个 parquet 流做固定种子抽样：
cycle 20k 按 `source` 分层取 5k/10k，direct RefCOCO 正例 30k 取 7.5k/15k，
direct gRefCOCO no-target 10k 取 2.5k/5k，DLC-QA 10k 取 2.5k/5k。
DLC-QA JSONL 按 `dam_source_id` 与所选 parquet 行重新配对，25% 子集嵌套于 50% 子集；
缩量文件写到各自 `RUN_ROOT/data`。保留 `112/224/56` 的每步 parent batch，
默认 `MAX_STEPS` 分别按约一轮数据设为 45/90（原 baseline 为 179），因此 warmup、
save cadence 等以 step 计的配置数值不变，但占训练总步数的比例会变化。
这两档是历史 70k 配方的数据量消融，不是论文原始 20k DenseWorld 训练配方。
`tools/train_supervised_70k_baseline_25pct_4gpu_tmp.sh` 是 25% 配方的一步本机 smoke：
GPU 0--2 训练、GPU 3 运行 judge，parent batch 改为可被 3 整除的 `114/228/57`，
`MAX_STEPS=1`、`SAVE_FREQ=1`、`SAVE_LIMIT=1`，结束后不启动 GPU hold。
它只用于验证缩量数据、Ray/judge 和训练链路能否启动及完成一步，不能替代正式 7+1
拓扑的显存与吞吐验证。
50% 正式 wrapper 未新增独立四卡文件；本机 smoke 通过环境变量覆盖其资源拓扑，使用同样的
GPU 0--2 训练、GPU 3 judge、`114/228/57` parent batch、`MAX_STEPS=1`、`SAVE_FREQ=1`、
`SAVE_LIMIT=1` 和 `HOLD_AFTER_EXIT=false`，因此仍保留正式 50% wrapper 的数据抽样与 baseline
SECA 关闭配置，同时不改变其默认 7+1/90-step 入口。
新增的 `tools/train_supervised_70k_seca_routing_l030_h085_4gpu_tmp.sh` 复用完整 70k
SECA 数据流、模型、算法和 routing 配置，仅把 routing 固定为 `low=0.30/high=0.85`，
并采用同样的 GPU 0--2 训练、GPU 3 judge、`114/228/57` parent batch、1 step 和不启动
结束占卡设置；它只用于本机验证该 routing 变体能进入真实 Ray/FSDP/vLLM 训练链路，不能
替代四个正式脚本的 7+1 拓扑或完整训练。
该 smoke 首次运行发现，当前 Transformers 4.57.0 枚举整个
`AutoModelForImageTextToText._model_mapping.keys()` 会导入无关的 Gemma3n 配置，
而服务器的 timm 0.4.12 不提供其所需的 `ImageNetInfo`。FSDP worker 现仅查询实际
`model_config` 对应的映射项，避免无关模型导入；Qwen3-VL 的 AutoModel 选择不变。
正式 25% 与 50% 八卡 wrapper 也在启动 Ray 前检查当前环境是否能解析 Qwen3-VL 的图文
AutoModel 映射，失败时直接返回原始 Python 错误；实际训练仍调用同一个已修复的 FSDP worker。
训练成功或失败后都会请求八卡 GPU hold，避免服务器因显存释放而回收节点；设置
`DRY_RUN=true` 只执行路径、行数和配置预检，不启动 Ray、Llama 或训练。

两套四组 H20 控制器都依赖平台预先创建 Ray cluster，不需要手工 SSH 启动 trainer 或配置 NCCL 网卡。纯自监督使用 `tools/multinode/launch_four_cycle_trials.sh`（其底层为
`launch_four_trials.sh`）：复制 `clusters.tsv.example`，每个非注释 TSV 行填写 `trial_id`、`ray_address`、
`ray_namespace`、`experiment_env`。它要求每个集群恰有两节点、16 张 Ray GPU；四个 20k trial 均为
`128/156/256`，分别比较 `official_source_aware/refcoco` prompt 与 `first_mask/union` 解码，且
`LOCAL_JUDGE_ENABLED=false`，GPU 0--7 全部用于训练。

混合四组实验使用独立的 `tools/multinode/launch_four_supervised_trials.sh`：复制
`supervised_clusters.tsv.example`，每行填写 `trial_id`、`ray_address`、`ray_namespace`、
`head_judge_base_url`（无 DLC-QA 的行填 `-`）和 `experiment_env`。控制器按 env 动态预检 Ray
拓扑：三个 20k 自监督 trial 使用单节点、8 张训练 GPU（8 GPU/trial），不启动 Llama；
70k 三流 trial 使用一台节点、7 张 Ray 训练 GPU，物理 GPU 7 由 node-affine actor 启动本机 Llama，
GPU 8--31 不使用。四个任务分别为 pixel-empty 20k、pixel-empty 70k（direct GRPO+CE+DLC-QA）、
official-bbox no-target 20k 和 official source-aware prompt 的 pixel-empty 20k：

| env | 主要目的 | main/direct/QA batch，step | direct CE | direct GRPO / DLC-QA |
|---|---|---|---|---|
| `supervised_trial_01` | pixel-empty 20k 自监督 | `128/128`，156 | — | 无 direct/CE/DLC-QA |
| `supervised_trial_02` | pixel-empty 70k 有监督 | `112/224/56`，179 | `0.005`，warmup `10--30`，含 no-target | `0.15 / 1.0` |
| `supervised_trial_03` | official-bbox no-target 20k | `128/128`，156 | — | 无 direct/CE/DLC-QA |
| `supervised_trial_04` | official source-aware prompt 20k | `128/128`，156 | — | 无 direct/CE/DLC-QA |

7 个 rank 上，70k batch `112/224/56` 对应每步每 rank `16/32/8` 个 main/direct/QA parent prompt；四卡 smoke 使用
3 个训练 rank 的可整除覆盖 `114/228/57`。20k 任务没有辅助流。
虽然环境中 `THREE_STREAM_2_4_1_ENABLED=false`，这是为了绕过该旧严格模式对 routing/EMA/teacher diagnosis 的互斥检查；
70k parent batch 仍保持 `2:4:1`，并由 mixed controller 显式校验，不代表关闭 v1/v2 诊断监督。两个控制器都以 `set -a` source env，使 trial 数据、运行名和开关传给
`nohup` 的训练子进程；PID 记录实际 training launcher。两者均有 `launch`、`status`、`stop` 和
`--dry-run`；stop 仅停止控制器所启动的 trainer，supervised stop 还关闭其管理的 detached judge actor。

入口以 `set -u` 运行时，未设置 `MAX_STEPS` 会采用正式 70k 默认值 `179`；显式设置为空字符串时不会向 Hydra
传入空位置参数并恢复完整 epoch，设置正整数时附加 `trainer.max_steps=<value>`。

火山引擎离线评测入口是 `projects/eval/qwen3vl_4b_volcengine.sh`。当前服务器默认将
`BASE_DIR` 固定为 `/volume/ybo/xyc`，使用 `/volume/ybo/xyc/envs/cyclegrpo`、
`/volume/ybo/xyc/Qwen3-VL-4B-SAMTok` 与 `global_step_714` 的评测输出目录；默认
`NUM_GPUS=7`，因此评测命令只应暴露 `cuda:0-6`，将 `cuda:7` 留给独立 Llama judge。
所有默认值仍可通过同名环境变量覆盖。训练 checkpoint 中的
`actor/model_world_size_<N>_rank_*.pt` 是 FSDP shard，`actor/huggingface/` 只包含配置和
processor；因此必须先执行 `export` action，并以与 shard 文件名相同的 `NUM_GPUS=N` 拓扑只加载 actor model shard 并导出
标准 safetensors HF 目录。之后 `refcoco`、`groundingsuite`、`gres` 和 `dlc` action 使用独立 CUDA
进程，不连接训练 Ray cluster。标准 RefCOCO 读取服务器的 `instances.json`、`refs(unc).p`
及 `train2014`，输出 cIoU/mIoU；批量生成固定使用 decoder-only 模型要求的 tokenizer left padding，避免
right-padding 导致不同长度多模态 prompt 的生成位置错位；生成默认最多 256 个新 token，可通过
`REFCOCO_MAX_NEW_TOKENS` 覆盖（例如设为 `128` 可复现旧评测上限）。RefCOCO prompt 可通过
`PROMPT_TEMPLATE` 覆盖，模板必须包含 `{phrase}`；逐样本 JSON 会保存实际模板，resume 只有在 prompt
模板与 `mask_protocol` 同时匹配时才跳过旧结果，避免 prompt ablation 混算。未设置时仍使用历史
RefCOCO 正例约束 prompt：
`The referring expression below describes an object that is present in the image. Locate and segment exactly that object. Do not answer "No target", "null", or refuse. Output only one mask group in this format: <|mt_start|><|mt_XXXX|><|mt_XXXX|><|mt_end|>. Expression: {phrase}`。
该 prompt 在全量 `step_156` RefCOCO val 上得到 cIoU `74.0443` / mIoU `76.0541`，而旧短 prompt 为
`29.4460` / `22.1497`；全量 response 审计中 literal no-target 从 `8027/10834` 降至 `111/10834`，新 prompt
的 10,834 条 JSON 均记录一致 prompt metadata，且无 malformed mask-token group。它不能由 GRES/gRefCOCO 脚本替代。GroundingSuite 与 GRES/gRefCOCO
也默认生成最多 256 个新 token，分别可通过 `GROUNDINGSUITE_MAX_NEW_TOKENS` 和 `GRES_MAX_NEW_TOKENS` 覆盖。GroundingSuite 接收其
数据根和可选 COCO 图像根，并在推理后保留逐样本 JSON 与合并 JSONL；仓库 metric 使用逐样本 JSON
目录计算 mask GIoU。当前服务器的默认 GroundingSuite 根目录是
`/volume/ybo/xyc/third_party/GroundingSuite`，其评测 JSONL 为
`/volume/ybo/xyc/third_party/GroundingSuite/GroundingSuite-Eval.jsonl`；可分别通过
`GROUNDINGSUITE_ROOT` 与 `GROUNDINGSUITE_DATASET` 覆盖。GroundingSuite 的 JSONL 若只保存
12 位 COCO image ID（如 `000000123456.jpg`），推理器会在 `data_root`、其 `assets/`、
`unlabeled2017/`、可用的 `train2014/` 子目录及 `coco_root/train2014` 中同时尝试该名称及官方的
`COCO_train2014_000000123456.jpg` 名称；无法
解析或读取的图像会立即令对应 shard 失败，不会经过多次退避后静默跳过并产生不完整结果。GRES 读取官方 `grefs(unc).json` 和 `instances.json`，以 `GRES_IMAGE_ROOT` 定位 COCO `train2014`；launcher 仅在全部 case 预测写完后汇总，输出 `N_acc`（no-target 拒识）、`T_acc`（有目标检测）、gIoU 和 cIoU。cIoU 按原协议累计有目标区域以及 no-target 的误检像素，正确的空预测不增加 union。无需重新推理即可用 `qwen3vl_gres_eval.py --metric-only --subset-report-file` 基于同一 case 编号和官方标注写出 JSONL 子集报告：精确的 no-target/single-instance/multi-instance 及基于 GT 面积的 small/medium/large；该模式要求完整 case 文件，并验证已保存的 `gres_<split>_samples.json` 与官方 refs 的顺序一致。DLC-Bench action 只产出 prediction JSON，最终语言 judge 需要
单独配置可用凭据。DLC caption inference 对全局图和可选 zoom-in 图使用与训练相同的正向 caption 指令
`Provide a detailed factual description of this region {SEG}.`；评测 prompt 不出现 `mask`、`token`、`JSON` 或
`reasoning` 等 segmentation 格式词，以免 SAMTok 将描述请求误解为定位请求。生成上限固定为 192 token。此协议是评测条件的一部分，比较任何 checkpoint 前都必须以同一版本重新推理。

该入口以项目 Conda 的明确解释器运行，并将仓库根目录加入 `PYTHONPATH`。顶层
`evaluation/*/*.py` 是按文件路径执行的脚本，Python 默认只会把其子目录加入 `sys.path`；若
遗漏该设置，`from projects...` 会因找不到仓库顶层包而失败。

当原始 FSDP world size 的 GPU 无法同时获得时，`tools/reassemble_fsdp_checkpoint.py` 是常规
`export` action 的明确离线替代：用 `torchrun --nproc_per_node=<checkpoint world size>` 建立同等数量的
CPU/Gloo rank，每 rank 读取一个同名 actor shard，rank 0 逐参数还原并写出 safetensors。它不使用 Ray、
rollout 或 VQ-SAM2，也不改变 checkpoint；代价是需要足够的 CPU RAM 容纳 rank 0 的完整模型和临时 shard。
该脚本不是降低 FSDP CUDA export world size 的通用开关，常规路径的 `NUM_GPUS` 仍必须匹配 shard 文件名。

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

`projects/rl/datasets/prepare_balanced_cyclegrpo_dataset.py` 将五份已转换的 parquet 混合为
可配置总量的图像/mask CycleGRPO 数据；当前默认配方是 20,000 条：8,000 条 RefCOCO 单实例 (`refcoco_cycle`)、4,000 条
gRefCOCO 多实例 union mask (`grefcoco_cycle`)、5,000 条 COCO-Stuff 语义区域
(`cocostuff_cycle`)、2,000 条 PACO-LVIS 真 part mask (`paco_part_cycle`) 和 1,000 条
gRefCOCO no-target (`gres_no_target`)。比例为 Single 40%、Multi 20%、Stuff 25%、
Part 10%、no-target 5%。`multi` 候选必须带 `grounding_instance_count>=2`；混合器在抽样前过滤，
候选不足会失败，绝不以 single expression 补足。可通过
`--single-count`、`--multi-count`、`--stuff-count`、`--part-count` 与 `--no-target-count` 覆盖各配额；
五项之和决定输出总量且必须为正，未传入时保持默认 20,000 条配方。

`prepare_dam_cycle_dataset.py` 将 Describe Anything 的 `mask_rle + caption` region
转换为当前 CycleGRPO 图像/mask parquet。`cocostuff_cycle` 直接读取 DAM
`COCOStuff/annotations.json`；`paco_part_cycle` 还必须与官方 PACO-LVIS train annotation
中 `id != obj_ann_id` 的 part 交集，避免将 parent object 混入 Part source。两类 source
默认每张图保留一个 region，并限制 mask 面积为图像的 1% 到 90%。输出保留原有
`source`、RLE 和 SAMTok token schema，额外写入不进入 prompt 的 `dam_source_id`；DAM
caption 写入独立 JSONL manifest，供后续 DLC QA 使用。

`generate_dam_caption_qa.py` 是该 manifest 的离线后续工具，不参与 dataloader、reward 或
主训练入口。它只以 DAM caption 作为文本证据，通过 OpenAI-compatible 本地 LLM 为每个
`dam_source_id` 生成两道四选一 positive QA，并可附加一道 `Yes=-1/No=0` 的 negative
hallucination QA；第二次 LLM 调用要求逐题验证显式文本蕴含、唯一正确项和无歧义干扰项。
本地 schema 检查会拒绝不满足这些积分约定的输出。结果以可断点续跑的 JSONL 写出，保留
caption、source、图像关联和生成模型元信息；可选额外导出与 DLC 无图 judge 兼容的 QA/class-name
JSON 映射。DAM caption 仍不进入 actor prompt。生成/验证 prompt 内的 JSON schema 使用 Python
`str.format` 的转义字面花括号，只有 caption 与 candidate JSON 是格式化字段；这保证请求在发送至
本地 LLM 前不会因 schema key 被误解析为 format field 而失败。

RefCOCO 与 gRefCOCO 转换器把原始 expression 另存为 `grounding_query`；它与被混合器清空的
`cap_answer` 严格分离。COCO-Stuff 转换器把官方 91 个 semantic-Stuff label 的原始 PNG 值
转换为 `the {label}`；其 GT 是同一图中该类别的完整 semantic union mask。PACO-LVIS v1 的 non-parent
part annotation 只保留 parent object `category_id`，没有逐 mask part-category id；转换器因此按同图、同 parent
object 类别合并全部 part mask，并构造 `the visible parts of the {parent}`，而不伪造细粒度 part 名称或任意选择
一个 part instance。二者写入 `grounding_query_kind=semantic_label/parent_parts_label`，
属于类别标签模板监督而不是人工 referring expression。gRefCOCO no-target 写入
`grounding_query_kind=no_target_referring`。DAM 行的 `grounding_query=null`，仅保留 `dam_source_id`。若向混合器传入
`--caption-qa-manifest`，其 JSONL 中的 Stuff/PACO ID 会在随机填充前强制纳入相应配额，随后训练 reward
actor 通过同一 ID 加载 QA sidecar，而不是将题目写进 parquet；混合器与 reward actor 都校验唯一 ID、source、
两道 positive 四选一题和可选 `Yes=-1/No=0` negative 题的 `1/0/-1` 契约。默认严格要求 sidecar 是 Stuff 3k +
PACO 2k（可用 `--qa-stuff-count` 与 `--qa-part-count` 显式覆盖）。

`prepare_paco_lvis_part_cycle_dataset.py` 只接受 `id != obj_ann_id` 的 PACO annotation，并且每张图
最多选一个 parent-category part union，因而不会把 parent object mask 混入 Part 配额或让少数密集标注图主导。
PACO-LVIS 需要其对应的 COCO 2017 图像；`--images-dir` 可指向图像父目录或直接指向
`train2017`/`val2017` split 目录。`prepare_cocostuff_cycle_dataset.py` 只从官方
`stuffthingmaps_trainval2017` PNG 的像素值 91..181 选取区域；这些值对应 COCO-Stuff 官方 label id
92..182，0..90 是 COCO thing、255 是 void，均不进入 Stuff 配额。Stuff 转换按类别频率优先选取，
仅保留面积占图像 1% 到 90% 的语义区域。

所有正样本只写入图像、目标 mask、由当前 SAMTok VQ-SAM2 编码出的 `seg_answer` 与从该 mask 构造的
caption prompt；混合器在写出最终 parquet 前统一清除 RefCOCO、gRefCOCO、PACO 与 COCO-Stuff
正样本继承的 `cap_answer`，因此不把人工 expression、COCO 全图 caption 或类别名输入 caption cycle；独立的
`grounding_query` 仅供 direct-grounding metadata 使用。`--require-grounding-query` 会在混合阶段拒绝任一
缺失 query 的 source，适用于要求五路样本均进入 direct segmentation anchor 的新 parquet。gRefCOCO
no-target 继续沿用已有 `gres_no_target` schema 与原有 rejection reward，其表达式是必须被拒识的 query；
其训练输入使用标准 GRES/RefCOCO 评测同构的 `Please segment {expression} in this image.`，而不是显式
提示模型当前样本必为 no-target。混合器写入 manifest，固定记录
输入 parquet、seed、五类数量与最终 source counts；任一来源不足请求数量时 fail-fast。该数据配方是当前
OPSD 扩展的 GroundingSuite 覆盖实验，并非论文原始 DenseWorld 数据或方法公式的一部分。

`prepare_grefcoco_cycle_dataset.py` 也支持 `--positive-samples 0 --no-target-samples N` 的纯
negative 导出：它不加载 VQ-SAM2、不写空的 positive parquet，而写出带人工 `grounding_query`、
`seg_answer=<answer>No target.</answer>` 和 `source=gres_no_target` 的 no-target parquet。这是 20k
RefCOCO 正例 + 20k gRefCOCO negative direct GRPO/SFT 流的负样本输入，不应混入主 20k CycleGRPO
parquet。该转换器也接受可重复的 `--exclude-parquet`：读取已有 cycle/direct parquet 的
`(COCO image_id, normalized grounding_query)` 与 `(COCO image_id, union-mask RLE)` 身份并在抽样前排除
对应 gRefCOCO ref，并在新候选内部按相同身份去重。用于新的 gRefCOCO direct multi 正例时，应同时传入主
20k cycle parquet 和既有 RefCOCO direct parquet；共享 COCO 图像本身不会被排除，只有同图同人工表达或同一
union target 才会被拒绝，输出 manifest 记录排除文件和两类身份数量。

## 3. 主训练调用链

### 3.1 启动与配置合并

1. `projects/rl/qwen3vl_4b_mt.sh` 或服务器入口 `qwen3vl_4b_refcoco10k_volcengine.sh` 调用 `python3 -m verl.trainer.main`；默认单机入口会清除不兼容的外部 `RAY_ADDRESS`。只有 `MULTINODE_ENABLED=true` 时，入口才保留由项目环境启动、并已验证为两节点 14-training-GPU 的私有 Ray 地址。
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
| `grounding_query` | 可选人工 referring expression；只用于独立 direct-grounding rollout，不进入 caption prompt、`R_Ci` 或 teacher target |
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
- `_build_gen_seg_messages` 把 actor caption 放进定位模板；图像 cycle 在 RefCOCO/GRES 的 `Please segment {caption} in this image.` 与 GroundingSuite 的 `Please carefully check ... detect the object this sentence describes: {caption}` 之间交替，bbox 和视频保留历史模板。
- 图像 caption 可使用多图，segmentation 只保留第一张图；视频保留帧率和帧数元数据。
- 返回 `cap_*` 和 `seg_*` 两套 input ids、attention mask、position ids、raw prompt ids 与多模态数据。

主 `data.train_files` 是唯一决定 CycleGRPO epoch 长度、global step 与 checkpoint cadence 的数据流。
开启外部监督时，`trainer/main.py` 会另建两个可恢复的 `StatefulDataLoader`：
`worker.supervised_anchors.direct_grounding.train_files` 专供 RefCOCO 正例和可选 gRefCOCO no-target 的
direct GRPO/SFT，
`worker.supervised_anchors.caption_qa.train_files` 专供 DLC-QA caption GRPO。两者各有独立
`batch_size`，每个主 step 只各读取一批，并循环遍历自身数据；它们绝不消费、重排或缩短主
CycleGRPO iterator。resume 时三个 loader 的 state 都保存到 checkpoint。

火山引擎 70k 三流入口默认使用 `ROLLOUT_BATCH_SIZE=112`、`DIRECT_BATCH_SIZE=224` 和
`CAPTION_QA_BATCH_SIZE=56`，并设置 `MAX_STEPS=179`。这使 20k 主 CycleGRPO、40k
direct supervision 和 10k DLC-QA 分别在约 179 个 optimizer step 内各消费一遍（parent-prompt
配额比例 `2:4:1`）。这些 batch 只决定各 loader 每步读取的 parent prompt 数，不改变 rollout 数、
loss weight 或三流在同一 optimizer step 中的梯度累积顺序；同名环境变量仍可覆盖默认值。
25%/50% baseline wrappers 复用该 batch 与训练链，仅把三流输入同时缩至原来的 25%/50%，
并将默认步数设为 45/90；common 启动器对各缩量行数及 DLC-QA sidecar 行数进行预检。

服务器入口的 `THREE_STREAM_2_4_1_ENABLED=true` 是固定配额模式：要求 `NUM_GPUS=7`，并要求
main/direct/DLC-QA 三个 parent-prompt batch 均能整除 7，且满足
`ROLLOUT_BATCH_SIZE:DIRECT_BATCH_SIZE:CAPTION_QA_BATCH_SIZE=2:4:1`。推荐值为
`28:56:14`，故每个 FSDP rank 概念上处理 `4:8:2` 个 parent prompt；经过 `G=6` caption 或
`K=6` localization rollout 后，各流仍按相同 rank 对齐。第八张物理 GPU 不属于 Ray/FSDP world，
只运行独立 Llama DLC judge service。`ACTOR_GLOBAL_BATCH_SIZE` 仍必须整除主
`ROLLOUT_BATCH_SIZE`，推荐保持 `28`，不能设为三流 parent batch 总和 `98`。
该模式还强制 20k 主流为纯 CycleGRPO + online pixel-IoU OPSD：`OPSD_ENABLED=true`、
`PIXEL_IOU_ENABLED=true`，但 routing、EMA teacher、teacher analysis 和 caption safety、
caption anchor KL 与 segmentation anchor KL 均关闭。这样 regenerate CE、privileged JSD、KL anchor
或 verifier 不会向 20k 流添加描述/分割辅助监督；唯一的描述 reward 来自独立 DLC-QA 流，唯一的
人工 referring/GT-mask 或 no-target refusal 监督来自独立 direct 流。
单节点 `7+1` 有监督 controller 的 world size 为 7：正式历史配置使用 `128:256:64`，而本机四卡
smoke 使用 `114:228:57` 以便被 3 个训练 rank 整除；`28:56:14` 是旧严格三流模式的配额。正式 controller 仍使用
`NUM_GPUS=7` 并在提交前检查三个 batch 都能被 7 整除。入口 attach 校验另允许 1--7 卡的外部-judge
开发 smoke；例如三卡可使用 `120:240:60`，但不得将其结果视为正式 7+1 实验。

### 3.3 Phase 1：caption rollout

`RayPPOTrainer._make_batch_data`：

1. 从 dataloader 取 batch，为原始 prompt 分配 `uid`，用 `cap_*` 字段构造 `task=caption` 的 `DataProto`。`data.cycle_prompt_mode=current` 保留 parquet 内 prompt；设置为 `official_source_aware` 时，`denseworld_single/refcoco_cycle` 使用单区域 `Provide a detailed description of this region {mask}`，`denseworld_multiple/grefcoco_cycle` 使用官方多区域 interleaved-mask 模板，`paco_part_cycle` 和 `cocostuff_cycle` 分别使用 visible-parts/semantic-region 模板；no-target 或未知 source 保留原 prompt。
2. `FSDPWorker.generate_sequences` 通过 `FSDPVLLMShardingManager` 把当前 actor 权重同步到 vLLM，再采样配置的 `G=6` 个回答。
3. 原样本按 `n` 重复并与 rollout 输出合并。
4. 对 image OPSD，像素 IoU 回写后 driver 用未跳过 special token 的实际 caption rollout 检查：非终止的 `<|...|>` special token、`mask_2d` JSON 和超过 `caption_safety.max_response_tokens` 的输出都标为不安全。默认强制将其 route 改为 `regenerate`；不安全 caption 不进入原始 caption GRPO 或 mid-route JSD，但 localization rollout/奖励仍保留。
5. 按 `source` 分流：`denseworld_single`、`denseworld_multiple`、`refcoco_cycle`、`grefcoco_cycle`、`cocostuff_cycle`、`paco_part_cycle`、`tg_multi_merged`、`dam_cyclegrpo` 和 `None` 进入 cycle batch；其他 source 进入 non-cycle batch。`grefcoco_cycle` 将 gRefCOCO 正样本的一个或多个 COCO instance mask 合并为 cycle target；`cocostuff_cycle` 是单类语义 Stuff 区域，`paco_part_cycle` 是真 object-part 区域；三者均走相同的 caption-to-localization rollout、真实 pixel IoU 与 CycleGRPO reward。其 `ann_id=[-1]` no-target 表达保留为 `gres_no_target`。`text` 与 `official_bbox` 保持 caption-only non-cycle；历史参考实验使用的 `pixel_empty` 则先从主 non-cycle batch 分离 `gres_no_target`，为其构造专用 `grounding_query` segmentation rollout，并由 `no_target_segmentation_loss_weight` 控制该 actor 的梯度权重。只有显式 direct grounding 才会再构造独立 direct segmentation batch；`consume_no_target_caption` 仍为 `false`，避免 direct 配置删除主 caption PPO。
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
3. 调用 dataset 的 `_gen_seg_preprocess`，把 caption 注入 localization prompt。默认图像 caption index 按偶/奇在 RefCOCO/GRES 与 GroundingSuite 两个 benchmark 同构模板之间交替；通过 `worker.opsd.pixel_iou.localization_prompt_mode` 可统一切换为 `refcoco`、`groundingsuite` 或旧的 `legacy` 模板。若 `data.cycle_prompt_mode=official_source_aware`，图像 mask localization 忽略该交替设置并统一采用官方长模板（包含 mask 格式和 `null` 说明）；视频/bbox 分支保持各自模板。`caption_text` 仍保存裸 student caption，供 CycleGRPO reward、OPSD 路由和 teacher 使用。
4. 从 `worker.opsd.localization_rollouts` 读取 `K`，用当前 actor 为每条 caption 采样定位结果。
   每条 localization rollout 的生成上限由 `pixel_iou.segmentation_max_response_tokens` 控制（当前默认
   `256`，可显式设置为历史 `32`）；合法的多个 mask group 是否共同表示一个区域由
   `pixel_iou.mask_decode_mode` 控制。这不改变 caption rollout 上限、`K` 或 no-target 拒识生成。
   当 parent 子批量不足 world size 时，输入 prompt 会临时补齐；生成结果去除补齐 response 后，展开的
   UID、`localization_index`、source 等元数据严格按未补齐 prompt 数量构造，并检查输出为
   `prompt_count × K`，避免 padding 后数量残留造成 DataProto 一致性错误。
5. vLLM offload 后再把 VQ-SAM2 移入 GPU；按原图分组，仅计算一次 SAM2 image embedding，并分 chunk 解码目标 token 与 `G*K` 个预测 response 中的合法 group。默认 `mask_decode_mode=union`，在每条 response 内取全部解码 mask 的像素 union；设置为 `first_mask` 时只保留 response 中第一个合法 group，用于复现原始训练语义。该开关同样作用于 no-target 的 `pixel_empty` 判定。
6. 非法、缺失或空 mask 记为 IoU `0`。优先使用可转换的 dense/PIL/COCO RLE/polygon 原始 GT；缺失时解码 `seg_answer` 的目标 token，并记录 `raw_gt` 或 `decoded_target` reference 来源。上一版基线不含正样本空 mask 或 no-target 非空 mask 的额外 penalty；其 no-target 正确性只由严格二值 `pixel_empty_reward`（显式 `No target.` 且 decoded union 为空）提供。
7. mask logits 双线性恢复原图尺寸并以 `0.5` 二值化；每条 caption 的 `K` 个 IoU 求均值得 `R_Ci`，再按当前入口的 `low_threshold/high_threshold` 严格分路由；未覆盖时仍为 `0.5/0.85`。
8. 视频 cycle 保留原 tIoU 与 GRPO 路径，不进入 image-only OPSD teacher 路由。
9. 恢复外层 rollout `n`，返回 `cycle_cap_batch` 和 `cycle_seg_batch`。

主 20k 的 `pixel_empty` no-target 行沿用上一版的专用 segmentation actor：trainer 先从 non-cycle
batch 分离每个唯一 `gres_no_target` UID，再复制为 `K=6` 个 localization rollout，并通过
`compute_no_target_pixel_empty` 计算严格二值 reward。只有显式 `No target.` 且 decoded union 为空时
得到 `1.0`，其他情况为 `0.0`；该批次与 cycle segmentation 一起进入 actor 更新，梯度乘以
`worker.opsd.no_target_segmentation_loss_weight`（默认 `1.0`）。因此这些行不会进入 caption batch
concat，也不需要 `no_target_pixel_empty`/`no_target_reward_mode` 的临时 metadata 清理。独立 direct
no-target segmentation rollout 仍按外部 direct supervision 配置运行。

启用 `worker.supervised_anchors.direct_grounding` 或 `direct_mask_ce` 时，trainer **只**从
`direct_grounding.train_files` 读取 standalone direct parent batch，不再从 cycle/non-cycle 主子批抽取。
这些文件可以是 `DIRECT_TRAIN_DATA` 的 RefCOCO 正例和可选
`DIRECT_NO_TARGET_TRAIN_DATA` 的 gRefCOCO no-target 行；按启用项，人工表达正例集合始终为
`{refcoco_cycle, grefcoco_cycle}`，可选 no-target 集合增加 `gres_no_target`，而
`include_label_sources=true` 额外允许 `{cocostuff_cycle, paco_part_cycle}`；其余 source 在首 step
显式报错。每个
parent expression 建立一个独立 `K=6`
text-to-mask rollout group，先裁到 world size 的整倍数；不足一个 rank-shard 时跳过并记录原因。
这最多丢弃 `world_size-1` 个 direct prompt，不影响主 caption/cycle batch 或其 reward。query 按偶/奇
index 在 RefCOCO/GRES `Please segment {query} in this image.` 与 GroundingSuite `Please carefully check ...`
模板之间 1:1 交替，也可由 `worker.opsd.pixel_iou.localization_prompt_mode` 统一设为
`refcoco`、`groundingsuite` 或 `legacy`。direct batch 的 UID、优势和日志独立，正例用原始 RLE pixel IoU，不调用 cycle
的 `R_Ci` 合并、OPSD routing、teacher regenerate 或 JSD。no-target direct GRPO 复用选定的
`text|official_bbox|pixel_empty` no-target reward；后者与离线 `N_acc` 一致。火山引擎入口默认关闭 direct anchor；
开启正例时必须传 `DIRECT_TRAIN_DATA` 与 `DIRECT_BATCH_SIZE`，开启 no-target GRPO 或 SFT 时还必须传
`DIRECT_NO_TARGET_TRAIN_DATA`。`consume_no_target_caption` 始终为 `false`，保证 direct 是对主 no-target
caption GRPO 的附加训练。

`direct_grounding.loss_weight` 是目标权重，不直接以固定值累积。开启该 anchor 后，
`direct_grounding.warmup_start_step=10` 前有效权重为零，`10..30` 之间线性升到目标值，
`warmup_end_step=30` 后保持目标值；因此推荐受控实验的 `loss_weight=0.15` 不会在早期直接压过
cycle localization。direct GRPO 仍按其单独 UID 的 `K=6` reward/advantage group 正规化，绝不能与 cycle
caption 或 localization reward 拼接后共同 whiten。

可选 `worker.supervised_anchors.direct_mask_ce` 是与 sampled direct GRPO 分开的 teacher-forcing
anchor。正例从 standalone RefCOCO parent batch 的每个原始 UID 建立
`grounding_query -> seg_answer`，target 是完整 GT SAMTok group；当
`direct_mask_ce.include_no_target=true` 时，gRefCOCO no-target row 同样建立一条
`grounding_query -> <answer>No target.</answer>` SFT 目标。二者服从同一个
`localization_prompt_mode` 配置（默认两种 localization prompt 交替），不接收 rollout response、IoU 或 advantage，也不会从 20k 主混合数据派生。CE loss mask 覆盖正例
的完整 mask token 或 no-target 的完整拒识文本 token，EOS/padding 只作为前向上下文。构造该 batch 时必须从
保留 batch 维度的 non-tensor object array 取每个 parent 的图像 metadata、GT 和 mask。主 cycle rollout
的 caption media 字段名是 `multi_modal_data`，独立 direct loader 则保留
`cap_multi_modal_data`；两者共享 `seg_multi_modal_data`，trainer 必须兼容这两种合法输入。不得先以
`DataProto[index]` 解包再用 `[0]` 索引字典。该项默认关闭，推荐开启时固定 `loss_weight=0.02`，并在同一
optimizer step 中独立累积，不通过 K 次 rollout 放大。

用于梯度冲突诊断时可设置
`worker.supervised_anchors.direct_mask_ce.record_base_gradient_cosine=true`。trainer 会先累积该 step
中所有非 CE 梯度（CycleGRPO caption/localization、direct GRPO、DLC-QA 以及其他已启用的 caption auxiliary），
由 FSDP 跨 rank 统计其与 direct mask CE 的 dot product、两侧范数和余弦相似度，并记录
`supervised_anchors/direct_ce_base_grad_cosine`、
`supervised_anchors/direct_ce_base_gradient_conflict`、
`supervised_anchors/direct_ce_base_grad_norm` 和
`supervised_anchors/direct_ce_grad_norm`。统计后会恢复原非 CE 梯度并原样加回 CE 梯度，因此该开关不改变
optimizer 更新；它与 `worker.opsd.asymmetric_gradient_projection` 互斥，避免 caption-first 投影暂存顺序
使 base 定义不明确。

`direct_mask_ce.warmup_start_step` 与 `warmup_end_step` 可将 CE 权重从 0 线性升到目标
`loss_weight`；例如 `10/30` 表示 step 1-10 不施加 CE，step 11-30 线性升至 `0.02`，之后保持。
默认均为 `0`，保持历史固定 CE 权重行为。实际每 step 权重记录为
`supervised_anchors/direct_mask_ce_weight_effective`，用于和梯度余弦、范数共同分析。

如需识别三流中的具体冲突，可启用
`worker.supervised_anchors.gradient_diagnostics.enabled=true`。在一个 optimizer step 内，trainer 按实际
反向累计中快照 `cycle_caption`、`cycle_segmentation`、`direct_grpo`、`direct_mask_ce`、`dlc_qa`，以及启用时的
`regenerate_ce`、`privileged_jsd` 加权梯度分量，并记录每个
`supervised_anchors/multitask_gradient/<component>_grad_norm`，以及任意已存在分量对的
`<left>_vs_<right>_cosine` 与二元 `..._conflict`（负内积为 1）。Cycle caption/segmentation 的 PPO 与其
同次 forward 中的 KL 属于同一分量。诊断只从已累积的 `.grad` 中扣除此前快照来观察新分量，不清零、投影、
重标定或改变 optimizer 输入；快照在该 step 的 optimizer 前释放。由于每个 FSDP rank 会暂存最多七份 gradient
shard，该开关仅用于 30--50 step 诊断，不应用于完整训练；它与
非对称梯度投影和旧的 direct-CE-vs-base 诊断互斥。
诊断开关在每个 trainer step 的公共路径读取，因而纯 CycleGRPO（没有 direct 或 DLC-QA 辅助 loader）也可安全
记录其实际存在的 cycle 分量；缺失的辅助分量不会被伪造，也不会改变正常的梯度累计顺序。

代码中存在 `generate_sequences_with_ref`，可临时把 vLLM 换成 reference policy 权重，但当前调用已注释，实际调用 `generate_sequences`。因此当前有效实现确实是“actor 作为自己的 critic”，而不是冻结的外部 critic。

### 3.5 奖励

`verl/workers/reward/function.py::BatchFunctionRewardManager` 动态调用 `projects/rl/reward_function/text2mask.py:compute_score`，并只把标量奖励写到 response 最后一个有效 token；之后优势会扩展到整个 response mask。

核心图像 cycle source 保留原 CycleGRPO 的倍率、格式和非重复项，只把原始 HTG
token matching 的 `s_i,k` 替换为在线 VQ-SAM2 解码后的真实像素 IoU：

```text
s_i,k = pixel_iou(union(decode(all legal groups in response_i,k)), GT_i)
m_i   = mean_k(s_i,k)

caption:
  R_cap_i = (non_repeat_i + 10*m_i) * valid_i + valid_i
  valid_i 同时检查没有 bbox/中文，且没有非终止 special token 或 mask_2d JSON；违规时正奖励被门控清零。

localization positive:
  R_loc_i,k = 10 * s_i,k * m_i + format_i,k + non_repeat_i,k
```

每条 response 的全部完整且 codebook 合法 depth-2 group 都会解码并 union；多组并非错误，
其额外区域自然反映在 union 的像素 IoU 中。`format_i,k` 仍要求至少存在一个完整 group；
`non_repeat_i,k` 保留原始规则，只在同一个完整 group 出现超过三次时从 `1` 降为 `0`，不置零 IoU
也不施加额外 group penalty。`supervised_grounding` 使用同样的 format/non-repeat 语义，奖励为
`10 * pixel_iou + format + non_repeat`。这与论文的 `R_cap_i=mean(s_i,k)`、
`R_loc_i,k=R_cap_i*s_i,k` 对应，但代码仍保留 `10x` 和两项一分的序列化正则。

`text2mask.py` 还保留多任务分支：

| `source` | 奖励行为 |
|---|---|
| `groundingme` / `denseworld_*` / `refcoco_cycle` / `grefcoco_cycle` / `cocostuff_cycle` / `paco_part_cycle` / `dam_cyclegrpo` / `None` | 图像 CycleGRPO 主分支 |
| `gres_no_target` | no-target/null 正确性 + 非重复奖励 |
| `tg_multi_merged` | 视频循环：tIoU、时间格式、段数门控、禁止 caption 泄漏时间 |
| `dam_captioning` / `tg_captioning` | 外部 OpenAI-compatible vLLM judge 的布尔 caption reward；仅当 batch 实际包含这些 source 时才初始化 judge client，不是主 CycleGRPO 路径 |
| `dam_grounding` / `tg_grounding` | 独立 grounding 任务，分别做 mask-token 或时间区间奖励 |
| `gcg`、`psg` 等 | grounded caption/scene graph 的 token、短语、格式奖励或保留分支 |

`worker.opsd.pixel_iou.no_target_reward_mode` 控制 `gres_no_target` 与历史可用的
`supervised_grounding_no_target` 的正确性项。默认 `text` 保持原来的 `1.0 / 0.2 / 0.0`
取值：响应必须含 `No target.`，且不含任何 SAMTok `<|mt_start|>`、`<|mt_####|>` 或
`<|mt_end|>` 片段；任一完整或残缺 mask-token 都会使该项为 `0.0`。选择 opt-in
`pixel_empty` 时，主 20k no-target 行仍计算 caption reward；FSDP worker 在该 non-cycle caption response
上解析全部完整、codebook 合法的 depth-2 group，以与离线 `legacy_union` 相同的 VQ-SAM2 和阈值解码并取像素 union；仅当 union
为空（包括无合法 group、残缺 group 或合法 group 解码为零像素）**且** response 以大小写不敏感的精确
`No target.` 短语显式拒识时，该正确性项才为 `1.0`。decoded union 非空时无论 response 是否写出拒识，默认
为 `-1.0`，以惩罚 no-target 的 mask 假阳性；将
`worker.opsd.pixel_iou.no_target_nonempty_mask_penalty`（入口环境变量
`NO_TARGET_NONEMPTY_MASK_PENALTY`）设为 `0.0` 可恢复该惩罚加入前的 `0.0`，同时仍执行 decoded-union
判定和正确拒识的 `+1.0`。仅在 union 为空但拒识格式无效时为 `0.0`。因此 `null`、普通解释、
空回复或仅 EOS 即使没有 mask 也不会得到 pixel-empty 奖励；`<answer>No target.</answer>` 仍是合法拒识格式。
`no_target_accuracy`/`seg_supervised_grounding_no_target` 继续记录正确拒识率，独立
`*_no_target_pixel_empty_reward` 与 `*_no_target_nonempty_mask_penalty` 记录实际三值 reward 和 false-positive
负分；惩罚系数为零时该负分指标为 `0.0`。该训练奖励比 GRES `N_acc` 的单独 `not pred_mask.any()` 更严格，目的是避免模型以空生成钻取 reward 空子集。
它要求 `worker.opsd.enabled=true` 和 `pixel_iou.enabled=true`，缺少 GPU 解码 metadata 会显式报错，不会退回
文本奖励。无论模式如何，第二项原有的非重复奖励均保持不变；主 no-target 更新始终作用于 captioner。独立
direct no-target segmentation rollout 仍按其自己的外部监督配置运行。

`worker.opsd.pixel_iou.positive_empty_mask_penalty`（入口环境变量
`POSITIVE_EMPTY_MASK_PENALTY`）默认是 `1.0`。对有非空 GT 的正例 segmentation rollout，若 response
包含大小写不敏感的 `No target.`，或其 decoded union 没有像素，则 `seg_overall` 额外加 `-1.0`；设为 `0.0`
可关闭。它不改变 `pixel_iou`、`R_Ci` 或离线 cIoU 指标，只通过独立的
`seg_positive_empty_mask_penalty` 奖励字段提供相对 GRPO 信号，也绝不作用于正确的 no-target 拒识。探索分支额外支持 `cycle_positive_empty_mask_penalty` 与 `direct_positive_empty_mask_penalty`，分别约束 cycle localization 和 direct grounding；未设置时回退到全局 `positive_empty_mask_penalty`。

当 `worker.supervised_anchors.caption_qa.enabled=true` 时，trainer 从独立
`caption_qa.train_files`（DLC-QA 10k parquet）采样 caption rollout，并将 source 改为
`supervised_caption_qa`。reward actor 在初始化时读取已验证的 QA JSONL，并严格按 `dam_source_id`
join 每一条 rollout；任何缺失 join 都会报错而不是静默给零分。独立 Llama judge 只看到学生 caption、题目和
选项；每条 rollout 对全部题目作答，`1/0/-1` 的均值乘 `reward_weight` 就是**全部** caption reward。
该流不计算或混入 cycle IoU、format/non-repeat、caption safety、`R_Ci`、OPSD routing、
teacher regenerate 或 JSD。服务超时、请求失败或无唯一选项时该题贡献 `0`；`caption_qa.loss_weight`
在 actor 累积时控制此独立 GRPO 梯度的实际比例，因为组内 GRPO 标准化不保留 reward 的常数缩放。
训练期每个 QA judge 请求还显式传递 Llama-3.1 的 `<|eot_id|>` `stop_token_ids=[128009]`。
当前转换后的 HF tokenizer 没有 chat template；不传该停止 token 时，vLLM 会在模型输出选项后继续生成
下一轮 `assistant` header（例如 `Aassistant`），从而被严格选项解析器判为失败。该参数与离线
`evaluation/dlc_bench/eval_llama_without_image.py` 的 judge 请求保持一致，不改变题目、选项或奖励公式。

`supervised_grounding` 的正例 segmentation reward 对一条 response 内所有合法 group 的 decoded union
计算 `10 * pixel_iou + format + non_repeat`；`supervised_grounding_no_target` 复用 `No target.`
拒识加非重复奖励。二者只出现在独立 batch。

`tg_reward.py` 是可配置的 temporal grounding 奖励库，支持 tIoU、format、precision/recall/F1、C-Acc、caption judge 和长度惩罚；当前 `text2mask.py` 的主要视频路径只直接复用其中少量逻辑或保留了注释调用。

当前主训练入口的非模块基线已重新对齐上一版实验 Git 快照 `06bf2bf`：主
`pixel_empty` no-target segmentation actor、`worker.opsd.no_target_segmentation_loss_weight=1.0`
和严格二值 `pixel_empty_reward` 均恢复；后续加入的正例空 mask penalty、no-target 非空 mask penalty、面积
比例/tau 以及 caption-only no-target 分流不属于该基线。EGCA 与 SECA 是此快照之上的唯一新增空间信用模块，
均由独立开关控制，默认关闭，不改变上述 no-target actor 或 reward 契约。

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

主代码将已验证的 Evidence Gate 与 Mask Credit 统一封装为 `verl/workers/opsd/seca.py` 的 **SECA (Spatial-Evidence Credit Assignment)** 模块。`worker.opsd.seca.enabled`（入口环境变量 `SECA_ENABLED`）默认关闭；开启后不改动 pixel-IoU、`R_Ci`、reward 或原始 GRPO：`spatial_evidence_weight` 用 IoU 与 reconstruction-only 面积比计算 `[min_weight,max_weight]` 门控，只乘到 mid-route privileged JSD 的 sample weight；`hierarchical_mask_token_weights` 在 direct mask CE 的有效 response mask token 中识别 SAMTok depth-2 code，将首个 coarse code 乘 `coarse_token_weight`、第二个 fine code 乘 `fine_token_weight`，其余 token 保持 1。新增 `worker.opsd.seca.self_supervised_enabled`（入口环境变量 `SECA_SELF_SUPERVISED_ENABLED`）明确打开 cycle-only evidence gate，供纯自监督 20k 使用；该路径仍是 detached 的 JSD sample weighting，不把 evidence 写入 reward 或 GRPO advantage。没有 direct mask CE batch 时，SECA 的 token-credit 子路径自然不产生更新；没有 mid-route 时，evidence 子路径自然不产生更新。`projects/rl/config.yaml` 保留两个开关默认关闭，`qwen3vl_4b_refcoco10k_volcengine.sh` 与 `qwen3vl_4b_mt.sh` 均暴露它们和五个参数，因而可在自监督或混合训练中复现实验分支的 Evidence+Mask Credit 行为。

主代码另外提供独立的动态 **EGCA (Evidence-Guided Code Attribution)** 路径，位于
`verl/workers/opsd/egca.py`，由 `worker.opsd.egca.enabled`（入口环境变量
`EGCA_ENABLED`）控制，默认关闭。它只作用于自监督 cycle localization，不绑定 direct
grounding、direct mask CE、DLC-QA 或 no-target。worker 仍在合法 SAMTok depth-2 mask
group 上解码 prefix/counterfactual，计算 coarse/fine Shapley 与 target/reconstruction
evidence；但当前默认 `update_mode=weighted_ce` 不把 signed credit 写入 GRPO advantage。
trainer 先按 `sample_uid` 聚合同一 prompt 的六个 localization rollout，再用其真实 pixel-IoU、
evidence gate 和 coarse/fine credit 生成 detached、非负的 sample/token weights。随后从同一
自监督行的 `seg_ground_truth` 构造独立 teacher-forcing batch，执行一次命名为
`opsd/egca_weighted_self_distill_loss` 的 GT SAMTok CE；证据权重覆盖完整 mask group，
coarse/fine 只允许增加对应 code-token 权重，负 Shapley 被截为零。这样循环仍提供 on-policy
探索和 reward，EGCA 只提供 rollout-conditioned self-distillation，不会形成额外的负向拒识通道。
`gres_no_target`、padding、非法 group 和 supervised source 均不会进入该 CE batch；它们仍保留
原有二值拒识、GRPO 或 direct supervision 语义。`EGCA_OPD_ENABLED` 仍可让 evidence gate
乘到 mid-route privileged OPD/JSD sample weight。

`update_mode=legacy_advantage` 与 `credit_mode=raw|contrastive` 仅保留给历史 ablation：该
兼容路径才会在 `compute_advantage` 后向 localization advantage 注入 signed credit，默认配置和
新的 20k 入口均不使用它。当前实现采用解码证据 gate 而非额外可训练 evidence head；因此不宣称
存在 learned calibration-head 参数更新。新入口 `tools/train_selfsupervised_20k_egca_weighted_ce_8gpu.sh`
使用 20k 自监督 parquet、8 卡、batch 128、1 epoch；对应的
`tools/train_selfsupervised_20k_egca_weighted_ce_4gpu_tmp.sh` 只用于本机 smoke test，覆盖为
4 卡和 `MAX_STEPS=1`，不改变生产脚本默认拓扑。
新模式 `reference_mode=target` 从每条样本的 GT SAMTok depth-2 group 动态提取 coarse/fine
code，作为 counterfactual reference；只有显式设置 `reference_mode=fixed` 时才使用旧的
code-0 reference，避免固定 code-0 在新训练中成为隐含的拒识偏置来源。

### 3.7 GRPO/OPSD token 监督诊断边界

`analysis/opsd_vs_grpo_diagnostic/` 是独立的离线诊断管线，不改变训练算法。`extract_signals.py` 只接受共同 student checkpoint 上导出的固定 rollout：必须同时包含完整 GRPO group 的 `advantages`、`response_mask`、`old_log_probs`、sampled `target_ids`、student logits、privileged teacher logits、route 和 `R_Ci`。脚本先验证完整 group（不能在路由子集上重新归一化 advantage），再固定筛选当前实现真正的 `on_policy_distill` 且 `R_Ci>=0.65` 分布教学分支，按 `(R_Ci,sample_id)` 排序，两种方法共用相同行和 mask。GRPO 调用 `verl.trainer.core_algos.compute_policy_loss` 的真实 clipped surrogate；OPSD 调用 `verl.workers.opsd.distillation.chunked_weighted_jsd_loss` 的真实 generalized-JSD、teacher entropy 权重、sample weight 和 response mask。两者均通过 autograd 计算 sampled-token 的 `-∂L/∂z[y]`，并执行有限差分检查。

当前仓库的 teacher diagnosis JSONL 和 FSDP checkpoint 没有保存这些逐 token logits/rollout 张量，因此严格脚本缺少输入时只写 `metadata.json` 的 blocked 状态，不生成随机或手工热力图。为满足日志级诊断需求，`plot_logged_proxy.py` 仍可从单个真实 OPSD run 生成 rollout/logged-loss proxy；新增 `plot_dual_proxy.py` 则读取真实 GRPO-only run 与 OPSD run，按 12 个时间阶段生成双热力图和两条独立的 logged-direction 路径：GRPO 面板把阶段聚合的一个 `pixel_iou_mean` 广播到六列，OPSD 面板保留每条 diagnosis 的六个真实 `pixel_ious`，并以该 diagnosis 的六次 rollout 均值为中心显示正负残差。因此该图直接展示 row-wise scalar credit 与 rollout-level evidence variation 的监督结构差异，但由于两套 run 不是共同 checkpoint/fixed rollout，不能作 loss 因果消融，也不能称为 token autograd 或参数更新图。`plot_signals.py` 的严格二维面板在没有 before/after 参数更新时明确使用允许的 local-logit-gradient fallback；只有 A/B 位置由更新前 teacher/student sampled-token log-prob 差异预注册阈值划分。严格图和两种日志代理图都只覆盖 OPSD 分布教学分支，不覆盖 regenerate、mask-code localization 或 SECA direct-CE 权重。 第一张增强双热力图由 `plot_ab_enhanced.py` 生成，使用统一对称色标和离散色阶；第二张配对更新图由 `plot_paired_update_direction.py` 严格要求共同 checkpoint/fixed rollout 的 `delta_logp_grpo`、`delta_logp_opsd`、`teacher_gap`、`group_id`，当前输入缺失时只写 blocked metadata，不生成模拟路径。

新增的 `analysis/token_update_scatter/` 是另一条只读、离线的论文图诊断管线。`extract_token_updates.py` 从文档记录的 disjoint RefCOCO parquet 固定抽取样本，用同一个 processor 对 SAMTok base、官方 GRPO checkpoint、Pixel-OPSD checkpoint 和独立 direct-supervision checkpoint 做真实 teacher-forced 多模态前向；只保留 GT depth-2 mask-code token，并保存四套 log-prob、有限值 mask、目标 token 位置和 `intervention_rollouts.jsonl`。由于历史 run 没有共同 sampled rollout、逐 token logits 或 before/after optimizer state，该管线采用明确命名的 `teacher_alignment` fallback：横轴是独立 teacher 与 base 的 log-prob 差，纵轴是历史 checkpoint 相对 base 的实际 `Δlog p`。这不是重新执行 GRPO/OPSD 更新，也不能将两套不同训练 parquet 的 checkpoint 差异表述为公平 objective 消融或因果 token credit。`plot_token_updates.py` 生成共享点、范围和色标的二维 GRPO/Pixel-OPSD 面板及同一批点的三维视图；三维 reliability 明确定义为 base 对 GT mask-code 序列的 `exp(mean log p)`，不是训练代码的 `R_Ci`。脚本、配置、审计、图注和真实数组均保存在该目录，且不修改论文、训练入口或 checkpoint。

为可观测性，`teacher_analysis` 可在每一步从 regenerate 和 mid route 各抽取一条最低 `R_Ci` 候选。EMA teacher 在独立 privileged prompt 中输出 JSON diagnosis：`failure_mode`、`missing_evidence`、`distractor_evidence`、`correction_focus`。driver 将其写入 checkpoint 根目录的 `teacher_diagnoses.jsonl`，记录 route、`R_Ci`、IoU 向量、student caption 和诊断文本；主标量日志只记录 `opsd/teacher_analysis_count`。诊断严格不进入 student prompt、teacher caption target、模型 checkpoint 或推理输出。该 pass 会增加一次小型 teacher rollout，设置 `worker.opsd.teacher_analysis.enabled=false` 可关闭。

当 captioner 和 segmenter 都启用时，trainer 不分别 optimizer step，而是：

```text
route-replacement: high GRPO + low CE + mid JSD
B: safe 全部原始 GRPO + low CE + mid JSD
辅助 CE/JSD 均按候选比例归一化，caption 侧再乘 caption_loss_weight=0.5
全部 route 的 localization GRPO，再乘 localization_loss_weight=0.5
可选 human-query anchor: lambda_direct(step) * direct GRPO + 0.02 * direct GT-mask CE
clip grad norm -> one optimizer.step()
optimizer.step 后原地执行 EMA shard 更新；当 `ema_teacher.decay=1.0` 时，更新为恒等映射，teacher
保持 worker 初始化时从初始 SAMTok actor 复制的参数。
```

这保证单一模型被两个方向联合优化。

direct-grounding 显式启用时，其 GRPO loss 在 cycle localization 后、同一次 optimizer step 前累积，权重为
`worker.supervised_anchors.direct_grounding.loss_weight`（generic YAML 默认 `0.25`）。火山引擎入口默认关闭该外部
supervised anchor；所有 no-target reward 模式下，主数据都保留原始 CycleGRPO 外层 caption GRPO 和两项拒识 reward，
而 pixel-empty 模式额外以 decoded empty-union 收紧其中的正确性项。若实验显式启用 no-target direct
group，它会额外使用独立 `K=6` group；`consume_no_target_caption=true` 已由配置校验拒绝，避免 no-target 仅依赖
direct rollout 而在同组正确拒识相同的情况下产生零 GRPO advantage。direct query 使用人工 expression 或类别模板，属于受控外部监督，不是 image-mask-only
CycleGRPO 的核心奖励或纯 on-policy self-distillation。

当同时开启两项 direct anchor 时，有效目标为
`L_total=0.5*L_cycle_caption + 0.5*L_cycle_segmentation + lambda_direct(step)*L_direct_GRPO + 0.02*L_direct_mask_CE + existing auxiliary losses`。
标量日志 `supervised_anchors/direct_loss_weight_{effective,target}`、
`supervised_anchors/direct_mask_ce_{weight,samples,loss}`、`direct_mask_ce_{positive,no_target}_samples`
必须与 direct single-mask/no-target reward
指标共同检查。GT-mask CE 是原论文 CycleGRPO 之外的显式外部有监督消融，不能称为 image-mask-only
cycle self-supervision。

三流实验中，主 20k 混合数据只贡献上式的 cycle caption/localization 项及其 OPSD auxiliary 项；它不会
产生 direct GRPO、SFT 或 DLC-QA reward。20k RefCOCO 加 20k gRefCOCO no-target 的独立 loader 只贡献
`lambda_direct(step)*L_direct_GRPO + L_direct_mask_CE`；DLC-QA 独立 loader 只贡献
`caption_qa.loss_weight*L_DLC_QA_GRPO`。三条梯度仍在每个主 step 的单次 optimizer step 前累计。
三个 `StatefulDataLoader` state 分别保存为主 `dataloader.pt` 与 checkpoint 内
`auxiliary_dataloaders.pt`，所以 resume 不会重置 40k/10k 流的采样位置。该三流外部监督是论文
image-mask-only CycleGRPO 之外的实验扩展。

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
| `README.md` | CycleGRPO 项目入口、环境/数据重建说明、四个 routing 70k 训练入口、评测命令、公开结果和路径占位符 |
| `README_EasyR1.md` | 上游 EasyR1/veRL 框架说明 |
| `tools/multinode/launch_four_trials.sh` / `tools/multinode/launch_four_cycle_trials.sh` | 四组纯 20k 两节点 Ray controller；读取 Ray 地址/namespace/试验 env，在就绪超时内预检每组 16 Ray GPU，以导出 env 启动并记录实际 trainer launcher PID；后者是面向用户的明确 pure-cycle 入口，不创建 SSH head/worker |
| `tools/multinode/clusters.tsv.example` | 纯自监督四组两节点 Ray 清单模板；填写平台 Ray 地址、namespace 和试验 env 文件 |
| `tools/multinode/launch_four_supervised_trials.sh` | 四组混合任务 controller；按 env 预检 2x8 或 1x7 Ray GPU 拓扑，仅 judge-enabled trial 在物理 GPU 7 启动本机 Llama，并支持 launch/status/stop 与 dry-run |
| `tools/multinode/supervised_clusters.tsv.example` | 混合四组清单模板；无 judge 行使用 `-`，其余填写本机 Llama URL、Ray 地址/namespace 和 env 文件 |
| `tools/multinode/local_llama_judge.py` / `tools/multinode/llama3_chat_template.jinja` | 在预期数量的 Ray 节点上用 CPU-only node-affine detached actor 管理 GPU 7 的本机 Llama vLLM；当前有监督 controller 要求单节点，健康检查要求 `/v1/models` 包含指定 served model，模板固定 Llama-3.1 对话格式 |
| `tools/cuda_keepalive.py` | 训练成功退出后的可选 CUDA 空闲卡监测/保活工具；用 `nvidia-smi` 监测整卡总显存，仅在低于 1 MiB 时为该可见卡预留约 40000 MiB，收到 SIGTERM/SIGINT 后释放 |
| `tools/gpu_power_hold.sh` | 显式 GPU 压力占用工具；默认仅对 GPU 0--3 各启动一个独立 worker，预留约 40000 MiB 并持续 BF16 矩阵乘以维持 GPU 利用率/功耗。支持 `start`、`status`、`stop`，只按自身 worker tag 停止 PID，绝不扫描或终止其他 CUDA 进程 |
| `tools/reassemble_fsdp_checkpoint.py` | CPU/Gloo 离线 FSDP checkpoint 重组器；按 checkpoint 文件名发现并启动同等数量 CPU rank，逐参数收集原 world-size shard 后在 rank 0 还原完整 state dict、复制 processor/config 并写出 HF safetensors。用于训练 world size 无法同时获得足量 GPU 时的评测导出 |
| `tools/run_official_cyclegrpo_keepalive.sh` | 调用未修改官方 CycleGRPO 训练入口；仅训练成功退出后启动 CUDA 保活工具，训练失败保留原退出码 |
| `tools/train_selfsupervised_40k_teacher_8gpu.sh` | 40k scaling 纯自监督 OPSD/teacher 单节点八卡入口；batch 128，pixel-empty 二值 no-target 惩罚与正例空 mask 惩罚开启，不启用 direct/DLC-QA |
| `tools/train_selfsupervised_80k_teacher_8gpu.sh` | 80k scaling 纯自监督 OPSD/teacher 单节点八卡入口；启动时原子合并 40k cycle 与 40k segmentation parquet，batch 128，pixel-empty 二值 no-target 惩罚与正例空 mask 惩罚开启 |
| `tools/train_selfsupervised_20k_egca_4gpu.sh` | 当前服务器四卡 EGCA 纯自监督入口；读取 20k raw cycle parquet，batch 128、1 epoch，开启 teacher/OPSD/pixel-empty/正例空 mask 惩罚，每 5 step 保存且最多保留 2 个 checkpoint，结束后启动 CUDA 保活 |
| `tools/train_selfsupervised_20k_opsd_all_samples_4gpu.sh` | 四卡 routing 消融入口；使用历史 20k 配置，关闭 SECA/EGCA，跳过 R_Ci 三路由分类并将全部 image-cycle 样本送入 OPSD privileged correction，保留原始 GRPO，结束后启动 CUDA 保活 |
| `tools/train_selfsupervised_20k_egca_weighted_ce_8gpu.sh` / `tools/train_selfsupervised_20k_egca_weighted_ce_4gpu_tmp.sh` | 新版 EGCA 20k 自监督入口与本机临时 smoke wrapper；生产脚本默认 8 卡、batch 128、1 epoch，rollout evidence 只生成非负 GT self-distillation CE 权重，临时 wrapper 覆盖为 4 卡/1 step 且不启动保活 |
| `tools/train_supervised_70k_common_8gpu.sh` | 70k 混合训练的共享启动器；默认校验 20k/30k/10k/10k，缩量 wrapper 下按固定种子和 cycle source 分层生成 25%/50% parquet 与匹配的 DLC-QA JSONL，校验行数；启动 7-GPU Ray head 与 GPU 7 的本地 Llama judge，传递历史 batch/anchor 配置，退出时清理自有 Ray session 并请求八卡 hold |
| `tools/train_supervised_70k_baseline_8gpu.sh` | 70k 历史 OPSD+有监督 baseline wrapper；SECA 和 cycle-only SECA 均关闭 |
| `tools/train_supervised_70k_baseline_25pct_8gpu.sh` / `tools/train_supervised_70k_baseline_50pct_8gpu.sh` | 历史 baseline 的全流数据量消融 wrapper；分别使用 25%/50% 分层子集和 45/90 默认训练步数，独立 run 名称、Ray/judge 端口与 Ray 临时目录 |
| `tools/train_supervised_70k_baseline_25pct_4gpu_tmp.sh` | 25% baseline 的本机四卡一步 smoke wrapper；GPU 0--2 训练、GPU 3 judge，batch `114/228/57`，独立端口和日志，不启动结束占卡 |
| `tools/train_supervised_70k_seca_8gpu.sh` | 70k 历史 OPSD+有监督 SECA wrapper；SECA evidence gate 与 direct mask CE token credit 开启，其他训练配置与 baseline 相同 |
| `tools/train_supervised_70k_seca_routing_l030_h085_8gpu.sh` / `tools/train_supervised_70k_seca_routing_l070_h085_8gpu.sh` / `tools/train_supervised_70k_seca_routing_l050_h065_8gpu.sh` / `tools/train_supervised_70k_seca_routing_l050_h100_8gpu.sh` | 基于 SECA 正式 70k wrapper 的四个 routing 阈值单因素消融；分别覆盖 `(low,high)=(0.30,0.85)/(0.70,0.85)/(0.50,0.65)/(0.50,1.00)`，其余配置不变 |
| `tools/train_supervised_70k_seca_routing_l030_h085_4gpu_tmp.sh` | `l030/h085` routing 变体的本机四卡一步 smoke wrapper；复用完整 70k SECA 数据和算法配置，GPU 0--2 训练、GPU 3 judge，batch `114/228/57`，不启动结束占卡 |
| `tools/demo_refcoco_compare_1000.sh` / `tools/demo_three_capabilities.sh` | 学生演示启动器；在同一 RefCOCO 1k 子集上比较 SAMTok/Ours，及展示最佳 ours mask captioning/VQA；推理期间 GPU 1--3 运行 hold，正常或异常退出时恢复四卡 hold |
| `tools/demo_refcoco_compare.sh` / `tools/demo_three_capabilities.sh` | 精简演示启动入口；静默调用 GPU hold helper，仅在发现 GPU 正被其他计算占用时输出一条通用提示，不打印逐卡状态表 |
| `tools/record_pixel_opsd_demo.sh` | 录屏用单图三任务启动器；读取 `logs/pixel_opsd_recording_demo/inputs/` 的三个 JSON 任务描述，在一个模型进程中运行指代分割、region captioning 和 general VQA，并把 JSON/PNG 结果写入对应 `outputs/` 目录；模型、VQ-SAM2 和 CUDA 设备可由环境变量覆盖 |
| `tools/train_supervised_70k_baseline_4gpu_tmp.sh` / `tools/train_supervised_70k_seca_4gpu_tmp.sh` | 本机临时 smoke wrapper；使用 CUDA 0--2 训练、CUDA 3 本地 Llama judge，batch `114/228/57` 仅跑 1 step，不启动结束占卡 |
| `tools/patch_official_final_validation.py` | 对官方 CycleGRPO trainer 做幂等的最小补丁，使 `trainer.val_freq<=0` 时跳过训练结束后的通用 validation |
| `experiments/pegc_ablation_20260902/tools/eval_direct_grpo_refusal_suite.sh` | 探索副本两版 direct-GRPO/Refusal Credit checkpoint 的四 bench 串行评测；使用本地 Llama-3.1 8B 评分 DLC，汇总 GRES 的 T_acc/N_acc/gIoU/cIoU，并在完成后恢复 GPU 占卡
| `experiments/pegc_ablation_20260902/tools/train_evidence_mask_credit_cycle_penalty_only_1of10_epoch1.sh` | 探索分支 Evidence+Mask Credit 的 cycle-only 正样本 no-target 惩罚对照；cycle=1.0，direct=0.0 |
| `TRAIN.md` | 旧的单/多节点 cold-start SFT 环境备忘，路径具有内部环境痕迹 |
| `setup.py` / `pyproject.toml` | 将仓库安装为 `verl`；ruff 规则和 Python `>=3.9` |
| `requirements.txt` | CUDA/PyTorch 之外的核心依赖；包括 VQ-SAM2/RefCOCO 转换所需的 Hydra、iopath、COCO RLE、COCO caption 评价和 torchvision；NumPy 限制在 2 以下以兼容当前 W&B，Transformers 锁定 `4.54-4.57`，vLLM `>=0.8` |
| `Makefile` | 上游开发命令 |
| `paper/iclr2027/pixelopsd/build_pdf.sh` | 论文构建入口；在论文目录执行三遍 pdflatex/BibTeX 并输出最终 PDF 页数 |
| `paper/iclr2027/pixelopsd_new/Pixel-OPSD-final/` | 由作者提供的 LaTeX 论文包整理出的可直接编译项目；包含 Figure 4 的真实 GroundingSuite 定性对比、`make_qualitative_figure.py` 图像生成器、样本来源说明和 `build_pdf.sh` |
| `tests/test_opsd_core.py` / `tests/test_tokenizer.py` / `tests/test_gres_subset_metrics.py` / `tests/test_no_target_reward.py` / `tests/test_balanced_cycle_dataset.py` / `tests/test_grefcoco_cycle_dataset.py` / `tests/test_dam_caption_qa.py` / `tests/test_supervised_anchors.py` / `tests/test_first_mask_diagnostic.py` / `tests/test_seca.py` / `tests/test_egca.py` | 无 GPU 单元测试；覆盖 OPSD、processor、GRES 指标、no-target、混合配额、gRefCOCO 排除抽样、DAM QA schema、anchor 配置边界、SECA 兼容路径和动态 EGCA Shapley/证据 gate |

### 5.2 `verl/`：RL 引擎

| 模块 | 实现职责 |
|---|---|
| `protocol.py` | `DataProto`：tensor/non-tensor/meta 三类数据的 select、union、repeat、concat、chunk、Ray 序列化 |
| `trainer/main.py` | CLI/YAML 配置合并、Ray runner、主 CycleGRPO 与独立 RefCOCO/DLC-QA loader、worker/reward/trainer 组装 |
| `trainer/ray_trainer.py` | Cycle/non-cycle 分流、双阶段 rollout、三条独立监督流的 reward/advantage 与梯度累积、验证与 checkpoint |
| `trainer/ray_trainer_old.py` | 上游/旧训练循环，仅供对照，不是主入口 |
| `trainer/core_algos.py` | GAE、GRPO、RLOO、ReMax、REINFORCE++，PPO clip loss、KL/value loss |
| `trainer/data_loader.py` | 主 train/val 与可恢复的独立辅助 train `RLHFDataset`、sampler/DataLoader |
| `trainer/metrics.py` | reward、length、timing、throughput 指标汇总 |
| `workers/fsdp_workers.py` | actor/ref/critic 构建，FSDP-vLLM 权重切换，多模态前处理，rollout 后 token/tIoU 评分，以及 regenerate/direct-mask CE/EGCA weighted self-distillation 的独立梯度累积调用 |
| `workers/actor/dp_actor.py` | log-prob 前向、动态 micro-batch、PPO loss、独立 caption anchor KL、命名的 teacher-forcing CE、梯度累积和 optimizer step |
| `workers/critic/dp_critic.py` | GAE/PPO 可选 value model；GRPO 主配置通常不启用 critic |
| `workers/rollout/vllm_rollout_spmd.py` | SPMD vLLM engine、采样参数、视觉输入和 response tensor 构造；caption task 动态以 logit bias 屏蔽 SAMTok/object-reference vocabulary |
| `workers/sharding_manager/fsdp_vllm.py` | FSDP 参数与 vLLM engine 同步/offload |
| `workers/sharding_manager/fsdp_ulysses.py` | sequence parallel 数据切分/还原 |
| `workers/reward/function.py` | 动态加载 sequential/batch 自定义 reward 并写 token-level score |
| `workers/opsd/config.py` | pixel IoU、路由、caption safety、EMA teacher、regenerate、distillation、SECA 自监督/兼容路径与动态 EGCA 配置及边界校验 |
| `workers/opsd/egca.py` | 动态 EGCA 的合法 SAMTok group 解析、coarse/fine Shapley counterfactual credit、decoded evidence gate、非负完整 group teacher-forcing 权重、legacy 对比式 token credit 与 OPD sample-weight gate |
| `workers/opsd/seca.py` | 统一的 SECA 空间证据信用分配：自监督/混合训练的 mid-route JSD evidence gate 与 direct mask CE hierarchical token credit；由 `worker.opsd.seca.enabled` 和自监督子开关控制 |
| `workers/opsd/distillation.py` | response-token 分块的 checkpointed generalized-JSD、teacher 置信度权重、caption 分割 special-token vocab 屏蔽和 distillation metrics |
| `analysis/opsd_vs_grpo_diagnostic/extract_signals.py` / `plot_signals.py` | 共同固定 rollout 的真实 GRPO/OPSD autograd token 梯度提取、相同行热力图和 local-gradient 二维诊断；缺少 logits 时拒绝生成图 |
| `analysis/opsd_vs_grpo_diagnostic/plot_dual_proxy.py` | 从真实 GRPO-only 与 OPSD experiment logs 生成双方法 rollout/logged-loss 代理图；渲染 12 个候选布局并选择单一双路径版本 |
| `analysis/opsd_vs_grpo_diagnostic/plot_ab_enhanced.py` | 从真实日志代理信号绘制增强双热力图，采用多级发散色阶和最小机制标注 |
| `analysis/opsd_vs_grpo_diagnostic/plot_paired_update_direction.py` | 从共同 checkpoint/fixed rollout 的实测 sampled-token `delta_logp` 生成 GRPO/OPSD 配对更新方向；缺少输入时阻止生成 |
| `analysis/token_update_scatter/extract_token_updates.py` / `estimate_token_contribution.py` | 在固定 disjoint RefCOCO 样本上对真实 HF checkpoint 做 teacher-forced mask-code log-prob 提取，并计算显式 teacher-alignment 横轴；不启动训练 |
| `analysis/token_update_scatter/plot_token_updates.py` | 以共同点、共同范围和实际 checkpoint `Δlog p` 渲染二维/三维 token-update 诊断图 |
| `analysis/token_update_scatter/implementation_audit.md` / `caption.tex` / `README_zh.md` / `config.yaml` | 记录 GRPO/OPSD 源码审计、坐标单位、可靠性定义、checkpoint/data provenance、限制和复现命令 |
| `workers/opsd/mask_iou.py` | 完整/合法 SAMTok group 解析和计数、原始 GT 转换、可复用 image embedding 的批量 `union`/`first_mask` 解码、尺寸恢复和像素 IoU |
| `workers/opsd/routing.py` | `R_Ci` 聚合、三路由边界、caption 特殊 token/JSON/长度安全检查、原始 GRPO 启用判定、packed mask context、GT/reconstruction teacher crop 构造、route 权重与泄漏过滤 |
| `models/monkey_patch.py` | 为多种 HF MLLM 注册 flash attention 和混合多模态 forward |
| `models/transformers/*.py` | Qwen2/3-VL、Qwen3.5、Gemma4 的 RoPE、embedding 与 forward 适配 |
| `single_controller/` | Ray worker、worker group、注册装饰器、资源/dispatch 管理 |
| `utils/dataset.py` | 本项目数据 schema、图像/视频处理、可配置 current/source-aware caption 与 official/RefCOCO/GroundingSuite localization prompt 构建和过滤 |
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
| `qwen3vl_4b_refcoco10k_volcengine.sh` | 火山引擎默认单节点 8 卡、可显式连接平台 1 或 2 节点 Ray cluster 的 OPSD 入口；pure controller 使用 `2 nodes / >=16 training GPUs`，有监督 controller 使用 `1 node / >=7 training GPUs`、`NUM_GPUS=7` 和本机 GPU 7 judge；`LOCAL_JUDGE_TRAIN_GPUS` 允许本地 smoke 参数化保留卡数；`DIRECT_TRAIN_DATA`/`DIRECT_BATCH_SIZE` 与 `CAPTION_QA_TRAIN_DATA`/`CAPTION_QA_BATCH_SIZE` 建立独立外部监督流 |
| `tools/train_selfsupervised_40k_teacher_8gpu.sh` / `tools/train_selfsupervised_80k_teacher_8gpu.sh` | scaling 纯自监督八卡入口，分别使用 40k cycle parquet 与 40k cycle + 40k segmentation 的 80k 合并 parquet；统一 batch 128 和 teacher/pixel-empty 配置 |
| `experiments/multinode/trial_01.env` 至 `trial_04.env` | 四个可审计纯 20k 两节点 H20 模板；GPU 0--7 全部训练、统一 `128/156/256`，分别测试 official/refcoco prompt 与 first/union decode |
| `experiments/multinode/supervised_trial_01.env` 至 `supervised_trial_04.env` | 四个混合任务 env：pixel-empty 20k、pixel-empty 70k（历史配置 main/direct/QA=`128/256/64`、7+1）、official-bbox 20k 和 official source-aware prompt 20k；20k 使用单节点 8 GPU、batch 128/156 step，70k 使用 1x7+GPU7 Llama、156 step，均保持 G=K=6 与 response 256 |
| `config.yaml` | CycleGRPO 的 data/algorithm/worker/reward/trainer 配置 |
| `format_prompt/non_thinking.jinja` | 原样输出 prompt；主入口使用 |
| `format_prompt/r1v.jinja` | 旧的 think/answer 包装模板 |
| `reward_function/text2mask.py` | 图像 mask、bbox、视频时间段、GCG/PSG/no-target 的总奖励路由 |
| `reward_function/tg_reward.py` | temporal grounding 可组合奖励库 |
| `reward_function/llm_judge_reward.py` | 可选外部 vLLM caption judge；包含无图 DLC-QA option judge，不占用训练 GPU |

`projects/eval/qwen3vl_4b_volcengine.sh` 是评测编排入口，支持 FSDP actor 导出及 RefCOCO、GroundingSuite、GRES/gRefCOCO、DLC-Bench 的服务器路径、Conda/Ray 环境隔离和输出目录约定。三项分割评测都接受 `MASK_PROTOCOL=legacy_union|first_mask`；默认 `legacy_union` 用完整合法 depth-2 group 的解码 union 复现历史 SAMTok baseline，`first_mask` 才会在第一个 `<|mt_end|>` 终止并只解码首组，供严格单 mask 格式实验单独报告。RefCOCO 额外支持仅用于与公开 CycleGRPO 脚本交叉验证的 `cyclegrpo_legacy`：它强制单样本、128-token、`skip_special_tokens=True`，按原脚本从 raw `<|mt_####|>` 片段配对、修复奇数 token，并允许第二码本无效时传入 `-1`。这不是标准 benchmark 协议，不能与 `legacy_union` 或 `first_mask` 的结果直接比较，也不提供给 GRES/GroundingSuite。逐样本 JSON 会保存 `mask_protocol`，resume 时协议不同或旧结果缺失该字段会自动重新推理，禁止混合两种口径；兼容模式另写 raw/decoded token 数和实际生成参数，必须使用新的输出目录。GRES 默认标注根为服务器实际目录 `${BASE_DIR}/gRefCOCO`；也可通过 `GRES_ROOT` 覆盖。GRES action 通过 `GRES_REFS_FILE`、`GRES_INSTANCES_FILE` 和 `GRES_IMAGE_ROOT` 指定官方标注与 COCO 图像，先生成 `gres_<split>_samples.json`，再将逐样本预测放到 `EVAL_ROOT/gres/`，最终指标写入 `EVAL_ROOT/gres_metrics.json`。

`projects/rl/datasets/` 全部是离线数据工具，不在 trainer 内自动运行：

- `prepare_dw_rl_dataset.py` / `prepare_dw_single_rl_dataset.py`：DenseWorld 多目标/单目标转 RL parquet，构造区域叠加图、caption/seg prompt 和 mask token。
- `prepare_grefcoco_cycle_dataset.py`：从 gRefCOCO `train` 按 seed 分层抽取 single/multi positive 与 `ann_id=[-1]` no-target 表达；可通过一个或多个既有 parquet 按 COCO 图像和规范化 expression 或 union-mask RLE 排除已使用记录。正样本合并多个 COCO instance mask 并编码为 `grefcoco_cycle`，no-target 保留为 `gres_no_target`，写出正样本、no-target、合并训练 parquet 与类别清单。gRefCOCO 不包含 part mask，清单明确记录 `part_instance=0`。
- `prepare_paco_lvis_part_cycle_dataset.py`：从 PACO-LVIS train 的 `id != obj_ann_id` annotation 确定性抽取真 part mask；由于 PACO v1 不提供逐 mask part label，同图同 parent object 类别的所有 part mask 取 union、每图最多保留一个 query，写入 `paco_part_cycle` parquet 与 parent-category manifest。
- `prepare_cocostuff_cycle_dataset.py`：从 COCO-Stuff 官方 stuffthingmaps 的真 Stuff 类别区域构造 `cocostuff_cycle` parquet；仅接受 PNG 值 91..181，并写入 canonical semantic-label `grounding_query`。
- `grounding_queries.py`：COCO-Stuff 官方 91 类 PNG-to-label 映射，以及 Stuff/PACO label-template query 的唯一构造器；不读取模型、图像或评测数据。
- `prepare_dam_cycle_dataset.py`：从 DAM `COCOStuff`/`PACO` 的 `mask_rle + caption` annotation 构造 DAM-backed `cocostuff_cycle` 或 `paco_part_cycle` parquet；PACO 与官方 part annotation 交叉校验，caption 写入独立 manifest。
- `generate_dam_caption_qa.py`：从 DAM caption manifest 离线生成、LLM 验证并可恢复写出 text-only DLC 风格 QA；可额外导出 DLC judge 兼容的 QA/class-name JSON，不读取图像且不修改训练数据或 reward。`--stop-token-id` 可将特定 vLLM token 作为生成/验证的请求级结束 token；Llama-3.1 使用 `128009`（`<|eot_id|>`）。
- `prepare_balanced_cyclegrpo_dataset.py`：按可配置配额抽取；默认 `8k/4k/5k/2k/1k`，multi 仅接受多实例 gRefCOCO，QA manifest ID 强制优先纳入 DAM 配额。
- `prepare_refcoco_rl_dataset.py`：标准 RefCOCO train split 转单目标 CycleGRPO parquet；编码 mask token、保留原始 RLE，并写出独立 `grounding_query`。
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
| `qwen3vl_4b_volcengine.sh` | export-only FSDP actor 转 HF safetensors，并顺序启动标准 RefCOCO、GroundingSuite mask GIoU、GRES/gRefCOCO 与 DLC-Bench prediction 生成；默认使用当前 `/volume/ybo/xyc` 服务器路径、7 卡评测和独立 cuda:7 judge |

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
| `gres/` | `qwen3vl_gres_eval.py` 从官方 gRefCOCO refs/instances 生成评测清单，解码 mask token、保存可恢复 shard，并计算全量与可选 JSONL 子集 gIoU/cIoU/N-acc/T-acc；分割生成默认最多 256 个新 token（可通过 `GRES_MAX_NEW_TOKENS` 覆盖）；mask 解析与 VQ-SAM2 构造共享模块级 `CODEBOOK_SIZE=256`、`CODEBOOK_DEPTH=2`，因此推理分片在首次生成 mask 时不会依赖 `main()` 局部变量；`subset_metrics.py` 复用官方 empty-target cIoU 语义，提供无模型依赖的累积器、multi annotation 数量、GT 面积分桶和 two-instance member coverage/geometry 分组；`run_gres_multigpu.sh` 负责多 GPU 分片和完整性检查 |
| `mask_protocol.py` | RefCOCO、GRES 和 GroundingSuite 共用的严格离线 SAMTok 协议：`legacy_union` 保留全部完整、codebook 合法的 depth-2 group 并 union，`first_mask` 仅保留首组且在生成时将 `<|mt_end|>` 加入 EOS；另提供仅供 RefCOCO 调用的公开 CycleGRPO raw-token 兼容解析 helper |
| `mr_baselines/` | 五个本地基线的 MR sampled 推理适配器（Sa2VA、PaDT、UniPixel、InstructSeg、EVF-SAM）、SAMTok/图片预检、PaDT RLE 转换、保留占卡显存的单模型启动器、独立收尾监控及严格全量 RLE scorer；四个推理适配器支持按全局样本序号从中断点续跑，续跑写入独立目录；`finalize_resumed.py` 在各后缀成功退出后逐条校验顺序、严格评分、填表并请求四卡 GPU hold |
| `refcoco/` | 标准 RefCOCO 的 `instances.json`/`refs(unc).p` 多 GPU 分片推理和 cIoU/mIoU 汇总；默认 `legacy_union` 解码全部合法 group，显式 `first_mask` 才在首个 `<|mt_end|>` 终止并只解码首组。`cyclegrpo_legacy` 是单独的公开脚本兼容模式，强制 batch 1、128 token、special-token skip 与宽松 raw-token parser，不得作为标准分数和严格协议对比。其他模式每个 GPU 的 VLM generation 通过 `EVAL_BATCH_SIZE` 批处理，默认 16，并固定使用 decoder-only 模型所需的 tokenizer left padding；逐样本 JSON 保存 group 数、协议与实际生成参数，协议不匹配时会重新生成。`demo_compare_1000.py` 在固定 seed 下抽取同一组 1,000 个 RefCOCO val 样本，按普通表达分割 SAMTok/Ours、实时显示百分比进度、保存实测 cIoU/mIoU、最大逐样本 IoU 增益对比图；`demo_three_capabilities.py` 从该预测目录选取 Ours IoU 最高样本，使用预测 mask token 做 mask-conditioned captioning，并在原图上运行传统 VQA。`record_three_tasks.py` 是独立的录屏入口：读取三个都指向同一图像的 JSON 输入，单进程依次执行 referring segmentation、region captioning、general VQA，输出三份结构化 JSON、预测/GT mask 和三张大字 PNG；固定录屏案例为 RefCOCO case `115981`（已有成功 IoU 约 `0.9747`） |
| `groundingsuite/` | Qwen3-VL 推理、按 task 分片和自动合并；支持显式 data root 与可选 COCO 图像根；分割生成默认上限为 256（可通过 `GROUNDINGSUITE_MAX_NEW_TOKENS` 覆盖），默认 `legacy_union`，可显式切为严格 `first_mask`，逐样本 JSON 保存协议且不打印逐样本 response |
| `gcg/` | 生成 interleaved text-mask，解码 mask 并保存 RLE/文本供官方 GCG 指标；数据根需替换 |
| `gar/` | VQA 和 detailed caption 两个推理入口；`gar_vqa_metrics.py` 汇总总体与属性类别准确率 |
| `dlc_bench/` | 多后端 caption inference、裁剪/区域输入、judge server、GPT-with-image/Llama-without-image 评测和绘图；Qwen3-VL 推理使用训练同构的正向 caption prompt、192-token 上限和 caption-only special-token logits blocker，另写 `.stats.json` 记录 leak rate |
| `bbox/` | Qwen2.5/3/3.5、InternVL、Gemma、Llama 的 bbox 输出泛化；解析 `[x1,y1,x2,y2]` 并按 0-1000 坐标还原 |

无图 Llama judge 通过 `eval_llama_without_image.py` 调用 OpenAI-compatible vLLM。对于未在 tokenizer 内声明 chat template 的 Llama-3.1 HF 权重，服务端须显式提供 Llama 3 chat template；评测器还必须传入 `stop_token_ids=[128009]`，将 `<|eot_id|>` 视为每道选择题的结束 token。单题只需输出一个选项，生成上限为 16 token，避免在正确答案后继续生成下一轮 `assistant` header。

`refcoco/demo_three_capabilities.py` 的 mask-captioning 与传统 VQA 仅在可视化和 `result.json` 中保留模型最终回答；会清除 `<think>...</think>` 推理块和孤立 think 标签，若清理后没有最终文本则显式报错，避免生成空白能力展示图。

录屏专用 `record_three_tasks.py` 将输入与输出都渲染为横向大字展板，按实际文本测量高度，
保留原图比例。caption/VQA 的 `answer` 和展板仅显示清理 think 标签后的最终文本，原始响应仍保存到
`raw_response`；无合法非空分割或无最终文本会报错。终端分三步报告进度，结束后报告一个质量指标
（单例真实像素 IoU）和两个文本非空检查，后者不是语义准确率。`summary.json` 记录实际 checkpoint、
UTC 开始时间、耗时和样例范围，`outputs/results.txt` 便于翻阅，shell 入口将完整 stdout/stderr 同时写入
`evaluation.log` 并保留 Python 退出码。固定 case 是从历史成功案例选出的定性展示，不能代表全量测试集；
region captioning 的蓝色 mask 是已保存的 SAMTok 区域输入，其 token 进入 prompt，GT mask 仅供分割 IoU。
该入口仍使用原有短分割 prompt 和 `legacy_union`，不改变论文训练或标准 benchmark 主执行路径。

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
15. **火山引擎注入的 Ray 集群与项目环境不兼容。** 当前平台集群使用 Python 3.12 / Ray 2.53，而项目 Conda 环境使用 Python 3.10 / Ray 2.56；默认单机入口必须清除继承的 `RAY_ADDRESS`，由当前解释器启动本地 Ray。两节点模式只能连接用同一 `$ENV_DIR/bin/ray` 创建的私有集群，八台训练节点的 Python/Ray/Torch/vLLM 版本必须一致，不能仅降级 Ray 而保留不同 Python 版本。Ray 的 `RAY_TMPDIR` 不能直接使用仓库长路径，否则 `session_*/sockets/plasma_store` 会超过 Linux `AF_UNIX` 的 107 字节限制；它也不能链接到使用率不低于 95% 的 workspace。多机控制器为每个节点使用独立短路径 `/dev/shm/cgrpo-<trial>-{h,w}`，入口仍检查临时盘利用率。
16. **RefCOCO parquet 的图像路径必须与当前服务器一致。** `images` 保存的是绝对路径；跨服务器复制 parquet 后必须重新导出或修复该列。火山引擎入口在初始化 Ray/FSDP/vLLM 前逐条验证 `images`，避免模型全部加载后才由 DataLoader 抛出 `FileNotFoundError`。
17. **Qwen3-VL checkpoint 必须使用复合 processor。** 自定义导出的 checkpoint 可能缺少让 `AutoProcessor` 识别 `Qwen3VLProcessor` 的元数据；loader 会根据 `config.json` 的 `model_type=qwen3_vl` 显式回退。若 `config.json` 也缺失或模型类型错误，必须先修正 checkpoint 元数据，不能用 tokenizer 或 image processor 代替，否则多模态 prompt 无法展开。
18. **FSDP checkpoint 不是可直接评测的 HF 模型。** `actor/huggingface/` 仅保存 config/generation config/processor；必须使用与保存 world size 相同的 export-only FSDP worker 恢复 shard 后导出。不要把原 cold-start `MODEL_PATH` 当作训练后模型传给评测脚本。
19. **caption safety 是当前 OPSD 的稳定化消融。** 它在 IoU 路由之后排除特殊 token、`mask_2d` JSON 和超长 caption 对原始 GRPO/mid JSD 的影响，并把它们导向 regenerate；这不改变论文的单 actor 双任务设计、privileged prompt 或 JSD 公式。比较该消融与历史实验时，必须同时报告 `CAPTION_MAX_RESPONSE_LENGTH` 和安全指标，不能仅比较最终 benchmark 分数。
20. **B 保留原始 GRPO 是另一项受控消融。** `PRESERVE_ORIGINAL_GRPO=true` 使低/中路由的 teacher CE/JSD 成为额外梯度，而非替代原 CycleGRPO caption 梯度；这会改变 caption 梯度总量和与 teacher 的相对权重，不能与 route-replacement 结果直接混合。必须检查 `caption_original_grpo_active_rate` 是否接近 `caption_safe_rate`，否则说明安全门控或 batch 组合没有按预期生效。
21. **C 当前同时处理 special-token 支持集、reference anchor、teacher 特权信息形态和共享梯度冲突。** JSD 屏蔽和 caption anchor KL 能阻止特权 token 分布写入 caption、并将安全 caption 拉回 frozen SAMTok。C2 保留 GT/reconstruction 的诊断信息，但仅以全图、GT crop 和 reconstruction crop 传给 teacher，不把 IoU、几何或 raw mask 文本写进 teacher prompt；student 不会看到这些图。为控制已观察到的纯 CycleGRPO text-to-mask 遗忘，C2 以 segmentation anchor KL 约束全部 cycle localization response，并可用非对称梯度投影移除 caption-side gradient 中与 localization gradient 冲突的分量；两者都不把 `seg_answer` 作为 student CE target，保持单 actor 的 cycle-only 训练信号。投影会额外保留一份本 rank 的 caption gradient，因此增加约一个 FSDP gradient shard 的显存；系数和冲突率必须通过 10-step RefCOCO/GroundingSuite 消融验证。屏蔽词表的实现不得把 logits 设为 `-inf` 后直接参与 entropy/JSD；必须保留有限 log-probability，且任何非有限 JSD 或 actor gradient 都必须 fail-fast，不能静默跳过 optimizer step。
22. **groundedness verifier 已从当前训练链路移除。** 当前 caption reward、regenerate、privileged JSD 与 DLC-QA 不再读取 groundedness 字段，也不会启动 groundedness verifier rollout、claim penalty、candidate/JSD gate 或 groundedness token mask/weight。保留的 teacher analysis、caption safety 和 teacher confidence 仍按各自配置运行；历史变更日志中的 groundedness 记录仅用于保留实验历史，不能视为当前可用配置。
23. **GRES/gRefCOCO 评测需要独立标注根目录。** `projects/eval/qwen3vl_4b_volcengine.sh gres` 不使用训练 parquet 作为评测集，而是由 `GRES_REFS_FILE`、`GRES_INSTANCES_FILE` 和 `GRES_IMAGE_ROOT` 生成固定的 `gres_<split>_samples.json`。推理逐样本写入 `EVAL_ROOT/gres/case_*.json`，确认所有 case 完成后才计算 `gres_metrics.json`；因此不能用部分 shard 或只存在旧 prediction 的目录计算 GRES 指标。离线子集报告同样拒绝不完整 case，并使用该固定样本 JSON 的逐项 phrase 对齐来确认官方 refs 的重建顺序；不能把不同 split、不同标注版本或不同评测清单的 case 混用。
24. **正、负样本必须共享完整的 localization prompt 分布。** 若 `Please segment {expression} in this image.` 只用于 no-target caption PPO，会使模型把 RefCOCO/GRES 的评测指令条件化为固定拒识。当前默认正 cycle caption 与 no-target direct segmentation query 都以 1:1 覆盖 RefCOCO/GRES 和 GroundingSuite 模板；也可用 `LOCALIZATION_PROMPT_MODE=refcoco` 让所有训练 localization 使用 RefCOCO 指令，或用 `legacy` 复现旧模板。二者的差别只能是查询内容和奖励，不能是未记录的外层 instruction。该措施只对齐外层 instruction，不能替代带关系表达的正 referring supervision；若开启 `include_positive_sources=true`，必须将其作为使用人工 expression 的外部 anchoring 消融报告。
25. **类别模板不是人工 referring expression。** `include_label_sources=true` 只允许 COCO-Stuff 的完整 semantic category mask 使用 `the {label}`，以及 PACO v1 的同图 parent-category part union 使用 `the visible parts of the {parent}`。它不得使用 COCO 五条全图 caption 直接配对 region mask，也不得把 PACO 的 parent object category 伪装成未提供的细粒度 part label。该开关是额外的 label-template direct grounding 消融，实验报告必须与 RefCOCO/gRefCOCO 人工 expression anchor 分开说明。
26. **正例 segmentation 的 mask 解码可配置。** 在线 CycleGRPO 与 direct reward 默认记录 `mask_group_count`、`valid_mask_group_count`，将一条 response 中全部完整、codebook 合法 group 的 decoded mask union 后计算 IoU 和 `R_Ci`。通过 `MASK_DECODE_MODE=first_mask` 可以恢复原始训练时只解码第一个合法 group 的语义；`union` 是当前默认值。多 group 是原始 CycleGRPO 允许的表达形式，不会被置零或逐组扣分；只有同一完整 group 出现超过三次时，原有 `non_repeat` 一分正则为零。训练日志应检查 `opsd/seg_multi_mask_rate`、`opsd/seg_mean_mask_group_count` 与 direct 对应指标，用于定位退化的重复输出。训练解码模式必须与离线评测协议单独记录，不能混合比较。
27. **三条监督流必须严格隔离。** `data.train_files` 只能是 20k image-mask cycle mix；不得把它传给 `DIRECT_TRAIN_DATA`、`DIRECT_NO_TARGET_TRAIN_DATA` 或 `CAPTION_QA_TRAIN_DATA`。`DIRECT_TRAIN_DATA` 必须是 RefCOCO 人工正 expression（`source=refcoco_cycle`）；启用 no-target direct GRPO/SFT 时，`DIRECT_NO_TARGET_TRAIN_DATA` 必须是 gRefCOCO no-target expression（`source=gres_no_target`），推荐各 20k。`CAPTION_QA_TRAIN_DATA` 必须含全部可 join 的 `dam_source_id`，并与 `CAPTION_QA_JSONL` 一一对应。三条 loader 的 batch size 各自独立，主训练 epoch/step/save cadence 只由 20k loader 决定；resume 必须保留 checkpoint 内 `auxiliary_dataloaders.pt`，否则两条外部流会从头开始。
28. **2:4:1 配额按 parent prompt 而不是生成 response 计数。** 在 `28:56:14`、`G=K=6`、7 个训练 rank 下，每 step 先采样 4 个主 cycle、8 个 RefCOCO direct、2 个 DLC-QA parent prompt/rank；随后主 caption 生成 24 条、main localization 生成 144 条、direct localization 生成 48 条、QA caption 生成 12 条 response/rank。它们的 loss 仍在同一次 optimizer step 累积，但 `caption_loss_weight`、`localization_loss_weight`、direct warmup/CE 权重和 `caption_qa.loss_weight` 继续决定实际梯度尺度，数据配额本身不等价于 loss 等权。`THREE_STREAM_2_4_1_ENABLED=true` 是旧的严格纯三流模式，会强制关闭 teacher routing/regenerate/JSD、caption/segmentation anchor KL 与 caption safety；当前有监督多机 sweep 不开启该 flag，而是让 controller 校验相同的 2:4:1 配额，以保留 v1/v2 routing、EMA teacher 和 teacher diagnosis。
29. **当前服务器 disjoint 诊断环境变量记录。** 固定基础变量为 `BASE_DIR=/volume/ybo/xyc`、`REPO_DIR=/volume/ybo/xyc/CycleGRPO-OPSD`、`ENV_DIR=/volume/ybo/xyc/envs/cyclegrpo`、`MODEL_PATH=/volume/ybo/xyc/Qwen3-VL-4B-SAMTok`，训练 GPU 为 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6`、`NUM_GPUS=7`。主 cycle 数据为 `/volume/ybo/xyc/datasets/cyclegrpo_20k_raw_seed20260820/cyclegrpo_20k_40_20_25_10_5_seed20260820.parquet`；70k 入口的 `DIRECT_TRAIN_DATA` 固定为 `/volume/ybo/xyc/datasets/direct_refcoco30k_disjoint_cycle20k/refcoco_train_30k_disjoint_cycle20k_seed20260823.parquet`，`DIRECT_NO_TARGET_TRAIN_DATA` 固定为 `/volume/ybo/xyc/datasets/grefcoco_no_target_direct_10k/grefcoco_train_0pos_10000notarget_seed20260821_no_target.parquet`；两条 direct 流合计 40k 分割监督样本（30k RefCOCO + 10k no-target）；DLC-QA 通过 `CAPTION_QA_TRAIN_DATA` 与 `CAPTION_QA_JSONL`。两个 direct 文件均须在路径重写后实际存在且保留 `source=refcoco_cycle`/`source=gres_no_target` 契约，启动前必须执行 `test -f`，不能假定历史命名或未核验路径。
30. **混合四组多机试验必须使用五列清单与动态资源契约。** `launch_four_supervised_trials.sh` 的每行是
`trial_id`、`ray_address`、`ray_namespace`、`head_judge_base_url`、`experiment_env`；无 DLC-QA 的 20k 行将
judge URL 填为 `-`。pixel-empty 20k、official-bbox 20k 和 official source-aware prompt 20k 均要求单节点、
8 Ray GPU（GPU 0--7 全训练），不启动 Llama；70k 混合任务要求一台节点、7 Ray GPU（GPU 0--6），
并由 node-affine actor 在 GPU 7 启动 Llama，GPU 8--31 不使用。controller 从 env 读取 `NNODES`、`NUM_GPUS`、
`LOCAL_JUDGE_ENABLED`，按实际拓扑预检 Ray、只对 judge-enabled 行执行 judge start/status/stop，并将 env
完整导出给 trainer。20k 任务保持 batch 128、156 step、G=K=6、response 256；70k 任务消费 20k cycle、
30k direct RefCOCO、10k no-target 和 10k DLC-QA，使用 batch `112/224/56`、179 step、pixel-empty、
direct GRPO/CE/DLC-QA 全开。平台预先提供隔离 Ray cluster；controller 不执行 SSH、`ray start`、`ray stop`
或 NCCL 网卡配置。真实启动前应执行 `launch --dry-run`，确认每个 trial 的节点/GPU/judge 分支。

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

### 2026-08-06 - 修复未设置 MAX_STEPS 时的训练入口启动失败

- 代码：修改 `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`。
- 文档：更新第 2.2 节训练入口的 `MAX_STEPS` 行为。
- 行为：在 `set -u` 下安全展开可选的 `TRAINER_MAX_STEPS_ARG` 数组；`MAX_STEPS` 为空时不再触发 unbound variable，且不会把空参数传给 Hydra。设置正整数时仍传入相同的 `trainer.max_steps` 覆盖。
- 验证：执行 `bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`，并用最小 Bash `set -u` 测试覆盖空数组和设置数组两种展开结果；未在本机运行 GPU/Ray/FSDP 训练。

### 2026-08-06 - 允许将 Ray 临时目录置于短路径本地数据盘

- 代码：修改 `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`。
- 文档：更新第 2.2 节 Ray 临时目录约束。
- 行为：`RAY_SHORT_ROOT` 不再限定在 `/tmp`，接受长度不超过 32 的绝对、非符号链接目录；默认值改为 `/dev/shm/cgrpo-ray-<uid>`。根分区空间不足时也可显式设为本地数据盘短路径（如 `/data5/ray-<uid>`），仍保留 95% 文件系统使用率检查以符合 Ray object spill 的要求。
- 验证：执行 `bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 与 `git diff --check`；未在本机运行 Ray。

### 2026-08-06 - 支持没有 W&B 的 file-only 训练日志

- 代码：修改 `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`。
- 文档：更新第 2.2 节日志配置。
- 行为：入口新增 `TRAINER_LOGGERS` 环境变量，默认仍为 `['file','wandb']`。设置 `TRAINER_LOGGERS='["file"]'` 后仅创建 checkpoint 目录中的 JSONL 与 generation 文件，不导入或调用 W&B，适用于未安装 `wandb` 的离线环境。
- 验证：执行 `bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 与 `git diff --check`；基于 `verl/utils/logger/logger.py` 和 `gen_logger.py` 静态确认 file backend 不依赖 W&B，未运行 GPU 训练。

### 2026-08-06 - 参数化 GroundingSuite 20k 数据混合配额

- 代码：修改 `projects/rl/datasets/prepare_balanced_cyclegrpo_dataset.py`。
- 文档：更新第 2.4、5.3 节。
- 行为：混合器新增 Single、Multi、Stuff、Part 与 no-target 五个可选计数参数，要求总数恒为 20,000；默认 `7k/5k/4k/2k/2k` 保持不变。可用于生成 `8k/4k/4k/3k/1k` 等受控配方，manifest 记录实际配额。
- 验证：执行 Python 语法编译、默认/自定义配额参数解析与 20k 总数校验；本机没有服务器侧 parquet、CUDA 或 VQ-SAM2 权重，未执行完整 token 编码。

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

### 2026-08-01 - 在 20k 混合器中清除正样本的继承 caption 标签

- 代码：修改 `projects/rl/datasets/prepare_balanced_cyclegrpo_dataset.py`。
- 文档：修正第 2.4 节最终 parquet 的数据契约。
- 行为：合并时无条件将四类正样本的 `cap_answer` 置为 `None`，包括旧 RefCOCO/gRefCOCO parquet 中可能遗留的 referring expression；no-target 的表达式仍只存在于 `cap_problem`。因此无需重跑 VQ-SAM2 编码，重新执行合并即可得到图像/mask-only 的 20k 文件。
- 验证：执行混合器 Python 语法编译与 `git diff --check`；本机没有服务器侧 parquet，未执行最终行数和路径校验。

### 2026-08-01 - 参数化训练 checkpoint 保留数量

- 代码：修改 `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`。
- 文档：更新第 2.2 节 checkpoint 保存配置。
- 行为：入口新增正整数环境变量 `SAVE_LIMIT`，默认保持既有的 20；传入 `SAVE_LIMIT=2` 时，训练仍按 `SAVE_FREQ` 创建 checkpoint，但 trainer 只保留最新两个，以控制持久磁盘占用。
- 验证：执行 shell 语法检查与 `git diff --check`；未在服务器执行 FSDP checkpoint 轮转。

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

### 2026-08-09 - 增加 caption groundedness verifier 与 caption 生成 special-token 屏蔽

- 代码：新增 `verl/workers/opsd/groundedness.py`；修改 `verl/workers/opsd/{__init__,config}.py`、`routing.py`、`distillation.py`、`verl/workers/rollout/vllm_rollout_spmd.py`、`verl/workers/fsdp_workers.py`、`verl/trainer/ray_trainer.py`、`verl/workers/reward/function.py`、`projects/rl/reward_function/text2mask.py`、`projects/rl/config.yaml`、`projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`、`evaluation/dlc_bench/inference.py` 和 `tests/test_opsd_core.py`。
- 文档：更新第 2.2、3.6、5.2、5.6、6 节及模块清单。
- 行为：caption、regenerate、privileged groundedness teacher 和 DLC 生成均动态屏蔽 SAMTok mask/object-reference response vocabulary；segmentation rollout 不受影响。frozen initial teacher 使用全图 + GT target crop 对全部有目标 cycle caption 输出原文 claim 的结构化 groundedness verdict；明确 unsupported/contradicted claim 进入 caption reward 惩罚，并过滤不可靠 regenerate CE/JSD teacher target。verifier 结果写入 `caption_groundedness.jsonl`，no-target 保留原拒识分支。首版只启用整句 reward 惩罚，token-level JSD extra weight 接口默认关闭。
- 论文边界：该功能是 caption factuality 的额外受控辅助消融，不是论文原始 CycleGRPO 的 pixel-IoU/cycle objective，也没有增加 RefCOCO CE 或改变 segmentation loss。
- 验证：所有修改 Python 文件通过 `python3 -m py_compile`，训练入口通过 `bash -n`，`git diff --check` 通过；本机 `python3 -m unittest tests.test_opsd_core` 因未安装 `torch` 无法运行，尚未进行服务器 Ray/FSDP/vLLM smoke training。

### 2026-08-09 - 修复 groundedness 与 GRES mixed batch 的合并边界

- 代码：修改 `verl/workers/opsd/config.py`、`verl/trainer/ray_trainer.py`、`projects/rl/config.yaml`、`projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 和 `tests/test_opsd_core.py`。
- 文档：更新第 3.6 节 groundedness 路由与 batch 生命周期说明。
- 行为：groundedness 启用时，mid-route privileged JSD 无条件要求 `R_Ci>=0.65`，不再依赖可关闭的历史 `teacher_confidence` 开关。cycle caption 的 verifier-only non-tensor verdict 与 token mask 在 reward/JSD 已消费后、主 PPO actor 更新前移除，防止与 GRES no-target 合并时 `DataProto.concat` 因两侧字段不对齐失败，并避免特权日志传给 actor worker；segmentation 和 caption PPO 行为不变。
- 验证：修改涉及的 Python 文件已通过 `python3 -m py_compile`，训练 shell 已通过 `bash -n`，`git diff --check` 通过；groundedness parser stdlib smoke test 通过。本机 `python3 -m unittest tests.test_opsd_core` 仍因未安装 `torch` 无法导入，未运行 Ray/FSDP/vLLM 10-step smoke test。

### 2026-08-09 - 将 GRES/gRefCOCO 接入统一离线评测入口

- 代码：修改 `evaluation/gres/qwen3vl_gres_eval.py`、`evaluation/gres/run_gres_multigpu.sh` 和 `projects/eval/qwen3vl_4b_volcengine.sh`。
- 文档：更新第 2.2、5.3.1、5.6、6 节及 GRES 模块职责。
- 行为：新增 `gres` action。评测入口接收官方 gRefCOCO refs/instances 与 COCO 图像根，先生成 split 固定的 inference JSON，再以当前 Conda Python 进行多 GPU 分片推理。预测按 `case_<id>.json` 可恢复保存；所有 case 完成后才汇总 `N_acc`、`T_acc`、gIoU、cIoU 和 target mIoU 到独立 `gres_metrics.json`。metric-only 不依赖 CUDA，并保持原 GRES 对 no-target 误检计入 cIoU union 的语义；缺图或尺寸不匹配会使 shard 显式失败。
- 论文边界：这是新增 GRES/gRefCOCO 下游评测入口，不改变训练、reward、数据配方或论文 CycleGRPO 优化目标。
- 验证：`python3 -m py_compile evaluation/gres/qwen3vl_gres_eval.py`、`bash -n evaluation/gres/run_gres_multigpu.sh`、`bash -n projects/eval/qwen3vl_4b_volcengine.sh` 和 `git diff --check` 通过；本机无 GRES/COCO 数据、CUDA 或模型，未运行端到端 benchmark。

### 2026-08-09 - 记录 groundedness verifier 解析失败原因

- 代码：修改 `verl/workers/opsd/groundedness.py`、`verl/trainer/ray_trainer.py` 和 `tests/test_opsd_core.py`。
- 文档：更新第 3.6 节、关键注意事项 22 及本变更日志。
- 行为：groundedness parser 在不改变 reward 或路由语义的前提下，为失败 verdict 标记顶层解析原因并统计无效 claim 的具体原因。`caption_groundedness.jsonl` 继续完整记录成功 verdict，并且每个 global step 额外写入最多 8 条失败样本，含学生 caption、解析原因、claim 丢弃统计和最多 2048 字符的 verifier 原始输出；no-target/全局禁用行不写入。诊断数据仅保留在 trainer driver 的短暂 batch 生命周期中，不传递至 actor PPO 更新。
- 验证：`python3 -m py_compile verl/workers/opsd/groundedness.py verl/trainer/ray_trainer.py tests/test_opsd_core.py`、groundedness parser stdlib smoke test 与 `git diff --check`；完整 `tests/test_opsd_core.py` 仍依赖本机未安装的 PyTorch，未运行 Ray/FSDP/vLLM smoke training。

### 2026-08-10 - 增加 DAM-backed CycleGRPO 数据转换器

- 代码：新增 `projects/rl/datasets/prepare_dam_cycle_dataset.py`。
- 文档：更新第 2.4、5.3 节，记录 DAM annotation 输入、PACO part 过滤、面积约束、输出 schema 与 caption manifest 边界。
- 行为：从 DAM `COCOStuff` 或 `PACO` annotation 读取 `mask_rle`，解析实际图像并重新编码 VQ-SAM2 mask token，输出原有 `cocostuff_cycle`/`paco_part_cycle` source；PACO 仅保留与官方 `id != obj_ann_id` 交集且每图最多一个 part。`dam_source_id` 进入 parquet 仅用于离线关联，DAM caption 不进入 actor prompt，而写入独立 JSONL 供 DLC QA 构造。
- 验证：执行新增脚本的 `python3 -m py_compile` 与 `git diff --check`；本机没有 DAM/COCO 图片、PyTorch/CUDA 或 SAMTok 权重，未执行实际 mask 编码和 parquet 导出。

### 2026-08-10 - 对齐 gRefCOCO no-target 训练与 GRES 评测 prompt

- 代码：修改 `projects/rl/datasets/prepare_grefcoco_cycle_dataset.py`。
- 文档：更新第 2.4 节及本变更日志。
- 行为：新导出的 `gres_no_target` 样本将输入从显式 no-target 指令改为 `Please segment {expression} in this image.`，与 RefCOCO/GRES 推理的用户指令一致；输出目标仍是 `No target.`，并继续使用未改动的原始 no-target accuracy + no-repeat reward。正样本仍不写入 referring expression，cycle prompt、分割 reward、OPSD/C2 辅助项和评测协议均不变。旧 parquet 已包含旧 prompt，不能用于这个受控消融。
- 验证：对转换器执行无缓存语法解析和常量断言，并执行 `git diff --check`；本机没有 PyTorch、datasets、服务器标注/图像或 CUDA，未运行 VQ-SAM2 parquet 导出和 8-GPU training。

### 2026-08-10 - 增加 GRES 离线子集指标汇总

- 代码：修改 `evaluation/gres/qwen3vl_gres_eval.py`，新增 `evaluation/gres/subset_metrics.py` 和 `tests/test_gres_subset_metrics.py`。
- 文档：更新第 2.2、5.6、关键注意事项 23 及本变更日志。
- 行为：metric-only evaluator 可通过 `--subset-report-file` 写出 JSONL。它重建与保存 `gres_<split>_samples.json` 完全一致的官方 case 顺序，逐条验证 phrase 对齐和预测完整性，再按标注实例数输出 no-target/single-instance/multi-instance，按 GT 像素面积输出 small(<5%)/medium(5%-25%)/large(>=25%)；每行复用全量 GRES 的 cIoU、gIoU、T/N-acc 和 target mIoU 语义。该功能只读取已有 prediction 和标注，不加载模型、不改变推理、训练或论文目标。
- 验证：新增纯 NumPy unit test 覆盖 empty-target cIoU 语义和面积边界；本机已通过 Python syntax 与 `git diff --check`，但本机 Python 未安装 NumPy，unit test 会显式 skip。仍需在项目 Conda 环境执行 unit test 与真实 server prediction 的 metric-only smoke test。

### 2026-08-10 - 细分 GRES multi-instance 标注数量

- 代码：修改 `evaluation/gres/qwen3vl_gres_eval.py`、`evaluation/gres/subset_metrics.py` 和 `tests/test_gres_subset_metrics.py`。
- 文档：更新第 5.6 节及本变更日志。
- 行为：JSONL 子集报告对 `multi_instance` case 额外写出 `multi_annotation_count=2_instances/3_instances/4plus_instances`。数量来自官方 reference 的正 annotation id，不以可能相连或重叠的像素组件猜测实例数；全量指标和既有子集的计算不变。
- 验证：补充 annotation 数量分桶的 NumPy unit test；仍需在项目 Conda 环境运行该测试与现有 GRES prediction 的 metric-only smoke test。

### 2026-08-10 - 增加 GRES 两实例与目标面积交叉汇总

- 代码：修改 `evaluation/gres/qwen3vl_gres_eval.py`。
- 文档：更新第 5.6 节及本变更日志。
- 行为：离线子集 JSONL 对精确含两个正 annotation 的 reference 额外输出 `two_instance_target_area=small_lt_5pct/medium_5_to_25pct/large_ge_25pct`，面积由现有 case 的 GT union mask 计算。它与总面积统计共用相同边界，不改变全量、cardinality 或 multi count 指标。
- 验证：需以已有完整 GRES prediction 在项目 Conda 环境运行 metric-only，确认三类样本数之和等于 `2_instances` 样本数。

### 2026-08-10 - 增加 GRES 两实例成员召回与几何诊断

- 代码：修改 `evaluation/gres/qwen3vl_gres_eval.py`、`evaluation/gres/subset_metrics.py` 和 `tests/test_gres_subset_metrics.py`。
- 文档：更新第 5.6 节及本变更日志。
- 行为：离线子集汇总额外从官方 `instances.json` 重建每个 two-instance reference 的两个原 annotation mask，并验证其 union 与保存 case GT 一致。JSONL 增加全体成员召回、small/large member 的平均覆盖和 recall、both/one/none member hit rate，并分别以较小/较大成员面积比 `<0.2`、`0.2-0.5`、`>=0.5` 及掩码质心距离 `<0.25`、`0.25-0.5`、`>=0.5` 图像对角线分桶。成员命中默认要求预测覆盖该成员至少 50%，可由 `--two-instance-member-recall-threshold` 调整；union cIoU 同时保留，避免过大预测仅靠 recall 获得误导性结论。
- 验证：新增 two-instance member coverage/geometry 的 NumPy unit test；仍需在项目 Conda 环境用已有完整 GRES prediction 运行 metric-only smoke test。

### 2026-08-10 - 修正 GRES no-target 的 mask-token 拒识奖励

- 代码：修改 `projects/rl/reward_function/text2mask.py`，新增 `tests/test_no_target_reward.py`。
- 文档：更新第 3.5 节 `gres_no_target` 奖励契约和第 5.1 节测试清单。
- 行为：保留原始两项 `no_target_accuracy + no_repeat_score` 及其数值：完整拒识为 `1.0`，无 mask 但未写 `No target.` 为 `0.2`，其余为 `0.0`。`gres_no_target` 现在调用适用于 SAMTok 的 `no_target_check`，任何完整或残缺的 `<|mt_start|>`、`<|mt_####|>`、`<|mt_end|>` 都使拒识准确性为零；此前误用 bbox 检查，mask-token 幻觉不会被处罚。bbox helper 保留供历史 bbox 代码，当前 GRES mask 训练不再调用它。
- 论文边界：这是对现有 GRES no-target reward 实现的格式语义修正，不增加额外 reward 项、不改变 CycleGRPO pixel-IoU、caption/segmentation loss、OPSD 辅助项或数据配方。
- 验证：新增 unit test 覆盖正确拒识、缺少拒识文本的无 mask 输出，以及完整/残缺 mask-token 输出；待在项目 Conda 环境运行该测试和 10-step prompt-aligned C2 smoke training，检查 `reward/no_target_accuracy` 与 GRES N-acc。

### 2026-08-10 - 允许五路 CycleGRPO 混合器生成 25k 等可配置总量

- 代码：修改 `projects/rl/datasets/prepare_balanced_cyclegrpo_dataset.py`，新增 `tests/test_balanced_cycle_dataset.py`。
- 文档：更新第 2.4、5.1、5.3 节的混合器总量契约。
- 行为：五个 `--*-count` 配额不再被硬编码为总和 20,000；任意正总量均可导出，manifest 的 `total` 与行数断言均使用实际配额和。默认未变，仍为 `7k/5k/4k/2k/2k`。该修改仅允许 controlled scaling，例如同 35/25/20/10/10 比例的 25k `8750/6250/5000/2500/2500`，不改变抽样、随机 seed、正样本 caption 清除或训练算法。
- 论文边界：这是当前受控数据配方工具的规模参数化，不属于论文固定约 20k DenseWorld 设置；实验报告须同时记录总量和五路比例。
- 验证：新增 unit test 覆盖 25k 配额与空配额拒绝；待在项目 Conda 环境运行该测试、检查 25k manifest 的 source counts，并在训练前确认传入的 gRefCOCO positive parquet 为 `single_fraction=0.0` 导出的 true multi-only 数据。

### 2026-08-10 - 增加 DAM caption 的离线 QA 生成与验证工具

- 代码：新增 `projects/rl/datasets/generate_dam_caption_qa.py` 和 `tests/test_dam_caption_qa.py`。
- 文档：更新第 2.4、5.1、5.3 节及本变更日志。
- 行为：工具读取 DAM-backed Stuff/PACO 导出时写出的 caption JSONL manifest，以本地 OpenAI-compatible LLM 生成恰好两道 positive 四选一 QA，可选一道 `Yes=-1/No=0` negative hallucination QA；每个候选再由 LLM 验证其题目事实被 caption 显式蕴含、正确项唯一且干扰项无歧义。本地 schema 检查、失败重试、JSONL 逐行落盘、`--resume` ID 边界校验和 rejected manifest 使长任务可恢复。可选 DLC QA/class-name JSON 仅供现有无图 judge 复用。该工具不读取图像、不写 actor prompt、不改变 parquet、训练或任何 reward；QA reward 接入属于后续独立实验。
- 验证：`python3 -m py_compile projects/rl/datasets/generate_dam_caption_qa.py tests/test_dam_caption_qa.py`、`python3 -m unittest tests.test_dam_caption_qa` 和 `git diff --check`；本机没有运行中的 vLLM/DAM manifest，未执行 LLM 端到端生成或人工 QA 审核。

### 2026-08-10 - 接入 DLC-QA 与人工 referring grounding 外部锚定

- 代码：修改 RefCOCO/gRefCOCO/DAM 与平衡混合转换器、`verl` reward/config/trainer、`text2mask.py`、judge client、主训练入口；新增 `verl/workers/supervised_anchors.py` 和 `tests/test_supervised_anchors.py`。
- 文档：更新第 2.4、3.2、3.4、3.5、3.6、5.1、5.3 节及本日志。
- 行为：最终默认混合为 `8k/4k/5k/2k/1k`，gRefCOCO multi 在抽样前强制 `grounding_instance_count>=2`；原始 expression 写入 `grounding_query`，而 `cap_answer` 仍被清空。DAM QA JSONL 仅按 `dam_source_id` sidecar join，学生 caption 对全部题目得到均值 `1/0/-1` 外部 reward；失败题归零。RefCOCO/gRefCOCO 的 human-query text-to-mask 使用独立 `K=2` rollout、UID、像素 IoU/no-target reward 和 `0.25` loss weight；不修改 cycle `R_Ci`、OPSD route、teacher CE/JSD 或 anchor KL。该功能是外部监督锚定，不应表述为纯 on-policy self-distillation。
- 验证：`python3 -m py_compile` 覆盖修改 Python、`bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`、`python3 -m unittest tests.test_balanced_cycle_dataset tests.test_dam_caption_qa tests.test_supervised_anchors`（11 tests）和 `git diff --check` 均通过。本机没有 Ray/FSDP/vLLM、CUDA、SAMTok 或独立 judge 服务，未运行多 GPU training smoke test。

### 2026-08-10 - 修复 direct grounding 的 gRefCOCO no-target 分流

- 代码：修改 `verl/trainer/ray_trainer.py`、`verl/workers/fsdp_workers.py`、`verl/workers/supervised_anchors.py` 和 `tests/test_supervised_anchors.py`。
- 文档：更新第 3.4 节 direct-grounding 的 cycle/non-cycle 数据边界。
- 行为：direct grounding 现在分别从 cycle 与 non-cycle rollout 子批提取带 `grounding_query` 的样本，因此 `gres_no_target` 不会因其正常训练分流而遗漏。no-target 使用独立 source 和现有 `No target.` reward；像素 decoder 显式跳过没有 GT mask 的该类 rollout。它仍不参与 cycle `R_Ci`、OPSD route、teacher CE/JSD 或 caption reward。
- 验证：`python3 -m unittest tests.test_balanced_cycle_dataset tests.test_dam_caption_qa tests.test_supervised_anchors`（12 tests）、受影响 Python 文件的 `python3 -m py_compile`、`bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 与 `git diff --check` 均通过；本机没有 Ray/FSDP/vLLM、CUDA、SAMTok 或独立 judge 服务，仍未执行服务器端 smoke training。

### 2026-08-10 - 严格校验 DLC-QA sidecar 奖励契约

- 代码：修改 `projects/rl/datasets/generate_dam_caption_qa.py`、`projects/rl/datasets/prepare_balanced_cyclegrpo_dataset.py`、`verl/workers/reward/function.py`、`tests/test_balanced_cycle_dataset.py` 和 `tests/test_supervised_anchors.py`。
- 文档：更新第 2.4 节 sidecar 的导出与训练载入边界。
- 行为：混合器和 reward actor 在使用 JSONL 前共同验证问题文本/选项唯一性、正题的四选一 `1/0`、可选负题的 `Yes=-1/No=0` 以及两道 positive 的数量。无效 sidecar 会在训练前失败，不会把任意标签数值交给 Llama judge 或加入 caption reward。
- 验证：`python3 -m unittest tests.test_balanced_cycle_dataset tests.test_dam_caption_qa tests.test_supervised_anchors`（13 tests）、受影响 Python 文件的 `python3 -m py_compile`、`bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 与 `git diff --check` 均通过；本机没有 Ray/FSDP/vLLM、CUDA、SAMTok 或独立 judge 服务，仍未执行服务器端 smoke training。

### 2026-08-11 - 对齐 cycle localization 与下游分割指令

- 代码：修改 `verl/utils/dataset.py`、`verl/trainer/ray_trainer.py`、`verl/workers/supervised_anchors.py`、`projects/rl/config.yaml`、`projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 和 `tests/test_supervised_anchors.py`。
- 文档：更新第 3.2、3.3、3.4、3.6 节及关键注意事项 24。
- 行为：有目标 image CycleGRPO localization 不再只使用历史长模板。对每个原图连续的 `G=6` 条 student caption，按组内偶/奇 index 精确使用 3 条 `Please segment {caption} in this image.` 与 3 条 GroundingSuite `Please carefully check ...` 模板；caption 本身仍来自 actor，不注入正样本人工 referring expression。火山引擎入口默认把 `gres_no_target` 建为独立 `task=segmentation` direct-grounding group，并按同一偶/奇规则以 1:1 覆盖这两种模板，使用既有两项拒识 reward；该 row 在 direct batch 构造后从 non-cycle caption PPO 移除，且 direct no-target 不加 segmentation anchor KL。`include_positive_sources=false` 保持该首版只对齐 no-target，开启正 expression direct grounding 时必须作为外部监督消融报告。
- 论文边界：no-target 分支使用 gRefCOCO expression，属于为评测 instruction 对齐的受控 GRES 辅助项；它不进入 caption cycle 的 `R_Ci`、OPSD route、teacher regenerate/JSD 或 image-mask-only 主训练信号。
- 验证：`python3 -m py_compile verl/trainer/ray_trainer.py verl/utils/dataset.py verl/workers/supervised_anchors.py tests/test_supervised_anchors.py`、`python3 -m unittest tests.test_supervised_anchors`（7 tests）、`bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 与 `git diff --check` 通过；本机缺少 PyTorch/Ray/FSDP/vLLM、CUDA、SAMTok 与服务器数据，未执行 8-GPU smoke training。

### 2026-08-11 - 修复禁用 caption-QA 时的 Hydra 空字符串覆盖

- 代码：修改 `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`。
- 文档：追加本变更日志。
- 行为：当 `SUPERVISED_CAPTION_QA_ENABLED=false` 时，入口只传递该 enabled flag，不再把空 `CAPTION_QA_JSONL`、judge URL/model 等命令行参数传入 OmegaConf。此前 Hydra 会将空字符串解析为 `None`，与 `CaptionQAConfig` 的 `str` 字段冲突并在 trainer 初始化前报错；启用 QA 时仍传递全部已校验参数。
- 验证：`bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 与 `git diff --check` 通过；该问题发生在 OmegaConf 启动期，仍需在服务器以 QA disabled 的训练命令完成一次配置加载验证。

### 2026-08-11 - 修正 GRES 评测默认标注目录大小写

- 代码：修改 `projects/eval/qwen3vl_4b_volcengine.sh`。
- 文档：更新第 5.3 的 GRES 评测路径约定并追加本日志。
- 行为：统一评测入口的默认 `GRES_ROOT` 从不存在的 `${BASE_DIR}/grefcoco` 改为服务器实际目录 `${BASE_DIR}/gRefCOCO`，使默认 `grefs(unc).json` 和 `instances.json` 解析正常；显式 `GRES_ROOT` 覆盖行为不变。
- 验证：`bash -n projects/eval/qwen3vl_4b_volcengine.sh` 与 `git diff --check` 通过；仍需在服务器重新运行 `gres` action 完成端到端验证。

### 2026-08-11 - 为五路混合增加类别模板 direct grounding

- 代码：新增 `projects/rl/datasets/grounding_queries.py` 和 `tests/test_grounding_queries.py`；修改 COCO-Stuff、PACO-LVIS、RefCOCO、gRefCOCO 与平衡 parquet 转换器，及 direct-grounding config/trainer/火山引擎入口与相关测试。
- 文档：更新第 2.4、3.4、5.3、6 节的 query schema、PACO union target、direct anchor 开关与论文边界。
- 行为：新导出的 RefCOCO/gRefCOCO/no-target 保留类型化人工 query；COCO-Stuff 写入由官方 91 类 semantic PNG value 映射的 `the {label}`；PACO 写入 `the {part} of the {parent}`，并合并同图同标签的 part mask。`include_label_sources=false` 默认保持旧训练不变；显式启用时 Stuff/PACO 与人工 query 一样进入独立 `K=2` pixel-IoU direct grounding batch，仍不进入 caption prompt、cycle `R_Ci`、OPSD route 或 teacher target。混合器的 `--require-grounding-query` 用于确保新五路 parquet 没有静默漏掉 direct 监督。
- 论文边界：类别模板 query 是从原始 semantic/part 标签确定性构造的额外监督，不是 GroundingSuite 训练数据、人工 referring expression 或原始 CycleGRPO 的 image-mask-only objective；COCO 全图 caption 仍不与单个 region mask 配对。
- 验证：`python3 -m py_compile` 覆盖转换器、anchor/trainer 与测试，`python3 -m unittest tests.test_grounding_queries tests.test_balanced_cycle_dataset tests.test_supervised_anchors`（17 tests）、`bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 和 `git diff --check` 通过。本机无 PACO/COCO 图像、SAMTok/VQ-SAM2、CUDA 与 Ray/vLLM，尚未执行 25k re-export 或 8-GPU smoke training。

### 2026-08-11 - 默认保留原始 CycleGRPO 的 GRES no-target 路径

- 代码：修改 `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`。
- 文档：更新第 3.3、3.4、3.6 节和本变更日志。
- 行为：火山引擎入口的 direct-grounding 默认从启用的 no-target-only 替换式辅助，改为完全关闭；`include_no_target=false`、`consume_no_target_caption=false`。因此 `gres_no_target` 默认只走原始的外层 caption `G` rollout、原有 `No target.` + no-repeat reward 和 GRPO，不进入 cycle `K` localization，也不会被 direct `K=2` batch 替代。RefCOCO/gRefCOCO/Stuff/PACO direct supervision 以及 no-target instruction-alignment 仍可通过环境变量显式启用，属于外部监督消融。
- 论文边界：这恢复原始 CycleGRPO 对 no-target 的训练拓扑；现有 SAMTok mask-token 拒识检查仍是本仓库的格式修复，不新增 reward 项。
- 验证：`bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`、针对默认值的静态断言和 `git diff --check`。

### 2026-08-11 - 将 direct grounding 固定为附加的 K=6 监督采样

- 代码：修改 `projects/rl/config.yaml`、`projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`、`verl/workers/supervised_anchors.py`、`verl/trainer/ray_trainer.py` 和 `tests/test_supervised_anchors.py`。
- 文档：更新第 3.3、3.4、3.6 节和本变更日志。
- 行为：direct grounding 的 generic YAML、dataclass 与火山引擎入口默认 rollout 数统一为 `K=6`。它继续在主 cycle `G=6,K=6` rollout 完成后以新 UID 独立生成并在同一 optimizer step 累积梯度；不会进入或修改 cycle `R_Ci`、路由、teacher 或主奖励。`gres_no_target` 由 source 显式映射为 `supervised_grounding_no_target`，其 direct reward 仅是既有无 mask `No target.` 拒识正确性加 non-repeat，VQ-SAM2/像素 IoU 对该组明确跳过。移除 trainer 中曾按 `consume_no_target_caption` 删除主 no-target PPO row 的实现，并在 shell/config 校验中拒绝该旧替换式设置，因此 direct 对所有 source 均只能是额外监督。
- 论文边界：direct K=6 是外部 query-to-mask supervised anchor 的采样规模调整，不属于原始 CycleGRPO 的 image-mask-only cycle；保留主 no-target outer-caption GRPO 则与原始训练拓扑一致。
- 验证：`python3 -m unittest tests.test_supervised_anchors`、受影响 Python 文件的 `python3 -m py_compile`、`bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 和 `git diff --check`。

### 2026-08-11 - 修正 PACO-LVIS v1 的 part 查询与 union 目标

- 代码：修改 `projects/rl/datasets/grounding_queries.py`、`projects/rl/datasets/prepare_paco_lvis_part_cycle_dataset.py` 和 `tests/test_grounding_queries.py`。
- 文档：更新第 2.4、模块清单、关键注意事项 25 和本变更日志。
- 行为：实际 PACO-LVIS v1 train 标注显示所有 annotation 的 `category_id` 都属于 object category；`id != obj_ann_id` 的 395,071 条 non-parent annotation 虽是 part mask，却没有逐 mask part-category 字段。转换器不再把 object ID 错当 `part_categories` ID，改为对同图同 parent object 类别的全部 part mask 取 union，并写入 `the visible parts of the {parent}` 与 `grounding_query_kind=parent_parts_label`。这恢复非空 PACO 候选，同时避免制造不存在的 `the {part} of the {parent}` 细粒度监督或从同类 part 中任意抽取一个 mask。
- 论文边界：该 PACO source 提供的是 parent-conditioned visible-part union supervision，粒度低于人工 part referring expression；它仍是额外 label-template direct grounding，不是 GroundingSuite 训练数据或原始 CycleGRPO image-mask-only cycle。
- 验证：`python3 -m py_compile` 覆盖转换器/query helper、`python3 -m unittest tests.test_grounding_queries` 和 `git diff --check`；服务器端仍需重新导出 PACO parquet，并检查 2,500 条输出与 `parent_parts_label` manifest。

### 2026-08-11 - 修复 PACO split 图像根目录解析

- 代码：修改 `projects/rl/datasets/prepare_paco_lvis_part_cycle_dataset.py`。
- 文档：更新第 2.4 节 PACO 图像目录契约并追加本日志。
- 行为：图像解析现在无条件依次尝试 metadata 相对路径、其 basename、以及 `train2017/` 和 `val2017/` 子目录。因此 `--images-dir` 既可传 `PACO-LVIS/images`，也可传已进入 split 的 `PACO-LVIS/images/train2017`。此前对后一种正确服务器路径，含 `train2017/` 前缀的 metadata 会被错误拼接为两层 split 目录，导致所有可用 PACO group 被计入 skipped。
- 验证：`python3 -m py_compile projects/rl/datasets/prepare_paco_lvis_part_cycle_dataset.py projects/rl/datasets/grounding_queries.py`、`python3 -m unittest tests.test_grounding_queries`（3 tests）、临时目录 image-root resolution smoke test（split root 与 parent root）和 `git diff --check` 均通过；服务器端仍需重跑 2,500 条 PACO 导出。

### 2026-08-11 - 对齐 direct-grounding 子批的分布式 prompt 数

- 代码：修改 `verl/workers/supervised_anchors.py`、`verl/trainer/ray_trainer.py` 和 `tests/test_supervised_anchors.py`。
- 文档：更新第 3.4 节 direct-grounding 分布式 dispatch 契约并追加本日志。
- 行为：每个 cycle/non-cycle direct parent 子批在 segmentation rollout 前裁到 rollout world size 的整数倍；少于一个完整 shard 时跳过并输出原因。此前主 cycle/non-cycle group 已按 8 卡对齐，但从它们筛出的 direct no-target 子批可保留 12 个 prompt，导致 `DataProto.chunk(8)` 在第 0 step 断言失败。现在该 case 裁为 8 个 prompt、`K=6` 产生 48 个 direct no-target rollout；cycle 主训练、no-target outer caption GRPO 和既有拒识 reward 不变。
- 验证：`python3 -m py_compile verl/workers/supervised_anchors.py verl/trainer/ray_trainer.py`、`python3 -m unittest tests.test_supervised_anchors`（10 tests）、aligned-prefix static integration assertion 与 `git diff --check` 均通过；服务器端需以 8 GPU 重启该 25k direct run，确认日志出现 trim/skip 信息后进入 step 1。

### 2026-08-12 - 增加 RefCOCO first-mask 离线诊断

- 代码：新增 `evaluation/refcoco/first_mask_diagnostic.py`、`evaluation/refcoco/run_first_mask_diagnostic_multigpu.sh` 和 `tests/test_first_mask_diagnostic.py`。
- 文档：更新第 5.1、5.6、关键注意事项 26 和本日志。
- 行为：诊断工具读取已有 RefCOCO 逐样本 response/GT JSON，只提取第一个完整且 codebook 合法的 SAMTok depth-2 mask group，以 VQ-SAM2 重新解码为独立目录的 mask，再输出 cIoU/mIoU、首 mask 缺失数。它不重新生成 VLM response、不修改原 union-mask prediction 或正式 benchmark 指标，可直接量化后续多 mask group 对当前 direct run 分割结果的污染。`--metric-only` 只需要 `--output-dir`；输入 response、RefCOCO 标注和 VQ-SAM2 路径只在重解码分片时校验。
- 验证：`python3 -m py_compile evaluation/refcoco/first_mask_diagnostic.py tests/test_first_mask_diagnostic.py`、`python3 -m unittest tests.test_first_mask_diagnostic`（4 tests）、`bash -n evaluation/refcoco/run_first_mask_diagnostic_multigpu.sh` 和 `git diff --check` 均通过；本机无 NumPy，未能执行依赖 pycocotools/NumPy 的 metric-only runtime，服务器项目环境需各运行 current-direct 与旧 C2 的 8-GPU re-decode 后比较 metrics。

### 2026-08-12 - 强制正例 segmentation 的单 mask 序列化并统一首 mask 评测

- 代码：修改 `verl/workers/opsd/mask_iou.py`、`verl/workers/opsd/config.py`、`verl/workers/opsd/__init__.py`、`verl/workers/fsdp_workers.py`、`verl/workers/reward/function.py`、`verl/trainer/ray_trainer.py`、`projects/rl/reward_function/text2mask.py`、`projects/rl/config.yaml`、`projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`、RefCOCO/GRES/GroundingSuite 评测脚本；扩展 `tests/test_opsd_core.py`。
- 文档：更新第 3.4、3.5、5.2、5.6、关键注意事项 26 和本日志。
- 行为：正例 cycle localization 与 `supervised_grounding` 现在要求 response 含恰好一个完整、codebook 合法的 SAMTok depth-2 group。online VQ-SAM2 仍解码首个合法 group 以记录 `first_mask_pixel_iou`，但若 group 数不是一，训练 IoU 为零、format/non-repeat 正项为零，并按额外完整 group 数施加 `-1.0` penalty；`supervised_grounding_no_target` 继续使用未改变的 `No target.` 零 mask 拒识 reward。新增 group-count metadata 和 cycle/direct one-mask/multi-mask 日志。为避免早期重复循环耗尽 rollout token，正例 localization 独立限制为 32 tokens。离线 RefCOCO/GRES/GroundingSuite 生成在第一个 `<|mt_end|>` 停止，只解码第一个完整合法 group；GroundingSuite 上限由 512 降为 128 且不再打印每条 response。新旧 checkpoint 必须在这一首 mask 协议下重新生成，不能与历史 union-mask 数值直接混比。
- 论文边界：原始 CycleGRPO 只奖励存在的 mask-token 格式；本修改是针对 direct-grounding 高权重训练触发的重复同一 code group 循环的格式稳定化扩展，不改变 `G=6`、`K=6`、多实例 union GT、pixel-IoU 定义或 no-target 两项 reward。
- 验证：本机 `python3 -m py_compile` 覆盖全部修改 Python 文件，`bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 和 `git diff --check` 通过。新增 unit test 覆盖单 group 正分、三次重复 group 的零 IoU/`-2` penalty 和首合法 code 保留；本机缺少 PyTorch、Hydra、vLLM、CUDA 与项目 Conda 环境，`tests.test_opsd_core` 和评测 parser runtime 必须在服务器 `$ENV_DIR/bin/python3` 运行，尚未执行 8-GPU smoke training。

### 2026-08-12 - 受控融合 CycleGRPO、direct GRPO 与 GT-mask CE

- 代码：修改 `verl/workers/supervised_anchors.py`、`verl/workers/config.py`、`verl/trainer/ray_trainer.py`、`verl/workers/fsdp_workers.py`、`verl/workers/actor/dp_actor.py`、`projects/rl/config.yaml`、`projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 和 `tests/test_supervised_anchors.py`。
- 文档：更新第 3.4、3.5、3.6、5.2 节和本日志。
- 行为：direct GRPO 的 `loss_weight` 现在是目标权重；默认 step `<=10` 为零、`11..29` 线性升高、step `>=30` 到达目标。新增默认关闭的 `direct_mask_ce`，仅从 RefCOCO/gRefCOCO 人工正 referring expression 的每个原始 UID 建立一条 query-to-GT-SAMTok teacher-forcing 样本；GRES no-target、Stuff/PACO label template 与 direct rollout response 不进入 CE。GT mask token 参与 CE，EOS 与 padding 只参与前向上下文且 loss mask 为零。两项 direct 梯度与现有 caption GRPO、localization GRPO、regenerate CE、JSD/KL 在同一次 optimizer step 累积；direct GRPO 的 UID/advantage group 不变。新增 effective/target direct 权重、direct CE 权重、样本数与独立 CE loss 日志。
- 论文边界：`lambda_direct(step)*L_direct_GRPO + 0.02*L_direct_mask_CE` 是当前 image-mask-only CycleGRPO 之外的外部人工 referring-expression 监督消融；它不修改 cycle `R_Ci`、三路由、pixel-IoU 定义或原有 no-target 两项 reward。
- 验证：本机 `python3 -m py_compile` 覆盖 anchor/config/trainer/FSDP/actor，`bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`、`python3 -m unittest tests.test_supervised_anchors`（13 tests）和 `git diff --check` 通过。本机缺少 PyTorch/Hydra/Ray/vLLM/CUDA，尚未运行完整 `tests.test_opsd_core` 或 8-GPU 20-step smoke training；服务器需确认 step 10/30 weight、CE sample count、single-mask rate、no-target reward 和 RefCOCO 首 mask指标。

### 2026-08-15 - 修复 DAM QA prompt 的 JSON schema 格式化

- 代码：修改 `projects/rl/datasets/generate_dam_caption_qa.py` 和 `tests/test_dam_caption_qa.py`。
- 文档：更新第 2.4 节 DAM QA prompt 渲染契约并追加本日志。
- 行为：生成与 LLM 验证 prompt 的字面 JSON 花括号现在以 `str.format` 规则转义，只有 `{caption}` 和 `{candidate_json}` 保留为格式化字段。此前 schema 内的 `"class_name"` 会在请求发送前触发 `KeyError`，导致全部 QA 记录在本地重试后进入 rejected JSONL；QA schema、判定规则、source 配额和训练 reward 均未改变。
- 验证：`python3 -m py_compile projects/rl/datasets/generate_dam_caption_qa.py tests/test_dam_caption_qa.py`、`python3 -m unittest tests.test_dam_caption_qa` 和 `git diff --check`。

### 2026-08-15 - 修复 direct GT-mask CE 的 non-tensor media 索引

- 代码：修改 `verl/trainer/ray_trainer.py`、`verl/workers/supervised_anchors.py` 和 `tests/test_supervised_anchors.py`。
- 文档：更新第 3.4 节 direct GT-mask CE 的 DataProto non-tensor 取值契约并追加本日志。
- 行为：direct mask CE 现在在原始 cycle batch 的 object array 上按 parent index 选择图像 media、GT mask、RLE 和 caption metadata。此前先执行 `DataProto[parent_index]` 会把 media row 解包为字典，代码随后对该字典执行 `[0]` 而在首个 CE batch 触发 `KeyError: 0`。CE source 筛选、target token、loss mask、权重和主 CycleGRPO/direct GRPO 均未改变。
- 验证：`python3 -m py_compile verl/workers/supervised_anchors.py verl/trainer/ray_trainer.py tests/test_supervised_anchors.py`、`python3 -m unittest tests.test_supervised_anchors` 和 `git diff --check`。

### 2026-08-18 - 重写 README 的环境、数据、权重与运行手册

- 代码：修改 `README.md`；不修改训练、评测、数据转换或奖励逻辑。
- 文档：README 现在以当前受维护的火山引擎训练入口和统一 FSDP 导出/评测入口为主，覆盖 CUDA/Python/vLLM 环境 profile、SAMTok/VQ-SAM2 文件契约、RefCOCO/gRefCOCO/Stuff/PACO/DAM/GroundingSuite/DLC 数据边界、公开 COCO/COCO-Stuff/RefCOCO/PACO-LVIS 的可恢复下载与目录整理命令、Parquet 导出、25k 混合、direct/CE、DAM QA、resume、导出和四项评测命令。监督章节现位于 checkpoint 导出和评测之前；direct 段新增完整 RefCOCO train（42,404 条）Parquet 导出命令，可作为直接 GRPO/GT-mask CE 专项训练输入。根据新服务器的实际解压结果，COCO-Stuff 路径更正为官方 archive 直接生成的 `COCO-Stuff/train2017`，PACO 下载更正为完整官方 `paco_lvis_v1.zip`。HF checkpoint 下载改为当前激活环境的 `snapshot_download`，RefCOCO expression 文件通过 `find` 适配官方压缩包的不同层级；DLC judge 改为直接 `vllm serve`，避免历史脚本忽略用户模型路径；direct 样例补齐独立运行所需的 GPU/model/data/run 变量，QA 片段明确为完整训练命令的附加变量。gRefCOCO、DAM、GroundingSuite 与 DLC 仅列出官方发布入口和目标目录，避免对可能受限或变动的第三方发布 URL 作出不可靠承诺。
- 行为：用户不再被旧通用脚本的占位符或“FSDP shard 可直接评测”的错误假设误导；README 明确当前外部有监督扩展与原始 image-mask-only CycleGRPO 的边界。文档中的服务器手动命令不使用 fail-fast shell 选项，以保留终端 traceback。
- 验证：逐段核对 README 的 bash block、变量名、数据源、转换器参数、训练/eval action 与 `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`、`projects/eval/qwen3vl_4b_volcengine.sh`、`projects/rl/datasets/*`、`evaluation/dlc_bench/serve_judge.sh`，并执行 `git diff --check`。

### 2026-08-19 - 批量化 RefCOCO 离线生成

- 代码：修改 `evaluation/refcoco/qwen3vl_refcoco_eval.py`、`evaluation/refcoco/run_refcoco_multigpu.sh` 与 `README.md`。
- 文档：更新第 5.6 节 RefCOCO 评测职责。
- 行为：RefCOCO 每个 GPU 不再逐条调用 Qwen3-VL `generate`；待评测样本按 `--batch_size` 组成多图多文本 batch，统一 padding 后执行 generation，再按原有逐样本路径解码首个合法 mask group、恢复原尺寸并写 JSON。launcher 从 `EVAL_BATCH_SIZE` 读取每卡 batch size，默认 16。已有 JSON 仍会跳过，指标格式、mask 解析和 VQ-SAM2 解码语义不变。H20 可先显式设为 32；OOM 时降低为 24 或 16。
- 验证：`python3 -m py_compile evaluation/refcoco/qwen3vl_refcoco_eval.py`、`bash -n evaluation/refcoco/run_refcoco_multigpu.sh` 和 `git diff --check` 通过。本机没有 Qwen3-VL/SAMTok、CUDA、RefCOCO 数据或 H20，尚未执行多图 processor/generation 的端到端 smoke test；服务器应先以一个 shard 和 `EVAL_BATCH_SIZE=16` 验证输出数与单样本协议一致，再提高 batch size。

### 2026-08-19 - 修复 GRES 与 GroundingSuite 首 mask 解析的 codebook 作用域

- 代码：修改 `evaluation/gres/qwen3vl_gres_eval.py` 与 `evaluation/groundingsuite/qwen3vl_groundingsuite_infer.py`。
- 文档：更新第 5.6 节 GRES 评测职责并追加本日志。
- 行为：将 `CODEBOOK_SIZE=256` 和 `CODEBOOK_DEPTH=2` 设为模块级常量，供首个 depth-2 SAMTok mask group 的合法性检查和 VQ-SAM2 构造共同使用，移除两个 `main()` 中遮蔽同名常量的局部声明。此前 GRES 的 `extract_mt_token_ids_v1()` 在模型首次返回完整 mask group 时引用未定义的局部作用域常量并触发 `NameError`，导致各 shard 退出；GroundingSuite 存在相同潜在错误。结果文件的恢复/跳过协议、首 mask 语义与指标计算均不变。
- 验证：`PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile evaluation/gres/qwen3vl_gres_eval.py evaluation/groundingsuite/qwen3vl_groundingsuite_infer.py`、GRES 与 GroundingSuite launcher 的 `bash -n`、以及 `git diff --check` 通过；本机没有 H20、模型和评测数据，服务器需以保留的 case JSON 续跑 GRES 来完成端到端验证。

### 2026-08-19 - 恢复 SAMTok 历史 union 分割评测协议

- 代码：新增 `evaluation/mask_protocol.py` 与 `tests/test_mask_protocol.py`；修改 `evaluation/refcoco/qwen3vl_refcoco_eval.py`、`evaluation/gres/qwen3vl_gres_eval.py`、`evaluation/groundingsuite/qwen3vl_groundingsuite_infer.py`、三个对应 multi-GPU launcher、`projects/eval/qwen3vl_4b_volcengine.sh` 和 `README.md`。
- 文档：更新第 2.2、5.6、关键注意事项 26、模块清单和本日志。
- 行为：三个离线分割评测器现在共享显式 `legacy_union|first_mask` 协议。默认 `legacy_union` 不将 `<|mt_end|>` 设为 EOS，解析全部完整、codebook 合法的 depth-2 mask group，逐组 VQ-SAM2 解码后取 union，以保证 SAMTok 历史 RefCOCO/GRES/GroundingSuite 基线可比；`first_mask` 保留此前严格首组语义，并将 `<|mt_end|>` 加入 EOS。每条预测写入 `mask_protocol`；已有 JSON 只有在协议相同才会 resume，旧的无协议或不同协议结果会重算，防止指标混合。在线 CycleGRPO/direct 的单 mask reward、额外 group penalty、codebook 作用域修复与 RefCOCO batch generation 均未改变。
- 论文边界：这是离线 benchmark 可比性修复，不放宽训练时正例 localization 的单 mask 序列化约束。使用 `first_mask` 的新 direct checkpoint 分数必须作为独立协议报告，不能和历史 union baseline 直接比较。
- 验证：`PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile evaluation/mask_protocol.py evaluation/refcoco/qwen3vl_refcoco_eval.py evaluation/gres/qwen3vl_gres_eval.py evaluation/groundingsuite/qwen3vl_groundingsuite_infer.py tests/test_mask_protocol.py`、`python3 -m unittest tests.test_mask_protocol`（4 tests）、三个 evaluator launcher 与 `projects/eval/qwen3vl_4b_volcengine.sh` 的 `bash -n`、以及 `git diff --check` 均通过。本机无 H20、checkpoint、CUDA 和 benchmark 数据，尚未运行端到端重新生成；服务器必须用独立 `EVAL_ROOT`（推荐）或允许协议字段触发目录内 JSON 重写后完成三项完整评测。

### 2026-08-19 - 恢复 CycleGRPO 多 group 训练语义并在线计算 union IoU

- 代码：修改 `verl/workers/opsd/mask_iou.py`、`verl/workers/opsd/config.py`、`verl/workers/fsdp_workers.py`、`verl/workers/reward/function.py`、`verl/trainer/ray_trainer.py`、`projects/rl/reward_function/text2mask.py`、`projects/rl/config.yaml`、`projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 和 `tests/test_opsd_core.py`。
- 文档：更新 `README.md`、第 3.4、3.5、5.2、关键注意事项 26 和本日志。
- 行为：正例 localization 不再要求恰好一个 group，也不再把多 group IoU 置零或按额外 group 扣分。在线 decoder 从同一 image embedding 批量解码一条 response 的全部完整、codebook 合法 depth-2 group，并 union 为一个 prediction；raw GT 可用时仍优先作为 IoU target，结果同时作为 cycle `s_i,k`、`R_Ci` routing 和 direct `pixel_iou`。奖励恢复原 CycleGRPO 的格式与 non-repeat 语义：至少一个完整 group 得 format 一分，只有同一完整 group 超过三次时 non-repeat 一分为零，IoU 不受此正则置零。保留可配置的 `segmentation_max_response_tokens` 作为生成长度上限和 group-count telemetry；移除严格单 group 配置、环境变量、Hydra override 与遥测。
- 论文边界：原始公开 CycleGRPO 以 HTG token matching 计算 `s_i,k`；当前实现唯一替换该测量为在线 decoded union 的像素 IoU，保留其在 `R_loc_i,k=10*s_i,k*mean_k(s_i,k)+format+non_repeat` 中的位置和其多 group 表达语义。此项取代本日志中 2026-08-12 的严格单 group 训练扩展；离线 `legacy_union|first_mask` 协议不变。
- 验证：修改 Python 文件的 AST 语法检查、`bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 和 `git diff --check` 通过。`python3 -m unittest tests.test_opsd_core` 在本机因缺少 `torch` 无法导入；测试已新增多 group 正奖励、四次相同 group 仅清零 non-repeat 项、以及 shared embedding 的 union decode 覆盖。服务器仍需以项目环境运行该单元测试和最小 batch smoke training。

### 2026-08-19 - 修复 union-IoU 训练入口的残留严格开关校验

- 代码：修改 `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`。
- 文档：追加本日志。
- 行为：移除布尔环境变量校验列表中已删除的 `REQUIRE_EXACTLY_ONE_MASK`。union-IoU 改动已不再定义该变量；其残留会在脚本的 `set -u` 下于模型、Ray 或训练日志初始化前报 `!bool_name: unbound variable`。训练参数、direct GRPO/CE、数据和 checkpoint 行为均不变。
- 验证：`bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 与 `git diff --check` 通过。

### 2026-08-20 - 隔离 CycleGRPO、RefCOCO direct 与 DLC-QA 三条训练流

- 代码：修改 `verl/workers/supervised_anchors.py`、`verl/workers/config.py`、`verl/trainer/data_loader.py`、`verl/trainer/main.py`、`verl/trainer/ray_trainer.py`、`verl/workers/reward/function.py`、`projects/rl/reward_function/text2mask.py`、`projects/rl/config.yaml`、`projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 和 `tests/test_supervised_anchors.py`。
- 文档：更新第 3.2、3.4、3.5、3.6、5.2、5.3、关键注意事项 27 及 `README.md` 的监督训练示例。
- 行为：主 `data.train_files` 的 20k image-mask mix 现在只执行 CycleGRPO/OPSD，不再从其 cycle 或 non-cycle 子批派生 direct GRPO、GT-mask CE 或 caption-QA。新增可恢复的独立 RefCOCO loader（`direct_grounding.train_files`/`batch_size`）和独立 DLC-QA loader（`caption_qa.train_files`/`batch_size`），每个主 step 各读取一批且不改变主 epoch/global step/save cadence；checkpoint 新增 `auxiliary_dataloaders.pt` 保存二者位置。direct loader 强制全是 `source=refcoco_cycle` 的 human-expression full RefCOCO 行，direct GRPO 与 GT-mask CE 都只从该流构造。DLC-QA loader rollout 标为 `supervised_caption_qa`，要求每条 `dam_source_id` 可 join 已验证 QA JSONL，并且其 reward 只等于 judge QA 得分，不混入 cycle IoU、格式、caption safety、groundedness、OPSD routing 或 teacher 辅助项；`caption_qa.loss_weight` 单独控制该 GRPO 梯度比例。三流仍在一个 optimizer step 前累计，因此这是论文 image-mask-only CycleGRPO 之外的外部监督扩展。
- 验证：`PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile verl/workers/supervised_anchors.py verl/workers/config.py verl/trainer/data_loader.py verl/trainer/main.py verl/trainer/ray_trainer.py verl/workers/reward/function.py projects/rl/reward_function/text2mask.py tests/test_supervised_anchors.py`、`PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_supervised_anchors`（15 tests）、`bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 与 `git diff --check` 均通过。当前机器无项目 GPU/Ray/vLLM 环境，尚未执行 8-GPU end-to-end smoke；服务器应先以 `MAX_STEPS=1` 验证三条 loader 的 size、direct source guard、QA join 和 `auxiliary_dataloaders.pt`。

### 2026-08-20 - 增加七卡训练的 2:4:1 三流 batch 配额校验

- 代码：修改 `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`。
- 文档：更新第 3.2、关键注意事项 28、`README.md` 的训练说明和本日志。
- 行为：新增 opt-in `THREE_STREAM_2_4_1_ENABLED=true`。该模式固定使用 7 个 Ray/FSDP 训练 GPU，要求 direct GRPO、direct GT-mask CE 和 DLC-QA 都启用，三条 parent-prompt batch 均可被 7 整除，并强制 `ROLLOUT_BATCH_SIZE:DIRECT_BATCH_SIZE:CAPTION_QA_BATCH_SIZE=2:4:1`。推荐 `28:56:14`，对应每 rank `4:8:2` 个 parent prompt；第八张 GPU 由外部 `vllm serve` 独占用于 DLC judge。该检查不把三条 parquet 拼接，不改变三条 loader 的独立循环、同 step 梯度累积、rollout 数或主 20k epoch 定义。为保证 20k 流只含主 CycleGRPO/OPSD，该模式要求 online pixel IoU 开启并关闭 routing/EMA teacher/regenerate-JSD 路径、caption safety、groundedness 与两项 anchor KL。`ACTOR_GLOBAL_BATCH_SIZE` 继续只约束主 rollout batch，必须整除 `ROLLOUT_BATCH_SIZE`。
- 验证：`bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 与 `git diff --check` 通过；当前机器无 8 张 H20、项目 Conda/Ray/vLLM 环境，服务器应先以 `MAX_STEPS=1` 确认 judge endpoint、日志中的 7 卡 world size 与 `4:8:2` 三流 loader batch。

### 2026-08-21 - 修复 DLC Llama judge 与 QA 生成的对话结束 token

- 代码：修改 `evaluation/dlc_bench/eval_llama_without_image.py` 和 `projects/rl/datasets/generate_dam_caption_qa.py`。
- 文档：更新第 5.6 节 DLC judge 约定并追加本日志。
- 行为：DLC evaluator 每个 OpenAI-compatible vLLM 请求显式传递 Llama-3.1 `<|eot_id|>` 的 `stop_token_ids=[128009]`，并将选择题生成上限从 300 降至 16 token。DAM QA 生成器新增可选 `--stop-token-id`，将相同结束 token 同时传给生成与 validator 请求，不影响其他服务的默认请求格式。缺失 tokenizer chat template 的本地 HF 权重此前会在生成 `A.` 或 JSON 后继续输出下一轮 `assistant` header，直至长度上限；此修复不改变 QA schema、选项解析或评分公式。
- 验证：服务端 `curl` 已确认 `stop_token_ids=[128009]` 时响应仅输出 `A.`；本地执行 `PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile evaluation/dlc_bench/eval_llama_without_image.py projects/rl/datasets/generate_dam_caption_qa.py`、`python3 -m unittest tests.test_dam_caption_qa` 与 `git diff --check`。

### 2026-08-21 - 增加与离线 GRES N_acc 对齐的 no-target reward 配置

- 代码：修改 `verl/workers/opsd/config.py`、`verl/workers/config.py`、`verl/workers/opsd/mask_iou.py`、`verl/workers/opsd/__init__.py`、`verl/workers/fsdp_workers.py`、`verl/trainer/ray_trainer.py`、`verl/workers/reward/function.py`、`projects/rl/reward_function/text2mask.py`、`projects/rl/config.yaml`、`projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 和 `tests/test_opsd_core.py`。
- 文档：更新第 3.5 节 no-target 奖励契约并追加本日志；未新增、移动或删除模块。
- 行为：新增 `worker.opsd.pixel_iou.no_target_reward_mode=text|pixel_empty`，默认 `text` 保持原有的 `No target.`/mask-fragment 文本代理和无额外 VQ-SAM2 解码。opt-in `pixel_empty` 要求 OPSD pixel-IoU 已启用；FSDP worker 在 reward 前解析每条 no-target response 的所有完整、codebook 合法 SAMTok group，用与离线 `legacy_union` 相同的 VQ-SAM2 threshold 解码并集。空并集（含无合法 group、残缺 group 与零像素解码）奖励 `1.0`，非空并集奖励 `0.0`，不依赖 `No target.` 文本，因此和离线 GRES `N_acc` 的 `not pred_mask.any()` 完全同义。此二元正确性项仍与既有 non-repeat 项相加，未改变其权重或正例 CycleGRPO reward。
- 验证：本机 `PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile` 覆盖全部改动 Python 文件、`bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 和 `git diff --check` 均通过。`python3 -m unittest tests.test_no_target_reward tests.test_opsd_core` 在导入前因本机 Python 未安装 `torch` 失败，未执行测试主体；完整 FSDP decoder smoke 仍必须在服务器以包含 `gres_no_target` 的 `MAX_STEPS=1` 运行确认。

### 2026-08-21 - 允许 20k RefCOCO 与 20k no-target 共同进行 direct GRPO 和 SFT

- 代码：修改 `projects/rl/datasets/prepare_grefcoco_cycle_dataset.py`、`verl/workers/supervised_anchors.py`、`verl/trainer/ray_trainer.py`、`projects/rl/config.yaml`、`projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 和 `tests/test_supervised_anchors.py`。
- 文档：更新第 3.4、3.5、关键注意事项 27 和本日志；未新增、移动或删除模块。
- 行为：direct loader 现在可拼接 RefCOCO 正例 `DIRECT_TRAIN_DATA` 与 gRefCOCO no-target `DIRECT_NO_TARGET_TRAIN_DATA`。允许 source 由启用的 positive/no-target 配置严格决定，拒绝 label/template 或未启用的 source。no-target direct GRPO 使用与主 GRES 相同的 text 或 opt-in pixel-empty reward。新增 `direct_mask_ce.include_no_target` / `DIRECT_MASK_CE_INCLUDE_NO_TARGET`；启用时 no-target SFT teacher-force 原始 `<answer>No target.</answer>` token，CE 覆盖该完整文本、仍屏蔽 EOS/padding，和正例 mask-token SFT 在同一步独立累积。训练日志新增 direct GRPO rollout 与 direct SFT sample 的正例/no-target 计数。gRefCOCO 转换器允许 `--positive-samples 0 --no-target-samples N`，纯 negative 导出不初始化 VQ-SAM2 且不写空正例 parquet。
- 验证：`PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile projects/rl/datasets/prepare_grefcoco_cycle_dataset.py verl/workers/supervised_anchors.py verl/trainer/ray_trainer.py tests/test_supervised_anchors.py`、`PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_supervised_anchors`（16 tests）、`bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 与 `git diff --check` 通过。本机因 Python 未安装 `torch`，`tests.test_no_target_reward tests.test_opsd_core` 无法导入；服务器仍需以两个各 20k parquet、`MAX_STEPS=1` 检查 no-target GRPO/SFT count 和 no-target reward。

### 2026-08-21 - 修复独立 direct loader 的多模态字段兼容性

- 代码：修改 `verl/workers/supervised_anchors.py`、`verl/trainer/ray_trainer.py` 和 `tests/test_supervised_anchors.py`。
- 文档：更新第 3.4 节独立 direct GRPO/SFT 的 media 字段契约并追加本日志；未新增、移动或删除模块。
- 行为：direct GRPO 和 direct SFT 构造 localization batch 时，现在同时接受主 cycle rollout 的 `multi_modal_data` 与独立 dataloader 保留的 `cap_multi_modal_data`，并始终以 `seg_multi_modal_data` 作为 segmentation media。此前独立 direct loader 在第一个 batch 尚未经过 caption rollout 字段重命名时，访问不存在的 `multi_modal_data`，使训练在 step 1 抛出 `KeyError`。数据比例、prompt、reward、SFT target 和梯度权重均未改变。
- 验证：`PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_supervised_anchors`（17 tests）、受影响 Python 的 AST 语法检查和 `git diff --check` 通过。本机缺少 PyTorch，无法构造真实 `DataProto`/FSDP batch；服务器以 `MAX_STEPS=1` 重新启动三流训练，确认通过 direct GRPO/SFT batch 构造。

### 2026-08-22 - 固定当前服务器的默认离线评测路径

- 代码：修改 `projects/eval/qwen3vl_4b_volcengine.sh`。
- 文档：更新第 2.2、5.3 节的评测入口默认值并追加本日志；未新增、移动或删除模块。
- 行为：统一评测入口默认使用 `/volume/ybo/xyc`、`envs/cyclegrpo`、`Qwen3-VL-4B-SAMTok`、`cyclegrpo20k_direct30k_notarget10k_dlcqa10k/checkpoints/global_step_714` 及对应 `evaluation/step_714` 输出目录；默认 FSDP/benchmark GPU 数改为 7，为 `cuda:7` 的 Llama judge 保留独立设备。所有路径和 `NUM_GPUS` 仍可通过环境变量覆盖。
- 验证：`bash -n projects/eval/qwen3vl_4b_volcengine.sh` 与 `git diff --check` 通过；服务器需确认 checkpoint shard 为 world-size 7 并依次运行 export、各 benchmark action。

### 2026-08-22 - 修正 GroundingSuite 发布包默认路径

- 代码：修改 `projects/eval/qwen3vl_4b_volcengine.sh` 和 `README.md`。
- 文档：更新第 2.2、导出与评测章节，明确当前服务器的 GroundingSuite JSONL 与资源根目录。
- 行为：GroundingSuite 默认根目录从旧的 `/volume/ybo/xyc/GSEval` 改为
  `/volume/ybo/xyc/third_party/GroundingSuite`，默认数据文件因此解析为该目录下的
  `GroundingSuite-Eval.jsonl`；仍可用 `GROUNDINGSUITE_ROOT` 和
  `GROUNDINGSUITE_DATASET` 覆盖。评测协议、mask 解码和输出目录不变。
- 验证：执行 `bash -n projects/eval/qwen3vl_4b_volcengine.sh` 与 `git diff --check`。

### 2026-08-24 - 增加 direct CE 与三流非 CE 梯度余弦诊断

- 代码：修改 `verl/workers/supervised_anchors.py`、`verl/workers/config.py`、`verl/workers/fsdp_workers.py`、`verl/trainer/ray_trainer.py`、`projects/rl/config.yaml` 和 `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`。
- 行为：新增 `direct_mask_ce.record_base_gradient_cosine` 及对应环境变量。启用且存在 direct CE batch 时，trainer 先累积 CycleGRPO caption/localization、direct GRPO、DLC-QA 和其他 caption auxiliary，FSDP 暂存完整非 CE 梯度；direct CE backward 后跨 rank 记录 base/CE 范数、dot-product cosine 与冲突指示，再恢复 base 并合并 CE，训练更新保持不变。该诊断与非对称 caption-to-segmentation 梯度投影互斥。
- 文档：更新第 3.4 节，说明诊断指标、包含的梯度范围和不改变 optimizer 更新的保证；未新增、移动或删除模块。
- 验证：执行受影响 Python 文件 AST/compile 检查、`bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 和 `git diff --check`；本机无 CUDA/Ray/FSDP 环境，未执行多卡端到端训练。

### 2026-08-24 - 记录 disjoint 诊断训练环境变量契约

- 代码：仅修改 `code.md`，未修改训练实现或默认配置。
- 行为：记录当前服务器的基础路径、7 卡拓扑、20k 主 cycle 数据、2:4:1 batch 配额、direct/CE/DLC-QA 环境变量及权重；明确 `DIRECT_NO_TARGET_TRAIN_DATA` 必须由服务器实际文件校验后传入，避免使用未验证的历史路径。
- 验证：检索 `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`、`README.md`、`code.md` 中的环境变量契约；此前启动失败输出确认 no-target 文件检查是唯一阻断项。

### 2026-08-24 - 增加 direct mask CE 线性 warmup

- 代码：修改 `verl/workers/supervised_anchors.py`、`verl/trainer/ray_trainer.py`、`projects/rl/config.yaml` 和 `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`。
- 行为：新增 `direct_mask_ce.warmup_start_step/end_step` 及对应环境变量。CE 在 start 及之前为零，在 start 与 end 之间线性升至 `loss_weight`，end 及之后保持目标权重；默认 `0/0`，保持历史固定 CE 权重。每 step 记录 `supervised_anchors/direct_mask_ce_weight_effective`。这用于消除诊断中 step 1-10 的高强度 CE 反向梯度窗口，不改数据、direct GRPO、DLC-QA 或后期 CE 权重。
- 文档：更新第 3.4 节，说明权重函数和日志指标；未新增、移动或删除模块。
- 验证：执行受影响 Python 文件 compile、训练入口 `bash -n` 和 `git diff --check`；本机没有 CUDA/Ray/FSDP，未运行端到端训练。

### 2026-08-24 - 修复训练期 DLC-QA Llama judge 的输出截断

- 代码：修改 `projects/rl/reward_function/llm_judge_reward.py` 和 `tests/test_supervised_anchors.py`。
- 文档：更新第 3.5 节 caption QA judge 的请求契约；未新增、移动或删除模块。
- 行为：训练期 QA 请求显式传递 Llama-3.1 `<|eot_id|>` 的 `stop_token_ids=[128009]`。没有该参数时，缺失
  chat template 的 HF tokenizer 会让服务在 `A` 后继续输出 `assistant`，严格 `_parse_option` 将其记为失败，
  造成 QA reward 被大量置零；现在与离线 DLC evaluator 使用相同的停止语义。
- 验证：新增 mock OpenAI client 单测，检查停止 token 且确认 `A` 可解析；执行
  `PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile projects/rl/reward_function/llm_judge_reward.py tests/test_supervised_anchors.py`、
  `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_supervised_anchors` 和 `git diff --check`。

### 2026-08-25 - 增加三流 pairwise 梯度冲突诊断

- 代码：修改 `verl/workers/supervised_anchors.py`、`verl/workers/config.py`、`verl/workers/fsdp_workers.py`、
  `verl/trainer/ray_trainer.py`、`projects/rl/config.yaml`、`projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 和
  `tests/test_supervised_anchors.py`。
- 文档：更新第 3.4 节的 direct/CE 梯度诊断契约；未新增、移动或删除模块。
- 行为：新增 opt-in `gradient_diagnostics.enabled`。开启后记录 Cycle caption、Cycle segmentation、direct GRPO、
  DLC-QA GRPO、direct mask CE，以及实际启用的 regenerate CE/JSD 的加权梯度范数、所有可用两两余弦和负内积
  冲突标志，不修改反向累计或 optimizer step。该模式与 caption-to-segmentation 投影、旧 direct CE/base 聚合
  诊断互斥，并在单步完成后释放 FSDP gradient snapshot；仅适合短期诊断运行。
- 验证：待执行受影响 Python compile、`python3 -m unittest tests.test_supervised_anchors`、入口 `bash -n`、
  `git diff --check`；服务器需以完整五流配置运行 30--50 step，确认日志包含 pairwise metrics 且没有 OOM。

### 2026-08-25 - 修复 pixel-empty no-target 与 cycle caption 合并失败

- 代码：修改 `verl/trainer/ray_trainer.py` 和 `tests/test_supervised_anchors.py`。
- 文档：更新第 3.5 节 no-target reward 的 metadata 生命周期，并追加本日志；未新增、移动或删除模块。
- 行为：`pixel_empty` no-target 解码产生的 `no_target_pixel_empty`、`no_target_reward_mode` 在其 reward/advantage
  已计算后、与正例 cycle caption batch 合并为 PPO actor batch 前从 `non_cycle_batch` 移除，并对 cycle 侧执行同一
  清理。此前字段只存在于 non-cycle 子 batch；`DataProto.concat` 未补齐 cycle 一侧字段，导致长度为 no-target
  rollout 数的数组被带入总 batch，并在 `check_consistency` 抛出断言。该修复不改变 decoded-union no-target 奖励、
  优势、PPO loss 或正例训练。
- 验证：`PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile verl/trainer/ray_trainer.py tests/test_supervised_anchors.py`、
  `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_supervised_anchors`（20 tests）与
  `git diff --check` 通过；本机无 PyTorch/CUDA/Ray/FSDP，服务器仍需以
  `NO_TARGET_REWARD_MODE=pixel_empty` 运行 `MAX_STEPS=1`，确认可越过 caption batch concat 并记录 no-target reward。

### 2026-08-25 - 修复纯 CycleGRPO 梯度诊断的开关作用域

- 代码：修改 `verl/trainer/ray_trainer.py` 和 `tests/test_supervised_anchors.py`。
- 文档：更新第 3.4 节梯度诊断的无辅助 loader 契约；未新增、移动或删除模块。
- 行为：`multitask_gradient_diagnostics_enabled` 改为在每个训练 step 的公共路径初始化，而非仅在
  `direct_parent_batch` 存在时初始化。启用诊断但只训练 CycleGRPO 时，cycle caption/segmentation 分量可以正常
  捕获；禁用诊断或不存在对应 batch 时保持无操作。direct GRPO、DLC-QA、CE、OPSD 及既有梯度累计顺序不变。
- 验证：`PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile verl/trainer/ray_trainer.py tests/test_supervised_anchors.py`、
  `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_supervised_anchors` 和 `git diff --check`。

### 2026-08-26 - 增加训练完成后的 CUDA 保活工具

- 代码：新增 `tools/cuda_keepalive.py`、`tools/run_official_cyclegrpo_keepalive.sh`。
- 文档：更新第 5.1 节模块清单；未修改官方 CycleGRPO 训练主循环或算法配置。
- 行为：官方训练包装入口调用未修改的 `/volume/ybo/xyc/CycleGRPO` 主循环，只有训练成功退出后才启动保活工具。
  保活工具按 `CUDA_VISIBLE_DEVICES` 遍历可见 GPU，逐卡预留可配置的小块显存并保持进程运行；训练失败时
  不会启动保活进程，收到 SIGTERM/SIGINT 后释放预留并退出。入口支持可选的 `MAX_STEPS` 环境变量；设置
  `MAX_STEPS=1` 时只运行一个官方训练 step，成功后立即进入保活，未设置时仍运行完整一轮 epoch。
- 验证：`PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile tools/cuda_keepalive.py`、
  `bash -n tools/run_official_cyclegrpo_keepalive.sh` 和 `git diff --check`；本机无 CUDA 运行时，未执行实际显存保活 smoke test。

### 2026-08-26 - 将 CUDA 保活默认显存调整为约 40 GiB

- 代码：修改 `tools/cuda_keepalive.py` 和 `tools/run_official_cyclegrpo_keepalive.sh`。
- 文档：更新第 5.1 节模块说明和本变更日志。
- 行为：保活工具和官方训练包装入口默认改为每张可见 GPU 预留 `40000 MiB`；仍可通过
  `KEEPALIVE_MEMORY_MB` 或 `--memory-mb` 覆盖。训练算法、数据和 checkpoint 行为不变。
- 验证：`PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile tools/cuda_keepalive.py`、
  `bash -n tools/run_official_cyclegrpo_keepalive.sh` 和 `git diff --check`；本机无 CUDA，未执行真实显存分配。

### 2026-08-26 - 支持官方训练一 step 后进入 CUDA 保活

- 代码：修改 `tools/run_official_cyclegrpo_keepalive.sh`。
- 文档：更新第 5.1 节包装入口说明和本变更日志。
- 行为：包装入口通过可选 `MAX_STEPS` 注入官方 `trainer.max_steps`；`MAX_STEPS=1` 可用于短时验证，
  训练成功后继续执行默认约 40000 MiB/卡的 CUDA 保活，未设置时保持完整 epoch 行为。
- 验证：`bash -n tools/run_official_cyclegrpo_keepalive.sh` 和 `git diff --check`；本机无服务器 CUDA/官方依赖，
  未执行实际 one-step training 或显存保活。

### 2026-08-26 - 修复官方训练结束验证阻断 CUDA 保活

- 代码：新增 `tools/patch_official_final_validation.py`。
- 文档：更新第 5.1 节模块清单和本变更日志；说明该补丁作用于独立的官方仓库，不改变 CycleGRPO 更新逻辑。
- 行为：幂等补丁将官方 trainer 的训练后 validation 改为仅在 `trainer.val_freq>0` 时执行。此前即使
  `val_freq=-1`，训练结束仍会调用 `_validate()`，当前官方 loader 的字段不兼容时会在 vLLM 预处理阶段
  触发 `AttributeError`，阻止最终 checkpoint 保存及后续 CUDA 保活。补丁不改变 step 内训练、reward、loss、
  optimizer 或显式正频率 validation；未知版本不会强行改写。
- 验证：`PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile tools/patch_official_final_validation.py` 和
  `git diff --check`；本机未修改官方仓库，未执行服务器训练。

### 2026-08-26 - 对齐 70k 三流训练的默认 batch 与 step

- 代码：修改 `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`。
- 文档：更新第 2.2、3.2 节及本变更日志。
- 行为：火山引擎入口默认主 CycleGRPO、direct supervision、DLC-QA parent batch 分别为
  `128`、`256`、`64`，并默认 `MAX_STEPS=156`，使 20k/40k/10k 三条独立 loader 在约
  156 个 optimizer step 内各消费一遍；显式环境变量仍可覆盖，`MAX_STEPS=""` 可恢复完整 epoch。
- 验证：`bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`、`git diff --check`，并用 Bash
  空/未设置变量检查确认默认 `156` 与显式空值的完整 epoch 分支；本机无 8 卡 Ray/FSDP 环境，未执行端到端训练。

### 2026-08-27 - 恢复历史 20k OPSD 的 localization 响应上限

- 代码：修改 `verl/workers/opsd/config.py`、`projects/rl/config.yaml` 和
  `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`。
- 文档：更新第 2.2 节与本变更日志；未新增、移动或删除模块。
- 行为：`pixel_iou.segmentation_max_response_tokens` 的 dataclass、YAML 与火山入口默认值统一从
  `32` 恢复为 `256`。历史 RefCOCO 77 运行没有专用 segmentation cap，localization rollout 因而
  继承全局 `data.max_response_length=256`；恢复后保持这一有效上限。该项只改变训练期
  localization 生成的最大长度，不改变 caption 上限、G/K、union 解码、IoU/reward、routing、数据或 loss。
- 验证：`PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile verl/workers/opsd/config.py`、
  `bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 与 `git diff --check` 通过；本机无
  服务器 CUDA/Ray/FSDP 环境，未执行端到端训练。

### 2026-08-28 - 增加四组两节点多机训练编排

- 代码：修改 `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`；新增
  `tools/multinode/launch_four_trials.sh`、`tools/multinode/clusters.tsv.example` 和
  `projects/rl/experiments/multinode/trial_01.env` 至 `trial_04.env`。
- 文档：更新第 2.2、3.1、5.1、5.3、6 节及本变更日志；模块清单已记录新增控制器、清单模板与 env
  模板。
- 行为：初始设计的 16-training-GPU/外部 judge 拓扑已被 2026-08-30 的每节点 `7+1` 拓扑替代；请以最新
  记录为准。默认单机仍清除平台注入的 `RAY_ADDRESS`，并固定 `trainer.nnodes=1`。多机控制器读取四行
  TSV，对每组独占的 head/worker 启动独立 Ray head/worker、namespace、短 `/dev/shm` 目录、训练
  launcher、PID/state 和控制日志；默认拒绝已有 Ray，仅 `CLEAN_RAY=true` 允许清理清单中的目标节点。
- 验证：执行 `bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`、
  `bash -n tools/multinode/launch_four_trials.sh`、`launch/status/stop --dry-run` 与
  `git diff --check`。本机没有八台 H20、共享 `/volume`、Ray/vLLM 或 SSH 目标，未执行两节点
  preflight、`MAX_STEPS=1` smoke test 或四 trial 并发实机验证。

### 2026-08-28 - 将 CUDA 保活改为按卡空闲监测

- 代码：修改 `tools/cuda_keepalive.py`。
- 文档：更新第 5.1 节工具说明及本变更日志；未新增、移动或删除模块。
- 行为：工具不再启动即为全部可见卡分配显存。它每 15 秒通过 `nvidia-smi --query-gpu=index,memory.used`
  检查整卡使用量，仅在使用量严格低于默认阈值 `1 MiB` 时，为该卡预留默认 `40000 MiB`；已有训练、vLLM
  或其他 CUDA 进程占用的卡保持未分配，待其释放后才补位。已由本工具预留的卡保持占用直到收到
  `SIGTERM`/`SIGINT`。工具要求使用数值形式且与 `nvidia-smi` 报告的物理卡一致的 `CUDA_VISIBLE_DEVICES`，例如
  `0,1,2,3`，以将 nvidia-smi 的物理卡统计正确映射到 PyTorch device。首次检查只调用
  `nvidia-smi`，不调用 `torch.cuda`；因此监测进程自身不会先创建 CUDA context 并将空卡误判为超过
  1 MiB，只有确认该卡空闲后才初始化 PyTorch/CUDA 并预留显存。
- 验证：执行 Python AST 语法解析、`python3 -B tools/cuda_keepalive.py --help`、
  `bash -n tools/run_official_cyclegrpo_keepalive.sh` 与 `git diff --check`；本机没有 CUDA/nvidia-smi，
  未执行真实显存监测或分配 smoke test。

### 2026-08-30 - 将 RefCOCO 评测响应上限对齐至 256

- 代码：修改 `evaluation/refcoco/qwen3vl_refcoco_eval.py`、
  `evaluation/refcoco/run_refcoco_multigpu.sh` 和 `projects/eval/qwen3vl_4b_volcengine.sh`；未新增、
  移动或删除模块。
- 文档：更新第 2.2、5.6 节及本变更日志。
- 行为：RefCOCO 推理从原先硬编码的 `max_new_tokens=128` 改为默认 `256`，并通过
  `REFCOCO_MAX_NEW_TOKENS`/`MAX_NEW_TOKENS` 逐层传递到多 GPU worker；评测默认与当前训练的
  response 256 上限一致，显式设为 128 仍可进行旧上限对照。输出协议、mask 解码、指标计算和其他
  benchmark 不变。
- 验证：执行 `PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile evaluation/refcoco/qwen3vl_refcoco_eval.py`、
  `bash -n evaluation/refcoco/run_refcoco_multigpu.sh`、`bash -n projects/eval/qwen3vl_4b_volcengine.sh`
  和 `git diff --check`；本机无 CUDA/模型数据，未执行服务器端到端评测。

### 2026-08-30 - 将 GroundingSuite 与 GRES 分割评测响应上限对齐至 256

- 代码：修改 `evaluation/groundingsuite/qwen3vl_groundingsuite_infer.py`、
  `evaluation/groundingsuite/run_groundingsuite_multigpu.sh`、`evaluation/gres/qwen3vl_gres_eval.py`、
  `evaluation/gres/run_gres_multigpu.sh` 和 `projects/eval/qwen3vl_4b_volcengine.sh`；未新增、移动或删除模块。
- 文档：更新第 2.2、5.6 节及本变更日志。
- 行为：GroundingSuite 和 GRES/gRefCOCO 的分割生成从硬编码 `128` 改为默认 `256`，并分别通过
  `GROUNDINGSUITE_MAX_NEW_TOKENS`、`GRES_MAX_NEW_TOKENS` 及对应 wrapper 参数透传；统一评测入口的
  RefCOCO、GroundingSuite、GRES 三类 mask 评测现在与训练期 localization 的 256 上限一致。DLC caption
  的 192-token 协议及 mask 解码、指标计算保持不变。
- 验证：执行受影响 Python 文件 compile、`bash -n` 检查两个多 GPU wrapper 和统一评测入口，以及
  `git diff --check`；本机无 CUDA/模型数据，未执行服务器端到端评测。

### 2026-08-30 - 将四组多机训练改为每节点 7 训练卡加 1 本机 Llama

- 代码：修改 `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`、
  `tools/multinode/launch_four_trials.sh` 和 `projects/rl/experiments/multinode/trial_01.env` 至
  `trial_04.env`；未新增、移动或删除模块。
- 文档：更新第 2.2、3.1、5.1、5.3、6 节及本变更日志。
- 行为：显式两节点模式从每节点 8 Ray GPU 改为每节点 7 Ray 训练 GPU，要求 `NNODES=2`、
  `NUM_GPUS=7`、`RAY_CLUSTER_EXPECTED_GPUS=14`。多机控制器以 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6`
  启动 Ray head/worker 和 trainer，并在每台机器用物理 GPU 7 后台启动本机 Llama-3.1 vLLM、检查
  tokenizer chat template、端口空闲与 `/v1/models` 健康状态；`status`/`stop` 同时管理两个 judge PID。
  四个模板改为 batch/step/segmentation-response 的 `28/714/32`、`28/714/256`、`112/178/32`、
  `112/178/256` 受控对照。DLC-QA 启用时当前单 URL reward 配置使用 trial head 的 Llama；worker
  Llama 不参与自动负载均衡。默认单机 `NUM_GPUS=8` 行为不变。
- 验证：执行训练入口和多机控制器 `bash -n`、四个 trial env 的 `bash -n`、四组 `launch/status/stop`
  dry-run，以及 `git diff --check`；本机无八台 H20、共享 `/volume`、Ray/vLLM 或 SSH 目标，未执行
  真正的 14-world-size smoke training、GPU 7 vLLM 启动或四 trial 并发验证。

### 2026-08-30 - 替换已运行的多机 response 条件为 128-token 对照

- 代码：修改 `projects/rl/experiments/multinode/trial_01.env` 与 `trial_04.env`；未新增、移动或删除模块。
- 文档：更新第 2.2 节的多机试验矩阵与模块清单，并追加本日志。
- 行为：已完成的 `28/714/32` 改为 `28/714/128`，已完成的 `112/178/256` 改为
  `112/178/128`；因此当前四个两节点 `7+1` trial 为 `28/714/128`、`28/714/256`、
  `112/178/32`、`112/178/128`。其他训练开关、batch、step、Ray/Llama GPU 拓扑均不变。
- 验证：执行四个 trial env 的 `bash -n`、多机控制器 `launch --dry-run` 与 `git diff --check`；
  未在真实集群启动。

### 2026-08-30 - 删除训练链路中的 groundedness verifier

- 代码：删除 `verl/workers/opsd/groundedness.py`；修改 `verl/workers/opsd/{__init__,config}.py`、
  `verl/workers/opsd/routing.py`、`verl/trainer/ray_trainer.py`、`verl/workers/fsdp_workers.py`、
  `verl/workers/reward/function.py`、`projects/rl/reward_function/text2mask.py`、
  `projects/rl/config.yaml`、`projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`、`tests/test_opsd_core.py`、
  `README.md`。
- 行为：移除 groundedness verifier rollout、claim penalty、candidate/JSD gate、token mask/weight 和
  相关配置；caption reward、regenerate、privileged JSD 与 DLC-QA 不再读取 groundedness 字段。
  现有 teacher analysis、caption safety 和 teacher confidence 保持不变。
- 模块清单：移除 `workers/opsd/groundedness.py` 条目。
- 验证：执行受影响 Python 文件 `py_compile`、训练入口 `bash -n`、`git diff --check`，并确认运行代码中无 groundedness 引用；未执行 Ray/FSDP/CUDA 训练。

### 2026-08-30 - 增加训练 mask 解码与 RefCOCO prompt 配置

- 代码：修改 `verl/workers/opsd/config.py`、`verl/workers/opsd/mask_iou.py`、`verl/workers/fsdp_workers.py`、
  `verl/workers/supervised_anchors.py`、`verl/trainer/ray_trainer.py`、
  `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 和 `tests/test_opsd_core.py`。
- 文档：更新第 2.2、3.4、5.1 节及关键注意事项 24、26；未新增、移动或删除模块。
- 行为：新增 `MASK_DECODE_MODE=union|first_mask`，默认保持完整合法 group 的 union，设置为
  `first_mask` 时只解码第一个合法 group，以复现原始单 group 训练语义；新增
  `LOCALIZATION_PROMPT_MODE=mixed|refcoco|groundingsuite|legacy`，默认保留 RefCOCO/GroundingSuite
  交替，设置为 `refcoco` 时 cycle、direct GRPO 和 direct CE 的图像 localization prompt 全部使用
  `Please segment ... in this image.`。两个开关均透传至 no-target pixel-empty 解码或 direct prompt 构造。
- 验证：执行受影响 Python 文件 `py_compile`、新增配置/解码/prompt 单元测试、训练入口 `bash -n` 和
  `git diff --check`；本机无 CUDA/Ray/FSDP，未执行端到端训练。

### 2026-08-30 - 增加 source-aware 官方两阶段 prompt 开关

- 代码：修改 `verl/utils/dataset.py`、`verl/trainer/config.py`、`verl/trainer/data_loader.py`、
  `projects/rl/config.yaml`、`projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 和
  `tests/test_opsd_core.py`。
- 行为：新增 `data.cycle_prompt_mode` 与环境变量 `CYCLE_PROMPT_MODE`。默认 `current` 完全保留现有
  parquet caption prompt 及 `LOCALIZATION_PROMPT_MODE` 行为；`official_source_aware` 按
  `refcoco_cycle/denseworld_single`、`grefcoco_cycle/denseworld_multiple`、`paco_part_cycle`、
  `cocostuff_cycle` 分别重建单区域、官方多区域 interleaved、visible-parts 和 semantic-region
  caption 模板，并将图像 localization 统一切换到官方长 segmentation 模板。no-target、未知 source、视频和 bbox
  分支不被错误套用图像 source 模板；开关同时作用于 train/val dataloader。
- 论文边界：这是可选的 prompt 分布对照，不改变 CycleGRPO reward、mask 解码、G/K、OPSD 路由或优化公式；
  source-aware PACO/Stuff 文案是当前数据类别语义模板，不宣称为官方论文新增 prompt。
- 验证：执行 `python3 -m py_compile verl/utils/dataset.py verl/trainer/config.py verl/trainer/data_loader.py tests/test_opsd_core.py`、
  `bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`、source-template 单测检查和
  `git diff --check`；本机缺少完整 PyTorch/Ray/vLLM/CUDA，未执行多卡训练。

### 2026-08-30 - 固定四组纯 20k 自监督 prompt/decode 消融

- 代码：修改 `projects/rl/experiments/multinode/trial_01.env` 至 `trial_04.env`、
  `tools/multinode/launch_four_trials.sh` 的模板注释和 `tools/multinode/clusters.tsv.example`；未新增、
  移动或删除模块。
- 行为：四个两节点 trial 统一只读取 20k `TRAIN_DATA`，关闭 `DIRECT_GROUNDING_ENABLED`、
  `DIRECT_MASK_CE_ENABLED`、`SUPERVISED_CAPTION_QA_ENABLED`，并设置 `LOCAL_JUDGE_ENABLED=false`，
  因而不启动 Llama、也不消费 40k/10k/DLC-QA 数据。四组均为 `ROLLOUT_BATCH_SIZE=128`、
  `ACTOR_GLOBAL_BATCH_SIZE=128`、`MAX_STEPS=156`、`CAPTION/SEGMENTATION response=256`、`G=K=6`，
  仅比较官方 source-aware 两阶段 prompt 或当前 RefCOCO prompt，以及 `first_mask` 或 `union` 解码：
  `official+first`、`refcoco+first`、`official+union`、`refcoco+union`。
- 资源：每个 Ray 集群为两节点、每节点 GPU 0--7 共 16 张训练卡；不启动 judge，四组共 64 张训练卡。
  OPSD、pixel-IoU、routing/regenerate 和 teacher 诊断仍保持各 env 原值，未关闭自监督
  CycleGRPO 主链路。
- 验证：四个 env 和 `tools/multinode/launch_four_trials.sh` 通过 `bash -n`；使用
  `INVENTORY=tools/multinode/clusters.tsv.example tools/multinode/launch_four_trials.sh launch --dry-run`
  验证四行 TSV、Ray 检查命令和 judge 条件分支；`git diff --check` 通过。未连接真实 H20 节点，未启动
  多机训练或 Llama 服务。

### 2026-08-30 - 四组纯自监督切换为 8 卡节点与 batch 128

- 代码：修改 `projects/rl/experiments/multinode/trial_01.env` 至 `trial_04.env`、
  `tools/multinode/launch_four_trials.sh`、`projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`、
  `tools/multinode/clusters.tsv.example`；未新增、移动或删除模块。
- 行为：当前四组纯 20k 实验统一使用每节点 8 张 Ray GPU（两节点共 16 张/实验），`ROLLOUT_BATCH_SIZE`
  和 `ACTOR_GLOBAL_BATCH_SIZE` 从 112 改为 128，完整一轮对应 `MAX_STEPS=156`；四组仍关闭 Llama、
  direct GRPO、direct CE 和 DLC-QA。控制器按 `LOCAL_JUDGE_ENABLED` 动态选择 8 卡纯训练拓扑，只有
  judge-enabled 辅助实验才回退到每节点 7 卡训练加 GPU 7 judge。
- 验证：四个 env、训练入口和控制器通过 `bash -n`；dry-run 验证 16-GPU Ray 参数和四组 TSV 解析，
  `git diff --check` 通过。未连接真实多机节点或执行训练。

### 2026-08-30 - 改用平台 Ray/verl 多机框架

- 代码：重写 `tools/multinode/launch_four_trials.sh` 和 `tools/multinode/clusters.tsv.example`；更新
  `code.md` 的多机说明与模块清单，未新增、移动或删除训练模块。
- 行为：控制器不再通过 SSH 启动 Ray head/worker，不再执行 `ray start`/`ray stop`，也不要求清单提供
  NCCL 网卡。平台需预先提供四个独立的两节点 Ray 集群；清单改为 `trial_id`、`ray_address`、
  `ray_namespace`、`experiment_env` 四列。控制器通过 Ray API 验证每个集群为 2 节点、16 GPU，再在
  提交机启动对应的 verl trainer；`status`/`stop` 只管理本地 trainer PID。四组仍为纯 20k、8 卡/节点、
  batch 128、156 steps、无 Llama。
- 验证：执行控制器和训练入口 `bash -n`、四个 env `bash -n`、平台 Ray 清单 `launch --dry-run` 解析及
  `git diff --check`；未连接真实 Ray 集群或启动训练。

### 2026-08-30 - 固定 RefCOCO 批量推理使用 left padding

- 代码：修改 `evaluation/refcoco/qwen3vl_refcoco_eval.py`；未新增、移动或删除模块。
- 文档：更新第 2.2、5.6 节及本变更日志；模块清单无变化。
- 行为：加载 `AutoProcessor` 后固定设置 `processor.tokenizer.padding_side="left"`。RefCOCO 的
  batch size 1 与 batch size 4 对照显示，right padding 的 cIoU 为 64.67，而 left padding 为
  79.17，后者与无 padding 的 batch size 1（79.16）一致；该修复只改变评测批处理方式，不改变
  EOS、mask protocol、模型权重或训练逻辑。
- 验证：执行 `PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile evaluation/refcoco/qwen3vl_refcoco_eval.py`、
  `bash -n evaluation/refcoco/run_refcoco_multigpu.sh` 和 `git diff --check`；服务器端已完成 8 卡
  batch 4 left-padding RefCOCO 全量评测，得到 `cIoU=79.1668`、`mIoU=78.8606`。

### 2026-08-30 - 修正四组多机控制器的环境、namespace 与 PID 传递

- 代码：修改 `tools/multinode/launch_four_trials.sh` 与 `verl/trainer/main.py`；未新增、移动或删除模块。
- 文档：更新第 2.2、5.1、6 节及本变更日志；模块清单无变化。
- 行为：控制器以 `set -a` source trial env，使 `RUN_NAME`、`TRAIN_DATA`、prompt/decode 配置和所有训练
  参数都导出到 `nohup` 训练子进程；此前未导出的 shell 变量可能使子进程回退到 launcher 默认值。后台启动
  现在直接记录 `nohup bash`（最终 `exec` 到 trainer）的 PID，`stop` 的 SIGTERM 能作用于实际训练链路。
  Ray 预检改为在 `RAY_READY_TIMEOUT_SECONDS`（默认 120 秒）内轮询两节点/16-GPU 就绪状态。`main.py` 的
  `ray.init` 读取 `RAY_NAMESPACE`，使 TSV 中每个 trial 的 namespace 真正作用于 Ray driver 与其 actor。
- 验证：执行 `PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile verl/trainer/main.py`、
  `bash -n tools/multinode/launch_four_trials.sh`、四个 env 的 `bash -n`、`launch/status/stop --dry-run`
  及 `git diff --check`；本机未连接真实两节点 Ray 集群，尚未执行 16-GPU smoke training。

### 2026-08-30 - 拆分纯自监督与 70k 有监督四组多机控制器

- 代码：新增 `tools/multinode/launch_four_cycle_trials.sh`、
  `tools/multinode/launch_four_supervised_trials.sh`、`tools/multinode/supervised_clusters.tsv.example`、
  `tools/multinode/local_llama_judge.py`、`tools/multinode/llama3_chat_template.jinja` 和
  `projects/rl/experiments/multinode/supervised_trial_01.env` 至 `supervised_trial_04.env`。
- 文档：更新第 2.2、5.1、5.3、6 节与本变更日志，区分两套 controller、清单格式、数据/超参数与 `8` 对 `7+1`
  的 GPU 拓扑。
- 行为：纯 20k 的 `launch_four_cycle_trials.sh` 是现有 16-training-GPU controller 的明确入口，继续使用
  四列清单和 `128/156/256` 的 prompt/decode 矩阵。新增的 supervised controller 使用独立五列清单和四个
  70k 三流 env：每 trial 两节点、每节点 7 张 Ray 训练 GPU，batch 为 `112/224/56`、178 step、response 256；
  每节点物理 GPU 7 由 CPU-only node-affine detached Ray actor 启动的 Llama vLLM 独占。DLC-QA 指向清单指定的
  head judge URL；所有本机及由提交端访问的 head judge `/v1/models` 健康检查还必须匹配
  `LOCAL_JUDGE_SERVED_MODEL_NAME`，避免错误复用已有服务。该 node-local judge 方案不改变 CycleGRPO、direct GRPO、direct CE、DLC-QA reward 或优化公式。
- 验证：执行 `PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile tools/multinode/local_llama_judge.py`，对两个
  controller 与全部八个 env 执行 `bash -n`，执行两类 controller 的 `launch/status/stop --dry-run` 和
  `git diff --check`；本机未连接实际 Ray/H20/vLLM，未运行 14-world-size 或四 trial 并发 smoke test。

### 2026-08-30 - 重设四个有监督多机 trial 为 CE、batch 与累计剂量消融

- 代码：修改 `projects/rl/experiments/multinode/supervised_trial_01.env` 至
  `supervised_trial_04.env`；未修改训练实现、数据加载、reward 或 controller。
- 文档：更新第 2.2、5.3、6 节及本变更日志，记录固定配置和四个受控变量。
- 行为：四个 trial 固定 70k 数据、`current + mixed localization prompt + union`、`G=K=6` 和
  response 256。trial 01 为 `bs112/178-step, CE=0.005`；trial 02 为 `bs28/714-step, CE=0.02`；trial 03
  为 `bs28/714-step, CE=0.005` 且只对正例 direct mask CE，no-target 继续仅接受 direct GRPO；trial 04 为
  `bs28/714-step, CE=0.005`，direct GRPO/DLC-QA loss 从 `0.15/1.0` 缩至 `0.0375/0.25`，近似匹配
  bs112/178 baseline 的累计辅助剂量。28-batch 的 warmup 设为 `40--120`，对应 112-batch 的 `10--30`
  训练进度比例。
- 验证：对四个 supervised env 执行 `bash -n`，运行 supervised controller 的
  `launch/status/stop --dry-run`，并执行 `git diff --check`；未连接真实 Ray/H20/vLLM，未进行训练。

### 2026-08-30 - 有监督多机 sweep 恢复 v1/v2 routing 与诊断监督

- 代码：修改 `tools/multinode/launch_four_supervised_trials.sh` 和
  `projects/rl/experiments/multinode/supervised_trial_01.env` 至 `supervised_trial_04.env`。
- 文档：更新第 2.2、5.3、6 节及本变更日志，修正此前错误的“关闭 routing”表述。
- 行为：四个 70k trial 均固定 `ROUTING_ENABLED=true`、`EMA_TEACHER_ENABLED=true`、
  `TEACHER_ANALYSIS_ENABLED=true` 与 `PRESERVE_ORIGINAL_GRPO=true`，保持历史 v1/v2 的 teacher-routing
  和诊断监督路径。controller 在 launch/status/stop 解析 env 时强制检查前三个开关；任一开关退化为 false
  会在启动前失败。`THREE_STREAM_2_4_1_ENABLED=false` 仅用于避开该旧严格模式对 routing 的互斥要求，
  controller 仍强制 main/direct/DLC-QA parent batch 的实际比例为 `2:4:1`。
- 验证：执行 supervised controller 与四个 env 的 `bash -n`、`launch/status/stop --dry-run` 及
  `git diff --check`；未连接真实 Ray/H20/vLLM，未启动训练。

### 2026-08-30 - 将有监督 sweep 改为单 32-GPU 节点的 7+1 拓扑

- 代码：修改 `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`、
  `tools/multinode/launch_four_supervised_trials.sh`、`tools/multinode/local_llama_judge.py` 和
  `tools/multinode/supervised_clusters.tsv.example`。
- 文档：更新第 2.2、3.2、5.1、5.3、6 节及本变更日志。
- 行为：四个 70k 有监督 trial 不再由两节点 14-world-size Ray 集群运行；每个 trial 改为一台独占的
  32-GPU 节点上的 7-world-size Ray 集群。控制器要求恰有 1 个 alive Ray node 和 7 张已登记 Ray GPU，
  trainer 以 `MULTINODE_ENABLED=true, NNODES=1, NUM_GPUS=7` 连接该集群。物理 GPU 0--6 用于训练、GPU 7
  由单个 node-affine Llama judge actor 使用，GPU 8--31 不被本 trial 调度。训练入口的显式 Ray attach
  模式现在接受 `NNODES=1|2`；纯 20k 两节点 controller 行为不变。对 7 个 rank，batch `112/224/56`
  对应每 rank `16/32/8` parent prompt，`28/56/14` 对应 `4/8/2`；2:4:1、routing 和所有四组 loss
  消融语义不变。
- 验证：执行训练入口、supervised controller 和四个 env 的 `bash -n`，执行 supervised controller 的
  `launch/status/stop --dry-run`、`local_llama_judge.py` 无缓存语法编译及 `git diff --check`；未连接
  真实 Ray/H20/vLLM，未启动训练。

### 2026-08-30 - 将 pixel-empty no-target reward 移入主 segmentation rollout

- 代码：修改 `verl/trainer/ray_trainer.py`；未新增、移动或删除模块。
- 行为：`NO_TARGET_REWARD_MODE=pixel_empty` 时，主 20k `gres_no_target` 行从外层 caption batch
  移除，按原始 `grounding_query` 每个 UID 建立标准 segmentation rollout，并以
  `supervised_grounding_no_target` source 使用 decoded-union 空 mask + non-repeat reward 计算
  segmentation GRPO advantage。该梯度只更新 segmentation policy，不再更新 caption policy；
  `text` 与 `official_bbox` 模式保持原 caption 分支行为。不同响应长度的 cycle/no-target segmentation
  batch 不拼接，而是在同一 optimizer step 独立累积并按样本数分配 segmentation loss weight。
- 文档：更新第 3.3、3.4、3.5 节，说明 pixel-empty 的新数据流、reward 生命周期和梯度归属；模块清单无变化。
- 验证：执行 AST 语法解析、`git diff --check`；本机无 PyTorch/Ray/CUDA，未执行 GPU rollout smoke test，
  服务器需用 `NO_TARGET_REWARD_MODE=pixel_empty MAX_STEPS=1` 验证 no-target segmentation 计数和 reward 指标。

### 2026-08-30 - 将有监督多机控制器改为四组混合多任务实验

- 代码：修改 `tools/multinode/launch_four_supervised_trials.sh`、`tools/multinode/supervised_clusters.tsv.example`、
  `projects/rl/experiments/multinode/supervised_trial_01.env` 至 `supervised_trial_04.env`；未新增、移动或删除模块。
- 行为：原有监督 controller 现在按每个 env 动态支持单节点 8-GPU 纯自监督或单节点 7-GPU+GPU7 Llama
  有监督拓扑；不再强制四组都启动 judge。四组分别运行 pixel-empty 20k 自监督、pixel-empty 70k
  （20k+30k+10k+DLC-QA，bs112，direct GRPO/CE/DLC-QA 全开）、official-bbox no-target 20k 自监督和
  official source-aware prompt 的 pixel-empty 20k 自监督；20k 任务均 batch 128、156 step、G=K=6、response 256。
- 文档：同步更新第 2.2、5.1、5.3 节和模块职责，说明清单中无 judge 使用 `-`，以及四组资源/数据契约。
- 验证：执行四个 env 与 controller 的 `bash -n`、混合清单 `launch/status/stop --dry-run` 和
  `git diff --check`；dry-run 确认三组使用 1 节点/8 GPU 且不启动 judge，70k 组使用 1 节点/7 GPU
  并绑定 GPU 7 judge；未连接真实 Ray/H20 集群，未启动训练或 Llama 服务。

### 2026-08-31 - 将混合多任务四组统一限制为单节点 8 卡

- 代码：修改 `projects/rl/experiments/multinode/supervised_trial_01.env`、`supervised_trial_03.env`、
  `supervised_trial_04.env` 和 `tools/multinode/supervised_clusters.tsv.example`；未修改训练算法或数据。
- 行为：三个 20k 自监督任务的 `NNODES` 从 2 改为 1，仍由 Ray 登记 GPU 0--7、batch 128、156 step；
70k 任务保持单节点 7 卡训练并在 GPU 7 启动 Llama，因此每个任务物理最多使用 8 张卡。
- 验证：执行四个 env 与 controller 的 `bash -n`、混合清单 `launch --dry-run`、确认 4 个 Ray 检查和仅 1 个
  judge 启动命令，并通过 `git diff --check`；未连接真实 Ray/H20 集群。

### 2026-08-31 - 修复小批量 pixel-empty segmentation 的多卡 dispatch

- 代码：修改 `verl/trainer/ray_trainer.py`；未新增、移动或删除模块。
- 行为：`_make_seg_batch_data_for_caption` 在 rollout dispatch 前将 parent prompt 临时补齐到
  `world_size` 的整倍数，并在生成后移除对应的 `n` 条合成 response。主 20k 的 5% no-target 子批次
  即使只有 1--7 条也不会再触发 `DataProto.chunk` 的 equal-chunk assertion；补齐样本不进入后续 reward、
  advantage、指标或 loss 权重。
- 文档：更新第 3.4 节，说明总数据量与每 step 子批量的区别及 padding 生命周期。
- 验证：执行 `PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile verl/trainer/ray_trainer.py` 和
  `git diff --check`；未执行 GPU/Ray rollout smoke test。

### 2026-08-31 - 修复 pixel-empty rollout 元数据错位

- 代码：修改 `verl/trainer/ray_trainer.py`；未新增、移动或删除模块。
- 行为：segmentation rollout 的输入 padding 在生成后已移除，但元数据展开仍误用 padding 后的 prompt
  数量，导致 `localization_index` 等 non-tensor 字段与 response tensor batch 不一致。现在统一使用未补齐
  的原始 prompt 数量，并在构造元数据前检查输出恰为 `prompt_count × K`；不一致时立即报告明确错误。
- 验证：执行 `PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile verl/trainer/ray_trainer.py` 与
  `git diff --check`；未连接真实 Ray/CUDA，需服务器用 `NO_TARGET_REWARD_MODE=pixel_empty MAX_STEPS=1`
  复跑 no-target segmentation smoke test。

### 2026-09-01 - 收紧 pixel-empty no-target 拒识奖励

- 代码：修改 `verl/workers/opsd/mask_iou.py`、`verl/workers/fsdp_workers.py` 与 `tests/test_opsd_core.py`；未新增、移动或删除模块。
- 文档：更新第 3.5 节 pixel-empty reward 语义及本日志；模块清单无变化。
- 行为：`pixel_empty` 不再只因 decoded union 为空就给分。现在每条 segmentation rollout 必须同时包含大小写不敏感的精确 `No target.` 拒识短语，并解码为零像素 union，才得到 `1.0`；任何 nonempty mask、`null`、自由文本、空输出或仅 EOS 都得到 `0.0`。`text` 和 `official_bbox` reward 模式未改变。此为当前训练实现相对 GRES 单独 empty-mask N-acc 更严格的约束。
- 验证：`PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile verl/workers/opsd/mask_iou.py verl/workers/fsdp_workers.py tests/test_opsd_core.py` 与 `git diff --check` 通过；本机执行 `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_opsd_core` 时因未安装 `torch` 无法导入。未执行 GPU/Ray rollout，服务器应以 `NO_TARGET_REWARD_MODE=pixel_empty MAX_STEPS=1` 检查 no-target segmentation reward 不再为普通空输出给分。

### 2026-09-01 - 参数化主 pixel-empty no-target segmentation loss

- 代码：修改 `verl/workers/opsd/config.py`、`verl/trainer/ray_trainer.py`、`projects/rl/config.yaml`、`projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 与 `tests/test_opsd_core.py`；未新增、移动或删除模块。
- 文档：更新第 3.5 节 pixel-empty reward/loss 边界及本日志；模块清单无变化。
- 行为：新增 `worker.opsd.no_target_segmentation_loss_weight` / `NO_TARGET_SEGMENTATION_LOSS_WEIGHT`，默认 `1.0` 完全保留既有 sample-proportional no-target gradient。设置 `0.25` 时，仅主 20k pixel-empty no-target segmentation actor loss 在现有 effective weight 上再乘 `0.25`，不改变 reward、组内 GRPO advantage、cycle segmentation、caption 或 auxiliary loss；双任务与 segmenter-only 更新路径均适用。日志新增 target/base/effective no-target 权重，负值在配置校验时拒绝。
- 验证：受影响 Python 文件通过无缓存 AST 语法解析，`bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 与 `git diff --check` 通过；本机执行 `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_opsd_core` 时因未安装 `torch` 无法导入。服务器应以 `NO_TARGET_REWARD_MODE=pixel_empty NO_TARGET_SEGMENTATION_LOSS_WEIGHT=0.25 MAX_STEPS=1` 检查 effective weight 为历史值的四分之一。

### 2026-09-01 - 增加公开 CycleGRPO RefCOCO 评测兼容协议

- 代码：修改 `evaluation/mask_protocol.py`、`evaluation/refcoco/qwen3vl_refcoco_eval.py` 与 `tests/test_mask_protocol.py`；未新增、移动或删除模块。
- 文档：更新第 2.2、5.6 节与本变更日志；模块清单无变化。
- 行为：RefCOCO 新增仅用于交叉验证公开 CycleGRPO 脚本的 `MASK_PROTOCOL=cyclegrpo_legacy`。该模式强制 `batch_size=1`、`max_new_tokens=128`、`skip_special_tokens=True`，并完全复用公开脚本的容错规则：直接从 raw `<|mt_####|>` 配对、奇数 token 时仅从修复后的完整 wrapper 重新提取、首码本越界丢弃、第二码本越界传入 `-1`，最后解码所有保留 pair 的像素 union。它不加入共享的 `legacy_union|first_mask` 协议集合，GRES/GroundingSuite 不会接受该参数；严格模式的 EOS、解析、batch 和默认 256-token 上限均不变。RefCOCO 输出额外记录 raw/decoded token 数、实际 batch 与 token cap，且 `mask_protocol` 继续作为 resume 隔离字段；兼容评测必须使用全新的输出目录，不能与标准 RefCOCO 分数比较或混写。
- 验证：`PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_mask_protocol`（6 tests）、`PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile evaluation/mask_protocol.py evaluation/refcoco/qwen3vl_refcoco_eval.py tests/test_mask_protocol.py`、`bash -n evaluation/refcoco/run_refcoco_multigpu.sh` 与 `git diff --check` 通过。本机无 CUDA、HF checkpoint、VQ-SAM2 权重或 RefCOCO assets，未进行端到端推理；服务器应以新的输出目录运行八卡 `cyclegrpo_legacy` 交叉验证。

### 2026-09-01 - 将 pixel-empty no-target 恢复为官方 non-cycle 路径

- 代码：修改 `verl/trainer/ray_trainer.py`、`verl/workers/opsd/__init__.py`、`verl/workers/opsd/config.py`、`verl/workers/opsd/mask_iou.py`、`verl/workers/fsdp_workers.py`、`projects/rl/reward_function/text2mask.py`、`projects/rl/config.yaml`、`projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 与 `tests/test_opsd_core.py`；未新增、移动或删除模块。
- 文档：更新第 3.3--3.5、6 节、`Agent.md` 与本日志，废弃此前主 pixel-empty segmentation rollout/loss-weight 的当前路径描述；较早的变更日志保留为实验历史。
- 行为：`NO_TARGET_REWARD_MODE=pixel_empty` 的主 `gres_no_target` 不再由 `grounding_query` 创建 K 次 `supervised_grounding_no_target` segmentation rollout，而是与公开 CycleGRPO 一样保持 caption-only non-cycle batch。FSDP worker 在 caption reward 前解码 response，只有精确 `No target.` 加零像素 union 才写入 `no_target_pixel_empty=1.0`，随后继续常规 caption GRPO；移除了不再适用的 `NO_TARGET_SEGMENTATION_LOSS_WEIGHT`。新增 `POSITIVE_EMPTY_MASK_PENALTY` / `pixel_iou.positive_empty_mask_penalty`，默认 `1.0`：对非空正例 GT，明确拒识或 decoded union 为空时在 `seg_overall` 额外扣分，同时保留真实 `pixel_iou` 并记录 `seg_positive_empty_mask_penalty`。该项不作用于正确 no-target 拒识。
- 验证：受影响 Python 文件已通过无字节码 AST 语法解析，训练 shell 通过 `bash -n`，`git diff --check` 通过。`python3 -m unittest tests.test_opsd_core` 因本机系统 Python 未安装 `torch` 无法导入；未执行 GPU/Ray smoke test。服务器应以 `NO_TARGET_REWARD_MODE=pixel_empty MAX_STEPS=1` 确认主 no-target 只出现在 non-cycle caption 指标，且正例空 mask 的 penalty 指标为非正值。

### 2026-09-02 - 新增四卡显存与功耗占用工具

- 代码：新增 `tools/gpu_power_hold.sh`；模块清单同步更新。
- 行为：该独立运维脚本默认仅启动物理 GPU 0--3，每卡 worker 预留约 40000 MiB，并持续执行 BF16
  matrix multiplication 以保持 GPU 利用率和功耗。`GPU_LIST`、`MEMORY_MIB`、`MATMUL_DIM`、`PYTHON_BIN`
  与 `STATE_DIR` 均可覆盖；`status` 只读显示显存/功耗/PID，`stop` 只停止命令行带自身 worker tag 的
  已记录 PID，拒绝杀死任何非本工具进程。该工具不参与训练、评测、Ray 或 checkpoint 行为。
- 验证：执行 `bash -n tools/gpu_power_hold.sh` 与 `git diff --check`；本机无 CUDA/H20，未执行实际
  显存分配或功耗负载。

### 2026-09-02 - 修复 pixel-empty no-target caption batch 合并

- 代码：修改 `verl/trainer/ray_trainer.py`；未新增、移动或删除模块。
- 文档：更新第 3.4 节的 pixel-empty no-target metadata 生命周期；模块清单无变化。
- 行为：主 `pixel_empty` no-target non-cycle batch 在完成 reward、KL 和 GRPO advantage 后，现会在与
  cycle caption batch 合并前清理 `no_target_pixel_empty` 与 `no_target_reward_mode`。此前字段只存在于
  no-target 一侧，`DataProto.concat` 会保留长度为 no-target rollout 数的数组并与总 tensor batch 长度冲突，
  在第一个训练 step 触发一致性断言。修复不改变 strict `No target.` + decoded-empty 奖励、正样本空 mask
  penalty、已计算 advantage 或任何 loss 权重。
- 验证：执行 `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_supervised_anchors`、
  `PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile verl/trainer/ray_trainer.py` 与 `git diff --check`；
  本机未运行 CUDA/Ray 训练，服务器应以 `NO_TARGET_REWARD_MODE=pixel_empty MAX_STEPS=1` 确认越过
  第 0 step 的 caption concat。

### 2026-09-02 - 增加 CPU/Gloo FSDP checkpoint 离线重组导出

- 代码：新增 `tools/reassemble_fsdp_checkpoint.py` 和 `tests/test_fsdp_reassemble.py`；模块清单同步更新。
- 行为：新增不依赖 CUDA 的 HF 导出路径。脚本从 `actor/model_world_size_<N>_rank_*.pt` 自动发现并验证
  原始 world size，要求以同样数量的 CPU/Gloo rank 启动；每 rank 只读取自己的 checkpoint 文件，rank 0
  按 ShardedTensor 或一维 DTensor metadata 逐参数重建完整 CPU tensor，然后以 checkpoint 的 processor/config
  和原始 SAMTok model config 写出 safetensors。它拒绝缺失/混合 rank 文件、非空输出目录、重复 shard、未覆盖的
  ShardedTensor 或不支持的多维 DTensor，而不会静默产生不完整权重。该工具是现有 CUDA/FSDP exporter 的离线运维
  替代，不改变训练、reward 或评测协议；它以 CPU RAM 换取 GPU，rank 0 需要容纳完整约 4B 权重及临时 gather 数据。
- 验证：执行 `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_fsdp_reassemble`、
  `PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile tools/reassemble_fsdp_checkpoint.py` 与 `git diff --check`；
  本机没有 PyTorch distributed/FSDP checkpoint，未执行真实 8-rank 重组，服务器首次运行应检查 manifest 的
  `source_world_size=8`，并用一个小型 RefCOCO shard 验证导出权重可加载。

### 2026-09-02 - 允许外部 judge 的小 world-size Ray smoke

- 代码：修改 `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`；未新增、移动或删除模块。
- 文档：更新第 2.2、3.2 节与本变更日志；模块清单无变化。
- 行为：显式 Ray attach 且 `LOCAL_JUDGE_ENABLED=true` 时，训练 world size 从原先硬编码的
  `NUM_GPUS=7` 放宽为 `1..7`。这使开发机可用例如三张训练卡加一张独立 Llama judge 对完整
  main/direct/DLC-QA 路径做 smoke；入口现在精确验证 Ray 节点/GPU 数与 `NUM_GPUS * NNODES` 一致，拒绝
  旧 Ray head 注册更多 GPU 时的静默附着。
  `LOCAL_JUDGE_ENABLED=false` 的 8 卡 attach 约束、正式 7+1 拓扑、训练算法和 loss 均不变。小 world-size
  调用方必须自行保证各 parent-prompt batch 能整除实际训练卡数；三流正式 7 卡 controller 仍使用
  `112:224:56`。同时，DLC-QA judge 启动前的 `/v1/models` 健康检查在
  `CAPTION_QA_JUDGE_API_KEY` 非空且非 `EMPTY` 时会发送对应的 Bearer token；这与 vLLM 的
  `--api-key` 认证兼容，未启用认证的 judge 行为不变。
- 验证：执行 `bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 与 `git diff --check`；本机未连接
  Ray/CUDA/Llama，尚未执行三卡端到端 smoke。

### 2026-09-05 - 惩罚 pixel-empty no-target 的非空解码 mask

- 代码：修改 `verl/workers/opsd/mask_iou.py`、`projects/rl/reward_function/text2mask.py` 与
  `tests/test_opsd_core.py`；未新增、移动或删除模块。
- 文档：更新第 3.4、3.5 节和本变更日志；模块清单无变化。
- 行为：`NO_TARGET_REWARD_MODE=pixel_empty` 现在对 decoded mask union 非空的 no-target rollout 固定返回
  `-1.0`，不论 response 是否声称 `No target.`；仅精确拒识且 union 为空得 `+1.0`，空 union 但拒识格式无效
  仍得 `0.0`。主 `gres_no_target` caption-only non-cycle 路径与显式启用的
  `supervised_grounding_no_target` direct GRPO 均采用该三值 reward。`cap_overall`/`seg_overall` 使用真实
  reward；`no_target_accuracy`/`seg_supervised_grounding_no_target` 保留二元正确拒识率，并新增实际 reward 与
  非空 mask 负分指标，避免训练日志把 `-1.0` 误称为 accuracy。`text` 与 `official_bbox` no-target 模式、
  正样本 `POSITIVE_EMPTY_MASK_PENALTY` 均未改变。
- 验证：执行 `PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile verl/workers/opsd/mask_iou.py
  projects/rl/reward_function/text2mask.py tests/test_opsd_core.py tests/test_no_target_reward.py` 与
  `git diff --check`。尝试 `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_opsd_core
  tests.test_no_target_reward`，但本机 Python 未安装 `torch`，测试在导入前失败；未执行 CUDA/Ray decoder smoke。

### 2026-09-05 - 支持 disjoint gRefCOCO multi direct 数据导出

- 代码：修改 `projects/rl/datasets/prepare_grefcoco_cycle_dataset.py`，新增
  `tests/test_grefcoco_cycle_dataset.py`。
- 文档：更新第 2.4、5.1、5.3 节和本变更日志，记录可审计的 cross-parquet 排除语义与测试模块。
- 行为：gRefCOCO 转换器新增可重复的 `--exclude-parquet`。它在按 seed 分层抽样前，从每个已有
  parquet 提取 `(COCO image_id, normalized grounding_query)` 与 `(COCO image_id, union-mask RLE)`，排除
  相同身份的 gRefCOCO ref；因此可导出 10k multi-instance positive direct 数据，同时避免与主 cycle 数据和
  既有 RefCOCO direct 数据复用同图同人工表达或同一 target union。共享 COCO 图像但表达和 target 均不同的
  记录仍可使用，避免不必要缩小 multi 候选池；新候选也按这两种身份去重。输出 manifest 记录排除文件和两类
  身份数。未指定该参数时不进行跨 parquet 排除，但仍执行候选内部去重。
- 验证：使用 `ast.parse` 解析转换器和新增单测，且 `git diff --check` 通过；本机 Python 缺少
  `numpy`/`torch`，无法导入转换器执行该单测，未在服务器执行 VQ-SAM2 编码或 parquet cross-check。

### 2026-09-05 - 参数化 pixel-empty no-target 非空 mask 负分

- 代码：修改 `verl/workers/opsd/mask_iou.py`、`verl/workers/opsd/config.py`、
  `verl/workers/fsdp_workers.py`、`projects/rl/config.yaml`、
  `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 与 `tests/test_opsd_core.py`；未新增、移动或删除模块。
- 文档：更新第 3.4、3.5 节和本变更日志，区分正样本 empty/refusal penalty 与 no-target nonempty-mask penalty。
- 行为：新增 `pixel_iou.no_target_nonempty_mask_penalty` / `NO_TARGET_NONEMPTY_MASK_PENALTY`，默认 `1.0`
  保持 decoded nonempty no-target union 的 `-1.0` 行为。设置为 `0.0` 时，`pixel_empty` 仍解码 no-target
  response，显式 `No target.` 加空 union 仍为 `+1.0`，但 nonempty union 恢复为负分加入前的 `0.0`。该开关
  不影响 `POSITIVE_EMPTY_MASK_PENALTY`，因此可单独保留正样本错误拒识/空 mask 的 `-1.0`。
- 验证：执行受影响 Python 的 AST 语法解析、`bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 与
  `git diff --check`；本机缺少 `torch`，未运行依赖模型模块的完整单测或 CUDA/Ray decoder smoke。

### 2026-09-05 - 修复 gRefCOCO positive direct GRPO 的前置 source 校验

- 代码：修改 `verl/workers/supervised_anchors.py`、`verl/trainer/ray_trainer.py` 与
  `tests/test_supervised_anchors.py`；未新增、移动或删除模块。
- 文档：更新第 3.4 节 direct loader 的 source 契约及本变更日志；模块清单无变化。
- 行为：将 direct loader 的首 step source 校验与实际 direct grounding/SFT source 路由统一。启用
  `include_positive_sources` 时，校验现在同时允许 `refcoco_cycle` 和 `grefcoco_cycle`，因此新的
  30k RefCOCO + 10k gRefCOCO multi positive parquet 可进入 direct GRPO。可选 no-target 和 label
  source 也从同一纯函数生成允许集合，避免校验与路由再次漂移；loss、数据抽样和现有 RefCOCO/no-target
  行为不变。
- 验证：执行 `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_supervised_anchors`、
  `PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile verl/workers/supervised_anchors.py`
  `verl/trainer/ray_trainer.py tests/test_supervised_anchors.py` 与 `git diff --check`；本机未运行 CUDA/Ray
  端到端训练，服务器应以当前三卡配置确认进入 step 1。

### 2026-09-07 - 将 Evidence+Mask Credit 迁移为主代码可切换的 SECA 模块

- 代码：新增 `verl/workers/opsd/seca.py`、`tests/test_seca.py`；修改 `verl/workers/opsd/{config,__init__}.py`、`verl/trainer/ray_trainer.py`、`verl/workers/actor/dp_actor.py`、`projects/rl/config.yaml`、`projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 和 `projects/rl/qwen3vl_4b_mt.sh`。
- 行为：将已验证分支的 Evidence Gate 与 Mask Credit 统一命名为 SECA（Spatial-Evidence Credit Assignment），由 `worker.opsd.seca.enabled` / `SECA_ENABLED` 单一开关控制，默认关闭以保持 baseline。开启后，mid-route privileged JSD 的 sample weight 按 IoU 与 reconstruction-only false-positive 面积门控；direct mask CE 的首个/第二个 SAMTok depth-2 code 分别使用 coarse/fine token credit；pixel-IoU、`R_Ci`、reward、原始 GRPO 和无 token-weight 监督批次均不变。
- 文档：更新第 3.6 节、模块清单与本日志，明确 SECA 的调用路径、适用的两条辅助梯度边界以及没有对应 batch 时的空路径语义。
- 验证：执行 `bash -n projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh projects/rl/qwen3vl_4b_mt.sh`、受影响 Python 文件的 `py_compile`、`PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_seca`（依赖当前训练环境的 PyTorch）、YAML/脚本静态检查和 `git diff --check`；未启动 GPU/Ray 训练，未修改占卡进程。

### 2026-09-07 - 隔离探索副本新增 no-target Refusal Credit 对照

- 代码：仅修改 `experiments/pegc_ablation_20260902/` 内的 supervised-anchor 配置、trainer/FSDP worker、branch launcher，并新增两套 3+1 卡 1/10 训练入口。
- 行为：实验组独立 teacher-force no-target `No target.` token，baseline 保持 direct GRPO + 正样本 empty-mask penalty；两者均关闭 direct mask CE 与 Mask Credit，并记录每个 no-target prompt 的 6-rollout all-mask/all-empty/同质率。根目录主训练入口不变。
- 验证：分支 Python `py_compile`、四个 shell `bash -n` 通过；训练与四 bench 评测待执行。

### 2026-09-07 - 探索副本追加 Mask Credit + Refusal Credit 联合训练

- 代码：仅新增 `experiments/pegc_ablation_20260902/tools/train_mask_refusal_credit_1of10_3gpu.sh`，根目录主训练入口不变。
- 行为：该入口与 Refusal Credit 对照共享 1/10 数据、三卡训练 + 一卡本地 Llama 和 direct GRPO 设置，仅额外打开 hierarchical Mask Credit，关闭 direct mask CE；用于和 baseline、Refusal Credit 两版统一评测 N acc 是否恢复。
- 验证：新增脚本通过 `bash -n`；训练与四 bench 评测待三版依次完成后执行。

### 2026-09-08 - 隔离探索副本增加干净 direct GRPO 对照入口

- 代码：新增 `experiments/pegc_ablation_20260902/tools/train_direct_grpo_only_baseline_1of10_3gpu.sh` 与 `experiments/pegc_ablation_20260902/tools/train_direct_grpo_only_refusal_credit_1of10_3gpu.sh`；根目录主训练入口不变。
- 行为：新增对照明确关闭 DLC-QA、Mask Credit、direct mask CE 及其他 PEGC 模块，仅保留正样本/no-target direct GRPO；实验组额外启用 Refusal Credit，direct grounding 从第 0 步生效，用于单独测量拒识梯度对 N acc 的影响。
- 验证：两个脚本通过 `bash -n`；训练与四 bench 评测在当前服务器依次执行，DLC-Bench 评测使用本地 Llama 3.1 8B judge。

### 2026-09-08 - 修复 70k 八卡命令的 Ray 环境不一致

- 代码：修改 `opsd_70k_positive_empty_penalty_only_8gpu.txt`。
- 文档：更新 Ray 环境约束说明和本变更日志。
- 行为：70k 入口现在把项目环境的 `$ENV_DIR/bin` 放在 PATH 前，并使用同一环境的 `ray start/stop`；启动前清理本机旧 Ray head，避免全局 Ray 2.35/Python 3.10.12 与训练环境 Ray 2.57/Python 3.10.20 混用导致版本校验失败。
- 验证：复现并确认原日志中的 Ray version mismatch 根因；执行 `bash -n opsd_70k_positive_empty_penalty_only_8gpu.txt`，检查命令包含环境内 `ray` 和 stale-head 清理。

### 2026-09-08 - 增加 Refusal Credit 对照的四 bench 评测编排

- 代码：新增 `experiments/pegc_ablation_20260902/tools/eval_direct_grpo_refusal_suite.sh`，根目录主训练入口不变。
- 行为：训练完成后从探索分支目录依次执行两版 HF 导出、RefCOCO、GroundingSuite、GRES 四项指标及 DLC-Bench 本地 Llama-3.1 8B judge；严格检查 benchmark 产物并在末尾恢复 GPU0--3 占卡。
- 验证：脚本通过 `bash -n`，训练链路继续运行中。


### 2026-09-09 - 修复 70k 离线训练缺少 wandb 依赖的启动失败

- 代码：修改 `opsd_70k_positive_empty_penalty_only_8gpu.txt`；未新增、移动或删除模块。
- 文档：更新第 2.2 节日志配置说明和本变更日志。
- 行为：70k 正样本 pixel-empty 训练改用 `TRAINER_LOGGERS='["file"]'`。当前环境未安装可选
  `wandb` 包，原 `['file','wandb']` 配置会在 `verl/utils/logger/logger.py` 的条件导入后调用
  不存在的 `wandb` 名称并触发 `NameError`；file-only 模式保留本地 experiment/generation JSONL，
  不改变训练、数据、reward 或 Ray 拓扑。
- 验证：根据 `logs/cyclegrpo70k_opsd_seca_positive_only_8gpu/train_20260909_031958.log` 的
  `NameError: name 'wandb' is not defined` 堆栈确认根因；执行 `bash -n opsd_70k_positive_empty_penalty_only_8gpu.txt`
  和 `git diff --check`，未重新启动训练。


### 2026-09-09 - 完善跨服务器 70k 全流程 README

- 代码：未修改训练、评测或数据处理实现；修改 `README.md`，无新增、移动、删除模块。
- 文档：将 RefCOCO 明确指向 `Untitled111/refcoco-train2014-assets`，其余已上传数据指向 `Untitled111/train-opsd`；固定 20k cycle、30k direct、10k no-target、10k DLC-QA 的实际 parquet 路径；补充绝对路径重写、DLC-QA join 检查、7+1 GPU 训练、四 bench 评测、本地 Llama-3.1-8B DLC judge 与权重上传流程。
- 行为：跨服务器使用者只需下载模型/数据、执行路径重写和 preflight，即可从任意工作目录调用单一训练脚本；脚本内部自动启动 Ray 和训练期 Llama judge。旧版公共下载段落保留为可选的原始数据重建说明，不再与已上传数据冲突。
- 验证：执行 `bash -n` 检查 70k 训练、RL launcher 和 eval launcher；抽取 README 全部 bash 代码块执行语法检查；执行 `git diff --check`；静态确认 README 不再包含旧 parquet 路径或“not mirrored”矛盾说明。未启动新的 70k 训练，避免占用当前正在进行的评测资源。

### 2026-09-09 - 修正 70k 训练入口的 direct parquet 路径

- 代码：修改 `opsd_70k_positive_empty_penalty_only_8gpu.txt`；未新增、移动或删除模块。
- 行为：`DIRECT_TRAIN_DATA` 和 `DIRECT_NO_TARGET_TRAIN_DATA` 现在指向 `train-opsd` 中实际存在的 30k direct 与 10k no-target parquet；避免跨服务器按 README 下载后因旧目录名导致启动前数据文件不存在。
- 验证：使用 `rg` 核对脚本路径与 README 第 3、4 节 canonical exports 一致；执行 `bash -n opsd_70k_positive_empty_penalty_only_8gpu.txt` 与 `git diff --check`，未启动新的 GPU 训练。

### 2026-09-09 - 补充 70k direct 数据路径契约

- 代码：未新增模块；同步更新 `code.md` 第 29 条数据/资源契约说明。
- 文档：明确 70k 入口实际使用的 30k direct 与 10k no-target parquet 文件名，并要求路径重写后执行存在性检查和 source 契约检查。
- 验证：核对 `opsd_70k_positive_empty_penalty_only_8gpu.txt`、`README.md` 与本机四个 parquet 的行数；执行 `git diff --check`。

### 2026-09-09 - 明确 40k 分割监督组成与 70k 运行时依赖

- 代码：未修改训练算法；同步修正 `code.md` 当前数据路径说明。
- 文档：明确 40k 分割监督由 30k `refcoco_cycle` 正例和 10k `gres_no_target` 组成；README 增加依赖 profile、系统工具要求与运行时 import smoke。
- 配置：确认 70k 使用 20k cycle + 40k direct 分割监督 + 10k DLC-QA；SECA、OPSD/pixel-IoU、routing、EMA teacher、teacher analysis/confidence、caption safety、direct GRPO、direct mask CE 均开启；main/rollout batch=112，direct=224，QA=56，GPU 0--6 为 Ray 训练、GPU 7 为本地 Llama judge，no-target reward=`pixel_empty`。
- 验证：实际检查两条监督 parquet 的 source/行数（30,000 + 10,000）、当前环境核心 import（torch 2.8.0+cu128、Ray 2.57.0、vLLM 0.11.0 等）和 `verl.trainer.main` import；执行训练/评测脚本 `bash -n`、README bash block 语法检查及 `git diff --check`。

### 2026-09-09 - 修正 70k README 预检开关与依赖说明

- 代码：未修改训练算法；修改 `README.md` 与 `code.md` 数据/环境说明。
- 文档：预检现在使用入口真实变量名 `DIRECT_GROUNDING_ENABLED`、`DIRECT_MASK_CE_ENABLED`、`SUPERVISED_CAPTION_QA_ENABLED`，并检查 OPSD、SECA、routing、EMA teacher、teacher analysis/confidence、caption safety 全部开启；明确合并 40k 文件仅作说明，正式入口使用 30k+10k split。
- 验证：README 全部 bash block、70k 训练入口和 eval 入口通过 `bash -n`；核心依赖 import smoke 通过；两条监督 parquet source 严格为 30,000 `refcoco_cycle` 与 10,000 `gres_no_target`；`git diff --check` 通过。
### 2026-09-13 — Pixel-OPSD manuscript iteration
- Changed files: `paper/iclr2027/pixelopsd/main.tex`, regenerated `paper/iclr2027/pixelopsd/main.pdf`.
- Behavioral impact: expanded Related Work, Method, OPD, EGCA, supervised mixing, training schedule, evaluation protocol, ablation rationale, cross-benchmark analysis, reproducibility, and conclusion. The body is nine pages with Conclusion on page 9; references occupy two pages and the appendix follows.
- Verification: three-pass pdflatex/BibTeX; 12 pages total. No fatal, undefined-reference, or overfull diagnostics in `main.log`; underfull vbox warnings remain from float balancing.

### 2026-09-13 — figure regeneration
- Changed files: `paper/iclr2027/pixelopsd/figures/overview.pdf`, `framework.pdf`, `results_bars.pdf`, `qualitative.pdf`, and regenerated `main.pdf`.
- Behavioral impact: regenerated all four manuscript figures from the maintained plotting script after layout updates; no training or evaluation behavior changed.
- Verification: figure generation completed successfully; three-pass LaTeX/BibTeX compile produced a 12-page artifact with no fatal, undefined-reference, or overfull diagnostics.

### 2026-09-13 — pagination command fix
- Changed files: `paper/iclr2027/pixelopsd/main.tex`, regenerated `paper/iclr2027/pixelopsd/main.pdf`.
- Behavioral impact: corrected the doubled-backslash conclusion break that printed literal LaTeX commands; the manuscript now has a clean nine-page body, with references starting after the conclusion and spanning two pages, followed by the appendix.
- Verification: three-pass pdflatex/BibTeX; 12 pages total; `pdftotext` confirms no literal `clearpage` or `sectionConclusion` text and `main.log` has no fatal, undefined-reference, or overfull diagnostics.

### 2026-09-13 — qualitative figure spacing refinement
- Changed files: `paper/iclr2027/pixelopsd/make_revised_figures.py`, regenerated the four figure PDFs and `main.pdf`.
- Behavioral impact: increased qualitative panel height and separated its legend from the image-row metadata to reduce top-row collisions; no model, training, or evaluation behavior changed.
- Verification: plotting script completed successfully; three-pass LaTeX/BibTeX compile produced 12 pages with no fatal, undefined-reference, or overfull diagnostics.

### 2026-09-13 — final body/reference pagination pass
- Changed files: `paper/iclr2027/pixelopsd/main.tex`, regenerated `main.pdf`.
- Behavioral impact: consolidated the analysis narrative before the conclusion, preserved a nine-page body with clean conclusion boundary, and increased bibliography leading so references occupy two physical pages.
- Verification: three-pass pdflatex/BibTeX; 12 pages total, no literal LaTeX command text, and no fatal, undefined-reference, or overfull diagnostics.

### 2026-09-13 — analysis-page density pass
- Changed files: `paper/iclr2027/pixelopsd/main.tex`, regenerated `main.pdf`.
- Behavioral impact: added design-implication, audit-trail, and reader-checklist prose before the conclusion to improve the previously sparse analysis page while preserving the nine-page body and two-page bibliography.
- Verification: three-pass pdflatex/BibTeX; 12 pages total; pages 7–9 now contain connected analysis/conclusion text, with no fatal, undefined-reference, or overfull diagnostics.

### 2026-09-13 — result-chart label margin
- Changed files: `paper/iclr2027/pixelopsd/make_revised_figures.py`, regenerated `results_bars.pdf` and `main.pdf`.
- Behavioral impact: widened the left plotting margin so the full `GroundingSuite` label remains visible at normal paper scale.
- Verification: plotting script and three-pass LaTeX/BibTeX compile succeeded; 12 pages and no fatal, undefined-reference, or overfull diagnostics.

### 2026-09-13 — algorithmic method clarification
- Changed files: `paper/iclr2027/pixelopsd/main.tex`, regenerated `main.pdf`.
- Behavioral impact: added explicit five-step training procedure and gradient/information-boundary description for OPD, EGCA, policy, and supervised losses; no training code or evaluation behavior changed.
- Verification: three-pass pdflatex/BibTeX; 12 pages with nine-page body and two-page references, no fatal, undefined-reference, or overfull diagnostics.

### 2026-09-13 — appendix and source-syntax cleanup
- Changed files: `paper/iclr2027/pixelopsd/main.tex`, regenerated `main.pdf`.
- Behavioral impact: expanded the appendix with rollout-record, evaluation-command, hyperparameter-disclosure, and inference-path details; corrected command escaping introduced during the expansion.
- Verification: three-pass pdflatex/BibTeX; 12 pages, nine-page body, two-page references, one-page appendix, and no fatal, undefined-reference, or overfull diagnostics.

### 2026-09-13 — introduction motivation and contribution roadmap
- Changed files: `paper/iclr2027/pixelopsd/main.tex`, regenerated `main.pdf`.
- Behavioral impact: added a literature-facing motivation paragraph, explicit OPD/EGCA/supervised contribution roadmap, and research questions; retained the nine-page body, two-page references, and one-page appendix.
- Verification: three-pass pdflatex/BibTeX; 12 pages with clean conclusion/reference boundary and no fatal, undefined-reference, or overfull diagnostics.

### 2026-09-13 — literature citation integration
- Changed files: `paper/iclr2027/pixelopsd/main.tex`, regenerated `main.pdf`.
- Behavioral impact: connected the motivation and Related Work claims to policy optimization, pixel MLLM, distillation, preference-learning, and multimodal pretraining references; no experimental values or training behavior changed.
- Verification: three-pass pdflatex/BibTeX; 12 pages, nine-page body and two-page references, no fatal, undefined-reference, or overfull diagnostics.

### 2026-09-13 — full-width table layout
- Changed files: `paper/iclr2027/pixelopsd/main.tex`, regenerated `main.pdf`.
- Behavioral impact: converted the main comparison and ablation exhibits to full-width elastic tables so columns use the available page width; metric ownership and provisional cells are unchanged.
- Verification: three-pass pdflatex/BibTeX; 12 pages with no fatal, undefined-reference, or overfull diagnostics.

### 2026-09-13 — paragraph command cleanup
- Changed files: `paper/iclr2027/pixelopsd/main.tex`, regenerated `main.pdf`.
- Behavioral impact: corrected escaped paragraph and math commands in the algorithmic method text so labels render as typography rather than literal strings.
- Verification: three-pass pdflatex/BibTeX; 12 pages, nine-page body, two-page references, one-page appendix, and no fatal, undefined-reference, overfull, or missing-character diagnostics.

### 2026-09-13 — ablation float environment correction
- Changed files: `paper/iclr2027/pixelopsd/main.tex`, regenerated `main.pdf`.
- Behavioral impact: corrected the ablation exhibit's closing environment to `table*`, ensuring the full-width table is placed and captioned as intended.
- Verification: three-pass pdflatex/BibTeX; 12 pages with nine-page body and two-page references, no fatal, undefined-reference, overfull, or missing diagnostics.

### 2026-09-13 — restrained table emphasis
- Changed files: `paper/iclr2027/pixelopsd/main.tex`, regenerated `main.pdf`.
- Behavioral impact: added light header and proposed-method row shading using `xcolor`/`colortbl` to improve table scanning without changing metric values or table ownership.
- Verification: three-pass pdflatex/BibTeX; 12 pages, nine-page body, two-page references, one-page appendix, and no fatal, undefined-reference, overfull, or missing diagnostics.

### 2026-09-13 — appendix reproducibility detail
- Changed files: `paper/iclr2027/pixelopsd/main.tex`, regenerated `main.pdf`.
- Behavioral impact: expanded the appendix with data-conversion, determinism, and logging-schema details so the supplementary page is a substantive reproducibility artifact rather than a short placeholder.
- Verification: three-pass pdflatex/BibTeX; 12 pages with nine-page body, two-page references, and one-page appendix; no fatal, undefined-reference, overfull, or missing diagnostics.

### 2026-09-13 — cross-reference labels
- Changed files: `paper/iclr2027/pixelopsd/main.tex`, regenerated `main.pdf`.
- Behavioral impact: added formal labels to all figures and tables and replaced informal figure/table mentions with resolvable cross-references.
- Verification: three-pass pdflatex/BibTeX; 12 pages, no fatal or undefined-reference diagnostics, and `git diff --check` passes.

### 2026-09-13 — equation cross-references
- Changed files: `paper/iclr2027/pixelopsd/main.tex`, regenerated `main.pdf`.
- Behavioral impact: labeled the cycle, OPD, and EGCA equations and connected their surrounding prose with explicit equation references.
- Verification: three-pass pdflatex/BibTeX; 12 pages with nine-page body and two-page references, no fatal, undefined-reference, overfull, or missing diagnostics.

### 2026-09-13 — ablation caption clarification
- Changed files: `paper/iclr2027/pixelopsd/main.tex`, regenerated `main.pdf`.
- Behavioral impact: clarified the ablation caption's fixed controls, progressive row semantics, diagnostic ownership, and meaning of provisional dashes.
- Verification: three-pass pdflatex/BibTeX; 12 pages, nine-page body, two-page references, one-page appendix, no fatal, undefined-reference, overfull, or missing diagnostics.

### 2026-09-13 — clean-build color package fix
- Changed files: `paper/iclr2027/pixelopsd/main.tex`, regenerated `main.pdf`.
- Behavioral impact: removed the duplicate `xcolor` option load already provided by the ICLR style, retaining `colortbl` for row shading. Clean builds now run BibTeX and resolve citations correctly.
- Verification: three-pass pdflatex/BibTeX from the paper directory; `main.bbl` generated, 12 pages produced, and no fatal, undefined-reference, overfull, missing, or option-clash diagnostics.

### 2026-09-13 — caveat deduplication
- Changed files: `paper/iclr2027/pixelopsd/main.tex`, regenerated `main.pdf`.
- Behavioral impact: consolidated repeated provisional/audit caveats in the body so the method-to-evidence narrative is less defensive; the explicit placeholder policy remains stated once in the evaluation section and appendix.
- Verification: three-pass pdflatex/BibTeX; 12 pages with nine-page body and two-page references, no fatal, undefined-reference, overfull, or missing diagnostics.

### 2026-09-13 — related-work citation anchoring
- Changed files: `paper/iclr2027/pixelopsd/main.tex`, regenerated `main.pdf`.
- Behavioral impact: attached direct citations to the SAMTok/pixel-MLLM, CycleGRPO, and supervised multimodal claims in Related Work, strengthening technical comparisons without adding unsupported claims.
- Verification: three-pass pdflatex/BibTeX; 12 pages, nine-page body, two-page references, no fatal, undefined-reference, overfull, or missing diagnostics.

### 2026-09-13 — abstract evaluation scope
- Changed files: `paper/iclr2027/pixelopsd/main.tex`, regenerated `main.pdf`.
- Behavioral impact: added a concise evaluation-scope sentence to the abstract and adjusted bibliography leading to retain exactly two reference pages after the added front-matter text.
- Verification: three-pass pdflatex/BibTeX; 12 pages, nine-page body, two-page references, one-page appendix, no fatal, undefined-reference, overfull, or missing diagnostics.

### 2026-09-13 — self-contained figure captions
- Changed files: `paper/iclr2027/pixelopsd/main.tex`, regenerated `main.pdf`.
- Behavioral impact: rewrote all four figure captions to state the visual encoding, evaluation condition, and intended takeaway; no figures or metric values changed.
- Verification: three-pass pdflatex/BibTeX; 12 pages with nine-page body and two-page references, no fatal, undefined-reference, overfull, or missing diagnostics.

### 2026-09-13 — result interpretation narrative
- Changed files: `paper/iclr2027/pixelopsd/main.tex`, regenerated `main.pdf`.
- Behavioral impact: added a main-findings subsection that interprets the headline table and signed transfer figure along separate spatial, semantic, and stability axes, while explicitly preserving provisional-value caveats.
- Verification: three-pass pdflatex/BibTeX; 12 pages with nine-page body and two-page references, no fatal, undefined-reference, overfull, or missing diagnostics.

### 2026-09-13 — benchmark-chart title margin
- Changed files: `paper/iclr2027/pixelopsd/make_revised_figures.py`, regenerated `results_bars.pdf` and `main.pdf`.
- Behavioral impact: moved benchmark titles and case-count labels inward so the full `GroundingSuite` text is visible and does not collide with the plot boundary.
- Verification: plotting script and three-pass LaTeX/BibTeX compile succeeded; 12 pages with no fatal, undefined-reference, overfull, or missing diagnostics.

### 2026-09-13 - 探索分支拆分 cycle/direct 正样本 no-target 惩罚

- 代码：更新 `experiments/pegc_ablation_20260902/verl/workers/opsd/config.py`、`verl/workers/fsdp_workers.py`、`projects/rl/config.yaml`、`projects/rl/qwen3vl_4b_pegc_ablation.sh`；新增 `tools/train_evidence_mask_credit_cycle_penalty_only_1of10_epoch1.sh`。
- 行为：探索分支支持独立的 cycle 与 direct 正样本空 mask 惩罚；本实验仅对 cycle localization 开启 `1.0`，direct GRPO 设为 `0.0`，Evidence、Mask Credit、DLC-QA 和 1/10 数据配方不变。根目录主训练入口不受本探索实验影响。
- 验证：探索分支配置 fallback/显式值断言、Python `py_compile`、两个 shell 入口 `bash -n` 通过；训练与四项评测待启动。

### 2026-09-13 - 固化拆分惩罚实验的本机四卡入口

- 代码：更新 `experiments/pegc_ablation_20260902/tools/train_evidence_mask_credit_cycle_penalty_only_1of10_epoch1.sh`。
- 行为：默认拓扑改为当前服务器 GPU0--2 训练、GPU3 启动本地 Llama judge，批大小调整为 `108/216/54`；8 卡服务器仍可通过 `TRAIN_GPU_LIST`、`TRAIN_NUM_GPUS`、`LLAMA_GPU` 覆盖，cycle/direct 惩罚值保持 `1.0/0.0`。
- 验证：脚本通过 `bash -n`；训练尚未启动。

### 2026-09-13 - 修正拆分惩罚入口启动提示

- 代码：更新 `experiments/pegc_ablation_20260902/tools/train_evidence_mask_credit_cycle_penalty_only_1of10_epoch1.sh`。
- 行为：恢复 Llama judge 启动提示中的实际 GPU 编号显示；不改变 CUDA 绑定、惩罚拆分或训练参数。
- 验证：脚本通过 `bash -n`。

### 2026-09-13 - 恢复 judge GPU 提示变量

- 代码：更新 `experiments/pegc_ablation_20260902/tools/train_evidence_mask_credit_cycle_penalty_only_1of10_epoch1.sh`。
- 行为：启动日志显示实际 `LLAMA_GPU` 编号，便于跨服务器确认 GPU 绑定。
- 验证：`bash -n` 通过。

### 2026-09-13 - 修复 branch launcher 的 conda 激活阻断

- 代码：更新 `experiments/pegc_ablation_20260902/projects/rl/qwen3vl_4b_pegc_ablation.sh`。
- 行为：训练入口跳过易受容器 CONDA_PREFIX/PATH 影响的 `conda activate`，直接使用 `/bin/python3` 与 `/bin/ray`，避免在 judge 健康后阻断 Ray 训练。
- 验证：launcher 通过 `bash -n`；待重新启动训练验证。

### 2026-09-13 - 缩短拆分惩罚实验 Ray 临时目录

- 代码：更新 experiments/pegc_ablation_20260902/tools/train_evidence_mask_credit_cycle_penalty_only_1of10_epoch1.sh。
- 行为：将默认 RAY_SHORT_ROOT 改为 /dev/shm/pegc-cycle，满足 launcher 的绝对路径和 32 字符限制，避免 judge 加载后训练入口退出。
- 验证：脚本通过 bash -n。

### 2026-09-13 - 加密消融展示并压缩论文重复分析

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`、`paper/iclr2027/pixelopsd/make_revised_figures.py`；新增 `paper/iclr2027/pixelopsd/figures/ablation_diagnostics.pdf`。
- 行为：消融表扩展为前缀一致性、粗/细 IoU、边界 F1、校准、QA 与拒答率等独立诊断；新增仅描述启用信号的消融图，不填充未导出的 Pixel-OPSD 结果。将重复的后段总结合并为方法分析、审计轨迹与评测交接三部分；参考文献仍保持两页，正文结构和 EGCA 唯一模块命名不变。
- 验证：运行 `python make_revised_figures.py && ./build_pdf.sh` 成功，生成 12 页 PDF（正文 9 页、参考文献 2 页、附录 1 页）；无致命 LaTeX 错误，交叉引用和图表资源均可解析。

### 2026-09-13 - 按 ICLR 浮动体约束调整首屏与正文比例

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`。
- 行为：将 overview 图从 Introduction 首段移至 Related Work 之后，避免 `figure* [t]` 回浮到标题页顶部；移除正文中的独立 Reproducibility 小节，将其审计内容保留在 Analysis 与附录；参考文献改为紧凑双页排版。
- 验证：`./build_pdf.sh` 成功生成 12 页 PDF，页序核验为正文 9 页、参考文献 2 页、附录 1 页；overview 从第 2 页开始，未发现致命 LaTeX 错误。

### 2026-09-13 - 修正首屏浮动体并收束正文后半段

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`。
- 行为：overview 图改在 Related Work 后排版，防止 ICLR 双栏浮动规则将其回置到第一页；正文删除独立重复的 Reproducibility 小节，改为机制诊断、范围与审计、评测交接三个连续段落；保留完整九页正文与两页参考文献。
- 验证：运行 `./build_pdf.sh` 成功，生成 12 页（正文 9 页、参考文献 2 页、附录 1 页）；overview 位于第 2 页，日志无致命错误。

### 2026-09-13 - 加深方法推导并平衡双页参考文献

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`。
- 行为：在 Method 中补充 token/code credit routing、teacher 因果顺序、优化成本及组件边界的细节；将后半段重复的跨基准/可复现总结压缩为单一 Interpretability and Reproducibility 段落与机制诊断段，避免页面填充式重复；调整 bibliography 间距使参考文献稳定占两页。
- 验证：运行 `./build_pdf.sh` 成功，生成 12 页 PDF；页序为正文 9 页、参考文献 2 页、附录 1 页，`main.log` 无 fatal、undefined-reference、overfull 或 missing-resource 诊断。

### 2026-09-13 - 扩展消融机制解释以平衡第七页

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`。
- 行为：在消融表和诊断图后补充各变体的因果预期、指标归属与导出统计协议，使消融章节成为独立实验分析而非空表占位；不改变任何占位数值或主表指标。
- 验证：`./build_pdf.sh` 成功，PDF 保持 12 页（正文 9、参考文献 2、附录 1），并完成第 7 页渲染检查，未见图文重叠或裁切。

### 2026-09-13 - 固定结论后参考文献双页边界

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`。
- 行为：在 bibliography 前加入显式分页，恢复较易读的 `small` 字号、1.05 行距和 12pt 条目间距，使参考文献完整占用第 10--11 页；附录从第 12 页开始，结论完整留在第 9 页。
- 验证：运行 `./build_pdf.sh` 成功生成 12 页；页序核验为正文 9 页、参考文献 2 页、附录 1 页，`main.log` 未出现 fatal、undefined-reference、overfull 或 missing-resource 诊断。

### 2026-09-13 - 压缩失败类型分析以保持九页正文

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`。
- 行为：在 Interpretability and Reproducibility 中加入紧凑的 selection/extent/relational failure taxonomy，解释两个 benchmark 的互补压力；控制段落长度，避免正文扩展到第十页。
- 验证：`./build_pdf.sh` 成功生成 12 页，正文仍为 9 页，参考文献 2 页、附录 1 页；第 9 页完成渲染检查。

### 2026-09-13 - 补充训练顺序与因果解释

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`。
- 行为：在结论前补充训练阶段顺序与可用证据的因果说明，解释 cycle warm-up、OPD、EGCA 和持续有监督锚点的关系；不新增模块名称或实验指标。
- 验证：运行 `./build_pdf.sh` 成功，仍为正文 9 页、参考文献 2 页、附录 1 页；构建日志无致命错误。

### 2026-09-13 - 平衡双页参考文献的垂直密度

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`。
- 行为：将参考文献条目间距从 12pt 调整为 18pt，在保留两页边界的前提下减少第 11 页底部空白；正文与附录内容不变。
- 验证：`./build_pdf.sh` 成功生成 12 页，页序仍为正文 9 页、参考文献 2 页、附录 1 页。

### 2026-09-13 - 修复结论段落句法断裂

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`。
- 行为：修正 Conclusion 中 headline metrics 句子的标点与大小写断裂，保持原有论断和占位结果不变。
- 验证：运行 `./build_pdf.sh` 成功生成 12 页；`main.log` 无 fatal、undefined-reference、overfull 或 missing-resource 诊断。

### 2026-09-13 - 放大定性证据并收紧图内留白

- 代码：更新 `paper/iclr2027/pixelopsd/make_revised_figures.py`，重新生成 `paper/iclr2027/pixelopsd/figures/qualitative.pdf`。
- 行为：提高定性图中四个 panel 的有效占比并上移图例，保留统一坐标系、目标/基线/本方法轮廓和四类错误覆盖；不改变画布尺寸与正文分页。
- 验证：运行 `python make_revised_figures.py && ./build_pdf.sh` 成功生成 12 页；第 8 页渲染检查无字体重叠、裁切或图例遮挡。

### 2026-09-13 - 在 overview 中显式标注推理边界

- 代码：更新 `paper/iclr2027/pixelopsd/make_revised_figures.py`，重新生成 `paper/iclr2027/pixelopsd/figures/overview.pdf`。
- 行为：在 outcome 卡片加入 “inference: image + query only” 标注，明确 EGCA 只参与训练期 credit assignment，不增加推理输入或路径；不改变方法定义和图尺寸。
- 验证：运行 `python make_revised_figures.py && ./build_pdf.sh` 成功生成 12 页；第 2 页渲染检查新增标注清晰且无重叠。

### 2026-09-13 - 提升 framework 跨栏图有效占比

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，将 `framework.pdf` 的跨栏宽度从 0.90 调整为 0.95 文本宽度。
- 行为：放大训练数据流、解码证据和目标函数节点，提升 token/pixel/更新路径的可读性；不改变图内容、推理路径或分页结构。
- 验证：运行 `python make_revised_figures.py && ./build_pdf.sh` 成功生成 12 页；第 5 页渲染检查无溢出、裁切或节点重叠。

### 2026-09-13 - 轻量放大 overview 跨栏图

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，将 `overview.pdf` 的跨栏宽度从 0.90 调整为 0.94 文本宽度。
- 行为：提高问题、OPD/EGCA 干预和 inference boundary 标注的可读性，不改变图内容或首屏位置。
- 验证：`./build_pdf.sh` 成功生成 12 页；第 2 页渲染检查通过，图内元素无重叠或裁切。

### 2026-09-13 - 明确 framework 的训练/推理信息边界

- 代码：更新 `paper/iclr2027/pixelopsd/make_revised_figures.py`，重新生成 `paper/iclr2027/pixelopsd/figures/framework.pdf`。
- 行为：在 rollout 区域标注 inference path，在 decoded-evidence 区域标注 target-conditioned training-only，强化目标掩码不进入推理路径的视觉证据。
- 验证：运行 `python make_revised_figures.py && ./build_pdf.sh` 成功生成 12 页；第 5 页渲染检查新增标注无重叠、裁切或越界。

### 2026-09-13 - 提升主表与消融表行距可读性

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`。
- 行为：为主比较表和独立消融表设置 1.35 的行距，在保持全宽和仅 eval 指标的前提下提高扫描性；更大的行距试验会破坏九页正文约束，未保留。
- 验证：`./build_pdf.sh` 成功生成 12 页，正文 9 页、参考文献 2 页、附录 1 页；第 5、7 页渲染检查无表格溢出。

### 2026-09-13 - 收紧实验叙事中的假设语气

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`。
- 行为：将机制解释从“expected to”改为明确的可检验问题，将未来时态的统计承诺改为评测记录描述，避免把未填结果写成已证实结论。
- 验证：运行 `./build_pdf.sh` 成功生成 12 页，正文/参考文献/附录页序不变。

### 2026-09-13 - 压缩 caption 以保持九页正文

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`。
- 行为：将消融图注压缩为独立且明确的诊断说明，避免冗余句子把 Conclusion 推到第十页；主表 caption 保持简洁，所有占位策略仍在表注/附录中说明。
- 验证：`./build_pdf.sh` 成功生成 12 页，正文 9 页、参考文献 2 页、附录 1 页。

### 2026-09-13 - 合并 Method 后段层级以改善主线阅读

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`。
- 行为：将 credit routing、teacher ordering、optimization cost 和 component boundaries 从连续编号小节改为段落级 signposts，保留全部技术内容但减少层级碎片，更接近 ICLR 论文的主线叙事。
- 验证：运行 `./build_pdf.sh` 成功生成 12 页，正文/参考文献/附录页序不变。

### 2026-09-13 - 回退 framework 重叠的 token strip 尝试

- 代码：更新 `paper/iclr2027/pixelopsd/make_revised_figures.py`。
- 行为：移除渲染检查中与 target-conditioned 标注和 EGCA 节点发生重叠的额外 token strip，保留无重叠的训练/推理边界标注版本。
- 验证：重新运行 `python make_revised_figures.py && ./build_pdf.sh` 成功生成 12 页；第 5 页复核无文字重叠或裁切。

### 2026-09-13 - 精修草稿式措辞与消融图注

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`。
- 行为：将 abstract、消融分析和附录中的 “current draft / expected effect / will report” 等草稿式表述改为正式、可验证的论文措辞；明确占位单元格仅等待最终 checkpoint 导出。
- 验证：`./build_pdf.sh` 成功生成 12 页，正文 9 页、参考文献 2 页、附录 1 页，构建无致命错误。

### 2026-09-13 - 增加 overview 纵向层次

- 代码：更新 `paper/iclr2027/pixelopsd/make_revised_figures.py`，将 overview 画布高度从 3.12 调整为 3.40 英寸。
- 行为：为问题、干预和结果卡片提供更充分的内部空间，提高示例图和关键标签的可读性；不改变图语义或正文页数。
- 验证：运行 `python make_revised_figures.py && ./build_pdf.sh` 成功生成 12 页；第 2 页渲染检查无图文重叠或裁切。

### 2026-09-13 - 收束结论并移除草稿式 checkpoint 收尾

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`。
- 行为：将 Conclusion 末句改为方法边界与部署接口的正式总结，移除“等待最终 checkpoint”措辞；占位说明继续保留在表注和附录中。
- 验证：运行 `./build_pdf.sh` 成功生成 12 页，正文 9 页、参考文献 2 页、附录 1 页。

### 2026-09-13 - 加密消融诊断展项并平衡第七页版面

- 代码：更新 `paper/iclr2027/pixelopsd/make_revised_figures.py`，重新生成 `paper/iclr2027/pixelopsd/figures/ablation_diagnostics.pdf`。
- 行为：扩大消融诊断画布，加入因果阶段说明和独立诊断 rail（semantic/prefix、coarse-to-fine overlap、boundary calibration、QA/refusal），仅描述启用的学习信号与评测维度，不引入新模块或未经核验的分数。
- 验证：运行 `python make_revised_figures.py && ./build_pdf.sh` 成功生成 12 页；日志仅保留参考文献页的既有 underfull vbox 提示，正文图表无 fatal、overfull、裁切或明显重叠。

### 2026-09-13 - 修正消融诊断 rail 的标签拥挤

- 代码：更新 `paper/iclr2027/pixelopsd/make_revised_figures.py`。
- 行为：缩短 diagnostic rail 的显示标签并重新分配横向位置，避免长标签在窄色块中发生截断或粘连；诊断含义不变。
- 验证：重新生成图并运行 `./build_pdf.sh` 成功生成 12 页；第 7 页渲染复核标签无遮挡、无重叠。

### 2026-09-13 - 收紧结果图与定性图的边界标注

- 代码：更新 `paper/iclr2027/pixelopsd/make_revised_figures.py`，重新生成 `results_bars.pdf` 和 `qualitative.pdf`。
- 行为：将结果图的 delta 文本和变化箭头从右边界向内收，将定性图例上移空间重新平衡，避免标注与边界或 case 标题产生视觉粘连；数据和图语义不变。
- 验证：运行 `python make_revised_figures.py && ./build_pdf.sh` 成功生成 12 页；第 6、8 页渲染检查无裁切、遮挡或文字重叠。

### 2026-09-13 - 记录 v27 严格视觉自评结果

- 代码：新增 `paper/iclr2027/pixelopsd/reviewer_self_audit_v27.txt`。
- 行为：将“编译/溢出检查通过”和“相对主页论文的视觉质量通过”分开记录，明确图表信息密度、页面填充和 section 比例仍未达到验收门槛；不把占位结果纳入审美评审。
- 验证：自评依据最新 `main.pdf`、`main.log` 及第 2、5、6、7、8 页渲染结果；当前 gate 保持 FAIL，继续迭代。

### 2026-09-13 - 提升结果表的最小行高

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，在文档级设置 `\extrarowheight=2pt`。
- 行为：增加主比较表和独立消融表的基础行间距，使通栏评测表更易扫描；不增加参数字段、不改变任何评测数值或表格列定义。
- 验证：运行 `./build_pdf.sh` 成功生成 12 页；`main.log` 无 fatal、undefined-reference 或 overfull 诊断，正文仍为 9 页。

### 2026-09-13 - 放大 overview 以匹配参考论文首屏信息密度

- 代码：更新 `paper/iclr2027/pixelopsd/make_revised_figures.py`，将 overview 画布高度从 3.40 调整为 4.05 英寸。
- 行为：尝试放大问题、OPD/EGCA 干预和 inference-boundary 三段式视觉摘要；4.05 英寸和 3.68 英寸版本都会使正文变为 13 页，违反 9 页正文约束，故回退到 3.40 英寸原版。
- 验证：回退后重新生成图并运行 `./build_pdf.sh`，恢复 12 页总页数和 9 页正文；该实验版本不作为最终图形提交。

### 2026-09-13 - 在固定画布内增加 overview 的具体状态

- 代码：更新 `paper/iclr2027/pixelopsd/make_revised_figures.py`，重新生成 `overview.pdf`。
- 行为：在问题卡加入 noun/relation/boundary 三类可混淆错误，在干预卡加入 coarse-to-fine partial-code 状态，在结果卡加入 diagnose/teach/anchor 三条收益轨；不改变方法定义、数据或画布高度。
- 验证：运行 `python make_revised_figures.py && ./build_pdf.sh` 后检查第 2 页图内元素、分页和日志诊断。

### 2026-09-13 - 修正 overview 错误标签的实际重叠

- 代码：更新 `paper/iclr2027/pixelopsd/make_revised_figures.py`。
- 行为：将问题卡中与 `R_C` 框重叠的 noun/relation/boundary 色块改为分散的短标签，避免新增状态说明遮挡核心标量框。
- 验证：重新生成图并运行 `./build_pdf.sh` 后复核第 2 页；标签与 `R_C`、照片和卡片边界均无重叠，正文仍为 9 页。

### 2026-09-13 - 为 qualitative 网格补充错误类型标题

- 代码：更新 `paper/iclr2027/pixelopsd/make_revised_figures.py`，重新生成 `qualitative.pdf`。
- 行为：四个 matched panel 的标题增加 coarse displacement、boundary drift、extent correction、stable prediction 语义标签，并改为两行排版；轮廓、IoU 和坐标系保持不变。
- 验证：运行 `python make_revised_figures.py && ./build_pdf.sh`，检查第 8 页标题换行无重叠、裁切或越界。

### 2026-09-13 - 重排 qualitative 图例以消除标题竞争

- 代码：更新 `paper/iclr2027/pixelopsd/make_revised_figures.py`。
- 行为：移除会占用 panel 标题空间的居中图例，改为标题下方的单行颜色说明，保持 target/baseline/Pixel-OPSD 的视觉编码不变。
- 验证：重新生成图并运行 `./build_pdf.sh`，第 8 页渲染检查 panel 标题、图例和图像边界无重叠。

### 2026-09-13 - 建立 section 长度比例审计基线

- 代码：新增 `paper/iclr2027/pixelopsd/section_ratio_audit_v28.txt`。
- 行为：记录 Pixel-OPSD 正文各 section 的页覆盖范围，并与本地 RMP-SAM、Grasp Any Region、OMG-Seg、SAMTok 的 section 分布对照；明确实验/视觉证据段落偏短，避免用页数相等替代实质内容。
- 验证：使用 `pdfinfo`、`pdftotext` 和渲染页核对页数与章节位置；该审计用于后续重排，当前 gate 保持 FAIL。

### 2026-09-13 - 将 Conclusion 锚定在正文第九页底部

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，在 Conclusion 前加入可伸缩垂直间距。
- 行为：保持正文内容和页数不变，将 Conclusion 稳定放置在第九页剩余空间的底部，避免结论悬在页面中部；不引入 filler 文本。
- 验证：运行 `./build_pdf.sh` 并渲染第 9 页，检查 Conclusion 未溢出到参考文献页。

### 2026-09-13 - 评估并回退主表字号扩展实验

- 代码：短暂将主表和消融表改为 `\normalsize` 与更大行距，随后恢复原有 `\small`/`1.35` 设置。
- 行为：字号扩展使文档变为 13 页，违反 9 页正文约束；回退后保留此前安全的全局 `\extrarowheight=2pt`，不改变表格字段或结果。
- 验证：回退后重新运行 `./build_pdf.sh`，确认恢复 12 页总页数和 9 页正文。

### 2026-09-13 - 重绘通栏主评测表

- 代码：更新 `paper/iclr2027/pixelopsd/make_revised_figures.py` 生成 `figures/main_table.pdf`，并更新 `main.tex` 使用该通栏评测展项。
- 行为：主表采用分组/换行表头、统一灰/青色行层次，仅保留现有 eval 指标；不增加参数列、不改变任何数值。为保持 9 页正文，表格画布压缩为 1.0 英寸并使用 8.7pt 字号；外层恢复为 `table*` 以保持 Table 编号语义。
- 验证：运行 `python make_revised_figures.py && ./build_pdf.sh` 成功生成 12 页；主表 PDF 资源加载正常，正文仍为 9 页，渲染检查无标题重叠。

### 2026-09-13 - 重绘独立消融评测表

- 代码：更新 `paper/iclr2027/pixelopsd/make_revised_figures.py` 生成 `figures/ablation_table.pdf`，并更新 `main.tex` 使用通栏 `Table 2`。
- 行为：消融表采用阶段行、换行表头和统一灰/青色层次，仅保留 prefix agreement、空间、校准和 QA/refusal 等独立诊断；占位单元格继续使用 `--`，不重复主表指标。
- 验证：运行 `python make_revised_figures.py && ./build_pdf.sh` 成功生成 12 页；第 7 页显示为 `Table 2`，表格与诊断图均无裁切、重叠，正文仍为 9 页。

### 2026-09-13 - 将参考文献压缩为单页

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，将 bibliography 组改为 `\scriptsize`、`\baselinestretch=0.95`、`\bibsep=0pt`。
- 行为：参考文献从两页压缩为一页，正文保持 9 页，附录顺序不变；不删除引用条目。
- 验证：运行 `./build_pdf.sh` 成功生成 11 页（9 页正文、1 页参考文献、1 页附录）；第 10 页渲染确认条目未裁切，日志无 `Overfull`，仅保留 bibliography 页的既有 `Underfull vbox` 提示。

### 2026-09-13 - 轻量放大 overview 跨栏宽度

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，将 `overview.pdf` 宽度从 0.94 调整为 0.98 文本宽度。
- 行为：提高第 2 页问题/干预/结果摘要的有效面积和标签可读性，不改变画布高度、图内布局或方法语义。
- 验证：运行 `./build_pdf.sh` 并渲染第 2 页，确认正文仍为 9 页且图形无横向裁切。

### 2026-09-13 - 移除 overview 中贴边的 partial-code 说明

- 代码：更新 `paper/iclr2027/pixelopsd/make_revised_figures.py`。
- 行为：删除贴近 EGCA 框上沿的冗余 `partial-mask states` 标签，保留独立的 `z1 / z1:z2 / z1...zK` 状态色条，避免任何细小文字粘连。
- 验证：重新生成图并运行 `./build_pdf.sh`，第 2 页渲染确认 overview 内部无文字重叠，正文仍为 9 页。

### 2026-09-13 - 启用模板级 flushbottom 页面对齐

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，在文档开始处启用 `\flushbottom`。
- 行为：让正文页的可用高度通过模板正常伸缩分配，稳定第 9 页 Conclusion 的底部位置；不新增文字或改变图表数据。
- 验证：运行 `./build_pdf.sh` 成功生成 11 页；第 9 页渲染确认 Conclusion 位于页底且无异常拉伸，日志无 `Overfull`。

### 2026-09-13 - 局部放宽 bibliography 的底部对齐

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，在 bibliography 分组内加入 `\raggedbottom`。
- 行为：参考文献保持单页和紧凑条目，不让全局 `\flushbottom` 强行拉伸参考文献页；正文页布局和 Conclusion 位置保持不变。
- 验证：运行 `./build_pdf.sh` 成功生成 11 页；无新增 `Overfull` 或引用错误，保留 page 5 浮动体的既有 `Underfull vbox` 提示待后续版面重排处理。

### 2026-09-13 - 改为全局 raggedbottom 以消除浮动体警告

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，将文档级 `\flushbottom` 改为 `\raggedbottom`，Conclusion 前的显式伸缩保持不变。
- 行为：消除 page 5 浮动体组合引起的 `Underfull vbox`，避免模板强行拉伸段落；正文页数、图表顺序和结论位置不变。
- 验证：运行 `./build_pdf.sh` 成功生成 11 页；`main.log` 不再包含 `Underfull`、`Overfull`、undefined-reference 或 fatal 诊断。

### 2026-09-13 - 评估并回退参考文献字号放大实验

- 代码：短暂将 bibliography 改为 `\footnotesize`/`0.80` 行距，随后恢复 `\scriptsize`/`0.95`。
- 行为：更大字号会使参考文献重新占用两页，违反单页参考文献约束；恢复后保留 9 页正文、1 页参考文献、1 页附录布局。
- 验证：回退后运行 `./build_pdf.sh` 成功生成 11 页，正文和引用顺序不变。

### 2026-09-13 - 评估并回退 Experiments 浮动体抑制实验

- 代码：短暂在 Experiments 前加入 `\suppressfloats[t]`，随后删除。
- 行为：该命令虽让章节标题先于表格出现，却把主表和结果图推迟到下一页，造成第 5 页大面积空白；已回退以保持整体页面填充。
- 验证：回退后运行 `./build_pdf.sh`，恢复 11 页布局；该实验版本不作为最终稿。

### 2026-09-13 - 移除主表中裁切的指标分组色带

- 代码：更新 `paper/iclr2027/pixelopsd/make_revised_figures.py`，重新生成 `figures/main_table.pdf`。
- 行为：删除在压缩画布中发生文字裁切的装饰性色带，保留清晰的换行表头、行层次和完整 eval 指标，避免任何视觉错误。
- 验证：运行 `python make_revised_figures.py && ./build_pdf.sh` 后复核第 5 页主表无裁切或重叠，正文仍为 9 页。

### 2026-09-13 - 执行 v31 完整交付门审计

- 代码：新增 `paper/iclr2027/pixelopsd/reviewer_self_audit_v31.txt`。
- 行为：一次性核对 PDF 页数、section/图表数量、禁止模块名、LaTeX 致命诊断和占位符策略，并将机械检查与相对参考论文的人工视觉 gate 分开。
- 验证：自动检查通过（11 页、9 页正文、无 fatal/overfull/undefined），人工视觉 gate 仍为 FAIL，继续迭代。

### 2026-09-13 - 补充附录 update pseudocode

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex` 附录，在 Inference path 后加入 Update pseudocode。
- 行为：以两句压缩摘要明确 rollout、prefix decode、teacher/evidence、四项 loss、归一化和共享 optimizer update 的可复现顺序；不新增正文模块或实验结果。
- 验证：运行 `./build_pdf.sh` 成功恢复 11 页（9 页正文、1 页参考文献、1 页附录），附录未溢出到新页。

### 2026-09-13 - 轻量放大 framework 跨栏宽度

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，将 `framework.pdf` 宽度从 0.95 调整为 0.98 文本宽度。
- 行为：增加训练/证据/目标函数节点的有效显示面积，提高第 4 页 framework 标签可读性；不改变图高、节点关系或分页。
- 验证：运行 `./build_pdf.sh` 并渲染第 4 页，确认无横向裁切、节点重叠，正文仍为 9 页。

### 2026-09-13 - 完成 11 页全页渲染完整性审计

- 代码：新增 `paper/iclr2027/pixelopsd/fullrender_audit_v33.txt`。
- 行为：统一 72 dpi 栅格化 PDF 全部 11 页，记录页面尺寸和正文带内像素占用，确认无空白页、尺寸异常或资源丢失；低占用页标记为后续版面优化目标。
- 验证：运行 `pdftoppm` 与只读 PIL 统计，11 页均为 612x792 且含有效内容；该审计不把渲染完整性误报为相对参考论文的质量通过。

### 2026-09-13 - 量化正文页面有效像素密度

- 代码：新增 `paper/iclr2027/pixelopsd/page_density_audit_v29.txt`。
- 行为：以统一 80 dpi 渲染和阈值统计正文 1--9 页的非页边像素占比，定位第 2、5 页为当前最稀疏页面；该诊断不把页边距误判为内容，也不通过 filler 提高密度。
- 验证：运行 `pdftoppm` 与只读 PIL 统计得到可复现密度表；当前 gate 保持 FAIL，后续针对第 2、5 页做有效内容重排。

### 2026-09-13 - 在 overview 后加入方法运行示例

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`。
- 行为：尝试在 Figure 1 后加入 running example，具体说明语义正确但边界过扩的轨迹如何分别由 OPD 与 EGCA 处理；该段即使压缩也会使正文变为 13 页，故删除，不作为最终稿内容。
- 验证：删除后运行 `./build_pdf.sh`，恢复 12 页总页数和 9 页正文；不改变图表或结果。

### 2026-09-13 - 适度放大 qualitative evidence 网格

- 代码：更新 `paper/iclr2027/pixelopsd/make_revised_figures.py`，将 qualitative 画布高度从 4.45 调整为 4.78 英寸。
- 行为：尝试扩大四个 matched panel 的有效图像面积；4.78 英寸版本会使正文变为 13 页，违反 9 页约束，故回退到 4.45 英寸原版。
- 验证：回退后重新生成图并运行 `./build_pdf.sh`，恢复 12 页总页数和 9 页正文；该实验版本不作为最终图形提交。

### 2026-09-13 - 修复 100k scaling runner 的磁盘峰值与 Llama judge 超时

- 代码：更新 `tools/run_100k_scaling_opsd_8gpu.sh`。
- 行为：将 checkpoint 周期从每 5 step 调整为每 25 step，并将保留数量从 2 改为 1；单个 FSDP checkpoint 约 69 GiB，避免下一次保存前双 checkpoint 峰值导致训练在 `experiment_log.jsonl` 写入时触发 `OSError: [Errno 28] No space left on device`。新增可覆盖的 `JUDGE_TIMEOUT_SECONDS`（默认 600 秒），用于覆盖本次 Llama vLLM 首次编译超过原 180 秒健康检查窗口的问题。训练仍使用 `RESUME=true`，数据、loss、Ray 拓扑和 1 epoch 配置不变。
- 验证：从 `run.log` 确认磁盘满根因与 vLLM 180 秒超时；运行 `bash -n tools/run_100k_scaling_opsd_8gpu.sh`、`git diff --check -- tools/run_100k_scaling_opsd_8gpu.sh`，并核对启动/清理两处均使用 600 秒默认超时。

### 2026-09-13 - 回退实验图表底部浮动试验

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，将主表和结果图的浮动参数从 `[!b]` 恢复为 `[t]`。
- 行为：底部浮动试验将实验标题推迟到图表之后，并在第 6 页产生大块无效留白；恢复顶部浮动以保留此前更紧凑的正文流和图表顺序。
- 验证：对试验版第 5--6 页执行栅格检查，确认试验版版面退化后回退；下一步重新编译并检查总页数、日志告警及第 5--6 页视觉布局。

### 2026-09-13 - 更新当前论文版式自审结论

- 代码：新增 `paper/iclr2027/pixelopsd/reviewer_self_audit_v34.txt`，记录恢复顶部浮动后的当前 PDF 审查结果。
- 行为：明确区分机械交付通过与相对主页论文的主观展示质量未通过，避免把无溢出误报为视觉优越性；记录第 5--6 页顺序改善及主表、消融表、图表信息密度仍需提升的差距。
- 验证：基于最新 `main.pdf`（11 页）及 90 dpi 渲染的第 5、6、9 页检查；`main.log` 无 Underfull/Overfull/Fatal/Undefined 告警。

### 2026-09-13 - 在九页约束内增密主结果表

- 代码：更新 `paper/iclr2027/pixelopsd/make_revised_figures.py`，将 `main_table.pdf` 画布由 1.48 英寸增至 1.78 英寸、表格有效框增高并将字体增至 9.2 pt；未添加未经核验的 baseline 数值。
- 行为：主表在跨栏页面中具有更大的行高和可读字号，减少窄条视觉问题，同时保持正文页数与现有指标范围不变。
- 验证：重新运行 `python make_revised_figures.py` 与 `./build_pdf.sh`；`main.pdf` 为 11 页，`main.log` 无 Underfull/Overfull/Fatal/Undefined 告警，并渲染第 5 页确认表格无裁切或重叠。该改动仍不足以宣称达到主页论文主表的半页信息密度。

### 2026-09-13 - 记录主表增密后的严格自审

- 代码：新增 `paper/iclr2027/pixelopsd/reviewer_self_audit_v35.txt`。
- 行为：以最新 11 页 PDF 重新评估图表可读性、页数、告警和 section 比例；明确主表可读性提升但仍未达到参考论文的信息密度，因此严格门槛保持 FAIL。
- 验证：基于第 5 页 90 dpi 栅格图、`pdfinfo` 页数和 `main.log` 诊断执行；未将机械检查结果误报为论文质量通过。

### 2026-09-13 - 回退主表过度增高试验

- 代码：再次调整 `paper/iclr2027/pixelopsd/make_revised_figures.py`，将主表画布从 2.28 英寸恢复为 1.78 英寸，并恢复有效表格框高度。
- 行为：2.28 英寸版本即使保持三行数据也将 Conclusion 推至第 10 页，违反正文 9 页约束；恢复 1.78 英寸以保持正文分页稳定。
- 验证：重新生成图并编译，`main.pdf` 恢复为 11 页，`main.log` 无 Underfull/Overfull/Fatal/Undefined 告警；该试验版本不作为最终版。

### 2026-09-13 - 回退主表分组装饰叠加试验

- 代码：更新 `paper/iclr2027/pixelopsd/make_revised_figures.py`，移除主表上方新增的分组线与标签。
- 行为：该装饰在栅格化 PDF 中落入表头/首行区域并造成视觉叠加，违反无重叠要求；移除后恢复稳定的无冲突表格。
- 验证：重新生成图并编译，`main.pdf` 为 11 页，`main.log` 无 Underfull/Overfull/Fatal/Undefined 告警；第 5 页视觉检查确认无叠加。

### 2026-09-13 - 回退消融表高度扩展试验

- 代码：更新 `paper/iclr2027/pixelopsd/make_revised_figures.py`，将 `ablation_table.pdf` 画布从 2.02 英寸恢复为 1.55 英寸。
- 行为：高度扩展会将正文推至第 10 页，且不能解决占位诊断值造成的信息密度不足；恢复稳定尺寸以保持 9 页正文约束。
- 验证：重新生成图并编译，`main.pdf` 恢复为 11 页，`main.log` 无 Underfull/Overfull/Fatal/Undefined 告警；第 7 页检查无裁切或重叠。

### 2026-09-13 - 完成正文全页严格自审

- 代码：新增 `paper/iclr2027/pixelopsd/reviewer_self_audit_v36.txt`。
- 行为：对正文 1--9 页进行 110 dpi 全量栅格检查，并核对七个图形资源；分别记录机械版式通过与相对主页论文的视觉/写作门槛未通过，避免将无碰撞误判为超越参考论文。
- 验证：`pdftoppm` 全页渲染、像素占用统计、`pdftotext` 禁用模块名检查及 `main.log` 诊断均完成；严格 gate 继续为 FAIL。

### 2026-09-13 - 提升消融诊断图层次与因果顺序可读性

- 代码：更新 `paper/iclr2027/pixelopsd/make_revised_figures.py`，增大消融诊断图 active/off 节点，加入不改变语义的紧凑 causal-order 轨道，并新增 `reviewer_self_audit_v37.txt`。
- 行为：在原有画布内提高行间层次和因果顺序可读性，未引入新模块或新指标，也未改变占位数据。
- 验证：重新生成图并编译 `main.pdf`（11 页）；90 dpi 检查第 7 页确认轨道、标签和诊断 rail 无重叠/裁切，日志无版式告警；全局参考论文优越性 gate 仍保持 FAIL。

### 2026-09-13 - 回退 qualitative 图宽度扩展试验

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，将 qualitative 跨栏图宽度从 0.95 恢复为 0.90 文本宽度。
- 行为：0.95 版本虽放大 panel，却将正文扩展到 10 页并破坏第 9 页 Conclusion 约束；恢复 0.90 以保持分页稳定。
- 验证：重新编译后 `main.pdf` 恢复 11 页，`main.log` 无 Underfull/Overfull/Fatal/Undefined 告警；试验版不作为最终图形布局。

### 2026-09-13 - 为 overview 增加具体失败状态证据条

- 代码：更新 `paper/iclr2027/pixelopsd/make_revised_figures.py`，在 Problem 卡片底部加入 noun、extent、edge 三个同风格缩略图，并移除被其替代的冗余说明行；新增 `reviewer_self_audit_v38.txt`。
- 行为：提高 overview 的视觉状态数量和下半区利用率，明确 scalar reward 对不同失败类型的混淆；不改变方法、指标或推理路径。
- 验证：重新生成图并编译 `main.pdf`（11 页）；110 dpi 检查第 2 页确认缩略图均在卡片内、与 scalar/标签/邻卡片无重叠，日志无版式告警；全局主页论文优越性 gate 仍为 FAIL。

### 2026-09-13 - 收紧逐图参考级评审门槛

- 代码：新增 `paper/iclr2027/pixelopsd/reviewer_self_audit_v39.txt`。
- 行为：将评审拆分为五项逐图门槛，并对 overview、framework、results、ablation、qualitative 逐一判定；机械无碰撞通过不再等同于参考级信息密度通过。
- 验证：基于当前 11 页 PDF、全量图资源和同尺度本地参考图复核；五张图均通过机械清洁项但未通过参考级密度项，global gate 明确为 FAIL。

### 2026-09-13 - 标注 framework 的 EGCA 反馈路径

- 代码：更新 `paper/iclr2027/pixelopsd/make_revised_figures.py`，在 EGCA 指向目标函数的反馈箭头旁加入 `code credit` 语义标签；新增 `reviewer_self_audit_v40.txt`。
- 行为：明确长反馈箭头表示代码级 credit 回流，提升方法图可解释性；不改变节点、公式、训练路径或推理接口。
- 验证：重新生成图并编译 `main.pdf`（11 页）；110 dpi 检查第 5 页确认标签不接触节点、分隔线、caption 或正文，日志无版式告警；全局严格 gate 仍为 FAIL。

### 2026-09-13 - 为 qualitative panels 增加局部放大证据

- 代码：更新 `paper/iclr2027/pixelopsd/make_revised_figures.py`，在每个 qualitative panel 内加入基于目标 mask 包围盒的 zoom inset，并新增 `reviewer_self_audit_v41.txt`。
- 行为：局部放大显示 boundary/extent 轮廓差异，提高证据图的信息密度；复用现有目标、CycleGRPO 和 Pixel-OPSD 轮廓，不增加指标或推理输入。
- 验证：重新生成图并编译 `main.pdf`（11 页）；110 dpi 检查第 8 页确认四个 inset 均位于各自 panel 内且不触碰标题、图例、相邻 panel 或 caption，日志无版式告警；全局严格 gate 仍为 FAIL。

### 2026-09-13 - 回退主表内部脚注叠加试验

- 代码：更新 `paper/iclr2027/pixelopsd/make_revised_figures.py`，移除放置在主表画布底部的 protocol 脚注。
- 行为：脚注经 PDF 缩放后落入 Pixel-OPSD 最后一行，造成可见叠加；移除后恢复已验证的无冲突表格，协议说明保留在 caption 和正文。
- 验证：重新生成图并编译，`main.pdf` 为 11 页，`main.log` 无 Underfull/Overfull/Fatal/Undefined 告警；失败试验不作为交付版。

### 2026-09-13 - 清理摘要中的过程性占位说明

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，移除摘要中关于 provisional checkpoint 单元的过程性说明；占位符解释保留在表格 caption 与附录。
- 行为：摘要聚焦问题、OPD、EGCA 和监督混合的研究叙事，避免将内部结果转移流程写入主摘要，不改变任何实验数据或方法定义。
- 验证：重新编译后 `main.pdf` 为 11 页，`main.log` 无 Underfull/Overfull/Fatal/Undefined 告警，`git diff --check -- main.tex` 通过；严格参考级 gate 仍为 FAIL。

### 2026-09-14 - 调整总览/方法图位置并合并实验叙述

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，将 overview 跨栏图移至 Related Work 前、使其浮动到正文第 2 页顶部；将 framework 跨栏图移至 Method 标题后、使其浮动到 Method 开头；将 Experiments 中逐条短 subsection 合并为三组 claim-bearing 段落并使用 inline signpost。
- 行为：恢复“总览图 -> 方法框架图 -> 实验证据”的读者路径，实验部分由碎片化短段改为较长的 evaluation design、headline comparison、controls and measurement 叙述；不改变方法、指标或训练实现。
- 验证：重新编译并渲染第 2、3、5、6 页，确认图位置和段落结构符合要求；`main.pdf` 11 页（正文 9 页），`main.log` 无 Underfull/Overfull/Fatal/Undefined 告警；详细结论记录于 `reviewer_self_audit_v44.txt`。

### 2026-09-14 - 将实验标题对齐主页论文命名风格

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，把自定义的 `Headline comparison`、`Evaluation design`、`Controls and measurement` 标题改为参考论文常用的 `Experiment Setup`、`Main Results` 和 `Ablation Study and Visual Analysis`；数据、基线、指标和实现检查归入 Experiment Setup，比较叙述归入 Main Results。
- 行为：实验 section 的标题层级和命名与本地 SAMTok/RMP-SAM 等主页论文一致，避免自行发明章节名；不改变实验内容或指标。
- 验证：重新编译并检查标题分页，`main.pdf` 为 11 页（正文 9 页），`main.log` 无 Underfull/Overfull/Fatal/Undefined 告警；详细审计记录于 `reviewer_self_audit_v45.txt`。

### 2026-09-14 - 按主页论文规范清理摘要、引言与相关工作结构

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，删除 Introduction 中自定义的 Motivation/Contributions/Research questions 加粗标题，删除 Related Work 中自定义 subsection 标题并改为连续比较叙述；在 Method 前保留分页边界，使 framework 位于 Method 开头；补充 related-work 比较段落以避免第 2 页无效空白。
- 行为：摘要、引言、相关工作恢复标准论文连续段落风格；总览图位于第 2 页顶部，framework 位于第 3 页 Method 开头；不改变方法、指标或训练实现。
- 验证：重新编译并渲染第 1--3 页，确认无自定义引言标题、总览/框架图位置正确且无碰撞；`main.pdf` 11 页（正文 9 页），`main.log` 无 Underfull/Overfull/Fatal/Undefined 告警；详细审计记录于 `reviewer_self_audit_v46.txt`。

### 2026-09-14 - 补充 related-work 方法边界比较

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，在连续 Related Work 叙述末尾加入与 adapter/reranking 方法的边界比较段落，避免使用自定义小标题并改善第 2 页有效内容密度。
- 行为：明确 Pixel-OPSD 保持 actor/decoder 接口和 sampled trajectory 的差异，不引入新模块或新实验主张。
- 验证：重新编译并渲染第 2--3 页，`main.pdf` 仍为 11 页（正文 9 页），overview 在第 2 页顶部、framework 在第 3 页 Method 开头，日志无 Underfull/Overfull/Fatal/Undefined 告警；详细审计记录于 `reviewer_self_audit_v47.txt`。

### 2026-09-14 - 将消融实验归入 Experiments 标准层级

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，把顶层 `Ablation Study and Visual Analysis` 改为 Experiments 下的 4.3 subsection，并保留 4.1 `Experiment Setup`、4.2 `Main Results`。
- 行为：实验标题层级对齐本地 RMP-SAM 等主页论文的组织方式；Introduction/Related Work/Conclusion 仍为连续标准段落，不增加自定义标题。
- 验证：重新编译并渲染第 7 页确认显示为标准 4.3 标题，`main.pdf` 为 11 页（正文 9 页），`main.log` 无 Underfull/Overfull/Fatal/Undefined 告警；详细审计记录于 `reviewer_self_audit_v48.txt`。

### 2026-09-14 - 完成摘要/引言/相关工作/结论结构复核

- 代码：新增 `paper/iclr2027/pixelopsd/reviewer_self_audit_v49.txt`，复核当前 `main.tex` 的 Abstract、Introduction、Related Work、Conclusion 和实验标题层级。
- 行为：确认摘要为单段标准叙述，引言为无自定义标题的连续段落，Related Work 为无自定义 subsection 的连续比较，Conclusion 为无内部 subsection 的标准段落；不改变科学内容。
- 验证：依据第 1--3、7 页 90 dpi 渲染及 `main.log`，确认总览/框架图位置、页数和告警均稳定；严格图形优越性仍未宣称通过。

### 2026-09-14 - 清理解释性章节残留自定义 subsection

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，删除 `Interpretability` subsection 标题，使 Interpretability and Reproducibility 以连续正文呈现，仅保留 mechanistic diagnostics 与 failure taxonomy 两个证据性段落 signpost。
- 行为：减少不必要的自创层级，使全文更接近主页论文的自然段落结构；不改变方法、实验或结论内容。
- 验证：重新编译并渲染第 9 页，`main.pdf` 为 11 页（正文 9 页），Conclusion 仍在第 9 页底部，`main.log` 无 Underfull/Overfull/Fatal/Undefined 告警；详细审计记录于 `reviewer_self_audit_v50.txt`。

### 2026-09-14 - 将 Conclusion 合并为主页论文式单段

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，将 Conclusion 的三个短段合并为一个连续段落，保留原有论断和第 9 页底部锚定。
- 行为：结论呈现更接近主页论文的紧凑叙述，不新增结果或主张。
- 验证：重新编译并渲染第 9 页，`main.pdf` 为 11 页（正文 9 页），无段落碰撞或裁切，`main.log` 无 Underfull/Overfull/Fatal/Undefined 告警；详细审计记录于 `reviewer_self_audit_v51.txt`。

### 2026-09-14 - 对齐实验后半段层级并恢复九页正文

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，将 Qualitative Analysis 与 Interpretability and Reproducibility 纳入 Experiments 子层级，改用 Qualitative Results 及其 subsubsection；删除重复的解释性诊断段落，并保留机制信息于附录记录。
- 行为：实验结构更接近主页论文的 4.1/4.2/4.3 组织，Conclusion 紧接实验并回到第 9 页；Abstract、Introduction、Related Work、Conclusion 仍无自定义标题分段，EGCA 仍是唯一空间模块。
- 验证：运行 `./build_pdf.sh`，`main.pdf` 为 11 页（正文 9 页、参考文献 1 页、附录 1 页），`main.log` 无 Underfull/Overfull/Undefined/Fatal 告警；已检查第 9--10 页分页与 Conclusion 位置。

### 2026-09-14 - 恢复主页论文式自然段边界

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，在 Introduction 与 Related Work 的主题转换处加入自然段空行，不添加任何自定义标题；保留 Abstract/Conclusion 单段和 Experiments 标准层级。
- 行为：引言按动机、误差归因、贡献、研究问题分成自然段；相关工作按 pixel MLLM、CycleGRPO/OPD、监督锚点与差异化定位组织，阅读节奏更接近主页论文。
- 验证：重新编译并渲染第 1--3 页，`main.pdf` 仍为 11 页（正文 9 页），总览图位于第 2 页顶部，`main.log` 无 Underfull/Overfull/Undefined/Fatal 告警；审计记录见 `reviewer_self_audit_v53.txt`。

### 2026-09-14 - 移除实验定性分析中的自定义小标题

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，将 Sensitivity and stability、Qualitative protocol、Threats to validity 合并为 Experiments 下的连续自然段，保留 `Qualitative Results` 作为标准实验子节。
- 行为：实验后半段不再引入主页论文中不存在的 subsubsection 标题，定性、稳定性和有效性讨论以连贯段落呈现；不改变指标、方法或图表。
- 验证：运行 `./build_pdf.sh`，`main.pdf` 保持 11 页（正文 9 页），`main.log` 无 Underfull/Overfull/Undefined/Fatal 告警。

### 2026-09-14 - 补充实验末尾可复现性段落

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，在 Qualitative Results 与 Conclusion 之间加入连续的 checkpoint、manifest、token-level audit 说明，不新增标题。
- 行为：填充第 9 页实验末尾的有效内容，明确诊断记录和统一评测协议，保持 Conclusion 在正文最后一页。
- 验证：运行 `./build_pdf.sh` 并渲染第 9 页，`main.pdf` 仍为 11 页（正文 9 页），无版面或引用警告。

### 2026-09-14 - 修复 Method 前空白页并定位框架图

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，移除 Related Work 与 Method 之间导致双栏浮动体延迟的强制 `clearpage`。
- 行为：Related Work 尾部与 Method 连续排版，framework 图稳定位于第 3 页顶部，消除原先仅有少量文字的空白页；不改变方法内容。
- 验证：运行 `./build_pdf.sh` 并渲染第 2--4 页，`main.pdf` 为 11 页（正文 9 页），`main.log` 无 Underfull/Overfull/Undefined/Fatal 告警。

### 2026-09-14 - 放大主评测表的版面占比

- 代码：更新 `paper/iclr2027/pixelopsd/make_revised_figures.py`，增大主表画布、字体、行高与单元格内边距；表格仍只包含评测指标。
- 行为：主表在双栏页面中具有更清晰的阅读尺寸和更接近主页论文的版面占比，不引入训练参数或额外字段。
- 验证：重新生成 figures 并运行 `./build_pdf.sh`，`main.pdf` 仍为 11 页（正文 9 页），第 6 页无重叠/溢出，`main.log` 无版面警告。

### 2026-09-14 - 细化 Main Results 的自然段组织

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，将 benchmark 定义、方法对比、训练协议与完整数据控制拆为少量较长自然段，不新增自定义小标题。
- 行为：实验叙述更接近主页论文的长段落风格，避免短句堆叠，同时保持主表指标独立、结果不变。
- 验证：运行 `./build_pdf.sh`，正文仍为 9 页（总 11 页），无 Underfull/Overfull/Undefined/Fatal 警告。

### 2026-09-14 - 扩展单段 Abstract 的信息密度

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，在不引入标题或分段的前提下补充 scalar-credit 问题、三阶段方法递进、共享接口和评测协议。
- 行为：摘要长度与主页论文常见摘要更接近，仍明确 OPD -> EGCA -> supervised mixing，且不新增结果或未经验证的数值。
- 验证：运行 `./build_pdf.sh`，`main.pdf` 仍为 11 页（正文 9 页），`main.log` 无版面或引用警告。

### 2026-09-14 - 进一步提升主表可读尺寸

- 代码：更新 `paper/iclr2027/pixelopsd/make_revised_figures.py`，进一步增加主表画布高度、字体与行间距，保持仅展示评测指标。
- 行为：主表在第 6 页的可读尺寸更接近主页论文展示，且不改变任何数值或比较对象。
- 验证：重新生成图表并运行 `./build_pdf.sh`，`main.pdf` 仍为 11 页（正文 9 页），第 6 页无重叠、裁切或溢出，日志无版面警告。

### 2026-09-14 - 清理 Conclusion 单段的源文件边界

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，移除 Conclusion 正文末尾多余的空段边界，确保源文件和渲染结果均为单段结论。
- 行为：不改变结论文字或分页，仅使单段结构检查无歧义。
- 验证：运行 `./build_pdf.sh`，`main.pdf` 仍为 11 页（正文 9 页），`main.log` 无版面或引用警告。

### 2026-09-14 - 将定性结果并入消融分析小节

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，移除额外的 `4.4 Qualitative Results` 标题，将定性图说明作为 `4.3 Ablation Study and Visual Analysis` 的连续正文。
- 行为：Experiments 现在严格保持 4.1/4.2/4.3 三个主页论文式小节，定性结果、消融与分析在同一小节内组织；不改变图表内容。
- 验证：运行 `./build_pdf.sh` 并检查 PDF 章节编号仅出现 4.1、4.2、4.3，正文仍 9 页且无版面警告。

### 2026-09-14 - 补足 Introduction 的动机论证段落

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，加入关于语义选择、空间细化和语言表达三类耦合决策及其对应干预的连续论证段落，不添加标题。
- 行为：Introduction 的篇幅与主页论文更接近，动机到 OPD/EGCA/监督混合递进关系更清楚；总览图仍位于第 2 页顶部。
- 验证：运行 `./build_pdf.sh` 并渲染第 1--2 页，正文保持 9 页（总 11 页），无版面或引用警告。

### 2026-09-14 - 消除可能被误判为自定义标题的措辞

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，将 Main Results 正文中的 “headline comparison” 改为普通的 “main comparison”，避免与用户禁止的自定义标题产生歧义。
- 行为：不改变实验含义或结构，仅统一正文措辞；章节层级保持主页论文式 4.1/4.2/4.3。
- 验证：运行 `./build_pdf.sh`，正文 9 页、总计 11 页，日志无版面或引用警告。

### 2026-09-14 - 启用 ICLR final 排版模式

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，加入模板提供的 `\\iclrfinalcopy`，关闭 review 模式左侧行号。
- 行为：PDF 版面与主页论文成稿一致，不再显示灰色逐行编号；章节、图表和正文内容不变。
- 验证：运行 `./build_pdf.sh` 并渲染第 1 页，确认行号消失；`main.pdf` 仍为 11 页（正文 9 页），`main.log` 无 Underfull/Overfull/Undefined/Fatal 警告。

### 2026-09-14 - 固定 framework 与 Method 的连续起始页

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，精简 Related Work 并在 Method 前保留分页，使 framework 图与 Method 标题从第 3 页同步开始；补充一段方法边界比较以改善第 2 页填充。
- 行为：消除 framework 位于 Related Work 尾部之后的顺序歧义，保持 overview 第 2 页顶部、framework 第 3 页顶部，章节组织更接近主页论文。
- 验证：重新编译并渲染第 2--3 页，正文仍 9 页（总 11 页），无版面或引用警告。

### 2026-09-14 - 完善第 2 页 Related Work 填充与顺序

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，补充一段无标题的 preference/on-policy distillation 比较，保持 Related Work 完整结束后再分页进入 Method。
- 行为：第 2 页内容填充更均衡；第 3 页 framework 图与 `3 Method` 标题同步置顶，避免浮动体与 Related Work 交错。
- 验证：运行 `./build_pdf.sh` 并渲染第 2--3 页，`main.pdf` 保持 11 页（正文 9 页），无 Underfull/Overfull/Undefined/Fatal 警告。

### 2026-09-14 - 扩展主表方法对比并修正长方法名排版

- 代码：更新 `paper/iclr2027/pixelopsd/make_revised_figures.py`，主表加入来自公开 SAMTok 验证表的五个以上先前方法，并扩大方法列、调整字号与单元格间距。
- 行为：主表现在同时比较 LISA、MLLMSeg、HiMTok、ARGenSeg、CycleGRPO、SAMTok、Qwen25VL-SAMTok (rl) 与 Pixel-OPSD；仅展示评测指标，不填入训练配置；不可比或未公开指标继续使用 `--`，长方法名不再贴边或溢出。
- 验证：重新运行 `make_revised_figures.py` 与 `build_pdf.sh`，`main.pdf` 共 11 页（正文 9 页）；渲染第 6 页检查主表、柱状图和图注，无 Underfull/Overfull/Undefined/Fatal 日志警告或可见文字裁切。

### 2026-09-14 - 补足第 9 页实验分析并保持结论收束

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，在 4.3 末尾补充跨基准解释、负结果报告原则、定性案例选择和训练/部署边界分析，移除结论前的弹性填充。
- 行为：正文实验分析不再过早结束；第 9 页由连续实验讨论自然过渡到 Conclusion，减少大面积空白，同时保持结论为正文最后一节。
- 验证：运行 `./build_pdf.sh`，`main.pdf` 仍为 11 页（正文 9 页）；渲染第 9 页确认新增段落与结论无重叠、无裁切，`main.log` 无版面警告。

### 2026-09-14 - 更新为作者指定论文标题

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex` 的 `\\title`。
- 行为：论文标题改为 `Pixel-OPSD: Beyond Scalar Rewards for Pixel-Level MLLMs via On-Policy Self-Distillation`，与作者指定版本一致；正文内容和章节结构不变。
- 验证：重新运行 `./build_pdf.sh`，检查首页标题换行、总页数及日志版面诊断。

### 2026-09-14 - 增加实验图表并细化段落节奏

- 代码：更新 `paper/iclr2027/pixelopsd/make_revised_figures.py` 与 `paper/iclr2027/pixelopsd/main.tex`，新增 evaluation map 和 failure-mode analysis 两类实验诊断图，并将实验、消融、引言和相关工作中的长段落拆分为中等长度自然段。
- 行为：实验部分现在包含主表、主结果柱状图、实验分析双面板、消融表、消融诊断图和定性图；新增图只表达评测协议与误差归因，不引入虚构指标或新模块。
- 验证：运行 `make_revised_figures.py` 与 `build_pdf.sh`，共 13 页（正文 11 页，含参考文献和附录）；`main.log` 无 Underfull/Overfull/Undefined/Fatal 警告，并检查实验图表无可见文字重叠或裁切。

### 2026-09-14 - 压缩实验尾部冗余并恢复结论收束

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，删除实验尾部重复的协议解释，保留定性案例、复现记录和结论所需论证。
- 行为：新增实验双面板图后，正文压缩为 10 页；结论回到实验正文最后一页，避免单独生成空白结论页；段落仍保持按论证功能拆分。
- 验证：运行 `build_pdf.sh`，`main.pdf` 共 12 页（正文 10 页、参考文献和附录），日志无 Underfull/Overfull/Undefined/Fatal 警告；第 10 页视觉检查确认结论与实验分析无重叠或裁切。

### 2026-09-14 - 恢复 ICLR 九页正文限制并合并实验诊断图

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，将消融诊断图与定性图合并为横向双面板，压缩实验尾部重复说明并保留一段可复现性边界说明。
- 行为：正文恢复为 9 页；第二页顶部贡献总览、Method 顶部框架图、实验主表和消融表均保留，实验诊断图与定性图共享一张横向展板，减少浮动体造成的空白页。
- 验证：运行 `build_pdf.sh`，`main.pdf` 共 11 页（正文 9 页、参考文献和附录）；`main.log` 无 Underfull/Overfull/Undefined/Fatal 警告，并渲染第 8--9 页确认图表无裁切或重叠。

### 2026-09-14 - 修复第 3 页空白并完成逐页正文审查

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，移除 Related Work 与 Method 之间导致孤立页的 `\\clearpage`，并补充必要的 Method 实现边界说明。
- 行为：第 3 页现在直接显示 Method 顶部框架图与方法正文；正文保持 9 页，第二页贡献总览图、主表和消融表均保留。
- 验证：重新运行 `build_pdf.sh`，渲染第 1--9 页逐页检查；确认无空白正文页、图表裁切或重叠，`main.log` 无 Underfull/Overfull/Undefined/Fatal 警告。

### 2026-09-14 - 按实际页面覆盖率补足第 9 页正文

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，在实验图表之后补充因果可证伪性、图表组织、复现边界和混合 batch 归一化说明。
- 行为：修正“页数为 9 但第 9 页仅有结论”的问题；正文仍为 9 页，Conclusion 保持在第 9 页末尾，新增内容均属于实验协议与实现解释。
- 验证：逐页渲染第 1--9 页并检查实际内容覆盖；`main.pdf` 共 11 页，`main.log` 无 Underfull/Overfull/Undefined/Fatal 警告。

### 2026-09-14 - 恢复参考文献默认字体与行距

- 代码：更新 `paper/iclr2027/pixelopsd/main.tex`，移除参考文献组中的 `\\scriptsize`、`\\baselinestretch=0.95`、`\\bibsep=0pt` 和 `\\selectfont` 压缩设置。
- 行为：参考文献现在使用 ICLR 模板默认字体、字号和条目间距；正文排版未被缩小。由于取消压缩，参考文献从一页扩展为两页，总 PDF 变为 12 页，但正文仍为 9 页。
- 验证：运行 `build_pdf.sh`，检查参考文献页渲染和正文页数；`main.log` 无 Underfull/Overfull/Undefined/Fatal 警告。

### 2026-09-14 - 整理 Overleaf 可直接导入的项目包

- 代码：生成 `paper/iclr2027/Pixel-OPSD-Overleaf.zip`，包含 `main.tex`、ICLR 模板 `.sty/.bst`、`references.bib`、`math_commands.tex` 和全部 `figures/*.pdf`。
- 行为：zip 根目录直接包含 `main.tex`，不包含本地 `.aux/.log/.out`、审计截图、训练日志或旧编译缓存，可直接上传 Overleaf 并以 `main.tex` 编译。
- 验证：检查 zip 清单共 15 个项目文件，确认主标题、参考文献路径、总览图、框架图、主表和消融表路径均存在。

### 2026-09-15 - 增加指定 step_178 权重的 GRES prompt 搜索临时入口

- 代码：新增 `evaluation/gres/qwen3vl_gres_eval_prompt_search_tmp.py`，仅用于当前 checkpoint 的离线 prompt ablation；维护中的 `qwen3vl_gres_eval.py` 未改变。
- 行为：通过 `PROMPT_VARIANT` 切换严格匹配、证据核验、目标保护、短 prompt cardinality、显式计数、视觉容错、多对象容错、按短语动态路由、动态放宽及动态属性严格规则等候选指令，统一要求 no-target 输出训练分布中的 `No target.`，并为每轮实验写入独立 case 目录和 metrics 文件。
- 验证：候选入口已通过 `PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile`；第 1--10 次候选均完成 14229/14229 样本并成功生成 N_acc/T_acc/gIoU/cIoU 指标；动态路由候选曾发现并修正词边界正则错误，修正后的有效动态候选继续使用同一完整数据和 `legacy_union` 解码协议；第 10 个 `dynamic_balanced` 候选按显式对象类别进行动态拒识并容忍关系措辞噪声，完整结果已与默认 prompt 对比。

### 2026-09-16 - 完成作者论文包 Figure 4 定性对比

- 代码：新增 `paper/iclr2027/pixelopsd_new/Pixel-OPSD-final/` LaTeX 项目包及 `make_qualitative_figure.py`；更新该包的 `main.tex`、`README.md` 和 `build_pdf.sh`，生成 `figures/qualitative_comparison.pdf`。
- 行为：Figure 4 不再使用 reserved/pending 占位布局，而是读取相同 GroundingSuite official-long-prompt manifest 的 3,715 条结果，固定展示 step-156 CycleGRPO 与 step-178 Pixel-OPSD 的四个真实案例（target recovery、extent refinement、failure、stable prediction），并在统一图像坐标中显示 target、baseline、ours 轮廓及逐例 IoU。正文明确记录评测协议、checkpoint 和样本类型；Figure 3 的待补数值占位保持不变。
- 验证：运行 `python3 make_qualitative_figure.py` 生成 PDF；渲染图4检查无文字/图像裁切或重叠；运行 `./build_pdf.sh` 成功生成 14 页 `main.pdf`，最终 `main.log` 无 Fatal、Undefined citation/reference 或 missing-file 错误（仅保留既有 Underfull 排版提示）；生成 `paper/iclr2027/pixelopsd_new/Pixel-OPSD-final-overleaf.zip`（zip 根目录直接包含 `main.tex`），并检查清单不含 `.aux/.log/.out` 编译缓存。

### 2026-09-16 - 按作者反馈校正 Figure 4 的 baseline 与样例选择

- 代码：更新 `paper/iclr2027/pixelopsd_new/Pixel-OPSD-final/main.tex`、`make_qualitative_figure.py` 和 `README.md`，重新生成 `figures/qualitative_comparison.pdf` 及可直接导入的论文包。
- 行为：论文明确将 SAMTok 作为 pretrained baseline/reference，将 CycleGRPO 定义为独立的 scalar-reward post-training comparator；Figure 4 使用同一 3,715 条 official-training-long-prompt manifest，仅展示相对 SAMTok 的四个大幅正增益样例（内部选择索引 374、633、158、2974，IoU 增益分别为 +0.970、+0.965、+0.964、+0.959），不在图中显示 case 编号、逐例 IoU、长 caption 或下降案例。消融表同时保留 SAMTok 和 CycleGRPO 两个参照行。
- 验证：重新运行 `python3 make_qualitative_figure.py` 和 `python3 -m py_compile make_qualitative_figure.py`；核验四条匹配输入的正增益；渲染 Figure 4 确认无文字/图像裁切或重叠；随后运行 `./build_pdf.sh`，检查 14 页 PDF、引用与文件路径诊断，并重新打包 Overleaf 项目。

### 2026-09-16 - 按 Oral qualitative figure 风格扩展 Figure 4 三模型对比

- 代码：更新 `paper/iclr2027/pixelopsd_new/Pixel-OPSD-final/make_qualitative_figure.py`、`main.tex` 和 `README.md`，加入官方 CycleGRPO 结果并重新生成 `figures/qualitative_comparison.pdf`。
- 行为：Figure 4 采用 `GT | SAMTok | CycleGRPO | Pixel-OPSD` 四列极简布局，方法名只出现一次，每个预测格仅保留 IoU 数值；不显示 query、case 编号、长 caption、图例解释或额外说明。六个匹配样例覆盖大幅恢复、中等提升、小幅提升和接近持平，避免只展示 0 到 0.9+ 的单一增益区间。
- 参考与依据：检索并核对 CVPR 2024 官方 Oral 日程及 10 篇公开 Oral 论文（Florence-2、LISA、InternVL、MMMU、Eyes Wide Shut、Visual Program Distillation、Learning to Segment Referred Objects、SceneFun3D、360+x、Describing Differences），归纳其共同的短列标题、同一输入横向对齐、低文字密度、指标贴近预测格和多难度样例组织方式。
- 验证：确认 SAMTok、CycleGRPO、Pixel-OPSD 三套结果均为同一 3,715 条 manifest 且 image/query 顺序完全一致；运行 `python3 -m py_compile make_qualitative_figure.py`、重新生成并渲染 Figure 4，随后运行 `./build_pdf.sh` 并重新打包 Overleaf 项目。

### 2026-09-16 - 完成 Figure 4 三模型最终版交付

- 代码：最终同步 `main.tex`、`make_qualitative_figure.py`、`README.md`、`figures/qualitative_comparison.pdf`、`submission_preview.pdf` 和 Overleaf zip。
- 行为：Figure 4 固定为 `GT | SAMTok | CycleGRPO | Pixel-OPSD`，六行样例同时保留大幅、中等、小幅和近似持平变化；每个预测格仅显示 IoU，方法名只显示在列头。
- 验证：`./build_pdf.sh` 成功生成 14 页 `main.pdf`；渲染第 10 页确认无裁切、重叠或过密文字；`main.log` 无 Fatal、Undefined citation/reference 或 missing-file 错误；zip 根目录包含 `main.tex` 且排除编译缓存。

### 2026-09-16 - 修正 Figure 4 样例约束并加入输入 caption

- 代码：更新 `paper/iclr2027/pixelopsd_new/Pixel-OPSD-final/make_qualitative_figure.py`、`main.tex` 和 `README.md`。
- 行为：六个样例全部满足 Pixel-OPSD IoU 同时高于 SAMTok 与 CycleGRPO 且优势超过 0.1；移除不满足该约束的原第三、六个样例。每行左侧展示真实 referring caption，右侧四个等宽面板依次为 GT、SAMTok、CycleGRPO、Pixel-OPSD，预测面板只保留 IoU 数值；新样例的轮廓差异经过逐格视觉筛选。
- 验证：核验样例 IoU 为 `0.44/0.25/0.97`、`0.57/0.37/0.98`、`0.45/0.08/0.86`、`0.37/0.47/0.89`、`0.51/0.66/0.93`、`0.52/0.27/0.96`（SAMTok/CycleGRPO/Pixel-OPSD）；重新生成并渲染 Figure 4，确认 caption、等宽图格和轮廓差异可读。

### 2026-09-16 - 参考 2026 Oral 风格完成 Figure 4 交付版

- 参考：核对 CVPR 2026 官方 Oral 日程及公开论文 CoSMo3D、GeoViS、RobotSeg、INSID3、PR-MaGIC、R2-Seg、VGGT-Segmentor、Molmo2、CURE、SegMoTE 的 qualitative panels，采用其等宽对比列、左侧输入描述、短列名和格内指标标注风格。
- 代码：最终更新 `make_qualitative_figure.py`、`main.tex`、`README.md`、`figures/qualitative_comparison.pdf`、`submission_preview.pdf` 和 Overleaf zip。
- 验证：运行 `python3 -m py_compile make_qualitative_figure.py` 与 `./build_pdf.sh`；生成 14 页 PDF，渲染第 10 页确认 caption、等宽图格和 mask 轮廓均可读，日志无 Fatal、Undefined citation/reference 或 missing-file 错误。

### 2026-09-16 - 优化 Figure 4 mask 可视化细节

- 代码：更新 `paper/iclr2027/pixelopsd_new/Pixel-OPSD-final/make_qualitative_figure.py`、`main.tex`、`README.md`，重新生成 `figures/qualitative_comparison.pdf`。
- 行为：预测与 GT mask 改为低透明度实心填充叠加，并使用 0.65pt 细边界；IoU 标记同步缩小并降低不透明度。该绘法保留原图纹理、目标内部细节和多目标之间的空间关系，避免粗描边遮挡内容。
- 验证：通过 `python3 -m py_compile make_qualitative_figure.py`；重新生成 Figure 4 并渲染检查六行样例，确认 mask 区域、边界、caption、IoU 和等宽面板均无裁切或重叠；随后运行 `./build_pdf.sh` 并更新 Overleaf zip。

### 2026-09-16 - 统一 Figure 4 面板宽度并进一步细化边界

- 代码：更新 `paper/iclr2027/pixelopsd_new/Pixel-OPSD-final/make_qualitative_figure.py`、`README.md`，重新生成 `figures/qualitative_comparison.pdf`。
- 行为：所有样例先放入相同 `1.33` 固定宽高比画布，再进入四个模型列，保留原图比例并消除竖幅样例造成的窄面板；mask 边界进一步细化为 `0.20pt`。
- 验证：运行 `python3 -m py_compile make_qualitative_figure.py`，渲染 Figure 4 检查六行四列外框等宽且无图像拉伸、裁切或文字重叠；运行 `./build_pdf.sh` 成功生成 14 页 PDF，并重新打包 Overleaf 项目。

### 2026-09-16 - 恢复 Figure 4 边界并提高原图清晰度

- 代码：更新 `paper/iclr2027/pixelopsd_new/Pixel-OPSD-final/make_qualitative_figure.py`、`README.md`，重新生成 `figures/qualitative_comparison.pdf`。
- 行为：mask 边界恢复为 `0.65pt`；原图与 overlay 使用 `interpolation="none"`，Figure 4 PDF 导出分辨率提升至 `1200dpi`，避免原始照片在导出时被低分辨率栅格化或平滑压缩。
- 验证：运行 `python3 -m py_compile make_qualitative_figure.py`，通过 `pdfimages -list` 核对 Figure 4 内嵌图像分辨率，再渲染检查边界、细节和等宽面板；运行 `./build_pdf.sh` 成功生成 14 页 PDF，并重新打包 Overleaf 项目。

### 2026-09-16 - 按作者选择保留 Figure 4 三个样例

- 代码：更新 `paper/iclr2027/pixelopsd_new/Pixel-OPSD-final/make_qualitative_figure.py`、`main.tex`、`README.md`，重新生成 `figures/qualitative_comparison.pdf`。
- 行为：Figure 4 仅保留原六行中的第三、第五、第六行（manifest 索引 `1525、1947、2050`），删除其余三个样例；三模型列、caption、IoU、固定宽高比画布和高分辨率图像渲染保持不变。
- 验证：运行 `python3 -m py_compile make_qualitative_figure.py`，核对生成图仅含三行并渲染检查无裁切或重叠；运行 `./build_pdf.sh` 成功生成 14 页 PDF，并重新打包 Overleaf 项目。

### 2026-09-16 - 按三行布局压缩 Figure 4 空白

- 代码：更新 `paper/iclr2027/pixelopsd_new/Pixel-OPSD-final/make_qualitative_figure.py`，重新生成 `figures/qualitative_comparison.pdf`。
- 行为：Figure 4 总高度根据保留的三行样例自适应，减少六行模板遗留的垂直空白并放大有效图格；样例索引、等宽面板、0.65pt 边界和高分辨率图像保持不变。
- 验证：重新渲染 Figure 4 检查三行间距、caption、mask 和 IoU 标注；运行 `./build_pdf.sh` 成功生成 14 页 PDF，并更新 Overleaf 项目包。

### 2026-09-16 - 将 Figure 4 调整为横向紧凑布局

- 代码：更新 `paper/iclr2027/pixelopsd_new/Pixel-OPSD-final/make_qualitative_figure.py`，重新生成 `figures/qualitative_comparison.pdf`。
- 行为：缩短三行 Figure 4 的总高度，标题行高度改为 `0.18`、行间距改为 `0.03`，并将 PDF 外边距设为 `0.01in`；样例行紧密排列，图像上下不再保留额外空白。
- 验证：重新渲染 Figure 4 检查横向比例、行间距、上下边界、caption 和等宽图格；运行 `./build_pdf.sh` 成功生成 14 页 PDF，并更新 Overleaf 项目包。

### 2026-09-16 - 移除 Figure 4 图像填充框并铺满面板

- 代码：更新 `paper/iclr2027/pixelopsd_new/Pixel-OPSD-final/make_qualitative_figure.py`、`README.md`，重新生成 `figures/qualitative_comparison.pdf`。
- 行为：删除固定画布的白色 padding，不再为不同原图比例补边；原图和对应 mask 直接铺满各自等宽面板，并通过 `aspect="auto"` 拉伸到统一面板尺寸，去除当前第三行的填充框。
- 验证：重新渲染 Figure 4 检查三行无填充边框、原图与 mask 对齐、列宽一致且无裁切；运行 `./build_pdf.sh` 成功生成 14 页 PDF，并更新 Overleaf 项目包。

### 2026-09-16 - 恢复 Figure 4 标题与样例行间距

- 代码：更新 `paper/iclr2027/pixelopsd_new/Pixel-OPSD-final/make_qualitative_figure.py`，重新生成 `figures/qualitative_comparison.pdf`。
- 行为：恢复六行版本使用的标题行比例 `0.22` 与行间距 `0.14`，并按三行内容设置总高度；GT、SAMTok、CycleGRPO、Pixel-OPSD 标题与第一行之间，以及各样例行之间重新保留清晰间隔。无填充框和原图拉伸铺满面板保持不变。
- 验证：重新渲染 Figure 4 检查标题、三行样例和底部边界的间隔；运行 `./build_pdf.sh` 成功生成 14 页 PDF，并更新 Overleaf 项目包。

### 2026-09-16 - 缩短 Figure 4 子图高度并平衡 caption 排版

- 代码：更新 `paper/iclr2027/pixelopsd_new/Pixel-OPSD-final/make_qualitative_figure.py`，重新生成 `figures/qualitative_comparison.pdf`。
- 行为：保持图宽和四列宽度不变，将三行子图高度调整为当前约四分之三；caption 改为左侧列内居中，字体改为略粗的 `semibold`，使左右留白更均衡。
- 验证：重新渲染 Figure 4 检查子图高度、caption 左右间距、字体可读性和行间距；运行 `./build_pdf.sh` 成功生成 14 页 PDF，并更新 Overleaf 项目包。

### 2026-09-16 - 微调 Figure 4 caption 字重

- 代码：更新 `paper/iclr2027/pixelopsd_new/Pixel-OPSD-final/make_qualitative_figure.py`，将 caption 字重调整为 `500` 并恢复 `6.5pt` 字号。
- 行为：caption 保持居中和左右均衡留白，但仅比常规正文略粗，避免 `semibold` 在当前字体中显示过重；子图高度、宽度和间隔不变。
- 验证：重新生成 Figure 4 并检查 caption 字重与三行图面布局，随后运行 `./build_pdf.sh` 并更新 Overleaf 项目包。

### 2026-09-16 - 增加 40k/80k scaling 纯自监督 teacher 八卡入口

- 代码：新增 `tools/train_selfsupervised_40k_teacher_8gpu.sh` 与
  `tools/train_selfsupervised_80k_teacher_8gpu.sh`。
- 行为：两份入口均调用主 `qwen3vl_4b_refcoco10k_volcengine.sh`，固定单节点 8 GPU、
  rollout/global batch `128`、完整 1 epoch，并显式开启 OPSD pixel-IoU、routing、caption safety、
  frozen EMA teacher、teacher confidence 和 teacher analysis；关闭 direct GRPO、direct mask CE 与 DLC-QA。
  no-target 使用 `pixel_empty` 的纯二值 decoded-union 判定，`NO_TARGET_NONEMPTY_MASK_PENALTY=0.0`，
  正例空 mask/拒识惩罚为 `POSITIVE_EMPTY_MASK_PENALTY=1.0`。40k 入口读取
  `cyclegrpo_selfsupervised_40k.parquet`；80k 入口将该文件和 `direct_supervised_40k.parquet`
  原子合并成 80,000 行缓存 parquet 后训练。
- 验证：两份脚本通过 `bash -n`；使用当前环境 PyArrow 17 按 80k 入口的相同逻辑验证两源 schema 可用
  `promote_options="default"` 合并且行数为 80,000；静态核对 batch、teacher、pixel-empty 与辅助流开关，
  并确认当前四卡 GPU power hold 仍正常。未在当前占卡状态下启动完整训练，避免中断已有 GPU 保活。

### 2026-09-16 - 调整 scaling 纯自监督 checkpoint 保存频率

- 代码：修改 `tools/train_selfsupervised_40k_teacher_8gpu.sh` 与
  `tools/train_selfsupervised_80k_teacher_8gpu.sh`。
- 行为：两份入口的默认 `SAVE_FREQ` 从 25 改为 5；`SAVE_LIMIT=1`、1 epoch、batch 128 和其余
  teacher/pixel-empty 配置保持不变。
- 验证：两份脚本通过 `bash -n`；确认主 launcher 将 `SAVE_FREQ` 传给
  `trainer.save_freq`，并核对 `verl/trainer/main.py` 通过 `ray.init(...)` 自动创建本地 Ray。

### 2026-09-16 - 将 scaling 纯自监督入口改为显式 Ray head 启动

- 代码：修改 `tools/train_selfsupervised_40k_teacher_8gpu.sh` 与
  `tools/train_selfsupervised_80k_teacher_8gpu.sh`。
- 行为：40k/80k 入口分别使用项目环境的 `ray start --head`（默认端口 `29679/29680`，避开 Ray 默认 worker 端口区间 `10002--19999`）、注册 8 张 GPU，
  以 `MULTINODE_ENABLED=true` attach 主 launcher，并在训练结束或中断时执行 `ray stop --force`。
  主训练仍是单节点 8 卡，数据、batch 128、teacher 和 pixel-empty 配置不变。
- 验证：两份脚本通过 `bash -n` 和 `git diff --check`；核对主 launcher 的单节点 attach 校验需要
  `RAY_ADDRESS`、`NNODES=1`、`RAY_CLUSTER_EXPECTED_GPUS=8`，与新增环境变量一致；未启动完整训练，保持现有 GPU 占卡。

### 2026-09-17 - 增加 GRPO/OPSD token 监督诊断提取与绘图管线

- 代码：新增 `analysis/opsd_vs_grpo_diagnostic/extract_signals.py`、`plot_signals.py`、`README.md` 和 `caption.tex`。
- 行为：离线管线只接受共同 checkpoint 的固定 rollout，验证完整 GRPO group、causal token 对齐、response/route mask、有限 logits，并直接调用实际 `compute_policy_loss` 与 `chunked_weighted_jsd_loss` 通过 autograd 提取 sampled-token local logit gradients；OPSD 只分析 `on_policy_distill` 且 `R_Ci>=0.65` 分布教学分支，固定共同排序和各自全局归一化。缺少逐 token logits 时生成 blocked metadata，禁止随机或手工热力图。
- 文档：更新第 3.7 节和模块清单，明确当前日志/checkpoint 没有固定 rollout logits，二维面板在没有 before/after 参数更新时只能标为 local-logit-gradient fallback，不能宣称 actual model update。
- 验证：两个脚本通过 `python -m py_compile`；无输入运行提取器后正确写出 `metadata.json` 的 `blocked_missing_diagnostic_data`；`git diff --check` 通过；未启动训练、未覆盖 checkpoint。

### 2026-09-17 - 修正二维局部方向面板的配对方法信号

- 代码：更新 `analysis/opsd_vs_grpo_diagnostic/extract_signals.py` 与 `plot_signals.py`。
- 行为：二维 A/B 诊断组现在同时保存并绘制 GRPO、OPSD 两套 local-logit-gradient 散点和均值箭头，保持同一预注册分组；不把单一方法的梯度误标为两方法方向。
- 验证：重新通过两个脚本的 `py_compile`、无输入阻塞检查和 `git diff --check`；没有生成模拟数据或启动训练。

### 2026-09-17 - 用真实训练日志生成监督信号代理图

- 代码：新增 `analysis/opsd_vs_grpo_diagnostic/plot_logged_proxy.py`，更新该目录的 README、caption、metadata 和三种图形输出。
- 行为：由于仓库仍没有逐 token logits，新增图严格使用真实 `teacher_diagnoses.jsonl` 的 178 条 mid-route、六次 localization IoU，以及同一 run `experiment_log.jsonl` 的 178 个 `pg_loss`/`distill_jsd` 记录。GRPO 面板显示去均值后的 `R_Ci` 广播信号，OPSD 面板显示实际 IoU 残差乘实现中的 mid-route 权重，二维面板显示 logged loss 的累计路径；明确标注为 rollout/logged-loss proxy，不冒充 token autograd 或参数更新。
- 验证：运行 `plot_logged_proxy.py` 生成 `figure_opsd_vs_grpo.pdf/svg/png`（PNG 400 dpi）和 `signals.npz`；查看图像确认三联图可读，运行 `py_compile` 与 `git diff --check`；未启动训练、未修改 checkpoint。

### 2026-09-17 - 比较 12 种布局并压缩 logged update path

- 代码：更新 `analysis/opsd_vs_grpo_diagnostic/plot_logged_proxy.py`、README、metadata 和图注，新增 `variants/` 候选图目录。
- 行为：自动渲染 12 个独立候选（6/8/10/12/14/16 个阶段，均值或累计路径），最终选用 12 阶段累计均值、11 个箭头的紧凑版本；热力图与路径面板增加边距、统一字号和独立横向色条，减少文字挤压与路径交叉。图注补充 (a)/(b)/(c) 的含义、OPSD 相对 GRPO 的证据残差优势及“日志代理而非 token/参数更新”的边界。
- 验证：重新生成 12 个候选和正式 PDF/SVG/400 dpi PNG，完成视觉检查；`py_compile` 与 `git diff --check` 通过，未启动训练、未覆盖 checkpoint。

### 2026-09-17 - 增加独立动态 EGCA：Shapley coarse/fine credit 与 evidence-gated OPD

- 代码：新增 `verl/workers/opsd/egca.py`、`tests/test_egca.py`；修改
  `verl/workers/opsd/{config,__init__,mask_iou}.py`、`verl/workers/fsdp_workers.py`、
  `verl/trainer/ray_trainer.py`、`projects/rl/config.yaml`、`projects/rl/qwen3vl_4b_mt.sh` 和
  `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`。
- 行为：新增默认关闭的 `worker.opsd.egca`。在自监督 cycle localization 的合法 SAMTok
  depth-2 group 上，worker 用固定 reference code 构造 coarse/fine 两个合法 counterfactual，
  计算独立 Shapley credit，经 decoded target/reconstruction evidence gate 缩放后写回真实
  coarse/fine token positions；cycle advantage 之后按 warmup/ramp/actor coefficient 注入。
  EGCA 不修改 pixel-IoU、`R_Ci`、reward、rollout、direct grounding、direct mask CE、DLC-QA
  或 no-target。开启 `egca.opd_enabled` 时，同一 evidence gate 额外乘到 mid-route privileged
  OPD/JSD sample weight；旧 SECA 的静态 direct-CE/JSD 兼容路径仍由 `SECA_ENABLED` 独立控制。
  当前实现使用 decoded evidence gate，不包含额外可训练 evidence head。
- 文档：更新第 3.6 节、模块清单，明确 EGCA 与 SECA 的边界以及动态 credit 的 target/teacher
  训练期信息边界。
- 验证：`PYTHONPATH=. /volume/ybo/xyc/envs/cyclegrpo/bin/python3 tests/test_egca.py`、
  `tests/test_seca.py` 均通过；受影响 Python 文件 `py_compile`、两个主训练 shell `bash -n`
  和 `git diff --check` 通过；未启动 GPU/Ray 训练，未修改占卡进程。

### 2026-09-18 - 新增四卡 20k EGCA 训练入口并允许本地 attach 消融

- 代码：新增 `tools/train_selfsupervised_20k_egca_4gpu.sh`；修改
  `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 的无 judge attach GPU 数校验。
- 行为：新入口固定读取 20k raw cycle parquet，验证行数为 20,000，启动项目环境的本地 Ray head
  并使用 CUDA 0--3、batch 128、1 epoch；开启 OPSD/pixel-IoU、动态 EGCA、routing、EMA teacher、
  teacher confidence/analysis、pixel-empty 二值 no-target reward 和正例空 mask 惩罚，关闭 direct
  GRPO、direct mask CE 与 DLC-QA；每 5 step 保存且最多保留 2 个 checkpoint，成功后停止 Ray 并
  启动 CUDA 保活。主入口允许 `MULTINODE_ENABLED=true` 且无 judge 时使用 1--8 张训练卡，保留正式
  8 卡与 7+1 judge 的拓扑约束语义。
- 模块清单：加入四卡训练 shell；未新增 Python 模块。
- 验证：新脚本和主入口通过 `bash -n`，`git diff --check` 通过；PyArrow 校验训练 parquet 为
  20,000 行；尚未启动完整训练，启动前需释放当前四卡占卡 worker。

### 2026-09-18 - 修复显式项目 Python 下的 Conda 激活失败

- 代码：修改 `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`。
- 行为：当 wrapper 已提供 `PYTHON_BIN` 时不再重复执行 Conda activate，避免当前服务器外层
  `base` shell 的 Conda PATH 栈触发 `IndexError`；未提供显式 Python 的历史入口仍保留原有激活行为。
- 验证：以前台启动四卡 EGCA 脚本捕获到的唯一失败为 Conda 激活器错误；修复后重新通过
  `bash -n` 与 `git diff --check`，下一次启动将继续进入 Ray/数据校验阶段。

### 2026-09-18 - 导出四卡 wrapper 的 Python/Ray 路径

- 代码：修改 `tools/train_selfsupervised_20k_egca_4gpu.sh`。
- 行为：将 `PYTHON_BIN` 与 `RAY_BIN` 导出给子 shell，使主入口能够识别显式项目 Python 并跳过
  外层 Conda 激活；训练参数与数据配方不变。
- 验证：脚本通过 `bash -n`；前台启动日志确认此前 Conda 错误来自子 shell 未继承
  `PYTHON_BIN`，修复后可进入主入口检查。

### 2026-09-18 - 修复四卡 Ray head 地址不一致

- 代码：修改 `tools/train_selfsupervised_20k_egca_4gpu.sh`。
- 行为：四卡 wrapper 现在从 `hostname -I` 读取本机实际 Ray bind IP，同时传给
  `ray start --node-ip-address` 和 `RAY_ADDRESS`；不再假设 `127.0.0.1`，避免 Ray GCS 实际绑定
  节点地址而 trainer 连接回环地址导致 90 秒超时。`RAY_BIND_IP` 仍可由调用方显式覆盖。
- 验证：前次启动的 `ray_start.log` 确认实际节点地址为 `172.16.189.147` 且连接失败来自
  `127.0.0.1:29677`；新脚本通过 `bash -n` 和 `git diff --check`，将以同一实际地址重新启动。

### 2026-09-18 - 将四卡 EGCA no-target reward 固定为二值

- 代码：修改 `verl/workers/opsd/config.py`、`projects/rl/config.yaml`、
  `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`、`tools/train_selfsupervised_20k_egca_4gpu.sh`、
  `tools/train_selfsupervised_40k_teacher_8gpu.sh`、`tools/train_selfsupervised_80k_teacher_8gpu.sh` 和
  `tests/test_no_target_reward.py`。
- 行为：`pixel_empty` 实验的 `NO_TARGET_EMPTY_AREA_TAU` 统一为 `0.0`，并明确该模式只使用
  decoded union 的空/非空判定，输出为 `1.0`、`0.0`（或显式 nonempty penalty 时的 `-1.0`），不按前景面积比例连续打分。
  配置校验允许 tau 为零，但显式 `pixel_empty_iou` 仍必须提供正 tau，避免连续模式被误配置为二值模式。
- 验证：新增严格 binary reward 单测；受影响脚本通过 `bash -n`，配置与 reward 代码通过 Python
  编译/单测，`git diff --check` 通过。此前失败的 Ray 启动没有产生训练进程，四卡将按修正后的入口重新启动。

### 2026-09-19 - 防止 EGCA common-mode credit 诱导拒识偏置

- 代码：修改 `verl/workers/opsd/egca.py`、`verl/workers/opsd/config.py`、
  `verl/workers/opsd/__init__.py`、`verl/workers/fsdp_workers.py`、`verl/trainer/ray_trainer.py`、
  `projects/rl/config.yaml`、`projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`、
  `projects/rl/qwen3vl_4b_mt.sh`、`tools/train_selfsupervised_20k_egca_4gpu.sh` 和
  `tests/test_egca.py`。
- 行为：EGCA 默认新增 `credit_mode=contrastive`。worker 显式记录每个实际 coarse/fine
  code position；trainer 在注入 actor advantage 前对每条 response 的这些位置做零均值投影，
  只保留 coarse/fine 相对 Shapley 差异，不重复计入已经由 GRPO 携带的完整 mask 轨迹收益。
  `credit_mode=raw` 保留旧行为用于历史复现；no-target/supervised/padding 仍不产生 EGCA credit。
  该修复不改变 pixel-empty 的二值 reward、`R_Ci`、OPD rollout 或 inference 路径。
- 分析：`cyclegrpo20k_egca_pixel_empty_4gpu` 的 EGCA run 与旧
  `cyclegrpo20k_pixel_empty_noncycle_positivepenalty1_bs128_response256` 并非单变量对照；后者
  额外启用了 no-target segmentation actor（日志含 `main_no_target_segmentation_*`），而当前
  EGCA 主路径是 caption-only pixel-empty 拒识。除此之外，旧实现把有符号 Shapley 的 common
  component 直接加到 mask token advantage；当该 component 偏负时会通过共享 actor 隐式压低
  正例 mask 生成，最终表现为 no-target 偏置。contrastive 投影消除这一非预期轨迹级通道。
- 验证：运行 `PYTHONPATH=. /volume/ybo/xyc/envs/cyclegrpo/bin/python3 -m unittest tests.test_egca`、
  `python -m py_compile verl/workers/opsd/egca.py verl/workers/opsd/config.py verl/workers/fsdp_workers.py verl/trainer/ray_trainer.py`、
  两个主训练入口的 `bash -n` 和 `git diff --check`；额外用 `DataProto` 做 trainer advantage 注入
  smoke test，确认 contrastive credit 每条 response 的和为零；未启动训练或停止现有 GPU 占卡进程。

### 2026-09-20 - 将 EGCA 默认更新改为独立的非负加权自蒸馏

- 代码：修改 `verl/workers/opsd/egca.py`、`verl/workers/opsd/config.py`、
  `verl/workers/opsd/__init__.py`、`verl/workers/fsdp_workers.py`、`verl/trainer/ray_trainer.py`、
  `projects/rl/config.yaml`、`projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`、
  `projects/rl/qwen3vl_4b_mt.sh` 和 `tests/test_egca.py`；新增
  `tools/train_selfsupervised_20k_egca_weighted_ce_8gpu.sh`、
  `tools/train_selfsupervised_20k_egca_weighted_ce_4gpu_tmp.sh`。
- 行为：新增 `egca.update_mode=weighted_ce`（默认）。cycle localization 仍计算 IoU、decoded
  evidence 与 coarse/fine Shapley，但不再把 signed credit 写入 GRPO advantage；trainer 按
  `sample_uid` 聚合 6 个 rollout，从同一 self-supervised 行的 GT SAMTok 序列构造独立
  teacher-forcing batch。evidence 生成非负 sample weight，正向 coarse/fine 贡献只增加完整 mask
  group 的 CE token weight，负向 credit 被截断为零；`gres_no_target`、supervised source、非法
  group 和无 mask rollout 不进入该 batch。OPD/JSD 仍可使用 evidence gate。`legacy_advantage`
  模式保留旧 signed-credit 行为用于历史 ablation。
- reference：新模式默认 `egca.reference_mode=target`，按每条 GT mask 的 coarse/fine code
  构造 counterfactual；`fixed` 仅用于复现固定 code-0 的历史消融。
- 训练入口：新增 8 卡 20k 自监督脚本，默认 batch 128、1 epoch、Ray head、pixel-empty 二值
  no-target reward、正样本空 mask penalty、teacher/OPD 开启、direct/DLC-QA 关闭；临时 4 卡
  wrapper 仅覆盖 CUDA 0--3、`MAX_STEPS=1`，并关闭训练结束后的 keepalive，便于本机 smoke。
- 验证：`tests.test_egca`（9 tests）、受影响 Python 文件 `py_compile`、四个 shell `bash -n`、
  `git diff --check` 均通过；尚未完成临时 4 卡 Ray/FSDP/vLLM step 验证，需在测试前暂时停止当前
  GPU power-hold worker，并在 smoke 结束后恢复占卡。

### 2026-09-20 - 完成 EGCA 加权自蒸馏四卡 smoke 并恢复占卡

- 验证：在停止占卡后使用临时四卡 wrapper 完成了完整的 Ray/FSDP/vLLM 单 step 训练，日志为
  `logs/cyclegrpo20k_egca_weighted_ce_4gpu_smoke2/train_20260921_110559.log`，并成功写出
  `checkpoints/global_step_1`；日志确认 `opsd/egca_weighted_self_distill_samples=120`、
  `opsd/egca_actor_advantage_injection=0.0` 以及加权 CE loss 均正常记录。随后针对 target-reference
  改动启动的第三次 smoke 已按用户要求主动停止，不将其误报为完成训练。
- 当前状态：已停止该 smoke 的 Ray/trainer/worker 进程，并重新启动 CUDA 0--3 的 GPU power-hold；
  四张卡各占用约 30,864 MiB、GPU 利用率约 100%，占卡 worker 持续运行。

### 2026-09-21 - 增加 SECA 纯自监督 20k 入口并完成四卡验证

- 代码：修改 `verl/workers/opsd/config.py`、`verl/workers/opsd/seca.py`、
  `verl/workers/opsd/__init__.py`、`verl/trainer/ray_trainer.py`、
  `projects/rl/config.yaml`、`projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`、
  `projects/rl/qwen3vl_4b_mt.sh` 和 `tests/test_seca.py`；新增
  `tools/train_selfsupervised_20k_seca_8gpu.sh`、
  `tools/train_selfsupervised_20k_seca_4gpu_tmp.sh`。
- 行为：新增 `worker.opsd.seca.self_supervised_enabled` /
  `SECA_SELF_SUPERVISED_ENABLED`。20k 自监督脚本使用 20k raw CycleGRPO parquet、8 卡 Ray
  head、batch 128、1 epoch、OPSD/pixel-IoU、routing/EMA teacher，并在 mid-route privileged
  JSD 上启用 detached SECA evidence gate；EGCA、direct GRPO、direct mask CE 和 DLC-QA 关闭。
  pixel-empty no-target 使用上一版的二值空 union reward 和专用 segmentation actor，
  `no_target_segmentation_loss_weight=1.0`；当前基线不再定义正样本空 mask 或 no-target 非空 penalty 字段。
- 验证：`tests.test_seca` 4 tests、受影响 Python `py_compile`、新旧入口 `bash -n` 与
  `git diff --check` 通过。停止占卡后，临时四卡脚本成功完成 1 step，并生成
  `logs/cyclegrpo20k_seca_selfsupervised_pixel_empty_4gpu_smoke/checkpoints/global_step_1`；
  `experiment_log.jsonl` 记录 `opsd/seca_evidence_weight_mean=0.828068`、
  `opsd/seca_evidence_active_count=151`、`distill_opsd/distill_jsd=0.038655`，无错误/traceback。
  训练结束后已恢复 CUDA 0--3 占卡，四卡各约 30,864 MiB、利用率约 100%。

### 2026-09-21 - 对齐 SECA 自监督入口与 20k noncycle 基准配置

- 代码：修改 `tools/train_selfsupervised_20k_seca_8gpu.sh`。
- 行为：除 `worker.opsd.seca.*` 外，入口继续沿用
  `logs/cyclegrpo20k_pixel_empty_noncycle_positivepenalty1_bs128_response256/checkpoints/experiment_config.json`
  的 20k 配置；显式固定 validation 使用同一 parquet、caption/localization rollout 均为 6，
  默认最大训练步数从空值改为 `156`，与该基准的 1 epoch、batch 128 以及
  actor/critic `training_steps=156` 一致。临时四卡 wrapper 仍可显式覆盖为 `MAX_STEPS=1`，
  因而不改变已完成的 smoke 验证；SECA 仍是唯一有意开启的额外模块。
- 验证：`bash -n tools/train_selfsupervised_20k_seca_8gpu.sh`
  与 `git diff --check` 通过；未启动新的训练，当前正在运行的评测进程未被中断。

### 2026-09-21 - 修正 SECA 入口的历史 no-target 配置对齐

- 代码：修改 `tools/train_selfsupervised_20k_seca_8gpu.sh`；更新本文件对应的 SECA 配置说明。
- 行为：历史参考 `experiment_config.json` 的 `worker.opsd.no_target_segmentation_loss_weight=1.0`
  属于已废弃的主 no-target segmentation actor 路径；当前有效代码已在 2026-09-01 恢复为
  caption-only non-cycle，因此该字段不再被当前入口读取。SECA 入口现在明确将当前代码仍支持的
  `positive_empty_mask_penalty` 与 `no_target_nonempty_mask_penalty` 都设为 `0.0`，不再误加任何
  penalty；`pixel_empty` 仍是严格二值空 union + `No target.` 拒识判定。SECA 参数是唯一有意新增的训练行为。
- 验证：`bash -n tools/train_selfsupervised_20k_seca_8gpu.sh`、
  `PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile verl/workers/opsd/config.py` 与
  `git diff --check`；未重启当前评测进程或训练。

### 2026-09-21 - 将主代码基线恢复到上一版实验并保留 EGCA/SECA

- 代码：依据 Git 快照 `06bf2bf`（`add strict text check in pixel empty reward and add no target loss weight`）
  对齐 `verl/trainer/ray_trainer.py`、`verl/workers/opsd/config.py`、`verl/workers/opsd/mask_iou.py`、
  `verl/workers/fsdp_workers.py`、`verl/workers/actor/dp_actor.py`、`verl/workers/opsd/__init__.py`、
  `verl/workers/supervised_anchors.py`、`verl/workers/config.py`、`verl/workers/reward/function.py`、
  `projects/rl/config.yaml`、`projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 及相关测试；保留
  `verl/workers/opsd/egca.py`、`verl/workers/opsd/seca.py` 及其 trainer/actor 接口。
- 行为：恢复主 `pixel_empty` `gres_no_target` 的专用 segmentation actor 和
  `worker.opsd.no_target_segmentation_loss_weight`，默认权重为 `1.0`；恢复严格二值规则（精确
  `No target.` 且 decoded union 为空才得 `1.0`，否则 `0.0`）。移除后来加入的
  `positive_empty_mask_penalty`、`no_target_nonempty_mask_penalty`、面积 tau/连续 IoU 配置及其
  metadata；这些不再进入当前基线。EGCA/SECA 仍可通过各自开关独立启用，默认关闭。
- 验证：`bash -n` 通过主训练入口和 SECA 入口；相关 Python 文件 `py_compile` 通过；
  `PYTHONPATH=. /volume/ybo/xyc/envs/cyclegrpo/bin/python3 -m unittest
  tests.test_opsd_core tests.test_no_target_reward tests.test_supervised_anchors tests.test_seca`
  共 52 项通过；`git diff --check` 通过。未停止正在运行的 GRES 评测进程。

### 2026-09-21 - 固定 SECA 入口的历史 no-target actor 权重

- 代码：修改 `tools/train_selfsupervised_20k_seca_8gpu.sh`。
- 行为：入口显式导出 `NO_TARGET_SEGMENTATION_LOSS_WEIGHT=1.0`，与上一版
  `experiment_config.json` 完全一致，避免调用环境中的同名变量改变 no-target segmentation actor 的梯度比例；
  未恢复任何已移除的 penalty/tau 字段。
- 验证：`bash -n tools/train_selfsupervised_20k_seca_8gpu.sh` 与 `git diff --check` 通过。

### 2026-09-21 - 将 SECA 自监督入口其余开关固定为历史基线

- 代码：修改 `tools/train_selfsupervised_20k_seca_8gpu.sh`。
- 行为：入口现在显式固定历史实验的保存频率/保留数（`5/2`）、direct grounding/CE、DLC-QA、
  direct batch、rollout、warmup、loss weight、source flags、梯度诊断和三流模式；这些非 SECA 设置不再
 受外部环境变量意外污染。`MAX_STEPS` 仍保留 `1`-step smoke 覆盖能力，正式默认仍为 `156`。
- 验证：脚本静态检查通过；历史配置关键值仍为 `pixel_empty`、no-target actor weight `1.0`、
  batch `128`、rollout `6`、teacher decay `1.0`、anchor KL `0.05/0.05`、保存 `5/2`、总步数 `156`。

### 2026-09-21 - 修正 SECA 四卡临时 smoke 的本地 Ray 模式

- 代码：修改 `tools/train_selfsupervised_20k_seca_4gpu_tmp.sh`。
- 行为：四卡临时 wrapper 现在显式设置 `NNODES=1`、`MULTINODE_ENABLED=false`，避免继承八卡正式入口的
  多节点校验；训练仍固定 CUDA 0--3、`MAX_STEPS=1`，正式八卡脚本不受影响。
- 验证：待本次四卡 smoke 完成后记录实际 Ray/FSDP/vLLM 结果。

### 2026-09-21 - 允许 SECA 八卡入口被四卡 smoke 显式切换为本地 Ray

- 代码：修改 `tools/train_selfsupervised_20k_seca_8gpu.sh`。
- 行为：正式入口默认仍为 `MULTINODE_ENABLED=true` 并启动 8-GPU Ray head；当临时 wrapper 显式传入
  `MULTINODE_ENABLED=false` 时，不启动外部 Ray head，由主 launcher 创建本地单节点 Ray，从而支持 CUDA 0--3
  的 1-step smoke，不改变正式八卡默认路径。
- 验证：已通过 shell 语法检查；四卡 smoke 正在重新启动，结果以独立日志记录。

### 2026-09-21 - 修正 SECA 四卡 smoke 的 Conda 环境继承

- 代码：修改 `tools/train_selfsupervised_20k_seca_4gpu_tmp.sh`。
- 行为：临时四卡 wrapper 显式导出项目 `ENV_DIR` 对应的 `CONDA_PREFIX`。由于 `PYTHON_BIN`/`RAY_BIN` 已显式指向项目环境，主 launcher 会跳过外层 base Conda 的重复激活，规避 Conda 激活器 `IndexError`；正式八卡入口默认行为不变。
- 验证：四卡本机 smoke 完成模型初始化、vLLM rollout、mask decode、log-prob 并进入 policy update，检查点目录已创建；随后按要求停止训练进程。日志中的 `KeyboardInterrupt` 仅来自人工停止，不是训练异常。

### 2026-09-21 - 固定四卡 smoke wrapper 的项目环境路径

- 代码：修改 `tools/train_selfsupervised_20k_seca_4gpu_tmp.sh`。
- 行为：wrapper 现在先解析 `ENV_DIR=${BASE_DIR}/envs/cyclegrpo` 默认值，再导出 `CONDA_PREFIX`；从任意外层 shell 启动时都能稳定跳过 base Conda 的重复激活，不依赖调用环境是否预先设置 `ENV_DIR`。
- 验证：`bash -n tools/train_selfsupervised_20k_seca_4gpu_tmp.sh tools/train_selfsupervised_20k_seca_8gpu.sh` 与 `git diff --check` 通过；本轮 smoke 已完成完整初始化、rollout、mask decode 和 log-prob 阶段并进入 policy update，之后按要求停止。

### 2026-09-21 - 固定正式 SECA 入口的 Conda 环境继承

- 代码：修改 `tools/train_selfsupervised_20k_seca_8gpu.sh`。
- 行为：正式八卡入口在调用主 Qwen launcher 前显式导出 `CONDA_PREFIX=${ENV_DIR}`，因此从 base shell 或任意外层 shell 启动时均不会触发错误的重复 `conda activate`；训练 Python/Ray 仍使用项目环境绝对路径。
- 验证：`bash -n` 与 `git diff --check` 通过；四卡临时入口已完成模型初始化、rollout、mask decode、log-prob 并进入 policy update，之后按要求停止。

### 2026-09-21 - 固定四卡 EGCA 入口的 Conda 环境继承

- 代码：修改 `tools/train_selfsupervised_20k_egca_4gpu.sh`。
- 行为：四卡 EGCA 正式入口在调用主 launcher 前显式导出 `CONDA_PREFIX=${ENV_DIR}`，避免从外层 base shell 启动时重复激活项目 Conda 环境；Ray/Python 仍使用项目环境绝对路径，训练配置不变。
- 验证：`bash -n tools/train_selfsupervised_20k_egca_4gpu.sh` 与 `git diff --check` 通过。

### 2026-09-21 - 修正四卡 EGCA 本地 Ray 拓扑

- 代码：修改 `tools/train_selfsupervised_20k_egca_4gpu.sh`。
- 行为：四卡入口默认使用 `MULTINODE_ENABLED=false`，不再把 4-GPU 本地运行误报为需要 8-GPU 的多节点集群；主 launcher 会创建匹配项目环境的本地 Ray 单节点。仅显式设置为 `true` 时才启动并连接入口自建 Ray head。训练数据、历史基线超参数、EGCA 与 no-target 配置不变。
- 验证：脚本通过 `bash -n`；此前失败原因为 launcher 的 4-GPU 多节点拓扑校验，已由该路径修正。

### 2026-09-21 - 重新生成 GRPO-only 与 OPSD 双路径监督诊断图

- 代码：新增 `analysis/opsd_vs_grpo_diagnostic/plot_dual_proxy.py`；更新该目录的 `README.md`、`caption.tex`、`figure_opsd_vs_grpo.{pdf,svg,png}`、`signals.npz`、`metadata.json`，并新增 `variants_dual/` 的 12 个候选布局。
- 行为：读取真实 GRPO-only 714-step 日志和 OPSD 178-step 日志；两套热力图均按 12 个时间阶段聚合，GRPO 显示广播的阶段标量，OPSD 保留每条 diagnosis 的六个真实 IoU；二维面板显示两条独立的 `-pg_loss`/pixel-IoU 累积路径。最终选择每种方法六个箭头的单一版本，不平均候选图，也不把结果宣称为共同 checkpoint 的 token-autograd 或参数更新因果比较。
- 验证：`python -m py_compile analysis/opsd_vs_grpo_diagnostic/plot_dual_proxy.py` 和绘图脚本运行通过；确认正式 PDF/SVG/400 dpi PNG、12 个候选图和 `metadata.json` 生成；`pdfinfo` 显示页面为 `911.424 x 299.415 pt`，`git diff --check` 通过；未启动训练、未覆盖 checkpoint。

### 2026-09-21 - 将双图 OPSD 热力图改为真实组内证据残差

- 代码：更新 `analysis/opsd_vs_grpo_diagnostic/plot_dual_proxy.py`、`README.md`、`caption.tex`、`metadata.json` 及正式三种图形输出。
- 行为：OPSD 面板对每条 teacher diagnosis 的六个真实 `pixel_ious` 减去该行六次 rollout 均值，再按 12 个阶段聚合；GRPO 面板仍为广播到六列的阶段标量。该中心化只改变显示基准，保留真实组内正负证据差异，不做逐行对比度拉伸。
- 验证：重新运行 dual plot 脚本，检查 `signals.npz` 有限值、正式 PDF/SVG/400 dpi PNG、12 个候选图和 metadata；`py_compile`、`pdfinfo` 与 `git diff --check` 通过，未启动训练、未覆盖 checkpoint。
### 2026-09-21 - 新增真实 token 更新二维/三维诊断图

- 代码：新增 `analysis/token_update_scatter/extract_token_updates.py`、`estimate_token_contribution.py`、`plot_token_updates.py`、`implementation_audit.md`、`config.yaml`、`caption.tex`、`README_zh.md`、`token_data.npz`、`intervention_rollouts.jsonl`、`metadata.json` 及 `token_update_{2d,3d}.{pdf,svg,png}`。
- 文档：更新第 3.7 节与第 5.2 节模块清单，明确该图使用 teacher-alignment fallback、真实 checkpoint `Δlog p`、SAMTok mask-code token、bounded sequence-reliability 定义，以及历史 GRPO/Pixel-OPSD run 使用不同 parquet、不能作 paired objective/causal ablation。
- 行为：从固定 seed=20260921 的 16 条 disjoint direct RefCOCO 行抽取 32 个合法 depth-2 mask-code token；对 base、GRPO、Pixel-OPSD 和独立 direct-supervision checkpoint 进行真实多模态 teacher-forced 前向，二维面板两侧共享点和坐标范围，三维面板复用相同数据。未启动重训练、未覆盖 checkpoint、未用 preview 数组。
- 验证：`python -m py_compile analysis/token_update_scatter/*.py`、`git diff --check`；GPU 离线 scoring 完成 4 个 checkpoint、16 行/32 token 且 finite coverage=1.0；绘图脚本成功生成 PDF/SVG/300 dpi PNG，`pdfinfo` 确认二维页面 `719.322 x 304.275 pt`、三维页面 `703.775 x 350.689 pt`。


### 2026-09-21 - 生成增强双热力图并建立配对更新方向严格入口

- 代码：新增 `analysis/opsd_vs_grpo_diagnostic/plot_ab_enhanced.py`、`plot_paired_update_direction.py`、`caption_ab.tex`、`caption_paired_update_direction.tex`、`metadata_ab.json`；生成 `figure_opsd_vs_grpo_ab.{pdf,svg,png}`；更新 README、模块清单和第 3.7 节。
- 行为：双热力图使用 16 个离散发散色阶、统一对称 99% 色标和简短 `shared credit`/`evidence-resolved` 标注。配对更新脚本要求共同 checkpoint/fixed rollout 的实测 `delta_logp`，当前因仓库缺失 sampled rollout/log-prob 只写 `paired_update_metadata.json` 的 blocked 状态，不生成模拟第二张图。
- 验证：两份脚本 `py_compile` 通过；增强 PNG 为 4142×1515、400 dpi；PDF/SVG/PNG 和 caption/metadata 均生成；配对脚本正确报告 `blocked_missing_common_rollout`；`git diff --check` 通过，未启动训练、未覆盖 checkpoint。

### 2026-09-21 - 新增独立 70k 历史基线与 SECA 有监督入口

- 代码：新增 `tools/train_supervised_70k_common_8gpu.sh`、
  `tools/train_supervised_70k_baseline_8gpu.sh` 和
  `tools/train_supervised_70k_seca_8gpu.sh`；更新第 2.2 节和第 5.1 节模块清单。
- 行为：两个公开 wrapper 均使用 20k raw CycleGRPO、30k RefCOCO positive direct、10k
  gRefCOCO no-target direct 和 10k DLC-QA 的历史 70k 配方，除新增监督流与 SECA 开关外对齐
  指定历史配置，固定 main/direct/QA batch `128/256/64`、156 step、anchor KL
  `0.05/0.05`、direct GRPO、direct mask CE、DLC-QA、pixel-empty 和 7+1
  Ray/Llama 拓扑。baseline 关闭 SECA；SECA wrapper 同时开启 cycle-only evidence gate 与
  direct mask CE token credit。共享入口自行启动项目环境 Ray head 与 GPU 7 的本地
  Llama-3.1-8B judge，退出时仅按独立 Ray session 清理，避免误杀其他本机 Ray 任务，随后
  请求八卡 GPU hold。四份 parquet 和 DLC-QA JSONL 在启动前逐一校验行数。
- 预检：共享入口支持 `DRY_RUN=true`，用于在不占用 GPU 的情况下完成上述路径、行数和配置检查。
- 验证：三个正式/共享脚本均通过 `bash -n`，`git diff --check` 通过；使用项目 Python 读取实际服务器
  数据，确认行数为 `20000/30000/10000/10000`，DLC-QA JSONL 为 10,000 行；未启动新的
  70k 训练，以避免中断当前四卡 EGCA 训练。

### 2026-09-21 - 对齐 70k 有监督入口并增加四卡 smoke wrapper

- 代码：修改 `tools/train_supervised_70k_common_8gpu.sh`；新增
  `tools/train_supervised_70k_baseline_4gpu_tmp.sh` 和
  `tools/train_supervised_70k_seca_4gpu_tmp.sh`；更新本节、模块清单和 70k 配置说明。
- 行为：正式 70k 两个入口除 direct/DLC-QA/SECA 相关字段外对齐指定
  `experiment_config.json` 的 `128/128` 主 batch、`256/64` 辅助 batch、156 steps、
  `caption_anchor_kl_coef=0.05`、`segmentation_anchor_kl_coef=0.05` 及其余历史开关。
  临时 wrapper 将拓扑改为 3 张训练卡（CUDA 0--2）+1 张 judge（CUDA 3），将三流 batch
  调整为可被 3 整除且保持 2:4:1 的 `114/228/57`，并限制为 1 step；生产脚本不受该
  smoke 覆盖影响。
- 验证：四个 shell 脚本 `bash -n` 与 `git diff --check` 通过；baseline 与 SECA 临时入口均在本机
  CUDA 0--2 训练、CUDA 3 Llama judge 下启动成功，均完成数据计数、Ray/FSDP/vLLM 初始化并进入
  rollout（约两分钟内无错误）；按要求未等待完整 step，随后停止并清理各自 Ray/judge。

### 2026-09-21 - 支持四卡本地监督 smoke 的动态 judge 拓扑

- 代码：修改 `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh` 与
  `tools/train_supervised_70k_common_8gpu.sh`。
- 行为：主 launcher 新增 `LOCAL_JUDGE_TRAIN_GPUS`，正式 7+1 入口默认仍要求 7 张 Ray 训练卡；
  四卡临时 wrapper 导出值 3，从而允许 3 张 Ray/FSDP 训练卡与第 4 张物理卡上的 Llama judge，
  不放宽正式 8 卡拓扑。共享入口的 Ray/judge 日志改为使用实际变量，避免 smoke 日志误报为 7 卡。
- 验证：baseline 和 SECA 临时脚本分别启动并通过 20k/30k/10k/10k 数据校验、Llama 健康检查、
  Ray 资源校验、FSDP 模型初始化和 vLLM rollout 预热；两次均在约两分钟无异常后停止。执行
  `bash -n`、`git diff --check`；后续恢复 CUDA 0--3 GPU hold。

### 2026-09-21 - 新增全样本 OPSD routing 消融并完成四卡验证

- 代码：修改 `verl/workers/opsd/config.py`、`verl/workers/fsdp_workers.py`、
  `verl/trainer/ray_trainer.py`、`projects/rl/config.yaml` 和
  `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`；新增
  `tools/train_selfsupervised_20k_opsd_all_samples_4gpu.sh`。
- 行为：新增 `worker.opsd.routing.all_samples_opsd`/`OPSD_ALL_SAMPLES_OPD` 开关。开启后不再按
  `R_Ci` 执行 regenerate/on-policy-distill/GRPO 三分类，所有 eligible image-cycle 样本统一使用
  privileged on-policy OPSD correction；`PRESERVE_ORIGINAL_GRPO=true` 仍保留所有安全 caption 的
  原始 CycleGRPO GRPO。该模式同时跳过原有 mid-route teacher-confidence 的 `R_Ci>=0.65` 过滤，
  使 OPSD 修正真正覆盖全量样本；no-target pixel-empty segmentation 分支保持独立不变。
- 验证：执行 shell/Python 语法检查与 `git diff --check`。本机四卡 1-step smoke 成功写入
  `logs/cyclegrpo20k_opsd_all_samples_routing_4gpu_smoke/checkpoints/experiment_log.jsonl`：
  `route_on_policy_distill_count=732`、`route_regenerate_count=0`、`route_grpo_count=0`、
  `distillation_confident_count=732`、`caption_original_grpo_active_rate=1.0`，并保存
  `global_step_1`。随后正式 20k/156-step 四卡训练已启动，日志为
  `logs/cyclegrpo20k_opsd_all_samples_routing_4gpu/train_20260921_202812.log`，当前 Ray/FSDP
  进程正常运行，训练结束后入口自动启动四卡 CUDA keepalive。

### 2026-09-21 - 修正 70k 正式 7+1 拓扑的 batch 整除约束

- 代码：修改 `tools/train_supervised_70k_common_8gpu.sh`；更新第 2.2、3.2、5.1 和关键注意事项中的正式 70k 配置说明。
- 行为：此前正式入口使用 `128/256/64`，但 Ray 训练 world size 为 7（GPU 0--6，GPU 7 留给 Llama judge），首步在 `DataProto.chunk(7)` 因 128 不能整除 7 而失败。现在默认改为 `112/224/56`，保持三流 `2:4:1` parent-prompt 比例；`MAX_STEPS` 改为 `179`，使 20k/40k/10k loader 近似各消费一轮。四卡 smoke 的 `114/228/57` 覆盖保持不变。
- 验证：检查两个失败的 `run.log` 均为 `Got size of DataProto 128 and chunk 7`；执行 `bash -n tools/train_supervised_70k_common_8gpu.sh tools/train_supervised_70k_baseline_8gpu.sh tools/train_supervised_70k_seca_8gpu.sh`、`DRY_RUN=true` 两个正式 wrapper 的数据/路径预检和 `git diff --check`。未在本轮自动重启正式 70k 训练。

### 2026-09-23 - 固化通过全量 RefCOCO 验证的正例 prompt

- 代码：修改 `evaluation/refcoco/qwen3vl_refcoco_eval.py` 和
  `evaluation/refcoco/run_refcoco_multigpu.sh`；同步更新第 5.6 节评测协议。
- 行为：RefCOCO evaluator 与多 GPU launcher 新增 `--prompt_template` / `PROMPT_TEMPLATE`，模板必须包含
  `{phrase}`；逐样本结果保存 prompt 模板，resume 要求 prompt 与 `mask_protocol` 均一致。以“表达必然有目标，
  禁止拒识/空回答，只输出一个合法 mask group”的正例 prompt 作为默认值，旧的短 prompt 仍可显式覆盖；
  mask decode、EOS、指标和训练 prompt 不变。
- 验证：`py_compile`、`bash -n`、`git diff --check` 通过。step-156 checkpoint 完成四卡 RefCOCO val
  10,834/10,834 推理；新 prompt cIoU/mIoU=`74.0443/76.0541`，旧 prompt=`29.4460/22.1497`。
  全量审计显示 10,723 条解出 mask group、11 条 literal `No target`、0 条 malformed group，10,834 条
  均具有匹配 prompt metadata。推理 shard 完成后立刻对空闲 GPU 补上保活。

### 2026-09-24 - 新增学生展示用 RefCOCO 与三能力脚本

- 代码：新增 `evaluation/refcoco/demo_compare_1000.py`、`evaluation/refcoco/demo_three_capabilities.py`、`tools/demo_refcoco_compare_1000.sh`、`tools/demo_three_capabilities.sh`。
- 文档：更新第 5.6 节 RefCOCO 入口说明和演示协议。
- 行为：首脚本用 seed 固定抽样，同一 1,000 条 RefCOCO val 上顺序比较 SAMTok/Ours，采用普通表达分割 prompt 与 `legacy_union`，输出真实 cIoU/mIoU、百分比进度、最佳逐样本 IoU 增益可视化；83.4/84.6 作为单独的教学视频展示参考值，不替代本次真实计算结果。次脚本从首脚本 Ours 预测中挑 IoU 最高样本，使用预测 mask token 做 mask-captioning、以原图做传统 VQA，并保存 mask 与三能力合成图。两个 shell 启动器只停止自身 GPU hold worker、推理时在 GPU 1--3 保持占用，并在退出后恢复 CUDA 0--3 hold。
- 验证：两份 Python 文件 AST 解析、从 `/tmp` 按绝对路径运行 `--help`、两份 shell 的 `bash -n` 与 `git diff --check` 通过。CUDA 0--3 正在运行其他四卡推理（约 26k/26.36k），未并行加载演示模型；完整 GPU 端到端 smoke 尚未执行。

### 2026-09-24 - 精简演示启动脚本的 GPU 状态输出

- 代码：修改 `tools/demo_refcoco_compare.sh`、`tools/demo_three_capabilities.sh`。
- 文档：更新第 5.1 节工具 inventory。
- 行为：GPU hold 的 stop/start 操作保持不变，但其状态表和 `nvidia-smi` 明细不再打印；占卡冲突时只显示一条通用提示，推理执行与输出流程不变。
- 验证：`bash -n` 和 `git diff --check` 通过；未启动推理。

### 2026-09-24 - 将三能力合成图拆为独立大字 PNG

- 代码：修改 `evaluation/refcoco/demo_three_capabilities.py`。
- 文档：更新第 5.6 节三能力演示产物说明。
- 行为：原单张三列合成图替换为三张独立可视化：表达分割（原图/GT/Ours mask 与 IoU）、预测 mask captioning（高亮 mask 与生成 caption）、传统 VQA（原图、问题和回答）。统一增大标题与正文，并按内容换行；模型推理、样本选择和 JSON/mask 保存不变。
- 验证：Python AST 解析、shell `bash -n` 和 `git diff --check` 通过；用合成输入 GPU-free 生成三张 PNG 并查看布局，尺寸为 `1560x712`、`1500x1060`、`1500x1144`，标题/说明未重叠；未启动模型推理。

### 2026-09-24 - 增加 Pixel-OPSD 三任务录屏工作区

- 代码：新增 `evaluation/refcoco/record_three_tasks.py`、`tools/record_pixel_opsd_demo.sh`；新增录屏输入资产 `logs/pixel_opsd_recording_demo/inputs/` 和说明文件 `README_recording.txt`。
- 行为：固定使用已验证的 RefCOCO case `115981` 和同一张图像，三个 JSON 输入分别描述 referring segmentation、region captioning（含可视化 region mask 与 SAMTok mask prompt）和 general VQA。新入口一次加载模型和 VQ-SAM2，依次运行三项推理，并在 `outputs/` 写入任务 JSON、预测/GT mask、三张大字结果 PNG 和 `summary.json`；分割若提供 GT mask 会在终端和结果中报告 IoU。该录屏入口不改变论文训练或标准 benchmark 评测协议。
- 验证：`record_three_tasks.py` 通过 `py_compile`，启动器通过 `bash -n`，`git diff --check` 通过；按实际成功的既有 `115981` 输出构造输入资产，GPU-free 校验确认三个 JSON 均引用同一 `shared_image.jpg`，RLE 解码尺寸为 `640x480`、前景像素数为 `37713`；完整 GPU smoke 因当前 GPU 0--3 正在执行既有评测而未中断该评测，已保留已有三任务模型输出作为案例基线。

### 2026-09-24 - 清理三能力展示中的推理标签

- 代码：修改 `evaluation/refcoco/demo_three_capabilities.py`；更新已有 `logs/demo_three_capabilities/result.json` 与 caption/VQA 两张 PNG。
- 文档：更新第 5.6 节三能力 demo 行为说明。
- 行为：mask-captioning 与传统 VQA 展示文本先移除 `<think>...</think>` 内容及孤立 think 标签，再写入 JSON 和可视化；若模型没有 think 块外的最终回答则报错，避免输出空白内容。重新使用已保存的模型答案生成 caption/VQA PNG，没有重新加载模型或启动 GPU 推理。
- 验证：对清理函数执行包含空 think、有内容 think、孤立标签和结束 token 的 GPU-free 样例检查；脚本 AST 解析、shell `bash -n`、输出 JSON 无 think 标记检查及 `git diff --check` 通过。

### 2026-09-24 - 恢复三任务录屏并修复展板文字截断

- 代码：修改 `evaluation/refcoco/record_three_tasks.py`、`tools/record_pixel_opsd_demo.sh`；更新 `logs/pixel_opsd_recording_demo/README_recording.txt` 和输入/输出展板。
- 文档：更新第 5.6 节录屏行为；未新增代码模块，既有模块清单继续适用。
- 行为：修复文字纵坐标重复累计造成的截断，改为等比例图片加横向文字展板；清理 think 展示而保留 raw response；新增三阶段进度、运行来源与耗时、可读结果文本及完整 tee 日志。质量指标与输出检查明确区分，案例选取范围明确为单例定性展示。
- 验证：原脚本与修改后的脚本均完成 GPU 0 三任务真实推理；最终脚本退出码 0，模型初始化与推理共 10.5 秒，IoU=0.9747474747474747、foreground=37713，两项文本输出与历史案例一致。独立重算 PNG mask IoU 并与 summary 对齐，核对 region RLE/PNG 一致、最终文本与 raw response 清洗一致；逐张查看全部六张 1600×1000 输入/输出展板，确认无文字裁切。Python 编译、shell 语法和 scoped git diff --check 通过；未改变训练、权重或其他 GPU 进程。

### 2026-09-24 - 固化本地五基线 MR 适配器与严格汇总

- 文件：新增 `evaluation/mr_baselines/{sa2va_infer,padt_infer,unipixel_infer,instructseg_infer,evfsam_infer,score}.py`，更新模块清单。
- 行为：从历史临时适配器迁入；修正 InstructSeg 的 DataArguments 来源、图片绝对路径和显式 vision tower；EVF-SAM 保留全部 User/Assistant 历史，不再仅取最后一问；UniPixel 遇到运行异常直接失败，不把异常伪装为空预测；scorer 拒绝错误、重复键、错误尺寸和不完整样本集。
- 协议边界：这些原生 dense-mask 基线使用串行化历史文本；Sa2VA/PaDT 额外要求回答最后问题的 mask。历史 SAMTok 作为文本保留，不声称原生模型具备 SAMTok 解码能力；与 CycleGRPO 的原生多轮 mask-conditioned 输入有差异。`mask_protocol.py` 适用于 SAMTok 序列，不用于强行解释原生 dense mask。最终指标是全量 sampled split 的 foreground CIoU / mean sample GIoU。
- 验证：已检查官方接口、历史适配器和全量 Sa2VA scorer 输出；新适配器尚在 smoke 验证，不宣称其余四模型已完成。

### 2026-09-24 - MR 数据预检与 Sa2VA 全量结果归档

- 文件：新增 `evaluation/mr_baselines/audit_data.py`；更新 `evaluation/mr_benchmarks_comparison_20260924.md` 和模块清单。
- 行为：使用 `evaluation/mask_protocol.py` 检查全部历史 SAMTok，记录数据 SHA256、各轮条数与图片完整性，生成每轮一个原样样本的 smoke 数据；Sa2VA 全量预测归档到 `logs/mr_baselines_20260924/sa2va/` 并填入明确标注串行历史适配协议的结果行。
- 验证：严格 scorer 重算 Sa2VA 26,360 条，通过计数、重复键、异常记录和 mask 尺寸检查；其余模型未填未经全量验证的指标。UniPixel/EVF-SAM 已分别产出两个 smoke 预测；InstructSeg 正在处理 CUDA 扩展导入路径。

### 2026-09-24 - 修复 PaDT ZeRO-3 视觉 token 词表边界

- 文件：`evaluation/mr_baselines/padt_infer.py`。
- 行为：按官方 RefCOCO 入口在读取 embedding 大小时使用 `deepspeed.zero.GatheredParameters`，避免分片参数形状导致 VRT token 词表边界错误；新增非零词表检查。
- 验证：对照官方 `inference_refcoco.py` 及 `padt_processor.py` 的视觉 token 分配逻辑；旧 smoke 仅有文本而没有任何 mask，不能作为已验证的 PaDT 分割输出，正在重新 smoke。

### 2026-09-24 - 增加 MR 单模型启动与 PaDT 原生输出转换

- 文件：新增 `evaluation/mr_baselines/run.sh`、`convert_padt.py`，同步模块清单。
- 行为：启动器针对固定 GPU 的已验证 hold worker 暂停矩阵计算但保留其显存，退出时恢复计算，避免评测前后空卡；按模型隔离依赖路径，支持每轮 smoke 与全量模式，不覆盖已有预测，成功后严格汇总。PaDT converter 要求完成记录精确覆盖 sample map，按原生 mask 并集转 RLE；无 mask 的已完成生成才算合法空预测，不把未完成推理算作空预测。
- 验证：shell 语法和 Python 编译检查；InstructSeg 经补充现有已编译 CUDA op 路径后两个 smoke 样本成功。全量结果仅在严格计数通过后填表。

### 2026-09-24 - 区分 UniPixel 无 mask 回答与运行异常

- 文件：`evaluation/mr_baselines/unipixel_infer.py`。
- 行为：成功完成生成但无原生分割输出时记录空 mask 和原始回答，附带 `native_mask_count`；CUDA、数据、接口异常仍直接失败，不生成假零分。保持官方首 mask 选择方式。
- 验证：UniPixel、EVF-SAM、InstructSeg 在四个数据集各十轮的 40 条 smoke 全部完成并通过 scorer；PaDT 修复后已实际生成 VRT 与 mask。完整数据预检 4,879 张图零缺失、100,808 个合法历史 group，0 个非法完整 group。

### 2026-09-24 - 四个剩余基线通过多轮 smoke 并启动全量

- 文档：更新对比表的运行状态与输出来源；纠正 UniPixel 原生 sentence 模板也附加分割要求的协议说明。
- 验证：PaDT、UniPixel、InstructSeg、EVF-SAM 各 40/40 smoke（四数据集 × 十轮）成功，原生 mask 转换与严格 scorer 全部退出 0。保存外部源码 commit/兼容补丁及模型配置哈希；全量分别使用 GPU 0/1/2/3，保留每卡 41GB hold 显存。全量指标仍待完成，不用 smoke 指标替代。

### 2026-09-24 - PaDT 全量评测启用已验证的 batch 8

- 文件：`evaluation/mr_baselines/run.sh`，新增 `PADT_BATCH_SIZE` 覆盖（默认 8）。
- 行为：使用官方原生 batch 推理接口提高吞吐，不修改 prompt、生成策略或 mask 解码；原 batch-1 尚未完成的短运行保留为独立 partial 目录，正式全量从头统一使用 batch 8，避免混合协议。
- 验证：batch-8 四数据集十轮共 40 条 smoke 推理退出 0，完成键和 mask 尺寸严格检查通过，scorer complete=true。BF16 批量 padding 会导致少量生成差异，不声称与 batch 1 位级相等。

### 2026-09-24 - 独立全量结果收尾监控

- 文件：新增 `evaluation/mr_baselines/finalize.py`，同步模块清单。
- 行为：按 `/proc` 的实际推理命令与输出路径确认进程结束，再独立执行 PaDT 转换、全量 scorer 和精确计数检查，只有 26,360 条全量完整结果才原子更新对应表格行。失败保留异常并退出非零，不伪造空预测，不重启存活任务。用于避免 launcher 尾部故障丢失汇总；不会修改正在运行的 shell。完成后仍需独立核查四卡占用。
- 验证：Python 编译检查；`--watch` 将首先识别四个存活推理进程，不把当前 partial 文件写入结果表。

### 2026-09-25 - 中断的 MR 基线全量推理按全局序号续跑

- 文件：修改 `evaluation/mr_baselines/{padt,unipixel,instructseg,evfsam}_infer.py`、`run.sh`；更新第 5.6 节模块说明。
- 行为：四个适配器新增 `--start-id`，仍按原四数据集、轮次及行顺序分配全局样本号，只对该号及之后的样本推理，保留原始 benchmark/round/row 键。`run.sh` 第四参数传递该起点；非零起点的独立输出目录只做原生推理及 PaDT 转换，不对不完整后缀执行全量 scorer。既有前缀文件不覆盖，前后缀合并后须严格检查 26,360 个键再填表。该续跑机制只影响离线 MR 评测，不改变论文训练或模型推理协议。
- 验证：四个 Python 适配器 `py_compile` 和 `run.sh` 的 `bash -n` 通过；旧运行进程实际已退出，前缀记录数分别为 PaDT 2,000、UniPixel 4,101、InstructSeg 3,901、EVF-SAM 6,901。全量续跑和合并仍在进行。

### 2026-09-25 - 恢复 MR 评测源码环境并自动收尾续跑结果

- 文件：修改 `evaluation/mr_baselines/unipixel_infer.py`、`run.sh`；新增 `evaluation/mr_baselines/finalize_resumed.py`；修补 `logs/mr_baselines_20260924/sources/` 下 PaDT 和 InstructSeg 本地源码兼容当前 Transformers/SigLIP API；更新第 5.6 节模块清单。
- 行为：由于原 `/tmp` 第三方源码 checkout 消失，按归档 commit 在持久日志目录恢复 PaDT、UniPixel、InstructSeg、EVF-SAM 并重放文本兼容补丁。`run.sh` 将项目环境 site-packages 放在本地依赖前，避免临时 vendor 的 NumPy 2 覆盖训练环境的 NumPy 1；InstructSeg 加入重新编译的当前 CUDA 扩展及 Detectron2 源码路径。UniPixel 每次生成前清空原生 `seg` 列表，以兼容新 Transformers 首步 DynamicCache；InstructSeg 从 SigLIP encoder 逐层取回原配置指定的中间层，因为当前 Transformers 的 vision forward 不再返回 `hidden_states`；PaDT 使用当前 flash-attn rotary 函数并适配 GenerationMixin 的缓存/停止条件参数。上述为本地评测环境兼容修补，仍使用原模型权重、输入序列和原生 mask 输出，不改变论文训练路径。收尾脚本仅在对应后缀退出码为 0 后合并已核验前缀和后缀，以官方 26,360 个样本键逐条检查后调用严格 scorer，成功才原子填表；全部结束后运行 `gpu_power_hold.sh start`。
- 验证：恢复的四份源码提交号与 `provenance/manifest.json` 一致；PaDT、UniPixel、InstructSeg 导入检查通过，InstructSeg CUDA 扩展用当前 CUDA/PyTorch 重新构建；PaDT、UniPixel、InstructSeg、EVF-SAM 的续跑分别已写出至少 120、801、801、4,901 条新预测，四张 GPU 均有显存占用。PaDT/InstructSeg 本地源码兼容补丁和源码信息归档在 `logs/mr_baselines_20260924/provenance/`，UniPixel 适配器修补保留在仓库脚本。`finalize_resumed.py` 已作为独立存活进程启动，负责在完成后严格评分、填表及运行四卡 GPU hold；未把任何 partial 数值填为最终指标。四个适配器编译、启动器语法、收尾脚本 `--help` 与 `git diff --check` 均通过。

### 2026-09-25 - 新增 70k baseline 的 25% 与 50% 全流数据量消融

- 文件：新增 `tools/train_supervised_70k_baseline_25pct_8gpu.sh`、`tools/train_supervised_70k_baseline_50pct_8gpu.sh`；修改 `tools/train_supervised_70k_common_8gpu.sh`；更新本文件第 2.2、3.2 节与工具模块清单。
- 行为：对 cycle、direct positive、direct no-target、DLC-QA 四个训练 parquet 分别取 25%/50%，cycle 按 source 分层，DLC JSONL 由 `dam_source_id` 精确配对；两档使用独立运行目录和端口，保留原 baseline 的训练配置与每步 batch，默认训练 45/90 step。原 70k baseline 与 SECA wrapper 默认仍使用完整 20k/30k/10k/10k 数据和 179 step。
- 验证：三份 shell 脚本 `bash -n` 通过；两档 `DRY_RUN=true HOLD_AFTER_EXIT=false` 均退出 0，行数分别为 5k/7.5k/2.5k/2.5k 与 10k/15k/5k/5k，JSONL 行数匹配；独立校验 cycle 的五个 source 配额、四流 25% 子集均包含于 50% 子集、DLC parquet/JSONL 的 `dam_source_id` 逐行对齐，`git diff --check` 通过。未启动 GPU 训练。

### 2026-09-25 - 新增 25% baseline 四卡一步训练 smoke

- 文件：新增 `tools/train_supervised_70k_baseline_25pct_4gpu_tmp.sh`，修改 `tools/train_supervised_70k_baseline_25pct_8gpu.sh`、`tools/train_supervised_70k_baseline_50pct_8gpu.sh`、`verl/workers/fsdp_workers.py`；更新第 2.2 节及工具模块清单。
- 行为：复用 25% 数据准备与 common 训练链路，以 GPU 0--2 训练、GPU 3 judge、可被 3 整除的 `114/228/57` batch 运行 1 step，并使用独立日志/端口；该 smoke 不启动结束占卡，不代表正式 7+1 拓扑性能。FSDP 模型类别判断改为仅查当前配置项，避免 Transformers 枚举所有图文模型时导入无关 Gemma3n 与旧 timm 冲突；Qwen3-VL 仍选用 `AutoModelForImageTextToText`。正式 25% 与 50% 八卡入口在启动 Ray 前验证此图文映射可解析。
- 验证：四卡脚本 `bash -n` 与 `DRY_RUN` 通过。首次真实运行通过缩量数据加载、3-GPU Ray attach、本地 judge 健康检查，在 FSDP 模型初始化时报 `timm.data.ImageNetInfo` 导入错误；修复映射查询后重跑，Qwen3-VL actor/reference/teacher FSDP、vLLM 与 judge 初始化均成功，进入首个 optimizer step 的 batch generation，按用户要求在该 step 内主动停止（退出码 143），未验证完整 step、loss 更新或 checkpoint。25% 与 50% 八卡 wrapper 的映射预检和 `DRY_RUN` 均退出 0，脚本语法、`fsdp_workers.py` 编译及 `git diff --check` 通过；独立 smoke Ray/judge 已清理，GPU 0--3 hold worker 已恢复并确认各约 41 GiB 显存和 100% 利用率。

### 2026-09-25 - 新增 SECA 70k routing 阈值四组正式入口

- 文件：新增 `tools/train_supervised_70k_seca_routing_l030_h085_8gpu.sh`、`tools/train_supervised_70k_seca_routing_l070_h085_8gpu.sh`、`tools/train_supervised_70k_seca_routing_l050_h065_8gpu.sh`、`tools/train_supervised_70k_seca_routing_l050_h100_8gpu.sh`；修改 `projects/rl/qwen3vl_4b_refcoco10k_volcengine.sh`。
- 文档：更新第 2.2、3.4、5.1 节及模块清单。
- 行为：四个新 wrapper 均继承 `train_supervised_70k_seca_8gpu.sh` 的完整 70k SECA 配置、数据流、112/224/56 batch、179 step 和 7+1 Ray/Llama 拓扑，只分别覆盖 `(low,high)` 为 `(0.30,0.85)`、`(0.70,0.85)`、`(0.50,0.65)`、`(0.50,1.00)`。主 launcher 新增 `ROUTING_LOW_THRESHOLD`/`ROUTING_HIGH_THRESHOLD` 环境变量，执行 `[0,1]` 及 `low<=high` 校验并传入 Hydra；默认值仍为 `0.5/0.85`。high 增量实验因上界限制从请求的 `+0.20` 截断为实际 `+0.15`。
- 验证：新增 wrapper 与主 launcher 的 `bash -n`、四组 `DRY_RUN=true HOLD_AFTER_EXIT=false` 预检和 `git diff --check` 均通过；四组预检均确认 `20000/30000/10000/10000` 数据行数和 DLC-QA JSONL 行数，未启动 GPU/Ray/FSDP 训练。

### 2026-09-25 - 新增 SECA routing l030/h085 四卡本机 smoke

- 文件：新增 `tools/train_supervised_70k_seca_routing_l030_h085_4gpu_tmp.sh`；更新本文件第 2.2 节和工具模块清单。
- 行为：临时入口只覆盖本机 smoke 所需的 3 张训练卡 + 1 张 judge、`114/228/57` parent batch、`MAX_STEPS=1`、独立端口/日志和 `HOLD_AFTER_EXIT=false`；完整复用 SECA 70k 数据流、模型与算法，并把 routing 阈值固定为 `0.30/0.85`，不作为正式 7+1 训练入口。
- 验证：`bash -n` 通过；实际本机运行完成四个 parquet 行数校验、Ray head、Llama judge 健康检查、routing Hydra 注入（`low=0.3/high=0.85`）、Qwen3-VL FSDP/vLLM/teacher 初始化并进入首个 batch generation。按用户确认后主动停止，未等待完整 optimizer step/checkpoint；独立 Ray/judge 已清理，GPU 0--3 hold worker 已恢复，`git diff --check` 待本次文档更新后复核。

### 2026-09-25 - README 收敛为四组 routing 消融复现实验

- 文件：修改 `README.md`；同步更新本文件第 2.2 节、README 模块说明和本变更日志；未新增、移动或删除模块。
- 行为：删除 README 中会引导到旧通用 20k/direct/DLC-QA 训练入口的示例，将 70k 训练说明收敛为四个正式 routing wrapper。README 现在明确四组 `(low,high)`、7 张 Ray/FSDP 训练卡 + GPU 7 judge 拓扑、20k/30k/10k/10k 数据契约、disjoint no-target 文件、模型/judge 路径变量、`DRY_RUN` 预检、顺序运行与 GPU hold 生命周期；DLC-QA 段仅保留 sidecar 数据契约。同步修正 7-rank FSDP checkpoint、`global_step_179` 和四组评测路径示例，避免与正式 70k 入口混淆。
- 验证：逐项核对四个 wrapper、`tools/train_supervised_70k_common_8gpu.sh`、主 RefCOCO launcher 与 README 的环境变量/数据路径/端口/拓扑；检索 README 无旧 `opsd_70k`、`gs25k`、`NUM_GPUS=8` 或其他替代训练命令；执行 README bash block 语法检查、四组 `DRY_RUN=true HOLD_AFTER_EXIT=false` 预检、`bash -n` 和 `git diff --check`。

### 2026-09-25 - README 增加并执行 50% baseline 控制实验

- 文件：修改 `README.md`、`code.md`；未新增、移动或删除模块。
- 行为：README 在四组 routing 阈值矩阵之外新增独立的 50% baseline 数据量控制实验，明确其关闭 SECA、使用 `10000/15000/5000/5000` 的确定性四流子集、匹配 5,000 条 DLC-QA JSONL、`112/224/56` parent batch、默认 90 step 和 7+1 拓扑；补充 baseline 的预检、启动命令及结果解释边界。
- 验证：`tools/train_supervised_70k_baseline_50pct_8gpu.sh` 的 `bash -n` 与 `DRY_RUN=true HOLD_AFTER_EXIT=false` 预检均退出 0，确认生成 `10000/15000/5000/5000` 四流子集及匹配的 5,000 条 JSONL sidecar；按用户要求未启动 Ray、judge、FSDP 或正式 GPU 训练，因此尚无 checkpoint、训练日志或 GPU hold 结果可报告。

### 2026-09-25 - 完成 50% baseline 四卡本机一步 smoke

- 文件：修改本节训练说明和变更日志；未新增或修改训练脚本、未改变正式 8 卡默认配置。
- 行为：通过环境变量覆盖 `tools/train_supervised_70k_baseline_50pct_8gpu.sh` 的本机资源，使用 GPU 0--2 训练、GPU 3 本地 Llama judge、`114/228/57` parent batch、`MAX_STEPS=1`、`SAVE_FREQ=1`、`SAVE_LIMIT=1` 与 `HOLD_AFTER_EXIT=false`；50% 子集与 baseline SECA 关闭配置保持不变。
- 验证：运行目录 `logs/cyclegrpo70k_historical_baseline_50pct_4gpu_smoke/` 的数据准备、四流行数校验、Ray head、judge 健康检查、Qwen3-VL actor/reference/teacher FSDP、vLLM、完整首个 optimizer step 和 `global_step_1` 三 rank actor checkpoint 均成功；`run.log` 记录 `[stage:done]` 与 `[stage:exit] status=0`，Ray/judge 已清理，GPU 0--3 已释放。该 smoke 验证四卡链路，不替代正式 7+1 训练。
