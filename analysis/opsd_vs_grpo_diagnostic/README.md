# GRPO vs. OPSD 监督诊断图

严格 autograd 管线只接受共同 student checkpoint 上导出的固定 rollout。它不会从评测
JSONL、IoU 汇总或 teacher diagnosis 推断 token logits，也不会用随机数补图。

## 当前数据审计与已生成图

仓库现有 `logs/**/teacher_diagnoses.jsonl` 只保存 `route`、`R_Ci`、caption 和分析文本；
现有 FSDP checkpoint 只保存参数分片。没有发现同时包含 `responses`、`response_mask`、
完整 GRPO group、`old_log_probs`、`advantages`、student logits 和 privileged teacher
logits 的固定 rollout 缓存。因此严格版本在缺少输入时会标记
`blocked_missing_diagnostic_data`。

当前正式输出由 `plot_dual_proxy.py` 生成：
`figure_opsd_vs_grpo.{pdf,svg,png}`。它只读取两套真实训练日志：

- GRPO-only：`logs/cyclegrpo20k_direct30k_notarget10k_dlcqa10k_nodirectsft`，714 个 step；
- OPSD：`logs/cyclegrpo20k_withnt_direct30k_notarget10k_dlcqa10k_bs112_directgrpo_ce005_pixel_empty_positivepenalty1`，178 个 step，另读取其 `teacher_diagnoses.jsonl`。

(a) 将 GRPO-only 每个阶段的一个 `pixel_iou_mean` 聚合值广播到六列，直观看到共享标量监督；(b) 保留 OPSD diagnosis 中六个真实 `pixel_ious`，对每条 diagnosis 减去其六次 rollout 均值后显示同一 caption 内的正负证据差异；(c) 分别把两套日志的 `-cap_actor.pg_loss` 与 pixel-IoU 变化标准化后累积，画成两条独立路径。脚本先渲染 12 个候选布局（`variants_dual/`），正式图选用每种方法六个箭头的单一版本，没有平均多版结果，也没有随机数或手工色块。

这是一张真实日志的 rollout/logged-loss 代理图：它能说明 OPSD 保留 rollout-level evidence、GRPO 日志中呈现 row-wise scalar credit 的监督结构差异，但不能隔离 loss 的因果作用，因为两套 run 不是共同 checkpoint/fixed rollout；它也不能称为 token-autograd 或 before/after parameter-update 图。严格共同 rollout 版本仍由 `extract_signals.py` / `plot_signals.py` 提供，缺少 logits 时会阻止生成。
## 导出数据契约

先在共同 checkpoint 上固定输入、seed、temperature=1、top-p=1、group size=6，保存一个
`fixed_rollout.npz`，至少包含：

`student_logits[N,T,V]`, `teacher_logits[N,T,V]`, `target_ids[N,T]`,
`response_mask[N,T]`, `old_log_probs[N,T]`, `advantages[N,T]`, `group_id[N]`,
`sample_id[N]`, `route[N]`, `R_Ci[N]`。logits 必须已经按 causal next-token 对齐，
即第 `t` 行预测 `target_ids[:,t]`；脚本不会悄悄再移位。可选的
`policy_loss_mask[N,T]` 会与 response mask 相乘。

所有完整 group 必须在 route 筛选前写入 NPZ，GRPO advantage 必须由完整 group 计算。
脚本随后固定筛选 `route == on_policy_distill` 且 `R_Ci >= 0.65` 的 OPSD 分布教学分支，
按 `(R_Ci, sample_id)` 排序，并在两张热力图中使用完全相同的行。

## 运行

```bash
python analysis/opsd_vs_grpo_diagnostic/extract_signals.py \
  --input /path/to/fixed_rollout.npz \
  --checkpoint /path/to/common_student \
  --teacher-checkpoint /path/to/ema_teacher \
  --dataset /path/to/dataset.parquet \
  --out-dir analysis/opsd_vs_grpo_diagnostic
python analysis/opsd_vs_grpo_diagnostic/plot_signals.py \
  --data-dir analysis/opsd_vs_grpo_diagnostic
python analysis/opsd_vs_grpo_diagnostic/plot_logged_proxy.py \
  --out-dir analysis/opsd_vs_grpo_diagnostic
python analysis/opsd_vs_grpo_diagnostic/plot_dual_proxy.py \
  --out-dir analysis/opsd_vs_grpo_diagnostic
```

提取阶段直接调用仓库的 `compute_policy_loss` 和
`chunked_weighted_jsd_loss`，用 autograd 得到
`s=-∂L/∂z[target]`；并对第一条轨迹做有限差分检查。图中热力图只做每种方法各自的
全局最大绝对值缩放，原始梯度和未分箱梯度也保存在 `signals.npz`。

严格版本的二维面板实现允许的 local-logit-gradient fallback：A/B 由更新前
teacher 与 student 对 sampled token 的 log-prob 差异（阈值默认 0.05）确定，并按完整
group 求均值，同时画出 GRPO 和 OPSD 两组配对箭头。只有额外提供互不重叠更新集和更新前后 `Δlog p` 后，才可以改成
“actual model updates”；本脚本不会把局部梯度称作参数更新。当前日志代理图的二维
面板则使用真实的 `-cap_actor.pg_loss` 与 `distill_opsd.distill_jsd` 逐 step 累积路径。

## 结果边界

图只分析 OPSD 的 `on_policy_distill` 分布教学分支，排除 regenerate replacement、
mask-code localization 和 SECA 的固定权重。teacher 的冻结/EMA 状态必须在导出元数据中
如实记录；脚本不假设 teacher 冻结。若没有上述真实张量，不能把该图作为实验结果使用。

## 新增两张图

第一张是增强的双热力图：
`figure_opsd_vs_grpo_ab.{pdf,svg,png}`，由 `plot_ab_enhanced.py` 从真实 `signals.npz` 生成。它使用 16 个离散颜色层级、统一的对称 99% 绝对值色标，并加入 `$k_1=\cdots=k_6$` / `$k_1\ne\cdots\ne k_6$` 以及 `shared credit` / `evidence-resolved` 的短标注。

第二张要求共同 checkpoint、共同 fixed rollout，并在分别执行一次真实 GRPO/OPSD 更新后测量 sampled-token `\Delta log p`。仓库当前只有 FSDP 参数分片和 dataloader iterator state，没有保存该 rollout 或更新前后 log-prob，因此 `plot_paired_update_direction.py` 只生成 `paired_update_metadata.json` 的 blocked 状态，不生成模拟图。准备好 `paired_updates.npz` 后，至少提供：

`delta_logp_grpo[N]`, `delta_logp_opsd[N]`, `teacher_gap[N]`, `group_id[N]`。

运行：

```bash
python analysis/opsd_vs_grpo_diagnostic/plot_ab_enhanced.py \
  --out-dir analysis/opsd_vs_grpo_diagnostic
python analysis/opsd_vs_grpo_diagnostic/plot_paired_update_direction.py \
  --input /path/to/paired_updates.npz \
  --out-dir analysis/opsd_vs_grpo_diagnostic
```

第二张脚本会按 `|teacher_gap|>0.05` 固定划分 A/B，要求每个诊断组同时含有两类位置，渲染 12 个候选版本后选择带配对散点和均值箭头的版本；它不会根据更新后结果重新分组。
