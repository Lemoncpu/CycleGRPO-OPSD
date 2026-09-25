# 真实 token 更新散点图

这组图不是压缩包里的 synthetic preview。`extract_token_updates.py` 从固定 seed 的 disjoint RefCOCO parquet
取样，分别对 SAMTok base、官方 GRPO checkpoint、Pixel-OPSD checkpoint 和独立 direct-supervision checkpoint
做真实多模态 teacher-forced forward。x 轴是允许的 `teacher_alignment` 机制诊断（独立 teacher 与 base 的
mask-code log-prob 差），y 轴是 checkpoint 相对 base 的实际 `Δlog p`。只保留 GT SAMTok depth-2 mask-code
token；不会把 caption token 与 mask token 混在一起。

运行命令（需 GPU；不会启动训练）：

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 /volume/ybo/xyc/envs/cyclegrpo/bin/python \
  analysis/token_update_scatter/extract_token_updates.py \
  --data /volume/ybo/xyc/datasets/direct_refcoco30k_disjoint_cycle20k/refcoco_train_30k_disjoint_cycle20k_seed20260823.parquet \
  --base /volume/ybo/xyc/Qwen3-VL-4B-SAMTok \
  --grpo /volume/ybo/xyc/CycleGRPO/logs/cyclegrpo_official_20k_g6k6/evaluation/step_156_response256/hf_global_step_156 \
  --opsd logs/cyclegrpo20k_pixel_empty_bs128_response256/evaluation/step_156_response256/hf_global_step_156 \
  --teacher logs/cyclegrpo20k_direct30k_notarget10k_dlcqa10k_nodirectsft/evaluation/step_714/hf_global_step_714 \
  --output-dir analysis/token_update_scatter
PYTHONPATH=. /volume/ybo/xyc/envs/cyclegrpo/bin/python \
  analysis/token_update_scatter/plot_token_updates.py \
  --input analysis/token_update_scatter/token_data.npz \
  --output-dir analysis/token_update_scatter
```

这是一项 checkpoint-to-checkpoint 机制诊断：两套历史训练使用了不同的训练 parquet，未重新执行配对
GRPO/OPSD optimizer step，因此不能表述为公平 objective 消融、独立 correctness 证明、benchmark 结果或
因果 credit。3D 的 reliability 明确定义为 base 对 GT mask-code 序列的 `exp(mean log p)`，不是训练代码的
`R_Ci`。若没有可用 GPU，脚本只应保留审计和配置，不得用 preview 数组替代真实数据。
