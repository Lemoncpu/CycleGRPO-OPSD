#!/bin/bash
# CycleGRPO — image-level RL training (Qwen3-VL-4B + mask-token / SAM2).
# Caption<->grounding cycle-consistency reward. See README for data / cold-start
# checkpoint setup. Replace the <PATH_TO_*> placeholders before running.

set -x
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE=offline
export WANDB_DIR='./verl_wandb_logs'

export MODELSCOPE_CACHE='./modelscope_cache/shared'
export HF_DATASETS_CACHE='./hf_dataset_cache'

# Cold-start (co-SFT) checkpoint that RL starts from. See README for how to obtain it.
MODEL_PATH="<PATH_TO_COLD_START_CKPT>"
SECA_ENABLED="${SECA_ENABLED:-false}"
SECA_SELF_SUPERVISED_ENABLED="${SECA_SELF_SUPERVISED_ENABLED:-false}"
EGCA_ENABLED="${EGCA_ENABLED:-false}"
EGCA_OPD_ENABLED="${EGCA_OPD_ENABLED:-true}"
EGCA_ACTOR_COEF="${EGCA_ACTOR_COEF:-0.2}"
EGCA_CREDIT_MODE="${EGCA_CREDIT_MODE:-contrastive}"
EGCA_UPDATE_MODE="${EGCA_UPDATE_MODE:-weighted_ce}"
EGCA_REFERENCE_MODE="${EGCA_REFERENCE_MODE:-target}"
EGCA_CE_LOSS_WEIGHT="${EGCA_CE_LOSS_WEIGHT:-0.05}"
EGCA_CE_MIN_WEIGHT="${EGCA_CE_MIN_WEIGHT:-0.25}"
EGCA_CE_MAX_WEIGHT="${EGCA_CE_MAX_WEIGHT:-1.0}"
EGCA_CE_TOKEN_CREDIT_SCALE="${EGCA_CE_TOKEN_CREDIT_SCALE:-0.5}"
EGCA_CE_TOKEN_WEIGHT_MAX="${EGCA_CE_TOKEN_WEIGHT_MAX:-2.0}"
EGCA_EVIDENCE_MIN_WEIGHT="${EGCA_EVIDENCE_MIN_WEIGHT:-0.5}"
EGCA_EVIDENCE_MAX_WEIGHT="${EGCA_EVIDENCE_MAX_WEIGHT:-1.0}"
EGCA_FALSE_POSITIVE_PENALTY="${EGCA_FALSE_POSITIVE_PENALTY:-2.0}"
EGCA_REFERENCE_COARSE_CODE="${EGCA_REFERENCE_COARSE_CODE:-0}"
EGCA_REFERENCE_FINE_CODE="${EGCA_REFERENCE_FINE_CODE:-0}"
EGCA_CREDIT_CLIP="${EGCA_CREDIT_CLIP:-1.0}"
EGCA_WARMUP_STEPS="${EGCA_WARMUP_STEPS:-0}"
EGCA_RAMP_STEPS="${EGCA_RAMP_STEPS:-0}"
EGCA_MAX_GROUPS="${EGCA_MAX_GROUPS:-8}"

python3 -m verl.trainer.main \
    config=projects/rl/config.yaml \
    data.train_files="['<PATH_TO_DATA>/denseworld_train.parquet', '<PATH_TO_DATA>/gres_no_target_train.parquet']" \
    data.val_files="['<PATH_TO_DATA>/val.parquet']" \
    data.format_prompt=./projects/rl/format_prompt/non_thinking.jinja \
    data.region_format=mask_token \
    worker.actor.model.freeze_vision_tower=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.optimize_captioner=true \
    worker.actor.optimize_segmenter=true \
    worker.opsd.seca.enabled=${SECA_ENABLED} \
    worker.opsd.seca.self_supervised_enabled=${SECA_SELF_SUPERVISED_ENABLED} \
    worker.opsd.egca.enabled=${EGCA_ENABLED} \
    worker.opsd.egca.opd_enabled=${EGCA_OPD_ENABLED} \
    worker.opsd.egca.actor_coef=${EGCA_ACTOR_COEF} \
    worker.opsd.egca.credit_mode=${EGCA_CREDIT_MODE} \
    worker.opsd.egca.update_mode=${EGCA_UPDATE_MODE} \
    worker.opsd.egca.reference_mode=${EGCA_REFERENCE_MODE} \
    worker.opsd.egca.ce_loss_weight=${EGCA_CE_LOSS_WEIGHT} \
    worker.opsd.egca.ce_min_weight=${EGCA_CE_MIN_WEIGHT} \
    worker.opsd.egca.ce_max_weight=${EGCA_CE_MAX_WEIGHT} \
    worker.opsd.egca.ce_token_credit_scale=${EGCA_CE_TOKEN_CREDIT_SCALE} \
    worker.opsd.egca.ce_token_weight_max=${EGCA_CE_TOKEN_WEIGHT_MAX} \
    worker.opsd.egca.evidence_min_weight=${EGCA_EVIDENCE_MIN_WEIGHT} \
    worker.opsd.egca.evidence_max_weight=${EGCA_EVIDENCE_MAX_WEIGHT} \
    worker.opsd.egca.false_positive_penalty=${EGCA_FALSE_POSITIVE_PENALTY} \
    worker.opsd.egca.reference_coarse_code=${EGCA_REFERENCE_COARSE_CODE} \
    worker.opsd.egca.reference_fine_code=${EGCA_REFERENCE_FINE_CODE} \
    worker.opsd.egca.credit_clip=${EGCA_CREDIT_CLIP} \
    worker.opsd.egca.warmup_steps=${EGCA_WARMUP_STEPS} \
    worker.opsd.egca.ramp_steps=${EGCA_RAMP_STEPS} \
    worker.opsd.egca.max_groups=${EGCA_MAX_GROUPS} \
    worker.rollout.n=6 \
    trainer.experiment_name=cyclegrpo_qwen3vl_4b \
    trainer.total_epochs=1 \
    trainer.val_freq=-1 \
    trainer.save_freq=5 \
    trainer.val_before_train=false \
    trainer.save_limit=20 \
    trainer.logger=["file","wandb"] \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.rollout_batch_size=128 \
    worker.actor.global_batch_size=128

# ---- Optional tuning for large multi-image samples / OOM (append as needed) ----
# Long multi-image prompts make the actor backward memory-heavy. If you hit OOM,
# add some of these (see README "Memory tuning"):
#    data.max_prompt_length=24576 \
#    worker.rollout.max_num_batched_tokens=32768 \
#    data.mini_rollout_batch_size=16 \
#    worker.actor.micro_batch_size_per_device_for_experience=1 \
#    worker.actor.micro_batch_size_per_device_for_update=1 \
#    trainer.max_try_make_batch=64 \
