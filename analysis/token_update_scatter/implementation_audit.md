# Implementation audit

- `verl/trainer/core_algos.py:compute_grpo_outcome_advantage` computes the complete six-sample group
  normalization `(r_i-mean_group)/(std_group+eps)` and broadcasts the scalar over `response_mask`.
- `verl/workers/actor/dp_actor.py:accumulate_actor_gradients` recomputes token log-probabilities and applies the
  repository clipped policy surrogate, ordinary KL, caption-anchor KL, and segmentation-anchor KL before one
  optimizer step. `update_supervised` is the weighted teacher-forcing CE path.
- `verl/workers/opsd/routing.py:aggregate_caption_rollouts` averages the six localization pixel IoUs into `R_Ci`
  and applies the configured `<0.5`, `[0.5,0.85]`, `>0.85` routes. `uses_original_grpo` preserves native GRPO on
  safe routes when `preserve_original_grpo=true`.
- `verl/workers/opsd/distillation.py:chunked_weighted_jsd_loss` implements the privileged generalized JSD with
  teacher entropy weighting, blocked caption special-token vocabulary, and response masks.
- `verl/trainer/ray_trainer.py` around the cycle update block accumulates caption GRPO/CE/JSD, localization GRPO,
  optional anchors, and EGCA/SECA auxiliaries before one actor optimizer step; EMA teacher parameters are updated
  afterwards. The current default teacher decay is configured as `1.0` in the historical comparison run, so it is
  effectively frozen after initialization.
- `verl/workers/opsd/egca.py` and `verl/workers/opsd/seca.py` are separate auxiliary credit/self-distillation
  modules; this figure does not label them as CycleGRPO+OPD and does not infer their token credit from scalar logs.

The repository does not retain common sampled rollout tensors or per-token before/after logits for the two historical
runs. The delivered figure therefore uses real checkpoint forward passes on a common held-out set and the explicitly
named teacher-alignment fallback. It does not claim a paired update experiment.
