# PEGC Ablation Workspace

This directory is an isolated copy of the CycleGRPO-OPSD training/evaluation code.
The repository-root implementation is intentionally untouched. All checkpoints,
logs, reduced datasets, and experimental edits belong under this directory.

The planned five-way comparison uses the SAMTok checkpoint as initialization and
the same deterministic one-fifth data subsets and training budget for every run:

1. `baseline`: current mixed supervised OPSD.
2. `evidence_gate`: positive/negative teacher evidence gating.
3. `mask_credit`: hierarchical counterfactual mask-token credit.
4. `adaptive_balance`: adaptive supervised/self-supervised gradient balancing only.
5. `cbba`: confidence-balanced branch allocation driven by current cycle grounding quality.
5. `cbba`: confidence-balanced branch allocation driven by current cycle grounding quality.

The reproducible suite is `tools/run_pegc_suite.sh`. It uses physical GPUs 0 and 2 for the two-rank trainer and GPU 3 for the local Llama judge at port 8007; GPU 1 is left untouched when occupied by another job, and GPUs 4-7 are never referenced. Every run uses the same 4k/8k/2k/2k subsets, 112/224/56 batches, and 36 optimizer steps.

Never launch the original root training script from this workspace.

Data preparation is deterministic:

```bash
python tools/prepare_pegc_subsets.py --cycle ... --direct ... \
  --no-target ... --qa-parquet ... --qa-jsonl ... --output-dir data
```
