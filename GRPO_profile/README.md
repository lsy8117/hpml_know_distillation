# GRPO_profile

This folder contains profiling helpers for GRPO-style RL training.

## Contents

- `grpo_profiler.py`: stage-level GPU/time/memory profiling utilities.
- `profiled_trainer.py`: a profiling subclass/wrapper around TRL `GRPOTrainer`.
- `profiled_train_grpo.py`: wrapper entry point that patches `GRPOTrainer` for short profiling runs.
- `train_grpo.py`: copied GRPO training entry used by the profiling workflow.
- `qwen25_3b_sft_grpo.yaml`: default profiling config.
- `train_bs-32-gc-True-vllm-*`: example resolved configs for vLLM on/off comparisons.
- `log/profile_logs/profile_results.png`: sample profiling visualization.

## Notes

This is a profiling workspace, not the canonical RL training implementation. The canonical GRPO/GSPO/DAPO implementation lives in `RL_GRPO_train/`.
