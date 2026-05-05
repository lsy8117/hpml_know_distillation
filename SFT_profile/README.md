# SFT_profile

This folder contains SFT profiling notebooks and compact profiling artifacts.

## Contents

- `train.ipynb`: original hand-written SFT training notebook copied for profiling comparison.
- `sft_profile*.ipynb`: SFT profiling notebook variants.
- `qwen25_3b_llamafactory_lora_sft_colab_train.ipynb`: LLaMA-Factory SFT notebook variant used for profiling.
- `train_qwen25_3b_lora_sft.yaml`: SFT profiling config.
- `data/`: profiling copy of the SFT dataset files.
- `success.jsonl`: teacher-generation sample copy used by the profiling workflow.

## Notes

This folder is for performance/profiling experiments. The canonical full SFT workflow lives in `SFT_train/`.
