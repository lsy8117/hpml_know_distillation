# RL_GRPO_train

This directory contains the RL post-training workflow for GSM8K mathematical reasoning. It starts from the existing full SFT LoRA adapter and compares three policy-optimization variants: GRPO, GSPO, and DAPO.

The current experiments share the same core setup: the base model is `Qwen/Qwen2.5-3B-Instruct`, the initial adapter is `SFT_train/outputs/qwen25_3b_gsm8k_lora_sft_full`, the training set is `SFT_train/data/gsm8k_sft_train.json`, the in-training validation set is `SFT_train/data/gsm8k_sft_val.json`, and the final evaluation uses the official GSM8K test split.

## Directory Layout

- `train_grpo.py`: training and evaluation entry point.
- `configs/qwen25_3b_sft_grpo.yaml`: GRPO configuration.
- `configs/qwen25_3b_sft_gspo.yaml`: GSPO configuration.
- `configs/qwen25_3b_sft_dapo.yaml`: DAPO configuration.
- `outputs/*/final_adapter/`: LoRA adapters saved after training.
- `outputs/*/best_eval_results.json`: best validation results during training.
- `outputs/*_final_test/final_test_eval_results.json`: final GSM8K test results.

## Usage

```bash
accelerate launch RL_GRPO_train/train_grpo.py \
  --config RL_GRPO_train/configs/qwen25_3b_sft_grpo.yaml
```

Run final-test evaluation only:

```bash
python RL_GRPO_train/train_grpo.py \
  --config RL_GRPO_train/configs/qwen25_3b_sft_grpo.yaml \
  --eval-only \
  --final-test
```

To run GSPO or DAPO, replace the config with `qwen25_3b_sft_gspo.yaml` or `qwen25_3b_sft_dapo.yaml`.

## Current Results

All three runs use 1000 training steps, `num_generations=16`, `seed=42`, and the same reward and evaluation scripts.

### Best Validation Checkpoint

| Algorithm | Best Step | SFT-Val Exact Match | Correct |
| --- | ---: | ---: | ---: |
| GRPO | 960 | 89.7931% | 651 / 725 |
| GSPO | 980 | 89.5172% | 649 / 725 |
| DAPO | 860 | 90.0690% | 653 / 725 |

### GSM8K Final Test

| Algorithm | Exact Match | Correct | Reward Mean | Avg Output Tokens | Length Penalty Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRPO | 79.3783% | 1047 / 1319 | 0.8035 | 93.14 | 10.2350% |
| GSPO | 80.5155% | 1062 / 1319 | 0.8135 | 93.43 | 11.5997% |
| DAPO | 79.3025% | 1046 / 1319 | 0.8041 | 90.08 | 8.8704% |

In the current single-seed comparison, GSPO has the highest final-test exact match and is about 1.14 percentage points above GRPO. DAPO reaches the highest best-validation metric and produces shorter outputs with the lowest length-penalty rate, but its final-test accuracy does not exceed GSPO.

## Algorithm Differences

- GRPO: the baseline group-relative policy optimization method, using group-relative advantages without a separate value model.
- GSPO: a sequence-level importance-sampling variant of GRPO that constrains the policy shift over the full completion.
- DAPO: uses `loss_type: dapo`, `mask_truncated_completions: true`, asymmetric clipping, and optional soft overlong punishment to reduce length bias and overlong outputs.

These are fixed-seed, single-run results. Stronger experimental claims should use multiple seeds and report mean and standard deviation for each algorithm.
