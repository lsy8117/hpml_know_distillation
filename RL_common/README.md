# RL_common

This package provides shared utilities for the GSM8K RL post-training stages. It centralizes answer parsing, reward construction, evaluation, dataset loading, prompt formatting, and model loading so the dry-run and RL training modules can reuse the same behavior.

## Contents

- `answers.py`: answer extraction and normalization. It first looks for `The answer is <number>`, then `\boxed{...}`, and finally falls back to the last numeric expression.
- `data.py`: GSM8K train/test loading plus SFT train/validation ID handling, keeping validation examples aligned with the SFT split.
- `rewards.py`: answer reward, format reward, parse/length penalties, and dry-run group diagnostics.
- `eval.py`: reusable exact-match evaluation for validation or test splits.
- `model.py`: helper for loading the base model, tokenizer, and optional LoRA adapter from YAML configuration.

## Colab Installation

```bash
cd /content/project
pip install -e RL_common
```

Model selection is mainly controlled by the caller YAML file. Example:

```yaml
model:
  base_model_name_or_path: Qwen/Qwen2.5-3B-Instruct
  adapter_path: SFT_train/outputs/qwen25_3b_gsm8k_lora_sft_full
  tokenizer_name_or_path: SFT_train/outputs/qwen25_3b_gsm8k_lora_sft_full
  prompt_template: qwen_chat
  system_prompt: ""
  include_empty_system: false
```

Leave `adapter_path` empty when evaluating or training without a LoRA adapter.
