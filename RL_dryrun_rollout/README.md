# RL_dryrun_rollout

This module is a pre-GRPO diagnostic stage. It does not train the model. Instead, it loads a student model with an optional LoRA adapter, samples multiple completions per GSM8K training prompt, and checks whether each question has a useful mix of correct and incorrect completions. That reward variance is the signal needed for the later RL stage to produce meaningful gradients.

## Directory Layout

- `configs/qwen25_3b_sft_dryrun.yaml`: experiment configuration.
- `run_rollout.py`: main rollout sampling script.
- `outputs/{run.name}/`: deterministic output directory generated from the YAML run name.

The default `run.name` is parameterized from the YAML:

```yaml
run:
  name: qwen25_3b_instruct_sft_dryrun_g{generation.num_generations}_train{dataset.limit}
```

For example, if `generation.num_generations: 8` and `dataset.limit: 500`, the output directory becomes:

```text
outputs/qwen25_3b_instruct_sft_dryrun_g8_train500/
```

## Colab Usage

Set required tokens in Colab Secrets, such as `HF_TOKEN`. The current model is public, so this token is often optional, but setting it can avoid rate-limit issues.

```python
from google.colab import userdata
import os

hf_token = userdata.get("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
```

The dry-run is usually launched through `run.ipynb`.

Outputs include:

- `rollouts.jsonl`: one line per prompt, containing sampled completions, extracted answers, and rewards. This is mainly for detailed analysis.
- `summary.json`: aggregate metrics such as `all_correct_rate`, `all_wrong_rate`, `mixed_rate`, `parse_fail_rate`, average length, and reward distribution.
- `sampled_outputs.md`: human-readable sampled examples.

## Changing The Model Or Adapter

Edit only the `model` section in the YAML file:

```yaml
model:
  base_model_name_or_path: Qwen/Qwen2.5-3B-Instruct
  adapter_path: SFT_train/outputs/qwen25_3b_gsm8k_lora_sft_full
  tokenizer_name_or_path: SFT_train/outputs/qwen25_3b_gsm8k_lora_sft_full
  prompt_template: qwen_chat
  system_prompt: ""
  include_empty_system: false
```

`include_empty_system: false` lets the Qwen chat template insert its default system prompt:

```text
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
```

This matches the `template: qwen` input format used during SFT training.

For a non-chat model, or if you do not want to use a chat template, set:

```yaml
prompt_template: plain
```
