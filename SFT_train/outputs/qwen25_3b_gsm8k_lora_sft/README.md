---
library_name: peft
license: other
base_model: Qwen/Qwen2.5-3B-Instruct
tags:
- base_model:adapter:Qwen/Qwen2.5-3B-Instruct
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: qwen25_3b_gsm8k_lora_sft
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen25_3b_gsm8k_lora_sft

This model is a fine-tuned version of [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) on the gsm8k_sft_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3162

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.03
- num_epochs: 3.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.8280        | 0.1479 | 25   | 0.5818          |
| 0.4262        | 0.2959 | 50   | 0.4181          |
| 0.3842        | 0.4438 | 75   | 0.3720          |
| 0.3386        | 0.5917 | 100  | 0.3549          |
| 0.3320        | 0.7396 | 125  | 0.3454          |
| 0.3020        | 0.8876 | 150  | 0.3372          |
| 0.3051        | 1.0355 | 175  | 0.3333          |
| 0.3001        | 1.1834 | 200  | 0.3291          |
| 0.3001        | 1.3314 | 225  | 0.3266          |
| 0.3030        | 1.4793 | 250  | 0.3228          |
| 0.2946        | 1.6272 | 275  | 0.3220          |
| 0.2968        | 1.7751 | 300  | 0.3192          |
| 0.2931        | 1.9231 | 325  | 0.3188          |
| 0.2842        | 2.0710 | 350  | 0.3176          |
| 0.2698        | 2.2189 | 375  | 0.3173          |
| 0.2638        | 2.3669 | 400  | 0.3172          |
| 0.2762        | 2.5148 | 425  | 0.3167          |
| 0.2893        | 2.6627 | 450  | 0.3162          |
| 0.2744        | 2.8107 | 475  | 0.3162          |
| 0.2931        | 2.9586 | 500  | 0.3163          |


### Framework versions

- PEFT 0.18.1
- Transformers 5.0.0
- Pytorch 2.10.0+cu128
- Datasets 4.0.0
- Tokenizers 0.22.2