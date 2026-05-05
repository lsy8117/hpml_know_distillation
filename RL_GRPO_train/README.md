# RL_GRPO_train

本目录用于在 GSM8K 数学推理任务上，从已有 SFT LoRA adapter 继续做 RL 训练，并对比 GRPO、GSPO、DAPO 三种策略优化方法。

当前实验设置基本一致：模型为 `Qwen/Qwen2.5-3B-Instruct`，初始 adapter 为 `SFT_train/outputs/qwen25_3b_gsm8k_lora_sft_full`，训练集来自 `SFT_train/data/gsm8k_sft_train.json`，训练中验证集为 `SFT_train/data/gsm8k_sft_val.json`，最终测试集为官方 GSM8K test。

## 文件结构

- `train_grpo.py`: 训练和评测入口。
- `configs/qwen25_3b_sft_grpo.yaml`: GRPO 配置。
- `configs/qwen25_3b_sft_gspo.yaml`: GSPO 配置。
- `configs/qwen25_3b_sft_dapo.yaml`: DAPO 配置。
- `outputs/*/final_adapter/`: 训练后保存的 LoRA adapter。
- `outputs/*/best_eval_results.json`: 训练中验证集最优结果。
- `outputs/*_final_test/final_test_eval_results.json`: 最终 GSM8K test 结果。

## 运行方式

```bash
accelerate launch RL_GRPO_train/train_grpo.py \
  --config RL_GRPO_train/configs/qwen25_3b_sft_grpo.yaml
```

只跑最终测试：

```bash
python RL_GRPO_train/train_grpo.py \
  --config RL_GRPO_train/configs/qwen25_3b_sft_grpo.yaml \
  --eval-only \
  --final-test
```

将配置文件换成 `qwen25_3b_sft_gspo.yaml` 或 `qwen25_3b_sft_dapo.yaml` 即可运行对应算法。

## 当前结果

训练均为 1000 steps，`num_generations=16`，`seed=42`，使用同一套 reward 和评测脚本。

### 验证集 best checkpoint

| 算法 | best step | sft_val exact match | 正确数 |
| --- | ---: | ---: | ---: |
| GRPO | 960 | 89.7931% | 651 / 725 |
| GSPO | 980 | 89.5172% | 649 / 725 |
| DAPO | 860 | 90.0690% | 653 / 725 |

### GSM8K final test

| 算法 | exact match | 正确数 | reward mean | 平均输出 tokens | 长度惩罚率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRPO | 79.3783% | 1047 / 1319 | 0.8035 | 93.14 | 10.2350% |
| GSPO | 80.5155% | 1062 / 1319 | 0.8135 | 93.43 | 11.5997% |
| DAPO | 79.3025% | 1046 / 1319 | 0.8041 | 90.08 | 8.8704% |

结论上，当前 single-seed 结果里 GSPO 的最终 test exact match 最高，比 GRPO 高约 1.14 个百分点；DAPO 在验证集 best metric 最高，同时输出更短、长度惩罚率最低，但最终 test accuracy 没有超过 GSPO。

## 算法差异

- GRPO: 标准 baseline，使用 group relative advantage，不需要额外 value model。
- GSPO: 在 GRPO 基础上使用 sequence-level importance sampling，更直接约束整段 completion 的策略变化。
- DAPO: 使用 `loss_type: dapo`、`mask_truncated_completions: true`、非对称 clip 和 soft overlong punishment，重点缓解长度偏置和过长输出问题。

当前结果是固定随机种子的单次对比，适合做阶段性判断；如果要写成更严谨的实验结论，建议每个算法补多 seed 的均值和标准差。
