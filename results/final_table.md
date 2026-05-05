# Benchmark Summary

Updated after integrating full SFT and RL post-training branches.

## GSM8K Test

| System | Model / Checkpoint | Accuracy / Exact Match (%) | Correct / Total | Avg Latency (s) | Avg Output Tokens | Source |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Teacher | qwen3.5-397b-a17b | 96.51 | 1273/1319 | - | - | `benchmark/gsm8k_benchmark_result/gsm8k_qwen3_5_results.json` |
| Base Student | Qwen/Qwen2.5-3B-Instruct | 79.61 | 1050/1319 | 14.47 | 272.30 | `benchmark/gsm8k_benchmark_result/gsm8k_qwen2_5_3b_results.json` |
| SFT Student (Orig-Orig) | Qwen2.5-3B + qwen2.5-3B-Orig-Orig | 80.21 | 1058/1319 | 30.50 | 281.90 | `benchmark/gsm8k_benchmark_result/gsm8k_student_qwen2.5-3B-Orig-Orig_results.json` |
| SFT Student (Orig-Quant) | Qwen2.5-3B + qwen2.5-3B-Orig-Quant | 77.48 | 1022/1319 | 26.21 | 246.20 | `benchmark/gsm8k_benchmark_result/gsm8k_student_qwen2.5-3B-Orig-Quant_results.json` |
| SFT Student (Distill-Orig) | Qwen2.5-3B + qwen2.5-3B-Distill-Orig | 78.62 | 1037/1319 | 15.95 | 140.70 | `benchmark/gsm8k_benchmark_result/gsm8k_student_qwen2.5-3B-Distill-Orig_results.json` |
| SFT Student (Distill-Quant) | Qwen2.5-3B + qwen2.5-3B-Distill-Quant | 78.70 | 1038/1319 | 16.20 | 147.20 | `benchmark/gsm8k_benchmark_result/gsm8k_student_qwen2.5-3B-Distill-Quant_results.json` |
| GRPO | Full SFT adapter + GRPO | 79.38 | 1047/1319 | - | 93.14 | `RL_GRPO_train/outputs/qwen25_3b_instruct_sft_grpo_g16_trainall_final_test/final_test_eval_results.json` |
| GSPO | Full SFT adapter + GSPO | 80.52 | 1062/1319 | - | 93.43 | `RL_GRPO_train/outputs/qwen25_3b_instruct_sft_gspo_g16_trainall_final_test/final_test_eval_results.json` |
| DAPO | Full SFT adapter + DAPO | 79.30 | 1046/1319 | - | 90.08 | `RL_GRPO_train/outputs/qwen25_3b_instruct_sft_dapo_g16_trainall_final_test/final_test_eval_results.json` |

## SFT Validation

The newer full SFT run is evaluated on a 725-example held-out subset from GSM8K train, so these numbers should not be compared directly with the full GSM8K test table above.

| System | Validation Accuracy (%) | Correct / Total | Source |
| --- | ---: | ---: | --- |
| Base Qwen2.5-3B-Instruct | 87.72 | 636/725 | `SFT_train/base_model_val_results.json` |
| Full SFT adapter | 80.41 | 583/725 | `SFT_train/val_results.json` |
| GRPO best validation checkpoint | 89.79 | 651/725 | `RL_GRPO_train/outputs/qwen25_3b_instruct_sft_grpo_g16_trainall/best_eval_results.json` |
| GSPO best validation checkpoint | 89.52 | 649/725 | `RL_GRPO_train/outputs/qwen25_3b_instruct_sft_gspo_g16_trainall/best_eval_results.json` |
| DAPO best validation checkpoint | 90.07 | 653/725 | `RL_GRPO_train/outputs/qwen25_3b_instruct_sft_dapo_g16_trainall/best_eval_results.json` |

## Notes

- The best current student-side GSM8K test result in this repository is GSPO at `80.52%`.
- The result is single-seed; multi-seed runs are needed before making a strong algorithmic claim.
- Full SFT lowers validation loss but does not improve validation exact match, so RL results should be interpreted carefully.
- RL outputs are much shorter than the earlier SFT/base benchmark outputs.
