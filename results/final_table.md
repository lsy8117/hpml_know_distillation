# Midterm Benchmark Summary

Generated at 2026-04-05 23:02:24Z from existing benchmark JSON files.

| System | Model / Checkpoint | Accuracy (%) | Correct / Total | Avg Latency (s) | Avg Output Tokens | Source |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Teacher | qwen3.5-397b-a17b | 96.51 | 1273/1319 | - | - | `benchmark/gsm8k_benchmark_result/gsm8k_qwen3_5_results.json` |
| Base Student | Qwen/Qwen2.5-3B-Instruct | 79.61 | 1050/1319 | 14.47 | 272.30 | `benchmark/gsm8k_benchmark_result/gsm8k_qwen2_5_3b_results.json` |
| SFT Student (Orig-Orig) | Qwen/Qwen2.5-3B-Instruct + qwen2.5-3B-Orig-Orig | 80.21 | 1058/1319 | 30.50 | 281.90 | `benchmark/gsm8k_benchmark_result/gsm8k_student_qwen2.5-3B-Orig-Orig_results.json` |
| SFT Student (Distill-Orig) | Qwen/Qwen2.5-3B-Instruct + qwen2.5-3B-Distill-Orig | 78.62 | 1037/1319 | 15.95 | 140.70 | `benchmark/gsm8k_benchmark_result/gsm8k_student_qwen2.5-3B-Distill-Orig_results.json` |
| SFT Student (Distill-Quant) | Qwen/Qwen2.5-3B-Instruct + qwen2.5-3B-Distill-Quant | 78.70 | 1038/1319 | 16.20 | 147.20 | `benchmark/gsm8k_benchmark_result/gsm8k_student_qwen2.5-3B-Distill-Quant_results.json` |

## Notes

- The table now includes all currently available student checkpoints with benchmark JSON files in the repository.
- Teacher: Teacher model on GSM8K benchmark.
- Base Student: Untuned Qwen2.5-3B-Instruct baseline.
- SFT Student (Orig-Orig): Fine-tuned on original GSM8K data without quantization; best current student result in the repo.
- SFT Student (Distill-Orig): Fine-tuned on distilled teacher outputs without quantization.
- SFT Student (Distill-Quant): Fine-tuned on distilled teacher outputs with NF4 quantization.
- Among the currently saved student checkpoints, `Orig-Orig` is above the base model, while both distilled variants are slightly below the base model on this benchmark.
