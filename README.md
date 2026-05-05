# hpml_know_distillation

This repository contains the HPML project on GSM8K mathematical reasoning post-training. The project now covers the full path from teacher data generation to SFT and RL post-training.

## Repository Map

- `SFT_data_generation/`: teacher-side GSM8K data generation with DashScope / OpenAI-compatible API.
- `train/`: original notebook-based SFT experiments for the midterm checkpoint.
- `SFT_train/`: LLaMA-Factory SFT workflow, full filtered SFT dataset, validation outputs, and LoRA adapter artifacts.
- `RL_common/`: shared answer parsing, prompt building, data loading, reward, model loading, and evaluation utilities for RL stages.
- `RL_dryrun_rollout/`: pre-RL rollout diagnostics to check reward variance before GRPO-style training.
- `RL_GRPO_train/`: GRPO, GSPO, and DAPO training/evaluation configs, scripts, and result artifacts.
- `GRPO_profile/`: GRPO profiling utilities and sample profiling outputs.
- `SFT_profile/`: SFT profiling notebooks and related artifacts.
- `benchmark/`: earlier benchmark notebooks and benchmarking environment notes.
- `results/`: summarized result tables.

## Current Pipeline

```text
teacher_gen
  -> build_full_sft_dataset
  -> train_sft_full
  -> rl_dryrun_rollout
  -> train_grpo/gspo/dapo
  -> eval_validation_and_gsm8k_test
  -> profile_and_summarize
```

The midterm pipeline and artifacts are still kept for comparison. The newer full-data pipeline uses filtered teacher generations from `SFT_data_generation/outputs/runs/20260426_132022`.

## Teacher Data Generation

Teacher generation reads GSM8K train questions and asks `qwen3.5-397b-a17b` to return structured `reasoning` and `answer` fields. The newer generation pass:

- targets the full GSM8K train split,
- filters teacher answers that mismatch official GSM8K answers,
- records teacher CoT token statistics,
- writes successful samples to `success.jsonl` and filtered samples to `bad.jsonl`.

Run:

```bash
cd SFT_data_generation
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export DASHSCOPE_API_KEY=YOUR_KEY
python -m teacher_data_gen.main --config configs/teacher_gsm8k.yaml
```

Current full-data run summary:

- target examples: `7473`
- successful examples: `7256`
- filtered answer mismatches / bad outputs: `217`
- average teacher CoT tokens: `61.35`

## SFT

`SFT_train/` contains the LLaMA-Factory full SFT setup.

- base model: `Qwen/Qwen2.5-3B-Instruct`
- dataset size: `7256`
- train / val split: `6531 / 725`
- LoRA rank / alpha / dropout: `32 / 64 / 0.05`
- epochs: `4`
- learning rate: `2e-5`
- output adapter: `SFT_train/outputs/qwen25_3b_gsm8k_lora_sft_full`

Validation results in this full SFT run:

| System | Validation Accuracy |
| --- | ---: |
| Base Qwen2.5-3B-Instruct | 87.72% |
| Full SFT adapter | 80.41% |

The SFT loss improves, but validation exact-match accuracy drops. This is an important caveat for interpreting later RL results.

## RL Post-Training

RL code lives in `RL_GRPO_train/` and reuses utilities from `RL_common/`.

Implemented algorithms:

- GRPO: group relative policy optimization baseline.
- GSPO: sequence-level importance sampling variant.
- DAPO: DAPO loss with truncated-completion masking, asymmetric clipping, and optional soft overlong punishment.

All three runs use the same full SFT adapter as initialization and evaluate on the same SFT validation set and official GSM8K test set.

Run example:

```bash
accelerate launch RL_GRPO_train/train_grpo.py \
  --config RL_GRPO_train/configs/qwen25_3b_sft_grpo.yaml
```

Final GSM8K test results:

| System | Exact Match | Correct / Total | Avg Output Tokens |
| --- | ---: | ---: | ---: |
| GRPO | 79.38% | 1047/1319 | 93.14 |
| GSPO | 80.52% | 1062/1319 | 93.43 |
| DAPO | 79.30% | 1046/1319 | 90.08 |

Current single-seed result: GSPO is the best RL post-training run and slightly exceeds the previous best SFT student result. More seeds are needed before treating this as a robust conclusion.

## Benchmark Summary

The combined result table is in:

```text
results/final_table.md
```

Important comparison points:

- Teacher remains far stronger at `96.51%`.
- Original midterm best SFT student: `80.21%`.
- Current best RL post-training result: GSPO at `80.52%`.
- GRPO and DAPO do not beat the older midterm best on GSM8K test.

## Environment

Install the shared environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r env/requirements.txt
```

For vLLM-backed TRL generation, install the optional TRL vLLM extra in the target runtime:

```bash
pip install "trl[vllm]>=0.25.0"
```

## Notes

- Large LoRA adapter weights are tracked through Git LFS where present.
- Do not commit `.DS_Store`, `__pycache__/`, or `*.pyc`.
- Existing midterm artifacts are kept intentionally as baselines.
- `benchmark/README.md` records the original benchmark environment and decoding settings.
