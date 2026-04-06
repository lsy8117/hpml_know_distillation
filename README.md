# hpml_know_distillation

## Current Midterm Integration Status

This repository already contains the main work from different members in separate folders:

- `SFT_data_generation/`: teacher-side GSM8K data generation
- `train/`: student SFT training workflow
- `benchmark/`: teacher / base / student evaluation notebooks and JSON outputs

For the midterm stage, the engineering integration goal is not to rewrite these parts, but to document how they connect and provide a thin orchestration entrypoint.

## Folder Responsibilities

### `SFT_data_generation/`

This folder contains the teacher data generation module.

- Entry config: `SFT_data_generation/configs/teacher_gsm8k.yaml`
- Main command:

```bash
cd SFT_data_generation
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python -m teacher_data_gen.main --config configs/teacher_gsm8k.yaml
```

- Output example:
  - `outputs/runs/20260402_201445/success.jsonl`
  - `outputs/runs/20260402_201445/run_stats.json`

Note: API key is now expected from the environment variable `DASHSCOPE_API_KEY`.

### `train/`

This folder contains the current student training workflow.

- Main file: `train/train.ipynb`
- Current status:
  - data formatting and train/val split logic are handled in the notebook
  - LoRA training is handled in the notebook
  - exported checkpoints are referenced in `train/README.md`

At the midterm stage, training is still notebook-based and has not yet been migrated to a standalone CLI script.

### `benchmark/`

This folder contains the current evaluation workflow.

- Teacher evaluation notebook: `benchmark/gsm8k_benchmark_teacher.ipynb`
- Base model evaluation notebook: `benchmark/gsm8k_benchmark_qwen2_5_3b.ipynb`
- Student evaluation notebook: `benchmark/gsm8k_student_distill.ipynb`
- Existing result JSON files:
  - `benchmark/gsm8k_benchmark_result/gsm8k_qwen3_5_results.json`
  - `benchmark/gsm8k_benchmark_result/gsm8k_qwen2_5_3b_results.json`
  - `benchmark/gsm8k_benchmark_result/gsm8k_student_qwen2.5-3B-Distill-Orig_results.json`
  - `benchmark/gsm8k_benchmark_result/gsm8k_student_qwen2.5-3B-Distill-Quant_results.json`
  - `benchmark/gsm8k_benchmark_result/gsm8k_student_qwen2.5-3B-Orig-Orig_results.json`

At the midterm stage, evaluation is also notebook-based.

## Midterm Pipeline

The current project pipeline is:

```text
prepare_data -> teacher_gen -> build_sft_dataset -> train_sft -> eval_teacher/base/sft -> summarize
```

How this maps to the existing repository:

- `prepare_data`
  - currently embedded in the existing data-processing / notebook workflow
- `teacher_gen`
  - implemented in `SFT_data_generation/`
- `build_sft_dataset`
  - currently handled in the training notebook workflow
- `train_sft`
  - implemented in `train/train.ipynb`
- `eval_teacher/base/sft`
  - implemented in the notebooks under `benchmark/`
- `summarize`
  - current midterm summary table is written manually to `results/final_table.md` using existing benchmark JSON outputs

## Thin Runner

A thin orchestration script is provided at `scripts/run_all.sh`.

It does not reimplement existing logic. Instead, it:

- runs teacher generation if `DASHSCOPE_API_KEY` is available
- points users to the existing notebook-based training workflow
- points users to the existing notebook-based benchmark workflow
- treats `results/final_table.md` as the current summary artifact

Run it with:

```bash
bash scripts/run_all.sh
```

## Environment

The current dependency list is in `env/requirements.txt`.

Recommended setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r env/requirements.txt
```

## Midterm Result Table

The current three-row comparison table is stored in:

- `results/final_table.md`

Current metrics from existing benchmark outputs:

| System | Accuracy |
| --- | ---: |
| Teacher | 96.51 |
| Base Student | 79.61 |
| SFT Student (Orig-Orig) | 80.21 |
| SFT Student (Distill-Orig) | 78.62 |
| SFT Student (Distill-Quant) | 78.70 |

