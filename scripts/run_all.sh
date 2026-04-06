#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TEACHER_CONFIG="${TEACHER_CONFIG:-$ROOT_DIR/SFT_data_generation/configs/teacher_gsm8k.yaml}"

stage() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

stage "teacher_gen"
if [[ -n "${DASHSCOPE_API_KEY:-}" ]]; then
  (
    cd "$ROOT_DIR/SFT_data_generation"
    export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
    "$PYTHON_BIN" -m teacher_data_gen.main --config "$TEACHER_CONFIG"
  )
else
  echo "Skipping teacher generation because DASHSCOPE_API_KEY is not set."
  echo "If you want to rerun it, install SFT_data_generation and export DASHSCOPE_API_KEY first."
fi

stage "prepare_data_and_build_sft_dataset"
echo "These two steps are already implemented inside the existing project files:"
echo "  - raw GSM8K subset selection / teacher output: $ROOT_DIR/SFT_data_generation"
echo "  - train/val split and SFT formatting logic: $ROOT_DIR/train/train.ipynb"

stage "train_sft"
echo "Training is currently notebook-based."
echo "Run the existing notebook: $ROOT_DIR/train/train.ipynb"

stage "eval_teacher/base/sft"
echo "Evaluation is currently notebook-based."
echo "Use the existing notebooks under: $ROOT_DIR/benchmark"
echo "Existing benchmark outputs are already stored in:"
echo "  $ROOT_DIR/benchmark/gsm8k_benchmark_result"

stage "done"
echo "  $ROOT_DIR/results/final_table.md"
echo "Read project integration notes in:"
echo "  $ROOT_DIR/README.md"
