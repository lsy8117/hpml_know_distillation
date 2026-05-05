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
echo "  - full train/val split and SFT formatting logic: $ROOT_DIR/SFT_train/qwen25_3b_llamafactory_lora_sft_colab_train.ipynb"

stage "train_sft"
echo "SFT training is currently notebook/config based."
echo "Run the full SFT notebook:"
echo "  $ROOT_DIR/SFT_train/qwen25_3b_llamafactory_lora_sft_colab_train.ipynb"
echo "Or inspect the LLaMA-Factory config:"
echo "  $ROOT_DIR/SFT_train/train_qwen25_3b_lora_sft.yaml"

stage "rl_post_training"
echo "RL post-training is implemented under:"
echo "  $ROOT_DIR/RL_GRPO_train"
echo "Available configs:"
echo "  $ROOT_DIR/RL_GRPO_train/configs/qwen25_3b_sft_grpo.yaml"
echo "  $ROOT_DIR/RL_GRPO_train/configs/qwen25_3b_sft_gspo.yaml"
echo "  $ROOT_DIR/RL_GRPO_train/configs/qwen25_3b_sft_dapo.yaml"
echo "Example command:"
echo "  accelerate launch RL_GRPO_train/train_grpo.py --config RL_GRPO_train/configs/qwen25_3b_sft_gspo.yaml"

stage "eval_teacher/base/sft"
echo "Legacy evaluation is notebook-based."
echo "Use the existing notebooks under: $ROOT_DIR/benchmark"
echo "Existing benchmark outputs are already stored in:"
echo "  $ROOT_DIR/benchmark/gsm8k_benchmark_result"
echo "RL final-test outputs are stored in:"
echo "  $ROOT_DIR/RL_GRPO_train/outputs"

stage "done"
echo "  $ROOT_DIR/results/final_table.md"
echo "Read project integration notes in:"
echo "  $ROOT_DIR/README.md"
