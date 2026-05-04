"""
Wrapper around train_grpo.py that injects ProfiledGRPOTrainer
before any TRL imports happen. Called by accelerate launch.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# ── ensure profiler modules are importable ────────────────────────────────────
sys.path.insert(0, "/content")

# ── ensure rl_common is importable (mirrors train_grpo.py's own path setup) ──
ROOT = Path(__file__).resolve().parents[1]

# Also try common Colab project locations
for candidate in [
    ROOT / "RL_common" / "src",
    Path("/content/drive/MyDrive/HPML_project/RL_common/src"),
    Path("/content/drive/MyDrive/HPML_project/project/RL_common/src"),
]:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
        break

# ── read profile config from environment variables ────────────────────────────
PROFILE_STEPS          = int(os.environ.get("PROFILE_STEPS",          "5"))
PROFILE_START_STEP     = int(os.environ.get("PROFILE_START_STEP",     "0"))
PROFILE_EVERY_N_STEPS  = int(os.environ.get("PROFILE_EVERY_N_STEPS",  "1"))
PRECISE_TIMING         = os.environ.get("PRECISE_TIMING", "1") == "1"
PROFILE_LOG_DIR        = os.environ.get("PROFILE_LOG_DIR", "/content/profile_logs")
MAX_STEPS              = PROFILE_START_STEP + PROFILE_STEPS

# ── patch trl.GRPOTrainer ─────────────────────────────────────────────────────
import trl
import importlib
from profiled_trainer import ProfiledGRPOTrainer

_orig_init = ProfiledGRPOTrainer.__init__
def _patched_init(self, *args, **kwargs):
    kwargs.setdefault("profile_steps",         PROFILE_STEPS)
    kwargs.setdefault("profile_start_step",    PROFILE_START_STEP)
    kwargs.setdefault("profile_every_n_steps", PROFILE_EVERY_N_STEPS)
    kwargs.setdefault("precise_timing",        PRECISE_TIMING)
    kwargs.setdefault("profile_log_dir",       PROFILE_LOG_DIR)
    _orig_init(self, *args, **kwargs)
ProfiledGRPOTrainer.__init__ = _patched_init

trl.GRPOTrainer = ProfiledGRPOTrainer
importlib.import_module("trl.trainer.grpo_trainer").GRPOTrainer = ProfiledGRPOTrainer

# ── patch load_config to cap max_steps and disable greedy eval ────────────────
import train_grpo

_orig_load_config = train_grpo.load_config
def _patched_load_config(path):
    cfg = _orig_load_config(path)
    cfg.setdefault("grpo", {})["max_steps"]   = MAX_STEPS
    cfg.setdefault("eval", {})["every_steps"] = MAX_STEPS + 1  # disable greedy eval
    return cfg
train_grpo.load_config = _patched_load_config

# ── hand off to train_grpo.main() ─────────────────────────────────────────────
train_grpo.main()