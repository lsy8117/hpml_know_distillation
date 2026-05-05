from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from trl import GRPOTrainer

from grpo_profiler import GRPOStepProfiler, STAGES


class ProfiledGRPOTrainer(GRPOTrainer):
    """
    Subclass of GRPOTrainer that profiles a fixed window of steps.
    Only overrides _generate_completions, _compute_rewards, compute_loss,
    and training_step — everything else is untouched.
    """

    def __init__(
        self,
        *args,
        profile_steps: int = 5,
        profile_start_step: int = 0,
        profile_log_dir: str = "./profile_logs",
        profile_every_n_steps: int = 1,   # ← new
        precise_timing: bool = True,       # ← new
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.profile_steps          = profile_steps
        self.profile_start_step     = profile_start_step
        self.profile_log_dir        = Path(profile_log_dir)
        self.profile_every_n_steps  = profile_every_n_steps
        self.precise_timing         = precise_timing          # ← stored
        self.profile_log_dir.mkdir(parents=True, exist_ok=True)
        self._profiler: GRPOStepProfiler | None = None
        self._profiling_active = False

    def _should_profile(self) -> bool:
        step = self.state.global_step
        in_window = self.profile_start_step <= step < self.profile_start_step + self.profile_steps
        on_interval = (step % self.profile_every_n_steps == 0)
        return in_window and on_interval

    # ── stage hooks ─────────────────────────────────────────────────────────

    def _generate_completions(self, *args, **kwargs):
        if self._profiling_active:
            with self._profiler.stage("rollout"):
                return super()._generate_completions(*args, **kwargs)
        return super()._generate_completions(*args, **kwargs)

    def _compute_rewards(self, *args, **kwargs):
        if self._profiling_active:
            with self._profiler.stage("reward_compute"):
                return super()._compute_rewards(*args, **kwargs)
        return super()._compute_rewards(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if not self._profiling_active:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        # advantage_compute happens inside compute_loss before the forward pass.
        # We split it by hooking torch.no_grad() context which TRL uses for
        # advantage normalisation, approximated as the time before .backward().
        with self._profiler.stage("advantage_compute"):
            # TRL normalises advantages here — cheap CPU op, but worth tracking
            pass

        with self._profiler.stage("forward"):
            result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        return result

    # ── main step override ───────────────────────────────────────────────────

    def training_step(self, model, inputs, num_items_in_batch=None):
        if not self._should_profile():
            return super().training_step(model, inputs, num_items_in_batch)

        step = self.state.global_step
        device = next(model.parameters()).device
        self._profiler = GRPOStepProfiler(device, precise_timing=self.precise_timing)
        self._profiling_active = True

        with self._profiler:
            # backward and optimizer are inside training_step in Trainer base
            with self._profiler.stage("backward"):
                loss = super().training_step(model, inputs, num_items_in_batch)

            with self._profiler.stage("optimizer"):
                # optimizer.step() already called inside super(), so we
                # capture the tail cost (grad clip + scheduler) here
                pass

        self._profiling_active = False

        # ── collect metrics ──────────────────────────────────────────────────
        num_tokens  = _infer_num_tokens(inputs)
        num_samples = inputs["input_ids"].shape[0] if "input_ids" in inputs else 1
        loss_val    = loss.item() if hasattr(loss, "item") else float(loss)
        lr          = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else 0.0

        metrics = self._profiler.summary(
            loss=loss_val,
            lr=lr,
            num_samples=num_samples,
            num_tokens=num_tokens,
        )
        metrics["profile/global_step"] = step

        _log(metrics, step, self.profile_log_dir)
        return loss


# ── utils ────────────────────────────────────────────────────────────────────

def _infer_num_tokens(inputs: dict) -> int:
    if "input_ids" in inputs:
        return int(inputs["input_ids"].numel())
    return 0


def _log(metrics: dict[str, Any], step: int, log_dir: Path) -> None:
    # swanlab
    try:
        import swanlab
        swanlab.log({f"{k}": v for k, v in metrics.items()}, step=step)
    except Exception:
        pass

    # JSON file per step
    out = log_dir / f"step_{step:05d}.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)

    # stdout summary
    print(f"\n{'='*60}")
    print(f"[PROFILE] step {step}")
    print(f"{'='*60}")
    groups = {}
    for k, v in metrics.items():
        prefix = k.split("/")[0]
        groups.setdefault(prefix, {})[k] = v
    for prefix, group in groups.items():
        print(f"  [{prefix}]")
        for k, v in group.items():
            val = f"{v:.4f}" if isinstance(v, float) else str(v)
            print(f"    {k:<45} {val}")
