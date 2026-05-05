from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any

import torch

STAGES = ["rollout", "reward_compute", "advantage_compute", "forward", "backward", "optimizer"]


# ── GPU Poller ──────────────────────────────────────────────────────────────

class GPUPoller:
    """Polls GPU utilization in a background thread."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _poll(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            while not self._stop.is_set():
                rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self._samples.append(float(rates.gpu))
                self._stop.wait(self.interval)
        except Exception:
            pass

    @property
    def mean(self) -> float:
        return sum(self._samples) / len(self._samples) if self._samples else 0.0

    @property
    def peak(self) -> float:
        return max(self._samples, default=0.0)

    def snapshot(self) -> dict:
        return {"mean": self.mean, "peak": self.peak, "samples": list(self._samples)}


# ── Per-stage snapshot ───────────────────────────────────────────────────────

class StageSnapshot:
    """Captures GPU metrics at the start and end of a stage."""

    def __init__(self, name: str, device: torch.device, precise_timing: bool = True):
        self.name = name
        self.device = device
        self.duration: float = 0.0
        self.peak_vram_gb: float = 0.0
        self.reserved_vram_gb: float = 0.0
        self.memory_traffic_gb: float = 0.0
        self.gpu_mean_util: float = 0.0
        self.gpu_peak_util: float = 0.0
        self.precise_timing = precise_timing

        # internals
        self._t0: float = 0.0
        self._alloc_before: int = 0
        self._poller = GPUPoller()

    def begin(self):
        if self.device.type == "cuda":
            if self.precise_timing:
                torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

            stats = torch.cuda.memory_stats(self.device)
            self._alloc_before = stats.get("allocation.all.allocated", 0)
        self._poller.start()
        self._t0 = time.perf_counter()

    def end(self):
        if self.device.type == "cuda":
            if self.precise_timing:
                torch.cuda.synchronize(self.device)

        self.duration = time.perf_counter() - self._t0
        self._poller.stop()

        if self.device.type == "cuda":
            self.peak_vram_gb     = torch.cuda.max_memory_allocated(self.device) / 1e9
            self.reserved_vram_gb = torch.cuda.memory_reserved(self.device)      / 1e9

            stats = torch.cuda.memory_stats(self.device)
            alloc_after  = stats.get("allocation.all.allocated", 0)
            # bytes touched = new allocations + frees during the stage (reads + writes)
            freed_bytes  = stats.get("allocation.all.freed", 0) * 4  # rough: each free ≈ one read
            new_bytes    = max(alloc_after - self._alloc_before, 0)
            self.memory_traffic_gb = (new_bytes + freed_bytes) / 1e9

        self.gpu_mean_util = self._poller.mean
        self.gpu_peak_util = self._poller.peak

    def as_dict(self) -> dict[str, Any]:
        """Keys are metric/stage_name so swanlab groups by metric across stages."""
        return {
            f"time/{self.name}":             self.duration,
            f"peak_vram/{self.name}":        self.peak_vram_gb,
            f"reserved_vram/{self.name}":    self.reserved_vram_gb,
            f"memory_traffic/{self.name}":   self.memory_traffic_gb,
            f"gpu_util_mean/{self.name}":    self.gpu_mean_util,
            f"gpu_util_peak/{self.name}":    self.gpu_peak_util,
        }


# ── Step-level profiler ──────────────────────────────────────────────────────

class GRPOStepProfiler:
    """
    Context manager that wraps a full GRPO training step.
    Individual stages are timed via the .stage() context manager.

    Usage:
        with GRPOStepProfiler(device) as prof:
            with prof.stage("rollout"):       ...
            with prof.stage("reward_compute"): ...
            with prof.stage("forward"):       ...
            with prof.stage("backward"):      ...
            with prof.stage("optimizer"):     ...
        metrics = prof.summary(loss, lr, num_samples, num_tokens)
    """

    def __init__(self, device: torch.device, precise_timing: bool = True):
        self.device = device
        self._snapshots: dict[str, StageSnapshot] = {}
        self._step_start: float = 0.0
        self._step_poller = GPUPoller()   # whole-step GPU util
        self.precise_timing = precise_timing

    def __enter__(self):
        if self.device.type == "cuda" and self.precise_timing:
            torch.cuda.synchronize(self.device)
        self._step_start = time.perf_counter()
        self._step_poller.start()
        return self

    def __exit__(self, *_):
        if self.device.type == "cuda" and self.precise_timing:
            torch.cuda.synchronize(self.device)
        self._step_poller.stop()

    @contextmanager
    def stage(self, name: str):
        snap = StageSnapshot(name, self.device, precise_timing=self.precise_timing)
        self._snapshots[name] = snap
        snap.begin()
        try:
            yield snap
        finally:
            snap.end()

    def summary(self, loss, lr, num_samples, num_tokens) -> dict[str, Any]:
        total_time   = time.perf_counter() - self._step_start
        compute_time = sum(
            s.duration for n, s in self._snapshots.items() if n in ("forward", "backward")
        )

        m: dict[str, Any] = {
            # step-level — their own namespace, no stage comparison needed
            "step/loss":                 loss,
            "step/lr":                   lr,
            "step/total_time_s":         total_time,
            "step/compute_time_s":       compute_time,
            "step/peak_vram_gb":         torch.cuda.max_memory_allocated(self.device) / 1e9
                                         if self.device.type == "cuda" else 0.0,
            "step/throughput_samples_s": num_samples / total_time   if total_time   > 0 else 0.0,
            "step/throughput_tokens_s":  num_tokens  / total_time   if total_time   > 0 else 0.0,
            "step/compute_tokens_s":     num_tokens  / compute_time if compute_time > 0 else 0.0,
            "step/gpu_mean_util":        self._step_poller.mean,
            "step/gpu_peak_util":        self._step_poller.peak,

            # time fractions — group together for bottleneck chart
            **{
                f"time_frac/{name}": snap.duration / total_time if total_time > 0 else 0.0
                for name, snap in self._snapshots.items()
            },

            # per-stage metrics — metric/stage_name for cross-stage comparison
            **{k: v for snap in self._snapshots.values() for k, v in snap.as_dict().items()},
        }

        return m
