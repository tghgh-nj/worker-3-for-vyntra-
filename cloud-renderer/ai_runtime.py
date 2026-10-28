"""Shared runtime selection for the CPU/GPU backends used by AI models."""
from __future__ import annotations

import math
import os
from pathlib import Path


def bounded_cpu_threads(affinity_count: int | None, quota_count: int | None, requested: int | None = None) -> int:
    candidates = [value for value in (affinity_count, quota_count) if value is not None and value > 0]
    available = min(candidates) if candidates else max(1, os.cpu_count() or 1)
    if requested is not None and requested > 0:
        available = min(available, requested)
    return max(1, min(16, available))


def _cpu_quota() -> int | None:
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            return max(1, math.ceil(int(quota) / int(period)))
    except (OSError, ValueError, ZeroDivisionError):
        pass
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text().strip())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip())
        if quota > 0 and period > 0:
            return max(1, math.ceil(quota / period))
    except (OSError, ValueError, ZeroDivisionError):
        pass
    return None


def cpu_thread_count() -> int:
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = os.cpu_count() or 1
    try:
        requested = int(os.getenv("VYNTRA_AI_CPU_THREADS", "0"))
    except ValueError:
        requested = 0
    return bounded_cpu_threads(affinity, _cpu_quota(), requested)


def configure_torch(torch) -> int:
    threads = cpu_thread_count()
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, min(4, threads // 2)))
    except RuntimeError:
        # PyTorch only permits changing inter-op threads before work starts.
        pass
    return threads


def select_torch_device(torch):
    requested = os.getenv("VYNTRA_AI_DEVICE", "auto").strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise RuntimeError("ai_device_configuration_invalid")
    if requested == "cpu":
        return torch.device("cpu")
    available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    if requested == "cuda" and not available:
        raise RuntimeError("ai_cuda_runtime_unavailable")
    return torch.device("cuda" if available else "cpu")
