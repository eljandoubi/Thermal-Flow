"""Efficiency metrics: parameter count, model size, inference latency, FLOPs.

Reporting these makes the project's "lightweight" claim measurable (project
proposal section 19).
"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    """Approximate in-memory size assuming 4 bytes/parameter (float32)."""
    return count_parameters(model) * 4 / (1024**2)


def _synchronize(device: torch.device | None) -> None:
    if device is None:
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


@torch.no_grad()
def measure_inference_time(
    forward_fn: Callable[[], object],
    num_iterations: int = 20,
    warmup_iterations: int = 5,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Average wall-clock time (seconds) per call to `forward_fn()` (no args).

    Pass `device` (the device the model/inputs live on) to synchronize CUDA/MPS
    around each call; otherwise GPU work queued asynchronously would make the
    timing measure kernel-launch overhead rather than actual execution time.
    """
    for _ in range(warmup_iterations):
        forward_fn()
    _synchronize(device)

    start = time.perf_counter()
    for _ in range(num_iterations):
        forward_fn()
    _synchronize(device)
    elapsed = time.perf_counter() - start

    seconds_per_call = elapsed / num_iterations
    images_per_second = 1.0 / seconds_per_call if seconds_per_call > 0 else float("inf")
    return {"seconds_per_call": seconds_per_call, "images_per_second": images_per_second}


def count_flops(model: nn.Module, *example_inputs: torch.Tensor) -> int:
    """Total FLOPs for one forward pass through `model(*example_inputs)`."""
    with FlopCounterMode(display=False) as flop_counter:
        model(*example_inputs)
    return flop_counter.get_total_flops()
