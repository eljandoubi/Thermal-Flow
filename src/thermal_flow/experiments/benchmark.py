"""Efficiency benchmarking across multiple models (project proposal section 19)."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from thermal_flow.evaluation.efficiency import (
    count_flops,
    count_parameters,
    measure_inference_time,
    model_size_mb,
)


def benchmark_model(
    model: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    num_iterations: int = 20,
    warmup_iterations: int = 5,
) -> dict[str, float]:
    """Parameter count, size, FLOPs, and inference latency for one model."""
    device = next(model.parameters()).device
    metrics = {
        "num_parameters": float(count_parameters(model)),
        "model_size_mb": model_size_mb(model),
        "flops": float(count_flops(model, *example_inputs)),
    }
    metrics.update(
        measure_inference_time(
            lambda: model(*example_inputs),
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations,
            device=device,
        )
    )
    return metrics


def benchmark_models(
    models: dict[str, nn.Module],
    example_inputs_fn: Callable[[str, nn.Module], tuple[torch.Tensor, ...]],
    num_iterations: int = 20,
    warmup_iterations: int = 5,
) -> dict[str, dict[str, float]]:
    """Benchmark several named models, each with its own example inputs.

    `example_inputs_fn(name, model)` builds the forward-pass arguments for each
    model, since different baselines take different call signatures (e.g. the
    Flow Matching U-Net takes `(x_t, t, x_rgb)` while the CNN baseline takes
    just `(x_rgb,)`).
    """
    return {
        name: benchmark_model(
            model, example_inputs_fn(name, model), num_iterations, warmup_iterations
        )
        for name, model in models.items()
    }
