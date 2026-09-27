import torch
from torch import nn

from thermal_flow.experiments.benchmark import benchmark_model, benchmark_models


def test_benchmark_model_returns_expected_keys() -> None:
    model = nn.Conv2d(3, 4, kernel_size=3, padding=1)
    x = torch.randn(1, 3, 8, 8)

    result = benchmark_model(model, (x,), num_iterations=2, warmup_iterations=1)

    assert result["num_parameters"] > 0
    assert result["model_size_mb"] > 0
    assert result["flops"] > 0
    assert result["seconds_per_call"] > 0
    assert result["images_per_second"] > 0


def test_benchmark_models_supports_heterogeneous_inputs() -> None:
    models: dict[str, nn.Module] = {
        "conv": nn.Conv2d(3, 4, kernel_size=3, padding=1),
        "linear": nn.Linear(8, 4),
    }

    def example_inputs_fn(name: str, model: nn.Module) -> tuple[torch.Tensor, ...]:
        if name == "conv":
            return (torch.randn(1, 3, 8, 8),)
        return (torch.randn(1, 8),)

    results = benchmark_models(models, example_inputs_fn, num_iterations=1, warmup_iterations=0)

    assert set(results) == {"conv", "linear"}
    assert results["conv"]["num_parameters"] > 0
    assert results["linear"]["num_parameters"] > 0
