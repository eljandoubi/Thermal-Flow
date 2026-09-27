import torch
from torch import nn

from thermal_flow.evaluation.efficiency import (
    count_flops,
    count_parameters,
    measure_inference_time,
    model_size_mb,
)


def test_count_parameters_matches_manual_sum() -> None:
    model = nn.Linear(4, 2)  # 4*2 weight + 2 bias = 10
    assert count_parameters(model) == 10


def test_count_parameters_trainable_only() -> None:
    model = nn.Linear(4, 2)
    model.bias.requires_grad = False
    assert count_parameters(model, trainable_only=True) == 8
    assert count_parameters(model, trainable_only=False) == 10


def test_model_size_mb_scales_with_parameter_count() -> None:
    small = nn.Linear(4, 2)
    large = nn.Linear(400, 200)
    assert model_size_mb(large) > model_size_mb(small)


def test_measure_inference_time_returns_positive_metrics() -> None:
    model = nn.Linear(8, 8)
    x = torch.randn(4, 8)

    result = measure_inference_time(lambda: model(x), num_iterations=5, warmup_iterations=2)

    assert result["seconds_per_call"] > 0
    assert result["images_per_second"] > 0


def test_count_flops_for_known_conv() -> None:
    model = nn.Conv2d(3, 8, kernel_size=3, padding=1)
    x = torch.randn(1, 3, 16, 16)

    # output_positions * (in_channels * kernel_h * kernel_w) * 2 (mul+add)
    expected = 16 * 16 * 8 * (3 * 3 * 3) * 2

    assert count_flops(model, x) == expected
