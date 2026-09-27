import pytest
import torch

from thermal_flow.baselines.cnn import CNNRegressionBaseline


def test_forward_shape_matches_thermal_channels() -> None:
    model = CNNRegressionBaseline(
        rgb_channels=3, thermal_channels=3, hidden_channels=8, num_layers=3
    )
    out = model(torch.randn(2, 3, 16, 16))
    assert out.shape == (2, 3, 16, 16)


def test_supports_mismatched_channel_counts() -> None:
    model = CNNRegressionBaseline(
        rgb_channels=3, thermal_channels=1, hidden_channels=8, num_layers=3
    )
    out = model(torch.randn(2, 3, 16, 16))
    assert out.shape == (2, 1, 16, 16)


def test_rejects_too_few_layers() -> None:
    with pytest.raises(ValueError):
        CNNRegressionBaseline(num_layers=1)


def test_gradients_flow() -> None:
    model = CNNRegressionBaseline(hidden_channels=8, num_layers=3)
    model(torch.randn(2, 3, 16, 16)).sum().backward()
    assert all(p.grad is not None for p in model.parameters())
