import pytest
import torch

from thermal_flow.inference.sample import sample
from thermal_flow.models.flow_matching import ConditionalFlowMatching
from thermal_flow.models.unet import ConditionalUNet


def _make_model() -> ConditionalFlowMatching:
    unet = ConditionalUNet(thermal_channels=3, rgb_channels=3, base_channels=16, time_dim=32)
    return ConditionalFlowMatching(unet)


def test_sample_output_shape_matches_rgb_input() -> None:
    model = _make_model()
    x_rgb = torch.randn(2, 3, 16, 16)

    out = sample(model, x_rgb, num_steps=2)

    assert out.shape == x_rgb.shape


def test_sample_does_not_track_gradients() -> None:
    model = _make_model()
    x_rgb = torch.randn(2, 3, 16, 16, requires_grad=True)

    out = sample(model, x_rgb, num_steps=2)

    assert not out.requires_grad


def test_sample_rejects_unknown_solver() -> None:
    model = _make_model()
    x_rgb = torch.randn(1, 3, 16, 16)
    with pytest.raises(ValueError):
        sample(model, x_rgb, solver="midpoint")


def test_sample_supports_rk4_solver() -> None:
    model = _make_model()
    x_rgb = torch.randn(1, 3, 16, 16)

    out = sample(model, x_rgb, num_steps=2, solver="rk4")

    assert out.shape == x_rgb.shape
