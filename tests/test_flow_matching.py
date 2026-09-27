import pytest
import torch

from thermal_flow.models.flow_matching import ConditionalFlowMatching
from thermal_flow.models.unet import ConditionalUNet


def _make_model() -> ConditionalFlowMatching:
    unet = ConditionalUNet(thermal_channels=3, rgb_channels=3, base_channels=16, time_dim=32)
    return ConditionalFlowMatching(unet)


def test_sample_path_boundary_conditions() -> None:
    x_rgb = torch.randn(2, 3, 16, 16)
    x_thermal = torch.randn(2, 3, 16, 16)

    x_t0, u = ConditionalFlowMatching.sample_path(x_rgb, x_thermal, torch.zeros(2))
    assert torch.allclose(x_t0, x_rgb)
    assert torch.allclose(u, x_thermal - x_rgb)

    x_t1, _ = ConditionalFlowMatching.sample_path(x_rgb, x_thermal, torch.ones(2))
    assert torch.allclose(x_t1, x_thermal)


def test_sample_path_rejects_shape_mismatch() -> None:
    x_rgb = torch.randn(2, 3, 16, 16)
    x_thermal = torch.randn(2, 1, 16, 16)
    with pytest.raises(ValueError):
        ConditionalFlowMatching.sample_path(x_rgb, x_thermal, torch.rand(2))


def test_loss_is_finite_and_backprops() -> None:
    model = _make_model()
    x_rgb = torch.randn(2, 3, 16, 16)
    x_thermal = torch.randn(2, 3, 16, 16)

    loss = model.loss(x_rgb, x_thermal)
    assert torch.isfinite(loss)

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert any(torch.any(g != 0) for g in grads)


def test_loss_accepts_explicit_timestep() -> None:
    model = _make_model()
    x_rgb = torch.randn(2, 3, 16, 16)
    x_thermal = torch.randn(2, 3, 16, 16)

    loss = model.loss(x_rgb, x_thermal, t=torch.full((2,), 0.5))

    assert torch.isfinite(loss)
