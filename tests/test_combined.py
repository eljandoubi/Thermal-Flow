import torch

from thermal_flow.losses.combined import CombinedFlowMatchingLoss
from thermal_flow.models.flow_matching import ConditionalFlowMatching
from thermal_flow.models.unet import ConditionalUNet


def _make_model() -> ConditionalFlowMatching:
    unet = ConditionalUNet(thermal_channels=3, rgb_channels=3, base_channels=16, time_dim=32)
    return ConditionalFlowMatching(unet)


def test_combined_loss_is_finite_and_backprops() -> None:
    model = _make_model()
    combined = CombinedFlowMatchingLoss(lambda_ssim=0.1, lambda_edge=0.05)
    x_rgb = torch.rand(2, 3, 32, 32)
    x_thermal = torch.rand(2, 3, 32, 32)

    losses = combined(model, x_rgb, x_thermal)

    assert set(losses) == {"total", "fm", "ssim", "edge"}
    for value in losses.values():
        assert torch.isfinite(value)

    losses["total"].backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters())


def test_total_equals_weighted_sum_of_components() -> None:
    model = _make_model()
    combined = CombinedFlowMatchingLoss(lambda_ssim=0.2, lambda_edge=0.1)
    x_rgb = torch.rand(2, 3, 32, 32)
    x_thermal = torch.rand(2, 3, 32, 32)
    t = torch.full((2,), 0.3)

    losses = combined(model, x_rgb, x_thermal, t=t)

    expected_total = losses["fm"] + 0.2 * losses["ssim"] + 0.1 * losses["edge"]
    assert torch.isclose(losses["total"], expected_total, atol=1e-6)


def test_zero_weights_reduce_to_flow_matching_loss() -> None:
    model = _make_model()
    combined = CombinedFlowMatchingLoss(lambda_ssim=0.0, lambda_edge=0.0)
    x_rgb = torch.rand(2, 3, 32, 32)
    x_thermal = torch.rand(2, 3, 32, 32)
    t = torch.full((2,), 0.3)

    losses = combined(model, x_rgb, x_thermal, t=t)

    assert torch.isclose(losses["total"], losses["fm"], atol=1e-6)
