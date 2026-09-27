import torch

from thermal_flow.baselines.unet import UNetBaseline


def test_forward_shape_matches_input_spatial_size() -> None:
    model = UNetBaseline(rgb_channels=3, thermal_channels=3, base_channels=8)
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 3, 32, 32)


def test_supports_mismatched_channel_counts() -> None:
    model = UNetBaseline(rgb_channels=3, thermal_channels=1, base_channels=8)
    out = model(torch.randn(1, 3, 32, 32))
    assert out.shape == (1, 1, 32, 32)


def test_gradients_flow() -> None:
    model = UNetBaseline(base_channels=8)
    model(torch.randn(1, 3, 32, 32)).sum().backward()
    assert all(p.grad is not None for p in model.parameters())
