import torch

from thermal_flow.models.unet import ConditionalUNet


def test_forward_shape_matches_thermal_input() -> None:
    model = ConditionalUNet(thermal_channels=3, rgb_channels=3, base_channels=16, time_dim=32)
    x_t = torch.randn(2, 3, 32, 32)
    x_rgb = torch.randn(2, 3, 32, 32)
    t = torch.rand(2)

    out = model(x_t, t, x_rgb)

    assert out.shape == x_t.shape


def test_handles_mismatched_thermal_and_rgb_channels() -> None:
    model = ConditionalUNet(thermal_channels=1, rgb_channels=3, base_channels=16, time_dim=32)
    x_t = torch.randn(2, 1, 32, 32)
    x_rgb = torch.randn(2, 3, 32, 32)
    t = torch.rand(2)

    out = model(x_t, t, x_rgb)

    assert out.shape == (2, 1, 32, 32)


def test_gradients_flow_to_all_parameters() -> None:
    model = ConditionalUNet(thermal_channels=3, rgb_channels=3, base_channels=16, time_dim=32)
    x_t = torch.randn(2, 3, 32, 32)
    x_rgb = torch.randn(2, 3, 32, 32)
    t = torch.rand(2)

    model(x_t, t, x_rgb).sum().backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"


def test_parameter_count_is_lightweight() -> None:
    model = ConditionalUNet()
    n_params = sum(p.numel() for p in model.parameters())
    assert 1_000_000 < n_params < 30_000_000
