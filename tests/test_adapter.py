import torch

from thermal_flow.models.adapter import (
    BottleneckAdapter,
    count_trainable_parameters,
    freeze_for_adapter_finetuning,
)
from thermal_flow.models.unet import ConditionalUNet


def test_adapter_is_identity_at_initialization() -> None:
    adapter = BottleneckAdapter(channels=8, bottleneck_dim=4)
    z = torch.randn(2, 8, 4, 4)
    assert torch.allclose(adapter(z), z)


def test_adapter_output_shape() -> None:
    adapter = BottleneckAdapter(channels=8, bottleneck_dim=4)
    z = torch.randn(2, 8, 4, 4)
    assert adapter(z).shape == z.shape


def _make_unet_with_adapter() -> ConditionalUNet:
    return ConditionalUNet(
        thermal_channels=3,
        rgb_channels=3,
        base_channels=16,
        time_dim=32,
        use_adapter=True,
        adapter_dim=4,
    )


def test_unet_with_adapter_forward_shape() -> None:
    model = _make_unet_with_adapter()
    x_t = torch.randn(2, 3, 32, 32)
    x_rgb = torch.randn(2, 3, 32, 32)
    t = torch.rand(2)

    out = model(x_t, t, x_rgb)

    assert out.shape == x_t.shape


def test_freeze_for_adapter_finetuning_only_trains_adapter() -> None:
    model = _make_unet_with_adapter()
    total_params = sum(p.numel() for p in model.parameters())

    freeze_for_adapter_finetuning(model)

    trainable = count_trainable_parameters(model)
    assert 0 < trainable < total_params
    for name, param in model.named_parameters():
        if "adapter" in name:
            assert param.requires_grad
        else:
            assert not param.requires_grad


def test_freeze_without_adapter_freezes_everything() -> None:
    model = ConditionalUNet(base_channels=16, time_dim=32)  # no adapter present

    freeze_for_adapter_finetuning(model)

    assert count_trainable_parameters(model) == 0
