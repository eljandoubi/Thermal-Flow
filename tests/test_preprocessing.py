import torch

from thermal_flow.datasets.preprocessing import normalize_to_unit_range, resize


def test_normalize_to_unit_range_scales_uint8() -> None:
    x = torch.tensor([[0, 128, 255]], dtype=torch.uint8)

    out = normalize_to_unit_range(x)

    assert out.dtype == torch.float32
    assert torch.allclose(out, torch.tensor([[0.0, 128 / 255, 1.0]]), atol=1e-6)


def test_resize_changes_spatial_size_and_keeps_channels() -> None:
    x = torch.rand(3, 8, 8)

    out = resize(x, 16)

    assert out.shape == (3, 16, 16)


def test_resize_is_identity_size_when_already_target() -> None:
    x = torch.rand(3, 10, 10)

    out = resize(x, 10)

    assert out.shape == x.shape
