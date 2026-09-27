import pytest
import torch

from thermal_flow.losses.structural import SSIMLoss, ssim


def test_ssim_of_identical_images_is_one() -> None:
    x = torch.rand(2, 3, 32, 32)
    assert torch.isclose(ssim(x, x), torch.tensor(1.0), atol=1e-4)


def test_ssim_loss_of_identical_images_is_zero() -> None:
    x = torch.rand(2, 3, 32, 32)
    loss = SSIMLoss()
    assert torch.isclose(loss(x, x), torch.tensor(0.0), atol=1e-4)


def test_ssim_decreases_for_dissimilar_images() -> None:
    x = torch.rand(2, 3, 32, 32)
    y = torch.rand(2, 3, 32, 32)
    assert ssim(x, y) < ssim(x, x)


def test_ssim_rejects_shape_mismatch() -> None:
    x = torch.rand(2, 3, 32, 32)
    y = torch.rand(2, 1, 32, 32)
    with pytest.raises(ValueError):
        ssim(x, y)
