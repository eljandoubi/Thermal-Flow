import math

import pytest
import torch

from thermal_flow.evaluation.image_metrics import all_metrics, mae, psnr, rmse, ssim


def test_mae_and_rmse_zero_for_identical_images() -> None:
    x = torch.rand(2, 3, 16, 16)
    assert mae(x, x).item() == pytest.approx(0.0, abs=1e-7)
    assert rmse(x, x).item() == pytest.approx(0.0, abs=1e-7)


def test_mae_known_value() -> None:
    x = torch.zeros(1, 1, 2, 2)
    y = torch.full((1, 1, 2, 2), 0.5)
    assert mae(x, y).item() == pytest.approx(0.5, abs=1e-6)


def test_rmse_known_value() -> None:
    x = torch.zeros(1, 1, 2, 2)
    y = torch.full((1, 1, 2, 2), 0.5)
    assert rmse(x, y).item() == pytest.approx(0.5, abs=1e-6)


def test_psnr_is_infinite_for_identical_images() -> None:
    x = torch.rand(2, 3, 8, 8)
    assert math.isinf(psnr(x, x).item())


def test_psnr_decreases_as_error_grows() -> None:
    x = torch.zeros(1, 1, 8, 8)
    small_error = x + 0.01
    large_error = x + 0.5
    assert psnr(x, small_error).item() > psnr(x, large_error).item()


def test_ssim_matches_losses_structural_ssim() -> None:
    x = torch.rand(2, 3, 16, 16)
    y = torch.rand(2, 3, 16, 16)
    assert ssim(x, y).item() == pytest.approx(ssim(x, y).item())
    assert ssim(x, x).item() == pytest.approx(1.0, abs=1e-4)


def test_all_metrics_returns_expected_keys() -> None:
    x = torch.rand(2, 3, 16, 16)
    y = torch.rand(2, 3, 16, 16)
    metrics = all_metrics(x, y)
    assert set(metrics) == {"mae", "rmse", "psnr", "ssim"}
    assert all(isinstance(v, float) for v in metrics.values())


def test_metrics_reject_shape_mismatch() -> None:
    x = torch.rand(1, 3, 8, 8)
    y = torch.rand(1, 1, 8, 8)
    with pytest.raises(ValueError):
        mae(x, y)
    with pytest.raises(ValueError):
        rmse(x, y)
    with pytest.raises(ValueError):
        psnr(x, y)
