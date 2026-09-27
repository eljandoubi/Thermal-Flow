"""Image-level reconstruction metrics."""

from __future__ import annotations

import torch

from thermal_flow.losses.structural import ssim as _ssim


def _check_shapes(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _check_shapes(pred, target)
    return (pred - target).abs().mean()


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _check_shapes(pred, target)
    return torch.sqrt(torch.mean((pred - target) ** 2))


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Peak signal-to-noise ratio in dB; +inf when `pred == target` exactly."""
    _check_shapes(pred, target)
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(float("inf"))
    return 20 * torch.log10(torch.tensor(data_range)) - 10 * torch.log10(mse)


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    return _ssim(pred, target, data_range=data_range)


def all_metrics(
    pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0
) -> dict[str, float]:
    """Compute MAE, RMSE, PSNR, SSIM in one call, returned as plain Python floats."""
    return {
        "mae": mae(pred, target).item(),
        "rmse": rmse(pred, target).item(),
        "psnr": psnr(pred, target, data_range=data_range).item(),
        "ssim": ssim(pred, target, data_range=data_range).item(),
    }
