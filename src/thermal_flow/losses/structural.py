"""Structural similarity (SSIM) loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _gaussian_1d(
    window_size: int, sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    return g / g.sum()


def _gaussian_window(
    window_size: int, channels: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    g1d = _gaussian_1d(window_size, sigma=1.5, device=device, dtype=dtype)
    window_2d = g1d[:, None] @ g1d[None, :]
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Mean structural similarity index between `pred` and `target`, shaped (B, C, H, W)."""
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    channels = pred.shape[1]
    window = _gaussian_window(window_size, channels, pred.device, pred.dtype)
    padding = window_size // 2

    mu_pred = F.conv2d(pred, window, padding=padding, groups=channels)
    mu_target = F.conv2d(target, window, padding=padding, groups=channels)
    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu_pred_sq
    sigma_target_sq = (
        F.conv2d(target * target, window, padding=padding, groups=channels) - mu_target_sq
    )
    sigma_pred_target = (
        F.conv2d(pred * target, window, padding=padding, groups=channels) - mu_pred_target
    )

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)

    return (numerator / denominator).mean()


class SSIMLoss(nn.Module):
    """L_SSIM = 1 - SSIM(pred, target)."""

    def __init__(self, window_size: int = 11, data_range: float = 1.0) -> None:
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - ssim(pred, target, window_size=self.window_size, data_range=self.data_range)
