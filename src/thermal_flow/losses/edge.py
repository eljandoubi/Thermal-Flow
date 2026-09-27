"""Edge-preservation loss based on Sobel image gradients."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

_SOBEL_X = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
_SOBEL_Y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]


def _sobel_kernels(
    channels: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    kx = torch.tensor(_SOBEL_X, device=device, dtype=dtype).expand(channels, 1, 3, 3).contiguous()
    ky = torch.tensor(_SOBEL_Y, device=device, dtype=dtype).expand(channels, 1, 3, 3).contiguous()
    return kx, ky


def image_gradients(x: torch.Tensor) -> torch.Tensor:
    """Sobel (grad_x, grad_y) stacked along the channel dim: (B, C, H, W) -> (B, 2*C, H, W)."""
    channels = x.shape[1]
    kx, ky = _sobel_kernels(channels, x.device, x.dtype)
    grad_x = F.conv2d(x, kx, padding=1, groups=channels)
    grad_y = F.conv2d(x, ky, padding=1, groups=channels)
    return torch.cat([grad_x, grad_y], dim=1)


class EdgeLoss(nn.Module):
    """L_edge = || grad(pred) - grad(target) ||_1."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
        return (image_gradients(pred) - image_gradients(target)).abs().mean()
