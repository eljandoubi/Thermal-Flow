"""Sinusoidal timestep embeddings for conditioning the flow matching U-Net."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Maps a scalar t in [0, 1] to a `dim`-dimensional sinusoidal embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) float tensor in [0, 1] -> (B, dim)."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        # scale to a wider range, similar to diffusion timestep embeddings
        args = t[:, None] * freqs[None, :] * 1000.0
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding followed by a learned 2-layer MLP projection."""

    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.sinusoidal = SinusoidalTimeEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_dim = hidden_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(t))
