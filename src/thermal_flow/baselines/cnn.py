"""Baseline 1 - simple CNN regression, RGB -> Thermal (project proposal section 14)."""

from __future__ import annotations

import torch
from torch import nn


class CNNRegressionBaseline(nn.Module):
    """A small stack of 3x3 convolutions directly regressing thermal from RGB.

    No downsampling/skip-connections: the simplest possible pixel-level baseline.
    """

    def __init__(
        self,
        rgb_channels: int = 3,
        thermal_channels: int = 3,
        hidden_channels: int = 64,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2, got {num_layers}")

        layers: list[nn.Module] = [
            nn.Conv2d(rgb_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Conv2d(hidden_channels, thermal_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
        return self.net(x_rgb)
