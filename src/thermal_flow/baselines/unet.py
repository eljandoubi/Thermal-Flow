"""Baseline 2 - standard encoder-decoder U-Net, RGB -> Thermal (proposal section 14).

Unlike `thermal_flow.models.unet.ConditionalUNet`, this is a plain, non-time-
conditioned direct regressor: it tests whether Flow Matching provides an advantage
over standard encoder-decoder translation.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Two 3x3 conv + ReLU layers."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetBaseline(nn.Module):
    """Plain U-Net baseline: direct RGB -> Thermal regression.

    Input height/width must be divisible by 4 (two pooling stages).
    """

    def __init__(
        self, rgb_channels: int = 3, thermal_channels: int = 3, base_channels: int = 64
    ) -> None:
        super().__init__()
        self.enc1 = ConvBlock(rgb_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        self.out_conv = nn.Conv2d(base_channels, thermal_channels, kernel_size=1)

    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x_rgb)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)
