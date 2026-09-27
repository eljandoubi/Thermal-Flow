"""Baseline 3 - Pix2Pix conditional GAN, RGB -> Thermal (proposal section 14).

A well-established image-to-image translation baseline: a U-Net generator trained
against a conditional PatchGAN discriminator.
"""

from __future__ import annotations

import torch
from torch import nn

from thermal_flow.baselines.unet import UNetBaseline


class Pix2PixGenerator(UNetBaseline):
    """U-Net generator (identical architecture to `UNetBaseline`)."""


class PatchGANDiscriminator(nn.Module):
    """Conditional PatchGAN discriminator: classifies (x_rgb, x_thermal) patches as real/fake."""

    def __init__(
        self, rgb_channels: int = 3, thermal_channels: int = 3, base_channels: int = 64
    ) -> None:
        super().__init__()

        def block(in_ch: int, out_ch: int, normalize: bool = True) -> list[nn.Module]:
            layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(rgb_channels + thermal_channels, base_channels, normalize=False),
            *block(base_channels, base_channels * 2),
            *block(base_channels * 2, base_channels * 4),
            nn.Conv2d(base_channels * 4, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x_rgb: torch.Tensor, x_thermal: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_rgb, x_thermal], dim=1))
