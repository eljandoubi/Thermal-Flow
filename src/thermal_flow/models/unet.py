"""Lightweight conditional U-Net used as the Flow Matching velocity network."""

from __future__ import annotations

import torch
from torch import nn

from thermal_flow.models.adapter import BottleneckAdapter
from thermal_flow.models.timestep import TimeEmbedding


class ResBlock(nn.Module):
    """Residual block with an additive FiLM-style time-embedding bias."""

    def __init__(self, channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class Downsample(nn.Module):
    """Stride-2 3x3 convolution halving spatial resolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    """Stride-2 4x4 transposed convolution doubling spatial resolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class ConditionalUNet(nn.Module):
    """Predicts the Flow Matching velocity field v_theta(x_t, t, x_rgb).

    `x_t` (interpolated thermal) and `x_rgb` (RGB conditioning) are concatenated
    channel-wise at the input; the output has `thermal_channels` channels, matching
    the target velocity `x_thermal - x_rgb`. Input height/width must be divisible by
    4 (two downsampling stages). `base_channels` controls model capacity: the default
    lands in the ~5-15M parameter "lightweight" range.

    If `use_adapter` is set, a `BottleneckAdapter` is inserted after the bottleneck
    ResBlock so the model can later be few-shot finetuned via
    `thermal_flow.models.adapter.freeze_for_adapter_finetuning`.
    """

    def __init__(
        self,
        thermal_channels: int = 3,
        rgb_channels: int = 3,
        base_channels: int = 96,
        time_dim: int = 128,
        use_adapter: bool = False,
        adapter_dim: int = 16,
    ) -> None:
        super().__init__()
        self.thermal_channels = thermal_channels
        self.time_embedding = TimeEmbedding(time_dim)
        emb_dim = self.time_embedding.out_dim
        self.adapter = BottleneckAdapter(base_channels, adapter_dim) if use_adapter else None

        self.in_conv = nn.Conv2d(
            thermal_channels + rgb_channels, base_channels, kernel_size=3, padding=1
        )

        self.down1 = ResBlock(base_channels, emb_dim)
        self.downsample1 = Downsample(base_channels)
        self.down2 = ResBlock(base_channels, emb_dim)
        self.downsample2 = Downsample(base_channels)

        self.bottleneck = ResBlock(base_channels, emb_dim)

        self.upsample1 = Upsample(base_channels)
        self.up1 = ResBlock(base_channels, emb_dim)
        self.upsample2 = Upsample(base_channels)
        self.up2 = ResBlock(base_channels, emb_dim)

        self.out_conv = nn.Conv2d(base_channels, thermal_channels, kernel_size=3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, x_rgb: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedding(t)
        h = self.in_conv(torch.cat([x_t, x_rgb], dim=1))

        h1 = self.down1(h, t_emb)
        h = self.downsample1(h1)
        h2 = self.down2(h, t_emb)
        h = self.downsample2(h2)

        h = self.bottleneck(h, t_emb)
        if self.adapter is not None:
            h = self.adapter(h)

        h = self.up1(self.upsample1(h) + h2, t_emb)
        h = self.up2(self.upsample2(h) + h1, t_emb)

        return self.out_conv(h)
