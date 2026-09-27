"""Parameter-efficient bottleneck adapter for few-shot fine-tuning.

Instead of retraining the full Flow Matching model when only a handful of
thermal images are available, freeze the pretrained encoder/decoder and only
train a small bottleneck adapter inserted at the U-Net bottleneck.
"""

from __future__ import annotations

import torch
from torch import nn


class BottleneckAdapter(nn.Module):
    """z' = z + W2 * sigma(W1 * z), applied per-pixel via 1x1 convolutions."""

    def __init__(self, channels: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.down_proj = nn.Conv2d(channels, bottleneck_dim, kernel_size=1)
        self.act = nn.SiLU()
        self.up_proj = nn.Conv2d(bottleneck_dim, channels, kernel_size=1)
        # zero-init the up-projection: the adapter starts as a no-op so
        # adapter finetuning begins from the pretrained model's behavior.
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.up_proj(self.act(self.down_proj(z)))


def freeze_for_adapter_finetuning(model: nn.Module) -> None:
    """Freeze every parameter except those belonging to `BottleneckAdapter` submodules."""
    adapter_param_ids = {
        id(param)
        for module in model.modules()
        if isinstance(module, BottleneckAdapter)
        for param in module.parameters()
    }
    for param in model.parameters():
        param.requires_grad = id(param) in adapter_param_ids


def count_trainable_parameters(model: nn.Module) -> int:
    """Number of parameters with `requires_grad=True`."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
