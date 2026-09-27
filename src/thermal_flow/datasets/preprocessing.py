"""Image preprocessing shared by all dataset loaders."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def normalize_to_unit_range(image: torch.Tensor) -> torch.Tensor:
    """Convert a uint8 [0, 255] image tensor to float32 [0, 1]."""
    return image.float() / 255.0


def resize(image: torch.Tensor, size: int) -> torch.Tensor:
    """Bilinearly resize a (C, H, W) tensor to (C, size, size)."""
    x = image.unsqueeze(0)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x.squeeze(0)
