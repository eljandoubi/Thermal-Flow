"""Qualitative RGB / real-thermal / synthetic-thermal comparison grids
(project proposal section 23: visualization/qualitative.py).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend, no display server required
import matplotlib.pyplot as plt
import torch


def _to_displayable(image: torch.Tensor) -> torch.Tensor:
    """(C, H, W) float tensor -> (H, W) or (H, W, C), clamped to [0, 1] for imshow."""
    image = image.detach().cpu().clamp(0, 1)
    if image.shape[0] == 1:
        return image.squeeze(0)
    return image.permute(1, 2, 0)


def save_comparison_grid(
    rgb: torch.Tensor,
    real_thermal: torch.Tensor,
    synthetic_thermal: torch.Tensor,
    output_path: str | Path,
    max_samples: int = 4,
) -> Path:
    """Save a grid with one row per sample: RGB | real thermal | synthetic thermal.

    Each input is a (B, C, H, W) batch of images in [0, 1]; at most `max_samples`
    rows are rendered.
    """
    if not (rgb.shape[0] == real_thermal.shape[0] == synthetic_thermal.shape[0]):
        raise ValueError("rgb, real_thermal and synthetic_thermal must have the same batch size")

    num_samples = min(rgb.shape[0], max_samples)
    columns = (rgb, real_thermal, synthetic_thermal)
    column_titles = ("RGB", "Real thermal", "Synthetic thermal")

    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples), squeeze=False)
    for row in range(num_samples):
        for col, (batch, col_title) in enumerate(zip(columns, column_titles, strict=True)):
            ax = axes[row][col]
            ax.imshow(_to_displayable(batch[row]), cmap="gray" if batch.shape[1] == 1 else None)
            ax.axis("off")
            if row == 0:
                ax.set_title(col_title)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
