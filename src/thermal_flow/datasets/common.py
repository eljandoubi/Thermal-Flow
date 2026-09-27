"""Generic paired RGB/Thermal image dataset over a normalized on-disk layout.

Expected layout (raw dataset downloads must be converted/symlinked into this form;
see `FLIRDataset` / `LLVIPDataset` for dataset-specific notes)::

    root/
      <split>/
        rgb/<name>.<ext>
        thermal/<name>.<ext>

RGB and thermal images are paired by identical filename stem within a split.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from thermal_flow.datasets.preprocessing import normalize_to_unit_range, resize

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _list_images(directory: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(directory.iterdir()) if p.suffix.lower() in _IMAGE_EXTENSIONS}


def _load_image(path: Path, channels: int) -> torch.Tensor:
    mode = "L" if channels == 1 else "RGB"
    with Image.open(path) as img:
        img = img.convert(mode)
        width, height = img.size
        data = img.tobytes()
    array = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    if channels == 1:
        return array.view(1, height, width)
    return array.view(height, width, channels).permute(2, 0, 1).contiguous()


class PairedRGBThermalDataset(Dataset):
    """Paired RGB/thermal samples for one dataset split, resized+normalized to [0, 1]."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 256,
        rgb_channels: int = 3,
        thermal_channels: int = 3,
    ) -> None:
        self.root = Path(root)
        self.split_dir = self.root / split
        self.rgb_dir = self.split_dir / "rgb"
        self.thermal_dir = self.split_dir / "thermal"
        if not self.rgb_dir.is_dir() or not self.thermal_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {self.rgb_dir} and {self.thermal_dir} to exist (paired "
                "RGB/thermal layout, see PairedRGBThermalDataset's docstring)."
            )

        rgb_images = _list_images(self.rgb_dir)
        thermal_images = _list_images(self.thermal_dir)
        shared = sorted(set(rgb_images) & set(thermal_images))
        if not shared:
            raise FileNotFoundError(
                f"No matching RGB/thermal filename pairs found under {self.split_dir}."
            )

        self.pairs = [(rgb_images[name], thermal_images[name]) for name in shared]
        self.image_size = image_size
        self.rgb_channels = rgb_channels
        self.thermal_channels = thermal_channels

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rgb_path, thermal_path = self.pairs[index]
        x_rgb = normalize_to_unit_range(_load_image(rgb_path, self.rgb_channels))
        x_thermal = normalize_to_unit_range(_load_image(thermal_path, self.thermal_channels))
        return {
            "rgb": resize(x_rgb, self.image_size),
            "thermal": resize(x_thermal, self.image_size),
        }
