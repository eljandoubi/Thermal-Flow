from pathlib import Path

import pytest
import torch
from PIL import Image

from thermal_flow.datasets.common import PairedRGBThermalDataset


def _write_image(path: Path, size: tuple[int, int], mode: str, color) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, color=color).save(path)


def _make_split(root: Path, split: str, names: list[str], size: tuple[int, int] = (12, 10)) -> None:
    for name in names:
        _write_image(root / split / "rgb" / f"{name}.png", size, "RGB", (10, 20, 30))
        _write_image(root / split / "thermal" / f"{name}.png", size, "RGB", (40, 50, 60))


def test_loads_matching_pairs_with_correct_shape_and_range(tmp_path: Path) -> None:
    _make_split(tmp_path, "train", ["a", "b", "c"])

    dataset = PairedRGBThermalDataset(tmp_path, split="train", image_size=8)

    assert len(dataset) == 3
    sample = dataset[0]
    assert sample["rgb"].shape == (3, 8, 8)
    assert sample["thermal"].shape == (3, 8, 8)
    assert sample["rgb"].dtype == torch.float32
    assert sample["rgb"].min() >= 0.0
    assert sample["rgb"].max() <= 1.0


def test_only_matching_filenames_are_paired(tmp_path: Path) -> None:
    _make_split(tmp_path, "train", ["a", "b"])
    _write_image(tmp_path / "train" / "rgb" / "unmatched.png", (12, 10), "RGB", (1, 2, 3))

    dataset = PairedRGBThermalDataset(tmp_path, split="train", image_size=8)

    assert len(dataset) == 2


def test_missing_split_directory_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PairedRGBThermalDataset(tmp_path, split="train", image_size=8)


def test_no_matching_pairs_raises_clear_error(tmp_path: Path) -> None:
    (tmp_path / "train" / "rgb").mkdir(parents=True)
    (tmp_path / "train" / "thermal").mkdir(parents=True)
    _write_image(tmp_path / "train" / "rgb" / "only_rgb.png", (12, 10), "RGB", (1, 2, 3))

    with pytest.raises(FileNotFoundError):
        PairedRGBThermalDataset(tmp_path, split="train", image_size=8)


def test_single_channel_thermal_loading(tmp_path: Path) -> None:
    _write_image(tmp_path / "train" / "rgb" / "a.png", (12, 10), "RGB", (1, 2, 3))
    _write_image(tmp_path / "train" / "thermal" / "a.png", (12, 10), "L", 100)

    dataset = PairedRGBThermalDataset(tmp_path, split="train", image_size=8, thermal_channels=1)

    assert dataset[0]["thermal"].shape == (1, 8, 8)
