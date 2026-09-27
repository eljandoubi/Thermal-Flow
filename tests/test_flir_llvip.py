from pathlib import Path

from PIL import Image

from thermal_flow.datasets.flir import FLIRDataset
from thermal_flow.datasets.llvip import LLVIPDataset


def _make_pair(root: Path, split: str, name: str) -> None:
    for kind in ("rgb", "thermal"):
        directory = root / split / kind
        directory.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 10), color=(5, 5, 5)).save(directory / f"{name}.png")


def test_flir_dataset_loads_pairs(tmp_path: Path) -> None:
    _make_pair(tmp_path, "train", "a")

    dataset = FLIRDataset(tmp_path, split="train", image_size=8)

    assert len(dataset) == 1


def test_llvip_dataset_loads_pairs(tmp_path: Path) -> None:
    _make_pair(tmp_path, "train", "a")

    dataset = LLVIPDataset(tmp_path, split="train", image_size=8)

    assert len(dataset) == 1
