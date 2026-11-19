import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset

from thermal_flow.experiments.study import (
    METHODS,
    CachedPairs,
    main,
    parse_sizes,
    run_study,
)


class _Ramp(Dataset):
    """Ten pairs whose pixel value is the pair index."""

    def __len__(self) -> int:
        return 10

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        value = torch.full((3, 2, 2), index / 255)
        return {"rgb": value, "thermal": value}


def _picked(cached: CachedPairs) -> list[int]:
    return [round(float(cached[i]["rgb"][0, 0, 0]) * 255) for i in range(len(cached))]


def test_cached_pairs_with_a_limit_are_evenly_spaced():
    assert _picked(CachedPairs(_Ramp(), limit=4)) == [0, 3, 6, 9]


def test_cached_pairs_keep_everything_without_a_limit():
    assert _picked(CachedPairs(_Ramp())) == list(range(10))
    assert _picked(CachedPairs(_Ramp(), limit=50)) == list(range(10))


def _make_split(root: Path, split: str, count: int) -> None:
    for i in range(count):
        for kind, color in (("rgb", (10 * i, 20, 30)), ("thermal", (40, 50 + 10 * i, 60))):
            directory = root / split / kind
            directory.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16), color=color).save(directory / f"{i}.png")


@pytest.fixture
def paired_root(tmp_path: Path) -> Path:
    _make_split(tmp_path, "train", 6)
    _make_split(tmp_path, "test", 4)
    return tmp_path


def test_parse_sizes_maps_full_to_none():
    assert parse_sizes(["10", "full", "all"]) == (10, None, None)


def test_study_runs_every_method_and_writes_the_outputs(paired_root: Path, tmp_path: Path):
    out = tmp_path / "out"
    rows = run_study(
        root=str(paired_root),
        sizes=(2, None),
        methods=METHODS,
        steps=2,
        batch_size=2,
        lr=1e-3,
        image_size=16,
        base_channels=8,
        test_images=4,
        seeds=(0,),
        output_dir=str(out),
        device_name="cpu",
    )

    assert len(rows) == 2 * len(METHODS)
    assert {r["method"] for r in rows} == set(METHODS)
    assert {r["train_pairs"] for r in rows} == {2, 6}
    for row in rows:
        assert {"mae", "rmse", "psnr", "ssim", "train_seconds"} <= set(row)
        assert 0 <= row["mae"] <= 1

    assert json.loads((out / "study.json").read_text()) == rows
    efficiency = json.loads((out / "efficiency.json").read_text())
    assert efficiency["flow_matching"]["network_evaluations"] == 8
    assert (out / "study.md").read_text().count("\n") == 2 + 2 * len(METHODS)
    assert (out / "grid_flow_matching_all.png").exists()
    assert (out / "psnr_cnn.png").exists()


def test_cli_entry_point(paired_root: Path, tmp_path: Path):
    out = tmp_path / "cli"
    main(
        [
            "--root",
            str(paired_root),
            "--sizes",
            "2",
            "--methods",
            "cnn",
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--image-size",
            "16",
            "--base-channels",
            "8",
            "--test-images",
            "2",
            "--output",
            str(out),
            "--device",
            "cpu",
        ]
    )
    assert (out / "study.json").exists()
