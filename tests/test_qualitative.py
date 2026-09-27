from pathlib import Path

import pytest
import torch

from thermal_flow.visualization.qualitative import save_comparison_grid


def test_save_comparison_grid_creates_file(tmp_path: Path) -> None:
    rgb = torch.rand(3, 3, 16, 16)
    real_thermal = torch.rand(3, 3, 16, 16)
    synthetic_thermal = torch.rand(3, 3, 16, 16)
    output_path = tmp_path / "grid.png"

    returned_path = save_comparison_grid(rgb, real_thermal, synthetic_thermal, output_path)

    assert returned_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_comparison_grid_respects_max_samples(tmp_path: Path) -> None:
    rgb = torch.rand(10, 3, 8, 8)
    real_thermal = torch.rand(10, 3, 8, 8)
    synthetic_thermal = torch.rand(10, 3, 8, 8)
    output_path = tmp_path / "grid_capped.png"

    save_comparison_grid(rgb, real_thermal, synthetic_thermal, output_path, max_samples=2)

    assert output_path.exists()


def test_save_comparison_grid_supports_single_channel_thermal(tmp_path: Path) -> None:
    rgb = torch.rand(2, 3, 8, 8)
    real_thermal = torch.rand(2, 1, 8, 8)
    synthetic_thermal = torch.rand(2, 1, 8, 8)
    output_path = tmp_path / "grid_1ch.png"

    save_comparison_grid(rgb, real_thermal, synthetic_thermal, output_path)

    assert output_path.exists()


def test_save_comparison_grid_rejects_batch_size_mismatch(tmp_path: Path) -> None:
    rgb = torch.rand(2, 3, 8, 8)
    real_thermal = torch.rand(3, 3, 8, 8)
    synthetic_thermal = torch.rand(2, 3, 8, 8)

    with pytest.raises(ValueError):
        save_comparison_grid(rgb, real_thermal, synthetic_thermal, tmp_path / "x.png")
