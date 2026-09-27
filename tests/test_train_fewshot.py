from pathlib import Path

from PIL import Image

from thermal_flow.training.config import TrainingConfig
from thermal_flow.training.train_fewshot import run_fewshot_training


def _make_dataset(root: Path, n: int) -> None:
    for i in range(n):
        for kind, color in (("rgb", (10, 20, 30)), ("thermal", (40, 50, 60))):
            directory = root / "train" / kind
            directory.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16), color=color).save(directory / f"{i}.png")


def test_run_fewshot_training_respects_num_samples_and_checkpoints(tmp_path: Path) -> None:
    _make_dataset(tmp_path, n=6)
    config = TrainingConfig.from_dict(
        {
            "dataset": {
                "root": str(tmp_path),
                "image_size": 8,
                "rgb_channels": 3,
                "thermal_channels": 3,
            },
            "model": {"base_channels": 8, "time_dim": 8},
            "few_shot": {"num_samples": 3, "seed": 0},
            "batch_size": 2,
            "num_epochs": 1,
            "num_workers": 0,
            "device": "cpu",
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        }
    )

    model = run_fewshot_training(config)

    assert sum(p.numel() for p in model.parameters()) > 0
    assert (tmp_path / "checkpoints" / "fewshot_epoch_0000.pt").exists()
