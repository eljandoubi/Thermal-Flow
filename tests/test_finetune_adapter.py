from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from thermal_flow.training.checkpoint import save_checkpoint
from thermal_flow.training.config import TrainingConfig
from thermal_flow.training.finetune_adapter import run_adapter_finetuning
from thermal_flow.training.train import build_model


def _make_dataset(root: Path, n: int) -> None:
    for i in range(n):
        for kind, color in (("rgb", (10, 20, 30)), ("thermal", (40, 50, 60))):
            directory = root / "train" / kind
            directory.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16), color=color).save(directory / f"{i}.png")


def _base_config_dict(tmp_path: Path, use_adapter: bool) -> dict[str, Any]:
    return {
        "dataset": {
            "root": str(tmp_path),
            "image_size": 8,
            "rgb_channels": 3,
            "thermal_channels": 3,
        },
        "model": {
            "base_channels": 8,
            "time_dim": 8,
            "use_adapter": use_adapter,
            "adapter_dim": 2,
        },
        "few_shot": {"num_samples": 2, "seed": 0},
        "batch_size": 2,
        "num_epochs": 1,
        "num_workers": 0,
        "device": "cpu",
        "checkpoint_dir": str(tmp_path / "checkpoints"),
    }


def test_run_adapter_finetuning_freezes_backbone_and_updates_adapter(tmp_path: Path) -> None:
    _make_dataset(tmp_path, n=4)

    pretrained_config = TrainingConfig.from_dict(_base_config_dict(tmp_path, use_adapter=False))
    pretrained_model = build_model(pretrained_config)
    pretrained_path = tmp_path / "pretrained.pt"
    save_checkpoint(pretrained_path, pretrained_model, step=0)
    backbone_before = {name: p.clone() for name, p in pretrained_model.named_parameters()}

    adapter_config = TrainingConfig.from_dict(_base_config_dict(tmp_path, use_adapter=True))
    model = run_adapter_finetuning(adapter_config, pretrained_path)

    adapter_moved = False
    for name, p in model.named_parameters():
        if "adapter" in name:
            if "up_proj" in name and not torch.allclose(p, torch.zeros_like(p)):
                adapter_moved = True
        else:
            assert torch.allclose(p, backbone_before[name]), f"backbone param {name} changed"
    assert adapter_moved, "expected adapter's zero-initialized up_proj to move during finetuning"


def test_run_adapter_finetuning_requires_adapter_enabled(tmp_path: Path) -> None:
    _make_dataset(tmp_path, n=2)
    config = TrainingConfig.from_dict(_base_config_dict(tmp_path, use_adapter=False))
    with pytest.raises(ValueError):
        run_adapter_finetuning(config, tmp_path / "does-not-matter.pt")
