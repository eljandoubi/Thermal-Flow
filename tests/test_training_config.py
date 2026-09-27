from pathlib import Path

import pytest

from thermal_flow.training.config import TrainingConfig


def test_from_dict_applies_defaults() -> None:
    config = TrainingConfig.from_dict({"dataset": {"root": "data/flir"}})

    assert config.dataset.root == "data/flir"
    assert config.dataset.image_size == 256
    assert config.model.base_channels == 96
    assert config.optimizer.lr == 1e-4
    assert config.few_shot.num_samples is None


def test_from_dict_overrides_nested_values() -> None:
    config = TrainingConfig.from_dict(
        {
            "dataset": {"root": "data/flir", "image_size": 128},
            "model": {"base_channels": 32},
            "few_shot": {"num_samples": 50, "seed": 7},
            "batch_size": 2,
        }
    )

    assert config.dataset.image_size == 128
    assert config.model.base_channels == 32
    assert config.few_shot.num_samples == 50
    assert config.few_shot.seed == 7
    assert config.batch_size == 2


def test_from_dict_requires_dataset_root() -> None:
    with pytest.raises(TypeError):
        TrainingConfig.from_dict({})


@pytest.mark.parametrize("name", ["flir.yaml", "llvip.yaml", "fewshot.yaml"])
def test_example_configs_are_valid(name: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = TrainingConfig.from_yaml(repo_root / "configs" / name)
    assert config.dataset.root
