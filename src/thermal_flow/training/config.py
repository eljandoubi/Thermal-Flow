"""Training configuration schema, loaded from YAML or a plain dict."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    root: str
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    image_size: int = 256
    rgb_channels: int = 3
    thermal_channels: int = 3


@dataclass
class ModelConfig:
    base_channels: int = 96
    time_dim: int = 128
    use_adapter: bool = False
    adapter_dim: int = 16


@dataclass
class OptimizerConfig:
    lr: float = 1e-4
    weight_decay: float = 0.0


@dataclass
class LossConfig:
    lambda_ssim: float = 0.1
    lambda_edge: float = 0.05


@dataclass
class FewShotConfig:
    num_samples: int | None = None
    seed: int = 0


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "thermal-flow"
    entity: str | None = None
    name: str | None = None
    mode: str = "online"  # "online", "offline", or "disabled"


@dataclass
class TrainingConfig:
    dataset: DatasetConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    few_shot: FewShotConfig = field(default_factory=FewShotConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    batch_size: int = 8
    num_epochs: int = 100
    num_workers: int = 4
    seed: int = 0
    device: str | None = None  # None => auto-detect (thermal_flow.device.get_device)
    checkpoint_dir: str = "checkpoints"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainingConfig:
        raw = dict(raw)
        dataset = DatasetConfig(**raw.pop("dataset", {}))
        model = ModelConfig(**raw.pop("model", {}))
        optimizer = OptimizerConfig(**raw.pop("optimizer", {}))
        loss = LossConfig(**raw.pop("loss", {}))
        few_shot = FewShotConfig(**raw.pop("few_shot", {}))
        wandb = WandbConfig(**raw.pop("wandb", {}))
        return cls(
            dataset=dataset,
            model=model,
            optimizer=optimizer,
            loss=loss,
            few_shot=few_shot,
            wandb=wandb,
            **raw,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)
