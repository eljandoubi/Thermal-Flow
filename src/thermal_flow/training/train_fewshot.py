"""Few-shot training entry point: restrict the training split to N thermal pairs
(10 / 50 / 100 / 500 / 1000 / Full).
"""

from __future__ import annotations

from torch.utils.data import Dataset

from thermal_flow.datasets.fewshot import few_shot_subset
from thermal_flow.device import get_device
from thermal_flow.models.flow_matching import ConditionalFlowMatching
from thermal_flow.training.config import TrainingConfig
from thermal_flow.training.train import build_model, fit, load_train_dataset


def load_few_shot_dataset(config: TrainingConfig) -> Dataset:
    return few_shot_subset(
        load_train_dataset(config),
        config.few_shot.num_samples,
        seed=config.few_shot.seed,
    )


def run_fewshot_training(config: TrainingConfig) -> ConditionalFlowMatching:
    """Same as `run_training`, but restricted to `config.few_shot.num_samples` pairs."""
    device = get_device(config.device)
    model = build_model(config).to(device)
    return fit(config, model, load_few_shot_dataset(config), device, "fewshot_epoch")
