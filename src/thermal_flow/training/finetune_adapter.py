"""Adapter finetuning entry point: freeze the pretrained backbone, train only the
bottleneck adapter on a small number of thermal pairs.
"""

from __future__ import annotations

from pathlib import Path

from thermal_flow.device import get_device
from thermal_flow.models.adapter import freeze_for_adapter_finetuning
from thermal_flow.models.flow_matching import ConditionalFlowMatching
from thermal_flow.training.checkpoint import load_checkpoint
from thermal_flow.training.config import TrainingConfig
from thermal_flow.training.train import build_model, fit
from thermal_flow.training.train_fewshot import load_few_shot_dataset


def run_adapter_finetuning(
    config: TrainingConfig, pretrained_checkpoint: str | Path
) -> ConditionalFlowMatching:
    """Load a backbone-only pretrained checkpoint into an adapter-enabled model,
    freeze everything but the adapter, and finetune on `config.few_shot.num_samples`
    thermal pairs.
    """
    if not config.model.use_adapter:
        raise ValueError("config.model.use_adapter must be True for adapter finetuning")

    device = get_device(config.device)
    model = build_model(config).to(device)
    # the pretrained checkpoint predates the adapter, so its keys are a strict
    # subset of the adapter-enabled model's state dict.
    load_checkpoint(pretrained_checkpoint, model, map_location=device, strict=False)
    freeze_for_adapter_finetuning(model)

    return fit(config, model, load_few_shot_dataset(config), device, "adapter_epoch")
