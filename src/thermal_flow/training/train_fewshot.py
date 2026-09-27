"""Few-shot training entry point: restrict the training split to N thermal pairs
(project proposal section 17: 10 / 50 / 100 / 500 / 1000 / Full).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from torch.utils.data import DataLoader

from thermal_flow.datasets.common import PairedRGBThermalDataset
from thermal_flow.datasets.fewshot import few_shot_subset
from thermal_flow.device import get_device
from thermal_flow.losses.combined import CombinedFlowMatchingLoss
from thermal_flow.models.flow_matching import ConditionalFlowMatching
from thermal_flow.training.checkpoint import save_checkpoint
from thermal_flow.training.config import TrainingConfig
from thermal_flow.training.tracking import WandbLogger
from thermal_flow.training.train import build_model, build_optimizer, train_one_epoch


def run_fewshot_training(config: TrainingConfig) -> ConditionalFlowMatching:
    """Same as `run_training`, but restricted to `config.few_shot.num_samples` pairs."""
    device = get_device(config.device)
    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)
    loss_fn = CombinedFlowMatchingLoss(
        lambda_ssim=config.loss.lambda_ssim, lambda_edge=config.loss.lambda_edge
    )

    full_train_dataset = PairedRGBThermalDataset(
        config.dataset.root,
        split=config.dataset.train_split,
        image_size=config.dataset.image_size,
        rgb_channels=config.dataset.rgb_channels,
        thermal_channels=config.dataset.thermal_channels,
    )
    train_dataset = few_shot_subset(
        full_train_dataset, config.few_shot.num_samples, seed=config.few_shot.seed
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    with WandbLogger(
        enabled=config.wandb.enabled,
        project=config.wandb.project,
        name=config.wandb.name,
        entity=config.wandb.entity,
        mode=config.wandb.mode,
        config=asdict(config),
    ) as logger:
        for epoch in range(config.num_epochs):
            avg_loss = train_one_epoch(model, loss_fn, dataloader, optimizer, device)
            logger.log({"train/loss": avg_loss}, step=epoch)
            checkpoint_path = Path(config.checkpoint_dir) / f"fewshot_epoch_{epoch:04d}.pt"
            save_checkpoint(checkpoint_path, model, optimizer, step=epoch)

    return model
