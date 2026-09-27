"""Core training loop and full-supervision training entry point."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from thermal_flow.datasets.common import PairedRGBThermalDataset
from thermal_flow.device import get_device
from thermal_flow.losses.combined import CombinedFlowMatchingLoss
from thermal_flow.models.flow_matching import ConditionalFlowMatching
from thermal_flow.models.unet import ConditionalUNet
from thermal_flow.training.checkpoint import save_checkpoint
from thermal_flow.training.config import TrainingConfig


def build_model(config: TrainingConfig) -> ConditionalFlowMatching:
    unet = ConditionalUNet(
        thermal_channels=config.dataset.thermal_channels,
        rgb_channels=config.dataset.rgb_channels,
        base_channels=config.model.base_channels,
        time_dim=config.model.time_dim,
        use_adapter=config.model.use_adapter,
        adapter_dim=config.model.adapter_dim,
    )
    return ConditionalFlowMatching(unet)


def build_optimizer(model: torch.nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        params, lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay
    )


def train_one_epoch(
    model: ConditionalFlowMatching,
    loss_fn: CombinedFlowMatchingLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one pass over `dataloader`, returning the average total loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch in dataloader:
        x_rgb = batch["rgb"].to(device)
        x_thermal = batch["thermal"].to(device)

        optimizer.zero_grad()
        losses = loss_fn(model, x_rgb, x_thermal)
        losses["total"].backward()
        optimizer.step()

        total_loss += losses["total"].item()
        num_batches += 1

    if num_batches == 0:
        raise ValueError("dataloader produced no batches")
    return total_loss / num_batches


def run_training(config: TrainingConfig) -> ConditionalFlowMatching:
    """Full-supervision training entry point (loads the entire training split)."""
    device = get_device(config.device)
    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)
    loss_fn = CombinedFlowMatchingLoss(
        lambda_ssim=config.loss.lambda_ssim, lambda_edge=config.loss.lambda_edge
    )

    train_dataset = PairedRGBThermalDataset(
        config.dataset.root,
        split=config.dataset.train_split,
        image_size=config.dataset.image_size,
        rgb_channels=config.dataset.rgb_channels,
        thermal_channels=config.dataset.thermal_channels,
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    for epoch in range(config.num_epochs):
        train_one_epoch(model, loss_fn, dataloader, optimizer, device)
        checkpoint_path = Path(config.checkpoint_dir) / f"epoch_{epoch:04d}.pt"
        save_checkpoint(checkpoint_path, model, optimizer, step=epoch)

    return model
