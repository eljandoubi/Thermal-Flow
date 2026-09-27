import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from thermal_flow.losses.combined import CombinedFlowMatchingLoss
from thermal_flow.training.config import TrainingConfig
from thermal_flow.training.train import build_model, build_optimizer, train_one_epoch


class _SyntheticPairedDataset(Dataset):
    """Tiny in-memory RGB/thermal dataset for fast, deterministic training tests."""

    def __init__(self, n: int, channels: int, size: int, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.rgb = torch.rand(n, channels, size, size, generator=generator)
        self.thermal = torch.rand(n, channels, size, size, generator=generator)

    def __len__(self) -> int:
        return self.rgb.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"rgb": self.rgb[index], "thermal": self.thermal[index]}


def _make_config() -> TrainingConfig:
    return TrainingConfig.from_dict(
        {
            "dataset": {
                "root": "unused",
                "image_size": 16,
                "rgb_channels": 3,
                "thermal_channels": 3,
            },
            "model": {"base_channels": 8, "time_dim": 16},
            "optimizer": {"lr": 1e-2},
            "batch_size": 4,
        }
    )


def test_build_model_matches_config() -> None:
    model = build_model(_make_config())
    assert sum(p.numel() for p in model.parameters()) > 0


def test_build_optimizer_only_includes_trainable_params() -> None:
    config = _make_config()
    model = build_model(config)
    next(model.parameters()).requires_grad = False

    optimizer = build_optimizer(model, config)

    n_optimized = sum(p.numel() for group in optimizer.param_groups for p in group["params"])
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_optimized == n_trainable
    assert n_trainable < sum(p.numel() for p in model.parameters())


def test_train_one_epoch_reduces_loss_on_synthetic_overfit_task() -> None:
    torch.manual_seed(0)
    config = _make_config()
    model = build_model(config)
    optimizer = build_optimizer(model, config)
    loss_fn = CombinedFlowMatchingLoss(lambda_ssim=0.0, lambda_edge=0.0)
    dataloader = DataLoader(_SyntheticPairedDataset(n=8, channels=3, size=16), batch_size=4)
    device = torch.device("cpu")

    first_loss = train_one_epoch(model, loss_fn, dataloader, optimizer, device)
    for _ in range(20):
        last_loss = train_one_epoch(model, loss_fn, dataloader, optimizer, device)

    assert last_loss < first_loss


def test_train_one_epoch_rejects_empty_dataloader() -> None:
    config = _make_config()
    model = build_model(config)
    optimizer = build_optimizer(model, config)
    loss_fn = CombinedFlowMatchingLoss()
    dataloader = DataLoader(_SyntheticPairedDataset(n=0, channels=3, size=16), batch_size=4)

    with pytest.raises(ValueError):
        train_one_epoch(model, loss_fn, dataloader, optimizer, torch.device("cpu"))
