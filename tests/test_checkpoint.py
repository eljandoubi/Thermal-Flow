from pathlib import Path

import torch

from thermal_flow.models.unet import ConditionalUNet
from thermal_flow.training.checkpoint import load_checkpoint, save_checkpoint


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(3, 4)).sum().backward()
    optimizer.step()

    checkpoint_path = tmp_path / "ckpt.pt"
    save_checkpoint(checkpoint_path, model, optimizer, step=5, extra={"note": "test"})

    new_model = torch.nn.Linear(4, 2)
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    metadata = load_checkpoint(checkpoint_path, new_model, new_optimizer)

    assert metadata["step"] == 5
    assert metadata["extra"] == {"note": "test"}
    for p1, p2 in zip(model.parameters(), new_model.parameters(), strict=True):
        assert torch.equal(p1, p2)


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    checkpoint_path = tmp_path / "nested" / "dir" / "ckpt.pt"

    save_checkpoint(checkpoint_path, model, step=0)

    assert checkpoint_path.exists()


def test_load_without_optimizer_only_restores_model(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    checkpoint_path = tmp_path / "ckpt.pt"
    save_checkpoint(checkpoint_path, model, step=1)

    new_model = torch.nn.Linear(2, 2)
    metadata = load_checkpoint(checkpoint_path, new_model)

    assert metadata["step"] == 1
    for p1, p2 in zip(model.parameters(), new_model.parameters(), strict=True):
        assert torch.equal(p1, p2)


def test_load_non_strict_for_adapter_finetuning_reports_only_adapter_missing(
    tmp_path: Path,
) -> None:
    base_model = ConditionalUNet(thermal_channels=3, rgb_channels=3, base_channels=8, time_dim=16)
    checkpoint_path = tmp_path / "pretrained.pt"
    save_checkpoint(checkpoint_path, base_model, step=10)

    adapter_model = ConditionalUNet(
        thermal_channels=3,
        rgb_channels=3,
        base_channels=8,
        time_dim=16,
        use_adapter=True,
        adapter_dim=4,
    )
    metadata = load_checkpoint(checkpoint_path, adapter_model, strict=False)

    assert metadata["step"] == 10
    assert metadata["missing_keys"]
    assert all("adapter" in key for key in metadata["missing_keys"])
    assert metadata["unexpected_keys"] == []
