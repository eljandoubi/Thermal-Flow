"""Checkpoint save/load for model + optimizer state.

`torch.load` is called with `weights_only=True` (the safe default): only plain
tensors/numbers/dicts/lists are restored, never arbitrary pickled objects. Keep
any `extra` payload passed to `save_checkpoint` limited to such simple types.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    step: int = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"model": model.state_dict(), "step": step}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    map_location: str | torch.device | None = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint into `model` (and `optimizer`, if given).

    `strict=False` allows loading a backbone-only checkpoint into a model that
    has extra submodules (e.g. an adapter); the missing/unexpected keys are
    returned so callers can verify only the expected keys differ.
    """
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    load_result = model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return {
        "step": payload.get("step", 0),
        "extra": payload.get("extra"),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }
