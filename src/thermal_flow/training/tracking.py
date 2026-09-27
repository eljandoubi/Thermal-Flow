"""Optional Weights & Biases experiment tracking.

`wandb` is only imported when actually enabled, so training/tests never pay its
import cost (or need an account) unless `WandbConfig.enabled=True`.
"""

from __future__ import annotations

from typing import Any


class WandbLogger:
    """Thin wrapper around the wandb SDK; a no-op when `enabled=False`."""

    def __init__(
        self,
        enabled: bool,
        project: str = "thermal-flow",
        name: str | None = None,
        entity: str | None = None,
        mode: str = "online",
        config: dict[str, Any] | None = None,
    ) -> None:
        self.enabled = enabled
        if not enabled:
            return

        import wandb

        self._wandb = wandb
        self._run = wandb.init(
            project=project, name=name, entity=entity, mode=mode, config=config or {}
        )

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled:
            return
        self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if not self.enabled:
            return
        self._wandb.finish()

    def __enter__(self) -> WandbLogger:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.finish()
