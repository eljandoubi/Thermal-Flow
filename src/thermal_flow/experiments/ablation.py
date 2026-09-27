"""Generic named-configuration ablation runner: Flow Matching vs. direct regression,
loss terms, adaptation strategy, ODE steps, and resolution are all "evaluate each
named configuration" studies.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeVar

T = TypeVar("T")


def run_ablation(
    configs: Mapping[str, T], evaluate_fn: Callable[[T], dict[str, float]]
) -> dict[str, dict[str, float]]:
    """Evaluate `evaluate_fn(config)` for every named config, e.g.:

    run_ablation(
        {"FM": fm_config, "FM+SSIM": fm_ssim_config, "FM+SSIM+Edge": full_config},
        evaluate_fn=train_and_evaluate,
    )
    """
    return {name: evaluate_fn(config) for name, config in configs.items()}
