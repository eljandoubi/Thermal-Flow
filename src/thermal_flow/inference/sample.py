"""End-to-end RGB -> synthetic thermal sampling via the learned Flow Matching field."""

from __future__ import annotations

from typing import Literal

import torch

from thermal_flow.inference.ode import euler_solve, rk4_solve
from thermal_flow.models.flow_matching import ConditionalFlowMatching

Solver = Literal["euler", "rk4"]

_SOLVERS = {"euler": euler_solve, "rk4": rk4_solve}


@torch.no_grad()
def sample(
    model: ConditionalFlowMatching,
    x_rgb: torch.Tensor,
    num_steps: int = 8,
    solver: Solver = "euler",
) -> torch.Tensor:
    """Generate a synthetic thermal image from an RGB image.

    Solves dx/dt = v_theta(x, t, x_rgb) from x(0) = x_rgb to x(1) ~= thermal, using
    `num_steps` steps of the requested ODE `solver`.
    """
    if solver not in _SOLVERS:
        raise ValueError(f"Unknown solver: {solver!r} (expected one of {tuple(_SOLVERS)})")
    model.eval()

    def velocity_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return model.predict_velocity(x, t, x_rgb)

    return _SOLVERS[solver](velocity_fn, x_rgb, num_steps=num_steps)
