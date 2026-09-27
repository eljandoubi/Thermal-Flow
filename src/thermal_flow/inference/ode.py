"""ODE solvers for integrating the Flow Matching velocity field at inference time."""

from __future__ import annotations

from collections.abc import Callable

import torch

VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@torch.no_grad()
def euler_solve(velocity_fn: VelocityFn, x0: torch.Tensor, num_steps: int = 8) -> torch.Tensor:
    """Integrate dx/dt = velocity_fn(x, t) from t=0 to t=1 with the explicit Euler method.

    `velocity_fn(x, t)` takes the current state and a (batch,) tensor of timesteps and
    returns a tensor shaped like `x`.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    dt = 1.0 / num_steps
    batch = x0.shape[0]
    x = x0
    for step in range(num_steps):
        t = torch.full((batch,), step * dt, device=x.device, dtype=x.dtype)
        x = x + dt * velocity_fn(x, t)
    return x


@torch.no_grad()
def rk4_solve(velocity_fn: VelocityFn, x0: torch.Tensor, num_steps: int = 8) -> torch.Tensor:
    """Integrate dx/dt = velocity_fn(x, t) from t=0 to t=1 with the classic RK4 method.

    More accurate than `euler_solve` for the same `num_steps`, at 4x the velocity
    network evaluations per step.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    dt = 1.0 / num_steps
    batch = x0.shape[0]
    x = x0
    for step in range(num_steps):
        t0 = step * dt
        t_start = torch.full((batch,), t0, device=x.device, dtype=x.dtype)
        t_mid = torch.full((batch,), t0 + 0.5 * dt, device=x.device, dtype=x.dtype)
        t_end = torch.full((batch,), t0 + dt, device=x.device, dtype=x.dtype)

        k1 = velocity_fn(x, t_start)
        k2 = velocity_fn(x + 0.5 * dt * k1, t_mid)
        k3 = velocity_fn(x + 0.5 * dt * k2, t_mid)
        k4 = velocity_fn(x + dt * k3, t_end)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x
