import math

import pytest
import torch

from thermal_flow.inference.ode import euler_solve, rk4_solve


def test_constant_velocity_field_is_exact_for_euler_and_rk4() -> None:
    x0 = torch.zeros(2, 3)
    velocity = torch.full_like(x0, 2.0)

    def velocity_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return velocity

    for solver in (euler_solve, rk4_solve):
        for steps in (1, 4, 16):
            x1 = solver(velocity_fn, x0, num_steps=steps)
            assert torch.allclose(x1, x0 + velocity, atol=1e-5)


def test_rk4_is_more_accurate_than_euler_for_exponential_growth() -> None:
    # dx/dt = x  =>  x(1) = x0 * e
    x0 = torch.ones(1, 4)

    def velocity_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x

    target = x0 * math.e
    euler_error = (euler_solve(velocity_fn, x0, num_steps=8) - target).abs().max()
    rk4_error = (rk4_solve(velocity_fn, x0, num_steps=8) - target).abs().max()

    assert rk4_error < euler_error


def test_solvers_reject_invalid_num_steps() -> None:
    def velocity_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x

    with pytest.raises(ValueError):
        euler_solve(velocity_fn, torch.zeros(1, 2), num_steps=0)
    with pytest.raises(ValueError):
        rk4_solve(velocity_fn, torch.zeros(1, 2), num_steps=0)


def test_output_preserves_shape() -> None:
    x0 = torch.randn(3, 3, 8, 8)

    def velocity_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    assert euler_solve(velocity_fn, x0, num_steps=4).shape == x0.shape
    assert rk4_solve(velocity_fn, x0, num_steps=4).shape == x0.shape
