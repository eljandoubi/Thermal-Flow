"""Conditional Flow Matching: probability path, velocity target, and training loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConditionalFlowMatching(nn.Module):
    """Wraps a velocity network v_theta(x_t, t, x_rgb) and implements the conditional
    Flow Matching objective from the project proposal:

        x_t = (1 - t) * x_rgb + t * x_thermal        (linear probability path)
        u_t = x_thermal - x_rgb                       (target velocity)
        L_FM = E[ || v_theta(x_t, t, x_rgb) - u_t ||^2 ]
    """

    def __init__(self, velocity_model: nn.Module) -> None:
        super().__init__()
        self.velocity_model = velocity_model

    @staticmethod
    def sample_path(
        x_rgb: torch.Tensor, x_thermal: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x_t, target_velocity) for the linear probability path.

        `x_rgb` and `x_thermal` must have the same shape (see README for why the
        conditional Flow Matching path requires source/target to be directly
        interpolable).
        """
        if x_rgb.shape != x_thermal.shape:
            raise ValueError(
                f"x_rgb and x_thermal must have the same shape, got {tuple(x_rgb.shape)} "
                f"and {tuple(x_thermal.shape)}"
            )
        t_ = t.view(-1, *([1] * (x_rgb.dim() - 1)))
        x_t = (1 - t_) * x_rgb + t_ * x_thermal
        target_velocity = x_thermal - x_rgb
        return x_t, target_velocity

    def predict_velocity(
        self, x_t: torch.Tensor, t: torch.Tensor, x_rgb: torch.Tensor
    ) -> torch.Tensor:
        return self.velocity_model(x_t, t, x_rgb)

    def loss(
        self,
        x_rgb: torch.Tensor,
        x_thermal: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample t ~ U(0, 1) if not given, and compute the Flow Matching MSE loss."""
        if t is None:
            t = torch.rand(x_rgb.shape[0], device=x_rgb.device, dtype=x_rgb.dtype)
        x_t, target_velocity = self.sample_path(x_rgb, x_thermal, t)
        predicted_velocity = self.predict_velocity(x_t, t, x_rgb)
        return F.mse_loss(predicted_velocity, target_velocity)
