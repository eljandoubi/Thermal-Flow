"""Combined Flow Matching + structural + edge objective from the project proposal.

L = L_FM + lambda_ssim * L_SSIM + lambda_edge * L_edge
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from thermal_flow.losses.edge import EdgeLoss
from thermal_flow.losses.structural import SSIMLoss
from thermal_flow.models.flow_matching import ConditionalFlowMatching


class CombinedFlowMatchingLoss(nn.Module):
    """Flow Matching loss plus structural/edge losses on the predicted endpoint.

    L_SSIM and L_edge are computed against a one-step estimate of the final thermal
    image, `x1_hat = x_t + (1 - t) * v_theta(x_t, t, x_rgb)`, which is exact when the
    predicted velocity matches the true (constant, along the linear path) velocity.
    """

    def __init__(self, lambda_ssim: float = 0.1, lambda_edge: float = 0.05) -> None:
        super().__init__()
        self.lambda_ssim = lambda_ssim
        self.lambda_edge = lambda_edge
        self.ssim_loss = SSIMLoss()
        self.edge_loss = EdgeLoss()

    def forward(
        self,
        flow_model: ConditionalFlowMatching,
        x_rgb: torch.Tensor,
        x_thermal: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if t is None:
            t = torch.rand(x_rgb.shape[0], device=x_rgb.device, dtype=x_rgb.dtype)
        x_t, target_velocity = flow_model.sample_path(x_rgb, x_thermal, t)
        predicted_velocity = flow_model.predict_velocity(x_t, t, x_rgb)

        fm_loss = F.mse_loss(predicted_velocity, target_velocity)

        t_ = t.view(-1, *([1] * (x_rgb.dim() - 1)))
        x1_hat = x_t + (1 - t_) * predicted_velocity
        ssim_loss = self.ssim_loss(x1_hat, x_thermal)
        edge_loss = self.edge_loss(x1_hat, x_thermal)

        total = fm_loss + self.lambda_ssim * ssim_loss + self.lambda_edge * edge_loss
        return {"total": total, "fm": fm_loss, "ssim": ssim_loss, "edge": edge_loss}
