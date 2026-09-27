import pytest
import torch

from thermal_flow.losses.edge import EdgeLoss, image_gradients


def test_image_gradients_zero_for_constant_image() -> None:
    x = torch.full((1, 3, 16, 16), 0.5)
    grads = image_gradients(x)
    # zero-padding introduces a false edge exactly at the border; check the interior.
    interior = grads[..., 1:-1, 1:-1]
    assert torch.allclose(interior, torch.zeros_like(interior), atol=1e-6)


def test_edge_loss_zero_for_identical_images() -> None:
    x = torch.rand(2, 3, 16, 16)
    loss = EdgeLoss()
    assert torch.isclose(loss(x, x), torch.tensor(0.0), atol=1e-6)


def test_edge_loss_positive_for_different_images() -> None:
    x = torch.zeros(1, 1, 16, 16)
    y = torch.zeros(1, 1, 16, 16)
    y[:, :, :, 8:] = 1.0  # sharp vertical edge

    assert EdgeLoss()(x, y) > 0


def test_edge_loss_rejects_shape_mismatch() -> None:
    x = torch.rand(1, 3, 16, 16)
    y = torch.rand(1, 1, 16, 16)
    with pytest.raises(ValueError):
        EdgeLoss()(x, y)
