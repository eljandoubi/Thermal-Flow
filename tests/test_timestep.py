import pytest
import torch

from thermal_flow.models.timestep import SinusoidalTimeEmbedding, TimeEmbedding


def test_sinusoidal_requires_even_dim() -> None:
    with pytest.raises(ValueError):
        SinusoidalTimeEmbedding(dim=7)


def test_sinusoidal_output_shape() -> None:
    embed = SinusoidalTimeEmbedding(dim=16)
    out = embed(torch.rand(4))
    assert out.shape == (4, 16)


def test_sinusoidal_is_deterministic() -> None:
    embed = SinusoidalTimeEmbedding(dim=16)
    t = torch.tensor([0.0, 0.5, 1.0])
    assert torch.equal(embed(t), embed(t))


def test_sinusoidal_distinguishes_timesteps() -> None:
    embed = SinusoidalTimeEmbedding(dim=16)
    out = embed(torch.tensor([0.0, 1.0]))
    assert not torch.allclose(out[0], out[1])


def test_time_embedding_output_shape_and_dim() -> None:
    embed = TimeEmbedding(dim=16, hidden_dim=32)
    out = embed(torch.rand(5))
    assert out.shape == (5, 32)
    assert embed.out_dim == 32


def test_time_embedding_default_hidden_dim() -> None:
    embed = TimeEmbedding(dim=8)
    assert embed.out_dim == 32
