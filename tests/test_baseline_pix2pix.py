import torch

from thermal_flow.baselines.pix2pix import PatchGANDiscriminator, Pix2PixGenerator


def test_generator_forward_shape() -> None:
    gen = Pix2PixGenerator(rgb_channels=3, thermal_channels=3, base_channels=8)
    out = gen(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 3, 32, 32)


def test_discriminator_output_is_a_patch_map() -> None:
    disc = PatchGANDiscriminator(rgb_channels=3, thermal_channels=3, base_channels=8)
    x_rgb = torch.randn(2, 3, 64, 64)
    x_thermal = torch.randn(2, 3, 64, 64)

    out = disc(x_rgb, x_thermal)

    assert out.shape[0] == 2
    assert out.shape[1] == 1
    assert out.shape[2] > 1
    assert out.shape[3] > 1


def test_discriminator_gradients_flow() -> None:
    disc = PatchGANDiscriminator(base_channels=8)
    x_rgb = torch.randn(1, 3, 64, 64)
    x_thermal = torch.randn(1, 3, 64, 64)

    disc(x_rgb, x_thermal).sum().backward()

    assert all(p.grad is not None for p in disc.parameters())
