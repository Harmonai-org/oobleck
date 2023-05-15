import auraloss
import pytest
import torch

from oobleck import losses


def test_debug():
    waveform = torch.randn(1, 1, 2**14)

    inputs = {"waveform": waveform, "reconstruction": waveform}
    loss = losses.DebugLoss("waveform", "reconstruction")(inputs)
    assert loss["generator_loss"].item() == 0.

    inputs = {
        "waveform": waveform,
        "reconstruction": torch.randn_like(waveform)
    }
    inputs = losses.DebugLoss("waveform", "reconstruction")(inputs)
    assert inputs["generator_loss"].item() > 0.


def test_debug_vae():
    waveform = torch.randn(1, 1, 2**14)
    latent_mean = torch.randn(1, 16, 2**10)
    latent_std = torch.randn(1, 16, 2**10).div(2).exp()

    inputs = {
        "waveform": waveform,
        "reconstruction": waveform,
        "latent_mean": torch.zeros_like(latent_mean),
        "latent_std": torch.ones_like(latent_std),
    }

    inputs = losses.DebugLossVae("waveform", "reconstruction")(inputs)
    assert inputs["generator_loss"].item() == 0.

    inputs = {
        "waveform": waveform,
        "reconstruction": torch.randn_like(waveform),
        "latent_mean": latent_mean,
        "latent_std": latent_std,
    }

    inputs = losses.DebugLossVae("waveform", "reconstruction")(inputs)
    assert inputs["generator_loss"].item() > 0.


auraloss_modules = [
    auraloss.freq.SumAndDifferenceSTFTLoss(
        fft_sizes=[512, 256, 128],
        hop_sizes=[128, 64, 32],
        win_lengths=[512, 256, 128],
    ),
    auraloss.freq.SpectralConvergenceLoss(),
    auraloss.freq.RandomResolutionSTFTLoss(),
]


@pytest.mark.parametrize(
    "loss_module",
    auraloss_modules,
    ids=map(lambda x: x.__class__.__name__, auraloss_modules),
)
def test_auraloss_wrapper(loss_module):
    waveform = torch.randn(1, 2, 2**16)

    inputs = {
        "waveform": waveform,
        "reconstruction": waveform,
    }

    wrapper = losses.AuralossWrapper("waveform", "reconstruction",
                                     lambda: loss_module)
    inputs = wrapper(inputs)
    assert inputs["generator_loss"].item() == 0.

    inputs = {
        "waveform": waveform,
        "reconstruction": torch.randn_like(waveform),
    }

    inputs = wrapper(inputs)
    assert inputs["generator_loss"].item() != 0.
