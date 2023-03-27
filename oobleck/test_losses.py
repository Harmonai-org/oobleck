import pytest
import torch

import auraloss
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
    loss = losses.DebugLoss("waveform", "reconstruction")(inputs)
    assert loss["generator_loss"].item() > 0.


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

    loss = losses.DebugLossVae("waveform", "reconstruction")(inputs)
    assert loss["generator_loss"].item() == 0.

    inputs = {
        "waveform": waveform,
        "reconstruction": torch.randn_like(waveform),
        "latent_mean": latent_mean,
        "latent_std": latent_std,
    }

    loss = losses.DebugLossVae("waveform", "reconstruction")(inputs)
    assert loss["generator_loss"].item() > 0.

def test_auraloss_wrapper():
    waveform = torch.randn(1, 2, 2**14)

    scales = [2048, 1024, 512, 256, 128]
    hop_sizes = []
    win_lengths = []
    overlap = 0.75
    for s in scales:
        hop_sizes.append(int(s * (1 - overlap)))
        win_lengths.append(s)

    auraloss_stft_loss = auraloss.freq.SumAndDifferenceSTFTLoss(
        fft_sizes=scales,
        hop_sizes=hop_sizes,
        win_lengths=win_lengths
    )

    inputs = {"waveform": waveform, "reconstruction": waveform}
    loss = losses.AuralossWrapper("waveform", "reconstruction", auraloss_stft_loss)(inputs)
    assert loss["generator_loss"].item() == 0.

    inputs = {
        "waveform": waveform,
        "reconstruction": torch.randn_like(waveform)
    }

    loss = losses.AuralossWrapper("waveform", "reconstruction", auraloss_stft_loss)(inputs)
    assert loss["generator_loss"].item() > 0.