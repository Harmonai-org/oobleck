from typing import Sequence, Tuple

import pytest
import torch
import torch.nn as nn

from oobleck import discriminators, utils


def test_multi_discriminator():

    num_features = 4

    class DummyDiscriminator(nn.Module):

        def forward(
                self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, Sequence[torch.Tensor]]:
            features = []
            for i in range(num_features):
                features.append(x.clone() * i)
            score = x.reshape(x.shape[0], -1).mean(-1)
            return score, features

    md = discriminators.MultiDiscriminator(
        [DummyDiscriminator, DummyDiscriminator],
        ["real", "fake"],
    )

    inputs = {"real": torch.ones(4, 1, 2), "fake": -torch.ones(4, 1, 2)}

    score, features = md(inputs)

    assert score["real"].mean().item() == 2.
    assert score["fake"].mean().item() == -2.

    for i, feature in enumerate(features["features_real"]):
        assert feature.mean() == i % num_features
    for i, feature in enumerate(features["features_fake"]):
        assert feature.mean() == -(i % num_features)


def test_multi_scale_discriminator():
    discriminator = discriminators.MultiScaleDiscriminator(
        n_scales=4,
        convnet=lambda: discriminators.SharedDiscriminatorConvNet(
            in_size=1,
            out_size=1,
            capacity=2,
            n_layers=4,
            kernel_size=15,
            stride=4,
            convolution=nn.Conv1d,
            activation=lambda: nn.LeakyReLU(.2),
            normalization=nn.utils.weight_norm,
        ),
    )

    x = torch.randn(1, 1, 2**14)
    score, features = discriminator(x)

    assert score.shape[0] == x.shape[0]


def test_multi_period_discriminator():
    discriminator = discriminators.MultiPeriodDiscriminator(
        periods=[2, 3, 5, 7],
        convnet=lambda: discriminators.SharedDiscriminatorConvNet(
            in_size=1,
            out_size=1,
            capacity=2,
            n_layers=4,
            kernel_size=15,
            stride=4,
            convolution=nn.Conv2d,
            activation=lambda: nn.LeakyReLU(.2),
            normalization=nn.utils.weight_norm,
        ),
    )

    x = torch.randn(1, 1, 2**14)
    score, features = discriminator(x)

    assert score.shape[0] == x.shape[0]


def test_multi_scale_spectral_discriminator():
    discriminator = discriminators.MultiScaleSpectralDiscriminator(
        scales=[128, 256, 512],
        convnet=lambda in_size: discriminators.SharedDiscriminatorConvNet(
            in_size=in_size,
            out_size=1,
            capacity=2,
            n_layers=4,
            kernel_size=15,
            stride=4,
            convolution=nn.Conv2d,
            activation=lambda: nn.LeakyReLU(.2),
            normalization=nn.utils.weight_norm,
        ),
        spectrogram=utils.get_spectrogram,
        use_2d_conv=True,
    )

    x = torch.randn(1, 1, 2**14)
    score, features = discriminator(x)

    assert score.shape[0] == x.shape[0]


def test_multi_discriminator_scale_period():
    multiscale = discriminators.MultiScaleDiscriminator(
        n_scales=4,
        convnet=lambda: discriminators.SharedDiscriminatorConvNet(
            in_size=1,
            out_size=1,
            capacity=2,
            n_layers=4,
            kernel_size=15,
            stride=4,
            convolution=nn.Conv1d,
            activation=lambda: nn.LeakyReLU(.2),
            normalization=nn.utils.weight_norm,
        ),
    )

    multiperiod = discriminators.MultiPeriodDiscriminator(
        periods=[2, 3, 5, 7],
        convnet=lambda: discriminators.SharedDiscriminatorConvNet(
            in_size=1,
            out_size=1,
            capacity=2,
            n_layers=4,
            kernel_size=15,
            stride=4,
            convolution=nn.Conv2d,
            activation=lambda: nn.LeakyReLU(.2),
            normalization=nn.utils.weight_norm,
        ),
    )

    discriminator = discriminators.MultiDiscriminator(
        [lambda: multiscale, lambda: multiperiod],
        ["real", "fake"],
    )

    scores, features = discriminator({
        "real": torch.randn(1, 1, 2**16),
        "fake": torch.randn(1, 1, 2**16),
    })
