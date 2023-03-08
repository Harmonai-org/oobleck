from typing import Sequence, Tuple

import pytest
import torch
import torch.nn as nn

from oobleck import discriminators


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
