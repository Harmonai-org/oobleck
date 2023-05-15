import os
import pathlib

import gin
import pytest
import torch

from oobleck import AudioAutoEncoder

gin.enter_interactive_mode()

configurations = pathlib.Path("oobleck/configs").rglob("*.gin")
configurations = list(map(str, configurations))
ids = map(os.path.basename, configurations)

input_data = {
    "waveform": torch.randn(1, 1, 2**16),
}


@pytest.mark.parametrize("config", configurations, ids=ids)
def test_models(config):
    gin.clear_config()
    gin.parse_config_file(config)

    model = AudioAutoEncoder()

    inputs = model.loss(input_data)

    assert "generator_loss" in inputs
    assert "discriminator_loss" in inputs
