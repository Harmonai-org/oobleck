import pathlib

import gin
import pytest
import torch

from oobleck import AudioAutoEncoder

gin.enter_interactive_mode()

configurations = pathlib.Path("oobleck/configs").rglob("*.gin")
configurations = list(map(str, configurations))

input_data = {
    "waveform": torch.randn(1, 1, 2**16),
}


@pytest.mark.parametrize("config", configurations)
def test_models(config):
    gin.clear_config()
    gin.parse_config_file(config)

    model = AudioAutoEncoder()

    model.loss(input_data)
