from typing import Callable, Dict, Optional, Tuple

import gin
import torch
import torch.nn as nn

from .blocks import ModuleFactory

TensorDict = Dict[str, torch.Tensor]


class SimpleTensorDictWrapper(nn.Module):

    def __init__(self, input_key: str, output_key: str,
                 model: ModuleFactory) -> None:
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.model = model()

    def __repr__(self):
        return self.model.__repr__()

    def forward(self, inputs: TensorDict):
        return {self.output_key: self.model(inputs[self.input_key])}


class VariationalEncoder(nn.Module):

    def __init__(self, input_key: str, model: ModuleFactory) -> None:
        super().__init__()
        self.input_key = input_key
        self.model = model()

    def forward(self, inputs: TensorDict) -> TensorDict:
        encoder_out: torch.Tensor = self.model(inputs[self.input_key])
        mean, std = encoder_out.chunk(2, 1)

        std = nn.functional.softplus(std) + 1e-5

        latent = torch.randn_like(mean) * std + mean

        return {"latent": latent, "latent_mean": mean, "latent_std": std}


@gin.configurable
class AudioAutoEncoder(nn.Module):

    def __init__(
        self,
        encoder: ModuleFactory,
        decoder: ModuleFactory,
        loss_module: ModuleFactory,
        discriminator: Optional[ModuleFactory] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder()
        self.decoder = decoder()

        if discriminator is not None:
            self.discriminator = discriminator()
        else:
            self.discriminator = None

        self.loss_module = loss_module()

    def forward(self, inputs: TensorDict) -> TensorDict:
        inputs.update(self.encoder(inputs))
        inputs.update(self.decoder(inputs))

        if self.discriminator is not None:
            inputs.update(self.discriminator(inputs))

        return inputs

    def loss(self, inputs: TensorDict) -> TensorDict:
        inputs = self.forward(inputs)
        inputs.update(self.loss_module(inputs))
        return inputs
