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
        loss_function: Callable[[TensorDict], torch.Tensor],
        discriminator: Optional[ModuleFactory] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder()
        self.decoder = decoder()

        if discriminator is not None:
            self.discriminator = discriminator()
        else:
            self.discriminator = None

        self.loss_function = loss_function

    def forward(self, inputs: TensorDict) -> TensorDict:
        encoder_out = self.encoder(inputs)
        inputs.update(encoder_out)

        decoder_out = self.decoder(inputs)
        inputs.update(decoder_out)

        if self.discriminator is not None:
            discriminator_out = self.discriminator(inputs)
            inputs.update(discriminator_out)

        return inputs

    def loss(self, inputs: TensorDict) -> Tuple[torch.Tensor, TensorDict]:
        outputs = self.forward(inputs)
        return self.loss_function(outputs), outputs
