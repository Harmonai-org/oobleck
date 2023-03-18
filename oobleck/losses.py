from typing import Dict

import torch
import torch.nn as nn

TensorDict = Dict[str, torch.Tensor]


class DebugLoss(nn.Module):

    def __init__(self, input_key: str, output_key: str) -> None:
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key

    def forward(self, inputs: TensorDict) -> torch.Tensor:
        l1 = (inputs[self.input_key] - inputs[self.output_key]).abs().mean()
        return {"generator_loss": l1}


class DebugLossVae(nn.Module):

    def __init__(self,
                 input_key: str,
                 output_key: str,
                 beta_kl: float = 1.) -> None:
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.beta_kl = beta_kl

    def forward(self, inputs: TensorDict) -> torch.Tensor:
        l1 = (inputs[self.input_key] - inputs[self.output_key]).abs().mean()

        mean, std = inputs["latent_mean"], inputs["latent_std"]
        var = std.pow(2)
        logvar = torch.log(var)

        kl = (mean.pow(2) + var - logvar - 1).sum(1).mean() * self.beta_kl
        return {"generator_loss": l1 + kl, "kl_divergence": kl}
