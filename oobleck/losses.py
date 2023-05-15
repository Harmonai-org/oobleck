from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from .blocks import ModuleFactory

TensorDict = Dict[str, torch.Tensor]


def accumulate_value(inputs: TensorDict, update: TensorDict):
    for k, v in update.items():
        if k in inputs:
            inputs[k] += v
        else:
            inputs[k] = v
    return inputs


class DebugLoss(nn.Module):

    def __init__(self,
                 input_key: str,
                 output_key: str,
                 weight: float = 1.) -> None:
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.weight = weight

    def forward(self, inputs: TensorDict) -> TensorDict:
        l1 = (inputs[self.input_key] - inputs[self.output_key]).abs().mean()
        inputs = accumulate_value(
            inputs,
            {
                "generator_loss": l1 * self.weight,
            },
        )
        return inputs


class DebugLossVae(nn.Module):

    def __init__(self,
                 input_key: str,
                 output_key: str,
                 beta_kl: float = 1.,
                 weight: float = 1.) -> None:
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.beta_kl = beta_kl
        self.weight = weight

    def forward(self, inputs: TensorDict) -> TensorDict:
        l1 = (inputs[self.input_key] - inputs[self.output_key]).abs().mean()

        mean, std = inputs["latent_mean"], inputs["latent_std"]
        var = std.pow(2)
        logvar = torch.log(var)

        kl = (mean.pow(2) + var - logvar - 1).sum(1).mean()
        inputs = accumulate_value(
            inputs,
            {
                "generator_loss": self.weight * (l1 + self.beta_kl * kl),
            },
        )
        inputs.update({"kl_divergence": kl})
        return inputs


class KLDivergenceVAE(nn.Module):

    def __init__(self, weight: float = 1.) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, inputs: TensorDict) -> TensorDict:
        mean, std = inputs["latent_mean"], inputs["latent_std"]
        var = std.pow(2)
        logvar = torch.log(var)

        kl = (mean.pow(2) + var - logvar - 1).sum(1).mean()

        inputs = accumulate_value(
            inputs,
            {
                "generator_loss": self.weight * kl,
            },
        )
        inputs.update({"kl": kl})

        return inputs


class HingeGan(nn.Module):

    def __init__(self,
                 real_key: str,
                 fake_key: str,
                 weight: float = 1.) -> None:
        super().__init__()
        self.real_key = real_key
        self.fake_key = fake_key
        self.weight = weight

    def forward(self, inputs: TensorDict) -> TensorDict:
        score_real = inputs[f"score_{self.real_key}"]
        score_fake = inputs[f"score_{self.fake_key}"]

        gen_loss = -score_fake.mean()
        dis_loss = torch.relu(1 - score_real).mean() + torch.relu(
            1 + score_fake).mean()

        inputs = accumulate_value(
            inputs,
            {
                "generator_loss": gen_loss * self.weight,
                "discriminator_loss": dis_loss,
            },
        )

        return inputs


class AuralossWrapper(nn.Module):

    def __init__(
        self,
        input_key: str,
        output_key: str,
        auraloss_module: ModuleFactory,
        weight: float = 1.,
        name: Optional[str] = None,
    ) -> None:

        super().__init__()
        self.input_key = input_key
        self.output_key = output_key

        self.loss = auraloss_module()
        self.weight = weight

        if name is None:
            name = self.loss.__class__.__name__

        self.name = name

    def forward(self, inputs: TensorDict) -> torch.Tensor:
        loss_value = self.loss(inputs[self.input_key], inputs[self.output_key])
        inputs = accumulate_value(
            inputs,
            {
                "generator_loss": loss_value * self.weight,
            },
        )
        inputs.update({self.name: loss_value})
        return inputs


class CombineLosses(nn.Module):

    def __init__(self, loss_modules: Sequence[ModuleFactory]) -> None:
        super().__init__()
        self.loss_modules = nn.ModuleList(
            [module() for module in loss_modules])

    def forward(self, inputs: TensorDict) -> TensorDict:
        for loss in self.loss_modules:
            inputs = loss(inputs)
        return inputs
