from typing import Dict

import torch

TensorDict = Dict[str, torch.Tensor]


def debug_loss(inputs: TensorDict, input_key: str,
               output_key: str) -> torch.Tensor:
    l1 = (inputs[input_key] - inputs[output_key]).abs().mean()
    return {"generator_loss": l1}


def debug_loss_vae(inputs: TensorDict,
                   input_key: str,
                   output_key: str,
                   beta_kl: float = 1.) -> torch.Tensor:
    l1 = (inputs[input_key] - inputs[output_key]).abs().mean()

    mean, std = inputs["latent_mean"], inputs["latent_std"]
    var = std.pow(2)
    logvar = torch.log(var)

    kl = (mean.pow(2) + var - logvar - 1).sum(1).mean() * beta_kl
    return {"generator_loss": l1 + kl, "kl_divergence": kl}
