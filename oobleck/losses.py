from typing import Dict

import torch

TensorDict = Dict[str, torch.Tensor]


def debug_loss(inputs: TensorDict) -> torch.Tensor:
    return (inputs["input"] - inputs["output"]).abs().mean()
