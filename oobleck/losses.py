from typing import Dict

import torch

TensorDict = Dict[str, torch.Tensor]


def debug_loss(inputs: TensorDict, input_key: str,
               output_key: str) -> torch.Tensor:
    return (inputs[input_key] - inputs[output_key]).abs().mean()
