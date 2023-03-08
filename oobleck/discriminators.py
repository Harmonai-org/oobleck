import functools
from typing import Dict, Sequence

import torch
import torch.nn as nn

from .blocks import ModuleFactory

TensorDict = Dict[str, torch.Tensor]


class MultiDiscriminator(nn.Module):

    def __init__(self, discriminator_list: Sequence[ModuleFactory],
                 keys: Sequence[str]) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList([d() for d in discriminator_list])
        self.keys = keys

    def unpack_tensor_to_dict(self, features: torch.Tensor) -> TensorDict:
        features = features.chunk(len(self.keys), 0)
        return {k: features[i] for i, k in enumerate(self.keys)}

    @staticmethod
    def concat_dicts(dict_a, dict_b):
        out_dict = {}
        keys = set(list(dict_a.keys()) + list(dict_b.keys()))
        for k in keys:
            out_dict[k] = []
            if k in dict_a:
                if isinstance(dict_a[k], list):
                    out_dict[k].extend(dict_a[k])
                else:
                    out_dict[k].append(dict_a[k])
            if k in dict_b:
                if isinstance(dict_b[k], list):
                    out_dict[k].extend(dict_b[k])
                else:
                    out_dict[k].append(dict_b[k])
        return out_dict

    @staticmethod
    def sum_dicts(dict_a, dict_b):
        out_dict = {}
        keys = set(list(dict_a.keys()) + list(dict_b.keys()))
        for k in keys:
            out_dict[k] = 0.
            if k in dict_a:
                out_dict[k] = out_dict[k] + dict_a[k]
            if k in dict_b:
                out_dict[k] = out_dict[k] + dict_b[k]
        return out_dict

    def forward(self, inputs: TensorDict) -> TensorDict:
        discriminator_input = torch.cat([inputs[k] for k in self.keys], 0)
        all_scores = []
        all_features = []

        for discriminator in self.discriminators:
            score, features = discriminator(discriminator_input)
            all_scores.append(self.unpack_tensor_to_dict(score))

            features = map(self.unpack_tensor_to_dict, features)
            features = functools.reduce(self.concat_dicts, features)
            features = {f"features_{k}": features[k] for k in features.keys()}
            all_features.append(features)

        all_scores = functools.reduce(self.sum_dicts, all_scores)
        all_features = functools.reduce(self.concat_dicts, all_features)
        return all_scores, all_features
