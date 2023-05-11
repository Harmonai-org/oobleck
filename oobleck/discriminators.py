import functools
from typing import Callable, Dict, Sequence, Tuple, Union

import cached_conv as cc
import numpy as np
import torch
import torch.nn as nn
import torchaudio

from .blocks import ModuleFactory

TensorDict = Dict[str, torch.Tensor]

IndividualDiscriminatorOut = Tuple[torch.Tensor, Sequence[torch.Tensor]]


class MultiDiscriminator(nn.Module):
    """
    Individual discriminators should take a single tensor as input (NxB C T) and
    return a tuple composed of a score tensor (NxB) and a Sequence of Features
    Sequence[NxB C' T'].
    """

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


class SharedDiscriminatorConvNet(nn.Module):

    def __init__(
        self,
        in_size: int,
        out_size: int,
        capacity: int,
        n_layers: int,
        kernel_size: int,
        stride: int,
        convolution: Union[nn.Conv1d, nn.Conv2d],
        activation: ModuleFactory,
        normalization: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> None:
        super().__init__()
        channels = [in_size]
        channels += list(capacity * 2**np.arange(n_layers))

        if isinstance(stride, int):
            stride = n_layers * [stride]

        net = []
        for i in range(n_layers):
            if isinstance(kernel_size, int):
                pad = kernel_size // 2
                s = stride[i]
            else:
                pad = kernel_size[0] // 2
                s = (stride[i], 1)

            net.append(
                normalization(
                    convolution(
                        channels[i],
                        channels[i + 1],
                        kernel_size,
                        stride=s,
                        padding=pad,
                    )))
            net.append(activation())

        net.append(convolution(channels[-1], out_size, 1))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        features = []
        for layer in self.net:
            x = layer(x)
            if isinstance(layer, nn.modules.conv._ConvNd):
                features.append(x)
        score = features.reshape(features.shape[0], -1).mean(-1)
        return score, features


class MultiScaleDiscriminator(nn.Module):

    def __init__(self,
                 n_scales: int,
                 convnet=Callable[[], SharedDiscriminatorConvNet]) -> None:
        super().__init__()
        layers = []
        for _ in range(n_scales):
            layers.append(convnet())
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> IndividualDiscriminatorOut:
        score = 0
        features = []
        for layer in self.layers:
            s, f = layer(x)
            score = score + s
            features.extend(f)
            x = nn.functional.avg_pool1d(x, 2)
        return score, features


class MultiPeriodDiscriminator(nn.Module):

    def __init__(self,
                 periods: Sequence[int],
                 convnet=Callable[[], SharedDiscriminatorConvNet]) -> None:
        super().__init__()
        layers = []
        self.periods = periods

        for _ in periods:
            layers.append(convnet())

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> IndividualDiscriminatorOut:
        score = 0
        features = []
        for layer, n in zip(self.layers, self.periods):
            s, f = layer(self.fold(x, n))
            score = score + s
            features.extend(f)
        return score, features

    def fold(self, x: torch.Tensor, n: int) -> torch.Tensor:
        pad = (n - (x.shape[-1] % n)) % n
        x = nn.functional.pad(x, (0, pad))
        return x.reshape(*x.shape[:2], -1, n)


class MultiScaleSpectralDiscriminator1d(nn.Module):

    def __init__(
        self,
        scales: Sequence[int],
        convnet: Callable[[int], SharedDiscriminatorConvNet],
        spectrogram: Callable[[int], torchaudio.transforms.Spectrogram],
    ) -> None:
        super().__init__()
        self.specs = nn.ModuleList([spectrogram(n) for n in scales])
        self.nets = nn.ModuleList([convnet(n + 2) for n in scales])

    def forward(self, x):
        score = 0
        features = []
        for spec, net in zip(self.specs, self.nets):
            spec_x = spec(x).squeeze(1)
            spec_x = torch.cat([spec_x.real, spec_x.imag], 1)
            s, f = net(spec_x)
            score = score + s
            features.extend(f)
        return features
