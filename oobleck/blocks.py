from typing import Callable, Sequence, Type, Union

import numpy as np
import torch
import torch.nn as nn

ModuleFactory = Union[Type[nn.Module], Callable[[], nn.Module]]


class FeedForwardModule(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.net = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Residual(nn.Module):

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x) + x


class DilatedConvolutionalUnit(FeedForwardModule):

    def __init__(
            self,
            hidden_dim: int,
            dilation: int,
            kernel_size: int,
            activation: ModuleFactory,
            normalization: Callable[[nn.Module],
                                    nn.Module] = lambda x: x) -> None:
        super().__init__()
        self.net = nn.Sequential(
            activation(),
            normalization(
                nn.Conv1d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=((kernel_size - 1) * dilation) // 2,
                )),
            activation(),
            nn.Conv1d(in_channels=hidden_dim,
                      out_channels=hidden_dim,
                      kernel_size=1),
        )


class UpsamplingUnit(FeedForwardModule):

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            stride: int,
            activation: ModuleFactory,
            normalization: Callable[[nn.Module],
                                    nn.Module] = lambda x: x) -> None:
        super().__init__()
        self.net = nn.Sequential(
            activation(),
            normalization(
                nn.ConvTranspose1d(
                    in_channels=input_dim,
                    out_channels=output_dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=stride // 2,
                )))


class DownsamplingUnit(FeedForwardModule):

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            stride: int,
            activation: ModuleFactory,
            normalization: Callable[[nn.Module],
                                    nn.Module] = lambda x: x) -> None:
        super().__init__()
        self.net = nn.Sequential(
            activation(),
            normalization(
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=output_dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=stride // 2,
                )))


class DilatedResidualEncoder(FeedForwardModule):

    def __init__(
            self,
            capacity: int,
            dilated_unit: Type[DilatedConvolutionalUnit],
            downsampling_unit: Type[DownsamplingUnit],
            ratios: Sequence[int],
            dilations: Union[Sequence[int], Sequence[Sequence[int]]],
            pre_network_conv: Type[nn.Conv1d],
            post_network_conv: Type[nn.Conv1d],
            normalization: Callable[[nn.Module],
                                    nn.Module] = lambda x: x) -> None:
        super().__init__()
        channels = capacity * 2**np.arange(len(ratios) + 1)

        dilations_list = self.normalize_dilations(dilations, ratios)

        net = [normalization(pre_network_conv(out_channels=channels[0]))]

        for ratio, dilations, input_dim, output_dim in zip(
                ratios, dilations_list, channels[:-1], channels[1:]):
            for dilation in dilations:
                net.append(Residual(dilated_unit(input_dim, dilation)))
            net.append(downsampling_unit(input_dim, output_dim, ratio))

        net.append(post_network_conv(in_channels=output_dim))

        self.net = nn.Sequential(*net)

    @staticmethod
    def normalize_dilations(dilations: Union[Sequence[int],
                                             Sequence[Sequence[int]]],
                            ratios: Sequence[int]):
        if isinstance(dilations[0], int):
            dilations = [dilations for _ in ratios]
        return dilations


class DilatedResidualDecoder(FeedForwardModule):

    def __init__(
            self,
            capacity: int,
            dilated_unit: Type[DilatedConvolutionalUnit],
            upsampling_unit: Type[UpsamplingUnit],
            ratios: Sequence[int],
            dilations: Union[Sequence[int], Sequence[Sequence[int]]],
            pre_network_conv: Type[nn.Conv1d],
            post_network_conv: Type[nn.Conv1d],
            normalization: Callable[[nn.Module],
                                    nn.Module] = lambda x: x) -> None:
        super().__init__()
        channels = capacity * 2**np.arange(len(ratios) + 1)
        channels = channels[::-1]

        dilations_list = self.normalize_dilations(dilations, ratios)
        dilations_list = dilations_list[::-1]

        net = [pre_network_conv(out_channels=channels[0])]

        for ratio, dilations, input_dim, output_dim in zip(
                ratios, dilations_list, channels[:-1], channels[1:]):
            net.append(upsampling_unit(input_dim, output_dim, ratio))
            for dilation in dilations:
                net.append(Residual(dilated_unit(output_dim, dilation)))

        net.append(normalization(post_network_conv(in_channels=output_dim)))

        self.net = nn.Sequential(*net)

    @staticmethod
    def normalize_dilations(dilations: Union[Sequence[int],
                                             Sequence[Sequence[int]]],
                            ratios: Sequence[int]):
        if isinstance(dilations[0], int):
            dilations = [dilations for _ in ratios]
        return dilations
