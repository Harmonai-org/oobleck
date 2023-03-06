from typing import Callable, Optional, Type, Union

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
            kernel_size: int,
            dilation: int,
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
