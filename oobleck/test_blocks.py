import itertools
from typing import Callable, Sequence, Union

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.nn import utils

from oobleck import blocks


def identity(module):
    return module


@pytest.mark.parametrize(
    "batch_size,residual,hidden_dim,kernel_size,dilation,activation,normalization",
    itertools.product([1, 4], [True, False], [16], [3, 5, 7], [1, 3, 9],
                      [nn.ReLU], [identity, utils.weight_norm]))
def test_dilated_convolutional_unit(batch_size: int, residual: bool,
                                    hidden_dim: int, kernel_size: int,
                                    dilation: int,
                                    activation: blocks.ModuleFactory,
                                    normalization: Callable[[nn.Module],
                                                            nn.Module]):
    model = blocks.DilatedConvolutionalUnit(hidden_dim, dilation, kernel_size,
                                            activation, normalization)

    if residual:
        model = blocks.Residual(model)

    x = torch.randn(batch_size, hidden_dim, 128)
    y = model(x)

    assert x.shape == y.shape


@pytest.mark.parametrize(
    "batch_size,input_dim,output_dim,stride,activation,normalization",
    itertools.product([1, 4], [4, 16], [4, 16], [2, 4, 8], [nn.ReLU],
                      [identity, utils.weight_norm]))
def test_upsampling_unit(batch_size: int, input_dim: int, output_dim: int,
                         stride: int, activation: blocks.ModuleFactory,
                         normalization: Callable[[nn.Module], nn.Module]):
    model = blocks.UpsamplingUnit(input_dim, output_dim, stride, activation,
                                  normalization)

    x = torch.randn(batch_size, input_dim, 128)
    y = model(x)

    assert x.shape[0] == y.shape[0]
    assert y.shape[1] == output_dim
    assert x.shape[-1] * stride == y.shape[-1]


@pytest.mark.parametrize(
    "batch_size,input_dim,output_dim,stride,activation,normalization",
    itertools.product([1, 4], [4, 16], [4, 16], [2, 4, 8], [nn.ReLU],
                      [identity, utils.weight_norm]))
def test_downsampling_unit(batch_size: int, input_dim: int, output_dim: int,
                           stride: int, activation: blocks.ModuleFactory,
                           normalization: Callable[[nn.Module], nn.Module]):
    model = blocks.DownsamplingUnit(input_dim, output_dim, stride, activation,
                                    normalization)

    x = torch.randn(batch_size, input_dim, 128)
    y = model(x)

    assert x.shape[0] == y.shape[0]
    assert y.shape[1] == output_dim
    assert x.shape[-1] // stride == y.shape[-1]


@pytest.mark.parametrize(
    "batch_size,data_size,latent_size,capacity,ratios,dilations",
    itertools.product([1, 4], [1, 2], [16, 32], [2, 4], [[4, 4, 2], [2, 2]],
                      [[1, 3, 9]]))
def test_dilated_residual_waveform_encoder(
        batch_size: int, data_size: int, latent_size: int, capacity: int,
        ratios: Sequence[int], dilations: Union[Sequence[int],
                                                Sequence[Sequence[int]]]):

    def dilated_unit(hidden_dim, dilation):
        return blocks.DilatedConvolutionalUnit(hidden_dim,
                                               dilation,
                                               kernel_size=3,
                                               activation=nn.ReLU,
                                               normalization=utils.weight_norm)

    def downsampling_unit(input_dim: int, output_dim: int, stride: int):
        return blocks.DownsamplingUnit(input_dim,
                                       output_dim,
                                       stride,
                                       nn.ReLU,
                                       normalization=utils.weight_norm)

    def pre_conv(out_channels):
        return nn.Conv1d(data_size, out_channels, 1)

    def post_conv(in_channels):
        return nn.Conv1d(in_channels, latent_size, 1)

    model = blocks.DilatedResidualEncoder(
        capacity=capacity,
        dilated_unit=dilated_unit,
        downsampling_unit=downsampling_unit,
        ratios=ratios,
        dilations=dilations,
        pre_network_conv=pre_conv,
        post_network_conv=post_conv,
    )

    x = torch.randn(batch_size, data_size, 256)
    y = model(x)

    assert y.shape[0] == batch_size
    assert y.shape[1] == latent_size
    assert x.shape[-1] // np.prod(ratios) == y.shape[-1]


@pytest.mark.parametrize(
    "batch_size,data_size,latent_size,capacity,ratios,dilations",
    itertools.product([1, 4], [1, 2], [16, 32], [2, 4], [[4, 4, 2], [2, 2]],
                      [[1, 3, 9]]))
def test_dilated_residual_waveform_decoder(
        batch_size: int, data_size: int, latent_size: int, capacity: int,
        ratios: Sequence[int], dilations: Union[Sequence[int],
                                                Sequence[Sequence[int]]]):

    def dilated_unit(hidden_dim, dilation):
        return blocks.DilatedConvolutionalUnit(hidden_dim,
                                               dilation,
                                               kernel_size=3,
                                               activation=nn.ReLU,
                                               normalization=utils.weight_norm)

    def upsampling_unit(input_dim: int, output_dim: int, stride: int):
        return blocks.UpsamplingUnit(input_dim,
                                     output_dim,
                                     stride,
                                     nn.ReLU,
                                     normalization=utils.weight_norm)

    def pre_conv(out_channels):
        return nn.Conv1d(latent_size, out_channels, 1)

    def post_conv(in_channels):
        return nn.Conv1d(in_channels, data_size, 1)

    model = blocks.DilatedResidualDecoder(
        capacity=capacity,
        dilated_unit=dilated_unit,
        upsampling_unit=upsampling_unit,
        ratios=ratios,
        dilations=dilations,
        pre_network_conv=pre_conv,
        post_network_conv=post_conv,
    )

    x = torch.randn(batch_size, latent_size, 4)
    y = model(x)

    assert y.shape[0] == batch_size
    assert y.shape[1] == data_size
    assert x.shape[-1] * np.prod(ratios) == y.shape[-1]
