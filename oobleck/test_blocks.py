import itertools
from typing import Callable

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
    model = blocks.DilatedConvolutionalUnit(hidden_dim, kernel_size, dilation,
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
