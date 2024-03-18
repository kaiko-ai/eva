"""MLP network tests."""

import itertools
from typing import Tuple

import pytest
import torch

from eva.core.models.networks import MLP


@pytest.mark.parametrize(
    "input_size, output_size, hidden_layer_sizes, dropout",
    list(itertools.product([8], [2, 4], [(), (5,), (5, 5)], [0.0, 0.5])),
)
def test_mlp_initialization(
    input_size: int, output_size: int, hidden_layer_sizes: Tuple[int, ...], dropout: float
):
    """Tests intitializing mlp with different parameters."""
    mlp = MLP(
        input_size=input_size,
        output_size=output_size,
        hidden_layer_sizes=hidden_layer_sizes,
        dropout=dropout,
        hidden_activation_fn=torch.nn.ReLU,
    )

    k = 3 if dropout > 0 else 2  # Linear, Activation, (Dropout)
    expected_n_layers = k * len(hidden_layer_sizes) + 1  # + 1 for the output layer
    assert len(list(mlp._network.modules())) == expected_n_layers + 1  # + 1 for the MLP itself


@pytest.mark.parametrize(
    "batch_size, input_size, output_size, hidden_layer_sizes, dropout",
    list(itertools.product([1, 4], [8], [1, 4], [(), (5, 5)], [0.5])),
)
def test_mlp_forward_pass(
    batch_size: int,
    input_size: int,
    output_size: int,
    hidden_layer_sizes: Tuple[int, ...],
    dropout: float,
):
    """Tests forward pass with different hyper parameters, input & output sizes & batch sizes."""
    mlp = MLP(
        input_size=input_size,
        output_size=output_size,
        hidden_layer_sizes=hidden_layer_sizes,
        dropout=dropout,
        hidden_activation_fn=torch.nn.ReLU,
    )

    # Create a dummy input tensor
    x = torch.randn(batch_size, input_size)
    output = mlp(x)

    # Check the output size
    assert output.shape == (batch_size, output_size)
