"""Multi-layer Perceptron (MLP) implemented in PyTorch."""

from typing import Type

import torch
import torch.nn as nn


class MLP(torch.nn.Sequential):
    """A Multi-layer Perceptron (MLP) network."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layer_sizes: tuple[int, ...] = (),
        dropout: float = 0.0,
        activation_fn: Type[torch.nn.Module] = nn.ReLU,
    ):
        """Initializes the MLP.

        Args:
            input_size: The number of input features.
            output_size: The number of output features.
            hidden_layer_sizes: A list specifying the number of units in each hidden layer.
            dropout: Dropout probability for hidden layers.
            activation_fn: Activation function to use for hidden layers. Default is ReLU.
        """
        super(MLP, self).__init__()

        self.activation_fn = activation_fn

        # Initialize hidden layers
        layers = []
        prev_size = input_size
        for size in hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(self.activation_fn())
            layers.append(nn.Dropout(dropout))
            prev_size = size

        layers.append(nn.Linear(prev_size, output_size))
        super().__init__(*layers)
