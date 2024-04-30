"""Multi-layer Perceptron (MLP) implemented in PyTorch."""

from typing import Tuple, Type

import torch
import torch.nn as nn


class MLP(nn.Module):
    """A Multi-layer Perceptron (MLP) network."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layer_sizes: Tuple[int, ...] | None = None,
        hidden_activation_fn: Type[torch.nn.Module] | None = nn.ReLU,
        output_activation_fn: Type[torch.nn.Module] | None = None,
        dropout: float = 0.0,
    ) -> None:
        """Initializes the MLP.

        Args:
            input_size: The number of input features.
            output_size: The number of output features.
            hidden_layer_sizes: A list specifying the number of units in each hidden layer.
            dropout: Dropout probability for hidden layers.
            hidden_activation_fn: Activation function to use for hidden layers. Default is ReLU.
            output_activation_fn: Activation function to use for the output layer. Default is None.
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes if hidden_layer_sizes is not None else ()
        self.hidden_activation_fn = hidden_activation_fn
        self.output_activation_fn = output_activation_fn
        self.dropout = dropout

        self._network = self._build_network()

    def _build_network(self) -> nn.Sequential:
        """Builds the neural network's layers and returns a nn.Sequential container."""
        layers = []
        prev_size = self.input_size
        for size in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            if self.hidden_activation_fn is not None:
                layers.append(self.hidden_activation_fn())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev_size = size

        layers.append(nn.Linear(prev_size, self.output_size))
        if self.output_activation_fn is not None:
            layers.append(self.output_activation_fn())

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the MLP.

        Args:
            x: The input tensor.

        Returns:
            The output of the network.
        """
        return self._network(x)
