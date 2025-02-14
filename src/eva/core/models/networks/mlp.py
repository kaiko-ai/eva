"""Multi-layer Perceptron (MLP) implemented in PyTorch."""

from typing import Tuple, Type

import torch
import torch.nn as nn


class MLP(nn.Module):
    """A Multi-layer Perceptron (MLP) network."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layer_sizes: Tuple[int, ...] | None = None,
        hidden_activation_fn: Type[torch.nn.Module] | None = nn.ReLU,
        output_activation_fn: Type[torch.nn.Module] | None = None,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
    ) -> None:
        """Initializes the MLP.

        Args:
            in_features: The number of input features.
            out_features: The number of output features.
            hidden_layer_sizes: A list specifying the number of units in each hidden layer.
            dropout: Dropout probability for hidden layers.
            hidden_activation_fn: Activation function to use for hidden layers. Default is ReLU.
            output_activation_fn: Activation function to use for the output layer. Default is None.
            use_batch_norm: Wether to apply batch norm after the hidden layers.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_layer_sizes = hidden_layer_sizes if hidden_layer_sizes is not None else ()
        self.hidden_activation_fn = hidden_activation_fn
        self.output_activation_fn = output_activation_fn
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        self._network = self._build_network()

    def _build_network(self) -> nn.Sequential:
        """Builds the neural network's layers and returns a nn.Sequential container."""
        layers = []
        prev_size = self.in_features
        for size in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(size))
            if self.hidden_activation_fn is not None:
                layers.append(self.hidden_activation_fn())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev_size = size

        layers.append(nn.Linear(prev_size, self.out_features))
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
