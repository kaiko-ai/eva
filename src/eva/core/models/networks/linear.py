"""A simple linear classifier as commonly used for linear probing."""

import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    """A simple classifier for linear probing."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_batch_norm: bool = True,
        learnable_batch_norm: bool = False,
    ) -> None:
        """Initializes the LinearClassifier.

        Args:
            in_features: The number of input features.
            out_features: The number of output features.
            use_batch_norm: Whether to apply batch normalization before the linear layer.
            learnable_batch_norm: Whether the batch norm parameters are learnable.
        """
        super().__init__()

        self._use_batch_norm = use_batch_norm

        self.batch_norm = nn.BatchNorm1d(in_features, affine=learnable_batch_norm)
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the LinearClassifier.

        Args:
            x: The input tensor.

        Returns:
            The output of the network.
        """
        if self._use_batch_norm:
            x = self.batch_norm(x)
        return self.linear(x)
