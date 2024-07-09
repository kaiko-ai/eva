"""Encoder base class."""

import abc
from typing import List

import torch
from torch import nn


class Encoder(nn.Module, abc.ABC):
    """Encoder base class."""

    @abc.abstractmethod
    def forward(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Returns the multi-level feature maps of the model.

        Args:
            tensor: The image tensor (batch_size, num_channels, height, width).

        Returns:
            The list of multi-level image features of shape (batch_size,
                hidden_size, num_patches_height, num_patches_width).
        """
