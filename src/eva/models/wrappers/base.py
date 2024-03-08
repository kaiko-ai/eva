"""Base class for model wrappers."""

import abc
from typing import Callable

import torch
import torch.nn as nn
from typing_extensions import override


class BaseModel(nn.Module):
    """Base class for model wrappers."""

    def __init__(self, tensor_transforms: Callable | None = None) -> None:
        """Initializes the model.

        Args:
            tensor_transforms: The transforms to apply to the output
                tensor produced by the model.
        """
        super().__init__()

        self._output_transforms = tensor_transforms

    @override
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.model_forward(tensor)
        return self._apply_transforms(tensor)

    @abc.abstractmethod
    def load_model(self) -> Callable[..., torch.Tensor]:
        """Loads the model."""
        raise NotImplementedError

    @abc.abstractmethod
    def model_forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Implements the forward pass of the model.

        Args:
            tensor: The input tensor to the model.
        """
        raise NotImplementedError

    def _apply_transforms(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._output_transforms is not None:
            tensor = self._output_transforms(tensor)
        return tensor
