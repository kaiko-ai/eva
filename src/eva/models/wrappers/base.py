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
            tensor_transforms: The transforms to apply to the output tensor produced by the model.
        """
        super().__init__()

        self._output_transforms = tensor_transforms

    @override
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self._forward(tensor)
        return self._apply_transforms(tensor)

    @abc.abstractmethod
    def _load_model(self) -> Callable[..., torch.Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _apply_transforms(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._output_transforms is not None:
            tensor = self._output_transforms(tensor)
        return tensor
