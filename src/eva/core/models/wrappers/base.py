"""Base class for model wrappers."""

import abc
from typing import Callable, Generic, TypeVar

import torch.nn as nn
from typing_extensions import override

InputType = TypeVar("InputType")
"""The input data type."""
OutputType = TypeVar("OutputType")
"""The output data type."""


class BaseModel(nn.Module, Generic[InputType, OutputType]):
    """Base class for model wrappers."""

    def __init__(self, transforms: Callable | None = None) -> None:
        """Initializes the model.

        Args:
            transforms: The transforms to apply to the output produced by the model.
        """
        super().__init__()

        self._output_transforms = transforms

        self._model: Callable[..., OutputType] | nn.Module

    @override
    def forward(self, tensor: InputType) -> OutputType:
        out = self.model_forward(tensor)
        return self._apply_transforms(out)

    @abc.abstractmethod
    def load_model(self) -> Callable[..., OutputType]:
        """Loads the model."""
        raise NotImplementedError

    def model_forward(self, tensor: InputType) -> OutputType:
        """Implements the forward pass of the model.

        Args:
            tensor: The input tensor to the model.
        """
        return self._model(tensor)

    def _apply_transforms(self, tensor: OutputType) -> OutputType:
        if self._output_transforms is not None:
            tensor = self._output_transforms(tensor)
        return tensor
