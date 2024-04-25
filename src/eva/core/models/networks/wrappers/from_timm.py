"""Helper function from models defined with a function."""

from typing import Any, Dict, List

import timm
import torch
from torch import nn
from typing_extensions import override


class TimmModel(nn.Module):
    """Wrapper class for timm models."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = False,
        checkpoint_path: str = "",
        model_arguments: Dict[str, Any] | None = None,
    ) -> None:
        """Initializes and constructs the model.

        Args:
            path: The path to the callable object (class or function).
            arguments: The extra callable function / class arguments.
            checkpoint_path: The path to the checkpoint to load the model
                weights from. This is currently only supported for torch
                model checkpoints. For other formats, the checkpoint loading
                should be handled within the provided callable object in <path>.
            tensor_transforms: The transforms to apply to the output tensor
                produced by the model.
        """
        super().__init__()

        self._model_name = model_name
        self._pretrained = pretrained
        self._checkpoint_path = checkpoint_path
        self._model_arguments = model_arguments or {}

        self._feature_extractor = self._load_model()

    def _load_model(self) -> nn.Module:
        """Builds, loads and returns the model."""
        return timm.create_model(
            model_name=self._model_name,
            pretrained=self._pretrained,
            checkpoint_path=self._checkpoint_path,
            **self._model_arguments,
        )

    @override
    def forward(self, tensor: torch.Tensor) -> torch.Tensor | List[torch.Tensor]:
        return self._feature_extractor(tensor)
