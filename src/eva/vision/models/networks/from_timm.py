"""Helper wrapper class for timm models."""

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
            model_name: Name of model to instantiate.
            pretrained: If set to `True`, load pretrained ImageNet-1k weights.
            checkpoint_path: Path of checkpoint to load.
            model_arguments: The extra callable function / class arguments.
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
