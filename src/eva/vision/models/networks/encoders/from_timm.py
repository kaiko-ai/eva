"""Encoder wrapper for timm models."""

from typing import Any, Dict, List, Tuple

import timm
import torch
from torch import nn
from typing_extensions import override

from eva.vision.models.networks.encoders import encoder


class TimmEncoder(encoder.Encoder):
    """Encoder wrapper for `timm` models.

    Note that only models with `forward_intermediates`
    method are currently only supported.
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = False,
        checkpoint_path: str = "",
        out_indices: int | Tuple[int, ...] | None = 1,
        model_arguments: Dict[str, Any] | None = None,
    ) -> None:
        """Initializes the encoder.

        Args:
            model_name: Name of model to instantiate.
            pretrained: If set to `True`, load pretrained ImageNet-1k weights.
            checkpoint_path: Path of checkpoint to load.
            out_indices: Returns last n blocks if `int`, all if `None`, select
                matching indices if sequence.
            model_arguments: Extra model arguments.
        """
        super().__init__()

        self._model_name = model_name
        self._pretrained = pretrained
        self._checkpoint_path = checkpoint_path
        self._out_indices = out_indices
        self._model_arguments = model_arguments or {}

        self._feature_extractor: nn.Module

        self.configure_model()

    def configure_model(self) -> None:
        """Builds and loads the timm model as feature extractor."""
        self._feature_extractor = timm.create_model(
            model_name=self._model_name,
            pretrained=self._pretrained,
            checkpoint_path=self._checkpoint_path,
            out_indices=self._out_indices,
            features_only=True,
            **self._model_arguments,
        )
        TimmEncoder.__name__ = self._model_name

    @override
    def forward(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        return self._feature_extractor(tensor)
