"""Vision backbone helper class."""

from typing import Any, Dict

import torch
from torch import nn

from eva.vision.models.networks.backbones._registry import BackboneModelRegistry


class VisionBackbone(nn.Module):
    """Vision backbone."""

    def __init__(
        self,
        model_name: str,
        arguments: Dict[str, Any] | None = None,
    ) -> None:
        """Initializes and constructs the model.

        Args:
            model_name: The name of the model to load.
            arguments: The arguments used for instantiating the model.
        """
        super().__init__()

        self._model_name = model_name
        self._arguments = arguments

        self._model: nn.Module

        self.configure_model()

    def configure_model(self) -> None:
        """Builds and loads the backbone."""
        self._model = BackboneModelRegistry.load_model(self._model_name, **self._arguments or {})
        VisionBackbone.__name__ = self._model_name

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Returns the embeddings of the model.

        Args:
            tensor: The image tensor (batch_size, num_channels, height, width).

        Returns:
            The image features.
        """
        return self._model(tensor)
