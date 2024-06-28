"""Encoder wrapper for vision models."""

from typing import Any, Callable, Dict, List

import jsonargparse
import torch
from torch import nn
from typing_extensions import override

from eva.core.models.networks import _utils
from eva.vision.models.networks.encoders import encoder


class EncoderWrapper(encoder.Encoder):
    """Encoder wrapper for torchvision vision_transformer models."""

    def __init__(
        self,
        path: Callable[..., nn.Module],
        arguments: Dict[str, Any] | None = None,
        checkpoint_path: str | None = None,
        tensor_transforms: Callable | None = None,
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

        self._path = path
        self._arguments = arguments
        self._checkpoint_path = checkpoint_path
        self._tensor_transforms = tensor_transforms

        self._feature_extractor: nn.Module

        self.configure_model()

    def configure_model(self) -> None:
        """Builds and loads the model."""
        class_path = jsonargparse.class_from_function(self._path, func_return=nn.Module)
        self._feature_extractor = class_path(**self._arguments or {})
        if self._checkpoint_path is not None:
            _utils.load_model_weights(self._feature_extractor, self._checkpoint_path)

    @override
    def forward(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        outputs = self._model(tensor)
        features = self._apply_transforms(outputs)
        return features if isinstance(features, list) else [features]
